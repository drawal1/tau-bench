# FastWorkflow Agent Adapter for Tau Bench
import contextlib
import json
import logging
import os
import time
import queue
import copy
from typing import List, Dict, Any, Optional, Tuple
from dotenv import dotenv_values

from tau_bench.agents.base import Agent
from tau_bench.types import SolveResult, Action as TauAction, RESPOND_ACTION_NAME, RESPOND_ACTION_FIELD_NAME
from tau_bench.envs.base import Env
 

logger = logging.getLogger(__name__)


def _json_deepcopy(obj: Any) -> Any:
    with contextlib.suppress(Exception):
        return json.loads(json.dumps(obj))
    return copy.deepcopy(obj)


def _collect_diffs(agent: Any, gt: Any, path: str, diffs: List[Dict[str, Any]], limit: int) -> None:
    if len(diffs) >= limit:
        return
    if type(agent) != type(gt):
        diffs.append({"path": path, "agent": str(agent), "gt": str(gt)})
        return
    if isinstance(agent, dict):
        keys = set(agent.keys()) | set(gt.keys())
        for k in sorted(keys):
            if len(diffs) >= limit:
                break
            np = f"{path}.{k}" if path else str(k)
            if k not in agent:
                diffs.append({"path": np, "agent": "<missing>", "gt": str(gt[k])})
            elif k not in gt:
                diffs.append({"path": np, "agent": str(agent[k]), "gt": "<missing>"})
            else:
                _collect_diffs(agent[k], gt[k], np, diffs, limit)
        return
    if isinstance(agent, list):
        if len(agent) != len(gt):
            diffs.append({"path": path, "agent_len": len(agent), "gt_len": len(gt)})
        # Compare element-wise up to limit
        for i, (av, gv) in enumerate(zip(agent, gt)):
            if len(diffs) >= limit:
                break
            _collect_diffs(av, gv, f"{path}[{i}]" if path else f"[{i}]", diffs, limit)
        return
    if agent != gt:
        diffs.append({"path": path, "agent": str(agent), "gt": str(gt)})


def _simulate_gt_data(env: Env) -> Dict[str, Any]:
    data = env.data_load_func()
    for action in env.task.actions:
        if action.name in getattr(env, "terminate_tools", []):
            continue
        tool_cls = env.tools_map.get(action.name)
        if tool_cls is None:
            continue
        with contextlib.suppress(Exception):
            tool_cls.invoke(data=data, **action.kwargs)
    return data


class FastWorkflowAgentAdapter(Agent):
    """
    FastWorkflow agent adapter that integrates with Tau Bench.
    
    This adapter now consumes FastWorkflow's command trace queue to obtain
    executed commands (name, parameters, responses) and steps the Tau Bench env
    directly to compute rewards, so Tau Bench's run.py works as-is.

    It also drains the FastWorkflow command_output_queue to relay agent
    questions to the Tau Bench user simulator, and pushes user replies into
    fastworkflow.chat_session.user_message_queue to avoid hangs.
    """
    
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str = "mistral-small-latest",
        provider: str = "mistral",
        temperature: float = 0.0,
        use_reasoning: bool = True,
        **kwargs
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        
        # Determine the environment type from the wiki content
        env_type = "retail"  # Default to retail
        if "airline" in wiki.lower():
            env_type = "airline"
        elif "retail" in wiki.lower():
            env_type = "retail"
        
        # Find the appropriate workflow path
        self.workflow_path = self._find_workflow_path(env_type)
        
        logger.info(f"FastWorkflow adapter initialized with model {model} from {provider}")
        logger.info(f"Using {env_type} workflow at: {self.workflow_path}")
    
    def _find_workflow_path(self, env_type: str) -> str:
        """Find the path to the workflow based on environment type."""
        current_dir = os.getcwd()
        workflow_path = os.path.join(current_dir, "examples", f"{env_type}_workflow")
        
        if os.path.exists(workflow_path):
            return workflow_path
        
        raise FileNotFoundError(
            f"Could not find {env_type} workflow. Expected at: {workflow_path}. "
            f"Run 'fastworkflow examples fetch {env_type}_workflow' to install it."
        )
    
    def _to_plain_kwargs(self, params: Any) -> Dict[str, Any]:
        """Best-effort conversion of parameters to a plain dict compatible with Tau Bench tools."""
        if params is None:
            return {}
        if isinstance(params, dict):
            return params
        # pydantic models
        with contextlib.suppress(Exception):
            return params.model_dump()
        with contextlib.suppress(Exception):
            return params.dict()
        # generic objects
        with contextlib.suppress(Exception):
            return dict(params)
        return {}
    
    def _drain_command_trace(
        self,
        fastworkflow,
        env: Env,
        max_drain: int,
        aggregated_response_texts: List[str],
        actions_executed: List[Dict[str, Any]],
    ) -> int:
        """Drain the command_trace_queue; step env for tool calls; collect texts.
        Returns number of items processed.
        """
        processed = 0
        while processed < max_drain:
            try:
                evt = fastworkflow.chat_session.command_trace_queue.get_nowait()
            except queue.Empty:
                break

            processed += 1

            is_agent_to_workflow = (
                getattr(evt, "direction", None)
                == getattr(fastworkflow, "CommandTraceEventDirection", None).AGENT_TO_WORKFLOW
                if hasattr(fastworkflow, "CommandTraceEventDirection")
                else False
            )

            if not is_agent_to_workflow:
                cmd_name = getattr(evt, "command_name", None)
                params = self._to_plain_kwargs(getattr(evt, "parameters", None))
                response_text = getattr(evt, "response_text", None)

                if isinstance(response_text, str) and len(response_text) > 0:
                    aggregated_response_texts.append(response_text)

                if isinstance(cmd_name, str) and len(cmd_name) > 0:
                    actions_executed.append({
                        "name": cmd_name,
                        "kwargs": params,
                        "response_text": response_text,
                        "success": getattr(evt, "success", None),
                    })
                    # Step Tau Bench env with tool
                    env.step(TauAction(name=cmd_name, kwargs=params))
        return processed

    def _drain_agent_outputs_and_respond(
        self,
        fastworkflow,
        env: Env,
        max_drain: int,
        aggregated_response_texts: List[str],
    ) -> Tuple[int, bool]:
        """Drain FastWorkflow's command_output_queue.
        For each agent output (question), generate a user reply via env.user and
        feed it back through user_message_queue. Returns (processed_count, done).
        """
        processed = 0
        done = False
        while processed < max_drain:
            try:
                out = fastworkflow.chat_session.command_output_queue.get_nowait()
            except queue.Empty:
                break

            processed += 1

            # CommandOutput objects carry one or more CommandResponses
            agent_texts: List[str] = []
            if hasattr(out, "command_responses") and isinstance(out.command_responses, list):
                for cr in out.command_responses:
                    txt = getattr(cr, "response", None)
                    if isinstance(txt, str) and txt.strip():
                        agent_texts.append(txt.strip())
            elif isinstance(out, str) and out.strip():
                agent_texts.append(out.strip())

            if not agent_texts:
                continue

            # Stitch and record
            agent_text = "\n".join(agent_texts)
            aggregated_response_texts.append(agent_text)

            # Let Tau Bench user simulator respond
            res = env.step(
                TauAction(name=RESPOND_ACTION_NAME, kwargs={RESPOND_ACTION_FIELD_NAME: agent_text})
            )
            # Feed user's reply back to FastWorkflow
            user_reply = res.observation
            if isinstance(user_reply, str) and user_reply:
                fastworkflow.chat_session.user_message_queue.put(user_reply)
                aggregated_response_texts.append(user_reply)
            if getattr(res, "done", False):
                done = True
                break
        return processed, done

    def solve(
        self, 
        env: Env, 
        task_index: Optional[int] = None, 
        max_num_steps: int = 200
    ) -> SolveResult:
        """
        Solve a task using FastWorkflow's agent, consuming the command trace and outputs to step the Tau Bench env and user.
        """
        try:
            # Load environment variables like CLI
            env_vars = {
                **dotenv_values('examples/fastworkflow.env'),
                **dotenv_values('examples/fastworkflow.passwords.env')
            }

            # Initialize FastWorkflow
            import fastworkflow
            fastworkflow.init(env_vars=env_vars)
            logger.info("✅ FastWorkflow initialized")

            # Clear any lingering workflow stack from prior runs (process-wide)
            with contextlib.suppress(Exception):
                fastworkflow.ChatSession.clear_workflow_stack()

            run_as_agent = True
            fastworkflow.chat_session = fastworkflow.ChatSession(run_as_agent=run_as_agent)
            logger.info("✅ Chat session created")

            # Start workflow with the task instruction; keep_alive for interactive loops
            initial_observation = env.tasks[task_index].instruction
            # Reset the environment state
            env.reset(task_index=task_index)
            logger.info(f"🎯 Starting task {task_index}: {initial_observation}")

            fastworkflow.chat_session.start_workflow(
                self.workflow_path, 
                workflow_context=None,
                startup_command=initial_observation, 
                startup_action=None, 
                keep_alive=True,
                project_folderpath=None
            )

            actions_executed: List[Dict[str, Any]] = []
            aggregated_response_texts: List[str] = []
            steps_taken = 0
            done = False

            # Allow for several idle cycles to let the agent produce the first events
            idle_cycles = 0
            idle_limit = 150  # increased patience (~22.5s at 0.15s sleep)

            # Bounded interaction loop
            for _ in range(max_num_steps):
                progressed = 0
                progressed += self._drain_command_trace(
                    fastworkflow,
                    env,
                    max_drain=200,
                    aggregated_response_texts=aggregated_response_texts,
                    actions_executed=actions_executed,
                )
                processed_out, out_done = self._drain_agent_outputs_and_respond(
                    fastworkflow,
                    env,
                    max_drain=200,
                    aggregated_response_texts=aggregated_response_texts,
                )
                progressed += processed_out
                if out_done:
                    done = True
                steps_taken += progressed
                if progressed == 0:
                    idle_cycles += 1
                else:
                    idle_cycles = 0
                if done or idle_cycles >= idle_limit:
                    break
                # Give the agent a moment to process the user reply
                time.sleep(0.15)

            # Capture agent-final data for diffing BEFORE calculate_reward mutates env.data
            agent_final_data = _json_deepcopy(env.data)
            gt_data_sim = _simulate_gt_data(env)
            # Build capped diffs
            diffs: List[Dict[str, Any]] = []
            _collect_diffs(agent_final_data, gt_data_sim, path="", diffs=diffs, limit=50)
            # Hashes for quick reference
            agent_hash = json.dumps(agent_final_data, sort_keys=True)
            gt_hash = json.dumps(gt_data_sim, sort_keys=True)

            # Compute reward against ground truth (will reset env.data to GT inside)
            reward_res = env.calculate_reward()
            reward = reward_res.reward

            # Build messages summary (minimal)
            combined_response = "\n".join(aggregated_response_texts).strip()
            messages = [
                {"role": "system", "content": self.wiki},
                {"role": "user", "content": initial_observation},
            ]
            if combined_response:
                messages.append({"role": "assistant", "content": combined_response})

            info: Dict[str, Any] = {
                "task_index": task_index,
                "steps": steps_taken,
                "executed_actions": actions_executed,
                "reward_info": reward_res.model_dump() if hasattr(reward_res, "model_dump") else {},
            }

            return SolveResult(reward=reward, messages=messages, info=info)

        except Exception as e:
            logger.error(f"❌ Error in FastWorkflow agent solve: {e}")
            import traceback
            traceback.print_exc()
            fallback_messages = [
                {"role": "system", "content": self.wiki},
            ]
            return SolveResult(
                reward=0.0,
                messages=fallback_messages,
                info={"error": str(e)},
            )
        finally:
            pass
