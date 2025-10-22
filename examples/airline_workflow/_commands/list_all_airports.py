from typing import List

from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

# Import business-logic helper
from ..airline_data import load_data
from ..tools.list_all_airports import ListAllAirports


class Signature:
    """List all airports"""
    """Metadata and parameter definitions for `list_all_airports`."""

    class Input(BaseModel):
        """No parameters expected for this command."""

    class Output(BaseModel):
        status: str = Field(
            description="List of airports and their cities as a JSON string representation.",
            json_schema_extra={
                "used_by": ["search_direct_flight", "search_onestop_flight"]
            }
        )

    # ---------------------------------------------------------------------
    # Utterances
    # ---------------------------------------------------------------------

    plain_utterances: List[str] = [
        "What airports do you fly to?",
        "Can you show me all airports you serve?",
        "I want to see all available airports.",
        "What are all the airports in your network?",
        "Which airports can I fly from or to?",
        "Show me all destinations you offer.",
        "What cities do you have flights to?",
        "List all airports and cities you serve.",
        "I need to see all available airports.",
        "What are your airport options?",
    ]

    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> List[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)

        from fastworkflow.train.generate_synthetic import generate_diverse_utterances

        return generate_diverse_utterances(utterances_obj.plain_utterances, command_name)


class ResponseGenerator:
    def __call__(
        self,
        workflow: Workflow,
        command: str,
        command_parameters: Signature.Input | None = None,
    ) -> CommandOutput:
        output = self._process_command(workflow)
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=f"Available airports: {output.status}")
            ],
        )

    def _process_command(self, workflow: Workflow) -> Signature.Output:
        """Run domain logic and wrap into `Signature.Output`."""
        data = load_data()
        result = ListAllAirports.invoke(data=data)
        return Signature.Output(status=result)
