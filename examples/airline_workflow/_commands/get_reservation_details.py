from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from ..airline_data import load_data
from ..tools.get_reservation_details import GetReservationDetails


class Signature:
    """Get reservation details"""
    class Input(BaseModel):
        reservation_id: str = Field(
            default="NOT_FOUND",
            description="The reservation ID to get details for",
            pattern=r"^([A-Z0-9]{6}|NOT_FOUND)$",
            examples=["8JX2WO"],
            json_schema_extra={
                "available_from": ["get_user_details", "search_direct_flight", "search_onestop_flight"]
            }
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        reservation_details: str = Field(
            description=(
                "Detailed reservation information such as "
                "reservation ID, user ID, origin, destination, flight type, cabin, "
                "flight details, passengers, payment history, baggage info, and insurance"
            ),
            json_schema_extra={
                "used_by": [
                    "cancel_reservation",
                    "update_reservation_baggages",
                    "update_reservation_flights", 
                    "update_reservation_passengers",
                    "send_certificate",
                ]
            },
        )

    plain_utterances: List[str] = [
        "Can you show me my reservation details?",
        "I want to see the details of my booking.",
        "What information do you have about my reservation?",
        "Can you pull up my flight reservation?",
        "I'd like to review my booking details.",
        "Show me my flight information.",
        "What are the details of my trip?",
        "retrieve reservation details"
    ]
    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> List[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        from fastworkflow.train.generate_synthetic import generate_diverse_utterances
        return generate_diverse_utterances(utterances_obj.plain_utterances, command_name)
        
class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        response = (
            f'Response: {output.reservation_details}'
        )
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[CommandResponse(response=response)],
        )

    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        data = load_data()
        reservation_details = GetReservationDetails.invoke(data=data, reservation_id=input.reservation_id)
        return Signature.Output(reservation_details=reservation_details)
