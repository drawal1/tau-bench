from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from ..airline_data import load_data
from ..tools.update_reservation_baggages import UpdateReservationBaggages


class Signature:
    """Update the baggage information of a reservation"""
    class Input(BaseModel):
        reservation_id: str = Field(
            default="NOT_FOUND",
            description="The reservation ID",
            pattern=r"^([A-Z0-9]{6}|NOT_FOUND)$",
            examples=["ZFA04Y", "4WQ150", "VAAOXJ"],
            json_schema_extra={
                "available_from": ["get_reservation_details"]
            }
        )
        total_baggages: int = Field(
            default=0,
            description="The updated total number of baggage items included in the reservation",
            ge=0,
            le=20,
            examples=[1, 2, 5],
            json_schema_extra={
                "available_from": ["get_reservation_details"]
            }
        )
        nonfree_baggages: int = Field(
            default=0,
            description="The updated number of non-free baggage items included in the reservation",
            ge=0,
            le=20,
            examples=[0, 1, 2],
            json_schema_extra={
                "available_from": ["get_reservation_details"]
            }
        )
        payment_id: str = Field(
            default="NOT_FOUND",
            description="The payment method ID for baggage fees",
            pattern=r"^((credit_card|gift_card|certificate)_\d+|NOT_FOUND)$",
            examples=["credit_card_4421486", "gift_card_1234567", "certificate_7504069"],
            json_schema_extra={
                "available_from": ["get_user_details", "get_reservation_details"]
            }
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        status: str = Field(description="Whether baggage update succeeded or error message.")

    plain_utterances: List[str] = [
        "I need to add more baggage to my reservation.",
        "Can I update the number of bags on my booking?",
        "Please change my baggage allowance for my flight.",
        "I want to modify the baggage count on my reservation.",
        "How can I add extra bags to my airline reservation?",
        "I need to increase the number of checked bags on my flight.",
        "Can I change my baggage from 2 bags to 5 bags?",
        "I want to add additional luggage to my booking.",
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
            f'Response: Baggage update result: {output.status}'
        )
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[CommandResponse(response=response)],
        )

    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        data = load_data()
        result = UpdateReservationBaggages.invoke(
            data=data,
            reservation_id=input.reservation_id,
            total_baggages=input.total_baggages,
            nonfree_baggages=input.nonfree_baggages,
            payment_id=input.payment_id,
        )
        return Signature.Output(status=result)
