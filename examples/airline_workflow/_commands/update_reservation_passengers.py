from typing import List, Optional

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from ..airline_data import load_data
from ..tools.update_reservation_passengers import UpdateReservationPassengers


class PassengerInfo(BaseModel):
    first_name: str = Field(
        description="First name of the passenger",
        examples=["John", "Sarah", "Michael"]
    )
    last_name: str = Field(
        description="Last name of the passenger", 
        examples=["Doe", "Smith", "Johnson"]
    )
    dob: Optional[str] = Field(
        default=None,
        description="Date of birth in YYYY-MM-DD format, or None if unknown",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        examples=["1990-01-01", "1985-05-15", "1992-12-31"]
    )


class Signature:
    """Update the passenger information of a reservation"""
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
        passengers: List[PassengerInfo] = Field(
            default=[],
            description="An array of objects containing details about each passenger",
            examples=[[{"first_name": "Noah", "last_name": "Brown", "dob": "1990-01-01"}, {"first_name": "Emma", "last_name": "Smith", "dob": "1985-05-15"}]]
        )
        

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        status: str = Field(description="Whether passenger update succeeded or error message.")

    plain_utterances: List[str] = [
        "I need to update the passenger information on my reservation.",
        "Can I change the passenger details for my booking?",
        "Please update the names and dates of birth for the travelers on my reservation.",
        "I want to modify the passenger information on my flight reservation.",
        "How can I update the passenger details for my airline reservation?",
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
            f'Response: Update result: {output.status}'
        )
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[CommandResponse(response=response)],
        )

    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        data = load_data()
        
        # Convert Pydantic models to dictionaries for the tool
        passengers_dict = [passenger.model_dump() for passenger in input.passengers]
        
        result = UpdateReservationPassengers.invoke(
            data=data,
            reservation_id=input.reservation_id,
            passengers=passengers_dict,
        )
        return Signature.Output(status=result)
