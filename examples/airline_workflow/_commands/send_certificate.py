from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances

from ..airline_data import load_data
from ..tools.send_certificate import SendCertificate


class Signature:
    """Send a gift certificate to a user"""

    class Input(BaseModel):
        user_id: str = Field(
            default="NOT_FOUND",
            description="The user ID to send the certificate to",
            pattern=r"^([a-z_0-9]+|NOT_FOUND)$",
            examples=["sara_doe_496", "john_smith_238"],
            json_schema_extra={
                "available_from": ["get_user_details"]
            }
        )
        amount: int = Field(
            description="Certificate amount in dollars",
            ge=1,
            le=1000,
            examples=[50, 100, 200],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        result: str = Field(
            description="Confirmation message or error message"
        )

    plain_utterances: List[str] = [
        "Send a gift certificate to Sara",
        "I'd like to issue a certificate for $100",
        "Give a travel voucher to user john_smith_238",
        "Send a $50 gift certificate to this account",
        "Issue a travel credit to my friend",
        "Can you send a gift card to this user?",
        "Create a flight certificate for $200",
        "I want to give a travel voucher as a gift",
        "Send a certificate worth $150 to this user ID",
        "Issue a travel credit as compensation"
    ]

    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> List[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        
        # Format the response based on success or error
        if output.result and output.result.startswith("Error:"):
            response = f"Failed to send certificate: {output.result}"
        else:
            response = f"Success! {output.result}"
            response += f"\n\nA gift certificate worth ${command_parameters.amount} has been added to the user's payment methods and is ready to use for booking flights."
            
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[CommandResponse(response=response)],
        )

    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        data = load_data()
        result = SendCertificate.invoke(
            data=data, 
            user_id=input.user_id, 
            amount=input.amount
        )
        return Signature.Output(result=result)