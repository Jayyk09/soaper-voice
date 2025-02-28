from config import begin_sentence, agent_prompt
import os
from custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)
from typing import List
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
import logging
import json

# Load environment variables from .env file
load_dotenv()

AZURE_OPENAI_SERVICE_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_SERVICE_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT_MODEL_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_MODEL_NAME')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')

logger = logging.getLogger(__name__)

client = AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_SERVICE_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_SERVICE_ENDPOINT,
    )

class LLMClient:
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_SERVICE_ENDPOINT,
        )

    def draft_begin_message(self):
        return ResponseResponse(
            response_id=0,
            content=begin_sentence,
            content_complete=True,
            end_call=False,
        )

    def convert_transcript_to_openai_messages(self, transcript: List[Utterance]):
        messages = []
        for utterance in transcript:
            role = "assistant" if utterance.role == "agent" else "user"
            messages.append({"role": role, "content": utterance.content})
        return messages

    def prepare_prompt(self, request: ResponseRequiredRequest):
        prompt = [{"role": "system", "content": agent_prompt}]
        prompt.extend(self.convert_transcript_to_openai_messages(request.transcript))

        if request.interaction_type == "reminder_required":
            prompt.append({
                "role": "user",
                "content": "(Now the user has not responded in a while, you would say:)"
            })
        
        return prompt

    async def draft_response(self, request: ResponseRequiredRequest, call_state=None):
        # Detect states in the conversation if not already running in a state-aware context
        if call_state is None:
            call_state = {}
            try:
                detected_states = await self.detect_state(request.transcript)
                call_state.update(detected_states)
            except Exception as e:
                print(f"State detection error: {e}")
        
        # Prepare prompt with state awareness
        prompt = self.prepare_prompt(request)
        
        # If we detected an appointment state, add it to the system message
        if call_state.get("appointment_scheduling", False):
            prompt[0]["content"] += f"\n\nThe user is currently discussing an appointment. Details: {json.dumps(call_state.get('appointment_details', {}))}"
        
        # Continue with regular response generation
        stream = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=prompt,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield ResponseResponse(
                    response_id=request.response_id,
                    content=chunk.choices[0].delta.content,
                    content_complete=False,
                    end_call=False,
                )

        yield ResponseResponse(
            response_id=request.response_id,
            content="",
            content_complete=True,
            end_call=False,
        )

        async def detect_state(self, transcript: List[Utterance]):
            # Detect various states in the conversation, including appointments
            # Take the most recent utterances to analyze
            recent_transcript = transcript[-5:] if len(transcript) > 5 else transcript
            
            # Prepare a focused prompt for state detection
            detection_prompt = [
                {"role": "system", "content": """
                Analyze this conversation fragment and identify if any of these states are present:
                1. Appointment scheduling/discussion
                2. Booking time and date detail
                3. Doctor's name
                 
                For each detected state, extract the relevant details in a structured format.
                Respond with a JSON object containing the detected states.
                """}
            ]
            
            # Add the recent transcript
            detection_prompt.extend(self.convert_transcript_to_openai_messages(recent_transcript))
            
            # Make a non-streaming call for this analysis task
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=detection_prompt,
                response_format={"type": "json_object"}
            )
            
            # Parse the detected states
            try:
                detected_states = json.loads(response.choices[0].message.content)
                return detected_states
            except json.JSONDecodeError:
                print("Error parsing state detection response")
                return {}
