from utils.config import begin_sentence, agent_prompt
import os
from utils.custom_types import (
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

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_SERVICE_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_MODEL_NAME = os.getenv('AZURE_OPENAI_MODEL_NAME')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
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
            model='gpt-4o',
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
            Respond with a JSON object with the following structure:
            {
                "appointment_scheduling": true/false,
                "appointment_details": {
                    "time": "extracted time or null",
                    "date": "extracted date or null",
                    "doctor": "doctor name or null"
                }
            }
            """}
        ]
        
        # Add the recent transcript
        detection_prompt.extend(self.convert_transcript_to_openai_messages(recent_transcript))
        
        try:
            # Make a non-streaming call for this analysis task
            response = await self.client.chat.completions.create(
                model='gpt-4o',
                messages=detection_prompt,
                response_format={"type": "json_object"}
            )
            
            # Parse the detected states
            if response.choices and len(response.choices) > 0:
                detected_states = json.loads(response.choices[0].message.content)
                return detected_states
            else:
                logger.warning("No choices in state detection response")
                return {"appointment_scheduling": False, "appointment_details": {}}
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing state detection response: {e}")
            return {"appointment_scheduling": False, "appointment_details": {}}
        except Exception as e:
            logger.error(f"Error in state detection: {e}")
            return {"appointment_scheduling": False, "appointment_details": {}}

    async def detect_appointment_intent(self, transcript: List[Utterance]):
        """Quickly check if there's an intent to book an appointment"""
        # Get the latest user message
        latest_user_msg = next((msg.content for msg in reversed(transcript) 
                              if msg.role == "user"), "")
        
        # Simple keyword check for booking intent
        booking_keywords = ["appointment", "schedule", "book", "reserve", "meet", 
                           "doctor", "consultation", "visit"]
        
        has_intent = any(keyword in latest_user_msg.lower() for keyword in booking_keywords)
        
        if has_intent:
            logger.info("Appointment booking intent detected")
            return True
        return False
    
    async def start_appointment_collection(self, transcript: List[Utterance]):
        """Start collecting appointment details after intent is detected"""
        # This begins the appointment collection state
        collection_prompt = [
            {"role": "system", "content": """
            The user wants to book an appointment. You need to collect:
            1. Preferred date
            2. Preferred time
            3. Doctor's name (if applicable)
            
            Guide the conversation to collect this information naturally.
            Keep track of what's been collected and what's still needed.
            """
            }
        ]
        
        # Include recent conversation context
        collection_prompt.extend(self.convert_transcript_to_openai_messages(transcript))
        
        # First response specifically designed to start the collection process
        response = await self.client.chat.completions.create(
            model='gpt-4o',
            messages=collection_prompt,
        )
        
        return {
            "in_appointment_collection": True,
            "appointment_details": {
                "date": None,
                "time": None,
                "doctor": None
            },
            "next_response": response.choices[0].message.content
        }
    
    async def update_appointment_details(self, transcript: List[Utterance], current_details):
        """Update appointment details based on the latest transcript"""
        update_prompt = [
            {"role": "system", "content": f"""
            You are collecting appointment details. Current status:
            - Date: {current_details.get('date') or 'Not yet provided'}
            - Time: {current_details.get('time') or 'Not yet provided'}
            - Doctor: {current_details.get('doctor') or 'Not yet provided'}
            
            Extract any new information from the latest messages.
            Respond with a JSON containing the updated fields and whether the collection is complete:
            {{
                "date": "extracted date or previous value",
                "time": "extracted time or previous value",
                "doctor": "doctor name or previous value",
                "collection_complete": true/false
            }}
            """
            }
        ]
        
        # Add recent conversation
        recent = transcript[-3:] if len(transcript) > 3 else transcript
        update_prompt.extend(self.convert_transcript_to_openai_messages(recent))
        
        # Get updates to the appointment details
        response = await self.client.chat.completions.create(
            model='gpt-4o',
            messages=update_prompt,
            response_format={"type": "json_object"}
        )
        
        try:
            updated_details = json.loads(response.choices[0].message.content)
            return updated_details
        except Exception as e:
            logger.error(f"Error parsing appointment details: {e}")
            return current_details
