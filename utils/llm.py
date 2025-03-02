import os
import asyncio
import logging
from typing import List, Dict, Any, AsyncGenerator
from utils.custom_types import ResponseRequiredRequest, ResponseResponse, Utterance
from agents.crew import MedicalOfficeVoiceApp, fallback_response

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Initial greeting message
begin_sentence = "Hello, thank you for calling. This is Joann from Soaper Medical Office! How can I help you today?"

class LLMClient:
    def __init__(self):
        # Initialize the crew for the current session
        self.medical_crew = MedicalOfficeVoiceApp().crew()
        
    def draft_begin_message(self):
        """Return the initial greeting message"""
        return ResponseResponse(
            response_id=0,
            content=begin_sentence,
            content_complete=True,
            end_call=False,
        )

    def convert_transcript_to_context(self, transcript: List[Utterance]) -> str:
        """Convert transcript to a context string for the CrewAI"""
        context = ""
        for utterance in transcript:
            speaker = "Joann" if utterance.role == "agent" else "Caller"
            context += f"{speaker}: {utterance.content}\n\n"
        return context

    async def draft_response(self, request: ResponseRequiredRequest) -> AsyncGenerator[ResponseResponse, None]:
        """Generate a response using the CrewAI system"""
        try:
            # Convert transcript to context
            context = self.convert_transcript_to_context(request.transcript)
            
            # Get the last user message
            last_user_message = ""
            for utterance in reversed(request.transcript):
                if utterance.role == "user":
                    last_user_message = utterance.content
                    break
            
            # Simple keyword check for appointment-related queries
            appointment_keywords = ["appointment", "book", "schedule", "doctor", "visit", "checkup"]
            is_appointment_request = any(keyword in last_user_message.lower() for keyword in appointment_keywords)
            
            # Prepare inputs for the crew
            inputs = {
                "conversation_context": context,
                "last_user_message": last_user_message,
                "is_appointment_request": is_appointment_request
            }
            
            # Log the request
            logger.info(f"Processing request with last message: '{last_user_message}'")
            logger.info(f"Is appointment request: {is_appointment_request}")
            
            try:
                # Run the crew with the sequential process
                crew_response = await asyncio.to_thread(
                    self.medical_crew.kickoff,
                    inputs=inputs
                )
                
                # Extract the response based on the format we now know works
                response_content = ""
                
                # For debugging
                logger.info(f"Crew response type: {type(crew_response)}")
                
                # Handle dict with 'response' key (our Pydantic model serializes to this)
                if isinstance(crew_response, dict) and "response" in crew_response:
                    response_content = crew_response["response"]
                # Handle Pydantic model directly
                elif hasattr(crew_response, "response"):
                    response_content = crew_response.response
                # Handle string response 
                elif isinstance(crew_response, str):
                    response_content = crew_response
                # Handle JSON string (sometimes the response is returned as JSON string)
                elif isinstance(crew_response, str) and "{" in crew_response and "response" in crew_response:
                    import json
                    try:
                        json_response = json.loads(crew_response)
                        if "response" in json_response:
                            response_content = json_response["response"]
                    except:
                        # If JSON parsing fails, use the string as is
                        response_content = crew_response
                
                # Log the extracted content
                logger.info(f"Extracted response content: {response_content[:100]}...")
                
            except Exception as e:
                logger.error(f"Error in crew execution: {str(e)}")
                # Fall back to keyword-based response
                if is_appointment_request:
                    response_content = "I'd be happy to help you book an appointment. Could you please provide your name and when you would like to come in?"
                else:
                    response_content = fallback_response(last_user_message)
            
            # If empty response, provide a fallback
            if not response_content:
                response_content = "I'm sorry, I didn't understand that. How can I assist you with your call today?"
            
            # For streaming effect, chunk the response
            chunk_size = 10  # characters per chunk
            for i in range(0, len(response_content), chunk_size):
                chunk = response_content[i:i+chunk_size]
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=chunk,
                    content_complete=False,
                    end_call=False,
                )
                yield response
                
                # Add small delay for more natural speech
                await asyncio.sleep(0.05)
                
            # Send final response
            yield ResponseResponse(
                response_id=request.response_id,
                content="",
                content_complete=True,
                end_call=False,
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Return a fallback response
            yield ResponseResponse(
                response_id=request.response_id,
                content="I'm sorry, I'm having trouble processing that right now. Could you please repeat your question?",
                content_complete=True,
                end_call=False,
            )