import os
import asyncio
import logging
import json
import re
from typing import List, Dict, Any, AsyncGenerator
from utils.custom_types import ResponseRequiredRequest, ResponseResponse, Utterance
from crewai_agents.crew import MedicalOfficeVoiceApp, fallback_response


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
        logger.info("Generating initial greeting")
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

    def _clean_response(self, response_text):
        """Clean the response text to ensure it's user-friendly"""
        # If it's already a string, check if it looks like JSON
        if isinstance(response_text, str):
            # If it looks like a JSON string, try to parse it
            if response_text.strip().startswith('{') and response_text.strip().endswith('}'):
                try:
                    json_obj = json.loads(response_text)
                    if isinstance(json_obj, dict) and 'response' in json_obj:
                        return json_obj['response']
                except json.JSONDecodeError:
                    pass
            
            # Return the raw string if it doesn't look like JSON or can't be parsed
            return response_text
        
        # If it's a dict, extract the response key
        elif isinstance(response_text, dict) and 'response' in response_text:
            return response_text['response']
        
        # If it has a response attribute
        elif hasattr(response_text, 'response'):
            return response_text.response
        
        # Convert to string as a last resort
        return str(response_text)

    def _extract_response_content(self, crew_response) -> str:
        """
        Extract the response content from crew output using multiple methods
        """
        try:
            # Method 1: If it's a dictionary with a 'response' key
            if isinstance(crew_response, dict) and 'response' in crew_response:
                return crew_response['response']
            
            # Method 2: If it's an object with a response attribute
            if hasattr(crew_response, 'response'):
                return crew_response.response
            
            # Method 3: If it's a string that might contain JSON
            if isinstance(crew_response, str):
                # Try direct return if it's not JSON
                if not ('{' in crew_response and '}' in crew_response):
                    return crew_response
                
                # Try parsing as JSON
                try:
                    json_obj = json.loads(crew_response)
                    if isinstance(json_obj, dict) and 'response' in json_obj:
                        return json_obj['response']
                except json.JSONDecodeError:
                    pass
                
                # Try regex extraction
                response_match = re.search(r'"response"\s*:\s*"(.+?)"', crew_response, re.DOTALL)
                if response_match:
                    # Handle escaped quotes and clean up the extracted string
                    extracted = response_match.group(1)
                    extracted = extracted.replace('\\"', '"').replace('\\n', '\n')
                    return extracted
            
            # Method 4: Convert to string and try again with regex
            str_response = str(crew_response)
            response_match = re.search(r'"response"\s*:\s*"(.+?)"', str_response, re.DOTALL)
            if response_match:
                extracted = response_match.group(1)
                extracted = extracted.replace('\\"', '"').replace('\\n', '\n')
                return extracted
            
            # Method 5: Look for the most likely response pattern
            final_answer_match = re.search(r'## Final Answer:.*?{.*?"response"\s*:\s*"(.+?)"', str_response, re.DOTALL)
            if final_answer_match:
                extracted = final_answer_match.group(1)
                extracted = extracted.replace('\\"', '"').replace('\\n', '\n')
                return extracted
            
            # Last resort: return the string version if all else fails
            return str_response
            
        except Exception as e:
            logger.error(f"Error extracting response: {str(e)}")
            return ""

    async def draft_response(self, request: ResponseRequiredRequest) -> AsyncGenerator[ResponseResponse, None]:
        """Generate a response using the CrewAI system with simplified response handling"""
        try:
            # Get the last user message
            last_user_message = ""
            for utterance in reversed(request.transcript):
                if utterance.role == "user":
                    last_user_message = utterance.content
                    break
            
            # Convert transcript to context
            context = self.convert_transcript_to_context(request.transcript)
            
            # Log what we're processing
            logger.info(f"Processing request with last message: '{last_user_message}'")
            
            # Check for appointment-related keywords
            appointment_keywords = ["appointment", "book", "schedule", "doctor", "visit", "checkup", "meeting"]
            is_appointment_request = any(keyword in last_user_message.lower() for keyword in appointment_keywords)
            logger.info(f"Is appointment request: {is_appointment_request}")
            
            # Default response in case something goes wrong
            final_response = "I'm sorry, I didn't understand that. How can I assist you with your call today?"
            
            try:
                # Run the crew with a timeout to prevent hanging
                crew_future = asyncio.get_event_loop().run_in_executor(
                    None,
                    self.medical_crew.kickoff,
                    {
                        "conversation_context": context,
                        "last_user_message": last_user_message,
                        "is_appointment_request": is_appointment_request
                    }
                )
                
                # Set a reasonable timeout (10 seconds)
                crew_response = await asyncio.wait_for(crew_future, timeout=10.0)
                
                # Log the raw response for debugging
                logger.info(f"Raw crew response type: {type(crew_response)}")
                logger.info(f"Raw crew response: {str(crew_response)[:200]}...")
                
                # Extract and clean the response 
                extracted_response = self._extract_response_content(crew_response)
                
                if extracted_response:
                    # Make sure the response is clean before sending
                    clean_response = self._clean_response(extracted_response)
                    logger.info(f"Cleaned response: {clean_response[:100]}...")
                    final_response = clean_response
                else:
                    logger.warning("Failed to extract response from crew output. Using fallback.")
                    final_response = fallback_response(last_user_message) if is_appointment_request else "I'm sorry, I didn't understand that. How can I assist you today?"
                
            except asyncio.TimeoutError:
                logger.error("Crew execution timed out")
                final_response = "I'm sorry for the delay. How can I help you with your appointment today?" if is_appointment_request else "I'm sorry for the delay. How can I assist you today?"
                
            except Exception as e:
                logger.error(f"Error in crew execution: {str(e)}")
                final_response = fallback_response(last_user_message)
            
            # For streaming effect, chunk the response
            chunk_size = 10  # characters per chunk
            for i in range(0, len(final_response), chunk_size):
                chunk = final_response[i:i+chunk_size]
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=chunk,
                    content_complete=False,
                    end_call=False,
                )
                
                # Check if this request is still the most recent one
                if hasattr(request, '_response_override') and request.response_id < request._response_override:
                    logger.info("Abandoning response due to new request")
                    break
                
                yield response
                
                # Small delay for more natural speech
                await asyncio.sleep(0.05)
                
            # Send final completion message
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