from utils.config import begin_sentence, agent_prompt
import os
from openai import AsyncAzureOpenAI
from utils.custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)
from typing import List
from dotenv import load_dotenv
import asyncio
import json

load_dotenv()

class LLMClient:
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
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
        prompt = [
            {"role": "system", 
            "content": '##Objective\nYou are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible\n\n## Style Guardrails\n- [Be concise] Keep your response succinct, short, and get to the point quickly. Address one question or action item at a time. Don\'t pack everything you want to say into one utterance.\n- [Do not repeat] Don\'t repeat what\'s in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.\n- [Be conversational] Speak like a human as though you\'re speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.\n- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don\'t be a pushover.\n- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.\n\n## Response Guideline\n- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn\'t catch that", "some noise", "pardon", "you\'re coming through choppy", "static in your speech", "voice is cutting in and out"). Do not ever mention "transcription error", and don\'t repeat yourself.\n- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don\'t repeat yourself in doing this. You should still be creative, human-like, and lively.\n- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.\n\n## Role\n'
            + agent_prompt
            },

        ]
        prompt.extend(self.convert_transcript_to_openai_messages(request.transcript))

        if request.interaction_type == "reminder_required":
            prompt.append({
                "role": "user",
                "content": "(Now the user has not responded in a while, you would say:)"
            })
        
        return prompt
    
        # Step 1: Prepare the function calling definition to the prompt
    def prepare_functions(self):
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "book_appointment",
                    "description": "Book an appointment with a doctor after collecting all required information.",
                    "parameters": {
                        "type": "object",
                        "properties": {  # This "properties" key was missing
                            "first_name": {
                                "type": "string",
                                "description": "Patient's first name",
                            },
                            "last_name": {
                                "type": "string",
                                "description": "Patient's last name",
                            },
                            "date": {
                                "type": "string",
                                "description": "Appointment date in YYYY-MM-DD format",
                            },
                            "time": {
                                "type": "string",
                                "description": "Appointment time in HH:MM format (24-hour)",
                            },
                            "doctor_name": {
                                "type": "string",
                                "description": "Name of the doctor to book with",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief reason for the appointment",
                            },
                        },
                        "required": ["first_name", "last_name", "date", "time", "doctor_name", "reason"],
                    },
                },
            },
        ]
        return functions

    async def draft_response(self, request: ResponseRequiredRequest):
        prompt = self.prepare_prompt(request)
        func_call = {}
        func_arguments = ""
        print(f"Sending prompt with {len(prompt)} messages")
        
        try:
            # Create the streaming request
            stream = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=prompt,
                stream=True,
                tools=self.prepare_functions(),
                tool_choice="auto",
            )

            # Process the stream
            async for chunk in stream:
                # Skip chunks with empty choices
                if not chunk.choices:
                    continue

                # Process function calling chunks
                if chunk.choices[0].delta.tool_calls:
                    print("Function calling detected in response")
                    tool_calls = chunk.choices[0].delta.tool_calls[0]
                    if tool_calls.id:
                        if func_call:
                            # Another function received, old function complete
                            break
                        func_call = {
                            "id": tool_calls.id,
                            "func_name": tool_calls.function.name or "",
                            "arguments": {},
                        }
                        print(f"Function call initiated: {func_call['func_name']}")
                    else:
                        # append argument
                        func_arguments += tool_calls.function.arguments or ""
                        print(f"Function arguments received: {tool_calls.function.arguments}")

                # Process content chunks
                if chunk.choices[0].delta.content:
                    print(f"Content chunk received: {chunk.choices[0].delta.content}")
                    yield ResponseResponse(
                        response_id=request.response_id,
                        content=chunk.choices[0].delta.content,
                        content_complete=False,
                        end_call=False,
                    )

            print(f"Streaming complete. Function call: {func_call}, Arguments collected: {func_arguments}")

            # Process function calls if present
            if func_call and func_call["func_name"] == "book_appointment":
                try:
                    print("Parsing function arguments...")
                    func_call["arguments"] = json.loads(func_arguments)
                    print(f"Parsed arguments: {func_call['arguments']}")
                    
                    # Make API call to book appointment
                    print("Calling booking API...")
                    api_result = await self.book_appointment(func_call["arguments"])
                    print(f"API result: {api_result}")
                    
                    # Generate response based on API result
                    if api_result.get("status") == "success":
                        doctor_name = func_call["arguments"].get("doctor_name", "the doctor")
                        content = f"Great news! I've booked your appointment with Dr. {doctor_name} on {func_call['arguments']['date']} at {func_call['arguments']['time']}. Your confirmation number is {api_result.get('appointment_id')}. Is there anything else I can help with?"
                    elif api_result.get("error_code") == "DOCTOR_NOT_AVAILABLE":
                        content = f"I'm sorry, but Dr. {func_call['arguments'].get('doctor_name', 'the doctor')} is not available on {func_call['arguments']['date']} at {func_call['arguments']['time']}. Would you like to try a different time or date?"
                    elif api_result.get("error_code") == "TIME_NOT_AVAILABLE":
                        content = f"I'm sorry, but that time slot ({func_call['arguments']['time']} on {func_call['arguments']['date']}) is not available. Would you like to try a different time?"
                    else:
                        error_message = api_result.get("message", "There was an error booking your appointment")
                        content = f"I'm sorry, but {error_message}. Would you like to try again?"
                    
                    print(f"Sending function response: {content}")
                    response = ResponseResponse(
                        response_id=request.response_id,
                        content=content,
                        content_complete=True,
                        end_call=False,
                    )
                    yield response
                except json.JSONDecodeError as e:
                    print(f"Error parsing function arguments: {str(e)}")
                    print(f"Raw arguments: {func_arguments}")
                    yield ResponseResponse(
                        response_id=request.response_id,
                        content="I'm sorry, I couldn't process your appointment request correctly. Let's try again. What date were you looking for?",
                        content_complete=True,
                        end_call=False,
                    )
            else:
                # No functions called, just complete the response
                print("No function called, completing response")
                response = ResponseResponse(
                    response_id=request.response_id,
                    content="",
                    content_complete=True,
                    end_call=False,
                )
                yield response

        except Exception as e:
            print(f"Error in draft_response: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Provide a fallback response in case of errors
            yield ResponseResponse(
                response_id=request.response_id,
                content="I'm sorry, I'm having trouble at the moment. Please try again.",
                content_complete=True,
                end_call=False,
            )

    async def book_appointment(self, appointment_data):
        """
        Make API call to booking service
        
        Args:
            appointment_data: Dictionary containing appointment details
            
        Returns:
            dict: API response with status and any error information
        """
        try:
            # This is where you would make the actual API call
            # For demonstration, simulating API response
            
            # In a real implementation, you would:
            # 1. Format the data as expected by your API
            # 2. Make the API request using aiohttp or similar
            # 3. Parse and return the response
            
            # Simulate API call (replace with actual implementation)
            import random
            
            # Simulate different response types for demonstration
            response_type = random.choice(["success", "doctor_not_available", "time_not_available"])
            
            if response_type == "success":
                return {
                    "status": "success",
                    "appointment_id": f"APT-{random.randint(10000, 99999)}",
                    "message": "Appointment booked successfully"
                }
            elif response_type == "doctor_not_available":
                return {
                    "status": "error",
                    "error_code": "DOCTOR_NOT_AVAILABLE",
                    "message": "The doctor is not available at this time"
                }
            else:
                return {
                    "status": "error",
                    "error_code": "TIME_NOT_AVAILABLE",
                    "message": "The requested time slot is not available"
                }
                
        except Exception as e:
            print(f"Error calling booking API: {str(e)}")
            return {
                "status": "error",
                "error_code": "API_ERROR",
                "message": "There was a problem connecting to the booking service"
            }