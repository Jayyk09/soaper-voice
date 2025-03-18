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
import aiohttp
load_dotenv()

class LLMClient:
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
        )

    patient_id = None
    physician_id = None
    available_slots = []

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
    
    def prepare_functions(self):
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "verify_or_create_patient",
                    "description": "When the user asks to book an appointment, first verify if a patient exists in the system or create a new patient record if they don't exist. Always call this function first in the booking workflow.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "first_name": {
                                "type": "string",
                                "description": "Patient's first name.",
                            },
                            "last_name": {
                                "type": "string",
                                "description": "Patient's last name",
                            },
                            "date_of_birth": {
                                "type": "string",
                                "description": "Patient's date of birth in YYYY-MM-DD format"
                            },
                        },
                        "required": ["first_name", "last_name", "date_of_birth"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_physician_id_by_name",
                    "description": "Get physician ID by first name and last name. Call this function after verifying/creating the patient.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "physician_first_name": {
                                "type": "string",
                                "description": "First name of the physician",
                            },
                            "physician_last_name": {
                                "type": "string",
                                "description": "Last name of the physician",
                            },
                        },
                        "required": ["physician_first_name", "physician_last_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_doctor_time_slots",
                    "description": "Get available appointment slots for a doctor. Call this after getting the physician ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "physician_id": {
                                "type": "string",
                                "description": "Physician ID obtained from get_physician_id_by_name function",
                            },
                            "date": {
                                "type": "string",
                                "description": "Date to check for available slots in YYYY-MM-DD format",
                            },
                        },
                        "required": ["physician_id", "date"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "book_appointment",
                    "description": "Book an appointment with a doctor after collecting all required information. Call this last after getting patient ID, physician ID, and selecting a time slot.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "string",
                                "description": "Patient ID obtained from verify_or_create_patient function",
                            },
                            "physician_id": {
                                "type": "string",
                                "description": "Physician ID obtained from get_physician_id_by_name function",
                            },
                            "date": {
                                "type": "string",
                                "description": "Appointment date in YYYY-MM-DD format",
                            },
                            "time": {
                                "type": "string",
                                "description": "Appointment time in HH:MM format (24-hour)",
                            },
                            "visit_type": {
                                "type": "string",
                                "description": "Type of visit (e.g. 'new patient', 'follow-up', 'annual checkup')",
                            },
                        },
                        "required": ["patient_id", "physician_id", "date", "time", "visit_type"],
                    },
                },
            },
        ]
        return functions

    async def draft_response(self, request: ResponseRequiredRequest):
        prompt = self.prepare_prompt(request)
        print(f"Sending prompt with {len(prompt)} messages")
        
        # Track conversation state between function calls
        conversation_state = self.get_conversation_state(request)
        
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
            func_call = {}
            func_arguments = ""
            
            async for chunk in stream:
                # Skip chunks with empty choices
                if not chunk.choices:
                    continue

                # Process function calling chunks
                if chunk.choices[0].delta.tool_calls:
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
            if func_call:
                try:
                    print("Parsing function arguments...")
                    func_args = json.loads(func_arguments)
                    print(f"Parsed arguments: {func_args}")
                    
                    # Handle different function calls in the sequence
                    if func_call["func_name"] == "verify_or_create_patient":
                        # Handle patient verification/creation
                        api_result = await self.verify_or_create_patient(func_args)
                        print(f"Patient verification result: {api_result}")
                        
                        if api_result.get("status") == "success":
                            # Store patient ID for future use
                            patient_id = api_result.get("patient_id")
                            patient_name = f"{func_args.get('first_name')} {func_args.get('last_name')}"
                            
                            # Update conversation state
                            conversation_state["patient_id"] = patient_id
                            conversation_state["patient_name"] = patient_name
                            self.save_conversation_state(request, conversation_state)
                            
                            # Send confirmation and ask about doctor
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"Thank you, {patient_name}. I've verified your information. Now, which doctor would you like to see? Please provide their first and last name.",
                                content_complete=True,
                                end_call=False,
                            )
                            
                            # Add result to conversation context
                            self.append_to_conversation(request, "function", "verify_or_create_patient", json.dumps({
                                "status": "success",
                                "patient_id": patient_id,
                                "patient_name": patient_name
                            }))
                            
                        else:
                            # Handle error
                            error_message = api_result.get("message", "There was an error verifying your information")
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"I'm sorry, but {error_message}. Can we try again with your information?",
                                content_complete=True,
                                end_call=False,
                            )
                            
                    elif func_call["func_name"] == "get_physician_id_by_name":
                        # Get physician ID
                        physician_first_name = func_args.get("physician_first_name")
                        physician_last_name = func_args.get("physician_last_name")
                        api_result = await self.get_physician_id_by_name(physician_first_name, physician_last_name)
                        print(f"Physician ID result: {api_result}")
                        
                        if api_result.get("status") == "success":
                            # Store physician information
                            physician_id = api_result.get("physician_id")
                            physician_name = f"Dr. {physician_first_name} {physician_last_name}"
                            
                            # Update conversation state
                            conversation_state["physician_id"] = physician_id
                            conversation_state["physician_name"] = physician_name
                            self.save_conversation_state(request, conversation_state)
                            
                            # Ask for preferred date
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"Great! You'd like to see {physician_name}. What date would you prefer for your appointment? Please provide the date in YYYY-MM-DD format.",
                                content_complete=True,
                                end_call=False,
                            )
                            
                            # Add result to conversation context
                            self.append_to_conversation(request, "function", "get_physician_id_by_name", json.dumps({
                                "status": "success",
                                "physician_id": physician_id,
                                "physician_name": physician_name
                            }))
                            
                        else:
                            # Handle error
                            error_message = api_result.get("message", "I couldn't find that doctor in our system")
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"I'm sorry, but {error_message}. Could you please check the spelling or provide a different doctor's name?",
                                content_complete=True,
                                end_call=False,
                            )
                    
                    elif func_call["func_name"] == "get_doctor_time_slots":
                        # Get available time slots
                        physician_id = func_args.get("physician_id")
                        date = func_args.get("date")
                        
                        # Ensure we have a physician ID from previous steps
                        if not physician_id and conversation_state.get("physician_id"):
                            physician_id = conversation_state.get("physician_id")
                        
                        api_result = await self.get_doctor_time_slots({"physician_id": physician_id, "date": date})
                        print(f"Time slots result: {api_result}")
                        
                        if api_result.get("success") and api_result.get("slots"):
                            # Store date and available slots
                            conversation_state["selected_date"] = date
                            slots = api_result.get("slots", [])
                            
                            # Format time slots for display
                            slot_options = []
                            for i, slot in enumerate(slots[:5], 1):  # Limit to first 5 slots
                                slot_time = slot.get("time")
                                slot_options.append(f"{i}. {slot_time}")
                            
                            # Update conversation state with available slots
                            conversation_state["available_slots"] = slots[:5]
                            self.save_conversation_state(request, conversation_state)
                            
                            # Present options to user
                            physician_name = conversation_state.get("physician_name", "the doctor")
                            slot_text = "\n".join(slot_options)
                            
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"I found the following available time slots for {physician_name} on {date}:\n\n{slot_text}\n\nPlease select a time slot by number, or type the time you prefer (in HH:MM format).",
                                content_complete=True,
                                end_call=False,
                            )
                            
                            # Add result to conversation context
                            self.append_to_conversation(request, "function", "get_doctor_time_slots", json.dumps({
                                "status": "success",
                                "date": date,
                                "slots": [slot.get("time") for slot in slots[:5]]
                            }))
                            
                        else:
                            # Handle no slots available
                            message = api_result.get("message", "No available appointments found for this date")
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"I'm sorry, but {message}. Would you like to try a different date?",
                                content_complete=True,
                                end_call=False,
                            )
                    
                    elif func_call["func_name"] == "book_appointment":
                        # Book the appointment
                        patient_id = func_args.get("patient_id") or conversation_state.get("patient_id")
                        physician_id = func_args.get("physician_id") or conversation_state.get("physician_id")
                        date = func_args.get("date") or conversation_state.get("selected_date")
                        time = func_args.get("time")
                        visit_type = func_args.get("visit_type")
                        
                        # Validate we have all required information
                        if not all([patient_id, physician_id, date, time, visit_type]):
                            missing_info = []
                            if not patient_id:
                                missing_info.append("patient information")
                            if not physician_id:
                                missing_info.append("doctor information")
                            if not date:
                                missing_info.append("appointment date")
                            if not time:
                                missing_info.append("appointment time")
                            if not visit_type:
                                missing_info.append("visit type")
                                
                            missing_text = ", ".join(missing_info)
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"I need a bit more information before booking your appointment. Could you please provide your {missing_text}?",
                                content_complete=True,
                                end_call=False,
                            )
                            return
                        
                        # All information available, make booking API call
                        booking_data = {
                            "patient_id": patient_id,
                            "physician_id": physician_id,
                            "date": date,
                            "time": time,
                            "visit_type": visit_type
                        }
                        
                        # Call booking API (mocked for this example)
                        api_result = {"status": "success", "appointment_id": "APT" + str(random.randint(10000, 99999))}
                        
                        if api_result.get("status") == "success":
                            # Booking successful
                            patient_name = conversation_state.get("patient_name", "")
                            physician_name = conversation_state.get("physician_name", "the doctor")
                            
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"Great news, {patient_name}! I've booked your {visit_type} appointment with {physician_name} on {date} at {time}. Your confirmation number is {api_result.get('appointment_id')}. Is there anything else I can help you with?",
                                content_complete=True,
                                end_call=False,
                            )
                            
                            # Clear conversation state after successful booking
                            self.save_conversation_state(request, {})
                            
                        else:
                            # Handle booking error
                            error_message = api_result.get("message", "There was an error booking your appointment")
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"I'm sorry, but {error_message}. Would you like to try again with a different time or date?",
                                content_complete=True,
                                end_call=False,
                            )
                            
                except json.JSONDecodeError as e:
                    print(f"Error parsing function arguments: {str(e)}")
                    print(f"Raw arguments: {func_arguments}")
                    yield ResponseResponse(
                        response_id=request.response_id,
                        content="I'm sorry, I couldn't process your request correctly. Let's try again. What information can I help you with for your appointment?",
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

    async def verify_or_create_patient(self, patient_data):
        """
        Make API call to patient verification service
        
        Args:
            patient_data (dict): Dictionary containing patient details (first_name, last_name, date_of_birth)
        """
        url = "https://ep.soaper.ai/api/v1/agent/patients/create"  # Update with your actual FastAPI server URL
        headers = {
            "Content-Type": "application/json",
            "X-Agent-API-Key": "sk-int-agent-PJNvT3BlbkFJe8ykcJe6kV1KQntXzgMW"  # Replace with actual API key
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=patient_data, headers=headers) as response:
                    response_data = await response.json()
                    
                    if response_data.get("success", False):
                        return {
                            "status": "success",
                            "message": response_data.get("message"),
                            "patient_id": response_data.get("patient", {}).get("id")
                        }
                    else:
                        return {
                            "status": "error",
                            "message": response_data.get("message", "Error creating patient")
                        }
        
        except Exception as e:
            print(f"Error calling patient creation API: {str(e)}")
            return {
                "status": "error",
                "message": f"There was a problem connecting to the patient creation service: {str(e)}"
            }
    
    async def get_physician_id_by_name(self, physician_first_name, physician_last_name):
        """
        Make API call to get a physician by first name and last name
        """
        url = f"https://ep.soaper.ai/api/v1/agent/appointments/physicians"
        headers = {
            "Content-Type": "application/json",
            "X-Agent-API-Key": "sk-int-agent-PJNvT3BlbkFJe8ykcJe6kV1KQntXzgMW"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    response_data = await response.json()
                    for physician in response_data.get("items", []):
                        if (physician.get("first_name") == physician_first_name and 
                            physician.get("last_name") == physician_last_name):
                            return {
                                "status": "success",
                                "physician_id": physician.get("id"),
                                "physician_fname": physician.get("first_name"),
                                "physician_lname": physician.get("last_name")
                            }
                    return {
                        "status": "error",
                        "message": "Physician not found"
                    }

        except Exception as e:
            print(f"Error calling physician API: {str(e)}")
            return {
                "status": "error",
                "message": f"There was a problem connecting to the physician service: {str(e)}"
            }
        
    async def get_doctor_time_slots(self, appointment_data):
        """
        Make API call to get next available appointment slots for an agent
        """
        url = f"https://ep.soaper.ai/api/v1/agent/appointments/next-available"
        headers = {
            "Content-Type": "application/json",
            "X-Agent-API-Key": "sk-int-agent-PJNvT3BlbkFJe8ykcJe6kV1KQntXzgMW"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=appointment_data, headers=headers) as response:
                    response_data = await response.json()
                    if response_data.get("success", False):
                        return {
                            "success": True,
                            "slots": response_data.get("slots", []),
                            "message": response_data.get("message", "Doctor time slots retrieved successfully")
                        }
                    else:
                        return {
                            "success": False,
                            "slots": [],
                            "message": response_data.get("message", "No available appointments found")
                        }
                    
        except Exception as e:
            print(f"Error calling next available slots API: {str(e)}")
            return {
                "success": False,
                "slots": [],
                "message": f"There was a problem connecting to the next available slots service: {str(e)}"
            }

    async def book_appointment(self, appointment_data):
        """
        Make API call to book an appointment
        
        Args:
            appointment_data (dict): Dictionary containing appointment details
        """
        url = "https://ep.soaper.ai/api/v1/agent/appointments/book"
        headers = {
            "Content-Type": "application/json",
            "X-Agent-API-Key": "sk-int-agent-PJNvT3BlbkFJe8ykcJe6kV1KQntXzgMW"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=appointment_data, headers=headers) as response:
                    response_data = await response.json()
                    
                    if response_data.get("success", False):
                        return {
                            "status": "success",
                            "message": response_data.get("message"),
                            "appointment_id": response_data.get("appointment", {}).get("id")
                        }
                    else:
                        error_code = response_data.get("error_code", "UNKNOWN_ERROR")
                        return {
                            "status": "error",
                            "error_code": error_code,
                            "message": response_data.get("message", "Error booking appointment")
                        }
        
        except Exception as e:
            print(f"Error calling booking API: {str(e)}")
            return {
                "status": "error",
                "error_code": "API_ERROR",
                "message": f"There was a problem connecting to the booking service: {str(e)}"
            }