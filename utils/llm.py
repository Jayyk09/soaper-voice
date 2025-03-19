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
    # Class-level attributes for state management
    patient_id = None
    patient_name = None
    physician_id = None
    physician_name = None
    selected_date = None
    available_slots = []
    
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
    
    def prepare_functions(self):
        # Redesigned functions with clear step prefixes
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "step1_collect_patient_and_doctor_info",
                    "description": "Step 1: Collect patient and doctor information for booking an appointment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_first_name": {
                                "type": "string",
                                "description": "Patient's first name."
                            },
                            "patient_last_name": {
                                "type": "string",
                                "description": "Patient's last name"
                            },
                            "date_of_birth": {
                                "type": "string",
                                "description": "Patient's date of birth in YYYY-MM-DD format"
                            },
                            "physician_first_name": {
                                "type": "string",
                                "description": "First name of the physician"
                            },
                            "physician_last_name": {
                                "type": "string",
                                "description": "Last name of the physician"
                            }
                        },
                        "required": ["patient_first_name", "patient_last_name", "date_of_birth", 
                                    "physician_first_name", "physician_last_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "step2_find_available_slots",
                    "description": "Step 2: Find available appointment slots for a doctor on a specific date.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "appointment_date": {
                                "type": "string",
                                "description": "Desired appointment date in YYYY-MM-DD format. The year is 2025."
                            }
                        },
                        "required": ["appointment_date"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "step3_book_appointment",
                    "description": "Step 3: Book an appointment using the selected time slot and visit type.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "slot_selection": {
                                "type": "string",
                                "description": "The selected time slot (can be slot number or time in HH:MM format)"
                            },
                            "visit_type": {
                                "type": "string",
                                "description": "Type of visit ('New Patient Consultation', 'Standard Office Visit', 'Virtual Visit', 'Follow-up Visit', 'Annual Physical', 'Injection/Vaccination', 'Lab Draw')"
                            }
                        },
                        "required": ["slot_selection", "visit_type"]
                    }
                }
            }
        ]
        return functions

    # Simplified method to get current conversation state
    def get_conversation_state(self, request):
        return {
            "patient_id": LLMClient.patient_id,
            "patient_name": LLMClient.patient_name,
            "physician_id": LLMClient.physician_id,
            "physician_name": LLMClient.physician_name,
            "selected_date": LLMClient.selected_date,
            "available_slots": LLMClient.available_slots
        }

    # Simplified method to save conversation state
    def save_conversation_state(self, request, state):
        if "patient_id" in state:
            LLMClient.patient_id = state["patient_id"]
        if "patient_name" in state:
            LLMClient.patient_name = state["patient_name"]
        if "physician_id" in state:
            LLMClient.physician_id = state["physician_id"]
        if "physician_name" in state:
            LLMClient.physician_name = state["physician_name"]
        if "selected_date" in state:
            LLMClient.selected_date = state["selected_date"]
        if "available_slots" in state:
            LLMClient.available_slots = state["available_slots"]

    # Simplified method to append to conversation
    def append_to_conversation(self, request, role, name, content):
        # Just log for now, since we're not tracking conversation history
        print(f"[{role}] {name}: {content}")

    async def draft_response(self, request: ResponseRequiredRequest):
        prompt = self.prepare_prompt(request)
        print(f"Sending prompt with {len(prompt)} messages")
        
        # Get current state
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
                    
                    # STEP 1: Collect patient and doctor info
                    if func_call["func_name"] == "step1_collect_patient_and_doctor_info":
                        # Extract patient and doctor info
                        patient_first_name = func_args.get("patient_first_name")
                        patient_last_name = func_args.get("patient_last_name")
                        date_of_birth = func_args.get("date_of_birth")
                        physician_first_name = func_args.get("physician_first_name")
                        physician_last_name = func_args.get("physician_last_name")
                        
                        # Step 1a: Verify or create patient
                        patient_data = {
                            "first_name": patient_first_name,
                            "last_name": patient_last_name,
                            "date_of_birth": date_of_birth
                        }
                        
                        patient_result = await self.verify_or_create_patient(patient_data)
                        print(f"Patient verification result: {patient_result}")
                        
                        if patient_result.get("status") != "success":
                            error_message = patient_result.get("message", "There was an error verifying your information")
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"I'm sorry, but {error_message}. Can we try again with your information?",
                                content_complete=True,
                                end_call=False,
                            )
                            return
                        
                        # Store patient info
                        LLMClient.patient_id = patient_result.get("patient_id")
                        LLMClient.patient_name = f"{patient_first_name} {patient_last_name}"
                        
                        # Step 1b: Get physician ID
                        physician_result = await self.get_physician_id_by_name(physician_first_name, physician_last_name)
                        print(f"Physician ID result: {physician_result}")
                        
                        if physician_result.get("status") != "success":
                            error_message = physician_result.get("message", "I couldn't find that doctor in our system")
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"Thank you, {LLMClient.patient_name}. I've verified your information, but {error_message}. Could you please check the spelling or provide a different doctor's name?",
                                content_complete=True,
                                end_call=False,
                            )
                            return
                        
                        # Store physician info
                        LLMClient.physician_id = physician_result.get("physician_id")
                        LLMClient.physician_name = f"Dr. {physician_first_name} {physician_last_name}"
                        
                        # After successful verification, ask for appointment date
                        yield ResponseResponse(
                            response_id=request.response_id,
                            content=f"Thank you, {LLMClient.patient_name}. I've verified your information and found Dr. {physician_first_name} {physician_last_name} in our system. What date would you prefer for your appointment? Please provide the date in YYYY-MM-DD format.",
                            content_complete=True,
                            end_call=False,
                        )
                    
                    # STEP 2: Find available slots
                    elif func_call["func_name"] == "step2_find_available_slots":
                        # Extract appointment date
                        appointment_date = func_args.get("appointment_date")
                        
                        # Ensure we have patient and physician info from step 1
                        if not LLMClient.patient_id or not LLMClient.physician_id:
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content="I need to collect your information and your doctor's information first. Could you please provide your full name, date of birth, and the name of the doctor you'd like to see?",
                                content_complete=True,
                                end_call=False,
                            )
                            return
                        
                        # Get available slots
                        slots_data = {
                            "patient_id": LLMClient.patient_id,
                            "physician_id": LLMClient.physician_id,
                            "date": appointment_date
                        }
                        
                        slots_result = await self.get_doctor_time_slots(slots_data)
                        print(f"Time slots result: {slots_result}")
                        
                        if not slots_result.get("success") or not slots_result.get("slots"):
                            message = slots_result.get("message", "No available appointments found for this date")
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"I'm sorry, but {message}. Would you like to try a different date?",
                                content_complete=True,
                                end_call=False,
                            )
                            return
                        
                        # Store date and available slots
                        LLMClient.selected_date = appointment_date
                        slots = slots_result.get("slots", [])
                        
                        # Format time slots for display
                        slot_options = []
                        for i, slot in enumerate(slots[:5], 1):  # Limit to first 5 slots
                            # Extract datetime and format it
                            slot_datetime = slot.get("datetime")
                            slot_time = slot_datetime.split("T")[1]  # Extract time part
                            slot_options.append(f"{i}. {slot_time}")
                            
                        # Store available slots with their indices
                        LLMClient.available_slots = [
                            {"index": i, "time": slot.get("datetime").split("T")[1]} 
                            for i, slot in enumerate(slots[:5], 1)
                        ]
                        
                        # Present options to user
                        slot_text = "\n".join(slot_options)
                        
                        yield ResponseResponse(
                            response_id=request.response_id,
                            content=f"I found the following available time slots for {LLMClient.physician_name} on {appointment_date}:\n\n{slot_text}\n\nPlease choose a time slot by number or specify the exact time, and tell me what type of visit you need (such as 'New Patient Consultation', 'Follow-up Visit', 'Annual Physical', etc.).",
                            content_complete=True,
                            end_call=False,
                        )
                    
                    # STEP 3: Book appointment
                    elif func_call["func_name"] == "step3_book_appointment":
                        # Extract slot selection and visit type
                        slot_selection = func_args.get("slot_selection")
                        visit_type = func_args.get("visit_type")
                        
                        # Ensure we have required info from previous steps
                        if not all([LLMClient.patient_id, LLMClient.physician_id, LLMClient.selected_date, LLMClient.available_slots]):
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content="I need to collect more information before booking your appointment. Let's start over. Could you please provide your name, date of birth, and the doctor you'd like to see?",
                                content_complete=True,
                                end_call=False,
                            )
                            return
                        
                        # Convert slot selection to time
                        time = slot_selection
                        if slot_selection.isdigit() and 1 <= int(slot_selection) <= len(LLMClient.available_slots):
                            slot_index = int(slot_selection)
                            for slot in LLMClient.available_slots:
                                if slot.get("index") == slot_index:
                                    time = slot.get("time")
                                    break
                        
                        # Book the appointment
                        booking_data = {
                            "patient_id": LLMClient.patient_id,
                            "physician_id": LLMClient.physician_id,
                            "date": LLMClient.selected_date,
                            "time": time,
                            "visit_type": visit_type
                        }
                        
                        booking_result = await self.book_appointment(booking_data)
                        
                        if booking_result.get("status") == "success":
                            # Booking successful
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"Great news, {LLMClient.patient_name}! I've booked your {visit_type} appointment with {LLMClient.physician_name} on {LLMClient.selected_date} at {time}. Your confirmation number is {booking_result.get('appointment_id')}. Is there anything else I can help you with?",
                                content_complete=True,
                                end_call=False,
                            )
                            
                            # Clear state after successful booking
                            LLMClient.patient_id = None
                            LLMClient.patient_name = None
                            LLMClient.physician_id = None
                            LLMClient.physician_name = None
                            LLMClient.selected_date = None
                            LLMClient.available_slots = []
                            
                        else:
                            # Handle booking error
                            error_message = booking_result.get("message", "There was an error booking your appointment")
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

    # API functions remain unchanged
    async def verify_or_create_patient(self, patient_data):
        """Make API call to patient verification service"""
        url = "https://ep.soaper.ai/api/v1/agent/patients/create"
        headers = {
            "Content-Type": "application/json",
            "X-Agent-API-Key": "sk-int-agent-PJNvT3BlbkFJe8ykcJe6kV1KQntXzgMW"
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
        """Make API call to get a physician by first name and last name"""
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
        """Make API call to get next available appointment slots for an agent"""
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
        """Make API call to book an appointment"""
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