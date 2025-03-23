from utils.config import agent_prompt
import os
from openai import AsyncAzureOpenAI
from utils.custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)
import datetime

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
    physician_matches = None
    visit_type = None
    
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
        )

    async def draft_begin_message(self):
        url = "https://ep.soaper.ai/api/v1/agent/appointments/physicians"
        headers = {
            "Content-Type": "application/json",
            "X-Agent-API-Key": "sk-int-agent-PJNvT3BlbkFJe8ykcJe6kV1KQntXzgMW"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                physicians = [physician['last_name'] for physician in (await response.json())['items']]

        physicians = ['Doctor ' + physician for physician in physicians]
        if len(physicians) > 2:
            begin_sentence = ', '.join(physicians[:-1]) + ', and ' + physicians[-1]
        else:
            begin_sentence = ' and '.join(physicians)

        begin_sentence = f"Hello, thank you for calling, you have reached the office of {begin_sentence}. If you're an existing patient, please use our mobile app for additional assistance. Would you like to schedule an appointment?"

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
    
    async def prepare_functions(self):
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "step1_collect_patient_and_doctor_info",
                    "description": "Step 1: Collect patient and doctor information for booking an appointment. First ask for the patient's first and last name, after getting that, ask for the date of birth, and finally the physician's name. MAKE SURE to tell the user TO wait a moment verifying their information before calling the function.",
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
                            "physician_name": {
                                "type": "string",
                                "description": "Name of the physician (can be first name, last name, or full name). Remove Dr. or doctor or anything else from the name if it is present. Ask the user for the name if they don't provide it."
                            }
                        },
                        "required": ["patient_first_name", "patient_last_name", "date_of_birth", "physician_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "select_physician_from_matches",
                    "description": "Select a physician from multiple matches based on user choice.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selection": {
                                "type": "string",
                                "description": "The selection number or doctor name chosen by the user"
                            }
                        },
                        "required": ["selection"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "step2_find_available_slots",
                    "description": f"""Step 2: Find available appointment slots for a doctor on a specific date. Make sure to tell that you will need to wait a moment while I check for available appointments. Ask the user for the date if they dont provide it. Convert the date into YYYY-MM-DD format.
                    If they say, first half of the month or first or third week of the month, then convert that into YYYY-MM-DD format. Today's date is {datetime.datetime.now().strftime("%Y-%m-%d")}.
                    """,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "appointment_date": {
                                "type": "string",
                                "description": "Desired appointment date."
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
                    "description": "Step 3: Book an appointment using the selected time slot from the previous step. Once that is done, ask the user if they would like book the appointment at that time.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "slot_selection": {
                                "type": "string",
                                "description": "The selected time slot"
                            },
                        },
                        "required": ["slot_selection"]
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
            "available_slots": LLMClient.available_slots,
            "physician_matches": LLMClient.physician_matches,
            "visit_type": LLMClient.visit_type
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
        if "physician_matches" in state:
            LLMClient.physician_matches = state["physician_matches"]

    # Simplified method to append to conversation
    def append_to_conversation(self, request, role, name, content):
        # Just log for now, since we're not tracking conversation history
        print(f"[{role}] {name}: {content}")

    # API methods
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
                            "patient_id": response_data.get("patient", {}).get("id"),
                            "is_new_patient": response_data.get("is_new_patient")
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
    
    async def get_physician_by_name(self, physician_name):
        """
        Make API call to get a physician by name, handling partial matches
        and disambiguation when needed.
        
        physician_name can be first name, last name, or full name.
        Returns the physician ID or prompts for disambiguation if needed.
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
                    physicians = response_data.get("items", [])
                    
                    # No physicians found
                    if not physicians:
                        return {
                            "status": "error",
                            "message": "No physicians found in our system."
                        }
                    
                    # Split the provided name to handle various input formats
                    name_parts = physician_name.strip().split()
                    
                    # Handle cases where only one name part is provided (first or last)
                    if len(name_parts) == 1:
                        single_name = name_parts[0].lower()
                        matches = []
                        
                        for physician in physicians:
                            if (single_name in physician.get("first_name", "").lower() or 
                                single_name in physician.get("last_name", "").lower()):
                                matches.append(physician)
                        
                        # Only one match found
                        if len(matches) == 1:
                            physician = matches[0]
                            return {
                                "status": "success",
                                "physician_id": physician.get("id"),
                                "physician_fname": physician.get("first_name"),
                                "physician_lname": physician.get("last_name")
                            }
                        
                        # Multiple matches, need disambiguation
                        elif len(matches) > 1:
                            match_descriptions = []
                            for i, p in enumerate(matches[:5], 1):  # Limit to 5 matches
                                specialty = p.get("specialty", "General Practitioner")
                                match_descriptions.append({
                                    "index": i,
                                    "id": p.get("id"),
                                    "name": f"Dr. {p.get('first_name')} {p.get('last_name')}",
                                    "specialty": specialty
                                })
                            
                            return {
                                "status": "disambiguation_required",
                                "message": f"We found multiple doctors matching '{physician_name}'.",
                                "matches": match_descriptions
                            }
                        
                        # No matches
                        else:
                            return {
                                "status": "error",
                                "message": f"No physicians found matching '{physician_name}'."
                            }
                    
                    # Full name provided (first and last or more)
                    else:
                        # Try exact match first with first and last name
                        first_name = name_parts[0]
                        last_name = name_parts[-1]
                        
                        for physician in physicians:
                            if (physician.get("first_name", "").lower() == first_name.lower() and 
                                physician.get("last_name", "").lower() == last_name.lower()):
                                return {
                                    "status": "success",
                                    "physician_id": physician.get("id"),
                                    "physician_fname": physician.get("first_name"),
                                    "physician_lname": physician.get("last_name")
                                }
                        
                        # Try partial match on first and last name
                        matches = []
                        for physician in physicians:
                            if (first_name.lower() in physician.get("first_name", "").lower() and 
                                last_name.lower() in physician.get("last_name", "").lower()):
                                matches.append(physician)
                        
                        if len(matches) == 1:
                            physician = matches[0]
                            return {
                                "status": "success",
                                "physician_id": physician.get("id"),
                                "physician_fname": physician.get("first_name"),
                                "physician_lname": physician.get("last_name")
                            }
                        
                        # Try matching just the last name if that fails
                        if not matches:
                            for physician in physicians:
                                if last_name.lower() in physician.get("last_name", "").lower():
                                    matches.append(physician)
                        
                        # Handle multiple matches or no matches
                        if len(matches) > 1:
                            match_descriptions = []
                            for i, p in enumerate(matches[:5], 1):
                                specialty = p.get("specialty", "General Practitioner")
                                match_descriptions.append({
                                    "index": i,
                                    "id": p.get("id"),
                                    "name": f"Dr. {p.get('first_name')} {p.get('last_name')}",
                                    "specialty": specialty
                                })
                            
                            return {
                                "status": "disambiguation_required",
                                "message": f"We found multiple doctors matching '{physician_name}'.",
                                "matches": match_descriptions
                            }
                        
                        elif len(matches) == 1:
                            physician = matches[0]
                            return {
                                "status": "success",
                                "physician_id": physician.get("id"),
                                "physician_fname": physician.get("first_name"),
                                "physician_lname": physician.get("last_name")
                            }
                        
                        else:
                            return {
                                "status": "error",
                                "message": f"No physicians found matching '{physician_name}'."
                            }

        except Exception as e:
            print(f"Error calling physician API: {str(e)}")
            return {
                "status": "error",
                "message": f"There was a problem connecting to the physician service: {str(e)}"
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
        """
        Make API call to book an appointment asynchronously.

        Args:
            appointment_data (dict): Appointment details.

        Returns:
            dict: Response containing booking status and appointment details.
        """
        agent_api_key = "sk-int-agent-PJNvT3BlbkFJe8ykcJe6kV1KQntXzgMW"
        base_url = "https://ep.soaper.ai/api/v1/agent/appointments/schedule"
        
        print(f"Booking appointment: {appointment_data}")

        headers = {
            "X-Agent-API-Key": agent_api_key,
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(base_url, json=appointment_data, headers=headers) as response:
                    # Handle HTTP errors
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"API Error: {response.status} - {error_text}")
                        return {
                            "status": "error",
                            "error_code": f"HTTP_{response.status}",
                            "message": "API request failed",
                            "details": error_text
                        }

                    # Parse JSON response safely
                    try:
                        response_data = await response.json()
                    except aiohttp.ContentTypeError:
                        raw_text = await response.text()
                        print(f"Invalid JSON response: {raw_text}")
                        return {
                            "status": "error",
                            "error_code": "INVALID_JSON",
                            "message": "Received invalid JSON from API",
                            "raw_response": raw_text
                        }

                    print(f"Booking response: {response_data}")

                    if response_data.get("success", False):
                        return {
                            "status": "success",
                            "message": response_data.get("message"),
                            "appointment_id": response_data.get("appointment_id"),
                            "datetime": response_data.get("datetime"),
                            "physician_name": response_data.get("physician_name"),
                            "visit_type": response_data.get("visit_type")
                        }
                    else:
                        return {
                            "status": "error",
                            "error_code": "BOOKING_FAILED",
                            "message": response_data.get("detail", "Error booking appointment")
                        }

            except Exception as e:
                print(f"Error calling booking API: {e}")
                return {
                    "status": "error",
                    "error_code": "API_ERROR",
                    "message": f"Connection issue with booking service: {str(e)}"
                }
                    
    async def draft_response(self, request: ResponseRequiredRequest):
        prompt = self.prepare_prompt(request)
        print(f"Sending prompt with {len(prompt)} messages")
        
        # Get current state
        conversation_state = self.get_conversation_state(request)
        
        try:
            # Create the streaming request
            functions = await self.prepare_functions()
            stream = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=prompt,
                stream=True,
                tools=functions,
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
                        physician_name = func_args.get("physician_name")
                        
                        # Step 1a: Verify or create patient
                        patient_data = {
                            "first_name": patient_first_name,
                            "last_name": patient_last_name,
                            "date_of_birth": date_of_birth
                        }
                        
                        patient_result = await self.verify_or_create_patient(patient_data)
                        print(f"Patient verification result: {patient_result}")
                        print(f"Patient result status: {patient_result.get('status')}")
                        
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
                        LLMClient.visit_type = "New Patient Consultation" if patient_result.get("is_new_patient") else "Follow-up Visit"
                        
                        # Step 1b: Get physician info with flexible name matching
                        physician_result = await self.get_physician_by_name(physician_name)
                        print(f"Physician lookup result: {physician_result}")
                        
                        if physician_result.get("status") == "success":
                            # Store physician info and continue
                            LLMClient.physician_id = physician_result.get("physician_id")
                            LLMClient.physician_name = f"Dr. {physician_result.get('physician_fname')} {physician_result.get('physician_lname')}"
                            
                            # After successful verification, ask for appointment date
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"Thank you, {patient_first_name}. I've verified your information and found {LLMClient.physician_name} in our system. Let's proceed now to find a date for your appointment. When would you like to schedule the appointment?",
                                content_complete=True,
                                end_call=False,
                            )
                        
                        elif physician_result.get("status") == "disambiguation_required":
                            # We need to clarify which doctor the patient wants
                            matches = physician_result.get("matches", [])
                            match_text = "\n".join([f"{m['index']}. {m['name']} - {m['specialty']}" for m in matches])
                            
                            # Create a new temporary state to store matches for the next interaction
                            LLMClient.physician_matches = matches
                            
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"Thank you, {LLMClient.patient_name}. I found multiple doctors matching '{physician_name}'. Could you please specify which one you'd like to see?\n\n{match_text}",
                                content_complete=True,
                                end_call=False,
                            )
                        
                        else:
                            # Error finding the physician
                            error_message = physician_result.get("message", "I couldn't find that doctor in our system")
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"Thank you, {patient_first_name}. I've verified your information, but {error_message}. Could you please check the spelling or provide a different doctor's name?",
                                content_complete=True,
                                end_call=False,
                            )
                    
                    # Handle physician selection from disambiguation
                    elif func_call["func_name"] == "select_physician_from_matches":
                        selection = func_args.get("selection")
                        
                        # Check if we have matches stored
                        if not LLMClient.physician_matches:
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content="I'm sorry, but I don't have any doctor matches to select from. Let's start over. Could you provide your information and the doctor you'd like to see?",
                                content_complete=True,
                                end_call=False,
                            )
                            return
                        
                        # Handle selection by number
                        if selection.isdigit():
                            index = int(selection)
                            matched_doctor = None
                            for doctor in LLMClient.physician_matches:
                                if doctor["index"] == index:
                                    matched_doctor = doctor
                                    break
                            
                            if matched_doctor:
                                # Store physician info
                                LLMClient.physician_id = matched_doctor["id"]
                                LLMClient.physician_name = matched_doctor["name"]
                                
                                # Clean up the matches
                                LLMClient.physician_matches = None
                                
                                # Proceed to date selection
                                yield ResponseResponse(
                                    response_id=request.response_id,
                                    content=f"Great! You've selected {LLMClient.physician_name}. Let's proceed now to find a date for your appointment.",
                                    content_complete=True,
                                    end_call=False,
                                )
                            else:
                                yield ResponseResponse(
                                    response_id=request.response_id,
                                    content=f"I'm sorry, but I couldn't find a doctor matching '{selection}'. Please choose one of the doctors from the list I provided.",
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
                        
                        # Only take the 1st and 5th slots if available
                        filtered_slots = []
                        if len(slots) >= 1:
                            filtered_slots.append(slots[0])  # Add 1st slot
                        if len(slots) >= 5:
                            filtered_slots.append(slots[4])  # Add 5th slot
                        
                        # Format time slots for display in a more conversational way
                        time_options = []
                        for i, slot in enumerate(filtered_slots, 1):  # Only use filtered slots
                            # Extract datetime and format it
                            slot_datetime = slot.get("datetime")
                            slot_time = slot_datetime.split("T")[1][:5]  # Extract time part HH:MM
                            
                            # Convert to AM/PM format
                            hour = int(slot_time.split(":")[0])
                            minute = slot_time.split(":")[1]
                            am_pm = "AM" if hour < 12 else "PM"
                            display_hour = hour if hour <= 12 else hour - 12
                            if display_hour == 0:
                                display_hour = 12
                            
                            time_display = f"{display_hour}:{minute} {am_pm}"
                            time_options.append(time_display)
                        
                        # Store available slots with their indices
                        LLMClient.available_slots = [
                            {
                                "index": i, 
                                "time": slot.get("datetime").split("T")[1][:5],
                                "datetime": slot.get("datetime")
                            }
                            for i, slot in enumerate(filtered_slots, 1)
                        ]
                        
                        # Present options to user in a conversational way
                        if len(time_options) == 1:
                            slot_text = f"I have one opening at {time_options[0]}"
                        elif len(time_options) == 2:
                            slot_text = f"I have openings at {time_options[0]} and {time_options[1]}"
                        else:
                            # This case won't happen with our current filtering, but keeping for robustness
                            last_option = time_options.pop()
                            slot_text = f"I have openings at {', '.join(time_options)}, and {last_option}"
                        
                        yield ResponseResponse(
                            response_id=request.response_id,
                            content=f"Great! For {LLMClient.physician_name} on {appointment_date}, {slot_text}. Which time works best for you?",
                            content_complete=True,
                            end_call=False,
                        )
                    
                    # STEP 3: Book appointment
                    elif func_call["func_name"] == "step3_book_appointment":
                        # Extract slot selection and visit type
                        slot_selection = func_args.get("slot_selection")
                        
                        # Ensure we have required info from previous steps
                        if not all([LLMClient.patient_id, LLMClient.physician_id, LLMClient.selected_date]):
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content="I need to collect more information before booking your appointment. Let's start over. Could you please provide your name, date of birth, and the doctor you'd like to see?",
                                content_complete=True,
                                end_call=False,
                            )
                            return
                        
                        # Convert slot selection to datetime
                        selected_datetime = None
                        
                        # If user provided a slot number
                        if slot_selection.isdigit() and 1 <= int(slot_selection) <= len(LLMClient.available_slots):
                            slot_index = int(slot_selection)
                            for slot in LLMClient.available_slots:
                                if slot.get("index") == slot_index:
                                    selected_datetime = slot.get("datetime")
                                    break
                        
                        # If user provided a time (HH:MM)
                        else:
                            # Try to match with available times
                            entered_time = slot_selection.strip()
                            # Handle various time formats (10:30, 10:30am, 10:30 am, etc.)
                            entered_time = ''.join(c for c in entered_time if c.isdigit() or c == ':').strip()
                            
                            for slot in LLMClient.available_slots:
                                if entered_time in slot.get("time"):
                                    selected_datetime = slot.get("datetime")
                                    break
                        
                        if not selected_datetime:
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"I'm sorry, but I couldn't find a time slot matching '{slot_selection}'. Please choose one of the time slots from the list I provided.",
                                content_complete=True,
                                end_call=False,
                            )
                            return
                        
                        # Book the appointment
                        booking_data = {
                            "patient_id": LLMClient.patient_id,
                            "physician_id": LLMClient.physician_id,
                            "datetime": selected_datetime,
                            "visit_type": LLMClient.visit_type,
                            "visit_notes": "Test scheduling via function call",
                            "duration_minutes": "60"
                        }

                        print(f"Booking data in step 3: {booking_data}")
                        
                        booking_result = await self.book_appointment(booking_data)
                        
                        if booking_result.get("status") == "success":
                            # Format the time for display
                            time_part = selected_datetime.split("T")[1][:5]
                            hour = int(time_part.split(":")[0])
                            minute = time_part.split(":")[1]
                            am_pm = "AM" if hour < 12 else "PM"
                            display_hour = hour if hour <= 12 else hour - 12
                            if display_hour == 0:
                                display_hour = 12
                            
                            formatted_time = f"{display_hour}:{minute} {am_pm}"
                            
                            # Booking successful
                            yield ResponseResponse(
                                response_id=request.response_id,
                                content=f"Great news! I've booked your appointment with {LLMClient.physician_name} on {LLMClient.selected_date} at {formatted_time}. Is there anything else I can help you with?",
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