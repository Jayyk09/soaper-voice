import uuid
import asyncio
import json
import base64
import os
import websockets
from urllib.parse import urlencode, urljoin

from quart import Quart, Response, request, json as quart_json
from logging import INFO
import re

from azure.eventgrid import EventGridEvent, SystemEventNames
from azure.communication.callautomation import (
    PhoneNumberIdentifier,
    CallMediaStreamingOptions,
    StreamingProtocol
)
from azure.communication.callautomation.aio import (
    CallAutomationClient
)
from azure.core.messaging import CloudEvent
from dotenv import load_dotenv

load_dotenv()

# Your ACS resource connection string
ACS_CONNECTION_STRING = os.getenv('ACS_CONNECTION_STRING')

# OpenAI API key and endpoint
OPENAI_API_KEY = os.getenv('OPENAI_REALTIME_API_KEY')
OPENAI_REALTIME_API_ENDPOINT = os.getenv('OPENAI_REALTIME_API_ENDPOINT')

# Callback events URI to handle callback events
CALLBACK_URI_HOST = os.getenv('CALLBACK_URI_HOST_WITH_PROTOCOL')
CALLBACK_EVENTS_URI = CALLBACK_URI_HOST + "/api/callbacks"

# OpenAI System Message - focused on appointment booking
SYSTEM_MESSAGE = """
You are an AI assistant working as a medical receptionist. Your job is to help callers book appointments.

Follow these steps:
1. Greet the caller and ask how you can help them.
2. If they want to book an appointment, collect:
   - Their full name
   - Their phone number 
   - The type of appointment or doctor they need to see
   - Preferred date and time

3. Once you have all the information, confirm the appointment details with the caller.
4. Thank them for calling and let them know they'll receive a confirmation text.

Keep the conversation friendly, professional, and focused on appointment booking.
"""

# Greeting prompt
HELLO_PROMPT = "Hello, thank you for calling our medical center. How can I help you today?"

# OpenAI voice options
VOICE = 'nova'  # Options: alloy, echo, fable, onyx, nova, shimmer

# Initialize call automation client
call_automation_client = CallAutomationClient.from_connection_string(ACS_CONNECTION_STRING)

# WebSocket connections dictionary to maintain active connections
active_connections = {}

app = Quart(__name__)

class StreamingConnection:
    def __init__(self, call_connection_id, caller_id):
        self.call_connection_id = call_connection_id
        self.caller_id = caller_id
        self.openai_ws = None
        self.acs_ws = None
        self.is_active = True
        self.appointment_data = {
            "name": None,
            "phone": None,
            "doctor": None,
            "datetime": None
        }

async def initialize_openai_session(openai_ws):
    """Initialize the OpenAI session with system message and settings"""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "pcm_s16le",  # ACS uses PCM 16-bit little-endian
            "output_audio_format": "pcm_s16le",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.7,
        }
    }
    app.logger.info('Sending OpenAI session update')
    await openai_ws.send(json.dumps(session_update))

    # Have the AI speak first
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": HELLO_PROMPT
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))

async def handle_media_streaming(stream_conn):
    """Handle the bidirectional media streaming between ACS and OpenAI"""
    try:
        # Initialize OpenAI WebSocket connection
        headers = {"api-key": OPENAI_API_KEY} if "api.openai.azure.com" in OPENAI_REALTIME_API_ENDPOINT else {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        
        async with websockets.connect(
            OPENAI_REALTIME_API_ENDPOINT,
            additional_headers=headers,
            open_timeout=30,
            ping_timeout=20,
            close_timeout=20
        ) as openai_ws:
            stream_conn.openai_ws = openai_ws
            await initialize_openai_session(openai_ws)
            
            # Set up tasks for bidirectional communication
            acs_to_openai = asyncio.create_task(forward_acs_to_openai(stream_conn))
            openai_to_acs = asyncio.create_task(forward_openai_to_acs(stream_conn))
            
            # Wait for either task to complete
            done, pending = await asyncio.wait(
                [acs_to_openai, openai_to_acs],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                
    except Exception as e:
        app.logger.error(f"Error in handle_media_streaming: {e}")
    finally:
        # Cleanup
        if stream_conn.call_connection_id in active_connections:
            del active_connections[stream_conn.call_connection_id]
        stream_conn.is_active = False

async def forward_acs_to_openai(stream_conn):
    """Forward audio from ACS to OpenAI"""
    try:
        async for message in stream_conn.acs_ws:
            if not stream_conn.is_active:
                break
                
            # Assume message is binary audio data from ACS
            # Convert ACS audio format to base64 for OpenAI
            audio_payload = base64.b64encode(message).decode('utf-8')
            
            # Append the audio to OpenAI's input buffer
            audio_append = {
                "type": "input_audio_buffer.append",
                "audio": audio_payload
            }
            await stream_conn.openai_ws.send(json.dumps(audio_append))
    except Exception as e:
        app.logger.error(f"Error in forward_acs_to_openai: {e}")

async def extract_appointment_data(text):
    """Extract appointment data from conversation text (for logging purposes)"""
    appointment_data = {}
    
    # Basic pattern matching - in a production system, use more robust NLP
    name_patterns = [
        r"name is (\w+\s\w+)",
        r"this is (\w+\s\w+)",
        r"I'm (\w+\s\w+)"
    ]
    
    phone_patterns = [
        r"phone number is (\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4})",
        r"(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4})"
    ]
    
    # Try to extract name
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            appointment_data["name"] = match.group(1)
            break
    
    # Try to extract phone
    for pattern in phone_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            appointment_data["phone"] = match.group(1)
            break
    
    return appointment_data

async def forward_openai_to_acs(stream_conn):
    """Forward audio from OpenAI to ACS and handle OpenAI events"""
    try:
        async for openai_message in stream_conn.openai_ws:
            if not stream_conn.is_active:
                break
                
            response = json.loads(openai_message)
            
            # Log important events
            if response.get('type') in ['error', 'response.content.done', 'response.done', 
                                       'input_audio_buffer.speech_started', 'session.created']:
                app.logger.info(f"OpenAI event: {response['type']}")
            
            # Handle audio data from OpenAI
            if response.get('type') == 'response.audio.delta' and 'delta' in response:
                # Convert base64 audio to binary for ACS
                audio_binary = base64.b64decode(response['delta'])
                await stream_conn.acs_ws.send(audio_binary)
            
            # Log text content for debugging and potential extraction of appointment details
            if response.get('type') == 'response.content.part' and response.get('content_type') == 'text':
                content_text = response.get('text', '')
                app.logger.info(f"OpenAI response text: {content_text}")
                
                # For logging purposes, try to extract appointment data
                appointment_data = await extract_appointment_data(content_text)
                if appointment_data:
                    app.logger.info(f"Extracted appointment data: {appointment_data}")
                    stream_conn.appointment_data.update(appointment_data)
                
    except Exception as e:
        app.logger.error(f"Error in forward_openai_to_acs: {e}")

async def setup_media_streaming(call_connection_id, caller_id):
    """Set up the media streaming for the call"""
    try:
        # Create a new streaming connection
        connection_client = call_automation_client.get_call_connection(call_connection_id)
        
        # Set up streaming options
        streaming_options = CallMediaStreamingOptions(
            streaming_protocol=StreamingProtocol.WEB_SOCKET,
            audio_enabled=True
        )
        
        # Start media streaming
        media_streaming_result = await connection_client.start_media_streaming(streaming_options)
        websocket_url = media_streaming_result.websocket_url
        
        # Connect to the ACS WebSocket
        acs_ws = await websockets.connect(websocket_url)
        
        # Create a streaming connection object
        stream_conn = StreamingConnection(call_connection_id, caller_id)
        stream_conn.acs_ws = acs_ws
        
        # Store the connection
        active_connections[call_connection_id] = stream_conn
        
        # Start handling media streaming in a background task
        asyncio.create_task(handle_media_streaming(stream_conn))
        
        return True
    except Exception as e:
        app.logger.error(f"Error setting up media streaming: {e}")
        return False

async def answer_call_async(incoming_call_context, callback_url):
    """Answer an incoming call"""
    return await call_automation_client.answer_call(
        incoming_call_context=incoming_call_context,
        callback_url=callback_url)

@app.route("/api/incomingCall", methods=['POST'])
async def incoming_call_handler():
    """Handle incoming call events"""
    try:
        for event_dict in await request.json:
            event = EventGridEvent.from_dict(event_dict)
            app.logger.info(f"Incoming event type: {event.event_type}")
            
            if event.event_type == SystemEventNames.EventGridSubscriptionValidationEventName:
                app.logger.info("Validating subscription")
                validation_code = event.data['validationCode']
                validation_response = {'validationResponse': validation_code}
                return Response(response=quart_json.dumps(validation_response), status=200)
                
            elif event.event_type == "Microsoft.Communication.IncomingCall":
                app.logger.info(f"Incoming call received: {event.data}")
                
                # Extract caller ID
                if event.data['from']['kind'] == "phoneNumber":
                    caller_id = event.data['from']["phoneNumber"]["value"]
                else:
                    caller_id = event.data['from']['rawId']
                app.logger.info(f"Caller ID: {caller_id}")
                
                # Generate callback URL
                incoming_call_context = event.data['incomingCallContext']
                guid = uuid.uuid4()
                query_parameters = urlencode({"callerId": caller_id})
                callback_uri = f"{CALLBACK_EVENTS_URI}/{guid}?{query_parameters}"
                app.logger.info(f"Callback URL: {callback_uri}")
                
                # Answer the call
                answer_call_result = await answer_call_async(incoming_call_context, callback_uri)
                app.logger.info(f"Answered call with connection ID: {answer_call_result.call_connection_id}")
                
                return Response(status=200)
    except Exception as e:
        app.logger.error(f"Error in incoming call handler: {e}")
        return Response(status=500)

@app.route("/api/callbacks/<context_id>", methods=["POST"])
async def handle_callback(context_id):
    """Handle callback events from ACS"""
    try:
        app.logger.info(f"Callback received for context ID: {context_id}")
        caller_id = request.args.get("callerId", "").strip()
        if "+" not in caller_id:
            caller_id = "+" + caller_id.strip()
            
        for event_dict in await request.json:
            event = CloudEvent.from_dict(event_dict)
            app.logger.info(f"Event type: {event.type} for call connection ID: {event.data['callConnectionId']}")
            call_connection_id = event.data['callConnectionId']
            
            # Handle call connected event - start media streaming
            if event.type == "Microsoft.Communication.CallConnected":
                app.logger.info(f"Call connected: {event.data}")
                # Set up media streaming for the call
                streaming_setup = await setup_media_streaming(call_connection_id, caller_id)
                if not streaming_setup:
                    app.logger.error("Failed to set up media streaming")
            
            # Handle call disconnected event
            elif event.type == "Microsoft.Communication.CallDisconnected":
                app.logger.info(f"Call disconnected: {event.data}")
                # Log any appointment data collected (in a real system, you'd save this to a database)
                if call_connection_id in active_connections:
                    stream_conn = active_connections[call_connection_id]
                    app.logger.info(f"Appointment data collected: {stream_conn.appointment_data}")
                    
                    # Clean up the connection
                    stream_conn.is_active = False
                    del active_connections[call_connection_id]
                    
        return Response(status=200)
    except Exception as e:
        app.logger.error(f"Error in handle_callback: {e}")
        return Response(status=500)

@app.route("/api/appointments", methods=["GET"])
async def get_appointments():
    """API endpoint to retrieve current appointments (for demonstration purposes)"""
    appointments = []
    for conn_id, conn in active_connections.items():
        if any(conn.appointment_data.values()):
            appointments.append({
                "call_id": conn_id,
                "caller_id": conn.caller_id,
                "appointment_data": conn.appointment_data
            })
    
    return Response(response=quart_json.dumps({"appointments": appointments}), status=200)

@app.route("/")
async def hello():
    return "Medical Appointment Booking System - Powered by ACS and OpenAI"

if __name__ == '__main__':
    app.logger.setLevel(INFO)
    app.run(port=8080)