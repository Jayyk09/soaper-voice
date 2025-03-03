import json
import os
import logging
import traceback  # Import traceback for detailed error info
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from concurrent.futures import TimeoutError as ConnectionTimeoutError
from pydantic import BaseModel
from retell import Retell
from utils.custom_types import ConfigResponse, ResponseRequiredRequest, ResponseResponse, Utterance
from utils.llm import LLMClient
from typing import List, Optional, Tuple
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)
retell_api_key = os.getenv("RETELL_API_KEY")
retell = Retell(api_key=retell_api_key)

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Handle webhook from Retell server
@app.post("/webhook")
async def handle_webhook(request: Request):
    try:
        post_data = await request.json()
        valid_signature = retell.verify(
            json.dumps(post_data, separators=(",", ":"), ensure_ascii=False),
            api_key=retell_api_key,
            signature=str(request.headers.get("X-Retell-Signature")),
        )
        
        if not valid_signature:
            print(f"Received Unauthorized {post_data['event']}")
            return JSONResponse(status_code=401, content={"message": "Unauthorized"})
            
        # Log different call events
        if post_data["event"] == "call_started":
            print(f"Call started: {post_data['call'].get('from_number', 'unknown')}")
            
        elif post_data["event"] == "call_ended":
            print(f"Call ended: {post_data['call'].get('id', 'unknown')}")
            
        elif post_data["event"] == "call_analyzed":
            print(f"Call analyzed: {post_data['call'].get('id', 'unknown')}")
                
        else:
            print(f"Unknown event: {post_data['event']}")
            
        return JSONResponse(status_code=200, content={"received": True})
        
    except Exception as err:
        print(f"Error in webhook: {str(err)}")
        traceback.print_exc()  # Print detailed traceback
        return JSONResponse(
            status_code=500, content={"message": "Internal Server Error"}
        )

@app.websocket("/llm-websocket/{call_id}")
async def websocket_handler(websocket: WebSocket, call_id: str):
    """Handle WebSocket connection for real-time voice interaction"""
    llm_client = None
    
    try:
        await websocket.accept()
        print(f"WebSocket connection opened for call {call_id}")
        
        # Debug: Print all request headers
        print(f"WebSocket headers: {dict(websocket.headers)}")
        
        # Initialize LLM client for this call
        try:
            llm_client = LLMClient()
            print(f"LLM client initialized successfully for call {call_id}")
        except Exception as e:
            print(f"Error initializing LLM client: {str(e)}")
            traceback.print_exc()
            raise
        
        # Send configuration to Retell server
        try:
            config = ConfigResponse(
                response_type="config",
                config={
                    "auto_reconnect": True,
                    "call_details": True,
                },
                response_id=1,
            )
            print(f"Sending config: {config}")
            await websocket.send_json(config.__dict__)
            print("Config sent successfully")
        except Exception as e:
            print(f"Error sending config: {str(e)}")
            traceback.print_exc()
            raise
        
        # Track response ID
        response_id = 0

        async def handle_message(request_json):
            nonlocal response_id
            nonlocal llm_client
            
            print(f"Received message with interaction_type: {request_json['interaction_type']}")
            
            try:
                # Verify llm_client is properly initialized
                if llm_client is None:
                    print(f"LLM client is None for call {call_id}, reinitializing")
                    llm_client = LLMClient()
                
                # Handle different interaction types
                if request_json["interaction_type"] == "call_details":
                    print(f"Handling call_details for {call_id}")
                    # Send initial greeting
                    try:
                        first_event = llm_client.draft_begin_message()
                        print(f"Initial greeting: {first_event.content}")
                        await websocket.send_json(first_event.__dict__)
                        print("Initial greeting sent successfully")
                    except Exception as e:
                        print(f"Error sending initial greeting: {str(e)}")
                        traceback.print_exc()
                    return
                    
                elif request_json["interaction_type"] == "ping_pong":
                    # Respond to keep-alive pings
                    ping_response = {
                        "response_type": "ping_pong",
                        "timestamp": request_json["timestamp"],
                    }
                    await websocket.send_json(ping_response)
                    print(f"Responded to ping_pong with timestamp {request_json['timestamp']}")
                    return
                    
                elif request_json["interaction_type"] == "update_only":
                    # No response needed for updates
                    print("Received update_only, no response needed")
                    return
                    
                elif request_json["interaction_type"] in ["response_required", "reminder_required"]:
                    print(f"Handling {request_json['interaction_type']} for {call_id}")
                    # Process response or reminder requests
                    response_id = request_json["response_id"]
                    
                    # Debug: Print the transcript structure
                    print(f"Transcript type: {type(request_json['transcript'])}")
                    if len(request_json['transcript']) > 0:
                        print(f"First transcript item type: {type(request_json['transcript'][0])}")
                    
                    # Create request object
                    try:
                        request = ResponseRequiredRequest(
                            interaction_type=request_json["interaction_type"],
                            response_id=response_id,
                            transcript=request_json["transcript"],
                        )
                        print(f"Created request object with response_id {response_id}")
                    except Exception as e:
                        print(f"Error creating request object: {str(e)}")
                        traceback.print_exc()
                        raise
                    
                    # Log the request
                    try:
                        last_user_message = ""
                        if request_json["transcript"] and len(request_json["transcript"]) > 0:
                            last_utterance = request_json["transcript"][-1]
                            print(f"Last utterance: {last_utterance}")
                            
                            if hasattr(last_utterance, 'content'):
                                last_user_message = last_utterance.content
                            elif isinstance(last_utterance, dict) and 'content' in last_utterance:
                                last_user_message = last_utterance['content']
                            
                        print(
                            f"Processing {request_json['interaction_type']}, "
                            f"response_id={response_id}, "
                            f"last_message='{last_user_message[:50]}...'"
                        )
                    except Exception as e:
                        print(f"Error accessing transcript: {str(e)}")
                        traceback.print_exc()

                    # Generate and stream response
                    try:
                        print("About to call draft_response")
                        async for event in llm_client.draft_response(request):
                            print(f"Got response chunk: {event.content}")
                            await websocket.send_json(event.__dict__)
                            # If a new response is needed, abandon current one
                            if request.response_id < response_id:
                                print("Abandoning response due to new request")
                                break
                        print("Finished streaming response")
                    except Exception as e:
                        print(f"Error generating or streaming response: {str(e)}")
                        traceback.print_exc()
                        raise
                        
                else:
                    print(f"Unknown interaction_type: {request_json['interaction_type']}")
                    
            except Exception as e:
                print(f"Error in handle_message: {str(e)}")
                traceback.print_exc()

        # Listen for messages
        print(f"Starting message listener for {call_id}")
        async for data in websocket.iter_json():
            print(f"Received data: {data}")
            # Process each message in a separate task
            asyncio.create_task(handle_message(data))

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for call {call_id}")
        
    except ConnectionTimeoutError:
        print(f"Connection timeout for call {call_id}")
        
    except Exception as e:
        print(f"Error in WebSocket handler: {str(e)} for call {call_id}")
        print("Detailed error information:")
        traceback.print_exc()
        try:
            await websocket.close(1011, "Server error")
        except:
            pass
            
    finally:
        print(f"WebSocket connection closed for call {call_id}")
