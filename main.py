import json
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from concurrent.futures import TimeoutError as ConnectionTimeoutError
from pydantic import BaseModel
from retell import Retell
from utils.custom_types import ConfigResponse, ResponseRequiredRequest, ResponseResponse
from utils.llm import LLMClient
from typing import List, Optional, Tuple

# Load environment variables
load_dotenv(override=True)
retell = Retell(api_key=os.getenv("RETELL_API_KEY"))

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Handle webhook from Retell server. This is used to receive events from Retell server.
# Including call_started, call_ended, call_analyzed
@app.post("/webhook")
async def handle_webhook(request: Request):
    try:
        post_data = await request.json()
        valid_signature = retell.verify(
            json.dumps(post_data, separators=(",", ":"), ensure_ascii=False),
            api_key=str(os.environ["RETELL_API_KEY"]),
            signature=str(request.headers.get("X-Retell-Signature")),
        )
        if not valid_signature:
            print(
                "Received Unauthorized",
                post_data["event"],
                post_data["data"]["call_id"],
            )
            return JSONResponse(status_code=401, content={"message": "Unauthorized"})
        if post_data["event"] == "call_started":
            print("Call started event", post_data['call']['from_number'])
        elif post_data["event"] == "call_ended":
            print("Call ended event", post_data)
        elif post_data["event"] == "call_analyzed":
            print("Call analyzed event", post_data['call']['transcript_object'])
        else:
            print("Unknown event", post_data["event"])
        return JSONResponse(status_code=200, content={"received": True})
    except Exception as err:
        print(f"Error in webhook: {err}")
        return JSONResponse(
            status_code=500, content={"message": "Internal Server Error"}
        )

@app.websocket("/llm-websocket/{call_id}")
async def websocket_handler(websocket: WebSocket, call_id: str):
    """Handles real-time communication with Retell's server over WebSocket."""
    try:
        await websocket.accept()
        llm_client = LLMClient()
        
        # Initialize call state
        call_state = {
            "in_appointment_collection": False,
            "appointment_details": {},
            "intent_checked": False
        }
        
        # Send initial configuration
        await websocket.send_json(ConfigResponse(
            response_type="config",
            config={"auto_reconnect": True, "call_details": True},
            response_id=1
        ).__dict__)
        
        async def handle_message(request_json):
            nonlocal call_state
            
            try:
                if websocket.client_state != WebSocketState.CONNECTED:
                    print("WebSocket disconnected.")
                    return
                
                interaction_type = request_json.get("interaction_type")
                response_id = request_json.get("response_id", 0)
                
                print(f"Handling interaction type: {interaction_type}")
                
                if interaction_type == "call_details":
                    await websocket.send_json(llm_client.draft_begin_message().__dict__)
                elif interaction_type == "ping_pong":
                    await websocket.send_json({"response_type": "ping_pong", "timestamp": request_json.get("timestamp")})
                elif interaction_type in ("response_required", "reminder_required"):
                    request = ResponseRequiredRequest(
                        interaction_type=interaction_type,
                        response_id=response_id,
                        transcript=request_json.get("transcript", []),
                    )
                    
                    # If we're not already collecting appointment info, check for intent
                    if not call_state["in_appointment_collection"] and not call_state["intent_checked"]:
                        call_state["intent_checked"] = True
                        has_intent = await llm_client.detect_appointment_intent(request.transcript)
                        
                        if has_intent:
                            # Start appointment collection process
                            collection_state = await llm_client.start_appointment_collection(request.transcript)
                            call_state.update(collection_state)
                            
                            # Send the appointment collection response
                            await websocket.send_json(ResponseResponse(
                                response_id=request.response_id,
                                content=call_state["next_response"],
                                content_complete=True,
                                end_call=False,
                            ).__dict__)
                            return
                    
                    # If we're in appointment collection mode, update details
                    elif call_state["in_appointment_collection"]:
                        updated_details = await llm_client.update_appointment_details(
                            request.transcript, 
                            call_state["appointment_details"]
                        )
                        
                        call_state["appointment_details"] = {
                            "date": updated_details.get("date"),
                            "time": updated_details.get("time"),
                            "doctor": updated_details.get("doctor")
                        }
                        
                        # Check if collection is complete
                        if updated_details.get("collection_complete", False):
                            call_state["in_appointment_collection"] = False
                            print(f"Appointment details collected: {call_state['appointment_details']}")
                            
                            # Add code here to save the appointment to your system
                            
                            # Send a confirmation message
                            confirmation = f"Great! I've scheduled your appointment for {updated_details.get('date')} at {updated_details.get('time')}. Is there anything else you'd like help with?"
                            
                            await websocket.send_json(ResponseResponse(
                                response_id=request.response_id,
                                content=confirmation,
                                content_complete=True,
                                end_call=False,
                            ).__dict__)
                            return
                    
                    # Reset intent check for new user messages
                    if interaction_type == "response_required" and not call_state["in_appointment_collection"]:
                        call_state["intent_checked"] = False
                    
                    # Handle normal response for non-appointment interactions
                    async for event in llm_client.draft_response(request):
                        await websocket.send_json(event.__dict__)
                        if request.response_id < response_id:
                            break
            except Exception as e:
                print(f"Error handling message: {e}")
        
        async for data in websocket.iter_json():
            await handle_message(data)
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for call {call_id}")
    except ConnectionTimeoutError:
        print(f"Connection timeout for call {call_id}")
    except Exception as e:
        print(f"WebSocket error for call {call_id}: {e}")
        await websocket.close(1011, "Server error")
    finally:
        print(f"WebSocket connection closed for call {call_id}")