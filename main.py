import json
import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, APIRouter
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
            print("Call started event", post_data['call'])
        elif post_data["event"] == "call_ended":
            print("Call ended event", post_data)
        elif post_data["event"] == "call_analyzed":
            print("Call analyzed event", post_data['call']['call_analysis']['custom_analysis_data'])

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
        
        # Send initial configuration
        await websocket.send_json(ConfigResponse(
            response_type="config",
            config={"auto_reconnect": True, "call_details": True},
            response_id=1
        ).__dict__)
        
        async def handle_message(request_json):
            heartbeat_task = asyncio.create_task(send_heartbeats(websocket))
            try:
                if websocket.client_state != WebSocketState.CONNECTED:
                    print("WebSocket disconnected.")
                    return
                
                interaction_type = request_json.get("interaction_type")
                response_id = request_json.get("response_id", 0)
                
                print(f"Handling interaction type: {interaction_type}")
                
                if interaction_type == "call_details":
                    response = await llm_client.draft_begin_message()
                    await websocket.send_json(response.__dict__)
                elif interaction_type == "ping_pong":
                    await websocket.send_json({"response_type": "ping_pong", "timestamp": request_json.get("timestamp")})
                elif interaction_type in ("response_required", "reminder_required"):
                    request = ResponseRequiredRequest(
                        interaction_type=interaction_type,
                        response_id=response_id,
                        transcript=request_json.get("transcript", []),
                    )
                    async for event in llm_client.draft_response(request):
                        await websocket.send_json(event.__dict__)
                        if request.response_id < response_id:
                            break
            except Exception as e:
                print(f"Error handling message: {e}")
            finally:
                heartbeat_task.cancel()

        async def send_heartbeats(websocket: WebSocket):
            """Send periodic pings to keep the connection alive."""
            try:
                while True:
                    await asyncio.sleep(15)  # Send heartbeat every 15 seconds
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json({"response_type": "ping_pong", "timestamp": int(time.time() * 1000)})
            except asyncio.CancelledError:
                # Task was cancelled, clean up
                pass
            except Exception as e:
                print(f"Error in heartbeat: {e}")

        
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
