from quart import Quart, Response, request, json, redirect, websocket
from azure.communication.callautomation import (
    MediaStreamingOptions, AudioFormat, MediaStreamingTransportType,
    MediaStreamingContentType, MediaStreamingAudioChannelType,
)
from azure.communication.callautomation.aio import CallAutomationClient
import uuid
import json
from openairealtime import RealTimeClient

ACS_CONNECTION_STRING = "ACS_CONNECTION_STRING"
CALLBACK_URI_HOST = "CALLBACK_URI_HOST"
CALLBACK_EVENTS_URI = CALLBACK_URI_HOST + "/api/callbacks"
AZURE_OPENAI_KEY = "AZURE_OPENAI_SERVICE_KEY"
AZURE_OPENAI_DEPLOYMENT = "AZURE_OPENAI_DEPLOYMENT_MODEL_NAME"

acs_client = CallAutomationClient.from_connection_string(ACS_CONNECTION_STRING)
app = Quart(__name__)
client = RealTimeClient(AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT)
active_websocket = None

@app.route("/api/incomingCall", methods=['POST'])
async def incoming_call_handler():
    for event_dict in await request.json:
        if event_dict['eventType'] == "Microsoft.Communication.IncomingCall":
            caller_id = event_dict['data']['from']['phoneNumber']['value']
            incoming_call_context = event_dict['data']['incomingCallContext']
            guid = str(uuid.uuid4())
            callback_uri = f"{CALLBACK_EVENTS_URI}/{guid}"

            media_streaming_options = MediaStreamingOptions(
                transport_url=f"wss://{CALLBACK_URI_HOST}/ws",
                transport_type=MediaStreamingTransportType.WEBSOCKET,
                content_type=MediaStreamingContentType.AUDIO,
                audio_channel_type=MediaStreamingAudioChannelType.MIXED,
                start_media_streaming=True,
                enable_bidirectional=True,
                audio_format=AudioFormat.PCM24_K_MONO
            )

            await acs_client.answer_call(
                incoming_call_context=incoming_call_context,
                operation_context="incomingCall",
                callback_url=callback_uri,
                media_streaming=media_streaming_options
            )

    return Response(status=200)

@app.websocket('/ws')
async def ws():
    global active_websocket
    active_websocket = websocket
    print("WebSocket Connected")
    await client.start()
    while True:
        try:
            data = await websocket.receive()
            response = await client.transcribe_and_generate(data)
            if response.audio:
                await websocket.send(json.dumps({"Kind": "AudioData", "AudioData": {"Data": response.audio}}))
        except Exception as e:
            print(f"WebSocket Disconnected: {e}")
            break

@app.route('/')
def home():
    return 'Hello ACS CallAutomation with OpenAI RealTime!'

if __name__ == '__main__':
    app.run(port=8080)
