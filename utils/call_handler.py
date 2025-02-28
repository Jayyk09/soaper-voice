from urllib.parse import urlencode
from quart import request, Response, json
from azure.eventgrid import EventGridEvent, SystemEventNames
from azure.core.messaging import CloudEvent
from utils.voice_service import answer_call_async
import uuid
# Global variables
caller_id = None
max_retry = 3  # Initialize max_retry variable

from config import (
    CALLBACK_EVENTS_URI, HELLO_PROMPT, TIMEOUT_SILENCE_PROMPT, GOODBYE_PROMPT,
    GOODBYE_CONTEXT
)

from utils.llm import get_chat_gpt_response
from utils.voice_service import (
    handle_recognize, handle_play, handle_hangup, answer_call_async,
    detect_meeting_booking_intent, process_meeting_details
)

async def setup_incoming_call_handler(app, call_automation_client):
    """Set up the incoming call handler"""

    @app.route("/incoming-call", methods=["POST"])
    async def incoming_call_handler():
        """Asynchronously handle incoming calls"""
        for event_dict in await request.json:
            event = EventGridEvent.from_dict(event_dict)
            app.logger.info("incoming event data --> %s", event.data)
            
            if event.event_type == SystemEventNames.EventGridSubscriptionValidationEventName:
                app.logger.info("Validating subscription")
                validation_code = event.data['validationCode']
                validation_response = {'validationResponse': validation_code}
                return Response(response=json.dumps(validation_response), status=200)
                
            elif event.event_type == "Microsoft.Communication.IncomingCall":
                app.logger.info("Incoming call received: data=%s", event.data)
                
                if event.data['from']['kind'] == "phoneNumber":
                    caller_id = event.data['from']["phoneNumber"]["value"]
                else:
                    caller_id = event.data['from']['rawId']
                    
                app.logger.info("incoming call handler caller id: %s", caller_id)
                
                incoming_call_context = event.data['incomingCallContext']
                guid = uuid.uuid4()
                query_parameters = urlencode({"callerId": caller_id})
                callback_uri = f"{CALLBACK_EVENTS_URI}/{guid}?{query_parameters}"

                app.logger.info("callback url: %s", callback_uri)

                answer_call_result = await answer_call_async(
                    call_automation_client,
                    incoming_call_context,
                    callback_uri
                )
                
                app.logger.info("Answered call for connection id: %s",
                                answer_call_result.call_connection_id)
                return Response(status=200)
                
        return Response(status=400)  # Bad request if no valid events

async def setup_callback_handler(app, call_automation_client):
    """Set up the route and handler for callbacks during the call"""
    
    @app.route("/api/callbacks/<context_id>", methods=["POST"])
    async def handle_callback(context_id):
        try:
            global caller_id, max_retry
            
            app.logger.info("Request Json: %s", await request.json)
            
            for event_dict in await request.json:
                event = CloudEvent.from_dict(event_dict)
                app.logger.info("%s event received for call connection id: %s", 
                               event.type, event.data['callConnectionId'])
                
                caller_id = request.args.get("callerId").strip()
                if "+" not in caller_id:
                    caller_id = "+" + caller_id.strip()

                app.logger.info("call connected : data=%s", event.data)
                
                if event.type == "Microsoft.Communication.CallConnected":
                    await handle_recognize(
                        call_automation_client,
                        HELLO_PROMPT,
                        caller_id, 
                        event.data['callConnectionId'],
                        context="GetFreeFormText"
                    )
                    
                elif event.type == "Microsoft.Communication.RecognizeCompleted":
                    if event.data['recognitionType'] == "speech":
                        speech_text = event.data['speechResult']['speech']
                        app.logger.info("Recognition completed, speech_text =%s", speech_text)
                        
                        if speech_text is not None and len(speech_text) > 0:
                            # Check for meeting booking intent
                            has_booking_intent = await detect_meeting_booking_intent(
                                speech_text=speech_text,
                                logger=app.logger
                            )
                            
                            if has_booking_intent:
                                # Handle meeting booking flow
                                await handle_recognize(
                                    call_automation_client,
                                    "I'd be happy to help you book a meeting. What date and time works for you?",
                                    caller_id,
                                    event.data['callConnectionId'],
                                    context="MeetingBookingFlow"
                                )
                            else:
                                # Continue with general conversation
                                chat_gpt_response = await get_chat_gpt_response(speech_text)
                                app.logger.info(f"Chat GPT response:{chat_gpt_response}")
                                
                                await handle_recognize(
                                    call_automation_client,
                                    chat_gpt_response,
                                    caller_id,
                                    event.data['callConnectionId'],
                                    context="OpenAISample"
                                )
                
                # Handle MeetingBookingFlow context in RecognizeCompleted
                elif event.type == "Microsoft.Communication.RecognizeCompleted" and event.data.get('operationContext') == "MeetingBookingFlow":
                    if event.data['recognitionType'] == "speech":
                        meeting_details = event.data['speechResult']['speech']
                        app.logger.info(f"Meeting details: {meeting_details}")
                        
                        # Process meeting details and confirm
                        confirmation_message = await process_meeting_details(meeting_details)
                        
                        await handle_play(
                            call_automation_client,
                            event.data['callConnectionId'],
                            confirmation_message,
                            "MeetingConfirmation"
                        )
                
                elif event.type == "Microsoft.Communication.RecognizeFailed":
                    resultInformation = event.data['resultInformation']
                    reasonCode = resultInformation['subCode']
                    context = event.data['operationContext']
                    
                    if reasonCode == 8510 and max_retry > 0:
                        await handle_recognize(
                            call_automation_client,
                            TIMEOUT_SILENCE_PROMPT,
                            caller_id,
                            event.data['callConnectionId']
                        )
                        max_retry -= 1
                    else:
                        await handle_play(
                            call_automation_client,
                            event.data['callConnectionId'],
                            GOODBYE_PROMPT, 
                            GOODBYE_CONTEXT
                        )
                    
                elif event.type == "Microsoft.Communication.PlayCompleted":
                    context = event.data['operationContext']
                    
                    if context.lower() == "meetingconfirmation":
                        # After confirming meeting, say goodbye and end call
                        await handle_play(
                            call_automation_client,
                            event.data['callConnectionId'],
                            "Your meeting has been booked. Thank you for calling!",
                            GOODBYE_CONTEXT
                        )
                    elif context.lower() == GOODBYE_CONTEXT.lower():
                        await handle_hangup(call_automation_client, event.data['callConnectionId'])
                    
            return Response(status=200)
            
        except Exception as ex:
            app.logger.error(f"Error in event handling: {ex}")
            return Response(status=500)
