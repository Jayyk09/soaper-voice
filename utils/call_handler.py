import uuid
import re
from urllib.parse import urlencode
from quart import request, Response, json
from azure.eventgrid import EventGridEvent, SystemEventNames
from azure.core.messaging import CloudEvent
from azure.communication.callautomation import PhoneNumberIdentifier
from utils.voice_service import answer_call_async
# Global variables
caller_id = None

from config import (
    CALLBACK_EVENTS_URI, HELLO_PROMPT, TIMEOUT_SILENCE_PROMPT, GOODBYE_PROMPT,
    CONNECT_AGENT_PROMPT, CALLTRANSFER_FAILURE_PROMPT, AGENT_PHONE_NUMBER_EMPTY_PROMPT,
    END_CALL_PHRASE_TO_CONNECT_AGENT, TRANSFER_FAILED_CONTEXT, CONNECT_AGENT_CONTEXT,
    GOODBYE_CONTEXT, CHAT_RESPONSE_EXTRACT_PATTERN, AGENT_PHONE_NUMBER
)
from utils.llm import get_chat_gpt_response, has_intent_async
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
                            # Check for meeting booking intent instead of escalation
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
                    elif context.lower() == TRANSFER_FAILED_CONTEXT.lower() or context.lower() == GOODBYE_CONTEXT.lower():
                        await handle_hangup(call_automation_client, event.data['callConnectionId'])
                    elif context.lower() == CONNECT_AGENT_CONTEXT.lower():
                        if not AGENT_PHONE_NUMBER or AGENT_PHONE_NUMBER.isspace():
                            app.logger.info(f"Agent phone number is empty")
                            await handle_play(
                                call_automation_client,
                                event.data['callConnectionId'],
                                AGENT_PHONE_NUMBER_EMPTY_PROMPT
                            )
                        else:
                            app.logger.info(f"Initializing the Call transfer...")
                            transfer_destination = PhoneNumberIdentifier(AGENT_PHONE_NUMBER)
                            call_connection_client = call_automation_client.get_call_connection(
                                call_connection_id=event.data['callConnectionId']
                            )
                            await call_connection_client.transfer_call_to_participant(
                                target_participant=transfer_destination
                            )
                            app.logger.info(f"Transfer call initiated: {context}")
                
                elif event.type == "Microsoft.Communication.CallTransferAccepted":
                    app.logger.info(f"Call transfer accepted event received for connection id: {event.data['callConnectionId']}")
                
                elif event.type == "Microsoft.Communication.CallTransferFailed":
                    app.logger.info(f"Call transfer failed event received for connection id: {event.data['callConnectionId']}")
                    resultInformation = event.data['resultInformation']
                    sub_code = resultInformation['subCode']
                    app.logger.info(f"Encountered error during call transfer, subCode={sub_code}")
                    await handle_play(
                        call_automation_client,
                        event.data['callConnectionId'],
                        CALLTRANSFER_FAILURE_PROMPT, 
                        TRANSFER_FAILED_CONTEXT
                    )
                    
            return Response(status=200)
            
        except Exception as ex:
            app.logger.error(f"Error in event handling: {ex}")
            return Response(status=500)