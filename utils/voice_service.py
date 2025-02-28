import asyncio
import re
from azure.communication.callautomation import PhoneNumberIdentifier, RecognizeInputType, TextSource
import os

from utils.llm import extract_meeting_details

COGNITIVE_SERVICE_ENDPOINT = os.getenv('COGNITIVE_SERVICE_ENDPOINT')

# New function to detect meeting booking intent
async def detect_meeting_booking_intent(speech_text, logger):
    """
    Detect if the user wants to book a meeting based on their speech
    """
    booking_keywords = [
        "book a meeting", "schedule a meeting", "set up a meeting", 
        "make an appointment", "book time", "schedule time",
        "calendar", "appointment", "booking", "schedule"
    ]
    
    speech_lower = speech_text.lower()
    
    # Simple keyword matching
    for keyword in booking_keywords:
        if keyword in speech_lower:
            logger.info(f"Meeting booking intent detected with keyword: {keyword}")
            return True
    return False
            
# New function to process meeting details
async def process_meeting_details(meeting_text):
    """
    Process the meeting details from user speech and return a confirmation message
    """
    meeting_details = await extract_meeting_details(meeting_text)

    return f"I've noted your meeting request. To confirm, you said: '{meeting_text}'. Your meeting has been scheduled for '{meeting_details}'."


async def handle_recognize(call_automation_client, reply_text, caller_id, call_connection_id, context=""):
    """Play prompt and start speech recognition"""
    play_source = TextSource(text=reply_text, voice_name="en-US-NancyNeural")
    connection_client = call_automation_client.get_call_connection(call_connection_id)
    
    try:
        recognize_result = await connection_client.start_recognizing_media(
            input_type=RecognizeInputType.SPEECH,
            target_participant=PhoneNumberIdentifier(caller_id),
            end_silence_timeout=10,
            play_prompt=play_source,
            operation_context=context
        )
        return recognize_result
    except Exception as ex:
        print(f"Error in recognize: {ex}")
        return None

async def handle_play(call_automation_client, call_connection_id, text_to_play, context):
    """Play text without expecting a response"""
    play_source = TextSource(text=text_to_play, voice_name="en-US-NancyNeural")
    await call_automation_client.get_call_connection(call_connection_id).play_media_to_all(
        play_source,
        operation_context=context
    )

async def handle_hangup(call_automation_client, call_connection_id):
    """Hang up the call"""
    await call_automation_client.get_call_connection(call_connection_id).hang_up(is_for_everyone=True)

async def answer_call_async(call_automation_client, incoming_call_context, callback_url):
    """Answer incoming call and set up cognitive services"""
    return await call_automation_client.answer_call(
        incoming_call_context=incoming_call_context,
        cognitive_services_endpoint=COGNITIVE_SERVICE_ENDPOINT,
        callback_url=callback_url
    )

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    bookMeeting = asyncio.run(detect_meeting_booking_intent("I want to book a meeting with John Doe on Monday at 10 AM", logger))
    print(bookMeeting)