import re
from azure.communication.callautomation import PhoneNumberIdentifier, RecognizeInputType, TextSource
from utils.config import COGNITIVE_SERVICE_ENDPOINT

def get_sentiment_score(sentiment_score):
    """Extract numerical sentiment score from text response"""
    pattern = r"(\d)+"
    regex = re.compile(pattern)
    match = regex.search(sentiment_score)
    return int(match.group()) if match else -1

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