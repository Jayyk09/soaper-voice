async def get_chat_completions_async(system_prompt,user_prompt): 
    openai.api_key = AZURE_OPENAI_SERVICE_KEY
    openai.api_base = AZURE_OPENAI_SERVICE_ENDPOINT # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
    openai.api_type = 'azure'
    openai.api_version = '2023-05-15' # this may change in the future
    
    # Define your chat completions request
    chat_request = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"In less than 200 characters: respond to this question: {user_prompt}?"}
    ]
  
    global response_content
    try:
        response = await ChatCompletion.acreate(model=AZURE_OPENAI_DEPLOYMENT_MODEL,deployment_id=AZURE_OPENAI_DEPLOYMENT_MODEL_NAME, messages=chat_request,max_tokens = 1000)
    except Exception as ex:
        app.logger.info("error in openai api call : %s",ex)
       
    # Extract the response content
    if response is not None :
         response_content  =  response['choices'][0]['message']['content']
    else :
         response_content=""    
    return response_content  

async def get_chat_gpt_response(speech_input):
   return await get_chat_completions_async(ANSWER_PROMPT_SYSTEM_TEMPLATE,speech_input)

async def handle_recognize(replyText,callerId,call_connection_id,context=""):
    play_source = TextSource(text=replyText, voice_name="en-US-NancyNeural")
    connection_client = call_automation_client.get_call_connection(call_connection_id)
    try:
        recognize_result = await connection_client.start_recognizing_media( 
        input_type=RecognizeInputType.SPEECH,
        target_participant=PhoneNumberIdentifier(callerId), 
        end_silence_timeout=10, 
        play_prompt=play_source,
        operation_context=context)
        app.logger.info("handle_recognize : data=%s",recognize_result)
    except Exception as ex:
        app.logger.info("Error in recognize: %s", ex)

async def handle_play(call_connection_id, text_to_play, context):     
    play_source = TextSource(text=text_to_play, voice_name= "en-US-NancyNeural") 
    await call_automation_client.get_call_connection(call_connection_id).play_media_to_all(
        play_source,
        operation_context=context)

async def handle_hangup(call_connection_id):     
    await call_automation_client.get_call_connection(call_connection_id).hang_up(is_for_everyone=True)   

async def detect_escalate_to_agent_intent(speech_text, logger):
    return await has_intent_async(user_query=speech_text, intent_description="talk to agent", logger=logger)

async def has_intent_async(user_query, intent_description, logger):
    is_match=False
    system_prompt = "You are a helpful assistant"
    combined_prompt = f"In 1 word: does {user_query} have a similar meaning as {intent_description}?"
    #combined_prompt = base_user_prompt.format(user_query, intent_description)
    response = await get_chat_completions_async(system_prompt, combined_prompt)
    if "yes" in response.lower():
        is_match =True        
    logger.info(f"OpenAI results: is_match={is_match}, customer_query='{user_query}', intent_description='{intent_description}'")
    return is_match

def get_sentiment_score(sentiment_score):
    pattern = r"(\d)+"
    regex = re.compile(pattern)
    match = regex.search(sentiment_score)
    return int(match.group()) if match else -1

async def answer_call_async(incoming_call_context,callback_url):
    return await call_automation_client.answer_call(
        incoming_call_context=incoming_call_context,
        cognitive_services_endpoint=COGNITIVE_SERVICE_ENDPOINT,
        callback_url=callback_url)
