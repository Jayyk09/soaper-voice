import openai
import logging
import json
from openai.api_resources import ChatCompletion
from config import ANSWER_PROMPT_SYSTEM_TEMPLATE

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

AZURE_OPENAI_SERVICE_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_SERVICE_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT_MODEL_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_MODEL_NAME')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')

logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai.api_key = AZURE_OPENAI_SERVICE_KEY
openai.api_base = AZURE_OPENAI_SERVICE_ENDPOINT
openai.api_type = 'azure'
openai.api_version = AZURE_OPENAI_API_VERSION

async def get_chat_completions_async(system_prompt, user_prompt, format_output=False):
    """Generate a response from OpenAI for the given prompts"""
    
    chat_request = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"In less than 200 characters: respond to this question: {user_prompt}?"}
    ]
  
    try:
        response = await ChatCompletion.acreate(
            model=AZURE_OPENAI_DEPLOYMENT_MODEL_NAME,
            deployment_id=AZURE_OPENAI_DEPLOYMENT_MODEL_NAME, 
            messages=chat_request,
            max_tokens=1000,
            response_format={"type": "json_object"} if format_output else None
        )
        
        if response is not None:
            response_content = response['choices'][0]['message']['content']
        else:
            response_content = ""
            
        return response_content
    
    except Exception as e:
        logger.error("Error in OpenAI API call: %s", e)
        return ""
    
async def get_chat_gpt_response(speech_input):
    """Process speech input through the OpenAI model with the standard template"""
    return await get_chat_completions_async(ANSWER_PROMPT_SYSTEM_TEMPLATE, speech_input)

async def extract_meeting_details(meeting_text):
    """
    Use Azure OpenAI to extract the time and date for the meeting from the given text
    """
    system_prompt = '''
    Extract the time and date for the meeting from the following text. Return the response in JSON format.
    The response should be in the following format:
    {
        "time": "10:00 AM",
        "date": "2025-01-01"
    }
    '''
    json_response = await get_chat_completions_async(system_prompt, meeting_text, format_output=True)
    try:
        # Parse the JSON string into a Python dictionary
        meeting_data = json.loads(json_response)
        return meeting_data
    except json.JSONDecodeError:
        # Return a default structure if JSON parsing fails
        return {"time": "unknown time", "date": "unknown date", "error": "Failed to parse meeting details"}


if __name__ == "__main__":
    import asyncio
    asyncio.run(get_chat_gpt_response("Hello, how are you?"))
