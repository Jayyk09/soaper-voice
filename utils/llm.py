import openai
import logging
from openai.api_resources import ChatCompletion
from utils.config import (
    AZURE_OPENAI_SERVICE_KEY, 
    AZURE_OPENAI_SERVICE_ENDPOINT, 
    AZURE_OPENAI_DEPLOYMENT_MODEL, 
    AZURE_OPENAI_DEPLOYMENT_MODEL_NAME,
    ANSWER_PROMPT_SYSTEM_TEMPLATE
)

logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai.api_key = AZURE_OPENAI_SERVICE_KEY
openai.api_base = AZURE_OPENAI_SERVICE_ENDPOINT
openai.api_type = 'azure'
openai.api_version = '2023-05-15'

async def get_chat_completions_async(system_prompt, user_prompt):
    """Generate a response from OpenAI for the given prompts"""
    
    chat_request = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"In less than 200 characters: respond to this question: {user_prompt}?"}
    ]
  
    try:
        response = await ChatCompletion.acreate(
            model=AZURE_OPENAI_DEPLOYMENT_MODEL,
            deployment_id=AZURE_OPENAI_DEPLOYMENT_MODEL_NAME, 
            messages=chat_request,
            max_tokens=1000
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

if __name__ == "__main__":
    print(get_chat_gpt_response("Hello, how are you?"))