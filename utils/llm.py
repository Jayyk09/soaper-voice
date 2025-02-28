import openai
import logging
import json
# from openai.api_resources import ChatCompletion
from openai import AsyncAzureOpenAI
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

client = AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_SERVICE_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_SERVICE_ENDPOINT,
    )

class LLMClient:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_SERVICE_ENDPOINT,
        )

    def draft_begin_message(self):
        return ResponseResponse(
            response_id=0,
            content=begin_sentence,
            content_complete=True,
            end_call=False,
        )

    def convert_transcript_to_openai_messages(self, transcript: List[Utterance]):
        messages = []
        for utterance in transcript:
            role = "assistant" if utterance.role == "agent" else "user"
            messages.append({"role": role, "content": utterance.content})
        return messages

    def prepare_prompt(self, request: ResponseRequiredRequest):
        prompt = [{"role": "system", "content": agent_prompt}]
        prompt.extend(self.convert_transcript_to_openai_messages(request.transcript))

        if request.interaction_type == "reminder_required":
            prompt.append({
                "role": "user",
                "content": "(Now the user has not responded in a while, you would say:)"
            })
        
        return prompt

    async def draft_response(self, request: ResponseRequiredRequest):
        prompt = self.prepare_prompt(request)
        stream = await self.client.chat.completions.create(
            model="gpt-4o",  # Or use a 3.5 model for speed
            messages=prompt,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield ResponseResponse(
                    response_id=request.response_id,
                    content=chunk.choices[0].delta.content,
                    content_complete=False,
                    end_call=False,
                )

        yield ResponseResponse(
            response_id=request.response_id,
            content="",
            content_complete=True,
            end_call=False,
        )