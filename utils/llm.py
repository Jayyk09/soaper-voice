from utils.config import begin_sentence, agent_prompt
import os
from utils.custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)
from typing import List
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
import logging
import json
from agents.crew import Voiceapp

# Load environment variables from .env file
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_SERVICE_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_MODEL_NAME = os.getenv('AZURE_OPENAI_MODEL_NAME')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.crew = Voiceapp().crew().kickoff()

        # self.client = AsyncAzureOpenAI(
        #     api_key=AZURE_OPENAI_API_KEY,
        #     api_version=AZURE_OPENAI_API_VERSION,
        #     azure_endpoint=AZURE_OPENAI_SERVICE_ENDPOINT,
        # )

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
        stream = self.crew.run(prompt, stream=True)

        for chunk in stream:
            if "content" in chunk and chunk['content']:
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=chunk['content'],
                    content_complete=False,
                    end_call=False,
                )
                yield response

        # Send final response with "content_complete" set to True to signal completion
        response = ResponseResponse(
            response_id=request.response_id,
            content="",
            content_complete=True,
            end_call=False,
        )
        yield response