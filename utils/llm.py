from utils.config import begin_sentence, agent_prompt
import os
from openai import AsyncAzureOpenAI
from utils.custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)
from typing import List
from dotenv import load_dotenv
import asyncio

load_dotenv()

class LLMClient:
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
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
        print(f"Sending prompt with {len(prompt)} messages")
        
        try:
            # Create the streaming request
            stream = await self.client.chat.completions.create(
                model="gpt-4o", 
                messages=prompt,
                stream=True,
            )

            # Process the stream
            async for chunk in stream:                
                # Skip chunks with empty choices (likely metadata/control chunks)
                if not chunk.choices:
                    continue
                    
                # Process content chunks
                if chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                    print(f"Content chunk received: {chunk.choices[0].delta.content}")
                    yield ResponseResponse(
                        response_id=request.response_id,
                        content=chunk.choices[0].delta.content,
                        content_complete=False,
                        end_call=False,
                    )
                else:
                    continue            
            # Signal that we've completed streaming
            print("Stream completed, sending content_complete=True")
            yield ResponseResponse(
                response_id=request.response_id,
                content="",
                content_complete=True,
                end_call=False,
            )
        except Exception as e:
            print(f"Error in draft_response: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Provide a fallback response in case of errors
            yield ResponseResponse(
                response_id=request.response_id,
                content="I'm sorry, I'm having trouble at the moment. Please try again.",
                content_complete=True,
                end_call=False,
            )