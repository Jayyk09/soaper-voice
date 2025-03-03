from openai import AsyncAzureOpenAI
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_version=os.getenv("AZURE_API_VERSION"),
)

async def main():
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())