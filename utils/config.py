import aiohttp
import asyncio

# System prompts for the agents
agent_prompt = """ 
You are a friendly voice assistant. Your role is to:
1. Greet callers warmly
2. Answer general questions about the practice, services, and office hours
3. Help callers book appointments efficiently

Dont't mention that you are a voice assistant, and if you have the information, go ahead and call the book_appointment function.
If you have the information, don't keep asking the user for the information again and again. 

APPOINTMENT BOOKING GUIDELINES:
When a caller wants to book an appointment, you should call function step1_collect_patient_and_doctor_info.
After that, call function step2_collect_appointment_detail, and then call book_appointment function.

FUNCTION CALLING INSTRUCTIONS:
1. Collect all required information through conversation
2. Once ALL details are collected, and only in sequence, call the necessary functions.

DON'T KEEP ASKING THE USER FOR THE INFORMATION AGAIN AND AGAIN.

DON'T REPEAT THE PATIENT NAME AGAIN AND AGAIN. Once you have the patient name, don't ask for it again.

Your responses should be clear, helpful, and guide the appointment booking process
in a conversational manner suitable for voice interaction. Keep responses concise and natural.

Always maintain a helpful, professional tone.
"""

async def get_all_physicians():
    url = "https://ep.soaper.ai/api/v1/agent/appointments/physicians"
    headers = {
        "Content-Type": "application/json",
        "X-Agent-API-Key": "sk-int-agent-PJNvT3BlbkFJe8ykcJe6kV1KQntXzgMW"
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            return [physician['last_name'] for physician in (await response.json())['items']]

# General message templates
async def generate_physicians_list():
    physicians = await get_all_physicians()
    physicians = ['Doctor ' + physician for physician in physicians]
    if len(physicians) > 2:
        return ', '.join(physicians[:-1]) + ', and ' + physicians[-1]
    else:
        return ' and '.join(physicians)

async def main():
    physicians_list = await generate_physicians_list()
    begin_sentence = f"Hello, thank you for calling, you have reached the office of {physicians_list}, I can help you schedule an appointment with them. If you're an existing patient, please use our mobile app for additional assistance."
    

timeout_silence_prompt = "I'm sorry, I didn't hear anything. If you need assistance, please let me know how I can help you."
goodbye_prompt = "Thank you for calling Soaper Medical Office! Have a great day!"

# Appointment-specific templates
appointment_transfer_message = "I'd be happy to help you book an appointment. Let me get some information from you."
appointment_confirmation = "Great! I've scheduled your appointment for {date} at {time} with {doctor}. You'll receive a confirmation text shortly. Is there anything else you need help with today?"

if __name__ == "__main__":
    asyncio.run(main())