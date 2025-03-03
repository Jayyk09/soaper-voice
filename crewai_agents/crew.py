import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from dotenv import load_dotenv

load_dotenv()

# Initialize Azure OpenAI client
azureLLM = LLM(
    api_key=os.getenv("AZURE_API_KEY"),
    api_base=os.getenv("AZURE_API_BASE"),
    api_version=os.getenv("AZURE_API_VERSION"),
    model="azure/gpt-4o",
    # api_version="2024-05-01-preview",
)

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class CallerInfo(BaseModel):
    name: Optional[str] = None
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None

class ReceptionistResponse(BaseModel):
    response: str = Field(description="The response to provide to the caller")
    extracted_info: Optional[CallerInfo] = Field(
        default=None, 
        description="Information extracted from the conversation"
    )

class AppointmentResponse(BaseModel):
    response: str = Field(description="The full response to provide to the caller")
    updated_info: Optional[CallerInfo] = Field(
        default=None, 
        description="Updated caller information after this interaction"
    )

@CrewBase
class MedicalOfficeVoiceApp:
    """Medical Office Voice Assistant Crew with Sequential Process"""

    @agent
    def receptionist(self) -> Agent:
        return Agent(
            config=self.agents_config['receptionist'],
            verbose=True,
            llm=azureLLM
        )
        
    @agent
    def appointment_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['appointment_specialist'],
            verbose=True,
            llm=azureLLM
        )
        
    @task
    def assess_request(self) -> Task:
        return Task(
            config=self.tasks_config['assess_request'],
            llm=azureLLM,
            output_json=ReceptionistResponse
        )
        
    @task
    def handle_appointment(self) -> Task:
        return Task(
            config=self.tasks_config['handle_appointment'],
            llm=azureLLM,
            output_json=AppointmentResponse
        )
        
    @crew
    def crew(self) -> Crew:
        """Creates the Medical Office Voice Assistant crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            llm=azureLLM
        )

# Simple function to create a basic response if CrewAI fails
def fallback_response(user_input):
    """Generate a simple response if CrewAI fails"""
    if "appointment" in user_input.lower():
        return "I'd be happy to help you book an appointment. Could you please provide your name and when you'd like to come in?"
    elif "hours" in user_input.lower():
        return "Our office hours are Monday to Friday, 8:00 AM to 5:00 PM, and Saturday from 9:00 AM to 1:00 PM."
    elif "location" in user_input.lower() or "address" in user_input.lower():
        return "We're located at 123 Health Boulevard in the Medical District."
    elif "doctors" in user_input.lower() or "physician" in user_input.lower():
        return "We have several doctors at our practice including Dr. Johnson, Dr. Chen, and Dr. Rodriguez."
    else:
        return "Thank you for calling Soaper Medical Office. How can I assist you today?"

if __name__ == "__main__":
    # Simple test to make sure configuration is working
    try:
        app = MedicalOfficeVoiceApp()
        crew = app.crew()
        
        print("Crew initialized successfully!")
        
        # Simple test with a basic input
        print("Testing crew with a basic input...")
        result = crew.kickoff(inputs={
            "conversation_context": "Caller: I'd like to book an appointment.\n\nJoann: I'd be happy to help you book an appointment. May I have your full name, please?\n\nCaller: My name is John Smith.\n",
            "last_user_message": "My name is John Smith.",
            "caller_info": {"name": "John Smith"}
        })
        
        print("type of result:", type(result) )
        print("result:", result)
        
    except Exception as e:
        import traceback
        print(f"Error during initialization: {str(e)}")
        traceback.print_exc()