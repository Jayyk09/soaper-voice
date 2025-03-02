from crewai import Agent, Crew, Process, Task, LLM
import os
from langchain_openai import AzureChatOpenAI

OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
OPENAI_SERVICE_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
OPENAI_MODEL_NAME = os.getenv('AZURE_OPENAI_MODEL_NAME', 'gpt-4o')
OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2023-07-01-preview')

# Verify environment variables are set
print(f"API Key available: {OPENAI_API_KEY is not None}")
print(f"Endpoint available: {OPENAI_SERVICE_ENDPOINT is not None}")
print(f"Model name: {OPENAI_MODEL_NAME}")
print(f"API version: {OPENAI_API_VERSION}")

# Initialize Azure OpenAI client
azureLLM = LLM(
    model="azure/gpt-4o",
    api_version="2024-05-01-preview",
)

class MedicalOfficeVoiceApp:
    """Medical Office Voice Assistant Crew"""

    def __init__(self):
        print("Initializing Medical Office Voice App...")
        
        # Create agents
        self.general_agent = Agent(
            role="Medical Office General Assistant",
            goal="Assist callers with general inquiries and delegate to appointment specialist when needed",
            backstory="You are Joann, a friendly assistant at Soaper Medical Office. You handle general questions and delegate appointment booking to the specialist.",
            verbose=True,
            llm=azureLLM,
            allow_delegation=True
        )
        
        self.appointment_specialist = Agent(
            role="Medical Appointment Booking Specialist",
            goal="Collect necessary information to book a doctor's appointment",
            backstory="You are the appointment specialist who collects patient information for booking medical appointments.",
            verbose=True,
            llm=azureLLM,
        )
        
        # Create a manager agent (required for hierarchical process)
        self.manager_agent = Agent(
            role="Office Manager",
            goal="Coordinate between the general assistant and appointment specialist",
            backstory="You are the office manager who ensures that caller inquiries are addressed properly.",
            verbose=True,
            llm=azureLLM,
        )
        
        # Create tasks
        self.handle_general_inquiry = Task(
            description="Handle general inquiries about the medical office",
            expected_output="A helpful response to the caller's inquiry",
            agent=self.general_agent
        )
        
        self.book_appointment = Task(
            description="Collect all necessary information to book a doctor's appointment",
            expected_output="Complete appointment details and confirmation",
            agent=self.appointment_specialist
        )
        
        print("Agents and tasks created successfully")

    def crew(self):
        """Creates the Medical Office Voice Assistant crew"""
        print("Creating crew...")
        return Crew(
            agents=[self.general_agent, self.appointment_specialist],
            tasks=[self.handle_general_inquiry, self.book_appointment],
            process=Process.hierarchical,  # Using hierarchical to enable delegation
            verbose=True,
            llm=azureLLM,
            manager_agent=self.manager_agent,  # Add the manager agent
        )

# Simple function to create a basic response if CrewAI fails
def fallback_response(user_input):
    """Generate a simple response if CrewAI fails"""
    if "appointment" in user_input.lower():
        return "I'd be happy to help you book an appointment. Could you please provide your name?"
    elif "hours" in user_input.lower():
        return "Our office hours are Monday to Friday, 8:00 AM to 5:00 PM."
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
            "conversation_context": "Caller: I'd like to know your office hours.\n",
            "last_user_message": "I'd like to know your office hours."
        })
        
        print("Test result:", result)
        
    except Exception as e:
        import traceback
        print(f"Error during initialization: {str(e)}")
        traceback.print_exc()