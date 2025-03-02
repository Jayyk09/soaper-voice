from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import os
from openai import AsyncAzureOpenAI

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_SERVICE_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_MODEL_NAME = os.getenv('AZURE_OPENAI_MODEL_NAME')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')

azureLLM = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_SERVICE_ENDPOINT,
    model=AZURE_OPENAI_MODEL_NAME,
)

@CrewBase
class MedicalOfficeVoiceApp():
    """Medical Office Voice Assistant Crew"""

    # Configuration files
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def general_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['general_agent'],
            verbose=True,
            llm=azureLLM,
            allow_delegation=True
        )

    @agent
    def appointment_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['appointment_specialist'],
            verbose=True,
            llm=azureLLM,
        )

    @task
    def handle_general_inquiry(self) -> Task:
        return Task(
            config=self.tasks_config['handle_general_inquiry'],
            agent=self.general_agent
        )

    @task
    def book_appointment(self) -> Task:
        return Task(
            config=self.tasks_config['book_appointment'],
            agent=self.appointment_specialist
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Medical Office Voice Assistant crew"""
        return Crew(
            agents=self.agents,
            tasks=[self.handle_general_inquiry, self.book_appointment],
            process=Process.hierarchical,  # Using hierarchical to enable delegation
            verbose=True,
            llm=azureLLM,
        )