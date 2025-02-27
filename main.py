import asyncio
from quart import Quart
from logging import INFO
from azure.communication.callautomation.aio import CallAutomationClient

from config import ACS_CONNECTION_STRING
from call_handler import setup_incoming_call_handler, setup_callback_handler

# Initialize the Quart application
app = Quart(__name__)
app.logger.setLevel(INFO)

# Initialize the call automation client
call_automation_client = CallAutomationClient.from_connection_string(ACS_CONNECTION_STRING)

@app.route("/")
def hello():
    return "Hello ACS CallAutomation!..test"

# Set up the routes
async def setup_routes():
    await setup_incoming_call_handler(app, call_automation_client)
    await setup_callback_handler(app, call_automation_client)

# Run the application
if __name__ == '__main__':
    asyncio.run(setup_routes())
    app.run(port=8080)