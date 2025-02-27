# Global variables
caller_id = None


async def setup_incoming_call_handler(app, call_automation_client):
    """Set up the incoming call handler"""

    @app.route("/incoming-call", methods=["POST"])
    async def incoming_call_handler():
        """Asynchronously handle incoming calls"""
        pass

async def setup_callback_handler(app, call_automation_client):
    """Set up the route and handler for callbacks during the call"""
    
    @app.route("/api/callbacks/<context_id>", methods=["POST"])
    async def handle_callback(context_id):
        """Handle the callback for the given context ID"""
        pass