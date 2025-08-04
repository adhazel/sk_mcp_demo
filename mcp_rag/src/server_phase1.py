"""
Basic MCP server setup with event storage and logging.
"""
from mcp.server.fastmcp import FastMCP
import logging
from src.utils import mcp_config as config
from src.utils.event_store import InMemoryEventStore

# Load environment variables
config = config.Config(environment="local", project_name="mcp_rag")

# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

# Create a logger for debugging
logger = logging.getLogger(__name__)

# Create event store for streamable HTTP
event_store = InMemoryEventStore(max_events_per_stream=1000)

# Use MCP configuration from environment variables
PORT = int(config.mcp_port)
HOST = config.mcp_host
URL = f"{HOST}:{PORT}"

##################################################
# Instantiate server
##################################################

mcp = FastMCP(
    name="Demo MCP Server", 
    host=HOST, 
    port=PORT,
    event_store=event_store  # Pass the event store for streamable HTTP support
)

##################################################
# Add internal resources, prompts, and tools
##################################################

# Add a tool to show detailed event information
@mcp.tool()
async def debug_event_details(count: int) -> object:
    """
    [DEBUG] A sample function that shows detailed information about the last N stored events in JSON format. 
    Known Issue: Does not filter out calling this tool.
    """
    import json
    
    total_events = len(event_store.event_index)
    recent_events = event_store.get_last_events(count)

    if total_events == 0:
        return {
            "total_events": 0,
            "showing_last": 0,
            "events": []
        }

    result = {
        "total_events": total_events,
        "showing_last": len(recent_events),
        "events": recent_events
    }
    
    return json.dumps(result, indent=2, default=str)

# @mcp.tool()
# def resume_from_event_id(last_event_id: str) -> str:
#     """Manually resume from a specific event ID"""
#     # Your app can implement custom resumption logic

##################################################
# Add business resources, prompts, and tools
##################################################

# Add an addition tool
@mcp.tool()
async def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Add a static resource
@mcp.resource("info://server")
async def get_server_info() -> str:
    """Get static server information"""
    return "Demo MCP Server v1.0 - This is a static resource that never changes"

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
async def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

##################################################
# Execution with streamable HTTP support for remote MCP servers
##################################################

if __name__ == "__main__":
    print(f"🏠 Local URL: http://{HOST}:{PORT}/mcp/")

    mcp.run(
        transport="streamable-http"
    )
