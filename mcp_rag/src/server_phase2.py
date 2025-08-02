"""
NAME: server_phase2.py
DESCRIPTION: This file implements a demo server using the Model Context Protocol (MCP) with streamable HTTP support. It uses mcp.server.fastmcp.FastMCP for the server implementation and session handling with an in-memory event store for resumability. The server provides tools and resources, including debugging tools to inspect stored events.

This server definition is phase 2 in a multi-phase tutorial series. This phase focuses on adding capabilities for the demonstration scenario:

- Search Chroma DB for internal product information
- Search SerpAPI Bing for external web results
- Generate an LLM response based on the search results (RAG)
- Evaluate the LLM response using a custom evaluation tool

AUTHOR: April Hazel
CREDIT: Derived from: 
    https://github.com/modelcontextprotocol/python-sdk/blob/959d4e39ae13e45d3059ec6d6ca82fb231039a91/examples/servers/simple-streamablehttp/mcp_simple_streamablehttp/server.py
HISTORY:
    - 20240730: Initial implementation based on the simple streamable HTTP server example.
"""
from mcp.server.fastmcp import FastMCP
import logging
from src.utils import mcp_config as config
from src.utils.event_store import InMemoryEventStore

# Load environment variables
config = config.Config(environment="local")

# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

# Create a logger for debugging
logger = logging.getLogger(__name__)

# Create event store for streamable HTTP
event_store = InMemoryEventStore(max_events_per_stream=1000)

# Create an MCP server with streamable HTTP support
PORT = 8002
HOST = "127.0.0.1"
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
