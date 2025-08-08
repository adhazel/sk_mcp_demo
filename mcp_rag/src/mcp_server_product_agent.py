"""
MCP server for searching an internal product database and the external web, with evaluation logic for RAG (Retrieval-Augmented Generation) workflows.
"""
from mcp.server.fastmcp import FastMCP
import logging
from typing import Optional
import sys
import json
from pathlib import Path
import os

# Add the current src directory to Python path to ensure we import from the right place
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from utils import McpConfig
from utils.event_store import InMemoryEventStore
from tools.chroma_search import ChromaDBSearcher
from tools.web_search import WebSearcher
from tools.rag_generator import RAGResponseGenerator
from tools.rag_evaluator import RAGEvaluator

# Load environment variables
config = McpConfig(project_name="mcp_rag")

# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

# Create a logger for debugging
logger = logging.getLogger(__name__)

def log_error_with_context(e: Exception, context_message: str = "") -> None:
    """
    Generic error logging function that captures standard error information.
    
    Args:
        e: The exception object
        context_message: Optional context message to include in the log
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    
    error_type = exc_type.__name__ if exc_type else type(e).__name__
    error_message = str(exc_value) if exc_value else str(e)
    function_name = exc_traceback.tb_frame.f_code.co_name if exc_traceback else "unknown"
    line_number = exc_traceback.tb_lineno if exc_traceback else "unknown"
    
    log_message = f"{context_message} - Function: {function_name}, Line: {line_number}, Error: {error_type}: {error_message}"
    logger.error(log_message)

# Create event store for streamable HTTP
event_store = InMemoryEventStore(max_events_per_stream=1000)

# Initialize tool instances
chroma_searcher = ChromaDBSearcher(config)
web_searcher = WebSearcher(config)
rag_generator = RAGResponseGenerator(config)
rag_evaluator = RAGEvaluator(config, rag_generator)  # Pass rag_generator to avoid recreation

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
    event_store=event_store,  # Pass the event store for streamable HTTP support
    request_timeout=120.0,    # 2 minute request timeout
    read_timeout=120.0,       # 2 minute read timeout
    write_timeout=120.0       # 2 minute write timeout
)

##################################################
# Add internal resources, prompts, and tools
##################################################

# Add a tool to show detailed event information
@mcp.tool()
async def debug_event_details(count: int) -> object:
    """
    Show the last N server events for debugging (includes all tool calls and responses).
    
    Args:
        count: Number of recent events to show
        
    Returns:
        JSON formatted list of recent server events
    """
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

##################################################
# Add business resources, prompts, and tools
##################################################

@mcp.tool()
async def search_internal_products(query: str, n_results: int = 5) -> list:
    """
    Search the internal product catalog using ChromaDB.

    Args:
        query (str): What products you're looking for
        n_results (int): Maximum number of results to return (default: 5)

    Returns:
        List of dictionaries (list): Internal search results, citations, and metadata. 
    """
    search_results = await chroma_searcher.search_chroma(query, n_results)
    return search_results

@mcp.tool()
async def search_external_web(user_query: str, n_results_per_search: int = 2, internal_context: Optional[list] = None) -> list:
    """
    Search the web using Bing to find additional information.
    This tool generates web search queries based on the user question and any internal context provided and then performs web searches returning the results.
    
    Args:
        user_query (str): The original question you want to research
        n_results_per_search (int): Maximum number of results to return per search query (default: 2)
        internal_context (list): Optional internal search results to inform web query generation. This value must be in the format returned by `search_internal_products` in its entirety.

    Returns:
       List of dictionaries (list): External search results, citations, and metadata.

    """
    try: 
        search_results = await web_searcher.search_bing_with_chat_and_context(
            user_query=user_query,
            internal_context=internal_context,
            n_results_per_search=n_results_per_search
        )
        return search_results
    except Exception as e:
        log_error_with_context(e)
        raise

@mcp.tool()
async def evaluate_response(user_query: str, response: str, context: str = "") -> dict:
    """
    Singleton evaluation tool - evaluates any response for accuracy and quality.
    Semantic Kernel can call this automatically when it wants to check response quality.

    Args:
        user_query (str): The original question
        response (str): The response to evaluate
        context (str): Optional context that was used to generate the response. This should be output from the `search_internal_products` or `search_external_web` tools.

    Returns:
        dict: Evaluation results with accuracy score and feedback
    """
    try:
        evaluation = await rag_evaluator.evaluate_rag_accuracy(
            user_query=user_query,
            answer=response,
            formatted_context=context or "<no context provided>"
        )
        
        return {
            "user_query": user_query,
            "evaluated_response": response,
            "evaluation": evaluation.get("evaluation", {}),
            "tool": "evaluate_response"
        }
        
    except Exception as e:
        logger.error(f"Error in evaluate_response: {e}")
        raise RuntimeError(f"Response evaluation failed: {str(e)}") from e


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