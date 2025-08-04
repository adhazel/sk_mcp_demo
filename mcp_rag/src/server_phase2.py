"""
MCP server with RAG tools for internal and external search.
"""
from mcp.server.fastmcp import FastMCP
import logging
from src.utils import mcp_config as config
from src.utils.event_store import InMemoryEventStore
from src.tools.chroma_search import ChromaDBSearcher
from src.tools.web_search import WebSearcher
from src.tools.rag_generator import RAGResponseGenerator
from src.tools.rag_evaluator import RAGEvaluator
import asyncio
import json
import time
import sys

# Load environment variables
config = config.Config(project_name="mcp_rag")

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

# @mcp.tool()
# def resume_from_event_id(last_event_id: str) -> str:
#     """Manually resume from a specific event ID"""
#     # Your app can implement custom resumption logic

##################################################
# Add business resources, prompts, and tools
##################################################

@mcp.tool()
async def search_internal_products(query: str, n_results: int = 5) -> list:
    """
    Search the internal product catalog using ChromaDB.

    Args:
        query: What products you're looking for
        n_results: Maximum number of results to return (default: 5)
        
    Returns:
        List of matching products with descriptions and metadata
    """
    search_results = await chroma_searcher.search_chroma(query, n_results)
    return search_results

@mcp.tool()
async def generate_web_search_queries(user_query: str, internal_context: list = None) -> object:
    """
    Generate smart web search queries to find additional information online.
    
    Args:
        user_query: The original question you want to research
        internal_context: Optional internal search results to improve web queries
        
    Returns:
        List of optimized search queries for web searching
    """
    search_queries = await web_searcher.get_web_search_queries(user_query, internal_context)
    return search_queries

@mcp.tool()
async def web_search(generated_queries: object, n_results: int = 5) -> object:
    """
    Search the web using Bing to find additional information.
    
    Args:
        generated_queries: Search queries (from generate_web_search_queries tool)
        n_results: Maximum results per search query (default: 5)
        
    Returns:
        Web search results with content, links, and metadata
    """
    try: 
        search_results = await web_searcher.search_serpapi_bing_with_generated_queries(generated_queries, n_results)
        return search_results
    except Exception as e:
        log_error_with_context(e)
        raise

@mcp.tool()
async def generate_chat_response(user_query: str, n_chroma_results: int = 5,n_web_results: int = 5, collection_name: str = "product_collection") -> object:
    """
    Generate a comprehensive answer by searching internal knowledge and the web.
    This combines internal product search, web research, and AI response generation.

    Args:
        user_query: Your question
        n_chroma_results: Max internal search results to use (default: 5)
        n_web_results: Max web search results per query (default: 5)
        collection_name: Internal database collection (default: "product_collection")

    Returns:
        Complete answer with sources and citations
    """
    start_time = time.time()
    
    try:
        # Set timeout to 2 minutes for RAG generation
        rag_response = await asyncio.wait_for(
            rag_generator.generate_chat_response(
                user_query=user_query,
                n_chroma_results=n_chroma_results,
                n_web_results=n_web_results,
                collection_name=collection_name
            ),
            timeout=120.0
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"RAG generation completed in {elapsed_time:.2f} seconds")
        return rag_response
        
    except asyncio.TimeoutError:
        elapsed_time = time.time() - start_time
        logger.error(f"RAG generation timed out after {elapsed_time:.2f} seconds (limit: 120s)")
        raise (asyncio.TimeoutError(
            f"Request timed out - RAG operation took longer than {elapsed_time:.2f} seconds"
        ))
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"RAG generation failed after {elapsed_time:.2f} seconds: {e}")
        raise (e)

@mcp.tool()
async def generate_evaluated_chat_response(user_query: str, n_chroma_results: int = 5,n_web_results: int = 5, collection_name: str = "product_collection") -> object:
    """
    Generate a comprehensive answer with quality evaluation included.
    This searches internal knowledge and the web, then evaluates the response accuracy.

    Args:
        user_query: Your question
        n_chroma_results: Max internal search results to use (default: 5)
        n_web_results: Max web search results per query (default: 5)
        collection_name: Internal database collection (default: "product_collection")

    Returns:
        Complete answer with sources, citations, and accuracy evaluation
    """
    start_time = time.time()
    
    try:
        # Set timeout to 2 minutes for RAG with evaluation
        rag_response = await asyncio.wait_for(
            rag_generator.generate_evaluated_chat_response(
                user_query=user_query,
                n_chroma_results=n_chroma_results,
                n_web_results=n_web_results,
                collection_name=collection_name
            ),
            timeout=120.0
        )

        elapsed_time = time.time() - start_time
        logger.info(f"RAG with evaluation completed in {elapsed_time:.2f} seconds")
        return rag_response
        
    except asyncio.TimeoutError:
        elapsed_time = time.time() - start_time
        logger.error(f"RAG with evaluation timed out after {elapsed_time:.2f} seconds (limit: 120s)")
        raise (asyncio.TimeoutError(
            f"Request timed out - RAG with evaluation took longer than {elapsed_time:.2f} seconds"
        ))
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"RAG with evaluation failed after {elapsed_time:.2f} seconds: {e}")
        raise (e)


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