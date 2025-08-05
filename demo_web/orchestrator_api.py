"""
Orchestrator Web Demo API
Advanced FastAPI server using ProductChatAgent with web interface.
"""

import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.utils.config import Config
from src.agents.sk_product_chat_agent import ProductChatAgent
from src.client.core.config import MCPConfig
from src.client.core.session import MCPSession
from src.client.core.discovery import MCPDiscovery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
orchestrator: Optional[ProductChatAgent] = None
mcp_session: Optional[MCPSession] = None
mcp_discovery: Optional[MCPDiscovery] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global orchestrator, mcp_session, mcp_discovery
    
    # Startup
    try:
        # Initialize main config
        config = Config()
        logger.info(f"🔧 Config initialized with MCP server URL: {config.mcp_server_url}")
        
        # Initialize ProductChatAgent for intelligent orchestration
        orchestrator = ProductChatAgent(config)
        logger.info("✅ ProductChatAgent initialized")
        
        # Initialize the agent and MCP plugin right away
        await orchestrator._get_or_create_agent()
        logger.info("✅ Agent and MCP plugin initialized")
        
        # Initialize MCP client for tools discovery (supplemental)
        try:
            # Use the same server URL as the orchestrator
            mcp_config = MCPConfig(server_url=config.mcp_server_url)
            logger.info(f"🔗 Initializing MCP session with endpoint: {mcp_config.mcp_endpoint}")
            
            mcp_session = MCPSession(mcp_config)
            
            # Initialize the session with proper error handling
            session_initialized = await mcp_session.initialize()
            if not session_initialized:
                raise Exception("Session initialization returned False")
            
            logger.info("✅ MCP session initialized successfully")
            
            # Test session connectivity
            if not mcp_session.is_initialized():
                raise Exception("Session not properly initialized after initialization call")
            
            # Initialize discovery
            mcp_discovery = MCPDiscovery(mcp_session)
            primitives = await mcp_discovery.discover_all()
            logger.info(f"✅ MCP client discovery initialized - found {len(primitives)} primitives")
            
        except Exception as e:
            logger.warning(f"⚠️ MCP client discovery failed (will use orchestrator fallback): {e}")
            logger.warning(f"⚠️ Error details: {type(e).__name__}: {str(e)}")
            mcp_session = None
            mcp_discovery = None
        
        yield
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise
    finally:
        # Shutdown
        try:
            if orchestrator and hasattr(orchestrator, 'cleanup'):
                await orchestrator.cleanup()
                logger.info("✅ ProductChatAgent cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during orchestrator cleanup: {e}")
        
        # MCP Session cleanup
        try:
            if mcp_session and hasattr(mcp_session, 'close'):
                await mcp_session.close()
                logger.info("✅ MCP client session closed")
            elif mcp_session:
                logger.info("✅ MCP client session cleanup (HTTP-based, no explicit close needed)")
        except Exception as e:
            logger.error(f"❌ Error during MCP session cleanup: {e}")


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="MCP Orchestrator Web Demo",
        description="Advanced web demo using ProductChatAgent with ChatCompletionAgent and MCPStreamableHttpPlugin",
        version="2.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


# Create the app instance
app = create_app()


# Pydantic models for request/response
class ChatRequest(BaseModel):
    """Request model for intelligent chat."""
    message: str = Field(..., description="The user's message")
    # Removed use_evaluation - Semantic Kernel will decide intelligently based on the message content


class ToolCallRequest(BaseModel):
    """Request model for direct tool calls."""
    tool_name: str = Field(..., description="Name of the MCP tool to call")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


class SearchRequest(BaseModel):
    """Request model for product search."""
    query: str = Field(..., description="Search query")
    limit: int = Field(5, description="Maximum number of results", ge=1, le=50)


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    orchestrator_status: Dict[str, Any]
    mcp_server_health: str


# Web Interface Routes
@app.get("/")
async def serve_frontend():
    """Serve the orchestrator web frontend."""
    import os
    html_path = os.path.join(os.getcwd(), "demo_web", "static", "orchestrator.html")
    return FileResponse(html_path)


@app.get("/favicon.ico")
async def favicon():
    """Simple favicon to prevent 404s."""
    return {"message": "No favicon"}


@app.get("/test.html")
async def serve_test():
    """Serve the test HTML file."""
    import os
    test_path = os.path.join(os.getcwd(), "demo_web", "static", "test.html")
    return FileResponse(test_path)


@app.get("/clean")
async def serve_clean_frontend():
    """Serve the clean orchestrator web frontend."""
    import os
    html_path = os.path.join(os.getcwd(), "demo_web", "static", "orchestrator_clean.html")
    return FileResponse(html_path)


# API Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the orchestrator and MCP discovery systems."""
    global orchestrator, mcp_discovery, mcp_session
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="ProductChatAgent not initialized")
    
    try:
        orchestrator_status = orchestrator.get_status()
        
        # Check MCP discovery health with detailed status
        mcp_health = "unknown"
        mcp_details = {}
        
        if mcp_session and mcp_discovery:
            try:
                # Check session initialization
                session_initialized = mcp_session.is_initialized()
                mcp_details["session_initialized"] = session_initialized
                
                # Check if we have discovered primitives
                has_primitives = hasattr(mcp_discovery, 'primitives') and mcp_discovery.primitives
                primitive_count = len(mcp_discovery.primitives) if has_primitives else 0
                mcp_details["primitives_count"] = primitive_count
                mcp_details["has_primitives"] = has_primitives
                
                # Test session connectivity by attempting to re-initialize
                try:
                    if session_initialized:
                        # Simply check if we can ensure the session (lightweight connectivity test)
                        await mcp_session.ensure_initialized()
                        mcp_details["connectivity_test"] = "passed"
                        mcp_health = "connected" if has_primitives else "connected_no_tools"
                    else:
                        mcp_details["connectivity_test"] = "session_not_initialized"
                        mcp_health = "not_initialized"
                except Exception as conn_e:
                    logger.debug(f"Connectivity test failed: {conn_e}")
                    mcp_details["connectivity_test"] = f"failed: {str(conn_e)}"
                    mcp_details["connectivity_error"] = str(conn_e)
                    if session_initialized and has_primitives:
                        mcp_health = "connected_with_issues"
                    elif session_initialized:
                        mcp_health = "initialized_no_connectivity" 
                    else:
                        mcp_health = "connection_failed"
                        
            except Exception as e:
                logger.warning(f"MCP health check error: {e}")
                mcp_health = "error"
                mcp_details["health_check_error"] = str(e)
                
        elif mcp_session and not mcp_discovery:
            mcp_health = "session_only"
            mcp_details["session_initialized"] = mcp_session.is_initialized() if mcp_session else False
            mcp_details["discovery_available"] = False
        elif mcp_discovery and not mcp_session:
            mcp_health = "discovery_only"
            mcp_details["session_available"] = False
        else:
            mcp_health = "not_available"
            mcp_details["session_available"] = False
            mcp_details["discovery_available"] = False
        
        # Add MCP details to the orchestrator status
        orchestrator_status["mcp_details"] = mcp_details
        
        return HealthResponse(
            status="healthy" if "ready" in orchestrator_status.get("status", "") else "degraded",
            orchestrator_status=orchestrator_status,
            mcp_server_health=mcp_health
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/chat")
async def intelligent_chat(request: ChatRequest) -> Dict[str, Any]:
    """Process a chat message using the orchestrator with intelligent tool selection."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        # Check if orchestrator has the process_question method
        if not hasattr(orchestrator, 'process_question'):
            logger.warning("Orchestrator doesn't have process_question method, falling back to simple_chat")
            if hasattr(orchestrator, 'simple_chat'):
                result = await orchestrator.simple_chat(
                    question=request.message,
                    use_evaluation=request.use_evaluation
                )
            else:
                return {
                    "user_message": request.message,
                    "intent_analysis": {
                        "tool": "error",
                        "reasoning": "Orchestrator methods not available"
                    },
                    "tool_result": {
                        "status": "error",
                        "error": "No available chat methods on orchestrator"
                    },
                    "success": False,
                    "orchestrator": "ProductChatAgent"
                }
        else:
            # Use the orchestrator's process_question method for intelligent handling
            # Semantic Kernel will decide whether evaluation is needed based on the message content
            result = await orchestrator.process_question(
                question=request.message,
                use_evaluation=False  # Let SK decide through AI orchestration
            )
        
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            result = {"response": str(result), "status": "completed"}
        
        if result.get("status") == "failed":
            return {
                "user_message": request.message,
                "intent_analysis": {
                    "tool": "orchestrated_response",
                    "reasoning": "ProductChatAgent processing failed"
                },
                "tool_result": result,
                "success": False,
                "orchestrator": "ProductChatAgent"
            }
        
        # Format the response for the web interface
        formatted_result = {
            "user_message": request.message,
            "intent_analysis": {
                "tool": "orchestrated_response",
                "reasoning": "Using ProductChatAgent for intelligent tool selection and response generation"
            },
            "tool_result": result,
            "success": True,
            "orchestrator": "ProductChatAgent"
        }
        
        return formatted_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        # Return a proper JSON error response
        return {
            "user_message": request.message,
            "intent_analysis": {
                "tool": "error",
                "reasoning": f"Chat processing failed: {str(e)}"
            },
            "tool_result": {
                "status": "error",
                "error": str(e)
            },
            "success": False,
            "orchestrator": "ProductChatAgent"
        }


@app.post("/call-tool")
async def call_tool_direct(request: ToolCallRequest) -> Dict[str, Any]:
    """
    Direct MCP tool execution using the Semantic Kernel plugin's call_mcp_tool function.
    
    This bypasses the agent's process_question and directly calls the tool
    for a cleaner testing experience.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    if not orchestrator.mcp_plugin:
        # Try to initialize the MCP plugin if not already done
        try:
            await orchestrator._get_or_create_agent()
            logger.info("✅ Successfully initialized MCP plugin on demand")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP plugin on demand: {e}")
            raise HTTPException(status_code=503, detail=f"MCP plugin not initialized and could not be initialized: {str(e)}")
    
    try:
        # Format parameters as JSON string for call_mcp_tool
        import json
        parameters_json = json.dumps(request.parameters) if request.parameters else None
        
        # Directly call the MCP tool through the plugin's call_mcp_tool function
        result = await orchestrator.mcp_plugin.call_mcp_tool(
            tool_name=request.tool_name,
            parameters=parameters_json
        )
        
        # Convert string result to dictionary if possible
        try:
            parsed_result = json.loads(result) if isinstance(result, str) else result
        except (json.JSONDecodeError, TypeError):
            parsed_result = {"result": result}
        
        # Add metadata to show this was direct execution
        if isinstance(parsed_result, dict):
            parsed_result["execution_method"] = "direct_mcp_tool_call"
            parsed_result["tool_name"] = request.tool_name
            parsed_result["tool_parameters"] = request.parameters
            
        return parsed_result
        
    except Exception as e:
        logger.error(f"❌ Error calling MCP tool directly: {e}")
        return {
            "status": "error",
            "error": str(e),
            "tool_name": request.tool_name,
            "execution_method": "direct_mcp_tool_call_failed"
        }
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI orchestration failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI orchestration failed: {str(e)}")


@app.get("/tools")
async def get_available_tools() -> Dict[str, Any]:
    """Get information about available MCP tools using the MCP discovery system."""
    global mcp_discovery, orchestrator
    
    # Try MCP discovery first (preferred method for tools listing)
    if mcp_discovery:
        try:
            tools_list = []
            
            # Get tools from MCP discovery primitives
            if hasattr(mcp_discovery, 'primitives') and mcp_discovery.primitives:
                for primitive in mcp_discovery.primitives:
                    if primitive.type == "tool":
                        tools_list.append({
                            "name": primitive.name,
                            "description": primitive.description or "No description available",
                            "schema": primitive.schema or {}
                        })
            
            return {
                "tools": tools_list,
                "total_count": len(tools_list),
                "source": "mcp_discovery",
                "mcp_discovery_info": {
                    "status": "connected",
                    "tools_count": len(tools_list),
                    "total_primitives": len(mcp_discovery.primitives) if hasattr(mcp_discovery, 'primitives') else 0,
                    "session_id": getattr(mcp_discovery.session, 'session_id', 'unknown') if hasattr(mcp_discovery, 'session') else 'unknown'
                }
            }
            
        except Exception as e:
            logger.error(f"MCP discovery failed: {e}")
            # Fall through to orchestrator fallback
    
    # Fallback to orchestrator method
    if not orchestrator:
        return {
            "tools": [],
            "total_count": 0,
            "error": "Neither MCP discovery nor orchestrator available",
            "source": "none"
        }
    
    try:
        # Check if orchestrator has the get_available_functions method
        if not hasattr(orchestrator, 'get_available_functions'):
            logger.warning("Orchestrator doesn't have get_available_functions method")
            return {
                "tools": [],
                "total_count": 0,
                "source": "orchestrator_fallback",
                "orchestrator_info": {
                    "status": "method_not_available",
                    "message": "get_available_functions method not available on orchestrator"
                }
            }
        
        functions_info = await orchestrator.get_available_functions()
        
        # Handle case where functions_info might be None or not a dict
        if not functions_info or not isinstance(functions_info, dict):
            logger.warning(f"Invalid functions_info received: {functions_info}")
            return {
                "tools": [],
                "total_count": 0,
                "source": "orchestrator_fallback",
                "orchestrator_info": {
                    "status": "no_functions",
                    "message": "No function information available"
                }
            }
        
        # Convert to the format expected by the web interface
        tools_list = []
        available_functions = functions_info.get("available_functions", {})
        
        if isinstance(available_functions, dict):
            for func_name, description in available_functions.items():
                tools_list.append({
                    "name": func_name,
                    "description": str(description) if description else "No description available",
                    "schema": {}  # The orchestrator doesn't expose detailed schemas
                })
        
        return {
            "tools": tools_list,
            "total_count": len(tools_list),
            "source": "orchestrator_fallback",
            "orchestrator_info": functions_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get tools from orchestrator: {e}")
        # Return a proper JSON error response instead of raising HTTPException
        return {
            "tools": [],
            "total_count": 0,
            "error": str(e),
            "source": "error",
            "orchestrator_info": {
                "status": "error",
                "message": f"Error getting tools: {str(e)}"
            }
        }


@app.post("/search")
async def search_products(request: SearchRequest) -> Dict[str, Any]:
    """Search for products in the internal database through the orchestrator."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        result = await orchestrator.simple_search(request.query, request.limit)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get the current status of the orchestrator."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        return orchestrator.get_status()
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


# Root API info endpoint
@app.get("/api")
async def api_info():
    """API information and available endpoints."""
    return {
        "message": "MCP Orchestrator Web Demo API",
        "version": "2.0.0",
        "approach": "ProductChatAgent → ChatCompletionAgent → MCPStreamableHttpPlugin → MCP Server",
        "endpoints": {
            "GET /": "Web interface",
            "GET /health": "System health check",
            "POST /chat": "Intelligent chat with automatic tool selection",
            "POST /call-tool": "Direct tool execution",
            "GET /tools": "List available MCP tools",
            "POST /search": "Search internal products",
            "GET /status": "Orchestrator status"
        },
        "features": [
            "Semantic Kernel Agents",
            "Intelligent tool orchestration",
            "Conversation thread management",
            "Production-ready error handling",
            "Advanced timeout handling"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting MCP Orchestrator Web Demo...")
    print("📱 Open your browser to: http://localhost:8001")
    print("🔧 Make sure your MCP server is running on http://127.0.0.1:8002")
    print("🧠 Using: ProductChatAgent → ChatCompletionAgent → MCPStreamableHttpPlugin")
    print()
    
    uvicorn.run(
        "demo_web.orchestrator_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
