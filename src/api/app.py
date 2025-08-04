"""
FastAPI application for the Semantic Kernel MCP Orchestrator.
Provides REST API endpoints to interact with the orchestrator.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.config import Config
from src.agents.sk_orchestrator import SemanticKernelOrchestrator


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator: Optional[SemanticKernelOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global orchestrator
    
    # Startup
    try:
        config = Config()
        orchestrator = SemanticKernelOrchestrator(config)
        logger.info("✅ Orchestrator initialized")
        yield
    except Exception as e:
        logger.error(f"❌ Failed to initialize orchestrator: {e}")
        raise
    finally:
        # Shutdown
        if orchestrator and hasattr(orchestrator, 'mcp_plugin'):
            try:
                await orchestrator.mcp_plugin.close()
                logger.info("✅ Orchestrator cleanup completed")
            except Exception as e:
                logger.error(f"❌ Error during cleanup: {e}")


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Semantic Kernel MCP Orchestrator API",
        description="REST API for the Semantic Kernel orchestrator that calls MCP RAG server",
        version="1.0.0",
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
class QuestionRequest(BaseModel):
    """Request model for question processing."""
    question: str = Field(..., description="The user's question")
    context: str = Field("", description="Additional context for the question")
    use_evaluation: bool = Field(True, description="Whether to use evaluation and risk scoring")


class SearchRequest(BaseModel):
    """Request model for product search."""
    query: str = Field(..., description="Search query")
    limit: int = Field(5, description="Maximum number of results", ge=1, le=50)


class ChatRequest(BaseModel):
    """Request model for simple chat."""
    question: str = Field(..., description="The user's question")
    context: str = Field("", description="Additional context")
    use_web_search: bool = Field(True, description="Whether to use web search")
    use_evaluation: bool = Field(False, description="Whether to use evaluation")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    orchestrator_status: Dict[str, Any]
    mcp_server_health: str


# API Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the orchestrator and MCP server."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        orchestrator_status = orchestrator.get_status()
        mcp_health = await orchestrator.mcp_plugin.health_check()
        
        return HealthResponse(
            status="healthy" if "accessible" in mcp_health else "degraded",
            orchestrator_status=orchestrator_status,
            mcp_server_health=mcp_health
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/ask")
async def ask_question(request: QuestionRequest) -> Dict[str, Any]:
    """Process a question using the full orchestrator capabilities."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        result = await orchestrator.process_question(
            question=request.question,
            context=request.context,
            use_evaluation=request.use_evaluation
        )
        
        if result.get("status") == "failed":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/search")
async def search_products(request: SearchRequest) -> Dict[str, Any]:
    """Search for products in the internal database."""
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


@app.post("/chat")
async def simple_chat(request: ChatRequest) -> Dict[str, Any]:
    """Generate a simple chat response without full orchestration."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        result = await orchestrator.simple_chat(
            question=request.question,
            context=request.context,
            use_web_search=request.use_web_search,
            use_evaluation=request.use_evaluation
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/functions")
async def get_functions() -> Dict[str, Any]:
    """Get information about available MCP functions."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        return await orchestrator.get_available_functions()
    except Exception as e:
        logger.error(f"Failed to get functions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get functions: {str(e)}")


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


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Semantic Kernel MCP Orchestrator API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health - Check system health",
            "ask": "POST /ask - Process question with full orchestration",
            "search": "POST /search - Search internal products",
            "chat": "POST /chat - Simple chat response",
            "functions": "GET /functions - List available functions",
            "status": "GET /status - Get orchestrator status"
        },
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    # For development only
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
