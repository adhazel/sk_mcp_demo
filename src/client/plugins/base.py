"""
Base MCP Plugin
Abstract base class for MCP client plugins across different frameworks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

from ..core.config import MCPConfig
from ..core.session import MCPSession
from ..core.discovery import MCPDiscovery, MCPPrimitive
from ..core.executor import MCPExecutor


class BaseMCPPlugin(ABC):
    """
    Abstract base class for MCP client plugins.
    
    This provides a common interface for MCP plugins across different
    frameworks (Semantic Kernel, LangChain, etc.).
    """
    
    def __init__(self, server_url: str = "http://127.0.0.1:8002"):
        """
        Initialize the base MCP plugin.
        
        Args:
            server_url: URL of the MCP server
        """
        self.config = MCPConfig(server_url=server_url)
        self.session = MCPSession(self.config)
        self.discovery = MCPDiscovery(self.session)
        self.executor = MCPExecutor(self.session)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.discovered_primitives: List[MCPPrimitive] = []
        self._initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize the plugin by discovering available primitives.
        
        Returns:
            True if initialization was successful
        """
        try:
            self.discovered_primitives = await self.discovery.discover_all()
            self._initialized = True
            
            self.logger.info(
                f"Plugin initialized with {len(self.discovered_primitives)} primitives"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin initialization failed: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the MCP connection and discover primitives.
        
        Returns:
            Health status information
        """
        try:
            if not self._initialized:
                success = await self.initialize()
                if not success:
                    return {
                        "status": "unhealthy",
                        "error": "Failed to initialize MCP connection"
                    }
            
            # Count primitives by type
            resources = [p for p in self.discovered_primitives if p.type == "resource"]
            tools = [p for p in self.discovered_primitives if p.type == "tool"]
            prompts = [p for p in self.discovered_primitives if p.type == "prompt"]
            
            return {
                "status": "healthy",
                "server_url": self.config.server_url,
                "session_id": self.session.session_id,
                "primitives": {
                    "total": len(self.discovered_primitives),
                    "resources": len(resources),
                    "tools": len(tools),
                    "prompts": len(prompts)
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    # Abstract methods that subclasses must implement
    
    @abstractmethod
    async def register_primitives(self):
        """Register discovered MCP primitives with the framework."""
        pass
    
    @abstractmethod
    def get_framework_functions(self) -> List[str]:
        """Get list of functions registered with the framework."""
        pass
    
    # Common utility methods
    
    def get_primitives_by_type(self, primitive_type: str) -> List[MCPPrimitive]:
        """Get primitives of a specific type."""
        return [p for p in self.discovered_primitives if p.type == primitive_type]
    
    def get_primitive_names(self, primitive_type: str = None) -> List[str]:
        """Get names of primitives, optionally filtered by type."""
        primitives = self.discovered_primitives
        if primitive_type:
            primitives = [p for p in primitives if p.type == primitive_type]
        return [p.name for p in primitives]
    
    async def call_tool_by_name(self, tool_name: str, **kwargs) -> Any:
        """Call a tool by name with keyword arguments."""
        return await self.executor.call_tool(tool_name, kwargs)
    
    async def read_resource_by_uri(self, uri: str) -> Any:
        """Read a resource by URI."""
        return await self.executor.read_resource(uri)
    
    async def get_prompt_by_name(self, prompt_name: str, **kwargs) -> Any:
        """Get a prompt by name with keyword arguments."""
        return await self.executor.get_prompt(prompt_name, kwargs)
