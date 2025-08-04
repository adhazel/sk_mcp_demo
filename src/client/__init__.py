"""
MCP Client Package
Provides clean, modular MCP (Model Context Protocol) client functionality.
"""

from .core.session import MCPSession
from .core.discovery import MCPDiscovery  
from .core.executor import MCPExecutor
from .core.config import MCPConfig
from .plugins.semantic_kernel import SimpleMCPPlugin

# Version info
__version__ = "1.0.0"
__author__ = "MCP Demo Team"

# Public API
__all__ = [
    "MCPSession",
    "MCPDiscovery", 
    "MCPExecutor",
    "MCPConfig",
    "SimpleMCPPlugin"
]
