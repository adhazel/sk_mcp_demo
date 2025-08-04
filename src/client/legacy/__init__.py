"""
Legacy Compatibility Layer
Maintains compatibility with existing FastMCP client plugin usage.
"""

# Import the original plugin class for backward compatibility
from ...services.fastmcp_client_plugin import FastMCPClientPlugin

# Convenience imports for common usage patterns
from ..core.config import MCPConfig
from ..core.session import MCPSession
from ..core.discovery import MCPDiscovery, MCPPrimitive
from ..core.executor import MCPExecutor

__all__ = [
    'FastMCPClientPlugin',
    'MCPConfig', 
    'MCPSession',
    'MCPDiscovery',
    'MCPPrimitive',
    'MCPExecutor'
]
