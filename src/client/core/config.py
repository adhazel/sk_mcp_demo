"""
MCP Configuration
Clean configuration management for MCP clients.
"""

from dataclasses import dataclass


@dataclass
class MCPConfig:
    """Configuration for MCP client operations."""
    
    # Server connection
    server_url: str = "http://127.0.0.1:8002"
    request_timeout: int = 30
    
    # MCP protocol
    protocol_version: str = "2024-11-05"
    client_name: str = "semantic-kernel-mcp-client"
    client_version: str = "1.0.0"
    
    # Capabilities
    supports_resources: bool = True
    supports_tools: bool = True
    supports_prompts: bool = True
    
    # Logging
    log_level: str = "INFO"
    debug_mode: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_url:
            raise ValueError("server_url is required")
        
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        
        # Ensure server_url doesn't end with slash
        self.server_url = self.server_url.rstrip('/')
    
    @property
    def mcp_endpoint(self) -> str:
        """Get the full MCP endpoint URL."""
        return f"{self.server_url}/mcp"
    
    @property
    def client_info(self) -> dict:
        """Get client info for MCP initialization."""
        return {
            "name": self.client_name,
            "version": self.client_version
        }
    
    @property
    def capabilities(self) -> dict:
        """Get capabilities for MCP initialization."""
        caps = {}
        if self.supports_resources:
            caps["resources"] = {}
        if self.supports_tools:
            caps["tools"] = {}
        if self.supports_prompts:
            caps["prompts"] = {}
        return caps
