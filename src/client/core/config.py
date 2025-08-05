"""
MCP Configuration
Clean configuration management for MCP clients.
"""

from dataclasses import dataclass
from ...utils.config import Config


@dataclass
class MCPConfig:
    """Configuration for MCP client operations."""
    
    def __init__(self, server_url: str = None):
        """Initialize MCP config, using main config if server_url not provided."""
        if server_url is None:
            main_config = Config()
            server_url = main_config.mcp_server_url
            
        # Server connection
        self.server_url: str = server_url
        self.request_timeout: int = 30
        
        # MCP protocol
        self.protocol_version: str = "2024-11-05"
        self.client_name: str = "semantic-kernel-mcp-client"
        self.client_version: str = "1.0.0"
        
        # Capabilities
        self.supports_resources: bool = True
        self.supports_tools: bool = True
        self.supports_prompts: bool = True
        
        # Logging
        self.log_level: str = "INFO"
        self.debug_mode: bool = False
        
        # Validate after initialization
        self.__post_init__()
    
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
