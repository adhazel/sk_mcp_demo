"""
MCP Session Management
Clean, focused session management for MCP protocol communication.
"""

import aiohttp
import logging
from typing import Optional

from .config import MCPConfig


class MCPSession:
    """
    Manages MCP session lifecycle with proper protocol handshake.
    
    This class handles:
    - Session initialization with MCP server
    - Protocol handshake (initialize + initialized notification)
    - Session ID management
    - Connection validation
    """
    
    def __init__(self, config: MCPConfig):
        """
        Initialize MCP session manager.
        
        Args:
            config: MCP configuration object
        """
        self.config = config
        self.session_id: Optional[str] = None
        self.logger = logging.getLogger(__name__)
        self._initialized = False
    
    async def initialize(self) -> str:
        """
        Initialize MCP session with proper protocol handshake.
        
        Returns:
            Session ID for subsequent requests
            
        Raises:
            Exception: If session initialization fails
        """
        if self._initialized and self.session_id:
            return self.session_id
        
        self.logger.info(f"Initializing MCP session with {self.config.server_url}")
        
        # Step 1: Send initialize request
        init_message = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": self.config.protocol_version,
                "capabilities": self.config.capabilities,
                "clientInfo": self.config.client_info
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.mcp_endpoint,
                    json=init_message,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream"
                    },
                    timeout=self.config.request_timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Session initialization failed: {response.status} - {error_text}")
                    
                    # Extract session ID from header or generate fallback
                    self.session_id = response.headers.get('Mcp-Session-Id')
                    if not self.session_id:
                        import time
                        self.session_id = f"session_{int(time.time()*1000)}"
                        self.logger.warning(f"Generated fallback session ID: {self.session_id}")
                    
                    # Step 2: Send initialized notification to complete handshake
                    await self._send_initialized_notification()
                    
                    self._initialized = True
                    self.logger.info(f"MCP session initialized successfully: {self.session_id}")
                    return self.session_id
                    
        except Exception as e:
            self.logger.error(f"Session initialization failed: {e}")
            raise Exception(f"Could not establish MCP session: {e}")
    
    async def _send_initialized_notification(self):
        """
        Send initialized notification to complete MCP handshake.
        This is required by the MCP protocol after successful initialization.
        """
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.mcp_endpoint,
                    json=notification,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                        "Mcp-Session-Id": self.session_id
                    },
                    timeout=self.config.request_timeout
                ) as response:
                    if response.status == 200:
                        self.logger.debug("Initialized notification sent successfully (200 OK)")
                    elif response.status == 202:
                        self.logger.debug("Initialized notification accepted successfully (202 Accepted)")
                    elif response.status == 406:
                        self.logger.debug("Server doesn't support initialized notifications (406) - continuing anyway")
                    else:
                        self.logger.warning(f"Initialized notification failed: {response.status}")
                        
        except Exception as e:
            self.logger.debug(f"Failed to send initialized notification: {e} - continuing anyway")
            # Don't raise - this is a notification, not critical for functionality
    
    def is_initialized(self) -> bool:
        """Check if session is properly initialized."""
        return self._initialized and self.session_id is not None
    
    def get_session_headers(self) -> dict:
        """
        Get headers for MCP requests including session ID.
        
        Returns:
            Dictionary of headers for MCP requests
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
            
        return headers
    
    async def ensure_initialized(self) -> str:
        """
        Ensure session is initialized, initializing if necessary.
        
        Returns:
            Session ID
        """
        if not self.is_initialized():
            return await self.initialize()
        return self.session_id
