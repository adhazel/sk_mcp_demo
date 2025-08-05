"""
MCP Discovery - Simplified for demo purposes
Discovers available MCP primitives (resources, tools, prompts) from servers.
"""

import aiohttp
import json
import logging
from typing import Dict, Any, List
from dataclasses import dataclass

from .session import MCPSession


@dataclass
class MCPPrimitive:
    """Represents a discovered MCP primitive (resource, tool, or prompt)."""
    name: str
    type: str  # 'resource', 'tool', or 'prompt'
    description: str
    schema: Dict[str, Any] = None


class MCPDiscovery:
    """Simplified MCP primitive discovery for demo purposes."""
    
    def __init__(self, session: MCPSession):
        self.session = session
        self.logger = logging.getLogger(__name__)
        self.primitives: List[MCPPrimitive] = []
    
    async def discover_all(self) -> List[MCPPrimitive]:
        """Discover all available MCP primitives."""
        await self.session.ensure_initialized()
        
        # Discover all types sequentially to avoid interference
        self.primitives = []
        for ptype in ["resources", "tools", "prompts"]:
            try:
                items = await self._discover_type(ptype)
                self.primitives.extend(items)
                self.logger.info(f"Discovered {len(items)} {ptype}")
            except Exception as e:
                self.logger.error(f"Failed to discover {ptype}: {e}")
        
        self.logger.info(f"Discovered {len(self.primitives)} MCP primitives total")
        return self.primitives
    
    async def _discover_type(self, primitive_type: str) -> List[MCPPrimitive]:
        """Discover primitives of a specific type with streaming support."""
        method = f"{primitive_type}/list"
        response = await self._make_streaming_request(method, {})
        items = response.get(primitive_type, [])
        
        result = []
        for item in items:
            # Simplified primitive creation
            name = item.get("uri" if primitive_type == "resources" else "name", "unknown")
            ptype = "resource" if primitive_type == "resources" else primitive_type[:-1]
            
            result.append(MCPPrimitive(
                name=name,
                type=ptype,
                description=item.get("description", f"MCP {ptype}: {name}"),
                schema=item
            ))
        
        return result
    
    async def _make_streaming_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a streaming request to the MCP server."""
        # Session should already be initialized by discover_all()
        if not self.session.is_initialized():
            await self.session.ensure_initialized()
        
        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        self.logger.debug(f"Making request to {method} with params: {params}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.session.config.mcp_endpoint,
                json=message,
                headers=self.session.get_session_headers(),
                timeout=self.session.config.request_timeout
            ) as response:
                self.logger.debug(f"Response status: {response.status}")
                self.logger.debug(f"Response headers: {dict(response.headers)}")
                
                response.raise_for_status()
                
                # Handle streaming response
                if response.headers.get('content-type', '').startswith('text/event-stream'):
                    result = await self._parse_streaming_response(response)
                    self.logger.debug(f"Streaming response result: {result}")
                    return result
                else:
                    # Fallback to regular JSON response
                    data = await response.json()
                    self.logger.debug(f"JSON response data: {data}")
                    return data.get("result", {})
    
    async def _parse_streaming_response(self, response) -> Dict[str, Any]:
        """Parse streaming response for demo - simplified version."""
        result = {}
        try:
            async for line in response.content:
                line_str = line.decode('utf-8').strip()
                self.logger.debug(f"Streaming line: {repr(line_str)}")
                if line_str.startswith('data: ') and len(line_str) > 6:
                    try:
                        json_str = line_str[6:]  # Remove 'data: ' prefix
                        data = json.loads(json_str)
                        self.logger.debug(f"Parsed streaming data: {data}")
                        if 'result' in data:
                            return data['result']
                        result.update(data)
                    except json.JSONDecodeError as e:
                        self.logger.debug(f"JSON decode error for line '{json_str}': {e}")
                        continue
        except Exception as e:
            self.logger.error(f"Error parsing streaming response: {e}")
        
        self.logger.debug(f"Final streaming result: {result}")
        return result
    
    def get_by_type(self, primitive_type: str) -> List[MCPPrimitive]:
        """Get primitives of a specific type."""
        return [p for p in self.primitives if p.type == primitive_type]
    
    def find_by_name(self, name: str) -> MCPPrimitive:
        """Find a primitive by name."""
        for primitive in self.primitives:
            if primitive.name == name:
                return primitive
        raise ValueError(f"Primitive '{name}' not found")
