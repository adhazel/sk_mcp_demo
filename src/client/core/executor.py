"""
MCP Executor
Executes MCP operations (tool calls, resource reads, prompt gets) with clean error handling.
"""

import aiohttp
import json
import logging
from typing import Dict, Any, Union

from .session import MCPSession


class MCPExecutor:
    """
    Executes MCP operations with clean error handling and response formatting.
    
    This class handles:
    - Tool execution (tools/call)
    - Resource reading (resources/read)
    - Prompt retrieval (prompts/get)
    - Response parsing and formatting
    - Error handling and logging
    """
    
    def __init__(self, session: MCPSession):
        """
        Initialize MCP executor.
        
        Args:
            session: Initialized MCP session
        """
        self.session = session
        self.logger = logging.getLogger(__name__)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            Exception: If tool execution fails
        """
        if arguments is None:
            arguments = {}
        
        await self.session.ensure_initialized()
        
        try:
            result = await self._make_request(
                method="tools/call",
                params={
                    "name": tool_name,
                    "arguments": arguments
                }
            )
            
            self.logger.info(f"Successfully executed tool '{tool_name}'")
            return result
            
        except Exception as e:
            error_msg = f"Failed to execute tool '{tool_name}': {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a resource from the MCP server.
        
        Args:
            uri: URI of the resource to read
            
        Returns:
            Resource content
            
        Raises:
            Exception: If resource reading fails
        """
        await self.session.ensure_initialized()
        
        try:
            result = await self._make_request(
                method="resources/read",
                params={"uri": uri}
            )
            
            self.logger.info(f"Successfully read resource '{uri}'")
            return result
            
        except Exception as e:
            error_msg = f"Failed to read resource '{uri}': {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    async def get_prompt(self, prompt_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Retrieve a prompt from the MCP server.
        
        Args:
            prompt_name: Name of the prompt to retrieve
            arguments: Arguments to pass to the prompt
            
        Returns:
            Prompt content
            
        Raises:
            Exception: If prompt retrieval fails
        """
        if arguments is None:
            arguments = {}
        
        await self.session.ensure_initialized()
        
        try:
            result = await self._make_request(
                method="prompts/get",
                params={
                    "name": prompt_name,
                    "arguments": arguments
                }
            )
            
            self.logger.info(f"Successfully retrieved prompt '{prompt_name}'")
            return result
            
        except Exception as e:
            error_msg = f"Failed to retrieve prompt '{prompt_name}': {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    async def _make_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a JSON-RPC request to the MCP server.
        
        Args:
            method: MCP method to call
            params: Parameters for the method
            
        Returns:
            Response data from the server
            
        Raises:
            Exception: If request fails
        """
        message = {
            "jsonrpc": "2.0",
            "id": hash(method + str(params)) % 10000,  # Simple ID generation
            "method": method,
            "params": params
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.session.config.mcp_endpoint,
                json=message,
                headers=self.session.get_session_headers(),
                timeout=self.session.config.request_timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"MCP request failed: {response.status} - {error_text}")
                
                response_text = await response.text()
                self.logger.debug(f"MCP response for {method}: {response_text}")
                
                return self._parse_response(response_text)
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse MCP server response (handles both JSON and Server-Sent Events).
        
        Args:
            response_text: Raw response text
            
        Returns:
            Parsed response data
            
        Raises:
            Exception: If response contains an error or cannot be parsed
        """
        # Try to parse as Server-Sent Events first
        lines = response_text.strip().split('\n')
        for line in lines:
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                    if 'result' in data:
                        return data['result']
                    elif 'error' in data:
                        raise Exception(f"MCP server error: {data['error']}")
                except json.JSONDecodeError:
                    continue
        
        # Fallback: try to parse as direct JSON
        try:
            data = json.loads(response_text)
            if 'result' in data:
                return data['result']
            elif 'error' in data:
                raise Exception(f"MCP server error: {data['error']}")
            return data
        except json.JSONDecodeError:
            raise Exception(f"Failed to parse MCP response: {response_text[:200]}...")
    
    def format_result_for_display(self, result: Union[Dict[str, Any], str], max_length: int = 1000) -> str:
        """
        Format execution result for demo/display purposes.
        
        Args:
            result: Result from MCP operation
            max_length: Maximum length of formatted output
            
        Returns:
            Nicely formatted string for display
        """
        if isinstance(result, str):
            return result[:max_length] + ("..." if len(result) > max_length else "")
        
        if isinstance(result, dict):
            # Handle common result patterns
            if "error" in result:
                return f"❌ Error: {result['error']}"
            
            if "content" in result and len(result) == 1:
                # Simple content response
                content = result["content"]
                if isinstance(content, str):
                    return content[:max_length] + ("..." if len(content) > max_length else "")
            
            # Pretty print JSON result
            try:
                formatted = json.dumps(result, indent=2, ensure_ascii=False)
                return formatted[:max_length] + ("..." if len(formatted) > max_length else "")
            except (TypeError, ValueError):
                return str(result)[:max_length]
        
        # Fallback for other types
        result_str = str(result)
        return result_str[:max_length] + ("..." if len(result_str) > max_length else "")
    
    def extract_key_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key information from MCP results for summary display.
        
        Args:
            result: Result from MCP operation
            
        Returns:
            Dictionary with key information
        """
        info = {"type": type(result).__name__}
        
        if isinstance(result, dict):
            # Count keys
            info["keys"] = list(result.keys())[:5]  # First 5 keys
            info["key_count"] = len(result)
            
            # Check for common patterns
            if "content" in result:
                content = result["content"]
                info["content_type"] = type(content).__name__
                if isinstance(content, str):
                    info["content_length"] = len(content)
                elif isinstance(content, list):
                    info["content_items"] = len(content)
            
            if "total_events" in result:
                info["events"] = {
                    "total": result.get("total_events"),
                    "shown": result.get("showing_last")
                }
        
        elif isinstance(result, list):
            info["item_count"] = len(result)
            if result:
                info["first_item_type"] = type(result[0]).__name__
        
        return info
