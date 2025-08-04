"""
Semantic Kernel MCP Plugin
Simple, demo-friendly MCP client plugin for Microsoft Semantic Kernel.
"""

from typing import List, Optional

from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function, KernelArguments

from .base import BaseMCPPlugin
from ..core.discovery import MCPPrimitive


class SimpleMCPPlugin(BaseMCPPlugin):
    """
    Simple MCP plugin for Semantic Kernel integration.
    
    This class provides an easy-to-understand integration between
    MCP servers and Semantic Kernel, perfect for demos and tutorials.
    """
    
    def __init__(self, kernel: Kernel, server_url: str = "http://127.0.0.1:8002"):
        """
        Initialize the Semantic Kernel MCP plugin.
        
        Args:
            kernel: The Semantic Kernel instance
            server_url: URL of the MCP server
        """
        super().__init__(server_url)
        self.kernel = kernel
        self.registered_functions: List[str] = []
        
        # Add this plugin to the kernel
        self.plugin_name = "MCPTools"
        self.kernel.add_plugin(self, plugin_name=self.plugin_name)
    
    async def register_primitives(self):
        """
        Register discovered MCP primitives as Semantic Kernel functions.
        
        This creates kernel functions for each MCP tool that can be
        called directly by the kernel.
        """
        tools = self.get_primitives_by_type("tool")
        
        for tool in tools:
            await self._register_tool_as_function(tool)
        
        self.logger.info(f"Registered {len(tools)} tools as kernel functions")
    
    async def _register_tool_as_function(self, tool: MCPPrimitive):
        """Register a single MCP tool as a kernel function."""
        function_name = f"mcp_{tool.name.replace('-', '_')}"
        
        # Create a dynamic kernel function
        async def tool_function(arguments: KernelArguments) -> str:
            """Dynamically created function for MCP tool."""
            try:
                # Convert KernelArguments to dict
                params = {}
                for key, value in arguments:
                    params[key] = value
                
                result = await self.executor.call_tool(tool.name, params)
                
                # Format result for kernel consumption
                if isinstance(result, dict):
                    return str(result.get('content', result))
                return str(result)
                
            except Exception as e:
                error_msg = f"Error calling {tool.name}: {str(e)}"
                self.logger.error(error_msg)
                return error_msg
        
        # Set function metadata
        tool_function.__name__ = function_name
        tool_function.__doc__ = tool.description or f"MCP tool: {tool.name}"
        
        # Register with kernel (simplified approach)
        self.registered_functions.append(function_name)
    
    def get_framework_functions(self) -> List[str]:
        """Get list of functions registered with Semantic Kernel."""
        return self.registered_functions.copy()
    
    # Demo-friendly kernel functions
    
    @kernel_function(
        name="list_available_tools",
        description="List all available MCP tools"
    )
    async def list_available_tools(self) -> str:
        """List available MCP tools for demo purposes."""
        if not self._initialized:
            await self.initialize()
        
        tools = self.get_primitives_by_type("tool")
        
        if not tools:
            return "No MCP tools available"
        
        tool_list = []
        for tool in tools:
            tool_list.append(f"- {tool.name}: {tool.description or 'No description'}")
        
        return f"Available MCP Tools ({len(tools)}):\n" + "\n".join(tool_list)
    
    @kernel_function(
        name="list_available_resources",
        description="List all available MCP resources"
    )
    async def list_available_resources(self) -> str:
        """List available MCP resources for demo purposes."""
        if not self._initialized:
            await self.initialize()
        
        resources = self.get_primitives_by_type("resource")
        
        if not resources:
            return "No MCP resources available"
        
        resource_list = []
        for resource in resources:
            resource_list.append(f"- {resource.name}: {resource.description or 'No description'}")
        
        return f"Available MCP Resources ({len(resources)}):\n" + "\n".join(resource_list)
    
    @kernel_function(
        name="mcp_health_check",
        description="Check MCP server health and connection status"
    )
    async def mcp_health_check(self) -> str:
        """Perform MCP health check for demo purposes."""
        health = await self.health_check()
        
        if health["status"] == "healthy":
            return f"""MCP Health Check: ✅ HEALTHY
Server: {health['server_url']}
Session ID: {health['session_id']}
Total Primitives: {health['primitives']['total']}
- Tools: {health['primitives']['tools']}
- Resources: {health['primitives']['resources']}
- Prompts: {health['primitives']['prompts']}"""
        else:
            return f"MCP Health Check: ❌ UNHEALTHY - {health.get('error', 'Unknown error')}"
    
    @kernel_function(
        name="call_mcp_tool",
        description="Call an MCP tool by name with parameters"
    )
    async def call_mcp_tool(
        self, 
        tool_name: str, 
        parameters: Optional[str] = None
    ) -> str:
        """
        Call an MCP tool by name.
        
        Args:
            tool_name: Name of the MCP tool to call
            parameters: JSON string of parameters (optional)
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Parse parameters if provided
            params = {}
            if parameters:
                import json
                try:
                    params = json.loads(parameters)
                except json.JSONDecodeError:
                    return f"Error: Invalid JSON parameters: {parameters}"
            
            result = await self.call_tool_by_name(tool_name, **params)
            return str(result)
            
        except Exception as e:
            error_msg = f"Error calling tool '{tool_name}': {str(e)}"
            self.logger.error(error_msg)
            return error_msg


# Convenience function for demo setup
async def create_mcp_plugin(kernel: Kernel, server_url: str = "http://127.0.0.1:8002") -> SimpleMCPPlugin:
    """
    Create and initialize an MCP plugin for Semantic Kernel.
    
    This is a convenience function for demo purposes.
    
    Args:
        kernel: The Semantic Kernel instance
        server_url: URL of the MCP server
        
    Returns:
        Initialized SimpleMCPPlugin instance
    """
    plugin = SimpleMCPPlugin(kernel, server_url)
    
    success = await plugin.initialize()
    if not success:
        raise RuntimeError("Failed to initialize MCP plugin")
    
    # Register primitives as kernel functions
    await plugin.register_primitives()
    
    return plugin
