"""
Demo Script: Semantic Kernel MCP Integration
Demonstrates the new SimpleMCPPlugin with Semantic Kernel.
"""

import asyncio
from semantic_kernel import Kernel

# Import the new Semantic Kernel MCP plugin
from src.client.plugins.semantic_kernel import create_mcp_plugin


class SemanticKernelMCPDemo:
    """
    Demo class for Semantic Kernel MCP plugin integration.
    
    Shows how to use the SimpleMCPPlugin with Semantic Kernel
    in a clean, presentation-friendly way.
    """
    
    def __init__(self, server_url: str = "http://127.0.0.1:8002"):
        """Initialize the demo."""
        self.server_url = server_url
        self.kernel = None
        self.mcp_plugin = None
        
        print("🧠 Semantic Kernel MCP Integration Demo")
        print(f"   Server: {server_url}")
        print()
    
    async def setup_kernel_with_mcp(self):
        """Set up Semantic Kernel with MCP plugin."""
        print("⚙️  Setting up Semantic Kernel with MCP...")
        
        try:
            # Create kernel
            self.kernel = Kernel()
            
            # Create and initialize MCP plugin
            self.mcp_plugin = await create_mcp_plugin(self.kernel, self.server_url)
            
            print("✅ Kernel setup complete!")
            print(f"   Plugin name: {self.mcp_plugin.plugin_name}")
            print(f"   Registered functions: {len(self.mcp_plugin.registered_functions)}")
            print()
            
        except Exception as e:
            print(f"❌ Kernel setup failed: {e}")
            print()
            raise
    
    async def demonstrate_built_in_functions(self):
        """Demonstrate the built-in MCP functions."""
        print("🔍 Testing Built-in MCP Functions...")
        
        try:
            # Test health check
            print("\n1. Health Check:")
            health_result = await self.mcp_plugin.mcp_health_check()
            print(f"   {health_result}")
            
            # Test listing tools
            print("\n2. Available Tools:")
            tools_result = await self.mcp_plugin.list_available_tools()
            print(f"   {tools_result}")
            
            # Test listing resources
            print("\n3. Available Resources:")
            resources_result = await self.mcp_plugin.list_available_resources()
            print(f"   {resources_result}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error testing built-in functions: {e}")
            print()
    
    async def demonstrate_direct_tool_calling(self):
        """Demonstrate calling MCP tools directly."""
        print("🛠️  Testing Direct Tool Calling...")
        
        try:
            # Get available tools
            tools = self.mcp_plugin.get_primitive_names("tool")
            
            if not tools:
                print("   No tools available for testing")
                return
            
            # Test the first available tool
            first_tool = tools[0]
            print(f"\n   Testing tool: {first_tool}")
            
            # Call tool without parameters first
            result = await self.mcp_plugin.call_mcp_tool(first_tool)
            print(f"   Result: {result}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error testing direct tool calling: {e}")
            print()
    
    async def demonstrate_kernel_invocation(self):
        """Demonstrate invoking MCP functions through the kernel."""
        print("🎯 Testing Kernel Function Invocation...")
        
        try:
            # Get the registered functions from the plugin directly
            registered_functions = self.mcp_plugin.get_framework_functions()
            
            if not registered_functions:
                print("   No MCP functions found in kernel")
                return
            
            print(f"\n   Available MCP functions in kernel: {len(registered_functions)}")
            for func_name in registered_functions:
                print(f"   • {func_name}")
            
            # Test invoking the health check function through the plugin directly
            print("\n   Invoking health check via plugin...")
            result = await self.mcp_plugin.mcp_health_check()
            print(f"   Result: {result}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error testing kernel invocation: {e}")
            print()
    
    async def run_full_demo(self):
        """Run the complete Semantic Kernel MCP demo."""
        print("=" * 70)
        print("🎬 SEMANTIC KERNEL MCP INTEGRATION - FULL DEMO")
        print("=" * 70)
        print()
        
        try:
            # Step 1: Setup
            await self.setup_kernel_with_mcp()
            
            # Step 2: Test built-in functions
            await self.demonstrate_built_in_functions()
            
            # Step 3: Test direct tool calling
            await self.demonstrate_direct_tool_calling()
            
            # Step 4: Test kernel invocation
            await self.demonstrate_kernel_invocation()
            
            print("=" * 70)
            print("🎉 Semantic Kernel MCP Demo completed successfully!")
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            print("=" * 70)


async def main():
    """Main demo function."""
    demo = SemanticKernelMCPDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    asyncio.run(main())
