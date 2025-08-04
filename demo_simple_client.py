"""
Demo Script: Simple MCP Client Testing
Clean, well-documented script for demonstrating MCP client functionality.
"""

import asyncio
import json
import logging
from typing import Dict, Any

# Import the new modular MCP client components
from src.client.core.config import MCPConfig
from src.client.core.session import MCPSession
from src.client.core.discovery import MCPDiscovery
from src.client.core.executor import MCPExecutor

# Set up debug logging
logging.basicConfig(level=logging.DEBUG)


class MCPClientDemo:
    """
    Simple demo class for testing MCP client functionality.
    
    This class demonstrates the modular MCP client architecture
    in a clean, easy-to-follow manner perfect for presentations.
    """
    
    def __init__(self, server_url: str = "http://127.0.0.1:8002"):
        """Initialize the demo with MCP server URL."""
        self.server_url = server_url
        print("Initializing MCP Client Demo")
        print(f"   Server: {server_url}")
        print()
    
    async def run_health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the MCP connection.
        
        Returns:
            Health status information
        """
        print("Running MCP Health Check...")
        
        try:
            # Initialize core components
            config = MCPConfig(server_url=self.server_url)
            session = MCPSession(config)
            discovery = MCPDiscovery(session)
            
            # Discover all available primitives
            primitives = await discovery.discover_all()
            
            # Count primitives by type
            resources = [p for p in primitives if p.type == "resource"]
            tools = [p for p in primitives if p.type == "tool"]
            prompts = [p for p in primitives if p.type == "prompt"]
            
            health_status = {
                "status": "healthy",
                "server_url": config.server_url,
                "session_id": session.session_id,
                "primitives": {
                    "total": len(primitives),
                    "resources": len(resources),
                    "tools": len(tools),
                    "prompts": len(prompts)
                }
            }
            
            # Pretty print results
            print("MCP Connection: HEALTHY")
            print(f"   Session ID: {session.session_id}")
            print(f"   Total Primitives: {len(primitives)}")
            print(f"   - Resources: {len(resources)}")
            print(f"   - Tools: {len(tools)}")
            print(f"   - Prompts: {len(prompts)}")
            print()
            
            return health_status
            
        except Exception as e:
            error_status = {
                "status": "unhealthy",
                "error": str(e)
            }
            
            print("❌ MCP Connection: UNHEALTHY")
            print(f"   Error: {e}")
            print()
            
            return error_status
    
    async def list_available_primitives(self):
        """List and display all available MCP primitives."""
        print("📋 Listing Available MCP Primitives...")
        
        try:
            # Initialize components
            config = MCPConfig(server_url=self.server_url)
            session = MCPSession(config)
            discovery = MCPDiscovery(session)
            
            # Discover primitives
            primitives = await discovery.discover_all()
            
            if not primitives:
                print("   No primitives found")
                return
            
            # Group by type
            by_type = {}
            for primitive in primitives:
                if primitive.type not in by_type:
                    by_type[primitive.type] = []
                by_type[primitive.type].append(primitive)
            
            # Display each type
            for ptype, items in by_type.items():
                print(f"\n🔧 {ptype.upper()}S ({len(items)}):")
                for item in items:
                    description = item.description[:60] + "..." if item.description and len(item.description) > 60 else item.description or "No description"
                    print(f"   • {item.name}: {description}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error listing primitives: {e}")
            print()
    
    async def test_tool_execution(self, tool_name: str, parameters: Dict[str, Any] = None):
        """
        Test executing a specific MCP tool.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Optional parameters for the tool
        """
        print(f"🛠️  Testing Tool Execution: {tool_name}")
        if parameters:
            print(f"   Parameters: {json.dumps(parameters, indent=2)}")
        
        try:
            # Initialize components
            config = MCPConfig(server_url=self.server_url)
            session = MCPSession(config)
            executor = MCPExecutor(session)
            
            # Execute the tool
            result = await executor.call_tool(tool_name, parameters or {})
            
            print("✅ Tool execution successful!")
            print(f"   Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
            print()
            
            return result
            
        except Exception as e:
            print(f"❌ Tool execution failed: {e}")
            print()
            return None
    
    async def run_full_demo(self):
        """Run the complete MCP client demo."""
        print("=" * 60)
        print("🎬 MCP CLIENT DEMO - FULL WALKTHROUGH")
        print("=" * 60)
        print()
        
        # Step 1: Health check
        health = await self.run_health_check()
        
        # Step 2: List primitives
        await self.list_available_primitives()
        
        # Step 3: Test a tool if available
        if health.get("status") == "healthy" and health.get("primitives", {}).get("tools", 0) > 0:
            print("🎯 Testing first available tool...")
            
            try:
                config = MCPConfig(server_url=self.server_url)
                session = MCPSession(config)
                discovery = MCPDiscovery(session)
                
                # Get first tool
                primitives = await discovery.discover_all()
                tools = [p for p in primitives if p.type == "tool"]
                
                if tools:
                    first_tool = tools[0]
                    print(f"   Selected: {first_tool.name}")
                    
                    # Try to execute without parameters first
                    await self.test_tool_execution(first_tool.name)
                
            except Exception as e:
                print(f"❌ Error testing tool: {e}")
                print()
        
        print("=" * 60)
        print("🎉 Demo completed!")
        print("=" * 60)


async def main():
    """Main demo function."""
    # Create and run the demo
    demo = MCPClientDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    asyncio.run(main())
