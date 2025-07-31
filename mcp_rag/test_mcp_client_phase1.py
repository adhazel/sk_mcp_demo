"""Test MCP Client to query a demo MCP server.

Stands up a basic MCP client to connect to a demo MCP server
and test its functionality, including tool calls and resource access.

    Server connectivity: The server responds correct on the specified port
    MCP Connection: The client connects to the MCP server
    Session Initialization: The client initializes a session with the server
    Tool Discovery: Finds tools and parameters
    Tool Execution: Calls tools with successful results
    Resource Access: Reads dynamic resources

Example:
    >>> poetry run python test_mcp_client.py
    Found 1 tool(s)...
    Testing add tool...
    Testing resource...

License: MIT
"""

import asyncio
import requests
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

async def test_demo_mcp():
    """Test the Demo MCP server."""
    
    print("🚀 Testing Demo MCP Server")
    print("=" * 50)
    
    # Use the correct port from your server.py
    server_url = "http://127.0.0.1:8002/mcp/"
    print(f"📡 Connecting to: {server_url}")
    
    try:
        # Create streamable HTTP client
        async with streamablehttp_client(server_url) as (read, write, get_session_id):
            print("✅ Connected to Demo MCP server")
            print(f"📋 Session ID: {get_session_id()}")
            
            # Create client session
            async with ClientSession(read, write) as session:
                print("🔄 Initializing session...")
                
                # Initialize the session
                await session.initialize()
                print("✅ Session initialized successfully")
                
                # List available tools
                print("\n🛠️  Listing available tools...")
                tools_result = await session.list_tools()
                
                if tools_result.tools:
                    print(f"📋 Found {len(tools_result.tools)} tool(s):")
                    for tool in tools_result.tools:
                        print(f"   - {tool.name}: {tool.description}")
                        if hasattr(tool, 'inputSchema') and tool.inputSchema:
                            properties = tool.inputSchema.get('properties', {})
                            if properties:
                                print(f"     Parameters: {list(properties.keys())}")
                else:
                    print("❌ No tools found")
                    return
                
                # Test the add tool (matching your server.py)
                print("\n🧮 Testing add tool...")
                test_case = {"a": 15, "b": 27}
                
                try:
                    print(f"   Testing: {test_case['a']} + {test_case['b']}")
                    result = await session.call_tool("add", test_case)
                    
                    if result.content:
                        for content in result.content:
                            if hasattr(content, 'text'):
                                print(f"   ✅ Result: {content.text}")
                            else:
                                print(f"   ✅ Result: {content}")
                    else:
                        print(f"   ⚠️  No content in result: {result}")
                        
                except Exception as tool_error:
                    print(f"   ❌ Tool call failed: {tool_error}")
                
                # Test resources if available
                print("\n📚 Listing available resources...")
                
                try:
                    resources_result = await session.list_resources()
                    if resources_result:
                        print(f"📋 Found {len(resources_result.resources)} static resource(s):")
                        for resource in resources_result.resources:
                            print(f"   - {resource.uri}: {resource.name}")

                            try:
                                static_result = await session.read_resource(resource.uri)
                                if static_result.contents:
                                    for content in static_result.contents:
                                        if hasattr(content, 'text'):
                                            print(f"   ✅ Static Content: {content.text}")
                                        else:
                                            print(f"   ✅ Static Content: {content}")
                                else:
                                    print("   ❌ Static resource returned no content")
                            except Exception as static_error:
                                print(f"   ❌ Static resource failed: {static_error}")
                    else:
                        print("   ❌ No static resources found")
                        return
                    
                    
                    resources_template_result = await session.list_resource_templates()

                    if resources_template_result:
                        print(f"📋 Found {len(resources_template_result.resourceTemplates)} template resource(s):")
                        for resource in resources_template_result.resourceTemplates:
                            print(f"   - {resource.uriTemplate} | {resource.name}: {resource.description} ")

                    # Test the dynamic greeting resource directly
                    print("\n👋 Testing dynamic greeting resource...")
                    try:
                        greeting_result = await session.read_resource("greeting://Alice")
                        if greeting_result.contents:
                            for content in greeting_result.contents:
                                if hasattr(content, 'text'):
                                    print(f"   ✅ Dynamic Greeting: {content.text}")
                                else:
                                    print(f"   ✅ Dynamic Greeting: {content}")
                                    
                        # Test with a different name to show it's dynamic
                        print(f"   Testing with different name...")
                        greeting_result2 = await session.read_resource("greeting://Bob")
                        if greeting_result2.contents:
                            for content in greeting_result2.contents:
                                if hasattr(content, 'text'):
                                    print(f"   ✅ Dynamic Greeting: {content.text}")
                                else:
                                    print(f"   ✅ Dynamic Greeting: {content}")
                        else:
                            print("   ⚠️  No content in greeting result")
                    except Exception as resource_error:
                        print(f"   ❌ Resource read failed: {resource_error}")
                        
                except Exception as e:
                    print(f"   ❌ Failed to list resources: {e}")
                    
                
            
                print("\n🎉 Demo MCP testing completed!")
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Print more detailed error information
        import traceback
        print("   Full traceback:")
        traceback.print_exc()
        
        print("\n💡 Troubleshooting tips:")
        print("   - Make sure your MCP server is running on port 8002")
        print(f"   - Check if the server URL is correct: {server_url}")
        print("   - Verify the server is using streamable-http transport")
        print("   - Try running: poetry run python src/server.py")
        print("   - Check if port 8002 is already in use: netstat -ano | findstr :8002")

def test_server_connectivity():
    """Quick test to check if the server is running."""
    
    print("🔍 Testing server connectivity...")
    server_url = "http://127.0.0.1:8002/mcp/"
    
    try:
        # Try a simple GET request first
        response = requests.get(server_url, timeout=5)
        print(f"   📡 GET {server_url} -> Status: {response.status_code}")
        
        # Try a POST request with proper MCP headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        response = requests.post(server_url, headers=headers, json=payload, timeout=10)
        print(f"   📡 POST {server_url} -> Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Server is responding correctly")
            return True
        else:
            print(f"   ❌ Server error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Connection refused - server is not running")
        return False
    except Exception as e:
        print(f"   ❌ Connectivity test failed: {e}")
        return False

if __name__ == "__main__":
    # First test basic connectivity
    if test_server_connectivity():
        # Test local server
        asyncio.run(test_demo_mcp())
    else:
        print("\n💡 Start your server first:")
        print("   poetry run python src/server.py")