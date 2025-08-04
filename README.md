# Agentic AI in Python - A Code Walkthrough

Curious how to bring agentic AI to life using Python? Want to see how Semantic Kernel can orchestrate real-world AI behavior?

In this hands-on walkthrough, we’ll walk through the code necessary to build a Python-based product Q&A chatbot that goes beyond basic RAG implementations. You’ll see how to give your AI agent the ability to:

- Pull product data from a vector store
- Enhance RAG answers with live web search
- Evaluate hallucination risk
- Respond with a trust score for transparency

We’ll explore how Semantic Kernel enables agentic behavior; and, along the way, you’ll learn how to ground responses in multiple data sources and implement an evaluation loop to keep your AI honest.

If you're a developer looking to move from experimentation to real-world AI applications, this walkthrough will give you the tools, patterns, and confidence to build smarter, more reliable agents in Python.

# Project Structure Overview

This demo showcases a clean, modular architecture for building production-ready AI agents:

```
sk_mcp_demo/
├── 🎯 demo_semantic_kernel.py     # Main demo script - start here!
├── 🌐 demo_web/                   # Web interface and API server
├── 📊 mcp_rag/                    # MCP server for RAG operations
└── 🧠 src/                        # Core application components
    ├── agents/                    # AI agent implementations
    ├── client/                    # MCP client integration
    │   ├── core/                  # Core MCP functionality
    │   └── plugins/               # Framework-specific integrations
    └── utils/                     # Configuration and utilities
```

## Key Technologies

- **Microsoft Semantic Kernel**: AI orchestration framework for building intelligent agents
- **Model Context Protocol (MCP)**: Standardized protocol for connecting AI systems to data sources
- **ChromaDB**: Vector database for semantic search and RAG operations
- **FastAPI**: Modern web framework for building APIs
- **Poetry**: Python dependency management

## Architecture & Data Flow

### Backend vs Frontend Structure

```
📁 src/                           🧠 Backend AI Orchestration Logic
├── agents/sk_product_chat_agent.py   - Main AI agent using Semantic Kernel
├── client/                           - MCP client integration layer
│   ├── plugins/                      - Framework-specific integrations
│   └── core/                         - Protocol-agnostic MCP functionality
└── utils/                            - Configuration and utilities

📁 demo_web/                      🌐 Frontend Web Interface
├── orchestrator_api.py               - FastAPI web server (REST endpoints)
└── static/orchestrator.html          - HTML/CSS/JavaScript frontend

📁 mcp_rag/                       📊 RAG Data Server
└── src/server_phase2.py              - MCP server for RAG operations
```

### Request Flow

```
User (Browser) 
    ↓ HTTP requests (chat, tools, health)
demo_web/orchestrator_api.py (FastAPI server)
    ↓ Python imports & function calls  
src/agents/sk_product_chat_agent.py (AI orchestration)
    ↓ MCP protocol over HTTP
mcp_rag/src/server_phase2.py (RAG operations)
    ↓ Vector search & data retrieval
ChromaDB + Product Database
```

**Key Benefits:**
- 🧩 **Clean Separation**: Frontend (UI/HTTP) separate from AI logic  
- 🔄 **Reusable Backend**: `src/` logic works with any frontend (CLI, API, etc.)
- 🧪 **Testable**: AI orchestration can be tested independently
- 📈 **Scalable**: Easy to add more frontends or backend agents

## Core Components

### 🎯 Demo Script (`demo_semantic_kernel.py`)
The main entry point that demonstrates Semantic Kernel + MCP integration:
- Clean, presentation-friendly code
- Shows dynamic tool discovery and registration
- Demonstrates health checks and tool invocation

### 🧠 Semantic Kernel Integration (`src/client/plugins/`)
- **SimpleMCPPlugin**: Clean integration between MCP and Semantic Kernel
- **Dynamic Tool Registration**: Automatically discovers and registers MCP tools as kernel functions
- **Demo Functions**: Built-in functions for health checks, tool listing, and direct tool calls

### 📊 MCP RAG Server (`mcp_rag/`)
Standalone MCP server providing:
- Vector database operations (ChromaDB)
- Product data retrieval
- Search capabilities
- RESTful API endpoints

### 🌐 Web Interface & API (`demo_web/`)
Combined web interface and REST API server:
- Interactive HTML demo interface
- Full REST API endpoints (`/chat`, `/search`, `/health`, `/tools`)
- Direct tool calling capabilities
- Real-time status monitoring

## Getting Started

### 1. Install Dependencies
```bash
poetry install
```

### 2. Start the MCP Server
```bash
# In terminal 1
cd mcp_rag
poetry run python src/server_phase2.py
```

### 3. Run the Demo
```bash
# In terminal 2
python demo_semantic_kernel.py
```

### 4. Try the Web Interface
```bash
# Start the web server
python demo_web/orchestrator_api.py

# Open browser to: http://localhost:8001
```

## Configuration

Configuration is managed through environment variables and `.env` files:

```bash
# .env.local
MCP_SERVER_URL=http://127.0.0.1:8002
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_API_KEY=your-key
...
```

## Key Features Demonstrated

### 🔄 Dynamic Tool Discovery
The system automatically discovers available MCP tools and registers them as Semantic Kernel functions:

```python
# Tools are discovered and registered automatically
plugin = await create_mcp_plugin(kernel)
tools = plugin.get_primitive_names("tool")
```

### 🎯 Semantic Kernel Integration
Clean integration that makes MCP tools available as native kernel functions:

```python
@kernel_function(name="mcp_health_check")
async def mcp_health_check(self) -> str:
    """Check MCP server health and connection status"""
```

### 🛠️ Flexible Architecture
Modular design that separates concerns:
- **Core MCP Logic**: Protocol handling, discovery, execution
- **Framework Integration**: Semantic Kernel-specific implementations
- **Configuration**: Centralized environment-based config

## MCP Client Architecture (`src/client/`)

```
client/
├── core/           # Protocol-agnostic MCP functionality
│   ├── config.py   # MCP configuration management
│   ├── session.py  # MCP session handling
│   ├── discovery.py# Tool/resource discovery
│   └── executor.py # Tool execution logic
└── plugins/        # Framework-specific integrations
    ├── base.py     # Abstract base for all frameworks
    └── semantic_kernel.py  # Semantic Kernel integration
```
## Next Steps

**Explore the Demo**: Run `demo_semantic_kernel.py` to see the integration in action

## Why This Matters

This project demonstrates how to build **production-ready agentic AI** that goes beyond simple chatbots:

- ✅ **Real-world Integration**: Connects to multiple data sources dynamically or via agentic orchestration.
- ✅ **Scalable Architecture**: Clean separation of concerns
- ✅ **Framework Agnostic**: Core MCP logic works with any AI framework
- ✅ **Demo-Friendly**: Clear, understandable code for demonstration purposes

Ready to build smarter AI agents? Start with `demo_semantic_kernel.py` and see the magic happen! 🎉

## Current Limitations & TODOs

This demo provides an entry point for agentic AI development, but several areas are planned for improvement. You're welcome to use and extend this project under the terms specified in [LICENSE.txt](LICENSE.txt).

**⚠️ Note:** In particular, the Web interface (`demo_web/`) was rapidly prototyped and needs significant refactoring. In addition, security mechanisms need to be considered fully.