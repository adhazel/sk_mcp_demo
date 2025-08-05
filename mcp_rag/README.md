# MCP RAG Server

A Model Context Protocol (MCP) server that provides RAG (Retrieval-Augmented Generation) capabilities with internal ChromaDB search and external web search via SerpAPI.

## Server Implementations

This project includes two server implementations to demonstrate MCP development progression:

### `mcp_server_fundamentals.py` - Basic MCP Server
- **Purpose**: Learn MCP fundamentals and instrumentation
- **Features**: Basic tool structure, simple examples, MCP framework setup
- **Use Case**: Getting started with MCP, understanding tool registration and basic patterns

### `mcp_server_product_agent.py` - Full RAG Scenario  
- **Purpose**: Complete RAG implementation with real-world tools
- **Features**: ChromaDB integration, web search, response evaluation, error handling
- **Use Case**: Demo-ready RAG server with comprehensive functionality

## mcp_server_product_agent Tools

- `search_internal_products` - Search internal product catalog using ChromaDB
- `search_external_web` - Search web using Bing via SerpAPI with query generation
- `evaluate_response` - Evaluate response accuracy and quality

## Additional Components

This project includes a `rag_generator` module for testing and demonstration purposes. It combines the MCP tools and calls an LLM to perform complete RAG operations.

**Note**: In production, this orchestration logic should reside in the client application (e.g., agent orchestrator's `/ask` endpoint) rather than within the MCP server.

## Requirements

- Python 3.8+
- Azure OpenAI API key
- SerpAPI key
- ChromaDB with populated data
- Populated .env or .env.local file

## Setup

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Configure environment variables, see [.env.local.example](./.env.local.example) for sample values.

3. Run servers:
   ```bash
   # Basic learning server
   poetry run python src/mcp_server_fundamentals.py
   
   # Full RAG server
   poetry run python src/mcp_server_product_agent.py
   ```

## Test

To test that each of the components are working, run the below.

```bash
poetry run python test_rag_components.py
```
