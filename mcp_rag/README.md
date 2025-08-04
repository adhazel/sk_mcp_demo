<!-- # MCP RAG Demo

This project demonstrates a Model Context Protocol (MCP) RAG system with a Chainlit UI featuring multi-agent coordination.

## Features

- **MCP Server**: Generic RAG tooling with ChromaDB and web search capabilities
- **RAG Tools**: Complete implementation of:
  - `search_internal_products` - ChromaDB semantic search for internal product information
  - `search_web_content` - SerpAPI Bing search for external web content  
  - `generate_rag_response` - Combined RAG response generation with both sources
  - `evaluate_response_accuracy` - AI-powered hallucination detection and accuracy scoring
- **Dynamic Resources**: Real-time collection info and product search results
- **Prompts**: Reusable templates for RAG generation and evaluation
- **Chainlit UI**: Interactive web interface with multi-agent system
- **GitHub Integration**: Analyze repositories using MCP GitHub server
- **Azure AI Services**: Integration with Azure OpenAI and Azure AI Search
- **Multi-Agent System**: Coordinated agents for GitHub analysis, hackathon recommendations, and event discovery

<!-- ## Project Structure

```
mcp_rag/
├── pyproject.toml           # Dependencies for both MCP server and Chainlit UI
├── README.md               # This file
├── .env.example           # Environment variables template
├── .chainlit              # Chainlit configuration
├── src/
│   ├── app.py             # Chainlit UI application
│   ├── chainlit.md        # Welcome screen content
│   ├── server.py          # MCP server implementation (to be implemented)
│   ├── event-descriptions.md  # Event data for RAG
│   ├── mcp_config.json    # MCP server configuration
│   ├── images/            # UI images
│   ├── tools/             # MCP tools (to be implemented)
│   ├── models/            # Data models
│   └── utils/
│       └── config.py      # Configuration utilities
├── data/
│   └── chroma_db/         # ChromaDB storage (to be created)
└── tests/                 # Test files
``` -->

## Setup Instructions

### 1. Environment Setup
```powershell
cd c:\Users\aprilhazel\Source\sk_mcp_demo\mcp_rag
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2. Configure Environment Variables
```powershell
cp .env.example .env
# Edit .env with your Azure and other service credentials
```

### 3. Run the Chainlit UI
```powershell
chainlit run src/app.py
```

### 4. Run the MCP Server (when implemented)
```powershell
python src/server_phase2.py
```

### 5. Test RAG Components
```powershell
python test_rag_components.py
```

## Testing

### Component Testing
Run the component test script to verify individual RAG tools:
```powershell
python test_rag_components.py
```

### MCP Inspector Testing
1. Start the MCP server:
   ```powershell
   poetry run python src/server_phase2.py
   ```

2. In another terminal, start the MCP Inspector:
   ```powershell
   poetry run mcp dev src/server_phase2.py
   ```

3. Test the RAG tools in the inspector interface:
   - Try `search_internal_products` with queries like "laptop" or "phone" 
   - Test `search_web_content` with any web search query
   - Use `generate_rag_response` for complete RAG responses
   - Evaluate responses with `evaluate_response_accuracy`

## Usage

### Chainlit UI
1. Open your browser to the Chainlit interface
2. Provide your GitHub username for analysis
3. Get personalized hackathon project recommendations
4. Discover relevant tech events

### MCP Server Tools
- **Search ChromaDB**: Query local vector databases with `search_internal_products`
- **Web Search**: Use SerpAPI for web search capabilities with `search_web_content`
- **RAG Generation**: Generate comprehensive responses with `generate_rag_response`
- **Response Evaluation**: Detect hallucinations and score accuracy with `evaluate_response_accuracy`

### MCP Server Resources
- **Static Resources**: 
  - `info://rag-capabilities` - RAG system capabilities documentation
  - `info://evaluation-metrics` - Response evaluation criteria and metrics
- **Dynamic Resources**:
  - `chroma://collection/{collection_name}` - ChromaDB collection information
  - `product://search/{query}` - Live product search results

### MCP Server Prompts
- **`rag_system_prompt`** - Template for RAG response generation
- **`evaluation_prompt`** - Template for response accuracy evaluation

## Environment Variables

Required environment variables (see `.env.example`):
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT_NAME`
- `AZURE_SEARCH_SERVICE_ENDPOINT`
- `AZURE_SEARCH_API_KEY`
- `GITHUB_TOKEN` (optional)

## Dependencies

The project includes dependencies for both the MCP server and Chainlit UI:
- MCP framework for server implementation
- Chainlit for web UI
- Semantic Kernel for agent orchestration
- Azure AI services integration
- ChromaDB for vector storage


# set up mcp inspector
run the mcp server poetry run python src/main.py
    It'll load to http://127.0.0.1:8001/mcp/ 
set DANGEROUSLY_OMIT_AUTH=true && npx @modelcontextprotocol/inspector
get Session token from terminal output
The browser should open to something like http://localhost:6274/#tools
    get the query parameters from the terminal as well 
        i.e., Query parameters: {"url":"http://127.0.0.1:8001/mcp/see","transportType":"streamable-http"}
In the MCP inspector, set transport type: streamable http and url (should end with see)
    Then, expand configuration and paste in the proxy session token
Use the dev


# ngrok
 -->



### New Run
1. Start MCP server: poetry run python src/server.py
1. Start MCP inspector: poetry run mcp dev src/server.py

# # Start the server
# poetry run python src/server.py

# # In another terminal, start the MCP Inspector
# poetry run mcp dev src/server.py

# Ensure you don't have anything alraedy using the port
Run the below in command prompt;
netstat -ano | findstr :8002






✨ Key Features:
Async Implementation: All tools use async/await for optimal performance
Error Handling: Comprehensive error handling with fallback responses
Input Validation: Parameter validation with sensible limits
Citation Tracking: All responses include proper source citations
Hallucination Detection: AI-powered accuracy evaluation
Structured Responses: Consistent JSON response format
Configuration: Uses your existing config system
🧪 Testing:
Created test_rag_components.py for component testing
Updated README with testing instructions
All code passes linting checks
📦 Dependencies:
All required dependencies are already included in your pyproject.toml:

chromadb - Vector database operations
aiohttp - Async HTTP requests
pydantic - Structured response parsing
openai - Azure OpenAI integration
The RAG capabilities are now fully integrated into your server_phase2.py and ready for testing with the MCP Inspector!