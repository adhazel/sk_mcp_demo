## Requirements

- Python
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

# Full server
poetry run python src/mcp_server_product_agent.py
```

## Test client

To test that each of the components are working, run the below.

```bash
poetry run python test_rag_components.py
```

## Test in MCP inspector

To test that each of the components are working, run the below.

```bash
poetry run mcp dev src/mcp_server_product_agent.py
```
