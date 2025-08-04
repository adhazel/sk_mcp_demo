@echo off
echo Starting MCP Orchestrator Web Demo...
echo.
echo Architecture: SemanticKernelOrchestrator → ChatCompletionAgent → MCPStreamableHttpPlugin → MCP Server
echo.
echo Make sure your MCP server is running on http://127.0.0.1:8002
echo.
echo Web interface will be available at: http://localhost:8001
echo API documentation at: http://localhost:8001/docs
echo.
poetry run python -m uvicorn demo_web.orchestrator_api:app --host 0.0.0.0 --port 8001 --reload
