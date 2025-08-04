@echo off
echo Starting MCP RAG Server...
echo.
echo Server will be available at: http://127.0.0.1:8002/mcp/
echo.
cd mcp_rag
poetry run python src/server_phase2.py
