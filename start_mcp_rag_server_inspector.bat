@REM Usage from the sk_mcp_demo root directory:
@REM .\start_mcp_rag_server.bat

echo off
echo Starting MCP RAG Server...
echo.

REM Check if we're in the correct directory
if not exist "mcp_rag\" (
    echo Error: This script must be run from the sk_mcp_demo root directory.
    echo Current directory: %CD%
    echo Expected to find: mcp_rag\
    pause
    exit /b 1
)

echo Server will be available at: http://127.0.0.1:8002/mcp/
echo.

REM Run with mcp_rag Poetry environment directly
REM Clear PYTHONPATH to avoid conflicts with root project
set PYTHONPATH=
cd mcp_rag && poetry run mcp dev src/mcp_server_product_agent.py
