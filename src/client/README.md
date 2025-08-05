# MCP Client Demo - Modular Architecture

This folder contains a semantic kernel AI agent application and a separate remote MCP server designed for demos.

**🚨 Important:** This is a newly created demo repository, created in-part through the power of vibe coding. Refactoring is needed prior to any implementation.

## 🏗️ Contents

```
demo_web/
├── static/
│   ├── index.html        # base web page + styling
│   ├── orchestrator.html # the MCP Orchestrator Web Demo container web page
├── orchestrator_api.py   # the front end api
src/agents/
├── sk_product_chat_agent.py # product chat agentic Orchestrator
src/client/
├── core/           # Core MCP functionality
│   ├── config.py   # Configuration management
│   ├── discovery.py # Primitive discovery
│   ├── executor.py # Tool/resource execution
│   └── session.py  # MCP session handling
├── plugins/        # Framework integrations
│   ├── base.py     # Abstract base plugin
│   └── semantic_kernel.py # Semantic Kernel plugin
src/utils/
├── caller.py # Gets the parent in the trace
├── config.py # App configuration
```
