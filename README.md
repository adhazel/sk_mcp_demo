# Agentic AI in Python - A Code Walkthrough

Curious how to bring agentic AI to life using Python? Want to see how Semantic Kernel can orchestrate real-world AI behavior?

In this hands-on walkthrough, we’ll walk through the code necessary to build a Python-based product Q&A chatbot that goes beyond basic RAG implementations. You’ll see how to give your AI agent the ability to:

- Pull product data from a vector store
- Enhance RAG answers with live web search
- Evaluate hallucination risk
- Respond with a trust score for transparency

We’ll explore how Semantic Kernel enables agentic behavior; and, along the way, you’ll learn how to ground responses in multiple data sources and implement an evaluation loop to keep your AI honest.

If you're a developer looking to move from experimentation to real-world AI applications, this walkthrough will give you the tools, patterns, and confidence to build smarter, more reliable agents in Python.

# MCP Client Demo - Modular Architecture

This folder contains a semantic kernel AI agent application and a separate remote MCP server designed for demos.

# **🚨 Important:** 
This is a newly created demo repository, created in-part through the power of vibe coding. Refactoring is needed prior to any implementation.

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
