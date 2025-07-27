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

This project includes both backend and frontend components:

- **src**: All backend capabilities. Python-based services using FastAPI, Semantic Kernel, and agentic AI patterns. Handles chat orchestration, product data, evaluation, and integrations.
- **Frontend**: React-based web application for user interaction, chat UI, and trust score visualization.

Below is a high-level directory structure for both backend and frontend codebases.

```
├── pyproject.toml                # Poetry configuration
├── README.md
├── .env.local                    # Local environment variables
├── data/                         # ChromaDB and other data storage
│   └── local_chroma_db/         # Local ChromaDB instance
├── notebooks/                    # Jupyter notebooks for development
│   ├── nb_quick_start.ipynb
│   └── notebook_utils.py
├── src/
│   ├── __init__.py
│   ├── main.py                   # Entry point selector (CLI/API)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py               # Chat API endpoints
│   │   ├── health.py             # Health check endpoints
│   │   └── middleware.py         # CORS, logging middleware
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vector_service.py     # ChromaDB operations
│   │   ├── search_service.py     # Web search integration
│   │   ├── evaluation_service.py # Hallucination detection
│   │   ├── chat_service.py       # Main chatbot orchestration
│   │   └── product_service.py    # Product data management
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── rag_agent.py          # RAG-specific agent
│   │   └── evaluation_agent.py   # Evaluation agent
│   ├── models/
│   │   ├── __init__.py
│   │   ├── api_models.py         # FastAPI request/response models
│   │   ├── product.py            # Product data models
│   │   ├── chat_message.py       # Chat message models
│   │   └── evaluation_result.py  # Trust score models
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # (move existing file here)
│       ├── semantic_kernel_setup.py # SK configuration
│       └── logging_utils.py      # Logging utilities
└── frontend/
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── index.html
    ├── public/
    │   └── favicon.ico
    └── src/
        ├── main.tsx              # React app entry
        ├── App.tsx               # Main app component
        ├── vite-env.d.ts        # Vite types
        ├── components/
        │   ├── Chat/
        │   │   ├── ChatContainer.tsx
        │   │   ├── ChatMessage.tsx
        │   │   ├── ChatInput.tsx
        │   │   ├── TrustScore.tsx
        │   │   └── SourcesList.tsx
        │   ├── Layout/
        │   │   ├── Header.tsx
        │   │   └── Layout.tsx
        │   └── UI/
        │       ├── Button.tsx
        │       ├── Input.tsx
        │       └── LoadingSpinner.tsx
        ├── hooks/
        │   ├── useChat.ts
        │   └── useApi.ts
        ├── services/
        │   ├── api.ts            # API client
        │   └── chatService.ts    # Chat-specific API calls
        ├── types/
        │   ├── chat.ts           # Chat-related types
        │   └── api.ts            # API response types
        ├── styles/
        │   ├── globals.css
        │   └── components.css
        └── utils/
            ├── constants.ts
            └── helpers.ts


# Architecture
TBD


# Running the Application

## Development:
```bash
# CLI mode
poetry run start --env dev
python src/main.py --env dev

# API mode  
poetry run start --env dev --mode api
python src/main.py --env dev --mode api
```

## Production:
```bash
# CLI mode
poetry run start --env prod
python src/main.py --env prod

# API mode
poetry run start --env prod --mode api
python src/main.py --env prod --mode api
```

## Local Development:
```bash
# CLI mode (default)
poetry run start
python src/main.py

# API mode
poetry run start --mode api
python src/main.py --mode api
```

> **Note**: Each environment loads its corresponding `.env` file (`.env.dev`, `.env.prod`, `.env.local`)