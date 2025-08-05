#!/usr/bin/env python3
"""
Test script for the RAG MCP server

Run in terminal: 
poetry run python test_rag_components.py
"""
import asyncio
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from utils.mcp_config import Config
from tools.chroma_search import ChromaDBSearcher  
from tools.web_search import WebSearcher
from tools.rag_generator import RAGResponseGenerator
from tools.rag_evaluator import RAGEvaluator

async def test_components():
    """Test individual components"""
    print("🧪 Testing Components...")
    try: 
        start = time.time()
        print("🔧⌛ Configuration loading...")
        config = Config(environment="local")
        elapsed = time.time() - start
        print(f"🔧✅ Configuration loaded: {config.environment} (Time: {elapsed:.2f}s)")
    except Exception as e:
        print(f"🔧❌ Error in configuration loading test: {e}")
        return
    
    try:
        start = time.time()
        print("🧪⌛ LLM connectivity test in progress...")
        llm_client = config.get_llm()
        # response = await llm.chat("Tell me a joke")
        response = llm_client.chat.completions.create(
                    model=config.azure_openai_deployment,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Tell me a one-liner joke."}
                    ],
                    max_tokens=300,
                    temperature=0.9,
                )
        elapsed = time.time() - start
        print(f"🧪✅ LLM connectivity suceeded: joke request response: {response.choices[0].message.content} (Time: {elapsed:.2f}s)")
    except Exception as e:
        print(f"🧪❌ Error in LLM connectivity test: {e}")
        return

    try: 
        start = time.time()
        print("📚⌛ ChromaDB collection exists...")
        chroma_searcher = ChromaDBSearcher(config)
        collection_info = await chroma_searcher.get_collection_info("product_collection")
        elapsed = time.time() - start
        if collection_info.get("exists", False):
            print(f"📚✅ ChromaDB collection does exists. (Time: {elapsed:.2f}s)")
        else:
            print(f"📚❌ ChromaDB collection does not exist. (Time: {elapsed:.2f}s)")
    except Exception as e:
        print(f"📚❌ Error in ChromaDB collection exists test: {e}")
        return

    try: 
        start = time.time()
        print("🔍⌛ ChromaDB searching in progress ...")
        chroma_searcher = ChromaDBSearcher(config)
        if collection_info.get("exists", False):
            chroma_results = await chroma_searcher.search_chroma("hiking", n_results=2)
            # chroma_results_list = chroma_results.to_list()
            elapsed = time.time() - start
            print(f"🔍✅ ChromaDB search found {len(chroma_results)} results (Time: {elapsed:.2f}s)")
        else:
            print("🔍❌ ChromaDB search failed")
    except Exception as e:
        print(f"🔍❌ Error in ChromaDB search test: {e}")
        return
    
    try: 
        start = time.time()
        print("❔⌛ Generating web queries test...")
        web_searcher = WebSearcher(config)
        generated_queries = await web_searcher._get_web_search_queries(user_query="hiking", internal_context=chroma_results)
        elapsed = time.time() - start
        print(f"❔✅ Generating web queries succeeded: {len(generated_queries)} queries (Time: {elapsed:.2f}s)")
    except Exception as e:
        print(f"❔❌ Error in Generating web queries test: {e}")
        return

    try: 
        start = time.time()
        print("🌐⌛ Web searching in progress ...")
        web_searcher = WebSearcher(config)
        web_searcher_results = await web_searcher.search_serpapi_bing_with_generated_queries(
            user_query="hiking",
            internal_context=chroma_results,
            n_results_per_search=3
        )
        elapsed = time.time() - start
        print(f"🌐✅ Web searches performed: {len(web_searcher_results)} searches (Time: {elapsed:.2f}s)")
    except Exception as e:
        print(f"🌐❌ Error in Web search test: {e}")
        return
    
    try: 
        start = time.time()
        print("🤖⌛ RAG generation in progress ...")
        rag_generator = RAGResponseGenerator(config)
        rag_results = await rag_generator.generate_chat_response(
            user_query= "tell me about hiking",
            n_chroma_results = 2,
            n_web_results = 3,
            collection_name= "product_collection")
        elapsed = time.time() - start
        print(f"🤖✅ RAG generation succeeded: response length: {len(rag_results['response'])} (Time: {elapsed:.2f}s)")
    except Exception as e:
        print(f"🤖❌ Error in RAG generation test: {e}")
        return
    
    try: 
        start = time.time()
        print("👩‍🔬⌛ RAG response with evaluation in progress ...")
        rag_with_evaluation = await rag_generator.generate_evaluated_chat_response(
            user_query= "tell me about hiking",
            n_chroma_results = 2,
            n_web_results = 3,
            collection_name= "product_collection")
        elapsed = time.time() - start
        print(f"👩‍🔬✅ RAG response with evaluation succeeded: evaluation score: {rag_with_evaluation['evaluation']['accuracy_score']} (Time: {elapsed:.2f}s)")
    except Exception as e:
        print(f"👩‍🔬❌ Error in RAG response with evaluation test: {e}")
        return



def main():
    """Main test function"""
    print("🚀 Starting RAG MCP Server Component Tests\n")
    asyncio.run(test_components())

if __name__ == "__main__":
    main()
