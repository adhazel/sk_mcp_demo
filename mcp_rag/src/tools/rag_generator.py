"""
RAG response generation combining internal database search, web research, and AI.

This module creates comprehensive answers by:
- Searching internal product database (ChromaDB)
- Finding additional information via web search (SerpAPI)
- Generating AI responses using Azure OpenAI
- Optionally evaluating response accuracy
"""
import logging
from typing import Dict, Any, List
from utils import McpConfig
from tools.chroma_search import ChromaDBSearcher
from tools.web_search import WebSearcher

logger = logging.getLogger(__name__)


class RAGResponseGenerator:
    """Generate comprehensive answers using internal database and web search."""
    
    def __init__(self, config: McpConfig):
        """
        Initialize the RAG response generator.
        
        Args:
            config: Configuration for database and API connections
        """
        self.config = config
        self.chroma_searcher = ChromaDBSearcher(config)
        self.web_searcher = WebSearcher(config)
        self.aoai_client = config.get_llm()
        self._rag_evaluator = None  # Lazy initialization

    def format_context_for_llm(self, context: List[Dict[str, Any]] = None) -> str:
        """Format search results for the AI to use."""
        # TODO: This is repeated in rag_generator.py, consider refactoring
        formatted_context = ""
        if not context:
            return "< No context available >"
        sections = {
            "internal": "\n## INTERNAL KNOWLEDGE BASE:\n",
            "external": "\n## EXTERNAL WEB:\n"
        }
        for source_type, header in sections.items():
            formatted_context += header
            for i, item in enumerate([c for c in context if c.get('source_type') == source_type], 1):
                formatted_context += f"\n{i}. Content: {item.get('content', 'N/A')}\n{item.get('citation', 'N/A')}\nMetadata: {item.get('metadata', {})}\n"
        return formatted_context

    def combine_search_results_to_list(self, search_results_list: List) -> List[Dict[str, Any]]:
        """
        Combine multiple search result lists into one.

        Args:
            search_results_list: List of search result lists

        Returns:
            Single combined list of all search results
        """
        all_context = []
        for search_results in search_results_list:
            all_context.extend(search_results)
        return all_context

    def _get_system_prompt(self) -> str:
        """Get the system prompt for RAG response generation"""
        return """You are a helpful assistant that provides comprehensive answers based on both internal knowledge base and external web sources.

INSTRUCTIONS:
1. Use the provided context to answer the user's question
2. Clearly distinguish between internal knowledge and external sources
3. Include relevant citations from the provided context
4. If information conflicts between sources, acknowledge the discrepancy
5. If you cannot find relevant information in the context, say so clearly
6. Provide a well-structured, informative response

CONTEXT:"""

    async def _generate_rag_internal(
        self,
        user_query: str,
        n_chroma_results: int = 5,
        n_web_results: int = 5,
        collection_name: str = "product_collection",
        include_full_context: bool = True
    ) -> Dict[str, Any]:
        """
        Internal method that handles the core RAG generation logic.
        
        Args:
            user_query: User's question
            n_chroma_results: Number of ChromaDB results to retrieve
            n_web_results: Number of web search results to retrieve  
            collection_name: ChromaDB collection name
            include_full_context: Whether to include full context objects (for evaluation)
            
        Returns:
            Dictionary with response and context (full or minimal based on flag)
        """
        try:
            # Step 1: Internal search
            internal_context = await self.chroma_searcher.search_chroma(
                query=user_query,
                n_results=n_chroma_results,
                collection_name=collection_name
            )
            logger.info(f"1️⃣ Internal search found {len(internal_context)} results")
            
            # Step 2: Perform web searches (generates queries internally)
            external_context = await self.web_searcher.search_bing_with_chat_and_context(
                user_query=user_query,
                internal_context=internal_context,
                n_results_per_search=n_web_results
            )
            logger.info(f"2️⃣ External search results found {len(external_context)}")
            
            # Step 3: Combine and format context
            combined_context = self.combine_search_results_to_list([internal_context, external_context])
            formatted_context = self.format_context_for_llm(combined_context)
            logger.info(f"4️⃣ Formatted context for LLM:\n{formatted_context[:200]}...")

            # Handle empty context
            if not formatted_context.strip():
                formatted_context = "<no_context>"

            # Step 5: Generate response
            system_prompt = self._get_system_prompt()
            messages = [
                {"role": "system", "content": system_prompt + formatted_context},
                {"role": "user", "content": user_query}
            ]
            
            response = self.aoai_client.chat.completions.create(
                model=self.config.azure_openai_deployment,
                messages=messages,
                max_tokens=8000,
                temperature=0.3
            )
            
            generated_response = response.choices[0].message.content
            logger.info(f"5️⃣ Generated RAG response:\n{generated_response[:200]}...")

            # Extract citations only (lightweight)
            citations = []
            for item in combined_context:
                citations.append({
                    "source_type": item.get("source_type", "unknown"),
                    "citation": item.get("citation", ""),
                    "context_id": item.get("context_id", -1)
                })

            # Base response (always included)
            result = {
                "user_query": user_query,
                "response": generated_response,
                "context_count": len(combined_context),
                "citations": citations  # Lightweight citations only
            }

            # Add full context only if needed (for evaluation)
            if include_full_context:
                result.update({
                    "formatted_context": formatted_context #,
                    # "combined_context": combined_context
                })

            return result
            
        except Exception as e:
            logger.error(f"Error in _generate_rag_internal: {e}")
            raise RuntimeError(f"RAG generation failed: {str(e)}") from e

    async def generate_chat_response(
        self,
        user_query: str,
        n_chroma_results: int = 5,
        n_web_results: int = 5,
        collection_name: str = "product_collection"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive answer using database and web search.
        
        Args:
            user_query: The question to answer
            n_chroma_results: Max internal database results to use
            n_web_results: Max web search results to use
            collection_name: Database collection to search
            
        Returns:
            Answer with sources and citations
        """
        return await self._generate_rag_internal(
            user_query=user_query,
            n_chroma_results=n_chroma_results,
            n_web_results=n_web_results,
            collection_name=collection_name,
            include_full_context=True  # Include full context for evaluation
        )
