"""
RAG Generator using LLM + ChromaDB + Web Search

This module provides RAG (Retrieval-Augmented Generation) capabilities that combine:
- Internal knowledge base search via ChromaDB
- External web search via SerpAPI
- LLM response generation using Azure OpenAI

INTEGRATION WITH RAG EVALUATOR:

The RAGResponseGenerator is designed to work seamlessly with the RAGEvaluator.
Key integration features:

1. Combined Context: The generate_chat_response method returns a 'combined_context' 
   field specifically formatted for evaluation.

2. Convenience Methods:
   - evaluate_generated_response(): Directly evaluate a RAG response
   - extract_evaluation_components(): Extract components needed for evaluation

3. Static Methods: Available for use without instantiation

Example Usage:
```python
from src.tools.rag_generator import RAGResponseGenerator
from src.tools.rag_evaluator import RAGEvaluator
from src.utils.mcp_config import Config

# Initialize
config = Config(environment="local")
rag_generator = RAGResponseGenerator(config=config)
rag_evaluator = RAGEvaluator(config=config)

# Generate response
rag_response = await rag_generator.generate_chat_response("What products are available?")

# Method 1: Use convenience method
evaluation = await rag_generator.evaluate_generated_response(rag_response, rag_evaluator)

# Method 2: Manual evaluation
components = RAGResponseGenerator.extract_evaluation_components(rag_response)
evaluation = await rag_evaluator.evaluate_rag_accuracy(
    user_query=components['user_query'],
    context=components['context'], 
    answer=components['answer']
)
```
"""
import logging
from typing import Dict, Any
from src.utils.mcp_config import Config
from src.tools.chroma_search import ChromaDBSearcher, SearchResults as ChromaSearchResults
from src.tools.web_search import WebSearcher, SearchResults as WebSearchResults

logger = logging.getLogger(__name__)

class RAGResponseGenerator:
    """Generate RAG responses combining internal and external search"""
    
    def __init__(self, config: Config):
        """
        Initialize RAG Response Generator
        
        Args:
            config: Configuration object (required if chroma_searcher and web_searcher not provided)
            chroma_searcher: Pre-initialized ChromaDB searcher (optional)
            web_searcher: Pre-initialized web searcher (optional)
        """
        self.config = config
        self.chroma_searcher = ChromaDBSearcher(config)
        self.web_searcher = WebSearcher(config)
        self.aoai_client = config.get_llm()

    def format_context_for_llm(self, internal_context: ChromaSearchResults, external_context: WebSearchResults) -> str:
        """Format search contexts for LLM prompt"""
        formatted_context = ""
        
        if internal_context and len(internal_context) > 0:
            formatted_context += "\n## INTERNAL KNOWLEDGE BASE:\n"
            for i, item in enumerate(internal_context.queries, 1):
                formatted_context += f"\n{i}. Content: {item.content}\n{item.citation}\nMetadata: {item.metadata}\n"
        
        if external_context and len(external_context) > 0:
            formatted_context += "\n## EXTERNAL WEB SOURCES:\n"
            for i, item in enumerate(external_context.queries, 1):
                formatted_context += f"\n{i}. Content: {item.content}\n{item.citation}\nMetadata: {item.metadata}\n"
        
        return formatted_context

    def combine_contexts_to_list(self, internal_context: ChromaSearchResults, external_context: WebSearchResults):
        """
        Prepare contexts for evaluation by the RAGEvaluator
        
        Args:
            internal_context: SearchResults from internal search
            external_context: SearchResults from external search
            
        Returns:
            Dictionary with internal and external contexts ready for evaluation
        """
        external_context = internal_context.to_list() + external_context.to_list()
        return external_context

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

    async def generate_chat_response(
        self,
        user_query: str,
        n_chroma_results: int = 5,
        n_web_results: int = 5,
        collection_name: str = "product_collection"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive RAG response using both internal and external search
        
        Args:
            user_query: User's question
            n_chroma_results: Number of ChromaDB results to retrieve
            n_web_results: Number of web search results to retrieve  
            collection_name: ChromaDB collection name
            
        Returns:
            Dictionary with response, metadata, and source information
        """
        try:
            # First, perform internal search
            internal_context = await self.chroma_searcher.search_chroma(
                query=user_query,
                n_results=n_chroma_results,
                collection_name=collection_name
            )
            logger.info(f"1️⃣ Internal search found {len(internal_context)} results")
            
            # Generate intelligent web search queries based on internal context
            generated_queries = await self.web_searcher.get_web_search_queries(
                user_query=user_query,
                internal_context=internal_context.to_list()
            )
            logger.info(f"2️⃣ Generated {len(generated_queries.queries) if generated_queries.queries else 0} web queries")
            
            # Perform web searches using generated queries
            external_context = None
            if generated_queries.queries:
                external_context = await self.web_searcher.search_serpapi_bing_with_generated_queries(
                    generated_queries=generated_queries,
                    n_results=n_web_results
                )
                logger.info(f"3️⃣ External search results found {len(external_context)}")
            else:
                logger.info("3️⃣ No intelligent web queries generated - skipping external search")
                # Create empty SearchResults for consistency
                external_context = WebSearchResults(queries=[])

            formatted_context = self.format_context_for_llm(internal_context, external_context)
            logger.info(f"4️⃣ Formatted context for LLM:\n{formatted_context[:200]}...")  # Log first 200 chars

            # If no context found, return early
            if not formatted_context.strip():
                formatted_context = "<no_context>"

            # Create system prompt
            system_prompt = self._get_system_prompt()

            # Generate response
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
            logger.info(f"5️⃣ Generated RAG response:\n{generated_response[:200]}...")  # Log first 200 chars
            combined_context = self.combine_contexts_to_list(internal_context, external_context)
            
            return {
                "user_query": user_query,
                "response": generated_response,
                "context_count": len(combined_context),
                "formatted_context": formatted_context,
                "combined_context": combined_context
            }
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            # Re-raise the exception instead of returning it as a success response
            raise RuntimeError(f"RAG response generation failed: {str(e)}") from e
