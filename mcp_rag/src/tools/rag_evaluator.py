"""
RAG Response Evaluation Tools

This module provides comprehensive evaluation capabilities for RAG (Retrieval-Augmented Generation) responses:
- Accuracy scoring based on context support
- Hallucination detection
- Claim-by-claim analysis
- Confidence assessment
"""
import logging
from typing import List, Dict, Any
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from src.utils.mcp_config import Config


logger = logging.getLogger(__name__)

# TODO: Consolidate models into a model folder

class SearchResult(BaseModel):
    """Model for search result"""
    query: str
    content: str
    citation: str
    metadata: Dict[str, Any]
    content_id: int = -1
    search_order: int = -1
    source_type: str = "external"  # Default source type for web search

class SearchResults(BaseModel):
    """Model for search results"""
    queries: List[SearchResult] = Field(default_factory=list)
    
    def __len__(self) -> int:
        """Return the number of search results"""
        return len(self.queries)
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert SearchResults to dictionary format for backward compatibility"""
        return [
            {
                "query": result.query,
                "content": result.content,
                "citation": result.citation,
                "metadata": result.metadata,
                "content_id": result.content_id,
                "search_order": result.search_order,
                "source_type": result.source_type
            }
            for result in self.queries
        ]
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Alias for to_dict() for clearer intent"""
        return self.to_dict()
    

class EvaluationResult(BaseModel):
    """Model for evaluation results with structured output constraints"""
    accuracy_score: float = Field(..., ge=0.0, le=1.0, description="Float between 0.0 and 1.0 indicating how well the answer is supported by context")
    is_hallucination: bool = Field(..., description="True if answer contains information not supported by context")
    evaluation_reasoning: str = Field(..., description="Explanation of the evaluation decision")
    supported_claims: List[str] = Field(..., description="Claims from the answer that are supported by context")
    unsupported_claims: List[str] = Field(..., description="Claims from the answer that are NOT supported by context")
    confidence_level: str = Field(..., description="Confidence in this evaluation", pattern="^(Low|Medium|High)$")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert EvaluationResult to dictionary format"""
        return {
            "accuracy_score": self.accuracy_score,
            "is_hallucination": self.is_hallucination,
            "evaluation_reasoning": self.evaluation_reasoning,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "confidence_level": self.confidence_level
        }

class RAGEvaluator:
    """Evaluate RAG response accuracy and detect hallucinations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.aoai_client = AzureOpenAI(
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_endpoint=config.azure_openai_endpoint
        )

    async def evaluate_rag_accuracy(
        self,
        user_query: str,
        answer: str,
        formatted_context: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate if RAG response is supported by retrieved context
        
        Args:
            user_query: Original user question
            formatted_context: Context string that was provided to the RAG generator
            answer: Generated RAG response to evaluate

        Returns:
            Dictionary with detailed evaluation results
        """
        try:
            # Create evaluation prompt
            evaluation_prompt = f"""You are an expert fact-checker evaluating the accuracy of an AI-generated response.

Your task is to determine if the given ANSWER is supported by the provided CONTEXT.

USER QUESTION: {user_query}

CONTEXT:
{formatted_context}

AI GENERATED ANSWER: {answer}

Please evaluate the answer using the following criteria:

1. accuracy_score: A float between 0.0 and 1.0 (0.0 = completely unsupported, 1.0 = fully supported)
2. is_hallucination: Boolean (true if answer contains information not supported by context)  
3. evaluation_reasoning: String explaining your evaluation
4. supported_claims: Array of claims from the answer that are supported by context
5. unsupported_claims: Array of claims from the answer that are NOT supported by context
6. confidence_level: String ("Low", "Medium", "High") indicating your confidence in this evaluation

Evaluation Criteria:
- Claims must be directly supported by the provided context
- Reasonable inferences are acceptable if clearly based on context
- General knowledge claims should be marked as unsupported unless found in context
- Consider both factual accuracy and completeness"""

            messages = [
                {"role": "system", "content": "You are a precise fact-checking assistant that evaluates AI-generated responses against provided context."},
                {"role": "user", "content": evaluation_prompt}
            ]
            
            response = self.aoai_client.chat.completions.parse(
                model=self.config.azure_openai_deployment,
                messages=messages,
                response_format=EvaluationResult,
                max_tokens=4000,
                temperature=0.1  # Low temperature for consistent evaluation
            )
            
            # Extract the parsed structured output
            evaluation_result = response.choices[0].message.parsed
            evaluation_data_dict = evaluation_result.to_dict()
            
            return {
                "user_query": user_query,
                "answer": answer,
                "evaluation": evaluation_data_dict
            }
            
        except Exception as e:
            logger.error(f"Error evaluating RAG accuracy: {e}")
            raise e
    
    async def evaluate_rag_response(self, rag_response: Dict[str, Any]) -> 'EvaluationResult':
        """
        Evaluate a RAG response that was generated by RAGResponseGenerator
        
        Args:
            rag_response: Dictionary returned by rag_generator.generate_chat_response()
            
        Returns:
            EvaluationResult object with evaluation details
        """
        try:
            # Extract components from RAG response
            user_query = rag_response.get("user_query", "")
            answer = rag_response.get("response", "")
            formatted_context = rag_response.get("formatted_context", "")
            
            # Call the main evaluation method
            evaluation_result = await self.evaluate_rag_accuracy(
                user_query=user_query,
                answer=answer,
                formatted_context=formatted_context
            )
            
            # Return just the evaluation part as EvaluationResult
            evaluation_data = evaluation_result.get("evaluation", {})
            
            return EvaluationResult(
                accuracy_score=evaluation_data.get("accuracy_score", 0.0),
                is_hallucination=evaluation_data.get("is_hallucination", True),
                evaluation_reasoning=evaluation_data.get("evaluation_reasoning", ""),
                supported_claims=evaluation_data.get("supported_claims", []),
                unsupported_claims=evaluation_data.get("unsupported_claims", []),
                confidence_level=evaluation_data.get("confidence_level", "Low")
            )
            
        except Exception as e:
            logger.error(f"Error evaluating RAG response: {e}")
            raise e

    async def generate_chat_response_and_evaluate(
        self,
        user_query: str,
        n_chroma_results: int = 3,
        n_web_results: int = 3,
        collection_name: str = "product_collection"
    ) -> Dict[str, Any]:
        """
        Generate a RAG response AND evaluate it in one step
        
        Args:
            user_query: The user's question
            n_chroma_results: Number of ChromaDB results to retrieve
            n_web_results: Number of web search results to retrieve
            collection_name: ChromaDB collection name
            
        Returns:
            Dictionary containing both the RAG response and evaluation
        """
        try:
            # Import here to avoid circular imports
            from src.tools.rag_generator import RAGResponseGenerator
            
            # Create RAG generator with same config
            rag_generator = RAGResponseGenerator(self.config)
            
            # Generate the RAG response
            rag_response = await rag_generator.generate_chat_response(
                user_query=user_query,
                n_chroma_results=n_chroma_results,
                n_web_results=n_web_results,
                collection_name=collection_name
            )
            
            # Evaluate the response
            evaluation = await self.evaluate_rag_response(rag_response)
            
            # Return combined result
            return {
                "rag_response": rag_response,
                "evaluation": evaluation.to_dict(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in generate_chat_response_and_evaluate: {e}")
            raise e