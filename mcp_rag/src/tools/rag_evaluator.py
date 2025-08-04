"""
Evaluate RAG response quality and accuracy.
"""
import logging
from typing import Dict, Any, TYPE_CHECKING
from openai import AzureOpenAI
from src.utils.mcp_config import Config
from src.models import EvaluationResult

if TYPE_CHECKING:
    from src.tools.rag_generator import RAGResponseGenerator


logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluate RAG response quality and accuracy."""
    
    def __init__(self, config: Config, rag_generator: 'RAGResponseGenerator' = None):
        self.config = config
        self.aoai_client = AzureOpenAI(
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_endpoint=config.azure_openai_endpoint
        )
        # Optional dependency injection for rag_generator
        self._rag_generator = rag_generator

    async def evaluate_rag_accuracy(
        self,
        user_query: str,
        answer: str,
        formatted_context: str = "",
    ) -> Dict[str, Any]:
        """
        Check if the answer is supported by the provided context.
        
        Args:
            user_query: Original user question
            answer: Generated response to evaluate
            formatted_context: Context that was used for generation

        Returns:
            Evaluation results with accuracy score and reasoning
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
2. evaluation_reasoning: String explaining your evaluation
3. supported_claims: Array of claims from the answer that are supported by context
4. unsupported_claims: Array of claims from the answer that are NOT supported by context
5. confidence_level: String ("Low", "Medium", "High") indicating your confidence in this evaluation

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
    
    async def evaluate_rag_response(self, rag_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a complete RAG response with all its components.
        
        Args:
            rag_response: Complete response from generate_chat_response()
            
        Returns:
            Evaluation results for the response
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
            
            # Return the evaluation data directly (it's already a dictionary)
            return evaluation_result["evaluation"]
            
        except Exception as e:
            logger.error(f"Error evaluating RAG response: {e}")
            raise e