"""RAG response evaluation models."""

from typing import Dict, Any, List
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Evaluation results for RAG response quality."""
    accuracy_score: float = Field(..., ge=0.0, le=1.0, description="Float between 0.0 and 1.0 indicating how well the answer is supported by context")
    evaluation_reasoning: str = Field(..., description="Explanation of the evaluation decision")
    supported_claims: List[str] = Field(..., description="Claims from the answer that are supported by context")
    unsupported_claims: List[str] = Field(..., description="Claims from the answer that are NOT supported by context")
    confidence_level: str = Field(..., description="Confidence in this evaluation", pattern="^(Low|Medium|High)$")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation to dictionary format."""
        return {
            "accuracy_score": self.accuracy_score,
            "evaluation_reasoning": self.evaluation_reasoning,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "confidence_level": self.confidence_level
        }
