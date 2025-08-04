"""Web search query models."""

from typing import Dict, Any, List
from pydantic import BaseModel, Field


class GeneratedSearchQuery(BaseModel):
    """A generated search query with metadata."""
    priority_rank: int
    search_query: str
    purpose: str


class GeneratedSearchQueries(BaseModel):
    """Collection of generated search queries."""
    queries: List[GeneratedSearchQuery] = Field(default_factory=list)

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert queries to dictionary format."""
        return [query.model_dump() for query in self.queries]
