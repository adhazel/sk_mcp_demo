"""Search result data models."""

from typing import Dict, Any, List
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Search result from internal database or external web."""
    query: str
    content: str
    citation: str
    metadata: Dict[str, Any]
    context_id: int = -1
    search_order: int = -1
    source_type: str = Field(default="external", description="Source type: 'internal' for ChromaDB, 'external' for web search")


class SearchResults(BaseModel):
    """Collection of search results."""
    queries: List[SearchResult] = Field(default_factory=list)
    
    def __len__(self) -> int:
        """Get number of results."""
        return len(self.queries)
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Convert results to list of dictionaries."""
        return [instance.model_dump() for instance in self.queries]
