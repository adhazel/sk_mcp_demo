"""
ChromaDB search functionality for retrieving internal product information.
"""
import asyncio
import logging
from typing import List, Dict, Any
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from src.utils.mcp_config import Config
from src.models import SearchResult, SearchResults

logger = logging.getLogger(__name__)

class ChromaDBSearcher:
    """Search internal product database using ChromaDB."""
    
    def __init__(self, config: Config):
        self.config = config
        self._client = None
        self._collection = None
    
    def _get_chroma_path(self) -> Path:
        """Get the ChromaDB storage path."""
        return Path(self.config.project_root / self.config.chroma_db_path.lstrip('./'))
    
    async def _get_client(self):
        """Get or create the ChromaDB client."""
        if self._client is None:
            chroma_path = self._get_chroma_path()
            logger.info(f"Connecting to ChromaDB at: {chroma_path}")
            
            # Ensure the directory exists
            chroma_path.mkdir(parents=True, exist_ok=True)
            
            # Use thread pool for synchronous ChromaDB operations
            loop = asyncio.get_event_loop()
            self._client = await loop.run_in_executor(
                None, 
                lambda: chromadb.PersistentClient(path=str(chroma_path))
            )
        return self._client
    
    def _get_embedding_function(self):
        """Create the embedding function for ChromaDB."""
        if self.config.openai_api_type == "azure":
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.config.azure_openai_embedding_api_key,
                api_base=self.config.azure_openai_embedding_endpoint,
                api_type=self.config.openai_api_type,
                api_version=self.config.azure_openai_embedding_api_version,
                model_name=self.config.azure_openai_embedding_model,
                deployment_id=self.config.azure_openai_embedding_deployment
            )
        else: 
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.config.openai_api_key,
                model_name=self.config.openai_model
            )
    
    async def _get_collection(self, collection_name: str = "product_collection"):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            client = await self._get_client()
            loop = asyncio.get_event_loop()
            
            chroma_embedding_function = self._get_embedding_function()

            try:
                # Use get_or_create_collection like working code
                self._collection = await loop.run_in_executor(
                    None,
                    lambda: client.get_or_create_collection(
                        name=collection_name,
                        embedding_function=chroma_embedding_function
                    )
                )
                logger.info(f"Successfully connected to collection '{collection_name}'")
            except Exception as e:
                logger.error(f"Error getting collection: {e}", exc_info=True)
                raise e
        return self._collection

    async def search_chroma(self, query: str, n_results: int = 5, collection_name: str = "product_collection") -> List[Dict[str, Any]]:
        """
        Search for relevant products in the database.
        
        Args:
            query: What to search for
            n_results: Maximum results to return
            collection_name: Database collection to search
            
        Returns:
            List of matching products with metadata
        """
        try:
            collection = await self._get_collection(collection_name)
            if collection is None:
                error_result = SearchResult(
                    query=query,
                    content="ChromaDB collection not found or not accessible",
                    citation="[Source: ChromaDB | Status: Collection Not Found]",
                    metadata={"error": "collection_not_found"},
                    context_id=-1,
                    source_type="internal"
                )
                return SearchResults(queries=[error_result]).to_list()
            
            # Perform similarity search using thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    include=['documents', 'metadatas', 'distances']
                )
            )
            
            # Transform results to consistent format
            search_results = []
            if results and 'documents' in results and results['documents']:
                documents = results['documents'][0]  # First query results
                metadatas = results.get('metadatas', [[]])[0]
                ids = results.get('ids', [[]])[0]
                distances = results.get('distances', [[]])[0]
                
                for i, (doc, metadata, doc_id, distance) in enumerate(zip(documents, metadatas, ids, distances)):
                    search_result = SearchResult(
                        query=query,
                        content=doc,
                        citation=f"[Source: ChromaDB | Collection: {collection_name} | ID: {doc_id} | Similarity: {1-distance:.3f}]",
                        metadata={
                            "collection": collection_name,
                            "document_id": doc_id,
                            "similarity_score": 1 - distance,
                            "distance": distance,
                            **metadata
                        },
                        context_id=i+1,
                        search_order=i + 1,
                        source_type="internal"  # Explicitly set for ChromaDB results
                    )
                    search_results.append(search_result)
            
            if search_results:
                return SearchResults(queries=search_results).to_list()
            else:
                no_results = SearchResult(
                    query=query,
                    content="No results found",
                    citation=f"[Source: ChromaDB | Collection: {collection_name} | Status: No Results]",
                    metadata={"collection": collection_name, "result_count": 0},
                    context_id=-1,
                    source_type="internal"
                )
                return SearchResults(queries=[no_results]).to_list()
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            raise e

    async def get_collection_info(self, collection_name: str = "product_collection") -> Dict[str, Any]:
        """
        Get information about a database collection.
        
        Args:
            collection_name: Name of the collection to inspect
            
        Returns:
            Collection metadata and document count
        """
        try:
            collection = await self._get_collection(collection_name)
            if collection is None:
                return {
                    "name": collection_name,
                    "exists": False,
                    "error": "Collection not found"
                }
            
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(None, collection.count)
            
            return {
                "name": collection_name,
                "exists": True,
                "document_count": count,
                "embedding_function": self.config.azure_openai_embedding_model,
                "description": f"ChromaDB collection '{collection_name}' with {count} documents"
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise e
