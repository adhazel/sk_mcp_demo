"""
Web Search Tools using SerpAPI Bing
"""
import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime
import aiohttp
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
    
class GeneratedSearchQuery(BaseModel):
    """Model for a generated search query"""
    priority_rank: int
    search_query: str
    purpose: str

class GeneratedSearchQueries(BaseModel):
    """Model for multiple generated search queries"""
    queries: List[GeneratedSearchQuery] = Field(default_factory=list)
    error: str = None

class WebSearcher:
    """Web search functionality using SerpAPI Bing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = "https://serpapi.com/search"
        # Create Azure OpenAI client for query generation
        self._llm_client = None
    
    def _get_llm_client(self):
        """Get or create LLM client from config"""
        if self._llm_client is None:
            self._llm_client = self.config.get_llm()
        return self._llm_client
    
    async def get_web_search_queries(self, user_query: str, internal_context: list = None) -> GeneratedSearchQueries:
        """
        Generate intelligent web search queries based on user query and internal context
        
        Args:
            user_query: The original user question
            internal_context: List of internal search results (ChromaDB results)
            
        Returns:
            GeneratedSearchQueries object with prioritized search queries
        """
        
        def _sync_query_generation():
            """Synchronous OpenAI API call to run in thread pool."""
            # System prompt for generating web search queries
            current_date = datetime.now().strftime("%B %d, %Y")
            system_message = f"""You are an assistant whose job is to transform a user's internal chatbot question plus the internal search results into one or more actionable web-search queries.

Inputs: 
  - today's date is: {current_date}
  - user_query (string): the original question the user posed to the chatbot
  - internal_results (array of search results with citations)

Steps: 
1. Intent Analysis
  - Determine what the user really wants, including any ambiguous or implied needs.
2. Context Review
  - Examine internal_results:
    - What sub-topics are already covered?
    - Where are the gaps or outdated pieces?
    - Is there jargon, acronyms or product names that need clarification?
3. External-Search Planning
  - For each gap or elaboration, craft a concise web search query.
  - Rank queries numerically by prioritization: highest (essential to answer), secondary, exploratory.
  - Aim for queries that return authoritative, up-to-date, and user-friendly results.
  - If the user includes and exact match to an internal product name and is asking for internal product information, do not generate a query. Internal context should be used to answer questions about internal products.
  - If the user asks questions about similar products or general information that is not specific to internal products, generate one more more queries.
  - When queries are generated, generate 1 to 3 queries, each with a clear purpose.
  - Generated search queries should include any relevant information from the internal context to optimize search results.

Output Format:
Return a JSON object with a "queries" array, each query object containing:
  - "priority_rank": integer (1, 2, 3, 4...)
  - "search_query": string
  - "purpose": brief description of what this query aims to find

Example:
{{
  "queries": [
    {{
      "priority_rank": 1,
      "search_query": "specific search terms",
      "purpose": "why this search helps"
    }}
  ]
}}
"""
            
            user_message = f"""User Query: {user_query}
Internal Context: {internal_context}

Should we search the web for additional information or is the internal context fully sufficient?"""
            
            try: 
                llm_client = self._get_llm_client()
                response = llm_client.chat.completions.parse(
                    model=self.config.azure_openai_deployment,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=4000,
                    temperature=0.4,
                    response_format=GeneratedSearchQueries
                )
                
                # Get parsed response
                parsed_response = response.choices[0].message.parsed
                if not parsed_response or not parsed_response.queries:
                    return GeneratedSearchQueries(queries=[])
                return GeneratedSearchQueries(queries=parsed_response.queries)
                
            except Exception as e:
                # Fallback: return an empty GeneratedSearchQueries object with error info
                return GeneratedSearchQueries(queries=[], error=str(e))
        
        # Run the synchronous query generation in a thread pool to make it async
        loop = asyncio.get_event_loop()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, _sync_query_generation)
    
    def transform_serpapi_bing_results(
        self, 
        serp_query: str,
        serp_response: Dict[str, Any],
        n_results: int = 5,
        includes: List[str] = None
    ) -> SearchResults:
        """
        Transform SerpAPI Bing search results into SearchResults format
        """
        if includes is None:
            includes = ["organic_results"]
        
        results = []
        engine_name = "Bing"
        
        def add_item(query: str, content: str, citation: str, metadata: Dict, cid: int, search_order: int = -1):
            if content:
                search_result = SearchResult(
                    query=query,
                    content=content,
                    citation=citation,
                    metadata=metadata,
                    content_id=cid,
                    search_order=search_order
                )
                results.append(search_result)
        
        # Answer box results
        if "answer_box" in includes:
            box = serp_response.get("answer_box")
            if box:
                links = [src["link"] for src in box.get("sources", []) if "link" in src]
                citation = (
                    f"[Source: {engine_name} Answer Box | "
                    f"Links: {', '.join(links)}]"
                )
                add_item(
                    query=serp_query,
                    content=box.get("snippet", ""),
                    citation=citation,
                    metadata={"highlighted_snippets": box.get("type")},
                    cid=1,
                    search_order=1
                )
        
        # Ads results
        if "ads" in includes:
            for idx, ad in enumerate(serp_response.get("ads", [])[:n_results], start=1):
                desc = ad.get("description")
                if not desc:
                    continue
                pos = int(ad.get("position", idx))
                citation = (
                    f"[Source: {engine_name} Ads | "
                    f"Link: {ad.get('tracking_link','')} | "
                    f"Position: {pos}]"
                )
                add_item(
                    query=serp_query,
                    content=desc,
                    citation=citation,
                    metadata={
                        "title": ad.get("title"),
                        "displayed_link": ad.get("displayed_link")
                    },
                    cid=pos,
                    search_order=pos
                )
        
        # Organic results
        if "organic_results" in includes:
            for idx, org in enumerate(serp_response.get("organic_results", [])[:n_results], start=1):
                snippet = org.get("snippet")
                link = org.get("link")
                if not (snippet and link):
                    continue
                pos = org.get("position", idx)
                citation = (
                    f"[Source: {engine_name} Search | "
                    f"Link: {link} | "
                    f"Position: {pos}]"
                )
                add_item(
                    query=serp_query,
                    content=snippet,
                    citation=citation,
                    metadata={
                        "title": org.get("title", "n/a"),
                        "displayed_link": org.get("displayed_link", "n/a"),
                        "snippet_highlighted_words": org.get("snippet_highlighted_words", []),
                    },
                    cid=pos,
                    search_order=pos
                )
        
        return SearchResults(queries=results)

    async def search_serpapi_bing(self, query: str, n_results: int = 5) -> SearchResults:
        """
        Perform web search using SerpAPI Bing
        
        Args:
            query: Search query string
            n_results: Number of results to return
            
        Returns:
            SearchResults object containing search results with content, citations, and metadata
        """
        try:
            params = {
                "engine": "bing",
                "q": f"{query} -site:ell.stackexchange.com -site:www.tenforums.com -site:https://ell.stackexchange.com -site:https://myaccount.mcafee.com/",
                "api_key": self.config.serp_api_key,
                "mkt": "en-us",
                "cc": "US", 
                "safeSearch": "on",
                "num": n_results * 2  # Request extra in case of filtering
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    response.raise_for_status()
                    json_data = await response.json()
                    results = self.transform_serpapi_bing_results(
                        serp_query=query,
                        serp_response=json_data,
                        n_results=n_results
                    )
                    # Limit to requested number of results
                    if len(results.queries) > n_results:
                        limited_results = SearchResults(queries=results.queries[:n_results])
                        return limited_results
                    return results
                    
        except Exception as e:
            logger.error(f"Error searching web for '{query}': {e}")
            error_result = SearchResult(
                query=query,
                content=f"Error performing web search: {str(e)}",
                citation=f"[Source: Bing Search | Error: {str(e)[:100]}]",
                metadata={"search_query": query, "error": str(e)},
                content_id=-1,
                search_order=1
            )
            return SearchResults(queries=[error_result])

    async def search_serpapi_bing_with_generated_queries(self, generated_queries: GeneratedSearchQueries, n_results: int = 5) -> SearchResults:
        """
        Perform web searches using generated queries from get_web_search_queries
        
        Args:
            generated_queries: GeneratedSearchQueries object with prioritized queries
            n_results: Number of results per query
            
        Returns:
            SearchResults object containing combined results from all generated queries
        """
        # Handle empty queries
        all_results = []
        total_queries = len(generated_queries.queries)
        
        if total_queries == 0:
            return SearchResults(queries=[])
        
        # Sort queries by priority rank to ensure proper ordering
        sorted_queries = sorted(generated_queries.queries, key=lambda q: q.priority_rank)
        
        # Run all searches concurrently
        search_tasks = [self.search_serpapi_bing(query.search_query, n_results) for query in sorted_queries]
        search_results_lists = await asyncio.gather(*search_tasks)
        
        # Flatten results from all queries
        for search_results in search_results_lists:
            all_results.extend(search_results.queries)
        
        # Update search_order to be sequential across all results
        for i, result in enumerate(all_results):
            result.search_order = i + 1
            if result.content_id != -1:
                result.content_id = i + 1
        
        return SearchResults(queries=all_results)

    async def search_multiple_queries(self, queries: List[str], n_results: int = 5) -> SearchResults:
        """
        Perform multiple web searches concurrently
        
        Args:
            queries: List of search query strings
            n_results: Number of results per query
            
        Returns:
            SearchResults object containing combined results from all queries
        """
        if not queries:
            return SearchResults(queries=[])
        
        # Run all searches concurrently
        search_tasks = [self.search_serpapi_bing(query, n_results) for query in queries]
        search_results_lists = await asyncio.gather(*search_tasks)
        
        # Flatten results from all queries
        all_results = []
        for search_results in search_results_lists:
            all_results.extend(search_results.queries)
        
        # Update search_order to be sequential across all results
        for i, result in enumerate(all_results):
            result.search_order = i + 1
            if result.content_id != -1:
                result.content_id = i + 1
        
        return SearchResults(queries=all_results)
