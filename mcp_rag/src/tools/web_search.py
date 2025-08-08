"""
Web search functionality using SerpAPI Bing for external research.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiohttp
from utils import McpConfig
from models import SearchResult, SearchResults, GeneratedSearchQueries

logger = logging.getLogger(__name__)

class WebSearcher:
    """Search the web using SerpAPI Bing."""

    def __init__(self, config: McpConfig):
        self.config = config
        self.base_url = "https://serpapi.com/search"
        # Create Azure OpenAI client for query generation
        self._llm_client = None
    
    def _get_llm_client(self):
        """Get the LLM client for query generation."""
        if self._llm_client is None:
            self._llm_client = self.config.get_llm()
        return self._llm_client

    def _format_context_for_llm(self, context: List[Dict[str, Any]] = None) -> str:
        """Format search results for LLM processing."""
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

    async def _get_web_search_queries(self, user_query: str, internal_context: Optional[list] = None) -> List[Dict[str, Any]]:
        """
        Generate smart web search queries based on a user question.
        
        Args:
            user_query: The question to research
            internal_context: Internal search results to inform web queries (can be None or empty list)
            
        Returns:
            List of optimized search queries with priorities
        """
        formatted_context = self._format_context_for_llm(internal_context)

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
  - If no internal context is provided, generate queries based on the user query alone.

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
Internal Context: 
{formatted_context}

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
                    return []
                return GeneratedSearchQueries(queries=parsed_response.queries).to_list()

            except Exception as e:
                logger.error(f"Error generating web search queries: {e}")
                raise e
        
        # Run the synchronous query generation in a thread pool to make it async
        loop = asyncio.get_event_loop()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, _sync_query_generation)
    
    def _transform_serpapi_bing_results(
        self, 
        serp_query: str,
        serp_response: Dict[str, Any],
        n_results: int = 5,
        includes: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Transform SerpAPI Bing search results into SearchResults format
        """
        if includes is None:
            includes = ["organic_results"]
        
        # Check if serp_response is a valid dictionary
        if not isinstance(serp_response, dict):
            logger.warning(f"Invalid SERP response format: expected dict, got {type(serp_response)}")
            return []
        
        # Check for API error responses
        if "error" in serp_response:
            logger.warning(f"SERP API error: {serp_response.get('error', 'Unknown error')}")
            return []
        
        results = []
        engine_name = "Bing"
        
        def add_item(query: str, content: str, citation: str, metadata: Dict, cid: int, search_order: int = -1):
            if content:
                search_result = SearchResult(
                    query=query,
                    content=content,
                    citation=citation,
                    metadata=metadata,
                    context_id=cid,
                    search_order=search_order,
                    source_type="external"  # Explicitly set for web search results
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
        
        return SearchResults(queries=results).to_list()

    async def search_serpapi_bing_with_query(self, query: str, n_results_per_search: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web using Bing via SerpAPI.
        
        Args:
            query: What to search for
            n_results_per_search: Maximum results to return
            
        Returns:
            List of web search results with content and links
        """
        if not query.strip():
            logger.warning("Empty search query provided")
            return []
            
        try:
            params = {
                "engine": "bing",
                "q": f"{query} -site:https://ell.stackexchange.com -site:ell.stackexchange.com -site:www.tenforums.com",  # Simplified exclusions
                "api_key": self.config.serp_api_key,
                "mkt": "en-us",
                "cc": "US", 
                "safeSearch": "on",
                "num": min(n_results_per_search * 2, 10)  # Cap max results to prevent slow responses
            }
            
            # Use shorter timeout and connection settings for faster responses
            timeout = aiohttp.ClientTimeout(total=12.0, connect=2.0)  # 12 second total, 2 second connect
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=10)
            
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                async with session.get(self.base_url, params=params) as response:
                    response.raise_for_status()
                    try:
                        json_data = await response.json()
                    except aiohttp.ContentTypeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        return []
                    except Exception as e:
                        logger.error(f"Unexpected error parsing response: {e}")
                        return []
                    
                    # Additional validation for the response
                    if not isinstance(json_data, dict):
                        logger.warning(f"Invalid response format: expected dict, got {type(json_data)}")
                        return []
                    
                    results = self._transform_serpapi_bing_results(
                        serp_query=query,
                        serp_response=json_data,
                        n_results=n_results_per_search
                    )
                    
                    # Ensure we don't exceed requested results
                    return results[:n_results_per_search] if len(results) > n_results_per_search else results

        except asyncio.TimeoutError:
            logger.warning(f"Search has timed out for query '{query[:50]}...'")
            return []
        except aiohttp.ClientError as e:
            logger.warning(f"HTTP error searching for '{query[:50]}...': {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching for '{query[:50]}...': {e}")
            return []

    async def search_bing_with_chat_and_context(self, user_query: str, internal_context: Optional[list] = None, n_results_per_search: int = 5) -> list:
        """
        Execute multiple web searches using generated queries.
        
        Args:
            user_query: The original user query
            internal_context: Contextual information to consider (can be None or empty list)
            n_results_per_search: Number of results to return per search

        Returns:
            Combined results from all searches
        """
        # Execute searches concurrently with error handling
        all_results = []

        # Handle None or empty list for internal_context
        if internal_context is None:
            internal_context = []

        generated_queries = await self._get_web_search_queries(user_query=user_query, internal_context=internal_context)
        # Early return for empty queries
        if not generated_queries:
            logger.info("No search queries provided, returning empty results")
            return []
        
        # Sort queries by priority rank but use all queries provided
        sorted_queries = sorted(generated_queries, key=lambda q: q.get('priority_rank', 1))
        
        logger.info(f"Running {len(sorted_queries)} web searches with {n_results_per_search} results each (max {len(sorted_queries) * n_results_per_search} total)")
        
        # Create search tasks with timeout
        search_tasks = []
        for i, query in enumerate(sorted_queries):
            search_query = query.get('search_query', '')
            if search_query.strip():  # Only search non-empty queries
                task = asyncio.create_task(
                    asyncio.wait_for(
                        self.search_serpapi_bing_with_query(search_query, n_results_per_search),
                        timeout=20.0  # 20 second timeout per search
                    )
                )
                search_tasks.append((i, task))
        
        
        if search_tasks:
            # Use asyncio.gather with return_exceptions=True to handle individual failures
            task_results = await asyncio.gather(
                *[task for _, task in search_tasks], 
                return_exceptions=True
            )
            
            # Process results and handle exceptions
            for (query_index, _), result in zip(search_tasks, task_results):
                if isinstance(result, Exception):
                    query_text = sorted_queries[query_index].get('search_query', 'unknown')
                    logger.warning(f"Skipping web search for query '{query_text}': {result}")
                    continue
                
                # Add results with updated metadata
                for search_result in result:
                    search_result['query_index'] = query_index
                    search_result['source_type'] = 'external'
                    all_results.append(search_result)
        
        # Update search_order to be sequential across all results
        for i, result in enumerate(all_results):
            result['search_order'] = i + 1
            # Keep original context_id if it exists, otherwise use search_order
            if result.get('context_id', -1) == -1:
                result['context_id'] = i + 1
        
        logger.info(f"Web search completed: {len(all_results)} total results from {len(sorted_queries)} queries")
        return all_results
