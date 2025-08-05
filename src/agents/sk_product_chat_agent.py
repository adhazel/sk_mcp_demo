"""
Semantic Kernel Product Chat Agent
This agent uses Semantic Kernel Agents to provide intelligent product assistance via MCP RAG server.
Specializes in product search, recommendations, and Q&A interactions.
"""

import json
import logging
from typing import Dict, Any, Optional

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion, 
    OpenAIChatCompletion,
    AzureTextEmbedding,
    OpenAITextEmbedding
)
from semantic_kernel.contents import ChatHistory

# Import Agent components
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread

# Import custom MCP plugin
from ..client.plugins.semantic_kernel import create_mcp_plugin

from ..utils.config import Config


class ProductChatAgent:
    """
    A specialized chat agent that provides intelligent product assistance using Semantic Kernel Agents.
    Coordinates calls to the MCP RAG server for product search, recommendations, and Q&A interactions.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Product Chat Agent.
        
        Args:
            config: Configuration object. If None, will create a new one.
        """
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize kernel
        self.kernel = Kernel()
        
        # Setup AI services
        self._setup_ai_services()
        
        # Initialize the agent (will be created when needed)
        self.agent = None
        self.mcp_plugin = None
        
        self.logger.info("✅ Product Chat Agent initialized")

    async def _get_or_create_agent(self) -> ChatCompletionAgent:
        """
        Get or create the ChatCompletionAgent with MCP plugin.
        
        Returns:
            ChatCompletionAgent instance
        """
        if self.agent is None:
            # Create MCP plugin if not exists
            if self.mcp_plugin is None:
                self.mcp_plugin = await create_mcp_plugin(
                    kernel=self.kernel,
                    server_url=self.config.mcp_server_url
                )
            
            # Get the chat service
            chat_service = self.kernel.get_service("chat")
            
            # Create agent with MCP plugin
            self.agent = ChatCompletionAgent(
                service=chat_service,
                name="RAGAgent",
                instructions=self._get_agent_instructions(),
                plugins=[self.mcp_plugin]
            )
            
            self.logger.info("✅ Chat Completion Agent created with custom MCP plugin")
        
        return self.agent

    def _setup_ai_services(self):
        """Setup AI services based on configuration."""
        try:
            if self.config.openai_api_type.lower() == "azure":
                # Azure OpenAI setup
                chat_service = AzureChatCompletion(
                    api_key=self.config.azure_openai_api_key,
                    endpoint=self.config.azure_openai_endpoint,
                    deployment_name=self.config.azure_openai_deployment,
                    api_version=self.config.azure_openai_api_version,
                    service_id="chat"
                )
                
                # Add embedding service if configured
                if (self.config.azure_openai_embedding_api_key and 
                    self.config.azure_openai_embedding_endpoint and
                    self.config.azure_openai_embedding_deployment):
                    
                    embedding_service = AzureTextEmbedding(
                        api_key=self.config.azure_openai_embedding_api_key,
                        endpoint=self.config.azure_openai_embedding_endpoint,
                        deployment_name=self.config.azure_openai_embedding_deployment,
                        api_version=self.config.azure_openai_embedding_api_version,
                        service_id="embedding"
                    )
                    self.kernel.add_service(embedding_service)
                    
            else:
                # OpenAI setup
                chat_service = OpenAIChatCompletion(
                    api_key=self.config.openai_api_key,
                    ai_model_id=self.config.openai_model or "gpt-4",
                    service_id="chat"
                )
                
                # Add embedding service
                embedding_service = OpenAITextEmbedding(
                    api_key=self.config.openai_api_key,
                    ai_model_id="text-embedding-3-small",
                    service_id="embedding"
                )
                self.kernel.add_service(embedding_service)
            
            self.kernel.add_service(chat_service)
            self.logger.info(f"✅ AI services configured for {self.config.openai_api_type}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup AI services: {e}")
            raise

    def _get_agent_instructions(self) -> str:
        """
        Get the instructions for the RAG agent.
        
        Returns:
            Agent instructions string
        """
        return """
You are a product chat assistant whose job is to help users with product-related questions using available MCP tools. You have access to MCP tools through the call_mcp_tool function.

AVAILABLE MCP TOOLS (accessed via call_mcp_tool):
1. **search_internal_products** - Search the internal product catalog using ChromaDB
   - Parameters: {"query": "search terms", "n_results": 5}
   - Returns: List of dictionaries with internal search results, citations, and metadata
   - Use this first for any product-related questions

2. **search_external_web** - Search the web for additional information
   - Parameters: {"user_query": "search terms", "n_results_per_search": 2, "internal_context": [optional results from search_internal_products]}
   - This tool automatically generates web search queries based on the user question and any internal context provided
   - Returns: List of dictionaries with external search results, citations, and metadata

3. **evaluate_response** - Evaluate response accuracy and quality
   - Parameters: {"user_query": "original question", "response": "response to evaluate", "context": "search results context"}
   - Use when users ask for evaluation, accuracy checking, or quality assessment
   - Examples: "How accurate is this?", "Please evaluate this answer", "Check the quality"

HOW TO USE MCP TOOLS:
- Use the call_mcp_tool function with tool_name and parameters (as JSON string)
- Example: call_mcp_tool(tool_name="search_internal_products", parameters='{"query": "hiking boots", "n_results": 5}')
- Example: call_mcp_tool(tool_name="search_external_web", parameters='{"user_query": "waterproof hiking boots", "n_results_per_search": 3}')

WORKFLOW:
1. **Always start with internal search** using search_internal_products for any product-related question
   - If results are relevant to the user's question, use them in your response
   - If no relevant results, inform the user that their question is outside the product catalog scope

2. **Follow up with external search for enhanced context**:
   - For most product questions, perform an external web search to provide comprehensive information
   - Use search_external_web with the user's question and pass internal results as internal_context
   - This demonstrates the power of combining internal product data with external market information
   - Skip external search ONLY if the question is completely unrelated to products or if the user explicitly requests internal-only information

3. **Combine information sources** to provide comprehensive, accurate responses that showcase both internal product knowledge and external market context

4. **Use evaluation tool** when specifically requested by the user or when you want to assess response quality

IMPORTANT NOTES:
- Always use call_mcp_tool to access MCP server functions
- Format parameters as valid JSON strings
- Prefer using BOTH internal and external searches to demonstrate multi-tool orchestration
- The search_external_web tool is intelligent - it generates its own web search queries based on the user question and internal context
- Always preserve the exact format of search results as they may be needed for evaluation
- Be helpful and professional in combining results from multiple sources
- Show users the value of comprehensive research by using multiple information sources

"""

    async def process_question(
        self, 
        question: str, 
        context: str = "",
        use_evaluation: bool = True,
        conversation_history: Optional[ChatHistory] = None,
        thread: Optional[ChatHistoryAgentThread] = None
    ) -> Dict[str, Any]:
        """
        Process a user question using the MCP RAG server with intelligent orchestration.
        
        Args:
            question: The user's question
            context: Additional context for the question
            use_evaluation: Whether to use evaluation and risk scoring
            conversation_history: Previous conversation history (legacy support)
            thread: Agent thread for conversation continuity
            
        Returns:
            Dict containing the response and metadata
        """
        self.logger.info(f"🤔 Processing question: {question}")
        
        try:
            # Get or create the agent
            agent = await self._get_or_create_agent()
            
            # Build the full question with context if provided
            full_question = question
            if context:
                full_question = f"Context: {context}\n\nQuestion: {question}"
            
            # Add instruction to encourage using both internal and external search tools
            full_question += "\n\nNote: Please use both internal product search AND external web search tools to demonstrate multiple tool usage. Show how combining information from both sources enhances the response."
            
            # Note: evaluation can be done using the evaluate_response MCP tool if needed
            
            # Use agent to get response with thread management
            response = await agent.get_response(messages=full_question, thread=thread)
            
            # Extract response data
            response_data = {
                "question": question,
                "response": {"response": str(response)},
                "context_used": context,
                "evaluation_enabled": use_evaluation,
                "status": "success",
                "orchestrator": "semantic_kernel_agent_mcp",
                "thread_id": response.thread.id if response.thread else None
            }
            
            self.logger.info("✅ Question processed successfully")
            return response_data
        
        except Exception as e:
            self.logger.error(f"❌ Error processing question: {e}")
            return {
                "error": str(e),
                "question": question,
                "status": "failed"
            }

    async def start_conversation(self) -> ChatHistoryAgentThread:
        """
        Start a new conversation thread.
        
        Returns:
            New ChatHistoryAgentThread (None initially, will be created on first message)
        """
        await self._get_or_create_agent()  # Ensure agent is ready
        return None  # Thread will be created automatically on first message

    async def continue_conversation(
        self, 
        question: str, 
        thread: Optional[ChatHistoryAgentThread] = None,
        use_evaluation: bool = True
    ) -> Dict[str, Any]:
        """
        Continue a conversation with the agent using a thread.
        
        Args:
            question: The user's question
            thread: Existing thread to continue, or None to start new
            use_evaluation: Whether to use evaluation
            
        Returns:
            Dict containing response and updated thread
        """
        try:
            agent = await self._get_or_create_agent()
            
            # Add instruction to encourage using both internal and external search tools
            full_question = question
            full_question += "\n\nNote: Please use both internal product search AND external web search tools to demonstrate multiple tool usage. Show how combining information from both sources enhances the response."
            
            # Note: evaluation can be done using the evaluate_response MCP tool if needed
            
            # Get response using the thread
            response = await agent.get_response(messages=full_question, thread=thread)
            
            return {
                "question": question,
                "response": {"response": str(response)},
                "status": "success",
                "thread": response.thread,  # Return the thread for continued conversation
                "thread_id": response.thread.id if response.thread else None,
                "orchestrator": "semantic_kernel_agent_mcp"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Conversation error: {e}")
            return {
                "error": str(e),
                "question": question,
                "status": "failed",
                "thread": thread
            }

    async def end_conversation(self, thread: Optional[ChatHistoryAgentThread]) -> None:
        """
        End a conversation and clean up the thread.
        
        Args:
            thread: Thread to clean up
        """
        if thread:
            try:
                await thread.delete()
                self.logger.info("✅ Conversation thread cleaned up")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to clean up thread: {e}")

    async def simple_search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Perform a simple internal product search using the agent.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            Search results
        """
        try:
            agent = await self._get_or_create_agent()
            
            # Create a focused search question
            search_question = f"Please search the internal product database for '{query}' and return up to {limit} results using search_internal_products."
            
            # Get response
            response = await agent.get_response(messages=search_question)
            
            # Try to parse the response as JSON if possible
            response_text = str(response)
            try:
                if response_text.strip().startswith('{') or response_text.strip().startswith('['):
                    return json.loads(response_text)
                else:
                    return {"results": response_text, "query": query, "limit": limit}
            except json.JSONDecodeError:
                return {"results": response_text, "query": query, "limit": limit}
                
        except Exception as e:
            self.logger.error(f"❌ Simple search failed: {e}")
            return {"error": str(e), "query": query}

    async def simple_chat(
        self, 
        question: str, 
        context: str = "",
        use_web_search: bool = True,
        use_evaluation: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a simple chat response using MCP tools for product assistance.
        
        Args:
            question: The user's question
            context: Additional context
            use_web_search: Whether to allow web search (defaults to True to encourage multi-tool usage)
            use_evaluation: Whether to use evaluation (if requested by user)
            
        Returns:
            Chat response
        """
        try:
            agent = await self._get_or_create_agent()
            
            # Build the question with context
            full_question = question
            if context:
                full_question = f"Context: {context}\n\nQuestion: {question}"
            
            # Add instruction to encourage both internal and external searches
            if use_web_search:
                full_question += "\n\nNote: Please use both internal product search AND external web search to demonstrate multiple tool usage. Show how combining information from both sources enhances the response."
            else:
                full_question += "\n\nNote: Please focus on internal product catalog only, avoid external web searches."
            
            # Get response
            response = await agent.get_response(messages=full_question)
            
            # Try to parse as JSON if possible
            response_text = str(response)
            try:
                if response_text.strip().startswith('{'):
                    return json.loads(response_text)
                else:
                    return {"response": response_text, "question": question}
            except json.JSONDecodeError:
                return {"response": response_text, "question": question}
            
        except Exception as e:
            self.logger.error(f"❌ Simple chat failed: {e}")
            return {"error": str(e), "question": question}

    async def get_available_functions(self) -> Dict[str, Any]:
        """
        Get information about available MCP functions from the agent.
        
        Returns:
            Dictionary of available functions and their descriptions
        """
        try:
            # Ensure agent is created (initializes MCP plugin)
            await self._get_or_create_agent()
            
            functions = {}
            
            # Use custom MCP plugin to discover functions
            if self.mcp_plugin:
                try:
                    # Get health check to ensure connection is working
                    health = await self.mcp_plugin.health_check()
                    
                    if health["status"] == "healthy":
                        # Get available tools from the plugin
                        tools = self.mcp_plugin.get_primitives_by_type("tool")
                        
                        for tool in tools:
                            functions[tool.name] = tool.description or f"MCP Tool: {tool.name}"
                        
                        # Add the built-in kernel functions from the plugin
                        kernel_functions = [
                            "list_available_tools",
                            "list_available_resources", 
                            "mcp_health_check",
                            "call_mcp_tool"
                        ]
                        
                        for func_name in kernel_functions:
                            functions[func_name] = f"MCP Plugin Function: {func_name.replace('_', ' ').title()}"
                    
                    else:
                        self.logger.warning(f"⚠️ MCP server health check failed: {health.get('error', 'Unknown error')}")
                        return {
                            "available_functions": {},
                            "mcp_server_url": self.config.mcp_server_url,
                            "status": "health_check_failed",
                            "error": health.get('error', 'Unknown error'),
                            "function_count": 0
                        }
                        
                except Exception as connect_error:
                    self.logger.warning(f"⚠️ Could not connect to MCP server to discover functions: {connect_error}")
                    return {
                        "available_functions": {},
                        "mcp_server_url": self.config.mcp_server_url,
                        "status": "connection_failed",
                        "error": str(connect_error),
                        "function_count": 0
                    }
            
            return {
                "available_functions": functions,
                "mcp_server_url": self.config.mcp_server_url,
                "status": "available" if functions else "no_functions_discovered",
                "function_count": len(functions)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get available functions: {e}")
            return {
                "error": str(e),
                "mcp_server_url": self.config.mcp_server_url,
                "status": "failed"
            }

    async def cleanup(self):
        """Clean up resources including MCP plugin connections."""
        try:
            if self.mcp_plugin:
                # Try to close the MCP plugin connection gracefully
                if hasattr(self.mcp_plugin, 'close'):
                    await self.mcp_plugin.close()
                self.logger.info("✅ MCP plugin cleaned up")
        except Exception as e:
            self.logger.warning(f"⚠️ Error during MCP plugin cleanup: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.cleanup()

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Product Chat Agent.
        
        Returns:
            Status information
        """
        # Get basic status info
        function_count = 0
        
        # Count MCP functions by checking for the actual available plugin functions
        if self.mcp_plugin:
            plugin_function_names = [
                'list_available_tools',
                'list_available_resources', 
                'mcp_health_check',
                'call_mcp_tool'
            ]
            
            for func_name in plugin_function_names:
                if hasattr(self.mcp_plugin, func_name) and callable(getattr(self.mcp_plugin, func_name)):
                    function_count += 1
        
        return {
            "orchestrator": "ProductChatAgent",
            "approach": "ChatCompletionAgent with MCPStreamableHttpPlugin",
            "mcp_server_url": self.config.mcp_server_url,
            "ai_service": self.config.openai_api_type,
            "max_iterations": self.config.sk_max_iterations,
            "temperature": self.config.sk_temperature,
            "agent_created": self.agent is not None,
            "mcp_plugin_created": self.mcp_plugin is not None,
            "functions_available": function_count,
            "status": "ready"
        }
