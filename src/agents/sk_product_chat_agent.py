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
You are an intelligent RAG (Retrieval Augmented Generation) assistant. Your goal is to answer user questions using the available MCP tools.

CHAT CONVERSATION STRATEGY:
For normal user conversations and questions, you should ONLY use these two functions:
1. generate_chat_response - For standard comprehensive answers
2. generate_evaluated_chat_response - For answers that need quality evaluation and accuracy scoring

These functions are complete solutions that internally handle:
- Searching the internal product database
- Generating and executing web search queries
- Combining information from multiple sources
- Creating comprehensive, well-cited responses

IMPORTANT CHAT GUIDELINES:
- DO NOT manually orchestrate multiple tools for chat conversations
- DO NOT use search_internal_products, generate_web_search_queries, web_search, or debug_event_details for chat
- Use generate_chat_response for most questions
- Use generate_evaluated_chat_response when evaluation/accuracy scoring is requested
- These chat functions will handle all the search and research steps internally
- Let the chat functions do the heavy lifting - they are complete RAG solutions

WHEN TO USE OTHER TOOLS:
The other tools (search_internal_products, etc.) are available for:
- Direct testing and debugging purposes
- When explicitly instructed to use a specific tool
- API endpoints that call tools directly (not chat conversations)

Remember: For chat conversations, use ONLY the generate_chat_response or generate_evaluated_chat_response functions!
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
            
            if use_evaluation:
                full_question += "\n\nPlease use generate_evaluated_chat_response for this question."
            
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
            
            # Add evaluation instruction if needed
            full_question = question
            if use_evaluation:
                full_question += "\n\nPlease use generate_evaluated_chat_response for this question."
            
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
        Generate a simple chat response using only the chat response functions.
        
        This method restricts the agent to ONLY use generate_chat_response or 
        generate_evaluated_chat_response for chat interactions, ensuring a 
        focused conversational experience.
        
        Args:
            question: The user's question
            context: Additional context
            use_web_search: Whether to use web search
            use_evaluation: Whether to use evaluation
            
        Returns:
            Chat response
        """
        try:
            agent = await self._get_or_create_agent()
            
            # Build the question with strict instructions
            full_question = question
            if context:
                full_question = f"Context: {context}\n\nQuestion: {question}"
            
            # Add VERY specific function instruction with restrictions
            func_name = "generate_evaluated_chat_response" if use_evaluation else "generate_chat_response"
            
            full_question += f"""

IMPORTANT INSTRUCTIONS:
- You MUST use the {func_name} function to answer this question
- Do NOT use any other MCP tools like search_internal_products, debug_event_details, or generate_web_search_queries
- The {func_name} function will handle all searching and research internally
- Simply call {func_name} with the user's question as the user_query parameter"""
            
            if not use_web_search:
                full_question += "\n- Set n_web_results to 0 to disable web search."
            
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
        
        # Count MCP functions by checking for known function names
        if self.mcp_plugin:
            mcp_function_names = [
                'search_internal_products',
                'web_search', 
                'generate_web_search_queries',
                'generate_chat_response',
                'generate_evaluated_chat_response'
            ]
            
            for func_name in mcp_function_names:
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
