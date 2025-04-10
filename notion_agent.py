from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
import asyncio
import logging
import json
from typing import Dict, Optional, List
from datetime import datetime

# Load environment variables and configure basic logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a proactive Notion assistant with access to various Notion operations through tools.
Your goal is to help users manage their Notion workspace efficiently by:

1. Taking initiative to complete multi-step tasks without asking for confirmation at each step
2. Breaking down complex requests into a sequence of operations
3. Using your memory of previous interactions to make informed decisions
4. Proactively suggesting related actions that might be helpful
5. Explaining your reasoning and planned steps before executing them

When you receive a request:
1. First analyze what needs to be done and break it down into steps
2. Execute all necessary steps in sequence without asking for permission for each one
3. Only ask for clarification if truly necessary
4. Keep track of what you've done and use that context for future actions
5. After completing the main task, suggest any relevant follow-up actions

Remember: You have access to Notion tools - use them proactively to gather information and take actions.
Don't ask for permission for basic operations - just execute them and report the results."""

class NotionAgentMemory:
    """Manages conversation history and API response caching for the Notion agent."""
    
    def __init__(self, cache_duration: int = 300):
        """
        Initialize memory systems.
        
        Args:
            cache_duration (int): Duration in seconds to keep API responses in cache (default: 300)
        """
        self.conversation_memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        self.api_cache = {}
        self.cache_file = "notion_api_cache.json"
        self.cache_duration = cache_duration
        self.action_history: List[Dict] = []  # Track actions taken
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load API cache from file if it exists."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.api_cache = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.api_cache = {}
    
    def _save_cache(self) -> None:
        """Save API cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.api_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def cache_api_result(self, tool_name: str, input_params: Dict, result: Dict) -> None:
        """
        Cache an API result with timestamp.
        
        Args:
            tool_name: Name of the API tool
            input_params: Parameters used in the API call
            result: API response to cache
        """
        cache_key = f"{tool_name}:{json.dumps(input_params, sort_keys=True)}"
        self.api_cache[cache_key] = {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        self._save_cache()
        
        # Track this action
        self.action_history.append({
            "tool": tool_name,
            "params": input_params,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_cached_result(self, tool_name: str, input_params: Dict) -> Optional[Dict]:
        """
        Get cached API result if it exists and is not expired.
        
        Args:
            tool_name: Name of the API tool
            input_params: Parameters used in the API call
            
        Returns:
            Cached result if valid, None otherwise
        """
        cache_key = f"{tool_name}:{json.dumps(input_params, sort_keys=True)}"
        if cache_key in self.api_cache:
            cached_data = self.api_cache[cache_key]
            cached_time = datetime.fromisoformat(cached_data["timestamp"])
            if (datetime.now() - cached_time).total_seconds() < self.cache_duration:
                return cached_data["result"]
        return None
    
    def add_to_memory(self, user_message: str, assistant_message: str) -> None:
        """Add a conversation turn to memory."""
        self.conversation_memory.chat_memory.add_user_message(user_message)
        self.conversation_memory.chat_memory.add_ai_message(assistant_message)
    
    def get_memory_variables(self) -> Dict:
        """Get current memory variables and action history."""
        memory_vars = self.conversation_memory.load_memory_variables({})
        
        # Add action history to the context
        if self.action_history:
            actions_summary = "\nRecent actions taken:\n"
            for action in self.action_history[-5:]:  # Last 5 actions
                actions_summary += f"- Used {action['tool']} at {action['timestamp']}\n"
            
            # Add this context to the last message if it exists
            if memory_vars.get("chat_history"):
                last_msg = memory_vars["chat_history"][-1]
                if isinstance(last_msg, AIMessage):
                    last_msg.content += actions_summary
        
        return memory_vars
    
    def clear_memory(self) -> None:
        """Clear both conversation memory and API cache."""
        self.conversation_memory.clear()
        self.api_cache = {}
        self.action_history = []
        self._save_cache()

async def create_notion_agent():
    """Create and initialize a Notion agent with memory capabilities."""
    
    # Initialize LLM with OpenRouter
    llm = ChatOpenAI(
        temperature=0,
        model="google/gemini-2.0-flash-001",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Notion MCP Agent"
        }
    )
    
    # Set up MCP client for Notion
    client = MultiServerMCPClient({
        "notion": {
            "command": "node",
            "args": ["/Applications/Utilities/notionmcp/build/index.js"],
            "transport": "stdio",
            "env": {
                "NOTION_API_TOKEN": os.getenv("NOTION_API_TOKEN")
            }
        }
    })
    
    await client.__aenter__()
    memory = NotionAgentMemory()
    
    # Create the agent with the MCP tools
    agent = create_react_agent(llm, client.get_tools())
    
    return agent, client, memory

async def process_message(message: str, agent, memory: NotionAgentMemory) -> str:
    """
    Process a message through the agent with memory context.
    
    Args:
        message: User's input message
        agent: The ReAct agent
        memory: Memory system for maintaining context
        
    Returns:
        Agent's response as a string
    """
    try:
        # Get conversation history and combine with current message
        chat_history = memory.get_memory_variables().get("chat_history", [])
        
        # Create message list with system prompt first
        messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ] + chat_history + [HumanMessage(content=message)]
        
        # Process through agent and update memory
        response = await agent.ainvoke({"messages": messages})
        response_content = response["messages"][-1].content
        memory.add_to_memory(message, response_content)
        
        return response_content
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return f"Error processing message: {str(e)}"

if __name__ == "__main__":
    asyncio.run(create_notion_agent())

    