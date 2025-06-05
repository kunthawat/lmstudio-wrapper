# app/llm_wrapper.py

import os
from langchain_openai import ChatOpenAI  # Updated import (no longer from langchain.chat_models)
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool  # Import Tool explicitly
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. Load Environment Variables
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://api.openai.com/v1") 
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "openai/gpt-3.5-turbo")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "missing-key")

# 2. Initialize LiteLLM-Backed LLM
llm = ChatOpenAI(
    model=LITELLM_MODEL,
    openai_api_key=LITELLM_API_KEY,
    openai_api_base=LITELLM_BASE_URL,
    temperature=0.7,
    max_tokens=8096
)

# 3. Define Tools (Example Tool)
def search(query: str) -> str:
    """Example tool - replace with real functionality"""
    return f"Search result for '{query}'"

# Create tool instance
search_tool = Tool(
    name="Search",
    func=search,
    description="Useful for searching information."
)

# 4. Initialize Agent
def create_agent():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    agent = initialize_agent(
        tools=[search_tool],
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True  # Important for robustness
    )
    
    return agent

# 5. Create Singleton Agent Instance
agent = create_agent()

# 6. Helper for Structured Tool Calls (Optional)
def format_tool_call(tool_name: str, input_str: str) -> str:
    """Format response to match n8n's expectations"""
    return f"Thought: Calling {tool_name}\nAction: {tool_name}\nAction Input: {input_str}"