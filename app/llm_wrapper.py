from langchain_openai import ChatOpenAI  # Updated import
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()

# Load env vars
LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL", "https://api.openai.com/v1") 
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "openai/gpt-3.5-turbo")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "missing-key")

# Initialize LiteLLM-backed LLM
llm = ChatOpenAI(
    model=LITELLM_MODEL,
    openai_api_key=LITELLM_API_KEY,
    openai_api_base=LITELLM_PROXY_URL,
    temperature=0.7,
    max_tokens=2048
)

# Define tools
from app.tools import search_tool

# Initialize agent
def create_agent():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    agent = initialize_agent(
        tools=[search_tool],
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )
    return agent

agent = create_agent()