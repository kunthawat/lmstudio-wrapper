from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()

# Load env vars
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
LITELLM_MODEL = os.getenv("LITELLM_MODEL")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")

if not all([LITELLM_BASE_URL, LITELLM_MODEL, LITELLM_API_KEY]):
    raise ValueError("Missing required LiteLLM environment variables")

# Initialize LiteLLM model
llm = ChatOpenAI(
    openai_api_key=LITELLM_API_KEY,
    base_url=LITELLM_BASE_URL,
    model_name=LITELLM_MODEL
)

# Sample tool
from app.tools import search_tool
tools = [search_tool]

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)