# app/main.py

# ✅ Required Imports
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from app.llm_wrapper import agent
from app.api_key_middleware import verify_api_key  # Ensure this exists

app = FastAPI()

# 🔧 OpenAI-Compatible Model List Endpoint
@app.get("/v1/models")
async def get_models(api_key: str = Depends(verify_api_key)):
    """
    Returns a list of available models from LiteLLM
    """
    import httpx
    
    LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{LITELLM_PROXY_URL}/v1/models")
            data = resp.json()
            
            # Optional: Filter/sanitize model names for n8n
            filtered_models = [
                {**model, "id": model["id"].replace("ollama_chat/", "")} 
                for model in data.get("data", [])
            ]
            
            return {"data": filtered_models}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch models: {str(e)}")

# 🔧 OpenAI-Compatible Chat Completion Endpoint
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 2048

@app.post("/v1/chat/completions")
async def openai_compatible_chat(
    request: ChatCompletionRequest, 
    api_key: str = Depends(verify_api_key)
):
    """
    Proxy for OpenAI-style chat completions
    """
    try:
        # Flatten messages into a single prompt (simplified)
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in request.messages])
        
        result = await agent.arun(prompt)
        
        # Format response to match OpenAI format
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": os.getenv("LITELLM_MODEL", "custom-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(result.split()),
                "total_tokens": len(prompt.split()) + len(result.split())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🧪 Health Check Endpoint
@app.get("/")
def health_check():
    return {"status": "Running", "model": os.getenv("LITELLM_MODEL", "unknown")}