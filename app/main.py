# app/main.py

from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import List, Dict, Any
from app.llm_wrapper import agent
from app.api_key_middleware import verify_api_key
import os
import time
import httpx

app = FastAPI()

# --- OpenAI-Compatible Model List Endpoint ---
@app.get("/v1/models")
async def get_models(
    x_api_key: str = Header(None),
    authorization: str = Header(None)
):
    """
    Proxy for LiteLLM's /models endpoint.
    Passes the Bearer token to LiteLLM.
    """
    # Validate wrapper API key first
    verify_api_key(x_api_key, authorization)

    LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL", "https://litellm.moreminimore.com/v1")

    async with httpx.AsyncClient() as client:
        try:
            # Pass the LiteLLM API key in headers
            headers = {
                "Authorization": f"Bearer {os.getenv('LITELLM_API_KEY', 'missing-key')}"
            }
            resp = await client.get(f"{LITELLM_PROXY_URL}/models", headers=headers)
            data = resp.json()
            
            # Optional: Sanitize model names
            filtered_models = [
                {**model, "id": model["id"].replace("ollama_chat/", "")} 
                for model in data.get("data", [])
            ]
            return {"data": filtered_models}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model fetch failed: {str(e)}")

# --- OpenAI-Compatible Chat Completion Endpoint ---
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 2048

@app.post("/v1/chat/completions")
async def openai_compatible_chat(
    request: ChatCompletionRequest,
    x_api_key: str = Header(None),
    authorization: str = Header(None)
):
    """
    Proxy for OpenAI-style chat completions.
    """
    verify_api_key(x_api_key, authorization)

    try:
        LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL", "http://litellm.moreminimore.com/v1")
        LITELLM_MODEL = os.getenv("LITELLM_MODEL", "ollama_chat/qwen3-30b-a3b")
        LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "missing-key")

        headers = {}
        if LITELLM_API_KEY:
            headers["Authorization"] = f"Bearer {LITELLM_API_KEY}"

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{LITELLM_PROXY_URL}/chat/completions",
                json={"model": LITELLM_MODEL, "messages": request.messages},
                headers=headers
            )
            return resp.json()  # Forward LiteLLM's response directly
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Health Check ---
@app.get("/")
def health_check():
    return {
        "status": "Running",
        "model": os.getenv("LITELLM_MODEL", "unknown"),
        "proxy_url": os.getenv("LITELLM_PROXY_URL", "unknown")
    }