# app/main.py
import logging



from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import List, Dict, Any
from app.llm_wrapper import agent
from app.api_key_middleware import verify_api_key
import os
import time
import httpx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    verify_api_key(x_api_key, authorization)

    LITELLM_PROXY_URL = os.getenv("LITELLM_PROXY_URL", "https://litellm.moreminimore.com/v1") 
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "missing-key")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LITELLM_API_KEY}"
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{LITELLM_PROXY_URL}/chat/completions",
                json=request.dict(),
                headers=headers,
                timeout=30.0
            )

            upstream_data = resp.json()

            # Ensure content and role are preserved
            cleaned_data = {
                "id": upstream_data.get("id"),
                "object": upstream_data.get("object"),
                "created": upstream_data.get("created"),
                "model": upstream_data.get("model"),
                "choices": [
                    {
                        "index": choice.get("index"),
                        "message": {
                            "role": choice["message"].get("role"),
                            "content": choice["message"].get("content")
                        },
                        "finish_reason": choice.get("finish_reason"),
                    }
                    for choice in upstream_data.get("choices", [])
                ],
                "usage": upstream_data.get("usage", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }),
            }

            return cleaned_data

        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")
        except KeyError as e:
            raise HTTPException(status_code=502, detail=f"Missing expected field in LiteLLM response: {str(e)}")
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

@app.get("/debug")
async def debug_env():
    return {
        "API_KEY": os.getenv("API_KEY"),
        "LITELLM_PROXY_URL": os.getenv("LITELLM_PROXY_URL"),
        "LITELLM_API_KEY": os.getenv("LITELLM_API_KEY")
    }