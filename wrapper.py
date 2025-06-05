# wrapper.py

import os
import json
from typing import Optional, List, Dict, Any, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

# -----------------------------------------------------------------------------
# 1) Load environment variables from .env (if present)
# -----------------------------------------------------------------------------
load_dotenv()

# Which backend to use: "lmstudio" or "litellm"
BACKEND: str = os.getenv("BACKEND", "lmstudio").lower()

# LM Studio chat endpoint (OpenAI‐compatible)
LMSTUDIO_URL: str = os.getenv(
    "LMSTUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions"
)

# LiteLLM chat endpoint (OpenAI‐compatible)
LITELLM_URL: str = os.getenv(
    "LITELLM_URL", "http://127.0.0.1:8080/v1/chat/completions"
)

# (Optional) LiteLLM API key, if you have configured LiteLLM to require one
LITELLM_API_KEY: Optional[str] = os.getenv("LITELLM_API_KEY", None)

WRAPPER_HOST: str = os.getenv("WRAPPER_HOST", "0.0.0.0")
WRAPPER_PORT: int = int(os.getenv("WRAPPER_PORT", "8000"))
DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

# -----------------------------------------------------------------------------
# 2) Define Pydantic models matching OpenAI's ChatCompletion schema
# -----------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None  # for function‐call messages

class FunctionSchema(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[FunctionSchema]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

# -----------------------------------------------------------------------------
# 3) Create FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="LM Studio / LiteLLM Wrapper", debug=DEBUG)

def get_backend_url() -> str:
    """
    Return the correct chat-completions URL based on BACKEND.
    """
    if BACKEND == "litellm":
        return LITELLM_URL
    return LMSTUDIO_URL

def get_auth_headers() -> Dict[str, str]\:
    """
    If we're using LiteLLM and an API key is set, return the Authorization header.
    Otherwise, return an empty dict.
    """
    if BACKEND == "litellm" and LITELLM_API_KEY:
        return {"Authorization": f"Bearer {LITELLM_API_KEY}"}
    return {}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    Receive an OpenAI‐compatible ChatCompletionRequest,
    forward it to LM Studio or LiteLLM (based on BACKEND),
    and return the raw JSON response.
    """
    target_url = get_backend_url()
    auth_headers = get_auth_headers()

    # Build payload for downstream LLM
    payload: Dict[str, Any] = {
        "model": req.model,
        "messages": [m.dict(exclude_none=True) for m in req.messages],
    }
    if req.functions is not None:
        payload["functions"] = [f.dict(exclude_none=True) for f in req.functions]
    if req.function_call is not None:
        payload["function_call"] = req.function_call
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens

    # Forward to chosen backend and catch errors
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            lm_resp = await client.post(
                target_url,
                json=payload,
                headers={**auth_headers, "Content-Type": "application/json"},
            )
            lm_resp.raise_for_status()
            downstream_json = lm_resp.json()
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error connecting to backend ({target_url}): {str(e)}",
        )
    except httpx.HTTPStatusError as e:
        content = e.response.text
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Backend ({target_url}) returned {e.response.status_code}: {content}",
        )

    # Return the backend’s JSON unchanged
    return downstream_json

# -----------------------------------------------------------------------------
# 4) Entrypoint for uvicorn
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "wrapper:app",
        host=WRAPPER_HOST,
        port=WRAPPER_PORT,
        reload=DEBUG,
    )
