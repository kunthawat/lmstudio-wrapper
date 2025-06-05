from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from app.llm_wrapper import agent
from app.api_key_middleware import verify_api_key  # Ensure this import is present
import os

app = FastAPI()

# Ensure this matches n8n's expectations
API_KEY = os.getenv("API_KEY", "fallback-key")

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 8096
    # Add other fields as needed

@app.post("/v1/chat/completions")
async def openai_compatible_chat(request: ChatCompletionRequest, api_key: str = Depends(verify_api_key)):
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