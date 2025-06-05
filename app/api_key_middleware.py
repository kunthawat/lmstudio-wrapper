from fastapi import Depends, HTTPException
import os

# Load API key from environment
API_KEY = os.getenv("API_KEY")

def verify_api_key(authorization: str = None):
    if not API_KEY:
        return  # No API key set, skip validation
    
    if authorization and authorization.startswith(f"Bearer {API_KEY}"):
        return  # Accept Bearer token format (used by n8n)

    raise HTTPException(status_code=403, detail="Invalid API Key")