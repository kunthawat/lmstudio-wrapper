from fastapi import Depends, HTTPException
import os

# Load API key from env
API_KEY = os.getenv("API_KEY")

def verify_api_key(api_key: str = None):
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")