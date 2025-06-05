# app/api_key_middleware.py
from fastapi import Depends, HTTPException
import os

API_KEY = os.getenv("API_KEY")

def verify_api_key(
    x_api_key: str = None,
    authorization: str = None
):
    if not API_KEY:
        return  # Skip if no key set

    # Accept either header
    if x_api_key == API_KEY:
        return
    
    if authorization and authorization.startswith(f"Bearer {API_KEY}"):
        return

    raise HTTPException(status_code=403, detail="Invalid API Key")