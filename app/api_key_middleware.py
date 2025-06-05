# app/api_key_middleware.py
from fastapi import Depends, HTTPException, Header
import os

API_KEY = os.getenv("API_KEY")

def verify_api_key(
    x_api_key: str = Header(None),
    authorization: str = Header(None)
):
    if not API_KEY:
        return  # Skip if no key is set

    if x_api_key == API_KEY:
        return

    if authorization and authorization.startswith(f"Bearer {API_KEY}"):
        return

    raise HTTPException(status_code=403, detail="Invalid API Key")