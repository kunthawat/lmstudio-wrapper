# LangChain + LiteLLM Wrapper

A lightweight LangChain-powered wrapper that uses a remote LiteLLM server.

## Setup

1. Copy `.env.example` to `.env.local` and fill in values.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `uvicorn app.main:app --reload`

## Endpoints

- `POST /run` — Accepts `{ "input": "your prompt here" }`
- Requires header: `x-api-key: your-secret-api-key`