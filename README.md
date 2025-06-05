# LM Studio / LiteLLM → OpenAI‐Style Wrapper

This repository provides a minimal FastAPI wrapper that accepts OpenAI‐compatible
ChatCompletion requests (including `functions` / `function_call`) and forwards them
to either LM Studio or LiteLLM, depending on an environment variable. The downstream
response is passed back unchanged so that n8n’s AI Agent (Tools Agent) or any other
LangChain-based client can detect function calls just as if you were talking to OpenAI.

---

## Repository Layout

