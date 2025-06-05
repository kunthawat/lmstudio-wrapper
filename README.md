# LM Studio / LiteLLM → OpenAI‐Style Wrapper

This FastAPI wrapper can proxy ChatCompletion calls to either LM Studio or LiteLLM.
If LiteLLM is protected by an API key, you can configure the wrapper to send that key automatically.

---

## 1. Prerequisites

- Python 3.9+  
- git  
- LM Studio and/or LiteLLM running locally or on a reachable host.  
- If LiteLLM is secured with a Bearer token, have that key ready.

---

## 2. Setup

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/lmstudio-wrapper.git
   cd lmstudio-wrapper
   ```
2. Create & activate a virtualenv & install:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. Run
    ```bash
    uvicorn wrapper:app --host "$WRAPPER_HOST" --port "$WRAPPER_PORT" --reload
    ```


