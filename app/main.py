from fastapi import FastAPI, Depends, HTTPException
from app.llm_wrapper import agent
from app.api_key_middleware import verify_api_key
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.post("/run")
async def run_agent(input_data: dict, api_key: str = Depends(verify_api_key)):
    user_input = input_data.get("input", "")
    result = await agent.arun(user_input)
    return {"output": result}
