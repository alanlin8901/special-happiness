from fastapi import APIRouter
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool  # 非同步跑同步函式用
from src.chains.agent_chain import agent

router = APIRouter()

class ChatRequest(BaseModel):
    prompt: str

@router.post("/chat")
async def chat(req: ChatRequest):
    answer = await run_in_threadpool(agent.run, req.prompt)
    return {"answer": answer}
