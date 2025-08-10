from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, List
from starlette.concurrency import run_in_threadpool
from src.chains.agent_chain import agent

router = APIRouter()

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    model: str = Field(..., description="ignored")
    messages: List[Message]

@router.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    try:
        user_msg = next(m.content for m in reversed(req.messages) if m.role == "user")
    except StopIteration:
        raise HTTPException(400, "No user message found")
    try:
        answer = await run_in_threadpool(agent.run, user_msg)
    except Exception as e:
        raise HTTPException(500, f"Agent error: {e!r}")
    return {
        "id": "local-chatcmpl",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
    }