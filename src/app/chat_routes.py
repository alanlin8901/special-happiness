from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
from src.chains.agent_chain import init_agent

router = APIRouter()
agent = init_agent()

class ChatRequest(BaseModel):
    model: str
    messages: list

@router.post("/chat")
async def chat(req: ChatRequest):
    user_message = next(
        (m["content"] for m in reversed(req.messages) if m["role"] == "user"),
        ""
    )
    answer = agent.run(user_message)
    return {
        "model": req.model,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "message": {
            "role": "assistant",
            "content": answer
        },
        "done": True
    }
