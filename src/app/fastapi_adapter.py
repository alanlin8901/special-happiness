from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Literal, Optional, Generator
from datetime import datetime
import json

from src.config import OLLAMA_MODEL   # 新增
from src.chains.agent_chain import init_agent, get_llm  # 若 get_llm 尚未 export 就補 export

router = APIRouter()

# 唯一模型 (id 與前端顯示一致)
AVAILABLE_MODELS = [
    {"id": OLLAMA_MODEL, "object": "model"}
]

agent = init_agent()

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = True

def _build_completion(answer: str, model: str):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": int(datetime.utcnow().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

def _select_model_answer(model: str, prompt: str, raw: bool) -> str:
    if model != OLLAMA_MODEL:
        return f"Unknown model: {model}"
    if raw:
        return get_llm()(prompt)
    return agent.run(prompt)

@router.post("/v1/chat/completions")
async def chat(req: ChatRequest, raw: bool = Query(False)):
    user_message = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    answer = _select_model_answer(req.model, user_message, raw)

    if not req.stream:
        return _build_completion(answer, req.model)

    def stream_gen() -> Generator[bytes, None, None]:
        created = int(datetime.utcnow().timestamp())
        delta = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": answer},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(delta, ensure_ascii=False)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(stream_gen(), media_type="text/event-stream")

@router.get("/v1/models")
def list_models():
    return {"object": "list", "data": AVAILABLE_MODELS}