from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Literal, Optional, Generator
from datetime import datetime
import json
from src.chains.agent_chain import init_agent

router = APIRouter()
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
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": int(datetime.utcnow().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

@router.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    user_message = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    answer = agent.run(user_message)

    if not req.stream:
        return _build_completion(answer, req.model)

    def stream_gen() -> Generator[bytes, None, None]:
        created = int(datetime.utcnow().timestamp())
        model = req.model
        for idx, chunk in enumerate([answer]):
            delta = {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant" if idx == 0 else None,
                            "content": chunk
                        },
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(delta, ensure_ascii=False)}\n\n".encode("utf-8")

        yield b"data: [DONE]\n\n"

    return StreamingResponse(stream_gen(), media_type="text/event-stream")

@router.get("/v1/models")
async def get_models():
    return {
        "data": [
            {"id": "ollama-model-name", "object": "model", "owned_by": "you"},
            {"id": "ss", "object": "model", "owned_by": "you"}
        ]
    }