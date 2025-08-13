from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Literal, Optional, Generator
from datetime import datetime
import json
import time
from hashlib import sha256
from threading import Lock

from src.config import OLLAMA_MODEL 
from src.chains.agent_chain import init_agent, get_llm

router = APIRouter()

AVAILABLE_MODELS = [
    {"id": OLLAMA_MODEL, "object": "model"}
]

agent = init_agent()

# --- dedup cache (短期快取，避免同一請求短時間重算) ---
_CACHE_TTL_SEC = 15
_CACHE: dict[str, dict] = {}
_CACHE_LOCK = Lock()

def _req_key(model: str, messages: List["Message"], raw: bool) -> str:
    payload = {
        "model": model,
        "raw": raw,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }
    s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256(s.encode("utf-8")).hexdigest()

def _cache_get(key: str) -> Optional[str]:
    now = time.time()
    with _CACHE_LOCK:
        item = _CACHE.get(key)
        if item and (now - item["ts"] <= _CACHE_TTL_SEC):
            return item["answer"]
    return None

def _cache_set(key: str, answer: str) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = {"ts": time.time(), "answer": answer}

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
    # 構建去重 key
    key = _req_key(req.model, req.messages, raw)
    cached = _cache_get(key)

    # 非串流：直接用快取或計算後回傳
    if not req.stream:
        if cached is None:
            user_message = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
            answer = _select_model_answer(req.model, user_message, raw)
            _cache_set(key, answer)
        else:
            answer = cached
        return _build_completion(answer, req.model)

    # 串流：先快速送出一個空增量，再計算，最後送出完整內容
    def stream_gen() -> Generator[bytes, None, None]:
        created = int(datetime.utcnow().timestamp())

        # 先送一個空增量，避免前端把連線當成無回應而重試
        init_chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(init_chunk, ensure_ascii=False)}\n\n".encode("utf-8")

        # 計算答案（可用快取）
        ans = _cache_get(key)
        if ans is None:
            user_message = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
            ans = _select_model_answer(req.model, user_message, raw)
            _cache_set(key, ans)

        # 實際內容
        delta = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ans},
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