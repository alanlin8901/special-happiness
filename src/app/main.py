from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.app.chat_routes import router as legacy_router
from src.app.fastapi_adapter import router as openai_router


app = FastAPI(title="Lab RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(legacy_router, prefix="/api")  # /api/chat 仍可用
app.include_router(openai_router)  # /v1/...
