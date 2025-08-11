from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.app.chat_routes import router as legacy_router
from src.app.fastapi_adapter import router as openai_router

app = FastAPI(title="Lab RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://140.118.115.195:3001"],  
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(legacy_router, prefix="/api")
app.include_router(openai_router)
