# src/llm_factory.py
from langchain_community.llms.llamacpp import LlamaCpp
from src.config import MODEL_PATH, N_CTX

MAX_TOKENS = 1024

def get_llm():
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        temperature=0.0,
        n_gpu_layers=-1,
        max_tokens=MAX_TOKENS
    )