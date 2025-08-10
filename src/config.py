"""Global settings pulled from .env"""
import os
from dotenv import load_dotenv
load_dotenv()

# LLM
# MODEL_PATH  = os.getenv("MODEL_PATH", "models/llm/phi-2.Q4_K_S.gguf")
# EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/e5-small-v2")
# USE_GPU     = os.getenv("USE_GPU", "true").lower() == "true"
# N_CTX       = int(os.getenv("N_CTX", "2024"))

# Milvus
MILVUS_URI = os.getenv("MILVUS_URI", "tcp://standalone:19530")
EMBED_DIM = int(os.getenv("EMBED_DIM", "2560"))
MILVUS_USER    = os.getenv("MILVUS_USER")
MILVUS_PASS    = os.getenv("MILVUS_PASS")

# MSSQL
SQL_SERVER  = os.getenv("MSSQL_SERVER")
SQL_DB      = os.getenv("MSSQL_DATABASE")
SQL_USER    = os.getenv("MSSQL_USER")
SQL_PWD     = os.getenv("MSSQL_PASSWORD")
SQL_CHARSET = os.getenv("MSSQL_CHARSET", "utf8")

# 新增 Ollama 相關配置
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")  # 或其他模型
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
