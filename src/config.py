"""Global settings pulled from .env"""
import os
from dotenv import load_dotenv
load_dotenv()

# Ollama Provider
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-alan:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Milvus
MILVUS_URI = os.getenv("MILVUS_URI", "tcp://milvus-standalone-alan:19530")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
MILVUS_USER    = os.getenv("MILVUS_USER", "root")
MILVUS_PASS    = os.getenv("MILVUS_PASS")
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus-standalone-alan")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# MSSQL
MSSQL_SERVER  = os.getenv("MSSQL_SERVER", "140.118.115.196")
MSSQL_DATABASE = os.getenv("MSSQL_DATABASE", "Northwind")
MSSQL_USER    = os.getenv("MSSQL_USER", "llm")
MSSQL_PASSWORD = os.getenv("MSSQL_PASSWORD", "1qaz2WSX")
MSSQL_CHARSET = os.getenv("MSSQL_CHARSET", "utf8")

# PDF 
PDF_DIRECTORY_PATH = os.getenv("PDF_DIRECTORY_PATH", "data/pdf")
# Collection 
PDF_COLLECTION_NAME = os.getenv("PDF_COLLECTION_NAME", "pdf_collection")
SQL_COLLECTION_NAME = os.getenv("SQL_COLLECTION_NAME", "sql_collection")