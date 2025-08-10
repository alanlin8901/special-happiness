"""
Thin helper to return a MilvusVectorStore properly configured
"""

from llama_index.vector_stores.milvus import MilvusVectorStore
from src.config import MILVUS_URI, EMBED_DIM


def get_vector_store(
    collection_name: str = "lab_papers",
    text_field: str = "text",
    embedding_field: str = "embedding",
    dim: int = EMBED_DIM,
) -> MilvusVectorStore:
    return MilvusVectorStore(
        uri=MILVUS_URI,
        collection_name=collection_name,
        text_field=text_field,
        embedding_field=embedding_field,
        dim=dim,
    )
