"""
Query engine over lab PDFs (Milvus + LlamaIndex)
"""

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from langchain_community.llms.llamacpp import LlamaCpp
from src.config import MODEL_PATH, EMBED_MODEL, MILVUS_URI, USE_GPU, N_CTX

MAX_TOKENS = 1024

def build_query_engine():
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.embed_model = embed_model

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        temperature=0.0,
        n_gpu_layers=-1,
        max_tokens=MAX_TOKENS
    )
    
    Settings.llm = llm

    vs = MilvusVectorStore(
        uri=MILVUS_URI,
        collection_name="lab_papers",
        dim=2560,
        embedding_field="embedding"
    )

    index = VectorStoreIndex.from_vector_store(vs)
    return index.as_query_engine(similarity_top_k=5)

