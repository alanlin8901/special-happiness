"""
One-shot PDF ingestion → Milvus collection.
Run inside container:  docker compose exec api python scripts/ingest_pdfs.py
"""

import time, socket
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.vector_store.milvus_client import get_vector_store
from src.config import EMBED_MODEL, MILVUS_URI


def wait_for_milvus(host: str = "standalone", port: int = 19530, timeout: int = 120):
    """Retry until Milvus at host:port accepts TCP connections."""
    start = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=5):
                print("✔ Milvus is reachable.")
                return
        except Exception:
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Cannot connect to Milvus at {host}:{port} after {timeout}s"
                )
            print("… waiting for Milvus to come up …", flush=True)
            time.sleep(3)


def main():
    # 1) wait until the standalone service is ready
    wait_for_milvus()

    # 2) set up embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # 3) load and ingest
    docs = SimpleDirectoryReader("data/pdf").load_data()
    vs = get_vector_store()  # uses MILVUS_URI -> tcp://standalone:19530
    VectorStoreIndex.from_documents(docs, vector_store=vs)
    print(f"Ingested {len(docs)} documents into Milvus.")


if __name__ == "__main__":
    main()
