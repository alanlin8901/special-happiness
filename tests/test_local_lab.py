from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import OllamaEmbeddings

# --- 1. 設定參數 (在本機執行) ---
# 連接到您對外映射的 Port
MILVUS_HOST = "localhost"
MILVUS_PORT = "19531" 
OLLAMA_BASE_URL = "http://localhost:11434"

# 您用來 ingest 的 collection 和 embedding 模型
COLLECTION_NAME = "pdf_collection"
OLLAMA_EMBED_MODEL = "nomic-embed-text" 

# --- 2. 初始化連線 ---
# 初始化 embedding 模型
embeddings = OllamaEmbeddings(
    model=OLLAMA_EMBED_MODEL,
    base_url=OLLAMA_BASE_URL
)

# 連接到 Milvus 中已存在的 collection
vector_store = Milvus(
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
)

# --- 3. 執行搜尋 ---
# 提出一個您認為 PDF 中應該能回答的問題
query = "What is the main contribution of the paper about attention mechanisms?"

print(f"\n正在用問題進行向量搜尋:\n'{query}'")

# 執行相似度搜尋，找出最相關的 4 個文件區塊
docs = vector_store.similarity_search(query, k=4)

print("\n--- 搜尋結果 ---")
if not docs:
    print("沒有找到相關的文件區塊。")
else:
    for i, doc in enumerate(docs):
        print(f"\n--- 結果 {i+1} ---")
        print("內容片段:", doc.page_content[:300] + "...")
        print("Metadata (來源):", doc.metadata)