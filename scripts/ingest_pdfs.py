import os
import re
import glob
import concurrent.futures
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema.document import Document

from src.config import (
    OLLAMA_BASE_URL, PDF_DIRECTORY_PATH, MILVUS_HOST,
    MILVUS_PORT, COLLECTION_NAME, OLLAMA_EMBED_MODEL,
)

# --- 1. 輔助函式 ---

def sanitize_metadata_keys(doc: Document) -> Document:
    if not doc.metadata:
        return doc
    new_meta = {}
    for k, v in doc.metadata.items():
        new_key = re.sub(r'[^a-zA-Z0-9_]', '_', k)
        new_meta[new_key] = v
    return Document(page_content=doc.page_content, metadata=new_meta)

def unify_metadata_keys(docs: List[Document]) -> List[Document]:
    all_keys = set()
    for doc in docs:
        if doc.metadata:
            all_keys.update(doc.metadata.keys())
    unified_docs = []
    for doc in docs:
        new_meta = {key: doc.metadata.get(key, "") for key in all_keys} if doc.metadata else {k: "" for k in all_keys}
        unified_docs.append(Document(page_content=doc.page_content, metadata=new_meta))
    return unified_docs

# --- 2. 平行處理 PDF 檔案切割 ---
def process_pdf(pdf_file: str) -> List[Document]:
    print(f" -> 正在處理檔案: {pdf_file}")
    try:
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        print(f"    -> 已切割成 {len(split_docs)} 個文件區塊。")
        return split_docs
    except Exception as e:
        print(f"    ❌ [錯誤] 處理檔案 {pdf_file} 時失敗: {e}")
        return []

def load_and_split_pdfs_parallel(directory_path: str) -> List[Document]:
    print(f"[偵錯] 正在搜尋路徑: {os.path.abspath(directory_path)}")
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    
    if not pdf_files:
        print(f"❌ [嚴重] 在 '{directory_path}' 中沒有找到任何 PDF 檔案。請檢查路徑和檔案權限。")
        return []

    print(f"✅ [成功] 找到 {len(pdf_files)} 個 PDF 檔案。開始平行處理...")

    all_docs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_pdf, pdf_files)
    for split_docs in results:
        all_docs.extend(split_docs)
    return all_docs

# --- 3. 主要執行流程 ---
if __name__ == "__main__":
    docs_from_all_pdfs = load_and_split_pdfs_parallel(PDF_DIRECTORY_PATH)

    if not docs_from_all_pdfs:
        print("\n[結束] 因為沒有讀取到任何文件區塊，程式已終止。")
        exit(1)
    else:
        print(f"\n✅ [成功] 所有 PDF 已成功切割成 {len(docs_from_all_pdfs)} 個文件區塊。")

    docs_sanitized = [sanitize_metadata_keys(doc) for doc in docs_from_all_pdfs]
    docs_unified = unify_metadata_keys(docs_sanitized)

    print("\n[偵錯] 準備寫入 Milvus 的第一個文件預覽：")
    print("  - Page Content (前100字):", docs_unified[0].page_content[:100] + "...")
    print("  - Metadata:", docs_unified[0].metadata)

    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    print(f"\n✅ [成功] 已連接到 Ollama Embedding 模型: {OLLAMA_EMBED_MODEL}")

    BATCH_SIZE = 500
    print(f"\n[開始] 準備將 {len(docs_unified)} 個文件以批次寫入 Milvus collection: {COLLECTION_NAME} ...")
    try:
        for i in range(0, len(docs_unified), BATCH_SIZE):
            batch = docs_unified[i:i+BATCH_SIZE]
            Milvus.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
                drop_old=(i == 0)  # 只第一次刪除重建 collection
            )
            print(f"  ✅ 已寫入第 {i} ~ {i+len(batch)-1} 筆文件。")
        print("✅ [成功] LangChain 'from_documents' 批次寫入 Milvus 完成！")
    except Exception as e:
        print(f"❌ [嚴重錯誤] 在寫入 Milvus 時發生錯誤: {e}")
        import traceback
        traceback.print_exc()

    print("\n[結束] Ingest 腳本執行完畢。")
