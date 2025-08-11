import pyodbc
from langchain.schema.document import Document
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import OllamaEmbeddings
from src.config import (
    MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL,
    MSSQL_SERVER, MSSQL_DATABASE, MSSQL_USER, MSSQL_PASSWORD
)

# MSSQL 連線字串
conn_str = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={MSSQL_SERVER};"
    f"DATABASE={MSSQL_DATABASE};"
    f"UID={MSSQL_USER};"
    f"PWD={MSSQL_PASSWORD};"
    f"TrustServerCertificate=yes;"
)
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# 查詢 Products 表，這裡取一些重要欄位
cursor.execute("""
    SELECT ProductID, ProductName, SupplierID, CategoryID, QuantityPerUnit, UnitPrice, UnitsInStock
    FROM Products
""")

docs = []
for row in cursor.fetchall():
    content = (
        f"Product Name: {row.ProductName}\n"
        f"Supplier ID: {row.SupplierID}\n"
        f"Category ID: {row.CategoryID}\n"
        f"Quantity Per Unit: {row.QuantityPerUnit}\n"
        f"Unit Price: {row.UnitPrice}\n"
        f"Units In Stock: {row.UnitsInStock}"
    )
    metadata = {"ProductID": row.ProductID}
    docs.append(Document(page_content=content, metadata=metadata))

conn.close()

# 建立 Embeddings
embeddings = OllamaEmbeddings(
    model=OLLAMA_EMBED_MODEL,
    base_url=OLLAMA_BASE_URL
)

# 將文件寫入 Milvus，並使用 drop_old=True 重建 collection schema
vector_store = Milvus.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    drop_old=True,
)

print(f"✅ 已將 {len(docs)} 筆 Products 資料寫入 Milvus")
