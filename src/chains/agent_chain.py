# agent.py

from langchain.agents import initialize_agent, AgentType, Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_experimental.sql.base import SQLDatabaseSequentialChain
from langchain_community.vectorstores import Milvus
from langchain_ollama import OllamaLLM
from sqlalchemy.exc import SQLAlchemyError
import re

from src.config import (
    COLLECTION_NAME, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE,
    OLLAMA_EMBED_MODEL, MILVUS_HOST, MILVUS_PORT,
    MSSQL_SERVER, MSSQL_DATABASE, MSSQL_USER, MSSQL_PASSWORD
)

def get_llm():
    return OllamaLLM(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE,
    )

def build_sql_db():
    uri = (
        f"mssql+pyodbc://{MSSQL_USER}:{MSSQL_PASSWORD}@{MSSQL_SERVER}/{MSSQL_DATABASE}"
        f"?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
    )
    engine = create_engine(uri)
    return SQLDatabase(engine)

def extract_sql(text: str) -> str:
    fence = re.search(r"```sql\s*(.+?)```", text, re.IGNORECASE | re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    else:
        txt = text.strip().strip("`")
        start = re.search(r"(select|with|insert|update|delete|exec)\b.*", txt, re.IGNORECASE | re.DOTALL)
        candidate = start.group(0).strip() if start else txt
    first = candidate.split(";")[0].strip()
    if not first:
        return first
    return first if first.endswith(";") else first + ";"

def build_pdf_vector_engine():
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    )
    return vector_store

def labpapersearch_fn(query: str):
    vector_store = build_pdf_vector_engine()
    docs = vector_store.similarity_search(query, k=3)
    summarized = []
    for i, d in enumerate(docs):
        meta_src = d.metadata.get("source", "paper") if hasattr(d, "metadata") else "paper"
        content = d.page_content if hasattr(d, "page_content") else str(d)
        summarized.append(f"[{i+1}] {meta_src}: {content[:220].replace('\n',' ')}...")
    return "\n".join(summarized) if summarized else "NO_MATCH"

def build_sql_vector_engine():
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name="mssql_data",
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    )
    return vector_store

def mssql_vector_search_fn(query: str):
    vector_store = build_sql_vector_engine()
    docs = vector_store.similarity_search(query, k=3)
    summarized = []
    for i, d in enumerate(docs):
        content = d.page_content if hasattr(d, "page_content") else str(d)
        summarized.append(f"[{i+1}] {content[:200].replace('\n',' ')}...")
    return "\n".join(summarized) if summarized else "NO_MATCH"

def llm_answer_fn(query: str):
    return get_llm()(query)

# 簡單路由：描述型 Northwind 問題不進工具推理，直接回答，避免解析錯誤循環
NORTHWIND_DESC_KEYWORDS = ["northwind", "purpose", "overview", "用途", "概述", "介紹", "是什麼"]

def maybe_direct_answer(question: str):
    q = question.lower()
    if "northwind" in q and any(k in q for k in ["purpose", "overview", "用途", "概述", "介紹", "是什麼"]):
        # 簡短描述，可自行再調整
        desc = (
            "Northwind 是常用的示範/教學交易資料庫，模擬一間進出口/批發公司，包含 Products, Categories, Suppliers, Customers, Employees, Orders, OrderDetails, Shippers, Region 等表，可用於展示查詢、統計、關聯與報表。"
        )
        return f"Final Answer: {desc}"
    return None

def init_agent():
    db = build_sql_db()

    def sql_query_tool(q: str):
        sql = extract_sql(q)
        if not sql:
            return "SQL_ERROR: empty SQL extracted from input. Provide a SELECT ... statement."
        try:
            return db.run(sql)
        except SQLAlchemyError as e:
            return f"SQL_ERROR: {e.__class__.__name__}: {e}"
        except Exception as e:
            return f"SQL_ERROR: {e}"

    def sql_schema_tool(_: str):
        try:
            return db.get_table_info()
        except Exception as e:
            return f"SCHEMA_ERROR: {e}"

    tools = [
        Tool(
            name="LabPaperSearch",
            func=labpapersearch_fn,
            description="Use for questions answerable from the lab's PDF corpus (RAG search). Returns short snippets."
        ),
        Tool(
            name="MSSQLVectorSearch",
            func=mssql_vector_search_fn,
            description="Use for semantic search over MSSQL data in Milvus (returns short snippets)."
        ),
        Tool(
            name="SQLSchema",
            func=sql_schema_tool,
            description="Get current MSSQL table structures before writing a query. Always call if unsure of column names."
        ),
        Tool(
            name="SQLQuery",
            func=sql_query_tool,
            description="Execute ONE MSSQL statement. Input MUST be pure SQL only (no explanation) ending with a semicolon."
        ),
        Tool(
            name="Python_REPL",
            func=PythonREPLTool().run,
            description="Execute Python code."
        ),
        Tool(
            name="LLMAnswer",
            func=llm_answer_fn,
            description="Use for general non-database, non-PDF questions or descriptive Northwind overview if no data retrieval needed."
        ),
    ]

    agent_kwargs = {
        "system_message": (
            "You are a helpful research assistant.\n"
            "If the user asks a general descriptive question about the Northwind database (overview / purpose / what it is) answer DIRECTLY using LLMAnswer WITHOUT calling SQL tools unless explicit data is requested.\n"
            "Tools:\n"
            "- LabPaperSearch: research / paper / dataset / algorithm questions needing PDF snippets.\n"
            "- MSSQLVectorSearch: semantic search over embedded MSSQL textual fragments.\n"
            "- SQLSchema: when unsure about table/column names BEFORE writing SQLQuery.\n"
            "- SQLQuery: ONE clean pure SQL statement (no prose) ending with a semicolon.\n"
            "- Python_REPL: calculations.\n"
            "- LLMAnswer: general reasoning / explanations / Northwind overview.\n"
            "If SQLQuery returns SQL_ERROR, fix once then proceed. Minimize tool calls.\n"
            "Format for reasoning steps (when using tools):\nThought: <reason>\nAction: <tool name>\nAction Input: <input>\nAfter final reasoning output exactly one line starting with 'Final Answer:' followed by the answer only.\n"
        )
    }

    agent = initialize_agent(
        tools=tools,
        llm=get_llm(),
        agent_type=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=10,
        handle_parsing_errors="Final Answer: 抱歉模型格式解析失敗，以下是根據目前資訊的最終回答。",
        verbose=True,
        agent_kwargs=agent_kwargs
    )

    # 包一層 run，先嘗試直接回答描述型問題
    class _Wrapped:
        def run(self, question: str):
            direct = maybe_direct_answer(question)
            if direct:
                return direct
            return agent.run(question)

    return _Wrapped()