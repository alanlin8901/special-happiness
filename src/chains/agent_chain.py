# agent.py

from langchain.agents import initialize_agent, AgentType, Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_ollama import OllamaLLM

from src.config import (
    COLLECTION_NAME, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE,
    OLLAMA_EMBED_MODEL, MILVUS_HOST, MILVUS_PORT
)

def get_llm():
    return OllamaLLM(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE,
    )

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

def init_agent():

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
        handle_parsing_errors="Final Answer: Sorry, the model format parsing failed. Here is the final answer based on the current information.",
        verbose=True,
        agent_kwargs=agent_kwargs
    )

    # run
    class _Wrapped:
        def run(self, question: str):
            return agent.run(question)

    return _Wrapped()