from src.chains.llm_factory import get_llm
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.prompts import PromptTemplate
from src.chains.pdf_chat import build_query_engine
from src.chains.sql_chat import sql_chain
from src.config import MODEL_PATH, USE_GPU, N_CTX

query_engine = build_query_engine()
llm = get_llm()

template = """
You are an intelligent assistant. When using a tool, respond with exactly two lines:

Action: <tool name>
Action Input: <input string>

Use plain text only, no types, no variable names, no code syntax.
For SQL queries, provide the raw SQL statement.

Question: {input}
"""

def labpapersearch_fn(q: str) -> str:
    try:
        result = query_engine.query(q)
        return str(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"[LabPaperSearch 錯誤] {e!r}"

prompt = PromptTemplate(template=template, input_variables=["input"])

agent = initialize_agent(
    [
        Tool(
            "LabPaperSearch",
            labpapersearch_fn,
            "Use for questions answerable from the lab's PDF corpus.",
        ),
        Tool(
            "SQLQuery",
            sql_chain.run,
            "Use for questions about the MSSQL 'course' database.",
        ),
        PythonREPLTool(),
    ],
    llm=llm,
    agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    max_iterations=30,
    verbose=True,
    prompt=prompt
)