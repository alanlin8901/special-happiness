from src.chains.llm_factory import get_llm
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql.base import SQLDatabaseSequentialChain
from src.config import (
    MODEL_PATH,
    USE_GPU,
    N_CTX,
    SQL_SERVER,
    SQL_DB,
    SQL_USER,
    SQL_PWD,
    SQL_CHARSET,
)

MAX_TOKENS = 1024

llm = get_llm()

uri = (
    f"mssql+pymssql://{SQL_USER}:{SQL_PWD}@{SQL_SERVER}/{SQL_DB}?charset={SQL_CHARSET}"
)
db = SQLDatabase(create_engine(uri))

sql_chain = SQLDatabaseSequentialChain.from_llm(
    llm, db, verbose=True, return_intermediate_steps=True
)
