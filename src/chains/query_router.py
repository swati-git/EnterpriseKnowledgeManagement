from src.models.model import llm_model
from pydantic import BaseModel
from typing import Literal
from langchain.prompts import ChatPromptTemplate


class QueryRouter(BaseModel):

    source: Literal["VECTORESTORE", "WEBSEARCH"]

system = """You are an expert at routing a user question to a vectorstore or web search.
Given the user question, decide whether to answer from VECTORESTORE or WEBSEARCH
Respond with a JSON object with a single field 'source' whose value is either 'VECTORESTORE' or 'WEBSEARCH'
The vectorstore contains documents related to Investment options.
Use the vectorstore for questions on these topics. For all else, use web-search."""

prompt = ChatPromptTemplate.from_messages(
    [
    ("system", system),
    ("user", "{query}")
     ]
     )

query_router = prompt | llm_model.with_structured_output(QueryRouter)

print("---QUERY ROUTER CHAIN INITIALIZED---")