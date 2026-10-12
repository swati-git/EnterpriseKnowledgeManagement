from typing import TypedDict, List, Annotated
from langgraph.graph import add_messages
from langchain.schema import Document

class GraphState(TypedDict):
    query: str
    llm_generated_response: str
    documents : List[Document]
    relevant: bool