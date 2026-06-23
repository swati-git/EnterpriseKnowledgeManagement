from typing import TypedDict, List, Annotated
from langgraph.graph import add_messages
from langchain.schema import Document

class GraphState(TypedDict):
    query: str
    ll_generated_response: Annotated[list, add_messages]
    documents : List[Document]
    grade: bool