from typing import TypedDict, List, Annotated
from langgraph.graph import add_messages
from langchain.schema import Document
from typing import List, Literal, TypedDict

class GraphState(TypedDict):
    source: Literal["VECTORSTORE", "WEBSEARCH"]
    query: str
    llm_generated_response: str
    documents : List[Document]
    relevant: bool
    risk: Literal["FAST", "STRICT"]
    risk_reasons: List[str]
    unsupported_claims: List[str]  # statements the groundedness check could not verify
    final_answer: str  # what the user sees: the answer with its sources, or a decline
    outcome: str  # ANSWERED_FAST, ANSWERED_VERIFIED, DECLINED_UNSUPPORTED, DECLINED_NO_RELEVANT_DOCS