from typing import Any, Dict

from src.create_vectorstore.create_vectorstore import get_retriever
from src.graph_state.state import GraphState


def vectorstore(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE FROM VECTORSTORE---")
    documents = get_retriever().invoke(state["query"])
    print(f"Retrieved {len(documents)} chunks.")
    return {"documents": documents}