from langchain_core.vectorstores.base import VectorStoreRetriever

from src.graph_state.state import GraphState
from typing import Any, Dict
from  src.create_vectorstore.create_vectorstore import initialize_retriever

def vectorstore(state: GraphState) -> Dict[str, Any]:
    retriever: VectorStoreRetriever = initialize_retriever()
    retrieved_docs = retriever.invoke(state["query"])
    for document in retrieved_docs:
      print(f"Vectorstore retrieved document: {document.page_content[:200]}...")  
    return {"documents": retrieved_docs}   
