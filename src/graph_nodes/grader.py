from typing import Any, Dict
from src.graph_state.state import GraphState
from src.chains.retrieval_grader import retrieval_grader

def grader(graph_state: GraphState) -> Dict[str, Any]:
    grade = retrieval_grader.invoke(
         {
             "query": graph_state["query"], 
             "documents": graph_state["documents"]
         }
        )
    print("---GRADER RECEIVED THE FOLLOWING DOCUMENTS---")
    list_of_documents = graph_state["documents"]
    for document in list_of_documents:
      print(f"Document : {document}")  
    print(f"Grader graded the documents as : {grade}")
    return {"relevant": grade}