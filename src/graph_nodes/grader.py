from typing import Any, Dict
from src.graph_state.state import GraphState
from src.chains.formatting import format_documents, source_label
from src.chains.retrieval_grader import RetrievalGraderOutput, retrieval_grader


def grader(graph_state: GraphState) -> Dict[str, Any]:
    documents = graph_state.get("documents", [])
    grade = retrieval_grader.invoke(
         {
             "query": graph_state["query"], 
             "documents": format_documents(documents)
         }
        )
    print("---GRADER RECEIVED THE FOLLOWING DOCUMENTS---")
    list_of_documents = graph_state["documents"]
    print(f"Grader graded the documents as : {grade}")
    return {"relevant": bool(grade and grade.relevant)}