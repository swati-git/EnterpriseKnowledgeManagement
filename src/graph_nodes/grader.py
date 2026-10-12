from typing import Any, Dict
from src.graph_state.state import GraphState
from src.chains.formatting import format_documents, source_label
from src.chains.retrieval_grader import RetrievalGraderOutput, retrieval_grader


def grader(graph_state: GraphState) -> Dict[str, Any]:
    documents = graph_state.get("documents", [])
    grades = retrieval_grader.batch(
        [{"query": graph_state["query"], "document": format_documents([doc])} for doc in documents],
        return_exceptions=True,
    )
    kept = []
    for document, grade in zip(documents, grades):
        is_relevant = isinstance(grade, RetrievalGraderOutput) and grade.relevant
        print(f"  {'keep' if is_relevant else 'drop'}: {source_label(document)}")
        if is_relevant:
            kept.append(document)
    return {"documents": kept, "relevant": bool(kept)}