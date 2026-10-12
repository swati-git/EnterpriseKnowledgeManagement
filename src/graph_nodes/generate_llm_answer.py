from typing import Any, Dict
from src.chains.llm_repsonse_generation_chain import response_formatter
from src.graph_state.state import GraphState
from src.chains.formatting import format_documents


NO_RELEVANT_DOCUMENTS = (
    "I couldn't find information relevant to this question, so I'm not going to guess. "
    "Try rephrasing it, or check with the team that owns the topic."
    )

def no_relevant_documents(state: GraphState) -> Dict[str, Any]:
        print("---NO RELEVANT DOCUMENTS---")
        return {"llm_generated_response": NO_RELEVANT_DOCUMENTS}


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    answer = response_formatter.invoke(
        {"query": state["query"], "documents": format_documents(state.get("documents", []))}
    )
    return {"llm_generated_response": answer}