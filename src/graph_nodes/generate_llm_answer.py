from typing import Any, Dict
from src.chains.llm_repsonse_generation_chain import response_formatter
from src.graph_state.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    """Generate answer using documents and question."""
    
    print("---GENERATE---")
    
    question = state["query"]
    documents = state["documents"]
    llm_generated_response = response_formatter.invoke({"documents": documents, "query": question})
    print(f"LLM Generated Response: {llm_generated_response}")

    return {
        "llm_generated_response": llm_generated_response
        }