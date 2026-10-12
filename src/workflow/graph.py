from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from src.chains.query_router import QueryRouter, query_router
from src.graph_nodes.generate_llm_answer import generate, no_relevant_documents
from src.graph_nodes.grader import grader
from src.graph_nodes.vectorstore import vectorstore
from src.graph_nodes.web_search import web_search
from src.graph_state.state import GraphState


def route_query(state: GraphState) -> Dict[str, Any]:
    """A node, not just an edge function, so the routing decision is kept in the state."""
    print("---ROUTE QUESTION---")
    try:
        decision = query_router.invoke({"query": state["query"]})
        source = decision.source if isinstance(decision, QueryRouter) else "VECTORSTORE"
    except Exception as exc:  # the router is a model call; don't let it crash the request
        print(f"Router failed ({type(exc).__name__}); defaulting to VECTORSTORE")
        source = "VECTORSTORE"
    print(f"Routed to: {source}")
    return {"source": source}


def after_routing(state: GraphState) -> str:
    return "WEBSEARCH" if state.get("source") == "WEBSEARCH" else "VECTORSTORE"


def after_grading(state: GraphState) -> str:
    """The grader's verdict now decides what happens next."""
    return "GENERATE_ANSWER" if state.get("relevant") else "NO_RELEVANT_DOCUMENTS"


workflow = StateGraph(GraphState)

workflow.add_node("ROUTE", route_query)
workflow.add_node("WEBSEARCH", web_search)
workflow.add_node("VECTORSTORE", vectorstore)
workflow.add_node("GRADER", grader)
workflow.add_node("GENERATE_ANSWER", generate)
workflow.add_node("NO_RELEVANT_DOCUMENTS", no_relevant_documents)

workflow.add_edge(START, "ROUTE")
workflow.add_conditional_edges("ROUTE", after_routing, {"WEBSEARCH": "WEBSEARCH", "VECTORSTORE": "VECTORSTORE"})
workflow.add_edge("WEBSEARCH", "GRADER")
workflow.add_edge("VECTORSTORE", "GRADER")
workflow.add_conditional_edges(
    "GRADER",
    after_grading,
    {"GENERATE_ANSWER": "GENERATE_ANSWER", "NO_RELEVANT_DOCUMENTS": "NO_RELEVANT_DOCUMENTS"},
)
workflow.add_edge("GENERATE_ANSWER", END)
workflow.add_edge("NO_RELEVANT_DOCUMENTS", END)

application = workflow.compile()