from typing import Any, Dict, List

from src.chains import risk_router
from src.chains.formatting import source_label
from src.graph_state.state import GraphState


def assess_risk(state: GraphState) -> Dict[str, Any]:
    """Decide how strictly this question's answer must be checked, before anything is retrieved."""
    risk, reasons = risk_router.assess_question_risk(state["query"])
    print(f"---RISK: {risk} ({'; '.join(reasons)})---")
    return {"risk": risk, "risk_reasons": reasons}


def check_answer_risk(state: GraphState) -> Dict[str, Any]:
    """Route on what was produced, not only on what was asked: risky answers escalate to STRICT."""
    if state.get("risk") == "STRICT":
        return {}
    reasons = risk_router.assess_answer_risk(state.get("llm_generated_response", ""))
    if not reasons:
        return {}
    print(f"---ESCALATED TO STRICT ({'; '.join(reasons)})---")
    return {"risk": "STRICT", "risk_reasons": state.get("risk_reasons", []) + reasons}


def _sources(documents) -> List[str]:
    """Unique source labels, in the order the documents were used."""
    labels: List[str] = []
    for document in documents or []:
        label = source_label(document)
        if label not in labels:
            labels.append(label)
    return labels


def finalize(state: GraphState) -> Dict[str, Any]:
    """Deliver the answer with its sources, and record how it was delivered."""
    outcome = "ANSWERED_FAST" if state.get("risk") == "FAST" else "ANSWERED_UNVERIFIED"
    answer = state.get("llm_generated_response", "")
    sources = _sources(state.get("documents"))
    if sources:
        answer += "\n\nSources: " + "; ".join(sources)
    print(f"---{outcome}---")
    return {"final_answer": answer, "outcome": outcome}
