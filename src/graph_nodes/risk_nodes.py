from typing import Any, Dict

from src.chains import risk_router
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
