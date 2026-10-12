from typing import Any, Dict

from src.chains import risk_router
from src.graph_state.state import GraphState


def assess_risk(state: GraphState) -> Dict[str, Any]:
    """Decide how strictly this question's answer must be checked, before anything is retrieved."""
    risk, reasons = risk_router.assess_question_risk(state["query"])
    print(f"---RISK: {risk} ({'; '.join(reasons)})---")
    return {"risk": risk, "risk_reasons": reasons}