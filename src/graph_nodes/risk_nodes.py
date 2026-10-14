from typing import Any, Dict, List

from src.chains import risk_router
from src.chains.formatting import format_documents, source_label
from src.chains.groundedness_checker import GroundednessOutput, groundedness_checker
from src.graph_state.state import GraphState


def assess_risk(state: GraphState) -> Dict[str, Any]:
    """Decide how strictly this question's answer must be checked, before anything is retrieved."""
    risk, reasons = risk_router.assess_question_risk(state["query"])
    print(f"---RISK: {risk} ({'; '.join(reasons)})---")
    update: Dict[str, Any] = {"risk": risk, "risk_reasons": reasons}
    failed = [r for r in reasons if r.startswith("classifier failed")]
    if failed:
        update["errors"] = [f"ASSESS_RISK: {r}" for r in failed]
    return update


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
    # Strict answers only reach this node after passing the groundedness check.
    # Without risk routing (the baseline), nothing is checked, and the outcome says so.
    risk = state.get("risk")
    outcome = {"FAST": "ANSWERED_FAST", "STRICT": "ANSWERED_VERIFIED"}.get(risk, "ANSWERED_UNCHECKED")
    answer = state.get("llm_generated_response", "")
    sources = _sources(state.get("documents"))
    if sources:
        answer += "\n\nSources: " + "; ".join(sources)
    print(f"---{outcome}---")
    return {"final_answer": answer, "outcome": outcome}


COULD_NOT_VERIFY = "The answer could not be checked against the documents."


def check_groundedness(state: GraphState) -> Dict[str, Any]:
    """On the strict path, list every claim the retrieved documents don't support. Fails closed."""
    print("---CHECK GROUNDEDNESS---")
    try:
        result = groundedness_checker.invoke(
            {
                "documents": format_documents(state.get("documents", [])),
                "answer": state.get("llm_generated_response", ""),
            }
        )
    except Exception as exc:  # a strict answer that can't be checked is not delivered
        print(f"Groundedness check failed ({type(exc).__name__}); failing closed")
        return {"unsupported_claims": [COULD_NOT_VERIFY], "errors": [f"CHECK_GROUNDEDNESS: {type(exc).__name__}"]}
    if not isinstance(result, GroundednessOutput):
        return {"unsupported_claims": [COULD_NOT_VERIFY]}
    claims = [c for c in result.unsupported_claims if c.strip()]
    print(f"Unsupported claims: {claims or 'none'}")
    return {"unsupported_claims": claims}


def decline_unsupported(state: GraphState) -> Dict[str, Any]:
    """Say why the answer is withheld, and which claims couldn't be verified."""
    if state.get("unsupported_claims") == [COULD_NOT_VERIFY]:
        # The check itself didn't run: say that, rather than blame the answer.
        print("---DECLINED_UNVERIFIED---")
        message = (
            "I drafted an answer, but I couldn't check it against the documents just now, "
            "so I'm not sharing it. Please try again shortly."
        )
        return {"final_answer": message, "outcome": "DECLINED_UNVERIFIED"}
    
    listed = "\n".join(f"- {claim}" for claim in state.get("unsupported_claims", []))
    message = (
        "I drafted an answer, but parts of it aren't supported by the documents, "
        f"so I'm not sharing it as fact.\nCouldn't verify:\n{listed}"
    )
    print("---DECLINED_UNSUPPORTED---")
    return {"final_answer": message, "outcome": "DECLINED_UNSUPPORTED"}
