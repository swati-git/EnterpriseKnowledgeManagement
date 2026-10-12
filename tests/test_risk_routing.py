"""Offline tests for routing by risk."""

import pytest

from src.chains import risk_router
from src.chains.retrieval_grader import RetrievalGraderOutput
from src.chains.risk_router import RiskAssessment, assess_question_risk
from tests.test_baseline import Stub, wired  # noqa: F401  (reuse the stubbed graph)


@pytest.mark.parametrize(
    "query",
    [
        "What is the lock-in period of our equity fund?",
        "What is the exit load if I redeem in 6 months?",
        "Should I move my savings into the debt fund?",
        "What fees does this plan charge?",
    ],
)
def test_high_risk_questions_go_strict_without_asking_the_model(query, monkeypatch):
    classifier = Stub(RiskAssessment(risk="FAST", reason="looks harmless"))
    monkeypatch.setattr(risk_router, "risk_classifier", classifier)
    risk, reasons = assess_question_risk(query)
    assert risk == "STRICT"
    assert all(r.startswith("rule:") for r in reasons)
    assert classifier.calls == []  # rules can't be overridden by the model


def test_low_risk_question_uses_the_classifier(monkeypatch):
    monkeypatch.setattr(risk_router, "risk_classifier", Stub(RiskAssessment(risk="FAST", reason="writing help")))
    assert assess_question_risk("How do I write a good out-of-office message?") == ("FAST", ["classifier: writing help"])


@pytest.mark.parametrize("classifier", [Stub(None), Stub(error=TimeoutError("slow"))])
def test_classifier_failure_fails_closed(classifier, monkeypatch):
    monkeypatch.setattr(risk_router, "risk_classifier", classifier)
    risk, reasons = assess_question_risk("Explain diversification")
    assert risk == "STRICT" and "failing closed" in reasons[0]


def test_risk_is_recorded_in_the_state(wired):  # noqa: F811
    app, _ = wired
    state = app.invoke({"query": "What is the lock-in period of our fund?"})
    assert state["risk"] == "STRICT"
    assert any("lock" in r or "terms" in r for r in state["risk_reasons"])


# ---------- R2: answer risk ----------

def test_answer_rules_flag_figures_directives_and_durations():
    assert risk_router.assess_answer_risk("Index funds have returned about 8% a year.")
    assert risk_router.assess_answer_risk("You should hold it for 3 years.")
    assert risk_router.assess_answer_risk("Diversification spreads risk across assets.") == []


def test_fast_question_with_figures_in_the_answer_is_escalated(wired):  # noqa: F811
    app, s = wired
    s["grader"].fn = lambda p: RetrievalGraderOutput(relevant=True)
    s["answer"].result = "Index funds have historically returned around 10% a year."
    state = app.invoke({"query": "How do index funds work?"})
    assert state["risk"] == "STRICT"
    assert state["risk_reasons"][0].startswith("classifier:")
    assert any("figures" in r for r in state["risk_reasons"])


def test_fast_question_with_a_plain_answer_stays_fast(wired):  # noqa: F811
    app, s = wired
    s["grader"].fn = lambda p: RetrievalGraderOutput(relevant=True)
    s["answer"].result = "Diversification spreads money across different assets."
    state = app.invoke({"query": "What does diversification mean?"})
    assert state["risk"] == "FAST"


# ---------- R3: outcomes and sources ----------

def test_answer_is_delivered_with_its_sources(wired):  # noqa: F811
    app, _ = wired
    state = app.invoke({"query": "What is the lock-in period of our fund?"})
    assert state["final_answer"].startswith("Fund A has a 3 year lock-in.")
    assert state["final_answer"].endswith("Sources: fund_a.pdf (page 1)")


def test_every_run_records_an_outcome(wired):  # noqa: F811
    app, s = wired
    s["grader"].fn = lambda p: RetrievalGraderOutput(relevant=True)
    s["answer"].result = "Diversification spreads money across different assets."
    assert app.invoke({"query": "What does diversification mean?"})["outcome"] == "ANSWERED_FAST"

    s["retriever"].result = []
    declined = app.invoke({"query": "What is the lock-in period of our fund?"})
    assert declined["outcome"] == "DECLINED_NO_RELEVANT_DOCS"
    assert declined["final_answer"] == declined["llm_generated_response"]
