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


# ---------- R4: verify strict answers ----------

def test_strict_supported_answer_is_verified_and_delivered(wired):  # noqa: F811
    app, s = wired
    state = app.invoke({"query": "What is the lock-in period of our fund?"})
    assert state["outcome"] == "ANSWERED_VERIFIED"
    assert state["final_answer"].endswith("Sources: fund_a.pdf (page 1)")
    assert len(s["grounded"].calls) == 1


def test_strict_answer_with_unsupported_claim_is_declined(wired):  # noqa: F811
    from src.chains.groundedness_checker import GroundednessOutput

    app, s = wired
    s["answer"].result = "Fund A has a 3 year lock-in and a 1% exit load."
    s["grounded"].result = GroundednessOutput(unsupported_claims=["Fund A has a 1% exit load."])
    state = app.invoke({"query": "What is the lock-in period of our fund?"})
    assert state["outcome"] == "DECLINED_UNSUPPORTED"
    assert "- Fund A has a 1% exit load." in state["final_answer"]
    assert "Sources:" not in state["final_answer"]


def test_groundedness_failure_fails_closed(wired):  # noqa: F811
    app, s = wired
    s["grounded"].error = TimeoutError("slow")
    state = app.invoke({"query": "What is the lock-in period of our fund?"})
    assert state["outcome"] == "DECLINED_UNSUPPORTED"
    assert "could not be checked" in state["final_answer"]


def test_fast_answer_skips_the_groundedness_check(wired):  # noqa: F811
    app, s = wired
    s["grader"].fn = lambda p: RetrievalGraderOutput(relevant=True)
    s["answer"].result = "Diversification spreads money across different assets."
    state = app.invoke({"query": "What does diversification mean?"})
    assert state["outcome"] == "ANSWERED_FAST"
    assert s["grounded"].calls == []


def test_escalated_answer_is_checked(wired):  # noqa: F811
    app, s = wired
    s["grader"].fn = lambda p: RetrievalGraderOutput(relevant=True)
    s["answer"].result = "Index funds have historically returned around 10% a year."
    state = app.invoke({"query": "How do index funds work?"})
    assert len(s["grounded"].calls) == 1
    assert state["outcome"] == "ANSWERED_VERIFIED"


# ---------- R5: switchable risk routing and the evaluation ----------

def test_baseline_skips_risk_routing(wired, monkeypatch):  # noqa: F811
    import src.workflow.graph as graph_mod

    _, s = wired
    state = graph_mod.baseline_application.invoke({"query": "What is the lock-in period of our fund?"})
    assert "risk" not in state
    assert state["outcome"] == "ANSWERED_UNCHECKED"
    assert state["final_answer"].endswith("Sources: fund_a.pdf (page 1)")
    assert s["grounded"].calls == []


def test_eval_runs_both_versions_and_judges_delivered_answers(wired, monkeypatch, tmp_path):  # noqa: F811
    import eval.run_eval as run_eval
    from src.chains.groundedness_checker import GroundednessOutput
    from src.observability.agent_trace import JsonlEventSink

    _, s = wired
    s["answer"].result = "Fund A has a 3 year lock-in and a 1% exit load."
    # The runtime check (strict path) and the judge both flag the exit load.
    s["grounded"].result = GroundednessOutput(unsupported_claims=["Fund A has a 1% exit load."])
    monkeypatch.setattr(run_eval, "groundedness_checker", s["grounded"])

    question = {"id": "q1", "query": "What is the lock-in period of our fund?",
                "expected_source": "VECTORSTORE", "expected_risk": "STRICT", "answerable_from_docs": True}
    sink = JsonlEventSink(tmp_path / "traces.jsonl")
    rows = [run_eval.run_one(app, question, mode, sink) for mode, app in run_eval.MODES]
    baseline, routed = rows

    assert baseline["outcome"] == "ANSWERED_UNCHECKED" and baseline["judge_unsupported_count"] == 1
    assert routed["outcome"] == "DECLINED_UNSUPPORTED" and routed["judge_unsupported_count"] == ""

    summary = run_eval.summarize(rows)
    assert "Delivered with unsupported claims:   1/1 (100%)" in summary  # baseline
    assert "Delivered with unsupported claims:   0/1 (0%)" in summary  # risk-routed
    assert "of which labelled answerable:      1" in summary
