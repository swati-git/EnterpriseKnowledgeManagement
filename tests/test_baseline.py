"""Offline tests for the baseline assistant. Every model, retriever and search call is stubbed."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from src.chains.formatting import format_documents
from src.chains.query_router import QueryRouter
from src.chains.retrieval_grader import RetrievalGraderOutput

class Stub:
    """Stands in for a chain: returns a fixed result, a computed one, or raises."""

    def __init__(self, result=None, error=None, fn=None):
        self.result, self.error, self.fn, self.calls = result, error, fn, []

    def invoke(self, payload, *_args, **_kwargs):
        self.calls.append(payload)
        if self.error:
            raise self.error
        return self.fn(payload) if self.fn else self.result

    def batch(self, payloads, *_args, return_exceptions=False, **_kwargs):
        results = []
        for payload in payloads:
            try:
                results.append(self.invoke(payload))
            except Exception as exc:
                if not return_exceptions:
                    raise
                results.append(exc)
        return results


FUND_A = Document(page_content="Fund A has a 3 year lock-in.", metadata={"source": "/data/fund_a.pdf", "page": 0})
WEATHER = Document(page_content="It will rain in Pune tomorrow.", metadata={"source": "/data/weather.pdf", "page": 2})


@pytest.fixture
def wired(monkeypatch):
    """Stub every external call the graph makes; return the stubs to configure per test."""
    import src.graph_nodes.generate_llm_answer as gen_mod
    import src.graph_nodes.grader as grader_mod
    import src.graph_nodes.vectorstore as vs_mod
    import src.graph_nodes.web_search as web_mod
    import src.workflow.graph as graph_mod

    stubs = {
        "router": Stub(QueryRouter(source="VECTORSTORE")),
        "retriever": Stub([FUND_A, WEATHER]),
        "grader": Stub(fn=lambda p: RetrievalGraderOutput(relevant="lock-in" in p["document"])),
        "answer": Stub("Fund A has a 3 year lock-in."),
        "web": Stub({"results": [{"url": "https://example.com/a", "title": "A", "content": "Diversification spreads risk."}]}),
    }
    monkeypatch.setattr(graph_mod, "query_router", stubs["router"])
    monkeypatch.setattr(vs_mod, "get_retriever", lambda: stubs["retriever"])
    monkeypatch.setattr(grader_mod, "retrieval_grader", stubs["grader"])
    monkeypatch.setattr(gen_mod, "response_formatter", stubs["answer"])
    monkeypatch.setattr(web_mod, "web_search_tool", stubs["web"])
    # Questions that no risk rule matches go to the risk classifier; keep it offline too.
    from src.chains import risk_router
    monkeypatch.setattr(risk_router, "risk_classifier", Stub(risk_router.RiskAssessment(risk="FAST", reason="test")))
    # Strict answers go through the groundedness check; by default every claim is supported.
    import src.graph_nodes.risk_nodes as risk_nodes_mod
    from src.chains.groundedness_checker import GroundednessOutput
    stubs["grounded"] = Stub(GroundednessOutput(unsupported_claims=[]))
    monkeypatch.setattr(risk_nodes_mod, "groundedness_checker", stubs["grounded"])

    return graph_mod.application, stubs


# ---------- Commit 1: state keys ----------

def test_answer_and_grade_survive_to_final_state(wired):
    """Regression: undeclared state keys used to be dropped, losing the grade and the answer."""
    app, _ = wired
    state = app.invoke({"query": "What is the lock-in of Fund A?"})
    assert state["llm_generated_response"] == "Fund A has a 3 year lock-in."
    assert state["relevant"] is True


# ---------- Commit 2: the grader gates generation ----------

def test_no_relevant_documents_skips_generation(wired):
    app, s = wired
    s["retriever"].result = [WEATHER]
    state = app.invoke({"query": "What is the lock-in of Fund A?"})
    assert state["relevant"] is False
    assert "couldn't find" in state["llm_generated_response"]
    assert s["answer"].calls == []


# ---------- Commit 3: the index is built once ----------

def test_index_is_built_once_not_on_every_start(tmp_path):
    """Regression: the old code re-embedded and appended every chunk on each start."""
    import src.create_vectorstore.create_vectorstore as cv

    cv.reset_vectorstore_cache()
    try:
        cv.get_vectorstore(embeddings=FakeEmbeddings(size=8), persist_directory=tmp_path)
        docs = [FUND_A, WEATHER]
        first = cv.build_vectorstore(documents=docs)
        second = cv.build_vectorstore(documents=docs)  # a second start
        rebuilt = cv.build_vectorstore(rebuild=True, documents=docs)
        assert first == second == rebuilt == 2
    finally:
        cv.reset_vectorstore_cache()


# ---------- Commit 4: web results ----------

def test_web_search_returns_one_document_per_result_with_url(wired):
    app, s = wired
    s["router"].result = QueryRouter(source="WEBSEARCH")
    s["grader"].fn = lambda p: RetrievalGraderOutput(relevant=True)
    state = app.invoke({"query": "What does diversification mean?"})
    assert isinstance(state["documents"], list)
    assert state["documents"][0].metadata["source"] == "https://example.com/a"


def test_web_search_with_no_results_declines(wired):
    app, s = wired
    s["router"].result = QueryRouter(source="WEBSEARCH")
    s["web"].result = {"results": []}
    state = app.invoke({"query": "anything"})
    assert state["relevant"] is False and s["answer"].calls == []


# ---------- Commit 5: clean prompt text ----------

def test_generator_gets_clean_text_not_object_reprs(wired):
    app, s = wired
    app.invoke({"query": "What is the lock-in of Fund A?"})
    prompt_documents = s["answer"].calls[0]["documents"]
    assert "Source: fund_a.pdf (page 1)" in prompt_documents
    assert "Document(" not in prompt_documents and "page_content=" not in prompt_documents
    assert "/data/" not in prompt_documents  # file name only, not the full path


def test_format_documents_handles_web_and_pdf_sources():
    web = Document(page_content="x", metadata={"source": "https://example.com/page"})
    text = format_documents([FUND_A, web])
    assert "[1] Source: fund_a.pdf (page 1)" in text
    assert "[2] Source: https://example.com/page" in text
    assert format_documents([]) == "(no documents)"


# ---------- Commit 6: per-document grading ----------

def test_grader_keeps_only_relevant_documents(wired):
    app, s = wired
    state = app.invoke({"query": "What is the lock-in of Fund A?"})
    assert state["documents"] == [FUND_A]
    assert len(s["grader"].calls) == 2  # one grade per document
    assert "rain in Pune" not in s["answer"].calls[0]["documents"]


def test_failed_grade_counts_as_not_relevant(wired):
    app, s = wired

    def flaky(payload):
        if "rain" in payload["document"]:
            raise TimeoutError("slow")
        return RetrievalGraderOutput(relevant=True)

    s["grader"].fn = flaky
    state = app.invoke({"query": "What is the lock-in of Fund A?"})
    assert state["documents"] == [FUND_A]


# ---------- Commit 7: routing node with a safe default ----------

def test_routing_decision_is_kept_in_state(wired):
    app, _ = wired
    assert app.invoke({"query": "What is the lock-in of Fund A?"})["source"] == "VECTORSTORE"


@pytest.mark.parametrize("router_stub", [Stub(None), Stub(error=TimeoutError("slow"))])
def test_router_failure_defaults_to_vectorstore(wired, monkeypatch, router_stub):
    import src.workflow.graph as graph_mod

    app, _ = wired
    monkeypatch.setattr(graph_mod, "query_router", router_stub)
    assert app.invoke({"query": "anything"})["source"] == "VECTORSTORE"


# ---------- Commit 8: the CLI checks for keys ----------

def test_missing_api_key_stops_with_a_clear_message(monkeypatch):
    from src.cli.main import check_environment

    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="COHERE_API_KEY"):
        check_environment()
