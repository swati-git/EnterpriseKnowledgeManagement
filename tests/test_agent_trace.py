"""Offline tests for the agent-trace-sdk adapter: a fake model runs inside a small LangGraph graph."""

import json
from typing import TypedDict

from agent_trace_sdk import AgentTracer
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph

from src.observability.agent_trace import AgentTraceCallback, JsonlEventSink


class FakeChatModel(BaseChatModel):
    """Reports token usage the way ChatCohere does, via usage_metadata."""

    input_tokens: int = 100

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(
            content="ok",
            usage_metadata={"input_tokens": self.input_tokens, "output_tokens": 5, "total_tokens": self.input_tokens + 5},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake"

    @property
    def _identifying_params(self):
        return {"model": "command-a-03-2025"}


class FakeRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content="x")] * 3


class State(TypedDict, total=False):
    query: str


def build_app():
    graph = StateGraph(State)
    graph.add_node("ROUTE", lambda s: (FakeChatModel(input_tokens=50).invoke("route"), {})[1])
    graph.add_node("VECTORSTORE", lambda s: (FakeRetriever().invoke(s["query"]), {})[1])
    graph.add_node("GENERATE_ANSWER", lambda s: (FakeChatModel(input_tokens=900).invoke("answer"), {})[1])
    graph.add_edge(START, "ROUTE")
    graph.add_edge("ROUTE", "VECTORSTORE")
    graph.add_edge("VECTORSTORE", "GENERATE_ANSWER")
    graph.add_edge("GENERATE_ANSWER", END)
    return graph.compile()


def run_one_question(tmp_path):
    sink = JsonlEventSink(tmp_path / "events.jsonl")
    tracer = AgentTracer(service_name="test", sink=sink)
    with tracer.session(agent_id="test", metadata={"query": "q"}):
        build_app().invoke({"query": "q"}, config={"callbacks": [AgentTraceCallback(tracer)]})
    events = [json.loads(line) for line in sink.path.read_text().splitlines()]
    return sink, events


def test_session_totals_count_every_model_and_tool_call(tmp_path):
    _, events = run_one_question(tmp_path)
    end = next(e for e in events if e["event_type"] == "session.end")
    assert end["total_llm_calls"] == 2
    assert end["total_tool_calls"] == 1
    assert end["total_input_tokens"] == 950
    assert end["total_output_tokens"] == 10


def test_spans_share_the_session_trace_and_record_their_node(tmp_path):
    _, events = run_one_question(tmp_path)
    assert len({e["trace_id"] for e in events}) == 1 and events[0]["trace_id"]
    nodes = [e["metadata"]["node"] for e in events if e["event_type"] == "llm_call.start"]
    assert nodes == ["ROUTE", "GENERATE_ANSWER"]
    models = {e["model"] for e in events if e["event_type"] == "llm_call.start"}
    assert models == {"command-a-03-2025"}


def test_console_summary_matches_the_session_event(tmp_path):
    sink, _ = run_one_question(tmp_path)
    summary = sink.session_summary()
    assert "2 LLM calls" in summary and "1 tool calls" in summary and "950 input" in summary
