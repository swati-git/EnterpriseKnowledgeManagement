"""Send LangGraph activity to agent-trace-sdk, stored in a local file (no Kafka, no cloud).

agent-trace-sdk (https://github.com/thefalc/agent-trace-sdk) models a run as a
session with LLM-call and tool-call spans. It has no LangChain integration, so
this module provides one: a callback that opens and closes those spans around
every model call and retrieval, and a sink that writes the events to JSONL.

Usage:
    tracer = AgentTracer(service_name="enterprise-knowledge-assistant", sink=JsonlEventSink())
    with tracer.session(agent_id="ekm", metadata={"query": q}):
        application.invoke({"query": q}, config={"callbacks": [AgentTraceCallback(tracer)]})
"""

from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

from agent_trace_sdk import AgentTracer
from agent_trace_sdk.models import BaseEvent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

TRACE_FILE = Path(__file__).resolve().parent.parent / "telemetry" / "agent_trace_events.jsonl"
MAX_TEXT = 2000  # keep stored responses and tool outputs readable, and the file small


class JsonlEventSink:
    """An agent-trace-sdk event sink that appends each event as one JSON line."""

    def __init__(self, path: Path = TRACE_FILE) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.last_session_end: Optional[BaseEvent] = None

    def emit(self, event: BaseEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
        if event.event_type == "session.end":
            self.last_session_end = event

    def session_summary(self) -> str:
        """One line for the console, from the same session.end event the dashboard reads."""
        e = self.last_session_end
        if e is None:
            return "No session recorded."
        return (
            f"Session {e.trace_id[:8]}: {e.total_llm_calls} LLM calls, {e.total_tool_calls} tool calls, "
            f"{e.total_input_tokens:,} input / {e.total_output_tokens:,} output tokens, "
            f"{e.duration_ms / 1000:.1f} s ({e.exit_reason})"
        )

    def flush(self, timeout: float = 10.0) -> None:
        pass

    def close(self) -> None:
        pass


def _usage(response: LLMResult) -> Dict[str, Optional[int]]:
    for generations in response.generations:
        for generation in generations:
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if usage:
                return {
                    "input": usage.get("input_tokens"),
                    "output": usage.get("output_tokens"),
                    "total": usage.get("total_tokens"),
                }
    return {"input": None, "output": None, "total": None}


def _text(response: LLMResult) -> str:
    """The answer text, or the structured-output tool call when there is no text."""
    for generations in response.generations:
        for generation in generations:
            message = getattr(generation, "message", None)
            if message is not None and getattr(message, "tool_calls", None):
                return str(message.tool_calls)[:MAX_TEXT]
            if generation.text:
                return generation.text[:MAX_TEXT]
    return ""


class AgentTraceCallback(BaseCallbackHandler):
    """Turns LangChain callbacks into agent-trace-sdk spans, tagged with the graph node."""

    def __init__(self, tracer: AgentTracer, provider: str = "cohere") -> None:
        self.tracer = tracer
        self.provider = provider
        self._spans: Dict[UUID, Any] = {}

    @staticmethod
    def _node(metadata: Optional[Dict[str, Any]]) -> str:
        return (metadata or {}).get("langgraph_node", "outside graph")

    # ---- model calls -> LLM spans ----
    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        params = kwargs.get("invocation_params") or {}
        span = self.tracer.llm_call(
            model=str(params.get("model") or params.get("model_name") or "unknown"),
            provider=self.provider,
            temperature=params.get("temperature"),
            metadata={"node": self._node(metadata)},
        )
        span.__enter__()
        self._spans[run_id] = span

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs):
        span = self._spans.pop(run_id, None)
        if span is None:
            return
        usage = _usage(response)
        span.set_response(
            input_tokens=usage["input"],
            output_tokens=usage["output"],
            total_tokens=usage["total"],
            response_content=_text(response),
        )
        span.__exit__(None, None, None)

    def on_llm_error(self, error: BaseException, *, run_id, **kwargs):
        span = self._spans.pop(run_id, None)
        if span is not None:
            span.__exit__(type(error), error, None)

    # ---- retrieval and web search -> tool spans ----
    def _start_tool(self, run_id: UUID, name: str, tool_input: Any, metadata) -> None:
        span = self.tracer.tool_call(
            tool_name=name, tool_input={"input": str(tool_input)[:MAX_TEXT]}, metadata={"node": self._node(metadata)}
        )
        span.__enter__()
        self._spans[run_id] = span

    def _end_tool(self, run_id: UUID, output: Any, error: Optional[BaseException] = None) -> None:
        span = self._spans.pop(run_id, None)
        if span is None:
            return
        span.set_output(str(output)[:MAX_TEXT], success=error is None)
        span.__exit__(type(error) if error else None, error, None)

    def on_retriever_start(self, serialized, query, *, run_id, metadata=None, **kwargs):
        self._start_tool(run_id, "vectorstore_retriever", query, metadata)

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        self._end_tool(run_id, f"{len(documents)} documents")

    def on_retriever_error(self, error, *, run_id, **kwargs):
        self._end_tool(run_id, None, error)

    def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, **kwargs):
        self._start_tool(run_id, (serialized or {}).get("name") or "tool", input_str, metadata)

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._end_tool(run_id, output)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._end_tool(run_id, None, error)