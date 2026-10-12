"""Local replacement for the dashboard's Confluent data access.

agent-trace-sdk's dashboard reads three tables that Flink SQL derives from the raw
event topic. This module derives the same tables, with the same column names, from
the local JSONL file written by src/observability/agent_trace.py. The joins mirror
flink/01_llm_call_completed.sql, 02_tool_call_completed.sql and 03_session_summaries.sql.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

TRACE_FILE = Path(__file__).resolve().parent.parent / "telemetry" / "agent_trace_events.jsonl"


def _events() -> pd.DataFrame:
    if not TRACE_FILE.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in TRACE_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def _upper(df: pd.DataFrame) -> pd.DataFrame:
    # The dashboard expects upper-case column names, except "timestamp".
    return df.rename(columns={c: c if c == "timestamp" else c.upper() for c in df.columns})


def _completed(events: pd.DataFrame, kind: str, start_cols: list, end_cols: list) -> pd.DataFrame:
    """Join <kind>.start and <kind>.end events on span_id, as the Flink jobs do."""
    if events.empty or "event_type" not in events:
        return pd.DataFrame()
    starts = events[events["event_type"] == f"{kind}.start"]
    ends = events[events["event_type"] == f"{kind}.end"]
    if starts.empty or ends.empty:
        return pd.DataFrame()
    joined = starts[["trace_id", "span_id", "parent_span_id", "service_name", "metadata", *start_cols]].merge(
        ends[["span_id", *end_cols]], on="span_id", how="inner"
    )
    joined["node"] = joined["metadata"].apply(lambda m: (m or {}).get("node", ""))
    return _upper(joined.drop(columns=["metadata"]))


@st.cache_data(ttl=5)
def load_sessions() -> pd.DataFrame:
    events = _events()
    if events.empty:
        return pd.DataFrame()
    cols = ["trace_id", "service_name", "timestamp", "duration_ms", "total_llm_calls", "total_tool_calls",
            "total_input_tokens", "total_output_tokens", "exit_reason", "error"]
    sessions = events[events["event_type"] == "session.end"][cols]
    starts = events[events["event_type"] == "session.start"][["trace_id", "metadata"]]
    sessions = sessions.merge(starts, on="trace_id", how="left")
    sessions["query"] = sessions["metadata"].apply(lambda m: (m or {}).get("query", "") if isinstance(m, dict) else "")
    return _upper(sessions.drop(columns=["metadata"]).sort_values("timestamp", ascending=False))


@st.cache_data(ttl=5)
def load_llm_calls() -> pd.DataFrame:
    calls = _completed(
        _events(), "llm_call",
        ["model", "provider"],
        ["duration_ms", "input_tokens", "output_tokens", "total_tokens", "response_content", "finish_reason", "error"],
    )
    if not calls.empty:
        # This app uses one model for every step, so "by model" says little.
        # Label each call with its graph node instead: the Models page then shows tokens per node.
        calls["MODEL"] = calls["NODE"].where(calls["NODE"] != "", calls["MODEL"]) + " · " + calls["MODEL"]
    return calls


@st.cache_data(ttl=5)
def load_tool_calls() -> pd.DataFrame:
    return _completed(
        _events(), "tool_call",
        ["tool_name", "tool_input"],
        ["duration_ms", "tool_output", "success", "error"],
    )


def load_trace_llm_calls(trace_id: str) -> pd.DataFrame:
    calls = load_llm_calls()
    return calls[calls["TRACE_ID"] == trace_id] if not calls.empty else calls


def load_trace_tool_calls(trace_id: str) -> pd.DataFrame:
    calls = load_tool_calls()
    return calls[calls["TRACE_ID"] == trace_id] if not calls.empty else calls
