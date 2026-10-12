"""Compare the baseline assistant with the risk-routed one on a labelled question set.

Usage (from the project root, with COHERE_API_KEY and TAVILY_API_KEY set):
    python -m eval.run_eval --questions eval/questions.jsonl --out eval/results

Every question runs through both versions. Each answer that is actually delivered
is then checked by the same groundedness judge, so both versions are measured the
same way. Token counts, model calls and latency come from the agent-trace session
of each run; the judge's own calls are not counted.

The judge is an LLM too: spot-check a sample of its verdicts by hand before
quoting the numbers.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from agent_trace_sdk import AgentTracer

from src.chains.formatting import format_documents
from src.chains.groundedness_checker import GroundednessOutput, groundedness_checker
from src.create_vectorstore.create_vectorstore import build_vectorstore, index_size
from src.observability.agent_trace import AgentTraceCallback, JsonlEventSink
from src.workflow.graph import application, baseline_application

MODES = (("baseline", baseline_application), ("risk_routed", application))


def judge(state: Dict) -> List[str]:
    """Unsupported claims in a delivered answer, according to the groundedness judge."""
    result = groundedness_checker.invoke(
        {
            "documents": format_documents(state.get("documents", [])),
            "answer": state.get("llm_generated_response", ""),
        }
    )
    if not isinstance(result, GroundednessOutput):
        return ["(judge returned no usable verdict)"]
    return [c for c in result.unsupported_claims if c.strip()]


def run_one(app, question: Dict, mode: str, sink: JsonlEventSink) -> Dict:
    tracer = AgentTracer(service_name="ekm-eval", sink=sink)
    metadata = {"query": question["query"], "mode": mode, "question_id": question["id"]}
    with tracer.session(agent_id="ekm-eval", metadata=metadata):
        state = app.invoke({"query": question["query"]}, config={"callbacks": [AgentTraceCallback(tracer)]})
    session = sink.last_session_end

    outcome = state.get("outcome", "")
    delivered = outcome.startswith("ANSWERED")
    claims = judge(state) if delivered else []
    return {
        "id": question["id"],
        "mode": mode,
        "query": question["query"],
        "expected_source": question.get("expected_source", ""),
        "expected_risk": question.get("expected_risk", ""),
        "answerable_from_docs": question.get("answerable_from_docs", ""),
        "source": state.get("source", ""),
        "risk": state.get("risk", ""),
        "risk_reasons": "; ".join(state.get("risk_reasons", [])),
        "outcome": outcome,
        "delivered": delivered,
        "judge_unsupported_count": len(claims) if delivered else "",
        "judge_unsupported_claims": " | ".join(claims),
        "llm_calls": session.total_llm_calls,
        "input_tokens": session.total_input_tokens,
        "output_tokens": session.total_output_tokens,
        "latency_s": round(session.duration_ms / 1000, 2),
        "final_answer": state.get("final_answer", ""),
    }


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


def _mean(rows: List[Dict], key: str) -> float:
    return statistics.mean(r[key] for r in rows) if rows else 0.0


def summarize(rows: List[Dict]) -> str:
    by_mode: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)

    lines = []
    for mode, rs in by_mode.items():
        n = len(rs)
        delivered = [r for r in rs if r["delivered"]]
        unsupported = [r for r in delivered if (r["judge_unsupported_count"] or 0) > 0]
        declined = [r for r in rs if not r["delivered"]]
        over_refused = [r for r in declined if str(r["answerable_from_docs"]).lower() == "true"]
        source_labelled = [r for r in rs if r["expected_source"]]
        source_ok = [r for r in source_labelled if r["source"] == r["expected_source"]]

        lines.append(f"\n== {mode} ({n} questions) ==")
        lines.append(f"Answers delivered:                   {_pct(len(delivered), n)}")
        lines.append(f"Delivered with unsupported claims:   {_pct(len(unsupported), n)}")
        lines.append(f"Declined:                            {_pct(len(declined), n)}")
        lines.append(f"  of which labelled answerable:      {len(over_refused)}")
        lines.append(f"Source routing matches label:        {_pct(len(source_ok), len(source_labelled))}")
        if mode == "risk_routed":
            risk_labelled = [r for r in rs if r["expected_risk"]]
            risk_ok = [r for r in risk_labelled if r["risk"] == r["expected_risk"]]
            missed = [r for r in risk_labelled if r["expected_risk"] == "STRICT" and r["risk"] != "STRICT"]
            lines.append(f"Risk routing matches label:          {_pct(len(risk_ok), len(risk_labelled))}")
            lines.append(f"  strict questions sent fast:        {len(missed)}")
        lines.append(f"Mean model calls per question:       {_mean(rs, 'llm_calls'):.1f}")
        lines.append(f"Mean input / output tokens:          {_mean(rs, 'input_tokens'):.0f} / {_mean(rs, 'output_tokens'):.0f}")
        lines.append(f"Mean latency (s):                    {_mean(rs, 'latency_s'):.2f}")
        if mode == "risk_routed":
            for path in ("FAST", "STRICT"):
                prs = [r for r in rs if r["risk"] == path]
                if prs:
                    lines.append(
                        f"  {path:<6} path: {len(prs)} questions, {_mean(prs, 'llm_calls'):.1f} calls, "
                        f"{_mean(prs, 'input_tokens'):.0f} input tokens, {_mean(prs, 'latency_s'):.2f}s"
                    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--out", default="eval/results")
    args = parser.parse_args()

    questions = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    if index_size() == 0:
        build_vectorstore()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sink = JsonlEventSink(out / "traces.jsonl")

    rows = []
    for question in questions:
        for mode, app in MODES:
            print(f"\n### {question['id']} [{mode}] {question['query']}")
            rows.append(run_one(app, question, mode, sink))

    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (out / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nWrote {out / 'results.csv'}, {out / 'summary.txt'} and {out / 'traces.jsonl'}")


if __name__ == "__main__":
    main()
