import os
import sys
from pathlib import Path
from typing import TypedDict

#sys.path.append(str(Path(__file__).parent.parent.parent))

from src.workflow.graph import application, baseline_application
from src.create_vectorstore.create_vectorstore import build_vectorstore, index_size
from agent_trace_sdk import AgentTracer
from src.observability.agent_trace import AgentTraceCallback, JsonlEventSink

REQUIRED_KEYS = ("COHERE_API_KEY", "TAVILY_API_KEY")


def check_environment() -> None:
    missing = [key for key in REQUIRED_KEYS if not os.getenv(key)]
    if missing:
        sys.exit(f"Missing environment variables: {', '.join(missing)}. Add them to .env and try again.")


def main():
    check_environment()
  
    if index_size() == 0:
        print("The document index is empty; building it once (this embeds every PDF).")
        build_vectorstore()
    
    query = input("Type your query:").strip()
    if not query:
        sys.exit("No question entered.")


    sink = JsonlEventSink()
    agent_tracer = AgentTracer(service_name="enterprise-knowledge-assistant", sink=sink)
    callbacks = [ AgentTraceCallback(agent_tracer)]

    # `python -m src.cli.main --baseline` runs the assistant without risk routing, for comparison.
    baseline = "--baseline" in sys.argv
    app = baseline_application if baseline else application
    mode = "baseline" if baseline else "risk_routed"

    with agent_tracer.session(agent_id="ekm-assistant", metadata={"query": query, "mode": mode}):
        state = app.invoke({"query": query}, config={"callbacks": callbacks})
                    

    print(f"\nSource: {state.get('source')} | Risk: {state.get('risk')} | Outcome: {state.get('outcome')}")
    print(f"Answer: {state.get('final_answer', 'No response generated.')}")
    print(sink.session_summary())

def format_response(result):
    """Extract response from workflow result."""
    print(f"type of result: {type(result)}")
    if isinstance(result, dict) and "llm_generated_response" in result:
        print("Formatting response from llm_generated_response")
        return result["llm_generated_response"]
    elif isinstance(result, dict) and "response" in result:
        print("Formatting response from response key")
        return result["response"]
    else:
        print("Formatting response from raw result")
        return str(result)
   
    
if __name__ == "__main__":
    main()