from src.graph_state.state import GraphState
from typing import Any, Dict
from langchain_tavily import TavilySearch
from langchain.schema import Document
from dotenv import load_dotenv

web_search_tool = TavilySearch(max_results=3)

load_dotenv()

def web_search(graph_state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
   
    tavily_results = web_search_tool.invoke({"query" : graph_state["query"]})
    results = tavily_results.get("results", []) if isinstance(tavily_results, dict) else []
    documents = [
        Document(
            page_content=r.get("content", ""),
            metadata={"source": r.get("url", "web"), "title": r.get("title", "")},
        )
        for r in results
        if r.get("content")
    ]
    print(f"Web search returned {len(documents)} results.")
    return {"documents": documents}

