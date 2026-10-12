from src.graph_state.state import GraphState
from typing import Any, Dict
from langchain_tavily import TavilySearch
from langchain.schema import Document
from dotenv import load_dotenv

web_search_tool = TavilySearch(max_results=3)

load_dotenv()

def web_search(graph_state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
   
    tavily_results = web_search_tool.invoke({"query" : graph_state["query"]})["results"]
    complete_result = "\n".join([tavily_result["content"] for tavily_result in tavily_results])
    web_results = Document(page_content=complete_result)
    return {"documents": web_results, 
            "query": graph_state["query"]}

