import sys
from pathlib import Path
from typing import TypedDict

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.workflow.graph import application
from src.create_vectorstore.create_vectorstore import build_vectorstore, index_size

def main():  
    if index_size() == 0:
        print("The document index is empty; building it once (this embeds every PDF).")
        build_vectorstore()
    
    query = input("Type your query:")
    print(f"You asked: {query}")

    print("Processing...")
    result = None
    state_input = {
        "query": query,
    }

    for output in application.stream(state_input) :
        #print("Intermediate output:", output)
        for key, value in output.items():
            result = value
                    
    
    if result:
        print(f"\nAnswer: {format_response(result)}")
    else:
        print("No response generated.")

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