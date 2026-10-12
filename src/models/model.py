from dotenv import load_dotenv
from langchain_cohere import ChatCohere
import os

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")

# Writes answers: a little variation is fine.
llm_model  = ChatCohere(model="command-a-03-2025",temperature=0.2)

# Routes and grades: decisions should be repeatable, so no sampling.
decision_model = ChatCohere(model="command-a-03-2025", temperature=0)