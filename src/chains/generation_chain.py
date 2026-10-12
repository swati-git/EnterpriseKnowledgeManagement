from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.models.model import llm_model, retry_on_rate_limit

llm = llm_model

prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Answer the question using only the context below.
    If the context doesn't contain the answer, say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Question: {query} 
    Context: {documents} 
    Answer:"""

)

response_formatter = retry_on_rate_limit(prompt | llm | StrOutputParser())