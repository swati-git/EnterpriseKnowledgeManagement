from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.models.model import llm_model

llm = llm_model

prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Question: {query} 
    Context: {documents} 
    Answer:"""

)

response_formatter = prompt | llm | StrOutputParser()