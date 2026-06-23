import re
from src.models.model import llm_model
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class RetrievalGraderOutput(BaseModel):
    relevant: bool = Field(
        description="Are Documents relevant to the question, 'true' or 'false'"
    )


retrieval_grader_template = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """
         You are a grader assessing relevance of a Document to a user question. 
         You will be provided with a question and one or more Document.
         If the Document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
         Your task is to grade as "true" ,when the Document is relevant to the question or grade as false when the Document is not relevant to the question. 
         """
         ),
        ("human", "Question: {query} Documents: {documents}")
    
    ]
)

llm_model_with_structured_output = llm_model.with_structured_output(RetrievalGraderOutput)

retrieval_grader =  retrieval_grader_template | llm_model_with_structured_output