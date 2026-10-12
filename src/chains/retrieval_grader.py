import re
from src.models.model import llm_model
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class RetrievalGraderOutput(BaseModel):
    relevant: bool = Field(
        description="Is the  Document relevant to the question, 'true' or 'false'"
    )


retrieval_grader_template = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """
         You are a grader assessing relevance of a Document to a user question. 
         You will be provided with a question and one or more Document.
         If the Document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
         You grade whether one retrieved document is relevant to a user question. If the document contains keywords or meaning related to the question, grade it relevant (true).
         Otherwise grade it not relevant (false). 
         """
         ),
        ("human", "Question: {query} Document: {document}")
    
    ]
)

llm_model_with_structured_output = llm_model.with_structured_output(RetrievalGraderOutput)

retrieval_grader =  retrieval_grader_template | llm_model_with_structured_output