import re
from src.models.model import decision_model, retry_on_rate_limit
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

retrieval_grader =  retry_on_rate_limit(retrieval_grader_template | decision_model.with_structured_output(RetrievalGraderOutput))