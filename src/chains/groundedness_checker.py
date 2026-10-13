"""Check that every factual statement in an answer is supported by the retrieved documents.

Used on the strict path as a gate. Deliberately simple: it checks claims against the
documents one statement at a time, so it can still pass an answer built from true
parts that are wrong together. Making claim checking robust is the next pattern.
"""

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.models.model import decision_model,retry_on_rate_limit


class GroundednessOutput(BaseModel):
    unsupported_claims: List[str] = Field(
        description="Each factual statement in the answer that the documents do not support. Empty if all are supported.",
    )


_system = """You verify an answer against source documents.
List every factual statement in the answer that cannot be verified from the documents.
Use only the documents, never outside knowledge. A statement that adds a number,
date, condition or name not present in the documents is unsupported.
If everything is supported, return an empty list."""

_prompt = ChatPromptTemplate.from_messages(
    [("system", _system), ("human", "Documents:\n{documents}\n\nAnswer:\n{answer}")]
)

groundedness_checker = retry_on_rate_limit(_prompt | decision_model.with_structured_output(GroundednessOutput))
