"""Route by risk: decide how much an answer must be checked before it is delivered.

The source router decides WHERE to look. This router decides HOW MUCH to verify.
Two rules shape it:

1. Rules run first and cannot be overridden. If a question matches a high-risk
   pattern, it goes STRICT however confident the classifier is. The model never
   gets to talk itself out of checking.
2. Fail closed. If the classifier errors or returns something unusable, the
   question goes STRICT.

The answer is classified too, not just the question: a harmless-sounding
question can still produce an answer full of figures someone will act on.
"""

import re
from typing import List, Literal, Tuple

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.models.model import decision_model, retry_on_rate_limit

Risk = Literal["FAST", "STRICT"]

# Tuned for investment documents. Adjust to your domain: the rules encode what a
# wrong answer would cost, not what the topic is.
QUESTION_RULES = {
    "numbers or money": r"\b(\d+(\.\d+)?\s?%|percent\w*|rate|returns?|yield|fees?|charges?|cost|price|nav|amount|minimum|maximum)\b",
    "our own products or policies": r"\b(our|my|this)\s+(funds?|plans?|schemes?|portfolios?|polic(y|ies)|accounts?|products?)\b",
    "terms, eligibility or deadlines": r"\b(eligib\w*|lock[- ]?in|tenure|maturity|penalt\w*|withdraw\w*|redeem\w*|redemption|exit load|tax\w*|deadline|limit)\b",
    "advice someone may act on": r"\b(should i|should we|do you recommend|best (fund|plan|option)|which (fund|plan|option))\b",
}

ANSWER_RULES = {
    "answer contains figures": r"(\d+(\.\d+)?\s?%|[$€£₹]\s?\d|\b\d{1,3}(,\d{3})+\b)",
    "answer gives a directive": r"\b(you should|you must|we recommend|it is advisable)\b",
    "answer states dates or durations": r"\b\d+\s+(days?|weeks?|months?|years?)\b",
}


def _matches(text: str, rules: dict) -> List[str]:
    return [name for name, pattern in rules.items() if re.search(pattern, text, re.IGNORECASE)]


class RiskAssessment(BaseModel):
    risk: Risk = Field(description="FAST for low-stakes questions, STRICT if a wrong answer could cause harm or a bad decision")
    reason: str = Field(description="One short sentence explaining the choice")


_system = """You decide how carefully an assistant's answer must be checked.
Return STRICT if a wrong answer could lead someone to make a financial, legal,
HR or compliance decision, or if the question is about the organisation's own
products, policies, numbers or terms. Return FAST for general explanations,
writing help or questions where a wrong answer is low-cost.
When unsure, return STRICT."""

_prompt = ChatPromptTemplate.from_messages([("system", _system), ("user", "{query}")])

risk_classifier = retry_on_rate_limit(_prompt | decision_model.with_structured_output(RiskAssessment))


def assess_question_risk(query: str) -> Tuple[Risk, List[str]]:
    """Return (risk, reasons) for a question. Rules first, then the classifier."""
    rule_hits = _matches(query, QUESTION_RULES)
    if rule_hits:
        return "STRICT", [f"rule: {hit}" for hit in rule_hits]

    try:
        result = risk_classifier.invoke({"query": query})
    except Exception as exc:  # fail closed on any classifier error
        return "STRICT", [f"classifier failed ({type(exc).__name__}); failing closed"]
    if not isinstance(result, RiskAssessment):
        return "STRICT", ["classifier returned no usable answer; failing closed"]
    return result.risk, [f"classifier: {result.reason}"]


def assess_answer_risk(answer: str) -> List[str]:
    """Return the reasons an answer itself needs strict checking (empty if none)."""
    return [f"rule: {hit}" for hit in _matches(answer or "", ANSWER_RULES)]
