from dotenv import load_dotenv
from langchain_cohere import ChatCohere
import os
from cohere.errors import TooManyRequestsError
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")

# Cohere trial keys allow 20 chat calls a minute (and 1,000 calls a month).
# Both models share one limiter, so calls are spaced out instead of failing with
# TooManyRequestsError. With a production key, raise COHERE_CALLS_PER_MINUTE.
CALLS_PER_MINUTE = float(os.getenv("COHERE_CALLS_PER_MINUTE", "18"))
rate_limiter = InMemoryRateLimiter(requests_per_second=CALLS_PER_MINUTE / 60, check_every_n_seconds=0.1, max_bucket_size=1)


# Writes answers: a little variation is fine.
llm_model  = ChatCohere(model="command-a-03-2025",temperature=0.2,rate_limiter=rate_limiter)

# Routes and grades: decisions should be repeatable, so no sampling.
decision_model = ChatCohere(model="command-a-03-2025", temperature=0,  rate_limiter=rate_limiter)

def retry_on_rate_limit(chain: Runnable, initial_wait: float = 5) -> Runnable:
    """Retry a chain with backoff when the API still says too many requests.

    Without this, a rate limit looks like a model failure: the router falls back,
    the grader drops the document, the groundedness check fails closed.
    """
    return chain.with_retry(
        retry_if_exception_type=(TooManyRequestsError,),
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": initial_wait, "max": 60},
        stop_after_attempt=5,
    )