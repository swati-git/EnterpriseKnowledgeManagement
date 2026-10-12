# Enterprise Knowledge Management

An enterprise knowledge assistant that answers employees' questions from internal investment documents, and uses web search for general questions. It checks that what it retrieved is relevant before answering, and says so when it finds nothing relevant instead of guessing.

## How it works

A LangGraph workflow:

1. **Route**: a model call picks the source, `VECTORSTORE` (internal investment PDFs) or `WEBSEARCH` (Tavily). If routing fails, it defaults to the vector store.
2. **Retrieve**: the top chunks from the vector store, or the top web results, each kept as its own document with its source (file and page, or URL).
3. **Grade**: each retrieved document is graded for relevance on its own; irrelevant ones are dropped.
4. **Answer or decline**: if relevant documents remain, the answer is generated from them only. Otherwise the assistant replies that it couldn't find relevant information.

Routing and grading use temperature 0, so the same question gets the same decisions.

## Usage examples

- "What is the lock-in period of our equity fund?" → vector store (internal documents)
- "What does diversification mean?" → web search

## Setup

1. Python 3.12 and [uv](https://docs.astral.sh/uv/).
2. `uv sync`
3. Create `.env` with `COHERE_API_KEY` and `TAVILY_API_KEY`.
4. Put the PDFs in `enterprise_data/`.
5. Build the index once:
   `uv run python -m src.create_vectorstore.create_vectorstore`
   Add `--rebuild` after changing the documents or the chunking settings.

## Run

```
uv run python -m src.cli.main
```

If the index is empty, the CLI builds it once before answering.

## Tracing and dashboard

Every question is recorded as a trace using [agent-trace-sdk](https://github.com/thefalc/agent-trace-sdk) by Sean Falconer (Apache-2.0), adapted to run locally with no Kafka or cloud services:

- Each question is a **session**; each model call is an **LLM span** and each retrieval or web search a **tool span**, tagged with the graph node that made it (ROUTE, GRADER, GENERATE_ANSWER…).
- Events are written to `telemetry/agent_trace_events.jsonl` on your machine. Nothing is sent to an external service.
- After each answer, the CLI prints a one-line summary: model calls, tool calls, input/output tokens and duration.

To view the dashboard:

```
uv sync --group dashboard
uv run streamlit run dashboard/app.py
```

Then open http://localhost:8501. The **Models** page groups model calls by graph node, which shows where tokens are spent.

`dashboard/app.py` is adapted from the SDK's Streamlit dashboard to read the local file instead of Confluent Cloud; see `dashboard/LICENSE-agent-trace-sdk`.

## Model specs

- Chat model: Cohere `command-a-03-2025` (temperature 0.2 for answers, 0 for routing and grading)
- Embeddings: Cohere `embed-english-v3.0`

## Framework

- LangGraph and LangChain
- Chroma vector store, persisted in `vectorstore_db/`
- Tavily for web search
- Fixed-size chunking: 800 characters, 150 overlap, top 4 chunks (starting values, not tuned)