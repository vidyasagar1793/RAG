building a RAG


Language/Framework: Python 3.12, FastAPI, Pydantic

AI Orchestration: Pure Python with LlamaIndex (avoid LangChain bloat)

Models: OpenAI (GPT-4o) for generation, OpenAI text-embedding-3-small for embeddings

Vector Database: Qdrant (running via Docker)

Reranker: BGE Reranker (via HuggingFace)

Database (Metadata/State): PostgreSQL

Observability & Eval: Arize Phoenix (or Langfuse) for tracing, Ragas for evaluation

Infrastructure: Docker, Docker Compose, Poetry (dependency management), Pytest