"""
Central configuration for the app.

Why this exists (Day 1 concept):
Hard-coding a database URL or API key directly in code is a common
enterprise anti-pattern — you already know this from Checkmarx/SonarQube
findings. Pydantic's BaseSettings reads from environment variables (or a
.env file locally), validates types, and gives the rest of the app a single
typed object to import instead of scattering os.environ.get() calls
everywhere.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_name: str = "Enterprise RAG Platform"
    environment: str = "development"

    # Database (Postgres + pgvector)
    database_url: str = (
        "postgresql+asyncpg://rag_user:rag_password@db:5432/rag_platform"
    )

    # Embeddings — dimension must match whatever embedding model we pick
    # in Day 2 (e.g. 384 for a small sentence-transformers model,
    # 1536 for OpenAI text-embedding-3-small). Kept configurable on
    # purpose — see project spec, section 5: "don't hard-code K/N values."
    embedding_dimension: int = 384

    # Retrieval defaults (also configurable, not hard-coded — Day 4+)
    retrieval_top_k: int = 10
    rerank_top_n: int = 5
    rrf_k: int = 60  # Reciprocal Rank Fusion constant — see hybrid_search.py

    # Chunking strategy — swappable, same pattern as LLM_PROVIDER.
    # "semantic" (default) groups sentences by embedding-similarity
    # topic shifts, not fixed word counts — see semantic_chunker.py.
    # "fixed" is the original Day 2 word-count splitter, kept available
    # for comparison/rollback.
    chunking_strategy: str = "semantic"
    semantic_similarity_threshold: float = 0.55
    semantic_chunk_max_words: int = 300
    semantic_chunk_min_words: int = 50

    # Agentic retrieval (Day 5) — the iteration cap is not optional to
    # set; every agent loop needs one, or a persistently weak query would
    # retry forever. See agentic_retrieval.py.
    max_retrieval_attempts: int = 2
    min_rerank_score: float = 0.5  # sigmoid-scaled, 0-1 — see reranker.py

    # Query expansion (v2 follow-up) — the fix documented in the README's
    # Known Limitations for the corpus-imbalance problem: search with a
    # few paraphrased variants per attempt, not just the literal query,
    # so a numerically small but relevant document class isn't
    # systematically out-competed by a larger one under one fixed
    # phrasing. Proactive (every attempt), unlike Day 5's reformulation
    # (reactive, only after a failed attempt) — see query_expansion.py.
    query_expansion_enabled: bool = True
    query_expansion_variants: int = 2

    # LLM (Day 3+) — provider is swappable via LLM_PROVIDER.
    # "ollama" = free, local, runs on your machine, no API key needed.
    # "groq" = free, cloud, no local compute needed — the right choice
    #          for a public deployment where Ollama can't run (see
    #          README Deployment section).
    # "anthropic" = real Claude, requires ANTHROPIC_API_KEY + billing.
    llm_provider: str = "ollama"
    llm_max_tokens: int = 1024

    # Ollama (local, free)
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3.1:8b"

    # Groq (cloud, free — no credit card required)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Anthropic (cloud, paid)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"


settings = Settings()
