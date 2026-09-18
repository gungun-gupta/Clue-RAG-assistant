import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2:3b"

    # Embedding Configuration
    EMBEDDING_PROVIDER: str = "ollama"  # "ollama" or "chroma_default"
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # Database & Storage paths
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma"
    DOCUMENTS_DIRECTORY: str = "./data/documents"

    # Hybrid Retrieval & Reranking Weights
    WEIGHT_SEMANTIC: float = 0.35
    WEIGHT_KEYWORD: float = 0.25
    WEIGHT_COMPLETENESS: float = 0.30
    WEIGHT_SECTION: float = 0.10
    COMPLETE_CODE_BOOST: float = 0.40
    MIN_RELEVANCE_THRESHOLD: float = 0.35

    # Token & Chunk Budgets
    MAX_CONTEXT_TOKENS: int = 2500
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 80
    TOP_K_RETRIEVAL: int = 15
    TOP_K_RERANKED: int = 5

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

settings = Settings()

# Ensure directories exist
Path(settings.CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path(settings.DOCUMENTS_DIRECTORY).mkdir(parents=True, exist_ok=True)
