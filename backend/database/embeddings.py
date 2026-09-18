import logging
import httpx
from typing import List, Union
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from backend.config import settings

logger = logging.getLogger(__name__)

class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    ChromaDB compatible embedding function that queries the local Ollama instance.
    Falls back gracefully to ChromaDB's default ONNX embedding if Ollama embedding is unavailable.
    """

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.EMBEDDING_MODEL
        self._fallback_function = None
        self._checked_model = False
        self._use_fallback = False

    def name(self) -> str:
        return f"ollama_{self.model}"

    def _get_fallback(self):
        if self._fallback_function is None:
            logger.warning("Initializing local ONNX fallback embedding function...")
            from chromadb.utils import embedding_functions
            self._fallback_function = embedding_functions.DefaultEmbeddingFunction()
        return self._fallback_function

    def __call__(self, input: Documents) -> Embeddings:
        if self._use_fallback or settings.EMBEDDING_PROVIDER != "ollama":
            return self._get_fallback()(input)

        if not input:
            return []

        url = f"{self.base_url}/api/embed"
        payload = {"model": self.model, "input": input}

        try:
            with httpx.Client(timeout=45.0) as client:
                response = client.post(url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    if "embeddings" in data:
                        return data["embeddings"]

                # Fallback to single-call endpoint if /api/embed failed
                embeddings = []
                for text in input:
                    single_url = f"{self.base_url}/api/embeddings"
                    single_payload = {"model": self.model, "prompt": text}
                    res = client.post(single_url, json=single_payload, timeout=30.0)
                    if res.status_code != 200:
                        raise RuntimeError(f"Ollama embedding error ({res.status_code}): {res.text}")
                    embeddings.append(res.json()["embedding"])
                return embeddings
        except Exception as e:
            logger.error(
                f"Failed to generate embedding with Ollama model '{self.model}': {e}. "
                f"Switching to built-in ONNX fallback embedding function."
            )
            self._use_fallback = True
            return self._get_fallback()(input)

def get_embedding_function() -> EmbeddingFunction[Documents]:
    """Factory to get configured embedding function."""
    if settings.EMBEDDING_PROVIDER == "chroma_default":
        from chromadb.utils import embedding_functions
        return embedding_functions.DefaultEmbeddingFunction()
    return OllamaEmbeddingFunction(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.EMBEDDING_MODEL
    )
