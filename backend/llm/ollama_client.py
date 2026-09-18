import json
import logging
from typing import AsyncIterator, Dict, Any, List, Optional
import httpx
from backend.config import settings
from backend.llm.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class OllamaClient:
    """Async and sync HTTP client for communicating with local Ollama instance."""

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.LLM_MODEL

    async def check_health(self) -> Dict[str, Any]:
        """Checks if Ollama is reachable and if the target LLM model is available."""
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    data = res.json()
                    models = [m["name"] for m in data.get("models", [])]
                    target_available = any(self.model in m for m in models)
                    return {
                        "status": "online",
                        "base_url": self.base_url,
                        "configured_model": self.model,
                        "model_installed": target_available,
                        "available_models": models
                    }
                return {
                    "status": "error",
                    "code": res.status_code,
                    "message": f"Ollama returned HTTP {res.status_code}"
                }
        except Exception as e:
            return {
                "status": "offline",
                "base_url": self.base_url,
                "configured_model": self.model,
                "model_installed": False,
                "error": str(e),
                "instructions": (
                    "Ollama is not running. Please start Ollama by opening the Ollama application "
                    f"or running 'ollama serve', and ensure '{self.model}' is installed via 'ollama pull {self.model}'."
                )
            }

    async def generate_stream(
        self,
        prompt: str,
        system: str = SYSTEM_PROMPT,
        temperature: float = 0.1
    ) -> AsyncIterator[str]:
        """Streams response tokens from Ollama generate endpoint."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": 4096
            }
        }

        async with httpx.AsyncClient(timeout=90.0) as client:
            try:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        yield f"Error from Ollama ({response.status_code}): {await response.aread()}"
                        return

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk_data = json.loads(line)
                            token = chunk_data.get("response", "")
                            if token:
                                yield token
                            if chunk_data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                yield f"\n\n[Connection Error: Failed to communicate with Ollama at {self.base_url}. Details: {str(e)}]"

    async def generate(
        self,
        prompt: str,
        system: str = SYSTEM_PROMPT,
        temperature: float = 0.1
    ) -> str:
        """Non-streaming generation from Ollama, implemented via streaming accumulation to prevent HTTP timeouts."""
        tokens = []
        async for token in self.generate_stream(prompt, system=system, temperature=temperature):
            tokens.append(token)
        return "".join(tokens)

ollama_client = OllamaClient()
