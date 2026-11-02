import httpx
from typing import List
import logging

logger = logging.getLogger(__name__)

class OllamaEmbeddingService:
    """Asynchronous client for generating text embeddings locally via Ollama."""

    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/embed"

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Sends a batch of text chunks to Ollama and returns their dense vector embeddings.
        """
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": self.model_name,
                "input": texts
            }
            try:
                response = await client.post(self.endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
                
                embeddings = data.get("embeddings")
                if not embeddings:
                    raise ValueError("Ollama response did not contain 'embeddings'.")
                
                return embeddings
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error during embedding generation: {e.response.text}")
                raise RuntimeError(f"Ollama embedding failed: {e.response.status_code}") from e
            except Exception as e:
                logger.error(f"Unexpected error connecting to Ollama: {str(e)}")
                raise RuntimeError(f"Ollama connection error: {str(e)}") from e