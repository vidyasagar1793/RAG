import os
import httpx
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, List
import logging

from dotenv import dotenv_values
from ollama import Client

logger = logging.getLogger(__name__)


class OllamaCloudChatClient:
    """Streaming chat client for models hosted by Ollama Cloud."""

    def __init__(
        self,
        model_name: str = "gpt-oss:120b",
        api_key: str | None = None,
        host: str = "https://ollama.com",
    ):
        self.model_name = model_name
        api_key = api_key or os.getenv("OLLAMA_API_KEY")
        if not api_key:
            raise ValueError("OLLAMA_API_KEY must be set to use Ollama Cloud.")

        self.client = Client(
            host=host,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def stream_chat(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> Iterator[str]:
        """Yield response text chunks from the configured Ollama Cloud model."""
        try:
            for part in self.client.chat(
                model=self.model_name,
                messages=list(messages),
                stream=True,
            ):
                content = part.message.content
                if content:
                    yield content
        except Exception as exc:
            logger.exception("Ollama Cloud chat request failed")
            raise RuntimeError("Ollama Cloud chat request failed") from exc

class OllamaEmbeddingService:
    """Asynchronous client for generating text embeddings through Ollama."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "https://ollama.com",
        api_key: str | None = None,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/api/embed"
        env_api_key = dotenv_values(".env").get("OLLAMA_API_KEY")
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY") or env_api_key

        if self.base_url == "https://ollama.com" and not self.api_key:
            raise ValueError("OLLAMA_API_KEY must be set to use Ollama Cloud.")

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Sends a batch of text chunks to Ollama and returns their dense vector embeddings.
        """
        headers = (
            {"Authorization": f"Bearer {self.api_key}"}
            if self.api_key
            else None
        )
        async with httpx.AsyncClient(timeout=60.0, headers=headers) as client:
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
