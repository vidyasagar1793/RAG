import logging
from typing import List
from openai import AsyncOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

class GitHubEmbeddingService:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        # Route standard OpenAI SDK to GitHub Models endpoint
        self.client = AsyncOpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=settings.GITHUB_TOKEN,
        )

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            # Ensure vectors preserve exact order
            embeddings = [data.embedding for data in sorted(response.data, key=lambda x: x.index)]
            return embeddings

        except Exception as e:
            logger.error(f"GitHub Models Embedding failed: {str(e)}")
            raise RuntimeError("Failed to generate embeddings from GitHub Models") from e