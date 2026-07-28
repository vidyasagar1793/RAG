import logging
from typing import AsyncGenerator

from qdrant_client import AsyncQdrantClient, models

from app.core.config import settings

logger = logging.getLogger(__name__)

qdrant_client = AsyncQdrantClient(
    url=f"http://{settings.qdrant_host}:{settings.qdrant_port}",
)


async def init_qdrant_collections() -> None:
    """Ensure required vector collections exist on startup."""
    collection_name = "document_chunks"

    exists = await qdrant_client.collection_exists(collection_name=collection_name)

    if not exists:
        logger.info("Creating Qdrant collection: '%s'...", collection_name)
        await qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1536,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info("Qdrant collection created successfully.")
    else:
        logger.info(
            "Qdrant collection '%s' already exists. Skipping creation.",
            collection_name,
        )


async def get_qdrant() -> AsyncGenerator[AsyncQdrantClient, None]:
    """FastAPI dependency injector for the Qdrant client."""
    yield qdrant_client
