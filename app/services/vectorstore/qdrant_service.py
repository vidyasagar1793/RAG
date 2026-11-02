import logging
import uuid
from typing import List

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from app.services.ingestion.models import ProcessedChunk

logger = logging.getLogger(__name__)


class QdrantVectorService:
    """Handles batch upserting of chunks and metadata payloads into Qdrant."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str = "document_chunks",
    ) -> None:
        self.client = client
        self.collection_name = collection_name

    async def upsert_chunks(
        self,
        chunks: List[ProcessedChunk],
        embeddings: List[List[float]],
    ) -> None:
        """Construct Qdrant points and upsert them to the vector collection."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Mismatch between number of text chunks and generated embeddings."
            )

        points = []
        for chunk, vector in zip(chunks, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))
            payload = {
                "text": chunk.text,
                "token_count": chunk.token_count,
                **chunk.metadata,
            }
            points.append(
                models.PointStruct(id=point_id, vector=vector, payload=payload)
            )

        if not points:
            return

        try:
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info(
                "Successfully upserted %d chunks into Qdrant collection '%s'.",
                len(points),
                self.collection_name,
            )
        except Exception as exc:
            logger.exception("Failed to upsert points into Qdrant")
            raise RuntimeError(f"Qdrant storage error: {exc}") from exc
