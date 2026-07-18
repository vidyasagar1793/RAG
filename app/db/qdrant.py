from qdrant_client import AsyncQdrantClient
from app.core.config import settings

# Initialize the async client
qdrant = AsyncQdrantClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
)