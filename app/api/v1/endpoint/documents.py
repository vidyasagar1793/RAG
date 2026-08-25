import logging
from fastapi import APIRouter, Depends, Query, HTTPException, status
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from typing import Optional

# Adjust these imports based on your exact internal routing as seen in image_53ab20.png
from app.db.qdrant import get_qdrant_client
from app.services.ingestion.embeddings import GitHubEmbeddingService
from app.schemas.search import SearchResponse, SearchResultItem

router = APIRouter()
logger = logging.getLogger(__name__)

# Assuming you have a dependency to provide the embedding service
def get_embedding_service() -> GitHubEmbeddingService:
    return GitHubEmbeddingService()

@router.get("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search_documents(
    q: str = Query(..., min_length=2, description="The semantic search query"),
    limit: int = Query(5, ge=1, le=50, description="Maximum number of results to return"),
    file_name: Optional[str] = Query(None, description="Optional exact file name to filter by"),
    qdrant_client: AsyncQdrantClient = Depends(get_qdrant_client),
    embedding_service: GitHubEmbeddingService = Depends(get_embedding_service)
):
    """
    Perform a semantic search over ingested document chunks.
    Allows optional metadata filtering by file_name.
    """
    try:
        # 1. Generate embeddings for the search query
        # We pass [q] because generate_embeddings expects a List[str]
        query_vectors = await embedding_service.generate_embeddings([q])
        
        if not query_vectors or len(query_vectors) == 0:
            raise ValueError("Embedding service failed to return a vector.")
            
        query_vector = query_vectors[0]

        # 2. Construct optional Qdrant metadata filters
        query_filter = None
        if file_name:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="file_name",
                        match=qmodels.MatchValue(value=file_name)
                    )
                ]
            )

        # 3. Execute Async Search against Qdrant
        search_results = await qdrant_client.search(
            collection_name="document_chunks",
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )

        # 4. Map Qdrant ScoredPoints to Pydantic Response schema
        results = []
        for point in search_results:
            payload = point.payload or {}
            results.append(
                SearchResultItem(
                    text=payload.get("text", ""),
                    score=point.score,
                    file_name=payload.get("file_name", "unknown"),
                    page_number=payload.get("page_number", 0),
                    document_id=payload.get("document_id", ""),
                    chunk_index=payload.get("chunk_index", 0)
                )
            )

        return SearchResponse(query=q, results=results)

    except ValueError as ve:
        logger.error(f"Validation/Embedding error during search: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request or embedding generation failed."
        )
    except Exception as e:
        logger.error(f"Search API encountered an unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while searching documents."
        )