import logging
from pathlib import Path
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update

from app.services.ingestion.parsers import PDFParser
from app.services.ingestion.chunker import TokenAwareChunker
from embeddings import GitHubEmbeddingService

logger = logging.getLogger(__name__)

class IngestionPipeline:
    def __init__(
        self, 
        db_session: AsyncSession, 
        qdrant_client: AsyncQdrantClient,
        collection_name: str = "document_chunks"
    ):
        self.db = db_session
        self.qdrant = qdrant_client
        self.collection_name = collection_name
        
        self.chunker = TokenAwareChunker(chunk_size=500, chunk_overlap=50)
        self.embedding_service = GitHubEmbeddingService()

    async def process_document(self, document_id: str, file_path: Path):
        """
        Full ingestion pipeline:
        1. Update Postgres state to PROCESSING
        """
        try:
            logger.info(f"Starting ingestion for document {document_id}")
            
            # 1. Update Postgres state to PROCESSING
            await self._update_doc_status(document_id, "PROCESSING")

            # 2. Parse File (Assuming PDF for now; you can add a router here later)
            pages = PDFParser.parse(file_path)

            # 3. Chunk Text
            chunks = self.chunker.process_document(pages, document_id)
            if not chunks:
                raise ValueError("No text extracted from document.")

            # 4. Generate Embeddings (Batch process in blocks of 100 to respect API limits)
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                texts = [c.text for c in batch]
                
                vectors = await self.embedding_service.generate_embeddings(texts)
                
                # 5. Push batch to Qdrant
                points = [
                    models.PointStruct(
                        id=chunk.id, 
                        vector=vector, 
                        payload={
                            "text": chunk.text,
                            **chunk.metadata
                        }
                    )
                    for chunk, vector in zip(batch, vectors)
                ]
                
                await self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True  # Ensure write is confirmed
                )

            # 6. Mark as COMPLETED in Postgres
            await self._update_doc_status(document_id, "COMPLETED", total_chunks=len(chunks))
            logger.info(f"Successfully processed and stored {len(chunks)} chunks for {document_id}")

        except Exception as e:
            logger.error(f"Pipeline failed for {document_id}: {str(e)}")
            # Mark as FAILED so the UI knows
            await self._update_doc_status(document_id, "FAILED")
            raise

    async def _update_doc_status(self, doc_id: str, status: str, total_chunks: int = 0):
        """Helper to safely update Postgres state"""
        stmt = (
            update(DocumentModel)
            .where(DocumentModel.id == doc_id)
            .values(status=status, total_chunks=total_chunks)
        )
        await self.db.execute(stmt)
        await self.db.commit()