import logging
from pathlib import Path
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ingestion.parsers import PDFParser
from app.services.ingestion.chunker import RecursiveTokenChunker
from app.services.embedding.ollama import OllamaEmbeddingService
from app.services.vectorstore.qdrant_service import QdrantVectorService

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
        
        self.chunker = RecursiveTokenChunker(
            model_name="nomic-embed-text",
            chunk_size=500,
            chunk_overlap=50,
        )
        self.embedding_service = OllamaEmbeddingService()
        self.vector_service = QdrantVectorService(
            client=qdrant_client,
            collection_name=collection_name,
        )

    async def run(self, file_path: str, document_id: str, original_filename: str):
        return await self.process_document(
            document_id=document_id,
            file_path=Path(file_path),
            original_filename=original_filename,
        )

    async def process_document(
        self,
        document_id: str,
        file_path: Path,
        original_filename: str | None = None,
    ):
        """
        Full ingestion pipeline:
        1. Update Postgres state to PROCESSING
        """
        try:
            logger.info(f"Starting ingestion for document {document_id}")
            
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
                
                # 5. Push vectors and metadata to Qdrant.
                await self.vector_service.upsert_chunks(batch, vectors)

            logger.info(f"Successfully processed and stored {len(chunks)} chunks for {document_id}")
            return {
                "document_id": document_id,
                "filename": original_filename or file_path.name,
                "page_count": len(pages),
                "chunk_count": len(chunks),
            }

        except Exception as e:
            logger.error(f"Pipeline failed for {document_id}: {str(e)}")
            raise


async def process_and_index_document(
    file_path: Path,
    document_id: str,
    qdrant_client: AsyncQdrantClient,
    model_name: str = "nomic-embed-text",
) -> None:
    """Parse, chunk, embed with Ollama, and store a document in Qdrant."""
    logger.info("Starting processing pipeline for document ID: %s", document_id)

    extracted_pages = PDFParser.parse(file_path)
    chunker = RecursiveTokenChunker(
        model_name=model_name,
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = chunker.process_document(extracted_pages, document_id=document_id)

    if not chunks:
        logger.warning("No text chunks found to process for document %s", document_id)
        return

    embedding_service = OllamaEmbeddingService(model_name=model_name)
    embeddings = await embedding_service.generate_embeddings(
        [chunk.text for chunk in chunks]
    )

    vector_service = QdrantVectorService(client=qdrant_client)
    await vector_service.upsert_chunks(chunks, embeddings)

    logger.info("Document %s successfully indexed into Qdrant", document_id)
