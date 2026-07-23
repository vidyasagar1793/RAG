import tiktoken
from typing import List, Dict, Any
from models import ExtractedPage, ProcessedChunk

class TokenAwareChunker:
    """Splits document pages into token-bounded, overlapping chunks."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Get the correct BPE tokenizer for the targeted OpenAI embedding model
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def chunk_page(self, page: ExtractedPage, document_id: str) -> List[ProcessedChunk]:
        """Splits an ExtractedPage into chunks respecting token bounds."""
        tokens = self.tokenizer.encode(page.text)
        total_tokens = len(tokens)
        
        if total_tokens <= self.chunk_size:
            # Entire page fits inside a single chunk
            return [
                ProcessedChunk(
                    text=page.text,
                    token_count=total_tokens,
                    metadata={
                        **page.metadata,
                        "document_id": document_id,
                        "page_number": page.page_number,
                        "chunk_index": 0
                    }
                )
            ]

        chunks: List[ProcessedChunk] = []
        start_idx = 0
        chunk_idx = 0

        # Sliding window across token sequence with overlap
        while start_idx < total_tokens:
            end_idx = min(start_idx + self.chunk_size, total_tokens)
            
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append(
                ProcessedChunk(
                    text=chunk_text,
                    token_count=len(chunk_tokens),
                    metadata={
                        **page.metadata,
                        "document_id": document_id,
                        "page_number": page.page_number,
                        "chunk_index": chunk_idx
                    }
                )
            )

            # Move window forward by chunk_size minus overlap
            start_idx += (self.chunk_size - self.chunk_overlap)
            chunk_idx += 1

        return chunks

    def process_document(self, pages: List[ExtractedPage], document_id: str) -> List[ProcessedChunk]:
        """Processes all pages of a document into a flat list of chunks."""
        all_chunks: List[ProcessedChunk] = []
        for page in pages:
            page_chunks = self.chunk_page(page, document_id=document_id)
            all_chunks.extend(page_chunks)
        return all_chunks