import tiktoken
import logging
from typing import List
from app.services.ingestion.models import ExtractedPage, ProcessedChunk

logger = logging.getLogger(__name__)

class RecursiveTokenChunker:
    """
    Recursively splits text by natural boundaries (paragraphs, sentences) 
    while strictly enforcing a maximum token limit and overlap.
    """
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        
        # Default boundary hierarchy: Paragraphs -> Sentences -> Words -> Characters
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively splits the text using the highest-priority separator available.
        """
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        # Find the highest priority separator that actually exists in the text
        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1:]
                break

        # Split the text
        splits = text.split(separator) if separator else list(text)

        # Re-attach the separator to preserve punctuation (except for the last split)
        good_splits = []
        for i, split in enumerate(splits):
            if i < len(splits) - 1 and separator:
                good_splits.append(split + separator)
            else:
                good_splits.append(split)

        # Recursively evaluate the splits
        for split in good_splits:
            if not split.strip():
                continue
            
            if self.count_tokens(split) <= self.chunk_size:
                final_chunks.append(split)
            elif new_separators:
                # If a piece is still too big, recurse deeper with the next separator
                final_chunks.extend(self._split_text(split, new_separators))
            else:
                # Absolute fallback: Brute-force slice by tokens if there are no spaces
                tokens = self.tokenizer.encode(split)
                for i in range(0, len(tokens), self.chunk_size):
                    chunk_tokens = tokens[i : i + self.chunk_size]
                    final_chunks.append(self.tokenizer.decode(chunk_tokens))

        return final_chunks

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Combines small text pieces together until they hit the chunk_size limit,
        and rewinds the buffer to create overlapping context windows.
        """
        docs = []
        current_doc = []
        total_tokens = 0

        for split in splits:
            split_tokens = self.count_tokens(split)
            
            # If adding this split exceeds the limit, flush the current document block
            if total_tokens + split_tokens > self.chunk_size and current_doc:
                merged_text = "".join(current_doc).strip()
                if merged_text:
                    docs.append(merged_text)
                
                # Backtrack to create the overlap
                while current_doc:
                    popped = current_doc.pop(0)
                    total_tokens -= self.count_tokens(popped)
                    # Stop dropping elements once we are within the target overlap size
                    if total_tokens <= self.chunk_overlap or not current_doc:
                        break
            
            current_doc.append(split)
            total_tokens += split_tokens
            
        # Flush the final document block
        if current_doc:
            merged_text = "".join(current_doc).strip()
            if merged_text:
                docs.append(merged_text)
                
        return docs

    def process_document(self, pages: List[ExtractedPage], document_id: str) -> List[ProcessedChunk]:
        """
        Main entry point. Replaces the old sliding window logic.
        """
        all_processed_chunks = []
        chunk_index = 0

        for page in pages:
            # 1. Break text strictly by natural boundaries
            raw_splits = self._split_text(page.text, self.separators)
            
            # 2. Intelligently merge them back together to fit the 500-token limit
            merged_texts = self._merge_splits(raw_splits)

            # 3. Format into the pipeline data model
            for text_chunk in merged_texts:
                all_processed_chunks.append(
                    ProcessedChunk(
                        text=text_chunk,
                        token_count=self.count_tokens(text_chunk),
                        metadata={
                            **page.metadata,
                            "document_id": document_id,
                            "page_number": page.page_number,
                            "chunk_index": chunk_index
                        }
                    )
                )
                chunk_index += 1

        logger.info(f"Generated {len(all_processed_chunks)} recursive chunks for doc {document_id}")
        return all_processed_chunks