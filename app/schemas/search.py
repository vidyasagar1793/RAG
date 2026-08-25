from pydantic import BaseModel, Field
from typing import List, Optional

class SearchResultItem(BaseModel):
    text: str = Field(..., description="The content of the document chunk")
    score: float = Field(..., description="Cosine similarity score")
    file_name: str = Field(..., description="Source file name")
    page_number: int = Field(..., description="Page number where the chunk is found")
    document_id: str = Field(..., description="UUID of the parent document in Postgres")
    chunk_index: int = Field(..., description="Index of the chunk within the document")

class SearchResponse(BaseModel):
    query: str = Field(..., description="The original search query")
    results: List[SearchResultItem] = Field(..., description="Ranked list of matching chunks")