import fitz  # PyMuPDF
from pathlib import Path
from typing import List
import logging
from models import ExtractedPage

logger = logging.getLogger(__name__)

class PDFParser:
    """Extracts text page-by-page from PDF files using PyMuPDF."""

    @staticmethod
    def parse(file_path: Path) -> List[ExtractedPage]:
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        pages: List[ExtractedPage] = []
        
        try:
            doc = fitz.open(file_path)
            for page_index in range(len(doc)):
                page = doc[page_index]
                # Extract clean text line-by-line
                raw_text = page.get_text("text")
                
                # Basic cleaning: strip null characters and excessive blank space
                cleaned_text = raw_text.replace("\x00", "").strip()
                
                if cleaned_text:
                    pages.append(
                        ExtractedPage(
                            text=cleaned_text,
                            page_number=page_index + 1,  # 1-indexed for human readability
                            metadata={
                                "total_pages": len(doc),
                                "file_name": file_path.name
                            }
                        )
                    )
            doc.close()
            logger.info(f"Successfully extracted {len(pages)} pages from {file_path.name}")
            return pages
            
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path.name}: {str(e)}")
            raise RuntimeError(f"PDF Parsing error: {str(e)}") from e