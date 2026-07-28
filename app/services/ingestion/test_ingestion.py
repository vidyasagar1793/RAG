import asyncio
from pathlib import Path
from parsers import PDFParser
from chunker import TokenAwareChunker

def run_test():
    # 1. Create a dummy PDF to test with (if you don't have one handy)
    test_pdf_path = Path("sample_test.pdf")
    if not test_pdf_path.exists():
        import pymupdf  # The new correct import name!
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), "This is a test document for our RAG system.\n" * 20)
        doc.save(test_pdf_path)
        doc.close()
        print(f"Created temporary test PDF at {test_pdf_path}")

    print("\n--- STEP 1: Testing PDF Parser ---")
    pages = PDFParser.parse(test_pdf_path)
    print(f"Successfully extracted {len(pages)} pages.")
    for p in pages:
        print(f"Page {p.page_number} length: {len(p.text)} characters. Metadata: {p.metadata}")

    print("\n--- STEP 2: Testing Token-Aware Chunker ---")
    # Using a small chunk size of 20 tokens to force it to split our small test document
    chunker = TokenAwareChunker(chunk_size=20, chunk_overlap=5)
    
    # We pass a fake database UUID for the document_id
    chunks = chunker.process_document(pages, document_id="uuid-test-1234")
    
    print(f"Successfully created {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks[:3]):  # Just print the first 3 chunks
        print(f"\nChunk {i}:")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Metadata: {chunk.metadata}")
        print(f"  Text: {chunk.text!r}")

if __name__ == "__main__":
    run_test()