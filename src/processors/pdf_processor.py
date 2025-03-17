"""
PDF Processor module for extracting and chunking text from PDF files.
"""
import os
from typing import List, Dict, Any, Optional

from pypdf import PdfReader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


class PDFProcessor:
    """
    A class for processing PDF documents with semantic chunking capabilities.
    
    This class extracts text from PDF files and applies semantic chunking to
    create contextually meaningful document chunks for later use in RAG systems.
    """
    
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        Initialize the PDF processor with embedding model and chunking parameters.
        
        Args:
            embedding_model_name: Name of the Hugging Face model to use for embeddings
            chunk_size: Target chunk size for semantic chunking
            chunk_overlap: Overlap size between chunks to maintain context
        """
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
    def extract_text_from_pdf(self, 
                              pdf_path: str, 
                              start_page: int = 0, 
                              end_page: Optional[int] = None) -> str:
        """
        Extract text from a PDF file within specified page range.
        
        Args:
            pdf_path: Path to the PDF file
            start_page: First page to extract (0-indexed)
            end_page: Last page to extract (inclusive, 0-indexed)
                      If None, extracts until the end of the document
                      
        Returns:
            Extracted text as a single string
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
        
        pdf_reader = PdfReader(pdf_path)
        total_pages = len(pdf_reader.pages)
        
        # Validate page ranges
        if start_page < 0:
            start_page = 0
        if end_page is None or end_page >= total_pages:
            end_page = total_pages - 1
            
        extracted_text = ""
        for page_num in range(start_page, end_page + 1):
            page = pdf_reader.pages[page_num]
            extracted_text += page.extract_text() + "\n\n"
            
        return extracted_text
    
    def create_semantic_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into semantic chunks using embeddings-based chunking.
        
        Args:
            text: The raw text to be split into chunks
            
        Returns:
            List of documents with chunked text and metadata
        """
        # Use SemanticChunker with percentile breakpoint threshold (default)
        text_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95.0,  # Default percentile
            min_chunk_size=self.chunk_size // 2  # Minimum size as half of target chunk size
        )
        
        chunks = text_splitter.create_documents([text])
        
        # Add additional metadata for each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            
        return chunks
    
    def process_pdf(self, 
                    pdf_path: str, 
                    start_page: int = 0, 
                    end_page: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process a PDF file by extracting text and creating semantic chunks.
        
        Args:
            pdf_path: Path to the PDF file
            start_page: First page to extract (0-indexed)
            end_page: Last page to extract (inclusive, 0-indexed)
                      If None, extracts until the end of the document
                      
        Returns:
            List of semantically chunked documents with metadata
        """
        text = self.extract_text_from_pdf(pdf_path, start_page, end_page)
        chunks = self.create_semantic_chunks(text)
        
        # Add source information to metadata
        for chunk in chunks:
            chunk.metadata["source"] = pdf_path
            chunk.metadata["page_range"] = f"{start_page}-{end_page}"
            
        return chunks


if __name__ == "__main__":
    # Simple test to verify the processor works
    processor = PDFProcessor()
    pdf_path = os.path.join("data", "ISLRv2.pdf")
    
    # Process only the first few pages for testing
    chunks = processor.process_pdf(pdf_path, start_page=15, end_page=20)
    
    print(f"Extracted {len(chunks)} chunks from the PDF")
    print("\nSample chunk content:")
    if chunks:
        print(f"Chunk {chunks[0].metadata['chunk_id']}:")
        print(chunks[0].page_content[:300] + "...") 