"""
Document ingestion service for parsing and chunking
"""
import logging
from typing import List, Dict, Any
from uuid import UUID, uuid4
import io

from pypdf import PdfReader
from docx import Document
import openpyxl
import tiktoken

from app.config import settings

logger = logging.getLogger(__name__)


class IngestionService:
    """Service for document parsing and intelligent chunking"""
    
    def __init__(self):
        # Initialize tiktoken for accurate token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    async def process_document(
        self,
        content: bytes,
        filename: str,
        file_type: str,
        resource_id: UUID,
        workspace_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Process a document into chunks
        
        Args:
            content: Raw file content
            filename: Original filename
            file_type: File extension
            resource_id: Resource UUID
            workspace_id: Workspace UUID
            
        Returns:
            List of chunk dictionaries
        """
        logger.info(f"Processing document: {filename} ({file_type})")
        
        # Extract text based on file type
        if file_type == "pdf":
            text = self._extract_pdf(content)
        elif file_type in ["docx", "doc"]:
            text = self._extract_docx(content)
        elif file_type in ["xlsx", "xls"]:
            text = self._extract_xlsx(content)
        elif file_type in ["txt", "md"]:
            text = content.decode("utf-8", errors="ignore")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into chunks
        chunks = self._create_chunks(
            text=text,
            resource_id=resource_id,
            workspace_id=workspace_id,
            metadata={"filename": filename, "file_type": file_type}
        )
        
        logger.info(f"Created {len(chunks)} chunks from {filename}")
        
        return chunks
    
    def _extract_pdf(self, content: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf_file = io.BytesIO(content)
            reader = PdfReader(pdf_file)
            
            text_parts = []
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    # Add page marker for better context
                    text_parts.append(f"[Page {page_num}]\n{text}")
            
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
    
    def _extract_docx(self, content: bytes) -> str:
        """Extract text from DOCX"""
        try:
            docx_file = io.BytesIO(content)
            document = Document(docx_file)
            
            text_parts = []
            for para in document.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
            
            # Also extract text from tables
            for table in document.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        text_parts.append(row_text)
            
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise
    
    def _extract_xlsx(self, content: bytes) -> str:
        """Extract text from XLSX"""
        try:
            xlsx_file = io.BytesIO(content)
            workbook = openpyxl.load_workbook(xlsx_file, data_only=True)
            
            text_parts = []
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text_parts.append(f"[Sheet: {sheet_name}]")
                
                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join(str(cell) for cell in row if cell is not None)
                    if row_text.strip():
                        text_parts.append(row_text)
            
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"XLSX extraction failed: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Replace multiple newlines with double newline
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove special characters that might interfere with processing
        # (but preserve important punctuation)
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    def _create_chunks(
        self,
        text: str,
        resource_id: UUID,
        workspace_id: UUID,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create intelligent chunks from text
        
        Uses tiktoken for accurate token counting and respects:
        - Paragraph boundaries
        - Sentence boundaries
        - Section headers
        - Semantic coherence
        """
        chunks = []
        
        # Split into paragraphs first
        paragraphs = text.split("\n\n")
        
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Count tokens in paragraph
            para_tokens = len(self.tokenizer.encode(para))
            
            # If single paragraph exceeds chunk size, split it
            if para_tokens > settings.DEFAULT_CHUNK_SIZE:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        content=" ".join(current_chunk),
                        chunk_index=chunk_index,
                        resource_id=resource_id,
                        workspace_id=workspace_id,
                        metadata=metadata
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0
                
                # Split large paragraph by sentences
                sentences = self._split_into_sentences(para)
                for sentence in sentences:
                    sentence_tokens = len(self.tokenizer.encode(sentence))
                    
                    if current_tokens + sentence_tokens > settings.DEFAULT_CHUNK_SIZE:
                        if current_chunk:
                            chunks.append(self._create_chunk_dict(
                                content=" ".join(current_chunk),
                                chunk_index=chunk_index,
                                resource_id=resource_id,
                                workspace_id=workspace_id,
                                metadata=metadata
                            ))
                            chunk_index += 1
                        
                        # Start new chunk with overlap
                        if len(current_chunk) > 1:
                            overlap_tokens = 0
                            overlap_sentences = []
                            for s in reversed(current_chunk):
                                s_tokens = len(self.tokenizer.encode(s))
                                if overlap_tokens + s_tokens <= settings.CHUNK_OVERLAP:
                                    overlap_sentences.insert(0, s)
                                    overlap_tokens += s_tokens
                                else:
                                    break
                            current_chunk = overlap_sentences
                            current_tokens = overlap_tokens
                        else:
                            current_chunk = []
                            current_tokens = 0
                    
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
            
            # Normal case: add paragraph to current chunk
            elif current_tokens + para_tokens > settings.DEFAULT_CHUNK_SIZE:
                # Save current chunk
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        content=" ".join(current_chunk),
                        chunk_index=chunk_index,
                        resource_id=resource_id,
                        workspace_id=workspace_id,
                        metadata=metadata
                    ))
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = " ".join(current_chunk[-2:]) if len(current_chunk) >= 2 else ""
                overlap_tokens = len(self.tokenizer.encode(overlap_text))
                
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_tokens = overlap_tokens + para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk_dict(
                content=" ".join(current_chunk),
                chunk_index=chunk_index,
                resource_id=resource_id,
                workspace_id=workspace_id,
                metadata=metadata
            ))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        
        # Simple sentence splitter
        # In production, consider using spaCy or nltk for better accuracy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunk_dict(
        self,
        content: str,
        chunk_index: int,
        resource_id: UUID,
        workspace_id: UUID,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a chunk dictionary"""
        token_count = len(self.tokenizer.encode(content))
        
        return {
            "id": uuid4(),
            "content": content,
            "chunk_index": chunk_index,
            "token_count": token_count,
            "resource_id": resource_id,
            "workspace_id": workspace_id,
            "metadata": {
                **metadata,
                "chunk_index": chunk_index,
                "token_count": token_count
            }
        }