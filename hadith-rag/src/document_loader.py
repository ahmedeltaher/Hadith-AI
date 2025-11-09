"""
Document loader module for Arabic Hadith texts.
Supports loading .txt, .md, .json files with proper metadata extraction.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llama_index.core import Document
from llama_index.core.schema import MetadataMode
from src.config import get_config


class HadithDocumentLoader:
    """Document loader for Hadith texts from various file formats."""
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """Initialize the document loader.
        
        Args:
            data_dir: Directory containing the Hadith documents
        """
        config = get_config()
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
    
    def load_documents(
        self, 
        file_patterns: Optional[List[str]] = None
    ) -> List[Document]:
        """Load all documents from the data directory.
        
        Args:
            file_patterns: List of file patterns to match (e.g., ['*.txt', '*.json'])
                         If None, loads all supported formats.
        
        Returns:
            List of LlamaIndex Document objects with metadata
        """
        documents = []
        
        # Default patterns for supported file types
        if file_patterns is None:
            file_patterns = ["*.txt", "*.md", "*.json"]
        
        for pattern in file_patterns:
            for file_path in self.data_dir.rglob(pattern):
                if file_path.is_file():
                    try:
                        docs = self._load_single_file(file_path)
                        documents.extend(docs)
                        print(f"✓ Loaded {len(docs)} documents from {file_path.name}")
                    except Exception as e:
                        print(f"✗ Failed to load {file_path.name}: {e}")
        
        print(f"\nTotal documents loaded: {len(documents)}")
        return documents
    
    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Load documents from a single file.
        
        Args:
            file_path: Path to the file to load
        
        Returns:
            List of Document objects
        """
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.json':
            return self._load_json_file(file_path)
        elif file_extension in ['.txt', '.md']:
            return self._load_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _load_json_file(self, file_path: Path) -> List[Document]:
        """Load documents from JSON file.
        
        Expected JSON structure:
        - {"hadiths": [{"text": "...", "metadata": {...}}, ...]}
        - [{"text": "...", "metadata": {...}}, ...]
        
        Args:
            file_path: Path to JSON file
        
        Returns:
            List of Document objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        base_metadata = self._create_base_metadata(file_path)
        
        # Handle different JSON structures
        if isinstance(data, dict) and "hadiths" in data:
            # Structure: {"hadiths": [...]}
            hadith_list = data["hadiths"]
        elif isinstance(data, list):
            # Structure: [{"text": "...", ...}, ...]
            hadith_list = data
        else:
            raise ValueError(f"Unsupported JSON structure in {file_path}")
        
        for i, hadith_data in enumerate(hadith_list):
            if isinstance(hadith_data, dict):
                # Extract text content
                text = (
                    hadith_data.get("text") or 
                    hadith_data.get("hadith") or 
                    hadith_data.get("content") or
                    str(hadith_data)
                )
                
                # Create document metadata
                doc_metadata = base_metadata.copy()
                doc_metadata.update({
                    "hadith_index": i,
                    "source_type": "json_hadith",
                })
                
                # Add any additional metadata from the hadith object
                for key, value in hadith_data.items():
                    if key not in ["text", "hadith", "content"] and value is not None:
                        doc_metadata[key] = str(value)
                
                document = Document(
                    text=text,
                    metadata=doc_metadata,
                    id_=f"{file_path.stem}_{i}"
                )
                documents.append(document)
        
        return documents
    
    def _load_text_file(self, file_path: Path) -> List[Document]:
        """Load document from text/markdown file.
        
        Args:
            file_path: Path to text file
        
        Returns:
            List containing single Document object
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines to separate individual hadiths if present
        sections = [section.strip() for section in content.split('\n\n') if section.strip()]
        
        documents = []
        base_metadata = self._create_base_metadata(file_path)
        
        if len(sections) == 1:
            # Single document
            document = Document(
                text=sections[0],
                metadata=base_metadata,
                id_=file_path.stem
            )
            documents.append(document)
        else:
            # Multiple sections
            for i, section in enumerate(sections):
                doc_metadata = base_metadata.copy()
                doc_metadata.update({
                    "section_index": i,
                    "source_type": "text_section",
                })
                
                document = Document(
                    text=section,
                    metadata=doc_metadata,
                    id_=f"{file_path.stem}_{i}"
                )
                documents.append(document)
        
        return documents
    
    def _create_base_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Create base metadata for a file.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Dictionary with base metadata
        """
        # Try to extract collection/book information from path
        path_parts = file_path.parts
        collection = "unknown"
        book = "unknown"
        
        # Look for common Hadith collection names in path
        hadith_collections = [
            "bukhari", "muslim", "abu_dawud", "tirmidhi", 
            "nasai", "ibn_majah", "malik", "ahmad"
        ]
        
        for part in path_parts:
            part_lower = part.lower()
            for coll in hadith_collections:
                if coll in part_lower:
                    collection = coll
                    break
        
        # Extract book name (usually the filename without extension)
        book = file_path.stem
        
        return {
            "filename": file_path.name,
            "collection": collection,
            "book": book,
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "source_type": "file"
        }
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about loaded documents.
        
        Args:
            documents: List of Document objects
        
        Returns:
            Dictionary with statistics
        """
        if not documents:
            return {"total_documents": 0}
        
        total_chars = sum(len(doc.text) for doc in documents)
        collections = set(doc.metadata.get("collection", "unknown") for doc in documents)
        books = set(doc.metadata.get("book", "unknown") for doc in documents)
        
        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "average_chars_per_doc": total_chars // len(documents),
            "unique_collections": len(collections),
            "unique_books": len(books),
            "collections": sorted(list(collections)),
            "books": sorted(list(books))
        }


def load_hadith_documents(
    data_dir: Optional[Union[str, Path]] = None,
    file_patterns: Optional[List[str]] = None
) -> List[Document]:
    """Convenience function to load Hadith documents.
    
    Args:
        data_dir: Directory containing documents
        file_patterns: File patterns to match
    
    Returns:
        List of loaded Document objects
    """
    loader = HadithDocumentLoader(data_dir)
    return loader.load_documents(file_patterns)