"""
Arabic Hadith RAG Pipeline

A comprehensive Retrieval-Augmented Generation system for Arabic Hadith texts
using LlamaIndex, Ollama, and Qdrant vector store.

Modules:
- config: Configuration management
- embeddings: Custom Ollama embedding wrapper
- document_loader: Document loading and processing
- index_builder: Vector index creation and management
- query_engine: Query processing and response generation
"""

__version__ = "1.0.0"
__author__ = "Arabic Hadith RAG Pipeline"

from .config import get_config, update_config
from .embeddings import create_embedding_model, OllamaEmbedding
from .document_loader import HadithDocumentLoader, load_hadith_documents
from .index_builder import HadithIndexBuilder, build_hadith_index
from .query_engine import HadithQueryEngine, create_query_engine

__all__ = [
    "get_config",
    "update_config", 
    "create_embedding_model",
    "OllamaEmbedding",
    "HadithDocumentLoader",
    "load_hadith_documents",
    "HadithIndexBuilder", 
    "build_hadith_index",
    "HadithQueryEngine",
    "create_query_engine",
]