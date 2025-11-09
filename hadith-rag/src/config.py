"""
Configuration settings for the Hadith RAG pipeline.
Centralizes all model configurations and system settings.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class RAGConfig(BaseSettings):
    """Configuration settings for the RAG pipeline."""
    
    # Directory paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    STORAGE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "storage")
    
    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "qwen3-embedding:4b"
    LLM_MODEL: str = "qwen2.5:7b"
    
    # Qdrant settings
    QDRANT_URL: Optional[str] = None  # Use None for in-memory
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "hadith_collection"
    
    # OpenAI settings (fallback)
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # Chunking settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    SEMANTIC_SPLITTER_BREAKPOINT_PERCENTILE_THRESHOLD: int = 90  # More sensitive for Hadith boundaries
    
    # Retrieval settings
    SIMILARITY_TOP_K: int = 5
    SENTENCE_WINDOW_SIZE: int = 3  # Context sentences around each chunk
    
    # System settings
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 1000
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global config instance
config = RAGConfig()


def get_config() -> RAGConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> RAGConfig:
    """Update configuration with new values."""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config