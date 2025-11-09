"""
Custom Ollama embedding wrapper for Arabic text processing.
Implements the LlamaIndex BaseEmbedding interface for Ollama models.
"""

import asyncio
import json
import requests
from typing import Any, Dict, List, Optional
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from src.config import get_config


class OllamaEmbedding(BaseEmbedding):
    """Custom embedding model using Ollama server."""
    
    model_name: str = Field(description="Name of the Ollama embedding model")
    base_url: str = Field(description="Base URL of the Ollama server")
    _session: Optional[requests.Session] = PrivateAttr(default=None)
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama embedding model."""
        config = get_config()
        
        super().__init__(
            model_name=model_name or config.EMBEDDING_MODEL,
            base_url=base_url or config.OLLAMA_BASE_URL,
            **kwargs,
        )
        
        # Initialize session for connection pooling
        self._session = requests.Session()
        
        # Verify model availability
        self._verify_model()
    
    def _verify_model(self) -> None:
        """Verify that the embedding model is available on Ollama server."""
        try:
            response = self._session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models = response.json()
            available_models = [model["name"] for model in models.get("models", [])]
            
            if self.model_name not in available_models:
                raise ValueError(
                    f"Model '{self.model_name}' not found. Available models: {available_models}"
                )
                
        except requests.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to Ollama server at {self.base_url}: {e}"
            )
    
    def _get_embedding(self, text: str) -> Embedding:
        """Get embedding for a single text."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            
            response = self._session.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get("embedding")
            
            if not embedding:
                raise ValueError(f"No embedding returned for text: {text[:50]}...")
                
            return embedding
            
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to get embedding from Ollama: {e}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid response format from Ollama: {e}")
    
    def _get_text_embedding(self, text: str) -> Embedding:
        """Get embedding for text (required by BaseEmbedding)."""
        return self._get_embedding(text)
    
    def _get_query_embedding(self, query: str) -> Embedding:
        """Get embedding for query (required by BaseEmbedding)."""
        return self._get_embedding(query)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """Get embeddings for multiple texts."""
        embeddings = []
        
        # Process texts in batches to avoid overwhelming the server
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for text in batch:
                embedding = self._get_embedding(text)
                embeddings.append(embedding)
        
        return embeddings
    
    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Async get embedding for text."""
        # For now, use sync method in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embedding, text)
    
    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Async get embedding for query."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_query_embedding, query)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """Async get embeddings for multiple texts."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embeddings, texts)
    
    @classmethod
    def class_name(cls) -> str:
        """Return class name for serialization."""
        return "OllamaEmbedding"
    
    def __del__(self) -> None:
        """Clean up session on deletion."""
        if self._session:
            self._session.close()


def create_embedding_model(
    model_name: Optional[str] = None,
    base_url: Optional[str] = None
) -> OllamaEmbedding:
    """Factory function to create Ollama embedding model."""
    return OllamaEmbedding(
        model_name=model_name,
        base_url=base_url
    )