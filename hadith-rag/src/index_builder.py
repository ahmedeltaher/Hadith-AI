"""
Index builder module for Arabic Hadith RAG pipeline.
Implements semantic chunking and sentence-window context expansion.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llama_index.core import (
    Document, 
    ServiceContext, 
    StorageContext, 
    VectorStoreIndex,
    Settings
)
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceWindowNodeParser
)
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from src.config import get_config
from src.embeddings import create_embedding_model
from src.document_loader import load_hadith_documents


class HadithIndexBuilder:
    """Builder for creating and managing Hadith document indices."""
    
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        use_sentence_window: bool = False,
        rebuild: bool = False
    ):
        """Initialize the index builder.
        
        Args:
            storage_dir: Directory to store the index
            use_sentence_window: Whether to apply sentence window context (semantic chunking is always used)
            rebuild: Whether to rebuild existing index
        """
        config = get_config()
        self.config = config
        self.storage_dir = Path(storage_dir) if storage_dir else config.STORAGE_DIR
        self.use_sentence_window = use_sentence_window
        self.rebuild = rebuild
        
        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embed_model = create_embedding_model()
        self.vector_store = self._setup_vector_store()
        self.node_parser = self._setup_node_parser()
        
        # Setup LlamaIndex settings
        Settings.embed_model = self.embed_model
        Settings.chunk_size = config.CHUNK_SIZE
        Settings.chunk_overlap = config.CHUNK_OVERLAP
    
    def _setup_vector_store(self) -> QdrantVectorStore:
        """Setup Qdrant vector store."""
        config = self.config
        
        # Initialize Qdrant client
        if config.QDRANT_URL:
            # Remote Qdrant instance
            client = QdrantClient(
                url=config.QDRANT_URL,
                port=config.QDRANT_PORT
            )
        else:
            # Local/in-memory Qdrant
            client = QdrantClient(":memory:")
        
        # Check if collection exists and handle rebuild
        collection_name = config.QDRANT_COLLECTION_NAME
        
        try:
            collections = client.get_collections()
            existing_collections = [c.name for c in collections.collections]
            
            if collection_name in existing_collections:
                if self.rebuild:
                    print(f"🗑️  Deleting existing collection: {collection_name}")
                    client.delete_collection(collection_name)
                else:
                    print(f"📚 Using existing collection: {collection_name}")
        except Exception as e:
            print(f"⚠️  Collection check failed: {e}")
        
        # Create vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name
        )
        
        return vector_store
    
    def _setup_node_parser(self) -> TransformComponent:
        """Setup node parser for document chunking with semantic awareness."""
        config = self.config
        
        # Always prioritize SemanticSplitterNodeParser for better Hadith boundaries
        # This ensures each Hadith becomes its own node based on semantic meaning
        node_parser = SemanticSplitterNodeParser(
            buffer_size=1,  # Small buffer for precise boundaries
            breakpoint_percentile_threshold=config.SEMANTIC_SPLITTER_BREAKPOINT_PERCENTILE_THRESHOLD,
            embed_model=self.embed_model,
        )
        print("📄 Using SemanticSplitterNodeParser for smart semantic chunking")
        print(f"   🎯 Breakpoint threshold: {config.SEMANTIC_SPLITTER_BREAKPOINT_PERCENTILE_THRESHOLD}%")
        print("   🧠 Smart semantic chunking to preserve meaning between Hadiths")
        
        # If sentence window is still needed, we can apply it as post-processing
        if self.use_sentence_window:
            print(f"   🪟 Sentence window context will be applied (size={config.SENTENCE_WINDOW_SIZE})")
        
        return node_parser
    
    def build_index(
        self,
        documents: Optional[List[Document]] = None,
        data_dir: Optional[Union[str, Path]] = None
    ) -> VectorStoreIndex:
        """Build or load the vector index.
        
        Args:
            documents: Pre-loaded documents (optional)
            data_dir: Directory to load documents from (if documents not provided)
        
        Returns:
            VectorStoreIndex instance
        """
        # Load documents if not provided
        if documents is None:
            print("📖 Loading documents...")
            documents = load_hadith_documents(data_dir or self.config.DATA_DIR)
            
            if not documents:
                raise ValueError("No documents found to index")
        
        print(f"📊 Processing {len(documents)} documents...")
        
        # Setup storage context
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Check if index already exists and not rebuilding
        index_path = self.storage_dir / "index"
        if index_path.exists() and not self.rebuild:
            print("📚 Loading existing index...")
            try:
                index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=storage_context,
                )
                print("✅ Successfully loaded existing index")
                return index
            except Exception as e:
                print(f"⚠️  Failed to load existing index: {e}")
                print("🔄 Building new index...")
        
        # Build new index
        print("🔨 Building new index...")
        
        # Transform documents to nodes
        nodes = self.node_parser.get_nodes_from_documents(
            documents, show_progress=True
        )
        
        print(f"📝 Created {len(nodes)} nodes from documents")
        
        # Create index
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        
        # Persist index
        try:
            index.storage_context.persist(persist_dir=str(self.storage_dir))
            print(f"💾 Index saved to {self.storage_dir}")
        except Exception as e:
            print(f"⚠️  Failed to persist index: {e}")
        
        return index
    
    def get_index_stats(self, index: VectorStoreIndex) -> Dict[str, Any]:
        """Get statistics about the built index.
        
        Args:
            index: The vector index
        
        Returns:
            Dictionary with index statistics
        """
        try:
            # Get nodes from the index
            retriever = index.as_retriever(similarity_top_k=1000)  # Get many to count
            nodes = retriever._get_nodes_with_embeddings([])  # This might not work in all versions
            
            # Alternative method: try to access vector store directly
            if hasattr(self.vector_store, 'client'):
                client = self.vector_store.client
                collection_info = client.get_collection(self.config.QDRANT_COLLECTION_NAME)
                
                return {
                    "total_nodes": collection_info.vectors_count,
                    "vector_size": collection_info.config.params.vectors.size,
                    "collection_name": self.config.QDRANT_COLLECTION_NAME,
                    "embedding_model": self.config.EMBEDDING_MODEL,
                    "node_parser_type": type(self.node_parser).__name__,
                    "sentence_window_enabled": self.use_sentence_window,
                }
        except Exception as e:
            print(f"⚠️  Could not get detailed index stats: {e}")
            return {
                "embedding_model": self.config.EMBEDDING_MODEL,
                "node_parser_type": type(self.node_parser).__name__,
                "sentence_window_enabled": self.use_sentence_window,
            }


def build_hadith_index(
    data_dir: Optional[Union[str, Path]] = None,
    storage_dir: Optional[Union[str, Path]] = None,
    use_sentence_window: bool = False,
    rebuild: bool = False
) -> VectorStoreIndex:
    """Convenience function to build Hadith index with semantic chunking.
    
    Args:
        data_dir: Directory containing documents
        storage_dir: Directory to store index
        use_sentence_window: Whether to apply sentence window context (semantic chunking always used)
        rebuild: Whether to rebuild existing index
    
    Returns:
        Built VectorStoreIndex with semantic chunking for optimal Hadith boundaries
    """
    builder = HadithIndexBuilder(
        storage_dir=storage_dir,
        use_sentence_window=use_sentence_window,
        rebuild=rebuild
    )
    
    return builder.build_index(data_dir=data_dir)