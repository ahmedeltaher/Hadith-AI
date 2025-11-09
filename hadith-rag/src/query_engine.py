"""
Retriever and query engine module for Arabic Hadith RAG pipeline.
Implements advanced retrieval with Ollama LLM integration.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

from src.config import get_config
from src.index_builder import HadithIndexBuilder, build_hadith_index


class HadithQueryEngine:
    """Query engine for Arabic Hadith retrieval and question answering."""
    
    def __init__(
        self,
        index: Optional[VectorStoreIndex] = None,
        similarity_top_k: Optional[int] = None,
        use_reranking: bool = False,
        response_mode: str = "tree_summarize"
    ):
        """Initialize the query engine.
        
        Args:
            index: Pre-built vector index (optional)
            similarity_top_k: Number of documents to retrieve
            use_reranking: Whether to use reranking for better results
            response_mode: Mode for response synthesis
        """
        config = get_config()
        self.config = config
        
        self.similarity_top_k = similarity_top_k or config.SIMILARITY_TOP_K
        self.use_reranking = use_reranking
        self.response_mode = response_mode
        
        # Setup LLM
        self.llm = self._setup_llm()
        Settings.llm = self.llm
        
        # Build or load index
        self.index = index or self._get_or_build_index()
        
        # Setup retriever and query engine
        self.retriever = self._setup_retriever()
        self.query_engine = self._setup_query_engine()
    
    def _setup_llm(self) -> Ollama:
        """Setup Ollama LLM for query processing."""
        config = self.config
        
        llm = Ollama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.TEMPERATURE,
            request_timeout=120,  # Longer timeout for Arabic processing
            additional_kwargs={
                "num_predict": config.MAX_TOKENS,
                "top_p": 0.9,
            }
        )
        
        print(f"🤖 Initialized Ollama LLM: {config.LLM_MODEL}")
        return llm
    
    def _get_or_build_index(self) -> VectorStoreIndex:
        """Get existing index or build new one."""
        try:
            # Try to load existing index with semantic chunking
            builder = HadithIndexBuilder(
                storage_dir=self.config.STORAGE_DIR,
                use_sentence_window=False,  # Semantic chunking prioritized
                rebuild=False
            )
            return builder.build_index()
        except Exception as e:
            print(f"⚠️  Could not load existing index: {e}")
            print("🔨 Building new index with semantic chunking...")
            return build_hadith_index()
    
    def _setup_retriever(self) -> VectorIndexRetriever:
        """Setup document retriever."""
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.similarity_top_k,
        )
        
        print(f"🔍 Retriever configured with top_k={self.similarity_top_k}")
        return retriever
    
    def _setup_query_engine(self) -> RetrieverQueryEngine:
        """Setup the complete query engine."""
        # Setup response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode=self.response_mode,
            use_async=False,
        )
        
        # Setup postprocessors
        node_postprocessors = []
        
        if self.use_reranking:
            try:
                # Add reranker for better relevance
                reranker = SentenceTransformerRerank(
                    model="cross-encoder/ms-marco-MiniLM-L-12-v2",
                    top_n=self.similarity_top_k // 2,  # Rerank top half
                )
                node_postprocessors.append(reranker)
                print("🎯 Reranker enabled")
            except Exception as e:
                print(f"⚠️  Could not setup reranker: {e}")
        
        # Create query engine
        query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
        )
        
        print("🚀 Query engine ready")
        return query_engine
    
    def query(
        self, 
        question: str, 
        include_sources: bool = True,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Execute a query against the Hadith corpus.
        
        Args:
            question: The question to ask
            include_sources: Whether to include source information
            stream: Whether to stream the response
        
        Returns:
            Dictionary with response and metadata
        """
        try:
            print(f"❓ Query: {question}")
            
            # Execute query
            if stream:
                response = self.query_engine.query(question)
                # For streaming, we'd need to implement async handling
                # For now, return regular response
            else:
                response = self.query_engine.query(question)
            
            # Process response
            result = {
                "question": question,
                "answer": str(response),
                "metadata": {}
            }
            
            # Add source information if available
            if include_sources and hasattr(response, 'source_nodes'):
                sources = []
                for i, node in enumerate(response.source_nodes):
                    source_info = {
                        "rank": i + 1,
                        "score": getattr(node, 'score', None),
                        "text_snippet": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "metadata": dict(node.metadata) if node.metadata else {}
                    }
                    sources.append(source_info)
                
                result["sources"] = sources
                result["metadata"]["num_sources"] = len(sources)
            
            # Add query metadata
            result["metadata"].update({
                "similarity_top_k": self.similarity_top_k,
                "llm_model": self.config.LLM_MODEL,
                "embedding_model": self.config.EMBEDDING_MODEL,
                "response_mode": self.response_mode,
            })
            
            return result
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "error": True,
                "metadata": {"error_details": str(e)}
            }
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Execute multiple queries in batch.
        
        Args:
            questions: List of questions to ask
        
        Returns:
            List of query results
        """
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
        return results
    
    def get_relevant_documents(
        self, 
        query: str, 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get relevant documents without generating an answer.
        
        Args:
            query: Search query
            top_k: Number of documents to return
        
        Returns:
            List of relevant documents with metadata
        """
        top_k = top_k or self.similarity_top_k
        
        # Update retriever for this query
        temp_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
        )
        
        # Retrieve documents
        nodes = temp_retriever.retrieve(query)
        
        documents = []
        for i, node in enumerate(nodes):
            doc_info = {
                "rank": i + 1,
                "score": getattr(node, 'score', None),
                "text": node.text,
                "metadata": dict(node.metadata) if node.metadata else {},
                "node_id": node.id_
            }
            documents.append(doc_info)
        
        return documents
    
    def update_settings(self, **kwargs):
        """Update query engine settings.
        
        Args:
            **kwargs: Settings to update (similarity_top_k, etc.)
        """
        if "similarity_top_k" in kwargs:
            self.similarity_top_k = kwargs["similarity_top_k"]
            self.retriever = self._setup_retriever()  # Recreate retriever
            self.query_engine = self._setup_query_engine()  # Recreate query engine
        
        if "response_mode" in kwargs:
            self.response_mode = kwargs["response_mode"]
            self.query_engine = self._setup_query_engine()  # Recreate query engine


def create_query_engine(
    index: Optional[VectorStoreIndex] = None,
    similarity_top_k: Optional[int] = None,
    use_reranking: bool = False,
    response_mode: str = "tree_summarize"
) -> HadithQueryEngine:
    """Factory function to create a Hadith query engine.
    
    Args:
        index: Pre-built vector index
        similarity_top_k: Number of documents to retrieve
        use_reranking: Whether to use reranking
        response_mode: Response synthesis mode
    
    Returns:
        Configured HadithQueryEngine instance
    """
    return HadithQueryEngine(
        index=index,
        similarity_top_k=similarity_top_k,
        use_reranking=use_reranking,
        response_mode=response_mode
    )