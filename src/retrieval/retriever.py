"""
Evidence retrieval service.

Retrieves relevant novel chunks for a given backstory query.
"""

from typing import List, Tuple, Optional

from ..config import RetrievalConfig, EmbeddingConfig, DatabaseConfig
from ..models.embeddings import get_embedding_provider, EmbeddingProvider
from ..storage.pgvector import VectorStore
from ..logger import get_logger

logger = get_logger("retrieval")


class EvidenceRetriever:
    """
    Retrieves relevant evidence from novels for backstory verification.
    
    Uses embedding similarity search to find chunks that are most
    relevant to the backstory being verified.
    """
    
    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
        config: RetrievalConfig = None,
        embedding_config: EmbeddingConfig = None,
        database_config: DatabaseConfig = None
    ):
        """
        Initialize the evidence retriever.
        
        Args:
            embedding_provider: Pre-configured embedding provider (optional)
            vector_store: Pre-configured vector store (optional)
            config: Retrieval configuration
            embedding_config: Embedding configuration (if no provider given)
            database_config: Database configuration (if no store given)
        """
        self.config = config or RetrievalConfig()
        
        # Use provided or create new embedding provider
        self.embedder = embedding_provider or get_embedding_provider(embedding_config)
        
        # Use provided or create new vector store
        self.store = vector_store or VectorStore(database_config)
        
        logger.info(f"Initialized EvidenceRetriever: top_k={self.config.top_k}")
    
    def retrieve(
        self,
        backstory: str,
        book_name: str,
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        Retrieve relevant evidence chunks for a backstory.
        
        Args:
            backstory: Character backstory to verify
            book_name: Name of the book to search
            top_k: Number of chunks to retrieve (overrides config)
            
        Returns:
            List of relevant text chunks from the novel
        """
        top_k = top_k or self.config.top_k
        
        logger.info(f"Retrieving evidence for backstory in '{book_name}'")
        logger.debug(f"Query: {backstory[:100]}...")
        
        # Embed the backstory query
        query_embedding = self.embedder.embed_query(backstory)
        
        # Search for similar chunks
        results = self.store.search_similar(
            query_embedding=query_embedding,
            book_name=book_name,
            top_k=top_k,
            threshold=self.config.similarity_threshold
        )
        
        # Extract just the content
        chunks = [content for content, score, idx in results]
        
        # Log similarity scores
        if results:
            scores = [score for _, score, _ in results]
            logger.info(f"Retrieved {len(chunks)} chunks (similarity: {min(scores):.3f} - {max(scores):.3f})")
        else:
            logger.warning(f"No chunks found for '{book_name}'")
        
        # Apply reranking if enabled (Phase 2)
        if self.config.use_reranking:
            chunks = self._rerank(backstory, chunks)
        
        return chunks
    
    def retrieve_with_scores(
        self,
        backstory: str,
        book_name: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve evidence with similarity scores.
        
        Args:
            backstory: Character backstory to verify
            book_name: Name of the book to search
            top_k: Number of chunks to retrieve
            
        Returns:
            List of tuples (chunk_content, similarity_score)
        """
        top_k = top_k or self.config.top_k
        
        # Embed the backstory query
        query_embedding = self.embedder.embed_query(backstory)
        
        # Search for similar chunks
        results = self.store.search_similar(
            query_embedding=query_embedding,
            book_name=book_name,
            top_k=top_k,
            threshold=self.config.similarity_threshold
        )
        
        return [(content, score) for content, score, idx in results]
    
    def _rerank(self, query: str, chunks: List[str]) -> List[str]:
        """
        Rerank chunks for better relevance (Phase 2 feature).
        
        Args:
            query: Query text
            chunks: Chunks to rerank
            
        Returns:
            Reranked chunks
        """
        # TODO: Implement cross-encoder reranking in Phase 2
        logger.debug("Reranking not yet implemented, returning original order")
        return chunks
    
    def close(self):
        """Close resources."""
        self.store.close()


def get_retriever(
    config: RetrievalConfig = None,
    embedding_config: EmbeddingConfig = None,
    database_config: DatabaseConfig = None
) -> EvidenceRetriever:
    """
    Factory function to create an evidence retriever.
    
    Args:
        config: Retrieval configuration
        embedding_config: Embedding configuration
        database_config: Database configuration
        
    Returns:
        Configured evidence retriever
    """
    return EvidenceRetriever(
        config=config,
        embedding_config=embedding_config,
        database_config=database_config
    )
