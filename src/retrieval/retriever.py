"""
Evidence retrieval service.

Retrieves relevant novel chunks for a given backstory query.
Supports hybrid search (embedding + BM25) and reranking.
"""

from typing import List, Tuple, Optional, Dict

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
    
    Supports:
    - Hybrid search: Combines embedding similarity with BM25
    - Reranking: Cross-encoder for improved relevance
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
        
        # Lazy-loaded hybrid searcher and reranker
        self._hybrid_searcher = None
        self._reranker = None
        self._book_chunks_cache: Dict[str, List[str]] = {}
        
        logger.info(f"Initialized EvidenceRetriever: top_k={self.config.top_k}, "
                   f"hybrid={self.config.use_hybrid}, rerank={self.config.use_reranking}")
    
    def _get_hybrid_searcher(self):
        """Lazy-load hybrid searcher."""
        if self._hybrid_searcher is None and self.config.use_hybrid:
            from .hybrid import HybridSearcher
            self._hybrid_searcher = HybridSearcher(alpha=self.config.hybrid_alpha)
        return self._hybrid_searcher
    
    def _get_reranker(self):
        """Lazy-load reranker."""
        if self._reranker is None and self.config.use_reranking:
            from .reranker import get_reranker
            self._reranker = get_reranker(self.config.rerank_model)
        return self._reranker
    
    def _fit_hybrid_for_book(self, book_name: str):
        """Fit hybrid searcher on book chunks if needed."""
        if not self.config.use_hybrid:
            return
        
        if book_name in self._book_chunks_cache:
            return
        
        # Load all chunks for this book
        chunks = self.store.get_all_chunks(book_name)
        if chunks:
            self._book_chunks_cache[book_name] = chunks
            hybrid = self._get_hybrid_searcher()
            if hybrid:
                hybrid.fit(chunks)
                logger.info(f"Fitted hybrid search on {len(chunks)} chunks for '{book_name}'")
    
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
        
        # Get more results if we're doing reranking
        initial_k = top_k * 2 if self.config.use_reranking else top_k
        
        # Search for similar chunks
        results = self.store.search_similar(
            query_embedding=query_embedding,
            book_name=book_name,
            top_k=initial_k,
            threshold=self.config.similarity_threshold
        )
        
        if not results:
            logger.warning(f"No chunks found for '{book_name}'")
            return []
        
        # Apply hybrid search if enabled
        if self.config.use_hybrid:
            self._fit_hybrid_for_book(book_name)
            hybrid = self._get_hybrid_searcher()
            if hybrid and hybrid._fitted:
                embedding_results = [(idx, score) for _, score, idx in results]
                hybrid_results = hybrid.search(backstory, embedding_results, initial_k)
                # Reorder results based on hybrid scores
                idx_to_result = {idx: (content, score, idx) for content, score, idx in results}
                results = [idx_to_result[idx] for idx, _ in hybrid_results if idx in idx_to_result]
                logger.info(f"Applied hybrid search, reordered {len(results)} chunks")
        
        # Extract chunks with indices for reranking
        chunks = [content for content, score, idx in results]
        indices = [idx for content, score, idx in results]
        
        # Log similarity scores
        scores = [score for _, score, _ in results]
        logger.info(f"Retrieved {len(chunks)} chunks (similarity: {min(scores):.3f} - {max(scores):.3f})")
        
        # Apply reranking if enabled
        if self.config.use_reranking:
            reranker = self._get_reranker()
            if reranker and reranker.available:
                rerank_k = self.config.rerank_top_k or top_k
                reranked = reranker.rerank(backstory, chunks, indices, top_k=rerank_k)
                chunks = [chunk for chunk, _, _ in reranked]
                logger.info(f"Reranked to top {len(chunks)} chunks")
            else:
                chunks = chunks[:top_k]
        else:
            chunks = chunks[:top_k]
        
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

