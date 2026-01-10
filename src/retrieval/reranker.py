"""
Reranking using cross-encoder models.

Reranks retrieved chunks for better relevance.
"""

from typing import List, Tuple
import os

from ..logger import get_logger

logger = get_logger("reranker")


class CrossEncoderReranker:
    """
    Reranker using cross-encoder models.
    
    Uses sentence-transformers cross-encoder for relevance scoring.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name
        self._model = None
        self._available = None
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self._available = True
            logger.info(f"Loaded cross-encoder: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
            self._available = False
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}")
            self._available = False
    
    @property
    def available(self) -> bool:
        """Check if reranker is available."""
        if self._available is None:
            self._load_model()
        return self._available
    
    def rerank(
        self,
        query: str,
        chunks: List[str],
        chunk_indices: List[int],
        top_k: int = 5
    ) -> List[Tuple[str, float, int]]:
        """
        Rerank chunks based on relevance to query.
        
        Args:
            query: Query text
            chunks: List of chunk texts
            chunk_indices: Original indices of chunks
            top_k: Number of results to return
            
        Returns:
            List of (chunk_text, score, original_index) tuples
        """
        if not self.available:
            logger.warning("Reranker not available, returning original order")
            return [(chunk, 1.0, idx) for chunk, idx in zip(chunks, chunk_indices)][:top_k]
        
        if not chunks:
            return []
        
        logger.debug(f"Reranking {len(chunks)} chunks")
        
        # Create query-document pairs
        pairs = [[query, chunk] for chunk in chunks]
        
        # Get scores from cross-encoder
        scores = self._model.predict(pairs)
        
        # Combine with indices
        results = list(zip(chunks, scores, chunk_indices))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Reranked: top score={results[0][1]:.3f}, bottom={results[-1][1]:.3f}")
        
        return results[:top_k]


def get_reranker(model_name: str = None) -> CrossEncoderReranker:
    """
    Factory function to get a reranker.
    
    Args:
        model_name: Model name (uses default if None)
        
    Returns:
        Configured reranker
    """
    model = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    return CrossEncoderReranker(model)
