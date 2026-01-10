"""
Hybrid search combining embedding similarity with BM25.

Provides configurable fusion of dense (embedding) and sparse (BM25) retrieval.
"""

import re
import math
from typing import List, Dict, Tuple
from collections import defaultdict

from ..logger import get_logger

logger = get_logger("hybrid")


class BM25Scorer:
    """
    BM25 scorer for sparse retrieval.
    
    Implements Okapi BM25 algorithm for text similarity scoring.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 scorer.
        
        Args:
            k1: Term frequency saturation parameter
            b: Document length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.corpus_size = 0
        self.documents = []
        self._tokenized_docs = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def fit(self, documents: List[str]):
        """
        Fit BM25 on a corpus of documents.
        
        Args:
            documents: List of document strings
        """
        self.documents = documents
        self.corpus_size = len(documents)
        self._tokenized_docs = []
        
        total_length = 0
        
        for doc in documents:
            tokens = self._tokenize(doc)
            self._tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)
            
            # Count document frequency for each unique term
            seen_terms = set()
            for token in tokens:
                if token not in seen_terms:
                    self.doc_freqs[token] += 1
                    seen_terms.add(token)
        
        self.avg_doc_length = total_length / self.corpus_size if self.corpus_size > 0 else 0
        logger.debug(f"BM25 fitted on {self.corpus_size} documents, avg_len={self.avg_doc_length:.1f}")
    
    def _score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a single document."""
        doc_tokens = self._tokenized_docs[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        
        # Term frequency in document
        tf = defaultdict(int)
        for token in doc_tokens:
            tf[token] += 1
        
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            
            # Inverse document frequency
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
            
            # Term frequency component
            term_freq = tf[term]
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            
            score += idf * (numerator / denominator)
        
        return score
    
    def score(self, query: str) -> List[Tuple[int, float]]:
        """
        Score all documents against a query.
        
        Args:
            query: Query string
            
        Returns:
            List of (doc_index, score) tuples, sorted by score descending
        """
        query_tokens = self._tokenize(query)
        
        scores = []
        for idx in range(self.corpus_size):
            score = self._score_document(query_tokens, idx)
            scores.append((idx, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def get_top_k(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Get top-k documents for a query."""
        return self.score(query)[:k]


class HybridSearcher:
    """
    Hybrid search combining embedding similarity with BM25.
    
    Uses reciprocal rank fusion or weighted combination.
    """
    
    def __init__(self, alpha: float = 0.7):
        """
        Initialize hybrid searcher.
        
        Args:
            alpha: Weight for embedding scores (1-alpha for BM25)
        """
        self.alpha = alpha
        self.bm25 = BM25Scorer()
        self._fitted = False
    
    def fit(self, documents: List[str]):
        """Fit BM25 on documents."""
        self.bm25.fit(documents)
        self._fitted = True
        logger.info(f"HybridSearcher fitted on {len(documents)} documents")
    
    def search(
        self,
        query: str,
        embedding_results: List[Tuple[int, float]],  # (chunk_idx, similarity)
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Perform hybrid search.
        
        Args:
            query: Query text
            embedding_results: Results from embedding search (idx, score)
            top_k: Number of results to return
            
        Returns:
            List of (chunk_index, combined_score) tuples
        """
        if not self._fitted:
            logger.warning("BM25 not fitted, returning embedding results only")
            return embedding_results[:top_k]
        
        # Get BM25 scores
        bm25_scores = {idx: score for idx, score in self.bm25.score(query)}
        
        # Normalize scores to 0-1 range
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
        if max_bm25 > 0:
            bm25_scores = {idx: score / max_bm25 for idx, score in bm25_scores.items()}
        
        # Combine scores
        combined = {}
        
        # Add embedding scores
        for idx, emb_score in embedding_results:
            combined[idx] = self.alpha * emb_score
        
        # Add BM25 scores
        for idx, bm25_score in bm25_scores.items():
            if idx in combined:
                combined[idx] += (1 - self.alpha) * bm25_score
            else:
                combined[idx] = (1 - self.alpha) * bm25_score
        
        # Sort and return top-k
        results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]
