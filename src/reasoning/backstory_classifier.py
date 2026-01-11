"""
Canonical Backstory Classifier.

Classifies backstories by comparing against stored canonical backstories.
Supports two modes:
- 'embedding': Fast, LLM-free using cosine similarity
- 'llm': Accurate, uses LLM to compare canonical vs input
"""

from typing import List, Tuple, Optional
import numpy as np

from ..config import LLMConfig
from ..storage.pgvector import VectorStore
from ..models.embeddings import EmbeddingProvider
from ..logger import get_logger

logger = get_logger("backstory_classifier")


CANONICAL_COMPARISON_PROMPT = """You are comparing a character backstory against the canonical backstory from the novel.

## Character: {character}
## Book: {book_name}

## Canonical Backstory (from the novel):
{canonical}

## Input Backstory (to verify):
{input}

## Task:
Determine if the input backstory CONTRADICTS the canonical backstory.

## Decision Rules:
- If input adds new details NOT in canonical → CONSISTENT (new details are allowed)
- If input CONTRADICTS a fact in canonical → INCONSISTENT
- If input is a subset of canonical → CONSISTENT
- If you cannot find a clear contradiction → CONSISTENT

## Output:
PREDICTION: [1 for consistent, 0 for inconsistent]
RATIONALE: [Brief explanation with specific contradiction if found]
"""


class CanonicalBackstoryClassifier:
    """
    Classifies backstories by comparing against canonical backstories.
    
    Supports two modes:
    - 'embedding': Fast, LLM-free using cosine similarity
    - 'llm': Accurate, uses LLM to compare canonical vs input
    """
    
    def __init__(
        self,
        config: LLMConfig,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        llm_provider = None,
        similarity_threshold: float = 0.7
    ):
        self.config = config
        self.store = vector_store
        self.embedder = embedding_provider
        self.llm = llm_provider
        self.similarity_threshold = similarity_threshold
        self.mode = getattr(config, 'canonical_mode', 'embedding')
        
        # Lazy-load LLM if needed (Ollama for local, Gemini for cloud)
        if self.mode == 'llm' and self.llm is None:
            if getattr(config, 'use_local', False):
                from ..models.ollama_llm import get_ollama_llm
                self.llm = get_ollama_llm(model=config.local_model, config=config)
            else:
                from ..models.llm import get_llm_provider
                self.llm = get_llm_provider(config)
        
        logger.info(f"Initialized CanonicalBackstoryClassifier (mode={self.mode}, threshold={similarity_threshold})")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _classify_embedding(
        self,
        backstory: str,
        canonical_text: str,
        canonical_embedding: List[float]
    ) -> Tuple[int, str]:
        """Classify using embedding similarity (fast, no LLM)."""
        input_embedding = self.embedder.embed_query(backstory)
        similarity = self._cosine_similarity(input_embedding, canonical_embedding)
        
        logger.info(f"Similarity to canonical: {similarity:.3f}")
        
        if similarity >= 0.85:
            return 1, f"High similarity ({similarity:.2f}) to canonical backstory - consistent"
        elif similarity < 0.5:
            return 0, f"Low similarity ({similarity:.2f}) to canonical backstory - likely contains contradictions"
        else:
            return 1, f"Moderate similarity ({similarity:.2f}) to canonical - no clear contradiction found"
    
    def _classify_llm(
        self,
        backstory: str,
        canonical_text: str,
        character: str,
        book_name: str
    ) -> Tuple[int, str]:
        """Classify using LLM comparison (accurate, uses API)."""
        import re
        
        prompt = CANONICAL_COMPARISON_PROMPT.format(
            character=character,
            book_name=book_name,
            canonical=canonical_text,
            input=backstory
        )
        
        response = self.llm.generate(prompt)
        
        # Parse response
        prediction = 1  # Default to consistent
        rationale = ""
        
        prediction_match = re.search(r"PREDICTION:\s*(\d)", response, re.IGNORECASE)
        if prediction_match:
            prediction = int(prediction_match.group(1))
        
        rationale_match = re.search(r"RATIONALE:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
        if rationale_match:
            rationale = rationale_match.group(1).strip()
        else:
            rationale = response[-200:]
        
        return prediction, rationale
    
    def classify(
        self,
        backstory: str,
        book_name: str,
        character: str
    ) -> Tuple[int, str]:
        """
        Classify whether a backstory is consistent with the canonical backstory.
        
        Uses mode from config:
        - 'embedding': Fast cosine similarity (no LLM)
        - 'llm': LLM compares canonical vs input (accurate)
        
        Args:
            backstory: Input backstory to verify
            book_name: Name of the book
            character: Character name
            
        Returns:
            Tuple of (prediction: 0 or 1, rationale)
        """
        logger.info(f"Classifying backstory for '{character}' in '{book_name}' (mode={self.mode})")
        
        # Get canonical backstory
        canonical = self.store.get_canonical_backstory(book_name, character)
        
        if not canonical:
            logger.warning(f"No canonical backstory found for '{character}', defaulting to consistent")
            return 1, "No canonical backstory available for comparison - defaulting to consistent"
        
        canonical_text, canonical_embedding = canonical
        
        # Route to appropriate method
        if self.mode == 'llm':
            return self._classify_llm(backstory, canonical_text, character, book_name)
        else:
            if not canonical_embedding:
                logger.warning("Canonical backstory has no embedding, defaulting to consistent")
                return 1, "Canonical backstory missing embedding - defaulting to consistent"
            return self._classify_embedding(backstory, canonical_text, canonical_embedding)
    
    def classify_with_details(
        self,
        backstory: str,
        book_name: str,
        character: str
    ) -> dict:
        """Classify with detailed analysis output."""
        canonical = self.store.get_canonical_backstory(book_name, character)
        
        if not canonical:
            return {
                "prediction": 1,
                "similarity": None,
                "canonical_preview": None,
                "rationale": "No canonical backstory found",
                "mode": self.mode
            }
        
        canonical_text, canonical_embedding = canonical
        
        prediction, rationale = self.classify(backstory, book_name, character)
        
        # Calculate similarity for reference (even in LLM mode)
        similarity = None
        if canonical_embedding:
            input_embedding = self.embedder.embed_query(backstory)
            similarity = self._cosine_similarity(input_embedding, canonical_embedding)
        
        return {
            "prediction": prediction,
            "similarity": similarity,
            "canonical_preview": canonical_text[:200] + "..." if len(canonical_text) > 200 else canonical_text,
            "rationale": rationale,
            "mode": self.mode
        }


def get_canonical_classifier(
    config: LLMConfig,
    vector_store: VectorStore,
    embedding_provider: EmbeddingProvider,
    llm_provider = None,
    similarity_threshold: float = 0.7
) -> CanonicalBackstoryClassifier:
    """Factory function for canonical backstory classifier."""
    return CanonicalBackstoryClassifier(
        config, vector_store, embedding_provider, llm_provider, similarity_threshold
    )

