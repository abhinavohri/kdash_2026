"""
Abstract base classes for model providers.

These interfaces enable plug-and-play swapping of embedding and LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Text to embed (backstory/query)
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.
        
        Args:
            texts: List of texts to embed (novel chunks)
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def classify_consistency(
        self, 
        backstory: str, 
        evidence: List[str],
        character: str,
        book_name: str
    ) -> Tuple[int, str]:
        """
        Classify whether a backstory is consistent with evidence.
        
        Args:
            backstory: Character backstory to verify
            evidence: List of relevant excerpts from the novel
            character: Character name
            book_name: Name of the book
            
        Returns:
            Tuple of (prediction: 0 or 1, rationale: explanation string)
        """
        pass
