"""
Text chunking strategies for processing novels.

Provides pluggable chunking implementations for different strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import re

from ..config import ChunkingConfig
from ..logger import get_logger

logger = get_logger("chunking")


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of text chunks
        """
        pass


class FixedOverlapChunker(ChunkingStrategy):
    """
    Fixed-size chunking with overlap.
    
    Simple and effective baseline strategy. Splits by word count
    with configurable overlap for context continuity.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize fixed overlap chunker.
        
        Args:
            chunk_size: Target words per chunk
            overlap: Words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info(f"Initialized FixedOverlapChunker: size={chunk_size}, overlap={overlap}")
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + self.chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            # Move start forward, accounting for overlap
            start = end - self.overlap
            
            # Prevent infinite loop if overlap >= chunk_size
            if start <= end - self.chunk_size:
                start = end
        
        logger.info(f"Created {len(chunks)} chunks from {len(words)} words")
        return chunks


class SentenceChunker(ChunkingStrategy):
    """
    Sentence-boundary aware chunking.
    
    Splits at sentence boundaries to preserve semantic coherence.
    Better for maintaining context but may result in variable chunk sizes.
    """
    
    def __init__(self, target_size: int = 1000, overlap_sentences: int = 2):
        """
        Initialize sentence chunker.
        
        Args:
            target_size: Target words per chunk (approximate)
            overlap_sentences: Number of sentences to overlap
        """
        self.target_size = target_size
        self.overlap_sentences = overlap_sentences
        logger.info(f"Initialized SentenceChunker: target={target_size}, overlap_sentences={overlap_sentences}")
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved with spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text at sentence boundaries.
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of text chunks
        """
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence exceeds target, save current chunk
            if current_size + sentence_words > self.target_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Keep overlap sentences
                overlap = current_chunk[-self.overlap_sentences:] if len(current_chunk) >= self.overlap_sentences else current_chunk
                current_chunk = overlap.copy()
                current_size = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_words
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Created {len(chunks)} chunks from {len(sentences)} sentences")
        return chunks


def get_chunker(config: ChunkingConfig = None) -> ChunkingStrategy:
    """
    Factory function to get a chunking strategy.
    
    Args:
        config: Chunking configuration
        
    Returns:
        Configured chunking strategy
        
    Raises:
        ValueError: If strategy is not supported
    """
    config = config or ChunkingConfig()
    
    if config.strategy == "fixed_overlap":
        return FixedOverlapChunker(
            chunk_size=config.chunk_size,
            overlap=config.overlap
        )
    elif config.strategy == "sentence":
        return SentenceChunker(
            target_size=config.chunk_size,
            overlap_sentences=config.overlap // 50  # Approximate sentences
        )
    else:
        raise ValueError(f"Unsupported chunking strategy: {config.strategy}")


def chunk_text(text: str, config: ChunkingConfig = None) -> List[str]:
    """
    Convenience function to chunk text with default config.
    
    Args:
        text: Text to chunk
        config: Optional chunking configuration
        
    Returns:
        List of text chunks
    """
    chunker = get_chunker(config)
    return chunker.chunk(text)
