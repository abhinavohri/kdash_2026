"""
Embedding providers implementation.

Supports Gemini embeddings with configurable task types and rate limiting.
"""

import os
import time
from typing import List
import google.generativeai as genai

from .base import EmbeddingProvider
from ..config import EmbeddingConfig
from ..logger import get_logger

logger = get_logger("embeddings")


class GeminiEmbedding(EmbeddingProvider):
    """
    Gemini embedding provider using gemini-embedding-001.
    
    Uses FACT_VERIFICATION task type for queries (backstories)
    and RETRIEVAL_DOCUMENT for indexing novel chunks.
    """
    
    def __init__(self, config: EmbeddingConfig = None):
        """
        Initialize Gemini embedding provider.
        
        Args:
            config: Embedding configuration (uses defaults if None)
        """
        self.config = config or EmbeddingConfig()
        
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = self.config.model
        
        logger.info(f"Initialized Gemini embedding: model={self.model}, dims={self.config.dimensions}")
    
    @property
    def dimensions(self) -> int:
        """Return configured embedding dimensions."""
        return self.config.dimensions
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query (backstory) using FACT_VERIFICATION task type.
        
        This task type is optimized for verifying factual claims.
        """
        logger.debug(f"Embedding query (task={self.config.task_type}): {text[:50]}...")
        
        result = genai.embed_content(
            model=f"models/{self.model}",
            content=text,
            task_type=self.config.task_type,
            output_dimensionality=self.config.dimensions
        )
        
        return result["embedding"]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents (novel chunks) using RETRIEVAL_DOCUMENT task type.
        
        This task type is optimized for document retrieval.
        Includes rate limiting and retry logic to respect API quotas.
        """
        if not texts:
            return []
        
        logger.info(f"Embedding {len(texts)} documents (task={self.config.document_task_type})")
        
        # Calculate delay based on RPM limit
        rpm_limit = self.config.rpm_limit
        delay_seconds = 60.0 / rpm_limit if rpm_limit > 0 else 0
        
        if delay_seconds > 0:
            logger.info(f"Rate limiting: {rpm_limit} RPM ({delay_seconds:.1f}s between requests)")
        
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            # Rate limiting delay (skip for first request)
            if i > 0 and delay_seconds > 0:
                time.sleep(delay_seconds)
            
            # Retry logic with exponential backoff
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    result = genai.embed_content(
                        model=f"models/{self.model}",
                        content=text,
                        task_type=self.config.document_task_type,
                        output_dimensionality=self.config.dimensions
                    )
                    embeddings.append(result["embedding"])
                    break  # Success, exit retry loop
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        # Rate limit hit - wait and retry
                        wait_time = (2 ** attempt) * 15  # 15, 30, 60, 120, 240 seconds
                        logger.warning(f"Rate limit hit. Waiting {wait_time}s before retry ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        raise  # Re-raise non-rate-limit errors
            else:
                # All retries failed
                raise Exception(f"Failed to embed after {max_retries} retries due to rate limits")
            
            # Log progress every 10 documents
            if (i + 1) % 10 == 0 or i == total - 1:
                logger.info(f"Embedded {i + 1}/{total} documents")
        
        return embeddings


def get_embedding_provider(config: EmbeddingConfig = None) -> EmbeddingProvider:
    """
    Factory function to get an embedding provider.
    
    Args:
        config: Embedding configuration
        
    Returns:
        Configured embedding provider
    
    Raises:
        ValueError: If provider is not supported
    """
    config = config or EmbeddingConfig()
    
    if config.provider == "gemini":
        return GeminiEmbedding(config)
    else:
        raise ValueError(f"Unsupported embedding provider: {config.provider}")
