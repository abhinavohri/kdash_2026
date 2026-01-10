"""
Central configuration for KDSH Pipeline.

All components use these dataclasses for configuration, enabling plug-and-play
swapping of models, strategies, and parameters with minimal code changes.
"""

import os
from dataclasses import dataclass, field
from typing import Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models.
    
    Attributes:
        provider: Embedding provider ("gemini", "openai", etc.)
        model: Model name/identifier
        task_type: Task type for embeddings (Gemini-specific)
        dimensions: Output embedding dimensions
        rpm_limit: Requests per minute limit for rate limiting
    """
    provider: Literal["gemini", "openai", "huggingface"] = "gemini"
    model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"))
    task_type: str = "FACT_VERIFICATION"  # Best for verifying backstory claims
    document_task_type: str = "RETRIEVAL_DOCUMENT"  # For indexing novel chunks
    dimensions: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSIONS", "768")))
    rpm_limit: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_RPM_LIMIT", "20")))  # 20 RPM (stay under 30K TPM with big chunks)


@dataclass
class LLMConfig:
    """Configuration for LLM reasoning.
    
    Attributes:
        provider: LLM provider ("gemini", "openai", etc.)
        model: Model name/identifier
        temperature: Sampling temperature (lower = more deterministic)
        max_tokens: Maximum tokens in response
        rpm_limit: Requests per minute limit (5 for free tier flash)
        prompt_strategy: Prompting strategy ('base', 'few_shot', 'cot', 'claim_extraction')
    """
    provider: Literal["gemini", "openai", "anthropic"] = "gemini"
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-2.5-flash"))
    temperature: float = 0.2
    max_tokens: int = 4096
    rpm_limit: int = field(default_factory=lambda: int(os.getenv("LLM_RPM_LIMIT", "5")))  # 5 RPM for flash
    prompt_strategy: str = "base"  # Options: base, few_shot, cot, claim_extraction
    
    # Local model settings
    use_local: bool = False  # Use Ollama instead of API
    local_model: str = "llama3.2"  # Ollama model name


@dataclass
class ChunkingConfig:
    """Configuration for text chunking strategies.
    
    Attributes:
        strategy: Chunking strategy name
        chunk_size: Target words per chunk
        overlap: Words to overlap between chunks
    """
    strategy: Literal["fixed_overlap", "semantic", "sentence"] = "fixed_overlap"
    chunk_size: int = 1000
    overlap: int = 200


@dataclass
class RetrievalConfig:
    """Configuration for evidence retrieval.
    
    Attributes:
        top_k: Number of chunks to retrieve
        use_hybrid: Whether to use hybrid search (embedding + BM25)
        hybrid_alpha: Weight for embedding score (1-alpha for BM25)
        use_reranking: Whether to apply reranking after retrieval
        rerank_model: Model for reranking (cross-encoder)
        similarity_threshold: Minimum similarity score (0-1)
    """
    top_k: int = 10
    
    # Hybrid search settings
    use_hybrid: bool = False
    hybrid_alpha: float = 0.7  # 0.7 embedding + 0.3 BM25
    
    # Reranking settings
    use_reranking: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 5  # Final top-k after reranking
    
    similarity_threshold: float = 0.0
    
    # Query expansion
    use_query_expansion: bool = False  # Prepend character name to query


@dataclass
class DatabaseConfig:
    """Configuration for PostgreSQL + pgvector.
    
    Supports both individual connection params and full DATABASE_URL.
    """
    # Full connection string (takes priority if set)
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    
    # Individual params (used if DATABASE_URL not set)
    host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5433")))
    database: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "kdsh"))
    user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "kdsh"))
    password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "kdsh_password"))
    
    @property
    def connection_string(self) -> str:
        """PostgreSQL connection string."""
        if self.database_url:
            return self.database_url
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class PipelineConfig:
    """Master configuration combining all component configs.
    
    This is the main configuration object passed throughout the pipeline.
    """
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Paths
    books_dir: str = "Dataset/Books"
    train_csv: str = "Dataset/train.csv"
    test_csv: str = "Dataset/test.csv"
    
    # Book filtering (None = all books, or list of book names)
    books_filter: list = None  # Set to ["In search of the castaways"] to index only one
    
    # Evaluation
    eval_rows: int = 10  # Number of rows to evaluate (configurable)
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create configuration from environment variables."""
        return cls()


# Default configuration instance
default_config = PipelineConfig()
