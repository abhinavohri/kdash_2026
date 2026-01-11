"""Central configuration for KDSH Pipeline."""

import os
from dataclasses import dataclass, field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    provider: Literal["gemini", "openai", "huggingface"] = "gemini"
    model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"))
    task_type: str = "FACT_VERIFICATION"
    document_task_type: str = "RETRIEVAL_DOCUMENT"
    dimensions: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSIONS", "768")))
    rpm_limit: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_RPM_LIMIT", "60")))


@dataclass
class LLMConfig:
    provider: Literal["gemini", "openai", "anthropic"] = "gemini"
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-2.5-flash"))
    temperature: float = 0.2
    max_tokens: int = 4096
    rpm_limit: int = field(default_factory=lambda: int(os.getenv("LLM_RPM_LIMIT", "5")))
    prompt_strategy: str = "conservative"
    optimistic: bool = False  # If True, default to consistent unless contradiction proof found
    
    use_local: bool = False
    local_model: str = "llama3.2"
    
    use_nli: bool = False
    use_hybrid: bool = False
    nli_confidence_threshold: float = 0.9
    min_contradictions: int = 2
    use_llm_fallback: bool = True
    
    # Canonical backstory approach (DEFAULT - better accuracy)
    use_canonical: bool = True  # Use pre-generated canonical backstories for classification
    canonical_mode: str = "llm"  # "embedding" (fast, no LLM) or "llm" (accurate, compares with LLM)


@dataclass
class ChunkingConfig:
    strategy: Literal["fixed_overlap", "semantic", "sentence"] = "fixed_overlap"
    chunk_size: int = 1000
    overlap: int = 200


@dataclass
class RetrievalConfig:
    top_k: int = 10
    use_hybrid: bool = False
    hybrid_alpha: float = 0.7
    use_reranking: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 5
    similarity_threshold: float = 0.0
    use_query_expansion: bool = False
    filter_by_character: bool = False


@dataclass
class DatabaseConfig:
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5433")))
    database: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "kdsh"))
    user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "kdsh"))
    password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "kdsh_password"))
    
    @property
    def connection_string(self) -> str:
        if self.database_url:
            return self.database_url
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class PipelineConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    books_dir: str = "Dataset/Books"
    train_csv: str = "Dataset/train.csv"
    test_csv: str = "Dataset/test.csv"
    books_filter: list = None
    eval_rows: int = 10
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls()


default_config = PipelineConfig()
