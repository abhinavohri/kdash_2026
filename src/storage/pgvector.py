"""
PostgreSQL + pgvector storage for novel chunks.

Provides vector similarity search using pgvector extension.
"""

from typing import List, Dict, Optional, Tuple
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

from ..config import DatabaseConfig
from ..logger import get_logger

logger = get_logger("storage")


class VectorStore:
    """
    PostgreSQL + pgvector storage for embeddings.
    
    Stores novel chunks with their embeddings and provides
    similarity search for evidence retrieval.
    """
    
    def __init__(self, config: DatabaseConfig = None):
        """
        Initialize vector store connection.
        
        Args:
            config: Database configuration (uses defaults if None)
        """
        self.config = config or DatabaseConfig()
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Establish database connection."""
        conn_str = self.config.connection_string
        # Mask password in log
        safe_str = conn_str.split('@')[-1] if '@' in conn_str else conn_str
        logger.info(f"Connecting to PostgreSQL: ...@{safe_str}")
        
        try:
            self.conn = psycopg2.connect(conn_str)
            
            # Register pgvector type
            register_vector(self.conn)
            
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _ensure_connection(self):
        """Ensure database connection is active."""
        if self.conn is None or self.conn.closed:
            self._connect()
    
    def init_schema(self):
        """Initialize database schema (create tables and extensions)."""
        self._ensure_connection()
        
        logger.info("Initializing database schema...")
        
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create chunks table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    book_name TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(768),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(book_name, chunk_index)
                )
            """)
            
            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_book_name_idx ON chunks(book_name)
            """)
        
        self.conn.commit()
        logger.info("Database schema initialized")
    
    def is_book_indexed(self, book_name: str) -> bool:
        """
        Check if a book has already been indexed.
        
        Args:
            book_name: Name of the book to check
            
        Returns:
            True if book has chunks in the database
        """
        self._ensure_connection()
        
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM chunks WHERE book_name = %s",
                (book_name,)
            )
            count = cur.fetchone()[0]
        
        return count > 0
    
    def get_indexed_books(self) -> List[str]:
        """
        Get list of books that have been indexed.
        
        Returns:
            List of indexed book names
        """
        self._ensure_connection()
        
        with self.conn.cursor() as cur:
            cur.execute("SELECT DISTINCT book_name FROM chunks")
            books = [row[0] for row in cur.fetchall()]
        
        return books
    
    def get_indexed_chunk_indices(self, book_name: str) -> set:
        """
        Get set of chunk indices already indexed for a book.
        
        Args:
            book_name: Name of the book
            
        Returns:
            Set of chunk indices that are already in the database
        """
        self._ensure_connection()
        
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_index FROM chunks WHERE book_name = %s",
                (book_name,)
            )
            indices = {row[0] for row in cur.fetchall()}
        
        return indices
    
    def insert_chunks(
        self, 
        chunks: List[Dict],
        embeddings: List[List[float]]
    ) -> int:
        """
        Insert chunks with embeddings into the database.
        
        Args:
            chunks: List of chunk dictionaries (book_name, chunk_index, content)
            embeddings: List of embedding vectors
            
        Returns:
            Number of chunks inserted
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        self._ensure_connection()
        
        logger.info(f"Inserting {len(chunks)} chunks into database")
        
        # Prepare data for insertion
        data = [
            (
                chunk["book_name"],
                chunk["chunk_index"],
                chunk["content"],
                embedding
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO chunks (book_name, chunk_index, content, embedding)
                VALUES %s
                ON CONFLICT (book_name, chunk_index) DO UPDATE
                SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
                """,
                data,
                template="(%s, %s, %s, %s::vector)"
            )
        
        self.conn.commit()
        logger.info(f"Successfully inserted {len(chunks)} chunks")
        
        return len(chunks)
    
    def search_similar(
        self,
        query_embedding: List[float],
        book_name: str,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[str, float, int]]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            book_name: Book to search within
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of tuples (content, similarity_score, chunk_index)
        """
        self._ensure_connection()
        
        logger.debug(f"Searching for {top_k} similar chunks in '{book_name}'")
        
        with self.conn.cursor() as cur:
            # Cosine similarity using pgvector (<=> is cosine distance)
            # Convert to similarity: 1 - distance
            cur.execute(
                """
                SELECT content, 1 - (embedding <=> %s::vector) as similarity, chunk_index
                FROM chunks
                WHERE book_name = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, book_name, query_embedding, top_k)
            )
            
            results = cur.fetchall()
        
        # Filter by threshold
        filtered = [(content, sim, idx) for content, sim, idx in results if sim >= threshold]
        
        logger.debug(f"Found {len(filtered)} chunks above threshold {threshold}")
        return filtered
    
    def get_chunk_count(self, book_name: Optional[str] = None) -> int:
        """
        Get count of chunks in database.
        
        Args:
            book_name: Optional book to filter by
            
        Returns:
            Number of chunks
        """
        self._ensure_connection()
        
        with self.conn.cursor() as cur:
            if book_name:
                cur.execute(
                    "SELECT COUNT(*) FROM chunks WHERE book_name = %s",
                    (book_name,)
                )
            else:
                cur.execute("SELECT COUNT(*) FROM chunks")
            
            return cur.fetchone()[0]
    
    def clear_book(self, book_name: str) -> int:
        """
        Delete all chunks for a specific book.
        
        Args:
            book_name: Book to clear
            
        Returns:
            Number of chunks deleted
        """
        self._ensure_connection()
        
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE book_name = %s RETURNING id",
                (book_name,)
            )
            deleted = cur.rowcount
        
        self.conn.commit()
        logger.info(f"Deleted {deleted} chunks for '{book_name}'")
        
        return deleted
    
    def clear_all(self) -> int:
        """
        Delete all chunks from database.
        
        Returns:
            Number of chunks deleted
        """
        self._ensure_connection()
        
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks RETURNING id")
            deleted = cur.rowcount
        
        self.conn.commit()
        logger.info(f"Deleted {deleted} total chunks")
        
        return deleted
    
    def close(self):
        """Close database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
