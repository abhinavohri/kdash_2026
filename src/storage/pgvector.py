"""
PostgreSQL + pgvector storage for novel chunks.

Provides vector similarity search using pgvector extension.
"""

from typing import List, Dict, Optional, Tuple, Any
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
            
            # Add metadata column if it doesn't exist
            cur.execute("""
                ALTER TABLE chunks 
                ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb
            """)
            
            # Create chunks table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    book_name TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(768),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(book_name, chunk_index)
                )
            """)
            
            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_book_name_idx ON chunks(book_name);
                CREATE INDEX IF NOT EXISTS chunks_metadata_idx ON chunks USING GIN (metadata);
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
    
    def get_all_chunks(self, book_name: str) -> List[str]:
        """
        Get all chunk contents for a book (for BM25 indexing).
        
        Args:
            book_name: Name of the book
            
        Returns:
            List of chunk contents ordered by chunk_index
        """
        self._ensure_connection()
        
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT content FROM chunks WHERE book_name = %s ORDER BY chunk_index",
                (book_name,)
            )
            chunks = [row[0] for row in cur.fetchall()]
        
        return chunks
    
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
        data = []
        import json
        
        for chunk, embedding in zip(chunks, embeddings):
            metadata = chunk.get("metadata", {})
            data.append((
                chunk["book_name"],
                chunk["chunk_index"],
                chunk["content"],
                embedding,
                json.dumps(metadata)
            ))
        
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO chunks (book_name, chunk_index, content, embedding, metadata)
                VALUES %s
                ON CONFLICT (book_name, chunk_index) DO UPDATE
                SET content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                data,
                template="(%s, %s, %s, %s::vector, %s::jsonb)"
            )
        
        self.conn.commit()
        logger.info(f"Successfully inserted {len(chunks)} chunks")
        
        return len(chunks)
    
    def update_chunk_metadata(self, chunk_id: int, metadata: Dict):
        """Update metadata for a specific chunk."""
        import json
        import psycopg2
        
        self._ensure_connection()
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "UPDATE chunks SET metadata = %s WHERE id = %s",
                    (json.dumps(metadata), chunk_id)
                )
            self.conn.commit()
        except psycopg2.errors.UndefinedColumn:
            # Auto-migration: Add column if it doesn't exist
            self.conn.rollback()
            logger.info("Migrating schema: Adding metadata column...")
            with self.conn.cursor() as cur:
                cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb")
                cur.execute("CREATE INDEX IF NOT EXISTS chunks_metadata_idx ON chunks USING GIN (metadata)")
            self.conn.commit()
            
            # Retry update
            self.update_chunk_metadata(chunk_id, metadata)

    def get_all_chunks_with_id(self) -> List[Tuple[int, str, str]]:
        """Get all chunks with ID, book_name, content."""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("SELECT id, book_name, content FROM chunks")
            return cur.fetchall()
    
    def search_similar(
        self,
        query_embedding: List[float],
        book_name: str,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[str, float, int, Dict]]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            book_name: Book to search within
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of tuples (content, similarity_score, chunk_index, metadata)
        """
        self._ensure_connection()
        
        logger.debug(f"Searching for {top_k} similar chunks in '{book_name}'")
        
        with self.conn.cursor() as cur:
            # Cosine similarity using pgvector (<=> is cosine distance)
            # Convert to similarity: 1 - distance
            cur.execute(
                """
                SELECT content, 1 - (embedding <=> %s::vector) as similarity, chunk_index, metadata
                FROM chunks
                WHERE book_name = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, book_name, query_embedding, top_k)
            )
            
            results = cur.fetchall()
        
        # Filter by threshold and format
        return [
            (row[0], float(row[1]), row[2], row[3] if len(row) > 3 else {}) 
            for row in results 
            if float(row[1]) >= threshold
        ]
    
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
    
    # =========================================================================
    # CANONICAL BACKSTORY METHODS
    # =========================================================================
    
    def init_canonical_schema(self):
        """Initialize canonical_backstories table."""
        self._ensure_connection()
        
        logger.info("Initializing canonical_backstories schema...")
        
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS canonical_backstories (
                    id SERIAL PRIMARY KEY,
                    book_name TEXT NOT NULL,
                    character_name TEXT NOT NULL,
                    backstory TEXT NOT NULL,
                    backstory_embedding vector(768),
                    version INT DEFAULT 1,
                    model_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(book_name, character_name, version)
                )
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS canonical_book_char_idx 
                ON canonical_backstories(book_name, character_name);
            """)
        
        self.conn.commit()
        logger.info("Canonical backstories schema initialized")
    
    def store_canonical_backstory(
        self,
        book_name: str,
        character_name: str,
        backstory: str,
        embedding: List[float],
        model_used: str = None,
        version: int = 1
    ) -> int:
        """
        Store a canonical backstory for a character.
        
        Args:
            book_name: Name of the book
            character_name: Character name
            backstory: Generated canonical backstory
            embedding: Backstory embedding vector
            model_used: Model used for generation
            version: Version number (for tracking)
            
        Returns:
            ID of inserted/updated row
        """
        self._ensure_connection()
        
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO canonical_backstories 
                    (book_name, character_name, backstory, backstory_embedding, model_used, version)
                VALUES (%s, %s, %s, %s::vector, %s, %s)
                ON CONFLICT (book_name, character_name, version) DO UPDATE
                SET backstory = EXCLUDED.backstory,
                    backstory_embedding = EXCLUDED.backstory_embedding,
                    model_used = EXCLUDED.model_used,
                    created_at = CURRENT_TIMESTAMP
                RETURNING id
            """, (book_name, character_name, backstory, embedding, model_used, version))
            
            row_id = cur.fetchone()[0]
        
        self.conn.commit()
        logger.info(f"Stored canonical backstory for '{character_name}' in '{book_name}' (v{version})")
        
        return row_id
    
    def get_canonical_backstory(
        self,
        book_name: str,
        character_name: str,
        version: int = None
    ) -> Optional[Tuple[str, List[float]]]:
        """
        Get canonical backstory for a character.
        
        Args:
            book_name: Name of the book
            character_name: Character name
            version: Specific version (None = latest)
            
        Returns:
            Tuple of (backstory, embedding) or None if not found
        """
        self._ensure_connection()
        
        with self.conn.cursor() as cur:
            if version:
                cur.execute("""
                    SELECT backstory, backstory_embedding
                    FROM canonical_backstories
                    WHERE book_name = %s AND character_name = %s AND version = %s
                """, (book_name, character_name, version))
            else:
                # Get latest version
                cur.execute("""
                    SELECT backstory, backstory_embedding
                    FROM canonical_backstories
                    WHERE book_name = %s AND character_name = %s
                    ORDER BY version DESC
                    LIMIT 1
                """, (book_name, character_name))
            
            row = cur.fetchone()
        
        if row:
            # row[1] is a numpy array from pgvector, check if it's not None
            embedding = list(row[1]) if row[1] is not None else None
            return row[0], embedding
        return None
    
    def get_all_canonical_backstories(self, book_name: str = None) -> List[Dict]:
        """Get all canonical backstories, optionally filtered by book."""
        self._ensure_connection()
        
        with self.conn.cursor() as cur:
            if book_name:
                cur.execute("""
                    SELECT book_name, character_name, backstory, version, model_used, created_at
                    FROM canonical_backstories
                    WHERE book_name = %s
                    ORDER BY character_name, version DESC
                """, (book_name,))
            else:
                cur.execute("""
                    SELECT book_name, character_name, backstory, version, model_used, created_at
                    FROM canonical_backstories
                    ORDER BY book_name, character_name, version DESC
                """)
            
            rows = cur.fetchall()
        
        return [
            {
                "book_name": row[0],
                "character_name": row[1],
                "backstory": row[2],
                "version": row[3],
                "model_used": row[4],
                "created_at": row[5]
            }
            for row in rows
        ]
    
    def search_by_character(
        self,
        book_name: str,
        character: str,
        top_k: int = 20
    ) -> List[str]:
        """
        Find chunks that mention a specific character.
        
        Uses simple text search on chunk content.
        
        Args:
            book_name: Name of the book
            character: Character name to search for
            top_k: Maximum chunks to return
            
        Returns:
            List of chunk contents mentioning the character
        """
        self._ensure_connection()
        
        # Handle character aliases (e.g., "Tom Ayrton/Ben Joyce")
        char_variants = [c.strip() for c in character.split('/')]
        
        with self.conn.cursor() as cur:
            # Build OR condition for all character variants
            conditions = " OR ".join(["content ILIKE %s" for _ in char_variants])
            params = [f"%{c}%" for c in char_variants]
            params.insert(0, book_name)
            
            query = f"""
                SELECT content, chunk_index
                FROM chunks
                WHERE book_name = %s AND ({conditions})
                ORDER BY chunk_index
                LIMIT %s
            """
            params.append(top_k)
            
            cur.execute(query, params)
            rows = cur.fetchall()
        
        logger.info(f"Found {len(rows)} chunks mentioning '{character}' in '{book_name}'")
        return [row[0] for row in rows]

