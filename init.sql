-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create chunks table for storing novel chunks with embeddings
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    book_name TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(book_name, chunk_index)
);

-- Create index for fast similarity search
CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
ON chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for book name lookups
CREATE INDEX IF NOT EXISTS chunks_book_name_idx ON chunks(book_name);
