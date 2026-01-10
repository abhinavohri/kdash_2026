"""
Main pipeline orchestration for KDSH Track A.

Coordinates all components: ingestion, embedding, storage, retrieval, and classification.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from .config import PipelineConfig
from .data.ingestion import PathwayNovelProcessor, load_all_novels
from .data.chunking import get_chunker
from .models.embeddings import get_embedding_provider
from .storage.pgvector import VectorStore
from .retrieval.retriever import EvidenceRetriever
from .reasoning.classifier import ConsistencyClassifier
from .logger import get_logger, logger

logger = get_logger("pipeline")


class KDSHPipeline:
    """
    Main pipeline for KDSH Track A.
    
    Orchestrates the complete flow:
    1. Ingest and index novels (if not already indexed)
    2. For each test case, retrieve relevant evidence
    3. Classify consistency using LLM
    4. Return predictions and rationales
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        
        logger.info("=" * 60)
        logger.info("Initializing KDSH Pipeline")
        logger.info("=" * 60)
        logger.info(f"LLM Model: {self.config.llm.model}")
        logger.info(f"Embedding Model: {self.config.embedding.model}")
        logger.info(f"Chunking: {self.config.chunking.strategy} (size={self.config.chunking.chunk_size})")
        logger.info(f"Retrieval: top_k={self.config.retrieval.top_k}")
        
        # Initialize components (lazy loading)
        self._embedder = None
        self._store = None
        self._retriever = None
        self._classifier = None
    
    @property
    def embedder(self):
        """Lazy-load embedding provider."""
        if self._embedder is None:
            self._embedder = get_embedding_provider(self.config.embedding)
        return self._embedder
    
    @property
    def store(self):
        """Lazy-load vector store."""
        if self._store is None:
            self._store = VectorStore(self.config.database)
        return self._store
    
    @property
    def retriever(self):
        """Lazy-load evidence retriever."""
        if self._retriever is None:
            self._retriever = EvidenceRetriever(
                embedding_provider=self.embedder,
                vector_store=self.store,
                config=self.config.retrieval
            )
        return self._retriever
    
    @property
    def classifier(self):
        """Lazy-load consistency classifier."""
        if self._classifier is None:
            self._classifier = ConsistencyClassifier(config=self.config.llm)
        return self._classifier
    
    def index_books(self, force_reindex: bool = False) -> Dict[str, int]:
        """
        Index all books in the books directory.
        
        Supports resumable indexing - skips already indexed chunks.
        
        Args:
            force_reindex: If True, reindex even if already indexed
            
        Returns:
            Dictionary mapping book names to chunk counts
        """
        logger.info("=" * 60)
        logger.info("Indexing Books (resumable)")
        logger.info("=" * 60)
        
        # Load all novels
        books = load_all_novels(self.config.books_dir)
        
        # Filter books if specified
        if self.config.books_filter:
            books = {k: v for k, v in books.items() if k in self.config.books_filter}
            logger.info(f"Filtering to books: {list(books.keys())}")
        
        chunk_counts = {}
        chunker = get_chunker(self.config.chunking)
        
        for book_name, book_text in books.items():
            # Clear existing if force reindex
            if force_reindex:
                self.store.clear_book(book_name)
            
            # Get already indexed chunk indices
            indexed_indices = self.store.get_indexed_chunk_indices(book_name)
            
            # Chunk the text
            chunks_text = chunker.chunk(book_text)
            total_chunks = len(chunks_text)
            
            # Find chunks that need indexing
            chunks_to_index = []
            texts_to_embed = []
            for i, content in enumerate(chunks_text):
                if i not in indexed_indices:
                    chunks_to_index.append({
                        "book_name": book_name,
                        "chunk_index": i,
                        "content": content
                    })
                    texts_to_embed.append(content)
            
            already_indexed = len(indexed_indices)
            remaining = len(chunks_to_index)
            
            if remaining == 0:
                logger.info(f"'{book_name}' fully indexed ({total_chunks} chunks)")
                chunk_counts[book_name] = total_chunks
                continue
            
            logger.info(f"'{book_name}': {already_indexed}/{total_chunks} already indexed, {remaining} remaining")
            
            # Generate embeddings for remaining chunks
            logger.info(f"Generating embeddings for {remaining} chunks...")
            embeddings = []
            
            # Process in batches with progress bar
            batch_size = 50
            for i in tqdm(range(0, len(texts_to_embed), batch_size), desc=f"Embedding {book_name[:20]}"):
                batch = texts_to_embed[i:i + batch_size]
                batch_embeddings = self.embedder.embed_documents(batch)
                embeddings.extend(batch_embeddings)
                
                # Save progress incrementally (in case of interruption)
                batch_chunks = chunks_to_index[i:i + len(batch_embeddings)]
                self.store.insert_chunks(batch_chunks, batch_embeddings[-len(batch_chunks):])
                logger.info(f"Saved batch {i // batch_size + 1} to database")
            
            chunk_counts[book_name] = total_chunks
            logger.info(f"Indexed '{book_name}': {total_chunks} total chunks")
        
        logger.info(f"Indexing complete. Total books: {len(chunk_counts)}")
        return chunk_counts
    
    def process_single(
        self,
        backstory: str,
        book_name: str,
        character: str
    ) -> Tuple[int, str, List[str]]:
        """
        Process a single backstory verification.
        
        Args:
            backstory: Character backstory to verify
            book_name: Name of the book
            character: Character name
            
        Returns:
            Tuple of (prediction, rationale, evidence_chunks)
        """
        logger.info(f"Processing: {character} in {book_name}")
        
        # Retrieve evidence
        evidence = self.retriever.retrieve(backstory, book_name)
        
        # Classify
        prediction, rationale = self.classifier.classify(
            backstory=backstory,
            evidence=evidence,
            character=character,
            book_name=book_name
        )
        
        return prediction, rationale, evidence
    
    def run_evaluation(
        self,
        csv_path: Optional[str] = None,
        num_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Run evaluation on the training dataset.
        
        Args:
            csv_path: Path to CSV file (uses config default if None)
            num_rows: Number of rows to evaluate (uses config default if None)
            
        Returns:
            DataFrame with predictions and metrics
        """
        csv_path = csv_path or self.config.train_csv
        num_rows = num_rows or self.config.eval_rows
        
        logger.info("=" * 60)
        logger.info(f"Running Evaluation on {num_rows} rows")
        logger.info("=" * 60)
        
        # Load data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        
        # Limit rows if specified
        if num_rows and num_rows < len(df):
            df = df.head(num_rows)
            logger.info(f"Evaluating first {num_rows} rows")
        
        # Ensure books are indexed
        self.index_books()
        
        # Process each row
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            # Map book_name to file name format
            book_name = self._normalize_book_name(row['book_name'])
            
            prediction, rationale, evidence = self.process_single(
                backstory=row['content'],
                book_name=book_name,
                character=row['char']
            )
            
            # Map label to int
            actual_label = 1 if row['label'] == 'consistent' else 0
            
            results.append({
                'id': row['id'],
                'book_name': row['book_name'],
                'character': row['char'],
                'backstory': row['content'][:100] + '...',
                'prediction': prediction,
                'actual': actual_label,
                'correct': prediction == actual_label,
                'rationale': rationale
            })
            
            # Log progress
            is_correct = "✓" if prediction == actual_label else "✗"
            logger.info(f"  [{is_correct}] {row['char']}: pred={prediction}, actual={actual_label}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        accuracy = results_df['correct'].mean()
        correct_count = results_df['correct'].sum()
        total_count = len(results_df)
        
        logger.info("=" * 60)
        logger.info("Evaluation Results")
        logger.info("=" * 60)
        logger.info(f"Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
        
        # Breakdown by label
        for label in [0, 1]:
            subset = results_df[results_df['actual'] == label]
            if len(subset) > 0:
                label_acc = subset['correct'].mean()
                label_name = 'consistent' if label == 1 else 'inconsistent'
                logger.info(f"  {label_name}: {label_acc:.2%} ({subset['correct'].sum()}/{len(subset)})")
        
        return results_df
    
    def _normalize_book_name(self, book_name: str) -> str:
        """
        Normalize book name to match file naming.
        
        Args:
            book_name: Book name from CSV
            
        Returns:
            Normalized book name matching file stem
        """
        # Map from CSV format to file format
        name_mapping = {
            "In Search of the Castaways": "In search of the castaways",
            "The Count of Monte Cristo": "The Count of Monte Cristo"
        }
        return name_mapping.get(book_name, book_name)
    
    def generate_results_csv(
        self,
        csv_path: Optional[str] = None,
        output_path: str = "results.csv"
    ) -> str:
        """
        Generate results CSV for submission.
        
        Args:
            csv_path: Path to test CSV (uses test.csv if None)
            output_path: Path to write results
            
        Returns:
            Path to generated results file
        """
        csv_path = csv_path or self.config.test_csv
        
        logger.info("=" * 60)
        logger.info("Generating Results CSV")
        logger.info("=" * 60)
        
        # Load test data
        df = pd.read_csv(csv_path)
        logger.info(f"Processing {len(df)} test examples")
        
        # Ensure books are indexed
        self.index_books()
        
        # Process each row
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            book_name = self._normalize_book_name(row['book_name'])
            
            prediction, rationale, _ = self.process_single(
                backstory=row['content'],
                book_name=book_name,
                character=row['char']
            )
            
            results.append({
                'Story ID': row['id'],
                'Prediction': prediction,
                'Rationale': rationale
            })
        
        # Write results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"Results written to {output_path}")
        return output_path
    
    def close(self):
        """Close all resources."""
        if self._store:
            self._store.close()
        logger.info("Pipeline resources closed")


def create_pipeline(config: PipelineConfig = None) -> KDSHPipeline:
    """
    Factory function to create a pipeline.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Configured pipeline
    """
    return KDSHPipeline(config)
