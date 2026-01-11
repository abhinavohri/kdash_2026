"""Main pipeline orchestration for KDSH Track A."""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os
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
    """Main pipeline for KDSH Track A."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        logger.info("=" * 60)
        logger.info("Initializing KDSH Pipeline")
        logger.info("=" * 60)
        logger.info(f"LLM Model: {self.config.llm.model}")
        logger.info(f"Embedding Model: {self.config.embedding.model}")
        logger.info(f"Chunking: {self.config.chunking.strategy} (size={self.config.chunking.chunk_size})")
        logger.info(f"Retrieval: top_k={self.config.retrieval.top_k}")
        
        self._embedder = None
        self._store = None
        self._retriever = None
        self._classifier = None
    
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedding_provider(self.config.embedding)
        return self._embedder
    
    @property
    def store(self):
        if self._store is None:
            self._store = VectorStore(self.config.database)
        return self._store
    
    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = EvidenceRetriever(
                embedding_provider=self.embedder,
                vector_store=self.store,
                config=self.config.retrieval
            )
        return self._retriever
    
    @property
    def classifier(self):
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
        
        # Load characters for metadata tagging
        try:
            train_df = pd.read_csv(self.config.train_csv)
            # Normalize character names (remove / variant names for simpler matching)
            known_characters = set()
            for char in train_df['char'].unique():
                # Split "Tom Ayrton/Ben Joyce" -> ["Tom Ayrton", "Ben Joyce"]
                parts = char.split('/')
                known_characters.update([p.strip() for p in parts])
            logger.info(f"Loaded {len(known_characters)} characters for metadata tagging")
        except Exception as e:
            logger.warning(f"Could not load characters for metadata: {e}")
            known_characters = set()
        
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
                    # simplistic character matching
                    chars_found = [c for c in known_characters if c in content]
                    
                    chunks_to_index.append({
                        "book_name": book_name,
                        "chunk_index": i,
                        "content": content,
                        "metadata": {"characters": chars_found}
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
    
    def backfill_metadata(self):
        """Backfill metadata for existing chunks without re-indexing."""
        logger.info("Starting metadata backfill...")
        
        # Load characters
        try:
            train_df = pd.read_csv(self.config.train_csv)
            known_characters = set()
            for char in train_df['char'].unique():
                parts = char.split('/')
                known_characters.update([p.strip() for p in parts])
            logger.info(f"Loaded {len(known_characters)} characters to tag")
        except Exception as e:
            logger.error(f"Failed to load characters: {e}")
            return

        # Get all chunks
        chunks = self.store.get_all_chunks_with_id()
        logger.info(f"Found {len(chunks)} total chunks in database")
        
        updated_count = 0
        for chunk_id, book_name, content in tqdm(chunks, desc="Backfilling"):
            chars_found = [c for c in known_characters if c in content]
            
            if chars_found:
                metadata = {"characters": chars_found}
                self.store.update_chunk_metadata(chunk_id, metadata)
                updated_count += 1
                
        logger.info(f"Backfill complete. Updated {updated_count}/{len(chunks)} chunks with metadata.")
    
    def generate_canonical_backstories(self, force_regenerate: bool = False):
        """
        Generate canonical backstories for all characters.
        
        Uses LLM to analyze character-specific chunks and generate
        a canonical backstory for each character.
        """
        from .reasoning.backstory_generator import CanonicalBackstoryGenerator
        from .models.llm import get_llm_provider
        
        logger.info("=" * 60)
        logger.info("Generating Canonical Backstories")
        logger.info("=" * 60)
        
        # Ensure books are indexed
        self.index_books()
        
        # Get LLM provider
        llm = get_llm_provider(self.config.llm)
        
        # Create generator
        generator = CanonicalBackstoryGenerator(
            config=self.config,
            vector_store=self.store,
            embedding_provider=self.embedder,
            llm_provider=llm
        )
        
        # Load characters from training data
        train_df = pd.read_csv(self.config.train_csv)
        
        # Group characters by book
        for book_name in train_df['book_name'].unique():
            book_chars = train_df[train_df['book_name'] == book_name]['char'].unique()
            normalized_book = self._normalize_book_name(book_name)
            
            logger.info(f"Generating backstories for '{book_name}': {list(book_chars)}")
            
            for character in book_chars:
                generator.generate_for_character(
                    book_name=normalized_book,
                    character=character,
                    force_regenerate=force_regenerate
                )
        
        # Show summary
        all_backstories = self.store.get_all_canonical_backstories()
        logger.info(f"Total canonical backstories: {len(all_backstories)}")
    
    def _ensure_canonical_backstories(self):
        """Ensure canonical backstories exist, generate if missing."""
        # First ensure books are indexed
        self.index_books()
        
        # Check if canonical backstories exist
        existing = self.store.get_all_canonical_backstories()
        
        if not existing:
            logger.info("No canonical backstories found - generating automatically...")
            self.generate_canonical_backstories()
        else:
            logger.info(f"Found {len(existing)} existing canonical backstories")
    
    @property
    def canonical_classifier(self):
        """Get canonical backstory classifier (lazy initialized with auto-generation)."""
        if not hasattr(self, '_canonical_classifier') or self._canonical_classifier is None:
            # Auto-generate backstories if needed when canonical mode is used
            self._ensure_canonical_backstories()
            
            from .reasoning.backstory_classifier import CanonicalBackstoryClassifier
            self._canonical_classifier = CanonicalBackstoryClassifier(
                config=self.config.llm,
                vector_store=self.store,
                embedding_provider=self.embedder
            )
        return self._canonical_classifier
    
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
        
        # Use canonical backstory classifier if enabled
        if self.config.llm.use_canonical:
            logger.info("Using canonical backstory classifier")
            prediction, rationale = self.canonical_classifier.classify(
                backstory=backstory,
                book_name=book_name,
                character=character
            )
            return prediction, rationale, []  # No evidence needed for canonical approach
        
        # Standard approach: retrieve evidence and classify
        evidence = self.retriever.retrieve(backstory, book_name, character=character)
        
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
        
        Filters to only evaluate rows from indexed books.
        Auto-exports results to JSON.
        
        Args:
            csv_path: Path to CSV file (uses config default if None)
            num_rows: Number of rows to evaluate (uses config default if None)
            
        Returns:
            DataFrame with predictions and metrics
        """
        import json
        
        csv_path = csv_path or self.config.train_csv
        num_rows = num_rows or self.config.eval_rows
        
        logger.info("=" * 60)
        logger.info(f"Running Evaluation")
        logger.info("=" * 60)
        
        # Ensure books are indexed first
        self.index_books()
        
        # Get list of indexed books
        indexed_books = self.store.get_indexed_books()
        
        # Apply filter if configured (e.g. --one-book)
        if self.config.books_filter:
            logger.info(f"Applying book filter: {self.config.books_filter}")
            # Normalize filter names
            filter_norm = [self._normalize_book_name(b) for b in self.config.books_filter]
            # Intersect with indexed books
            indexed_books = [b for b in indexed_books if b in filter_norm]
            
        logger.info(f"Evaluating on books: {indexed_books}")
        
        # Load data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        
        # Filter to only indexed books
        df['normalized_book'] = df['book_name'].apply(self._normalize_book_name)
        df = df[df['normalized_book'].isin(indexed_books)]
        logger.info(f"Filtered to {len(df)} rows from indexed books")
        
        # Limit rows if specified - use balanced sampling (half consistent, half inconsistent)
        if num_rows and num_rows < len(df):
            # Split by label and take equal numbers from each
            consistent_df = df[df['label'] == 'consistent']
            inconsistent_df = df[df['label'] != 'consistent']  # 'contradict' maps to inconsistent
            
            half = num_rows // 2
            # Sample or take first N from each group
            consistent_sample = consistent_df.head(half)
            inconsistent_sample = inconsistent_df.head(half)
            
            # If odd number, add one more from whichever has more
            if num_rows % 2 == 1:
                if len(consistent_df) > half:
                    consistent_sample = consistent_df.head(half + 1)
                elif len(inconsistent_df) > half:
                    inconsistent_sample = inconsistent_df.head(half + 1)
            
            df = pd.concat([consistent_sample, inconsistent_sample]).sort_index()
            logger.info(f"Balanced evaluation: {len(consistent_sample)} consistent + {len(inconsistent_sample)} inconsistent = {len(df)} rows")
        
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
        metrics = {
            'total_samples': int(total_count),
            'correct': int(correct_count),
            'accuracy': float(accuracy),
            'by_label': {}
        }
        
        for label in [0, 1]:
            subset = results_df[results_df['actual'] == label]
            if len(subset) > 0:
                label_acc = subset['correct'].mean()
                label_name = 'consistent' if label == 1 else 'inconsistent'
                logger.info(f"  {label_name}: {label_acc:.2%} ({subset['correct'].sum()}/{len(subset)})")
                metrics['by_label'][label_name] = {
                    'total': len(subset),
                    'correct': int(subset['correct'].sum()),
                    'accuracy': float(label_acc)
                }
        
        # Auto-export to JSON with experiment name
        from datetime import datetime
        import os
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build experiment name with all relevant settings
        # Start with model identifier
        if self.config.llm.use_nli:
            model_id = "nli"
        elif self.config.llm.use_local:
            # Clean up model name for filename (e.g., llama3.1:8b -> llama3.1_8b)
            model_id = self.config.llm.local_model.replace(":", "_").replace(".", "")
        else:
            model_id = self.config.llm.model.replace("-", "_").replace(".", "")
        
        exp_parts = [model_id, f"topk{self.config.retrieval.top_k}"]
        if self.config.retrieval.use_reranking:
            exp_parts.append("rerank")
        if self.config.retrieval.use_hybrid:
            exp_parts.append("hybrid")
        if self.config.llm.prompt_strategy != "base":
            exp_parts.append(self.config.llm.prompt_strategy)
        exp_parts.append(f"chunk{self.config.chunking.chunk_size}")
        
        exp_name = "_".join(exp_parts)
        filename = f"experiments/exp_{timestamp}_{exp_name}.json"
        
        os.makedirs("experiments", exist_ok=True)
        
        json_output = {
            'run_info': {
                'timestamp': timestamp,
                'rows': int(total_count),
                'books': indexed_books,
                'model': self.config.llm.model,
                'prompt_strategy': self.config.llm.prompt_strategy,
                'top_k': self.config.retrieval.top_k,
                'use_hybrid': self.config.retrieval.use_hybrid,
                'use_reranking': self.config.retrieval.use_reranking,
                'chunk_size': self.config.chunking.chunk_size,
                'chunk_strategy': self.config.chunking.strategy
            },
            'metrics': metrics,
            'predictions': results
        }
        
        with open(filename, 'w') as f:
            json.dump(json_output, f, indent=2)
        
        # Also update latest results
        with open('evaluation_results.json', 'w') as f:
            json.dump(json_output, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        
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
        """Generate results CSV for submission. Resumable from existing progress."""
        csv_path = csv_path or self.config.test_csv
        
        logger.info("=" * 60)
        logger.info("Generating Results CSV")
        logger.info("=" * 60)
        
        df = pd.read_csv(csv_path)
        logger.info(f"Total test examples: {len(df)}")
        
        self.index_books()
        
        # Check for existing results to resume
        existing_ids = set()
        results = []
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            existing_ids = set(existing_df['Story ID'].tolist())
            results = existing_df.to_dict('records')
            logger.info(f"Resuming from {len(existing_ids)} existing results")
        
        pending = df[~df['id'].isin(existing_ids)]
        logger.info(f"Processing {len(pending)} remaining examples")
        
        for idx, row in tqdm(pending.iterrows(), total=len(pending), desc="Processing"):
            book_name = self._normalize_book_name(row['book_name'])
            
            try:
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
                
                # Save progress after each row
                results_df = pd.DataFrame(results)
                results_df.to_csv(output_path, index=False)
                
            except Exception as e:
                logger.error(f"Error processing row {row['id']}: {e}")
                logger.info(f"Progress saved. Resume with same command.")
                raise
        
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
