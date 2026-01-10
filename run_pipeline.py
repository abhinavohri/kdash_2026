#!/usr/bin/env python3
"""
KDSH Track A Pipeline Entry Point.

Usage:
    python run_pipeline.py --evaluate --rows 10
    python run_pipeline.py --generate-results
    python run_pipeline.py --index-only
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import PipelineConfig
from src.pipeline import KDSHPipeline
from src.logger import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="KDSH Track A Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --evaluate --rows 10    # Evaluate on 10 training examples
  python run_pipeline.py --evaluate --rows 50    # Evaluate on 50 training examples
  python run_pipeline.py --generate-results      # Generate submission results.csv
  python run_pipeline.py --index-only            # Only index books, no evaluation
  python run_pipeline.py --reindex               # Force reindex all books
        """
    )
    
    # Action selection
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate on training data (uses train.csv)"
    )
    action.add_argument(
        "--generate-results",
        action="store_true",
        help="Generate results.csv for submission (uses test.csv)"
    )
    action.add_argument(
        "--index-only",
        action="store_true",
        help="Only index books, no evaluation"
    )
    
    # Configuration
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Number of rows to evaluate (default: 10)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model to use (e.g., gemini-2.5-flash, gemini-2.5-pro)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model to use (default: gemini-embedding-001)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve (default: 10)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in words (default: 1000)"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reindex all books"
    )
    parser.add_argument(
        "--one-book",
        action="store_true",
        help="Index only one book (smaller dataset for faster testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.csv",
        help="Output file path for results (default: results.csv)"
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable hybrid search (embedding + BM25)"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable reranking with cross-encoder"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        choices=["base", "few_shot", "cot", "claim_extraction"],
        default="base",
        help="Prompting strategy: base, few_shot (examples), cot (chain-of-thought), claim_extraction"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local Ollama model instead of Gemini API"
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default="llama3.2",
        help="Ollama model to use (default: llama3.2). Pull with: ollama pull <model>"
    )
    parser.add_argument(
        "--nli",
        action="store_true",
        help="Use NLI model for classification instead of LLM (runs locally, no API needed)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Build configuration
    config = PipelineConfig()
    config.eval_rows = args.rows
    
    # Override model if specified
    if args.model:
        config.llm.model = args.model
        logger.info(f"Using LLM model: {args.model}")
    
    if args.embedding_model:
        config.embedding.model = args.embedding_model
        logger.info(f"Using embedding model: {args.embedding_model}")
    
    config.retrieval.top_k = args.top_k
    config.chunking.chunk_size = args.chunk_size
    
    # One book mode - use smaller book for faster testing
    if args.one_book:
        config.books_filter = ["In search of the castaways"]
        logger.info("One-book mode: indexing only 'In search of the castaways'")
    
    # Hybrid search and reranking
    if args.hybrid:
        config.retrieval.use_hybrid = True
        logger.info("Hybrid search enabled (embedding + BM25)")
    
    if args.rerank:
        config.retrieval.use_reranking = True
        logger.info("Reranking enabled (cross-encoder)")
    
    # Prompting strategy
    if args.prompt != "base":
        config.llm.prompt_strategy = args.prompt
        logger.info(f"Using prompt strategy: {args.prompt}")
    
    # Local model (Ollama)
    if args.local:
        config.llm.use_local = True
        config.llm.local_model = args.local_model
        logger.info(f"Using local model: {args.local_model} (Ollama)")
    
    # NLI-based classification
    if args.nli:
        config.llm.use_nli = True
        logger.info("Using NLI model for classification (DeBERTa)")
    
    # Create pipeline
    pipeline = KDSHPipeline(config)
    
    try:
        if args.index_only:
            # Just index books
            pipeline.index_books(force_reindex=args.reindex)
            logger.info("Indexing complete!")
            
        elif args.evaluate:
            # Run evaluation
            results_df = pipeline.run_evaluation(num_rows=args.rows)
            
            # Save evaluation results
            eval_output = f"eval_results_{args.rows}rows.csv"
            results_df.to_csv(eval_output, index=False)
            logger.info(f"Evaluation results saved to {eval_output}")
            
        elif args.generate_results:
            # Generate submission file
            output_path = pipeline.generate_results_csv(output_path=args.output)
            logger.info(f"Submission file generated: {output_path}")
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
