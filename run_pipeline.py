#!/usr/bin/env python3
"""KDSH Track A Pipeline Entry Point."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import PipelineConfig
from src.pipeline import KDSHPipeline
from src.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="KDSH Track A Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --evaluate --rows 10
  python run_pipeline.py --generate-results --input Dataset/test.csv
  python run_pipeline.py --index-only
        """
    )
    
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--evaluate", action="store_true", help="Evaluate on training data")
    action.add_argument("--generate-results", action="store_true", help="Generate results.csv for submission")
    action.add_argument("--index-only", action="store_true", help="Only index books")
    action.add_argument("--backfill-metadata", action="store_true", help="Backfill metadata for existing chunks")
    action.add_argument("--generate-backstories", action="store_true", help="Generate canonical backstories for all characters")
    
    parser.add_argument("--rows", type=int, default=10, help="Number of rows to evaluate")
    parser.add_argument("--model", type=str, default=None, help="LLM model to use")
    parser.add_argument("--embedding-model", type=str, default=None, help="Embedding model to use")
    parser.add_argument("--top-k", type=int, default=10, help="Number of chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in words")
    parser.add_argument("--reindex", action="store_true", help="Force reindex all books")
    parser.add_argument("--one-book", action="store_true", help="Index only one book for faster testing")
    parser.add_argument("--output", type=str, default="results.csv", help="Output file path")
    parser.add_argument("--input", type=str, default=None, help="Input CSV file path")
    parser.add_argument("--hybrid", action="store_true", help="Enable hybrid search (embedding + BM25)")
    parser.add_argument("--rerank", action="store_true", help="Enable reranking with cross-encoder")
    parser.add_argument("--prompt", type=str, choices=["base", "few_shot", "cot", "claim_extraction", "track_a", "conservative", "optimistic", "evidence_dossier"], default="conservative")
    parser.add_argument("--local", action="store_true", help="Use local Ollama model")
    parser.add_argument("--local-model", type=str, default="llama3.2", help="Ollama model name")
    parser.add_argument("--nli", action="store_true", help="Use NLI model for classification")
    parser.add_argument("--use-hybrid", action="store_true", help="Use hybrid NLI + LLM classification")
    parser.add_argument("--nli-threshold", type=float, default=0.9, help="NLI confidence threshold")
    parser.add_argument("--min-contradictions", type=int, default=2, help="Min contradictions for INCONSISTENT")
    parser.add_argument("--optimistic", action="store_true", help="Optimistic mode: default to consistent unless specific contradiction proof found")
    parser.add_argument("--use-canonical", action="store_true", help="Use pre-generated canonical backstories for classification")
    parser.add_argument("--canonical-mode", type=str, choices=["embedding", "llm"], default="embedding", help="Canonical mode: 'embedding' (fast, no LLM) or 'llm' (accurate)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = PipelineConfig()
    config.eval_rows = args.rows
    
    if args.model:
        config.llm.model = args.model
        logger.info(f"Using LLM model: {args.model}")
    
    if args.embedding_model:
        config.embedding.model = args.embedding_model
        logger.info(f"Using embedding model: {args.embedding_model}")
    
    config.retrieval.top_k = args.top_k
    config.chunking.chunk_size = args.chunk_size
    
    if args.one_book:
        config.books_filter = ["In search of the castaways"]
        logger.info("One-book mode: indexing only 'In search of the castaways'")
    
    if args.hybrid:
        config.retrieval.use_hybrid = True
        logger.info("Hybrid search enabled")
    
    if args.rerank:
        config.retrieval.use_reranking = True
        logger.info("Reranking enabled")
    
    if args.prompt != "base":
        config.llm.prompt_strategy = args.prompt
        logger.info(f"Using prompt strategy: {args.prompt}")
    
    if args.local:
        config.llm.use_local = True
        config.llm.local_model = args.local_model
        logger.info(f"Using local model: {args.local_model}")
    
    if args.nli:
        config.llm.use_nli = True
        logger.info("Using NLI model for classification")
    
    if args.use_hybrid:
        config.llm.use_hybrid = True
        config.llm.nli_confidence_threshold = args.nli_threshold
        config.llm.min_contradictions = args.min_contradictions
        logger.info(f"Using hybrid NLI + LLM classification")
    
    if args.optimistic:
        config.llm.optimistic = True
        config.llm.prompt_strategy = "optimistic"  # Auto-set prompt strategy
        logger.info("Optimistic mode: defaulting to consistent unless contradiction proof found")
    
    if hasattr(args, 'use_canonical') and args.use_canonical:
        config.llm.use_canonical = True
        if hasattr(args, 'canonical_mode'):
            config.llm.canonical_mode = args.canonical_mode
        logger.info(f"Canonical backstory mode: {config.llm.canonical_mode}")
    
    pipeline = KDSHPipeline(config)
    
    try:
        if args.index_only:
            pipeline.index_books(force_reindex=args.reindex)
            logger.info("Indexing complete!")
            
        elif args.backfill_metadata:
            pipeline.backfill_metadata()
        
        elif hasattr(args, 'generate_backstories') and args.generate_backstories:
            pipeline.generate_canonical_backstories()
            logger.info("Canonical backstory generation complete!")
            
        elif args.evaluate:
            results_df = pipeline.run_evaluation(num_rows=args.rows)
            eval_output = f"eval_results_{args.rows}rows.csv"
            results_df.to_csv(eval_output, index=False)
            logger.info(f"Evaluation results saved to {eval_output}")
            
        elif args.generate_results:
            output_path = pipeline.generate_results_csv(csv_path=args.input, output_path=args.output)
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
