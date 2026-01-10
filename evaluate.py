#!/usr/bin/env python3
"""
Quick evaluation script for KDSH Pipeline.

Simplified evaluation runner for rapid testing.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import PipelineConfig
from src.pipeline import KDSHPipeline
from src.logger import logger


def main():
    parser = argparse.ArgumentParser(description="Evaluate KDSH Pipeline")
    parser.add_argument("--rows", type=int, default=10, help="Number of rows to evaluate")
    parser.add_argument("--model", type=str, default=None, help="LLM model to use")
    args = parser.parse_args()
    
    # Configure
    config = PipelineConfig()
    config.eval_rows = args.rows
    
    if args.model:
        config.llm.model = args.model
    
    # Run
    pipeline = KDSHPipeline(config)
    
    try:
        results = pipeline.run_evaluation(num_rows=args.rows)
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(results)}")
        print(f"Correct: {results['correct'].sum()}")
        print(f"Accuracy: {results['correct'].mean():.2%}")
        
        # Show incorrect predictions
        incorrect = results[~results['correct']]
        if len(incorrect) > 0:
            print(f"\nIncorrect predictions ({len(incorrect)}):")
            for _, row in incorrect.iterrows():
                print(f"  - {row['character']}: predicted {row['prediction']}, actual {row['actual']}")
        
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
