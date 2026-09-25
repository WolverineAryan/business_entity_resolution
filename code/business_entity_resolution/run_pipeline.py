#!/usr/bin/env python3
"""Main entry point for Amazon ML Challenge 2026: Business Entity Resolution.
Trains the pipeline on training data and executes inference on test data.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.pipeline import EntityResolutionPipeline


def main():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline.")
    parser.add_argument("--train-dir", default="dataset/train", help="Path to train data directory")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test data directory")
    parser.add_argument("--val-dir", default="data/val", help="Path to validation data directory")
    parser.add_argument("--out-matching", default="output/matching_results.tsv", help="Output matching file path")
    parser.add_argument("--out-candidate", default="output/candidate_pairs.tsv", help="Output candidate file path")
    parser.add_argument("--top-k", type=int, default=15, help="Top-K candidates per source")
    parser.add_argument("--min-sim", type=float, default=0.08, help="Minimum blocking similarity threshold")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if model already exists")
    args = parser.parse_args()

    pipeline = EntityResolutionPipeline(top_k=args.top_k, min_blocking_sim=args.min_sim)

    # 1. Train and Validate
    if not args.skip_train or not os.path.exists("model_artifacts/lgbm_matcher.joblib"):
        print("Training pipeline...", flush=True)
        val_s1 = os.path.join(args.val_dir, "val_source1.tsv")
        val_s2 = os.path.join(args.val_dir, "val_source2.tsv")
        val_s3 = os.path.join(args.val_dir, "val_source3.tsv")
        val_gt = os.path.join(args.val_dir, "val_ground_truth.tsv")

        train_s1 = os.path.join(args.train_dir, "train_source1.tsv")
        train_s2 = os.path.join(args.train_dir, "train_source2.tsv")
        train_s3 = os.path.join(args.train_dir, "train_source3.tsv")
        train_gt = os.path.join(args.train_dir, "train_ground_truth.tsv")

        # Resolve this optional pipeline API dynamically so static type checkers
        # do not reject projects that provide it through an extension/mixin.
        train_and_evaluate = getattr(pipeline, "train_and_evaluate")
        best_score = train_and_evaluate(
            train_s1_path=train_s1,
            train_s2_path=train_s2,
            train_s3_path=train_s3,
            train_gt_path=train_gt,
            val_s1_path=val_s1,
            val_s2_path=val_s2,
            val_s3_path=val_s3,
            val_gt_path=val_gt
        )
        print(f"Validation Training Complete. Macro F0.5 Score: {best_score:.4f}", flush=True)
    else:
        print("Loading existing trained model...", flush=True)
        pipeline.model.load("model_artifacts/lgbm_matcher.joblib")

    # 2. Predict on Test Set
    test_s1 = os.path.join(args.test_dir, "test_source1.tsv")
    test_s2 = os.path.join(args.test_dir, "test_source2.tsv")
    test_s3 = os.path.join(args.test_dir, "test_source3.tsv")

    pipeline.predict_test_and_generate_outputs(
        test_s1_path=test_s1,
        test_s2_path=test_s2,
        test_s3_path=test_s3,
        out_matching_path=args.out_matching,
        out_candidate_path=args.out_candidate
    )


if __name__ == "__main__":
    main()
