# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

## Overview
This repository contains the complete, self-contained, reproducible pipeline for the Amazon ML Challenge 2026: Business Entity Resolution Challenge.

## Directory Structure
```text
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Locality-aware string cleaners & legal suffix normalizer
│   ├── blocking.py         # Multi-tier candidate generation & sparse TF-IDF indexer
│   ├── features.py         # RapidFuzz string, token, and numeric feature extractors
│   ├── model.py            # LightGBM classifier & Macro F0.5 threshold optimizer
│   ├── evaluate.py         # Macro F0.5, Precision, Recall, and singleton evaluation
│   └── pipeline.py         # End-to-end pipeline coordinator
├── run_pipeline.py         # Master CLI runner
├── requirements.txt        # Pinned dependencies
└── README.md               # Reproduction guide
```

## Setup & Requirements
Python 3.10+ is recommended. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Pipeline End-to-End

To execute the complete pipeline (training on train/val data and generating submission files for the test set):

```bash
python run_pipeline.py \
    --train-dir ../../train \
    --test-dir ../../test \
    --val-dir ../../data/val \
    --out-matching ../../output/matching_results.tsv \
    --out-candidate ../../output/candidate_pairs.tsv
```

## Validating Output Submission
To verify that the generated outputs strictly conform to all official evaluation rules:
```bash
python ../../validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../test
```
