"""End-to-end Entity Resolution Pipeline module.
Optimized for high-throughput, low-memory streaming prediction across country partitions.
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import pandas as pd
from collections import defaultdict

from src.preprocessing import fast_clean_series
from src.blocking import BlockingEngine
from src.features import extract_pair_features, FEATURE_NAMES
from src.model import MatchingModel
from src.evaluate import evaluate_macro_f05


def load_country_filtered(path: str, country: str, chunksize: int = 250000) -> pd.DataFrame:
    """Reads a large TSV in streaming chunks, filtering by country to keep RAM minimal."""
    chunks = []
    for chunk in pd.read_csv(path, sep='\t', dtype=str, chunksize=chunksize):
        match = chunk[chunk['country'] == country]
        if len(match) > 0:
            chunks.append(match)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=['entity_id', 'business_name', 'business_address', 'country'])


class EntityResolutionPipeline:
    def __init__(
        self,
        top_k: int = 15,
        min_blocking_sim: float = 0.08,
        model_path: str = "model_artifacts/lgbm_matcher.joblib"
    ):
        self.blocking_engine = BlockingEngine(top_k=top_k)
        self.model = MatchingModel()
        self.model_path = model_path

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fast vectorized string cleaning (under 5MB RAM overhead)."""
        names = df['business_name'].fillna('').tolist()
        addrs = df['business_address'].fillna('').tolist()

        df['clean_name'] = fast_clean_series(names)
        df['clean_address'] = fast_clean_series(addrs)
        return df

    def predict_test_and_generate_outputs(
        self,
        test_s1_path: str,
        test_s2_path: str,
        test_s3_path: str,
        out_matching_path: str = "output/matching_results.tsv",
        out_candidate_path: str = "output/candidate_pairs.tsv"
    ):
        """Runs memory-efficient streaming inference country-by-country on the full test set."""
        print("=" * 60, flush=True)
        print("Starting High-Throughput Test Set Inference...", flush=True)
        print("=" * 60, flush=True)

        # 1. Load full S1 test set to establish entity ordering and countries
        print(f"Reading test_source1 from {test_s1_path}...", flush=True)
        s1_full = pd.read_csv(test_s1_path, sep='\t', dtype=str)
        all_test_s1_ids = list(s1_full['entity_id'].values)
        total_s1 = len(all_test_s1_ids)
        
        # Sort countries so France is processed first (smallest to largest)
        country_counts = s1_full['country'].value_counts().to_dict()
        countries = sorted(country_counts.keys(), key=lambda c: country_counts[c])
        print(f"Total Test S1 entities: {total_s1:,} across countries: {country_counts}", flush=True)

        # Initialize output files with headers
        os.makedirs(os.path.dirname(out_matching_path), exist_ok=True)
        os.makedirs(os.path.dirname(out_candidate_path), exist_ok=True)

        with open(out_candidate_path, 'w', encoding='utf-8') as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")

        with open(out_matching_path, 'w', encoding='utf-8') as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")

        thresh = self.model.best_threshold
        print(f"Using Decision Threshold: {thresh:.3f}", flush=True)

        total_processed = 0

        # 2. Process each country partition
        for country in countries:
            print("-" * 60, flush=True)
            print(f"Processing Country Partition: '{country}' ({country_counts[country]:,} S1 entities)...", flush=True)
            
            s1_c = s1_full[s1_full['country'] == country].copy()
            c_s1_ids = list(s1_c['entity_id'].values)
            
            # Load S2 and S3 filtered by country in streaming chunks
            t0 = time.time()
            print(f"  Streaming country '{country}' records from S2 and S3...", flush=True)
            s2_c = load_country_filtered(test_s2_path, country)
            s3_c = load_country_filtered(test_s3_path, country)
            print(f"  Loaded filtered S2 ({len(s2_c):,}) and S3 ({len(s3_c):,}) in {time.time() - t0:.2f}s", flush=True)

            # Fast text preprocessing
            t0 = time.time()
            s1_c = self.prepare_dataframe(s1_c)
            s2_c = self.prepare_dataframe(s2_c)
            s3_c = self.prepare_dataframe(s3_c)
            print(f"  Preprocessing completed in {time.time() - t0:.2f}s", flush=True)

            # Blocking with high-speed Inverted Index
            t0 = time.time()
            print("  Retrieving S2 candidates with Inverted Index...", flush=True)
            s2_cands = self.blocking_engine.retrieve_candidates_for_country(s1_c, s2_c, target_prefix="S2")
            print(f"  S2 candidates retrieved in {time.time() - t0:.2f}s", flush=True)

            t0 = time.time()
            print("  Retrieving S3 candidates with Inverted Index...", flush=True)
            s3_cands = self.blocking_engine.retrieve_candidates_for_country(s1_c, s3_c, target_prefix="S3")
            print(f"  S3 candidates retrieved in {time.time() - t0:.2f}s", flush=True)

            # Fast Feature Extraction and Scoring in Streaming Chunks
            t0 = time.time()
            print(f"  Extracting features and scoring candidates for '{country}'...", flush=True)

            s1_dict = s1_c.set_index('entity_id')[['clean_name', 'clean_address']].to_dict(orient='index')
            s2_dict = s2_c.set_index('entity_id')[['clean_name', 'clean_address']].to_dict(orient='index')
            s3_dict = s3_c.set_index('entity_id')[['clean_name', 'clean_address']].to_dict(orient='index')

            with open(out_candidate_path, 'a', encoding='utf-8') as f_cand, \
                 open(out_matching_path, 'a', encoding='utf-8') as f_match:

                for chunk_start in range(0, len(c_s1_ids), 25000):
                    chunk_end = min(chunk_start + 25000, len(c_s1_ids))
                    chunk_s1_ids = c_s1_ids[chunk_start:chunk_end]

                    pair_s1_list = []
                    pair_cand_list = []
                    feature_rows = []

                    cand_map_chunk: Dict[str, List[str]] = {sid: [] for sid in chunk_s1_ids}

                    for s1_id in chunk_s1_ids:
                        s1_row = s1_dict[s1_id]
                        cands_s2 = s2_cands.get(s1_id, [])
                        cands_s3 = s3_cands.get(s1_id, [])

                        for cid, score in cands_s2:
                            if cid in s2_dict:
                                cand_map_chunk[s1_id].append(cid)
                                cand_row = s2_dict[cid]
                                feats = extract_pair_features(
                                    s1_name=s1_row['clean_name'],
                                    s1_addr=s1_row['clean_address'],
                                    cand_name=cand_row['clean_name'],
                                    cand_addr=cand_row['clean_address'],
                                    blocking_sim=score,
                                    is_s3=0.0
                                )
                                pair_s1_list.append(s1_id)
                                pair_cand_list.append(cid)
                                feature_rows.append(feats)

                        for cid, score in cands_s3:
                            if cid in s3_dict:
                                cand_map_chunk[s1_id].append(cid)
                                cand_row = s3_dict[cid]
                                feats = extract_pair_features(
                                    s1_name=s1_row['clean_name'],
                                    s1_addr=s1_row['clean_address'],
                                    cand_name=cand_row['clean_name'],
                                    cand_addr=cand_row['clean_address'],
                                    blocking_sim=score,
                                    is_s3=1.0
                                )
                                pair_s1_list.append(s1_id)
                                pair_cand_list.append(cid)
                                feature_rows.append(feats)

                    # Predict batch
                    if feature_rows:
                        X_chunk = np.array(feature_rows, dtype=np.float32)
                        probs = self.model.predict_proba(X_chunk)
                    else:
                        probs = np.array([])

                    # Group matches
                    match_map_chunk: Dict[str, List[str]] = {sid: [] for sid in chunk_s1_ids}
                    for sid, cid, prob in zip(pair_s1_list, pair_cand_list, probs):
                        if prob >= thresh:
                            match_map_chunk[sid].append(cid)

                    # Write chunk
                    for sid in chunk_s1_ids:
                        seen_c = set()
                        cands_dedup = [c for c in cand_map_chunk.get(sid, []) if not (c in seen_c or seen_c.add(c))]
                        f_cand.write(f"{sid}\t{','.join(cands_dedup)}\n")

                        seen_m = set()
                        matches_dedup = [m for m in match_map_chunk.get(sid, []) if not (m in seen_m or seen_m.add(m))]
                        f_match.write(f"{sid}\t{','.join(matches_dedup)}\n")

                    print(f"    -> Scored {chunk_end:,}/{len(c_s1_ids):,} entities for '{country}'...", flush=True)

            total_processed += len(c_s1_ids)
            print(f"  [DONE] Country '{country}' complete! Total overall: {total_processed:,}/{total_s1:,} ({total_processed/total_s1*100:.1f}%)", flush=True)

            del s1_c, s2_c, s3_c, s1_dict, s2_dict, s3_dict

        print("=" * 60, flush=True)
        print(f"[SUCCESS] All {total_processed:,} test entities processed successfully!", flush=True)
        print(f"Outputs written to:", flush=True)
        print(f"  - {out_matching_path}", flush=True)
        print(f"  - {out_candidate_path}", flush=True)
