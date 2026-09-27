"""
Amazon ML Challenge 2026 — Calibrated Pure Ensemble Production Pipeline
========================================================================
Architecture:
1. Multi-Core Vectorized Flat Inverted Index Blocking (Top-35 candidates, ~98% recall)
2. 35-Feature Discriminative Engine (Bug-Free Domain Matching, French Legal Normalization, Year Filtering)
3. Tri-Model Gradient Boosting Ensemble (LightGBM 0.40 + CatBoost 0.35 + XGBoost 0.25)
4. Calibrated Dynamic Ratio Decision Rule (Sweet Spot for Macro F0.5):
   - France: base_p=0.80, rel_ratio=0.77, top_k=35
   - India:  base_p=0.81, rel_ratio=0.76, top_k=35
   - US:     base_p=0.81, rel_ratio=0.77, top_k=35
   - Match Cap: Up to 12 matches (accommodating multi-branch enterprise clusters)
5. Full compliance verification via validate_submission.py --check-ids
"""

import os
import sys
import gc
import time
import re
import unicodedata
import argparse
import joblib
import subprocess
import multiprocessing as mp
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from rapidfuzz import fuzz, distance

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('code/business_entity_resolution'))

from src.blocking import BlockingEngine, extract_index_tokens
from src.features_v3 import (
    precompute_entity_v3, extract_features_v3, get_char_ngrams, FEATURE_NAMES_V3
)

# Calibrated Pure Ensemble Configurations validated for Max Macro F0.5
CONFIGS = {
    'France': {'base_p': 0.80, 'rel_ratio': 0.77, 'top_k': 35},
    'India':  {'base_p': 0.81, 'rel_ratio': 0.76, 'top_k': 35},
    'US':     {'base_p': 0.81, 'rel_ratio': 0.77, 'top_k': 35},
}

def run_country_pipeline(country: str, ensemble_data: dict, s1_c_df: pd.DataFrame, test_s2_path: str, test_s3_path: str):
    cfg = CONFIGS[country]
    base_p = cfg['base_p']
    rel_ratio = cfg['rel_ratio']
    top_k = cfg['top_k']

    lgb_m = ensemble_data['lgb_model']
    cat_m = ensemble_data['cat_model']
    xgb_m = ensemble_data['xgb_model']
    w_lgb, w_cat, w_xgb = ensemble_data['weights']

    print("\n" + "=" * 80, flush=True)
    print(f"PROCESSING COUNTRY: {country.upper()} ({len(s1_c_df):,} Source-1 Queries)", flush=True)
    print(f"Configuration: base_p={base_p:.2f}, rel_ratio={rel_ratio:.2f}, top_k={top_k}", flush=True)
    print("=" * 80, flush=True)
    t_start = time.time()

    # 1. Load S2 and S3 for this country
    t0 = time.time()
    print(f"[{country}] Loading target records from test_source2.tsv & test_source3.tsv...", flush=True)
    cols = ['entity_id', 'business_name', 'business_address', 'country']
    s2_c = pd.read_csv(test_s2_path, sep='\t', dtype=str, usecols=cols)
    s2_c = s2_c[s2_c['country'] == country]

    s3_c = pd.read_csv(test_s3_path, sep='\t', dtype=str, usecols=cols)
    s3_c = s3_c[s3_c['country'] == country]

    target_df = pd.concat([s2_c, s3_c], ignore_index=True)
    del s2_c, s3_c
    gc.collect()
    print(f"[{country}] Loaded {len(target_df):,} target records in {time.time()-t0:.2f}s", flush=True)

    # 2. Multi-Core Vectorized Candidate Generation (Top-35)
    t0 = time.time()
    print(f"[{country}] Building Vectorized Inverted Index & Querying Candidates (Top-{top_k})...", flush=True)
    num_workers = min(12, max(1, mp.cpu_count()))
    blocking_engine = BlockingEngine(top_k=top_k, max_posting_len=20000, num_workers=num_workers)
    cand_map_raw = blocking_engine.retrieve_candidates_for_country(s1_c_df, target_df)
    cand_map = {sid: [cid for cid, _ in c_list] for sid, c_list in cand_map_raw.items()}
    del cand_map_raw, blocking_engine
    gc.collect()
    print(f"[{country}] Candidate generation completed in {time.time()-t0:.2f}s", flush=True)

    # 3. Precompute candidate lookups
    t0 = time.time()
    all_cands_needed = set()
    for clist in cand_map.values():
        all_cands_needed.update(clist)
    print(f"[{country}] Distinct candidates referenced: {len(all_cands_needed):,}", flush=True)

    target_filtered = target_df[target_df['entity_id'].isin(all_cands_needed)]
    cand_lookup: Dict[str, Tuple] = {
        eid: precompute_entity_v3(n, a)
        for eid, n, a in zip(
            target_filtered['entity_id'],
            target_filtered['business_name'].fillna(''),
            target_filtered['business_address'].fillna('')
        )
    }
    s1_lookup: Dict[str, Tuple] = {
        sid: precompute_entity_v3(n, a)
        for sid, n, a in zip(
            s1_c_df['entity_id'],
            s1_c_df['business_name'].fillna(''),
            s1_c_df['business_address'].fillna('')
        )
    }
    s1_ngrams = {sid: get_char_ngrams(tup[0], 3) for sid, tup in s1_lookup.items()}

    del target_df, target_filtered, all_cands_needed
    gc.collect()
    print(f"[{country}] Precomputed lookups in {time.time()-t0:.2f}s", flush=True)

    # 4. Stream Batch Scoring with Pure Tri-Model Ensemble
    t0 = time.time()
    batch_size = 15000
    s1_ids = list(s1_c_df['entity_id'].values)

    match_results = {}
    candidate_results = {}
    total_evaluated_pairs = 0
    total_accepted_matches = 0
    empty_matches = 0

    print(f"[{country}] Scoring candidate pairs with Tri-Model Ensemble (LGB+Cat+XGB)...", flush=True)
    for b_idx in range(0, len(s1_ids), batch_size):
        chunk_s1 = s1_ids[b_idx: b_idx + batch_size]
        pair_s1, pair_cid, pair_feats = [], [], []

        for sid in chunk_s1:
            s1_tup = s1_lookup[sid]
            s1_ng = s1_ngrams[sid]
            cands = cand_map.get(sid, [])
            candidate_results[sid] = cands

            for cid in cands:
                c_tup = cand_lookup.get(cid)
                if not c_tup: continue

                # Fast pre-filter dead candidates (no common tokens and very low fuzzy)
                if not c_tup[11]: # not indic
                    if len(s1_tup[2].intersection(c_tup[2])) == 0 and len(s1_tup[8].intersection(c_tup[8])) <= 1:
                        if fuzz.token_set_ratio(s1_tup[0], c_tup[0]) < 28 and fuzz.token_set_ratio(s1_tup[1], c_tup[1]) < 30:
                            continue

                is_s3 = 1.0 if cid.startswith('S3-') else 0.0
                feats = extract_features_v3(s1_tup, s1_ng, c_tup, is_s3)

                pair_s1.append(sid)
                pair_cid.append(cid)
                pair_feats.append(feats)

        total_evaluated_pairs += len(pair_feats)

        if pair_feats:
            X_b = np.array(pair_feats, dtype=np.float32)
            p_lgb = lgb_m.predict_proba(X_b)[:, 1]
            p_cat = cat_m.predict_proba(X_b)[:, 1]
            p_xgb = xgb_m.predict_proba(X_b)[:, 1]
            probs = w_lgb * p_lgb + w_cat * p_cat + w_xgb * p_xgb
        else:
            probs = np.array([])

        chunk_cand_scores: Dict[str, List[Tuple[str, float]]] = {sid: [] for sid in chunk_s1}
        for sid, cid, prob in zip(pair_s1, pair_cid, probs):
            chunk_cand_scores[sid].append((cid, float(prob)))

        # Pure Calibrated Decision Rule
        for sid in chunk_s1:
            c_list = chunk_cand_scores[sid]
            if not c_list:
                match_results[sid] = []
                empty_matches += 1
                continue

            sorted_c = sorted(c_list, key=lambda x: x[1], reverse=True)
            top_prob = sorted_c[0][1]

            if top_prob < base_p:
                match_results[sid] = []
                empty_matches += 1
                continue

            accepted = []
            for cid, prob in sorted_c:
                if prob < base_p: break
                if prob < top_prob * rel_ratio: break
                accepted.append(cid)
                if len(accepted) >= 12: break

            if not accepted:
                match_results[sid] = []
                empty_matches += 1
            else:
                seen = set()
                dedup = [m for m in accepted if not (m in seen or seen.add(m))]
                match_results[sid] = dedup
                total_accepted_matches += len(dedup)

        processed = min(b_idx + batch_size, len(s1_ids))
        rate = total_evaluated_pairs / max(1e-5, time.time() - t0)
        print(f"[{country}] Scored {processed:,}/{len(s1_ids):,} entities ({rate:,.0f} pairs/sec)...", flush=True)

    del cand_lookup, s1_lookup, s1_ngrams, cand_map
    gc.collect()

    elapsed = time.time() - t_start
    print(f"\n[{country} PIPELINE COMPLETED in {elapsed/60:.2f} mins]", flush=True)
    print(f"  Evaluated Candidate Pairs: {total_evaluated_pairs:,}", flush=True)
    print(f"  Total Accepted Matches:   {total_accepted_matches:,}", flush=True)
    print(f"  Singletons (0 match):     {empty_matches:,} ({empty_matches/len(s1_ids)*100:.2f}%)", flush=True)
    print(f"  Avg Matches/Non-Empty:    {total_accepted_matches/max(1, len(s1_ids)-empty_matches):.2f}", flush=True)

    return match_results, candidate_results

def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 — Calibrated Pure Ensemble Production Pipeline")
    parser.add_argument('--test-dir', type=str, default='dataset/test', help="Directory containing test_source1/2/3.tsv")
    parser.add_argument('--model-path', type=str, default='model_artifacts/ensemble_v3.joblib', help="Path to ensemble artifact")
    parser.add_argument('--output-dir', type=str, default='output', help="Directory to save submission files")
    parser.add_argument('--skip-validation', action='store_true', help="Skip validate_submission.py check")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_matching = os.path.join(args.output_dir, 'matching_results.tsv')
    out_candidate = os.path.join(args.output_dir, 'candidate_pairs.tsv')

    print("=" * 85, flush=True)
    print("AMAZON ML CHALLENGE 2026 — CALIBRATED PURE ENSEMBLE PRODUCTION PIPELINE", flush=True)
    print("=" * 85, flush=True)

    t_global = time.time()

    # 1. Load Ensemble Artifact
    print(f"Loading Tri-Model Ensemble from {args.model_path}...", flush=True)
    ensemble_data = joblib.load(args.model_path)
    print(f"Loaded ensemble with weights: {ensemble_data['weights']} (Validation F0.5: {ensemble_data['val_score']:.5f})", flush=True)

    # 2. Auto-Detect and Load test_source1.tsv
    actual_test_dir = args.test_dir
    if not os.path.exists(os.path.join(actual_test_dir, 'test_source1.tsv')):
        # Auto-scan /kaggle/input and common paths
        found = False
        search_roots = ['/kaggle/input', 'dataset/test', 'dataset', '.']
        for sroot in search_roots:
            if os.path.exists(sroot):
                for root, dirs, files in os.walk(sroot):
                    if 'test_source1.tsv' in files:
                        actual_test_dir = root
                        found = True
                        break
                if found: break

    s1_path = os.path.join(actual_test_dir, 'test_source1.tsv')
    print(f"Loading reference test entities from {s1_path}...", flush=True)
    t0 = time.time()
    s1_full = pd.read_csv(s1_path, sep='\t', dtype=str)
    all_s1_ordered = list(s1_full['entity_id'].values)
    print(f"Loaded {len(all_s1_ordered):,} test entities in {time.time()-t0:.2f}s", flush=True)

    all_matches = {}
    all_candidates = {}

    test_s2_path = os.path.join(actual_test_dir, 'test_source2.tsv')
    test_s3_path = os.path.join(actual_test_dir, 'test_source3.tsv')

    for country in ['France', 'India', 'US']:
        s1_c_df = s1_full[s1_full['country'] == country][['entity_id', 'business_name', 'business_address']].copy()
        c_matches, c_cands = run_country_pipeline(
            country=country,
            ensemble_data=ensemble_data,
            s1_c_df=s1_c_df,
            test_s2_path=test_s2_path,
            test_s3_path=test_s3_path
        )
        all_matches.update(c_matches)
        all_candidates.update(c_cands)
        del s1_c_df, c_matches, c_cands
        gc.collect()

    del s1_full
    gc.collect()

    # 3. Write candidate_pairs.tsv
    print("\n" + "=" * 85, flush=True)
    print(f"Writing {out_candidate} in exact test_source1 ordering...", flush=True)
    print("=" * 85, flush=True)
    t0 = time.time()
    with open(out_candidate, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1_ordered:
            cands = all_candidates.get(sid, [])
            f.write(f"{sid}\t{','.join(cands)}\n")
    print(f"Written {len(all_s1_ordered):,} rows to {out_candidate} in {time.time()-t0:.2f}s ({os.path.getsize(out_candidate)/1e6:.2f} MB)", flush=True)

    # 4. Write matching_results.tsv
    print("\n" + "=" * 85, flush=True)
    print(f"Writing {out_matching} in exact test_source1 ordering...", flush=True)
    print("=" * 85, flush=True)
    t0 = time.time()
    total_written_matches = 0
    total_singletons = 0
    with open(out_matching, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ordered:
            matches = all_matches.get(sid, [])
            f.write(f"{sid}\t{','.join(matches)}\n")
            if matches:
                total_written_matches += len(matches)
            else:
                total_singletons += 1
    print(f"Written {len(all_s1_ordered):,} rows to {out_matching} in {time.time()-t0:.2f}s ({os.path.getsize(out_matching)/1e6:.2f} MB)", flush=True)
    print(f"  Total Matches Written:      {total_written_matches:,}", flush=True)
    print(f"  Total Singletons (0 match): {total_singletons:,} ({total_singletons/len(all_s1_ordered)*100:.2f}%)", flush=True)
    print(f"  Avg Matches / Non-Empty:    {total_written_matches/max(1, len(all_s1_ordered)-total_singletons):.2f}", flush=True)

    # 5. Validation Check
    if not args.skip_validation and os.path.exists('validate_submission.py'):
        print("\n" + "=" * 85, flush=True)
        print("Running Official Submission Validator with --check-ids...", flush=True)
        print("=" * 85, flush=True)
        val_cmd = [
            sys.executable,
            'validate_submission.py',
            '--matching', out_matching,
            '--candidate', out_candidate,
            '--test-dir', args.test_dir,
            '--check-ids'
        ]
        res = subprocess.run(val_cmd, capture_output=True, text=True)
        print(res.stdout, flush=True)
        if res.stderr:
            print("Validator STDERR:", res.stderr, flush=True)

    total_time = (time.time() - t_global) / 60
    print(f"\n>>> PIPELINE EXECUTION FINISHED IN {total_time:.2f} MINUTES! <<<\n", flush=True)

if __name__ == '__main__':
    main()
