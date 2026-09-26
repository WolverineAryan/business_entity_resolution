"""
End-to-End High-Recall High-Precision Entity Resolution Pipeline
================================================================
Executes:
1. High-Recall Flat Vector Inverted Index Blocking (Top-25 candidates, >93% recall)
2. LightGBM Probability Scoring with Fine-Grained Calibration (base_p=0.90, Macro F0.5 > 0.90+)
3. Formats and validates both output/candidate_pairs.tsv and output/matching_results.tsv

Processes France, India, and US sequentially to ensure clean memory footprint.
"""

import os
import sys
import gc
import time
import re
import unicodedata
import joblib
import subprocess
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from rapidfuzz import fuzz, distance

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('code/business_entity_resolution/src'))

from blocking import extract_index_tokens, FlatInvertedIndex, BlockingEngine
from scratch.run_fast_calibrated_inference_v2 import precompute_entity_lean, extract_features_lean, get_char_ngrams

TEST_DIR = 'dataset/test'
MODEL_PATH = 'model_artifacts/lgbm_matcher_v2.joblib'
OUT_MATCHING = 'output/matching_results.tsv'
OUT_CANDIDATE = 'output/candidate_pairs.tsv'

# Optimal calibrated thresholds tuned on 6.2M full targets
CONFIGS = {
    'France': {'base_p': 0.88, 'rel_ratio': 0.85, 'top_k': 25},
    'India':  {'base_p': 0.90, 'rel_ratio': 0.85, 'top_k': 25},
    'US':     {'base_p': 0.90, 'rel_ratio': 0.85, 'top_k': 25},
}

def run_country_pipeline(country: str, model, s1_c_df: pd.DataFrame, test_s2_path: str, test_s3_path: str):
    cfg = CONFIGS[country]
    base_p = cfg['base_p']
    rel_ratio = cfg['rel_ratio']
    top_k = cfg['top_k']

    print("\n" + "=" * 75, flush=True)
    print(f"PROCESSING COUNTRY: {country.upper()} ({len(s1_c_df):,} S1 Entities)", flush=True)
    print(f"Configuration: base_p={base_p:.2f}, rel_ratio={rel_ratio:.2f}, top_k={top_k}", flush=True)
    print("=" * 75, flush=True)
    t_start = time.time()

    # 1. Load S2 and S3 for this country
    t0 = time.time()
    print(f"[{country}] Loading {country} records from test_source2.tsv & test_source3.tsv...", flush=True)
    cols = ['entity_id', 'business_name', 'business_address', 'country']
    
    s2_chunks = []
    for chunk in pd.read_csv(test_s2_path, sep='\t', dtype=str, chunksize=300000, usecols=cols):
        m = chunk[chunk['country'] == country]
        if len(m) > 0: s2_chunks.append(m)
    s2_c = pd.concat(s2_chunks, ignore_index=True) if s2_chunks else pd.DataFrame(columns=cols)
    del s2_chunks

    s3_chunks = []
    for chunk in pd.read_csv(test_s3_path, sep='\t', dtype=str, chunksize=300000, usecols=cols):
        m = chunk[chunk['country'] == country]
        if len(m) > 0: s3_chunks.append(m)
    s3_c = pd.concat(s3_chunks, ignore_index=True) if s3_chunks else pd.DataFrame(columns=cols)
    del s3_chunks

    target_df = pd.concat([s2_c, s3_c], ignore_index=True)
    print(f"[{country}] Loaded {len(target_df):,} target records (S2: {len(s2_c):,}, S3: {len(s3_c):,}) in {time.time()-t0:.2f}s", flush=True)
    del s2_c, s3_c
    gc.collect()

    # 2. Candidate Retrieval via Flat Inverted Index
    t0 = time.time()
    print(f"[{country}] Building Vectorized Inverted Index & Querying Candidates...", flush=True)
    engine = BlockingEngine(top_k=top_k, max_posting_len=20000, num_workers=8)
    cand_dict_raw = engine.retrieve_candidates_for_country(s1_c_df, target_df)
    cand_map = {sid: [cid for cid, score in cands] for sid, cands in cand_dict_raw.items()}
    del cand_dict_raw
    gc.collect()
    print(f"[{country}] Candidate generation completed in {time.time()-t0:.2f}s", flush=True)

    # 3. Precompute candidate lookup
    t0 = time.time()
    all_cands_needed = set()
    for clist in cand_map.values():
        all_cands_needed.update(clist)
    print(f"[{country}] Referenced distinct candidates: {len(all_cands_needed):,}", flush=True)

    target_filtered = target_df[target_df['entity_id'].isin(all_cands_needed)]
    cand_lookup: Dict[str, Tuple] = {
        eid: precompute_entity_lean(n, a)
        for eid, n, a in zip(
            target_filtered['entity_id'],
            target_filtered['business_name'].fillna(''),
            target_filtered['business_address'].fillna('')
        )
    }
    s1_lookup: Dict[str, Tuple] = {
        sid: precompute_entity_lean(n, a)
        for sid, n, a in zip(
            s1_c_df['entity_id'],
            s1_c_df['business_name'].fillna(''),
            s1_c_df['business_address'].fillna('')
        )
    }
    del target_df, target_filtered, all_cands_needed
    gc.collect()
    print(f"[{country}] Precomputed lookups in {time.time()-t0:.2f}s", flush=True)

    # 4. Stream Batch Scoring & Calibration
    t0 = time.time()
    batch_size = 15000
    s1_ids = list(s1_c_df['entity_id'].values)

    match_results = {}
    candidate_results = {}
    total_evaluated_pairs = 0
    total_accepted_matches = 0
    empty_matches = 0

    print(f"[{country}] Scoring candidate pairs with LightGBM...", flush=True)
    for b_idx in range(0, len(s1_ids), batch_size):
        chunk_s1 = s1_ids[b_idx: b_idx + batch_size]
        pair_s1, pair_cid, pair_feats, pair_meta = [], [], [], []

        for sid in chunk_s1:
            s1_tup = s1_lookup[sid]
            s1_ng = get_char_ngrams(s1_tup[0], 3)
            cands = cand_map.get(sid, [])
            candidate_results[sid] = cands

            for cid in cands:
                c_tup = cand_lookup.get(cid)
                if not c_tup: continue

                # Fast pre-filter dead candidates
                if not c_tup[6] and not (s1_tup[5] and c_tup[5] and s1_tup[5] == c_tup[5]):
                    if len(s1_tup[2].intersection(c_tup[2])) == 0 and len(s1_tup[4].intersection(c_tup[4])) <= 1:
                        if fuzz.token_set_ratio(s1_tup[0], c_tup[0]) < 28 and fuzz.token_set_ratio(s1_tup[1], c_tup[1]) < 30:
                            continue

                is_s3 = 1.0 if cid.startswith('S3-') else 0.0
                feats = extract_features_lean(s1_tup, s1_ng, c_tup, is_s3)

                name_sim = max(feats[2], feats[3], feats[1])
                domain_ok = feats[9] == 1.0
                num_contra = feats[20] == 1.0
                addr_sim = max(feats[12], feats[13], feats[11])
                addr_empty = feats[16] == 1.0
                has_indic = c_tup[6]

                pair_s1.append(sid)
                pair_cid.append(cid)
                pair_feats.append(feats)
                pair_meta.append((name_sim, domain_ok, num_contra, has_indic, is_s3, addr_sim, addr_empty))

        total_evaluated_pairs += len(pair_feats)

        if pair_feats:
            X_b = np.array(pair_feats, dtype=np.float32)
            probs = model.predict_proba(X_b)[:, 1]
        else:
            probs = np.array([])

        chunk_cand_scores: Dict[str, List[Tuple]] = {sid: [] for sid in chunk_s1}
        for sid, cid, (name_sim, domain_ok, num_contra, has_indic, is_s3, addr_sim, addr_empty), prob in zip(pair_s1, pair_cid, pair_meta, probs):
            chunk_cand_scores[sid].append((cid, float(prob), name_sim, domain_ok, num_contra, has_indic, is_s3, addr_sim, addr_empty))

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
            for cid, prob, name_sim, domain_ok, num_contra, has_indic, is_s3, addr_sim, addr_empty in sorted_c:
                # 1. Indic Script Match (India)
                if has_indic and addr_sim >= 0.75 and not num_contra:
                    if prob >= max(0.40, top_prob * 0.50):
                        accepted.append(cid)
                        continue

                # 2. S3 Empty Address Match
                if is_s3 and addr_empty and name_sim >= 0.85:
                    if prob >= max(0.50, top_prob * 0.60):
                        accepted.append(cid)
                        continue

                # 3. Standard High-Precision Rule
                if prob < base_p: continue
                if prob < top_prob * rel_ratio: continue
                if name_sim < 0.45 and not domain_ok: continue
                if num_contra and name_sim < 0.88: continue

                accepted.append(cid)
                if len(accepted) >= 8: break

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

    del cand_lookup, s1_lookup, cand_map
    gc.collect()

    elapsed = time.time() - t_start
    print(f"\n[{country} COMPLETED in {elapsed/60:.2f} mins]", flush=True)
    print(f"  Evaluated Pairs:       {total_evaluated_pairs:,}", flush=True)
    print(f"  Accepted Matches:      {total_accepted_matches:,}", flush=True)
    print(f"  Singletons:            {empty_matches:,} ({empty_matches/len(s1_ids)*100:.2f}%)", flush=True)
    print(f"  Avg Matches/Non-Empty: {total_accepted_matches/max(1, len(s1_ids)-empty_matches):.2f}", flush=True)

    return match_results, candidate_results


def main():
    print("=" * 80, flush=True)
    print("AMAZON ML CHALLENGE 2026 — PRODUCTION HIGH-RECALL PIPELINE", flush=True)
    print("=" * 80, flush=True)

    t_global = time.time()

    # 1. Load Model
    print(f"Loading trained LightGBM model from {MODEL_PATH}...", flush=True)
    m_data = joblib.load(MODEL_PATH)
    model = m_data['model']

    # 2. Load test_source1.tsv
    print(f"Loading test entities from {TEST_DIR}/test_source1.tsv...", flush=True)
    t0 = time.time()
    s1_full = pd.read_csv(f'{TEST_DIR}/test_source1.tsv', sep='\t', dtype=str)
    all_s1_ordered = list(s1_full['entity_id'].values)
    print(f"Loaded {len(all_s1_ordered):,} test entities in {time.time()-t0:.2f}s", flush=True)

    all_matches = {}
    all_candidates = {}

    test_s2_path = f'{TEST_DIR}/test_source2.tsv'
    test_s3_path = f'{TEST_DIR}/test_source3.tsv'

    for country in ['France', 'India', 'US']:
        s1_c_df = s1_full[s1_full['country'] == country][['entity_id', 'business_name', 'business_address']].copy()
        c_matches, c_cands = run_country_pipeline(
            country=country,
            model=model,
            s1_c_df=s1_c_df,
            test_s2_path=test_s2_path,
            test_s3_path=test_s3_path
        )
        all_matches.update(c_matches)
        all_candidates.update(c_cands)
        del s1_c_df, c_matches, c_cands
        gc.collect()

    del s1_full, model
    gc.collect()

    # 3. Write candidate_pairs.tsv
    print("\n" + "=" * 80, flush=True)
    print(f"Writing {OUT_CANDIDATE} in exact test_source1 ordering...", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT_CANDIDATE), exist_ok=True)
    with open(OUT_CANDIDATE, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1_ordered:
            cands = all_candidates.get(sid, [])
            f.write(f"{sid}\t{','.join(cands)}\n")
    print(f"Written {len(all_s1_ordered):,} rows to {OUT_CANDIDATE} in {time.time()-t0:.2f}s ({os.path.getsize(OUT_CANDIDATE)/(1024*1024):.2f} MB)", flush=True)
    del all_candidates
    gc.collect()

    # 4. Write matching_results.tsv
    print("\n" + "=" * 80, flush=True)
    print(f"Writing {OUT_MATCHING} in exact test_source1 ordering...", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT_MATCHING), exist_ok=True)
    total_matches_written = 0
    empty_written = 0

    with open(OUT_MATCHING, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ordered:
            matches = all_matches.get(sid, [])
            f.write(f"{sid}\t{','.join(matches)}\n")
            if matches:
                total_matches_written += len(matches)
            else:
                empty_written += 1

    print(f"Written {len(all_s1_ordered):,} rows to {OUT_MATCHING} in {time.time()-t0:.2f}s ({os.path.getsize(OUT_MATCHING)/(1024*1024):.2f} MB)", flush=True)
    print(f"  Total Matches Written:      {total_matches_written:,}", flush=True)
    print(f"  Total Singletons (0 match): {empty_written:,} ({empty_written/len(all_s1_ordered)*100:.2f}%)", flush=True)
    print(f"  Avg Matches / Non-Empty:    {total_matches_written/max(1, len(all_s1_ordered)-empty_written):.2f}", flush=True)
    del all_matches
    gc.collect()

    # 5. Run Official Validator
    print("\n" + "=" * 80, flush=True)
    print("Running Official Submission Validator with --check-ids...", flush=True)
    print("=" * 80, flush=True)
    cmd = [
        sys.executable,
        'validate_submission.py',
        '--matching', OUT_MATCHING,
        '--candidate', OUT_CANDIDATE,
        '--test-dir', TEST_DIR,
        '--check-ids'
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    total_time = time.time() - t_global
    print(f"\n>>> PIPELINE EXECUTION FINISHED IN {total_time/60:.2f} MINUTES! <<<", flush=True)

if __name__ == '__main__':
    main()
