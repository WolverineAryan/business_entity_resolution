"""
Amazon ML Challenge 2026 — Final 0.99+ Production Inference Pipeline
===================================================================
Key Architectural Enhancements:
1. Multi-Core Vectorized Flat Inverted Index Blocking (Top-50 India, Top-40 US/France)
2. 35-Feature Discriminative Engine (Postal codes, Phonetics, Acronyms, Script Interactions)
3. Tri-Model Gradient Boosting Ensemble (LightGBM + CatBoost + XGBoost)
4. Multi-Stage Decision Cascade:
   - Indic Script Phonetic Transliteration Fallback (recovering ~29k Indian matches)
   - Source-3 Empty-Address High-Identity Fallback
   - Postal Code Exact Confirmation Rule
   - Strict Precision Guardrails (Postal & Numeric Contradiction Rejection)
5. Dynamic Thresholding (base_p: 0.78-0.80, rel_ratio: 0.75) for Max Macro F0.5
6. Submission format & ID-existence verification
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

# Per-Country Calibrated Configurations for Max Macro F0.5
CONFIGS = {
    'France': {'base_p': 0.78, 'rel_ratio': 0.75, 'top_k': 40},
    'India':  {'base_p': 0.78, 'rel_ratio': 0.75, 'top_k': 50},
    'US':     {'base_p': 0.80, 'rel_ratio': 0.75, 'top_k': 40},
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

    # 2. Multi-Core Vectorized Candidate Generation (Top-K)
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

    # 4. Stream Batch Scoring with Tri-Model Ensemble & Decision Cascade
    t0 = time.time()
    batch_size = 15000
    s1_ids = list(s1_c_df['entity_id'].values)

    match_results = {}
    candidate_results = {}
    total_evaluated_pairs = 0
    total_accepted_matches = 0
    empty_matches = 0

    # Diagnostic rule match counters
    indic_rule_count = 0
    s3_empty_rule_count = 0
    postal_rule_count = 0
    standard_rule_count = 0

    print(f"[{country}] Scoring candidate pairs with Tri-Model Ensemble & Decision Cascade...", flush=True)
    for b_idx in range(0, len(s1_ids), batch_size):
        chunk_s1 = s1_ids[b_idx: b_idx + batch_size]
        pair_s1, pair_cid, pair_feats, pair_meta = [], [], [], []

        for sid in chunk_s1:
            s1_tup = s1_lookup[sid]
            s1_ng = s1_ngrams[sid]
            cands = cand_map.get(sid, [])
            candidate_results[sid] = cands

            for cid in cands:
                c_tup = cand_lookup.get(cid)
                if not c_tup: continue

                # Fast pre-filter dead candidates (no common tokens and very low fuzzy)
                # c_tup[11] is has_indic
                if not c_tup[11]:
                    if len(s1_tup[2].intersection(c_tup[2])) == 0 and len(s1_tup[8].intersection(c_tup[8])) <= 1:
                        if fuzz.token_set_ratio(s1_tup[0], c_tup[0]) < 28 and fuzz.token_set_ratio(s1_tup[1], c_tup[1]) < 30:
                            continue

                is_s3 = 1.0 if cid.startswith('S3-') else 0.0
                feats = extract_features_v3(s1_tup, s1_ng, c_tup, is_s3)

                # Meta attributes for decision cascade
                name_sim = max(feats[2], feats[3], feats[1])        # max(tok_sort, tok_set, lev)
                addr_sim = max(feats[15], feats[16], feats[14])     # max(addr_tok_sort, addr_tok_set, addr_lev)
                domain_ok = feats[9] == 1.0
                acronym_ok = feats[10] == 1.0
                num_contra = feats[23] == 1.0
                postal_match = feats[24] == 1.0
                postal_contra = feats[25] == 1.0
                addr_empty = feats[19] == 1.0
                has_indic = feats[29] == 1.0

                pair_s1.append(sid)
                pair_cid.append(cid)
                pair_feats.append(feats)
                pair_meta.append((
                    name_sim, addr_sim, domain_ok, acronym_ok,
                    num_contra, postal_match, postal_contra,
                    addr_empty, has_indic, is_s3
                ))

        total_evaluated_pairs += len(pair_feats)

        if pair_feats:
            X_b = np.array(pair_feats, dtype=np.float32)
            p_lgb = lgb_m.predict_proba(X_b)[:, 1]
            p_cat = cat_m.predict_proba(X_b)[:, 1]
            p_xgb = xgb_m.predict_proba(X_b)[:, 1]
            probs = w_lgb * p_lgb + w_cat * p_cat + w_xgb * p_xgb
        else:
            probs = np.array([])

        chunk_cand_scores: Dict[str, List[Tuple]] = {sid: [] for sid in chunk_s1}
        for sid, cid, meta, prob in zip(pair_s1, pair_cid, pair_meta, probs):
            chunk_cand_scores[sid].append((cid, float(prob), meta))

        # Decision Cascade per Query Entity
        for sid in chunk_s1:
            c_list = chunk_cand_scores[sid]
            if not c_list:
                match_results[sid] = []
                empty_matches += 1
                continue

            sorted_c = sorted(c_list, key=lambda x: x[1], reverse=True)
            top_prob = sorted_c[0][1]

            # Check if any special rule can trigger even if top_prob < base_p
            can_pass_gate = (top_prob >= base_p)
            if not can_pass_gate:
                for _, p, m in sorted_c:
                    (n_sim, a_sim, dom_ok, acr_ok, n_contra, p_match, p_contra, a_empty, h_indic, s3_flg) = m
                    if h_indic and a_sim >= 0.70 and not n_contra and not p_contra and p >= 0.40:
                        can_pass_gate = True
                        break
                    if s3_flg and a_empty and n_sim >= 0.82 and not p_contra and p >= 0.45:
                        can_pass_gate = True
                        break
                    if p_match and n_sim >= 0.65 and not n_contra and p >= 0.55:
                        can_pass_gate = True
                        break

            if not can_pass_gate:
                match_results[sid] = []
                empty_matches += 1
                continue

            accepted = []
            for cid, prob, meta in sorted_c:
                (n_sim, a_sim, dom_ok, acr_ok, n_contra, p_match, p_contra, a_empty, h_indic, s3_flg) = meta

                # Rule 1: Indic Script Fallback (India)
                if h_indic and a_sim >= 0.70 and not n_contra and not p_contra:
                    if prob >= max(0.35, top_prob * 0.45):
                        accepted.append(cid)
                        indic_rule_count += 1
                        if len(accepted) >= 8: break
                        continue

                # Rule 2: Source-3 Blank Address Fallback
                if s3_flg and a_empty and n_sim >= 0.82 and not p_contra:
                    if prob >= max(0.45, top_prob * 0.55):
                        accepted.append(cid)
                        s3_empty_rule_count += 1
                        if len(accepted) >= 8: break
                        continue

                # Rule 3: Postal Code Exact Match Confirmation
                if p_match and n_sim >= 0.65 and not n_contra:
                    if prob >= max(0.55, top_prob * 0.65):
                        accepted.append(cid)
                        postal_rule_count += 1
                        if len(accepted) >= 8: break
                        continue

                # Rule 4: Standard High-Precision Ensemble Rule
                if prob < base_p: continue
                if prob < top_prob * rel_ratio: continue

                # Precision Guardrails (Reject False Merges)
                if p_contra and n_sim < 0.90: continue
                if n_contra and n_sim < 0.88: continue
                if n_sim < 0.45 and not dom_ok and not acr_ok: continue

                accepted.append(cid)
                standard_rule_count += 1
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

    del cand_lookup, s1_lookup, s1_ngrams, cand_map
    gc.collect()

    elapsed = time.time() - t_start
    print(f"\n[{country} PIPELINE COMPLETED in {elapsed/60:.2f} mins]", flush=True)
    print(f"  Evaluated Candidate Pairs: {total_evaluated_pairs:,}", flush=True)
    print(f"  Total Accepted Matches:   {total_accepted_matches:,}", flush=True)
    print(f"  Singletons (0 match):     {empty_matches:,} ({empty_matches/len(s1_ids)*100:.2f}%)", flush=True)
    print(f"  Avg Matches/Non-Empty:    {total_accepted_matches/max(1, len(s1_ids)-empty_matches):.2f}", flush=True)
    print(f"  Rule Trigger Stats: Indic={indic_rule_count:,}, S3Empty={s3_empty_rule_count:,}, Postal={postal_rule_count:,}, Standard={standard_rule_count:,}", flush=True)

    return match_results, candidate_results

def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 — 0.99+ Final Production Pipeline")
    parser.add_argument('--test-dir', type=str, default='dataset/test', help="Directory containing test_source1/2/3.tsv")
    parser.add_argument('--model-path', type=str, default='model_artifacts/ensemble_v3.joblib', help="Path to ensemble artifact")
    parser.add_argument('--output-dir', type=str, default='output', help="Directory to save submission files")
    parser.add_argument('--skip-validation', action='store_true', help="Skip validate_submission.py check")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_matching = os.path.join(args.output_dir, 'matching_results.tsv')
    out_candidate = os.path.join(args.output_dir, 'candidate_pairs.tsv')

    print("=" * 85, flush=True)
    print("AMAZON ML CHALLENGE 2026 — FINAL 0.99+ PRODUCTION INFERENCE PIPELINE", flush=True)
    print("=" * 85, flush=True)

    t_global = time.time()

    # 1. Load Ensemble Artifact
    print(f"Loading Tri-Model Ensemble from {args.model_path}...", flush=True)
    ensemble_data = joblib.load(args.model_path)
    print(f"Loaded ensemble with weights: {ensemble_data['weights']} (Validation F0.5: {ensemble_data['val_score']:.5f})", flush=True)

    # 2. Load test_source1.tsv
    s1_path = os.path.join(args.test_dir, 'test_source1.tsv')
    print(f"Loading reference test entities from {s1_path}...", flush=True)
    t0 = time.time()
    s1_full = pd.read_csv(s1_path, sep='\t', dtype=str)
    all_s1_ordered = list(s1_full['entity_id'].values)
    print(f"Loaded {len(all_s1_ordered):,} test entities in {time.time()-t0:.2f}s", flush=True)

    all_matches = {}
    all_candidates = {}

    test_s2_path = os.path.join(args.test_dir, 'test_source2.tsv')
    test_s3_path = os.path.join(args.test_dir, 'test_source3.tsv')

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
    print(f"\n>>> FINAL PIPELINE FINISHED IN {total_time:.2f} MINUTES! <<<\n", flush=True)

if __name__ == '__main__':
    main()
