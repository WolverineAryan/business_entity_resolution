"""
Amazon ML Challenge 2026 — 500,000 Queries Large-Scale Tri-Model Training
========================================================================
Scales training from 150k -> 500k queries with 1,000 estimators per model.
Uses the updated 35-feature engine with French legal entity normalization and domain fix.
"""

import os
import sys
import gc
import time
import re
import joblib
import multiprocessing as mp
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from typing import Dict, List, Tuple, Set
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('code/business_entity_resolution'))

from src.blocking import BlockingEngine, extract_index_tokens
from src.features_v3 import (
    precompute_entity_v3, extract_features_v3, get_char_ngrams, FEATURE_NAMES_V3
)

def evaluate_macro_f05(preds: Dict[str, Set[str]], gt: Dict[str, Set[str]]) -> Tuple[float, float, float]:
    f05_list, p_list, r_list = [], [], []
    for sid, true_matches in gt.items():
        pred_matches = preds.get(sid, set())
        tp = len(pred_matches.intersection(true_matches))
        fp = len(pred_matches - true_matches)
        fn = len(true_matches - pred_matches)

        if not true_matches and not pred_matches:
            f05_list.append(1.0); p_list.append(1.0); r_list.append(1.0)
            continue
        if not true_matches and pred_matches:
            f05_list.append(0.0); p_list.append(0.0); r_list.append(1.0)
            continue
        if true_matches and not pred_matches:
            f05_list.append(0.0); p_list.append(1.0); r_list.append(0.0)
            continue

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        beta2 = 0.5 ** 2
        f05 = (1 + beta2) * (prec * rec) / (beta2 * prec + rec) if (beta2 * prec + rec) > 0 else 0.0

        f05_list.append(f05)
        p_list.append(prec)
        r_list.append(rec)

    return float(np.mean(f05_list)), float(np.mean(p_list)), float(np.mean(r_list))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train 500k Scaled Tri-Model Ensemble")
    parser.add_argument('--train-dir', type=str, default='dataset/train', help="Directory with train_source1/2/3.tsv and ground truth")
    parser.add_argument('--output-model', type=str, default='model_artifacts/ensemble_v4.joblib', help="Output model artifact path")
    parser.add_argument('--num-samples', type=int, default=300000, help="Number of S1 training queries (default: 300k)")
    args = parser.parse_args()

    os.makedirs('model_artifacts', exist_ok=True)
    t_global = time.time()

    print("=" * 80)
    print(f"AMAZON ML CHALLENGE 2026 — TRAINING SCALED ENSEMBLE ({args.num_samples:,} QUERIES)")
    print("=" * 80)

    # 1. Load train_source1.tsv
    print("1. Loading train_source1.tsv...")
    t0 = time.time()
    s1_full = pd.read_csv(f'{args.train_dir}/train_source1.tsv', sep='\t', dtype=str)
    print(f"   Loaded {len(s1_full):,} Source-1 records in {time.time()-t0:.2f}s")

    # Stratified sampling
    n_per_country = args.num_samples // 2
    rng = np.random.RandomState(42)
    sample_dfs = []
    for c in ['India', 'US']:
        sub = s1_full[s1_full['country'] == c]
        take = min(n_per_country, len(sub))
        idx = rng.choice(sub.index, size=take, replace=False)
        sample_dfs.append(sub.loc[idx])
    
    sample_s1 = pd.concat(sample_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
    del s1_full
    gc.collect()
    print(f"   Selected {len(sample_s1):,} stratified queries ({len(sample_s1)//2:,} India, {len(sample_s1)//2:,} US)")

    # 2. Load ground truth
    print("2. Loading train_ground_truth.tsv...")
    t0 = time.time()
    gt_df = pd.read_csv(f'{args.train_dir}/train_ground_truth.tsv', sep='\t', dtype=str)
    sample_sid_set = set(sample_s1['entity_id'].values)
    gt_df_sample = gt_df[gt_df['source1_entity_id'].isin(sample_sid_set)]
    
    gt_map: Dict[str, Set[str]] = {sid: set() for sid in sample_sid_set}
    for sid, m_str in zip(gt_df_sample['source1_entity_id'], gt_df_sample['matched_entity_ids'].fillna('')):
        if m_str:
            gt_map[sid] = set(m_str.split(','))
    del gt_df, gt_df_sample
    gc.collect()
    print(f"   Loaded ground truth for {len(gt_map):,} queries in {time.time()-t0:.2f}s")

    # 3. Precompute queries
    print("3. Precomputing Entity Tuples with updated 35-feature engine...")
    t0 = time.time()
    s1_lookup = {
        sid: precompute_entity_v3(n, a)
        for sid, n, a in zip(
            sample_s1['entity_id'],
            sample_s1['business_name'].fillna(''),
            sample_s1['business_address'].fillna('')
        )
    }
    s1_ngrams = {sid: get_char_ngrams(tup[0], 3) for sid, tup in s1_lookup.items()}
    print(f"   Precomputed {len(s1_lookup):,} queries in {time.time()-t0:.2f}s")

    # Train/Val split
    all_s1_ids = list(sample_s1['entity_id'].values)
    rng.shuffle(all_s1_ids)
    split_idx = int(len(all_s1_ids) * 0.85)
    train_sid_set = set(all_s1_ids[:split_idx])
    val_sid_set = set(all_s1_ids[split_idx:])
    print(f"   Split: {len(train_sid_set):,} Train entities | {len(val_sid_set):,} Validation entities")

    all_pairs_X = []
    all_pairs_y = []
    all_pairs_meta = []

    num_workers = min(12, max(1, mp.cpu_count()))
    blocking_engine = BlockingEngine(top_k=35, max_posting_len=20000, num_workers=num_workers)

    for country in ['India', 'US']:
        print(f"\n[{country}] Mining candidates and hard negatives from full target pool...")
        t0 = time.time()
        c_s1 = sample_s1[sample_s1['country'] == country][['entity_id', 'business_name', 'business_address']].copy()
        c_s1_ids = list(c_s1['entity_id'].values)

        cols = ['entity_id', 'business_name', 'business_address', 'country']
        s2_c = pd.read_csv(f'{args.train_dir}/train_source2.tsv', sep='\t', dtype=str, usecols=cols)
        s2_c = s2_c[s2_c['country'] == country]

        s3_c = pd.read_csv(f'{args.train_dir}/train_source3.tsv', sep='\t', dtype=str, usecols=cols)
        s3_c = s3_c[s3_c['country'] == country]

        target_df = pd.concat([s2_c, s3_c], ignore_index=True)
        del s2_c, s3_c
        gc.collect()

        c_cand_map_raw = blocking_engine.retrieve_candidates_for_country(c_s1, target_df)
        c_cand_map = {sid: [cid for cid, _ in c_list] for sid, c_list in c_cand_map_raw.items()}
        del c_cand_map_raw
        gc.collect()

        needed_cands = set()
        for clist in c_cand_map.values():
            needed_cands.update(clist)
        for sid in c_s1_ids:
            needed_cands.update(gt_map.get(sid, set()))

        target_filt = target_df[target_df['entity_id'].isin(needed_cands)]
        cand_lookup = {
            eid: precompute_entity_v3(n, a)
            for eid, n, a in zip(
                target_filt['entity_id'],
                target_filt['business_name'].fillna(''),
                target_filt['business_address'].fillna('')
            )
        }
        del target_df, target_filt
        gc.collect()

        country_pairs = 0
        for sid in c_s1_ids:
            s1_tup = s1_lookup[sid]
            s1_ng = s1_ngrams[sid]
            true_m = gt_map.get(sid, set())
            is_val = sid in val_sid_set

            retrieved = c_cand_map.get(sid, [])
            all_eval_cands = list(dict.fromkeys(list(true_m) + retrieved))

            neg_count = 0
            for cid in all_eval_cands:
                c_tup = cand_lookup.get(cid)
                if not c_tup: continue

                is_pos = 1 if cid in true_m else 0
                if not is_pos:
                    if not is_val and neg_count >= 10:
                        continue
                    neg_count += 1

                is_s3 = 1.0 if cid.startswith('S3-') else 0.0
                feats = extract_features_v3(s1_tup, s1_ng, c_tup, is_s3)

                all_pairs_X.append(feats)
                all_pairs_y.append(is_pos)
                all_pairs_meta.append((sid, cid, is_pos, 1 if is_val else 0))
                country_pairs += 1

        del cand_lookup, c_cand_map, c_s1
        gc.collect()
        print(f"[{country}] Extracted {country_pairs:,} feature pairs in {time.time()-t0:.2f}s")

    # Arrays
    X_all = np.array(all_pairs_X, dtype=np.float32)
    y_all = np.array(all_pairs_y, dtype=np.int32)
    meta_df = pd.DataFrame(all_pairs_meta, columns=['sid', 'cid', 'is_pos', 'is_val'])
    del all_pairs_X, all_pairs_y, all_pairs_meta
    gc.collect()

    train_mask = (meta_df['is_val'] == 0).values
    val_mask = (meta_df['is_val'] == 1).values

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    val_meta = meta_df[meta_df['is_val'] == 1].reset_index(drop=True)
    del X_all, y_all, meta_df
    gc.collect()

    print(f"\nTrain Matrix: {len(X_train):,} pairs (Pos: {y_train.sum():,})")
    print(f"Val Matrix:   {len(X_val):,} pairs (Pos: {y_val.sum():,})")

    # 4. Train LightGBM (1,000 estimators)
    print("\n--- Training LightGBM v4 (1,000 trees) ---")
    t0 = time.time()
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=9,
        subsample=0.85,
        colsample_bytree=0.80,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_clf.fit(X_train, y_train)
    print(f"LightGBM trained in {time.time()-t0:.2f}s")
    lgb_val_probs = lgb_clf.predict_proba(X_val)[:, 1]

    # 5. Train CatBoost (800 iterations)
    print("\n--- Training CatBoost v4 (800 trees) ---")
    t0 = time.time()
    cat_clf = CatBoostClassifier(
        iterations=800,
        learning_rate=0.035,
        depth=7,
        l2_leaf_reg=3.0,
        random_seed=42,
        thread_count=-1,
        verbose=200
    )
    cat_clf.fit(X_train, y_train)
    print(f"CatBoost trained in {time.time()-t0:.2f}s")
    cat_val_probs = cat_clf.predict_proba(X_val)[:, 1]

    # 6. Train XGBoost (800 estimators)
    print("\n--- Training XGBoost v4 (800 trees) ---")
    t0 = time.time()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.80,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    xgb_clf.fit(X_train, y_train)
    print(f"XGBoost trained in {time.time()-t0:.2f}s")
    xgb_val_probs = xgb_clf.predict_proba(X_val)[:, 1]

    # 7. Evaluate Ensemble
    print("\n--- Evaluating Tri-Model Ensemble Calibration ---")
    ensemble_val_probs = 0.40 * lgb_val_probs + 0.35 * cat_val_probs + 0.25 * xgb_val_probs
    val_cand_by_s1: Dict[str, List[Tuple]] = defaultdict(list)
    for (sid, cid, is_pos, _), prob in zip(val_meta.values, ensemble_val_probs):
        val_cand_by_s1[sid].append((cid, float(prob)))

    val_gt = {sid: gt_map[sid] for sid in val_sid_set}
    best_f05 = 0.0
    best_cfg = None

    for base_p in [0.78, 0.80, 0.81, 0.82, 0.85]:
        for rel_ratio in [0.75, 0.76, 0.77, 0.78, 0.80]:
            preds = {}
            for sid, true_matches in val_gt.items():
                c_list = val_cand_by_s1.get(sid, [])
                if not c_list:
                    preds[sid] = set()
                    continue
                sorted_c = sorted(c_list, key=lambda x: x[1], reverse=True)
                top_p = sorted_c[0][1]
                if top_p < base_p:
                    preds[sid] = set()
                    continue
                accepted = []
                for cid, prob in sorted_c:
                    if prob < base_p: break
                    if prob < top_p * rel_ratio: break
                    accepted.append(cid)
                    if len(accepted) >= 12: break
                preds[sid] = set(accepted)

            f05, prec, rec = evaluate_macro_f05(preds, val_gt)
            print(f"  base_p={base_p:.2f}, rel_ratio={rel_ratio:.2f} -> Macro F0.5: {f05:.5f} (Prec: {prec:.4f}, Rec: {rec:.4f})")
            if f05 > best_f05:
                best_f05 = f05
                best_cfg = (base_p, rel_ratio)

    print(f"\nOPTIMAL VALIDATION MACRO F0.5: {best_f05:.5f} (base_p={best_cfg[0]}, rel_ratio={best_cfg[1]})")

    # 8. Save Model
    joblib.dump({
        'lgb_model': lgb_clf,
        'cat_model': cat_clf,
        'xgb_model': xgb_clf,
        'weights': [0.40, 0.35, 0.25],
        'feature_names': FEATURE_NAMES_V3,
        'best_config': best_cfg,
        'val_score': best_f05
    }, args.output_model, compress=3)
    print(f"Ensemble saved to {args.output_model} (Size: {os.path.getsize(args.output_model)/1e6:.2f} MB)")
    print(f"Total training time: {(time.time()-t_global)/60:.2f} mins")

if __name__ == '__main__':
    main()
