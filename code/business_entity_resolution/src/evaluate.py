"""Evaluation metrics module for Amazon ML Challenge 2026: Business Entity Resolution.
Computes Macro F0.5 score across all Source 1 entities according to challenge specs.
"""

from typing import Dict, Set, List
import pandas as pd
import numpy as np


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Computes F_beta score given precision and recall."""
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """Computes F0.5 score for a single Source 1 entity.
    
    Rules:
    - True singleton (empty ground truth):
        - Empty prediction: 1.0
        - Non-empty prediction: 0.0 (false positive penalty)
    - Non-singleton (non-empty ground truth):
        - Empty prediction: 0.0 (missed all matches)
        - Non-empty prediction: standard F0.5 based on precision and recall
    """
    is_true_singleton = len(true_matches) == 0
    is_pred_singleton = len(pred_matches) == 0

    if is_true_singleton:
        return 1.0 if is_pred_singleton else 0.0

    if is_pred_singleton:
        return 0.0

    true_positives = len(true_matches.intersection(pred_matches))
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(pred_matches)
    recall = true_positives / len(true_matches)

    return compute_f_beta(precision, recall, beta=0.5)


def evaluate_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Dict[str, float]:
    """Computes Macro F0.5, Precision, Recall, and Singleton statistics.
    
    Args:
        ground_truth: mapping of s1_id -> set of true matched entity IDs
        predictions: mapping of s1_id -> set of predicted matched entity IDs
        
    Returns:
        dictionary containing macro_f05, singleton_accuracy, precision, recall
    """
    scores = []
    singleton_scores = []
    non_singleton_scores = []
    total_tp = 0
    total_pred = 0
    total_true = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(true_set, pred_set)
        scores.append(score)

        if len(true_set) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)
            tp = len(true_set.intersection(pred_set))
            total_tp += tp
            total_pred += len(pred_set)
            total_true += len(true_set)

    micro_precision = total_tp / total_pred if total_pred > 0 else 0.0
    micro_recall = total_tp / total_true if total_true > 0 else 0.0

    return {
        "macro_f05": float(np.mean(scores)),
        "singleton_accuracy": float(np.mean(singleton_scores)) if singleton_scores else 0.0,
        "non_singleton_macro_f05": float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "total_entities": len(ground_truth),
        "total_singletons": len(singleton_scores),
    }


def evaluate_from_files(gt_file: str, pred_file: str) -> Dict[str, float]:
    """Evaluates prediction TSV file against ground truth TSV file."""
    gt_df = pd.read_csv(gt_file, sep='\t', dtype=str).fillna('')
    pred_df = pd.read_csv(pred_file, sep='\t', dtype=str).fillna('')

    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id'].strip()
        matched = row['matched_entity_ids'].strip()
        gt_dict[s1_id] = set(matched.split(',')) if matched else set()

    pred_dict = {}
    for _, row in pred_df.iterrows():
        s1_id = row['source1_entity_id'].strip()
        matched = row['matched_entity_ids'].strip()
        pred_dict[s1_id] = set(matched.split(',')) if matched else set()

    return evaluate_macro_f05(gt_dict, pred_dict)
