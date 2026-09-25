"""Machine Learning matching model module.
Implements LightGBM-based binary classification, probability ranking, and Macro F0.5 threshold optimization.
"""

import os
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from src.evaluate import evaluate_macro_f05


class MatchingModel:
    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        self.model: Optional[lgb.LGBMClassifier] = None
        self.best_threshold: float = 0.50

    def fit(self, X: np.ndarray, y: np.ndarray, val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Trains the LightGBM classifier on candidate pair features."""
        self.model = lgb.LGBMClassifier(**self.params)
        eval_set = [val_data] if val_data is not None else None
        self.model.fit(
            X,
            y,
            eval_set=eval_set
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns match probability for candidate pairs."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict_proba(X)[:, 1]

    def optimize_threshold(
        self,
        candidate_predictions: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, Set[str]],
        threshold_range: Tuple[float, float, float] = (0.30, 0.85, 0.02)
    ) -> Tuple[float, float]:
        """Grid searches over decision thresholds to find the threshold that maximizes Macro F0.5.
        
        Args:
            candidate_predictions: s1_id -> list of (cand_id, probability)
            ground_truth: s1_id -> set of true matched IDs
            threshold_range: (min_thresh, max_thresh, step)
        
        Returns:
            (best_threshold, best_macro_f05)
        """
        min_t, max_t, step = threshold_range
        thresholds = np.arange(min_t, max_t + 1e-5, step)

        best_score = -1.0
        best_thresh = 0.50

        print(f"Optimizing threshold over {len(thresholds)} values for Macro F0.5...")

        for thresh in thresholds:
            pred_dict: Dict[str, Set[str]] = {}
            for s1_id, cands in candidate_predictions.items():
                matched = [cid for cid, prob in cands if prob >= thresh]
                pred_dict[s1_id] = set(matched)

            eval_res = evaluate_macro_f05(ground_truth, pred_dict)
            score = eval_res['macro_f05']

            if score > best_score:
                best_score = score
                best_thresh = float(thresh)

        self.best_threshold = best_thresh
        print(f"Optimal Threshold: {best_thresh:.3f} -> Best Macro F0.5: {best_score:.4f}")
        return best_thresh, best_score

    def save(self, filepath: str):
        """Saves trained model and threshold."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.best_threshold}, filepath)

    def load(self, filepath: str):
        """Loads trained model and threshold."""
        data = joblib.load(filepath)
        self.model = data["model"]
        self.best_threshold = data.get("threshold", 0.50)
