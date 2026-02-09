"""
Institutional-Grade Model Training Pipeline (v2)
==================================================

Replaces the naive 200-round LightGBM with proper financial ML training:
purged walk-forward cross-validation, Information Coefficient metrics
(not accuracy), SHAP per-prediction explanations, probability calibration,
and early stopping with a chronological validation set.

Key innovations over v1:
1. **PurgedWalkForwardCV** (de Prado Ch.7): removes training samples
   whose label period overlaps with test → no data leakage.
2. **FinancialMetrics**: IC, Rank IC, IR, Profit Factor — accuracy is
   irrelevant in finance.
3. **Early stopping**: validation carved from END of training data
   (chronological, not random).
4. **SHAP per-prediction**: explains WHY each trade was triggered.
5. **Isotonic calibration**: raw LightGBM probabilities are poorly
   calibrated — isotonic regression maps them to empirical frequencies
   for correct Kelly sizing.

Dependencies:
    pip install lightgbm shap scikit-learn scipy numpy pandas joblib

Author: NUBLE ML Pipeline
Version: 2.0.0
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy import stats as sp_stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_fscore_support

from .features_v2 import FeaturePipeline, build_features
from .labeling import create_labels, create_meta_labels

logger = logging.getLogger(__name__)

# Suppress verbose LightGBM / SHAP output
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")


# ══════════════════════════════════════════════════════════════
# 1. Purged Walk-Forward Cross-Validation  (de Prado Ch.7)
# ══════════════════════════════════════════════════════════════


class PurgedWalkForwardCV:
    """
    Walk-Forward Cross-Validation with Purging and Embargo.

    Standard walk-forward: train on [0, t], test on [t+1, t+k].
    Problem: if labels overlap (triple barrier with 10-day holding period),
    training data near the boundary contains information about test labels.

    Purging (de Prado Chapter 7):
    Remove training samples whose label period overlaps with ANY test sample.
    If test starts at t+1 with 10-day labels, purge training samples from
    [t-9, t] because their labels extend into the test period.

    Embargo (additional safety):
    After purging, also remove an additional buffer of ``embargo_pct`` of
    training samples at the boundary.  This handles any residual serial
    correlation in features.

    This CV produces CONSERVATIVE performance estimates — if your model
    performs well here, it's likely real.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = 252,
        max_holding_period: int = 10,
        embargo_pct: float = 0.01,
        expanding: bool = True,
    ):
        """
        Args:
            n_splits:             Number of walk-forward folds.
            min_train_size:       Minimum training observations (~1 year).
            max_holding_period:   Must match triple-barrier holding period.
            embargo_pct:          Additional buffer (fraction of training).
            expanding:            True = expanding window, False = sliding.
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.max_holding_period = max_holding_period
        self.embargo_pct = embargo_pct
        self.expanding = expanding

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        label_end_dates: pd.Series | None = None,
    ):
        """
        Generate train/test index splits with purging and embargo.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame with DatetimeIndex.
        y : pd.Series | None
            Labels (unused but required by sklearn interface).
        label_end_dates : pd.Series | None
            Series of barrier-touch dates for each observation.  If
            provided, used for precise purging.  If ``None``, assumes
            ``max_holding_period`` for all labels.

        Yields
        ------
        (train_indices, test_indices) : tuple[np.ndarray, np.ndarray]
            Integer position arrays.

        Algorithm
        ---------
        1. total = len(X)
        2. test_size = (total − min_train_size) // n_splits
        3. For split *i*:
           a. test_start = min_train_size + i × test_size
           b. test_end   = test_start + test_size  (or end for last split)
           c. train_end_raw = test_start
           d. PURGE: remove training samples whose label extends
              past X.index[test_start].
           e. EMBARGO: remove an additional ``embargo_pct × train_len``
              samples from the end of training.
           f. If expanding: train_start = 0.
              If sliding:   train_start = max(0, purged_end − min_train_size).
           g. Validate & yield.
        """
        n = len(X)
        available = n - self.min_train_size
        if available < self.n_splits * 21:
            raise ValueError(
                f"Not enough data for {self.n_splits} folds with "
                f"min_train_size={self.min_train_size}. "
                f"Have {n} observations, need at least "
                f"{self.min_train_size + self.n_splits * 21}."
            )

        test_size = available // self.n_splits

        for i in range(self.n_splits):
            test_start = self.min_train_size + i * test_size
            if i == self.n_splits - 1:
                test_end = n  # last fold gets remaining data
            else:
                test_end = test_start + test_size

            if test_end - test_start < 21:
                logger.warning(
                    "Fold %d has only %d test samples — skipping.",
                    i, test_end - test_start,
                )
                continue

            # ── Purging ───────────────────────────────────────
            train_end = test_start  # raw boundary

            if label_end_dates is not None:
                # Precise purging: remove samples whose label extends
                # into the test period.
                test_start_date = X.index[test_start]
                train_candidates = np.arange(0, train_end)
                end_dates = label_end_dates.iloc[train_candidates]
                keep_mask = end_dates < test_start_date
                purged_train = train_candidates[keep_mask.values]
            else:
                # Conservative purging: remove last max_holding_period
                purge_n = self.max_holding_period
                purged_end = max(0, train_end - purge_n)
                purged_train = np.arange(0, purged_end)

            # ── Embargo ───────────────────────────────────────
            if len(purged_train) > 0:
                embargo_n = max(1, int(len(purged_train) * self.embargo_pct))
                purged_train = purged_train[:-embargo_n]

            # ── Sliding window ────────────────────────────────
            if not self.expanding and len(purged_train) > self.min_train_size:
                purged_train = purged_train[-self.min_train_size:]

            # ── Validate ──────────────────────────────────────
            if len(purged_train) < self.min_train_size:
                logger.warning(
                    "Fold %d: only %d training samples after purge "
                    "(need %d) — skipping.",
                    i, len(purged_train), self.min_train_size,
                )
                continue

            test_indices = np.arange(test_start, test_end)

            # Final check: no overlap
            assert len(np.intersect1d(purged_train, test_indices)) == 0, (
                f"Fold {i}: train/test overlap detected!"
            )

            logger.info(
                "Fold %d: train=%d [%d..%d], test=%d [%d..%d], "
                "purged=%d, embargo=%d",
                i,
                len(purged_train),
                int(purged_train[0]),
                int(purged_train[-1]),
                len(test_indices),
                test_start,
                test_end - 1,
                train_end - (int(purged_train[-1]) + 1),
                max(1, int((int(purged_train[-1]) + 1) * self.embargo_pct)),
            )

            yield purged_train, test_indices

    def get_n_splits(self) -> int:
        """Return the number of splits."""
        return self.n_splits


# ══════════════════════════════════════════════════════════════
# 2. Financial Metrics
# ══════════════════════════════════════════════════════════════


class FinancialMetrics:
    """
    Financial ML metrics that actually matter.

    Accuracy is MEANINGLESS in finance:
    - Markets go up ~53% of days → always predicting "up" = 53% accuracy.
    - A model with 51% accuracy but strong IC can be extremely profitable.
    - A model with 55% accuracy only on low-vol days is worthless.

    Information Coefficient (IC) and Rank IC are the gold standards
    in institutional quant research.
    """

    @staticmethod
    def information_coefficient(
        predictions: pd.Series,
        actual_returns: pd.Series,
    ) -> tuple[float, float]:
        """
        Pearson correlation between predicted scores and actual returns.

        IC > 0.02 is considered good in financial ML.
        IC > 0.05 is excellent.
        IC > 0.10 is suspicious (likely overfitting or data leakage).

        Returns
        -------
        (ic, p_value) : tuple[float, float]
        """
        common = predictions.dropna().index.intersection(actual_returns.dropna().index)
        if len(common) < 3:
            return 0.0, 1.0
        pred = predictions.reindex(common)
        actual = actual_returns.reindex(common)
        ic, pval = sp_stats.pearsonr(pred, actual)
        return float(ic), float(pval)

    @staticmethod
    def rank_information_coefficient(
        predictions: pd.Series,
        actual_returns: pd.Series,
    ) -> tuple[float, float]:
        """
        Spearman rank correlation between predictions and actual returns.
        More robust to outliers than IC.

        Returns
        -------
        (rank_ic, p_value) : tuple[float, float]
        """
        common = predictions.dropna().index.intersection(actual_returns.dropna().index)
        if len(common) < 3:
            return 0.0, 1.0
        pred = predictions.reindex(common)
        actual = actual_returns.reindex(common)
        ric, pval = sp_stats.spearmanr(pred, actual)
        return float(ric), float(pval)

    @staticmethod
    def information_ratio(
        predictions: pd.Series,
        actual_returns: pd.Series,
        frequency: int = 252,
    ) -> float:
        """
        Annualized IC / std(IC).
        Measures consistency of prediction skill.

        Simplified annualisation: IC × √frequency.
        """
        ic, _ = FinancialMetrics.information_coefficient(predictions, actual_returns)
        return float(ic * np.sqrt(frequency))

    @staticmethod
    def hit_rate(
        predictions: pd.Series,
        actual_returns: pd.Series,
    ) -> float:
        """
        Percentage of correctly predicted directions.
        sign(prediction) == sign(actual_return).
        """
        common = predictions.dropna().index.intersection(actual_returns.dropna().index)
        if len(common) == 0:
            return 0.0
        pred_sign = np.sign(predictions.reindex(common))
        actual_sign = np.sign(actual_returns.reindex(common))
        # Exclude zeros in actual (no movement → direction undefined)
        valid = actual_sign != 0
        if valid.sum() == 0:
            return 0.0
        return float((pred_sign[valid] == actual_sign[valid]).mean())

    @staticmethod
    def profit_factor(
        predictions: pd.Series,
        actual_returns: pd.Series,
    ) -> float:
        """
        Gross profit / gross loss from trading in the predicted direction.

        P&L per trade = sign(prediction) × actual_return.
        Profit factor = sum(positive P&L) / |sum(negative P&L)|.

        Profit factor > 1.0 means the model makes money (ignoring costs).
        Profit factor > 2.0 is excellent.
        """
        common = predictions.dropna().index.intersection(actual_returns.dropna().index)
        if len(common) == 0:
            return 0.0
        pred = predictions.reindex(common)
        actual = actual_returns.reindex(common)

        # P&L = sign(pred) * actual: positive when correct direction
        pnl = np.sign(pred) * actual
        gross_profit = pnl[pnl > 0].sum()
        gross_loss = abs(pnl[pnl < 0].sum())

        if gross_loss < 1e-12:
            return float("inf") if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    @staticmethod
    def evaluate_all(
        predictions: pd.Series,
        actual_returns: pd.Series,
        labels: pd.Series | None = None,
    ) -> dict:
        """
        Compute all metrics and return as dict.

        Returns
        -------
        dict with keys:
            ic, ic_pvalue, rank_ic, rank_ic_pvalue, ir,
            hit_rate, profit_factor, n_predictions, n_positive,
            n_negative, avg_return_correct, avg_return_wrong.
            If *labels* provided: precision_per_class, recall_per_class,
            f1_per_class.
        """
        common = predictions.dropna().index.intersection(
            actual_returns.dropna().index
        )
        pred = predictions.reindex(common)
        actual = actual_returns.reindex(common)

        ic, ic_pval = FinancialMetrics.information_coefficient(pred, actual)
        ric, ric_pval = FinancialMetrics.rank_information_coefficient(pred, actual)

        # P&L per trade = sign(prediction) × actual return
        pnl = np.sign(pred) * actual
        correct_mask = pnl > 0
        avg_ret_correct = float(pnl[correct_mask].mean()) if correct_mask.any() else 0.0
        avg_ret_wrong = float(pnl[~correct_mask].mean()) if (~correct_mask).any() else 0.0

        metrics: dict[str, Any] = {
            "ic": ic,
            "ic_pvalue": ic_pval,
            "rank_ic": ric,
            "rank_ic_pvalue": ric_pval,
            "ir": FinancialMetrics.information_ratio(pred, actual),
            "hit_rate": FinancialMetrics.hit_rate(pred, actual),
            "profit_factor": FinancialMetrics.profit_factor(pred, actual),
            "n_predictions": len(common),
            "n_positive": int((pred > 0).sum()),
            "n_negative": int((pred < 0).sum()),
            "avg_return_correct": avg_ret_correct,
            "avg_return_wrong": avg_ret_wrong,
        }

        # Per-class precision / recall / F1 if labels provided
        if labels is not None:
            lab = labels.reindex(common).dropna()
            pred_lab = pred.reindex(lab.index)
            if len(lab) > 0:
                pred_class = np.sign(pred_lab).astype(int)
                true_class = lab.astype(int)
                classes = sorted(set(true_class.unique()) | set(pred_class.unique()))
                prec, rec, f1, _ = precision_recall_fscore_support(
                    true_class, pred_class, labels=classes, zero_division=0,
                )
                metrics["precision_per_class"] = {
                    int(c): round(float(p), 4) for c, p in zip(classes, prec)
                }
                metrics["recall_per_class"] = {
                    int(c): round(float(r), 4) for c, r in zip(classes, rec)
                }
                metrics["f1_per_class"] = {
                    int(c): round(float(f), 4) for c, f in zip(classes, f1)
                }

        return metrics

    @staticmethod
    def print_report(metrics: dict) -> None:
        """Pretty-print financial metrics report."""
        def _sig(pval: float) -> str:
            return "✓" if pval < 0.05 else "✗"

        ic = metrics.get("ic", 0.0)
        ic_p = metrics.get("ic_pvalue", 1.0)
        ric = metrics.get("rank_ic", 0.0)
        ric_p = metrics.get("rank_ic_pvalue", 1.0)

        print()
        print("═" * 50)
        print("  FINANCIAL ML EVALUATION REPORT")
        print("═" * 50)
        print(f"  Information Coefficient:   {ic:+.4f}  (p={ic_p:.3f}) {_sig(ic_p)}")
        print(f"  Rank IC:                   {ric:+.4f}  (p={ric_p:.3f}) {_sig(ric_p)}")
        print(f"  Information Ratio:         {metrics.get('ir', 0.0):.2f}")
        print(f"  Hit Rate:                  {metrics.get('hit_rate', 0.0) * 100:.1f}%")
        pf = metrics.get("profit_factor", 0.0)
        pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
        print(f"  Profit Factor:             {pf_str}")
        print("─" * 50)
        n = metrics.get("n_predictions", 0)
        n_pos = metrics.get("n_positive", 0)
        n_neg = metrics.get("n_negative", 0)
        print(f"  Predictions: {n} ({n_pos} long, {n_neg} short)")
        print(f"  Avg Return (correct):      {metrics.get('avg_return_correct', 0.0):+.4f}")
        print(f"  Avg Return (wrong):        {metrics.get('avg_return_wrong', 0.0):+.4f}")

        if "precision_per_class" in metrics:
            print("─" * 50)
            print("  Per-Class Precision:", metrics["precision_per_class"])
            print("  Per-Class Recall:   ", metrics["recall_per_class"])
            print("  Per-Class F1:       ", metrics["f1_per_class"])

        print("═" * 50)
        print()


# ══════════════════════════════════════════════════════════════
# 3. Model Trainer
# ══════════════════════════════════════════════════════════════


class ModelTrainer:
    """
    LightGBM trainer with financial ML best practices.

    Key differences from the v1 implementation:
    1. Early stopping with purged validation set (not fixed 200 rounds).
    2. Sample weights from label uniqueness (not equal weights).
    3. SHAP values per prediction (not just global importance).
    4. IC-based evaluation (not accuracy).
    5. Probability calibration (isotonic regression).
    6. Hyperparameter ranges tuned for financial data.
    """

    # ── Default hyperparameters ───────────────────────────────
    DEFAULT_PARAMS: dict[str, Any] = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "max_depth": 6,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    # Override for binary meta-labeling
    BINARY_PARAMS: dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_class": 1,
    }

    # Label mapping: {-1, 0, +1} → {0, 1, 2} for LightGBM
    _LABEL_MAP = {-1: 0, 0: 1, 1: 2}
    _LABEL_UNMAP = {0: -1, 1: 0, 2: 1}

    def __init__(
        self,
        params: dict | None = None,
        n_estimators: int = 1000,
        early_stopping_rounds: int = 50,
        val_fraction: float = 0.15,
        calibrate: bool = True,
        binary_mode: bool = False,
    ):
        """
        Args:
            params:                 LightGBM params override.
            n_estimators:           Max boosting rounds (early stop will cut).
            early_stopping_rounds:  Stop if no improvement for N rounds.
            val_fraction:           Fraction of training for validation.
            calibrate:              Apply isotonic probability calibration.
            binary_mode:            True for meta-labeling (binary {0,1}).
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        if binary_mode:
            self.params.update(self.BINARY_PARAMS)
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.calibrate = calibrate
        self.binary_mode = binary_mode

        self.model: lgb.Booster | None = None
        self.calibrators: dict[int, IsotonicRegression] = {}
        self.feature_names: list[str] = []
        self.shap_explainer: shap.TreeExplainer | None = None
        self.training_metrics: list[dict] = []
        self.best_iteration: int = 0

    # ── Validation split ──────────────────────────────────────

    def _split_validation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        w_train: pd.Series | None = None,
    ) -> tuple:
        """
        Carve validation set from END of training data (chronological).
        NOT random split — financial data is time-ordered.

        Returns
        -------
        (X_tr, X_val, y_tr, y_val, w_tr, w_val)
        """
        n = len(X_train)
        val_size = max(21, int(n * self.val_fraction))
        split_point = n - val_size

        X_tr = X_train.iloc[:split_point]
        X_val = X_train.iloc[split_point:]
        y_tr = y_train.iloc[:split_point]
        y_val = y_train.iloc[split_point:]

        w_tr = w_train.iloc[:split_point] if w_train is not None else None
        w_val = w_train.iloc[split_point:] if w_train is not None else None

        return X_tr, X_val, y_tr, y_val, w_tr, w_val

    # ── Training ──────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weights: pd.Series | None = None,
    ) -> ModelTrainer:
        """
        Train LightGBM with early stopping on a chronological
        validation set.

        Steps:
        1. Split validation from end of training data.
        2. Remap labels to 0-indexed for LightGBM (multiclass).
        3. Create LightGBM Datasets with sample weights.
        4. Train with early_stopping and log_evaluation callbacks.
        5. If calibrate: fit isotonic regression on validation probs.
        6. Initialise SHAP TreeExplainer.
        """
        self.feature_names = X_train.columns.tolist()

        # Split validation
        X_tr, X_val, y_tr, y_val, w_tr, w_val = self._split_validation(
            X_train, y_train, sample_weights
        )

        # Remap labels for multiclass
        if not self.binary_mode:
            y_tr_lgb = y_tr.map(self._LABEL_MAP)
            y_val_lgb = y_val.map(self._LABEL_MAP)
            # Safety: any unmapped values → 1 (neutral)
            y_tr_lgb = y_tr_lgb.fillna(1).astype(int)
            y_val_lgb = y_val_lgb.fillna(1).astype(int)
        else:
            y_tr_lgb = y_tr.astype(int)
            y_val_lgb = y_val.astype(int)

        # Build datasets
        d_train = lgb.Dataset(
            X_tr,
            label=y_tr_lgb,
            weight=w_tr.values if w_tr is not None else None,
            feature_name=self.feature_names,
            free_raw_data=False,
        )
        d_val = lgb.Dataset(
            X_val,
            label=y_val_lgb,
            weight=w_val.values if w_val is not None else None,
            reference=d_train,
            free_raw_data=False,
        )

        # Train with callbacks
        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=50),
        ]

        self.model = lgb.train(
            self.params,
            d_train,
            num_boost_round=self.n_estimators,
            valid_sets=[d_train, d_val],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        self.best_iteration = self.model.best_iteration
        logger.info(
            "Training complete: %d rounds (best iteration %d)",
            self.model.current_iteration(),
            self.best_iteration,
        )

        # ── Isotonic calibration ──────────────────────────────
        if self.calibrate:
            self._fit_calibrators(X_val, y_val_lgb)

        # ── SHAP explainer ────────────────────────────────────
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
        except Exception as exc:
            logger.warning("SHAP TreeExplainer init failed: %s", exc)
            self.shap_explainer = None

        return self

    def _fit_calibrators(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """
        Fit isotonic regression calibrators on validation predictions.

        For multiclass: one calibrator per class.
        For binary: single calibrator for P(class=1).
        """
        raw_proba = self.model.predict(
            X_val, num_iteration=self.best_iteration
        )

        if self.binary_mode:
            # raw_proba is 1-D for binary
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_proba, y_val.values)
            self.calibrators[1] = iso
        else:
            # raw_proba shape = (n, num_class)
            n_classes = raw_proba.shape[1]
            for c in range(n_classes):
                binary_target = (y_val.values == c).astype(int)
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(raw_proba[:, c], binary_target)
                self.calibrators[c] = iso

    # ── Prediction ────────────────────────────────────────────

    def predict(
        self,
        X: pd.DataFrame,
        return_proba: bool = True,
    ) -> pd.DataFrame:
        """
        Make predictions with optional isotonic calibration.

        Returns
        -------
        pd.DataFrame with columns:
            prediction : predicted class (-1, 0, +1 or 0/1).
            confidence : max probability across classes.
            proba_down / proba_neutral / proba_up : [multiclass only]
            proba_profitable : [binary only]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        raw_proba = self.model.predict(X, num_iteration=self.best_iteration)

        if self.binary_mode:
            proba = self._calibrate_binary(raw_proba)
            prediction = (proba >= 0.5).astype(int)
            result = pd.DataFrame(
                {
                    "prediction": prediction,
                    "confidence": np.where(proba >= 0.5, proba, 1.0 - proba),
                    "proba_profitable": proba,
                },
                index=X.index,
            )
        else:
            proba = self._calibrate_multiclass(raw_proba)
            pred_idx = proba.argmax(axis=1)
            prediction = np.array([self._LABEL_UNMAP[p] for p in pred_idx])
            result = pd.DataFrame(
                {
                    "prediction": prediction,
                    "confidence": proba.max(axis=1),
                    "proba_down": proba[:, 0],
                    "proba_neutral": proba[:, 1],
                    "proba_up": proba[:, 2],
                },
                index=X.index,
            )

        return result

    def _calibrate_binary(self, raw_proba: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration for binary mode."""
        if self.calibrate and 1 in self.calibrators:
            return self.calibrators[1].transform(raw_proba)
        return raw_proba

    def _calibrate_multiclass(self, raw_proba: np.ndarray) -> np.ndarray:
        """Apply per-class isotonic calibration and renormalise."""
        if not self.calibrate or not self.calibrators:
            return raw_proba

        calibrated = np.zeros_like(raw_proba)
        for c in range(raw_proba.shape[1]):
            if c in self.calibrators:
                calibrated[:, c] = self.calibrators[c].transform(raw_proba[:, c])
            else:
                calibrated[:, c] = raw_proba[:, c]

        # Renormalise rows to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        calibrated = calibrated / row_sums

        return calibrated

    # ── SHAP explanations ─────────────────────────────────────

    def explain(
        self,
        X: pd.DataFrame,
        top_k: int = 5,
    ) -> list[dict]:
        """
        SHAP per-prediction explanations.

        For each row in X returns a dict with:
            prediction, confidence, top_features (sorted by |SHAP|),
            base_value.

        Uses shap.TreeExplainer (natively supported for LightGBM).
        """
        if self.shap_explainer is None:
            logger.warning("SHAP explainer not available — returning empty.")
            return [{"prediction": 0, "confidence": 0.0, "top_features": [],
                      "base_value": 0.0}] * len(X)

        predictions = self.predict(X)
        shap_values = self.shap_explainer.shap_values(X)

        explanations: list[dict] = []
        for i in range(len(X)):
            pred = int(predictions["prediction"].iloc[i])
            conf = float(predictions["confidence"].iloc[i])

            if self.binary_mode:
                # Binary SHAP: list of 2 arrays or (n, features) or (n, features, 2)
                if isinstance(shap_values, list):
                    sv = shap_values[1][i]  # class 1 SHAP
                elif shap_values.ndim == 3:
                    sv = shap_values[i, :, 1]
                else:
                    sv = shap_values[i]
                ev = self.shap_explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    base = float(ev[1] if len(ev) > 1 else ev[0])
                else:
                    base = float(ev)
            else:
                # Multiclass: list of 3 arrays or (n, features, 3)
                pred_class_idx = self._LABEL_MAP.get(pred, 1)
                if isinstance(shap_values, list):
                    sv = shap_values[pred_class_idx][i]
                elif shap_values.ndim == 3:
                    sv = shap_values[i, :, pred_class_idx]
                else:
                    sv = shap_values[i]  # fallback
                ev = self.shap_explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    base = float(ev[pred_class_idx])
                else:
                    base = float(ev)

            # Sort features by |SHAP value| descending
            feature_indices = np.argsort(np.abs(sv))[::-1][:top_k]
            top_features = []
            for idx in feature_indices:
                fname = self.feature_names[idx] if idx < len(self.feature_names) else f"f{idx}"
                shap_val = float(sv[idx])
                feat_val = float(X.iloc[i, idx])
                top_features.append({
                    "feature": fname,
                    "shap_value": round(shap_val, 6),
                    "direction": "bullish" if shap_val > 0 else "bearish",
                    "feature_value": round(feat_val, 6),
                })

            explanations.append({
                "prediction": pred,
                "confidence": round(conf, 4),
                "top_features": top_features,
                "base_value": round(base, 6),
            })

        return explanations

    # ── Global importance ─────────────────────────────────────

    def get_global_importance(
        self,
        importance_type: str = "shap",
        X: pd.DataFrame | None = None,
    ) -> pd.Series:
        """
        Global feature importance.

        importance_type:
            "shap"  → mean(|SHAP values|) across X (requires X).
            "gain"  → LightGBM built-in gain importance.
            "split" → LightGBM built-in split count.

        Returns pd.Series sorted descending.
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        if importance_type == "shap":
            if X is None:
                raise ValueError("X required for SHAP importance.")
            if self.shap_explainer is None:
                logger.warning("SHAP unavailable, falling back to gain.")
                importance_type = "gain"
            else:
                shap_values = self.shap_explainer.shap_values(X)
                if isinstance(shap_values, list):
                    # Older SHAP: list of arrays, one per class
                    # Each element shape: (n_samples, n_features)
                    abs_shap = np.mean(
                        [np.abs(sv) for sv in shap_values], axis=0
                    )
                    # abs_shap shape: (n_samples, n_features)
                elif shap_values.ndim == 3:
                    # Newer SHAP: (n_samples, n_features, n_classes)
                    abs_shap = np.abs(shap_values).mean(axis=2)
                    # abs_shap shape: (n_samples, n_features)
                else:
                    # Binary: (n_samples, n_features)
                    abs_shap = np.abs(shap_values)
                mean_abs = abs_shap.mean(axis=0)  # (n_features,)
                return pd.Series(
                    mean_abs, index=self.feature_names, name="shap_importance"
                ).sort_values(ascending=False)

        # LightGBM built-in
        imp = self.model.feature_importance(
            importance_type=importance_type
        )
        return pd.Series(
            imp, index=self.feature_names, name=f"{importance_type}_importance"
        ).sort_values(ascending=False)


# ══════════════════════════════════════════════════════════════
# 4. Training Results Container
# ══════════════════════════════════════════════════════════════


class TrainingResults:
    """Container for cross-validated training results."""

    def __init__(self, symbol: str = ""):
        self.symbol = symbol
        self.fold_metrics: list[dict] = []
        self.aggregate_metrics: dict[str, Any] = {}
        self.is_significant: bool = False
        self.p_value: float = 1.0
        self.n_folds: int = 0

    def add_fold(self, metrics: dict, fold_num: int) -> None:
        """Add metrics from a single CV fold."""
        metrics["fold"] = fold_num
        self.fold_metrics.append(metrics)
        self.n_folds = len(self.fold_metrics)

    def compute_aggregate(self) -> None:
        """
        Compute aggregate statistics across folds.

        Statistical test: t-test whether mean IC > 0.
        """
        if not self.fold_metrics:
            return

        ics = [m["ic"] for m in self.fold_metrics]
        rics = [m["rank_ic"] for m in self.fold_metrics]
        hrs = [m["hit_rate"] for m in self.fold_metrics]
        pfs = [m["profit_factor"] for m in self.fold_metrics
               if m["profit_factor"] != float("inf")]

        self.aggregate_metrics = {
            "mean_ic": float(np.mean(ics)),
            "std_ic": float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0,
            "mean_rank_ic": float(np.mean(rics)),
            "std_rank_ic": float(np.std(rics, ddof=1)) if len(rics) > 1 else 0.0,
            "mean_hit_rate": float(np.mean(hrs)),
            "mean_profit_factor": float(np.mean(pfs)) if pfs else 0.0,
            "best_fold_ic": float(np.max(ics)),
            "worst_fold_ic": float(np.min(ics)),
            "n_folds": self.n_folds,
        }

        # Statistical significance: is mean IC > 0?
        if len(ics) >= 2:
            t_stat, p_value = sp_stats.ttest_1samp(ics, 0.0)
            self.p_value = float(p_value)
            self.is_significant = self.p_value < 0.05 and np.mean(ics) > 0
        else:
            self.p_value = 1.0
            self.is_significant = False

        self.aggregate_metrics["ic_pvalue"] = self.p_value
        self.aggregate_metrics["is_significant"] = self.is_significant

    def print_report(self) -> None:
        """Pretty-print comprehensive training report."""
        if not self.fold_metrics:
            print("No fold results available.")
            return

        self.compute_aggregate()

        print()
        print("═" * 60)
        print(f"  TRAINING REPORT: {self.symbol} "
              f"({self.n_folds}-Fold Purged Walk-Forward)")
        print("═" * 60)
        print()

        # Per-fold table
        print("  Per-Fold Results:")
        print("  ┌──────┬─────────┬──────────┬──────────┬───────────────┐")
        print("  │ Fold │   IC    │ Rank IC  │ Hit Rate │ Profit Factor │")
        print("  ├──────┼─────────┼──────────┼──────────┼───────────────┤")
        for m in self.fold_metrics:
            f = m["fold"]
            ic = m["ic"]
            ric = m["rank_ic"]
            hr = m["hit_rate"] * 100
            pf = m["profit_factor"]
            pf_str = f"{pf:8.2f}" if pf != float("inf") else "      ∞ "
            print(f"  │  {f:2d}  │ {ic:+.4f} │ {ric:+.4f}  │  {hr:5.1f}%  │{pf_str}      │")
        print("  └──────┴─────────┴──────────┴──────────┴───────────────┘")
        print()

        # Aggregate
        agg = self.aggregate_metrics
        sig_str = "✓ SIGNIFICANT" if self.is_significant else "✗ NOT significant"
        print(f"  Aggregate:")
        print(f"    Mean IC:            {agg['mean_ic']:+.4f} ± {agg['std_ic']:.4f}")
        print(f"    Mean Rank IC:       {agg['mean_rank_ic']:+.4f} ± {agg['std_rank_ic']:.4f}")
        print(f"    Mean Hit Rate:      {agg['mean_hit_rate'] * 100:.1f}%")
        print(f"    Mean Profit Factor: {agg['mean_profit_factor']:.2f}")
        print(f"    IC Significance:    p={self.p_value:.4f} {sig_str}")
        print()
        print("═" * 60)
        print()


# ══════════════════════════════════════════════════════════════
# 5. Training Pipeline  (Main Orchestrator)
# ══════════════════════════════════════════════════════════════


class TrainingPipeline:
    """
    End-to-end training pipeline that ties everything together.

    Usage::

        pipeline = TrainingPipeline(symbol="SPY")
        results = pipeline.run(df)   # df has OHLCV columns
        results.print_report()
        prediction = pipeline.predict_latest(current_features)
    """

    def __init__(
        self,
        symbol: str,
        feature_pipeline: FeaturePipeline | None = None,
        tp_multiplier: float = 2.0,
        sl_multiplier: float = 2.0,
        max_holding_period: int = 10,
        n_cv_splits: int = 5,
        binary_mode: bool = False,
        model_dir: str = "models/",
        data_fetcher=None,
    ):
        """
        Args:
            symbol:              Ticker symbol (e.g. "SPY").
            feature_pipeline:    Pre-configured FeaturePipeline (or None).
            tp_multiplier:       Take-profit in daily-vol units.
            sl_multiplier:       Stop-loss in daily-vol units.
            max_holding_period:  Vertical barrier in trading days.
            n_cv_splits:         Number of walk-forward CV folds.
            binary_mode:         True for meta-labeling.
            model_dir:           Directory to save trained models.
            data_fetcher:        Async callable for cross-asset features.
        """
        self.symbol = symbol
        self.feature_pipeline = feature_pipeline
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.max_holding_period = max_holding_period
        self.n_cv_splits = n_cv_splits
        self.binary_mode = binary_mode
        self.model_dir = model_dir
        self.data_fetcher = data_fetcher

        self.cv_results: list[dict] = []
        self.final_model: ModelTrainer | None = None
        self.final_feature_pipeline: FeaturePipeline | None = None
        self._training_date: str = ""

    # ── Main training loop ────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        primary_side: pd.Series | None = None,
    ) -> TrainingResults:
        """
        Full training pipeline.

        Steps:
        1. Create triple-barrier labels (or meta-labels).
        2. Compute sample weights.
        3. Set up PurgedWalkForwardCV.
        4. For each CV fold:
           a. Fit FeaturePipeline on training data ONLY.
           b. Transform train and test features.
           c. Train ModelTrainer with sample weights + early stopping.
           d. Predict on test set.
           e. Compute FinancialMetrics on test predictions.
           f. Store fold results.
        5. Aggregate CV metrics.
        6. Train FINAL model on ALL data.
        7. Save model + pipeline to disk.

        Returns TrainingResults.
        """
        self._training_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── 1. Labels ─────────────────────────────────────────
        if self.binary_mode and primary_side is not None:
            labeled = create_meta_labels(
                df,
                primary_side,
                tp_multiplier=self.tp_multiplier,
                sl_multiplier=self.sl_multiplier,
                max_holding_period=self.max_holding_period,
            )
            y_col = "meta_label"
        else:
            labeled = create_labels(
                df,
                tp_multiplier=self.tp_multiplier,
                sl_multiplier=self.sl_multiplier,
                max_holding_period=self.max_holding_period,
            )
            y_col = "label"

        if labeled.empty or len(labeled) < 100:
            logger.error(
                "Labeling produced only %d samples — aborting.",
                len(labeled),
            )
            return TrainingResults(self.symbol)

        y = labeled[y_col].astype(int)
        w = labeled["sample_weight"] if "sample_weight" in labeled.columns else None
        # Actual returns for IC computation
        actual_returns = (
            labeled["return_pct"]
            if "return_pct" in labeled.columns
            else pd.Series(0.0, index=labeled.index)
        )
        # Label end dates for precise purging
        label_end_dates = (
            pd.Series(
                pd.DatetimeIndex(labeled["barrier_date"]),
                index=labeled.index,
            )
            if "barrier_date" in labeled.columns
            else None
        )

        logger.info(
            "Labels: %d samples, distribution: %s",
            len(y), y.value_counts().to_dict(),
        )

        # ── 2. CV setup ──────────────────────────────────────
        cv = PurgedWalkForwardCV(
            n_splits=self.n_cv_splits,
            min_train_size=max(252, len(labeled) // 4),
            max_holding_period=self.max_holding_period,
            embargo_pct=0.01,
            expanding=True,
        )

        results = TrainingResults(symbol=self.symbol)

        # We need the original OHLCV data aligned to labels for
        # feature computation inside each fold.
        df_labeled = df.loc[df.index.isin(labeled.index)].copy()

        # ── 3. Cross-validated training ───────────────────────
        fold_num = 0
        for train_idx, test_idx in cv.split(
            df_labeled, y, label_end_dates=label_end_dates
        ):
            fold_num += 1
            logger.info("═" * 40)
            logger.info("FOLD %d", fold_num)
            logger.info("═" * 40)

            # Slice data
            df_train = df_labeled.iloc[train_idx]
            df_test = df_labeled.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            w_train = w.iloc[train_idx] if w is not None else None
            ret_test = actual_returns.iloc[test_idx]

            # ── a. Feature engineering (train/test isolation) ─
            try:
                X_train, pipe = build_features(
                    df_train, self.symbol, is_training=True,
                    data_fetcher=self.data_fetcher,
                )
                X_test, _ = build_features(
                    df_test, self.symbol, is_training=False,
                    pipeline=pipe,
                )
            except Exception as exc:
                logger.warning("Fold %d feature error: %s — skipping.", fold_num, exc)
                continue

            if X_train.empty or X_test.empty:
                logger.warning("Fold %d: empty features — skipping.", fold_num)
                continue

            # Align labels/weights to feature index
            y_tr = y_train.reindex(X_train.index).dropna()
            y_te = y_test.reindex(X_test.index).dropna()
            X_train = X_train.loc[y_tr.index]
            X_test = X_test.loc[y_te.index]
            ret_te = ret_test.reindex(X_test.index)
            w_tr = w_train.reindex(X_train.index) if w_train is not None else None

            if len(X_train) < 50 or len(X_test) < 21:
                logger.warning(
                    "Fold %d: insufficient data (train=%d, test=%d) — skipping.",
                    fold_num, len(X_train), len(X_test),
                )
                continue

            # ── b. Train model ────────────────────────────────
            trainer = ModelTrainer(
                binary_mode=self.binary_mode,
                calibrate=True,
            )
            try:
                trainer.train(X_train, y_tr, sample_weights=w_tr)
            except Exception as exc:
                logger.warning("Fold %d training error: %s — skipping.", fold_num, exc)
                continue

            # ── c. Predict on test ────────────────────────────
            pred_df = trainer.predict(X_test)

            # Convert predictions to signed scores for IC
            if self.binary_mode:
                # For meta: proba_profitable − 0.5 → signed score
                pred_scores = pred_df["proba_profitable"] - 0.5
            else:
                # For multiclass: P(up) − P(down) → signed score
                pred_scores = pred_df["proba_up"] - pred_df["proba_down"]

            # ── d. Evaluate ───────────────────────────────────
            fold_metrics = FinancialMetrics.evaluate_all(
                pred_scores, ret_te, labels=y_te,
            )
            fold_metrics["n_estimators_used"] = trainer.best_iteration
            fold_metrics["n_features"] = len(trainer.feature_names)

            results.add_fold(fold_metrics, fold_num)

            logger.info(
                "Fold %d: IC=%.4f (p=%.3f), Hit=%.1f%%, PF=%.2f, "
                "rounds=%d",
                fold_num,
                fold_metrics["ic"],
                fold_metrics["ic_pvalue"],
                fold_metrics["hit_rate"] * 100,
                fold_metrics["profit_factor"],
                trainer.best_iteration,
            )

        # ── 4. Aggregate ─────────────────────────────────────
        results.compute_aggregate()

        # ── 5. Train FINAL model on ALL data ──────────────────
        logger.info("Training final model on ALL data…")
        try:
            X_all, final_pipe = build_features(
                df_labeled, self.symbol, is_training=True,
                data_fetcher=self.data_fetcher,
            )
            y_all = y.reindex(X_all.index).dropna()
            X_all = X_all.loc[y_all.index]
            w_all = w.reindex(X_all.index) if w is not None else None

            self.final_model = ModelTrainer(
                binary_mode=self.binary_mode,
                calibrate=True,
            )
            self.final_model.train(X_all, y_all, sample_weights=w_all)
            self.final_feature_pipeline = final_pipe
        except Exception as exc:
            logger.error("Final model training failed: %s", exc)

        # ── 6. Save to disk ───────────────────────────────────
        try:
            self.save()
        except Exception as exc:
            logger.warning("Model save failed: %s", exc)

        return results

    # ── Predict latest ────────────────────────────────────────

    def predict_latest(self, df: pd.DataFrame) -> dict:
        """
        Make a prediction on the latest data using the final model.

        Args:
            df: Recent OHLCV data (must be enough for feature computation).

        Returns
        -------
        dict with keys: symbol, prediction, confidence, probabilities,
        explanation, model_info.
        """
        if self.final_model is None or self.final_feature_pipeline is None:
            raise ValueError("No final model. Call run() first or load().")

        # Transform features using final pipeline (no re-fitting)
        features = self.final_feature_pipeline.transform(df, self.symbol)

        if features.empty:
            return {
                "symbol": self.symbol,
                "prediction": 0,
                "confidence": 0.0,
                "probabilities": {},
                "explanation": {"top_features": [], "base_value": 0.0},
                "model_info": {},
            }

        # Use only the last row
        X_latest = features.iloc[[-1]]

        pred_df = self.final_model.predict(X_latest)
        explanations = self.final_model.explain(X_latest)

        prediction = int(pred_df["prediction"].iloc[0])
        confidence = float(pred_df["confidence"].iloc[0])

        if self.binary_mode:
            proba = {"proba_profitable": float(pred_df["proba_profitable"].iloc[0])}
        else:
            proba = {
                "proba_down": float(pred_df["proba_down"].iloc[0]),
                "proba_neutral": float(pred_df["proba_neutral"].iloc[0]),
                "proba_up": float(pred_df["proba_up"].iloc[0]),
            }

        return {
            "symbol": self.symbol,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": proba,
            "explanation": {
                "top_features": explanations[0]["top_features"] if explanations else [],
                "base_value": explanations[0]["base_value"] if explanations else 0.0,
            },
            "model_info": {
                "cv_mean_ic": self.cv_results[-1]["ic"] if self.cv_results else 0.0,
                "n_estimators_used": self.final_model.best_iteration,
                "training_date": self._training_date,
                "binary_mode": self.binary_mode,
            },
        }

    # ── Save / Load ───────────────────────────────────────────

    def save(self, path: str | None = None) -> str:
        """
        Save trained model, feature pipeline, and metadata to disk.

        Returns the save directory path.
        """
        if self.final_model is None:
            raise ValueError("No final model to save.")

        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(
                self.model_dir,
                f"{self.symbol}_{timestamp}",
            )

        Path(path).mkdir(parents=True, exist_ok=True)

        # 1. Save LightGBM model
        model_path = os.path.join(path, "model.txt")
        self.final_model.model.save_model(model_path)

        # 2. Save calibrators
        if self.final_model.calibrators:
            cal_path = os.path.join(path, "calibrators.pkl")
            joblib.dump(self.final_model.calibrators, cal_path)

        # 3. Save feature pipeline
        pipe_path = os.path.join(path, "feature_pipeline.pkl")
        joblib.dump(self.final_feature_pipeline, pipe_path)

        # 4. Save metadata
        metadata = {
            "symbol": self.symbol,
            "binary_mode": self.binary_mode,
            "tp_multiplier": self.tp_multiplier,
            "sl_multiplier": self.sl_multiplier,
            "max_holding_period": self.max_holding_period,
            "n_cv_splits": self.n_cv_splits,
            "training_date": self._training_date,
            "feature_names": self.final_model.feature_names,
            "best_iteration": self.final_model.best_iteration,
            "params": self.final_model.params,
        }
        meta_path = os.path.join(path, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info("Model saved to %s", path)
        return path

    @classmethod
    def load(cls, path: str) -> TrainingPipeline:
        """
        Load a saved pipeline from disk.

        Returns a TrainingPipeline ready for ``predict_latest()``.
        """
        # Load metadata
        meta_path = os.path.join(path, "metadata.json")
        with open(meta_path) as f:
            metadata = json.load(f)

        pipeline = cls(
            symbol=metadata["symbol"],
            binary_mode=metadata.get("binary_mode", False),
            tp_multiplier=metadata.get("tp_multiplier", 2.0),
            sl_multiplier=metadata.get("sl_multiplier", 2.0),
            max_holding_period=metadata.get("max_holding_period", 10),
            n_cv_splits=metadata.get("n_cv_splits", 5),
        )
        pipeline._training_date = metadata.get("training_date", "")

        # Load LightGBM model
        model_path = os.path.join(path, "model.txt")
        booster = lgb.Booster(model_file=model_path)

        trainer = ModelTrainer(
            params=metadata.get("params", {}),
            binary_mode=metadata.get("binary_mode", False),
        )
        trainer.model = booster
        trainer.feature_names = metadata.get("feature_names", [])
        trainer.best_iteration = metadata.get("best_iteration", 0)

        # Load calibrators
        cal_path = os.path.join(path, "calibrators.pkl")
        if os.path.exists(cal_path):
            trainer.calibrators = joblib.load(cal_path)

        # Initialise SHAP explainer
        try:
            trainer.shap_explainer = shap.TreeExplainer(booster)
        except Exception:
            trainer.shap_explainer = None

        pipeline.final_model = trainer

        # Load feature pipeline
        pipe_path = os.path.join(path, "feature_pipeline.pkl")
        if os.path.exists(pipe_path):
            pipeline.final_feature_pipeline = joblib.load(pipe_path)

        logger.info("Model loaded from %s", path)
        return pipeline
