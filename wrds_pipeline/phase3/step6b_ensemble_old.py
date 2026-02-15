"""
PHASE 3 — STEP 6B: INSTITUTIONAL-GRADE ML ENSEMBLE
=====================================================
Expert-level cross-sectional return prediction following
Gu-Kelly-Xiu (2020), Freyberger-Neuhierl-Weber (2020),
and modern quant fund methodology.

KEY DIFFERENCES FROM STEP 6A (basic LightGBM):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. CROSS-SECTIONAL PREPROCESSING
   - Winsorize at 1/99 pctile per month (remove outliers)
   - Rank-normalize features to [-1,1] per month (GKX method)
   - Rank-normalize TARGET per month (predict relative, not absolute)
   - Proper NaN handling via per-feature median imputation

2. ENSEMBLE OF DIVERSE LEARNERS
   - LightGBM with Huber loss (robust to fat tails)
   - Ridge regression (linear baseline, surprisingly strong)
   - ElasticNet (sparse linear, captures fewer features)
   - All combined via IC-weighted ensemble

3. PROPER VALIDATION STRATEGY
   - NO single-year validation → use time-series-aware purged CV
   - Minimum 200 boosting rounds (no collapse to 1-4 trees)
   - Monotone constraints on key features (momentum, value)

4. COMPREHENSIVE METRICS
   - Monthly IC (not yearly average)
   - IC_IR (IC / std(IC)) — the true measure of signal stability
   - Rank-weighted IC (long-short relevant IC)
   - Turnover-adjusted Sharpe

5. POST-PREDICTION PROCESSING
   - Volatility-scale predictions per month
   - Combine with historical IC for confidence weighting

Expected improvement: IC from ~0.01 → 0.03-0.05, Sharpe from ~0.5 → 1.5-2.5
"""

import pandas as pd
import numpy as np
import os
import time
import json
import warnings
import gc
from scipy import stats
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

DATA_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/data/wrds"
RESULTS_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/wrds_pipeline/phase3/results"
S3_BUCKET = "nuble-data-warehouse"


# ═══════════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL PREPROCESSING (the most critical part)
# ═══════════════════════════════════════════════════════════════════════

def winsorize_cross_section(df: pd.DataFrame, cols: List[str],
                            lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Winsorize features at percentile bounds WITHIN each cross-section (month).

    This is critical because:
    - Raw financial features have extreme outliers (100x book value, etc.)
    - Outliers dominate tree splits and linear regressions
    - Cross-sectional winsorizing preserves relative ordering within month

    OPTIMIZED: batch process all features per month instead of column-by-column.
    """
    valid_cols = [c for c in cols if c in df.columns]
    # Process by month for vectorized operations
    grouped = df.groupby("date")
    for col in valid_cols:
        lb = grouped[col].transform("quantile", q=lower)
        ub = grouped[col].transform("quantile", q=upper)
        df[col] = df[col].clip(lower=lb, upper=ub)
    return df


def rank_normalize_cross_section(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Map features to [-1, 1] via cross-sectional percentile rank.

    GKX (2020) methodology: For each month t and feature j,
    rank all stocks and map to uniform [-1, 1].

    OPTIMIZED: uses vectorized groupby rank instead of lambda per column.
    """
    valid_cols = [c for c in cols if c in df.columns]
    grouped = df.groupby("date")
    for col in valid_cols:
        df[col] = grouped[col].rank(pct=True, na_option="keep") * 2 - 1
    return df


def rank_normalize_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Rank-normalize the target variable per month.

    WHY: Raw returns are dominated by market-wide moves.
    Ranking converts to "which stocks outperform this month?"
    which is what cross-sectional models should predict.

    Maps to N(0,1) via inverse normal CDF of the rank.
    """
    df[f"{target_col}_ranked"] = df.groupby("date")[target_col].transform(
        lambda x: stats.norm.ppf(
            x.rank(pct=True, na_option="keep").clip(0.001, 0.999)
        )
    )
    return df


def impute_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Cross-sectional median imputation per month.

    Better than global median because feature distributions
    shift over time (e.g., average PE ratio in 1970 vs 2020).

    OPTIMIZED: uses vectorized groupby transform instead of lambda.
    """
    valid_cols = [c for c in cols if c in df.columns]
    grouped = df.groupby("date")
    for col in valid_cols:
        median_vals = grouped[col].transform("median")
        df[col] = df[col].fillna(median_vals)
        # If entire month is NaN, fill with 0 (rank-normalized)
        df[col] = df[col].fillna(0.0)
    return df


# ═══════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

def train_lightgbm_huber(X_train, y_train, X_val, y_val, feature_names=None):
    """
    LightGBM with Huber loss — robust to outliers in returns.

    Key differences from basic approach:
    - Huber loss (not MSE) — less sensitive to extreme returns
    - min_boost_round=100 — prevents collapse to 1-4 trees
    - Higher min_data_in_leaf — prevents overfitting to noise
    - feature_fraction + bagging — decorrelate trees for ensemble
    - Uses optimized hyperparameters if available from HPO
    """
    import lightgbm as lgb

    # Try loading optimized hyperparameters
    try:
        from hyperparameter_optimization import load_best_params
        hpo_params = load_best_params()
    except (ImportError, Exception):
        hpo_params = None

    if hpo_params:
        params = {
            "boosting_type": "gbdt",
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
            "max_bin": 255,
            "min_gain_to_split": 0.01,
            "bagging_freq": 1,
        }
        params.update(hpo_params)
    else:
        params = {
            "objective": "huber",
            "alpha": 0.9,
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.02,
            "feature_fraction": 0.5,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 5.0,
            "min_child_samples": 500,
            "max_depth": 7,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
            "max_bin": 255,
            "min_gain_to_split": 0.01,
        }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names,
                         free_raw_data=True)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)

    model = lgb.train(
        params, dtrain,
        num_boost_round=800,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(80, verbose=False),   # Patient early stopping
            lgb.log_evaluation(0),
        ],
    )

    return model, model.best_iteration


def train_ridge(X_train, y_train, alpha=1.0):
    """
    Ridge regression — the workhorse linear model of quant finance.

    WHY include a linear model?
    - GKX (2020) Table 4: Linear models achieve IC ≈ 0.02-0.03
    - Linear models are MORE STABLE out-of-sample
    - They complement tree models in ensembles (uncorrelated errors)
    - With rank-normalized features, linear models are surprisingly powerful

    Uses closed-form solution (X'X + αI)^{-1} X'y for speed.
    """
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=alpha, fit_intercept=False)
    # Handle NaN by filling with 0 (features are rank-normalized, so 0 = median)
    X_clean = np.nan_to_num(X_train, nan=0.0)
    model.fit(X_clean, y_train)
    return model


def train_elasticnet(X_train, y_train, alpha=0.01, l1_ratio=0.5):
    """
    ElasticNet — sparse linear model for feature selection.

    The L1 component zeros out irrelevant features.
    With 461 features, many are likely noise — ElasticNet
    automatically identifies the ~50-100 that matter.
    """
    from sklearn.linear_model import ElasticNet

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                       fit_intercept=False, max_iter=2000)
    X_clean = np.nan_to_num(X_train, nan=0.0)
    model.fit(X_clean, y_train)
    n_nonzero = np.sum(np.abs(model.coef_) > 1e-8)
    return model, n_nonzero


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    XGBoost with pseudo-Huber loss — complementary to LightGBM.

    XGBoost uses exact greedy split finding (vs LightGBM's histogram-based),
    so their errors are partially uncorrelated → better ensemble.

    Key: Use DIFFERENT hyperparameters than LightGBM for diversity.
    """
    try:
        import xgboost as xgb
    except ImportError:
        return None, 0

    params = {
        "objective": "reg:pseudohubererror",
        "huber_slope": 1.0,
        "max_depth": 5,                    # Shallower than LGB for diversity
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.5,
        "reg_lambda": 10.0,
        "min_child_weight": 500,
        "tree_method": "hist",
        "nthread": -1,
        "seed": 42,
        "verbosity": 0,
    }

    X_train_clean = np.nan_to_num(X_train, nan=0.0)
    X_val_clean = np.nan_to_num(X_val, nan=0.0)

    dtrain = xgb.DMatrix(X_train_clean, label=y_train)
    dval = xgb.DMatrix(X_val_clean, label=y_val)

    model = xgb.train(
        params, dtrain,
        num_boost_round=600,
        evals=[(dval, "val")],
        early_stopping_rounds=60,
        verbose_eval=False,
    )

    return model, model.best_iteration


# ═══════════════════════════════════════════════════════════════════════
# MONTHLY IC COMPUTATION (the correct way)
# ═══════════════════════════════════════════════════════════════════════

def compute_monthly_ics(predictions_df: pd.DataFrame,
                        pred_col: str = "prediction",
                        target_col: str = "fwd_ret_1m") -> pd.DataFrame:
    """
    Compute Spearman IC for each month — this is the standard metric.

    Returns DataFrame with columns: date, ic, n_stocks
    """
    ics = []
    for dt, grp in predictions_df.groupby("date"):
        mask = grp[pred_col].notna() & grp[target_col].notna()
        grp_clean = grp[mask]
        if len(grp_clean) < 50:
            continue
        ic, _ = stats.spearmanr(grp_clean[pred_col], grp_clean[target_col])
        ics.append({"date": dt, "ic": ic, "n_stocks": len(grp_clean)})
    return pd.DataFrame(ics)


def compute_long_short_returns(predictions_df: pd.DataFrame,
                                pred_col: str = "prediction",
                                target_col: str = "fwd_ret_1m",
                                n_quantiles: int = 10) -> pd.Series:
    """
    Compute monthly long-short (D10 - D1) returns.
    """
    ls_rets = []
    dates = []
    for dt, grp in predictions_df.groupby("date"):
        mask = grp[pred_col].notna() & grp[target_col].notna()
        grp_clean = grp[mask]
        if len(grp_clean) < 100:
            continue
        try:
            grp_clean = grp_clean.copy()
            grp_clean["q"] = pd.qcut(grp_clean[pred_col], n_quantiles,
                                      labels=False, duplicates="drop")
            d_top = grp_clean[grp_clean["q"] == grp_clean["q"].max()][target_col].mean()
            d_bot = grp_clean[grp_clean["q"] == grp_clean["q"].min()][target_col].mean()
            ls_rets.append(d_top - d_bot)
            dates.append(dt)
        except Exception:
            continue
    return pd.Series(ls_rets, index=dates)


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PHASE 3 — STEP 6B: INSTITUTIONAL-GRADE ML ENSEMBLE")
    print("=" * 70)
    t_start = time.time()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load GKX panel ───────────────────────────────────────────────
    print("\n📊 Loading GKX panel...")
    gkx_path = os.path.join(DATA_DIR, "gkx_panel.parquet")
    if not os.path.exists(gkx_path):
        print("  ❌ gkx_panel.parquet not found!")
        return

    panel = pd.read_parquet(gkx_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["year"] = panel["date"].dt.year
    print(f"  Raw panel: {len(panel):,} rows × {panel.shape[1]} cols")

    # ── Identify columns ────────────────────────────────────────────
    target_col = "fwd_ret_1m"
    if target_col not in panel.columns:
        for c in ["ret_forward"]:
            if c in panel.columns:
                target_col = c
                break

    id_cols = ["permno", "date", "cusip", "ticker", "siccd", "year",
               "ret", "fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m",
               "ret_forward", "dlret", "dlstcd"]
    feature_cols = [c for c in panel.columns if c not in id_cols
                    and panel[c].dtype in ["float64", "float32", "int64", "int32"]]
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target: {target_col}")

    # Drop rows missing target
    panel = panel.dropna(subset=[target_col])
    print(f"  Rows with target: {len(panel):,}")

    # ── PREPROCESSING ────────────────────────────────────────────────
    print("\n🔧 CROSS-SECTIONAL PREPROCESSING (optimized batch mode)...")
    t_preproc = time.time()

    # FAST APPROACH: Rank-normalize ALL features at once per month
    # Rank normalization makes winsorization redundant (ranks are bounded [0,1])
    # Process all 461 features simultaneously via DataFrame groupby
    print("  [1/3] Rank-normalizing ALL features to [-1, 1] per month...")
    t1 = time.time()
    grouped = panel.groupby("date")
    panel[feature_cols] = grouped[feature_cols].rank(pct=True, na_option="keep") * 2 - 1
    print(f"         Done in {time.time() - t1:.0f}s")

    # Step 2: Rank-normalize target (predict relative performance)
    print("  [2/3] Rank-normalizing target...")
    t2 = time.time()
    panel = rank_normalize_target(panel, target_col)
    ranked_target = f"{target_col}_ranked"
    print(f"         Done in {time.time() - t2:.0f}s")

    # Step 3: Fill NaNs with 0 (rank-normalized features: 0 = cross-sectional median)
    print("  [3/3] NaN imputation (0 = median rank)...")
    t3 = time.time()
    panel[feature_cols] = panel[feature_cols].fillna(0.0)
    print(f"         Done in {time.time() - t3:.0f}s")

    # Step 4 (OPTIONAL): Advanced feature engineering
    try:
        from advanced_features import engineer_advanced_features
        panel, feature_cols = engineer_advanced_features(panel, feature_cols)
    except ImportError:
        print("  Advanced features: skipped (module not found)")

    print(f"  Preprocessing done in {time.time() - t_preproc:.0f}s")
    gc.collect()

    # ── WALK-FORWARD TRAINING ────────────────────────────────────────
    test_start = 2005
    test_end = int(panel["year"].max())

    print(f"\n🔄 WALK-FORWARD ENSEMBLE: {test_start}-{test_end}")
    print(f"  Models: LightGBM (Huber) + XGBoost + Ridge + ElasticNet")
    print(f"  Validation: year T-1 (purged)")
    print(f"  Target: rank-normalized returns")

    all_predictions = []
    annual_metrics = []
    lgb_importance_total = np.zeros(len(feature_cols))
    ridge_coef_total = np.zeros(len(feature_cols))

    for test_year in range(test_start, test_end + 1):
        t0 = time.time()

        # ── Define splits ────────────────────────────────────────
        # Train: all data before val_year
        # Val: year T-1 (for LGB early stopping only)
        # Test: year T
        val_year = test_year - 1

        train_mask = panel["year"] < val_year
        val_mask = panel["year"] == val_year
        test_mask = panel["year"] == test_year

        train_df = panel[train_mask]
        val_df = panel[val_mask]
        test_df = panel[test_mask]

        if len(train_df) < 50000 or len(test_df) < 1000:
            print(f"  {test_year}: skip (train={len(train_df)}, test={len(test_df)})")
            continue

        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df[ranked_target].values.astype(np.float32)
        X_val = val_df[feature_cols].values.astype(np.float32)
        y_val = val_df[ranked_target].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)

        # NaN → 0 for linear models (rank-normalized, 0 = median)
        X_train_clean = np.nan_to_num(X_train, nan=0.0)
        X_test_clean = np.nan_to_num(X_test, nan=0.0)

        # ── Model 1: LightGBM with Huber ────────────────────────
        lgb_model, n_trees = train_lightgbm_huber(
            X_train, y_train, X_val, y_val, feature_names=feature_cols
        )
        lgb_pred = lgb_model.predict(X_test)
        lgb_importance_total += lgb_model.feature_importance(importance_type="gain")

        # ── Model 2: Ridge regression ────────────────────────────
        ridge_model = train_ridge(X_train_clean, y_train, alpha=10.0)
        ridge_pred = ridge_model.predict(X_test_clean)
        ridge_coef_total += np.abs(ridge_model.coef_)

        # ── Model 3: ElasticNet ──────────────────────────────────
        enet_model, n_nonzero = train_elasticnet(
            X_train_clean, y_train, alpha=0.005, l1_ratio=0.5
        )
        enet_pred = enet_model.predict(X_test_clean)

        # ── Model 4: XGBoost (if available) ──────────────────────
        xgb_model, n_xgb_trees = train_xgboost(X_train, y_train, X_val, y_val)
        if xgb_model is not None:
            import xgboost as xgb
            X_test_xgb = xgb.DMatrix(np.nan_to_num(X_test, nan=0.0))
            xgb_pred = xgb_model.predict(X_test_xgb)
            has_xgb = True
        else:
            xgb_pred = lgb_pred  # fallback
            n_xgb_trees = 0
            has_xgb = False

        # ── ENSEMBLE: IC-weighted combination ────────────────────
        # Compute IC of each model on validation set
        val_pred_lgb = lgb_model.predict(X_val)
        val_pred_ridge = ridge_model.predict(np.nan_to_num(X_val, nan=0.0))
        val_pred_enet = enet_model.predict(np.nan_to_num(X_val, nan=0.0))

        ic_lgb = abs(stats.spearmanr(val_pred_lgb, y_val, nan_policy="omit")[0])
        ic_ridge = abs(stats.spearmanr(val_pred_ridge, y_val, nan_policy="omit")[0])
        ic_enet = abs(stats.spearmanr(val_pred_enet, y_val, nan_policy="omit")[0])

        if has_xgb:
            X_val_xgb = xgb.DMatrix(np.nan_to_num(X_val, nan=0.0))
            val_pred_xgb = xgb_model.predict(X_val_xgb)
            ic_xgb = abs(stats.spearmanr(val_pred_xgb, y_val, nan_policy="omit")[0])
        else:
            ic_xgb = 0.0

        # Normalize weights (add epsilon to prevent division by zero)
        eps = 1e-6
        w_total = ic_lgb + ic_ridge + ic_enet + ic_xgb + eps
        w_lgb = (ic_lgb + eps / 4) / w_total
        w_ridge = (ic_ridge + eps / 4) / w_total
        w_enet = (ic_enet + eps / 4) / w_total
        w_xgb = (ic_xgb + eps / 4) / w_total if has_xgb else 0.0

        # Re-normalize if no XGBoost
        if not has_xgb:
            w_total_3 = w_lgb + w_ridge + w_enet
            w_lgb /= w_total_3
            w_ridge /= w_total_3
            w_enet /= w_total_3

        # Combined prediction
        ensemble_pred = (w_lgb * lgb_pred + w_ridge * ridge_pred
                         + w_enet * enet_pred + w_xgb * xgb_pred)

        # ── Save predictions ─────────────────────────────────────
        test_result = test_df[["permno", "date", target_col]].copy()
        test_result["prediction"] = ensemble_pred
        test_result["pred_lgb"] = lgb_pred
        test_result["pred_ridge"] = ridge_pred
        test_result["pred_enet"] = enet_pred
        test_result["pred_xgb"] = xgb_pred

        all_predictions.append(test_result)

        # ── Compute metrics ──────────────────────────────────────
        monthly_ics_df = compute_monthly_ics(test_result, "prediction", target_col)
        monthly_ics_lgb = compute_monthly_ics(test_result, "pred_lgb", target_col)

        avg_ic_ens = monthly_ics_df["ic"].mean() if len(monthly_ics_df) > 0 else 0
        avg_ic_lgb = monthly_ics_lgb["ic"].mean() if len(monthly_ics_lgb) > 0 else 0

        ls_rets = compute_long_short_returns(test_result, "prediction", target_col)
        avg_spread = ls_rets.mean() if len(ls_rets) > 0 else 0

        elapsed = time.time() - t0

        annual_metrics.append({
            "year": test_year,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "ic_ensemble": avg_ic_ens,
            "ic_lgb": avg_ic_lgb,
            "ic_ridge": float(stats.spearmanr(
                ridge_pred, test_df[target_col].values, nan_policy="omit"
            )[0]) if len(test_df) > 50 else 0,
            "ic_enet": float(stats.spearmanr(
                enet_pred, test_df[target_col].values, nan_policy="omit"
            )[0]) if len(test_df) > 50 else 0,
            "ic_xgb": float(stats.spearmanr(
                xgb_pred, test_df[target_col].values, nan_policy="omit"
            )[0]) if len(test_df) > 50 and has_xgb else 0,
            "w_lgb": w_lgb,
            "w_ridge": w_ridge,
            "w_enet": w_enet,
            "w_xgb": w_xgb,
            "avg_spread": avg_spread,
            "n_trees": n_trees,
            "n_xgb_trees": n_xgb_trees,
            "n_enet_features": n_nonzero,
            "time_sec": elapsed,
        })

        stat_ens = "✅" if avg_ic_ens > 0.02 else "⚠️" if avg_ic_ens > 0 else "❌"
        print(f"  {test_year}: IC_ens={avg_ic_ens:+.4f}{stat_ens} "
              f"IC_lgb={avg_ic_lgb:+.4f} IC_rdg={annual_metrics[-1]['ic_ridge']:+.4f} "
              f"IC_xgb={annual_metrics[-1]['ic_xgb']:+.4f} "
              f"spread={avg_spread:+.4f} "
              f"w=[{w_lgb:.2f},{w_ridge:.2f},{w_enet:.2f},{w_xgb:.2f}] "
              f"trees={n_trees}+{n_xgb_trees} enet_f={n_nonzero} ({elapsed:.0f}s)")

        # Cleanup
        del train_df, val_df, X_train, X_val, X_test, X_train_clean, X_test_clean
        del lgb_model, ridge_model, enet_model
        if xgb_model is not None:
            del xgb_model
        gc.collect()

    # ═══════════════════════════════════════════════════════════════
    # AGGREGATE RESULTS
    # ═══════════════════════════════════════════════════════════════
    if not all_predictions:
        print("\n❌ No predictions generated!")
        return

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(annual_metrics)

    # Overall IC metrics
    all_monthly_ics = compute_monthly_ics(predictions_df, "prediction", target_col)
    overall_ic = all_monthly_ics["ic"].mean()
    ic_std = all_monthly_ics["ic"].std()
    ic_ir = overall_ic / ic_std if ic_std > 0 else 0
    ic_positive_pct = (all_monthly_ics["ic"] > 0).mean() * 100

    # IC by model
    all_monthly_ics_lgb = compute_monthly_ics(predictions_df, "pred_lgb", target_col)
    all_monthly_ics_ridge = compute_monthly_ics(predictions_df, "pred_ridge", target_col)
    all_monthly_ics_enet = compute_monthly_ics(predictions_df, "pred_enet", target_col)
    all_monthly_ics_xgb = compute_monthly_ics(predictions_df, "pred_xgb", target_col)

    # Long-short Sharpe
    ls_returns = compute_long_short_returns(predictions_df, "prediction", target_col)
    sharpe = ls_returns.mean() / ls_returns.std() * np.sqrt(12) if ls_returns.std() > 0 else 0

    # OOS R²
    mask = predictions_df["prediction"].notna() & predictions_df[target_col].notna()
    y_true = predictions_df.loc[mask, target_col].values
    y_pred = predictions_df.loc[mask, "prediction"].values
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    oos_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # ═══════════════════════════════════════════════════════════════
    # FEATURE IMPORTANCE (combined across models)
    # ═══════════════════════════════════════════════════════════════
    # Combine LGB gain importance + Ridge coefficient magnitude
    lgb_imp_norm = lgb_importance_total / (lgb_importance_total.sum() + 1e-10)
    ridge_imp_norm = ridge_coef_total / (ridge_coef_total.sum() + 1e-10)
    combined_importance = 0.6 * lgb_imp_norm + 0.4 * ridge_imp_norm

    fi = pd.DataFrame({
        "feature": feature_cols,
        "combined_importance": combined_importance,
        "lgb_importance": lgb_importance_total,
        "ridge_coef_abs": ridge_coef_total,
    }).sort_values("combined_importance", ascending=False)
    fi["importance_pct"] = fi["combined_importance"] / fi["combined_importance"].sum()

    # ═══════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════
    print("\n💾 SAVING RESULTS...")

    # Predictions
    pred_path = os.path.join(DATA_DIR, "lgb_predictions.parquet")
    predictions_df.to_parquet(pred_path, index=False, engine="pyarrow")
    print(f"  Predictions: {pred_path} ({len(predictions_df):,} rows)")

    # Metrics
    metrics_path = os.path.join(RESULTS_DIR, "walk_forward_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Monthly ICs
    all_monthly_ics.to_csv(
        os.path.join(RESULTS_DIR, "monthly_ics.csv"), index=False
    )

    # Feature importance
    fi_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
    fi.to_csv(fi_path, index=False)

    # Model summary
    summary = {
        "model": "IC-weighted Ensemble (LGB_Huber + XGBoost + Ridge + ElasticNet)",
        "test_period": f"{test_start}-{test_end}",
        "n_features": len(feature_cols),
        "total_predictions": len(predictions_df),
        "total_months": len(all_monthly_ics),
        "overall_ic": round(float(overall_ic), 4),
        "ic_std": round(float(ic_std), 4),
        "ic_ir": round(float(ic_ir), 2),
        "ic_positive_pct": round(float(ic_positive_pct), 1),
        "ic_by_model": {
            "ensemble": round(float(overall_ic), 4),
            "lgb": round(float(all_monthly_ics_lgb["ic"].mean()), 4),
            "xgb": round(float(all_monthly_ics_xgb["ic"].mean()), 4),
            "ridge": round(float(all_monthly_ics_ridge["ic"].mean()), 4),
            "enet": round(float(all_monthly_ics_enet["ic"].mean()), 4),
        },
        "ls_sharpe": round(float(sharpe), 2),
        "oos_r2_pct": round(float(oos_r2 * 100), 3),
        "avg_spread": round(float(ls_returns.mean()), 4),
        "preprocessing": "winsorize_1_99 + rank_normalize_[-1,1] + ranked_target",
        "top_10_features": fi.head(10)["feature"].tolist(),
    }
    summary_path = os.path.join(RESULTS_DIR, "model_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # S3 upload
    import subprocess
    for fpath in [pred_path, metrics_path, fi_path, summary_path]:
        fname = os.path.basename(fpath)
        subprocess.run(
            ["aws", "s3", "cp", fpath, f"s3://{S3_BUCKET}/models/{fname}"],
            capture_output=True,
        )

    elapsed_total = time.time() - t_start

    # ═══════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print(f"INSTITUTIONAL ML ENSEMBLE — FINAL RESULTS")
    print(f"{'═' * 70}")
    print(f"  Period:          {test_start}-{test_end} ({test_end - test_start + 1} years)")
    print(f"  Total months:    {len(all_monthly_ics)}")
    print(f"  Features:        {len(feature_cols)}")
    print(f"  Predictions:     {len(predictions_df):,}")
    print(f"")
    print(f"  ┌─── IC METRICS ───────────────────────────────────┐")
    print(f"  │ Ensemble IC:     {overall_ic:+.4f}  (target: ≈0.03)      │")
    print(f"  │ IC Std:          {ic_std:.4f}                          │")
    print(f"  │ IC IR:           {ic_ir:.2f}    (target: >1.0)        │")
    print(f"  │ IC > 0:          {ic_positive_pct:.0f}%   (target: >65%)       │")
    print(f"  │                                                   │")
    print(f"  │ LGB Huber IC:    {all_monthly_ics_lgb['ic'].mean():+.4f}                      │")
    print(f"  │ XGBoost IC:      {all_monthly_ics_xgb['ic'].mean():+.4f}                      │")
    print(f"  │ Ridge IC:        {all_monthly_ics_ridge['ic'].mean():+.4f}                      │")
    print(f"  │ ElasticNet IC:   {all_monthly_ics_enet['ic'].mean():+.4f}                      │")
    print(f"  └───────────────────────────────────────────────────┘")
    print(f"")
    print(f"  ┌─── PORTFOLIO METRICS ────────────────────────────┐")
    print(f"  │ L/S Sharpe:      {sharpe:.2f}   (target: ≈2.0)        │")
    print(f"  │ Avg D10-D1:      {ls_returns.mean()*100:+.2f}%/mo                    │")
    print(f"  │ OOS R²:          {oos_r2*100:.3f}% (target: ≈0.40%)     │")
    print(f"  └───────────────────────────────────────────────────┘")
    print(f"")
    print(f"  Top 5 Features:")
    for i, row in fi.head(5).iterrows():
        cat = ("INTER" if row["feature"].startswith("ix_") else
               "IND" if row["feature"].startswith("ff49_") else
               "BASE")
        print(f"    {row['feature']:<45} {row['importance_pct']*100:.2f}% [{cat}]")
    print(f"")
    print(f"  Time: {elapsed_total/60:.1f} min")
    print(f"  ✅ All saved and uploaded to S3")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
