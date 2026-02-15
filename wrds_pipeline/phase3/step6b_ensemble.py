"""
PHASE 3 - STEP 6B: INSTITUTIONAL-GRADE ML ENSEMBLE
Memory-efficient: preprocess per-iteration, not globally.
Ensemble: LGB(Huber) + XGB + Ridge + ElasticNet, IC-weighted.
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import json
import warnings
import gc
from scipy import stats
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
S3_BUCKET = "nuble-data-warehouse"


# ============================================================
# CROSS-SECTIONAL PREPROCESSING (memory-efficient, per-slice)
# ============================================================

def rank_normalize_slice(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Rank-normalize features to [-1, 1] per month for a SLICE of data.
    Process in chunks of 100 features to keep memory bounded.
    """
    grouped = df.groupby("date")
    chunk_size = 100
    for i in range(0, len(feature_cols), chunk_size):
        chunk = feature_cols[i:i + chunk_size]
        valid = [c for c in chunk if c in df.columns]
        if valid:
            df[valid] = grouped[valid].rank(pct=True, na_option="keep") * 2 - 1
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


def rank_normalize_target_slice(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Rank-normalize target per month -> N(0,1) via inverse-normal of rank."""
    ranked = f"{target_col}_ranked"
    df[ranked] = df.groupby("date")[target_col].transform(
        lambda x: stats.norm.ppf(
            x.rank(pct=True, na_option="keep").clip(0.001, 0.999)
        )
    )
    return df


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def train_lightgbm_huber(X_train, y_train, X_val, y_val, feature_names=None):
    """LightGBM with Huber loss. Patient early stopping (80 rounds)."""
    import lightgbm as lgb

    hpo_params = None
    try:
        from hyperparameter_optimization import load_best_params
        hpo_params = load_best_params()
    except Exception:
        pass

    if hpo_params:
        params = {
            "boosting_type": "gbdt", "verbose": -1, "n_jobs": -1,
            "seed": 42, "max_bin": 255, "min_gain_to_split": 0.01,
            "bagging_freq": 1,
        }
        params.update(hpo_params)
    else:
        params = {
            "objective": "huber", "alpha": 0.9,
            "boosting_type": "gbdt", "num_leaves": 63,
            "learning_rate": 0.02, "feature_fraction": 0.5,
            "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l1": 0.1, "lambda_l2": 5.0,
            "min_child_samples": 500, "max_depth": 7,
            "verbose": -1, "n_jobs": -1, "seed": 42,
            "max_bin": 255, "min_gain_to_split": 0.01,
        }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names,
                         free_raw_data=True)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)

    model = lgb.train(
        params, dtrain, num_boost_round=800,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    return model, model.best_iteration


def train_xgboost(X_train, y_train, X_val, y_val):
    """XGBoost with pseudo-Huber loss."""
    try:
        import xgboost as xgb
    except ImportError:
        return None, 0

    params = {
        "objective": "reg:pseudohubererror", "huber_slope": 1.0,
        "max_depth": 5, "learning_rate": 0.03,
        "subsample": 0.7, "colsample_bytree": 0.6,
        "reg_alpha": 0.5, "reg_lambda": 10.0,
        "min_child_weight": 500, "tree_method": "hist",
        "nthread": -1, "seed": 42, "verbosity": 0,
    }

    dtrain = xgb.DMatrix(np.nan_to_num(X_train, nan=0.0), label=y_train)
    dval = xgb.DMatrix(np.nan_to_num(X_val, nan=0.0), label=y_val)

    model = xgb.train(
        params, dtrain, num_boost_round=600,
        evals=[(dval, "val")], early_stopping_rounds=60, verbose_eval=False,
    )
    return model, model.best_iteration


def train_ridge(X_train, y_train, alpha=10.0):
    """Ridge regression - stable linear baseline."""
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(np.nan_to_num(X_train, nan=0.0), y_train)
    return model


def train_elasticnet(X_train, y_train, alpha=0.005, l1_ratio=0.5):
    """ElasticNet - sparse linear with feature selection."""
    from sklearn.linear_model import ElasticNet
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                       fit_intercept=False, max_iter=2000)
    model.fit(np.nan_to_num(X_train, nan=0.0), y_train)
    n_nonzero = np.sum(np.abs(model.coef_) > 1e-8)
    return model, n_nonzero


# ============================================================
# METRICS
# ============================================================

def compute_monthly_ics(df, pred_col="prediction", target_col="fwd_ret_1m"):
    """Compute Spearman IC for each month."""
    ics = []
    for dt, grp in df.groupby("date"):
        m = grp[pred_col].notna() & grp[target_col].notna()
        g = grp[m]
        if len(g) < 50:
            continue
        ic, _ = stats.spearmanr(g[pred_col], g[target_col])
        ics.append({"date": dt, "ic": ic, "n_stocks": len(g)})
    return pd.DataFrame(ics)


def compute_long_short_returns(df, pred_col="prediction",
                                target_col="fwd_ret_1m", n_q=10):
    """Monthly D10 - D1 long-short returns."""
    ls_vals, dates = [], []
    for dt, grp in df.groupby("date"):
        m = grp[pred_col].notna() & grp[target_col].notna()
        g = grp[m]
        if len(g) < 100:
            continue
        try:
            g = g.copy()
            g["q"] = pd.qcut(g[pred_col], n_q, labels=False, duplicates="drop")
            top = g[g["q"] == g["q"].max()][target_col].mean()
            bot = g[g["q"] == g["q"].min()][target_col].mean()
            ls_vals.append(top - bot)
            dates.append(dt)
        except Exception:
            continue
    return pd.Series(ls_vals, index=dates)


# ============================================================
# MAIN PIPELINE - MEMORY-EFFICIENT WALK-FORWARD
# ============================================================

def main():
    print("=" * 70)
    print("PHASE 3 - STEP 6B: INSTITUTIONAL-GRADE ML ENSEMBLE")
    print("  Memory-efficient: preprocess per-iteration, not globally")
    print("=" * 70)
    t_start = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load GKX panel
    print("\nLoading GKX panel...")
    gkx_path = os.path.join(DATA_DIR, "gkx_panel.parquet")
    if not os.path.exists(gkx_path):
        print("  ERROR: gkx_panel.parquet not found!")
        return

    panel = pd.read_parquet(gkx_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["year"] = panel["date"].dt.year
    print(f"  Panel: {len(panel):,} rows x {panel.shape[1]} cols")

    # Identify columns
    target_col = "fwd_ret_1m"
    if target_col not in panel.columns:
        for c in ["ret_forward"]:
            if c in panel.columns:
                target_col = c
                break

    id_cols = {"permno", "date", "cusip", "ticker", "siccd", "year",
               "ret", "fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m",
               "ret_forward", "dlret", "dlstcd"}
    feature_cols = [c for c in panel.columns if c not in id_cols
                    and panel[c].dtype in ["float64", "float32", "int64", "int32"]]

    panel = panel.dropna(subset=[target_col])
    print(f"  Features: {len(feature_cols)}, Target: {target_col}")
    print(f"  Rows with target: {len(panel):,}")

    # Convert to float32 to save ~50% memory
    print("  Converting to float32 to save memory...")
    for col in feature_cols:
        panel[col] = panel[col].astype(np.float32)
    panel[target_col] = panel[target_col].astype(np.float32)
    gc.collect()
    mem_gb = panel.memory_usage(deep=True).sum() / 1e9
    print(f"  Memory after float32: {mem_gb:.1f} GB")

    # ONE-TIME rank normalization (in-place, chunked for memory safety)
    print("  Rank-normalizing features per month (chunked, one-time)...")
    t_rank = time.time()
    rank_normalize_slice(panel, feature_cols)  # modifies in-place
    rank_normalize_target_slice(panel, target_col)
    ranked_target = f"{target_col}_ranked"
    gc.collect()
    print(f"  Rank normalization done in {time.time() - t_rank:.0f}s")

    # Drop all columns we don't need to free memory
    keep = ["permno", "date", "year", target_col, ranked_target] + feature_cols
    panel = panel[keep]
    gc.collect()
    mem_gb = panel.memory_usage(deep=True).sum() / 1e9
    print(f"  Memory after dropping unused cols: {mem_gb:.1f} GB")

    # WALK-FORWARD
    test_start = 2005
    test_end = int(panel["year"].max())

    print(f"\nWALK-FORWARD: {test_start}-{test_end}")
    print(f"  Ensemble: LGB(Huber) + XGB + Ridge + ElasticNet")
    print(f"  Preprocessing: already rank-normalized")
    print(f"  Target: inverse-normal of cross-sectional rank")

    all_predictions = []
    annual_metrics = []
    lgb_importance_total = np.zeros(len(feature_cols))

    for test_year in range(test_start, test_end + 1):
        t0 = time.time()
        val_year = test_year - 1

        # Slice the already-normalized panel (views, no copy)
        train_df = panel[panel["year"] < val_year]
        val_df = panel[panel["year"] == val_year]
        test_df = panel[panel["year"] == test_year]

        if len(train_df) < 50000 or len(test_df) < 1000:
            print(f"  {test_year}: skip (train={len(train_df)}, "
                  f"test={len(test_df)})")
            continue

        # Extract arrays — use contiguous copies for speed
        # For linear models, subsample training to save memory
        X_va = val_df[feature_cols].values
        y_va = val_df[ranked_target].values
        X_te = test_df[feature_cols].values
        y_te_raw = test_df[target_col].values  # for OOS metrics

        # === LightGBM (subsample if needed for memory) ===
        import lightgbm as lgb
        max_lgb_rows = 500000  # 500K rows is plenty for tree models
        if len(train_df) > max_lgb_rows:
            lgb_idx = np.random.choice(
                len(train_df), max_lgb_rows, replace=False)
            lgb_X_tr = train_df.iloc[lgb_idx][feature_cols].values
            lgb_y_tr = train_df.iloc[lgb_idx][ranked_target].values
        else:
            lgb_X_tr = train_df[feature_cols].values
            lgb_y_tr = train_df[ranked_target].values

        lgb_model, n_trees = train_lightgbm_huber(
            lgb_X_tr, lgb_y_tr, X_va, y_va,
            feature_names=feature_cols)
        lgb_pred = lgb_model.predict(X_te)
        lgb_importance_total += lgb_model.feature_importance(
            importance_type="gain")
        lgb_va_pred = lgb_model.predict(X_va)
        del lgb_X_tr, lgb_y_tr
        gc.collect()

        # === Subsample for XGB / Ridge / ElasticNet ===
        max_linear_rows = 300000
        if len(train_df) > max_linear_rows:
            sub_idx = np.random.choice(
                len(train_df), max_linear_rows, replace=False)
            X_tr_sub = train_df.iloc[sub_idx][feature_cols].values
            y_tr_sub = train_df.iloc[sub_idx][ranked_target].values
        else:
            X_tr_sub = train_df[feature_cols].values
            y_tr_sub = train_df[ranked_target].values

        # Replace NaN for linear/XGB models
        X_tr_clean = np.nan_to_num(X_tr_sub, nan=0.0)
        X_va_clean = np.nan_to_num(X_va, nan=0.0)
        X_te_clean = np.nan_to_num(X_te, nan=0.0)

        # === XGBoost ===
        xgb_model, n_xgb = train_xgboost(
            X_tr_clean, y_tr_sub, X_va_clean, y_va)
        has_xgb = xgb_model is not None
        if has_xgb:
            import xgboost as xgb_lib
            xgb_pred = xgb_model.predict(xgb_lib.DMatrix(X_te_clean))
            xgb_va_pred = xgb_model.predict(xgb_lib.DMatrix(X_va_clean))
        else:
            xgb_pred = lgb_pred.copy()
            xgb_va_pred = lgb_va_pred.copy()
            n_xgb = 0

        # === Ridge ===
        ridge_model = train_ridge(X_tr_clean, y_tr_sub, alpha=10.0)
        ridge_pred = ridge_model.predict(X_te_clean)
        ridge_va_pred = ridge_model.predict(X_va_clean)

        # === ElasticNet ===
        enet_model, n_enet_f = train_elasticnet(X_tr_clean, y_tr_sub)
        enet_pred = enet_model.predict(X_te_clean)
        enet_va_pred = enet_model.predict(X_va_clean)

        # Free subsample arrays
        del X_tr_sub, y_tr_sub, X_tr_clean
        gc.collect()

        # IC-weighted ensemble using validation performance
        ic_l = abs(stats.spearmanr(lgb_va_pred, y_va,
                                    nan_policy="omit")[0])
        ic_r = abs(stats.spearmanr(ridge_va_pred, y_va,
                                    nan_policy="omit")[0])
        ic_e = abs(stats.spearmanr(enet_va_pred, y_va,
                                    nan_policy="omit")[0])
        if has_xgb:
            ic_x = abs(stats.spearmanr(xgb_va_pred, y_va,
                                        nan_policy="omit")[0])
        else:
            ic_x = 0.0

        # Compute IC-weighted combination
        eps = 1e-6
        w_tot = ic_l + ic_r + ic_e + ic_x + eps
        w_l = (ic_l + eps / 4) / w_tot
        w_r = (ic_r + eps / 4) / w_tot
        w_e = (ic_e + eps / 4) / w_tot
        w_x = (ic_x + eps / 4) / w_tot if has_xgb else 0.0

        if not has_xgb:
            ws = w_l + w_r + w_e
            w_l /= ws
            w_r /= ws
            w_e /= ws

        ensemble_pred = (w_l * lgb_pred + w_r * ridge_pred +
                         w_e * enet_pred + w_x * xgb_pred)

        # Collect predictions
        res = test_df[["permno", "date", target_col]].copy()
        res["prediction"] = ensemble_pred
        res["pred_lgb"] = lgb_pred
        res["pred_ridge"] = ridge_pred
        res["pred_enet"] = enet_pred
        res["pred_xgb"] = xgb_pred
        all_predictions.append(res)

        # Compute test-year metrics
        m_ics = compute_monthly_ics(res, "prediction", target_col)
        m_ics_l = compute_monthly_ics(res, "pred_lgb", target_col)
        avg_ic = float(m_ics["ic"].mean()) if len(m_ics) > 0 else 0
        avg_ic_l = float(m_ics_l["ic"].mean()) if len(m_ics_l) > 0 else 0

        ic_rdg_t = float(stats.spearmanr(
            ridge_pred, y_te_raw, nan_policy="omit")[0])
        ic_xgb_t = float(stats.spearmanr(
            xgb_pred, y_te_raw,
            nan_policy="omit")[0]) if has_xgb else 0.0

        ls = compute_long_short_returns(res, "prediction", target_col)
        spread = float(ls.mean()) if len(ls) > 0 else 0

        elapsed = time.time() - t0

        annual_metrics.append({
            "year": test_year, "n_train": len(train_df),
            "n_val": len(val_df), "n_test": len(test_df),
            "ic_ensemble": avg_ic, "ic_lgb": avg_ic_l,
            "ic_ridge": ic_rdg_t, "ic_xgb": ic_xgb_t,
            "w_lgb": float(w_l), "w_ridge": float(w_r),
            "w_enet": float(w_e), "w_xgb": float(w_x),
            "avg_spread": spread, "n_trees": n_trees,
            "n_xgb_trees": n_xgb, "n_enet_features": int(n_enet_f),
            "prep_sec": 0,
            "time_sec": round(elapsed, 1),
        })

        marker = "[OK]" if avg_ic > 0.02 else "[--]" if avg_ic > 0 else "[XX]"
        print(f"  {test_year}: IC_ens={avg_ic:+.4f}{marker} "
              f"IC_lgb={avg_ic_l:+.4f} IC_rdg={ic_rdg_t:+.4f} "
              f"IC_xgb={ic_xgb_t:+.4f} "
              f"spread={spread:+.4f} "
              f"w=[{w_l:.2f},{w_r:.2f},{w_e:.2f},{w_x:.2f}] "
              f"trees={n_trees}+{n_xgb} enet_f={n_enet_f} "
              f"({elapsed:.0f}s)")

        # Cleanup to free memory before next iteration
        del X_va, X_te, X_va_clean, X_te_clean
        del lgb_model, ridge_model, enet_model
        del lgb_va_pred, ridge_va_pred, enet_va_pred
        if xgb_model is not None:
            del xgb_model, xgb_va_pred
        gc.collect()

    # ============================================================
    # AGGREGATE & REPORT
    # ============================================================
    if not all_predictions:
        print("\nNo predictions generated!")
        return

    preds = pd.concat(all_predictions, ignore_index=True)
    mdf = pd.DataFrame(annual_metrics)

    all_ics = compute_monthly_ics(preds, "prediction", target_col)
    overall_ic = float(all_ics["ic"].mean())
    ic_std = float(all_ics["ic"].std())
    ic_ir = overall_ic / ic_std if ic_std > 0 else 0
    ic_pos = float((all_ics["ic"] > 0).mean() * 100)

    ics_l = compute_monthly_ics(preds, "pred_lgb", target_col)
    ics_r = compute_monthly_ics(preds, "pred_ridge", target_col)
    ics_e = compute_monthly_ics(preds, "pred_enet", target_col)
    ics_x = compute_monthly_ics(preds, "pred_xgb", target_col)

    ls_ret = compute_long_short_returns(preds, "prediction", target_col)
    sharpe = (float(ls_ret.mean()) / float(ls_ret.std()) * np.sqrt(12)
              if ls_ret.std() > 0 else 0)

    mask = preds["prediction"].notna() & preds[target_col].notna()
    yt = preds.loc[mask, target_col].values
    yp = preds.loc[mask, "prediction"].values
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    oos_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Feature importance
    imp = lgb_importance_total / (lgb_importance_total.sum() + 1e-10)
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": imp,
        "raw": lgb_importance_total
    }).sort_values("importance", ascending=False)
    fi["pct"] = fi["importance"] / fi["importance"].sum()

    # Save everything
    print("\nSAVING RESULTS...")

    pred_path = os.path.join(DATA_DIR, "ensemble_predictions.parquet")
    preds.to_parquet(pred_path, index=False, engine="pyarrow")

    compat_path = os.path.join(DATA_DIR, "lgb_predictions.parquet")
    preds.to_parquet(compat_path, index=False, engine="pyarrow")
    print(f"  Predictions: {pred_path} ({len(preds):,} rows)")

    mdf.to_csv(os.path.join(RESULTS_DIR, "ensemble_metrics.csv"), index=False)
    all_ics.to_csv(os.path.join(RESULTS_DIR, "ensemble_monthly_ics.csv"),
                   index=False)
    fi.to_csv(os.path.join(RESULTS_DIR, "ensemble_feature_importance.csv"),
              index=False)

    summary = {
        "model": "IC-weighted Ensemble (LGB_Huber + XGBoost + Ridge + ElasticNet)",
        "test_period": f"{test_start}-{test_end}",
        "n_features": len(feature_cols),
        "total_predictions": len(preds),
        "total_months": len(all_ics),
        "overall_ic": round(overall_ic, 4),
        "ic_std": round(ic_std, 4),
        "ic_ir": round(ic_ir, 2),
        "ic_positive_pct": round(ic_pos, 1),
        "ic_by_model": {
            "ensemble": round(overall_ic, 4),
            "lgb_huber": round(float(ics_l["ic"].mean()), 4),
            "xgboost": round(float(ics_x["ic"].mean()), 4),
            "ridge": round(float(ics_r["ic"].mean()), 4),
            "elasticnet": round(float(ics_e["ic"].mean()), 4),
        },
        "ls_sharpe": round(sharpe, 2),
        "oos_r2_pct": round(oos_r2 * 100, 3),
        "avg_spread": round(float(ls_ret.mean()), 4),
        "preprocessing": "rank_normalize_[-1,1]_per_month + ranked_target_invnorm",
        "baseline_ic": 0.0136,
        "top_10_features": fi.head(10)["feature"].tolist(),
    }
    with open(os.path.join(RESULTS_DIR, "ensemble_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # S3 upload (best-effort, non-blocking)
    try:
        import subprocess
        for fp in [pred_path,
                   os.path.join(RESULTS_DIR, "ensemble_summary.json")]:
            subprocess.run(
                ["aws", "s3", "cp", fp,
                 f"s3://{S3_BUCKET}/models/{os.path.basename(fp)}"],
                capture_output=True, timeout=30)
    except Exception:
        pass

    elapsed_total = time.time() - t_start

    # Final report
    print(f"\n{'=' * 70}")
    print(f"INSTITUTIONAL ML ENSEMBLE - FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"  Period:       {test_start}-{test_end} "
          f"({test_end - test_start + 1} years)")
    print(f"  Months:       {len(all_ics)}")
    print(f"  Features:     {len(feature_cols)}")
    print(f"  Predictions:  {len(preds):,}")
    print(f"")
    print(f"  --- IC METRICS ---")
    print(f"  Ensemble IC:    {overall_ic:+.4f}  (target: ~0.03)")
    print(f"  IC Std:         {ic_std:.4f}")
    print(f"  IC IR:          {ic_ir:.2f}    (target: >1.0)")
    print(f"  IC > 0:         {ic_pos:.0f}%   (target: >65%)")
    print(f"")
    print(f"  LGB Huber:      {ics_l['ic'].mean():+.4f}")
    print(f"  XGBoost:        {ics_x['ic'].mean():+.4f}")
    print(f"  Ridge:          {ics_r['ic'].mean():+.4f}")
    print(f"  ElasticNet:     {ics_e['ic'].mean():+.4f}")
    print(f"")
    print(f"  --- PORTFOLIO METRICS ---")
    print(f"  L/S Sharpe:     {sharpe:.2f}   (target: ~2.0)")
    print(f"  Avg D10-D1:     {ls_ret.mean() * 100:+.2f}%/mo")
    print(f"  OOS R2:         {oos_r2 * 100:.3f}% (target: ~0.40%)")
    print(f"")
    print(f"  --- vs BASELINE (Step 6) ---")
    print(f"  Baseline IC:    +0.0136  IR=0.28")
    print(f"  Ensemble IC:    {overall_ic:+.4f}  IR={ic_ir:.2f}")
    pct_chg = ((overall_ic - 0.0136) / abs(0.0136) * 100
               if abs(0.0136) > 0 else 0)
    print(f"  Improvement:    {pct_chg:+.0f}%")
    print(f"")
    print(f"  Top 5 Features:")
    for _, row in fi.head(5).iterrows():
        print(f"    {row['feature']:<40} {row['pct'] * 100:.2f}%")
    print(f"")
    print(f"  Time: {elapsed_total / 60:.1f} min")
    print(f"  All saved to {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
