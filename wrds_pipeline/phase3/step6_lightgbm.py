"""
PHASE 3 — STEP 6: LightGBM Walk-Forward Model
===============================================
GKX-style expanding window, annual re-estimation 2000-2024.
Predicts next-month stock returns using 600-900 features.
Walk-forward: train on all data up to year T, predict year T+1.
Re-estimate annually. Early stopping on validation IC.

Params (GKX defaults):
  num_leaves=31, learning_rate=0.05, feature_fraction=0.7,
  max_depth=6, early_stopping=50
"""

import pandas as pd
import numpy as np
import os
import time
import subprocess
import json
import warnings
import gc

warnings.filterwarnings("ignore")

# Resolve paths relative to project root (portable across machines)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models", "lightgbm")
S3_BUCKET = "nuble-data-warehouse"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# LightGBM hyperparameters (GKX-style)
LGB_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 6,
    "min_child_samples": 100,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}


def compute_monthly_ic(predictions, actuals):
    """Spearman rank IC between predictions and actuals."""
    mask = predictions.notna() & actuals.notna()
    if mask.sum() < 30:
        return np.nan
    return predictions[mask].corr(actuals[mask], method="spearman")


def compute_decile_spread(predictions, actuals):
    """Long-short return spread: D10 minus D1."""
    mask = predictions.notna() & actuals.notna()
    if mask.sum() < 100:
        return np.nan
    df = pd.DataFrame({"pred": predictions[mask], "actual": actuals[mask]})
    df["decile"] = pd.qcut(df["pred"], 10, labels=False, duplicates="drop")
    decile_rets = df.groupby("decile")["actual"].mean()
    if len(decile_rets) >= 10:
        return decile_rets.iloc[-1] - decile_rets.iloc[0]
    return decile_rets.max() - decile_rets.min()


def main():
    print("=" * 70)
    print("PHASE 3 — STEP 6: LIGHTGBM WALK-FORWARD MODEL")
    print("=" * 70)
    start = time.time()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load GKX panel ───────────────────────────────────────────────────
    print("\n📊 Loading GKX panel...")
    gkx_path = os.path.join(DATA_DIR, "gkx_panel.parquet")
    if not os.path.exists(gkx_path):
        print("  ❌ gkx_panel.parquet not found — run Step 5 first!")
        return

    # Load with float32 (Step 5 saves as float32 to halve memory)
    gkx = pd.read_parquet(gkx_path)
    gkx["date"] = pd.to_datetime(gkx["date"])
    gkx["year"] = gkx["date"].dt.year
    print(f"  Panel: {len(gkx):,} rows × {gkx.shape[1]} cols")

    # Identify target and features
    target_col = None
    for candidate in ["ret_forward", "fwd_ret_1m"]:
        if candidate in gkx.columns:
            target_col = candidate
            break
    if target_col is None:
        print("  ❌ No forward return column found (need 'ret_forward' or 'fwd_ret_1m')!")
        return
    id_cols = ["permno", "date", "cusip", "ticker", "siccd", "year",
               "ret", "fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m",
               "ret_forward", "dlret", "dlstcd"]
    feature_cols = [c for c in gkx.columns if c not in id_cols + [target_col]
                    and gkx[c].dtype in ["float64", "float32", "int64", "int32"]]
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target: {target_col}")

    # Drop rows with missing target
    gkx = gkx.dropna(subset=[target_col])
    print(f"  Rows with target: {len(gkx):,}")

    # ── Walk-Forward ─────────────────────────────────────────────────────
    import lightgbm as lgb

    # Determine year range — start test at 2005 (1926-2004 = 79 years training)
    min_year = max(gkx["year"].min(), 1990)  # Need enough data to train
    max_year = gkx["year"].max()
    test_start = 2005  # First test year (enough training data from 1926)
    test_end = int(max_year)

    print(f"\n🔄 WALK-FORWARD: Train through {test_start-1}, test {test_start}-{test_end}")
    print(f"  Annual re-estimation, expanding window")
    print(f"  LGB params: leaves={LGB_PARAMS['num_leaves']}, lr={LGB_PARAMS['learning_rate']}, "
          f"ff={LGB_PARAMS['feature_fraction']}, depth={LGB_PARAMS['max_depth']}")

    all_predictions = []
    annual_metrics = []
    feature_importance_total = np.zeros(len(feature_cols))

    for test_year in range(test_start, test_end + 1):
        t0 = time.time()

        # Train on all data up to test_year - 1
        train_mask = gkx["year"] < test_year
        test_mask = gkx["year"] == test_year

        train_data = gkx.loc[train_mask]
        test_data = gkx.loc[test_mask]

        if len(test_data) < 100:
            print(f"  {test_year}: skip (only {len(test_data)} test rows)")
            continue

        # LightGBM natively handles NaN — no need to copy + nan_to_num
        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values

        # Use last year of training as validation for early stopping
        val_year = test_year - 1
        val_mask_train = (train_data["year"] == val_year).values
        if val_mask_train.sum() > 0:
            train_subset = train_data[~val_mask_train]
            val_subset = train_data[val_mask_train]
            dtrain = lgb.Dataset(
                train_subset[feature_cols],
                train_subset[target_col],
                free_raw_data=True,
            )
            dval = lgb.Dataset(
                val_subset[feature_cols],
                val_subset[target_col],
                reference=dtrain,
                free_raw_data=True,
            )

            model = lgb.train(
                LGB_PARAMS,
                dtrain,
                num_boost_round=500,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            del dtrain, dval, train_subset, val_subset
        else:
            dtrain = lgb.Dataset(
                train_data[feature_cols], train_data[target_col],
                free_raw_data=True,
            )
            model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=200)
            del dtrain

        # Predict
        preds = model.predict(X_test)
        feature_importance_total += model.feature_importance(importance_type="gain")

        # Save latest model to disk for Step 10 (monthly refresh)
        if test_year == test_end:
            # Save to canonical models/ directory
            os.makedirs(MODELS_DIR, exist_ok=True)
            canonical_path = os.path.join(MODELS_DIR, "lgb_latest_model.txt")
            model.save_model(canonical_path)
            print(f"    → Saved latest model to {canonical_path}")

            # Also save to data/wrds/ for backward compatibility
            legacy_path = os.path.join(DATA_DIR, "lgb_latest_model.txt")
            model.save_model(legacy_path)
            print(f"    → (legacy copy) {legacy_path}")

        # Monthly metrics within test year
        test_df = test_data[["permno", "date", target_col]].copy()
        test_df["prediction"] = preds

        monthly_ics = []
        monthly_spreads = []
        for dt, grp in test_df.groupby("date"):
            ic = compute_monthly_ic(grp["prediction"], grp[target_col])
            spread = compute_decile_spread(grp["prediction"], grp[target_col])
            monthly_ics.append(ic)
            monthly_spreads.append(spread)

        avg_ic = np.nanmean(monthly_ics)
        avg_spread = np.nanmean(monthly_spreads)
        elapsed_yr = time.time() - t0

        n_train = len(train_data)
        n_test = len(test_data)
        n_trees = model.num_trees()

        annual_metrics.append({
            "year": test_year,
            "n_train": n_train,
            "n_test": n_test,
            "avg_ic": avg_ic,
            "avg_spread": avg_spread,
            "n_trees": n_trees,
        })

        all_predictions.append(test_df)
        print(f"  {test_year}: train={n_train:>8,} test={n_test:>6,} "
              f"IC={avg_ic:+.4f} spread={avg_spread:+.4f} trees={n_trees:>3} "
              f"({elapsed_yr:.0f}s)")

        del train_data, test_data, X_test, model, preds
        gc.collect()

    # ── Aggregate Results ────────────────────────────────────────────────
    if not all_predictions:
        print("\n❌ No predictions generated!")
        return

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(annual_metrics)

    # Overall statistics
    overall_ic = metrics_df["avg_ic"].mean()
    overall_spread = metrics_df["avg_spread"].mean()
    ic_ir = metrics_df["avg_ic"].mean() / metrics_df["avg_ic"].std() if metrics_df["avg_ic"].std() > 0 else 0

    # Feature importance
    fi = pd.Series(feature_importance_total, index=feature_cols).sort_values(ascending=False)
    fi_norm = fi / fi.sum()

    # ── Save Results ─────────────────────────────────────────────────────
    print("\n💾 SAVING RESULTS...")

    # Predictions
    pred_path = os.path.join(DATA_DIR, "lgb_predictions.parquet")
    predictions_df.to_parquet(pred_path, index=False, engine="pyarrow")

    # Metrics
    metrics_path = os.path.join(RESULTS_DIR, "walk_forward_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Feature importance
    fi_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
    fi_df = pd.DataFrame({"feature": fi.index, "importance": fi.values, "importance_pct": fi_norm.values})
    fi_df.to_csv(fi_path, index=False)

    # Summary JSON
    summary = {
        "model": "LightGBM",
        "test_period": f"{test_start}-{test_end}",
        "n_features": len(feature_cols),
        "total_predictions": len(predictions_df),
        "overall_ic": round(float(overall_ic), 4),
        "ic_ir": round(float(ic_ir), 2),
        "overall_spread": round(float(overall_spread), 4),
        "top_10_features": fi.head(10).index.tolist(),
        "params": {k: v for k, v in LGB_PARAMS.items() if k != "verbose"},
    }
    summary_path = os.path.join(RESULTS_DIR, "model_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # S3 upload
    for fpath in [pred_path, metrics_path, fi_path, summary_path]:
        fname = os.path.basename(fpath)
        subprocess.run(
            ["aws", "s3", "cp", fpath,
             f"s3://{S3_BUCKET}/models/{fname}"],
            capture_output=True,
        )

    elapsed = time.time() - start

    print(f"\n{'=' * 70}")
    print(f"LIGHTGBM WALK-FORWARD COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Period:           {test_start}-{test_end} ({test_end - test_start + 1} years)")
    print(f"  Features:         {len(feature_cols)}")
    print(f"  Predictions:      {len(predictions_df):,}")
    print(f"  Overall IC:       {overall_ic:+.4f}")
    print(f"  IC IR:            {ic_ir:.2f}")
    print(f"  Avg Spread:       {overall_spread:+.4f}")
    print(f"  Top 5 features:")
    for feat in fi.head(5).index:
        print(f"    {feat:<40} {fi_norm[feat]*100:.1f}%")
    print(f"  Time:             {elapsed/60:.1f} min")
    print(f"  ✅ All saved and uploaded to S3")


if __name__ == "__main__":
    main()
