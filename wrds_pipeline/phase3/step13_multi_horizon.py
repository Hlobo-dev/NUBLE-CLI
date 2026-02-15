"""
PHASE 3 — STEP 13: Multi-Horizon Ensemble
==========================================
Separate LightGBM models for 1m, 3m, 6m, 12m horizons.
Weighted combination for final prediction.
Each horizon captures different alpha sources:
  1m  → short-term momentum/reversal
  3m  → medium-term momentum
  6m  → fundamental value
  12m → long-term mean reversion
"""

import pandas as pd
import numpy as np
import os
import time
import json
import gc
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/data/wrds"
RESULTS_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/wrds_pipeline/phase3/results"


def compute_forward_returns(panel):
    """Compute multi-horizon forward returns using proper cumulative compounding."""
    panel = panel.sort_values(["permno", "date"]).copy()

    # 1-month forward (already exists as ret_forward)
    if "ret_forward" in panel.columns:
        panel["ret_1m"] = panel["ret_forward"]
    elif "ret" in panel.columns:
        panel["ret_1m"] = panel.groupby("permno")["ret"].shift(-1)
    else:
        panel["ret_1m"] = np.nan

    # Multi-month forward returns: compound individual forward returns
    if "ret" in panel.columns:
        # First compute 1-period forward returns properly
        panel["_fwd1"] = panel.groupby("permno")["ret"].shift(-1)
        panel["_fwd2"] = panel.groupby("permno")["ret"].shift(-2)
        panel["_fwd3"] = panel.groupby("permno")["ret"].shift(-3)
        panel["_fwd4"] = panel.groupby("permno")["ret"].shift(-4)
        panel["_fwd5"] = panel.groupby("permno")["ret"].shift(-5)
        panel["_fwd6"] = panel.groupby("permno")["ret"].shift(-6)

        # 3-month forward = compound of next 3 months
        panel["ret_3m"] = (1 + panel["_fwd1"].fillna(0)) * \
                          (1 + panel["_fwd2"].fillna(0)) * \
                          (1 + panel["_fwd3"].fillna(0)) - 1

        # 6-month forward = compound of next 6 months
        panel["ret_6m"] = panel["ret_3m"].copy()
        for k in range(4, 7):
            col = f"_fwd{k}"
            if col in panel.columns:
                panel["ret_6m"] = (1 + panel["ret_6m"]) * (1 + panel[col].fillna(0)) - 1

        # 12-month: compound next 12 individual months
        fwd_cols = []
        for k in range(1, 13):
            col = f"_fwd{k}"
            if col not in panel.columns:
                panel[col] = panel.groupby("permno")["ret"].shift(-k)
            fwd_cols.append(col)

        panel["ret_12m"] = 1.0
        for col in fwd_cols:
            panel["ret_12m"] = panel["ret_12m"] * (1 + panel[col].fillna(0))
        panel["ret_12m"] = panel["ret_12m"] - 1

        # Null out where we don't have enough future data
        panel.loc[panel["_fwd3"].isna(), "ret_3m"] = np.nan
        panel.loc[panel["_fwd6"].isna(), "ret_6m"] = np.nan
        panel.loc[panel["_fwd12"].isna() if "_fwd12" in panel.columns else panel["_fwd6"].isna(), "ret_12m"] = np.nan

        # Drop temp columns
        drop_cols = [c for c in panel.columns if c.startswith("_fwd")]
        panel.drop(columns=drop_cols, inplace=True)
    else:
        panel["ret_3m"] = panel["ret_1m"]
        panel["ret_6m"] = panel["ret_1m"]
        panel["ret_12m"] = panel["ret_1m"]

    return panel


def train_horizon_model(panel, feature_cols, target_col, horizon_name, test_start=2005):
    """Train LightGBM for a specific horizon."""
    import lightgbm as lgb

    params = {
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

    panel = panel.dropna(subset=[target_col])
    panel["year"] = panel["date"].dt.year
    max_year = int(panel["year"].max())

    predictions_all = []
    ics = []

    for test_year in range(test_start, max_year + 1):
        train = panel[panel["year"] < test_year]
        test = panel[panel["year"] == test_year]

        if len(test) < 100 or len(train) < 1000:
            continue

        X_train = np.nan_to_num(train[feature_cols].values, nan=0.0)
        y_train = train[target_col].values
        X_test = np.nan_to_num(test[feature_cols].values, nan=0.0)

        # Validation = last year of training
        val_mask = train["year"] == test_year - 1
        if val_mask.sum() > 0:
            dtrain = lgb.Dataset(X_train[~val_mask.values], y_train[~val_mask.values])
            dval = lgb.Dataset(X_train[val_mask.values], y_train[val_mask.values], reference=dtrain)
            model = lgb.train(params, dtrain, num_boost_round=300,
                              valid_sets=[dval],
                              callbacks=[lgb.early_stopping(30, verbose=False)])
        else:
            dtrain = lgb.Dataset(X_train, y_train)
            model = lgb.train(params, dtrain, num_boost_round=200)

        preds = model.predict(X_test)

        # Monthly IC
        test_df = test[["permno", "date", target_col]].copy()
        test_df["prediction"] = preds
        for dt, grp in test_df.groupby("date"):
            ic = grp["prediction"].corr(grp[target_col], method="spearman")
            if not np.isnan(ic):
                ics.append(ic)

        predictions_all.append(test_df)
        gc.collect()

    avg_ic = np.mean(ics) if ics else 0

    if predictions_all:
        all_preds = pd.concat(predictions_all, ignore_index=True)
        all_preds = all_preds.rename(columns={"prediction": f"pred_{horizon_name}"})
    else:
        all_preds = pd.DataFrame()

    return all_preds, avg_ic


def main():
    print("=" * 70)
    print("PHASE 3 — STEP 13: MULTI-HORIZON ENSEMBLE")
    print("=" * 70)
    start = time.time()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load panel
    gkx_path = os.path.join(DATA_DIR, "gkx_panel.parquet")
    if not os.path.exists(gkx_path):
        # Fallback to training panel
        gkx_path = os.path.join(DATA_DIR, "training_panel.parquet")

    if not os.path.exists(gkx_path):
        print("  ❌ No panel data found!")
        return

    panel = pd.read_parquet(gkx_path)
    panel["date"] = pd.to_datetime(panel["date"])
    print(f"  Panel: {len(panel):,} rows × {panel.shape[1]} cols")

    # Compute multi-horizon returns
    print("\n📊 Computing multi-horizon forward returns...")
    panel = compute_forward_returns(panel)

    # Feature columns
    id_cols = ["permno", "date", "cusip", "ticker", "siccd", "year",
               "ret", "ret_forward", "ret_1m", "ret_3m", "ret_6m", "ret_12m"]
    feature_cols = [c for c in panel.columns if c not in id_cols
                    and panel[c].dtype in ["float64", "float32", "int64"]]
    print(f"  Features: {len(feature_cols)}")

    # Train models for each horizon
    horizons = [
        ("1m", "ret_1m", 0.40),    # weight
        ("3m", "ret_3m", 0.30),
        ("6m", "ret_6m", 0.20),
        ("12m", "ret_12m", 0.10),
    ]

    horizon_results = {}
    all_preds = None

    for horizon_name, target_col, weight in horizons:
        print(f"\n🔄 Training {horizon_name} horizon model (target={target_col}, weight={weight})...")

        if target_col not in panel.columns:
            print(f"  ⚠️ {target_col} not available, skipping")
            continue

        preds, avg_ic = train_horizon_model(panel, feature_cols, target_col, horizon_name)

        if len(preds) > 0:
            horizon_results[horizon_name] = {
                "avg_ic": round(avg_ic, 4),
                "weight": weight,
                "n_predictions": len(preds),
            }
            print(f"  {horizon_name}: IC={avg_ic:+.4f}, {len(preds):,} predictions")

            if all_preds is None:
                all_preds = preds[["permno", "date", f"pred_{horizon_name}"]].copy()
            else:
                all_preds = all_preds.merge(
                    preds[["permno", "date", f"pred_{horizon_name}"]],
                    on=["permno", "date"], how="outer"
                )
        else:
            print(f"  ⚠️ No predictions for {horizon_name}")

    if all_preds is None or len(all_preds) == 0:
        print("\n❌ No ensemble predictions!")
        return

    # Weighted ensemble
    print("\n📊 BUILDING ENSEMBLE...")
    pred_cols = [c for c in all_preds.columns if c.startswith("pred_")]
    weights = {}
    for h_name, _, w in horizons:
        col = f"pred_{h_name}"
        if col in pred_cols:
            weights[col] = w

    # Normalize weights for available horizons
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    all_preds["ensemble_prediction"] = 0
    for col, w in weights.items():
        all_preds["ensemble_prediction"] += all_preds[col].fillna(0) * w

    # Compute ensemble IC
    if "ret_1m" in panel.columns:
        merged = all_preds.merge(
            panel[["permno", "date", "ret_1m"]].rename(columns={"ret_1m": "actual"}),
            on=["permno", "date"], how="inner"
        )
        ensemble_ics = []
        for dt, grp in merged.groupby("date"):
            ic = grp["ensemble_prediction"].corr(grp["actual"], method="spearman")
            if not np.isnan(ic):
                ensemble_ics.append(ic)
        ensemble_ic = np.mean(ensemble_ics) if ensemble_ics else 0
    else:
        ensemble_ic = 0

    # Save
    output_path = os.path.join(DATA_DIR, "ensemble_predictions.parquet")
    all_preds.to_parquet(output_path, index=False, engine="pyarrow")

    summary = {
        "model": "Multi-Horizon LightGBM Ensemble",
        "horizons": horizon_results,
        "weights": {k: round(v, 2) for k, v in weights.items()},
        "ensemble_ic": round(float(ensemble_ic), 4),
        "n_predictions": len(all_preds),
        "n_features": len(feature_cols),
    }
    with open(os.path.join(RESULTS_DIR, "ensemble_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"MULTI-HORIZON ENSEMBLE COMPLETE")
    print(f"{'=' * 70}")
    for h_name, metrics in horizon_results.items():
        print(f"  {h_name}: IC={metrics['avg_ic']:+.4f}, weight={metrics['weight']:.0%}")
    print(f"  Ensemble IC:  {ensemble_ic:+.4f}")
    print(f"  Predictions:  {len(all_preds):,}")
    print(f"  Time:         {elapsed/60:.1f} min")
    print(f"  ✅ Saved to {output_path}")


if __name__ == "__main__":
    main()
