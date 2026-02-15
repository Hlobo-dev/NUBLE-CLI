"""
Save Tier-Specific LightGBM Models
====================================
Extracts and trains 4 tier-specific LightGBM models from the GKX panel
using the curated feature lists from step6d/step6h.

The step6d_curated_features.py trains per-tier models inline but only saves
predictions as parquet — NOT the model files. This script re-trains the final
model for each tier on ALL available data (no walk-forward, just the last
training window) and saves them as:
    models/lightgbm/lgb_mega.txt
    models/lightgbm/lgb_large.txt
    models/lightgbm/lgb_mid.txt
    models/lightgbm/lgb_small.txt

These model files are loaded by WRDSPredictor v2 for real-time scoring.

Run ONCE:
    cd ~/Desktop/NUBLE-CLI
    .venv/bin/python wrds_pipeline/save_tier_models.py
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models", "lightgbm")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "wrds_pipeline", "phase3", "results")

# ── Tier definitions (matching step6d/step6f/step6h) ──────────────────
TIER_DEFS = {
    "mega":  {"label": "Mega-Cap (>$10B)",    "min_lmc": 9.21,  "max_lmc": np.inf},
    "large": {"label": "Large-Cap ($2-10B)",  "min_lmc": 7.60,  "max_lmc": 9.21},
    "mid":   {"label": "Mid-Cap ($500M-2B)",  "min_lmc": 6.21,  "max_lmc": 7.60},
    "small": {"label": "Small-Cap (<$500M)",  "min_lmc": 4.61,  "max_lmc": 6.21},
}

# ── LightGBM params per tier (from step6d/step6h — production configs) ─
TIER_LGB_PARAMS = {
    "mega": {
        "objective": "huber", "alpha": 0.9,
        "boosting_type": "gbdt",
        "num_leaves": 31, "min_child_samples": 200,
        "max_depth": 5, "learning_rate": 0.01,
        "feature_fraction": 0.5, "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.5, "lambda_l2": 10.0,
        "verbose": -1, "n_jobs": -1, "seed": 42,
        "max_bin": 255, "min_gain_to_split": 0.05,
        "num_boost_round": 600,
    },
    "large": {
        "objective": "huber", "alpha": 0.9,
        "boosting_type": "gbdt",
        "num_leaves": 31, "min_child_samples": 200,
        "max_depth": 5, "learning_rate": 0.01,
        "feature_fraction": 0.5, "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.5, "lambda_l2": 10.0,
        "verbose": -1, "n_jobs": -1, "seed": 42,
        "max_bin": 255, "min_gain_to_split": 0.05,
        "num_boost_round": 600,
    },
    "mid": {
        "objective": "huber", "alpha": 0.9,
        "boosting_type": "gbdt",
        "num_leaves": 63, "min_child_samples": 300,
        "max_depth": 7, "learning_rate": 0.015,
        "feature_fraction": 0.5, "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1, "lambda_l2": 5.0,
        "verbose": -1, "n_jobs": -1, "seed": 42,
        "max_bin": 255, "min_gain_to_split": 0.01,
        "num_boost_round": 900,
    },
    "small": {
        "objective": "huber", "alpha": 0.9,
        "boosting_type": "gbdt",
        "num_leaves": 63, "min_child_samples": 500,
        "max_depth": 7, "learning_rate": 0.015,
        "feature_fraction": 0.5, "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "lambda_l1": 0.1, "lambda_l2": 5.0,
        "verbose": -1, "n_jobs": -1, "seed": 42,
        "max_bin": 255, "min_gain_to_split": 0.01,
        "num_boost_round": 900,
    },
}


def load_feature_lists() -> dict:
    """Load per-tier feature lists from step6d curated summary or step6h IC-first summary."""
    # Prefer ic_first_fix_summary (step6h — more refined for mega/large)
    ic_first_path = os.path.join(RESULTS_DIR, "ic_first_fix_summary.json")
    curated_path = os.path.join(RESULTS_DIR, "curated_multi_universe_summary.json")

    feature_lists = {}

    # Step6h (IC-first) has better mega/large features
    if os.path.exists(ic_first_path):
        with open(ic_first_path) as f:
            ic_summary = json.load(f)
        for tier_name, feats in ic_summary.get("feature_lists", {}).items():
            feature_lists[tier_name] = feats
            print(f"  {tier_name} features (IC-first): {len(feats)}")

    # Step6d (curated) has mid/small features
    if os.path.exists(curated_path):
        with open(curated_path) as f:
            curated_summary = json.load(f)
        for tier_name, feats in curated_summary.get("feature_lists", {}).items():
            if tier_name not in feature_lists:
                feature_lists[tier_name] = feats
                print(f"  {tier_name} features (curated): {len(feats)}")

    if not feature_lists:
        raise RuntimeError("No feature lists found! Run step6d or step6h first.")

    return feature_lists


def compute_spearman_ic(preds, actuals):
    """Compute Spearman rank IC."""
    from scipy import stats
    mask = np.isfinite(preds) & np.isfinite(actuals)
    if mask.sum() < 30:
        return np.nan
    ic, _ = stats.spearmanr(preds[mask], actuals[mask])
    return ic


def main():
    print("=" * 70)
    print("SAVE TIER-SPECIFIC LIGHTGBM MODELS")
    print("=" * 70)
    t_start = time.time()

    import lightgbm as lgb

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load feature lists ──
    print("\n📋 Loading curated feature lists...")
    feature_lists = load_feature_lists()

    # ── Load GKX panel ──
    print("\n📊 Loading GKX panel...")
    gkx_path = os.path.join(DATA_DIR, "gkx_panel.parquet")
    if not os.path.exists(gkx_path):
        print("  ❌ gkx_panel.parquet not found!")
        return

    panel = pd.read_parquet(gkx_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["year"] = panel["date"].dt.year
    print(f"  Panel: {len(panel):,} rows × {panel.shape[1]} cols")

    # Identify target
    target_col = None
    for candidate in ["fwd_ret_1m", "ret_forward"]:
        if candidate in panel.columns:
            target_col = candidate
            break
    if target_col is None:
        print("  ❌ No forward return column found!")
        return
    print(f"  Target: {target_col}")

    panel = panel.dropna(subset=[target_col])
    print(f"  Rows with target: {len(panel):,}")

    # Ensure log_market_cap exists
    if "log_market_cap" not in panel.columns:
        if "market_cap" in panel.columns:
            panel["log_market_cap"] = np.log(panel["market_cap"].clip(lower=1))
        elif "mvel1" in panel.columns:
            panel["log_market_cap"] = np.log(panel["mvel1"].clip(lower=1) * 1e6)
        else:
            print("  ❌ No market cap column found!")
            return

    # ── Train and save per-tier models ──
    max_year = int(panel["year"].max())
    train_window = 10  # Use last 10 years of data

    saved_models = {}

    for tier_name, tier_def in TIER_DEFS.items():
        print(f"\n{'=' * 60}")
        print(f"TIER: {tier_def['label']}")
        print(f"{'=' * 60}")

        if tier_name not in feature_lists:
            print(f"  ⚠️  No feature list for {tier_name} — skipping")
            continue

        tier_features = feature_lists[tier_name]

        # Filter features to those actually in the panel
        available_features = [f for f in tier_features if f in panel.columns]
        missing_features = [f for f in tier_features if f not in panel.columns]
        if missing_features:
            print(f"  ⚠️  {len(missing_features)} features not in panel: "
                  f"{missing_features[:5]}...")
        print(f"  Using {len(available_features)} features (of {len(tier_features)} specified)")

        if len(available_features) < 10:
            print(f"  ❌ Too few features — skipping {tier_name}")
            continue

        # Filter to tier
        tier_mask = panel["log_market_cap"] >= tier_def["min_lmc"]
        if tier_def["max_lmc"] != np.inf:
            tier_mask &= panel["log_market_cap"] < tier_def["max_lmc"]
        tier_panel = panel[tier_mask].copy()

        print(f"  Tier rows: {len(tier_panel):,}")
        print(f"  Year range: {tier_panel['year'].min()}-{tier_panel['year'].max()}")

        if len(tier_panel) < 5000:
            print(f"  ❌ Too few rows — skipping {tier_name}")
            continue

        # Training data: use expanding window up to max_year
        train_start = max(max_year - train_window, tier_panel["year"].min())
        train_mask = (tier_panel["year"] >= train_start) & (tier_panel["year"] <= max_year)
        train_data = tier_panel[train_mask]

        # Validation: last year
        val_mask = train_data["year"] == max_year
        train_final = train_data[~val_mask]
        val_final = train_data[val_mask]

        print(f"  Train: {len(train_final):,} rows ({train_start}-{max_year - 1})")
        print(f"  Val:   {len(val_final):,} rows ({max_year})")

        # Rank-normalize target within each date cross-section
        from scipy import stats as sp_stats

        train_final = train_final.copy()
        val_final = val_final.copy()
        train_final["target_ranked"] = train_final.groupby("date")[target_col].transform(
            lambda x: sp_stats.norm.ppf(
                x.rank(pct=True, na_option="keep").clip(0.001, 0.999)
            )
        ).astype(np.float32)
        val_final["target_ranked"] = val_final.groupby("date")[target_col].transform(
            lambda x: sp_stats.norm.ppf(
                x.rank(pct=True, na_option="keep").clip(0.001, 0.999)
            )
        ).astype(np.float32)

        X_train = train_final[available_features].values
        y_train = train_final["target_ranked"].values
        X_val = val_final[available_features].values
        y_val = val_final["target_ranked"].values

        # Get tier-specific LGB params
        tp = TIER_LGB_PARAMS[tier_name].copy()
        num_boost_round = tp.pop("num_boost_round")

        # Build datasets
        dtrain = lgb.Dataset(X_train, label=y_train,
                             feature_name=available_features, free_raw_data=True)
        dval = lgb.Dataset(X_val, label=y_val,
                           reference=dtrain, free_raw_data=True)

        # Train
        print(f"  Training LightGBM...")
        t0 = time.time()
        model = lgb.train(
            tp, dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        train_time = time.time() - t0
        print(f"  Training time: {train_time:.1f}s, best iteration: {model.best_iteration}")

        # Evaluate
        val_preds = model.predict(X_val)
        val_ic = compute_spearman_ic(val_preds, val_final[target_col].values)
        print(f"  Validation IC: {val_ic:+.4f}")

        # Feature importance
        imp = model.feature_importance(importance_type="gain")
        feat_imp = sorted(zip(available_features, imp), key=lambda x: -x[1])
        print(f"  Top 5 features:")
        for fname, fval in feat_imp[:5]:
            print(f"    {fname}: {fval / imp.sum() * 100:.1f}%")

        # Save model
        model_path = os.path.join(MODELS_DIR, f"lgb_{tier_name}.txt")
        model.save_model(model_path)
        print(f"  ✅ Saved: {model_path}")
        print(f"     Model size: {os.path.getsize(model_path) / 1024:.0f} KB")
        print(f"     Trees: {model.num_trees()}")

        saved_models[tier_name] = {
            "path": model_path,
            "n_features": len(available_features),
            "features": available_features,
            "n_trees": model.num_trees(),
            "best_iteration": model.best_iteration,
            "val_ic": round(val_ic, 4) if not np.isnan(val_ic) else 0.0,
            "train_rows": len(train_final),
            "val_rows": len(val_final),
            "train_years": f"{train_start}-{max_year - 1}",
            "val_year": max_year,
            "top_features": [{"name": n, "importance": round(v / imp.sum() * 100, 2)}
                             for n, v in feat_imp[:10]],
        }

        del model, dtrain, dval, X_train, y_train, X_val, y_val
        import gc
        gc.collect()

    # ── Summary ──
    total_time = (time.time() - t_start) / 60
    print(f"\n{'=' * 70}")
    print("TIER MODEL SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Tier':<10} {'Features':>10} {'Trees':>8} {'Val IC':>10} {'Size':>10}")
    print("-" * 50)
    for tier_name in ["mega", "large", "mid", "small"]:
        if tier_name in saved_models:
            m = saved_models[tier_name]
            size_kb = os.path.getsize(m["path"]) / 1024
            print(f"{tier_name:<10} {m['n_features']:>10} {m['n_trees']:>8} "
                  f"{m['val_ic']:>+10.4f} {size_kb:>8.0f}KB")
    print("-" * 50)
    print(f"Total time: {total_time:.1f} min")

    # Save model registry
    registry = {
        "method": "tier_specific_lgb",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_min": round(total_time, 1),
        "models": {},
    }
    for tier_name, info in saved_models.items():
        registry["models"][tier_name] = {
            k: v for k, v in info.items() if k != "path"
        }

    registry_path = os.path.join(MODELS_DIR, "tier_model_registry.json")
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2, default=str)
    print(f"\n  Registry: {registry_path}")
    print(f"\n✅ All tier models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
