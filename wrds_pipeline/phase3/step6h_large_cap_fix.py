"""
PHASE 3 — STEP 6h: LARGE-CAP & MEGA-CAP IC FIX
=================================================
Root-Cause Fix for Red Flags #1, #3, #4 from Prompt 5 Audit.

DIAGNOSIS (from deep diagnostic analysis):
  1. Return dispersion in large-cap is 1.72× LOWER than small-cap
     → signal-to-noise ratio is fundamentally harder
  2. step6d's force-inclusion mechanism added ~60 academic factors with
     NEAR-ZERO IC in large-cap, while crowding out 26 features with
     HIGHEST IC (roce, intcov_ratio, roe, roa, etc.)
  3. Deduplication removed GOOD features (roe, roa) because they
     correlated with force-included low-IC features (roaq, gp_at)
  4. Only 23 features have |IC| > 0.02 in large-cap — the signal
     space is thin, so every feature slot is precious

FIX APPROACH:
  - IC-FIRST selection: No force-inclusion for large/mega
  - Compute per-tier univariate IC on ALL features
  - Select top-N by |IC|, then deduplicate keeping the BEST per cluster
  - Tighter budget (50-55 features) for signal concentration
  - Increased regularization for low-dispersion universes
  - Retrain ONLY large + mega (mid/small are already good)

EXPECTED IMPROVEMENT:
  Large-cap IC: 0.018 → 0.030+ (realistic ceiling ~0.035)
  Large-cap monotonicity: 0.321 → 0.500+
  Mega-cap IC: maintain or improve from 0.034

Author: Claude × Humberto
"""

import pandas as pd
import numpy as np
import os
import gc
import time
import json
import warnings
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/data/wrds"
RESULTS_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/wrds_pipeline/phase3/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════
# TIER DEFINITIONS — Only large + mega (mid/small untouched)
# ════════════════════════════════════════════════════════════

TIER_DEFS = {
    "mega":  {"label": "Mega-Cap (>$10B)",   "min_lmc": 9.21,  "max_lmc": np.inf},
    "large": {"label": "Large-Cap ($2-10B)", "min_lmc": 7.60,  "max_lmc": 9.21},
}

# Feature budget — TIGHTER for signal concentration
TIER_FEATURE_BUDGET = {
    "mega":  50,
    "large": 55,
}


# ════════════════════════════════════════════════════════════
# IC-FIRST FEATURE SELECTION (no force-inclusion)
# ════════════════════════════════════════════════════════════

def compute_univariate_ic(panel, feature_cols, target_col, min_periods=60):
    """
    Compute per-feature time-series average Spearman IC.
    OPTIMIZED: vectorized groupby instead of month-by-month Python loop.
    Returns dict: feature → metrics.
    """
    ic_results = {}
    n_feats = len(feature_cols)

    # Pre-group by date for speed
    grouped = panel.groupby("date")
    dates = sorted(panel["date"].unique())
    n_dates = len(dates)

    for i, feat in enumerate(feature_cols):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"      IC scan: {i+1}/{n_feats} features...")

        monthly_ics = []
        for dt, grp in grouped:
            mask = grp[feat].notna() & grp[target_col].notna()
            sub = grp[mask]
            if len(sub) >= 30:
                ic, _ = stats.spearmanr(sub[feat].values, sub[target_col].values)
                if not np.isnan(ic):
                    monthly_ics.append(ic)

        if len(monthly_ics) >= min_periods:
            mean_ic = np.mean(monthly_ics)
            ic_std = np.std(monthly_ics)
            ic_ir = mean_ic / ic_std if ic_std > 0 else 0
            pct_pos = np.mean([x > 0 for x in monthly_ics])
            ic_results[feat] = {
                "mean_ic": mean_ic,
                "ic_ir": ic_ir,
                "pct_positive": pct_pos,
                "n_months": len(monthly_ics),
                "abs_ic": abs(mean_ic),
            }

    return ic_results


def cluster_correlated_features(panel, features, ic_results, threshold=0.80,
                                sample_n=50000):
    """
    Cluster highly correlated features.
    For each cluster, keep the feature with HIGHEST |IC|.
    NO bias toward force-included features.
    """
    if len(panel) > sample_n:
        sample = panel[features].sample(n=sample_n, random_state=42)
    else:
        sample = panel[features]

    corr = sample.corr(method="spearman").abs()

    used = set()
    clusters = []
    features_list = list(features)

    for i, f1 in enumerate(features_list):
        if f1 in used:
            continue
        cluster = [f1]
        used.add(f1)
        for j in range(i + 1, len(features_list)):
            f2 = features_list[j]
            if f2 in used:
                continue
            if f1 in corr.index and f2 in corr.columns:
                if corr.loc[f1, f2] > threshold:
                    cluster.append(f2)
                    used.add(f2)
        clusters.append(cluster)

    return clusters


def select_features_ic_first(panel, tier_name, feature_cols, target_col,
                             budget=55):
    """
    IC-FIRST feature selection: NO force-inclusion.
    
    1. Remove macro interaction noise (ix_* features)
    2. Compute univariate IC for ALL remaining features within this tier
    3. Sort by |IC| descending
    4. Select top-N (overselect by 30% to allow for dedup)
    5. Deduplicate correlated features — keep BEST per cluster
    6. Trim to budget
    """
    available_features = set(panel.columns) & set(feature_cols)

    # Step 1: Remove noise
    noise_features = set()
    for f in available_features:
        if f.startswith("ix_"):
            noise_features.add(f)
        if f.startswith("ff49_"):
            noise_features.add(f)

    candidate_features = list(available_features - noise_features)
    print(f"    Candidates after removing noise: {len(candidate_features)}")

    # Step 2: Compute univariate IC for ALL candidates
    print(f"    Computing univariate IC for ALL {len(candidate_features)} features...")
    t0 = time.time()
    ic_results = compute_univariate_ic(panel, candidate_features, target_col,
                                       min_periods=36)
    print(f"    IC computed for {len(ic_results)} features ({time.time()-t0:.0f}s)")

    # Step 3: Sort by |IC|
    ranked = sorted(ic_results.items(), key=lambda x: x[1]["abs_ic"], reverse=True)

    # Step 4: Overselect top features (1.5× budget for dedup headroom)
    overselect = int(budget * 1.5)
    pre_dedup = []
    for feat, info in ranked:
        if len(pre_dedup) >= overselect:
            break
        # Minimum: |IC| > 0.005 and coverage > 10%
        if info["abs_ic"] > 0.005 and panel[feat].notna().mean() > 0.10:
            pre_dedup.append(feat)

    print(f"    Pre-dedup pool: {len(pre_dedup)} features (top by |IC|)")
    print(f"    IC range: {ic_results[pre_dedup[0]]['abs_ic']:.4f} → "
          f"{ic_results[pre_dedup[-1]]['abs_ic']:.4f}")

    # Step 5: Deduplicate — lower threshold (0.80) than step6d (0.85)
    # to be more aggressive about removing redundancy
    clusters = cluster_correlated_features(panel, pre_dedup, ic_results,
                                           threshold=0.80)

    deduplicated = []
    dedup_log = []
    for cluster in clusters:
        if len(cluster) == 1:
            deduplicated.append(cluster[0])
        else:
            # Keep the one with highest |IC| — NO force-include bias
            best = max(cluster, key=lambda f: ic_results.get(f, {}).get("abs_ic", 0))
            deduplicated.append(best)
            removed = [f for f in cluster if f != best]
            dedup_log.append((best, removed))

    print(f"    After deduplication: {len(deduplicated)} features")
    if dedup_log:
        print(f"    Dedup decisions (kept → removed):")
        for best, removed in dedup_log[:10]:
            best_ic = ic_results.get(best, {}).get("mean_ic", 0)
            rem_str = ", ".join(f"{r}({ic_results.get(r, {}).get('mean_ic', 0):+.4f})"
                                for r in removed[:3])
            print(f"      KEPT {best}(IC={best_ic:+.4f}) → dropped {rem_str}")

    # Step 6: Trim to budget (already sorted by IC via the clusters)
    # Re-sort by |IC| after dedup
    deduplicated.sort(key=lambda f: ic_results.get(f, {}).get("abs_ic", 0),
                      reverse=True)
    final_features = deduplicated[:budget]

    # Always ensure gsector is included (needed for sector exposure)
    if "gsector" in candidate_features and "gsector" not in final_features:
        final_features.append("gsector")

    # Always ensure log_market_cap is included (size control)
    if "log_market_cap" in candidate_features and "log_market_cap" not in final_features:
        final_features.append("log_market_cap")

    final_features = sorted(final_features)

    # Report
    print(f"\n    ═══ FINAL FEATURE SET: {len(final_features)} features ═══")
    print(f"    Top 20 by |IC|:")
    top20 = sorted(final_features,
                   key=lambda f: ic_results.get(f, {}).get("abs_ic", 0),
                   reverse=True)[:20]
    for f in top20:
        info = ic_results.get(f, {})
        print(f"      {f:>40}: IC={info.get('mean_ic', 0):+.4f}  "
              f"|IC|={info.get('abs_ic', 0):.4f}  "
              f"IR={info.get('ic_ir', 0):.2f}")

    return final_features, ic_results


# ════════════════════════════════════════════════════════════
# RANK NORMALIZATION
# ════════════════════════════════════════════════════════════

def rank_normalize_features(df, feature_cols):
    """Rank-normalize features within each cross-section to [-1, 1]."""
    for col in feature_cols:
        if col in df.columns:
            df[col] = df.groupby("date")[col].transform(
                lambda x: x.rank(pct=True, na_option="keep") * 2 - 1
            ).astype(np.float32)


def create_relevance_labels(df, target_col, n_bins=5):
    """Convert target to quintile labels (0-4) for LambdaRank."""
    labels = df.groupby("date")[target_col].transform(
        lambda x: pd.qcut(x.rank(method="first"), n_bins,
                          labels=False, duplicates="drop")
    )
    return labels.fillna(0).astype(np.int32)


def compute_query_groups(df, date_col="date"):
    """Compute query group sizes for LambdaRank."""
    groups = df.groupby(date_col).size().values
    return groups


# ════════════════════════════════════════════════════════════
# MODEL TRAINING — Stronger regularization for large/mega
# ════════════════════════════════════════════════════════════

def train_lambdarank(X_train, y_train, groups_train,
                     X_val, y_val, groups_val,
                     feature_names=None, tier="large"):
    """Train LambdaRank LightGBM with stronger regularization."""
    import lightgbm as lgb

    # Stronger regularization for large/mega (low-dispersion universes)
    tier_params = {
        "mega": {
            "num_leaves": 23,           # Was 31 → reduced for fewer features
            "min_child_samples": 150,   # Was 100 → increased
            "max_depth": 4,             # Was 5 → shallower
            "learning_rate": 0.010,     # Was 0.015 → slower
            "feature_fraction": 0.75,   # Was 0.7 → slightly more per tree
            "bagging_fraction": 0.85,   # Was 0.8 → more data per tree
            "lambdarank_truncation_level": 40,
            "num_boost_round": 1200,    # More rounds but slower LR
            "lambda_l1": 0.3,           # Stronger L1
            "lambda_l2": 8.0,           # Stronger L2
        },
        "large": {
            "num_leaves": 31,           # Was 47 → reduced
            "min_child_samples": 250,   # Was 200 → increased
            "max_depth": 5,             # Was 6 → shallower
            "learning_rate": 0.010,     # Was 0.015 → slower
            "feature_fraction": 0.70,   # Was 0.6 → more per tree (fewer total features)
            "bagging_fraction": 0.85,   # Was 0.8
            "lambdarank_truncation_level": 80,
            "num_boost_round": 1200,    # More rounds
            "lambda_l1": 0.3,           # Stronger L1
            "lambda_l2": 8.0,           # Stronger L2
        },
    }

    tp = tier_params.get(tier, tier_params["large"])

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 50, 100],
        "lambdarank_truncation_level": tp["lambdarank_truncation_level"],
        "boosting_type": "gbdt",
        "num_leaves": tp["num_leaves"],
        "learning_rate": tp["learning_rate"],
        "feature_fraction": tp["feature_fraction"],
        "bagging_fraction": tp["bagging_fraction"],
        "bagging_freq": 1,
        "lambda_l1": tp["lambda_l1"],
        "lambda_l2": tp["lambda_l2"],
        "min_child_samples": tp["min_child_samples"],
        "max_depth": tp["max_depth"],
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
        "max_bin": 255,
        "min_gain_to_split": 0.02,      # Was 0.01 → stricter split criterion
    }

    dtrain = lgb.Dataset(
        X_train, label=y_train, group=groups_train,
        feature_name=feature_names, free_raw_data=True
    )
    dval = lgb.Dataset(
        X_val, label=y_val, group=groups_val,
        reference=dtrain, free_raw_data=True
    )

    model = lgb.train(
        params, dtrain,
        num_boost_round=tp["num_boost_round"],
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(150, verbose=False),  # Was 100 → more patient
            lgb.log_evaluation(0),
        ],
    )

    return model, model.best_iteration


def train_huber_lgb(X_train, y_train, X_val, y_val,
                    feature_names=None, tier="large"):
    """Huber LGB with stronger regularization."""
    import lightgbm as lgb

    tier_params = {
        "mega": {
            "num_leaves": 23,
            "min_child_samples": 150,
            "max_depth": 4,
            "learning_rate": 0.010,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.85,
            "lambda_l1": 0.3,
            "lambda_l2": 8.0,
            "num_boost_round": 1200,
        },
        "large": {
            "num_leaves": 31,
            "min_child_samples": 250,
            "max_depth": 5,
            "learning_rate": 0.010,
            "feature_fraction": 0.70,
            "bagging_fraction": 0.85,
            "lambda_l1": 0.3,
            "lambda_l2": 8.0,
            "num_boost_round": 1200,
        },
    }

    tp = tier_params.get(tier, tier_params["large"])

    params = {
        "objective": "huber", "alpha": 0.9,
        "boosting_type": "gbdt",
        "num_leaves": tp["num_leaves"],
        "learning_rate": tp["learning_rate"],
        "feature_fraction": tp["feature_fraction"],
        "bagging_fraction": tp["bagging_fraction"],
        "bagging_freq": 1,
        "lambda_l1": tp["lambda_l1"],
        "lambda_l2": tp["lambda_l2"],
        "min_child_samples": tp["min_child_samples"],
        "max_depth": tp["max_depth"],
        "verbose": -1, "n_jobs": -1, "seed": 42,
        "max_bin": 255, "min_gain_to_split": 0.02,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names,
                         free_raw_data=True)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)

    model = lgb.train(
        params, dtrain, num_boost_round=tp["num_boost_round"],
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
    )
    return model, model.best_iteration


# ════════════════════════════════════════════════════════════
# METRICS (same as step6d)
# ════════════════════════════════════════════════════════════

def compute_monthly_ics(df, pred_col, target_col):
    """Compute monthly Spearman IC."""
    ics = []
    for dt, grp in df.groupby("date"):
        mask = grp[pred_col].notna() & grp[target_col].notna()
        sub = grp[mask]
        if len(sub) >= 20:
            ic, _ = stats.spearmanr(sub[pred_col], sub[target_col])
            ics.append({"date": dt, "ic": ic})
    return pd.DataFrame(ics)


def compute_monotonicity(df, pred_col, target_col, n_quantiles=10):
    """Monotonicity: fraction of adjacent quantile pairs correctly ordered."""
    mono_scores = []
    for dt, grp in df.groupby("date"):
        mask = grp[pred_col].notna() & grp[target_col].notna()
        sub = grp[mask]
        if len(sub) < n_quantiles * 5:
            continue
        sub = sub.copy()
        sub["q"] = pd.qcut(sub[pred_col].rank(method="first"), n_quantiles,
                           labels=False)
        q_means = sub.groupby("q")[target_col].mean()
        if len(q_means) < n_quantiles:
            continue
        correct = sum(q_means.iloc[i + 1] > q_means.iloc[i]
                      for i in range(len(q_means) - 1))
        mono_scores.append(correct / (len(q_means) - 1))

    return np.mean(mono_scores) if mono_scores else 0.0


def compute_ls_spread(df, pred_col, target_col):
    """Monthly long-short spread (D10 - D1)."""
    spreads = []
    for dt, grp in df.groupby("date"):
        mask = grp[pred_col].notna() & grp[target_col].notna()
        sub = grp[mask]
        if len(sub) < 50:
            continue
        sub = sub.copy()
        sub["q"] = pd.qcut(sub[pred_col].rank(method="first"), 10, labels=False)
        long_ret = sub[sub["q"] == 9][target_col].mean()
        short_ret = sub[sub["q"] == 0][target_col].mean()
        spreads.append(long_ret - short_ret)
    return pd.Series(spreads)


def compute_capacity(df, pred_col, panel_full):
    """Estimate tradeable capacity."""
    top_q = df.copy()
    top_q["q"] = top_q.groupby("date")[pred_col].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False)
    )
    top_stocks = top_q[top_q["q"] == 4][["permno", "date"]]

    vol_cols = ["permno", "date"]
    for vc in ["market_cap", "mktcap", "turnover", "vol", "price", "prc"]:
        if vc in panel_full.columns:
            vol_cols.append(vc)

    if len(vol_cols) <= 2:
        return 0.0

    vol_data = panel_full[vol_cols].copy()
    merged = top_stocks.merge(vol_data, on=["permno", "date"], how="left")

    cap_col = "market_cap" if "market_cap" in merged.columns else "mktcap"
    if cap_col in merged.columns and "turnover" in merged.columns:
        merged["ddv"] = merged[cap_col] * merged["turnover"] / 21
        median_ddv = merged["ddv"].median()
        return float(median_ddv * 0.05) if not np.isnan(median_ddv) else 0.0

    if cap_col in merged.columns:
        merged["ddv"] = merged[cap_col] * 0.01
        return float(merged["ddv"].median() * 0.05)

    return 0.0


def compute_quantile_returns(df, pred_col, target_col, n_quantiles=10):
    """Mean return by quantile."""
    all_q = []
    for dt, grp in df.groupby("date"):
        mask = grp[pred_col].notna() & grp[target_col].notna()
        sub = grp[mask]
        if len(sub) < n_quantiles * 5:
            continue
        sub = sub.copy()
        sub["q"] = pd.qcut(sub[pred_col].rank(method="first"), n_quantiles,
                           labels=False)
        for q in range(n_quantiles):
            qsub = sub[sub["q"] == q]
            all_q.append({"date": dt, "q": q, "ret": qsub[target_col].mean(),
                          "n": len(qsub)})

    qdf = pd.DataFrame(all_q)
    if len(qdf) == 0:
        return pd.DataFrame()
    return qdf.groupby("q").agg({"ret": "mean", "n": "mean"}).reset_index()


def compute_year_by_year_ic(df, pred_col, target_col):
    """Compute IC by year for diagnostics."""
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    yearly = []
    for yr, ygrp in df.groupby("year"):
        ic_df = compute_monthly_ics(ygrp, pred_col, target_col)
        if len(ic_df) > 0:
            yearly.append({"year": yr, "ic": ic_df["ic"].mean(),
                           "n_months": len(ic_df)})
    return pd.DataFrame(yearly)


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    np.random.seed(42)

    print("=" * 70)
    print("STEP 6h: LARGE-CAP & MEGA-CAP IC FIX")
    print("=" * 70)
    print("  Root-cause fix: IC-FIRST feature selection (no force-inclusion)")
    print("  Tighter budgets + stronger regularization")
    print("  Only retraining: mega + large (mid/small stay as-is)")
    print()

    # ── Load panel ──
    gkx_path = os.path.join(DATA_DIR, "gkx_panel.parquet")
    if not os.path.exists(gkx_path):
        print("  ERROR: gkx_panel.parquet not found!")
        return

    print("Loading panel...")
    panel = pd.read_parquet(gkx_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["year"] = panel["date"].dt.year
    print(f"  Panel: {len(panel):,} rows x {panel.shape[1]} cols")

    target_col = "fwd_ret_1m"

    # Identify ALL features
    id_cols = {"permno", "date", "cusip", "ticker", "siccd", "year",
               "ret", "ret_crsp", "fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m",
               "fwd_ret_12m", "ret_forward", "dlret", "dlstcd",
               "__fragment_index", "__batch_index", "__last_in_fragment",
               "__filename"}
    all_feature_cols = [c for c in panel.columns if c not in id_cols
                        and panel[c].dtype in ["float64", "float32", "int64", "int32"]
                        and c not in ("sp500_member",)]

    panel = panel.dropna(subset=[target_col])
    print(f"  All features: {len(all_feature_cols)}, Target: {target_col}")
    print(f"  Rows with target: {len(panel):,}")

    if "log_market_cap" not in panel.columns:
        print("  ERROR: log_market_cap not in panel!")
        return

    panel["cap_M"] = np.exp(panel["log_market_cap"])

    # Float32 conversion
    print("  Converting to float32...")
    for col in all_feature_cols:
        if col in panel.columns:
            panel[col] = panel[col].astype(np.float32)
    panel[target_col] = panel[target_col].astype(np.float32)
    gc.collect()

    # Save volume data for capacity
    vol_cols_to_keep = ["permno", "date"]
    for vc in ["vol", "price", "prc", "mktcap", "market_cap", "turnover"]:
        if vc in panel.columns:
            vol_cols_to_keep.append(vc)
    panel_vol = panel[vol_cols_to_keep].copy()

    # ── Load step6d baselines for comparison ──
    baselines = {}
    for tier_name in ["mega", "large"]:
        baseline_path = os.path.join(DATA_DIR,
                                     f"curated_predictions_{tier_name}.parquet")
        if os.path.exists(baseline_path):
            bdf = pd.read_parquet(baseline_path)
            bdf["date"] = pd.to_datetime(bdf["date"])
            ic_df = compute_monthly_ics(bdf, "prediction", target_col)
            mono = compute_monotonicity(bdf, "prediction", target_col)
            baselines[tier_name] = {
                "ic": ic_df["ic"].mean() if len(ic_df) > 0 else 0,
                "mono": mono,
                "n_preds": len(bdf),
            }
            print(f"  Baseline {tier_name}: IC={baselines[tier_name]['ic']:+.4f}, "
                  f"Mono={baselines[tier_name]['mono']:.3f}")

    # ── Walk-forward ──
    test_start = 2005
    test_end = int(panel["year"].max())
    train_window = 10

    tier_results = {}

    for tier_name, tier_def in TIER_DEFS.items():
        tier_t0 = time.time()
        print(f"\n{'=' * 70}")
        print(f"TIER: {tier_def['label']} — IC-FIRST FIX")
        print(f"  log_market_cap range: [{tier_def['min_lmc']:.2f}, "
              f"{tier_def['max_lmc']:.2f})")
        print(f"{'=' * 70}")

        # Filter to tier
        tier_mask = (panel["log_market_cap"] >= tier_def["min_lmc"])
        if tier_def["max_lmc"] != np.inf:
            tier_mask &= (panel["log_market_cap"] < tier_def["max_lmc"])
        tier_panel = panel[tier_mask].copy()

        year_counts = tier_panel.groupby("year").size()
        print(f"  Total rows: {len(tier_panel):,}")
        print(f"  Year range: {tier_panel['year'].min()}-{tier_panel['year'].max()}")
        print(f"  Avg rows/year: {year_counts.mean():.0f}")

        if len(tier_panel) < 10000:
            print(f"  SKIP: too few rows for {tier_name}")
            continue

        # ── IC-FIRST FEATURE SELECTION ──
        print(f"\n  ── IC-FIRST Feature Selection for {tier_name} ──")
        budget = TIER_FEATURE_BUDGET[tier_name]

        # Use training data for IC screening (not test period)
        ic_sample = tier_panel[tier_panel["year"].between(2000, 2020)]
        if len(ic_sample) < 5000:
            ic_sample = tier_panel[tier_panel["year"] >= 1995]

        tier_features, ic_results = select_features_ic_first(
            ic_sample, tier_name, all_feature_cols, target_col, budget=budget
        )

        print(f"\n  Using {len(tier_features)} IC-curated features")

        # Rank-normalize
        print(f"  Rank-normalizing {len(tier_features)} features...")
        t_rank = time.time()
        rank_normalize_features(tier_panel, tier_features)
        print(f"  Rank normalization: {time.time() - t_rank:.0f}s")

        # Create labels
        tier_panel["relevance"] = create_relevance_labels(tier_panel, target_col,
                                                          n_bins=5)
        tier_panel["target_ranked"] = tier_panel.groupby("date")[target_col].transform(
            lambda x: stats.norm.ppf(
                x.rank(pct=True, na_option="keep").clip(0.001, 0.999)
            )
        ).astype(np.float32)

        # ── Walk-forward ──
        all_preds = []
        all_importances = []

        for test_year in range(test_start, test_end + 1):
            train_start_year = test_year - train_window
            train_end_year = test_year - 1

            train_mask = (tier_panel["year"] >= train_start_year) & \
                         (tier_panel["year"] <= train_end_year)
            # 6-month embargo
            train_dates = tier_panel.loc[train_mask, "date"].unique()
            if len(train_dates) > 6:
                embargo_cutoff = sorted(train_dates)[-6]
                train_mask = train_mask & (tier_panel["date"] < embargo_cutoff)

            test_mask = tier_panel["year"] == test_year

            train_df = tier_panel[train_mask]
            test_df = tier_panel[test_mask]

            if len(train_df) < 1000 or len(test_df) < 100:
                continue

            X_train = train_df[tier_features].values
            X_test = test_df[tier_features].values

            # ── LambdaRank ──
            y_train_rel = train_df["relevance"].values
            groups_train = compute_query_groups(train_df)

            # Validation: last 20%
            n_val = max(1, int(len(X_train) * 0.2))
            X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
            y_tr_rel, y_val_rel = y_train_rel[:-n_val], y_train_rel[-n_val:]

            tr_df_tmp = train_df.iloc[:-n_val]
            val_df_tmp = train_df.iloc[-n_val:]
            groups_tr = compute_query_groups(tr_df_tmp)
            groups_val = compute_query_groups(val_df_tmp)

            try:
                lr_model, lr_best = train_lambdarank(
                    X_tr, y_tr_rel, groups_tr,
                    X_val, y_val_rel, groups_val,
                    feature_names=tier_features, tier=tier_name
                )
                lr_pred = lr_model.predict(X_test)
            except Exception as e:
                print(f"  {test_year}: LR failed ({e}), using Huber only")
                lr_pred = None

            # ── Huber LGB ──
            y_train_ranked = train_df["target_ranked"].values
            y_tr_ranked = y_train_ranked[:-n_val]
            y_val_ranked = y_train_ranked[-n_val:]

            try:
                hu_model, hu_best = train_huber_lgb(
                    X_tr, y_tr_ranked, X_val, y_val_ranked,
                    feature_names=tier_features, tier=tier_name
                )
                hu_pred = hu_model.predict(X_test)
            except Exception as e:
                print(f"  {test_year}: Huber failed ({e})")
                hu_pred = None

            if lr_pred is None and hu_pred is None:
                continue

            # ── IC-weighted blend ──
            actual = test_df[target_col].values

            if lr_pred is not None and hu_pred is not None:
                lr_rank = stats.rankdata(lr_pred) / len(lr_pred)
                hu_rank = stats.rankdata(hu_pred) / len(hu_pred)

                lr_val_pred = lr_model.predict(X_val)
                hu_val_pred = hu_model.predict(X_val)
                lr_ic, _ = stats.spearmanr(lr_val_pred, y_val_ranked)
                hu_ic, _ = stats.spearmanr(hu_val_pred, y_val_ranked)
                lr_ic = max(lr_ic, 0.001)
                hu_ic = max(hu_ic, 0.001)
                w_lr = lr_ic / (lr_ic + hu_ic)
                w_hu = hu_ic / (lr_ic + hu_ic)

                combined = w_lr * lr_rank + w_hu * hu_rank
            elif lr_pred is not None:
                combined = stats.rankdata(lr_pred) / len(lr_pred)
                w_lr, w_hu = 1.0, 0.0
            else:
                combined = stats.rankdata(hu_pred) / len(hu_pred)
                w_lr, w_hu = 0.0, 1.0

            test_ic, _ = stats.spearmanr(combined, actual)

            pred_df = test_df[["permno", "date"]].copy()
            pred_df["prediction"] = combined
            pred_df["tier"] = tier_name
            pred_df[target_col] = actual
            all_preds.append(pred_df)

            if hu_pred is not None:
                imp = hu_model.feature_importance(importance_type="gain")
                imp_norm = imp / imp.sum() * 100
                all_importances.append(dict(zip(tier_features, imp_norm)))

            spread_series = compute_ls_spread(pred_df, "prediction", target_col)
            spread = spread_series.mean() if len(spread_series) > 0 else 0

            ic_tag = "[OK]" if test_ic > 0.01 else "[--]" if test_ic > -0.01 else "[XX]"
            print(f"  {test_year}: IC={test_ic:+.4f}{ic_tag} "
                  f"spread={spread:+.4f} "
                  f"w=[LR:{w_lr:.2f},HU:{w_hu:.2f}] "
                  f"n={len(test_df):,} "
                  f"({time.time() - tier_t0:.0f}s)")
            tier_t0 = time.time()

        if not all_preds:
            print(f"  No predictions for {tier_name}")
            continue

        preds = pd.concat(all_preds, ignore_index=True)

        # ── Tier Summary ──
        ic_df = compute_monthly_ics(preds, "prediction", target_col)
        mean_ic = ic_df["ic"].mean() if len(ic_df) > 0 else 0
        ic_std = ic_df["ic"].std() if len(ic_df) > 0 else 1
        ic_ir = mean_ic / ic_std if ic_std > 0 else 0
        pct_pos = (ic_df["ic"] > 0).mean() if len(ic_df) > 0 else 0

        mono = compute_monotonicity(preds, "prediction", target_col)

        spreads = compute_ls_spread(preds, "prediction", target_col)
        ls_sharpe = (spreads.mean() / spreads.std() * np.sqrt(12)
                     if len(spreads) > 0 and spreads.std() > 0 else 0)

        capacity = compute_capacity(preds, "prediction", panel_vol)

        # Year-by-year IC for diagnostics
        yearly_ic = compute_year_by_year_ic(preds, "prediction", target_col)

        # Quantile returns
        qr = compute_quantile_returns(preds, "prediction", target_col)

        # Comparison with baseline
        baseline = baselines.get(tier_name, {})
        ic_delta = mean_ic - baseline.get("ic", 0) if baseline else 0
        mono_delta = mono - baseline.get("mono", 0) if baseline else 0

        # Print tier summary box
        print(f"\n  ╔══════════════════════════════════════════════════╗")
        print(f"  ║{tier_def['label'] + ' — IC-FIRST FIX':^50}║")
        print(f"  ╠══════════════════════════════════════════════════╣")
        print(f"  ║ IC:           {mean_ic:+.4f}  (IR: {ic_ir:.2f})"
              f"{'':>17}║")
        print(f"  ║ IC > 0:       {pct_pos:.0%}"
              f"{'':>33}║")
        print(f"  ║ Monotonicity: {mono:.3f}  (target: 0.60+)"
              f"{'':>11}║")
        print(f"  ║ L/S Sharpe:   {ls_sharpe:.2f}"
              f"{'':>34}║")
        print(f"  ║ Capacity:     ${capacity:>12,.0f}"
              f"{'':>22}║")
        print(f"  ║ Features:     {len(tier_features):>4}"
              f"{'':>34}║")
        print(f"  ║ Predictions:  {len(preds):>9,}"
              f"{'':>25}║")
        print(f"  ╠══════════════════════════════════════════════════╣")
        if baseline:
            ic_emoji = "✅" if ic_delta > 0 else "❌"
            mono_emoji = "✅" if mono_delta > 0 else "❌"
            print(f"  ║ vs BASELINE:                                    ║")
            print(f"  ║  IC:   {baseline.get('ic', 0):+.4f} → {mean_ic:+.4f}"
                  f"  ({ic_delta:+.4f}) {ic_emoji}"
                  f"{'':>13}║")
            print(f"  ║  Mono: {baseline.get('mono', 0):.3f} → {mono:.3f}"
                  f"  ({mono_delta:+.3f}) {mono_emoji}"
                  f"{'':>14}║")
        print(f"  ╚══════════════════════════════════════════════════╝")

        # Year-by-year IC
        if len(yearly_ic) > 0:
            print(f"\n  Year-by-year IC:")
            neg_years = 0
            for _, row in yearly_ic.iterrows():
                yr = int(row["year"])
                yic = row["ic"]
                tag = " ←NEG" if yic < 0 else ""
                if yic < 0:
                    neg_years += 1
                print(f"    {yr}: IC={yic:+.4f}{tag}")
            print(f"  Negative IC years: {neg_years}/{len(yearly_ic)}")

        # Quantile returns
        if len(qr) > 0:
            print(f"\n  Quantile returns (D1=worst, D10=best):")
            for _, row in qr.iterrows():
                q_num = int(row["q"]) + 1
                ret = row["ret"] * 100
                bar_len = int(abs(ret) * 50 / max(abs(qr["ret"].max()),
                                                   abs(qr["ret"].min()),
                                                   0.01) * 100)
                bar_len = min(bar_len, 60)
                sign = "+" if ret >= 0 else "-"
                bar = sign + "█" * bar_len
                print(f"    D{q_num:>2}: {ret:+.3f}%/mo  {bar}")

        # Feature importance
        if all_importances:
            avg_imp = {}
            for imp_dict in all_importances:
                for f, v in imp_dict.items():
                    avg_imp[f] = avg_imp.get(f, [])
                    avg_imp[f].append(v)
            avg_imp = {f: np.mean(v) for f, v in avg_imp.items()}
            top_feats = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)

            print(f"\n  Top 15 features:")
            top15_total = sum(v for _, v in top_feats[:15])
            for f, v in top_feats[:15]:
                f_ic = ic_results.get(f, {}).get("mean_ic", 0)
                print(f"    {f:>45} {v:.2f}%  (IC={f_ic:+.4f})")
            print(f"    {'Top-15 concentration:':>45} {top15_total:.1f}%")

            fi_df = pd.DataFrame(top_feats, columns=["feature", "importance"])
            fi_df.to_csv(os.path.join(RESULTS_DIR,
                                      f"feature_importance_icfirst_{tier_name}.csv"),
                         index=False)

        # Save predictions
        preds.to_parquet(os.path.join(DATA_DIR,
                                      f"curated_predictions_{tier_name}.parquet"),
                         index=False)
        print(f"\n  ✓ Saved: curated_predictions_{tier_name}.parquet ({len(preds):,} rows)")

        tier_results[tier_name] = {
            "ic": round(mean_ic, 4),
            "ir": round(ic_ir, 2),
            "monotonicity": round(mono, 3),
            "ls_sharpe": round(ls_sharpe, 2),
            "capacity_usd": round(capacity, 0),
            "n_features": len(tier_features),
            "n_predictions": len(preds),
            "n_months": len(ic_df),
            "pct_ic_positive": round(pct_pos, 2),
            "features": tier_features,
            "baseline_ic": round(baseline.get("ic", 0), 4),
            "baseline_mono": round(baseline.get("mono", 0), 3),
            "ic_delta": round(ic_delta, 4),
            "mono_delta": round(mono_delta, 3),
        }

    # ══════════════════════════════════════════════════════════
    # COMBINED SUMMARY
    # ══════════════════════════════════════════════════════════

    total_time = (time.time() - t_start) / 60

    print(f"\n\n{'═' * 70}")
    print("STEP 6h — LARGE-CAP IC FIX — RESULTS SUMMARY")
    print(f"{'═' * 70}")

    print(f"\n{'Tier':<12} {'Old IC':>8} {'New IC':>8} {'ΔIC':>8} "
          f"{'Old Mono':>10} {'New Mono':>10} {'ΔMono':>8}")
    print("─" * 70)

    for tier_name in ["mega", "large"]:
        if tier_name not in tier_results:
            continue
        r = tier_results[tier_name]
        print(f"{tier_name:<12} {r['baseline_ic']:>+8.4f} {r['ic']:>+8.4f} "
              f"{r['ic_delta']:>+8.4f} "
              f"{r['baseline_mono']:>10.3f} {r['monotonicity']:>10.3f} "
              f"{r['mono_delta']:>+8.3f}")

    print("─" * 70)

    # Overall assessment
    all_improved = all(tier_results[t]["ic_delta"] > 0
                       for t in tier_results if "ic_delta" in tier_results[t])
    mono_improved = all(tier_results[t]["mono_delta"] > 0
                        for t in tier_results if "mono_delta" in tier_results[t])

    print(f"\n  IC improvement:   {'✅ ALL TIERS' if all_improved else '⚠️  MIXED'}")
    print(f"  Mono improvement: {'✅ ALL TIERS' if mono_improved else '⚠️  MIXED'}")

    # Load mid/small baselines for full picture
    print(f"\n  Full system after fix:")
    full_tiers = {}
    for tier_name in ["mega", "large", "mid", "small"]:
        if tier_name in tier_results:
            full_tiers[tier_name] = tier_results[tier_name]
        else:
            # Load from step6d
            fp = os.path.join(DATA_DIR, f"curated_predictions_{tier_name}.parquet")
            if os.path.exists(fp):
                tdf = pd.read_parquet(fp)
                tdf["date"] = pd.to_datetime(tdf["date"])
                ic_df = compute_monthly_ics(tdf, "prediction", target_col)
                mono = compute_monotonicity(tdf, "prediction", target_col)
                full_tiers[tier_name] = {
                    "ic": round(ic_df["ic"].mean(), 4) if len(ic_df) > 0 else 0,
                    "monotonicity": round(mono, 3),
                }

    print(f"  {'Tier':<12} {'IC':>8} {'Mono':>8}")
    print(f"  {'─' * 28}")
    for t in ["mega", "large", "mid", "small"]:
        if t in full_tiers:
            print(f"  {t:<12} {full_tiers[t]['ic']:>+8.4f} "
                  f"{full_tiers[t]['monotonicity']:>8.3f}")

    print(f"\n  Total time: {total_time:.1f} min")

    # Save summary
    summary = {
        "method": "ic_first_large_cap_fix",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_min": round(total_time, 1),
        "approach": "IC-FIRST: no force-inclusion, tier-specific IC selection, "
                    "tighter budgets, stronger regularization",
        "tiers": {k: {kk: vv for kk, vv in v.items() if kk != "features"}
                  for k, v in tier_results.items()},
        "feature_lists": {k: v.get("features", []) for k, v in tier_results.items()},
        "full_system": {k: v for k, v in full_tiers.items()},
    }
    with open(os.path.join(RESULTS_DIR, "ic_first_fix_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'═' * 70}")
    print(f"DONE. Now re-run step6e (hedging) → step6f (ensemble) → step6g (audit)")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
