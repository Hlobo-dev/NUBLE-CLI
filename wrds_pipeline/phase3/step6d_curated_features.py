"""
PHASE 3 — STEP 6d: CURATED FEATURE MULTI-UNIVERSE
====================================================
Prompt 2 Execution: Tier-specific curated feature sets.

PROBLEM DIAGNOSIS:
  - 533 features overwhelm the model → most are macro interaction noise
  - Level 3 features (our best work) contribute ~0% importance
  - Model defaults to turnover/vol → these don't predict in large/mega cap
  - Feature importance is flat at 0.01-0.05% per feature = no signal

SOLUTION:
  1. Per-tier univariate IC screening (keep features with |IC| > threshold)
  2. Academic alpha factor force-inclusion (earnings momentum, accrual quality,
     institutional flow, value composites, quality metrics)
  3. Cap features at 60-80 per tier (signal concentration)
  4. Feature clustering + deduplication (remove redundant turnover_3m/6m/12m)
  5. Same LambdaRank + Huber ensemble from step6c

TARGETS:
  Mega-cap IC: 0.010 → 0.030+
  Large-cap IC: -0.008 → 0.020+
  Monotonicity: 0.50 → 0.60+
  Feature importance concentration: top-10 features > 20% total

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
# TIER DEFINITIONS (same as step6c)
# ════════════════════════════════════════════════════════════

TIER_DEFS = {
    "mega":  {"label": "Mega-Cap (>$10B)",     "min_lmc": 9.21,  "max_lmc": np.inf},
    "large": {"label": "Large-Cap ($2-10B)",   "min_lmc": 7.60,  "max_lmc": 9.21},
    "mid":   {"label": "Mid-Cap ($500M-2B)",   "min_lmc": 6.21,  "max_lmc": 7.60},
    "small": {"label": "Small-Cap ($100M-500M)","min_lmc": 4.61,  "max_lmc": 6.21},
}

# ════════════════════════════════════════════════════════════
# ACADEMIC ALPHA FACTORS — FORCE-INCLUDED PER TIER
# ════════════════════════════════════════════════════════════
# These are the factors with the strongest academic evidence.
# We force-include them even if univariate IC is low, because
# they have conditional alpha that LGB can extract.

# Core factors that MUST be in every tier
CORE_FACTORS = [
    # ── Value (Fama-French, Asness) ──
    "bm", "ep", "cashpr", "dy", "cfp",
    "pe_op_basic", "pe_exi", "peg_1yrforward", "peg_ltgforward",
    
    # ── Momentum (Jegadeesh-Titman, Novy-Marx) ──
    "mom_12m", "mom_6m", "mom_1m", "mom_12_2",
    
    # ── Size ──
    "log_market_cap", "market_cap",
    
    # ── Profitability (Novy-Marx 2013, Fama-French 2015) ──
    "roaq", "roeq", "gp_at",  # gross profitability
    
    # ── Investment (Fama-French 2015, Hou-Xue-Zhang 2015) ──
    "agr",  # asset growth rate
    
    # ── Liquidity / Turnover ──
    "turnover", "amihud_illiq",
    
    # ── Volatility ──
    "realized_vol", "idio_vol", "beta",
    
    # ── Industry ──
    "gsector",
]

# Tier 1 features: earnings momentum, accrual quality, institutional flow
# These are SPECIFICALLY designed for large/mega cap where price-based
# factors fail and fundamental information matters more.
TIER1_EARNINGS_MOMENTUM = [
    # ── IBES Analyst Dynamics (our Level 3 features) ──
    "eps_revision_1m",         # 1-month EPS estimate change
    "eps_revision_3m",         # 3-month EPS estimate change
    "revision_breadth",        # (up - down) / total analysts
    "eps_dispersion",          # analyst disagreement
    "eps_dispersion_trend",    # change in disagreement
    "eps_estimate_momentum",   # 3-month revision momentum
    "sue_ibes",                # standardized unexpected earnings
    "beat_miss_streak",        # consecutive beats/misses
    "num_analysts_fy1",        # analyst coverage
    # ── GKX original analyst features ──
    "analyst_revision",        # GKX original
    "analyst_dispersion",      # GKX original
    "sue",                     # GKX original SUE
    "num_analysts",            # GKX original coverage
]

TIER1_ACCRUAL_QUALITY = [
    # ── Sloan/Richardson Accruals ──
    "total_accruals",          # Sloan 1996
    "working_capital_accruals", # Richardson 2005
    "non_current_accruals",    # Richardson 2005
    "accruals_to_cash_flow",   # CF vs accruals ratio
    "cf_earnings_divergence",  # OCF - NI divergence
    "accrual",                 # GKX original
    "accruals_vs_industry",    # Industry-relative accruals
    # ── Beneish M-Score Components ──
    "beneish_m_score",         # Composite manipulation score
    "beneish_gmi",             # Gross margin index
    "beneish_aqi",             # Asset quality index
    "beneish_sgi",             # Sales growth index
    "beneish_tata",            # Total accruals to total assets
    "dsri",                    # Days sales in receivables index
    # ── Quality Scores ──
    "piotroski_f_score",       # Piotroski 2000 (9-point)
    "altman_z_score",          # Altman 1968
    "ohlson_o_score",          # Ohlson 1980
    "montier_c_score",         # Montier quality
    "piotroski_vs_industry",   # Industry-relative Piotroski
    "earnings_persistence",    # EPS autocorrelation
    "earnings_smoothness",     # σ(NI) / σ(OCF)
    "net_operating_assets",    # Hirshleifer 2004
]

TIER1_INSTITUTIONAL_FLOW = [
    # ── Institutional Holdings ──
    "inst_ownership_change",   # QoQ institutional ownership change
    "inst_breadth",            # Change in number of institutions
    "inst_hhi",                # Institutional concentration
    "total_inst_shares",       # Total institutional shares held
    "num_institutions",        # Number of institutions
    # ── Insider Trading ──
    "insider_buy_ratio_6m",    # 6-month buy ratio
    "insider_cluster_buy",     # Multiple insiders buying
    "insider_ceo_buy",         # CEO buying signal
    "insider_buy_ratio",       # Monthly buy ratio
    "insider_num_buys",        # Number of insider buys
    "insider_num_sells",       # Number of insider sells
]

TIER1_FINANCIAL_DYNAMICS = [
    # ── Revenue & Growth ──
    "revenue_growth_yoy",      # Year-over-year revenue growth
    "revenue_acceleration",    # Change in YoY growth
    "revenue_cagr_2yr",        # 2-year CAGR
    "operating_leverage",      # %ΔEBIT / %ΔRevenue
    # ── Margin Trajectories ──
    "gross_margin_trend",      # 4Q slope of gross margin
    "operating_margin_trend",  # 4Q slope of operating margin
    "net_margin_trend",        # 4Q slope of net margin
    "margin_divergence",       # Gross vs net margin divergence
    "gross_margin_vol",        # Margin stability
    # ── Working Capital ──
    "cash_conversion_cycle",   # DSO + DIO - DPO
    "ccc_trend",               # CCC trajectory
    # ── Capital Allocation ──
    "capex_to_depreciation",   # Investment rate
    "capex_intensity",         # CapEx/Revenue
    "rd_intensity",            # R&D/Revenue
    "fcf_to_revenue",          # FCF margin
    "fcf_to_revenue_trend",    # FCF margin trajectory
    # ── Leverage ──
    "debt_to_equity_change",   # Leverage trajectory
    "interest_coverage_trend", # Coverage trajectory
    "net_debt_to_ebitda",      # Leverage ratio
    "debt_maturity_risk",      # Short-term debt / total debt
]

# Features that should be EXCLUDED because they add noise
NOISE_FEATURES = {
    # Macro interaction features — these are the main noise source
    # They're 400+ features like ix_realized_vol_x_manufacturing_employment
    # that just add collinearity without real signal
}

# Per-tier feature budgets
TIER_FEATURE_BUDGET = {
    "mega":  70,   # Fewer features for smaller universe
    "large": 80,
    "mid":   80,
    "small": 80,
}

# ════════════════════════════════════════════════════════════
# FEATURE SELECTION ENGINE
# ════════════════════════════════════════════════════════════

def compute_univariate_ic(panel, feature_cols, target_col, min_periods=60):
    """
    Compute per-feature time-series average Spearman IC.
    Returns dict: feature → (mean_ic, ic_ir, pct_positive).
    """
    ic_results = {}
    dates = sorted(panel["date"].unique())
    
    for feat in feature_cols:
        monthly_ics = []
        for dt in dates:
            month_data = panel[panel["date"] == dt]
            mask = month_data[feat].notna() & month_data[target_col].notna()
            sub = month_data[mask]
            if len(sub) >= 30:
                ic, _ = stats.spearmanr(sub[feat], sub[target_col])
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


def cluster_correlated_features(panel, features, threshold=0.8, sample_n=50000):
    """
    Cluster highly correlated features, keep only the one with highest IC per cluster.
    """
    # Sample for speed
    if len(panel) > sample_n:
        sample = panel[features].sample(n=sample_n, random_state=42)
    else:
        sample = panel[features]
    
    # Correlation matrix
    corr = sample.corr(method="spearman").abs()
    
    # Greedy clustering
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


def select_features_for_tier(panel, tier_name, feature_cols, target_col,
                             budget=80, ic_results=None):
    """
    Select curated feature set for a specific tier.
    
    Strategy:
    1. Force-include academic alpha factors (if they exist in panel)
    2. Add univariate IC-screened features up to budget
    3. Remove highly correlated duplicates within each cluster
    4. Exclude macro interaction noise
    """
    available_features = set(panel.columns) & set(feature_cols)
    
    # Step 1: Identify macro interaction noise
    noise_features = set()
    for f in available_features:
        if f.startswith("ix_"):  # All macro interaction features
            noise_features.add(f)
        if f.startswith("ff49_"):  # Individual FF49 dummies (keep ffi10/ffi12/ffi30/ffi49)
            noise_features.add(f)
    
    candidate_features = available_features - noise_features
    print(f"    Candidates after removing noise: {len(candidate_features)} "
          f"(removed {len(noise_features)} macro interactions + FF49 dummies)")
    
    # Step 2: Force-include academic alpha factors
    force_include = set()
    all_forced = (CORE_FACTORS + TIER1_EARNINGS_MOMENTUM + TIER1_ACCRUAL_QUALITY +
                  TIER1_INSTITUTIONAL_FLOW + TIER1_FINANCIAL_DYNAMICS)
    
    for f in all_forced:
        if f in candidate_features:
            force_include.add(f)
    
    # Check coverage of forced features
    forced_coverage = {}
    for f in force_include:
        cov = panel[f].notna().mean()
        forced_coverage[f] = cov
    
    # Only force-include features with >10% coverage
    force_include = {f for f in force_include if forced_coverage.get(f, 0) > 0.10}
    print(f"    Force-included alpha factors: {len(force_include)} (>10% coverage)")
    
    # Step 3: Compute univariate IC for remaining candidates
    remaining = candidate_features - force_include
    if ic_results is None:
        print(f"    Computing univariate IC for {len(remaining)} features...")
        ic_results = compute_univariate_ic(panel, list(remaining), target_col,
                                           min_periods=36)
    
    # Step 4: Rank by |IC| and fill up to budget
    ranked = sorted(ic_results.items(), key=lambda x: x[1]["abs_ic"], reverse=True)
    
    selected = set(force_include)
    remaining_budget = budget - len(selected)
    
    for feat, info in ranked:
        if len(selected) >= budget:
            break
        if feat not in selected and feat in candidate_features:
            # Minimum IC threshold: |IC| > 0.005 (non-trivial)
            if info["abs_ic"] > 0.005:
                selected.add(feat)
    
    # Step 5: Deduplicate highly correlated features
    sel_list = list(selected)
    if len(sel_list) > 20:
        clusters = cluster_correlated_features(panel, sel_list, threshold=0.85)
        
        # For each cluster, keep the one with highest IC
        deduplicated = set()
        for cluster in clusters:
            if len(cluster) == 1:
                deduplicated.add(cluster[0])
            else:
                # Pick the one with highest |IC|
                best = None
                best_ic = -1
                for f in cluster:
                    f_ic = ic_results.get(f, {}).get("abs_ic", 0)
                    if f in force_include:
                        f_ic += 0.1  # Bias toward force-included
                    if f_ic > best_ic:
                        best = f
                        best_ic = f_ic
                deduplicated.add(best)
        
        removed = len(selected) - len(deduplicated)
        if removed > 0:
            print(f"    Deduplicated: removed {removed} correlated features")
        selected = deduplicated
    
    # Final sort
    final_features = sorted(selected)
    
    # Report
    n_forced = len(force_include & selected)
    n_screened = len(selected - force_include)
    print(f"    Final feature set: {len(final_features)} features "
          f"({n_forced} forced + {n_screened} IC-screened)")
    
    # Report IC of forced features
    forced_ics = []
    for f in sorted(force_include & selected):
        f_ic = ic_results.get(f, {}).get("mean_ic", None)
        if f_ic is not None:
            forced_ics.append((f, f_ic))
    if forced_ics:
        forced_ics.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"    Top forced features by IC:")
        for f, ic in forced_ics[:10]:
            print(f"      {f:>35}: IC={ic:+.4f}")
    
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
# MODEL TRAINING (same as step6c but uses curated features)
# ════════════════════════════════════════════════════════════

def train_lambdarank(X_train, y_train, groups_train,
                     X_val, y_val, groups_val,
                     feature_names=None, tier="large"):
    """Train LambdaRank LightGBM model."""
    import lightgbm as lgb
    
    tier_params = {
        "mega": {
            "num_leaves": 31, "min_child_samples": 100,
            "max_depth": 5, "learning_rate": 0.015,
            "feature_fraction": 0.7, "bagging_fraction": 0.8,
            "lambdarank_truncation_level": 50,
            "num_boost_round": 800,
        },
        "large": {
            "num_leaves": 47, "min_child_samples": 200,
            "max_depth": 6, "learning_rate": 0.015,
            "feature_fraction": 0.6, "bagging_fraction": 0.8,
            "lambdarank_truncation_level": 100,
            "num_boost_round": 900,
        },
        "mid": {
            "num_leaves": 63, "min_child_samples": 300,
            "max_depth": 7, "learning_rate": 0.015,
            "feature_fraction": 0.5, "bagging_fraction": 0.8,
            "lambdarank_truncation_level": 100,
            "num_boost_round": 900,
        },
        "small": {
            "num_leaves": 63, "min_child_samples": 500,
            "max_depth": 7, "learning_rate": 0.015,
            "feature_fraction": 0.5, "bagging_fraction": 0.7,
            "lambdarank_truncation_level": 200,
            "num_boost_round": 900,
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
        "lambda_l1": 0.1,
        "lambda_l2": 5.0,
        "min_child_samples": tp["min_child_samples"],
        "max_depth": tp["max_depth"],
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
        "max_bin": 255,
        "min_gain_to_split": 0.01,
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
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    
    return model, model.best_iteration


def train_huber_lgb(X_train, y_train, X_val, y_val,
                    feature_names=None, tier="large"):
    """Fallback Huber LGB for ensembling."""
    import lightgbm as lgb
    
    params = {
        "objective": "huber", "alpha": 0.9,
        "boosting_type": "gbdt", "num_leaves": 47,
        "learning_rate": 0.015, "feature_fraction": 0.6,
        "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l1": 0.1, "lambda_l2": 5.0,
        "min_child_samples": 300, "max_depth": 6,
        "verbose": -1, "n_jobs": -1, "seed": 42,
        "max_bin": 255, "min_gain_to_split": 0.01,
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names,
                         free_raw_data=True)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)
    
    model = lgb.train(
        params, dtrain, num_boost_round=800,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model, model.best_iteration


# ════════════════════════════════════════════════════════════
# METRICS
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
    """Estimate tradeable capacity: top-Q median DDV × 5% participation."""
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


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    np.random.seed(42)

    print("=" * 70)
    print("CURATED FEATURE MULTI-UNIVERSE (Prompt 2)")
    print("=" * 70)
    print("  Per-tier curated features + LambdaRank + Huber ensemble")
    print("  Academic alpha factors force-included")
    print("  Macro interaction noise removed")
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

    # Market cap for tier assignment
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

    # ── Walk-forward ──
    test_start = 2005
    test_end = int(panel["year"].max())
    train_window = 10

    tier_results = {}

    for tier_name, tier_def in TIER_DEFS.items():
        tier_t0 = time.time()
        print(f"\n{'=' * 70}")
        print(f"TIER: {tier_def['label']}")
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

        # ── FEATURE SELECTION (the key Prompt 2 innovation) ──
        print(f"\n  ── Feature Selection for {tier_name} ──")
        budget = TIER_FEATURE_BUDGET[tier_name]

        # Use a sample of the training data for IC screening
        ic_sample = tier_panel[tier_panel["year"].between(2000, 2020)]
        if len(ic_sample) < 5000:
            ic_sample = tier_panel[tier_panel["year"] >= 1995]

        tier_features, ic_results = select_features_for_tier(
            ic_sample, tier_name, all_feature_cols, target_col, budget=budget
        )

        print(f"\n  Using {len(tier_features)} curated features (was 533)")
        print(f"  Feature list: {tier_features[:10]}...")

        # Rank-normalize ONLY the curated features
        print(f"  Rank-normalizing {len(tier_features)} features within {tier_name} tier...")
        t_rank = time.time()
        rank_normalize_features(tier_panel, tier_features)
        print(f"  Rank normalization: {time.time() - t_rank:.0f}s")

        # Create labels
        tier_panel["relevance"] = create_relevance_labels(tier_panel, target_col, n_bins=5)
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
            train_end_year = test_year - 1  # 6-month embargo handled via month filtering

            train_mask = (tier_panel["year"] >= train_start_year) & \
                         (tier_panel["year"] <= train_end_year)
            # 6-month embargo: remove last 6 months of training
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

            # Validation: last 20% of training
            n_val = max(1, int(len(X_train) * 0.2))
            X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
            y_tr_rel, y_val_rel = y_train_rel[:-n_val], y_train_rel[-n_val:]

            train_dates_arr = train_df["date"].values
            tr_dates = train_dates_arr[:-n_val]
            val_dates = train_dates_arr[-n_val:]

            # Recompute groups for split
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
                # Use rank-transform before blending
                lr_rank = stats.rankdata(lr_pred) / len(lr_pred)
                hu_rank = stats.rankdata(hu_pred) / len(hu_pred)

                # Compute val IC for weights
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

            # Compute test IC
            test_ic, _ = stats.spearmanr(combined, actual)

            # Store predictions
            pred_df = test_df[["permno", "date"]].copy()
            pred_df["prediction"] = combined
            pred_df["tier"] = tier_name
            pred_df[target_col] = actual
            all_preds.append(pred_df)

            # Feature importance (from Huber model, more interpretable)
            if hu_pred is not None:
                imp = hu_model.feature_importance(importance_type="gain")
                imp_norm = imp / imp.sum() * 100
                all_importances.append(dict(zip(tier_features, imp_norm)))

            # Report
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

        tier_time = time.time() - (t_start if tier_name == "mega" else tier_t0)

        # Print tier summary box
        print(f"\n  ╔══════════════════════════════════════════════════╗")
        print(f"  ║{tier_def['label']:^50}║")
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
        print(f"  ╚══════════════════════════════════════════════════╝")

        # Quantile returns
        qr = compute_quantile_returns(preds, "prediction", target_col)
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

            print(f"\n  Top 10 features:")
            top10_total = sum(v for _, v in top_feats[:10])
            for f, v in top_feats[:10]:
                print(f"    {f:>45} {v:.2f}%")
            print(f"    {'Top-10 concentration:':>45} {top10_total:.1f}%")

            # Save feature importance
            fi_df = pd.DataFrame(top_feats, columns=["feature", "importance"])
            fi_df.to_csv(os.path.join(RESULTS_DIR,
                                      f"feature_importance_curated_{tier_name}.csv"),
                         index=False)

        # Save predictions
        preds.to_parquet(os.path.join(DATA_DIR,
                                      f"curated_predictions_{tier_name}.parquet"),
                         index=False)

        # Save tier metrics
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
        }

    # ══════════════════════════════════════════════════════════
    # COMBINED SUMMARY
    # ══════════════════════════════════════════════════════════

    total_time = (time.time() - t_start) / 60

    print(f"\n\n{'═' * 70}")
    print("CURATED FEATURE MULTI-UNIVERSE — RESULTS SUMMARY")
    print(f"{'═' * 70}")
    print(f"{'Tier':<12} {'IC':>8} {'IR':>8} {'Mono':>8} {'Sharpe':>8} "
          f"{'Capacity':>12} {'#Feats':>8} {'N months':>10}")
    print("─" * 70)

    for tier_name in ["mega", "large", "mid", "small"]:
        if tier_name not in tier_results:
            continue
        r = tier_results[tier_name]
        print(f"{tier_name:<12} {r['ic']:>+8.4f} {r['ir']:>8.2f} {r['monotonicity']:>8.3f} "
              f"{r['ls_sharpe']:>8.2f} {r['capacity_usd']:>11,.0f} "
              f"{r['n_features']:>8} {r['n_months']:>10}")

    print("─" * 70)
    print(f"Total time: {total_time:.1f} min")

    # Compare with step6c baseline
    print(f"\n  vs step6c BASELINE:")
    print(f"  {'Metric':<25} {'step6c':>10} {'step6d':>10} {'Delta':>10}")
    print(f"  {'─' * 55}")

    baseline = {
        "mega":  {"ic": 0.0095, "mono": 0.504, "sharpe": 0.13},
        "large": {"ic": -0.0084, "mono": 0.495, "sharpe": -0.22},
        "mid":   {"ic": 0.0255, "mono": 0.514, "sharpe": 0.68},
        "small": {"ic": 0.0729, "mono": 0.547, "sharpe": 1.02},
    }

    for tier_name in ["mega", "large", "mid", "small"]:
        if tier_name not in tier_results:
            continue
        r = tier_results[tier_name]
        b = baseline[tier_name]
        print(f"  {tier_name + ' IC':<25} {b['ic']:>+10.4f} {r['ic']:>+10.4f} "
              f"{r['ic'] - b['ic']:>+10.4f}")
        print(f"  {tier_name + ' Mono':<25} {b['mono']:>10.3f} {r['monotonicity']:>10.3f} "
              f"{r['monotonicity'] - b['mono']:>+10.3f}")
        print(f"  {tier_name + ' Sharpe':<25} {b['sharpe']:>10.2f} {r['ls_sharpe']:>10.2f} "
              f"{r['ls_sharpe'] - b['sharpe']:>+10.2f}")
        print()

    # Save combined predictions
    all_tier_preds = []
    for tier_name in ["mega", "large", "mid", "small"]:
        fp = os.path.join(DATA_DIR, f"curated_predictions_{tier_name}.parquet")
        if os.path.exists(fp):
            all_tier_preds.append(pd.read_parquet(fp))

    if all_tier_preds:
        combined = pd.concat(all_tier_preds, ignore_index=True)
        combined.to_parquet(os.path.join(DATA_DIR,
                                        "curated_multi_universe_predictions.parquet"),
                            index=False)
        print(f"  Combined predictions: {len(combined):,} rows")

    # Save summary
    summary = {
        "method": "curated_features_multi_universe",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_min": round(total_time, 1),
        "tiers": {k: {kk: vv for kk, vv in v.items() if kk != "features"}
                  for k, v in tier_results.items()},
        "feature_lists": {k: v.get("features", []) for k, v in tier_results.items()},
    }
    with open(os.path.join(RESULTS_DIR, "curated_multi_universe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary: {os.path.join(RESULTS_DIR, 'curated_multi_universe_summary.json')}")

    print(f"\n{'═' * 70}")
    print(f"DONE. Total time: {total_time:.1f} min")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
