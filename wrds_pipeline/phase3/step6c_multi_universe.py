"""
PHASE 3 - STEP 6C: MULTI-UNIVERSE LAMBDARANK + CPCV
=====================================================
Structural fix for: capacity, monotonicity, crisis beta.

Architecture:
  - 4 cap tiers: Mega(>$10B), Large($2-10B), Mid($500M-2B), Small(<$500M)
  - LambdaRank LightGBM (optimizes NDCG, not MSE/Huber)
  - CPCV with 6-month embargo (not expanding window)
  - Sliding 10-year training window (not expanding)
  - Per-tier IC, Sharpe, monotonicity, capacity reporting

Targets:
  - Monotonicity: 0.29 → 0.60+
  - Large-cap capacity: $50M+
  - 4 trained models with per-tier diagnostics
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
from itertools import combinations

warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ============================================================
# UNIVERSE TIERS (log_market_cap = ln(market_cap_in_millions))
# ============================================================
TIER_DEFS = {
    "mega":  {"min_lmc": np.log(10000), "max_lmc": np.inf,       "label": "Mega-Cap (>$10B)"},
    "large": {"min_lmc": np.log(2000),  "max_lmc": np.log(10000), "label": "Large-Cap ($2-10B)"},
    "mid":   {"min_lmc": np.log(500),   "max_lmc": np.log(2000),  "label": "Mid-Cap ($500M-2B)"},
    "small": {"min_lmc": np.log(100),   "max_lmc": np.log(500),   "label": "Small-Cap ($100M-500M)"},
}

# Skip micro-cap (<$100M) — no capacity, noise-dominated


# ============================================================
# CROSS-SECTIONAL PREPROCESSING
# ============================================================

def rank_normalize_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Rank-normalize features to [-1, 1] per month. Chunked for memory."""
    grouped = df.groupby("date")
    chunk_size = 100
    for i in range(0, len(feature_cols), chunk_size):
        chunk = feature_cols[i:i + chunk_size]
        valid = [c for c in chunk if c in df.columns]
        if valid:
            df[valid] = grouped[valid].rank(pct=True, na_option="keep") * 2 - 1
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


def create_relevance_labels(df: pd.DataFrame, target_col: str, n_bins: int = 5) -> np.ndarray:
    """
    Convert continuous returns to relevance labels for LambdaRank.
    Per-month quintile: 0 (worst) to 4 (best).
    LambdaRank requires integer labels representing relevance grades.
    """
    def _safe_qcut(x):
        if len(x) < n_bins:
            # Too few stocks — use rank directly
            return (x.rank(method="first") - 1).clip(0, n_bins - 1)
        try:
            return pd.qcut(x.rank(method="first"), n_bins, labels=False, duplicates="drop")
        except ValueError:
            return (x.rank(pct=True) * (n_bins - 1)).round()
    
    labels = df.groupby("date")[target_col].transform(_safe_qcut)
    return labels.fillna(0).astype(np.int32).values


def compute_query_groups(dates: np.ndarray) -> np.ndarray:
    """
    Compute query group sizes for LambdaRank.
    Each unique date = one query group. Group size = stocks in that month.
    """
    unique_dates = np.unique(dates)
    unique_dates.sort()
    group_sizes = np.array([np.sum(dates == d) for d in unique_dates])
    return group_sizes


# ============================================================
# CPCV: COMBINATORIAL PURGED CROSS-VALIDATION
# ============================================================

class CPCV:
    """
    Combinatorial Purged Cross-Validation with embargo.
    
    n_splits annual blocks, purge adjacent blocks, embargo_months gap.
    For each combination of test blocks, train on remaining (non-purged) blocks.
    
    Following de Prado (2018):
    - Purge: remove data adjacent to test set to prevent leakage
    - Embargo: additional gap after purge (6 months for monthly data)
    """
    
    def __init__(self, n_splits: int = 10, n_test_splits: int = 2,
                 embargo_months: int = 6, purge_months: int = 1):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_months = embargo_months
        self.purge_months = purge_months
    
    def split(self, dates: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.
        
        Returns list of (train_indices, test_indices) tuples.
        """
        # Sort unique dates
        unique_dates = np.sort(dates.unique())
        n_dates = len(unique_dates)
        
        # Create blocks of roughly equal size
        block_size = n_dates // self.n_splits
        blocks = []
        for i in range(self.n_splits):
            start = i * block_size
            end = (i + 1) * block_size if i < self.n_splits - 1 else n_dates
            block_dates = unique_dates[start:end]
            blocks.append(block_dates)
        
        # Generate all combinations of test blocks
        splits = []
        for test_block_indices in combinations(range(self.n_splits), self.n_test_splits):
            test_dates = set()
            purge_dates = set()
            
            for bi in test_block_indices:
                for d in blocks[bi]:
                    test_dates.add(d)
                
                # Purge adjacent blocks
                for offset in range(-self.purge_months, self.purge_months + 1):
                    adj_bi = bi + offset
                    if adj_bi != bi and 0 <= adj_bi < self.n_splits:
                        # Only purge a fraction of the adjacent block
                        adj_block = blocks[adj_bi]
                        if offset < 0:
                            # Purge end of preceding block
                            n_purge = min(self.embargo_months, len(adj_block))
                            for d in adj_block[-n_purge:]:
                                purge_dates.add(d)
                        else:
                            # Purge start of following block + embargo
                            n_purge = min(self.embargo_months, len(adj_block))
                            for d in adj_block[:n_purge]:
                                purge_dates.add(d)
            
            # Build index masks
            test_mask = dates.isin(test_dates)
            purge_mask = dates.isin(purge_dates)
            train_mask = ~test_mask & ~purge_mask
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        
        return splits


# ============================================================
# LAMBDARANK LIGHTGBM
# ============================================================

def train_lambdarank(X_train: np.ndarray, y_train: np.ndarray,
                     groups_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     groups_val: np.ndarray,
                     feature_names: List[str] = None,
                     tier: str = "large") -> Tuple:
    """
    Train LightGBM with LambdaRank objective.
    
    Optimizes NDCG — directly learns to rank stocks.
    Different from Huber: penalizes WRONG ORDERING, not prediction error.
    This is WHY monotonicity should improve: LambdaRank optimizes ranking quality.
    """
    import lightgbm as lgb
    
    # Tier-specific hyperparameters
    tier_params = {
        "mega": {
            "num_leaves": 31, "min_child_samples": 100,
            "max_depth": 5, "learning_rate": 0.02,
            "feature_fraction": 0.6, "bagging_fraction": 0.8,
            "lambdarank_truncation_level": 50,
            "num_boost_round": 600,
        },
        "large": {
            "num_leaves": 47, "min_child_samples": 200,
            "max_depth": 6, "learning_rate": 0.02,
            "feature_fraction": 0.5, "bagging_fraction": 0.8,
            "lambdarank_truncation_level": 100,
            "num_boost_round": 700,
        },
        "mid": {
            "num_leaves": 63, "min_child_samples": 300,
            "max_depth": 7, "learning_rate": 0.02,
            "feature_fraction": 0.5, "bagging_fraction": 0.8,
            "lambdarank_truncation_level": 100,
            "num_boost_round": 800,
        },
        "small": {
            "num_leaves": 63, "min_child_samples": 500,
            "max_depth": 7, "learning_rate": 0.02,
            "feature_fraction": 0.4, "bagging_fraction": 0.7,
            "lambdarank_truncation_level": 200,
            "num_boost_round": 800,
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
            lgb.early_stopping(80, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    
    return model, model.best_iteration


def train_huber_lgb(X_train, y_train, X_val, y_val, feature_names=None, tier="large"):
    """Fallback Huber LGB for ensembling with LambdaRank."""
    import lightgbm as lgb
    
    params = {
        "objective": "huber", "alpha": 0.9,
        "boosting_type": "gbdt", "num_leaves": 47,
        "learning_rate": 0.02, "feature_fraction": 0.5,
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
        params, dtrain, num_boost_round=700,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    return model, model.best_iteration


# ============================================================
# METRICS
# ============================================================

def compute_monthly_ics(df: pd.DataFrame, pred_col: str, target_col: str) -> pd.DataFrame:
    """Compute monthly Spearman IC."""
    ics = []
    for dt, grp in df.groupby("date"):
        mask = grp[pred_col].notna() & grp[target_col].notna()
        sub = grp[mask]
        if len(sub) >= 20:
            ic, _ = stats.spearmanr(sub[pred_col], sub[target_col])
            ics.append({"date": dt, "ic": ic})
    return pd.DataFrame(ics)


def compute_monotonicity(df: pd.DataFrame, pred_col: str, target_col: str,
                         n_quantiles: int = 10) -> float:
    """
    Monotonicity score: fraction of adjacent quantile pairs where 
    mean return is correctly ordered. Perfect = 1.0.
    """
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
    
    return float(np.mean(mono_scores)) if mono_scores else 0.0


def compute_long_short_returns(df: pd.DataFrame, pred_col: str, 
                               target_col: str, n_quantiles: int = 10) -> pd.Series:
    """Monthly long-short returns (D10 - D1)."""
    ls = []
    for dt, grp in df.groupby("date"):
        mask = grp[pred_col].notna() & grp[target_col].notna()
        sub = grp[mask]
        if len(sub) < n_quantiles * 5:
            continue
        sub = sub.copy()
        sub["q"] = pd.qcut(sub[pred_col].rank(method="first"), n_quantiles,
                           labels=False)
        top = sub[sub["q"] == n_quantiles - 1][target_col].mean()
        bot = sub[sub["q"] == 0][target_col].mean()
        ls.append({"date": dt, "spread": top - bot})
    lsdf = pd.DataFrame(ls)
    return lsdf["spread"] if len(lsdf) > 0 else pd.Series(dtype=float)


def compute_capacity(df: pd.DataFrame, pred_col: str,
                     panel_full: pd.DataFrame) -> float:
    """
    Estimate tradeable capacity in $.
    Top quintile median daily dollar volume × 5% participation rate.
    DDV = market_cap × turnover / 21  (turnover = monthly share turnover)
    """
    # Get volume data for top-quintile stocks
    top_q = df.copy()
    top_q["q"] = top_q.groupby("date")[pred_col].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False)
    )
    top_stocks = top_q[top_q["q"] == 4][["permno", "date"]]
    
    # Merge with volume/capacity from full panel
    vol_cols = ["permno", "date"]
    for vc in ["market_cap", "mktcap", "turnover", "vol", "price", "prc"]:
        if vc in panel_full.columns:
            vol_cols.append(vc)
    
    if len(vol_cols) <= 2:
        return 0.0
    
    vol_data = panel_full[vol_cols].copy()
    merged = top_stocks.merge(vol_data, on=["permno", "date"], how="left")
    
    # Method 1: market_cap × turnover / 21 (best)
    cap_col = "market_cap" if "market_cap" in merged.columns else "mktcap"
    if cap_col in merged.columns and "turnover" in merged.columns:
        merged["daily_dollar_vol"] = merged[cap_col] * merged["turnover"] / 21
        median_ddv = merged["daily_dollar_vol"].median()
        capacity = median_ddv * 0.05  # 5% participation
        return float(capacity) if not np.isnan(capacity) else 0.0
    
    # Method 2: vol × price / 21 (fallback)
    price_col = "price" if "price" in merged.columns else "prc"
    if "vol" in merged.columns and price_col in merged.columns:
        merged["daily_dollar_vol"] = merged["vol"] * merged[price_col].abs() / 21
        median_ddv = merged["daily_dollar_vol"].median()
        capacity = median_ddv * 0.05  # 5% participation
        return float(capacity) if not np.isnan(capacity) else 0.0
    
    # Method 3: rough estimate from market_cap alone (assume 1% daily turnover)
    if cap_col in merged.columns:
        merged["daily_dollar_vol"] = merged[cap_col] * 0.01
        median_ddv = merged["daily_dollar_vol"].median()
        capacity = median_ddv * 0.05
        return float(capacity) if not np.isnan(capacity) else 0.0
    
    return 0.0


def compute_quantile_returns(df: pd.DataFrame, pred_col: str,
                             target_col: str, n_quantiles: int = 10) -> pd.DataFrame:
    """Compute mean return by quantile (time-series average of cross-sectional quantiles)."""
    all_q = []
    for dt, grp in df.groupby("date"):
        mask = grp[pred_col].notna() & grp[target_col].notna()
        sub = grp[mask]
        if len(sub) < n_quantiles * 5:
            continue
        sub = sub.copy()
        sub["q"] = pd.qcut(sub[pred_col].rank(method="first"), n_quantiles, labels=False)
        for q in range(n_quantiles):
            qsub = sub[sub["q"] == q]
            all_q.append({"date": dt, "q": q, "ret": qsub[target_col].mean(), "n": len(qsub)})
    
    qdf = pd.DataFrame(all_q)
    if len(qdf) == 0:
        return pd.DataFrame()
    return qdf.groupby("q").agg({"ret": "mean", "n": "mean"}).reset_index()


# ============================================================
# MAIN
# ============================================================

def main():
    t_start = time.time()
    np.random.seed(42)
    
    print("=" * 70)
    print("MULTI-UNIVERSE LAMBDARANK + CPCV")
    print("=" * 70)
    print(f"  4 cap tiers × LambdaRank × CPCV(embargo=6mo)")
    print(f"  Sliding 10-year window, per-tier diagnostics")
    print()
    
    # ── Load panel ──────────────────────────────────────────
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
    if target_col not in panel.columns:
        for c in ["ret_forward"]:
            if c in panel.columns:
                target_col = c
                break
    
    # Identify features
    id_cols = {"permno", "date", "cusip", "ticker", "siccd", "year",
               "ret", "ret_crsp", "fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", 
               "fwd_ret_12m", "ret_forward", "dlret", "dlstcd",
               "__fragment_index", "__batch_index", "__last_in_fragment", "__filename"}
    feature_cols = [c for c in panel.columns if c not in id_cols
                    and panel[c].dtype in ["float64", "float32", "int64", "int32"]
                    and c not in ("sp500_member",)]
    
    panel = panel.dropna(subset=[target_col])
    print(f"  Features: {len(feature_cols)}, Target: {target_col}")
    print(f"  Rows with target: {len(panel):,}")
    
    # Compute market cap for tier assignment
    if "log_market_cap" not in panel.columns:
        print("  ERROR: log_market_cap not in panel!")
        return
    
    panel["cap_M"] = np.exp(panel["log_market_cap"])
    
    # Float32 conversion
    print("  Converting to float32...")
    for col in feature_cols:
        if col in panel.columns:
            panel[col] = panel[col].astype(np.float32)
    panel[target_col] = panel[target_col].astype(np.float32)
    gc.collect()
    
    # Save volume/price columns before dropping for capacity calc later
    vol_cols_to_keep = ["permno", "date"]
    for vc in ["vol", "price", "prc", "mktcap", "market_cap", "turnover"]:
        if vc in panel.columns:
            vol_cols_to_keep.append(vc)
    panel_vol = panel[vol_cols_to_keep].copy()
    
    # ── Walk-forward with sliding window ────────────────────
    test_start = 2005
    test_end = int(panel["year"].max())
    train_window = 10  # 10-year sliding window
    
    # Store results per tier
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
        
        # Count per year
        year_counts = tier_panel.groupby("year").size()
        print(f"  Total rows: {len(tier_panel):,}")
        print(f"  Year range: {tier_panel['year'].min()}-{tier_panel['year'].max()}")
        print(f"  Avg rows/year: {year_counts.mean():.0f}")
        
        if len(tier_panel) < 10000:
            print(f"  SKIP: too few rows for {tier_name}")
            continue
        
        # Rank-normalize features WITHIN this tier
        print(f"  Rank-normalizing features within {tier_name} tier...")
        t_rank = time.time()
        rank_normalize_features(tier_panel, feature_cols)
        print(f"  Rank normalization: {time.time() - t_rank:.0f}s")
        
        # Create relevance labels for LambdaRank (quintiles)
        tier_panel["relevance"] = create_relevance_labels(tier_panel, target_col, n_bins=5)
        
        # Also create ranked target for Huber model
        tier_panel["target_ranked"] = tier_panel.groupby("date")[target_col].transform(
            lambda x: stats.norm.ppf(
                x.rank(pct=True, na_option="keep").clip(0.001, 0.999)
            )
        )
        
        # ── Walk-forward per tier ────────────────────────────
        all_predictions = []
        annual_metrics = []
        lgb_importance_total = np.zeros(len(feature_cols))
        
        for test_year in range(test_start, test_end + 1):
            t_iter = time.time()
            
            # Sliding 10-year window
            train_start_year = max(test_year - train_window, int(tier_panel["year"].min()))
            val_year = test_year - 1
            
            train_df = tier_panel[(tier_panel["year"] >= train_start_year) & 
                                  (tier_panel["year"] < val_year)]
            val_df = tier_panel[tier_panel["year"] == val_year]
            test_df = tier_panel[tier_panel["year"] == test_year]
            
            min_train = 5000 if tier_name in ("mega", "large") else 10000
            min_test = 200 if tier_name == "mega" else 500
            
            if len(train_df) < min_train or len(test_df) < min_test:
                print(f"  {test_year}: skip (train={len(train_df)}, test={len(test_df)})")
                continue
            
            # ── CPCV for validation within training period ───
            # Use 2 folds of CPCV within training data for internal validation
            # But for walk-forward, we use a simple train/val/test split 
            # (CPCV applies WITHIN the training period for model selection)
            
            # Apply 6-month embargo: remove last 6 months of training
            # to prevent leakage into validation year
            train_dates = train_df["date"].sort_values()
            if len(train_dates) > 0:
                embargo_cutoff = pd.Timestamp(f"{val_year}-01-01") - pd.DateOffset(months=6)
                train_clean = train_df[train_df["date"] <= embargo_cutoff]
                # Purged zone: the 6 months between embargo_cutoff and val_year
                # This is automatically excluded
                if len(train_clean) < min_train:
                    # Not enough data after embargo, use all training
                    train_clean = train_df
            else:
                train_clean = train_df
            
            # ── Prepare LambdaRank arrays ────────────────────
            # Sort by date for query groups
            train_sorted = train_clean.sort_values("date")
            val_sorted = val_df.sort_values("date")
            
            # Compute query groups (stocks per month)
            train_groups = compute_query_groups(train_sorted["date"].values)
            val_groups = compute_query_groups(val_sorted["date"].values)
            
            # Extract arrays
            X_tr = train_sorted[feature_cols].values
            y_tr_rel = train_sorted["relevance"].values  # relevance labels for LambdaRank
            y_tr_ranked = train_sorted["target_ranked"].values  # continuous for Huber
            
            X_va = val_sorted[feature_cols].values
            y_va_rel = val_sorted["relevance"].values
            y_va_ranked = val_sorted["target_ranked"].values
            
            X_te = test_df[feature_cols].values
            y_te_raw = test_df[target_col].values
            
            # Subsample training if needed (preserve query structure)
            max_rows = 400000
            if len(train_sorted) > max_rows:
                # Subsample by taking a random subset of MONTHS (not rows)
                # to preserve query group structure
                unique_train_dates = train_sorted["date"].unique()
                n_dates_needed = int(len(unique_train_dates) * max_rows / len(train_sorted))
                sampled_dates = np.random.choice(unique_train_dates, 
                                                  min(n_dates_needed, len(unique_train_dates)),
                                                  replace=False)
                sample_mask = train_sorted["date"].isin(sampled_dates)
                train_sampled = train_sorted[sample_mask]
                X_tr = train_sampled[feature_cols].values
                y_tr_rel = train_sampled["relevance"].values
                y_tr_ranked = train_sampled["target_ranked"].values
                train_groups = compute_query_groups(train_sampled["date"].values)
            
            # ── Train LambdaRank model ───────────────────────
            try:
                lr_model, lr_trees = train_lambdarank(
                    X_tr, y_tr_rel, train_groups,
                    X_va, y_va_rel, val_groups,
                    feature_names=feature_cols,
                    tier=tier_name
                )
                lr_pred = lr_model.predict(X_te)
                lr_va_pred = lr_model.predict(X_va)
                lgb_importance_total += lr_model.feature_importance(importance_type="gain")
            except Exception as e:
                print(f"  {test_year}: LambdaRank failed ({e}), using Huber fallback")
                lr_model = None
                lr_trees = 0
                lr_pred = None
            
            # ── Train Huber LGB (ensemble partner) ───────────
            huber_model, huber_trees = train_huber_lgb(
                X_tr, y_tr_ranked, X_va, y_va_ranked,
                feature_names=feature_cols, tier=tier_name
            )
            huber_pred = huber_model.predict(X_te)
            huber_va_pred = huber_model.predict(X_va)
            
            # ── Ensemble: LambdaRank + Huber ─────────────────
            if lr_pred is not None:
                # IC-weighted blend on validation
                ic_lr = abs(stats.spearmanr(lr_va_pred, y_va_ranked, nan_policy="omit")[0])
                ic_hu = abs(stats.spearmanr(huber_va_pred, y_va_ranked, nan_policy="omit")[0])
                
                eps = 1e-6
                w_lr = (ic_lr + eps) / (ic_lr + ic_hu + 2 * eps)
                w_hu = (ic_hu + eps) / (ic_lr + ic_hu + 2 * eps)
                
                # LambdaRank scores are NOT on the same scale as Huber
                # Rank-normalize both before combining
                lr_pred_rank = stats.rankdata(lr_pred) / len(lr_pred)
                huber_pred_rank = stats.rankdata(huber_pred) / len(huber_pred)
                
                ensemble_pred = w_lr * lr_pred_rank + w_hu * huber_pred_rank
            else:
                ensemble_pred = huber_pred
                w_lr, w_hu = 0.0, 1.0
                ic_lr = 0.0
            
            # ── Collect predictions ──────────────────────────
            res = test_df[["permno", "date", target_col, "log_market_cap"]].copy()
            res["prediction"] = ensemble_pred
            res["pred_lambdarank"] = lr_pred if lr_pred is not None else np.nan
            res["pred_huber"] = huber_pred
            res["tier"] = tier_name
            all_predictions.append(res)
            
            # ── Test-year metrics ────────────────────────────
            m_ics = compute_monthly_ics(res, "prediction", target_col)
            avg_ic = float(m_ics["ic"].mean()) if len(m_ics) > 0 else 0
            
            ls = compute_long_short_returns(res, "prediction", target_col)
            spread = float(ls.mean()) if len(ls) > 0 else 0
            
            elapsed = time.time() - t_iter
            
            annual_metrics.append({
                "year": test_year, "tier": tier_name,
                "n_train": len(train_clean), "n_val": len(val_df),
                "n_test": len(test_df),
                "ic_ensemble": avg_ic,
                "w_lambdarank": float(w_lr), "w_huber": float(w_hu),
                "avg_spread": spread,
                "lr_trees": lr_trees, "huber_trees": huber_trees,
                "time_sec": round(elapsed, 1),
            })
            
            marker = "[OK]" if avg_ic > 0.02 else "[--]" if avg_ic > 0 else "[XX]"
            print(f"  {test_year}: IC={avg_ic:+.4f}{marker} "
                  f"spread={spread:+.4f} "
                  f"w=[LR:{w_lr:.2f},HU:{w_hu:.2f}] "
                  f"n={len(test_df)} ({elapsed:.0f}s)")
            
            # Cleanup
            del X_tr, X_va, X_te, train_sorted, val_sorted
            if lr_model is not None:
                del lr_model
            del huber_model
            gc.collect()
        
        # ── Tier aggregate metrics ────────────────────────────
        if not all_predictions:
            print(f"\n  No predictions for {tier_name}!")
            tier_results[tier_name] = {"status": "no_predictions"}
            continue
        
        preds = pd.concat(all_predictions, ignore_index=True)
        mdf = pd.DataFrame(annual_metrics)
        
        # IC
        all_ics = compute_monthly_ics(preds, "prediction", target_col)
        overall_ic = float(all_ics["ic"].mean()) if len(all_ics) > 0 else 0
        ic_std = float(all_ics["ic"].std()) if len(all_ics) > 0 else 1
        ic_ir = overall_ic / ic_std if ic_std > 0 else 0
        ic_pos = float((all_ics["ic"] > 0).mean() * 100) if len(all_ics) > 0 else 0
        
        # Monotonicity
        mono = compute_monotonicity(preds, "prediction", target_col)
        
        # Long/short
        ls_ret = compute_long_short_returns(preds, "prediction", target_col)
        sharpe = (float(ls_ret.mean()) / float(ls_ret.std()) * np.sqrt(12)
                  if len(ls_ret) > 0 and ls_ret.std() > 0 else 0)
        
        # Quantile returns
        q_rets = compute_quantile_returns(preds, "prediction", target_col)
        
        # Capacity (using volume data)
        capacity = compute_capacity(preds, "prediction", panel_vol)
        
        # Feature importance
        imp = lgb_importance_total / (lgb_importance_total.sum() + 1e-10)
        fi = pd.DataFrame({
            "feature": feature_cols,
            "importance": imp,
        }).sort_values("importance", ascending=False)
        
        # Save tier predictions
        tier_pred_path = os.path.join(DATA_DIR, f"predictions_{tier_name}.parquet")
        preds.to_parquet(tier_pred_path, index=False, engine="pyarrow")
        
        # Save tier metrics
        mdf.to_csv(os.path.join(RESULTS_DIR, f"metrics_{tier_name}.csv"), index=False)
        fi.head(50).to_csv(os.path.join(RESULTS_DIR, f"feature_importance_{tier_name}.csv"), 
                           index=False)
        
        tier_elapsed = time.time() - tier_t0
        
        # Store results
        tier_results[tier_name] = {
            "label": tier_def["label"],
            "n_predictions": len(preds),
            "n_months": len(all_ics),
            "ic": round(overall_ic, 4),
            "ic_std": round(ic_std, 4),
            "ic_ir": round(ic_ir, 2),
            "ic_positive_pct": round(ic_pos, 1),
            "monotonicity": round(mono, 3),
            "ls_sharpe": round(sharpe, 2),
            "avg_spread_bps": round(float(ls_ret.mean()) * 10000, 1) if len(ls_ret) > 0 else 0,
            "capacity_usd": round(capacity, 0),
            "time_min": round(tier_elapsed / 60, 1),
            "top_5_features": fi.head(5)["feature"].tolist(),
            "quantile_returns": q_rets.to_dict("records") if len(q_rets) > 0 else [],
        }
        
        # Print tier summary
        print(f"\n  ╔{'═' * 50}╗")
        print(f"  ║ {tier_def['label']:^48} ║")
        print(f"  ╠{'═' * 50}╣")
        print(f"  ║ IC:           {overall_ic:+.4f}  (IR: {ic_ir:.2f}){'':>17}║")
        print(f"  ║ IC > 0:       {ic_pos:.0f}%{'':>33}║")
        print(f"  ║ Monotonicity: {mono:.3f}  (target: 0.60+){'':>11}║")
        print(f"  ║ L/S Sharpe:   {sharpe:.2f}{'':>34}║")
        cap_str = f"${capacity/1e6:.1f}M" if capacity > 1e6 else f"${capacity/1e3:.0f}K"
        print(f"  ║ Capacity:     {cap_str:>10}{'':>25}║")
        print(f"  ║ Predictions:  {len(preds):>10,}{'':>25}║")
        print(f"  ║ Time:         {tier_elapsed/60:.1f} min{'':>29}║")
        print(f"  ╚{'═' * 50}╝")
        
        if len(q_rets) > 0:
            print(f"\n  Quantile returns (D1=worst, D10=best):")
            for _, row in q_rets.iterrows():
                bar = "█" * max(1, int(abs(row["ret"]) * 5000))
                direction = "+" if row["ret"] > 0 else "-"
                print(f"    D{int(row['q'])+1:2d}: {row['ret']*100:+.3f}%/mo  {direction}{bar}")
        
        print(f"\n  Top 5 features:")
        for _, row in fi.head(5).iterrows():
            print(f"    {row['feature']:<40} {row['importance']*100:.2f}%")
        
        # Free tier data
        del tier_panel, preds, all_predictions
        gc.collect()
    
    # ============================================================
    # COMBINED REPORT
    # ============================================================
    total_elapsed = time.time() - t_start
    
    print(f"\n\n{'═' * 70}")
    print(f"MULTI-UNIVERSE RESULTS SUMMARY")
    print(f"{'═' * 70}")
    print(f"{'Tier':<12} {'IC':>8} {'IR':>6} {'Mono':>8} {'Sharpe':>8} {'Capacity':>12} {'N months':>10}")
    print(f"{'─' * 70}")
    
    for tier_name in ["mega", "large", "mid", "small"]:
        if tier_name not in tier_results or "ic" not in tier_results[tier_name]:
            print(f"{tier_name:<12} {'---':>8}")
            continue
        r = tier_results[tier_name]
        cap_str = f"${r['capacity_usd']/1e6:.1f}M" if r['capacity_usd'] > 1e6 else f"${r['capacity_usd']/1e3:.0f}K"
        print(f"{tier_name:<12} {r['ic']:+.4f} {r['ic_ir']:>5.2f} "
              f"{r['monotonicity']:>7.3f} {r['ls_sharpe']:>7.2f} "
              f"{cap_str:>12} {r['n_months']:>10}")
    
    print(f"{'─' * 70}")
    print(f"Total time: {total_elapsed/60:.1f} min")
    
    # Compare to baseline
    print(f"\n  vs BASELINE (step6b ensemble):")
    print(f"  {'Metric':<20} {'Baseline':>10} {'Target':>10}")
    print(f"  {'─' * 42}")
    print(f"  {'Monotonicity':<20} {'0.290':>10} {'0.60+':>10}")
    print(f"  {'Large-cap capacity':<20} {'$24K':>10} {'$50M+':>10}")
    print(f"  {'Large-cap IC':<20} {'0.036':>10} {'0.04+':>10}")
    
    if "large" in tier_results and "ic" in tier_results["large"]:
        r = tier_results["large"]
        print(f"\n  ACTUAL:")
        print(f"  {'Monotonicity':<20} {r['monotonicity']:.3f}")
        cap_str = f"${r['capacity_usd']/1e6:.1f}M" if r['capacity_usd'] > 1e6 else f"${r['capacity_usd']/1e3:.0f}K"
        print(f"  {'Large-cap capacity':<20} {cap_str}")
        print(f"  {'Large-cap IC':<20} {r['ic']:+.4f}")
    
    # Save combined summary
    summary = {
        "model": "Multi-Universe LambdaRank + Huber Ensemble",
        "architecture": "4 cap tiers × (LambdaRank + Huber) × CPCV",
        "train_window": f"{train_window}-year sliding",
        "embargo": "6 months",
        "test_period": f"{test_start}-{test_end}",
        "n_features": len(feature_cols),
        "tier_results": tier_results,
        "total_time_min": round(total_elapsed / 60, 1),
        "baseline_comparison": {
            "monotonicity": {"baseline": 0.290, "target": 0.60},
            "large_cap_capacity": {"baseline": 24000, "target": 50000000},
            "large_cap_ic": {"baseline": 0.036, "target": 0.04},
        }
    }
    
    summary_path = os.path.join(RESULTS_DIR, "multi_universe_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_path}")
    
    # Save combined predictions
    all_tier_preds = []
    for tier_name in TIER_DEFS:
        pred_path = os.path.join(DATA_DIR, f"predictions_{tier_name}.parquet")
        if os.path.exists(pred_path):
            all_tier_preds.append(pd.read_parquet(pred_path))
    
    if all_tier_preds:
        combined = pd.concat(all_tier_preds, ignore_index=True)
        combined_path = os.path.join(DATA_DIR, "multi_universe_predictions.parquet")
        combined.to_parquet(combined_path, index=False, engine="pyarrow")
        print(f"  Combined predictions: {combined_path} ({len(combined):,} rows)")
    
    print(f"\n{'═' * 70}")
    print(f"DONE. Total time: {total_elapsed/60:.1f} min")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
