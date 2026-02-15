"""
PHASE 2 — STEP 5: Merge Daily Features into Training Panel
============================================================
Loads the training panel from Step 3, merges daily-derived features
from Step 4, applies winsorization, and re-saves.

Input:  data/wrds/training_panel.parquet + data/wrds/daily_features.parquet
Output: data/wrds/training_panel.parquet (updated in place)
"""

import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")

PANEL_PATH = "data/wrds/training_panel.parquet"
DAILY_PATH = "data/wrds/daily_features.parquet"


def merge_daily_features():
    print("=" * 70)
    print("PHASE 2 — STEP 5: MERGE DAILY FEATURES + WINSORIZE")
    print("=" * 70)
    start_time = time.time()

    # Load training panel
    print("\n[1/4] Loading training panel...")
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    print(f"  Panel: {len(panel):,} rows × {panel.shape[1]} cols")

    # Load daily features
    print("\n[2/4] Loading daily features...")
    if not os.path.exists(DAILY_PATH):
        print(f"  ⚠️  {DAILY_PATH} not found — run phase2_step4_daily_features.py first")
        print("  Skipping daily feature merge, proceeding with winsorization...")
    else:
        daily = pd.read_parquet(DAILY_PATH)
        daily["date"] = pd.to_datetime(daily["date"])

        # Select feature columns (exclude permno, date, num_trading_days, mean_daily_ret)
        daily_feature_cols = [
            "max_daily_ret", "min_daily_ret", "realized_vol", "amihud_illiq",
            "zero_vol_days", "intraday_range", "return_skewness", "return_kurtosis",
            "down_vol", "up_vol"
        ]
        daily_feature_cols = [c for c in daily_feature_cols if c in daily.columns]

        daily_merge = daily[["permno", "date"] + daily_feature_cols].copy()
        daily_merge = daily_merge.drop_duplicates(subset=["permno", "date"], keep="last")

        # Merge into panel
        before_cols = panel.shape[1]
        panel = panel.merge(daily_merge, on=["permno", "date"], how="left")
        after_cols = panel.shape[1]
        daily_match = panel["max_daily_ret"].notna().mean() if "max_daily_ret" in panel.columns else 0

        print(f"  + Daily features: {before_cols} → {after_cols} cols | "
              f"Match: {daily_match:.1%}")

    # ─── Winsorization ───
    print("\n[3/4] Winsorizing features at 1st/99th percentile...")

    # Identify numeric feature columns (exclude identifiers and labels)
    exclude_cols = [
        "permno", "date", "ret", "market_cap",
        "fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m",
        "excess_ret_1m", "rf",
        "exchcd", "shrcd", "siccd", "sp500_member", "dlstcd",
        "n_months", "num_trading_days"
    ]
    feature_cols = [c for c in panel.columns
                    if c not in exclude_cols
                    and panel[c].dtype in ["float64", "float32", "int64", "int32"]]

    winsorized_count = 0
    for col in feature_cols:
        if panel[col].notna().sum() < 100:
            continue
        lo = panel[col].quantile(0.01)
        hi = panel[col].quantile(0.99)
        if lo < hi:
            before = panel[col].copy()
            panel[col] = panel[col].clip(lower=lo, upper=hi)
            changed = (panel[col] != before).sum()
            if changed > 0:
                winsorized_count += 1

    print(f"  Winsorized {winsorized_count} features")

    # ─── Save ───
    print("\n[4/4] Saving updated training panel...")
    panel.to_parquet(PANEL_PATH, index=False, engine="pyarrow")
    file_size_gb = os.path.getsize(PANEL_PATH) / (1024 ** 3)

    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"MERGE + WINSORIZE COMPLETE!")
    print(f"{'='*70}")
    print(f"  Final rows:    {len(panel):,}")
    print(f"  Final columns: {panel.shape[1]}")
    print(f"  File size:     {file_size_gb:.2f} GB")
    print(f"  Time:          {total_time/60:.1f} minutes")

    return panel


if __name__ == "__main__":
    merge_daily_features()
