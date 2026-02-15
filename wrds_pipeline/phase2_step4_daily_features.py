"""
PHASE 2 — STEP 4: Daily-Derived Monthly Features
==================================================
Computes high-frequency features from CRSP daily stock data,
aggregated to monthly frequency for the training panel.

Input:  data/wrds/crsp_daily/crsp_daily_{year}.parquet
Output: data/wrds/daily_features.parquet

Features computed per (permno, month):
  1. max_daily_ret     — Maximum single-day return (lottery demand, Bali 2011)
  2. min_daily_ret     — Minimum single-day return (crash risk)
  3. realized_vol      — Std of daily returns × sqrt(252)
  4. amihud_illiq      — Mean(|ret| / dollar_volume) × 1e6 (Amihud 2002)
  5. zero_vol_days     — Fraction of days with zero volume (liquidity proxy)
  6. intraday_range    — Mean((askhi-bidlo)/abs(prc)) (Parkinson vol proxy)
  7. return_skewness   — Skewness of daily returns
  8. return_kurtosis   — Kurtosis of daily returns
  9. down_vol          — Volatility of negative-return days only
  10. up_vol           — Volatility of positive-return days only

Expected:
  - ~2-5M (permno × month) observations
  - Date range: 1926-2024
  - Runtime: 30-90 minutes
"""

import pandas as pd
import numpy as np
import os
import gc
import time
import glob
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

DAILY_DIR = "data/wrds/crsp_daily"
OUTPUT_PATH = "data/wrds/daily_features.parquet"


def compute_monthly_features(df):
    """Compute all daily-derived monthly features for one year of daily data.
    
    Optimized: avoids slow .apply() and lambda-based .agg() by using
    vectorized groupby operations and separate aggregations.
    """
    df = df.copy()
    for col in ["ret", "vol", "prc", "shrout", "bidlo", "askhi"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Dollar volume = |price| × volume (in shares)
    df["abs_prc"] = np.abs(df["prc"])
    df["dollar_vol"] = df["abs_prc"] * df["vol"]
    df["abs_ret"] = np.abs(df["ret"])

    # Year-month for grouping
    df["ym"] = df["date"].dt.to_period("M")
    gkeys = ["permno", "ym"]

    # ---- Fast aggregation: built-in functions only ----
    grouped = df.groupby(gkeys)["ret"]
    results = pd.DataFrame({
        "max_daily_ret": grouped.max(),
        "min_daily_ret": grouped.min(),
        "mean_daily_ret": grouped.mean(),
        "ret_std": grouped.std(),
        "num_trading_days": grouped.count(),
    })
    results["realized_vol"] = np.where(
        results["num_trading_days"] >= 5,
        results["ret_std"] * np.sqrt(252),
        np.nan
    )
    results = results.drop(columns=["ret_std"]).reset_index()

    # ---- Skewness & Kurtosis (vectorized via apply on small groups) ----
    def _skew_kurt(x):
        x = x.dropna()
        if len(x) < 5:
            return pd.Series({"return_skewness": np.nan, "return_kurtosis": np.nan})
        return pd.Series({
            "return_skewness": stats.skew(x),
            "return_kurtosis": stats.kurtosis(x),
        })
    sk = df.groupby(gkeys)["ret"].apply(_skew_kurt).unstack().reset_index()
    results = results.merge(sk, on=gkeys, how="left")

    # ---- Zero-volume days ----
    vol_g = df.groupby(gkeys)["vol"]
    zero_vol = pd.DataFrame({
        "zero_count": vol_g.apply(lambda x: (x == 0).sum()),
        "total_count": vol_g.count(),
    }).reset_index()
    zero_vol["zero_vol_days"] = zero_vol["zero_count"] / zero_vol["total_count"]
    results = results.merge(zero_vol[gkeys + ["zero_vol_days"]], on=gkeys, how="left")

    # ---- Amihud illiquidity: mean(|ret| / dollar_volume) × 10^6 ----
    valid_dv = df[df["dollar_vol"] > 0].copy()
    valid_dv["amihud_ratio"] = valid_dv["abs_ret"] / valid_dv["dollar_vol"]
    amihud_g = valid_dv.groupby(gkeys)
    amihud = pd.DataFrame({
        "amihud_illiq": amihud_g["amihud_ratio"].mean() * 1e6,
        "amihud_count": amihud_g["amihud_ratio"].count(),
    }).reset_index()
    amihud.loc[amihud["amihud_count"] < 5, "amihud_illiq"] = np.nan
    results = results.merge(amihud[gkeys + ["amihud_illiq"]], on=gkeys, how="left")

    # ---- Intraday range: mean((askhi - bidlo) / |prc|) ----
    valid_range = df[(df["askhi"] > 0) & (df["bidlo"] > 0) & (df["abs_prc"] > 0)].copy()
    if len(valid_range) > 0:
        valid_range["daily_range"] = (valid_range["askhi"] - valid_range["bidlo"]) / valid_range["abs_prc"]
        intraday = valid_range.groupby(gkeys)["daily_range"].mean().reset_index(name="intraday_range")
        results = results.merge(intraday, on=gkeys, how="left")
    else:
        results["intraday_range"] = np.nan

    # ---- Down-vol and up-vol ----
    down_df = df[df["ret"] < 0]
    down_g = down_df.groupby(gkeys)["ret"]
    down = pd.DataFrame({
        "down_vol_std": down_g.std(),
        "down_count": down_g.count(),
    }).reset_index()
    down["down_vol"] = np.where(down["down_count"] >= 3, down["down_vol_std"] * np.sqrt(252), np.nan)
    results = results.merge(down[gkeys + ["down_vol"]], on=gkeys, how="left")

    up_df = df[df["ret"] > 0]
    up_g = up_df.groupby(gkeys)["ret"]
    up = pd.DataFrame({
        "up_vol_std": up_g.std(),
        "up_count": up_g.count(),
    }).reset_index()
    up["up_vol"] = np.where(up["up_count"] >= 3, up["up_vol_std"] * np.sqrt(252), np.nan)
    results = results.merge(up[gkeys + ["up_vol"]], on=gkeys, how="left")

    # Convert period to end-of-month date
    results["date"] = results["ym"].dt.to_timestamp("M") + pd.offsets.MonthEnd(0)
    results = results.drop(columns=["ym"])

    return results


def main():
    print("=" * 70)
    print("PHASE 2 — STEP 4: DAILY-DERIVED MONTHLY FEATURES")
    print("=" * 70)
    start_time = time.time()

    # Find all daily parquet files
    pattern = os.path.join(DAILY_DIR, "crsp_daily_*.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"❌ No daily parquet files found in {DAILY_DIR}/")
        print("   Run phase2_step1_crsp_daily.py first!")
        return None

    print(f"Found {len(files)} yearly daily files")

    all_features = []
    total_daily_rows = 0

    for idx, filepath in enumerate(files):
        year = os.path.basename(filepath).replace("crsp_daily_", "").replace(".parquet", "")
        year_start = time.time()

        try:
            # Load one year of daily data
            df = pd.read_parquet(filepath, columns=[
                "permno", "date", "ret", "vol", "prc", "shrout", "bidlo", "askhi"
            ])
            df["date"] = pd.to_datetime(df["date"])
            daily_rows = len(df)
            total_daily_rows += daily_rows

            # Compute monthly features
            features = compute_monthly_features(df)
            all_features.append(features)

            year_time = time.time() - year_start
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 1
            remaining = (len(files) - idx - 1) / rate / 60

            print(f"  [{idx+1}/{len(files)}] {year}: "
                  f"{daily_rows:>10,} daily → {len(features):>7,} monthly | "
                  f"{year_time:.1f}s | ETA: {remaining:.0f}m")

            del df
            gc.collect()

        except Exception as e:
            print(f"  [{idx+1}/{len(files)}] {year}: ❌ ERROR: {str(e)[:80]}")

    if not all_features:
        print("❌ No features computed!")
        return None

    # Combine all years
    print(f"\nCombining {len(all_features)} years...")
    result = pd.concat(all_features, ignore_index=True)
    result = result.sort_values(["permno", "date"]).reset_index(drop=True)
    result = result.drop_duplicates(subset=["permno", "date"], keep="last")

    # Save
    print(f"Saving to {OUTPUT_PATH}...")
    result.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)

    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"DAILY FEATURES COMPLETE!")
    print(f"{'='*70}")
    print(f"  Total daily rows processed: {total_daily_rows:,}")
    print(f"  Monthly feature rows:       {len(result):,}")
    print(f"  Unique PERMNOs:             {result['permno'].nunique():,}")
    print(f"  Date range:                 {result['date'].min().date()} to {result['date'].max().date()}")
    print(f"  File size:                  {file_size_mb:.1f} MB")
    print(f"  Time:                       {total_time/60:.1f} minutes")

    # Sanity checks
    print(f"\n  Feature means (sanity):")
    for col in ["max_daily_ret", "realized_vol", "amihud_illiq", "return_skewness",
                "return_kurtosis", "zero_vol_days", "intraday_range"]:
        if col in result.columns:
            val = result[col].mean()
            print(f"    {col:<25} mean={val:.4f}")

    return result


if __name__ == "__main__":
    main()
