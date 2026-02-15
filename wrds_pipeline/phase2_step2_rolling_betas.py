"""
PHASE 2 — STEP 2: Rolling Factor Betas
=======================================
Computes 60-month rolling FF5+UMD factor betas for every stock-month.

Input:  data/wrds/crsp_monthly.parquet + data/wrds/ff_factors_monthly.parquet
Output: data/wrds/rolling_betas.parquet

For each (permno, month) with at least 36 months of history:
  - Regress excess stock return on [Mkt-RF, SMB, HML, RMW, CMA, UMD]
  - Store: alpha, beta_mkt, beta_smb, beta_hml, beta_rmw, beta_cma, beta_umd,
           idio_vol, total_vol, r_squared, n_months

Expected:
  - ~2-4M stock-month observations
  - Date range: ~1931-2024 (60 months after earliest data)
  - Runtime: 15-40 minutes
"""

import pandas as pd
import numpy as np
import time
import gc
import os
import warnings
warnings.filterwarnings("ignore")

OUTPUT_PATH = "data/wrds/rolling_betas.parquet"
WINDOW = 60       # Rolling window in months
MIN_OBS = 36      # Minimum observations required


def compute_rolling_betas():
    print("=" * 70)
    print("PHASE 2 — STEP 2: ROLLING FACTOR BETAS")
    print("=" * 70)
    start_time = time.time()

    # ─── Load CRSP Monthly ───
    print("\n[1/4] Loading CRSP monthly returns...")
    crsp = pd.read_parquet("data/wrds/crsp_monthly.parquet",
                           columns=["permno", "date", "ret"])
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp["date"] = crsp["date"] + pd.offsets.MonthEnd(0)
    crsp = crsp.dropna(subset=["ret"])
    crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
    crsp = crsp.dropna(subset=["ret"])
    print(f"  {len(crsp):,} rows | {crsp['permno'].nunique():,} stocks | "
          f"{crsp['date'].min().date()} to {crsp['date'].max().date()}")

    # ─── Load FF Factors ───
    print("\n[2/4] Loading Fama-French factors...")
    ff = pd.read_parquet("data/wrds/ff_factors_monthly.parquet")
    ff["date"] = pd.to_datetime(ff["date"])
    ff["date"] = ff["date"] + pd.offsets.MonthEnd(0)

    # FF factors from WRDS are in percentages — convert to decimals
    factor_cols = ["mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"]
    for col in factor_cols:
        if col in ff.columns:
            ff[col] = pd.to_numeric(ff[col], errors="coerce")
            # If values look like percentages (abs mean > 1), divide by 100
            if ff[col].abs().mean() > 0.5:
                ff[col] = ff[col] / 100.0

    print(f"  {len(ff):,} months | {ff['date'].min().date()} to {ff['date'].max().date()}")
    print(f"  Mkt-RF mean: {ff['mktrf'].mean():.4f} (should be ~0.005-0.008/month)")

    # ─── Merge ───
    print("\n[3/4] Merging and computing betas...")
    merged = crsp.merge(ff[["date", "mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"]],
                        on="date", how="inner")
    merged["excess_ret"] = merged["ret"] - merged["rf"]
    merged = merged.sort_values(["permno", "date"]).reset_index(drop=True)

    print(f"  Merged: {len(merged):,} stock-month obs")

    # ─── Rolling OLS per stock ───
    print(f"\n[4/4] Computing {WINDOW}-month rolling betas (min {MIN_OBS} obs)...")

    factor_names = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
    all_results = []
    permnos = merged["permno"].unique()
    n_permnos = len(permnos)

    batch_start = time.time()
    processed = 0

    for i, perm in enumerate(permnos):
        sub = merged[merged["permno"] == perm].sort_values("date")

        if len(sub) < MIN_OBS:
            continue

        dates = sub["date"].values
        y = sub["excess_ret"].values
        X = sub[factor_names].values

        # Rolling window
        for j in range(MIN_OBS, len(sub) + 1):
            start_idx = max(0, j - WINDOW)
            y_win = y[start_idx:j]
            X_win = X[start_idx:j]
            n = len(y_win)

            if n < MIN_OBS:
                continue

            # Check for NaN
            valid = ~(np.isnan(y_win) | np.isnan(X_win).any(axis=1))
            if valid.sum() < MIN_OBS:
                continue

            y_clean = y_win[valid]
            X_clean = X_win[valid]
            n_valid = len(y_clean)

            # Add intercept
            X_with_const = np.column_stack([np.ones(n_valid), X_clean])

            try:
                result, residuals, _, _ = np.linalg.lstsq(X_with_const, y_clean, rcond=None)
                alpha = result[0]
                betas = result[1:]

                # Predicted and residual
                y_pred = X_with_const @ result
                resid = y_clean - y_pred

                # Volatilities (annualized)
                idio_vol = np.std(resid) * np.sqrt(12)
                total_vol = np.std(y_clean) * np.sqrt(12)

                # R-squared
                ss_res = np.sum(resid ** 2)
                ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

                all_results.append({
                    "permno": perm,
                    "date": dates[j - 1],
                    "alpha": float(alpha),
                    "beta_mkt": float(betas[0]),
                    "beta_smb": float(betas[1]),
                    "beta_hml": float(betas[2]),
                    "beta_rmw": float(betas[3]),
                    "beta_cma": float(betas[4]),
                    "beta_umd": float(betas[5]),
                    "idio_vol": float(idio_vol),
                    "total_vol": float(total_vol),
                    "r_squared": float(r_squared),
                    "n_months": n_valid,
                })
            except Exception:
                continue

        processed += 1
        if (i + 1) % 2000 == 0 or i == n_permnos - 1:
            elapsed = time.time() - batch_start
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (n_permnos - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  Progress: {i+1:>6,}/{n_permnos:,} stocks | "
                  f"{len(all_results):>8,} beta obs | "
                  f"{rate:.0f} stocks/s | "
                  f"ETA: {remaining:.0f}m")

    # ─── Save ───
    print(f"\n  Building DataFrame from {len(all_results):,} observations...")
    df_betas = pd.DataFrame(all_results)
    df_betas["date"] = pd.to_datetime(df_betas["date"])
    df_betas = df_betas.sort_values(["permno", "date"]).reset_index(drop=True)

    print(f"  Saving to {OUTPUT_PATH}...")
    df_betas.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)

    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"ROLLING BETAS COMPLETE!")
    print(f"{'='*70}")
    print(f"  Rows:          {len(df_betas):,}")
    print(f"  Unique stocks: {df_betas['permno'].nunique():,}")
    print(f"  Date range:    {df_betas['date'].min().date()} to {df_betas['date'].max().date()}")
    print(f"  File size:     {file_size_mb:.1f} MB")
    print(f"  Time:          {total_time/60:.1f} minutes")

    # Sanity checks
    print(f"\n  Sanity checks:")
    print(f"    beta_mkt mean:  {df_betas['beta_mkt'].mean():.3f} (should be ~1.0)")
    print(f"    beta_mkt std:   {df_betas['beta_mkt'].std():.3f}")
    print(f"    idio_vol mean:  {df_betas['idio_vol'].mean():.3f} (should be ~0.3-0.5)")
    print(f"    r_squared mean: {df_betas['r_squared'].mean():.3f}")
    print(f"    n_months mean:  {df_betas['n_months'].mean():.1f}")

    return df_betas


if __name__ == "__main__":
    compute_rolling_betas()
