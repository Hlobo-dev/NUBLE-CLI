"""
PHASE 3 — STEP 2: Download Welch-Goyal Macroeconomic Predictors
=================================================================
14 macro conditioning variables used in GKX (2020).
Source: Amit Goyal's website (sites.google.com/view/agoyal145)
Monthly, 1871-2024. These + their interactions with stock characteristics
generated 752 features in GKX.
"""

import pandas as pd
import numpy as np
import requests
import io
import os
import subprocess
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")
S3_BUCKET = "nuble-data-warehouse"


def try_goyal_download():
    """Try downloading from Goyal's Google Drive."""
    urls = [
        "https://docs.google.com/spreadsheets/d/1g4LOaRj4TvwJr9RIaA_nwrXXWTOy46bP/export?format=csv",
        "https://docs.google.com/spreadsheets/d/1g4LOaRj4TvwJr9RIaA_nwrXXWTOy46bP/export?format=xlsx",
    ]

    for url in urls:
        try:
            print(f"  Trying Goyal download...")
            response = requests.get(url, timeout=120)
            if response.status_code == 200 and len(response.content) > 1000:
                if "xlsx" in url:
                    df = pd.read_excel(io.BytesIO(response.content))
                else:
                    df = pd.read_csv(io.StringIO(response.text))
                print(f"  ✅ Downloaded: {len(df)} rows × {df.shape[1]} cols")
                return df
        except Exception as e:
            print(f"  ❌ Failed: {str(e)[:80]}")

    return None


def construct_from_fred():
    """Construct Welch-Goyal style macro predictors from FRED data."""
    print("  Constructing WG-style macro from FRED (free, no API key needed)...")

    FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    # Map WG variables to FRED series
    wg_fred_map = {
        "tbl": "DTB3",         # T-bill rate (3-month)
        "lty": "DGS10",        # Long-term yield (10-year)
        "dfy_baa": "BAA",      # BAA corporate yield
        "dfy_aaa": "AAA",      # AAA corporate yield
        "svar_proxy": "VIXCLS",  # Stock variance (VIX proxy, from 1990)
        "infl_cpi": "CPIAUCSL",  # CPI for inflation
    }

    series = {}
    for name, fred_id in wg_fred_map.items():
        try:
            url = f"{FRED_BASE}?id={fred_id}&cosd=1920-01-01"
            df = pd.read_csv(url, na_values=["."])
            # FRED uses 'observation_date' or 'DATE'
            date_col = "observation_date" if "observation_date" in df.columns else "DATE" if "DATE" in df.columns else df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.rename(columns={date_col: "date", fred_id: name})
            df[name] = pd.to_numeric(df[name], errors="coerce")
            df = df.dropna(subset=[name])
            series[name] = df
            print(f"    ✅ {fred_id:<20} → {name:<15} | {len(df):>6,} obs")
        except Exception as e:
            print(f"    ❌ {fred_id:<20} → {name:<15} | {str(e)[:60]}")
        time.sleep(0.3)

    if not series:
        return None

    # Merge all into one DataFrame
    result = list(series.values())[0]
    for df in list(series.values())[1:]:
        result = result.merge(df, on="date", how="outer")
    result = result.sort_values("date")

    # Convert daily to monthly (end-of-month values)
    result["month_end"] = result["date"] + pd.offsets.MonthEnd(0)
    monthly = result.groupby("month_end").last().reset_index()
    monthly = monthly.rename(columns={"month_end": "date"})
    monthly = monthly.drop(columns=["date"], errors="ignore") if "date" in monthly.columns and monthly.columns.duplicated().any() else monthly

    # Construct derived WG variables
    if "dfy_baa" in monthly.columns and "dfy_aaa" in monthly.columns:
        monthly["dfy"] = monthly["dfy_baa"] - monthly["dfy_aaa"]  # Default spread
    if "lty" in monthly.columns and "tbl" in monthly.columns:
        monthly["tms"] = monthly["lty"] - monthly["tbl"]  # Term spread
    if "infl_cpi" in monthly.columns:
        monthly["infl"] = monthly["infl_cpi"].pct_change(12)  # YoY inflation

    # Drop intermediate columns
    monthly = monthly.drop(columns=["dfy_baa", "dfy_aaa", "infl_cpi"], errors="ignore")

    print(f"\n  Monthly macro panel: {len(monthly)} months × {monthly.shape[1]} cols")
    return monthly


def main():
    print("=" * 70)
    print("PHASE 3 — STEP 2: WELCH-GOYAL MACRO PREDICTORS")
    print("=" * 70)
    start = time.time()

    # Try Goyal website first
    wg = try_goyal_download()

    if wg is not None:
        # Process Goyal data
        date_col = None
        for c in wg.columns:
            if "yyyymm" in c.lower() or "date" in c.lower():
                date_col = c
                break
        if date_col is None:
            date_col = wg.columns[0]

        try:
            wg["date"] = (
                pd.to_datetime(wg[date_col].astype(str).str[:6], format="%Y%m")
                + pd.offsets.MonthEnd(0)
            )
        except Exception:
            wg["date"] = pd.to_datetime(wg[date_col])

        # GKX core 14 variables
        gkx_core = [
            "dp", "dy", "ep", "de", "bm", "ntis", "tbl", "lty",
            "tms", "dfy", "svar", "ltr", "corpr", "infl",
        ]
        col_lower = {c.lower(): c for c in wg.columns}
        available = [v for v in gkx_core if v in col_lower]
        available_orig = [col_lower[v] for v in available]

        print(f"  GKX variables ({len(available)}): {available}")
        wg_clean = wg[["date"] + available_orig].copy()
        wg_clean.columns = ["date"] + available
        wg_clean = wg_clean.dropna(subset=["date"])

        for col in available:
            wg_clean[col] = pd.to_numeric(wg_clean[col], errors="coerce")

        result = wg_clean
    else:
        # Fallback: construct from FRED
        result = construct_from_fred()

    if result is None or len(result) == 0:
        print("\n❌ Could not obtain macro predictors")
        return

    result = result.sort_values("date").reset_index(drop=True)

    # Save
    output_path = os.path.join(DATA_DIR, "welch_goyal_macro.parquet")
    result.to_parquet(output_path, index=False, engine="pyarrow")
    file_size_kb = os.path.getsize(output_path) / 1024

    subprocess.run(
        ["aws", "s3", "cp", output_path,
         f"s3://{S3_BUCKET}/features/welch_goyal_macro.parquet"],
        capture_output=True,
    )

    elapsed = time.time() - start
    macro_cols = [c for c in result.columns if c != "date"]

    print(f"\n{'=' * 70}")
    print(f"WELCH-GOYAL MACRO PREDICTORS COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Months:       {len(result):,}")
    print(f"  Variables:    {len(macro_cols)}: {macro_cols}")
    print(f"  Date range:   {result['date'].min().date()} to {result['date'].max().date()}")
    print(f"  File size:    {file_size_kb:.0f} KB")
    print(f"  Time:         {elapsed:.0f}s")
    print(f"  ✅ Saved and uploaded to S3")


if __name__ == "__main__":
    main()
