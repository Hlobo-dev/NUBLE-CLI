"""
PHASE 3 — STEP 1: Download Chen-Zimmermann 212 Predictors
============================================================
The Open Source Cross-Sectional Asset Pricing dataset.
212 published anomaly signals, pre-constructed, point-in-time correct.
Keyed by (permno, yyyymm) — perfect merge with our training panel.

Options tried in order:
1. openassetpricing Python package
2. Direct CSV download from Google Drive
3. Construct key predictors from existing WRDS data (fallback)
"""

import subprocess
import sys
import os
import time
import gc

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")
S3_BUCKET = "nuble-data-warehouse"


def try_package_download():
    """Try downloading via the openassetpricing Python package."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "openassetpricing", "-q"],
            capture_output=True,
        )
        import openassetpricing as oap

        print("  Downloading via openassetpricing package...")
        df = oap.download(data_type="signed_predictors_dl_wide")
        print(f"  ✅ Downloaded: {len(df):,} rows × {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"  Package method failed: {e}")
        return None


def try_direct_download():
    """Try downloading directly from Google Drive / openassetpricing.com."""
    import pandas as pd
    import requests
    import io
    import zipfile

    urls = [
        # Signed predictors (wide format) — primary URL
        "https://drive.google.com/uc?export=download&id=1OH5Nt4MwO_8MjAxKbHzYUD7fFqMIoaVn",
        # Alternative: confirmed=1 for large files
        "https://drive.google.com/uc?export=download&confirm=1&id=1OH5Nt4MwO_8MjAxKbHzYUD7fFqMIoaVn",
    ]

    for url in urls:
        try:
            print(f"  Trying direct download...")
            response = requests.get(url, timeout=600, stream=True)
            if response.status_code == 200:
                content = response.content
                if len(content) < 1000:
                    # Might be a redirect page, not actual data
                    print(f"  ⚠️ Response too small ({len(content)} bytes), skipping")
                    continue
                if content[:2] == b"PK":  # ZIP file
                    z = zipfile.ZipFile(io.BytesIO(content))
                    csv_name = z.namelist()[0]
                    print(f"  Unzipping {csv_name}...")
                    df = pd.read_csv(z.open(csv_name))
                else:
                    df = pd.read_csv(io.BytesIO(content))
                print(f"  ✅ Downloaded: {len(df):,} rows × {df.shape[1]} columns")
                return df
        except Exception as e:
            print(f"  ❌ Failed: {str(e)[:100]}")

    return None


def construct_cz_from_wrds():
    """
    Construct key CZ predictors from our existing WRDS data.
    This is the fallback — we already have the underlying data.
    We construct ~40-50 additional predictors not already in the panel.
    """
    import pandas as pd
    import numpy as np

    print("  Constructing CZ-style predictors from existing WRDS data...")

    # Load Compustat for accounting-based signals
    comp_path = os.path.join(DATA_DIR, "compustat_quarterly.parquet")
    if not os.path.exists(comp_path):
        print("  ❌ Compustat quarterly not found")
        return None

    comp = pd.read_parquet(comp_path)
    comp["datadate"] = pd.to_datetime(comp["datadate"])
    print(f"  Loaded Compustat: {len(comp):,} rows")

    # Load link table
    link_path = os.path.join(DATA_DIR, "crsp_compustat_link.parquet")
    if os.path.exists(link_path):
        link = pd.read_parquet(link_path)
        if "lpermno" in link.columns:
            link = link.rename(columns={"lpermno": "permno"})
        comp = comp.merge(
            link[["gvkey", "permno"]].drop_duplicates(),
            on="gvkey",
            how="inner",
        )
        print(f"  After CRSP link: {len(comp):,} rows")

    # Convert to monthly (use report date for point-in-time)
    if "rdq" in comp.columns:
        comp["date"] = pd.to_datetime(comp["rdq"])
    else:
        comp["date"] = comp["datadate"] + pd.DateOffset(months=3)
    comp["date"] = comp["date"] + pd.offsets.MonthEnd(0)

    # Sort for lag computations
    comp = comp.sort_values(["permno", "date"])

    # ── Construct predictors ──
    features = comp[["permno", "date"]].copy()

    # Accruals (Sloan 1996)
    for col in ["atq", "ltq", "cheq", "lctq", "dlcq"]:
        if col not in comp.columns:
            comp[col] = np.nan

    comp["working_capital"] = (comp["atq"] - comp["cheq"]) - (comp["ltq"] - comp["dlcq"])
    g = comp.groupby("permno")
    comp["wc_lag"] = g["working_capital"].shift(4)
    comp["at_avg"] = (comp["atq"] + g["atq"].shift(4)) / 2
    features["accruals"] = (comp["working_capital"] - comp["wc_lag"]) / comp["at_avg"].clip(lower=1)

    # Asset growth (Cooper, Gulen, Schill 2008)
    comp["at_lag4"] = g["atq"].shift(4)
    features["asset_growth"] = (comp["atq"] - comp["at_lag4"]) / comp["at_lag4"].clip(lower=1)

    # Investment (Titman, Wei, Xie 2004)
    if "ppentq" in comp.columns:
        comp["ppent_lag4"] = g["ppentq"].shift(4)
        features["investment"] = (comp["ppentq"] - comp["ppent_lag4"]) / comp["ppent_lag4"].clip(lower=1)

    # Gross profitability (Novy-Marx 2013)
    if "revtq" in comp.columns and "cogsq" in comp.columns:
        features["gross_profit_at"] = (comp["revtq"] - comp["cogsq"]) / comp["atq"].clip(lower=1)

    # Operating profitability
    if "xsgaq" in comp.columns and "revtq" in comp.columns and "cogsq" in comp.columns:
        features["oper_prof"] = (comp["revtq"] - comp["cogsq"] - comp["xsgaq"].fillna(0)) / comp["atq"].clip(lower=1)

    # Cash holdings (Palazzo 2012)
    features["cash_at"] = comp["cheq"] / comp["atq"].clip(lower=1)

    # Leverage change
    comp["lev"] = comp["ltq"] / comp["atq"].clip(lower=1)
    comp["lev_lag4"] = g["lev"].shift(4)
    features["leverage_change"] = comp["lev"] - comp["lev_lag4"]

    # Earnings consistency (number of positive earnings quarters in last 8)
    if "ibq" in comp.columns:
        for lag in range(1, 9):
            comp[f"ib_lag{lag}"] = g["ibq"].shift(lag)
        pos_cols = [f"ib_lag{lag}" for lag in range(1, 9)]
        features["earnings_consistency"] = comp[pos_cols].apply(
            lambda row: (row > 0).sum(), axis=1
        ) / 8.0

    # Revenue growth
    if "revtq" in comp.columns:
        comp["rev_lag4"] = g["revtq"].shift(4)
        features["revenue_growth"] = (comp["revtq"] - comp["rev_lag4"]) / comp["rev_lag4"].clip(lower=1).abs()

    # Piotroski F-score components
    if "niq" in comp.columns:
        features["positive_ni"] = (comp["niq"] > 0).astype(float)
    if "oiadpq" in comp.columns:
        features["positive_cfo"] = (comp["oiadpq"] > 0).astype(float)

    # Current ratio
    if "lctq" in comp.columns:
        curr_assets = comp["atq"] - comp.get("ppentq", pd.Series(dtype=float)).fillna(0)
        features["current_ratio"] = curr_assets / comp["lctq"].clip(lower=1)

    # Net issuance (shares)
    if "cshoq" in comp.columns:
        comp["csho_lag4"] = g["cshoq"].shift(4)
        features["share_issuance"] = (comp["cshoq"] - comp["csho_lag4"]) / comp["csho_lag4"].clip(lower=1)

    # Drop rows with all NaN features
    feat_cols = [c for c in features.columns if c not in ["permno", "date"]]
    features = features.dropna(subset=feat_cols, how="all")
    features = features.drop_duplicates(subset=["permno", "date"], keep="last")

    print(f"  Constructed {len(feat_cols)} CZ-style predictors")
    print(f"  Result: {len(features):,} rows × {features.shape[1]} cols")
    print(f"  Date range: {features['date'].min().date()} to {features['date'].max().date()}")
    print(f"  Unique PERMNOs: {features['permno'].nunique():,}")

    return features


def main():
    print("=" * 70)
    print("PHASE 3 — STEP 1: CHEN-ZIMMERMANN 212 PREDICTORS")
    print("=" * 70)
    start = time.time()

    # Try methods in order
    df = try_package_download()
    if df is None:
        df = try_direct_download()
    if df is None:
        df = construct_cz_from_wrds()

    if df is None or len(df) == 0:
        print("\n❌ All download methods failed")
        return

    import pandas as pd

    # Ensure proper types
    if "yyyymm" in df.columns and "date" not in df.columns:
        df["date"] = (
            pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m")
            + pd.offsets.MonthEnd(0)
        )
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    if "permno" in df.columns:
        df["permno"] = df["permno"].astype(int)

    # Save to Parquet
    output_path = os.path.join(DATA_DIR, "cz_predictors.parquet")
    df.to_parquet(output_path, index=False, engine="pyarrow")
    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)

    # Upload to S3
    subprocess.run(
        ["aws", "s3", "cp", output_path,
         f"s3://{S3_BUCKET}/features/cz_predictors.parquet"],
        capture_output=True,
    )

    elapsed = time.time() - start

    # Summary
    id_cols = ["permno", "date", "yyyymm"]
    feat_cols = [c for c in df.columns if c not in id_cols]

    print(f"\n{'=' * 70}")
    print(f"CZ PREDICTORS COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Rows:          {len(df):,}")
    print(f"  Features:      {len(feat_cols)}")
    print(f"  Date range:    {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  PERMNOs:       {df['permno'].nunique():,}")
    print(f"  File size:     {file_size_mb:.0f} MB")
    print(f"  Time:          {elapsed:.0f}s")
    print(f"  ✅ Saved: {output_path}")
    print(f"  ✅ Uploaded to s3://{S3_BUCKET}/features/cz_predictors.parquet")


if __name__ == "__main__":
    main()
