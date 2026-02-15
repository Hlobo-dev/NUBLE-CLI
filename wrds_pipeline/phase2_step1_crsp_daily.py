"""
PHASE 2 — STEP 1: CRSP Daily Stock Data Download
=================================================
Downloads ALL CRSP daily stock data (1926-2024) from WRDS to local Parquet.
~107M rows, ~20 columns, stored as yearly Parquet files for memory management.

Output:
  data/wrds/crsp_daily/crsp_daily_{year}.parquet  (one file per year)
  data/wrds/crsp_daily_manifest.json              (download manifest)

Expected:
  - 107M+ total rows
  - 1926-2024 (99 years)
  - 30K+ unique PERMNOs
  - ~8-12 GB total Parquet on disk
  - Runtime: 2-4 hours depending on WRDS load

IDEMPOTENT: Safe to re-run. Skips years already downloaded.
"""

import wrds
import pandas as pd
import numpy as np
import os
import gc
import json
import time
from datetime import datetime

# ─── Configuration ───
WRDS_USERNAME = "hlobo"
OUTPUT_DIR = "data/wrds/crsp_daily"
MANIFEST_PATH = "data/wrds/crsp_daily_manifest.json"
START_YEAR = 1926
END_YEAR = 2024

# Columns to download from crsp.dsf
COLUMNS = [
    "permno", "permco", "date", "cusip",
    "bidlo", "askhi", "prc", "vol", "ret", "retx",
    "shrout", "openprc", "numtrd",
    "hexcd", "hsiccd",
    "cfacpr", "cfacshr"
]


def load_manifest():
    """Load download manifest (tracks which years are done)."""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {"completed_years": {}, "total_rows": 0, "started": datetime.now().isoformat()}


def save_manifest(manifest):
    """Save download manifest."""
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def download_year(db, year):
    """Download one year of CRSP daily data."""
    query = f"""
        SELECT {', '.join(COLUMNS)}
        FROM crsp.dsf
        WHERE date >= '{year}-01-01'
          AND date <= '{year}-12-31'
        ORDER BY permno, date
    """
    df = db.raw_sql(query, date_cols=["date"])
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = load_manifest()
    completed = set(manifest["completed_years"].keys())

    print("=" * 70)
    print("PHASE 2 — STEP 1: CRSP DAILY DOWNLOAD")
    print("=" * 70)
    print(f"Source:      crsp.dsf (WRDS)")
    print(f"Destination: {OUTPUT_DIR}/ (local Parquet)")
    print(f"Range:       {START_YEAR}-{END_YEAR}")
    print(f"Already:     {len(completed)} years completed")
    print()

    # Determine years to download
    all_years = list(range(START_YEAR, END_YEAR + 1))
    pending_years = [y for y in all_years if str(y) not in completed]

    if not pending_years:
        print("✅ All years already downloaded!")
        print(f"   Total rows: {manifest['total_rows']:,}")
        return

    print(f"Years to download: {len(pending_years)}")
    print(f"  First: {pending_years[0]}, Last: {pending_years[-1]}")
    print()

    # Connect to WRDS
    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    print("Connected!\n")

    total_start = time.time()
    total_rows_this_run = 0
    errors = []

    for idx, year in enumerate(pending_years):
        year_start = time.time()
        output_path = os.path.join(OUTPUT_DIR, f"crsp_daily_{year}.parquet")

        try:
            print(f"[{idx+1}/{len(pending_years)}] Downloading {year}...", end=" ", flush=True)

            df = download_year(db, year)
            rows = len(df)

            if rows == 0:
                print(f"⚠️  0 rows (skipping)")
                manifest["completed_years"][str(year)] = {
                    "rows": 0,
                    "timestamp": datetime.now().isoformat()
                }
                save_manifest(manifest)
                continue

            # Data quality: ensure numeric types
            for col in ["ret", "retx", "prc", "vol", "shrout", "bidlo", "askhi", "openprc"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Save to parquet
            df.to_parquet(output_path, index=False, engine="pyarrow")
            file_size_mb = os.path.getsize(output_path) / (1024 ** 2)

            # Update manifest
            manifest["completed_years"][str(year)] = {
                "rows": rows,
                "file_size_mb": round(file_size_mb, 1),
                "permnos": int(df["permno"].nunique()),
                "date_min": str(df["date"].min().date()),
                "date_max": str(df["date"].max().date()),
                "timestamp": datetime.now().isoformat()
            }
            manifest["total_rows"] = sum(
                v["rows"] for v in manifest["completed_years"].values()
            )
            save_manifest(manifest)

            year_time = time.time() - year_start
            total_rows_this_run += rows
            elapsed = time.time() - total_start
            remaining = len(pending_years) - (idx + 1)
            eta_mins = (elapsed / (idx + 1) * remaining) / 60 if idx > 0 else 0

            print(
                f"✅ {rows:>10,} rows | "
                f"{df['permno'].nunique():>5,} stocks | "
                f"{file_size_mb:>6.1f} MB | "
                f"{year_time:.0f}s | "
                f"ETA: {eta_mins:.0f}m"
            )

            # Memory cleanup
            del df
            gc.collect()

        except Exception as e:
            year_time = time.time() - year_start
            error_msg = f"{year}: {str(e)[:100]}"
            errors.append(error_msg)
            print(f"❌ ERROR ({year_time:.0f}s): {str(e)[:80]}")

            # Try to reconnect if connection lost
            try:
                db.close()
            except:
                pass
            try:
                print("  Reconnecting to WRDS...", end=" ", flush=True)
                db = wrds.Connection(wrds_username=WRDS_USERNAME)
                print("OK")
            except Exception as e2:
                print(f"FAILED: {e2}")
                break

    # Close connection
    try:
        db.close()
    except:
        pass

    # Final summary
    total_time = time.time() - total_start
    total_completed = len(manifest["completed_years"])

    print("\n" + "=" * 70)
    print("CRSP DAILY DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Years completed:     {total_completed}/{len(all_years)}")
    print(f"Rows this run:       {total_rows_this_run:,}")
    print(f"Total rows all time: {manifest['total_rows']:,}")
    print(f"Time:                {total_time/60:.1f} minutes")
    print(f"Errors:              {len(errors)}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  ❌ {e}")

    # Disk usage
    total_size_mb = sum(
        v.get("file_size_mb", 0) for v in manifest["completed_years"].values()
    )
    print(f"\nDisk usage:          {total_size_mb/1024:.2f} GB")

    # Verification
    total_permnos = set()
    for year_str, info in manifest["completed_years"].items():
        total_permnos_count = info.get("permnos", 0)

    print(f"\n{'='*70}")
    if total_completed == len(all_years):
        print("✅ CRSP DAILY DOWNLOAD COMPLETE!")
    else:
        print(f"⚠️  {len(all_years) - total_completed} years remaining. Re-run to continue.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
