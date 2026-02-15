"""
PHASE 3 — STEP 4: OptionMetrics (Attempt)
==========================================
Try downloading implied volatility surface from WRDS OptionMetrics.
Graceful failure is acceptable — many institutions don't subscribe.
If successful: monthly IV skew, term structure slope, at-the-money IV.
"""

import pandas as pd
import numpy as np
import os
import time
import subprocess

DATA_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/data/wrds"
S3_BUCKET = "nuble-data-warehouse"

WRDS_PARAMS = {
    "host": os.environ.get("WRDS_PGHOST", "wrds-pgdata.wharton.upenn.edu"),
    "port": int(os.environ.get("WRDS_PGPORT", "9737")),
    "dbname": os.environ.get("WRDS_PGDATABASE", "wrds"),
    "user": os.environ.get("WRDS_USERNAME", "hlobo"),
    "password": os.environ.get("WRDS_PASSWORD", ""),
}


def try_optionmetrics():
    """Attempt OptionMetrics download from WRDS."""
    import psycopg2

    conn = psycopg2.connect(**WRDS_PARAMS)

    # Check if we have access
    try:
        check = pd.read_sql(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'optionm' LIMIT 5",
            conn,
        )
        if len(check) == 0:
            print("  ⚠️ No OptionMetrics tables accessible (subscription may not include it)")
            conn.close()
            return None
        print(f"  Found OptionMetrics tables: {check['table_name'].tolist()}")
    except Exception as e:
        print(f"  ⚠️ OptionMetrics not accessible: {str(e)[:80]}")
        conn.close()
        return None

    # Try securd (security-level daily volatility surface)
    try:
        query = """
        SELECT secid, date, days, delta, impl_volatility, volume, open_interest
        FROM optionm.securd
        WHERE date >= '2000-01-01'
          AND delta IN (50, 25, 75)    -- ATM and wings
          AND days BETWEEN 20 AND 365  -- Near-term to 1-year
          AND cp_flag = 'C'
        ORDER BY date, secid
        """
        print("  Downloading securd (IV surface)... this may take a while...")
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"  ✅ Downloaded: {len(df):,} rows")
        return df
    except Exception as e:
        print(f"  ⚠️ securd query failed: {str(e)[:80]}")

    # Try opprcd (option pricing)
    try:
        query = """
        SELECT secid, date, exdate, cp_flag, strike_price, best_bid, best_offer,
               impl_volatility, delta, volume, open_interest
        FROM optionm.opprcd
        WHERE date >= '2010-01-01'
          AND abs(delta) BETWEEN 0.20 AND 0.80
        LIMIT 1000000
        """
        print("  Trying opprcd...")
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"  ✅ Downloaded: {len(df):,} rows")
        return df
    except Exception as e:
        print(f"  ⚠️ opprcd failed: {str(e)[:80]}")

    conn.close()
    return None


def compute_iv_features(df):
    """Compute monthly IV features from daily option data."""
    df["date"] = pd.to_datetime(df["date"])
    df["month_end"] = df["date"] + pd.offsets.MonthEnd(0)

    features = []
    for (secid, month), group in df.groupby(["secid", "month_end"]):
        atm = group[group["delta"] == 50]
        otm_put = group[group["delta"] == 25]
        otm_call = group[group["delta"] == 75]

        feat = {"secid": secid, "date": month}

        # ATM implied volatility (30-day)
        near = atm[atm["days"].between(20, 40)]
        if len(near) > 0:
            feat["iv_atm_30d"] = near["impl_volatility"].mean()

        # IV term structure slope (long minus short)
        short = atm[atm["days"] < 60]
        long_term = atm[atm["days"] >= 180]
        if len(short) > 0 and len(long_term) > 0:
            feat["iv_term_slope"] = long_term["impl_volatility"].mean() - short["impl_volatility"].mean()

        # IV skew (OTM put vs ATM)
        if len(otm_put) > 0 and len(near) > 0:
            feat["iv_skew"] = otm_put["impl_volatility"].mean() - near["impl_volatility"].mean()

        # Put-call IV spread
        if len(otm_put) > 0 and len(otm_call) > 0:
            feat["iv_put_call_spread"] = otm_put["impl_volatility"].mean() - otm_call["impl_volatility"].mean()

        features.append(feat)

    return pd.DataFrame(features)


def main():
    print("=" * 70)
    print("PHASE 3 — STEP 4: OPTIONMETRICS (ATTEMPT)")
    print("=" * 70)
    start = time.time()

    df = try_optionmetrics()

    if df is not None and len(df) > 0:
        print(f"\n  Computing IV features...")
        iv_features = compute_iv_features(df)

        output_path = os.path.join(DATA_DIR, "optionmetrics_iv.parquet")
        iv_features.to_parquet(output_path, index=False, engine="pyarrow")
        file_size = os.path.getsize(output_path) / (1024 * 1024)

        subprocess.run(
            ["aws", "s3", "cp", output_path,
             f"s3://{S3_BUCKET}/features/optionmetrics_iv.parquet"],
            capture_output=True,
        )

        elapsed = time.time() - start
        print(f"\n  ✅ OptionMetrics IV features: {len(iv_features):,} rows, {file_size:.1f} MB")
        print(f"  Time: {elapsed:.0f}s")
    else:
        elapsed = time.time() - start
        print(f"\n  ⚠️ OptionMetrics not available — this is OK!")
        print(f"  The GKX panel will still have 600-900 features without options data.")
        print(f"  Time: {elapsed:.0f}s")

    print(f"\n{'=' * 70}")
    print(f"STEP 4 COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
