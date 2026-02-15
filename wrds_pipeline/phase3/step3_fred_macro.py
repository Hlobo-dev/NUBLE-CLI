"""
PHASE 3 — STEP 3: FRED Macro Series (35+ indicators)
=====================================================
Daily + Monthly macro data from FRED (no API key needed).
Categories: interest rates, employment, inflation, real activity,
credit spreads, financial markets, NBER recession.
"""

import pandas as pd
import numpy as np
import os
import time
import subprocess

DATA_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/data/wrds"
S3_BUCKET = "nuble-data-warehouse"
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


# ── SERIES DEFINITIONS ──────────────────────────────────────────────────
DAILY_SERIES = {
    # Interest rates
    "DFF":     "fed_funds_rate",
    "DTB3":    "tbill_3m",
    "DGS2":    "treasury_2y",
    "DGS5":    "treasury_5y",
    "DGS10":   "treasury_10y",
    "DGS30":   "treasury_30y",
    # Credit spreads
    "BAA":     "baa_yield",
    "AAA":     "aaa_yield",
    "BAMLC0A0CMEY": "corp_spread_hy",
    "BAMLC0A4CBBBEY": "corp_spread_bbb",
    # Volatility
    "VIXCLS":  "vix",
    # Dollar
    "DTWEXBGS": "trade_weighted_usd",
    # Oil
    "DCOILWTICO": "wti_crude",
    # Gold
    "GOLDPMGBD228NLBM": "gold_price",
}

MONTHLY_SERIES = {
    # Employment
    "UNRATE":       "unemployment_rate",
    "PAYEMS":       "nonfarm_payrolls",
    "ICSA":         "initial_claims",
    "LNS14000006":  "unemp_rate_black",
    # Inflation
    "CPIAUCSL":     "cpi",
    "CPILFESL":     "core_cpi",
    "PCEPI":        "pce_deflator",
    "PCEPILFE":     "core_pce",
    "T10YIE":       "breakeven_10y",
    # Real activity
    "INDPRO":       "industrial_production",
    "TCU":          "capacity_utilization",
    "RSAFS":        "retail_sales",
    "UMCSENT":      "consumer_sentiment",
    "DGORDER":      "durable_goods_orders",
    "HOUST":        "housing_starts",
    "PERMIT":       "building_permits",
    "CSUSHPINSA":   "case_shiller_hpi",
    # Money & credit
    "M2SL":         "m2_money_supply",
    "TOTALSL":      "consumer_credit",
    "BUSLOANS":     "commercial_loans",
    # Yield curve
    "T10Y2Y":       "yield_curve_10y2y",
    "T10Y3M":       "yield_curve_10y3m",
    # ISM
    "MANEMP":       "manufacturing_employment",
    # Leading indicators
    "USSLIND":      "leading_index",
    # NBER recession
    "USREC":        "nber_recession",
}


def download_fred_series(fred_id, name, start="1920-01-01"):
    """Download single FRED series as DataFrame."""
    url = f"{FRED_BASE}?id={fred_id}&cosd={start}"
    try:
        df = pd.read_csv(url, na_values=[".", ""])
        # FRED uses 'observation_date' or 'DATE' as date column
        date_col = None
        for candidate in ["observation_date", "DATE", "date"]:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.rename(columns={date_col: "date", fred_id: name})
        df[name] = pd.to_numeric(df[name], errors="coerce")
        df = df.dropna(subset=[name])
        return df
    except Exception as e:
        print(f"    ❌ {fred_id:<20} | {str(e)[:60]}")
        return None


def main():
    print("=" * 70)
    print("PHASE 3 — STEP 3: FRED MACRO SERIES (35+ INDICATORS)")
    print("=" * 70)
    start = time.time()

    # ── DAILY SERIES ─────────────────────────────────────────────────────
    print("\n📊 DAILY SERIES")
    daily_frames = []
    for fred_id, name in DAILY_SERIES.items():
        df = download_fred_series(fred_id, name)
        if df is not None:
            daily_frames.append(df)
            print(f"    ✅ {fred_id:<20} → {name:<25} | {len(df):>6,} obs")
        time.sleep(0.3)

    # Merge daily
    if daily_frames:
        daily = daily_frames[0]
        for df in daily_frames[1:]:
            daily = daily.merge(df, on="date", how="outer")
        daily = daily.sort_values("date").reset_index(drop=True)

        # Derived daily features
        if "baa_yield" in daily.columns and "aaa_yield" in daily.columns:
            daily["credit_spread"] = daily["baa_yield"] - daily["aaa_yield"]
        if "treasury_10y" in daily.columns and "treasury_2y" in daily.columns:
            daily["term_spread_10y2y"] = daily["treasury_10y"] - daily["treasury_2y"]
        if "treasury_10y" in daily.columns and "tbill_3m" in daily.columns:
            daily["term_spread_10y3m"] = daily["treasury_10y"] - daily["tbill_3m"]

        daily_path = os.path.join(DATA_DIR, "fred_daily.parquet")
        daily.to_parquet(daily_path, index=False, engine="pyarrow")
        daily_size = os.path.getsize(daily_path) / (1024 * 1024)
        print(f"\n  Daily panel: {len(daily):,} days × {daily.shape[1]} cols, {daily_size:.1f} MB")
    else:
        daily = pd.DataFrame()
        print("\n  ⚠️ No daily series downloaded")

    # ── MONTHLY SERIES ───────────────────────────────────────────────────
    print("\n📊 MONTHLY SERIES")
    monthly_frames = []
    for fred_id, name in MONTHLY_SERIES.items():
        df = download_fred_series(fred_id, name)
        if df is not None:
            # Snap to month-end
            df["date"] = df["date"] + pd.offsets.MonthEnd(0)
            df = df.groupby("date").last().reset_index()
            monthly_frames.append(df)
            print(f"    ✅ {fred_id:<20} → {name:<25} | {len(df):>6,} obs")
        time.sleep(0.3)

    # Merge monthly
    if monthly_frames:
        monthly = monthly_frames[0]
        for df in monthly_frames[1:]:
            monthly = monthly.merge(df, on="date", how="outer")
        monthly = monthly.sort_values("date").reset_index(drop=True)

        # YoY changes for level variables
        yoy_vars = [
            "cpi", "core_cpi", "pce_deflator", "core_pce",
            "industrial_production", "retail_sales", "nonfarm_payrolls",
            "m2_money_supply", "consumer_credit", "commercial_loans",
            "housing_starts", "durable_goods_orders",
        ]
        for v in yoy_vars:
            if v in monthly.columns:
                monthly[f"{v}_yoy"] = monthly[v].pct_change(12) * 100

        # MoM changes
        mom_vars = ["industrial_production", "retail_sales", "nonfarm_payrolls"]
        for v in mom_vars:
            if v in monthly.columns:
                monthly[f"{v}_mom"] = monthly[v].pct_change(1) * 100

        monthly_path = os.path.join(DATA_DIR, "fred_monthly.parquet")
        monthly.to_parquet(monthly_path, index=False, engine="pyarrow")
        monthly_size = os.path.getsize(monthly_path) / (1024 * 1024)
        print(f"\n  Monthly panel: {len(monthly):,} months × {monthly.shape[1]} cols, {monthly_size:.1f} MB")
    else:
        monthly = pd.DataFrame()
        print("\n  ⚠️ No monthly series downloaded")

    # ── COMBINED MACRO PREDICTORS ────────────────────────────────────────
    # Convert daily to monthly for one unified file
    if len(daily) > 0:
        daily_monthly = daily.copy()
        daily_monthly["month_end"] = daily_monthly["date"] + pd.offsets.MonthEnd(0)
        daily_monthly_agg = daily_monthly.groupby("month_end").agg("last").reset_index()
        daily_monthly_agg = daily_monthly_agg.drop(columns=["date"], errors="ignore")
        daily_monthly_agg = daily_monthly_agg.rename(columns={"month_end": "date"})

        if len(monthly) > 0:
            macro = daily_monthly_agg.merge(monthly, on="date", how="outer", suffixes=("_daily", "_monthly"))
        else:
            macro = daily_monthly_agg
    else:
        macro = monthly if len(monthly) > 0 else pd.DataFrame()

    if len(macro) > 0:
        macro = macro.sort_values("date").reset_index(drop=True)
        macro_path = os.path.join(DATA_DIR, "macro_predictors.parquet")
        macro.to_parquet(macro_path, index=False, engine="pyarrow")
        macro_size = os.path.getsize(macro_path) / (1024 * 1024)
        print(f"\n  Combined macro: {len(macro):,} months × {macro.shape[1]} cols, {macro_size:.1f} MB")

    # ── S3 UPLOAD ────────────────────────────────────────────────────────
    print("\n📤 S3 UPLOAD")
    for fname in ["fred_daily.parquet", "fred_monthly.parquet", "macro_predictors.parquet"]:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            subprocess.run(
                ["aws", "s3", "cp", fpath,
                 f"s3://{S3_BUCKET}/features/{fname}"],
                capture_output=True,
            )
            print(f"  ✅ Uploaded {fname}")

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"FRED MACRO SERIES COMPLETE")
    print(f"{'=' * 70}")
    n_daily = len(daily) if len(daily) > 0 else 0
    n_monthly = len(monthly) if len(monthly) > 0 else 0
    n_combined = len(macro) if len(macro) > 0 else 0
    print(f"  Daily:    {n_daily:>8,} obs × {daily.shape[1] if n_daily else 0} cols")
    print(f"  Monthly:  {n_monthly:>8,} obs × {monthly.shape[1] if n_monthly else 0} cols")
    print(f"  Combined: {n_combined:>8,} obs × {macro.shape[1] if n_combined else 0} cols")
    print(f"  Time:     {elapsed:.0f}s")
    print(f"  ✅ All saved and uploaded to S3")


if __name__ == "__main__":
    main()
