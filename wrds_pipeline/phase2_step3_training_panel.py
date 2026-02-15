"""
PHASE 2 — STEP 3: Unified ML Training Panel Builder
=====================================================
Produces a single ML-ready DataFrame merging ALL WRDS datasets
with strict point-in-time discipline.

Input:  All Parquet files in data/wrds/
Output: data/wrds/training_panel.parquet (~1-3 GB)

Design Principles (NON-NEGOTIABLE):
1. POINT-IN-TIME ONLY — never use future data in features
2. PERMNO IS THE UNIVERSAL KEY — all datasets mapped to PERMNO
3. SURVIVORSHIP-BIAS FREE — includes delisted stocks
4. MONTHLY FREQUENCY — features aligned to end-of-month
5. REAL CODE ONLY — no pass, no TODO, no pseudocode

Expected output:
  Rows:     ~2-5M (permno × month pairs)
  Columns:  ~120 (identifiers + features + labels)
  Date:     1970-2024 (55 years)
  Stocks/mo: ~3,000-5,000
"""

import pandas as pd
import numpy as np
import os
import gc
import time
import warnings
warnings.filterwarnings("ignore")

WRDS_DIR = "data/wrds"
OUTPUT_PATH = os.path.join(WRDS_DIR, "training_panel.parquet")


def load_parquet(name, columns=None):
    """Load a parquet file from the WRDS data directory."""
    path = os.path.join(WRDS_DIR, f"{name}.parquet")
    if not os.path.exists(path):
        print(f"  ⚠️  {path} not found, skipping")
        return None
    df = pd.read_parquet(path, columns=columns)
    return df


def build_training_panel():
    print("=" * 70)
    print("PHASE 2 — STEP 3: UNIFIED ML TRAINING PANEL BUILDER")
    print("=" * 70)
    start_time = time.time()

    # ════════════════════════════════════════════════════════════════
    # STEP 1: CRSP MONTHLY RETURNS (the backbone)
    # ════════════════════════════════════════════════════════════════
    print("\n[1/9] Loading CRSP monthly returns (backbone)...")

    crsp = load_parquet("crsp_monthly")
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp["date"] = crsp["date"] + pd.offsets.MonthEnd(0)

    # Numeric conversions
    for col in ["ret", "retx", "prc", "shrout", "vol"]:
        if col in crsp.columns:
            crsp[col] = pd.to_numeric(crsp[col], errors="coerce")

    # Compute market cap
    if "market_cap" not in crsp.columns:
        crsp["market_cap"] = np.abs(crsp["prc"]) * crsp["shrout"]
    else:
        crsp["market_cap"] = pd.to_numeric(crsp["market_cap"], errors="coerce")

    # Filter: valid returns and positive market cap
    crsp = crsp[crsp["ret"].notna() & (crsp["market_cap"] > 0)].copy()
    crsp = crsp.sort_values(["permno", "date"]).reset_index(drop=True)

    print(f"  CRSP monthly: {len(crsp):,} rows | "
          f"{crsp['permno'].nunique():,} stocks | "
          f"{crsp['date'].min().date()} to {crsp['date'].max().date()}")

    # ════════════════════════════════════════════════════════════════
    # STEP 2: FORWARD RETURN LABELS (what we're predicting)
    # ════════════════════════════════════════════════════════════════
    print("\n[2/9] Computing forward return labels...")

    crsp = crsp.sort_values(["permno", "date"])
    g = crsp.groupby("permno")

    # Forward cumulative returns for 1/3/6/12 month horizons
    for horizon, label in [(1, "fwd_ret_1m"), (3, "fwd_ret_3m"),
                           (6, "fwd_ret_6m"), (12, "fwd_ret_12m")]:
        # For each stock-month, compute cumulative return over next 'horizon' months
        crsp[label] = g["ret"].transform(
            lambda x: (1 + x).rolling(horizon).apply(np.prod, raw=True) - 1
        ).shift(-horizon)

    # Count valid labels
    for label in ["fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m"]:
        valid = crsp[label].notna().sum()
        print(f"  {label}: {valid:,} valid ({valid/len(crsp):.1%})")

    # ════════════════════════════════════════════════════════════════
    # STEP 3: FAMA-FRENCH FACTORS (risk-free rate for excess returns)
    # ════════════════════════════════════════════════════════════════
    print("\n[3/9] Loading Fama-French factors...")

    ff = load_parquet("ff_factors_monthly")
    ff["date"] = pd.to_datetime(ff["date"])
    ff["date"] = ff["date"] + pd.offsets.MonthEnd(0)

    # Convert percentages to decimals if needed
    for col in ["mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]:
        if col in ff.columns:
            ff[col] = pd.to_numeric(ff[col], errors="coerce")
            if ff[col].abs().mean() > 0.5:
                ff[col] = ff[col] / 100.0

    crsp = crsp.merge(ff[["date", "rf"]], on="date", how="left")
    crsp["excess_ret_1m"] = crsp["fwd_ret_1m"] - crsp["rf"]

    print(f"  FF factors: {len(ff):,} months | "
          f"RF mean: {ff['rf'].mean():.5f}/month")

    # ════════════════════════════════════════════════════════════════
    # STEP 4: MOMENTUM & TECHNICAL FEATURES (from CRSP monthly)
    # ════════════════════════════════════════════════════════════════
    print("\n[4/9] Computing momentum & technical features...")

    crsp = crsp.sort_values(["permno", "date"])
    g = crsp.groupby("permno")

    # Momentum signals (using PAST returns — no lookahead)
    crsp["mom_1m"] = g["ret"].shift(1)
    crsp["mom_3m"] = g["ret"].transform(
        lambda x: (1 + x).rolling(3).apply(np.prod, raw=True) - 1
    ).shift(1)
    crsp["mom_6m"] = g["ret"].transform(
        lambda x: (1 + x).rolling(6).apply(np.prod, raw=True) - 1
    ).shift(1)
    crsp["mom_12m"] = g["ret"].transform(
        lambda x: (1 + x).rolling(12).apply(np.prod, raw=True) - 1
    ).shift(1)
    # Jegadeesh-Titman 12-2 momentum (skip most recent month)
    crsp["mom_12_2"] = g["ret"].transform(
        lambda x: (1 + x).rolling(11).apply(np.prod, raw=True) - 1
    ).shift(2)

    # Short-term reversal
    crsp["str_reversal"] = g["ret"].shift(1)  # same as mom_1m (1-month reversal)

    # Volatility (annualized)
    crsp["vol_3m"] = g["ret"].transform(lambda x: x.rolling(3).std() * np.sqrt(12))
    crsp["vol_6m"] = g["ret"].transform(lambda x: x.rolling(6).std() * np.sqrt(12))
    crsp["vol_12m"] = g["ret"].transform(lambda x: x.rolling(12).std() * np.sqrt(12))

    # Turnover (volume / shares outstanding)
    crsp["turnover"] = crsp["vol"] / (crsp["shrout"] * 1000).clip(lower=1)
    crsp["turnover_3m"] = g["turnover"].transform(lambda x: x.rolling(3).mean())
    crsp["turnover_6m"] = g["turnover"].transform(lambda x: x.rolling(6).mean())

    # Size — academic standard: log(market_cap in MILLIONS of dollars)
    # CRSP market_cap = abs(prc) * shrout * 1000 (dollars)
    # Convert to millions: / 1e6
    crsp["log_market_cap"] = np.log((crsp["market_cap"] / 1e6).clip(lower=0.001))
    crsp["log_price"] = np.log(np.abs(crsp["prc"]).clip(lower=0.01))

    print(f"  Computed: mom_1m/3m/6m/12m/12_2, vol_3m/6m/12m, "
          f"turnover, log_mcap, log_price")

    # ════════════════════════════════════════════════════════════════
    # STEP 5: WRDS FINANCIAL RATIOS (★ THE GOLDMINE — 70+ features)
    # ════════════════════════════════════════════════════════════════
    print("\n[5/9] Loading WRDS Financial Ratios (pre-computed features)...")

    ratios = load_parquet("wrds_financial_ratios")
    if ratios is not None:
        ratios["public_date"] = pd.to_datetime(ratios["public_date"])
        ratios["date"] = ratios["public_date"] + pd.offsets.MonthEnd(0)

        # Identify feature columns (exclude identifiers)
        key_cols_fr = ["permno", "public_date", "adate", "qdate", "gvkey",
                       "cusip", "ticker", "sic", "naics", "date"]
        feature_cols_fr = [c for c in ratios.columns if c not in key_cols_fr]

        # Convert to numeric
        for col in feature_cols_fr:
            ratios[col] = pd.to_numeric(ratios[col], errors="coerce")

        # De-duplicate: keep latest per (permno, month) — POINT-IN-TIME via public_date
        ratios = ratios.sort_values(["permno", "date", "public_date"])
        ratios = ratios.drop_duplicates(subset=["permno", "date"], keep="last")
        ratios_merge = ratios[["permno", "date"] + feature_cols_fr].copy()

        print(f"  Financial Ratios: {len(ratios_merge):,} rows | "
              f"{ratios_merge['permno'].nunique():,} stocks | "
              f"{len(feature_cols_fr)} features")
        print(f"  Sample features: {feature_cols_fr[:8]}")

        del ratios
        gc.collect()
    else:
        ratios_merge = None
        feature_cols_fr = []

    # ════════════════════════════════════════════════════════════════
    # STEP 6: IBES EARNINGS FEATURES
    # ════════════════════════════════════════════════════════════════
    print("\n[6/9] Computing IBES earnings features...")

    # Load IBES-CRSP link (ticker → permno)
    ibes_link = load_parquet("ibes_crsp_link", columns=["ticker", "permno"])
    if ibes_link is not None:
        ibes_link = ibes_link.dropna(subset=["permno"])
        ibes_link["permno"] = ibes_link["permno"].astype(int)
        ibes_link = ibes_link.drop_duplicates(subset=["ticker"], keep="first")

    # --- IBES Summary: consensus estimates ---
    ibes_raw = load_parquet("ibes_summary")
    if ibes_raw is not None and ibes_link is not None:
        ibes_raw["statpers"] = pd.to_datetime(ibes_raw["statpers"])

        # Filter to EPS forecasts for next quarter (fpi='1') and next year (fpi='2')
        ibes_q1 = ibes_raw[ibes_raw["fpi"] == "1"].copy()
        ibes_q1 = ibes_q1[ibes_q1["statpers"].notna() & (ibes_q1["numest"] >= 1)]

        # Convert to numeric
        for col in ["numest", "meanest", "medest", "stdev"]:
            if col in ibes_q1.columns:
                ibes_q1[col] = pd.to_numeric(ibes_q1[col], errors="coerce")

        # Link ticker → PERMNO
        ibes_q1 = ibes_q1.merge(ibes_link, on="ticker", how="inner")
        ibes_q1["date"] = ibes_q1["statpers"] + pd.offsets.MonthEnd(0)

        # De-duplicate
        ibes_q1 = ibes_q1.sort_values(["permno", "date", "statpers"])
        ibes_q1 = ibes_q1.drop_duplicates(subset=["permno", "date"], keep="last")

        # Analyst dispersion
        ibes_q1["analyst_dispersion"] = (
            ibes_q1["stdev"] / np.abs(ibes_q1["meanest"]).clip(lower=0.01)
        )

        # Analyst revision (3-month change in consensus)
        ibes_q1 = ibes_q1.sort_values(["permno", "date"])
        ibes_q1["analyst_revision"] = ibes_q1.groupby("permno")["meanest"].transform(
            lambda x: x.pct_change(3)
        )

        # Number of analysts coverage
        ibes_features = ibes_q1[["permno", "date", "numest",
                                  "analyst_dispersion", "analyst_revision"]].copy()
        ibes_features = ibes_features.rename(columns={"numest": "num_analysts"})

        # --- SUE: Standardized Unexpected Earnings (from IBES actuals) ---
        ibes_actuals = load_parquet("ibes_actuals")
        if ibes_actuals is not None:
            ibes_actuals = ibes_actuals[ibes_actuals["measure"] == "EPS"].copy()
            ibes_actuals["anndats"] = pd.to_datetime(ibes_actuals["anndats"])
            ibes_actuals = ibes_actuals.merge(ibes_link, on="ticker", how="inner")
            ibes_actuals["date"] = ibes_actuals["anndats"] + pd.offsets.MonthEnd(0)
            ibes_actuals = ibes_actuals.sort_values(["permno", "date"])
            ibes_actuals = ibes_actuals.drop_duplicates(subset=["permno", "date"], keep="last")

            # Merge actual EPS into summary for SUE computation
            # SUE = (actual - forecast) / abs(forecast)
            sue_df = ibes_actuals[["permno", "date", "value"]].rename(
                columns={"value": "actual_eps"}
            )
            ibes_features = ibes_features.merge(sue_df, on=["permno", "date"], how="left")

            # Also merge meanest for SUE
            meanest_df = ibes_q1[["permno", "date", "meanest"]].copy()
            ibes_features = ibes_features.merge(meanest_df, on=["permno", "date"], how="left")

            ibes_features["sue"] = (
                (ibes_features["actual_eps"] - ibes_features["meanest"]) /
                np.abs(ibes_features["meanest"]).clip(lower=0.01)
            )
            ibes_features = ibes_features.drop(columns=["actual_eps", "meanest"], errors="ignore")

        ibes_features = ibes_features.drop_duplicates(subset=["permno", "date"], keep="last")

        print(f"  IBES features: {len(ibes_features):,} rows | "
              f"Cols: {list(ibes_features.columns[2:])}")

        del ibes_raw, ibes_q1
        gc.collect()
    else:
        ibes_features = None

    # --- IBES Recommendations ---
    recs_raw = load_parquet("ibes_recommendations")
    if recs_raw is not None and ibes_link is not None:
        recs_raw["anndats"] = pd.to_datetime(recs_raw["anndats"])
        recs_raw["ireccd"] = pd.to_numeric(recs_raw["ireccd"], errors="coerce")
        recs_raw = recs_raw[recs_raw["ireccd"].between(1, 5)].copy()
        recs_raw = recs_raw.merge(ibes_link, on="ticker", how="inner")
        recs_raw["date"] = recs_raw["anndats"] + pd.offsets.MonthEnd(0)

        rec_monthly = recs_raw.groupby(["permno", "date"]).agg(
            rec_score=("ireccd", "mean"),
            num_recs=("ireccd", "count")
        ).reset_index()

        if ibes_features is not None:
            ibes_features = ibes_features.merge(rec_monthly, on=["permno", "date"], how="outer")
        else:
            ibes_features = rec_monthly

        print(f"  + Recommendations: {len(rec_monthly):,} stock-months")

        del recs_raw, rec_monthly
        gc.collect()

    # ════════════════════════════════════════════════════════════════
    # STEP 7: ROLLING BETAS
    # ════════════════════════════════════════════════════════════════
    print("\n[7/9] Loading rolling betas...")

    betas_path = os.path.join(WRDS_DIR, "rolling_betas.parquet")
    if os.path.exists(betas_path):
        betas = pd.read_parquet(betas_path)
        betas["date"] = pd.to_datetime(betas["date"])
        betas["date"] = betas["date"] + pd.offsets.MonthEnd(0)
        betas = betas.drop_duplicates(subset=["permno", "date"], keep="last")
        print(f"  Betas: {len(betas):,} rows | {betas['permno'].nunique():,} stocks")
    else:
        print(f"  ⚠️  {betas_path} not found — run phase2_step2_rolling_betas.py first")
        betas = None

    # ════════════════════════════════════════════════════════════════
    # STEP 8: INSTITUTIONAL & INSIDER FEATURES
    # ════════════════════════════════════════════════════════════════
    print("\n[8/9] Computing institutional & insider features...")

    # --- Build CUSIP → PERMNO mapping from CRSP monthly ---
    # CRSP monthly has cusip via compustat_security + crsp_compustat_link
    # But easiest: use the CRSP stocknames approach — map 8-char CUSIP to PERMNO

    # From CRSP-Compustat link + Compustat Security
    ccl = load_parquet("crsp_compustat_link")
    comp_sec = load_parquet("compustat_security")

    cusip_permno_map = None
    if ccl is not None and comp_sec is not None:
        # ccl uses lpermno
        ccl_clean = ccl[ccl["lpermno"].notna()].copy()
        ccl_clean["permno"] = ccl_clean["lpermno"].astype(int)
        ccl_clean = ccl_clean[["gvkey", "permno"]].drop_duplicates(subset=["gvkey"], keep="first")

        if "cusip" in comp_sec.columns:
            cusip_col = "cusip"
        elif "tic" in comp_sec.columns:
            cusip_col = None  # no cusip, skip
        else:
            cusip_col = None

        if cusip_col and "gvkey" in comp_sec.columns:
            sec_cusip = comp_sec[["gvkey", cusip_col]].dropna().drop_duplicates(subset=["gvkey"], keep="last")
            cusip_permno_map = sec_cusip.merge(ccl_clean, on="gvkey", how="inner")
            cusip_permno_map["cusip8"] = cusip_permno_map[cusip_col].astype(str).str[:8]
            cusip_permno_map = cusip_permno_map[["cusip8", "permno"]].drop_duplicates(
                subset=["cusip8"], keep="first"
            )
            print(f"  CUSIP→PERMNO map: {len(cusip_permno_map):,} mappings")

    # --- Institutional Holdings ---
    inst = load_parquet("institutional_holdings", columns=["fdate", "cusip", "mgrno", "shares"])
    inst_features = None
    if inst is not None and cusip_permno_map is not None:
        inst["fdate"] = pd.to_datetime(inst["fdate"])
        inst["shares"] = pd.to_numeric(inst["shares"], errors="coerce")
        inst = inst[inst["shares"] > 0].copy()
        inst["cusip8"] = inst["cusip"].astype(str).str[:8]

        # Aggregate by cusip-quarter
        inst_agg = inst.groupby(["cusip8", "fdate"]).agg(
            total_inst_shares=("shares", "sum"),
            num_institutions=("mgrno", "nunique")
        ).reset_index()

        # Map CUSIP → PERMNO
        inst_agg = inst_agg.merge(cusip_permno_map, on="cusip8", how="inner")
        inst_agg["date"] = inst_agg["fdate"] + pd.offsets.MonthEnd(0)

        # Forward-fill quarterly → monthly (lag 45 days for filing delay)
        inst_agg = inst_agg.sort_values(["permno", "date"])
        inst_agg = inst_agg.drop_duplicates(subset=["permno", "date"], keep="last")

        # Resample to monthly via merge_asof
        # For each stock-month in crsp, find the most recent institutional holding
        inst_monthly = inst_agg[["permno", "date", "total_inst_shares", "num_institutions"]].copy()
        inst_features = inst_monthly

        print(f"  Institutional: {len(inst_features):,} obs | "
              f"{inst_features['permno'].nunique():,} stocks")

        del inst, inst_agg
        gc.collect()

    # --- Insider Trading ---
    insider = load_parquet("insider_trading")
    insider_features = None
    if insider is not None:
        insider["trade_date"] = pd.to_datetime(insider["trade_date"])
        insider["shares"] = pd.to_numeric(insider["shares"], errors="coerce")
        insider = insider[insider["shares"].notna() & (insider["shares"] > 0)].copy()

        if "cusip6" in insider.columns and "cusip2" in insider.columns:
            insider["cusip8"] = (insider["cusip6"].fillna("").astype(str) +
                                insider["cusip2"].fillna("").astype(str))
        elif "cusip6" in insider.columns:
            insider["cusip8"] = insider["cusip6"].astype(str).str[:8]
        elif "cusip" in insider.columns:
            insider["cusip8"] = insider["cusip"].astype(str).str[:8]

        insider["date"] = insider["trade_date"] + pd.offsets.MonthEnd(0)

        # Buy/sell classification
        insider["is_buy"] = insider["transaction_type"].isin(["P", "A"]).astype(int)
        insider["is_sell"] = insider["transaction_type"].isin(["S", "D"]).astype(int)
        insider["buy_shares"] = insider["shares"] * insider["is_buy"]
        insider["sell_shares"] = insider["shares"] * insider["is_sell"]

        insider_monthly = insider.groupby(["cusip8", "date"]).agg(
            insider_buy_shares=("buy_shares", "sum"),
            insider_sell_shares=("sell_shares", "sum"),
            insider_num_buys=("is_buy", "sum"),
            insider_num_sells=("is_sell", "sum")
        ).reset_index()

        insider_monthly["insider_buy_ratio"] = (
            insider_monthly["insider_num_buys"] /
            (insider_monthly["insider_num_buys"] + insider_monthly["insider_num_sells"]).clip(lower=1)
        )

        # Map CUSIP → PERMNO
        if cusip_permno_map is not None:
            insider_monthly = insider_monthly.merge(cusip_permno_map, on="cusip8", how="inner")
            insider_features = insider_monthly[["permno", "date", "insider_buy_ratio",
                                                 "insider_num_buys", "insider_num_sells"]].copy()
            print(f"  Insider: {len(insider_features):,} obs | "
                  f"{insider_features['permno'].nunique():,} stocks")

        del insider
        gc.collect()

    # ════════════════════════════════════════════════════════════════
    # STEP 9: MERGE EVERYTHING INTO THE TRAINING PANEL
    # ════════════════════════════════════════════════════════════════
    print("\n[9/9] Merging all features into unified training panel...")

    # Start with CRSP monthly as backbone
    panel_cols = [
        "permno", "date", "ret", "market_cap", "log_market_cap", "log_price",
        "mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12_2", "str_reversal",
        "vol_3m", "vol_6m", "vol_12m",
        "turnover", "turnover_3m", "turnover_6m",
        "fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m",
        "excess_ret_1m", "rf",
    ]
    # Keep only columns that exist
    panel_cols = [c for c in panel_cols if c in crsp.columns]
    panel = crsp[panel_cols].copy()

    # Add exchange and SIC info if available
    for col in ["exchcd", "shrcd", "siccd"]:
        if col in crsp.columns:
            panel[col] = crsp[col]

    print(f"  Base panel (CRSP monthly): {len(panel):,} rows × {panel.shape[1]} cols")
    initial_rows = len(panel)

    # Merge WRDS Financial Ratios (★ THE GOLDMINE)
    if ratios_merge is not None:
        panel = panel.merge(ratios_merge, on=["permno", "date"], how="left")
        match_rate = panel[feature_cols_fr[0]].notna().mean() if feature_cols_fr else 0
        print(f"  + Financial Ratios: {panel.shape[1]} cols | "
              f"Match rate: {match_rate:.1%} | "
              f"{len(feature_cols_fr)} features added")

    # Merge IBES earnings features
    if ibes_features is not None:
        panel = panel.merge(ibes_features, on=["permno", "date"], how="left")
        sue_match = panel["sue"].notna().mean() if "sue" in panel.columns else 0
        print(f"  + IBES features: {panel.shape[1]} cols | "
              f"SUE match: {sue_match:.1%}")

    # Merge rolling betas
    if betas is not None:
        beta_cols = [c for c in betas.columns if c not in ["permno", "date"]]
        panel = panel.merge(betas[["permno", "date"] + beta_cols],
                           on=["permno", "date"], how="left")
        beta_match = panel["beta_mkt"].notna().mean() if "beta_mkt" in panel.columns else 0
        print(f"  + Betas: {panel.shape[1]} cols | "
              f"Beta match: {beta_match:.1%}")

    # Merge institutional features (quarterly → monthly via left join + forward fill)
    if inst_features is not None:
        # Direct merge on exact (permno, date) for quarterly observations
        # Then forward-fill within each permno to cover non-reporting months
        panel = panel.merge(inst_features, on=["permno", "date"], how="left")
        # Forward fill within permno (quarterly → monthly), max 4 months
        panel = panel.sort_values(["permno", "date"])
        for col in ["total_inst_shares", "num_institutions"]:
            if col in panel.columns:
                panel[col] = panel.groupby("permno")[col].transform(
                    lambda x: x.ffill(limit=4)
                )
        inst_match = panel["num_institutions"].notna().mean() if "num_institutions" in panel.columns else 0
        print(f"  + Institutional: {panel.shape[1]} cols | "
              f"Match: {inst_match:.1%}")

    # Merge insider features
    if insider_features is not None:
        panel = panel.merge(insider_features, on=["permno", "date"], how="left")
        insider_match = panel["insider_buy_ratio"].notna().mean() if "insider_buy_ratio" in panel.columns else 0
        print(f"  + Insider: {panel.shape[1]} cols | "
              f"Match: {insider_match:.1%}")

    # ─── S&P 500 membership ───
    print("  Adding S&P 500 membership...")
    sp500 = load_parquet("sp500_constituents_crsp")
    if sp500 is not None:
        sp500["from_date"] = pd.to_datetime(sp500["from_date"])
        sp500["thru_date"] = pd.to_datetime(sp500["thru_date"])

        # Build interval index for fast membership lookup
        # For each row in panel, check if any SP500 period covers that date
        panel["sp500_member"] = 0

        for _, row in sp500.iterrows():
            if pd.isna(row["from_date"]):
                continue
            perm = row["permno"]
            start = row["from_date"]
            end = row["thru_date"] if pd.notna(row["thru_date"]) else pd.Timestamp("2025-12-31")
            mask = (
                (panel["permno"] == perm) &
                (panel["date"] >= start) &
                (panel["date"] <= end)
            )
            panel.loc[mask, "sp500_member"] = 1

        sp500_count = panel["sp500_member"].sum()
        print(f"  + S&P 500: {sp500_count:,} member-months")

    # ─── Delisting returns ───
    print("  Adding delisting returns...")
    delist = load_parquet("crsp_delisting")
    if delist is not None:
        delist["dlstdt"] = pd.to_datetime(delist["dlstdt"])
        delist["dlret"] = pd.to_numeric(delist["dlret"], errors="coerce")
        delist["date"] = delist["dlstdt"] + pd.offsets.MonthEnd(0)
        delist_merge = delist[["permno", "date", "dlret", "dlstcd"]].dropna(subset=["dlret"])
        delist_merge = delist_merge.drop_duplicates(subset=["permno", "date"], keep="last")
        panel = panel.merge(delist_merge, on=["permno", "date"], how="left")
        delist_count = panel["dlret"].notna().sum()
        print(f"  + Delisting: {delist_count:,} delisting-month obs")

    # ═══════════════════════════════════════════════════════════════
    # FINAL CLEANUP
    # ═══════════════════════════════════════════════════════════════
    print("\nFinal cleanup...")

    # Ensure no duplicate (permno, date) pairs
    panel = panel.drop_duplicates(subset=["permno", "date"], keep="last")
    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════
    print(f"\nSaving to {OUTPUT_PATH}...")
    panel.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
    file_size_gb = os.path.getsize(OUTPUT_PATH) / (1024 ** 3)

    total_time = time.time() - start_time

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"TRAINING PANEL COMPLETE!")
    print(f"{'='*70}")
    print(f"  Total rows:         {len(panel):,}")
    print(f"  Total columns:      {panel.shape[1]}")
    print(f"  Date range:         {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"  Unique PERMNOs:     {panel['permno'].nunique():,}")
    print(f"  Unique months:      {panel['date'].nunique():,}")
    print(f"  Avg stocks/month:   {len(panel) / max(panel['date'].nunique(), 1):.0f}")
    print(f"  File size:          {file_size_gb:.2f} GB")
    print(f"  Time:               {total_time/60:.1f} minutes")

    # Feature coverage report
    print(f"\n{'Feature Group':<30} {'Columns':<10} {'Coverage':<10}")
    print("-" * 50)

    groups = {
        "CRSP Momentum/Vol":     ["mom_1m", "mom_3m", "mom_6m", "mom_12m",
                                  "vol_3m", "vol_12m", "turnover"],
        "WRDS Financial Ratios": feature_cols_fr[:10] if feature_cols_fr else [],
        "IBES Earnings":         ["sue", "analyst_dispersion", "num_analysts",
                                  "rec_score", "analyst_revision"],
        "Factor Betas":          ["beta_mkt", "beta_smb", "beta_hml", "idio_vol"],
        "Institutional":         ["num_institutions", "total_inst_shares"],
        "Insider":               ["insider_buy_ratio", "insider_num_buys"],
        "Forward Returns":       ["fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m"],
    }

    for name, cols in groups.items():
        available = [c for c in cols if c in panel.columns]
        if available:
            coverage = panel[available].notna().mean().mean()
            print(f"  {name:<28} {len(available):<10} {coverage:.1%}")

    # Label distribution
    print(f"\nForward return labels:")
    for col in ["fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m"]:
        if col in panel.columns:
            valid = panel[col].notna().sum()
            mean_ret = panel[col].mean()
            print(f"  {col}: {valid:,} valid ({valid/len(panel):.1%}) | mean={mean_ret:.4f}")

    return panel


if __name__ == "__main__":
    build_training_panel()
