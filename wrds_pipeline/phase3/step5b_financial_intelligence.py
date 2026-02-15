"""
PHASE 3 — STEP 5b: FINANCIAL STATEMENT INTELLIGENCE ENGINE (Level 3)
====================================================================
Implements the Level 3 evolution from ROketalpha — deep analysis of
income statement, balance sheet, cash flow DYNAMICS, plus analyst,
insider, and institutional signals.

FEATURE GROUPS:
  A. Financial Dynamics          ~30 features
  B. Earnings Quality            ~20 features  (Sloan, Beneish, Richardson)
  C. Composite Scoring Models    ~6  features  (Piotroski, Altman, Ohlson, Montier)
  D. Analyst Dynamics            ~12 features  (from IBES)
  E. Institutional & Insider     ~10 features
  F. Industry-Relative           ~10 features
  ─────────────────────────────────────────────
  Total:                         ~88 new features

DATA SOURCES (all already downloaded):
  - Compustat Quarterly      (comp_quarterly.parquet)    2.1M rows × 42 cols
  - IBES Summary             (ibes_summary.parquet)      8.3M rows × 16 cols
  - IBES Actuals             (ibes_actuals.parquet)      1.3M rows × 6  cols
  - Insider Trading          (insider_trading.parquet)   17.1M rows × 17 cols
  - Institutional Holdings   (institutional_holdings)   124.8M rows × 9  cols
  - WRDS Financial Ratios    (wrds_ratios.parquet)       2.8M rows × 100 cols
  - CRSP-Compustat Link      (crsp_compustat_link)       40K  rows
  - IBES-CRSP Link           (ibes_crsp_link)            37K  rows
  - GKX Panel                (gkx_panel.parquet)         3.76M rows × 465 cols

MERGE KEYS:
  Compustat → GKX:  gvkey → lpermno (via crsp_compustat_link) → permno
  IBES → GKX:       IBES ticker → permno (via ibes_crsp_link)
  Insider → GKX:    ticker → permno (via ibes_crsp_link or CRSP)
  Institutional:    cusip → permno (via CRSP mapping in training_panel)

OUTPUT: gkx_panel.parquet (enriched with ~88 new features)

Author: Claude × Humberto Lobo
Date: 2026
"""

import pandas as pd
import numpy as np
import os
import gc
import time
import subprocess
import warnings

warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")
S3_BUCKET = "nuble-data-warehouse"

# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def safe_div(a, b, fill=np.nan):
    """Division with protection against zero/nan denominators."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) > 1e-8, a / b, fill)
    return result.astype(np.float32)


def quarterly_slope(df, col, n_quarters=4):
    """
    OLS slope of `col` over last `n_quarters` within each firm.
    Vectorized using rolling covariance: slope = Cov(x,y) / Var(x).
    """
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float32)
    
    x = np.arange(n_quarters, dtype=np.float64)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    
    if x_var == 0:
        return pd.Series(np.nan, index=df.index, dtype=np.float32)
    
    # Use rolling sum trick: slope = (n*Σ(i*y) - Σi*Σy) / (n*Σ(i²) - (Σi)²)
    # This is equivalent to Cov(x,y)/Var(x) but fully vectorized with rolling sums
    y = df.groupby("permno")[col]
    
    # Rolling mean and rolling weighted sum
    roll_mean = y.transform(lambda s: s.rolling(n_quarters, min_periods=n_quarters).mean())
    
    # For the covariance, we need Σ(x_i * y_i) over the window
    # x_i = [0, 1, 2, 3] for n=4, so we need sum of y_{t}, y_{t-1}*1, y_{t-2}*2, y_{t-3}*3
    # Build weighted sum via shifted lags
    weighted_sum = pd.Series(np.zeros(len(df)), index=df.index)
    for i in range(n_quarters):
        shifted = y.shift(i)
        weight = float(n_quarters - 1 - i)  # x values: [0,...,n-1] → shift 0 = x[n-1]
        weighted_sum += shifted.fillna(0) * weight
    
    # Mark where we don't have enough data
    valid_count = y.transform(lambda s: s.rolling(n_quarters, min_periods=n_quarters).count())
    
    # slope = (weighted_sum - n * x_mean * roll_mean) / (n * x_var / n)
    # Simplified: slope = (weighted_sum/n - x_mean * roll_mean) / (x_var / n)
    cov_xy = weighted_sum / n_quarters - x_mean * roll_mean
    slope = cov_xy / (x_var / n_quarters)
    slope[valid_count < n_quarters] = np.nan
    
    return slope.astype(np.float32)


def quarterly_vol(df, col, n_quarters=8):
    """Standard deviation over last n_quarters within each firm."""
    return df.groupby("permno")[col].transform(
        lambda s: s.rolling(n_quarters, min_periods=4).std()
    )


def pct_change_yoy(df, col):
    """Year-over-year percent change (4 quarters back) within firm."""
    result = df.groupby("permno")[col].transform(
        lambda s: s.pct_change(periods=4)
    )
    return result.replace([np.inf, -np.inf], np.nan)


def pct_change_qoq(df, col):
    """Quarter-over-quarter percent change within firm."""
    result = df.groupby("permno")[col].transform(
        lambda s: s.pct_change(periods=1)
    )
    return result.replace([np.inf, -np.inf], np.nan)


def lag(df, col, periods=1):
    """Lag within firm group."""
    return df.groupby("permno")[col].shift(periods)


def diff(df, col, periods=1):
    """Difference within firm group."""
    return df.groupby("permno")[col].diff(periods)


def rolling_sum_4q(df, col):
    """Trailing 4-quarter sum (TTM) within firm."""
    return df.groupby("permno")[col].transform(
        lambda s: s.rolling(4, min_periods=4).sum()
    )


def annual_from_cumulative(df, col):
    """
    Convert annual cumulative values (e.g., capxy, oancfy)
    to quarterly values by differencing within fiscal year,
    resetting at Q1.
    """
    # Within each firm-fiscal year, the value accumulates Q1→Q4
    # Quarterly value = cumulative(t) - cumulative(t-1), reset at fqtr==1
    df = df.copy()
    prev = df.groupby("permno")[col].shift(1)
    fq1_mask = df["fqtr"] == 1
    quarterly = df[col] - prev
    quarterly[fq1_mask] = df.loc[fq1_mask, col]
    return quarterly


# ══════════════════════════════════════════════════════════════════════════
# A. FINANCIAL DYNAMICS (~30 features)
# ══════════════════════════════════════════════════════════════════════════

def compute_financial_dynamics(cq):
    """
    Revenue & Growth, Margins, Working Capital, Capital Allocation,
    Debt & Leverage dynamics from Compustat Quarterly.
    """
    print("  📊 A.1 Revenue & Growth Dynamics...")
    
    # ── Revenue & Growth ──
    cq["revenue_growth_qoq"] = pct_change_qoq(cq, "revtq")
    cq["revenue_growth_yoy"] = pct_change_yoy(cq, "revtq")
    
    # 2-year CAGR: (rev_t / rev_t-8)^(1/2) - 1
    rev_lag8 = lag(cq, "revtq", 8)
    cq["revenue_cagr_2yr"] = safe_div(cq["revtq"].values, rev_lag8.values)
    mask = cq["revenue_cagr_2yr"].notna() & (cq["revenue_cagr_2yr"] > 0)
    cq.loc[mask, "revenue_cagr_2yr"] = cq.loc[mask, "revenue_cagr_2yr"] ** 0.5 - 1
    cq.loc[~mask, "revenue_cagr_2yr"] = np.nan
    
    # Revenue acceleration: change in YoY growth rate
    yoy_lag4 = lag(cq, "revenue_growth_yoy", 4)
    cq["revenue_acceleration"] = cq["revenue_growth_yoy"] - yoy_lag4
    
    # Operating leverage: %ΔEBIT / %ΔRevenue
    ebit_growth = pct_change_yoy(cq, "oiadpq")
    cq["operating_leverage"] = safe_div(ebit_growth.values, cq["revenue_growth_yoy"].values)
    # Clip extreme values
    cq["operating_leverage"] = cq["operating_leverage"].clip(-10, 10)
    
    print("  📊 A.2 Margin Trajectory...")
    
    # ── Margins ──
    cq["gross_margin"] = safe_div(
        (cq["revtq"] - cq["cogsq"]).values, cq["revtq"].values
    )
    cq["operating_margin"] = safe_div(cq["oiadpq"].values, cq["revtq"].values)
    cq["net_margin"] = safe_div(cq["niq"].values, cq["revtq"].values)
    
    # Margin trends (4-quarter slopes)
    cq["gross_margin_trend"] = quarterly_slope(cq, "gross_margin", 4)
    cq["operating_margin_trend"] = quarterly_slope(cq, "operating_margin", 4)
    cq["net_margin_trend"] = quarterly_slope(cq, "net_margin", 4)
    
    # Margin stability
    cq["gross_margin_vol"] = quarterly_vol(cq, "gross_margin", 8)
    
    # Cross-margin divergence: gross improving but net declining
    cq["margin_divergence"] = cq["gross_margin_trend"] - cq["net_margin_trend"]
    
    print("  📊 A.3 Working Capital Efficiency...")
    
    # ── Working Capital ──
    # Days Sales Outstanding
    cq["dso"] = safe_div(cq["rectq"].values * 91.25, cq["revtq"].values)
    # Days Inventory Outstanding
    cq["dio"] = safe_div(cq["invtq"].values * 91.25, cq["cogsq"].values)
    # Days Payable Outstanding — need AP, approximate from CL - short-term debt
    ap_approx = cq["lctq"] - cq["dlcq"].fillna(0)
    cq["dpo"] = safe_div(ap_approx.values * 91.25, cq["cogsq"].values)
    # Cash Conversion Cycle
    cq["cash_conversion_cycle"] = cq["dso"] + cq["dio"] - cq["dpo"]
    # CCC trend
    cq["ccc_trend"] = quarterly_slope(cq, "cash_conversion_cycle", 4)
    # DSRI — Beneish component
    dso_lag1 = lag(cq, "dso", 1)
    cq["dsri"] = safe_div(cq["dso"].values, dso_lag1.values)
    
    print("  📊 A.4 Capital Allocation...")
    
    # ── Capital Allocation ──
    # Quarterly CapEx from annual cumulative capxy
    cq["capxq"] = annual_from_cumulative(cq, "capxy")
    cq["capxq"] = cq["capxq"].abs()  # capxy is negative in Compustat
    
    cq["capex_to_depreciation"] = safe_div(cq["capxq"].values, cq["dpq"].values)
    cq["capex_intensity"] = safe_div(cq["capxq"].values, cq["revtq"].values)
    cq["capex_intensity_trend"] = quarterly_slope(cq, "capex_intensity", 4)
    
    # R&D intensity
    cq["rd_intensity"] = safe_div(cq["xrdq"].values, cq["revtq"].values)
    cq["rd_intensity_trend"] = quarterly_slope(cq, "rd_intensity", 4)
    
    # SGA efficiency
    cq["sga_efficiency"] = safe_div(cq["revtq"].values, cq["xsgaq"].values)
    cq["sga_efficiency_trend"] = quarterly_slope(cq, "sga_efficiency", 4)
    
    print("  📊 A.5 Debt & Leverage Dynamics...")
    
    # ── Debt & Leverage ──
    total_debt = cq["dlttq"].fillna(0) + cq["dlcq"].fillna(0)
    equity = cq["seqq"].fillna(cq["ceqq"])
    cq["debt_to_equity"] = safe_div(total_debt.values, equity.values)
    cq["debt_to_equity_change"] = diff(cq, "debt_to_equity", 1)
    
    # Interest coverage trend
    cq["interest_coverage"] = safe_div(cq["oiadpq"].values, cq["xintq"].values)
    cq["interest_coverage"] = cq["interest_coverage"].clip(-50, 50)
    cq["interest_coverage_trend"] = quarterly_slope(cq, "interest_coverage", 4)
    
    # Net debt to EBITDA (TTM)
    ebitda_ttm = rolling_sum_4q(cq, "oiadpq") + rolling_sum_4q(cq, "dpq")
    net_debt = total_debt - cq["cheq"].fillna(0)
    cq["net_debt_to_ebitda"] = safe_div(net_debt.values, ebitda_ttm.values)
    cq["net_debt_to_ebitda"] = cq["net_debt_to_ebitda"].clip(-20, 20)
    
    # Debt maturity risk
    cq["debt_maturity_risk"] = safe_div(cq["dlcq"].values, total_debt.values)
    
    # Quarterly operating cash flow from annual cumulative
    cq["oancfq"] = annual_from_cumulative(cq, "oancfy")
    # Quarterly dividends from annual cumulative
    cq["dvq"] = annual_from_cumulative(cq, "dvy")
    
    # FCF = Operating CF - CapEx
    cq["fcfq"] = cq["oancfq"] - cq["capxq"]
    cq["fcf_to_revenue"] = safe_div(cq["fcfq"].values, cq["revtq"].values)
    cq["fcf_to_revenue_trend"] = quarterly_slope(cq, "fcf_to_revenue", 4)
    
    dynamics_features = [
        "revenue_growth_qoq", "revenue_growth_yoy", "revenue_cagr_2yr",
        "revenue_acceleration", "operating_leverage",
        "gross_margin_trend", "operating_margin_trend", "net_margin_trend",
        "gross_margin_vol", "margin_divergence",
        "dso", "dio", "dpo", "cash_conversion_cycle", "ccc_trend", "dsri",
        "capex_to_depreciation", "capex_intensity", "capex_intensity_trend",
        "rd_intensity", "rd_intensity_trend",
        "sga_efficiency", "sga_efficiency_trend",
        "debt_to_equity_change", "interest_coverage_trend",
        "net_debt_to_ebitda", "debt_maturity_risk",
        "fcf_to_revenue", "fcf_to_revenue_trend",
    ]
    print(f"  ✅ Financial Dynamics: {len(dynamics_features)} features")
    return cq, dynamics_features


# ══════════════════════════════════════════════════════════════════════════
# B. EARNINGS QUALITY & MANIPULATION DETECTION (~20 features)
# ══════════════════════════════════════════════════════════════════════════

def compute_earnings_quality(cq):
    """
    Sloan accruals, Beneish M-Score components, Richardson decomposition,
    CF-earnings divergence.
    """
    print("  📊 B.1 Sloan Accruals (1996)...")
    
    # ── Sloan Total Accruals ──
    # ΔCA - ΔCash - ΔCL + ΔSTD - Dep, scaled by average total assets
    d_ca = diff(cq, "actq", 1)
    d_cash = diff(cq, "cheq", 1)
    d_cl = diff(cq, "lctq", 1)
    d_std = diff(cq, "dlcq", 1)  # short-term debt
    avg_assets = (cq["atq"] + lag(cq, "atq", 1)) / 2
    
    sloan_accruals = (d_ca - d_cash - d_cl + d_std.fillna(0) - cq["dpq"].fillna(0))
    cq["total_accruals"] = safe_div(sloan_accruals.values, avg_assets.values)
    cq["total_accruals"] = cq["total_accruals"].clip(-2, 2)
    
    print("  📊 B.2 Richardson Decomposition (2005)...")
    
    # ── Richardson Decomposition ──
    # Working capital accruals: Δ(CA-Cash) - Δ(CL-STD)
    wc_accruals = (d_ca - d_cash) - (d_cl - d_std.fillna(0))
    cq["working_capital_accruals"] = safe_div(wc_accruals.values, avg_assets.values)
    
    # Non-current operating accruals: Δ(TA-CA-Investments) - Δ(TL-CL-LTD)
    # Approximate: Δ(PPE + Other non-current) - Δ(Non-current liabilities)
    nco_assets = cq["atq"] - cq["actq"]
    nco_liab = cq["ltq"] - cq["lctq"]
    d_nco_a = diff(cq, "atq", 1) - d_ca
    d_nco_l = diff(cq, "ltq", 1) - d_cl
    cq["non_current_accruals"] = safe_div(
        (d_nco_a - d_nco_l).values, avg_assets.values
    )
    
    # Accruals to cash flow ratio
    cq["accruals_to_cash_flow"] = safe_div(
        sloan_accruals.values, np.abs(cq["oancfq"].values) + 1e-6
    )
    cq["accruals_to_cash_flow"] = cq["accruals_to_cash_flow"].clip(-5, 5)
    
    # CF-earnings divergence
    cq["cf_earnings_divergence"] = safe_div(
        (cq["oancfq"] - cq["niq"]).values,
        (np.abs(cq["niq"].values) + 1e-6)
    )
    cq["cf_earnings_divergence"] = cq["cf_earnings_divergence"].clip(-5, 5)
    
    print("  📊 B.3 Beneish M-Score Components (1999)...")
    
    # ── Beneish M-Score — All 8 Components ──
    rev_lag1 = lag(cq, "revtq", 1)
    cogs_lag1 = lag(cq, "cogsq", 1)
    at_lag1 = lag(cq, "atq", 1)
    ppe_lag1 = lag(cq, "ppentq", 1)
    dep_lag1 = lag(cq, "dpq", 1)
    sga_lag1 = lag(cq, "xsgaq", 1)
    
    # DSRI — already computed in dynamics
    # GMI — Gross Margin Index
    gm_prev = safe_div((rev_lag1 - cogs_lag1).values, rev_lag1.values)
    gm_curr = safe_div((cq["revtq"] - cq["cogsq"]).values, cq["revtq"].values)
    cq["beneish_gmi"] = safe_div(gm_prev, gm_curr)
    
    # AQI — Asset Quality Index
    hard_assets = cq["ppentq"].fillna(0) + cq["actq"].fillna(0)
    hard_assets_lag = ppe_lag1.fillna(0) + lag(cq, "actq", 1).fillna(0)
    aq_curr = 1 - safe_div(hard_assets.values, cq["atq"].values)
    aq_prev = 1 - safe_div(hard_assets_lag.values, at_lag1.values)
    cq["beneish_aqi"] = safe_div(aq_curr, aq_prev)
    
    # SGI — Sales Growth Index
    cq["beneish_sgi"] = safe_div(cq["revtq"].values, rev_lag1.values)
    
    # DEPI — Depreciation Index
    dep_rate_prev = safe_div(dep_lag1.values, (dep_lag1 + ppe_lag1).values)
    dep_rate_curr = safe_div(cq["dpq"].values, (cq["dpq"] + cq["ppentq"]).values)
    cq["beneish_depi"] = safe_div(dep_rate_prev, dep_rate_curr)
    
    # SGAI — SGA Index
    sga_rev_prev = safe_div(sga_lag1.values, rev_lag1.values)
    sga_rev_curr = safe_div(cq["xsgaq"].values, cq["revtq"].values)
    cq["beneish_sgai"] = safe_div(sga_rev_curr, sga_rev_prev)
    
    # TATA — Total Accruals to Total Assets
    cq["beneish_tata"] = cq["total_accruals"].copy()  # Already computed
    
    # LVGI — Leverage Index
    lev_curr = safe_div(
        (cq["dlttq"].fillna(0) + cq["dlcq"].fillna(0)).values,
        cq["atq"].values
    )
    lev_prev = safe_div(
        (lag(cq, "dlttq", 1).fillna(0) + lag(cq, "dlcq", 1).fillna(0)).values,
        at_lag1.values
    )
    cq["beneish_lvgi"] = safe_div(lev_curr, lev_prev)
    
    # Composite M-Score
    cq["beneish_m_score"] = (
        -4.84
        + 0.920 * cq["dsri"].fillna(1)
        + 0.528 * cq["beneish_gmi"].fillna(1)
        + 0.404 * cq["beneish_aqi"].fillna(1)
        + 0.892 * cq["beneish_sgi"].fillna(1)
        + 0.115 * cq["beneish_depi"].fillna(1)
        - 0.172 * cq["beneish_sgai"].fillna(1)
        + 4.679 * cq["beneish_tata"].fillna(0)
        - 0.327 * cq["beneish_lvgi"].fillna(1)
    )
    
    print("  📊 B.4 Additional Earnings Quality...")
    
    # ── Additional EQ Features ──
    # Earnings persistence: autocorrelation of EPS over 8Q
    eps = cq["epspxq"].copy()
    cq["earnings_persistence"] = cq.groupby("permno")["epspxq"].transform(
        lambda s: s.rolling(8, min_periods=6).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 6 else np.nan,
            raw=False
        )
    )
    
    # Earnings smoothness: σ(NI) / σ(OCF)
    ni_vol = cq.groupby("permno")["niq"].transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    ocf_vol = cq.groupby("permno")["oancfq"].transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    cq["earnings_smoothness"] = safe_div(ni_vol.values, ocf_vol.values)
    cq["earnings_smoothness"] = cq["earnings_smoothness"].clip(0, 5)
    
    # Net Operating Assets (Hirshleifer 2004)
    noa = (cq["atq"] - cq["cheq"].fillna(0)) - (cq["ltq"] - cq["dlttq"].fillna(0) - cq["dlcq"].fillna(0))
    cq["net_operating_assets"] = safe_div(noa.values, at_lag1.values)
    cq["net_operating_assets"] = cq["net_operating_assets"].clip(-5, 5)
    
    eq_features = [
        "total_accruals", "working_capital_accruals", "non_current_accruals",
        "accruals_to_cash_flow", "cf_earnings_divergence",
        "beneish_gmi", "beneish_aqi", "beneish_sgi",
        "beneish_depi", "beneish_sgai", "beneish_tata", "beneish_lvgi",
        "beneish_m_score",
        "earnings_persistence", "earnings_smoothness", "net_operating_assets",
    ]
    print(f"  ✅ Earnings Quality: {len(eq_features)} features")
    return cq, eq_features


# ══════════════════════════════════════════════════════════════════════════
# C. COMPOSITE SCORING MODELS (~6 features)
# ══════════════════════════════════════════════════════════════════════════

def compute_composite_scores(cq):
    """Piotroski F-Score, Altman Z-Score, Ohlson O-Score, Montier C-Score."""
    print("  📊 C.1 Piotroski F-Score (2000)...")
    
    # ── Piotroski F-Score (9 binary → 0-9) ──
    # Profitability (4):
    roa = safe_div(cq["niq"].values, cq["atq"].values)
    roa_lag = safe_div(lag(cq, "niq", 1).values, lag(cq, "atq", 1).values)
    
    f1 = (roa > 0).astype(np.float32)                        # ROA > 0
    f2 = (cq["oancfq"] > 0).astype(np.float32)               # CFO > 0
    f3 = (roa > roa_lag).astype(np.float32)                   # ROA improving
    f4 = (safe_div(cq["oancfq"].values, cq["atq"].values) > roa).astype(np.float32)  # CFO > ROA
    
    # Leverage/Liquidity (3):
    lt_debt_ratio = safe_div(cq["dlttq"].values, cq["atq"].values)
    lt_debt_ratio_lag = safe_div(lag(cq, "dlttq", 1).values, lag(cq, "atq", 1).values)
    curr_ratio = safe_div(cq["actq"].values, cq["lctq"].values)
    curr_ratio_lag = safe_div(lag(cq, "actq", 1).values, lag(cq, "lctq", 1).values)
    
    f5 = (lt_debt_ratio < lt_debt_ratio_lag).astype(np.float32)   # Leverage declining
    f6 = (curr_ratio > curr_ratio_lag).astype(np.float32)         # Liquidity improving
    # No dilution: shares outstanding not increasing
    csho_change = diff(cq, "cshoq", 1)
    f7 = (csho_change.fillna(0) <= 0).astype(np.float32)         # No new equity
    
    # Efficiency (2):
    gm = safe_div((cq["revtq"] - cq["cogsq"]).values, cq["revtq"].values)
    gm_lag = safe_div(
        (lag(cq, "revtq", 1) - lag(cq, "cogsq", 1)).values,
        lag(cq, "revtq", 1).values
    )
    at_turn = safe_div(cq["revtq"].values, cq["atq"].values)
    at_turn_lag = safe_div(lag(cq, "revtq", 1).values, lag(cq, "atq", 1).values)
    
    f8 = (gm > gm_lag).astype(np.float32)                        # Margin improving
    f9 = (at_turn > at_turn_lag).astype(np.float32)               # Turnover improving
    
    cq["piotroski_f_score"] = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
    
    print("  📊 C.2 Altman Z-Score (1968)...")
    
    # ── Altman Z-Score ──
    # Z = 1.2×WC/TA + 1.4×RE/TA + 3.3×EBIT/TA + 0.6×MVE/TL + 1.0×Sales/TA
    wc = cq["actq"] - cq["lctq"]
    # Retained earnings: approximate as equity - paid-in capital (common stock par)
    # seqq includes retained earnings; use seqq - pstkq as proxy for RE component
    re_proxy = cq["seqq"].fillna(cq["ceqq"]) - cq["pstkq"].fillna(0)
    
    z1 = safe_div(wc.values, cq["atq"].values) * 1.2
    z2 = safe_div(re_proxy.values, cq["atq"].values) * 1.4
    z3 = safe_div(cq["oiadpq"].values, cq["atq"].values) * 3.3
    # MVE component — use market cap from GKX panel later; for now skip or use book
    # We'll merge market_cap from GKX, for now use book equity as placeholder
    z4 = safe_div(cq["seqq"].fillna(cq["ceqq"]).values, cq["ltq"].values) * 0.6
    z5 = safe_div(cq["revtq"].values, cq["atq"].values) * 1.0
    
    cq["altman_z_score"] = z1 + z2 + z3 + z4 + z5
    cq["altman_z_score"] = cq["altman_z_score"].clip(-10, 20)
    
    print("  📊 C.3 Ohlson O-Score (1980)...")
    
    # ── Ohlson O-Score ──
    # Simplified (omit GNP deflator, use log(TA) directly)
    log_ta = np.log(cq["atq"].clip(lower=1).values)
    tl_ta = safe_div(cq["ltq"].values, cq["atq"].values)
    wc_ta = safe_div(wc.values, cq["atq"].values)
    cl_ca = safe_div(cq["lctq"].values, cq["actq"].values)
    x_flag = (cq["ltq"] > cq["atq"]).astype(np.float32).values  # TL > TA
    ni_ta = safe_div(cq["niq"].values, cq["atq"].values)
    ffo_tl = safe_div(cq["oancfq"].values, cq["ltq"].values)
    # Y = 1 if net loss for last 2 quarters
    ni_lag1 = lag(cq, "niq", 1)
    y_flag = ((cq["niq"] < 0) & (ni_lag1 < 0)).astype(np.float32).values
    # NI change
    ni_change = safe_div(
        (cq["niq"] - ni_lag1).values,
        (np.abs(cq["niq"].values) + np.abs(ni_lag1.values) + 1e-6)
    )
    
    cq["ohlson_o_score"] = (
        -1.32
        - 0.407 * log_ta
        + 6.03 * tl_ta
        - 1.43 * wc_ta
        + 0.076 * cl_ca
        - 1.72 * x_flag
        - 2.37 * ni_ta
        - 1.83 * ffo_tl
        + 0.285 * y_flag
        - 0.521 * ni_change
    )
    cq["ohlson_o_score"] = cq["ohlson_o_score"].clip(-10, 10)
    
    print("  📊 C.4 Montier C-Score (6-point)...")
    
    # ── Montier C-Score (accounting quality red flags) ──
    # c1: growing net assets (accruals increasing)
    accruals_lag1 = lag(cq, "total_accruals", 1)
    c1 = (cq["total_accruals"] > accruals_lag1).astype(np.float32)
    
    # c2: cash from operations declining vs net income
    ocf_ni = safe_div(cq["oancfq"].values, (np.abs(cq["niq"].values) + 1e-6))
    ocf_ni_lag = safe_div(
        lag(cq, "oancfq", 1).values,
        (np.abs(lag(cq, "niq", 1).values) + 1e-6)
    )
    c2 = (ocf_ni < ocf_ni_lag).astype(np.float32)
    
    # c3: receivables growing faster than revenue
    rect_growth = pct_change_qoq(cq, "rectq")
    c3 = (rect_growth > cq["revenue_growth_qoq"]).astype(np.float32)
    
    # c4: inventory growing faster than COGS
    inv_growth = pct_change_qoq(cq, "invtq")
    cogs_growth = pct_change_qoq(cq, "cogsq")
    c4 = (inv_growth > cogs_growth).astype(np.float32)
    
    # c5: SGA growing faster than revenue
    sga_growth = pct_change_qoq(cq, "xsgaq")
    c5 = (sga_growth > cq["revenue_growth_qoq"]).astype(np.float32)
    
    # c6: depreciation declining as % of PP&E
    dep_ppe = safe_div(cq["dpq"].values, cq["ppentq"].values)
    dep_ppe_lag = safe_div(lag(cq, "dpq", 1).values, lag(cq, "ppentq", 1).values)
    c6 = (dep_ppe < dep_ppe_lag).astype(np.float32)
    
    cq["montier_c_score"] = c1 + c2 + c3 + c4 + c5 + c6
    
    composite_features = [
        "piotroski_f_score", "altman_z_score",
        "ohlson_o_score", "montier_c_score",
    ]
    print(f"  ✅ Composite Scores: {len(composite_features)} features")
    return cq, composite_features


# ══════════════════════════════════════════════════════════════════════════
# D. ANALYST DYNAMICS (~12 features from IBES)
# ══════════════════════════════════════════════════════════════════════════

def compute_analyst_dynamics(ibes_summary, ibes_actuals, ibes_link):
    """
    EPS revisions, dispersion, SUE, beat/miss streaks from IBES.
    Returns a panel at (permno, month) grain.
    """
    print("  📊 D.1 Loading & Linking IBES data...")
    
    # ── Link IBES ticker → permno ──
    link = ibes_link[["ticker", "permno"]].drop_duplicates("ticker")
    
    # ── IBES Summary: focus on FY1 EPS (fpi='1') ──
    fy1 = ibes_summary[ibes_summary["fpi"] == "1"].copy()
    fy1 = fy1.merge(link, on="ticker", how="inner")
    fy1["statpers"] = pd.to_datetime(fy1["statpers"])
    fy1["month"] = fy1["statpers"].dt.to_period("M")
    
    # Keep latest summary per permno-month
    fy1 = fy1.sort_values("statpers").drop_duplicates(
        subset=["permno", "month"], keep="last"
    )
    
    print(f"    FY1 EPS linked: {len(fy1):,} rows, {fy1['permno'].nunique():,} stocks")
    
    print("  📊 D.2 EPS Revision Dynamics...")
    
    # ── Revision features ──
    # Net revisions: (up - down) / total
    fy1["revision_breadth"] = safe_div(
        (fy1["numup"] - fy1["numdown"]).values,
        fy1["numest"].values
    )
    
    # EPS estimate momentum: change in mean estimate over 1 month
    fy1 = fy1.sort_values(["permno", "statpers"]).reset_index(drop=True)
    mean_lag1 = fy1.groupby("permno")["meanest"].shift(1)
    fy1["eps_revision_1m"] = safe_div(
        (fy1["meanest"] - mean_lag1).values,
        (np.abs(mean_lag1.values) + 1e-4)
    )
    fy1["eps_revision_1m"] = fy1["eps_revision_1m"].clip(-2, 2)
    
    # 3-month revision
    mean_lag3 = fy1.groupby("permno")["meanest"].shift(3)
    fy1["eps_revision_3m"] = safe_div(
        (fy1["meanest"] - mean_lag3).values,
        (np.abs(mean_lag3.values) + 1e-4)
    )
    fy1["eps_revision_3m"] = fy1["eps_revision_3m"].clip(-2, 2)
    
    print("  📊 D.3 Dispersion & Estimate Properties...")
    
    # ── Dispersion ──
    fy1["eps_dispersion"] = safe_div(
        fy1["stdev"].values,
        (np.abs(fy1["meanest"].values) + 1e-4)
    )
    fy1["eps_dispersion"] = fy1["eps_dispersion"].clip(0, 5)
    
    # Dispersion trend
    fy1["eps_dispersion_trend"] = fy1.groupby("permno")["eps_dispersion"].transform(
        lambda s: s.diff(1)
    )
    
    # Estimate momentum (same as revision but 3m window — kept for compatibility)
    fy1["eps_estimate_momentum"] = fy1["eps_revision_3m"].copy()
    
    # Number of analysts (coverage proxy)
    fy1["num_analysts_fy1"] = fy1["numest"].astype(np.float32)
    
    print("  📊 D.4 SUE & Beat/Miss from Actuals...")
    
    # ── SUE from IBES Actuals ──
    act_q = ibes_actuals[ibes_actuals["pdicity"] == "QTR"].copy()
    act_q = act_q.merge(link, on="ticker", how="inner")
    act_q["anndats"] = pd.to_datetime(act_q["anndats"])
    act_q["pends"] = pd.to_datetime(act_q["pends"])
    act_q = act_q.sort_values(["permno", "pends"]).drop_duplicates(
        subset=["permno", "pends"], keep="last"
    )
    
    # Merge actuals with most recent consensus before announcement
    # For each actual, find the latest fy1 estimate
    act_q["month"] = act_q["anndats"].dt.to_period("M")
    
    # Merge latest consensus before announcement date
    # Use merge_asof on statpers <= anndats
    fy1_for_sue = fy1[["permno", "statpers", "meanest", "stdev"]].copy()
    fy1_for_sue = fy1_for_sue.rename(columns={"statpers": "est_date"})
    # Drop rows with null merge keys before merge_asof
    act_q = act_q.dropna(subset=["anndats", "permno"])
    fy1_for_sue = fy1_for_sue.dropna(subset=["est_date", "permno"])
    act_q_sorted = act_q.sort_values("anndats")
    fy1_sorted = fy1_for_sue.sort_values("est_date")
    
    sue_df = pd.merge_asof(
        act_q_sorted[["permno", "anndats", "pends", "value"]],
        fy1_sorted,
        left_on="anndats", right_on="est_date",
        by="permno",
        direction="backward"
    )
    
    # SUE = (Actual - Estimate) / σ(estimates)
    sue_df["sue_ibes"] = safe_div(
        (sue_df["value"] - sue_df["meanest"]).values,
        (sue_df["stdev"].values + 1e-4)
    )
    sue_df["sue_ibes"] = sue_df["sue_ibes"].clip(-5, 5)
    
    # Beat/miss indicator
    sue_df["beat_miss"] = np.sign(sue_df["value"] - sue_df["meanest"]).astype(np.float32)
    
    # Beat/miss streak: consecutive beats or misses
    sue_df = sue_df.sort_values(["permno", "pends"]).reset_index(drop=True)
    
    def _streak(s):
        """Count consecutive same-sign values."""
        streak = np.zeros(len(s), dtype=np.float32)
        for i in range(len(s)):
            if i == 0 or np.isnan(s.iloc[i]):
                streak[i] = s.iloc[i] if not np.isnan(s.iloc[i]) else 0
            elif np.sign(s.iloc[i]) == np.sign(s.iloc[i-1]) and not np.isnan(s.iloc[i-1]):
                streak[i] = streak[i-1] + np.sign(s.iloc[i])
            else:
                streak[i] = np.sign(s.iloc[i])
        return pd.Series(streak, index=s.index)
    
    sue_df["beat_miss_streak"] = sue_df.groupby("permno")["beat_miss"].transform(_streak)
    sue_df["beat_miss_streak"] = sue_df["beat_miss_streak"].clip(-8, 8)
    
    # Map SUE back to month grain for merge
    sue_df["month"] = sue_df["anndats"].dt.to_period("M")
    sue_monthly = sue_df.groupby(["permno", "month"]).agg({
        "sue_ibes": "last",
        "beat_miss_streak": "last",
    }).reset_index()
    
    # ── Combine all IBES features into monthly panel ──
    analyst_panel = fy1[[
        "permno", "month",
        "revision_breadth", "eps_revision_1m", "eps_revision_3m",
        "eps_dispersion", "eps_dispersion_trend", "eps_estimate_momentum",
        "num_analysts_fy1",
    ]].copy()
    
    # Merge SUE
    analyst_panel = analyst_panel.merge(
        sue_monthly, on=["permno", "month"], how="left"
    )
    
    # Convert month period to date for merge with GKX
    # Use how="start" then MonthEnd to get clean midnight timestamps matching GKX
    analyst_panel["date"] = analyst_panel["month"].dt.to_timestamp(how="start")
    analyst_panel["date"] = analyst_panel["date"] + pd.offsets.MonthEnd(0)
    analyst_panel = analyst_panel.drop(columns=["month"])
    
    analyst_features = [
        "revision_breadth", "eps_revision_1m", "eps_revision_3m",
        "eps_dispersion", "eps_dispersion_trend", "eps_estimate_momentum",
        "num_analysts_fy1", "sue_ibes", "beat_miss_streak",
    ]
    print(f"  ✅ Analyst Dynamics: {len(analyst_features)} features")
    return analyst_panel, analyst_features


# ══════════════════════════════════════════════════════════════════════════
# E. INSTITUTIONAL & INSIDER SIGNALS (~10 features)
# ══════════════════════════════════════════════════════════════════════════

def compute_insider_signals(insider_df, ibes_link, cusip_map=None):
    """
    Insider buy ratio, cluster buying, CEO buys from insider transactions.
    Returns monthly panel at (permno, date).
    """
    print("  📊 E.1 Insider Transaction Signals...")
    
    # Link insider ticker → permno via IBES link
    link = ibes_link[["ticker", "permno"]].drop_duplicates("ticker")
    ins = insider_df.merge(link, on="ticker", how="inner")
    
    # Also link via cusip if available (insider_trading has cusip6)
    if cusip_map is not None and "cusip6" in insider_df.columns:
        unlinked = insider_df[~insider_df["ticker"].isin(link["ticker"])]
        if len(unlinked) > 0:
            # cusip6 in insider is 6-digit; cusip_map has 8-digit cusip
            cusip6_map = cusip_map.copy()
            cusip6_map["cusip6"] = cusip6_map["cusip"].str[:6]
            cusip6_map = cusip6_map[["cusip6", "permno"]].drop_duplicates("cusip6")
            ins_cusip = unlinked.merge(cusip6_map, on="cusip6", how="inner")
            if len(ins_cusip) > 0:
                print(f"    Additional {len(ins_cusip):,} insider rows linked via cusip")
                ins = pd.concat([ins, ins_cusip], ignore_index=True)
    
    # Parse dates
    ins["trade_date"] = pd.to_datetime(ins["trade_date"], errors="coerce")
    ins = ins.dropna(subset=["trade_date"])
    ins["month"] = ins["trade_date"].dt.to_period("M")
    
    # Filter to open market transactions (Form 4, type P=purchase, S=sale)
    ins = ins[ins["formtype"].isin(["4"])].copy()
    
    # Classify buys vs sells
    ins["is_buy"] = (ins["acqdisp"] == "A").astype(int)
    ins["is_sell"] = (ins["acqdisp"] == "D").astype(int)
    ins["is_ceo_buy"] = ((ins["acqdisp"] == "A") & (ins["rolecode1"] == "CEO")).astype(int)
    
    # Aggregate per stock-month
    monthly = ins.groupby(["permno", "month"]).agg(
        insider_num_buys_raw=("is_buy", "sum"),
        insider_num_sells_raw=("is_sell", "sum"),
        insider_total_txns=("is_buy", "count"),
        insider_ceo_buy_raw=("is_ceo_buy", "sum"),
    ).reset_index()
    
    # Count unique buyers (distinct insiders who bought)
    unique_buyers = ins[ins["is_buy"] == 1].groupby(
        ["permno", "month"]
    ).size().reset_index(name="insider_unique_buyers")
    monthly = monthly.merge(unique_buyers, on=["permno", "month"], how="left")
    monthly["insider_unique_buyers"] = monthly["insider_unique_buyers"].fillna(0)
    
    # Rolling 6-month signals (more robust than single month)
    monthly = monthly.sort_values(["permno", "month"]).reset_index(drop=True)
    
    # Buy ratio: buys / (buys + sells) over trailing 6 months
    for col in ["insider_num_buys_raw", "insider_num_sells_raw", "insider_ceo_buy_raw"]:
        monthly[f"{col}_6m"] = monthly.groupby("permno")[col].transform(
            lambda s: s.rolling(6, min_periods=1).sum()
        )
    
    total_6m = monthly["insider_num_buys_raw_6m"] + monthly["insider_num_sells_raw_6m"]
    monthly["insider_buy_ratio_6m"] = safe_div(
        monthly["insider_num_buys_raw_6m"].values, total_6m.values
    )
    
    # Cluster buying: multiple insiders buying in same month
    monthly["insider_cluster_buy"] = (monthly["insider_unique_buyers"] >= 3).astype(np.float32)
    
    # CEO buy signal
    monthly["insider_ceo_buy"] = (monthly["insider_ceo_buy_raw_6m"] > 0).astype(np.float32)
    
    # Convert to date — use how="start" then MonthEnd for clean midnight timestamps
    monthly["date"] = monthly["month"].dt.to_timestamp(how="start")
    monthly["date"] = monthly["date"] + pd.offsets.MonthEnd(0)
    
    insider_panel = monthly[[
        "permno", "date",
        "insider_buy_ratio_6m", "insider_cluster_buy", "insider_ceo_buy",
    ]].copy()
    
    insider_features = [
        "insider_buy_ratio_6m", "insider_cluster_buy", "insider_ceo_buy",
    ]
    print(f"  ✅ Insider Signals: {len(insider_features)} features")
    return insider_panel, insider_features


def compute_institutional_signals(inst_df, gkx_cusip_map):
    """
    Institutional ownership dynamics from 13F filings.
    Returns quarterly panel at (permno, date).
    
    NOTE: inst_df is very large (124M rows). We process in aggregated form.
    """
    print("  📊 E.2 Institutional Holdings Signals...")
    
    # Parse dates
    inst_df["fdate"] = pd.to_datetime(inst_df["fdate"], errors="coerce")
    inst_df = inst_df.dropna(subset=["fdate", "cusip"])
    
    # Aggregate per cusip-quarter
    print("    Aggregating 124M rows by cusip-quarter...")
    quarterly = inst_df.groupby(["cusip", "fdate"]).agg(
        total_shares=("shares", "sum"),
        num_institutions=("mgrno", "nunique"),
        # HHI: sum of squared shares (compute after getting total)
    ).reset_index()
    
    # Compute HHI from individual holdings
    # For memory efficiency, compute in chunks
    print("    Computing institutional concentration (HHI)...")
    inst_df["share_sq"] = inst_df["shares"] ** 2
    hhi_df = inst_df.groupby(["cusip", "fdate"])["share_sq"].sum().reset_index()
    hhi_df = hhi_df.rename(columns={"share_sq": "sum_shares_sq"})
    quarterly = quarterly.merge(hhi_df, on=["cusip", "fdate"], how="left")
    quarterly["inst_hhi"] = safe_div(
        quarterly["sum_shares_sq"].values,
        (quarterly["total_shares"].values ** 2 + 1e-6)
    )
    del inst_df, hhi_df
    gc.collect()
    
    # Map cusip → permno using the training panel mapping
    if gkx_cusip_map is not None and len(gkx_cusip_map) > 0:
        # cusip in institutional holdings is 8-digit
        quarterly = quarterly.merge(gkx_cusip_map, on="cusip", how="inner")
    else:
        print("    ⚠️ No cusip→permno mapping available, skipping institutional features")
        return pd.DataFrame(columns=["permno", "date"]), []
    
    # Sort and compute dynamics
    quarterly = quarterly.sort_values(["permno", "fdate"]).reset_index(drop=True)
    
    # QoQ changes
    quarterly["inst_ownership_change"] = quarterly.groupby("permno")["total_shares"].transform(
        lambda s: s.pct_change(1)
    ).clip(-2, 2)
    
    quarterly["num_institutions_chg"] = quarterly.groupby("permno")["num_institutions"].transform(
        lambda s: s.diff(1)
    )
    
    # Institutional breadth: change in number of institutions (proxy for buy vs sell)
    quarterly["inst_breadth"] = quarterly["num_institutions_chg"].apply(
        lambda x: np.sign(x) if not np.isnan(x) else 0
    ).astype(np.float32)
    
    # Rename for merge
    quarterly["date"] = quarterly["fdate"] + pd.offsets.MonthEnd(0)
    
    inst_panel = quarterly[[
        "permno", "date",
        "inst_hhi", "inst_ownership_change", "inst_breadth",
    ]].copy()
    
    inst_features = [
        "inst_hhi", "inst_ownership_change", "inst_breadth",
    ]
    print(f"  ✅ Institutional Signals: {len(inst_features)} features")
    return inst_panel, inst_features


# ══════════════════════════════════════════════════════════════════════════
# F. INDUSTRY-RELATIVE FEATURES (~10 features)
# ══════════════════════════════════════════════════════════════════════════

def compute_industry_relative(cq_merged, gkx):
    """
    Compute features relative to FF49 industry median.
    Must be called after Compustat features are merged into GKX.
    """
    print("  📊 F. Industry-Relative Features...")
    
    if "siccd" not in gkx.columns:
        print("    ⚠️ No siccd in GKX panel, skipping industry-relative features")
        return gkx, []
    
    # Import FF49 mapping — handle running from different directories
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from step5_gkx_panel import sic_to_ff49
    except ImportError:
        print("    ⚠️ Cannot import sic_to_ff49, using inline implementation")
        # Inline fallback: simplified FF49 mapper using siccd ranges
        def sic_to_ff49(sic):
            if pd.isna(sic) or sic == 0:
                return 0
            return int(sic) // 100  # Rough approximation by 2-digit SIC
    
    if "ff49_industry" not in gkx.columns:
        gkx["ff49_industry"] = gkx["siccd"].apply(sic_to_ff49)
    
    industry_relative_features = []
    
    # Features to compare against industry
    feature_pairs = [
        ("roa", "roa_vs_industry"),
        ("roe", "roe_vs_industry"),
        ("gpm", "gpm_vs_industry"),
        ("npm", "npm_vs_industry"),
        ("revenue_growth_yoy", "growth_vs_industry"),
        ("total_accruals", "accruals_vs_industry"),
        ("piotroski_f_score", "piotroski_vs_industry"),
        ("net_debt_to_ebitda", "leverage_vs_industry"),
    ]
    
    for feat, out_name in feature_pairs:
        if feat in gkx.columns:
            # Compute industry-month median
            industry_med = gkx.groupby(["ff49_industry", "date"])[feat].transform("median")
            gkx[out_name] = gkx[feat] - industry_med
            industry_relative_features.append(out_name)
    
    # Revenue share within industry (competitive position proxy)
    if "revtq" in gkx.columns:
        industry_total = gkx.groupby(["ff49_industry", "date"])["revtq"].transform("sum")
        gkx["revenue_share"] = safe_div(gkx["revtq"].values, industry_total.values)
        gkx["revenue_share_trend"] = gkx.groupby("permno")["revenue_share"].transform(
            lambda s: s.diff(1)
        )
        industry_relative_features.extend(["revenue_share", "revenue_share_trend"])
    
    # Clean up temp column
    if "ff49_industry" in gkx.columns:
        gkx = gkx.drop(columns=["ff49_industry"])
    
    print(f"  ✅ Industry-Relative: {len(industry_relative_features)} features")
    return gkx, industry_relative_features


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("PHASE 3 — STEP 5b: FINANCIAL STATEMENT INTELLIGENCE ENGINE (Level 3)")
    print("=" * 78)
    start = time.time()
    
    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Load & Link Data Sources
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("STAGE 1: LOADING DATA SOURCES")
    print("─" * 70)
    
    # ── Load GKX Panel ──
    print("\n📂 Loading GKX Panel...")
    gkx_path = os.path.join(DATA_DIR, "gkx_panel.parquet")
    gkx = pd.read_parquet(gkx_path)
    gkx["date"] = pd.to_datetime(gkx["date"])
    existing_cols = set(gkx.columns)
    print(f"  GKX: {len(gkx):,} rows × {gkx.shape[1]} cols")
    print(f"  Date range: {gkx['date'].min().date()} to {gkx['date'].max().date()}")
    print(f"  Stocks: {gkx['permno'].nunique():,}")
    
    # ── Load CRSP-Compustat Link ──
    print("\n📂 Loading CRSP-Compustat Link...")
    link_path = os.path.join(DATA_DIR, "crsp_compustat_link.parquet")
    cclink = pd.read_parquet(link_path)
    # gvkey → lpermno mapping (keep best links)
    cclink = cclink[["gvkey", "lpermno"]].drop_duplicates("gvkey")
    cclink = cclink.rename(columns={"lpermno": "permno"})
    cclink["permno"] = cclink["permno"].astype(np.int64)
    print(f"  Link: {len(cclink):,} gvkey→permno mappings")
    
    # ── Load IBES-CRSP Link ──
    print("\n📂 Loading IBES-CRSP Link...")
    ibes_link_path = os.path.join(DATA_DIR, "ibes_crsp_link.parquet")
    ibes_link = pd.read_parquet(ibes_link_path)
    # Clean: drop rows without permno, convert to int
    ibes_link = ibes_link.dropna(subset=["permno"])
    ibes_link["permno"] = ibes_link["permno"].astype(np.int64)
    print(f"  IBES Link: {len(ibes_link):,} ticker→permno mappings")
    
    # ── Load Compustat Quarterly ──
    print("\n📂 Loading Compustat Quarterly...")
    cq_path = os.path.join(DATA_DIR, "compustat_quarterly.parquet")
    cq = pd.read_parquet(cq_path)
    cq["datadate"] = pd.to_datetime(cq["datadate"])
    # Merge permno via link
    cq = cq.merge(cclink, on="gvkey", how="inner")
    cq = cq.sort_values(["permno", "datadate"]).reset_index(drop=True)
    print(f"  Compustat Q (linked): {len(cq):,} rows, {cq['permno'].nunique():,} firms")
    
    # ── Load IBES ──
    print("\n📂 Loading IBES Summary & Actuals...")
    ibes_summary = pd.read_parquet(os.path.join(DATA_DIR, "ibes_summary.parquet"))
    ibes_actuals = pd.read_parquet(os.path.join(DATA_DIR, "ibes_actuals.parquet"))
    print(f"  IBES Summary: {len(ibes_summary):,} rows")
    print(f"  IBES Actuals: {len(ibes_actuals):,} rows")
    
    # ── Load Insider Trading ──
    print("\n📂 Loading Insider Trading...")
    insider = pd.read_parquet(os.path.join(DATA_DIR, "insider_trading.parquet"))
    print(f"  Insider: {len(insider):,} rows")
    
    # ── Load Institutional Holdings ──
    print("\n📂 Loading Institutional Holdings...")
    inst_path = os.path.join(DATA_DIR, "institutional_holdings.parquet")
    inst = pd.read_parquet(inst_path)
    print(f"  Institutional: {len(inst):,} rows")
    
    # ── Build cusip→permno mapping from Compustat Security + CRSP link ──
    print("\n📂 Building cusip→permno mapping...")
    cusip_map = None
    sec_path = os.path.join(DATA_DIR, "compustat_security.parquet")
    if os.path.exists(sec_path):
        sec = pd.read_parquet(sec_path, columns=["gvkey", "cusip"]).dropna()
        sec["cusip"] = sec["cusip"].str[:8]
        sec = sec.merge(cclink, on="gvkey", how="inner")
        cusip_map = sec[["cusip", "permno"]].drop_duplicates("cusip")
        cusip_map["permno"] = cusip_map["permno"].astype(np.int64)
        print(f"  cusip→permno: {len(cusip_map):,} mappings")
        del sec
    elif "ncusip" in ibes_link.columns:
        cusip_map = ibes_link[["ncusip", "permno"]].dropna().rename(
            columns={"ncusip": "cusip"}
        )
        cusip_map["cusip"] = cusip_map["cusip"].str[:8]
        cusip_map = cusip_map.drop_duplicates("cusip")
        print(f"  cusip→permno (from IBES): {len(cusip_map):,} mappings")
    else:
        print("  ⚠️ No cusip→permno mapping available")
    
    gc.collect()
    
    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Compute Compustat-Based Features (A, B, C)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("STAGE 2: COMPUTING COMPUSTAT FEATURES")
    print("─" * 70)
    
    # A. Financial Dynamics
    print("\n🔬 [A] Financial Dynamics (~29 features)")
    cq, dynamics_feats = compute_financial_dynamics(cq)
    gc.collect()
    
    # B. Earnings Quality
    print("\n🔬 [B] Earnings Quality (~17 features)")
    cq, eq_feats = compute_earnings_quality(cq)
    gc.collect()
    
    # C. Composite Scores
    print("\n🔬 [C] Composite Scoring Models (~4 features)")
    cq, composite_feats = compute_composite_scores(cq)
    gc.collect()
    
    all_compustat_feats = dynamics_feats + eq_feats + composite_feats
    
    # ── Map Compustat datadate → GKX month ──
    # GKX panel dates are month-end; Compustat datadate is fiscal quarter end.
    # We lag by 3 months to ensure data is available (filing delay).
    print("\n📅 Aligning Compustat dates to GKX (3-month lag for filing delay)...")
    
    cq["report_date"] = cq["datadate"] + pd.DateOffset(months=3)
    cq["report_date"] = cq["report_date"] + pd.offsets.MonthEnd(0)
    
    # De-duplicate: keep latest report per permno-month
    cq = cq.sort_values(["permno", "report_date", "datadate"]).drop_duplicates(
        subset=["permno", "report_date"], keep="last"
    )
    
    # Select columns for merge
    # Avoid overwriting existing GKX columns
    new_compustat_feats = [f for f in all_compustat_feats if f not in existing_cols]
    skipped = [f for f in all_compustat_feats if f in existing_cols]
    if skipped:
        print(f"  ⚠️ Skipping {len(skipped)} features already in GKX: {skipped[:5]}...")
    
    merge_cols = ["permno", "report_date"] + new_compustat_feats
    # Also carry revtq for industry-relative features
    extra_carry = [c for c in ["revtq", "cogsq"] if c not in existing_cols]
    merge_cols.extend(extra_carry)
    merge_cols = [c for c in merge_cols if c in cq.columns]
    cq_merge = cq[merge_cols].rename(columns={"report_date": "date"})
    
    # Forward-fill Compustat features: quarterly data → monthly panel
    # For each permno, carry forward the latest quarterly observation
    print("  Forward-filling quarterly → monthly...")
    gkx = gkx.merge(cq_merge, on=["permno", "date"], how="left")
    
    # Forward fill within firm (quarterly data only updates every 3 months)
    ff_cols = [c for c in new_compustat_feats + extra_carry if c in gkx.columns]
    gkx = gkx.sort_values(["permno", "date"]).reset_index(drop=True)
    for col in ff_cols:
        gkx[col] = gkx.groupby("permno")[col].ffill(limit=5)
    
    print(f"  Merged {len(new_compustat_feats)} Compustat features into GKX")
    
    del cq, cq_merge
    gc.collect()
    
    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3: IBES Analyst Dynamics (D)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("STAGE 3: COMPUTING ANALYST DYNAMICS (IBES)")
    print("─" * 70)
    
    print("\n🔬 [D] Analyst Dynamics (~9 features)")
    analyst_panel, analyst_feats = compute_analyst_dynamics(
        ibes_summary, ibes_actuals, ibes_link
    )
    
    del ibes_summary, ibes_actuals
    gc.collect()
    
    # Merge analyst features
    new_analyst_feats = [f for f in analyst_feats if f not in existing_cols]
    if not new_analyst_feats:
        new_analyst_feats = []
    if new_analyst_feats:
        analyst_merge = analyst_panel[["permno", "date"] + new_analyst_feats].copy()
        analyst_merge["date"] = pd.to_datetime(analyst_merge["date"])
        
        # Align to month-end
        analyst_merge["date"] = analyst_merge["date"] + pd.offsets.MonthEnd(0)
        analyst_merge = analyst_merge.drop_duplicates(
            subset=["permno", "date"], keep="last"
        )
        
        gkx = gkx.merge(analyst_merge, on=["permno", "date"], how="left")
        
        # Forward fill analyst features (updated monthly but may have gaps)
        for col in new_analyst_feats:
            if col in gkx.columns:
                gkx[col] = gkx.groupby("permno")[col].ffill(limit=3)
        
        print(f"  Merged {len(new_analyst_feats)} analyst features into GKX")
    
    del analyst_panel
    gc.collect()
    
    # ══════════════════════════════════════════════════════════════════════
    # STAGE 4: Insider & Institutional Signals (E)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("STAGE 4: COMPUTING INSIDER & INSTITUTIONAL SIGNALS")
    print("─" * 70)
    
    # ── Insider ──
    print("\n🔬 [E.1] Insider Signals (~3 features)")
    insider_panel, insider_feats = compute_insider_signals(insider, ibes_link, cusip_map)
    
    new_insider_feats = [f for f in insider_feats if f not in existing_cols]
    if not new_insider_feats:
        new_insider_feats = []
    if new_insider_feats:
        insider_merge = insider_panel[["permno", "date"] + new_insider_feats].copy()
        insider_merge["date"] = pd.to_datetime(insider_merge["date"])
        insider_merge["date"] = insider_merge["date"] + pd.offsets.MonthEnd(0)
        insider_merge = insider_merge.drop_duplicates(
            subset=["permno", "date"], keep="last"
        )
        
        gkx = gkx.merge(insider_merge, on=["permno", "date"], how="left")
        
        for col in new_insider_feats:
            if col in gkx.columns:
                gkx[col] = gkx.groupby("permno")[col].ffill(limit=6)
        
        print(f"  Merged {len(new_insider_feats)} insider features into GKX")
    
    del insider, insider_panel
    gc.collect()
    
    # ── Institutional ──
    print("\n🔬 [E.2] Institutional Signals (~3 features)")
    inst_panel, inst_feats = compute_institutional_signals(inst, cusip_map)
    
    new_inst_feats = [f for f in inst_feats if f not in existing_cols]
    if not new_inst_feats:
        new_inst_feats = []
    if new_inst_feats:
        inst_merge = inst_panel[["permno", "date"] + new_inst_feats].copy()
        inst_merge["date"] = pd.to_datetime(inst_merge["date"])
        inst_merge["date"] = inst_merge["date"] + pd.offsets.MonthEnd(0)
        inst_merge = inst_merge.drop_duplicates(
            subset=["permno", "date"], keep="last"
        )
        
        gkx = gkx.merge(inst_merge, on=["permno", "date"], how="left")
        
        for col in new_inst_feats:
            if col in gkx.columns:
                gkx[col] = gkx.groupby("permno")[col].ffill(limit=4)
        
        print(f"  Merged {len(new_inst_feats)} institutional features into GKX")
    
    del inst, inst_panel
    gc.collect()
    
    # ══════════════════════════════════════════════════════════════════════
    # STAGE 5: Industry-Relative Features (F)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("STAGE 5: COMPUTING INDUSTRY-RELATIVE FEATURES")
    print("─" * 70)
    
    print("\n🔬 [F] Industry-Relative Features")
    gkx, industry_feats = compute_industry_relative(None, gkx)
    gc.collect()
    
    # ══════════════════════════════════════════════════════════════════════
    # STAGE 6: Final Cleanup & Save
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("STAGE 6: FINAL CLEANUP & SAVE")
    print("─" * 70)
    
    # Collect all new features
    all_new_features = (
        new_compustat_feats
        + new_analyst_feats
        + new_insider_feats
        + new_inst_feats
        + industry_feats
    )
    
    # Remove any temporary columns
    temp_cols = ["revtq", "cogsq"]
    for tc in temp_cols:
        if tc in gkx.columns and tc not in existing_cols:
            gkx = gkx.drop(columns=[tc])
            if tc in all_new_features:
                all_new_features.remove(tc)
    
    # Clip extreme values on all new features
    print("\n  Clipping extreme values...")
    for col in all_new_features:
        if col in gkx.columns and gkx[col].dtype in [np.float32, np.float64]:
            # Replace inf/-inf with NaN before clipping
            gkx[col] = gkx[col].replace([np.inf, -np.inf], np.nan)
            p01 = gkx[col].quantile(0.001)
            p99 = gkx[col].quantile(0.999)
            gkx[col] = gkx[col].clip(p01, p99)
    
    # Downcast to float32
    print("  Downcasting to float32...")
    float64_cols = gkx.select_dtypes(include=["float64"]).columns
    gkx[float64_cols] = gkx[float64_cols].astype(np.float32)
    
    # Coverage report
    print("\n  📊 FEATURE COVERAGE REPORT:")
    print(f"  {'Feature':<35} {'Coverage %':>10} {'Mean':>10} {'Std':>10}")
    print(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*10}")
    for feat in sorted(all_new_features):
        if feat in gkx.columns:
            cov = gkx[feat].notna().mean() * 100
            mn = gkx[feat].mean()
            sd = gkx[feat].std()
            print(f"  {feat:<35} {cov:>9.1f}% {mn:>10.4f} {sd:>10.4f}")
    
    # Save
    print("\n💾 SAVING ENRICHED GKX PANEL...")
    output_path = os.path.join(DATA_DIR, "gkx_panel.parquet")
    
    # Backup original
    backup_path = os.path.join(DATA_DIR, "gkx_panel_pre_level3.parquet")
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(output_path, backup_path)
        print(f"  📦 Backed up original to gkx_panel_pre_level3.parquet")
    
    gkx.to_parquet(output_path, index=False, engine="pyarrow")
    file_size = os.path.getsize(output_path) / (1024 ** 3)
    
    # Upload to S3 (non-blocking, best-effort)
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", output_path,
             f"s3://{S3_BUCKET}/features/gkx_panel.parquet"],
            capture_output=True, timeout=120,
        )
        s3_ok = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        s3_ok = False
    
    elapsed = time.time() - start
    
    print(f"\n{'=' * 78}")
    print(f"LEVEL 3: FINANCIAL STATEMENT INTELLIGENCE — COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Panel size:      {len(gkx):,} rows × {gkx.shape[1]} cols")
    print(f"  Original cols:   {len(existing_cols)}")
    print(f"  New features:    {len(all_new_features)}")
    print(f"  Total features:  {gkx.shape[1]}")
    print(f"  Feature groups:")
    print(f"    [A] Financial Dynamics:     {len([f for f in new_compustat_feats if f in dynamics_feats])}")
    print(f"    [B] Earnings Quality:       {len([f for f in new_compustat_feats if f in eq_feats])}")
    print(f"    [C] Composite Scores:       {len([f for f in new_compustat_feats if f in composite_feats])}")
    print(f"    [D] Analyst Dynamics:       {len(new_analyst_feats)}")
    print(f"    [E] Insider & Institutional: {len(new_insider_feats) + len(new_inst_feats)}")
    print(f"    [F] Industry-Relative:      {len(industry_feats)}")
    print(f"  File size:       {file_size:.2f} GB")
    print(f"  Time elapsed:    {elapsed/60:.1f} min")
    print(f"  ✅ Saved to {output_path}")
    if s3_ok:
        print(f"  ✅ Uploaded to S3")
    else:
        print(f"  ⚠️ S3 upload skipped (AWS CLI not configured or timed out)")
    print(f"\n  🚀 NEXT: Re-run step6b_ensemble.py to measure IC improvement")
    print(f"     Expected IC: 0.05-0.07 (up from 0.0136 baseline)")


if __name__ == "__main__":
    main()
