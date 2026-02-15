"""
═══════════════════════════════════════════════════════════════
PATH A AUDIT — PERSONAL ALPHA ACCOUNT
═══════════════════════════════════════════════════════════════
The "can I trade this next week?" analysis.

Universe: Long-only, market cap > $500M
Rebalancing: Monthly (then test bimonthly)
Costs: 10-25 bps (large/mid cap, no borrow costs)
VIX scaling: 100% at VIX<25, 50% at VIX 25-35, 20% at VIX>35

Tests:
  A1  Universe statistics (how many stocks, median cap)
  A2  IC and IR on filtered universe
  A3  Quintile returns (Q5 long vs EW market)
  A4  FF6 alpha on long-only Q5
  A5  Net Sharpe after realistic costs
  A6  VIX-scaled net Sharpe
  A7  Max drawdown + crisis months
  A8  Bimonthly rebalancing comparison
  A9  Capacity on filtered universe
  A10 Current month portfolio (the actual trade list)
"""

import pandas as pd
import numpy as np
import os
import time
import warnings
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/data/wrds"
RESULTS_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/wrds_pipeline/phase3/results"

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

# log_market_cap = ln(market_cap_in_millions)
# $500M → ln(500)  = 6.215
# $1B   → ln(1000) = 6.908
# $2B   → ln(2000) = 7.601
# $10B  → ln(10000)= 9.210
MIN_CAP_LOG = np.log(500)   # $500M floor
COST_BPS = 15               # avg cost for >$500M stocks (10-25 bps)
BORROW_BPS = 0              # long-only, no borrow
N_QUINTILES = 5              # Q5 = top 20% — more stocks, more diversification


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def spearman_ic(pred, actual):
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < 30:
        return np.nan
    return stats.spearmanr(pred[mask], actual[mask])[0]


def pr(test_id, title, desc=""):
    print(f"\n{'═' * 70}")
    print(f"TEST {test_id}: {title}")
    if desc:
        print(f"  {desc}")
    print(f"{'═' * 70}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("PATH A AUDIT — PERSONAL ALPHA ACCOUNT")
    print("Long-only, >$500M market cap, realistic costs")
    print("=" * 70)

    # ── LOAD ──────────────────────────────────────────────────
    print("\n[LOAD] Loading data...")

    preds = pd.read_parquet(os.path.join(DATA_DIR, "ensemble_predictions.parquet"))
    preds["date"] = pd.to_datetime(preds["date"])

    import pyarrow.parquet as pq
    panel_path = os.path.join(DATA_DIR, "gkx_panel.parquet")
    avail = pq.read_schema(panel_path).names

    need = ["permno", "date", "fwd_ret_1m", "ret_crsp",
            "log_market_cap", "market_cap", "bm", "mom_12m", "mom_12_2",
            "mom_1m", "mom_6m", "realized_vol", "turnover", "turnover_6m",
            "roaq", "beta", "siccd", "log_price", "ep", "lev",
            "idio_vol", "ffi49"]
    need = [c for c in need if c in avail]
    need = list(dict.fromkeys(need))

    panel = pd.read_parquet(panel_path, columns=need)
    panel["date"] = pd.to_datetime(panel["date"])

    merged = preds.merge(panel, on=["permno", "date"], how="left",
                         suffixes=("", "_panel"))
    for c in list(merged.columns):
        if c.endswith("_panel"):
            merged.drop(columns=[c], inplace=True)

    # CRSP for volume
    crsp = pd.read_parquet(os.path.join(DATA_DIR, "crsp_monthly.parquet"),
                           columns=["permno", "date", "vol", "prc"])
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp["daily_dollar_vol"] = (crsp["vol"].fillna(0) * crsp["prc"].abs().fillna(0)) / 21.0
    crsp.loc[crsp["daily_dollar_vol"] < 100, "daily_dollar_vol"] = np.nan
    merged = merged.merge(crsp[["permno", "date", "daily_dollar_vol"]],
                          on=["permno", "date"], how="left")

    # FF factors — normalize dates to end-of-month to match predictions
    ff = pd.read_parquet(os.path.join(DATA_DIR, "ff_factors_monthly.parquet"))
    ff["date"] = pd.to_datetime(ff["date"])
    ff["ym"] = ff["date"].dt.to_period("M")

    # VIX
    macro = pd.read_parquet(os.path.join(DATA_DIR, "macro_predictors.parquet"))
    macro["date"] = pd.to_datetime(macro["date"])
    vix_monthly = macro[["date", "vix"]].dropna() if "vix" in macro.columns else pd.DataFrame(columns=["date", "vix"])

    # Permno → ticker for final trade list
    permno_map_path = os.path.join(DATA_DIR, "permno_ticker_map.parquet")
    if os.path.exists(permno_map_path):
        pm = pd.read_parquet(permno_map_path)
        permno_to_ticker = dict(zip(pm["permno"].astype(int), pm["ticker"]))
    else:
        permno_to_ticker = {}

    print(f"  Full universe: {len(merged):,} stock-months")

    # ══════════════════════════════════════════════════════════
    # FILTER: >$500M market cap only
    # ══════════════════════════════════════════════════════════

    filtered = merged[merged["log_market_cap"] >= MIN_CAP_LOG].copy()
    filtered = filtered.dropna(subset=["prediction", "fwd_ret_1m"])

    print(f"  Filtered (>$500M): {len(filtered):,} stock-months "
          f"({len(filtered)/len(merged)*100:.1f}% of universe)")

    R = {}  # results dict

    # ══════════════════════════════════════════════════════════
    # TEST A1: UNIVERSE STATISTICS
    # ══════════════════════════════════════════════════════════
    pr("A1", "FILTERED UNIVERSE STATISTICS",
       f"Market cap > $500M (ln > {MIN_CAP_LOG:.2f})")

    monthly_stats = []
    for dt, grp in filtered.groupby("date"):
        cap_millions = np.exp(grp["log_market_cap"])
        monthly_stats.append({
            "date": dt,
            "n_stocks": len(grp),
            "median_cap_M": cap_millions.median(),
            "p25_cap_M": cap_millions.quantile(0.25),
            "p75_cap_M": cap_millions.quantile(0.75),
            "mean_ret": grp["fwd_ret_1m"].mean(),
        })
    ms = pd.DataFrame(monthly_stats)

    print(f"  Months: {len(ms)}")
    print(f"  Avg stocks/month: {ms['n_stocks'].mean():.0f}")
    print(f"  Min/Max stocks:   {ms['n_stocks'].min():.0f} / {ms['n_stocks'].max():.0f}")
    print(f"  Median market cap: ${ms['median_cap_M'].mean():.0f}M")
    print(f"  IQR market cap:    ${ms['p25_cap_M'].mean():.0f}M — ${ms['p75_cap_M'].mean():.0f}M")

    R["a1_avg_stocks"] = ms["n_stocks"].mean()
    R["a1_median_cap"] = ms["median_cap_M"].mean()

    # ══════════════════════════════════════════════════════════
    # TEST A2: IC AND IR ON FILTERED UNIVERSE
    # ══════════════════════════════════════════════════════════
    pr("A2", "IC & IR — FILTERED UNIVERSE",
       "Does alpha survive when you remove micro-caps?")

    ics = []
    for dt, grp in filtered.groupby("date"):
        if len(grp) < 50:
            continue
        ic = spearman_ic(grp["prediction"].values, grp["fwd_ret_1m"].values)
        if not np.isnan(ic):
            ics.append({"date": dt, "ic": ic, "n": len(grp)})
    ic_df = pd.DataFrame(ics)

    filt_ic = ic_df["ic"].mean()
    filt_ir = filt_ic / ic_df["ic"].std() if ic_df["ic"].std() > 0 else 0
    pct_positive = (ic_df["ic"] > 0).mean() * 100

    print(f"  IC:  {filt_ic:+.4f}  (full universe: +0.1136)")
    print(f"  IR:  {filt_ir:.2f}    (full universe: 0.98)")
    print(f"  % months IC > 0: {pct_positive:.1f}%")
    print(f"  IC std: {ic_df['ic'].std():.4f}")

    if filt_ic > 0.03:
        print(f"  🟢 IC {filt_ic:.4f} > 0.03 — viable signal")
    elif filt_ic > 0.015:
        print(f"  🟡 IC {filt_ic:.4f} — marginal, needs feature engineering")
    else:
        print(f"  🔴 IC {filt_ic:.4f} — signal too weak for this universe")

    R["a2_ic"] = filt_ic
    R["a2_ir"] = filt_ir

    # Factor-neutral IC on filtered
    factor_cols = [c for c in ["log_market_cap", "bm", "mom_12m", "realized_vol"]
                   if c in filtered.columns]
    if factor_cols:
        neutral_ics = []
        for dt, grp in filtered.groupby("date"):
            g = grp.dropna(subset=["prediction", "fwd_ret_1m"] + factor_cols)
            if len(g) < 50:
                continue
            from numpy.linalg import lstsq
            X = g[factor_cols].values
            X = np.column_stack([np.ones(len(X)), X])
            pred_resid = g["prediction"].values - X @ lstsq(X, g["prediction"].values, rcond=None)[0]
            ic = spearman_ic(pred_resid, g["fwd_ret_1m"].values)
            if not np.isnan(ic):
                neutral_ics.append(ic)
        if neutral_ics:
            fn_ic = np.mean(neutral_ics)
            print(f"\n  Factor-neutral IC: {fn_ic:+.4f}  ({fn_ic/filt_ic*100:.0f}% of raw)")
            R["a2_neutral_ic"] = fn_ic

    # ══════════════════════════════════════════════════════════
    # TEST A3: QUINTILE RETURNS — Q5 LONG vs EW MARKET
    # ══════════════════════════════════════════════════════════
    pr("A3", "QUINTILE SPREAD — Q5 (TOP 20%) vs EQUAL-WEIGHT MARKET",
       "The long-only alpha: Q5 return minus EW market return.")

    q_records = []
    for dt, grp in filtered.groupby("date"):
        if len(grp) < 50:
            continue
        g = grp.copy()
        try:
            g["q"] = pd.qcut(g["prediction"], N_QUINTILES, labels=False, duplicates="drop")
        except ValueError:
            continue

        mkt = g["fwd_ret_1m"].mean()
        for q_val in sorted(g["q"].unique()):
            qg = g[g["q"] == q_val]
            q_records.append({
                "date": dt,
                "quintile": int(q_val) + 1,
                "ret": qg["fwd_ret_1m"].mean(),
                "n": len(qg),
                "mkt": mkt,
            })

    qdf = pd.DataFrame(q_records)

    print(f"  Avg return by quintile:")
    for q in sorted(qdf["quintile"].unique()):
        qr = qdf[qdf["quintile"] == q]["ret"].mean() * 100
        n = qdf[qdf["quintile"] == q]["n"].mean()
        bar = "█" * max(1, int((qr + 2) * 5))
        print(f"    Q{q}: {qr:+.3f}%/mo  ({n:.0f} stocks)  {bar}")

    q5_rets = qdf[qdf["quintile"] == qdf["quintile"].max()].copy()
    q5_rets = q5_rets.sort_values("date")
    q5_alpha = (q5_rets["ret"] - q5_rets["mkt"]).values
    q5_gross_alpha = np.mean(q5_alpha)

    q1_rets = qdf[qdf["quintile"] == qdf["quintile"].min()].copy()
    q1_rets = q1_rets.sort_values("date")
    q5_q1_spread = q5_rets["ret"].mean() - q1_rets["ret"].mean()

    print(f"\n  Q5 avg return:    {q5_rets['ret'].mean()*100:+.3f}%/mo")
    print(f"  EW market return: {q5_rets['mkt'].mean()*100:+.3f}%/mo")
    print(f"  Q5 alpha:         {q5_gross_alpha*100:+.3f}%/mo  ({q5_gross_alpha*1200:+.2f}% ann)")
    print(f"  Q5-Q1 spread:     {q5_q1_spread*100:+.3f}%/mo")

    R["a3_q5_alpha_mo"] = q5_gross_alpha
    R["a3_q5_q1_spread"] = q5_q1_spread

    # ══════════════════════════════════════════════════════════
    # TEST A4: FF6 ALPHA ON LONG-ONLY Q5
    # ══════════════════════════════════════════════════════════
    pr("A4", "FF6 ALPHA — LONG-ONLY Q5 (>$500M)",
       "Risk-adjusted alpha after all 6 Fama-French factors.")

    # Build monthly Q5 return series (excess of risk-free)
    q5_monthly = q5_rets[["date", "ret"]].copy()
    q5_monthly["ym"] = pd.to_datetime(q5_monthly["date"]).dt.to_period("M")
    q5_monthly = q5_monthly.merge(ff, on="ym", how="inner", suffixes=("", "_ff"))
    q5_monthly["excess"] = q5_monthly["ret"] - q5_monthly["rf"]

    # Also build Q5 alpha series (vs EW market)
    q5_alpha_series = q5_rets[["date", "ret", "mkt"]].copy()
    q5_alpha_series["alpha_ret"] = q5_alpha_series["ret"] - q5_alpha_series["mkt"]
    q5_alpha_series["ym"] = pd.to_datetime(q5_alpha_series["date"]).dt.to_period("M")
    q5_alpha_series = q5_alpha_series.merge(ff, on="ym", how="inner", suffixes=("", "_ff"))

    # FF6 regression on Q5 excess return
    factors = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
    factors_avail = [f for f in factors if f in q5_monthly.columns]
    if factors_avail:
        Y = q5_monthly["excess"].values
        X = q5_monthly[factors_avail].values
        X = np.column_stack([np.ones(len(X)), X])
        from numpy.linalg import lstsq
        beta, _, _, _ = lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        alpha = beta[0]
        # Robust standard error using pseudoinverse
        try:
            XtX_inv = np.linalg.pinv(X.T @ X)
        except Exception:
            XtX_inv = np.eye(X.shape[1]) * 1e-6
        se_alpha = np.sqrt(max(np.sum(resid**2) / max(len(Y) - len(beta), 1) * XtX_inv[0, 0], 1e-20))
        t_alpha = alpha / se_alpha if se_alpha > 0 else 0

        print(f"  ── Q5 EXCESS RETURN vs FF6 ──")
        print(f"  Monthly α: {alpha*100:+.3f}%  (annual: {alpha*1200:+.1f}%)")
        print(f"  α t-stat:  {t_alpha:.2f}")
        print(f"  Loadings:")
        for i, f in enumerate(factors_avail):
            print(f"    {f:8s}: {beta[i+1]:+.3f}")
        print(f"  R²: {1 - np.var(resid)/np.var(Y):.3f}")

        R["a4_alpha"] = alpha
        R["a4_alpha_t"] = t_alpha

    # FF6 regression on Q5 alpha (vs market)
    if factors_avail:
        Y2 = q5_alpha_series["alpha_ret"].values
        X2 = q5_alpha_series[factors_avail].values
        X2 = np.column_stack([np.ones(len(X2)), X2])
        beta2, _, _, _ = lstsq(X2, Y2, rcond=None)
        resid2 = Y2 - X2 @ beta2
        try:
            XtX_inv2 = np.linalg.pinv(X2.T @ X2)
        except Exception:
            XtX_inv2 = np.eye(X2.shape[1]) * 1e-6
        se_alpha2 = np.sqrt(max(np.sum(resid2**2) / max(len(Y2) - len(beta2), 1) * XtX_inv2[0, 0], 1e-20))
        t_alpha2 = beta2[0] / se_alpha2 if se_alpha2 > 0 else 0

        print(f"\n  ── Q5 ALPHA (vs EW market) vs FF6 ──")
        print(f"  Monthly α: {beta2[0]*100:+.3f}%  (annual: {beta2[0]*1200:+.1f}%)")
        print(f"  α t-stat:  {t_alpha2:.2f}")
        print(f"  Loadings:")
        for i, f in enumerate(factors_avail):
            print(f"    {f:8s}: {beta2[i+1]:+.3f}")

        R["a4_alpha_vs_mkt"] = beta2[0]
        R["a4_alpha_vs_mkt_t"] = t_alpha2

    # ══════════════════════════════════════════════════════════
    # TEST A5: NET SHARPE AFTER COSTS (MONTHLY REBAL)
    # ══════════════════════════════════════════════════════════
    pr("A5", "NET SHARPE — MONTHLY REBALANCING",
       f"Costs: {COST_BPS} bps round-trip. Long-only Q5 vs EW market.")

    # Q5 alpha series with costs
    q5_ts = q5_rets[["date", "ret", "mkt"]].sort_values("date").copy()
    q5_ts["alpha"] = q5_ts["ret"] - q5_ts["mkt"]

    # Estimate turnover from rank changes
    # For quintile-based, typical turnover is ~20-30% per month
    # We'll compute actual turnover from portfolio composition changes
    prev_permnos = set()
    turnover_list = []
    for dt, grp in filtered.groupby("date"):
        if len(grp) < 50:
            continue
        g = grp.copy()
        try:
            g["q"] = pd.qcut(g["prediction"], N_QUINTILES, labels=False, duplicates="drop")
        except ValueError:
            continue
        q5_permnos = set(g[g["q"] == g["q"].max()]["permno"].values)
        if prev_permnos:
            overlap = len(q5_permnos & prev_permnos)
            total = max(len(q5_permnos), len(prev_permnos))
            turnover = 1 - overlap / total if total > 0 else 0
            turnover_list.append({"date": dt, "turnover": turnover})
        prev_permnos = q5_permnos

    turnover_df = pd.DataFrame(turnover_list)
    avg_turnover = turnover_df["turnover"].mean() if len(turnover_df) > 0 else 0.25

    # Net alpha = gross alpha - costs
    cost_per_month = avg_turnover * (COST_BPS / 10000) * 2  # buy + sell
    q5_ts["alpha_net"] = q5_ts["alpha"] - cost_per_month

    gross_sharpe = q5_ts["alpha"].mean() / q5_ts["alpha"].std() * np.sqrt(12) if q5_ts["alpha"].std() > 0 else 0
    net_sharpe = q5_ts["alpha_net"].mean() / q5_ts["alpha_net"].std() * np.sqrt(12) if q5_ts["alpha_net"].std() > 0 else 0

    print(f"  Avg monthly turnover:  {avg_turnover*100:.1f}%")
    print(f"  Cost per rebalance:    {cost_per_month*10000:.1f} bps")
    print(f"  Gross alpha:           {q5_ts['alpha'].mean()*100:+.3f}%/mo")
    print(f"  Net alpha:             {q5_ts['alpha_net'].mean()*100:+.3f}%/mo  ({q5_ts['alpha_net'].mean()*1200:+.2f}% ann)")
    print(f"  Gross Sharpe:          {gross_sharpe:.2f}")
    print(f"  Net Sharpe:            {net_sharpe:.2f}")

    if net_sharpe > 0.7:
        print(f"  🟢 Net Sharpe {net_sharpe:.2f} — strong personal alpha")
    elif net_sharpe > 0.4:
        print(f"  🟡 Net Sharpe {net_sharpe:.2f} — viable personal alpha")
    else:
        print(f"  🔴 Net Sharpe {net_sharpe:.2f} — marginal")

    R["a5_turnover"] = avg_turnover
    R["a5_gross_sharpe"] = gross_sharpe
    R["a5_net_sharpe"] = net_sharpe
    R["a5_net_alpha_ann"] = q5_ts["alpha_net"].mean() * 12

    # ══════════════════════════════════════════════════════════
    # TEST A6: VIX-SCALED NET SHARPE
    # ══════════════════════════════════════════════════════════
    pr("A6", "VIX-SCALED PORTFOLIO",
       "Reduce exposure: 100% at VIX<25, 50% at VIX 25-35, 20% at VIX>35.")

    if len(vix_monthly) > 0:
        vix_m = vix_monthly.copy()
        vix_m["ym"] = pd.to_datetime(vix_m["date"]).dt.to_period("M")
        q5_vix = q5_ts.copy()
        q5_vix["ym"] = pd.to_datetime(q5_vix["date"]).dt.to_period("M")
        q5_vix = q5_vix.merge(vix_m[["ym", "vix"]], on="ym", how="left")
        q5_vix["vix"] = q5_vix["vix"].ffill()

        def vix_scale(vix_val):
            if pd.isna(vix_val):
                return 1.0
            if vix_val > 35:
                return 0.20
            if vix_val > 25:
                return 0.50
            return 1.0

        q5_vix["scale"] = q5_vix["vix"].apply(vix_scale)
        q5_vix["alpha_scaled"] = q5_vix["alpha_net"] * q5_vix["scale"]

        scaled_sharpe = (q5_vix["alpha_scaled"].mean() / q5_vix["alpha_scaled"].std() * np.sqrt(12)
                         if q5_vix["alpha_scaled"].std() > 0 else 0)

        # Compute drawdown comparison
        unscaled_cum = (1 + q5_vix["alpha_net"]).cumprod()
        scaled_cum = (1 + q5_vix["alpha_scaled"]).cumprod()
        unscaled_dd = (unscaled_cum / unscaled_cum.cummax() - 1).min() * 100
        scaled_dd = (scaled_cum / scaled_cum.cummax() - 1).min() * 100

        n_reduced = (q5_vix["scale"] < 1).sum()
        pct_reduced = n_reduced / len(q5_vix) * 100

        print(f"  Months at 100% exposure: {(q5_vix['scale']==1).sum()} ({(q5_vix['scale']==1).mean()*100:.0f}%)")
        print(f"  Months at 50% exposure:  {(q5_vix['scale']==0.5).sum()}")
        print(f"  Months at 20% exposure:  {(q5_vix['scale']==0.2).sum()}")
        print(f"")
        print(f"  WITHOUT VIX scaling:")
        print(f"    Net Sharpe: {net_sharpe:.2f}")
        print(f"    Max DD:     {unscaled_dd:.1f}%")
        print(f"  WITH VIX scaling:")
        print(f"    Net Sharpe: {scaled_sharpe:.2f}")
        print(f"    Max DD:     {scaled_dd:.1f}%")
        print(f"    DD improvement: {(1 - scaled_dd/unscaled_dd)*100:.0f}%")

        if scaled_sharpe > net_sharpe * 0.9 and scaled_dd > unscaled_dd * 0.7:
            print(f"  🟢 VIX scaling improves risk-adjusted: Sharpe {scaled_sharpe:.2f}, DD {scaled_dd:.1f}%")
        else:
            print(f"  🟡 VIX scaling trades some return for drawdown protection")

        R["a6_vix_sharpe"] = scaled_sharpe
        R["a6_vix_dd"] = scaled_dd
        R["a6_unscaled_dd"] = unscaled_dd
    else:
        print(f"  ⚠️  No VIX data available")
        R["a6_vix_sharpe"] = net_sharpe
        R["a6_vix_dd"] = 0
        R["a6_unscaled_dd"] = 0

    # ══════════════════════════════════════════════════════════
    # TEST A7: CRISIS MONTHS + MAX DRAWDOWN
    # ══════════════════════════════════════════════════════════
    pr("A7", "CRISIS PERFORMANCE — LONG-ONLY Q5",
       "How does the long-only Q5 portfolio behave when markets crash?")

    # Q5 absolute return (not alpha) during crisis periods
    crisis_periods = {
        "2008 Q4 (Financial Crisis)": ("2008-10-01", "2008-12-31"),
        "2009 Q1 (March Bottom)": ("2009-01-01", "2009-03-31"),
        "2011 Q3 (Debt Ceiling)": ("2011-07-01", "2011-09-30"),
        "2015 Q3 (China Deval)": ("2015-07-01", "2015-09-30"),
        "2018 Q4 (Vol Spike)": ("2018-10-01", "2018-12-31"),
        "2020 Q1 (COVID Crash)": ("2020-01-01", "2020-03-31"),
        "2022 H1 (Rate Hikes)": ("2022-01-01", "2022-06-30"),
    }

    for label, (start, end) in crisis_periods.items():
        period = q5_ts[(q5_ts["date"] >= start) & (q5_ts["date"] <= end)]
        if len(period) == 0:
            continue
        q5_ret = period["ret"].sum() * 100
        mkt_ret = period["mkt"].sum() * 100
        alpha = (q5_ret - mkt_ret)
        print(f"  {label}:")
        print(f"    Q5 return: {q5_ret:+.1f}%  Market: {mkt_ret:+.1f}%  Alpha: {alpha:+.1f}%")

    # Max drawdown of Q5 total return
    q5_ts_sorted = q5_ts.sort_values("date")
    q5_cum = (1 + q5_ts_sorted["ret"]).cumprod()
    q5_dd = (q5_cum / q5_cum.cummax() - 1)
    max_dd = q5_dd.min() * 100
    dd_idx = q5_dd.idxmin()
    dd_date = q5_ts_sorted.loc[dd_idx, "date"] if dd_idx in q5_ts_sorted.index else "?"

    # Max drawdown of Q5 alpha (relative to market)
    alpha_cum = (1 + q5_ts_sorted["alpha"]).cumprod()
    alpha_dd = (alpha_cum / alpha_cum.cummax() - 1).min() * 100

    print(f"\n  Max drawdown (total return): {max_dd:.1f}%")
    print(f"  Max drawdown (alpha):        {alpha_dd:.1f}%")
    print(f"  Worst drawdown date:         {dd_date}")

    R["a7_max_dd_total"] = max_dd
    R["a7_max_dd_alpha"] = alpha_dd

    # Conditional beta (Q5 in down vs up markets)
    q5_ff = q5_ts.copy()
    q5_ff["ym"] = pd.to_datetime(q5_ff["date"]).dt.to_period("M")
    q5_ff = q5_ff.merge(ff[["ym", "mktrf"]], on="ym", how="inner")
    down = q5_ff[q5_ff["mktrf"] < -0.05]
    up = q5_ff[q5_ff["mktrf"] > 0]

    if len(down) > 5:
        beta_down = np.polyfit(down["mktrf"], down["alpha"], 1)[0]
        print(f"\n  Conditional β (alpha, MKT < -5%): {beta_down:+.3f}")
        R["a7_beta_down"] = beta_down
    if len(up) > 5:
        beta_up = np.polyfit(up["mktrf"], up["alpha"], 1)[0]
        print(f"  Conditional β (alpha, MKT > 0%):  {beta_up:+.3f}")

    # ══════════════════════════════════════════════════════════
    # TEST A8: BIMONTHLY REBALANCING
    # ══════════════════════════════════════════════════════════
    pr("A8", "BIMONTHLY vs MONTHLY REBALANCING",
       "Signal half-life is 8.8 months — monthly rebal may be over-trading.")

    # Simulate bimonthly: only trade on odd months (Jan, Mar, May, Jul, Sep, Nov)
    q5_ts_sorted = q5_ts.sort_values("date").copy()
    q5_ts_sorted["month"] = pd.to_datetime(q5_ts_sorted["date"]).dt.month
    q5_ts_sorted["year"] = pd.to_datetime(q5_ts_sorted["date"]).dt.year

    # For bimonthly: use the prediction from previous rebalance month
    # On rebalance months (odd), we trade. On skip months (even), we hold.
    # The alpha decays but costs are halved.
    bimonthly_alpha = q5_ts_sorted["alpha"].copy()
    # On non-rebalance months, alpha is slightly lower (signal decay)
    # T+2 IC is 85% of T+1, so alpha on skip months is ~85% of fresh
    is_skip = q5_ts_sorted["month"].isin([2, 4, 6, 8, 10, 12])
    bimonthly_alpha[is_skip] = bimonthly_alpha[is_skip] * 0.85

    # Costs are halved (trade only 6× per year instead of 12×)
    bimonthly_cost = cost_per_month / 2
    bimonthly_net = bimonthly_alpha - bimonthly_cost
    bimonthly_sharpe = (bimonthly_net.mean() / bimonthly_net.std() * np.sqrt(12)
                        if bimonthly_net.std() > 0 else 0)

    # Quarterly
    quarterly_alpha = q5_ts_sorted["alpha"].copy()
    is_skip_q = ~q5_ts_sorted["month"].isin([1, 4, 7, 10])
    quarterly_alpha[is_skip_q] = quarterly_alpha[is_skip_q] * 0.80
    quarterly_cost = cost_per_month / 3
    quarterly_net = quarterly_alpha - quarterly_cost
    quarterly_sharpe = (quarterly_net.mean() / quarterly_net.std() * np.sqrt(12)
                        if quarterly_net.std() > 0 else 0)

    print(f"  Monthly rebal:    Net Sharpe = {net_sharpe:.2f}  Cost/mo = {cost_per_month*10000:.1f} bps")
    print(f"  Bimonthly rebal:  Net Sharpe = {bimonthly_sharpe:.2f}  Cost/mo = {bimonthly_cost*10000:.1f} bps")
    print(f"  Quarterly rebal:  Net Sharpe = {quarterly_sharpe:.2f}  Cost/mo = {quarterly_cost*10000:.1f} bps")

    best_freq = "monthly"
    best_sharpe = net_sharpe
    if bimonthly_sharpe > best_sharpe:
        best_freq = "bimonthly"
        best_sharpe = bimonthly_sharpe
    if quarterly_sharpe > best_sharpe:
        best_freq = "quarterly"
        best_sharpe = quarterly_sharpe

    print(f"\n  → Best rebalancing frequency: {best_freq} (Sharpe {best_sharpe:.2f})")

    R["a8_bimonthly_sharpe"] = bimonthly_sharpe
    R["a8_quarterly_sharpe"] = quarterly_sharpe
    R["a8_best_freq"] = best_freq
    R["a8_best_sharpe"] = best_sharpe

    # ══════════════════════════════════════════════════════════
    # TEST A9: CAPACITY ON FILTERED UNIVERSE
    # ══════════════════════════════════════════════════════════
    pr("A9", "CAPACITY — FILTERED UNIVERSE (>$500M)",
       "1% of daily volume × N stocks in Q5.")

    cap_records = []
    for dt, grp in filtered.dropna(subset=["daily_dollar_vol"]).groupby("date"):
        if len(grp) < 50:
            continue
        g = grp.copy()
        try:
            g["q"] = pd.qcut(g["prediction"], N_QUINTILES, labels=False, duplicates="drop")
        except ValueError:
            continue
        q5 = g[g["q"] == g["q"].max()]
        # 1% of daily volume per stock × N stocks
        cap = q5["daily_dollar_vol"].median() * 0.01 * len(q5)
        cap_records.append({"date": dt, "capacity": cap,
                            "median_ddv": q5["daily_dollar_vol"].median(),
                            "n_q5": len(q5)})

    if cap_records:
        cap_df = pd.DataFrame(cap_records)
        med_cap = cap_df["capacity"].median()
        med_ddv = cap_df["median_ddv"].median()
        avg_n = cap_df["n_q5"].mean()

        def fmt_cap(v):
            if pd.isna(v) or v == 0: return "N/A"
            if abs(v) < 1e6: return f"${v/1e3:.0f}K"
            return f"${v/1e6:.1f}M"

        print(f"  Avg Q5 stocks:       {avg_n:.0f}")
        print(f"  Median daily $ vol:  {fmt_cap(med_ddv)}")
        print(f"  Capacity (1% rule):  {fmt_cap(med_cap)}")
        print(f"  Safe deployment:     {fmt_cap(med_cap * 0.5)}  (50% of capacity)")

        if med_cap > 1e6:
            print(f"  🟢 Capacity {fmt_cap(med_cap)} — viable personal account")
        elif med_cap > 100e3:
            print(f"  🟡 Capacity {fmt_cap(med_cap)} — small personal account")
        else:
            print(f"  🔴 Capacity {fmt_cap(med_cap)} — too constrained")

        R["a9_capacity"] = med_cap
    else:
        print(f"  ⚠️  No volume data for capacity estimation")
        R["a9_capacity"] = 0

    # ══════════════════════════════════════════════════════════
    # TEST A10: CURRENT MONTH PORTFOLIO — THE TRADE LIST
    # ══════════════════════════════════════════════════════════
    pr("A10", "CURRENT MONTH PORTFOLIO — TRADE LIST",
       "The actual stocks to buy if you were trading this today.")

    latest_date = filtered["date"].max()
    latest = filtered[filtered["date"] == latest_date].copy()

    if len(latest) > 0:
        try:
            latest["q"] = pd.qcut(latest["prediction"], N_QUINTILES, labels=False, duplicates="drop")
        except ValueError:
            latest["q"] = 0

        q5_current = latest[latest["q"] == latest["q"].max()].copy()
        q5_current = q5_current.sort_values("prediction", ascending=False)
        q5_current["ticker"] = q5_current["permno"].map(permno_to_ticker)
        q5_current["cap_M"] = np.exp(q5_current["log_market_cap"]).round(0)

        print(f"  Date: {latest_date.date()}")
        print(f"  Universe: {len(latest)} stocks (>$500M)")
        print(f"  Q5 portfolio: {len(q5_current)} stocks")
        print(f"  Equal weight: {100/len(q5_current):.2f}% per stock")
        print(f"")

        # Cap tiers in Q5
        if "log_market_cap" in q5_current.columns:
            large = q5_current[q5_current["log_market_cap"] >= np.log(10000)]
            mid = q5_current[(q5_current["log_market_cap"] >= np.log(2000)) &
                             (q5_current["log_market_cap"] < np.log(10000))]
            small = q5_current[q5_current["log_market_cap"] < np.log(2000)]
            print(f"  Cap breakdown: {len(large)} large (>$10B), "
                  f"{len(mid)} mid ($2-10B), {len(small)} small ($500M-2B)")

        # Top 20 holdings
        print(f"\n  TOP 20 HOLDINGS (highest prediction):")
        print(f"  {'#':>3} {'Ticker':<8} {'Pred':>8} {'Cap($M)':>10} {'Mom12m':>8} {'Beta':>6}")
        print(f"  {'─'*3} {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*6}")
        for i, (_, row) in enumerate(q5_current.head(20).iterrows()):
            ticker = row.get("ticker", str(int(row["permno"])))
            if pd.isna(ticker):
                ticker = str(int(row["permno"]))
            pred = row["prediction"]
            cap = row.get("cap_M", 0)
            mom = row.get("mom_12m", np.nan)
            beta = row.get("beta", np.nan)
            mom_s = f"{mom:+.3f}" if pd.notna(mom) else "  N/A"
            beta_s = f"{beta:.2f}" if pd.notna(beta) else " N/A"
            print(f"  {i+1:3d} {ticker:<8} {pred:+.4f} {cap:>10,.0f} {mom_s:>8} {beta_s:>6}")

        # Bottom 20 (Q1 — what the model says to avoid)
        q1_current = latest[latest["q"] == latest["q"].min()].copy()
        q1_current = q1_current.sort_values("prediction", ascending=True)
        q1_current["ticker"] = q1_current["permno"].map(permno_to_ticker)
        q1_current["cap_M"] = np.exp(q1_current["log_market_cap"]).round(0)

        print(f"\n  BOTTOM 20 — AVOID THESE (Q1, lowest prediction):")
        print(f"  {'#':>3} {'Ticker':<8} {'Pred':>8} {'Cap($M)':>10}")
        print(f"  {'─'*3} {'─'*8} {'─'*8} {'─'*10}")
        for i, (_, row) in enumerate(q1_current.head(20).iterrows()):
            ticker = row.get("ticker", str(int(row["permno"])))
            if pd.isna(ticker):
                ticker = str(int(row["permno"]))
            pred = row["prediction"]
            cap = row.get("cap_M", 0)
            print(f"  {i+1:3d} {ticker:<8} {pred:+.4f} {cap:>10,.0f}")

        R["a10_n_q5"] = len(q5_current)
        R["a10_date"] = str(latest_date.date())

    # ══════════════════════════════════════════════════════════
    # TEST A11: SUBSAMPLE STABILITY (5-YEAR PERIODS)
    # ══════════════════════════════════════════════════════════
    pr("A11", "SUBSAMPLE STABILITY — FILTERED UNIVERSE",
       "Does the long-only alpha work in every market regime?")

    q5_ts_sorted = q5_ts.sort_values("date").copy()
    q5_ts_sorted["year"] = pd.to_datetime(q5_ts_sorted["date"]).dt.year

    periods = [
        ("2005-2009", 2005, 2009),
        ("2010-2014", 2010, 2014),
        ("2015-2019", 2015, 2019),
        ("2020-2024", 2020, 2024),
    ]

    for label, y_start, y_end in periods:
        subset = q5_ts_sorted[(q5_ts_sorted["year"] >= y_start) & (q5_ts_sorted["year"] <= y_end)]
        if len(subset) < 12:
            continue
        sub_alpha = subset["alpha"].mean()
        sub_sharpe = subset["alpha"].mean() / subset["alpha"].std() * np.sqrt(12) if subset["alpha"].std() > 0 else 0
        sub_ic = ic_df[ic_df["date"].dt.year.between(y_start, y_end)]["ic"].mean() if len(ic_df) > 0 else 0
        flag = "🟢" if sub_sharpe > 0.3 else "🔴" if sub_sharpe < 0 else "🟡"
        print(f"  {label}: IC={sub_ic:+.4f}  Alpha={sub_alpha*100:+.3f}%/mo  Sharpe={sub_sharpe:.2f}  {flag}")

    # ══════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'═' * 70}")
    print(f"PATH A VERDICT — {elapsed:.0f}s")
    print(f"{'═' * 70}")

    best_s = R.get("a8_best_sharpe", R.get("a5_net_sharpe", 0))
    best_f = R.get("a8_best_freq", "monthly")
    vix_s = R.get("a6_vix_sharpe", 0)
    vix_dd = R.get("a6_vix_dd", 0)

    def fc(v):
        if pd.isna(v) or v == 0: return "N/A"
        if abs(v) < 1e6: return f"${v/1e3:.0f}K"
        return f"${v/1e6:.1f}M"

    print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  PATH A: PERSONAL ALPHA ACCOUNT                          │
  │  Universe: Long-only Q5, >$500M market cap               │
  ├──────────────────────────────────────────────────────────┤
  │  IC on filtered universe:    {R.get('a2_ic',0):+.4f}                      │
  │  Factor-neutral IC:          {R.get('a2_neutral_ic',0):+.4f}                      │
  │  Q5 gross alpha:             {R.get('a3_q5_alpha_mo',0)*100:+.3f}%/mo ({R.get('a3_q5_alpha_mo',0)*1200:+.1f}% ann)    │
  │  FF6 alpha t-stat:           {R.get('a4_alpha_t',0):.2f}                         │
  │                                                          │
  │  Net Sharpe (monthly):       {R.get('a5_net_sharpe',0):.2f}                         │
  │  Net Sharpe (best freq):     {best_s:.2f} ({best_f})            │
  │  VIX-scaled Sharpe:          {vix_s:.2f}                         │
  │                                                          │
  │  Max DD (total return):      {R.get('a7_max_dd_total',0):.1f}%                      │
  │  Max DD (alpha):             {R.get('a7_max_dd_alpha',0):.1f}%                      │
  │  Max DD (VIX-scaled alpha):  {vix_dd:.1f}%                      │
  │                                                          │
  │  Capacity:                   {fc(R.get('a9_capacity',0)):>10}                   │
  │  Safe deployment:            {fc(R.get('a9_capacity',0)*0.5):>10}                   │
  │  Avg stocks in Q5:           {R.get('a1_avg_stocks',0)/N_QUINTILES:.0f}                        │
  └──────────────────────────────────────────────────────────┘
""")

    # Grade
    ns = R.get("a5_net_sharpe", 0)
    if ns > 0.7:
        verdict = "GO — DEPLOY PERSONAL CAPITAL"
        emoji = "🟢"
    elif ns > 0.4:
        verdict = "CONDITIONAL GO — VIABLE WITH VIX SCALING"
        emoji = "🟡"
    elif ns > 0.2:
        verdict = "MARGINAL — NEEDS FEATURE IMPROVEMENT FOR THIS UNIVERSE"
        emoji = "🟡"
    else:
        verdict = "NO GO — ALPHA DOESN'T SURVIVE IN TRADEABLE UNIVERSE"
        emoji = "🔴"

    print(f"  {emoji} {verdict}")

    if ns > 0.3:
        print(f"""
  NEXT STEPS:
  1. Set up paper trading account (IBKR or similar)
  2. Deploy Q5 portfolio with equal weights
  3. Rebalance {best_f} on the 1st trading day
  4. Enable VIX scaling: 50% at VIX>25, 20% at VIX>35
  5. Track live slippage vs backtest assumptions
  6. Run for 3 months before committing real capital
""")

    print(f"{'═' * 70}")
    print(f"END OF PATH A AUDIT")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
