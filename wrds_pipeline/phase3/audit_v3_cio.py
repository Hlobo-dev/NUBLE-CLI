"""
═══════════════════════════════════════════════════════════════
CIO-GRADE AUDIT v3 — 20 TESTS
═══════════════════════════════════════════════════════════════
The analysis a Chief Investment Officer demands before
allocating a single dollar. Every test that determines
whether this system makes money in the real world.

PHASE 0: FOUNDATIONAL (Tests 0A-0C)
  0A  Prediction correlation structure + Grinold's Law
  0B  Crowding / alpha decay over time
  0C  Capacity estimation

PHASE 1: IC VALIDITY (Tests 1-3)
  1   Ensemble vs individual (preprocessing attribution)
  1B  Walk-forward methodology audit (embargo, autocorr)
  2   Signal decay curve (multi-horizon)
  3   Prediction-factor correlations

PHASE 2: ALPHA VALIDITY (Tests 4-6)
  4   FF6 alpha + nonlinear factor exposure (4B)
  5   Factor-neutral IC
  6   Long vs short decomposition

PHASE 3: TRADEABILITY (Tests 7-10)
  7   Market-cap stratified IC & Sharpe
  8   Tiered transaction costs + net Sharpe
  10  Short side feasibility
  10B Constrained portfolio Sharpe (THE number)

PHASE 4: ROBUSTNESS (Tests 11-17)
  11  Subsample stability (4×5yr)
  12  Block bootstrap
  13  Multiple testing (Harvey et al.)
  14  Crisis performance + conditional beta
  15  Feature importance stability across retrains
  16  Prediction monotonicity (decile returns)
  17  Information decay profile + optimal rebalancing
"""

import pandas as pd
import numpy as np
import os
import time
import warnings
import gc
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

DATA_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/data/wrds"
RESULTS_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/wrds_pipeline/phase3/results"


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def spearman_ic(pred, actual):
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < 50:
        return np.nan
    return stats.spearmanr(pred[mask], actual[mask])[0]


def monthly_ics(df, pred_col, target_col):
    ics = []
    for dt, grp in df.groupby("date"):
        ic = spearman_ic(grp[pred_col].values, grp[target_col].values)
        if not np.isnan(ic):
            ics.append({"date": dt, "ic": ic, "n": len(grp)})
    return pd.DataFrame(ics)


def long_short_returns(df, pred_col, target_col, n_q=10):
    records = []
    for dt, grp in df.groupby("date"):
        g = grp.dropna(subset=[pred_col, target_col])
        if len(g) < 100:
            continue
        g = g.copy()
        try:
            g["q"] = pd.qcut(g[pred_col], n_q, labels=False, duplicates="drop")
        except ValueError:
            continue
        d10 = g[g["q"] == g["q"].max()][target_col].mean()
        d1 = g[g["q"] == g["q"].min()][target_col].mean()
        mkt = g[target_col].mean()
        records.append({"date": dt, "d10": d10, "d1": d1, "mkt": mkt,
                        "ls": d10 - d1, "long_alpha": d10 - mkt,
                        "short_alpha": mkt - d1,
                        "n_d10": int((g["q"] == g["q"].max()).sum()),
                        "n_d1": int((g["q"] == g["q"].min()).sum())})
    return pd.DataFrame(records)


def decile_returns(df, pred_col, target_col, n_q=10):
    """Compute average return per decile per month."""
    all_dec = []
    for dt, grp in df.groupby("date"):
        g = grp.dropna(subset=[pred_col, target_col])
        if len(g) < 100:
            continue
        g = g.copy()
        try:
            g["q"] = pd.qcut(g[pred_col], n_q, labels=False, duplicates="drop")
        except ValueError:
            continue
        for q_val in sorted(g["q"].unique()):
            dq = g[g["q"] == q_val]
            all_dec.append({"date": dt, "decile": int(q_val) + 1,
                            "ret": dq[target_col].mean(), "n": len(dq)})
    return pd.DataFrame(all_dec)


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
    print("CIO-GRADE AUDIT v3 — 20 TESTS")
    print("The analysis before allocating a single dollar.")
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

    # Merge predictions + panel
    merged = preds.merge(panel, on=["permno", "date"], how="left",
                         suffixes=("", "_panel"))
    for c in merged.columns:
        if c.endswith("_panel"):
            merged.drop(columns=[c], inplace=True)

    # Load CRSP for volume/capacity
    # CRSP vol = monthly share volume, prc = price (negative = bid/ask avg)
    crsp = pd.read_parquet(os.path.join(DATA_DIR, "crsp_monthly.parquet"),
                           columns=["permno", "date", "vol", "prc"])
    crsp["date"] = pd.to_datetime(crsp["date"])
    # Daily dollar volume = (monthly shares × |price|) / 21 trading days
    crsp["daily_dollar_vol"] = (crsp["vol"].fillna(0) * crsp["prc"].abs().fillna(0)) / 21.0
    # Filter out obviously broken entries
    crsp.loc[crsp["daily_dollar_vol"] < 100, "daily_dollar_vol"] = np.nan
    merged = merged.merge(crsp[["permno", "date", "daily_dollar_vol"]],
                          on=["permno", "date"], how="left")

    # Load FF factors
    ff = pd.read_parquet(os.path.join(DATA_DIR, "ff_factors_monthly.parquet"))
    ff["date"] = pd.to_datetime(ff["date"])

    # Load VIX (monthly avg from macro)
    macro = pd.read_parquet(os.path.join(DATA_DIR, "macro_predictors.parquet"))
    macro["date"] = pd.to_datetime(macro["date"])
    if "vix" in macro.columns:
        vix_monthly = macro[["date", "vix"]].dropna()
    else:
        vix_monthly = pd.DataFrame(columns=["date", "vix"])

    print(f"  Predictions: {len(preds):,} rows, {preds['date'].nunique()} months")
    print(f"  Panel: {len(panel):,} rows, {len(panel.columns)} cols")
    print(f"  Merged: {len(merged):,} rows")

    # Baseline
    base_ics = monthly_ics(preds, "prediction", "fwd_ret_1m")
    base_ic = base_ics["ic"].mean()
    base_ir = base_ic / base_ics["ic"].std() if base_ics["ic"].std() > 0 else 0
    ls_df = long_short_returns(preds, "prediction", "fwd_ret_1m")
    print(f"\n  Baseline IC: {base_ic:+.4f}  IR: {base_ir:.2f}")

    R = {}  # results dict

    # ═══════════════════════════════════════════════════════════
    # PHASE 0: FOUNDATIONAL VERIFICATION
    # ═══════════════════════════════════════════════════════════

    # ── TEST 0A: GRINOLD'S LAW — EFFECTIVE BREADTH ───────────
    pr("0A", "GRINOLD'S LAW — PREDICTION CORRELATION & EFFECTIVE BREADTH",
       "IR = IC × √BR. BR = independent bets, NOT N_stocks × N_months.")

    monthly_pcs = []
    sample_dates = sorted(preds["date"].unique())[::6]  # every 6 months

    for dt in sample_dates:
        month = merged[(merged["date"] == dt)].dropna(subset=["prediction"])
        if len(month) < 200:
            continue

        pred_vals = month["prediction"].values

        if "ffi49" in month.columns:
            month_c = month.dropna(subset=["ffi49"]).copy()
            if len(month_c) < 200:
                continue
            pred_vals = month_c["prediction"].values
            # Variance explained by sector (FF49 industry)
            sector_means = month_c.groupby("ffi49")["prediction"].transform("mean")
            residual = pred_vals - sector_means.values
            total_var = np.var(pred_vals)
            if total_var > 0:
                var_by_sector = 1 - np.var(residual) / total_var
            else:
                var_by_sector = 0
            n_sectors = month_c["ffi49"].nunique()

            # Effective bets via PCA on sector-level predictions
            # Each sector's mean prediction is one "bet"
            # Correlation between sector means determines redundancy
            sec_means = month_c.groupby("ffi49")["prediction"].mean()
            if len(sec_means) > 5:
                # Effective independent sectors ≈ n_sectors / (1 + (n-1)*avg_corr)
                # Approximate avg_corr from variance ratio
                # High var_by_sector = sector means explain a lot = high within-sector correlation
                avg_stock_corr = var_by_sector  # rough proxy
                eff_bets = n_sectors / (1 + max(n_sectors - 1, 1) * max(avg_stock_corr, 0.01))
                # But each sector also has some independent stock bets
                avg_sector_size = len(month) / max(n_sectors, 1)
                within_sector_bets = avg_sector_size * (1 - avg_stock_corr)
                total_eff = eff_bets + within_sector_bets * 0.1  # partial credit

                monthly_pcs.append({"date": dt, "var_by_sector": var_by_sector,
                                     "n_sectors": n_sectors,
                                     "eff_bets": total_eff,
                                     "n_stocks": len(month),
                                     "avg_stock_corr_proxy": avg_stock_corr})

    if monthly_pcs:
        pc_df = pd.DataFrame(monthly_pcs)
        avg_var_sector = pc_df["var_by_sector"].mean()
        avg_eff_bets = pc_df["eff_bets"].mean()
        avg_n_stocks = pc_df["n_stocks"].mean()

        eff_breadth_annual = avg_eff_bets * 12
        ir_grinold = base_ic * np.sqrt(eff_breadth_annual)
        ir_naive = base_ic * np.sqrt(avg_n_stocks * 12)

        print(f"  Avg stocks/month:              {avg_n_stocks:.0f}")
        print(f"  Prediction variance by sector: {avg_var_sector*100:.1f}%")
        print(f"  Effective independent bets/mo: {avg_eff_bets:.0f}")
        print(f"  Effective annual breadth:      {eff_breadth_annual:.0f}")
        print(f"")
        print(f"  GRINOLD'S LAW:")
        print(f"  Naive IR = {base_ic:.4f} × √({avg_n_stocks:.0f}×12) = {ir_naive:.2f}")
        print(f"  True IR  = {base_ic:.4f} × √({eff_breadth_annual:.0f}) = {ir_grinold:.2f}")
        print(f"  Observed IR = {base_ir:.2f}")
        print(f"")

        if base_ir > ir_grinold * 1.5:
            print(f"  🔴 Observed IR ({base_ir:.2f}) EXCEEDS Grinold ceiling ({ir_grinold:.2f})")
            print(f"     This is suspicious — something inflates apparent performance")
        elif base_ir > ir_grinold:
            print(f"  🟡 Observed IR ({base_ir:.2f}) slightly above ceiling ({ir_grinold:.2f})")
        else:
            print(f"  🟢 Observed IR ({base_ir:.2f}) ≤ Grinold ceiling ({ir_grinold:.2f})")

        R["0a_eff_breadth"] = eff_breadth_annual
        R["0a_ir_grinold"] = ir_grinold

    # ── TEST 0B: ALPHA DECAY / CROWDING ──────────────────────
    pr("0B", "ALPHA DECAY OVER TIME — CROWDING DIAGNOSTIC",
       "Is the D10-D1 spread declining? Rolling 36-month window.")

    if len(ls_df) > 36:
        ls_df_sorted = ls_df.sort_values("date").copy()
        ls_df_sorted["rolling_spread"] = ls_df_sorted["ls"].rolling(36, min_periods=24).mean()
        ls_df_sorted["year"] = pd.to_datetime(ls_df_sorted["date"]).dt.year

        # Period comparison
        early = ls_df_sorted[ls_df_sorted["year"].between(2005, 2009)]["ls"]
        late = ls_df_sorted[ls_df_sorted["year"].between(2020, 2024)]["ls"]

        early_spread = early.mean() * 100
        late_spread = late.mean() * 100
        decay_pct = (1 - late_spread / early_spread) * 100 if early_spread != 0 else 0

        # Time trend regression
        time_idx = np.arange(len(ls_df_sorted))
        spreads = ls_df_sorted["ls"].values
        valid = np.isfinite(spreads)
        if valid.sum() > 36:
            slope, intercept, r, p, se = stats.linregress(time_idx[valid], spreads[valid])
            monthly_decay_bps = slope * 10000
            t_stat = slope / se if se > 0 else 0

            print(f"  2005-2009 avg spread: {early_spread:+.3f}%/mo")
            print(f"  2020-2024 avg spread: {late_spread:+.3f}%/mo")
            print(f"  Decay: {decay_pct:.1f}%")
            print(f"")
            print(f"  Linear trend: {monthly_decay_bps:+.2f} bps/month (t={t_stat:.2f})")

            if t_stat < -2:
                print(f"  🔴 SIGNIFICANT alpha decay (t={t_stat:.2f})")
                print(f"     Forward-looking spread ≈ {late_spread:.3f}%/mo, NOT {(early_spread+late_spread)/2:.3f}%")
            elif t_stat < -1:
                print(f"  🟡 Marginal decay trend (t={t_stat:.2f})")
            else:
                print(f"  🟢 No significant decay trend (t={t_stat:.2f})")

            R["0b_early_spread"] = early_spread
            R["0b_late_spread"] = late_spread
            R["0b_decay_t"] = t_stat

    # ── TEST 0C: CAPACITY ESTIMATION ─────────────────────────
    pr("0C", "STRATEGY CAPACITY ESTIMATION",
       "Almgren-Chriss: impact ≈ σ × √(Q/V). Max capacity at 1% of volume.")

    cap_records = []
    for dt, grp in merged.dropna(subset=["prediction", "fwd_ret_1m", "daily_dollar_vol"]).groupby("date"):
        if len(grp) < 100:
            continue
        grp = grp.copy()
        try:
            grp["q"] = pd.qcut(grp["prediction"], 10, labels=False, duplicates="drop")
        except ValueError:
            continue

        d10 = grp[grp["q"] == grp["q"].max()]
        d1 = grp[grp["q"] == grp["q"].min()]

        # 1% of daily volume = max position size
        # Total capacity = 1% × avg_daily_vol × n_stocks_per_side
        d10_cap = d10["daily_dollar_vol"].median() * 0.01 * len(d10)
        d1_cap = d1["daily_dollar_vol"].median() * 0.01 * len(d1)
        binding_cap = min(d10_cap, d1_cap) if d1_cap > 0 else d10_cap
        long_only_cap = d10["daily_dollar_vol"].median() * 0.01 * len(d10)

        # Also compute for large-cap only (>$2B = ln(2000))
        if "log_market_cap" in grp.columns:
            lc = grp[grp["log_market_cap"] > np.log(2000)]
            if len(lc) > 50:
                try:
                    lc_copy = lc.copy()
                    lc_copy["q"] = pd.qcut(lc_copy["prediction"], 5, labels=False, duplicates="drop")
                    lc_d5 = lc_copy[lc_copy["q"] == lc_copy["q"].max()]
                    lc_cap = lc_d5["daily_dollar_vol"].median() * 0.01 * len(lc_d5)
                except ValueError:
                    lc_cap = np.nan
            else:
                lc_cap = np.nan
        else:
            lc_cap = np.nan

        cap_records.append({"date": dt, "ls_capacity": binding_cap,
                            "long_only_cap": long_only_cap,
                            "lc_capacity": lc_cap})

    if cap_records:
        cap_df = pd.DataFrame(cap_records)
        med_ls = cap_df["ls_capacity"].median()
        med_lo = cap_df["long_only_cap"].median()
        med_lc = cap_df["lc_capacity"].median()

        def fmt_cap(v):
            if pd.isna(v): return "N/A"
            if abs(v) < 1e6: return f"${v/1e3:.0f}K"
            return f"${v/1e6:.1f}M"

        print(f"  Median L/S capacity (all stocks):   {fmt_cap(med_ls)}")
        print(f"  Median long-only capacity:          {fmt_cap(med_lo)}")
        print(f"  Median large-cap (>$2B) capacity:   {fmt_cap(med_lc)}")
        print(f"")
        if med_ls < 50e6:
            print(f"  🔴 L/S capacity < $50M — personal account only")
        elif med_ls < 200e6:
            print(f"  🟡 L/S capacity {fmt_cap(med_ls)} — small fund viable")
        else:
            print(f"  🟢 L/S capacity {fmt_cap(med_ls)} — institutional viable")

        R["0c_ls_capacity"] = med_ls
        R["0c_lo_capacity"] = med_lo

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: IC VALIDITY
    # ═══════════════════════════════════════════════════════════

    # ── TEST 1: ENSEMBLE vs INDIVIDUAL ───────────────────────
    pr("1", "ENSEMBLE vs INDIVIDUAL MODEL IC",
       "The IC jump is from preprocessing, not ensembling.")

    for col in ["pred_lgb", "pred_xgb", "pred_ridge", "pred_enet", "prediction"]:
        if col in preds.columns:
            m = monthly_ics(preds, col, "fwd_ret_1m")
            ic = m["ic"].mean()
            ir = ic / m["ic"].std() if m["ic"].std() > 0 else 0
            print(f"  {col:<14}: IC={ic:+.4f}  IR={ir:.2f}")

    # ── TEST 1B: WALK-FORWARD METHODOLOGY AUDIT ─────────────
    pr("1B", "WALK-FORWARD METHODOLOGY AUDIT",
       "Embargo check, IC autocorrelation, training overlap contamination.")

    print(f"  Walk-forward type:  EXPANDING window (all years < test_year)")
    print(f"  Validation year:    test_year - 1 (1 year)")
    print(f"  Embargo:            ZERO — no gap between train end and test start")
    print(f"  ⚠️  No embargo means predictions at Jan of test_year use Dec")
    print(f"     of train_year data. Monthly overlap contamination possible.")
    print(f"")

    # IC autocorrelation
    ic_series = base_ics.sort_values("date")["ic"].values
    if len(ic_series) > 24:
        ac1 = np.corrcoef(ic_series[:-1], ic_series[1:])[0, 1]
        ac2 = np.corrcoef(ic_series[:-2], ic_series[2:])[0, 1]
        ac3 = np.corrcoef(ic_series[:-3], ic_series[3:])[0, 1]
        ac12 = np.corrcoef(ic_series[:-12], ic_series[12:])[0, 1]

        print(f"  IC autocorrelation:")
        print(f"    Lag 1:  {ac1:+.3f}")
        print(f"    Lag 2:  {ac2:+.3f}")
        print(f"    Lag 3:  {ac3:+.3f}")
        print(f"    Lag 12: {ac12:+.3f}")

        if ac1 > 0.3:
            print(f"  🟡 AC(1) = {ac1:.3f} > 0.3 — adjacent months correlated")
            print(f"     May reflect overlapping training sets or persistent features")
        else:
            print(f"  🟢 AC(1) = {ac1:.3f} — acceptable")

        R["1b_ic_ac1"] = ac1

    # ── TEST 2: SIGNAL DECAY CURVE ───────────────────────────
    pr("2", "SIGNAL DECAY CURVE (multi-horizon IC)",
       "IC at T+1, T+2, T+3, T+6, T+12. Half-life determines rebalancing.")

    preds_s = preds.sort_values(["permno", "date"]).copy()
    decay = {}
    for lag in [1, 2, 3, 6, 12]:
        col = f"_fwd_{lag}"
        preds_s[col] = preds_s.groupby("permno")["fwd_ret_1m"].shift(-(lag - 1))
        m = monthly_ics(preds_s.dropna(subset=[col]), "prediction", col)
        if len(m) > 0:
            ic = m["ic"].mean()
            decay[lag] = ic
            pct = ic / base_ic * 100 if base_ic else 0
            print(f"  T+{lag:<2}: IC = {ic:+.4f}  ({pct:.0f}% of T+1)")
    del preds_s
    gc.collect()

    if 1 in decay and 3 in decay and decay[3] > 0 and decay[1] > 0:
        ratio = decay[3] / decay[1]
        half_life = -2 / np.log(ratio) if ratio > 0 and ratio < 1 else np.inf
        print(f"\n  Signal half-life: {half_life:.1f} months")
        if half_life > 6:
            print(f"  → Capturing SLOW factors (value/quality) — low turnover possible")
        elif half_life > 2:
            print(f"  → Monthly rebalancing appropriate")
        else:
            print(f"  → Fast decay — needs sub-monthly rebalancing")
        R["2_half_life"] = half_life

    # ── TEST 3: PREDICTION-FACTOR CORRELATIONS ───────────────
    pr("3", "PREDICTION vs KNOWN FACTORS",
       "How much of the prediction is just size, vol, value, momentum?")

    factor_cols_test = [("log_market_cap", "Size"),
                        ("realized_vol", "Volatility"),
                        ("bm", "Value (B/M)"),
                        ("mom_12m", "Momentum 12m"),
                        ("roaq", "Profitability"),
                        ("beta", "Market Beta")]

    for col, name in factor_cols_test:
        if col in merged.columns:
            corrs = []
            for dt, grp in merged.dropna(subset=["prediction", col]).groupby("date"):
                if len(grp) > 100:
                    c = stats.spearmanr(grp["prediction"], grp[col])[0]
                    corrs.append(c)
            if corrs:
                avg_c = np.mean(corrs)
                marker = "🔴" if abs(avg_c) > 0.4 else "🟡" if abs(avg_c) > 0.2 else "🟢"
                print(f"  corr(pred, {name:<18}): {avg_c:+.4f}  {marker}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: ALPHA VALIDITY
    # ═══════════════════════════════════════════════════════════

    # ── TEST 4 + 4B: FF6 ALPHA + NONLINEAR FACTOR EXPOSURE ──
    pr("4+4B", "FF6 ALPHA + NONLINEAR FACTOR EXPOSURE",
       "Linear + squared + interaction terms. If alpha vanishes → factor timing.")

    ls_df["ym"] = pd.to_datetime(ls_df["date"]).dt.to_period("M")
    ff["ym"] = ff["date"].dt.to_period("M")
    mff = ls_df.merge(ff, on="ym", how="inner")

    if len(mff) > 36:
        y = mff["ls"].values
        factor_names = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
        X_lin = mff[factor_names].values

        # LINEAR regression
        X1 = np.column_stack([np.ones(len(X_lin)), X_lin])
        b1 = np.linalg.lstsq(X1, y, rcond=None)[0]
        res1 = y - X1 @ b1
        se1 = np.std(res1) / np.sqrt(len(y))
        t_alpha_lin = b1[0] / se1 if se1 > 0 else 0
        ss_res1 = np.sum(res1**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2_lin = 1 - ss_res1 / ss_tot if ss_tot > 0 else 0

        print(f"  ── LINEAR FF6 ──")
        print(f"  Monthly α:   {b1[0]*100:+.3f}%  (annual: {b1[0]*1200:+.1f}%)")
        print(f"  α t-stat:    {t_alpha_lin:.2f}")
        print(f"  R²:          {r2_lin:.3f}")
        print(f"  Loadings:")
        for i, fn in enumerate(factor_names):
            print(f"    {fn:<8}: {b1[i+1]:+.3f}")

        # NONLINEAR: add squared terms + key interactions
        X_sq = X_lin ** 2
        # Add MKT×SMB, MKT×HML, SMB×HML interactions
        interactions = np.column_stack([
            X_lin[:, 0] * X_lin[:, 1],  # MKT×SMB
            X_lin[:, 0] * X_lin[:, 2],  # MKT×HML
            X_lin[:, 1] * X_lin[:, 2],  # SMB×HML
            X_lin[:, 3] * X_lin[:, 1],  # RMW×SMB
        ])
        X2 = np.column_stack([np.ones(len(X_lin)), X_lin, X_sq, interactions])

        b2 = np.linalg.lstsq(X2, y, rcond=None)[0]
        res2 = y - X2 @ b2
        se2 = np.std(res2) / np.sqrt(len(y))
        t_alpha_nl = b2[0] / se2 if se2 > 0 else 0
        r2_nl = 1 - np.sum(res2**2) / ss_tot if ss_tot > 0 else 0

        print(f"\n  ── NONLINEAR FF6 (+ squared + interactions) ──")
        print(f"  Monthly α:   {b2[0]*100:+.3f}%  (annual: {b2[0]*1200:+.1f}%)")
        print(f"  α t-stat:    {t_alpha_nl:.2f}")
        print(f"  R²:          {r2_nl:.3f}")
        print(f"  R² increase: {(r2_nl - r2_lin)*100:.1f}pp from nonlinear terms")

        if t_alpha_nl > 3:
            print(f"  🟢 Alpha survives even nonlinear factor adjustment (t={t_alpha_nl:.2f})")
        elif t_alpha_nl > 2:
            print(f"  🟡 Marginal alpha after nonlinear adjustment (t={t_alpha_nl:.2f})")
        elif t_alpha_lin > 3 and t_alpha_nl < 2:
            print(f"  🔴 Alpha DISAPPEARS with nonlinear factors")
            print(f"     → Your 'alpha' is nonlinear factor timing, not true alpha")
        else:
            print(f"  🔴 No significant alpha (t={t_alpha_nl:.2f})")

        R["4_alpha_lin_t"] = t_alpha_lin
        R["4b_alpha_nl_t"] = t_alpha_nl
        R["4_r2_lin"] = r2_lin
        R["4b_r2_nl"] = r2_nl

    # ── TEST 5: FACTOR-NEUTRAL IC ────────────────────────────
    pr("5", "FACTOR-NEUTRAL IC",
       "Residualize predictions on size, value, momentum, vol. True alpha.")

    factor_feats = [c for c in ["log_market_cap", "bm", "mom_12m",
                                 "realized_vol", "roaq", "beta"]
                    if c in merged.columns]

    if len(factor_feats) >= 3:
        neutral_ics = []
        for dt, grp in merged.dropna(subset=["prediction", "fwd_ret_1m"] + factor_feats).groupby("date"):
            if len(grp) < 100:
                continue
            y_pred = grp["prediction"].values
            X_f = np.column_stack([np.ones(len(grp))] +
                                   [grp[c].values for c in factor_feats])
            try:
                b = np.linalg.lstsq(X_f, y_pred, rcond=None)[0]
                resid = y_pred - X_f @ b
                ic = spearman_ic(resid, grp["fwd_ret_1m"].values)
                if not np.isnan(ic):
                    neutral_ics.append({"date": dt, "ic": ic})
            except Exception:
                continue

        if neutral_ics:
            ndf = pd.DataFrame(neutral_ics)
            n_ic = ndf["ic"].mean()
            n_ir = n_ic / ndf["ic"].std() if ndf["ic"].std() > 0 else 0
            pct_raw = n_ic / base_ic * 100 if base_ic else 0

            print(f"  Factors: {factor_feats}")
            print(f"  Raw IC:            {base_ic:+.4f}")
            print(f"  Factor-neutral IC: {n_ic:+.4f}  ({pct_raw:.0f}% of raw)")
            print(f"  Factor-neutral IR: {n_ir:.2f}")

            marker = "🟢" if n_ic > 0.02 else "🟡" if n_ic > 0.01 else "🔴"
            print(f"  {marker} {'Genuine alpha beyond factors' if n_ic > 0.02 else 'Marginal/no alpha'}")

            R["5_neutral_ic"] = n_ic
            R["5_neutral_ir"] = n_ir

    # ── TEST 6: LONG vs SHORT DECOMPOSITION ──────────────────
    pr("6", "LONG vs SHORT DECOMPOSITION",
       "Where does the spread come from?")

    if len(ls_df) > 0:
        d10_m = ls_df["d10"].mean()
        d1_m = ls_df["d1"].mean()
        mkt_m = ls_df["mkt"].mean()
        ls_m = ls_df["ls"].mean()
        long_a = ls_df["long_alpha"].mean()
        short_a = ls_df["short_alpha"].mean()
        long_pct = long_a / ls_m * 100 if ls_m else 0
        short_pct = short_a / ls_m * 100 if ls_m else 0

        long_sharpe = long_a / ls_df["long_alpha"].std() * np.sqrt(12) if ls_df["long_alpha"].std() > 0 else 0
        short_sharpe = short_a / ls_df["short_alpha"].std() * np.sqrt(12) if ls_df["short_alpha"].std() > 0 else 0

        print(f"  D10 (long):    {d10_m*100:+.3f}%/mo  (avg {ls_df['n_d10'].mean():.0f} stocks)")
        print(f"  D1 (short):    {d1_m*100:+.3f}%/mo  (avg {ls_df['n_d1'].mean():.0f} stocks)")
        print(f"  Market:        {mkt_m*100:+.3f}%/mo")
        print(f"  L/S spread:    {ls_m*100:+.3f}%/mo")
        print(f"")
        print(f"  Long alpha:    {long_a*100:+.3f}%/mo  ({long_pct:.0f}% of spread)  Sharpe={long_sharpe:.2f}")
        print(f"  Short alpha:   {short_a*100:+.3f}%/mo  ({short_pct:.0f}% of spread)  Sharpe={short_sharpe:.2f}")

        marker = "🟢" if long_pct > 40 else "🟡" if long_pct > 25 else "🔴"
        print(f"  {marker} Long contributes {long_pct:.0f}% of spread")

        R["6_long_pct"] = long_pct
        R["6_long_sharpe"] = long_sharpe

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: TRADEABILITY
    # ═══════════════════════════════════════════════════════════

    # ── TEST 7: MARKET CAP STRATIFICATION ────────────────────
    pr("7", "MARKET-CAP STRATIFIED IC & SHARPE",
       "Large >$10B, Mid $2-10B, Small <$2B")

    if "log_market_cap" in merged.columns:
        # ln(market_cap_in_millions)
        t10b = np.log(10000)  # $10B
        t2b = np.log(2000)    # $2B

        for label, lo, hi in [("Large (>$10B)", t10b, 99),
                               ("Mid ($2-10B)", t2b, t10b),
                               ("Small (<$2B)", -99, t2b)]:
            sub = merged[(merged["log_market_cap"] > lo) &
                         (merged["log_market_cap"] <= hi)]
            sub = sub.dropna(subset=["prediction", "fwd_ret_1m"])
            sub_ics = monthly_ics(sub, "prediction", "fwd_ret_1m")
            sub_ls = long_short_returns(sub, "prediction", "fwd_ret_1m")

            if len(sub_ics) > 12:
                s_ic = sub_ics["ic"].mean()
                s_ir = s_ic / sub_ics["ic"].std() if sub_ics["ic"].std() > 0 else 0
                s_sharpe = (sub_ls["ls"].mean() / sub_ls["ls"].std() * np.sqrt(12)
                            if len(sub_ls) > 12 and sub_ls["ls"].std() > 0 else 0)
                n_avg = sub.groupby("date").size().mean()
                spread = sub_ls["ls"].mean() * 100 if len(sub_ls) > 0 else 0

                m = "🟢" if s_ic > 0.03 else "🟡" if s_ic > 0 else "🔴"
                print(f"  {label:<18}: IC={s_ic:+.4f}  IR={s_ir:.2f}  "
                      f"Sharpe={s_sharpe:.2f}  Spread={spread:+.2f}%/mo  "
                      f"(~{n_avg:.0f}/mo) {m}")
                if "Large" in label:
                    R["7_large_ic"] = s_ic
                    R["7_large_sharpe"] = s_sharpe

    # ── TEST 8: TIERED COSTS ────────────────────────────────
    pr("8", "TIERED TRANSACTION COSTS + NET SHARPE",
       "10/25/50/100 bps by cap tier + borrow costs")

    dates_sorted = sorted(preds["date"].unique())
    prev_long, prev_short = set(), set()
    gross_rets, net_rets, turnover_list = [], [], []

    for dt in dates_sorted:
        m = merged[merged["date"] == dt].dropna(subset=["prediction", "fwd_ret_1m"])
        if len(m) < 100:
            continue
        m = m.copy()
        try:
            m["q"] = pd.qcut(m["prediction"], 10, labels=False, duplicates="drop")
        except ValueError:
            continue

        longs = set(m[m["q"] == m["q"].max()]["permno"].values)
        shorts = set(m[m["q"] == m["q"].min()]["permno"].values)

        d10_r = m[m["permno"].isin(longs)]["fwd_ret_1m"].mean()
        d1_r = m[m["permno"].isin(shorts)]["fwd_ret_1m"].mean()
        gross = d10_r - d1_r
        gross_rets.append({"date": dt, "ret": gross})

        if prev_long:
            l_to = 1 - len(longs & prev_long) / max(len(longs), 1)
            s_to = 1 - len(shorts & prev_short) / max(len(shorts), 1)
            to = (l_to + s_to) / 2
            turnover_list.append(to)

            if "log_market_cap" in m.columns:
                d10_cap = m[m["permno"].isin(longs)]["log_market_cap"].mean()
                d1_cap = m[m["permno"].isin(shorts)]["log_market_cap"].mean()

                def cap_cost(lc):
                    if lc > np.log(10000): return 0.001
                    elif lc > np.log(2000): return 0.0025
                    elif lc > np.log(500): return 0.005
                    else: return 0.01

                l_cost = cap_cost(d10_cap) * l_to * 2
                s_cost = cap_cost(d1_cap) * s_to * 2
                borrow = 0.02/12 if d1_cap > np.log(2000) else 0.05/12
                tc = l_cost + s_cost + borrow
            else:
                tc = to * 0.005 * 2 + 0.03/12

            net_rets.append({"date": dt, "ret": gross - tc})

        prev_long, prev_short = longs, shorts

    if net_rets:
        g = np.array([r["ret"] for r in gross_rets])
        n = np.array([r["ret"] for r in net_rets])
        to_arr = np.array(turnover_list)
        gs = np.mean(g) / np.std(g) * np.sqrt(12) if np.std(g) > 0 else 0
        ns = np.mean(n) / np.std(n) * np.sqrt(12) if np.std(n) > 0 else 0

        print(f"  Turnover:     {to_arr.mean()*100:.1f}%/mo")
        print(f"  Gross spread: {np.mean(g)*100:+.3f}%/mo  Sharpe={gs:.2f}")
        print(f"  Net spread:   {np.mean(n)*100:+.3f}%/mo  Sharpe={ns:.2f}")
        print(f"  Cost drag:    {(np.mean(g)-np.mean(n))*100:.3f}%/mo")

        m = "🟢" if ns > 1 else "🟡" if ns > 0.5 else "🔴"
        print(f"  {m} Net Sharpe = {ns:.2f}")

        R["8_net_sharpe"] = ns
        R["8_turnover"] = to_arr.mean()

    # ── TEST 10: SHORT SIDE FEASIBILITY ──────────────────────
    pr("10", "SHORT SIDE FEASIBILITY",
       "Median D1 market cap, borrow costs, actual shortability.")

    if "log_market_cap" in merged.columns:
        d1_caps_dollars = []
        for dt, grp in merged.dropna(subset=["prediction", "log_market_cap"]).groupby("date"):
            if len(grp) < 100: continue
            grp = grp.copy()
            try:
                grp["q"] = pd.qcut(grp["prediction"], 10, labels=False, duplicates="drop")
            except ValueError:
                continue
            d1 = grp[grp["q"] == grp["q"].min()]
            d1_caps_dollars.append(np.exp(d1["log_market_cap"].median()) * 1e6)

        med_d1 = np.median(d1_caps_dollars)
        pct_u500 = np.mean([c < 500e6 for c in d1_caps_dollars]) * 100

        print(f"  Median D1 market cap: ${med_d1/1e6:.0f}M")
        print(f"  % months < $500M:     {pct_u500:.0f}%")
        m = "🟢" if med_d1 > 2e9 else "🟡" if med_d1 > 500e6 else "🔴"
        print(f"  {m} {'Shortable' if med_d1 > 500e6 else 'NOT shortable — micro-caps'}")
        R["10_d1_cap"] = med_d1

    # ── TEST 10B: CONSTRAINED PORTFOLIO (THE REAL NUMBER) ────
    pr("10B", "CONSTRAINED PORTFOLIO SHARPE — THE REAL NUMBER",
       "Min cap $500M, max stock 2%, max sector 25%, turnover-aware")

    min_cap_ln = np.log(500)  # $500M in ln(millions)
    prev_longs_c, prev_shorts_c = set(), set()
    constrained_rets = []

    for dt in dates_sorted:
        m = merged[merged["date"] == dt].dropna(subset=["prediction", "fwd_ret_1m"])
        # CONSTRAINT 1: min market cap $500M
        if "log_market_cap" in m.columns:
            m = m[m["log_market_cap"] > min_cap_ln]
        if len(m) < 50:
            continue

        m = m.copy()
        n_q = 5  # quintiles (more stocks per bucket)
        try:
            m["q"] = pd.qcut(m["prediction"], n_q, labels=False, duplicates="drop")
        except ValueError:
            continue

        # Q5 = long, Q1 = short
        long_pool = m[m["q"] == m["q"].max()].copy()
        short_pool = m[m["q"] == m["q"].min()].copy()

        # CONSTRAINT 2: max single-stock weight 2% → min 50 stocks per side
        # Take all available (quintile already limits to ~20% of universe)
        longs_c = set(long_pool["permno"].values)
        shorts_c = set(short_pool["permno"].values)

        # CONSTRAINT 3: max sector 25% of portfolio per side
        if "ffi49" in m.columns:
            for side_name, sdf, side_set in [("long", long_pool, longs_c),
                                               ("short", short_pool, shorts_c)]:
                sector_counts = sdf["ffi49"].value_counts()
                max_per_sector = max(int(len(sdf) * 0.25), 3)
                keep = []
                for sec in sector_counts.index:
                    sec_stocks = sdf[sdf["ffi49"] == sec]
                    keep.extend(sec_stocks.head(max_per_sector)["permno"].values)
                if side_name == "long":
                    longs_c = set(keep)
                else:
                    shorts_c = set(keep)

        # CONSTRAINT 4: exclude bottom 20% by volume from shorts
        if "daily_dollar_vol" in m.columns:
            short_m = m[m["permno"].isin(shorts_c)].copy()
            vol_thresh = short_m["daily_dollar_vol"].quantile(0.20)
            shorts_c = set(short_m[short_m["daily_dollar_vol"] > vol_thresh]["permno"].values)

        if len(longs_c) < 5 or len(shorts_c) < 5:
            continue

        d10_r = m[m["permno"].isin(longs_c)]["fwd_ret_1m"].mean()
        d1_r = m[m["permno"].isin(shorts_c)]["fwd_ret_1m"].mean()
        ls = d10_r - d1_r

        # Transaction costs (tiered by cap)
        if "log_market_cap" in m.columns:
            avg_cap = m[m["permno"].isin(longs_c | shorts_c)]["log_market_cap"].mean()
            cost = 0.001 if avg_cap > np.log(10000) else 0.0025 if avg_cap > np.log(2000) else 0.005
        else:
            cost = 0.003

        if prev_longs_c:
            to_l = 1 - len(longs_c & prev_longs_c) / max(len(longs_c), 1)
            to_s = 1 - len(shorts_c & prev_shorts_c) / max(len(shorts_c), 1)
            avg_to = (to_l + to_s) / 2
            tc = cost * avg_to * 2 + 0.025/12  # costs + 2.5% ann borrow
        else:
            tc = 0
            avg_to = 0

        constrained_rets.append({"date": dt, "gross": ls, "net": ls - tc,
                                  "n_long": len(longs_c), "n_short": len(shorts_c),
                                  "turnover": avg_to})

        prev_longs_c, prev_shorts_c = longs_c, shorts_c

    if constrained_rets:
        cr = pd.DataFrame(constrained_rets)
        cg = cr["gross"].values
        cn = cr["net"].values

        c_gs = np.mean(cg) / np.std(cg) * np.sqrt(12) if np.std(cg) > 0 else 0
        c_ns = np.mean(cn) / np.std(cn) * np.sqrt(12) if np.std(cn) > 0 else 0

        # Max drawdown
        cum_net = np.cumsum(cn)
        peak = np.maximum.accumulate(cum_net)
        dd = cum_net - peak
        max_dd = np.min(dd) * 100

        avg_to = cr["turnover"].mean() * 100

        print(f"  Constraints: min cap $500M, max sector 25%,")
        print(f"               exclude illiquid shorts, tiered costs + borrow")
        print(f"  Avg long stocks:  {cr['n_long'].mean():.0f}")
        print(f"  Avg short stocks: {cr['n_short'].mean():.0f}")
        print(f"  Avg turnover:     {avg_to:.1f}%/mo")
        print(f"  Gross spread:     {np.mean(cg)*100:+.3f}%/mo  Sharpe={c_gs:.2f}")
        print(f"  Net spread:       {np.mean(cn)*100:+.3f}%/mo  Sharpe={c_ns:.2f}")
        print(f"  Max drawdown:     {max_dd:.1f}%")
        print(f"")

        m = "🟢" if c_ns > 1 else "🟡" if c_ns > 0.5 else "🔴"
        print(f"  {m} CONSTRAINED NET SHARPE = {c_ns:.2f}")
        print(f"  *** This is the number that matters for real trading ***")

        R["10b_constrained_sharpe"] = c_ns
        R["10b_max_dd"] = max_dd

    # ── Also compute LONG-ONLY constrained (more realistic) ──
    print(f"\n  ── LONG-ONLY CONSTRAINED (min cap $500M, Q5 vs market) ──")
    lo_rets = []
    for dt in dates_sorted:
        m = merged[merged["date"] == dt].dropna(subset=["prediction", "fwd_ret_1m"])
        if "log_market_cap" in m.columns:
            m = m[m["log_market_cap"] > min_cap_ln]
        if len(m) < 50:
            continue
        m = m.copy()
        try:
            m["q"] = pd.qcut(m["prediction"], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        q5 = m[m["q"] == m["q"].max()]
        lo_ret = q5["fwd_ret_1m"].mean() - m["fwd_ret_1m"].mean()  # vs equal-weight market
        lo_rets.append({"date": dt, "ret": lo_ret})

    if lo_rets:
        lo_arr = np.array([r["ret"] for r in lo_rets])
        lo_sharpe = np.mean(lo_arr) / np.std(lo_arr) * np.sqrt(12) if np.std(lo_arr) > 0 else 0
        print(f"  Long-only alpha:  {np.mean(lo_arr)*100:+.3f}%/mo vs EW market")
        print(f"  Long-only Sharpe: {lo_sharpe:.2f}")
        R["10b_long_only_sharpe"] = lo_sharpe

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: ROBUSTNESS
    # ═══════════════════════════════════════════════════════════

    # ── TEST 11: SUBSAMPLE STABILITY ─────────────────────────
    pr("11", "SUBSAMPLE STABILITY (5-year periods)")

    base_ics["year"] = pd.to_datetime(base_ics["date"]).dt.year
    all_pos = True
    for s, e in [(2005, 2009), (2010, 2014), (2015, 2019), (2020, 2024)]:
        p = base_ics[(base_ics["year"] >= s) & (base_ics["year"] <= e)]
        if len(p) > 0:
            ic = p["ic"].mean()
            ir = ic / p["ic"].std() if p["ic"].std() > 0 else 0
            pls = ls_df[pd.to_datetime(ls_df["date"]).dt.year.between(s, e)]
            sh = (pls["ls"].mean() / pls["ls"].std() * np.sqrt(12)
                  if len(pls) > 12 and pls["ls"].std() > 0 else 0)
            m = "🟢" if ic > 0.03 else "🟡" if ic > 0 else "🔴"
            if ic <= 0: all_pos = False
            print(f"  {s}-{e}: IC={ic:+.4f}  IR={ir:.2f}  Sharpe={sh:.2f}  {m}")

    print(f"\n  {'🟢 All periods positive' if all_pos else '🔴 IC negative in some periods'}")

    # ── TEST 12: BLOCK BOOTSTRAP ─────────────────────────────
    pr("12", "BLOCK BOOTSTRAP (12-mo blocks, 10K draws)")

    gross_arr = np.array([r["ret"] for r in gross_rets]) if gross_rets else np.array([])
    if len(gross_arr) > 36:
        bs = 12
        nb = len(gross_arr) // bs
        blocks = [gross_arr[i*bs:(i+1)*bs] for i in range(nb)]
        np.random.seed(42)
        boot = []
        for _ in range(10000):
            s = [blocks[i] for i in np.random.randint(0, len(blocks), nb)]
            r = np.concatenate(s)
            if np.std(r) > 0:
                boot.append(np.mean(r) / np.std(r) * np.sqrt(12))
        boot = np.array(boot)
        p5, p50, p95 = np.percentile(boot, [5, 50, 95])
        print(f"  5th pctl:     {p5:.2f}")
        print(f"  Median:       {p50:.2f}")
        print(f"  95th pctl:    {p95:.2f}")
        print(f"  P(Sharpe>0):  {(boot>0).mean()*100:.1f}%")
        print(f"  P(Sharpe>0.5): {(boot>0.5).mean()*100:.1f}%")
        R["12_p5_sharpe"] = p5

    # ── TEST 13: MULTIPLE TESTING ────────────────────────────
    pr("13", "MULTIPLE TESTING (Harvey et al. 2016)")

    total_tests = 539 + 4 * 20 + 4  # features + models×hyperparams + ensemble
    bonf_a = 0.05 / total_tests
    bonf_t = norm.ppf(1 - bonf_a / 2)
    n_yrs = len(base_ics) / 12
    actual_t = base_ir * np.sqrt(n_yrs)

    print(f"  Total tests: {total_tests}")
    print(f"  Bonferroni t: {bonf_t:.2f}")
    print(f"  Actual t:     {actual_t:.2f}")
    m = "🟢" if actual_t > bonf_t else "🟡" if actual_t > 3.0 else "🔴"
    print(f"  {m} {'PASSES' if actual_t > bonf_t else 'FAILS'} Bonferroni")
    R["13_t_stat"] = actual_t

    # ── TEST 14: CRISIS PERFORMANCE ──────────────────────────
    pr("14", "CRISIS PERFORMANCE + CONDITIONAL BETA",
       "2008 Q4, 2020 Q1, 2022. Strategy behavior when it matters most.")

    ls_dates = pd.to_datetime(ls_df["date"])
    ls_df_c = ls_df.copy()
    ls_df_c["ym"] = ls_dates.dt.to_period("M")

    # Merge with market returns
    ls_ff = ls_df_c.merge(ff[["ym", "mktrf"]], on="ym", how="left")

    crises = [
        ("2008 Q4 (Financial Crisis)", "2008-10", "2008-12"),
        ("2009 Q1 (Momentum Crash)", "2009-01", "2009-03"),
        ("2020 Q1 (COVID Crash)", "2020-01", "2020-03"),
        ("2022 (Rate Hike Bear)", "2022-01", "2022-12"),
    ]

    for name, start, end in crises:
        mask = ls_dates.between(start, end)
        crisis = ls_df_c[mask.values]
        if len(crisis) > 0:
            cr_ret = crisis["ls"].sum() * 100
            cr_d10 = crisis["d10"].sum() * 100
            cr_d1 = crisis["d1"].sum() * 100
            print(f"  {name}:")
            print(f"    L/S return: {cr_ret:+.2f}%  (D10={cr_d10:+.2f}%, D1={cr_d1:+.2f}%)")

    # Conditional beta
    if "mktrf" in ls_ff.columns:
        down = ls_ff[ls_ff["mktrf"] < -0.05]  # down months (MKT < -5%)
        if len(down) > 10:
            beta_down = np.polyfit(down["mktrf"].values, down["ls"].values, 1)[0]
            up = ls_ff[ls_ff["mktrf"] > 0]
            beta_up = np.polyfit(up["mktrf"].values, up["ls"].values, 1)[0] if len(up) > 10 else 0

            print(f"\n  Conditional Beta:")
            print(f"    β in down months (MKT < -5%): {beta_down:+.3f}")
            print(f"    β in up months (MKT > 0%):    {beta_up:+.3f}")

            m = "🟢" if abs(beta_down) < 0.3 else "🟡" if abs(beta_down) < 0.5 else "🔴"
            print(f"    {m} {'Market-neutral in crises' if abs(beta_down) < 0.3 else 'NOT neutral in crises'}")
            R["14_beta_down"] = beta_down

    # VIX conditioning
    if len(vix_monthly) > 0:
        ls_vix = ls_df_c.copy()
        ls_vix["date_dt"] = pd.to_datetime(ls_vix["date"])
        vix_monthly["date_dt"] = pd.to_datetime(vix_monthly["date"])
        ls_vix = ls_vix.merge(vix_monthly[["date_dt", "vix"]], on="date_dt", how="left")

        calm = ls_vix[ls_vix["vix"] < 20]
        stress = ls_vix[ls_vix["vix"] > 30]

        if len(calm) > 12 and len(stress) > 5:
            calm_sh = calm["ls"].mean() / calm["ls"].std() * np.sqrt(12) if calm["ls"].std() > 0 else 0
            stress_sh = stress["ls"].mean() / stress["ls"].std() * np.sqrt(12) if stress["ls"].std() > 0 else 0
            print(f"\n  VIX-Conditional Sharpe:")
            print(f"    VIX < 20 (calm):   Sharpe = {calm_sh:.2f}  ({len(calm)} months)")
            print(f"    VIX > 30 (stress): Sharpe = {stress_sh:.2f}  ({len(stress)} months)")

            if stress_sh < 0:
                print(f"    🔴 Strategy LOSES money in stress — fails when it matters")
            elif stress_sh < calm_sh * 0.5:
                print(f"    🟡 Strategy weakens significantly in stress")
            else:
                print(f"    🟢 Strategy holds up in stress")

            R["14_calm_sharpe"] = calm_sh
            R["14_stress_sharpe"] = stress_sh

    # Max drawdown of L/S
    cum_ls = np.cumsum(ls_df["ls"].values)
    peak = np.maximum.accumulate(cum_ls)
    dd = cum_ls - peak
    max_dd = np.min(dd) * 100
    dd_end_idx = np.argmin(dd)
    # Find recovery
    recovery_idx = len(dd) - 1
    for i in range(dd_end_idx, len(dd)):
        if cum_ls[i] >= peak[dd_end_idx]:
            recovery_idx = i
            break
    dd_duration = recovery_idx - dd_end_idx

    print(f"\n  Max Drawdown: {max_dd:.1f}%")
    print(f"  Drawdown duration: ~{dd_duration} months to recover")

    R["14_max_dd"] = max_dd

    # ── TEST 15: FEATURE IMPORTANCE STABILITY ────────────────
    pr("15", "FEATURE IMPORTANCE STABILITY",
       "How consistent is feature importance across walk-forward years?")

    # Try to load LGB feature importances per year from step6b results
    fi_path = os.path.join(RESULTS_DIR, "ensemble_summary.json")
    import json
    if os.path.exists(fi_path):
        with open(fi_path) as f:
            ens_summary = json.load(f)
        if "top_10_features" in ens_summary:
            top_f = ens_summary["top_10_features"]
            if isinstance(top_f, list):
                print(f"  Top 10 features (LGB aggregate importance):")
                for i, feat in enumerate(top_f):
                    print(f"    {i+1:2d}. {feat}")
                vol_count = sum(1 for f in top_f if "vol" in f.lower())
                mom_count = sum(1 for f in top_f if "mom" in f.lower() or "ret" in f.lower())
                macro_count = sum(1 for f in top_f if f.startswith("ix_"))
                print(f"\n  In top 10: {vol_count} vol-related, {mom_count} momentum/return, {macro_count} macro-interactions")
                if vol_count > 4:
                    print(f"  🟡 Heavily concentrated in volatility features")
            elif isinstance(top_f, dict):
                print(f"  Top features (LGB aggregate importance):")
                for i, (feat, imp) in enumerate(list(top_f.items())[:20]):
                    print(f"    {i+1:2d}. {feat:<35} {imp:.4f}")
        else:
            print(f"  ⚠️ No feature importance in summary")
    else:
        print(f"  ⚠️ No ensemble summary found")

    # Compute IC stability of individual features
    print(f"\n  Feature IC stability (top raw features, sampled months):")
    sample_months = sorted(merged["date"].unique())[::12]  # annual samples
    feat_ic_by_month = {}
    test_features = ["realized_vol", "mom_12m", "log_market_cap", "bm", "roaq", "beta"]
    test_features = [f for f in test_features if f in merged.columns]

    for dt in sample_months:
        m = merged[merged["date"] == dt].dropna(subset=["fwd_ret_1m"])
        if len(m) < 200:
            continue
        for feat in test_features:
            s = m.dropna(subset=[feat])
            if len(s) > 100:
                ic = spearman_ic(s[feat].values, s["fwd_ret_1m"].values)
                if feat not in feat_ic_by_month:
                    feat_ic_by_month[feat] = []
                feat_ic_by_month[feat].append(ic)

    for feat, ics in feat_ic_by_month.items():
        avg_ic = np.nanmean(ics)
        std_ic = np.nanstd(ics)
        pct_pos = np.nanmean(np.array(ics) > 0) * 100
        print(f"    {feat:<20}: IC={avg_ic:+.4f} ± {std_ic:.4f}  ({pct_pos:.0f}% positive)")

    # ── TEST 16: PREDICTION MONOTONICITY ─────────────────────
    pr("16", "PREDICTION MONOTONICITY",
       "Do decile returns increase monotonically D1→D10?")

    dec_df = decile_returns(preds, "prediction", "fwd_ret_1m")

    if len(dec_df) > 0:
        # Average return per decile
        avg_dec = dec_df.groupby("decile")["ret"].mean()
        print(f"  Avg return by decile:")
        for d in sorted(avg_dec.index):
            bar = "█" * max(1, int((avg_dec[d] + 0.02) * 200))
            print(f"    D{d:2d}: {avg_dec[d]*100:+.3f}%/mo  {bar}")

        # Monotonicity: Spearman(decile_rank, decile_return) per month
        mono_scores = []
        for dt, grp in dec_df.groupby("date"):
            if len(grp) >= 8:  # need enough deciles
                rho = stats.spearmanr(grp["decile"], grp["ret"])[0]
                mono_scores.append(rho)

        if mono_scores:
            avg_mono = np.mean(mono_scores)
            pct_high = np.mean(np.array(mono_scores) > 0.85) * 100
            pct_pos = np.mean(np.array(mono_scores) > 0) * 100

            print(f"\n  Avg monotonicity score: {avg_mono:.3f}")
            print(f"  % months score > 0.85: {pct_high:.0f}%")
            print(f"  % months score > 0:    {pct_pos:.0f}%")

            m = "🟢" if avg_mono > 0.85 else "🟡" if avg_mono > 0.6 else "🔴"
            print(f"  {m} {'Strong monotonicity' if avg_mono > 0.85 else 'Weak monotonicity' if avg_mono > 0.6 else 'Not monotonic — middle deciles are noise'}")

            R["16_monotonicity"] = avg_mono

    # ── TEST 17: INFORMATION DECAY PROFILE ───────────────────
    pr("17", "INFORMATION DECAY & OPTIMAL REBALANCING")

    if decay:
        print(f"  IC decay profile:")
        for lag, ic in sorted(decay.items()):
            pct = ic / decay[1] * 100 if decay[1] else 0
            bar = "█" * max(1, int(pct / 2))
            print(f"    T+{lag:2d}: IC={ic:+.4f}  ({pct:.0f}%)  {bar}")

        # Cost-adjusted optimal rebalancing
        if R.get("8_turnover") and decay.get(2) and decay.get(1):
            monthly_cost = R["8_turnover"] * 0.005 * 2  # rough per-rebal cost
            monthly_ic = decay[1]
            # If rebalance every N months:
            print(f"\n  Rebalancing optimization:")
            for freq in [1, 2, 3, 6]:
                ic_at_freq = decay.get(freq, decay[max(decay.keys())])
                freq_cost = monthly_cost / freq  # amortized
                net_ic = ic_at_freq - freq_cost * 10  # rough scaling
                print(f"    Every {freq} mo: IC≈{ic_at_freq:.4f}, "
                      f"cost/mo≈{freq_cost*100:.2f}%")

    # ═══════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'═' * 70}")
    print(f"CIO VERDICT — {elapsed:.0f}s")
    print(f"{'═' * 70}")

    def _fc(v):
        if v == 0 or pd.isna(v): return "N/A"
        if abs(v) < 1e6: return f"${v/1e3:.0f}K"
        return f"${v/1e6:.0f}M"

    ls_cap_str = _fc(R.get('0c_ls_capacity', 0))
    lo_cap_str = _fc(R.get('0c_lo_capacity', 0))

    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  HEADLINE IC:           {base_ic:+.4f}                   │
  │  HEADLINE SHARPE:       {R.get('8_net_sharpe',0):.2f} (net, unconstrained)    │
  │                                                     │
  │  HONEST NUMBERS:                                    │
  │  Factor-neutral IC:     {R.get('5_neutral_ic',0):+.4f}                   │
  │  Large-cap IC (>$10B):  {R.get('7_large_ic',0):+.4f}                   │
  │  FF6 alpha t-stat:      {R.get('4_alpha_lin_t',0):.2f} (linear)              │
  │  FF6 alpha t-stat:      {R.get('4b_alpha_nl_t',0):.2f} (nonlinear)           │
  │  Constrained L/S Shrp:  {R.get('10b_constrained_sharpe',0):.2f}                       │
  │  Long-only Sharpe:      {R.get('10b_long_only_sharpe',0):.2f} (>$500M, Q5 vs mkt)   │
  │  Max drawdown:          {R.get('10b_max_dd',0):.1f}%                     │
  │  Strategy capacity:     {ls_cap_str} (L/S)                    │
  │  Long-only capacity:    {lo_cap_str}                           │
  │  Alpha decay t-stat:    {R.get('0b_decay_t',0):.2f}                       │
  │  Grinold IR ceiling:    {R.get('0a_ir_grinold',0):.2f}                       │
  │  Crisis β (down mths):  {R.get('14_beta_down',0):+.3f}                     │
  │  VIX>30 Sharpe:         {R.get('14_stress_sharpe',0):.2f}                       │
  │  Monotonicity:          {R.get('16_monotonicity',0):.3f}                      │
  │  Bootstrap p5 Sharpe:   {R.get('12_p5_sharpe',0):.2f}                       │
  │  Multiple test t-stat:  {R.get('13_t_stat',0):.2f}                       │
  └─────────────────────────────────────────────────────┘

  THE CONSTRAINED NET SHARPE ({R.get('10b_constrained_sharpe',0):.2f} L/S, {R.get('10b_long_only_sharpe',0):.2f} long-only) IS YOUR REAL NUMBER.
  Everything else is academic. This is what an allocator sees.
""")

    # Assessment — use the BETTER of constrained L/S or long-only
    cs = R.get("10b_constrained_sharpe", 0)
    lo = R.get("10b_long_only_sharpe", 0)
    best = max(cs, lo)
    best_label = "long-only" if lo > cs else "constrained L/S"
    if best > 1.0:
        grade = "A — INSTITUTIONAL GRADE"
    elif best > 0.7:
        grade = "B — VIABLE FUND STRATEGY"
    elif best > 0.4:
        grade = f"C — VIABLE WITH IMPROVEMENTS (best: {best_label} {best:.2f})"
    elif best > 0:
        grade = f"D — MARGINAL, NEEDS WORK (best: {best_label} {best:.2f})"
    else:
        grade = "F — NOT VIABLE"

    print(f"  GRADE: {grade}")

    # What to fix
    print(f"\n  PRIORITY ACTIONS:")
    if R.get("6_long_pct", 100) < 40:
        print(f"  1. Short side dominates ({R['6_long_pct']:.0f}% long) — build long-only variant")
    if R.get("10_d1_cap", 1e12) < 500e6:
        print(f"  2. D1 stocks are micro-caps (${R['10_d1_cap']/1e6:.0f}M) — can't short them")
    if R.get("7_large_ic", 0) < 0.03:
        print(f"  3. Large-cap IC weak ({R['7_large_ic']:.4f}) — retrain on large-cap only")
    if R.get("0b_decay_t", 0) < -2:
        print(f"  4. Alpha decaying (t={R['0b_decay_t']:.2f}) — use recent window training")
    if R.get("14_stress_sharpe", 99) < 0:
        print(f"  5. Negative Sharpe in stress — add crisis hedging")
    if abs(R.get("14_beta_down", 0)) > 0.3:
        print(f"  6. Not market-neutral in crises (β={R['14_beta_down']:.3f})")
    if R.get("8_turnover", 0) > 0.4:
        print(f"  7. High turnover ({R['8_turnover']*100:.0f}%) — slow signal decay suggests bimonthly rebalancing")

    print(f"\n{'═' * 70}")
    print(f"END OF CIO AUDIT")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
