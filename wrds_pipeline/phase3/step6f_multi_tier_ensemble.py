"""
PHASE 3 — STEP 6f: MULTI-TIER ENSEMBLE
==========================================
Prompt 4 Execution: Optimal tier-level strategy selection → IC-weighted
ensemble → capacity-aware portfolio construction → full evaluation.

INPUTS (from Prompts 1-3):
  - curated_predictions_{tier}.parquet  (step6d)  — stock-level predictions
  - hedged_returns_{tier}.parquet       (step6e)  — tier L/S return streams
  - blended_hedged_returns.parquet      (step6e)  — preview blend
  - gkx_panel.parquet                             — panel with market_cap/turnover
  - ff_factors_monthly.parquet                    — Fama-French 6 factors
  - fred_daily.parquet                            — VIX for regime conditioning

STRATEGY:
  1. Per-tier best strategy selection (not all tiers benefit from hedging)
  2. Rolling 12-month IC-weighted ensemble (adaptive weights)
  3. Risk-parity variant (equalize vol contribution across tiers)
  4. Model conviction scoring (cross-tier agreement = high conviction)
  5. Capacity estimation with realistic assumptions
  6. Comprehensive evaluation:
     - Fama-French 6-factor alpha + t-stat
     - Conditional beta (VIX > 25)
     - Drawdown analysis (max DD, Calmar ratio)
     - Year-by-year P&L
     - Monotonicity (decile spread)
     - Turnover estimation
  7. Final comparison: Raw blend vs. Best-strategy blend vs. Risk-parity

RESULTS FROM STEP 6e (recap):
  Mega:  VIX=0.24, Regime=0.25, Hedged=0.17 → BEST: Regime
  Large: VIX=0.28, Regime=0.25, Hedged=0.29 → BEST: Hedged
  Mid:   VIX=0.72, Regime=0.74, Hedged=0.73 → BEST: Regime (or raw 0.76)
  Small: VIX=1.94, Regime=1.90, Hedged=1.77 → BEST: VIX-Scaled

Author: Claude × Humberto
"""

import pandas as pd
import numpy as np
import os
import gc
import time
import json
import warnings
from scipy import stats

warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TIER_NAMES = ["mega", "large", "mid", "small"]
TIER_LABELS = {
    "mega": "Mega-Cap (>$10B)",
    "large": "Large-Cap ($2-10B)",
    "mid": "Mid-Cap ($500M-2B)",
    "small": "Small-Cap ($100M-500M)",
}

TIER_LMC = {
    "mega":  (9.21,  np.inf),
    "large": (7.60,  9.21),
    "mid":   (6.21,  7.60),
    "small": (4.61,  6.21),
}

# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def compute_performance(returns, label=""):
    """Comprehensive performance metrics from a monthly return series."""
    r = returns.dropna()
    if len(r) < 12:
        return {'sharpe': 0, 'max_dd': 0, 'ann_return': 0, 'ann_vol': 0,
                'sortino': 0, 'calmar': 0, 'n_months': len(r), 'pct_positive': 0}

    mean_m = r.mean()
    std_m = r.std()
    sharpe = (mean_m / std_m * np.sqrt(12)) if std_m > 0 else 0

    # Sortino
    downside = r[r < 0].std()
    sortino = (mean_m / downside * np.sqrt(12)) if downside > 0 else 0

    # Max Drawdown
    cum = (1 + r).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min()

    # Calmar ratio
    ann_ret = mean_m * 12
    calmar = (ann_ret / abs(max_dd)) if max_dd != 0 else 0

    # Percent positive months
    pct_pos = (r > 0).mean()

    if label:
        print(f"  {label}: Sharpe={sharpe:.2f}, Ann={ann_ret:.1%}, "
              f"MaxDD={max_dd:.1%}, Sortino={sortino:.2f}, "
              f"Calmar={calmar:.2f}, %Pos={pct_pos:.0%}, n={len(r)}")

    return {
        'sharpe': sharpe, 'max_dd': max_dd, 'ann_return': ann_ret,
        'ann_vol': std_m * np.sqrt(12), 'sortino': sortino,
        'calmar': calmar, 'n_months': len(r), 'pct_positive': pct_pos,
        'mean_monthly': mean_m, 'std_monthly': std_m,
    }


def compute_conditional_beta(strat_returns, mkt_returns, vix_ts, vix_thresh=25):
    """Beta of strategy vs market, conditional on VIX level."""
    common = strat_returns.index.intersection(mkt_returns.index).intersection(vix_ts.index)
    if len(common) < 24:
        return 0, 0

    s = strat_returns.loc[common]
    m = mkt_returns.loc[common]
    v = vix_ts.loc[common]

    # Full-sample beta
    cov_full = np.cov(s, m)
    beta_full = cov_full[0, 1] / cov_full[1, 1] if cov_full[1, 1] > 0 else 0

    # Crisis beta (VIX > threshold)
    crisis_mask = v > vix_thresh
    if crisis_mask.sum() >= 6:
        s_c, m_c = s[crisis_mask], m[crisis_mask]
        cov_c = np.cov(s_c, m_c)
        beta_crisis = cov_c[0, 1] / cov_c[1, 1] if cov_c[1, 1] > 0 else 0
    else:
        beta_crisis = beta_full

    return beta_full, beta_crisis


def compute_ff6_alpha(strat_returns, ff_factors):
    """Regress strategy returns on Fama-French 6 factors. Returns alpha, t-stat, R²."""
    common = strat_returns.index.intersection(ff_factors.index)
    if len(common) < 36:
        return 0, 0, 0

    y = strat_returns.loc[common].values
    factor_cols = [c for c in ['mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd'] if c in ff_factors.columns]
    X = ff_factors.loc[common, factor_cols].values
    X = np.column_stack([np.ones(len(X)), X])

    try:
        beta, resid, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        alpha = beta[0]
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # t-stat for alpha
        n = len(y)
        k = X.shape[1]
        mse = ss_res / (n - k) if n > k else ss_res / max(n, 1)
        XtX_inv = np.linalg.inv(X.T @ X)
        se_alpha = np.sqrt(mse * XtX_inv[0, 0])
        t_alpha = alpha / se_alpha if se_alpha > 0 else 0

        return alpha, t_alpha, r_sq
    except Exception:
        return 0, 0, 0


def compute_ic_by_date(predictions_df):
    """Compute rank IC per date from stock-level predictions."""
    ics = {}
    for dt, grp in predictions_df.groupby('date'):
        if len(grp) >= 20 and grp['fwd_ret_1m'].notna().sum() >= 10:
            valid = grp.dropna(subset=['prediction', 'fwd_ret_1m'])
            if len(valid) >= 10:
                ic = stats.spearmanr(valid['prediction'], valid['fwd_ret_1m'])[0]
                ics[dt] = ic
    return pd.Series(ics).sort_index()


def year_by_year_table(returns, label=""):
    """Year-by-year performance breakdown."""
    r_df = returns.to_frame('ret')
    r_df['year'] = r_df.index.year
    yearly = r_df.groupby('year')['ret'].agg(['mean', 'std', 'count'])
    yearly['ann_ret'] = yearly['mean'] * 12
    yearly['sharpe'] = (yearly['mean'] / yearly['std'] * np.sqrt(12)).round(2)

    if label:
        print(f"\n  Year-by-Year [{label}]:")
    n_positive = 0
    n_total = 0
    for yr, row in yearly.iterrows():
        ann = row['ann_ret']
        sh = row['sharpe']
        n = int(row['count'])
        marker = '[OK]' if ann > 0 else '[XX]'
        if ann > 0:
            n_positive += 1
        n_total += 1
        if label:
            print(f"    {yr}: {ann:+.1%} (Sharpe={sh:+.2f}, n={n}) {marker}")

    return yearly, n_positive, n_total


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("=" * 70)
    print("STEP 6f — MULTI-TIER ENSEMBLE (Prompt 4)")
    print("=" * 70)

    # ══════════════════════════════════════════════════════
    # STEP 1: LOAD ALL DATA
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1: LOAD DATA")
    print("=" * 70)

    # 1a. Per-tier curated predictions (stock-level)
    tier_preds = {}
    for tn in TIER_NAMES:
        fp = os.path.join(DATA_DIR, f"curated_predictions_{tn}.parquet")
        if os.path.exists(fp):
            tier_preds[tn] = pd.read_parquet(fp)
            print(f"  {tn}: {tier_preds[tn].shape[0]:,} predictions, "
                  f"{tier_preds[tn].permno.nunique()} permnos")
        else:
            print(f"  ⚠ Missing: {fp}")

    # 1b. Per-tier hedged returns (tier-aggregate L/S)
    tier_returns = {}
    for tn in TIER_NAMES:
        fp = os.path.join(DATA_DIR, f"hedged_returns_{tn}.parquet")
        if os.path.exists(fp):
            tier_returns[tn] = pd.read_parquet(fp)
            print(f"  {tn} returns: {tier_returns[tn].shape[0]} months, "
                  f"cols={list(tier_returns[tn].columns)[:6]}...")
        else:
            print(f"  ⚠ Missing: {fp}")

    # 1c. Fama-French factors
    ff_path = os.path.join(DATA_DIR, "ff_factors_monthly.parquet")
    ff_factors = pd.read_parquet(ff_path)
    # date may be a column, not the index
    if 'date' in ff_factors.columns:
        ff_factors['date'] = pd.to_datetime(ff_factors['date'])
        ff_factors = ff_factors.set_index('date')
    else:
        ff_factors.index = pd.to_datetime(ff_factors.index)
    # Snap to month-end to align with strategy returns
    ff_factors.index = ff_factors.index + pd.offsets.MonthEnd(0)
    # Deduplicate in case snapping created duplicates
    ff_factors = ff_factors[~ff_factors.index.duplicated(keep='first')]
    print(f"  FF factors: {ff_factors.shape[0]} months, cols={list(ff_factors.columns)}")
    print(f"    Index range: {ff_factors.index.min()} to {ff_factors.index.max()}")

    # Market returns for beta computation
    if 'mktrf' in ff_factors.columns and 'rf' in ff_factors.columns:
        mkt_returns = ff_factors['mktrf'] + ff_factors['rf']
    elif 'mktrf' in ff_factors.columns:
        mkt_returns = ff_factors['mktrf']
    else:
        mkt_returns = None
    if mkt_returns is not None:
        print(f"    Mkt returns: {len(mkt_returns)} months, mean={mkt_returns.mean():.4f}")

    # 1d. VIX for conditional analysis
    fred_daily = pd.read_parquet(os.path.join(DATA_DIR, "fred_daily.parquet"))
    if 'date' in fred_daily.columns:
        fred_daily['date'] = pd.to_datetime(fred_daily['date'])
        fred_daily = fred_daily.set_index('date')
    else:
        fred_daily.index = pd.to_datetime(fred_daily.index)

    if 'vix' in fred_daily.columns:
        vix_monthly = fred_daily['vix'].dropna().resample('ME').last()
        vix_monthly.index = vix_monthly.index + pd.offsets.MonthEnd(0)
        print(f"  VIX: {vix_monthly.shape[0]} months, latest={vix_monthly.iloc[-1]:.1f}")
    elif 'VIXCLS' in fred_daily.columns:
        vix_monthly = fred_daily['VIXCLS'].dropna().resample('ME').last()
        vix_monthly.index = vix_monthly.index + pd.offsets.MonthEnd(0)
        print(f"  VIX: {vix_monthly.shape[0]} months")
    else:
        vix_monthly = None
        print("  ⚠ No VIX column found")

    # 1e. Panel for capacity/turnover
    panel_cols = ['permno', 'date', 'log_market_cap', 'turnover']
    panel = pd.read_parquet(os.path.join(DATA_DIR, "gkx_panel.parquet"), columns=panel_cols)
    panel['date'] = pd.to_datetime(panel['date'])
    panel['date'] = panel['date'] + pd.offsets.MonthEnd(0)
    print(f"  Panel: {panel.shape[0]:,} rows for capacity estimation")

    # ══════════════════════════════════════════════════════
    # STEP 2: COMPUTE PER-TIER ICs (STOCK-LEVEL)
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2: STOCK-LEVEL IC PER TIER")
    print("=" * 70)

    tier_ic_series = {}
    tier_ic_stats = {}

    for tn in TIER_NAMES:
        if tn not in tier_preds:
            continue

        ic_series = compute_ic_by_date(tier_preds[tn])
        tier_ic_series[tn] = ic_series

        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        ir = mean_ic / std_ic if std_ic > 0 else 0
        pct_pos = (ic_series > 0).mean()

        tier_ic_stats[tn] = {
            'mean_ic': mean_ic,
            'std_ic': std_ic,
            'ir': ir,
            'pct_positive': pct_pos,
            'n_months': len(ic_series),
        }

        print(f"  {tn:6s}: IC={mean_ic:+.4f}, IR={ir:.2f}, "
              f"{pct_pos:.0%} positive, n={len(ic_series)}")

    # ══════════════════════════════════════════════════════
    # STEP 3: PER-TIER BEST STRATEGY SELECTION
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3: PER-TIER BEST STRATEGY SELECTION")
    print("=" * 70)
    print("  Selecting best return stream for each tier based on")
    print("  risk-adjusted return (Sharpe) AND drawdown control (MaxDD)")

    strategy_cols = {
        'raw':     'ls_return',
        'vix':     'vix_scaled_return',
        'regime':  'regime_scaled_return',
        'hedged':  'hedged_return',
    }

    tier_best_strategy = {}
    tier_best_returns = {}

    for tn in TIER_NAMES:
        if tn not in tier_returns:
            continue

        df = tier_returns[tn]
        print(f"\n  {TIER_LABELS[tn]}:")

        best_score = -999
        best_name = 'raw'
        best_col = 'ls_return'

        for sname, col in strategy_cols.items():
            if col not in df.columns:
                continue
            r = df[col].dropna()
            if len(r) < 12:
                continue
            perf = compute_performance(r)

            # Score = Sharpe + 0.3 * Calmar (reward drawdown control)
            # This penalizes strategies with deep drawdowns
            score = perf['sharpe'] + 0.3 * perf['calmar']

            marker = ""
            if score > best_score:
                best_score = score
                best_name = sname
                best_col = col
                marker = " ★ BEST"

            print(f"    {sname:8s}: Sharpe={perf['sharpe']:.2f}, "
                  f"MaxDD={perf['max_dd']:.1%}, Calmar={perf['calmar']:.2f}, "
                  f"Score={score:.2f}{marker}")

        tier_best_strategy[tn] = best_name
        tier_best_returns[tn] = df[best_col].dropna()
        print(f"    → Selected: {best_name}")

    print(f"\n  Strategy map: {tier_best_strategy}")

    # ══════════════════════════════════════════════════════
    # STEP 4: ENSEMBLE BLENDING (MULTIPLE METHODS)
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 4: ENSEMBLE BLENDING")
    print("=" * 70)

    # Get common dates across all tiers
    all_dates = set()
    for tn, r in tier_best_returns.items():
        all_dates.update(r.index)
    all_dates = sorted(all_dates)
    date_idx = pd.DatetimeIndex(all_dates)

    # ── Method 1: Equal Weight ──
    print("\n  Method 1: Equal Weight")
    eq_blend = pd.Series(0.0, index=date_idx)
    eq_count = pd.Series(0, index=date_idx)

    for tn, r in tier_best_returns.items():
        for dt in r.index:
            if dt in eq_blend.index:
                eq_blend[dt] += r[dt]
                eq_count[dt] += 1

    mask_eq = eq_count > 0
    eq_blend[mask_eq] = eq_blend[mask_eq] / eq_count[mask_eq]
    eq_blend = eq_blend[mask_eq].dropna()

    perf_eq = compute_performance(eq_blend, "Equal-Weight")

    # ── Method 2: Full-Sample IC-Weighted ──
    print("\n  Method 2: Full-Sample IC-Weighted")
    total_ic = sum(max(s['mean_ic'], 0.001) for s in tier_ic_stats.values())
    ic_weights = {}
    for tn in TIER_NAMES:
        if tn in tier_ic_stats:
            ic_weights[tn] = max(tier_ic_stats[tn]['mean_ic'], 0.001) / total_ic
    print(f"    Weights: {dict((k, round(v, 3)) for k, v in ic_weights.items())}")

    ic_blend = pd.Series(0.0, index=date_idx)
    ic_wsum = pd.Series(0.0, index=date_idx)

    for tn, r in tier_best_returns.items():
        w = ic_weights.get(tn, 0)
        for dt in r.index:
            if dt in ic_blend.index:
                ic_blend[dt] += w * r[dt]
                ic_wsum[dt] += w

    mask_ic = ic_wsum > 0
    ic_blend[mask_ic] = ic_blend[mask_ic] / ic_wsum[mask_ic]
    ic_blend = ic_blend[mask_ic].dropna()

    perf_ic = compute_performance(ic_blend, "IC-Weighted")

    # ── Method 3: Rolling 12-Month IC-Weighted (Adaptive) ──
    print("\n  Method 3: Rolling 12-Month IC-Weighted (Adaptive)")

    # Build rolling IC per tier
    rolling_ic_weights = {}
    for tn in TIER_NAMES:
        if tn in tier_ic_series:
            rolling_ic_weights[tn] = tier_ic_series[tn].rolling(12, min_periods=6).mean()

    roll_blend = pd.Series(0.0, index=date_idx)
    roll_wsum = pd.Series(0.0, index=date_idx)

    for dt in date_idx:
        total_w = 0
        contributions = {}

        for tn, r in tier_best_returns.items():
            if dt not in r.index:
                continue
            if tn in rolling_ic_weights and dt in rolling_ic_weights[tn].index:
                w = max(rolling_ic_weights[tn].loc[dt], 0.001)
            else:
                w = max(tier_ic_stats.get(tn, {}).get('mean_ic', 0.001), 0.001)

            contributions[tn] = (w, r[dt])
            total_w += w

        if total_w > 0:
            roll_blend[dt] = sum(w * ret / total_w for w, ret in contributions.values())
            roll_wsum[dt] = total_w

    mask_roll = roll_wsum > 0
    roll_blend = roll_blend[mask_roll].dropna()

    perf_roll = compute_performance(roll_blend, "Rolling-IC-Weighted")

    # ── Method 4: Risk-Parity (Equalize Vol Contribution) ──
    print("\n  Method 4: Risk-Parity (Inverse-Vol Weighted)")

    # Compute rolling 12-month vol per tier
    tier_vols = {}
    for tn, r in tier_best_returns.items():
        tier_vols[tn] = r.rolling(12, min_periods=6).std()

    rp_blend = pd.Series(0.0, index=date_idx)
    rp_wsum = pd.Series(0.0, index=date_idx)

    for dt in date_idx:
        inv_vols = {}
        for tn, r in tier_best_returns.items():
            if dt not in r.index:
                continue
            vol = tier_vols[tn].get(dt, None)
            if vol is not None and vol > 0 and not np.isnan(vol):
                inv_vols[tn] = 1.0 / vol
            else:
                inv_vols[tn] = 1.0  # fallback

        total_inv = sum(inv_vols.values())
        if total_inv > 0:
            for tn, inv_v in inv_vols.items():
                if dt in tier_best_returns[tn].index:
                    w = inv_v / total_inv
                    rp_blend[dt] += w * tier_best_returns[tn][dt]
                    rp_wsum[dt] += w

    mask_rp = rp_wsum > 0
    rp_blend[mask_rp] = rp_blend[mask_rp]  # already weighted
    rp_blend = rp_blend[mask_rp].dropna()

    perf_rp = compute_performance(rp_blend, "Risk-Parity")

    # ── Method 5: IC-Risk-Parity Hybrid (IC × InvVol) ──
    print("\n  Method 5: IC × Risk-Parity Hybrid")

    hybrid_blend = pd.Series(0.0, index=date_idx)
    hybrid_wsum = pd.Series(0.0, index=date_idx)

    for dt in date_idx:
        weights = {}
        for tn, r in tier_best_returns.items():
            if dt not in r.index:
                continue
            # IC component
            if tn in rolling_ic_weights and dt in rolling_ic_weights[tn].index:
                ic_w = max(rolling_ic_weights[tn].loc[dt], 0.001)
            else:
                ic_w = max(tier_ic_stats.get(tn, {}).get('mean_ic', 0.001), 0.001)

            # Vol component
            vol = tier_vols.get(tn, pd.Series(dtype=float)).get(dt, None)
            if vol is not None and vol > 0 and not np.isnan(vol):
                vol_w = 1.0 / vol
            else:
                vol_w = 1.0

            weights[tn] = ic_w * vol_w

        total_w = sum(weights.values())
        if total_w > 0:
            for tn, w in weights.items():
                if dt in tier_best_returns[tn].index:
                    hybrid_blend[dt] += (w / total_w) * tier_best_returns[tn][dt]
                    hybrid_wsum[dt] += w / total_w

    mask_hyb = hybrid_wsum > 0
    hybrid_blend = hybrid_blend[mask_hyb].dropna()

    perf_hyb = compute_performance(hybrid_blend, "IC×Risk-Parity")

    # ══════════════════════════════════════════════════════
    # STEP 5: SELECT BEST ENSEMBLE METHOD
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 5: ENSEMBLE METHOD COMPARISON")
    print("=" * 70)

    methods = {
        'equal':        (eq_blend,     perf_eq),
        'ic_weighted':  (ic_blend,     perf_ic),
        'rolling_ic':   (roll_blend,   perf_roll),
        'risk_parity':  (rp_blend,     perf_rp),
        'ic_riskpar':   (hybrid_blend, perf_hyb),
    }

    print(f"\n  {'Method':<18s} {'Sharpe':>7s} {'MaxDD':>7s} {'Sortino':>8s} "
          f"{'Calmar':>7s} {'Ann Ret':>8s} {'%Pos':>5s}")
    print(f"  {'-'*65}")

    best_score = -999
    best_method = 'ic_weighted'

    for mname, (ret_series, perf) in methods.items():
        score = perf['sharpe'] + 0.3 * perf['calmar']
        marker = ""
        if score > best_score:
            best_score = score
            best_method = mname
            marker = " ★"

        print(f"  {mname:<18s} {perf['sharpe']:>7.2f} {perf['max_dd']:>7.1%} "
              f"{perf['sortino']:>8.2f} {perf['calmar']:>7.2f} "
              f"{perf['ann_return']:>8.1%} {perf['pct_positive']:>5.0%}{marker}")

    best_returns = methods[best_method][0]
    best_perf = methods[best_method][1]
    print(f"\n  ★ SELECTED: {best_method} (Score={best_score:.2f})")

    # ══════════════════════════════════════════════════════
    # STEP 6: COMPREHENSIVE EVALUATION OF BEST ENSEMBLE
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"STEP 6: COMPREHENSIVE EVALUATION — {best_method.upper()}")
    print("=" * 70)

    # 6a. Fama-French 6-Factor Alpha
    print("\n  ── Fama-French 6-Factor Alpha ──")
    alpha, t_alpha, r_sq = compute_ff6_alpha(best_returns, ff_factors)
    alpha_ann = alpha * 12
    print(f"  Monthly α = {alpha:.4f} ({alpha_ann:.2%} annualized)")
    print(f"  t(α) = {t_alpha:.2f}  {'✓ SIGNIFICANT' if abs(t_alpha) > 1.96 else '✗ not significant'}")
    print(f"  R² = {r_sq:.3f}")

    # Also do per-tier FF alpha
    print("\n  Per-Tier FF6 Alpha:")
    for tn in TIER_NAMES:
        if tn not in tier_best_returns:
            continue
        a, t, r2 = compute_ff6_alpha(tier_best_returns[tn], ff_factors)
        sig = '✓' if abs(t) > 1.96 else '✗'
        print(f"    {tn:6s}: α={a*12:.2%}/yr, t={t:.2f} {sig}, R²={r2:.3f}")

    # 6b. Conditional Beta
    print("\n  ── Conditional Beta ──")
    if mkt_returns is not None and vix_monthly is not None:
        beta_full, beta_crisis = compute_conditional_beta(
            best_returns, mkt_returns, vix_monthly, 25)
        print(f"  β_full     = {beta_full:+.3f}")
        print(f"  β_crisis   = {beta_crisis:+.3f} (VIX > 25)")
        print(f"  {'✓ LOW BETA' if abs(beta_crisis) < 0.20 else '✗ High beta'}")

        # Also at VIX > 30
        _, beta_extreme = compute_conditional_beta(
            best_returns, mkt_returns, vix_monthly, 30)
        print(f"  β_extreme  = {beta_extreme:+.3f} (VIX > 30)")

    # 6c. Drawdown Analysis
    print("\n  ── Drawdown Analysis ──")
    cum = (1 + best_returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max

    # Worst drawdowns
    dd_sorted = dd.sort_values()
    print(f"  Max Drawdown: {dd_sorted.iloc[0]:.1%} ({dd_sorted.index[0].strftime('%Y-%m')})")
    if len(dd_sorted) >= 3:
        print(f"  2nd worst:    {dd_sorted.iloc[1]:.1%} ({dd_sorted.index[1].strftime('%Y-%m')})")
        print(f"  3rd worst:    {dd_sorted.iloc[2]:.1%} ({dd_sorted.index[2].strftime('%Y-%m')})")

    # Recovery time
    in_dd = dd < 0
    if in_dd.any():
        dd_periods = []
        start = None
        for i, (dt, val) in enumerate(dd.items()):
            if val < 0 and start is None:
                start = dt
            elif val >= 0 and start is not None:
                dd_periods.append((start, dt, (dt - start).days))
                start = None
        if start is not None:
            dd_periods.append((start, dd.index[-1], (dd.index[-1] - start).days))
        if dd_periods:
            longest = max(dd_periods, key=lambda x: x[2])
            print(f"  Longest DD period: {longest[2]} days "
                  f"({longest[0].strftime('%Y-%m')} to {longest[1].strftime('%Y-%m')})")

    # 6d. Year-by-Year
    print("\n  ── Year-by-Year Performance ──")
    yearly, n_pos, n_tot = year_by_year_table(best_returns, best_method)
    print(f"\n  Positive years: {n_pos}/{n_tot} ({n_pos/n_tot:.0%})")

    # 6e. Worst months
    print("\n  ── Worst / Best Months ──")
    sorted_rets = best_returns.sort_values()
    print(f"  5 Worst Months:")
    for i in range(min(5, len(sorted_rets))):
        dt = sorted_rets.index[i]
        print(f"    {dt.strftime('%Y-%m')}: {sorted_rets.iloc[i]:+.2%}")
    print(f"  5 Best Months:")
    for i in range(min(5, len(sorted_rets))):
        dt = sorted_rets.index[-(i+1)]
        print(f"    {dt.strftime('%Y-%m')}: {sorted_rets.iloc[-(i+1)]:+.2%}")

    # 6f. Rolling Sharpe
    print("\n  ── Rolling 36-Month Sharpe ──")
    rolling_sharpe = (best_returns.rolling(36, min_periods=24).mean() /
                      best_returns.rolling(36, min_periods=24).std() * np.sqrt(12))
    rs_stats = rolling_sharpe.dropna()
    if len(rs_stats) > 0:
        print(f"  Mean Rolling Sharpe:   {rs_stats.mean():.2f}")
        print(f"  Min Rolling Sharpe:    {rs_stats.min():.2f} ({rs_stats.idxmin().strftime('%Y-%m')})")
        print(f"  Max Rolling Sharpe:    {rs_stats.max():.2f} ({rs_stats.idxmax().strftime('%Y-%m')})")
        print(f"  % Periods Sharpe > 0:  {(rs_stats > 0).mean():.0%}")
        print(f"  % Periods Sharpe > 1:  {(rs_stats > 1).mean():.0%}")

    # ══════════════════════════════════════════════════════
    # STEP 7: STOCK-LEVEL CROSS-TIER CONVICTION
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 7: CROSS-TIER CONVICTION ANALYSIS")
    print("=" * 70)

    # Some stocks appear in multiple tiers over time (as they cross boundaries)
    # But more importantly: cross-tier prediction agreement = conviction
    # For each date, merge predictions across overlapping permnos

    # Load all predictions into one frame
    all_preds = pd.concat(tier_preds.values(), ignore_index=True)
    print(f"  Total predictions: {all_preds.shape[0]:,}")

    # Monotonicity analysis per tier
    print("\n  ── Decile Analysis Per Tier ──")
    for tn in TIER_NAMES:
        if tn not in tier_preds:
            continue
        pred_df = tier_preds[tn]
        valid = pred_df.dropna(subset=['prediction', 'fwd_ret_1m'])

        # Compute decile returns
        decile_rets = []
        for dt, grp in valid.groupby('date'):
            if len(grp) < 20:
                continue
            try:
                grp = grp.copy()
                grp['decile'] = pd.qcut(grp['prediction'], 10, labels=False, duplicates='drop')
                dec_r = grp.groupby('decile')['fwd_ret_1m'].mean()
                decile_rets.append(dec_r)
            except Exception:
                continue

        if not decile_rets:
            continue

        dec_df = pd.DataFrame(decile_rets)
        avg_dec = dec_df.mean()

        # Monotonicity = rank correlation of decile number vs. decile return
        mono = stats.spearmanr(range(len(avg_dec)), avg_dec.values)[0]

        # Top decile vs. bottom decile spread
        spread = avg_dec.iloc[-1] - avg_dec.iloc[0]

        print(f"  {tn:6s}: Mono={mono:.3f}, Spread={spread:.4f}/m "
              f"({spread*12:.2%}/yr), D10={avg_dec.iloc[-1]:.4f}, D1={avg_dec.iloc[0]:.4f}")

    # ══════════════════════════════════════════════════════
    # STEP 8: CAPACITY ESTIMATION
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 8: CAPACITY ESTIMATION")
    print("=" * 70)

    # For each tier, estimate how much $ can be deployed
    # Capacity = median daily dollar volume of stocks in top/bottom decile × participation rate × 21 days
    participation_rate = 0.05  # 5% of daily volume (conservative)

    for tn in TIER_NAMES:
        if tn not in tier_preds:
            continue

        pred_df = tier_preds[tn].copy()

        # Merge with panel for market_cap and turnover
        pred_df = pred_df.merge(
            panel[['permno', 'date', 'log_market_cap', 'turnover']],
            on=['permno', 'date'],
            how='left'
        )

        # Market cap in $ (log_market_cap = ln(market_cap_in_millions))
        pred_df['market_cap_M'] = np.exp(pred_df['log_market_cap'])

        # Daily dollar volume = market_cap × turnover / 21 (approx)
        pred_df['daily_dollar_vol_M'] = pred_df['market_cap_M'] * pred_df['turnover'] / 21

        # Top decile stocks
        top_decile_ddv = []
        for dt, grp in pred_df.groupby('date'):
            if len(grp) < 20:
                continue
            valid = grp.dropna(subset=['prediction', 'daily_dollar_vol_M'])
            if len(valid) < 10:
                continue
            try:
                valid = valid.copy()
                valid['decile'] = pd.qcut(valid['prediction'], 10, labels=False, duplicates='drop')
                top = valid[valid['decile'] == valid['decile'].max()]
                if len(top) > 0:
                    top_decile_ddv.append(top['daily_dollar_vol_M'].median())
            except Exception:
                continue

        if top_decile_ddv:
            median_ddv_M = np.median(top_decile_ddv)
            # Capacity per side = median DDV × participation × 21 days
            capacity_per_side_M = median_ddv_M * participation_rate * 21
            # Total (long + short) = 2 × per_side
            total_capacity_M = 2 * capacity_per_side_M

            # Market cap range
            min_lmc, max_lmc = TIER_LMC[tn]
            mc_min_M = np.exp(min_lmc)
            mc_max_M = np.exp(max_lmc) if max_lmc < 100 else np.inf

            print(f"  {tn:6s}: Mkt Cap={mc_min_M:.0f}M-{mc_max_M:.0f}M, "
                  f"Median DDV={median_ddv_M:.2f}M, "
                  f"Capacity/side=${capacity_per_side_M:.2f}M, "
                  f"Total=${total_capacity_M:.2f}M")
        else:
            print(f"  {tn:6s}: Could not compute capacity")

    # Blended capacity (IC-weighted sum)
    print(f"\n  NOTE: True capacity = sum across tiers (diversified L/S portfolio)")
    print(f"  Individual tier capacity is LOW because we're measuring")
    print(f"  median DDV of TOP decile × 5% participation.")
    print(f"  In practice, trading over 5-10 days instead of 1 day")
    print(f"  increases capacity by 5-10x.")

    # ══════════════════════════════════════════════════════
    # STEP 9: TURNOVER ESTIMATION
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 9: TURNOVER ESTIMATION")
    print("=" * 70)

    for tn in TIER_NAMES:
        if tn not in tier_preds:
            continue

        pred_df = tier_preds[tn]
        dates = sorted(pred_df['date'].unique())

        turnovers = []
        prev_top = set()
        prev_bot = set()

        for dt in dates:
            grp = pred_df[pred_df['date'] == dt]
            if len(grp) < 20:
                continue
            try:
                grp = grp.copy()
                grp['decile'] = pd.qcut(grp['prediction'], 10, labels=False, duplicates='drop')
                max_dec = grp['decile'].max()
                min_dec = grp['decile'].min()
                cur_top = set(grp[grp['decile'] == max_dec]['permno'])
                cur_bot = set(grp[grp['decile'] == min_dec]['permno'])

                if prev_top and prev_bot:
                    # Turnover = % of names replaced
                    top_turn = 1 - len(cur_top & prev_top) / max(len(cur_top), 1)
                    bot_turn = 1 - len(cur_bot & prev_bot) / max(len(cur_bot), 1)
                    turnovers.append((top_turn + bot_turn) / 2)

                prev_top = cur_top
                prev_bot = cur_bot
            except Exception:
                continue

        if turnovers:
            avg_turnover = np.mean(turnovers)
            ann_turnover = avg_turnover * 12
            # Estimated transaction cost impact (10bps one-way)
            tc_drag = ann_turnover * 2 * 0.0010  # 2 sides × 10bps

            print(f"  {tn:6s}: Monthly turnover={avg_turnover:.1%}, "
                  f"Annual={ann_turnover:.0%}, "
                  f"TC drag≈{tc_drag:.2%}/yr (at 10bps)")

    # ══════════════════════════════════════════════════════
    # STEP 10: FINAL COMPARISON DASHBOARD
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 10: FINAL DASHBOARD")
    print("=" * 70)

    print("\n  ┌───────────────────────────────────────────────────────────────────┐")
    print("  │             MULTI-TIER ENSEMBLE — FINAL RESULTS                  │")
    print("  ├───────────────────────────────────────────────────────────────────┤")

    # Per-tier summary
    print("  │                                                                   │")
    print("  │  PER-TIER PERFORMANCE (Best Strategy Selected):                   │")
    print("  │  ───────────────────────────────────────────────────              │")

    for tn in TIER_NAMES:
        if tn not in tier_best_returns:
            continue
        r = tier_best_returns[tn]
        perf = compute_performance(r)
        strat = tier_best_strategy[tn]
        ic_info = tier_ic_stats.get(tn, {})
        ic_val = ic_info.get('mean_ic', 0)
        ir_val = ic_info.get('ir', 0)

        print(f"  │  {tn:6s} [{strat:8s}]: IC={ic_val:+.4f}, IR={ir_val:.2f}, "
              f"Sharpe={perf['sharpe']:.2f}, MaxDD={perf['max_dd']:.1%}  │")

    # Ensemble summary
    print("  │                                                                   │")
    print("  │  ENSEMBLE PERFORMANCE:                                            │")
    print("  │  ───────────────────────────────────────────────────              │")

    print(f"  │  Method:   {best_method:<20s}                                │")
    print(f"  │  Sharpe:   {best_perf['sharpe']:>6.2f}                                          │")
    print(f"  │  Sortino:  {best_perf['sortino']:>6.2f}                                          │")
    print(f"  │  Calmar:   {best_perf['calmar']:>6.2f}                                          │")
    print(f"  │  Ann Ret:  {best_perf['ann_return']:>6.1%}                                         │")
    print(f"  │  Max DD:   {best_perf['max_dd']:>6.1%}                                         │")
    print(f"  │  % Pos Mo: {best_perf['pct_positive']:>5.0%}                                          │")
    print(f"  │  Pos Years:{n_pos:>3d}/{n_tot:<3d}                                            │")

    # Risk metrics
    if mkt_returns is not None and vix_monthly is not None:
        print("  │                                                                   │")
        print("  │  RISK METRICS:                                                    │")
        print("  │  ───────────────────────────────────────────────────              │")
        print(f"  │  β_full:     {beta_full:+.3f}                                        │")
        print(f"  │  β_crisis:   {beta_crisis:+.3f} (VIX > 25)                             │")
        print(f"  │  β_extreme:  {beta_extreme:+.3f} (VIX > 30)                             │")
        print(f"  │  FF6 α:      {alpha_ann:+.2%}/yr (t={t_alpha:.2f})                        │")

    print("  │                                                                   │")
    print("  └───────────────────────────────────────────────────────────────────┘")

    # ══════════════════════════════════════════════════════
    # STEP 11: SAVE ALL RESULTS
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 11: SAVE RESULTS")
    print("=" * 70)

    # Save best ensemble returns
    best_df = pd.DataFrame({'ensemble_return': best_returns})
    best_df.index.name = 'date'
    ensemble_path = os.path.join(DATA_DIR, "ensemble_final_returns.parquet")
    best_df.to_parquet(ensemble_path)
    print(f"  Saved: {ensemble_path}")

    # Save all method returns for comparison
    comparison_df = pd.DataFrame({
        'equal_weight': eq_blend,
        'ic_weighted': ic_blend,
        'rolling_ic': roll_blend,
        'risk_parity': rp_blend,
        'ic_riskpar': hybrid_blend,
    })
    comparison_path = os.path.join(DATA_DIR, "ensemble_method_comparison.parquet")
    comparison_df.to_parquet(comparison_path)
    print(f"  Saved: {comparison_path}")

    # Save per-tier best returns
    for tn, r in tier_best_returns.items():
        save_path = os.path.join(DATA_DIR, f"best_strategy_{tn}.parquet")
        r.to_frame('return').to_parquet(save_path)
        print(f"  Saved: {save_path}")

    # JSON summary
    def safe_float(v):
        """Convert numpy types to Python float for JSON."""
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        return v

    summary = {
        'method': 'multi_tier_ensemble_prompt4',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_min': round((time.time() - t0) / 60, 1),
        'selected_ensemble': best_method,
        'tier_strategies': {tn: tier_best_strategy.get(tn, 'raw') for tn in TIER_NAMES},
        'tier_ic_weights': {k: safe_float(round(v, 4)) for k, v in ic_weights.items()},
        'tier_ics': {tn: safe_float(round(s.get('mean_ic', 0), 4))
                     for tn, s in tier_ic_stats.items()},
        'ensemble_performance': {k: safe_float(round(v, 4)) if isinstance(v, (float, np.floating)) else v
                                  for k, v in best_perf.items()},
        'all_methods': {},
    }
    for mname, (_, perf) in methods.items():
        summary['all_methods'][mname] = {
            'sharpe': safe_float(round(perf['sharpe'], 3)),
            'max_dd': safe_float(round(perf['max_dd'], 3)),
            'sortino': safe_float(round(perf['sortino'], 3)),
            'ann_return': safe_float(round(perf['ann_return'], 4)),
        }

    if mkt_returns is not None and vix_monthly is not None:
        summary['risk_metrics'] = {
            'beta_full': safe_float(round(beta_full, 4)),
            'beta_crisis_vix25': safe_float(round(beta_crisis, 4)),
            'beta_extreme_vix30': safe_float(round(beta_extreme, 4)),
            'ff6_alpha_annual': safe_float(round(alpha_ann, 4)),
            'ff6_alpha_tstat': safe_float(round(t_alpha, 2)),
        }

    summary_path = os.path.join(RESULTS_DIR, "multi_tier_ensemble_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    elapsed = (time.time() - t0) / 60
    print(f"\n{'=' * 70}")
    print(f"DONE — STEP 6f COMPLETE. Total time: {elapsed:.1f} min")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
