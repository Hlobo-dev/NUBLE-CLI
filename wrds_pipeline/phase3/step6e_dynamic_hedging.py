"""
PHASE 3 — STEP 6e: DYNAMIC HEDGING + REGIME-ADAPTIVE POSITION SIZING
======================================================================
Prompt 3 Execution.

PROBLEM:
  step6d produced strong per-tier signals (IC 0.02-0.10) but the L/S
  portfolio has no risk management. It takes full exposure regardless
  of market regime, meaning:
  - In high VIX (2008, 2020, 2022): large drawdowns erase gains
  - Conditional beta is too high (β>0.30 in crises)
  - Max drawdown > -25%
  - No factor hedging (exposed to MKT, SMB, HML, UMD)

SOLUTION:
  1. VIX-based exposure scaling (reduce exposure as VIX rises)
  2. Regime detection via macro indicators (HMM-like rule-based)
  3. Factor hedging: neutralize MKT + SMB + HML + UMD exposures
  4. Tail hedging: hard cutoff when VIX > 35 (crisis = cash)
  5. Rolling conditional beta targeting (keep β < 0.20)

EXPECTED IMPROVEMENTS:
  - Max drawdown: > -25% → < -15%
  - Conditional beta: 0.30+ → < 0.20
  - Net Sharpe: improve by reducing left tail
  - Crisis performance: outperform when VIX > 30

Author: Claude × Humberto
"""

import pandas as pd
import numpy as np
import os
import json
import time
import warnings
from scipy import stats

warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════
# TIER DEFINITIONS
# ════════════════════════════════════════════════════════════

TIER_DEFS = {
    "mega":  {"label": "Mega-Cap (>$10B)",     "min_lmc": 9.21, "max_lmc": np.inf},
    "large": {"label": "Large-Cap ($2-10B)",   "min_lmc": 7.60, "max_lmc": 9.21},
    "mid":   {"label": "Mid-Cap ($500M-2B)",   "min_lmc": 6.21, "max_lmc": 7.60},
    "small": {"label": "Small-Cap ($100M-500M)","min_lmc": 4.61, "max_lmc": 6.21},
}

# ════════════════════════════════════════════════════════════
# VIX REGIME THRESHOLDS (monthly data)
# ════════════════════════════════════════════════════════════
# Based on VIX historical percentiles:
#   VIX < 15: Low vol (calm markets) → full exposure
#   15-20: Normal → full exposure
#   20-25: Elevated → reduce to 70%
#   25-30: High → reduce to 40%
#   30-35: Stress → reduce to 20%
#   > 35: Crisis → reduce to 0% (cash)

VIX_EXPOSURE_MAP = [
    (15, 1.00),   # VIX ≤ 15: full exposure
    (20, 1.00),   # 15-20: full exposure
    (25, 0.70),   # 20-25: reduce to 70%
    (30, 0.40),   # 25-30: reduce to 40%
    (35, 0.20),   # 30-35: reduce to 20%
    (np.inf, 0.0), # > 35: zero exposure (cash)
]


def vix_to_exposure(vix_val):
    """Map VIX level to exposure fraction [0, 1]."""
    if pd.isna(vix_val):
        return 0.80  # conservative default if VIX unavailable
    for threshold, exposure in VIX_EXPOSURE_MAP:
        if vix_val <= threshold:
            return exposure
    return 0.0


# ════════════════════════════════════════════════════════════
# REGIME DETECTION (Rule-Based, No Lookahead)
# ════════════════════════════════════════════════════════════

def detect_regime(macro_row):
    """
    Classify macro regime from monthly FRED data.
    Returns regime label and risk_level (0-1).

    Regimes:
      'expansion': normal growth, low vol → full exposure
      'late_cycle': rising rates, tight labor → reduce slightly
      'slowdown': falling leading indicators → caution
      'recession': NBER recession or inverted yield curve → defensive
      'crisis': VIX spike + recession → minimal exposure
    """
    regime = 'expansion'
    risk_level = 0.2

    vix = macro_row.get('vix', 20)
    yield_curve = macro_row.get('yield_curve_10y3m', 1.0)
    nber = macro_row.get('nber_recession', 0)
    leading = macro_row.get('leading_index', 0)
    credit_spread = macro_row.get('credit_spread', 1.0)

    # NBER recession flag
    if nber == 1:
        regime = 'recession'
        risk_level = 0.7

    # Inverted yield curve (< 0 for 2+ months is historically predictive)
    if not pd.isna(yield_curve) and yield_curve < 0:
        if regime == 'expansion':
            regime = 'late_cycle'
        risk_level = max(risk_level, 0.5)

    # Credit spread widening (> 2.5% is stress, > 3.5% is crisis)
    if not pd.isna(credit_spread):
        if credit_spread > 3.5:
            regime = 'crisis'
            risk_level = max(risk_level, 0.9)
        elif credit_spread > 2.5:
            regime = 'slowdown'
            risk_level = max(risk_level, 0.6)

    # VIX override (regime detection)
    if not pd.isna(vix):
        if vix > 35:
            regime = 'crisis'
            risk_level = max(risk_level, 0.95)
        elif vix > 30:
            regime = 'recession' if regime in ['slowdown', 'crisis'] else 'slowdown'
            risk_level = max(risk_level, 0.7)

    # Leading index declining
    if not pd.isna(leading) and leading < -0.5:
        if regime == 'expansion':
            regime = 'slowdown'
        risk_level = max(risk_level, 0.4)

    return regime, risk_level


def regime_to_exposure(regime, risk_level):
    """Convert regime + risk level to exposure multiplier."""
    regime_base = {
        'expansion': 1.00,
        'late_cycle': 0.80,
        'slowdown': 0.60,
        'recession': 0.35,
        'crisis': 0.10,
    }
    base = regime_base.get(regime, 0.70)
    # Further scale by risk_level
    exposure = base * (1.0 - 0.3 * risk_level)  # risk_level modulates down
    return max(0.0, min(1.0, exposure))


# ════════════════════════════════════════════════════════════
# FACTOR HEDGING
# ════════════════════════════════════════════════════════════

def compute_rolling_betas(returns, factors, window=36):
    """
    Compute rolling factor betas using 36-month OLS.
    Returns DataFrame of rolling betas for each factor.
    No lookahead — uses only past data.
    """
    factor_cols = [c for c in factors.columns if c != 'date']
    betas = pd.DataFrame(index=returns.index, columns=factor_cols, dtype=float)

    for i in range(window, len(returns)):
        y = returns.iloc[i-window:i].values
        X = factors[factor_cols].iloc[i-window:i].values
        # Add intercept
        X_with_const = np.column_stack([np.ones(len(X)), X])
        try:
            coef = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            betas.iloc[i] = coef[1:]  # skip intercept
        except:
            pass

    return betas


def hedge_factor_exposure(ls_return, factor_returns, betas_row):
    """
    Compute hedged return by subtracting factor exposures.
    hedged_return = raw_return - Σ(beta_i × factor_i_return)
    """
    if betas_row is None or betas_row.isna().all():
        return ls_return

    hedge = 0
    for factor in betas_row.index:
        if factor in factor_returns.index and not pd.isna(betas_row[factor]):
            hedge += betas_row[factor] * factor_returns[factor]
    return ls_return - hedge


# ════════════════════════════════════════════════════════════
# PORTFOLIO CONSTRUCTION
# ════════════════════════════════════════════════════════════

def build_ls_returns_per_tier(predictions_df, n_quantiles=10):
    """
    Build monthly long-short returns from predictions.
    Long top decile, short bottom decile.
    Returns Series of monthly L/S returns.
    """
    monthly_returns = []

    for date, group in predictions_df.groupby('date'):
        if len(group) < n_quantiles * 2:
            continue

        # Rank by prediction
        group = group.copy()
        try:
            group['decile'] = pd.qcut(group['prediction'], n_quantiles,
                                       labels=False, duplicates='drop')
        except:
            continue

        n_dec = group['decile'].nunique()
        if n_dec < 2:
            continue

        top = group[group['decile'] == group['decile'].max()]
        bot = group[group['decile'] == group['decile'].min()]

        long_ret = top['fwd_ret_1m'].mean()
        short_ret = bot['fwd_ret_1m'].mean()
        ls_ret = long_ret - short_ret

        monthly_returns.append({
            'date': date,
            'ls_return': ls_ret,
            'long_return': long_ret,
            'short_return': short_ret,
            'n_long': len(top),
            'n_short': len(bot),
            'n_total': len(group),
        })

    return pd.DataFrame(monthly_returns)


# ════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ════════════════════════════════════════════════════════════

def compute_performance(returns_series, label=""):
    """Compute performance metrics for a monthly return series."""
    r = returns_series.dropna()
    if len(r) < 12:
        return {}

    ann_return = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0

    # Sortino (downside vol)
    neg_returns = r[r < 0]
    downside_vol = neg_returns.std() * np.sqrt(12) if len(neg_returns) > 0 else ann_vol
    sortino = ann_return / downside_vol if downside_vol > 0 else 0

    # Skewness
    skew = r.skew()

    return {
        'label': label,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'calmar': calmar,
        'skew': skew,
        'n_months': len(r),
        'pct_positive': (r > 0).mean(),
        'mean_monthly': r.mean(),
        'worst_month': r.min(),
        'best_month': r.max(),
    }


def compute_conditional_beta(ls_returns, mkt_returns, vix_series,
                              vix_threshold=25):
    """
    Compute beta conditional on VIX > threshold.
    This measures crisis exposure.
    """
    # Align
    common = ls_returns.index.intersection(mkt_returns.index).intersection(vix_series.index)
    if len(common) < 12:
        return np.nan, np.nan

    ls = ls_returns.loc[common]
    mkt = mkt_returns.loc[common]
    vix = vix_series.loc[common]

    # Overall beta
    mask_all = ~(ls.isna() | mkt.isna())
    if mask_all.sum() < 12:
        return np.nan, np.nan
    try:
        beta_full = np.cov(ls[mask_all], mkt[mask_all])[0, 1] / np.var(mkt[mask_all])
    except:
        beta_full = np.nan

    # Conditional beta (VIX > threshold)
    crisis = vix > vix_threshold
    mask_crisis = crisis & ~(ls.isna() | mkt.isna())
    if mask_crisis.sum() < 6:
        return beta_full, np.nan

    try:
        beta_crisis = np.cov(ls[mask_crisis], mkt[mask_crisis])[0, 1] / np.var(mkt[mask_crisis])
    except:
        beta_crisis = np.nan

    return beta_full, beta_crisis


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("=" * 70)
    print("DYNAMIC HEDGING + REGIME-ADAPTIVE SIZING (Prompt 3)")
    print("=" * 70)
    print("  VIX-based exposure scaling")
    print("  Macro regime detection (yield curve, credit spreads, NBER)")
    print("  Rolling factor hedging (FF6)")
    print("  Tail risk management (VIX > 35 → cash)")
    print()

    # ── Load predictions from step6d ──────────────────────────────────
    print("Loading step6d predictions...")
    all_preds = []
    for tier_name in TIER_DEFS:
        path = os.path.join(DATA_DIR, f"curated_predictions_{tier_name}.parquet")
        if not os.path.exists(path):
            print(f"  ⚠️ Missing {path}, skipping {tier_name}")
            continue
        df = pd.read_parquet(path)
        df['date'] = pd.to_datetime(df['date'])
        all_preds.append(df)
        print(f"  {tier_name}: {len(df):,} predictions")

    if not all_preds:
        print("ERROR: No predictions found. Run step6d first.")
        return

    # ── Load FRED macro data (monthly) ────────────────────────────────
    print("\nLoading macro data...")
    fred_monthly_path = os.path.join(DATA_DIR, "fred_monthly.parquet")
    fred_daily_path = os.path.join(DATA_DIR, "fred_daily.parquet")

    macro_monthly = pd.read_parquet(fred_monthly_path)
    macro_monthly['date'] = pd.to_datetime(macro_monthly['date'])
    print(f"  Monthly macro: {len(macro_monthly):,} months × {macro_monthly.shape[1]} cols")

    fred_daily = pd.read_parquet(fred_daily_path)
    fred_daily['date'] = pd.to_datetime(fred_daily['date'])
    print(f"  Daily macro: {len(fred_daily):,} days × {fred_daily.shape[1]} cols")

    # Compute monthly VIX (average of daily VIX within each month)
    fred_daily['month_end'] = fred_daily['date'] + pd.offsets.MonthEnd(0)
    vix_monthly = fred_daily.groupby('month_end')['vix'].mean().reset_index()
    vix_monthly = vix_monthly.rename(columns={'month_end': 'date'})
    print(f"  Monthly VIX: {vix_monthly.dropna().shape[0]} months")

    # Merge VIX into macro_monthly
    macro = macro_monthly.merge(vix_monthly, on='date', how='left', suffixes=('_orig', ''))
    if 'vix_orig' in macro.columns:
        macro['vix'] = macro['vix'].fillna(macro['vix_orig'])
        macro = macro.drop(columns=['vix_orig'])

    # ── Load FF factors (for factor hedging) ──────────────────────────
    print("\nLoading Fama-French factors...")
    ff_path = os.path.join(DATA_DIR, "ff_factors_monthly.parquet")
    if not os.path.exists(ff_path):
        ff_path = os.path.join(DATA_DIR, "ff_monthly.parquet")
    if os.path.exists(ff_path):
        ff = pd.read_parquet(ff_path)
        ff['date'] = pd.to_datetime(ff['date'])
        # Standardize column names
        ff.columns = [c.lower().strip() for c in ff.columns]
        # Snap dates to month-end to align with predictions
        ff['date'] = ff['date'] + pd.offsets.MonthEnd(0)
        print(f"  FF factors: {len(ff):,} months, cols={ff.columns.tolist()}")
        # Check scale — values should be in decimal
        sample = ff[[c for c in ff.columns if c not in ['date', 'rf']]].abs().median()
        if sample.mean() > 0.5:  # likely in percent
            for c in ff.columns:
                if c not in ['date']:
                    ff[c] = ff[c] / 100
            print("  Converted factor returns from % to decimal")
        else:
            print(f"  Factor returns already in decimal (median abs={sample.mean():.4f})")
    else:
        print("  ⚠️ FF factors not found, will skip factor hedging")
        ff = None

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: BUILD UNHEDGED L/S RETURNS PER TIER
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1: BUILD UNHEDGED L/S RETURNS PER TIER")
    print("=" * 70)

    tier_ls = {}
    for tier_name in TIER_DEFS:
        path = os.path.join(DATA_DIR, f"curated_predictions_{tier_name}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        df['date'] = pd.to_datetime(df['date'])

        ls_df = build_ls_returns_per_tier(df)
        if ls_df.empty:
            continue

        ls_df['date'] = pd.to_datetime(ls_df['date'])
        ls_df = ls_df.set_index('date').sort_index()
        tier_ls[tier_name] = ls_df

        perf = compute_performance(ls_df['ls_return'], label=tier_name)
        print(f"\n  {TIER_DEFS[tier_name]['label']}: "
              f"Sharpe={perf['sharpe']:.2f}, MaxDD={perf['max_dd']:.1%}, "
              f"Ann.Ret={perf['ann_return']:.1%}, n={perf['n_months']}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: VIX-BASED EXPOSURE SCALING
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2: VIX-BASED EXPOSURE SCALING")
    print("=" * 70)

    vix_ts = vix_monthly.set_index('date')['vix'].sort_index()
    print(f"  VIX range: {vix_ts.min():.1f} to {vix_ts.max():.1f}")
    print(f"  VIX > 25: {(vix_ts > 25).sum()} months")
    print(f"  VIX > 30: {(vix_ts > 30).sum()} months")
    print(f"  VIX > 35: {(vix_ts > 35).sum()} months")

    for tier_name, ls_df in tier_ls.items():
        # Align VIX to L/S dates
        exposure = ls_df.index.map(lambda d: vix_to_exposure(
            vix_ts.get(d, vix_ts.asof(d) if d >= vix_ts.index.min() else np.nan)
        ))
        ls_df['vix_exposure'] = exposure.values
        ls_df['vix_scaled_return'] = ls_df['ls_return'] * ls_df['vix_exposure']

        # Report
        avg_exp = ls_df['vix_exposure'].mean()
        perf_raw = compute_performance(ls_df['ls_return'], 'raw')
        perf_scaled = compute_performance(ls_df['vix_scaled_return'], 'vix_scaled')
        print(f"\n  {tier_name}: avg_exposure={avg_exp:.2f}")
        print(f"    Raw:    Sharpe={perf_raw['sharpe']:.2f}, MaxDD={perf_raw['max_dd']:.1%}")
        print(f"    Scaled: Sharpe={perf_scaled['sharpe']:.2f}, MaxDD={perf_scaled['max_dd']:.1%}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: MACRO REGIME DETECTION
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3: MACRO REGIME DETECTION")
    print("=" * 70)

    # Compute credit spread from daily data → monthly
    if 'credit_spread' in fred_daily.columns:
        cs_monthly = fred_daily.groupby('month_end')['credit_spread'].mean().reset_index()
        cs_monthly = cs_monthly.rename(columns={'month_end': 'date'})
        macro = macro.merge(cs_monthly, on='date', how='left', suffixes=('_orig', ''))
        if 'credit_spread_orig' in macro.columns:
            macro['credit_spread'] = macro['credit_spread'].fillna(macro['credit_spread_orig'])
            macro = macro.drop(columns=['credit_spread_orig'])

    # Detect regime for each month
    macro = macro.sort_values('date').reset_index(drop=True)
    regimes = []
    for _, row in macro.iterrows():
        regime, risk = detect_regime(row)
        regimes.append({'date': row['date'], 'regime': regime,
                        'risk_level': risk,
                        'regime_exposure': regime_to_exposure(regime, risk)})
    regime_df = pd.DataFrame(regimes).set_index('date')

    # Report regime distribution
    print(f"\n  Regime distribution (full history):")
    regime_counts = regime_df['regime'].value_counts()
    for r, cnt in regime_counts.items():
        pct = cnt / len(regime_df) * 100
        avg_exp = regime_df.loc[regime_df['regime'] == r, 'regime_exposure'].mean()
        print(f"    {r:>15}: {cnt:>5} months ({pct:5.1f}%), avg exposure={avg_exp:.2f}")

    # Apply regime exposure to L/S returns
    for tier_name, ls_df in tier_ls.items():
        regime_exp = ls_df.index.map(
            lambda d: regime_df.loc[regime_df.index.asof(d), 'regime_exposure']
            if d >= regime_df.index.min() else 0.80
        )
        ls_df['regime_exposure'] = regime_exp.values

        # Combined exposure = min(VIX, Regime) — take the more conservative
        ls_df['combined_exposure'] = np.minimum(
            ls_df['vix_exposure'], ls_df['regime_exposure']
        )
        ls_df['regime_scaled_return'] = ls_df['ls_return'] * ls_df['combined_exposure']

        perf = compute_performance(ls_df['regime_scaled_return'], 'regime')
        avg_comb = ls_df['combined_exposure'].mean()
        print(f"\n  {tier_name}: avg_combined_exposure={avg_comb:.2f}")
        print(f"    Combined: Sharpe={perf['sharpe']:.2f}, MaxDD={perf['max_dd']:.1%}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: FACTOR HEDGING (FF6)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 4: FACTOR HEDGING (FF6)")
    print("=" * 70)

    if ff is not None:
        # Find factor columns (exclude rf and date)
        factor_cols = [c for c in ff.columns if c not in ['date', 'rf']]
        # Rename Mkt-RF → mktrf if needed
        rename_map = {'mkt-rf': 'mktrf', 'mom   ': 'umd'}
        for old, new in rename_map.items():
            if old in ff.columns and new not in ff.columns:
                ff = ff.rename(columns={old: new})
                factor_cols = [new if c == old else c for c in factor_cols]

        ff_aligned = ff.set_index('date')[factor_cols].sort_index()
        # Remove duplicate dates if any
        ff_aligned = ff_aligned[~ff_aligned.index.duplicated(keep='last')]
        print(f"  Factor columns: {factor_cols}")
        print(f"  Factor date range: {ff_aligned.index.min()} to {ff_aligned.index.max()}")

        for tier_name, ls_df in tier_ls.items():
            # Align dates
            common_dates = ls_df.index.intersection(ff_aligned.index)
            if len(common_dates) < 48:
                print(f"  {tier_name}: not enough overlapping dates for factor hedging")
                continue

            ls_returns = ls_df.loc[common_dates, 'regime_scaled_return']
            ff_sub = ff_aligned.loc[common_dates]

            # Rolling betas (36-month window)
            betas = compute_rolling_betas(ls_returns, ff_sub, window=36)

            # Compute hedged returns
            hedged = []
            for dt in common_dates:
                if dt not in betas.index or betas.loc[dt].isna().all():
                    hedged.append(ls_returns.loc[dt])
                else:
                    h = hedge_factor_exposure(
                        ls_returns.loc[dt],
                        ff_sub.loc[dt],
                        betas.loc[dt]
                    )
                    hedged.append(h)

            ls_df.loc[common_dates, 'hedged_return'] = hedged

            # Fill dates without hedging
            ls_df['hedged_return'] = ls_df['hedged_return'].fillna(
                ls_df['regime_scaled_return']
            )

            perf_pre = compute_performance(ls_df['regime_scaled_return'], 'pre-hedge')
            perf_post = compute_performance(ls_df['hedged_return'], 'post-hedge')
            print(f"\n  {tier_name}:")
            print(f"    Pre-hedge:  Sharpe={perf_pre['sharpe']:.2f}, MaxDD={perf_pre['max_dd']:.1%}")
            print(f"    Post-hedge: Sharpe={perf_post['sharpe']:.2f}, MaxDD={perf_post['max_dd']:.1%}")

            # Report average betas
            avg_betas = betas.mean()
            print(f"    Avg betas: {dict(avg_betas.round(3).dropna())}")
    else:
        # No factor data — use regime-scaled as final
        for tier_name, ls_df in tier_ls.items():
            ls_df['hedged_return'] = ls_df['regime_scaled_return']
        print("  Skipping factor hedging (no FF data)")

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: FINAL COMPARISON — ALL STRATEGIES
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 5: FINAL COMPARISON")
    print("=" * 70)

    for tier_name, ls_df in tier_ls.items():
        label = TIER_DEFS[tier_name]['label']
        print(f"\n  ╔══════════════════════════════════════════════════╗")
        print(f"  ║  {label:^46}  ║")
        print(f"  ╠══════════════════════════════════════════════════╣")

        strategies = [
            ('Raw L/S',          'ls_return'),
            ('VIX-Scaled',       'vix_scaled_return'),
            ('Regime-Scaled',    'regime_scaled_return'),
            ('Factor-Hedged',    'hedged_return'),
        ]

        for strat_name, col in strategies:
            if col not in ls_df.columns:
                continue
            p = compute_performance(ls_df[col], strat_name)
            if not p:
                continue
            print(f"  ║  {strat_name:<16} Sharpe={p['sharpe']:>5.2f}  "
                  f"MaxDD={p['max_dd']:>6.1%}  Sortino={p['sortino']:>5.2f}  ║")

        print(f"  ╚══════════════════════════════════════════════════╝")

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: CONDITIONAL BETA + CRISIS ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 6: CONDITIONAL BETA + CRISIS ANALYSIS")
    print("=" * 70)

    # Get market returns
    mkt_returns = None
    if ff is not None and 'mktrf' in ff_aligned.columns:
        mkt_returns = ff_aligned['mktrf']

    for tier_name, ls_df in tier_ls.items():
        label = TIER_DEFS[tier_name]['label']

        # Raw vs hedged conditional beta
        if mkt_returns is not None:
            common = ls_df.index.intersection(mkt_returns.index).intersection(vix_ts.index)
            if len(common) >= 24:
                # Raw
                beta_raw, cond_beta_raw = compute_conditional_beta(
                    ls_df.loc[common, 'ls_return'],
                    mkt_returns.loc[common],
                    vix_ts.loc[common], 25)

                # Hedged
                beta_hedged, cond_beta_hedged = compute_conditional_beta(
                    ls_df.loc[common, 'hedged_return'],
                    mkt_returns.loc[common],
                    vix_ts.loc[common], 25)

                print(f"\n  {label}:")
                print(f"    Raw:    β_full={beta_raw:+.3f}, β_crisis(VIX>25)={cond_beta_raw:+.3f}")
                print(f"    Hedged: β_full={beta_hedged:+.3f}, β_crisis(VIX>25)={cond_beta_hedged:+.3f}")

        # Crisis period analysis
        crisis_dates = vix_ts[vix_ts > 25].index
        crisis_in_tier = ls_df.index.intersection(crisis_dates)
        if len(crisis_in_tier) > 0:
            raw_crisis = ls_df.loc[crisis_in_tier, 'ls_return'].mean() * 12
            hedged_crisis = ls_df.loc[crisis_in_tier, 'hedged_return'].mean() * 12
            n_crisis = len(crisis_in_tier)
            print(f"    Crisis months (n={n_crisis}): raw_ann={raw_crisis:+.1%}, hedged_ann={hedged_crisis:+.1%}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: IC-WEIGHTED MULTI-TIER BLEND (Preview for Prompt 4)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 7: IC-WEIGHTED MULTI-TIER BLEND (Preview)")
    print("=" * 70)

    # Load step6d results for IC weights
    summary_path = os.path.join(RESULTS_DIR, "curated_multi_universe_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            step6d_results = json.load(f)

        tier_ics = {}
        for tn in tier_ls:
            if tn in step6d_results.get('tiers', {}):
                tier_ics[tn] = max(0.001, step6d_results['tiers'][tn]['ic'])

        total_ic = sum(tier_ics.values())
        ic_weights = {tn: ic / total_ic for tn, ic in tier_ics.items()}
        print(f"  IC weights: {dict((k, round(v, 3)) for k, v in ic_weights.items())}")

        # Build blended return
        # Align all tier returns to common dates
        all_dates = set()
        for ls_df in tier_ls.values():
            all_dates.update(ls_df.index)
        all_dates = sorted(all_dates)

        blended_raw = pd.Series(0.0, index=pd.DatetimeIndex(all_dates))
        blended_hedged = pd.Series(0.0, index=pd.DatetimeIndex(all_dates))
        weight_sum = pd.Series(0.0, index=pd.DatetimeIndex(all_dates))

        for tn, ls_df in tier_ls.items():
            w = ic_weights.get(tn, 0)
            for dt in ls_df.index:
                if dt in blended_raw.index:
                    blended_raw[dt] += w * ls_df.loc[dt, 'ls_return']
                    blended_hedged[dt] += w * ls_df.loc[dt, 'hedged_return']
                    weight_sum[dt] += w

        # Normalize (some dates may not have all tiers)
        mask = weight_sum > 0
        blended_raw[mask] = blended_raw[mask] / weight_sum[mask]
        blended_hedged[mask] = blended_hedged[mask] / weight_sum[mask]

        # Only keep dates with data
        blended_raw = blended_raw[mask].dropna()
        blended_hedged = blended_hedged[mask].dropna()

        perf_blend_raw = compute_performance(blended_raw, 'IC-weighted raw')
        perf_blend_hedged = compute_performance(blended_hedged, 'IC-weighted hedged')

        print(f"\n  IC-Weighted Blend (Raw):    Sharpe={perf_blend_raw['sharpe']:.2f}, "
              f"MaxDD={perf_blend_raw['max_dd']:.1%}, Sortino={perf_blend_raw['sortino']:.2f}")
        print(f"  IC-Weighted Blend (Hedged): Sharpe={perf_blend_hedged['sharpe']:.2f}, "
              f"MaxDD={perf_blend_hedged['max_dd']:.1%}, Sortino={perf_blend_hedged['sortino']:.2f}")

        # Conditional beta of blended
        if mkt_returns is not None:
            common = blended_hedged.index.intersection(mkt_returns.index).intersection(vix_ts.index)
            if len(common) >= 24:
                beta_full, beta_crisis = compute_conditional_beta(
                    blended_hedged.loc[common],
                    mkt_returns.loc[common],
                    vix_ts.loc[common], 25)
                print(f"  Blend β_full={beta_full:+.3f}, β_crisis(VIX>25)={beta_crisis:+.3f}")

        # Year-by-year breakdown of hedged blend
        print(f"\n  Year-by-Year Hedged Blend:")
        blended_hedged_df = blended_hedged.to_frame('ret')
        blended_hedged_df['year'] = blended_hedged_df.index.year
        yearly = blended_hedged_df.groupby('year')['ret'].agg(['mean', 'std', 'count'])
        yearly['ann_ret'] = yearly['mean'] * 12
        yearly['sharpe'] = (yearly['mean'] / yearly['std'] * np.sqrt(12)).round(2)
        for yr, row in yearly.iterrows():
            ann = row['ann_ret']
            sh = row['sharpe']
            n = int(row['count'])
            marker = '[OK]' if ann > 0 else '[XX]'
            print(f"    {yr}: {ann:+.1%} (Sharpe={sh:+.2f}, n={n}) {marker}")

    # ══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save per-tier hedged returns
    for tier_name, ls_df in tier_ls.items():
        save_path = os.path.join(DATA_DIR, f"hedged_returns_{tier_name}.parquet")
        ls_df.to_parquet(save_path)
        print(f"  Saved: {save_path}")

    # Save blended returns
    if 'blended_hedged' in dir():
        blend_df = pd.DataFrame({
            'hedged_return': blended_hedged,
            'raw_return': blended_raw
        })
        blend_path = os.path.join(DATA_DIR, "blended_hedged_returns.parquet")
        blend_df.to_parquet(blend_path)
        print(f"  Saved: {blend_path}")

    # Save summary
    summary = {
        'method': 'dynamic_hedging_prompt3',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_min': round((time.time() - t0) / 60, 1),
        'tiers': {},
    }
    for tier_name, ls_df in tier_ls.items():
        for strat, col in [('raw', 'ls_return'), ('hedged', 'hedged_return')]:
            if col in ls_df.columns:
                p = compute_performance(ls_df[col])
                summary['tiers'][f"{tier_name}_{strat}"] = {
                    'sharpe': round(float(p.get('sharpe', 0)), 3),
                    'max_dd': round(float(p.get('max_dd', 0)), 3),
                    'sortino': round(float(p.get('sortino', 0)), 3),
                    'ann_return': round(float(p.get('ann_return', 0)), 4),
                    'n_months': int(p.get('n_months', 0)),
                }

    if 'perf_blend_hedged' in dir():
        summary['blend_hedged'] = {
            'sharpe': round(float(perf_blend_hedged.get('sharpe', 0)), 3),
            'max_dd': round(float(perf_blend_hedged.get('max_dd', 0)), 3),
            'sortino': round(float(perf_blend_hedged.get('sortino', 0)), 3),
        }

    summary_path = os.path.join(RESULTS_DIR, "dynamic_hedging_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")

    elapsed = (time.time() - t0) / 60
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {elapsed:.1f} min")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
