#!/usr/bin/env python3
"""
Train Production Models — ONLY Polygon+FRED-available features
================================================================
Trains 4 production-grade LightGBM models using ONLY the 221 features
that exist in BOTH the GKX panel AND the Polygon feature engine output.

After this, LivePredictor achieves ~90-100% feature coverage instead of 23.5%.

Research models (lgb_mega.txt etc.) are KEPT for monthly strategic signals.
Production models are used for ALL real-time live predictions.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# ── Configuration ────────────────────────────────────────────────────────

LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 31,
    'min_child_samples': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

# Tier thresholds by log_market_cap (matching System B)
TIER_LMC = {
    'mega':  (9.21, float('inf')),   # >$10B
    'large': (7.60, 9.21),           # $2-10B
    'mid':   (6.21, 7.60),           # $500M-2B
    'small': (0.0,  6.21),           # <$500M
}

N_BOOST_ROUNDS = 500
EARLY_STOP_ROUNDS = 50
TRAIN_WINDOW_YEARS = 10
EMBARGO_MONTHS = 6

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'models', 'production')


def load_live_features():
    """Load the definitive list of features available in both panel and Polygon."""
    path = os.path.join(PROJECT_ROOT, 'wrds_pipeline', 'phase3',
                        'production_models', 'live_available_features.json')
    with open(path) as f:
        data = json.load(f)
    return data['features']


def find_target_column(panel):
    """Find the forward return column to use as target."""
    candidates = ['fwd_ret_1m', 'ret_exc_lead1m', 'ret_lead1m', 'ret_fwd_1m', 'ret_excess_1m']
    for c in candidates:
        if c in panel.columns:
            non_null = panel[c].notna().sum()
            print(f"  Target candidate '{c}': {non_null:,} non-null")
            if non_null > 100000:
                return c
    # Fallback: try to compute from ret_1m shifted
    if 'ret_1m' in panel.columns:
        print("  Computing ret_lead1m from ret_1m shift...")
        return 'ret_1m_shifted'
    raise ValueError("No target column found!")


def train_tier_model(panel, tier, live_features, target_col):
    """Train a production LightGBM for one tier on live-available features only."""
    print(f"\n{'='*60}")
    print(f"  TRAINING PRODUCTION MODEL: {tier.upper()}")
    print(f"{'='*60}")

    # Filter to tier by log_market_cap
    lmc_lo, lmc_hi = TIER_LMC[tier]
    tier_panel = panel[(panel['log_market_cap'] >= lmc_lo) &
                       (panel['log_market_cap'] < lmc_hi)].copy()
    print(f"  Tier observations: {len(tier_panel):,}")

    if len(tier_panel) < 5000:
        print(f"  ❌ Insufficient data for {tier}")
        return None

    # Features that actually exist in panel
    existing = [f for f in live_features if f in tier_panel.columns]
    print(f"  Live features in panel: {len(existing)}/{len(live_features)}")

    # Drop features with >90% NaN in this tier
    good_features = []
    for f in existing:
        null_pct = tier_panel[f].isna().mean()
        if null_pct < 0.90:
            good_features.append(f)
    print(f"  After dropping >90% NaN: {len(good_features)}")

    if len(good_features) < 10:
        print(f"  ❌ Too few usable features for {tier}")
        return None

    # Build X and y
    tier_panel = tier_panel.dropna(subset=[target_col])
    X = tier_panel[good_features].copy()
    y = tier_panel[target_col].copy()
    dates = pd.to_datetime(tier_panel['date'])

    # Fill NaN: cross-sectional median per date, then 0
    for col in good_features:
        X[col] = X.groupby(dates.dt.to_period('M'))[col].transform(
            lambda s: s.fillna(s.median())
        )
    X = X.fillna(0)

    print(f"  Feature matrix: {X.shape}")
    print(f"  Target: mean={y.mean():.4f}, std={y.std():.4f}")

    # ── Walk-forward validation ──────────────────────────────────────
    unique_months = sorted(dates.dt.to_period('M').unique())
    min_train_months = TRAIN_WINDOW_YEARS * 12
    oos_ics = []

    print(f"  Walk-forward validation ({len(unique_months)} months)...")

    # Test on last 60 months (5 years) for speed
    test_start_idx = max(min_train_months + EMBARGO_MONTHS, len(unique_months) - 60)

    for test_idx in range(test_start_idx, len(unique_months)):
        test_month = unique_months[test_idx]
        train_end_idx = test_idx - EMBARGO_MONTHS
        train_start_idx = max(0, train_end_idx - min_train_months)

        if train_end_idx <= train_start_idx:
            continue

        train_months = set(unique_months[train_start_idx:train_end_idx])
        date_months = dates.dt.to_period('M')

        train_mask = date_months.isin(train_months)
        test_mask = date_months == test_month

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(X_train) < 500 or len(X_test) < 20:
            continue

        dtrain = lgb.Dataset(X_train, label=y_train)
        dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

        model = lgb.train(
            LIGHTGBM_PARAMS,
            dtrain,
            num_boost_round=N_BOOST_ROUNDS,
            valid_sets=[dtest],
            callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                       lgb.log_evaluation(0)],
        )

        preds = model.predict(X_test)
        ic, _ = spearmanr(preds, y_test.values)
        if not np.isnan(ic):
            oos_ics.append(ic)

    if oos_ics:
        mean_ic = np.mean(oos_ics)
        std_ic = np.std(oos_ics)
        ir = mean_ic / (std_ic + 1e-8)
        pct_pos = (np.array(oos_ics) > 0).mean() * 100
        print(f"  OOS IC: {mean_ic:.4f} (std: {std_ic:.4f}, IR: {ir:.2f})")
        print(f"  % positive: {pct_pos:.0f}% over {len(oos_ics)} months")
    else:
        mean_ic = 0
        std_ic = 0
        ir = 0
        pct_pos = 0

    # ── Train FINAL model on all data ────────────────────────────────
    print(f"  Training final model on {len(X):,} observations...")
    dtrain_full = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        LIGHTGBM_PARAMS,
        dtrain_full,
        num_boost_round=N_BOOST_ROUNDS,
    )

    # Save model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, f'{tier}_production.txt')
    final_model.save_model(model_path)

    # Feature importance
    importance = final_model.feature_importance(importance_type='gain')
    imp_pairs = sorted(zip(good_features, importance), key=lambda x: -x[1])
    total_imp = sum(importance)

    print(f"\n  Top 10 features:")
    for i, (name, imp) in enumerate(imp_pairs[:10]):
        print(f"    {i+1:2d}. {name}: {imp:.0f} ({imp/total_imp*100:.1f}%)")

    # Save metadata
    meta = {
        'tier': tier,
        'features': good_features,
        'n_features': len(good_features),
        'n_observations': len(X),
        'date_range': [str(dates.min().date()), str(dates.max().date())],
        'target': target_col,
        'oos_ic_mean': round(float(mean_ic), 4),
        'oos_ic_std': round(float(std_ic), 4),
        'oos_ir': round(float(ir), 2),
        'n_test_months': len(oos_ics),
        'pct_positive_ic': round(float(pct_pos), 1),
        'top_features': [{'name': n, 'importance_pct': round(i/total_imp*100, 2)}
                         for n, i in imp_pairs[:20]],
        'params': LIGHTGBM_PARAMS,
        'trained_at': datetime.now().isoformat(),
        'model_file': model_path,
    }

    meta_path = os.path.join(OUTPUT_DIR, f'{tier}_production_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✅ Saved: {model_path}")
    print(f"  ✅ Saved: {meta_path}")
    return meta


def main():
    print("Loading GKX panel...")
    panel = pd.read_parquet(os.path.join(PROJECT_ROOT, 'data', 'wrds', 'gkx_panel.parquet'))
    print(f"  Shape: {panel.shape}")
    print(f"  Date range: {panel['date'].min()} to {panel['date'].max()}")

    # Find target
    target_col = find_target_column(panel)
    if target_col == 'ret_1m_shifted':
        panel['ret_1m_shifted'] = panel.groupby('permno')['ret_1m'].shift(-1)
        target_col = 'ret_1m_shifted'

    # Filter to post-1990 for relevance
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel[panel['date'] >= '1990-01-01']
    print(f"  Post-1990: {len(panel):,} rows")

    # Load live-available features
    live_features = load_live_features()
    print(f"  Live-available features: {len(live_features)}")

    results = {}
    for tier in ['mega', 'large', 'mid', 'small']:
        meta = train_tier_model(panel, tier, live_features, target_col)
        if meta:
            results[tier] = meta

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PRODUCTION MODEL TRAINING COMPLETE")
    print(f"{'='*60}")

    # Research model ICs for comparison
    research_ic = {'mega': 0.029, 'large': 0.046, 'mid': 0.084, 'small': 0.129}
    research_feats = {'mega': 51, 'large': 55, 'mid': 68, 'small': 69}

    print(f"\n  {'Tier':<8} {'Research':>12} {'Production':>12} {'Retention':>10}")
    print(f"  {'':─<8} {'IC (feats)':─>12} {'IC (feats)':─>12} {'':─>10}")

    for tier in ['mega', 'large', 'mid', 'small']:
        if tier in results:
            prod_ic = results[tier]['oos_ic_mean']
            prod_feats = results[tier]['n_features']
            res_ic = research_ic[tier]
            res_feats = research_feats[tier]
            retention = prod_ic / res_ic * 100 if res_ic > 0 else 0
            print(f"  {tier:<8} {res_ic:.4f} ({res_feats:3d})  {prod_ic:.4f} ({prod_feats:3d})  {retention:6.0f}%")

    # Save the registry
    registry = {
        'type': 'production',
        'description': 'Models trained on ONLY Polygon+FRED-available features',
        'trained_at': datetime.now().isoformat(),
        'tiers': {}
    }
    for tier, meta in results.items():
        registry['tiers'][tier] = {
            'model_file': meta['model_file'],
            'features': meta['features'],
            'n_features': meta['n_features'],
            'oos_ic': meta['oos_ic_mean'],
        }

    reg_path = os.path.join(OUTPUT_DIR, 'production_registry.json')
    with open(reg_path, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"\n  Registry: {reg_path}")


if __name__ == '__main__':
    main()
