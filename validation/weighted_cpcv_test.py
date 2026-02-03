#!/usr/bin/env python3
"""
CPCV + PBO Test with AFML Sample Weighting
Compare PBO between weighted and unweighted training.
"""
import sys
sys.path.insert(0, '/Users/humbertolobo/Desktop/bolt.new-main/NUBLE-CLI')

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

from src.institutional.validation.sample_weights import AFMLSampleWeights


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate simple features."""
    features = pd.DataFrame(index=df.index)
    
    for window in [5, 10, 20, 50]:
        features[f'ret_{window}'] = df['close'].pct_change(window)
        features[f'sma_ratio_{window}'] = df['close'] / df['close'].rolling(window).mean()
        features[f'vol_{window}'] = df['close'].pct_change().rolling(window).std()
    
    if 'volume' in df.columns:
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    return features.dropna()


def generate_labels(close: pd.Series, threshold: float = 0.01, max_hold: int = 10) -> pd.Series:
    """Generate triple-barrier-like labels."""
    labels = pd.Series(index=close.index, dtype=float)
    
    for i in range(len(close) - max_hold):
        entry_price = close.iloc[i]
        future_prices = close.iloc[i+1:i+max_hold+1]
        
        for j, price in enumerate(future_prices):
            ret = (price - entry_price) / entry_price
            if ret >= threshold:
                labels.iloc[i] = 1
                break
            elif ret <= -threshold:
                labels.iloc[i] = -1
                break
    
    return labels.dropna()


def compute_t1_times(df: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Compute t1 times for sample weighting."""
    t1 = pd.Series(index=labels.index, dtype='datetime64[ns]')
    
    for idx in labels.index:
        loc = df.index.get_loc(idx)
        end_loc = min(loc + 10, len(df) - 1)
        t1.loc[idx] = df.index[end_loc]
    
    return t1


def cpcv_pbo_analysis(df: pd.DataFrame, use_weights: bool = True, n_splits: int = 10, verbose: bool = False):
    """
    Combinatorial Purged Cross-Validation with Probability of Backtest Overfitting.
    
    Returns PBO and performance metrics.
    """
    # Prepare data
    features = generate_features(df)
    labels = generate_labels(df['close'], threshold=0.01, max_hold=10)
    labels = labels[labels.isin([-1, 1])].dropna()
    
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]
    
    if len(X) < 200:
        return {'pbo': 1.0, 'sharpe': 0.0, 'status': 'insufficient_data'}
    
    # Compute weights
    weighter = AFMLSampleWeights(decay_factor=0.95)
    if use_weights:
        try:
            t1 = compute_t1_times(df, y)
            weights = weighter.get_sample_weights(
                event_times=common_idx,
                t1_times=t1.loc[common_idx],
                close_idx=df.index
            )
            sample_weights = weights.reindex(common_idx).fillna(1.0).values
        except Exception as e:
            if verbose:
                print(f"  Weight computation failed: {e}")
            sample_weights = np.ones(len(y))
    else:
        sample_weights = None
    
    # Split data into n_splits groups
    n_samples = len(X)
    indices = np.arange(n_samples)
    split_size = n_samples // n_splits
    groups = [indices[i*split_size:(i+1)*split_size] for i in range(n_splits)]
    
    # Generate all combinations for train/test
    n_train = n_splits // 2
    train_combos = list(combinations(range(n_splits), n_train))
    
    # Limit number of combinations for speed
    max_combos = 50
    if len(train_combos) > max_combos:
        np.random.seed(42)
        train_combos = [train_combos[i] for i in np.random.choice(len(train_combos), max_combos, replace=False)]
    
    is_returns = []  # In-sample returns
    oos_returns = []  # Out-of-sample returns
    
    scaler = StandardScaler()
    
    for combo in train_combos:
        # Get train/test indices
        train_idx = np.concatenate([groups[i] for i in combo])
        test_groups = [i for i in range(n_splits) if i not in combo]
        test_idx = np.concatenate([groups[i] for i in test_groups])
        
        # Purge: remove samples near boundary
        purge = 5
        train_mask = np.ones(len(train_idx), dtype=bool)
        for tg in test_groups:
            for ti in combo:
                if abs(ti - tg) == 1:
                    boundary = groups[ti][-purge:] if ti < tg else groups[ti][:purge]
                    train_mask &= ~np.isin(train_idx, boundary)
        
        train_idx = train_idx[train_mask]
        
        if len(train_idx) < 50 or len(test_idx) < 20:
            continue
        
        # Prepare data
        X_train = X.iloc[train_idx].values
        y_train = y.iloc[train_idx].values
        X_test = X.iloc[test_idx].values
        y_test = y.iloc[test_idx].values
        
        # Scale
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=20, random_state=42)
        
        if sample_weights is not None and use_weights:
            w_train = sample_weights[train_idx]
            model.fit(X_train_scaled, y_train, sample_weight=w_train)
        else:
            model.fit(X_train_scaled, y_train)
        
        # Predict probabilities
        is_proba = model.predict_proba(X_train_scaled)
        oos_proba = model.predict_proba(X_test_scaled)
        
        # Compute returns (simplified)
        is_pred = model.predict(X_train_scaled)
        oos_pred = model.predict(X_test_scaled)
        
        # Accuracy as proxy for return
        is_acc = np.mean(is_pred == y_train)
        oos_acc = np.mean(oos_pred == y_test)
        
        is_returns.append(is_acc)
        oos_returns.append(oos_acc)
    
    if len(is_returns) < 5:
        return {'pbo': 1.0, 'sharpe': 0.0, 'status': 'insufficient_combos'}
    
    # Compute PBO
    is_returns = np.array(is_returns)
    oos_returns = np.array(oos_returns)
    
    # Rank correlation
    is_ranks = np.argsort(np.argsort(-is_returns))  # Higher is better
    oos_ranks = np.argsort(np.argsort(-oos_returns))
    
    # For each IS-optimal strategy, check if OOS is below median
    n_combos = len(is_returns)
    pbo_count = 0
    
    for i in range(n_combos):
        if is_ranks[i] < n_combos // 4:  # Top 25% IS performers
            if oos_ranks[i] >= n_combos // 2:  # Below median OOS
                pbo_count += 1
    
    top_is = max(1, n_combos // 4)
    pbo = pbo_count / top_is
    
    # Compute Sharpe from OOS accuracy
    avg_oos_acc = np.mean(oos_returns)
    std_oos_acc = np.std(oos_returns)
    
    # Convert accuracy to approximate Sharpe
    # If accuracy = 50%, Sharpe ≈ 0; if accuracy = 55%, Sharpe ≈ 0.5
    sharpe_approx = (avg_oos_acc - 0.5) * 10  # Rough conversion
    
    return {
        'pbo': pbo,
        'sharpe': sharpe_approx,
        'is_mean': np.mean(is_returns),
        'oos_mean': np.mean(oos_returns),
        'rank_corr': spearmanr(is_ranks, oos_ranks)[0],
        'n_combos': n_combos,
        'status': 'success'
    }


def main():
    print("="*70)
    print("CPCV + PBO ANALYSIS: WEIGHTED vs UNWEIGHTED")
    print("="*70)
    print("\nComparing overfitting risk with AFML sample weighting")
    print("Target: PBO < 0.5 (reduced overfitting)")
    print("="*70)
    
    symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA']
    results = []
    
    for symbol in symbols:
        try:
            df = pd.read_csv(
                f'/Users/humbertolobo/Desktop/bolt.new-main/NUBLE-CLI/data/train/{symbol}.csv',
                index_col=0, parse_dates=True
            )
            
            if len(df) < 600:
                print(f"\n{symbol}: Insufficient data")
                continue
            
            print(f"\n{symbol}: {len(df)} days")
            
            # Weighted analysis
            result_w = cpcv_pbo_analysis(df, use_weights=True, n_splits=10)
            print(f"  Weighted:   PBO={result_w['pbo']:.1%}, OOS Acc={result_w['oos_mean']:.1%}")
            
            # Unweighted analysis
            result_uw = cpcv_pbo_analysis(df, use_weights=False, n_splits=10)
            print(f"  Unweighted: PBO={result_uw['pbo']:.1%}, OOS Acc={result_uw['oos_mean']:.1%}")
            
            # Improvement
            pbo_diff = result_uw['pbo'] - result_w['pbo']
            status = "✓" if pbo_diff > 0 else "✗"
            print(f"  PBO Improvement: {pbo_diff:+.1%} {status}")
            
            results.append({
                'symbol': symbol,
                'pbo_weighted': result_w['pbo'],
                'pbo_unweighted': result_uw['pbo'],
                'pbo_improvement': pbo_diff
            })
            
        except Exception as e:
            print(f"\n{symbol}: Error - {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if results:
        avg_pbo_w = np.mean([r['pbo_weighted'] for r in results])
        avg_pbo_uw = np.mean([r['pbo_unweighted'] for r in results])
        
        print(f"\nWeighted Average PBO:   {avg_pbo_w:.1%}")
        print(f"Unweighted Average PBO: {avg_pbo_uw:.1%}")
        print(f"PBO Reduction:          {avg_pbo_uw - avg_pbo_w:+.1%}")
        
        if avg_pbo_w < avg_pbo_uw:
            print("\n✓ Sample weighting REDUCED overfitting risk")
        else:
            print("\n✗ Sample weighting did not reduce overfitting")
        
        if avg_pbo_w < 0.5:
            print(f"\n✓ PBO < 50% - Ready for OOS testing!")
        else:
            print(f"\n✗ PBO still > 50% - Need more improvements")


if __name__ == "__main__":
    main()
