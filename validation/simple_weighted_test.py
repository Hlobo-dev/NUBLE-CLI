"""
SIMPLIFIED SAMPLE-WEIGHTED VALIDATION
======================================

Tests the effect of AFML sample weighting on a simple ML model
without the complex Meta-Labeler integration.

This validates Priority 3: Sample Weighting independently.
"""

import sys
sys.path.insert(0, '/Users/humbertolobo/Desktop/bolt.new-main/NUBLE-CLI')

import os
os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable numba for stability

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import sample weights
from src.institutional.validation.sample_weights import AFMLSampleWeights

print("Imports successful")


def generate_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate simple features for ML model."""
    features = pd.DataFrame(index=df.index)
    close = df['close']
    
    # Momentum
    for period in [5, 10, 20, 60]:
        features[f'mom_{period}'] = close.pct_change(period)
    
    # Moving average distances
    for period in [10, 20, 50]:
        ma = close.rolling(period).mean()
        features[f'dist_ma{period}'] = (close - ma) / ma
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Volatility
    features['vol_20'] = close.pct_change().rolling(20).std() * np.sqrt(252)
    
    return features.fillna(0)


def generate_labels(close: pd.Series, threshold: float = 0.02, max_hold: int = 10) -> pd.Series:
    """Generate simple labels based on forward returns."""
    labels = pd.Series(index=close.index, dtype=float)
    
    for i in range(len(close) - max_hold):
        future_ret = close.iloc[i + max_hold] / close.iloc[i] - 1
        if future_ret > threshold:
            labels.iloc[i] = 1
        elif future_ret < -threshold:
            labels.iloc[i] = -1
        else:
            labels.iloc[i] = 0
    
    return labels


def compute_t1_times(df: pd.DataFrame, labels: pd.Series, max_hold: int = 10) -> pd.Series:
    """Compute label end times."""
    t1 = pd.Series(index=labels.index, dtype='datetime64[ns]')
    
    for idx in labels.dropna().index:
        pos = df.index.get_loc(idx)
        end_pos = min(pos + max_hold, len(df) - 1)
        t1[idx] = df.index[end_pos]
    
    return t1


def walk_forward_test(df: pd.DataFrame, use_weights: bool = True, verbose: bool = False):
    """
    Simple walk-forward test comparing weighted vs unweighted.
    """
    train_size = 504
    test_size = 63
    purge = 5
    
    # Initialize
    weighter = AFMLSampleWeights(decay_factor=0.95)
    results = {'returns': [], 'weights': []}
    
    start = 0
    n_windows = 0
    
    while start + train_size + purge + test_size <= len(df):
        # Split
        train_df = df.iloc[start:start+train_size]
        test_df = df.iloc[start+train_size+purge:start+train_size+purge+test_size]
        
        # Features
        train_features = generate_simple_features(train_df)
        test_features = generate_simple_features(test_df)
        
        # Labels - use looser threshold
        train_labels = generate_labels(train_df['close'], threshold=0.01, max_hold=10)
        train_labels = train_labels[train_labels.isin([-1, 1])].dropna()
        
        if len(train_labels) < 50:
            start += test_size
            continue
        
        # Align
        common_idx = train_features.index.intersection(train_labels.index)
        X_train = train_features.loc[common_idx]
        y_train = train_labels.loc[common_idx]
        
        # Compute weights
        if use_weights:
            try:
                t1 = compute_t1_times(train_df, train_labels)
                weights = weighter.get_sample_weights(
                    event_times=common_idx,
                    t1_times=t1.loc[common_idx],
                    close_idx=train_df.index
                )
                sample_weights = weights.reindex(common_idx).fillna(1.0).values
                results['weights'].append(sample_weights.mean())
            except Exception as e:
                if verbose:
                    print(f"  Warning: weights failed - {e}")
                sample_weights = None
        else:
            sample_weights = None
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(test_features)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=20,
            random_state=42
        )
        
        try:
            if sample_weights is not None and len(sample_weights) == len(y_train):
                model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train_scaled, y_train)
        except Exception as e:
            if verbose:
                print(f"  Warning: fit failed - {e}")
            start += test_size
            continue
        
        # Predict
        predictions = model.predict(X_test_scaled)
        positions = pd.Series(predictions, index=test_features.index)
        positions = positions.ffill().fillna(0)
        
        # Returns
        test_returns = test_df['close'].pct_change().fillna(0)
        aligned_returns = test_returns.reindex(positions.index).fillna(0)
        strategy_returns = positions.shift(1).fillna(0) * aligned_returns
        
        # Transaction costs
        costs = positions.diff().abs() * 0.001
        strategy_returns = strategy_returns - costs.fillna(0)
        
        results['returns'].extend(strategy_returns.dropna().values)
        n_windows += 1
        
        start += test_size
    
    # Compute metrics
    returns = np.array([r for r in results['returns'] if np.isfinite(r)])
    
    if len(returns) < 10:
        return {'sharpe': 0.0, 'total_return': 0.0, 'avg_weight': 1.0, 'n_windows': n_windows}
    
    ret_std = np.std(returns)
    if ret_std == 0 or not np.isfinite(ret_std):
        sharpe = 0.0
    else:
        sharpe = np.mean(returns) / ret_std * np.sqrt(252)
    
    total_ret = np.prod(1 + returns) - 1
    avg_weight = np.mean(results['weights']) if results['weights'] else 1.0
    
    return {
        'sharpe': float(sharpe) if np.isfinite(sharpe) else 0.0,
        'total_return': float(total_ret) if np.isfinite(total_ret) else 0.0,
        'avg_weight': avg_weight,
        'n_windows': n_windows
    }


def main():
    print("="*70)
    print("PRIORITY 3: SIMPLIFIED SAMPLE-WEIGHTED VALIDATION")
    print("="*70)
    print("\nTests effect of AFML sample weighting on simple ML model")
    print("="*70)
    
    symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'MSFT', 'NVDA']
    results_w = {}
    results_uw = {}
    
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
            
            # Weighted
            result_w = walk_forward_test(df, use_weights=True)
            results_w[symbol] = result_w
            print(f"  Weighted:   Sharpe={result_w['sharpe']:+.2f}, Return={result_w['total_return']*100:+.1f}%")
            
            # Unweighted
            result_uw = walk_forward_test(df, use_weights=False)
            results_uw[symbol] = result_uw
            print(f"  Unweighted: Sharpe={result_uw['sharpe']:+.2f}, Return={result_uw['total_return']*100:+.1f}%")
            
            # Improvement
            diff = result_w['sharpe'] - result_uw['sharpe']
            status = "✓" if diff > 0 else "✗"
            print(f"  Improvement: {diff:+.2f} {status}")
            
        except Exception as e:
            print(f"\n{symbol}: Error - {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if results_w and results_uw:
        avg_w = np.mean([r['sharpe'] for r in results_w.values()])
        avg_uw = np.mean([r['sharpe'] for r in results_uw.values()])
        
        print(f"\nWeighted Average Sharpe:   {avg_w:+.2f}")
        print(f"Unweighted Average Sharpe: {avg_uw:+.2f}")
        print(f"Improvement:               {avg_w - avg_uw:+.2f}")
        
        if avg_w > avg_uw:
            print("\n✓ Sample weighting IMPROVED performance")
        else:
            print("\n✗ Sample weighting did not improve this configuration")


if __name__ == "__main__":
    main()
