#!/usr/bin/env python3
"""
ROBUST ENSEMBLE TEST - Per AFML Chapters 3, 4, 7

Key Insight: High PBO is caused by NOISY LABELS, not model complexity.
The solution is NOT more regularization, but BETTER LABELING + ENSEMBLING.

Strategy:
1. Label Smoothing: Use probability targets instead of hard 0/1
2. Multiple Triple Barrier configs averaged together
3. Bagging of Walk-Forward paths (AFML Chapter 7)
4. Optimal depth search per symbol

Goal: Sharpe > 0.4, PBO < 50%
"""

import sys
sys.path.insert(0, '/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI')

print("Imports loading...")
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from itertools import combinations

print("Imports successful")


def load_data(symbol: str) -> pd.DataFrame:
    """Load CSV data from train folder"""
    path = Path(f'/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI/data/train/{symbol}.csv')
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Handle column names
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df.columns = df.columns.str.lower()
    df = df.sort_index()
    return df


def triple_barrier_labels(close: pd.Series, 
                          tp_mult: float = 1.5,
                          sl_mult: float = 1.0,
                          horizon: int = 10) -> pd.Series:
    """Generate triple barrier labels"""
    returns = close.pct_change()
    vol = returns.rolling(20).std()
    
    labels = []
    for i in range(len(close) - horizon):
        current = close.iloc[i]
        sigma = vol.iloc[i]
        
        if pd.isna(sigma) or sigma == 0:
            labels.append(0)
            continue
        
        upper = current * (1 + tp_mult * sigma)
        lower = current * (1 - sl_mult * sigma)
        
        # Check horizon
        future = close.iloc[i+1:i+1+horizon]
        hit_upper = (future >= upper).any()
        hit_lower = (future <= lower).any()
        
        if hit_upper and not hit_lower:
            labels.append(1)
        elif hit_lower and not hit_upper:
            labels.append(0)
        elif hit_upper and hit_lower:
            # First to hit
            upper_idx = (future >= upper).argmax()
            lower_idx = (future <= lower).argmax()
            labels.append(1 if upper_idx < lower_idx else 0)
        else:
            # Timeout - use return sign
            final_ret = (future.iloc[-1] / current) - 1 if len(future) > 0 else 0
            labels.append(1 if final_ret > 0 else 0)
    
    # Pad end
    labels.extend([0] * horizon)
    return pd.Series(labels, index=close.index)


def ensemble_labels(close: pd.Series) -> pd.Series:
    """
    AFML Chapter 3: Create robust labels by averaging multiple configs.
    This reduces noise in the labels themselves.
    """
    configs = [
        {'tp_mult': 1.0, 'sl_mult': 1.0, 'horizon': 5},
        {'tp_mult': 1.5, 'sl_mult': 1.0, 'horizon': 10},
        {'tp_mult': 2.0, 'sl_mult': 1.0, 'horizon': 10},
        {'tp_mult': 1.5, 'sl_mult': 1.5, 'horizon': 15},
        {'tp_mult': 2.0, 'sl_mult': 1.5, 'horizon': 20},
    ]
    
    label_sets = []
    for cfg in configs:
        labels = triple_barrier_labels(close, **cfg)
        label_sets.append(labels)
    
    # Average - this creates "soft" labels between 0 and 1
    ensemble = pd.concat(label_sets, axis=1).mean(axis=1)
    return ensemble


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create minimal, robust features - fewer features = less overfitting"""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    features = pd.DataFrame(index=df.index)
    
    # Only the most robust, theory-driven features:
    
    # 1. Momentum (proven in literature)
    features['mom_5'] = close.pct_change(5)
    features['mom_20'] = close.pct_change(20)
    
    # 2. Volatility regime
    ret = close.pct_change()
    features['vol_20'] = ret.rolling(20).std()
    features['vol_ratio'] = ret.rolling(5).std() / ret.rolling(20).std()
    
    # 3. Mean reversion indicator
    sma20 = close.rolling(20).mean()
    features['dist_sma'] = (close - sma20) / sma20
    
    # 4. Volume confirmation
    vol_sma = volume.rolling(20).mean()
    features['vol_surge'] = volume / vol_sma
    
    # That's it - only 6 features, all with economic rationale
    
    return features.dropna()


def walk_forward_sharpe(features: pd.DataFrame,
                        labels: pd.Series,
                        close: pd.Series,
                        model,
                        train_size: int = 504,
                        test_size: int = 63,
                        purge: int = 10,
                        embargo: int = 10,
                        costs: float = 0.001) -> Tuple[float, List]:
    """Walk-forward with proper purge/embargo"""
    
    # Align data
    common_idx = features.index.intersection(labels.index).intersection(close.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]
    close = close.loc[common_idx]
    
    all_returns = []
    
    start = 0
    while start + train_size + purge + embargo + test_size <= len(features):
        # Indices
        train_end = start + train_size
        test_start = train_end + purge + embargo
        test_end = test_start + test_size
        
        # Data splits
        X_train = features.iloc[start:train_end]
        
        # Convert soft labels to hard for training, but keep weights
        soft_labels = labels.iloc[start:train_end]
        y_train = (soft_labels > 0.5).astype(int)
        
        # Weight by label confidence (closer to 0.5 = less confident)
        label_confidence = np.abs(soft_labels - 0.5) * 2  # 0 to 1
        
        # Simple time-decay weights (more recent = higher weight) 
        # This is a simplified version of AFML Ch 4 approach
        n = len(X_train)
        time_decay = np.linspace(0.5, 1.0, n)  # Half weight at start, full at end
        
        # Combine time decay with label confidence
        combined_weights = time_decay * (0.5 + label_confidence.values)
        combined_weights = combined_weights / combined_weights.mean()
        
        X_test = features.iloc[test_start:test_end]
        test_close = close.iloc[test_start:test_end]
        
        if len(X_train) < 100 or len(X_test) < 20:
            start += test_size
            continue
        
        # Scale
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        # Fit
        try:
            model.fit(X_train_sc, y_train, sample_weight=combined_weights)
        except:
            model.fit(X_train_sc, y_train)
        
        # Predict
        probs = model.predict_proba(X_test_sc)[:, 1]
        signals = (probs > 0.5).astype(int)
        
        # Strategy returns
        daily_ret = test_close.pct_change().fillna(0)
        strat_ret = signals[:-1] * daily_ret.values[1:]
        
        # Costs on trades
        trades = np.diff(np.concatenate([[0], signals]))
        trade_costs = np.abs(trades[:-1]) * costs
        net_ret = strat_ret - trade_costs
        
        all_returns.extend(net_ret)
        start += test_size
    
    if len(all_returns) < 50:
        return np.nan, []
    
    returns = np.array(all_returns)
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    return sharpe, returns


def cpcv_pbo(features: pd.DataFrame,
             labels: pd.Series,
             close: pd.Series,
             model_params: dict,
             n_splits: int = 10,
             n_test: int = 2) -> Tuple[float, float, float]:
    """CPCV + PBO with proper implementation"""
    
    # Align data
    common_idx = features.index.intersection(labels.index).intersection(close.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]
    close = close.loc[common_idx]
    
    # Create splits
    split_size = len(features) // n_splits
    splits = []
    for i in range(n_splits):
        start = i * split_size
        end = start + split_size if i < n_splits - 1 else len(features)
        splits.append((start, end))
    
    # All combinations
    combos = list(combinations(range(n_splits), n_test))
    
    is_sharpes = []
    oos_sharpes = []
    
    for combo in combos[:20]:  # Limit for speed
        test_indices = []
        for split_idx in combo:
            start, end = splits[split_idx]
            test_indices.extend(range(start, end))
        
        train_indices = [i for i in range(len(features)) if i not in test_indices]
        
        if len(train_indices) < 200 or len(test_indices) < 50:
            continue
        
        X_train = features.iloc[train_indices]
        soft_labels_train = labels.iloc[train_indices]
        y_train = (soft_labels_train > 0.5).astype(int)
        
        X_test = features.iloc[test_indices]
        soft_labels_test = labels.iloc[test_indices]
        y_test = (soft_labels_test > 0.5).astype(int)
        
        # Simple time decay weights
        n = len(X_train)
        weights = np.linspace(0.5, 1.0, n)
        weights = weights / weights.mean()
        
        # Scale
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        # Model
        model = RandomForestClassifier(**model_params, random_state=42, n_jobs=-1)
        model.fit(X_train_sc, y_train, sample_weight=weights)
        
        # IS performance
        is_probs = model.predict_proba(X_train_sc)[:, 1]
        is_preds = (is_probs > 0.5).astype(int)
        is_acc = (is_preds == y_train).mean()
        is_sharpes.append(is_acc)
        
        # OOS performance
        oos_probs = model.predict_proba(X_test_sc)[:, 1]
        oos_preds = (oos_probs > 0.5).astype(int)
        oos_acc = (oos_preds == y_test).mean()
        oos_sharpes.append(oos_acc)
    
    if len(is_sharpes) < 5:
        return np.nan, np.nan, np.nan
    
    # PBO = probability that OOS rank < IS rank
    is_ranks = np.argsort(np.argsort(is_sharpes))
    oos_ranks = np.argsort(np.argsort(oos_sharpes))
    pbo = (oos_ranks < is_ranks).mean()
    
    return pbo, np.mean(is_sharpes), np.mean(oos_sharpes)


def find_optimal_depth(features: pd.DataFrame,
                       labels: pd.Series,
                       close: pd.Series,
                       depths: List[int] = [2, 3, 4, 5]) -> Tuple[int, Dict]:
    """Find optimal depth per symbol via cross-validation"""
    
    results = {}
    
    for depth in depths:
        params = {
            'max_depth': depth,
            'n_estimators': 100,  # Back to 100 for stability
            'min_samples_leaf': 30,  # Moderate
            'max_features': 'sqrt',
            'class_weight': 'balanced'
        }
        
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        
        sharpe, returns = walk_forward_sharpe(
            features, labels, close, model,
            train_size=504, test_size=63, purge=10, embargo=10
        )
        
        # Only compute PBO for promising configs
        if sharpe > 0.3:
            pbo, is_acc, oos_acc = cpcv_pbo(features, labels, close, params)
        else:
            pbo = np.nan
            is_acc = np.nan
            oos_acc = np.nan
        
        results[depth] = {
            'sharpe': sharpe,
            'pbo': pbo,
            'is_acc': is_acc,
            'oos_acc': oos_acc,
            'returns': returns
        }
        
        print(f"    depth={depth}: Sharpe={sharpe:+.2f}, PBO={pbo*100:.1f}%" if not np.isnan(pbo) 
              else f"    depth={depth}: Sharpe={sharpe:+.2f}")
    
    # Find best: maximize Sharpe where PBO < 0.6
    valid = {k: v for k, v in results.items() 
             if not np.isnan(v['sharpe']) and v['sharpe'] > 0}
    
    if not valid:
        # Just take highest Sharpe
        best_depth = max(results.keys(), key=lambda k: results[k]['sharpe'] if not np.isnan(results[k]['sharpe']) else -999)
    else:
        # Prefer lower PBO, then higher Sharpe
        best_depth = min(valid.keys(), 
                        key=lambda k: (valid[k]['pbo'] if not np.isnan(valid[k]['pbo']) else 0.99, 
                                      -valid[k]['sharpe']))
    
    return best_depth, results[best_depth]


def main():
    print("="*70)
    print("ROBUST ENSEMBLE TEST - AFML Best Practices")
    print("="*70)
    print()
    print("Key Changes from Previous Attempt:")
    print("  1. Ensemble labels (5 triple barrier configs averaged)")
    print("  2. Label confidence weighting")
    print("  3. Minimal features (6) with economic rationale")
    print("  4. Per-symbol optimal depth search")
    print("  5. Back to n_estimators=100 for stability")
    print("="*70)
    
    symbols = ['AAPL', 'MSFT', 'SPY', 'QQQ', 'NVDA']
    
    results = {}
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"{symbol}")
        print(f"{'='*50}")
        
        # Load data
        df = load_data(symbol)
        if df is None:
            print(f"  Data not found")
            continue
        
        # Train only (2015-2022)
        train_df = df[df.index < '2023-01-01']
        print(f"  Training period: {train_df.index[0].date()} to {train_df.index[-1].date()}")
        print(f"  {len(train_df)} days")
        
        # Features
        features = create_features(train_df)
        print(f"  Features: {features.shape[1]}")
        
        # Ensemble labels
        print("\n  [1] Creating ensemble labels (5 configs)...")
        labels = ensemble_labels(train_df['close'])
        
        # Align
        common_idx = features.index.intersection(labels.index)
        features = features.loc[common_idx]
        labels = labels.loc[common_idx]
        close = train_df['close'].loc[common_idx]
        
        print(f"      Label distribution: {(labels > 0.5).mean():.1%} bullish")
        print(f"      Label avg confidence: {(np.abs(labels - 0.5) * 2).mean():.2f}")
        
        # Find optimal depth
        print("\n  [2] Finding optimal depth...")
        best_depth, best_result = find_optimal_depth(features, labels, close)
        
        print(f"\n  Best config: depth={best_depth}")
        print(f"    Sharpe: {best_result['sharpe']:+.2f}")
        if not np.isnan(best_result['pbo']):
            print(f"    PBO: {best_result['pbo']*100:.1f}%")
            print(f"    IS Accuracy: {best_result['is_acc']:.1%}")
            print(f"    OOS Accuracy: {best_result['oos_acc']:.1%}")
        
        # Assess
        sharpe = best_result['sharpe']
        pbo = best_result['pbo']
        
        if np.isnan(sharpe) or sharpe < 0.4:
            status = "❌ FAIL (Sharpe < 0.4)"
        elif np.isnan(pbo):
            status = "⚠️ NEEDS PBO CHECK"
        elif pbo < 0.5:
            status = "✅ PASS"
        elif pbo < 0.6:
            status = "⚠️ MARGINAL (PBO 50-60%)"
        else:
            status = "❌ FAIL (PBO > 60%)"
        
        print(f"\n  Status: {status}")
        
        results[symbol] = {
            'sharpe': sharpe,
            'pbo': pbo,
            'best_depth': best_depth,
            'status': status
        }
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print()
    print(f"{'SYMBOL':<8} | {'Sharpe':>10} | {'PBO':>8} | {'Depth':>5} | Status")
    print("-"*70)
    
    passes = 0
    for symbol, r in results.items():
        sharpe_str = f"{r['sharpe']:+.2f}" if not np.isnan(r['sharpe']) else "N/A"
        pbo_str = f"{r['pbo']*100:.1f}%" if not np.isnan(r['pbo']) else "N/A"
        print(f"{symbol:<8} | {sharpe_str:>10} | {pbo_str:>8} | {r['best_depth']:>5} | {r['status']}")
        
        if 'PASS' in r['status']:
            passes += 1
    
    print()
    print("="*70)
    print("ASSESSMENT")
    print("="*70)
    
    if passes >= 3:
        print(f"\n✅ SUCCESS: {passes}/5 symbols pass!")
        print("   Ready for OOS testing (2023-2026)")
    else:
        print(f"\n⚠️ Need more work: Only {passes}/5 pass")
        print("   Recommendations:")
        print("   1. Try different feature sets")
        print("   2. Consider regime-conditional models")
        print("   3. May need longer training data")


if __name__ == '__main__':
    main()
