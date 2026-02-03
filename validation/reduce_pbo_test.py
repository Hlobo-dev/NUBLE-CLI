#!/usr/bin/env python3
"""
REDUCE PBO BELOW 50% TEST
=========================
Implements aggressive regularization and feature selection
to reduce overfitting while maintaining Sharpe > 0.4

Target: PBO < 50%, Sharpe > 0.4
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

print("Imports successful")

# =============================================================================
# STEP 1: AGGRESSIVE REGULARIZATION PARAMETERS
# =============================================================================

# RESTRICTIVE model parameters - key to reducing PBO
MODEL_PARAMS = {
    'n_estimators': 50,        # Reduced from 100
    'max_depth': 3,            # CRITICAL: Reduced from 5
    'min_samples_leaf': 50,    # Increased from default 20
    'min_samples_split': 100,  # Increased from default
    'max_features': 0.3,       # Only 30% of features per tree
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

# INCREASED purge/embargo for triple barrier overlap
WALK_FORWARD_CONFIG = {
    'train_size': 504,         # 2 years
    'test_size': 63,           # 3 months
    'purge_size': 10,          # INCREASED from 5
    'embargo_size': 10,        # INCREASED from 5
}

# CPCV config with larger gaps
CPCV_CONFIG = {
    'n_splits': 6,
    'purge_pct': 0.02,         # INCREASED from 0.01
    'embargo_pct': 0.02,       # INCREASED from 0.01
}

# Feature importance threshold for selection
FEATURE_IMPORTANCE_THRESHOLD = 0.05


# =============================================================================
# FEATURE GENERATION (REDUCED SET)
# =============================================================================

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate REDUCED feature set - only proven predictors."""
    features = pd.DataFrame(index=df.index)
    
    close = df['close']
    
    # Core momentum features (5 features)
    features['ret_5'] = close.pct_change(5)
    features['ret_10'] = close.pct_change(10)
    features['ret_20'] = close.pct_change(20)
    
    # SMA ratios (3 features)
    features['sma_ratio_10'] = close / close.rolling(10).mean()
    features['sma_ratio_20'] = close / close.rolling(20).mean()
    features['sma_ratio_50'] = close / close.rolling(50).mean()
    
    # Volatility (3 features)
    features['vol_10'] = close.pct_change().rolling(10).std()
    features['vol_20'] = close.pct_change().rolling(20).std()
    features['vol_ratio'] = features['vol_10'] / (features['vol_20'] + 1e-8)
    
    # RSI-like (2 features)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    features['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    features['rsi_zscore'] = (features['rsi_14'] - 50) / 20
    
    # Volume if available (2 features)
    if 'volume' in df.columns:
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
    
    # Total: 15 features max
    return features.dropna()


def select_top_features(model, X: pd.DataFrame, threshold: float = 0.05) -> list:
    """Keep only features with importance > threshold."""
    importances = model.feature_importances_
    feature_names = X.columns.tolist()
    
    # Keep features above threshold
    selected = []
    for name, imp in zip(feature_names, importances):
        if imp > threshold:
            selected.append(name)
    
    return selected


# =============================================================================
# LABEL GENERATION
# =============================================================================

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
    
    return labels


def compute_t1_times(df: pd.DataFrame, labels: pd.Series, max_hold: int = 10) -> pd.Series:
    """Compute t1 times for sample weighting."""
    t1 = pd.Series(index=labels.index, dtype='datetime64[ns]')
    
    for idx in labels.index:
        loc = df.index.get_loc(idx)
        end_loc = min(loc + max_hold, len(df) - 1)
        t1.loc[idx] = df.index[end_loc]
    
    return t1


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def run_walk_forward(df: pd.DataFrame, use_feature_selection: bool = True, verbose: bool = False):
    """
    Walk-forward validation with aggressive regularization.
    """
    config = WALK_FORWARD_CONFIG
    train_size = config['train_size']
    test_size = config['test_size']
    purge = config['purge_size']
    embargo = config['embargo_size']
    
    # Initialize
    weighter = AFMLSampleWeights(decay_factor=0.95)
    results = {'returns': [], 'n_trades': 0}
    selected_features = None
    
    start = 0
    n_windows = 0
    
    while start + train_size + purge + embargo + test_size <= len(df):
        # Split with increased purge/embargo
        train_end = start + train_size
        test_start = train_end + purge + embargo
        test_end = test_start + test_size
        
        train_df = df.iloc[start:train_end]
        test_df = df.iloc[test_start:test_end]
        
        # Generate features
        train_features = generate_features(train_df)
        test_features = generate_features(test_df)
        
        # Generate labels with looser threshold
        train_labels = generate_labels(train_df['close'], threshold=0.01, max_hold=10)
        train_labels = train_labels[train_labels.isin([-1, 1])].dropna()
        
        if len(train_labels) < 100:
            start += test_size
            continue
        
        # Align
        common_idx = train_features.index.intersection(train_labels.index)
        X_train = train_features.loc[common_idx]
        y_train = train_labels.loc[common_idx]
        
        # Compute sample weights
        try:
            t1 = compute_t1_times(train_df, y_train)
            weights = weighter.get_sample_weights(
                event_times=common_idx,
                t1_times=t1.loc[common_idx],
                close_idx=train_df.index
            )
            sample_weights = weights.reindex(common_idx).fillna(1.0).values
        except:
            sample_weights = np.ones(len(y_train))
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train with AGGRESSIVE regularization
        model = RandomForestClassifier(**MODEL_PARAMS)
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
        # Feature selection on first window
        if use_feature_selection and selected_features is None and n_windows == 0:
            selected_features = select_top_features(model, X_train, FEATURE_IMPORTANCE_THRESHOLD)
            if verbose:
                print(f"  Selected {len(selected_features)}/{len(X_train.columns)} features")
        
        # Use selected features if available
        if selected_features and len(selected_features) >= 3:
            X_train_sel = X_train[selected_features]
            X_test_sel = test_features[selected_features] if all(f in test_features.columns for f in selected_features) else test_features
            
            # Retrain on selected features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            X_test_scaled = scaler.transform(X_test_sel)
        else:
            X_test_scaled = scaler.transform(test_features)
        
        # Predict
        predictions = model.predict(X_test_scaled)
        positions = pd.Series(predictions, index=test_features.index)
        
        # Count trades
        results['n_trades'] += (positions.diff().abs() > 0).sum()
        
        # Calculate returns
        test_returns = test_df['close'].pct_change().fillna(0)
        aligned_returns = test_returns.reindex(positions.index).fillna(0)
        strategy_returns = positions.shift(1).fillna(0) * aligned_returns
        
        # Transaction costs (0.1% round-trip)
        costs = positions.diff().abs() * 0.001
        strategy_returns = strategy_returns - costs.fillna(0)
        
        results['returns'].extend(strategy_returns.dropna().values)
        n_windows += 1
        
        start += test_size
    
    # Compute metrics
    returns = np.array([r for r in results['returns'] if np.isfinite(r)])
    
    if len(returns) < 20:
        return {'sharpe': 0.0, 'total_return': 0.0, 'n_windows': n_windows, 'n_trades': 0}
    
    ret_std = np.std(returns)
    if ret_std == 0 or not np.isfinite(ret_std):
        sharpe = 0.0
    else:
        sharpe = np.mean(returns) / ret_std * np.sqrt(252)
    
    total_ret = np.prod(1 + returns) - 1
    
    return {
        'sharpe': float(sharpe) if np.isfinite(sharpe) else 0.0,
        'total_return': float(total_ret) if np.isfinite(total_ret) else 0.0,
        'n_windows': n_windows,
        'n_trades': results['n_trades'],
        'selected_features': selected_features
    }


# =============================================================================
# CPCV + PBO ANALYSIS
# =============================================================================

def run_cpcv_pbo(df: pd.DataFrame, selected_features: list = None, verbose: bool = False):
    """
    Combinatorial Purged Cross-Validation with PBO.
    Uses aggressive regularization and increased purge.
    """
    config = CPCV_CONFIG
    n_splits = config['n_splits']
    purge_pct = config['purge_pct']
    embargo_pct = config['embargo_pct']
    
    # Prepare data
    features = generate_features(df)
    labels = generate_labels(df['close'], threshold=0.01, max_hold=10)
    labels = labels[labels.isin([-1, 1])].dropna()
    
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]
    
    # Use selected features if provided
    if selected_features and all(f in X.columns for f in selected_features):
        X = X[selected_features]
    
    if len(X) < 300:
        return {'pbo': 1.0, 'sharpe': 0.0, 'status': 'insufficient_data'}
    
    # Compute sample weights
    weighter = AFMLSampleWeights(decay_factor=0.95)
    try:
        t1 = compute_t1_times(df, y)
        weights = weighter.get_sample_weights(
            event_times=common_idx,
            t1_times=t1.loc[common_idx],
            close_idx=df.index
        )
        sample_weights = weights.reindex(common_idx).fillna(1.0).values
    except:
        sample_weights = np.ones(len(y))
    
    # Split data
    n_samples = len(X)
    purge_size = int(n_samples * purge_pct)
    embargo_size = int(n_samples * embargo_pct)
    
    indices = np.arange(n_samples)
    split_size = n_samples // n_splits
    groups = [indices[i*split_size:(i+1)*split_size] for i in range(n_splits)]
    
    # Generate train/test combinations
    n_train = n_splits // 2
    train_combos = list(combinations(range(n_splits), n_train))
    
    # Limit combinations for speed
    max_combos = 30
    if len(train_combos) > max_combos:
        np.random.seed(42)
        train_combos = [train_combos[i] for i in np.random.choice(len(train_combos), max_combos, replace=False)]
    
    is_returns = []  # In-sample accuracy
    oos_returns = []  # Out-of-sample accuracy
    
    scaler = StandardScaler()
    
    for combo in train_combos:
        # Get indices
        train_idx = np.concatenate([groups[i] for i in combo])
        test_groups = [i for i in range(n_splits) if i not in combo]
        test_idx = np.concatenate([groups[i] for i in test_groups])
        
        # Apply purge at boundaries
        train_mask = np.ones(len(train_idx), dtype=bool)
        for tg in test_groups:
            for ti in combo:
                if abs(ti - tg) == 1:
                    # Remove samples near boundary
                    boundary_start = groups[ti][-purge_size:] if ti < tg else groups[ti][:purge_size]
                    train_mask &= ~np.isin(train_idx, boundary_start)
        
        train_idx = train_idx[train_mask]
        
        # Apply embargo
        if embargo_size > 0:
            test_idx = test_idx[embargo_size:]
        
        if len(train_idx) < 50 or len(test_idx) < 20:
            continue
        
        # Prepare data
        X_train = X.iloc[train_idx].values
        y_train = y.iloc[train_idx].values
        X_test = X.iloc[test_idx].values
        y_test = y.iloc[test_idx].values
        w_train = sample_weights[train_idx]
        
        # Scale
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train with AGGRESSIVE regularization
        model = RandomForestClassifier(**MODEL_PARAMS)
        model.fit(X_train_scaled, y_train, sample_weight=w_train)
        
        # Get accuracies
        is_acc = np.mean(model.predict(X_train_scaled) == y_train)
        oos_acc = np.mean(model.predict(X_test_scaled) == y_test)
        
        is_returns.append(is_acc)
        oos_returns.append(oos_acc)
    
    if len(is_returns) < 5:
        return {'pbo': 1.0, 'sharpe': 0.0, 'status': 'insufficient_combos'}
    
    # Compute PBO
    is_returns = np.array(is_returns)
    oos_returns = np.array(oos_returns)
    
    # Rank strategies
    is_ranks = np.argsort(np.argsort(-is_returns))
    oos_ranks = np.argsort(np.argsort(-oos_returns))
    
    # PBO: How often does the best IS performer underperform OOS?
    n_combos = len(is_returns)
    n_top = max(1, n_combos // 4)
    
    pbo_count = 0
    for i in range(n_combos):
        if is_ranks[i] < n_top:  # Top 25% IS
            if oos_ranks[i] >= n_combos // 2:  # Below median OOS
                pbo_count += 1
    
    pbo = pbo_count / n_top
    
    # Approximate Sharpe from OOS accuracy
    avg_oos = np.mean(oos_returns)
    sharpe_approx = (avg_oos - 0.5) * 10
    
    return {
        'pbo': pbo,
        'sharpe': sharpe_approx,
        'is_mean': np.mean(is_returns),
        'oos_mean': avg_oos,
        'rank_corr': spearmanr(is_ranks, oos_ranks)[0],
        'n_combos': n_combos,
        'status': 'success'
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    print("="*70)
    print("REDUCE PBO TEST - AGGRESSIVE REGULARIZATION")
    print("="*70)
    print(f"\nModel Parameters:")
    print(f"  max_depth: {MODEL_PARAMS['max_depth']} (reduced from 5)")
    print(f"  n_estimators: {MODEL_PARAMS['n_estimators']} (reduced from 100)")
    print(f"  min_samples_leaf: {MODEL_PARAMS['min_samples_leaf']} (increased)")
    print(f"  max_features: {MODEL_PARAMS['max_features']} (restricted)")
    print(f"\nWalk-Forward Config:")
    print(f"  purge: {WALK_FORWARD_CONFIG['purge_size']} days (increased from 5)")
    print(f"  embargo: {WALK_FORWARD_CONFIG['embargo_size']} days (increased from 5)")
    print("="*70)
    
    symbols = ['AAPL', 'MSFT', 'SPY', 'QQQ', 'NVDA']
    results_table = []
    
    for symbol in symbols:
        try:
            # Load data
            df = pd.read_csv(
                f'/Users/humbertolobo/Desktop/bolt.new-main/NUBLE-CLI/data/train/{symbol}.csv',
                index_col=0, parse_dates=True
            )
            
            if len(df) < 700:
                print(f"\n{symbol}: Insufficient data ({len(df)} days)")
                continue
            
            print(f"\n{'='*50}")
            print(f"{symbol}: {len(df)} days")
            print('='*50)
            
            # STEP 1: Walk-Forward with feature selection
            print("\n[1] Walk-Forward Test...")
            wf_result = run_walk_forward(df, use_feature_selection=True, verbose=True)
            wf_sharpe = wf_result['sharpe']
            selected_features = wf_result.get('selected_features', None)
            
            print(f"    Sharpe: {wf_sharpe:+.2f}")
            print(f"    Return: {wf_result['total_return']*100:+.1f}%")
            print(f"    Windows: {wf_result['n_windows']}")
            if selected_features:
                print(f"    Features: {len(selected_features)} selected")
            
            # Skip if Sharpe too low
            if wf_sharpe < 0.3:
                print(f"\n    SKIP CPCV - Sharpe too low ({wf_sharpe:.2f} < 0.30)")
                results_table.append({
                    'symbol': symbol,
                    'wf_sharpe': wf_sharpe,
                    'pbo': None,
                    'status': 'SKIP (Low Sharpe)'
                })
                continue
            
            # STEP 2: CPCV + PBO
            print("\n[2] CPCV + PBO Analysis...")
            cpcv_result = run_cpcv_pbo(df, selected_features=selected_features, verbose=True)
            pbo = cpcv_result['pbo']
            
            print(f"    PBO: {pbo:.1%}")
            print(f"    IS Accuracy: {cpcv_result['is_mean']:.1%}")
            print(f"    OOS Accuracy: {cpcv_result['oos_mean']:.1%}")
            print(f"    Rank Correlation: {cpcv_result['rank_corr']:.2f}")
            
            # Determine status
            if wf_sharpe >= 0.4 and pbo < 0.5:
                status = "✅ PASS - Ready for OOS"
            elif wf_sharpe >= 0.4 and pbo >= 0.5:
                status = "⚠️ High PBO - More regularization needed"
            elif wf_sharpe < 0.4 and pbo < 0.5:
                status = "⚠️ Weak but not overfit"
            else:
                status = "❌ FAIL - Overfit and weak"
            
            print(f"\n    Status: {status}")
            
            results_table.append({
                'symbol': symbol,
                'wf_sharpe': wf_sharpe,
                'pbo': pbo,
                'oos_acc': cpcv_result['oos_mean'],
                'status': status
            })
            
        except Exception as e:
            print(f"\n{symbol}: Error - {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'SYMBOL':<8} | {'WF Sharpe':<10} | {'PBO':<8} | {'Status'}")
    print("-"*70)
    
    for r in results_table:
        pbo_str = f"{r['pbo']:.1%}" if r['pbo'] is not None else "N/A"
        print(f"{r['symbol']:<8} | {r['wf_sharpe']:+.2f}      | {pbo_str:<8} | {r['status']}")
    
    # Overall assessment
    print("\n" + "="*70)
    print("ASSESSMENT")
    print("="*70)
    
    passing = [r for r in results_table if r['pbo'] is not None and r['wf_sharpe'] >= 0.4 and r['pbo'] < 0.5]
    
    if len(passing) >= 3:
        print(f"\n✅ {len(passing)}/5 symbols PASS - Ready for OOS test!")
        print(f"   Passing symbols: {[r['symbol'] for r in passing]}")
    else:
        print(f"\n⚠️ Only {len(passing)}/5 symbols pass (need ≥3)")
        
        # Recommendations
        high_pbo = [r for r in results_table if r['pbo'] is not None and r['pbo'] >= 0.5]
        if high_pbo:
            avg_pbo = np.mean([r['pbo'] for r in high_pbo])
            print(f"\n   High PBO symbols: {[r['symbol'] for r in high_pbo]}")
            print(f"   Average PBO: {avg_pbo:.1%}")
            print("\n   Recommendations:")
            print("   1. Further reduce max_depth to 2")
            print("   2. Increase min_samples_leaf to 100")
            print("   3. Use only top 3-5 features")


if __name__ == "__main__":
    main()
