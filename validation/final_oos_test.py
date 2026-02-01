"""
============================================================
FINAL OUT-OF-SAMPLE TEST (Priority 4)
============================================================

This is the FINAL validation step. We test on data that has 
NEVER been touched during development (2023-2026).

Configuration (from robust_ensemble_test.py results):
- AAPL: max_depth=3, Sharpe +0.92, PBO 50.0%
- MSFT: max_depth=2, Sharpe +0.71, PBO 40.0%  
- NVDA: max_depth=4, Sharpe +0.92, PBO 35.0%

This simulates real production deployment.
============================================================
"""

import sys
sys.path.insert(0, '/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI')

print("Loading imports...")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

print("Imports successful")

# ============================================================
# CONFIGURATION - LOCKED FROM VALIDATION
# ============================================================

OOS_CONFIG = {
    'AAPL': {'max_depth': 3},
    'MSFT': {'max_depth': 2},
    'NVDA': {'max_depth': 4},
}

TRAIN_START = '2015-01-01'
TRAIN_END = '2022-12-31'
TEST_START = '2023-01-01'
TEST_END = '2026-01-30'

TRANSACTION_COST = 0.001  # 0.1% round-trip

# ============================================================
# FEATURE ENGINEERING (EXACT SAME AS VALIDATION)
# ============================================================

def compute_minimal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    6 minimal features with economic rationale.
    MUST be identical to validation.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    features = pd.DataFrame(index=df.index)
    
    # 1. Momentum (20-day) - trend following
    features['momentum_20'] = close.pct_change(20)
    
    # 2. Volatility (20-day) - risk measure
    features['volatility_20'] = close.pct_change().rolling(20).std()
    
    # 3. RSI-like (normalized) - mean reversion
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    features['rsi_norm'] = (gain / (gain + loss + 1e-10)) - 0.5
    
    # 4. Volume trend - institutional activity
    features['volume_trend'] = (
        volume.rolling(5).mean() / volume.rolling(20).mean() - 1
    )
    
    # 5. Price vs MA - trend position
    features['price_vs_ma'] = close / close.rolling(50).mean() - 1
    
    # 6. Range compression - volatility regime
    atr = (high - low).rolling(20).mean()
    features['range_compression'] = atr / close
    
    return features.dropna()


def create_ensemble_labels(df: pd.DataFrame, horizon: int = 10) -> tuple:
    """
    Ensemble labels from 5 triple barrier configurations.
    MUST be identical to validation.
    """
    close = df['close']
    
    # 5 triple barrier configurations
    configs = [
        {'pt': 0.02, 'sl': 0.02},
        {'pt': 0.025, 'sl': 0.015},
        {'pt': 0.03, 'sl': 0.02},
        {'pt': 0.015, 'sl': 0.025},
        {'pt': 0.02, 'sl': 0.03},
    ]
    
    all_labels = []
    
    for config in configs:
        labels = []
        for i in range(len(close) - horizon):
            entry = close.iloc[i]
            future = close.iloc[i+1:i+1+horizon]
            
            pt_price = entry * (1 + config['pt'])
            sl_price = entry * (1 - config['sl'])
            
            label = 0  # Neutral
            for price in future:
                if price >= pt_price:
                    label = 1  # Bullish
                    break
                elif price <= sl_price:
                    label = -1  # Bearish
                    break
            
            labels.append(label)
        
        all_labels.append(labels)
    
    # Average labels
    all_labels = np.array(all_labels)
    avg_labels = all_labels.mean(axis=0)
    
    # Convert to binary with confidence
    final_labels = (avg_labels > 0).astype(int)
    confidence = np.abs(avg_labels)
    
    return final_labels, confidence


# ============================================================
# MAIN TEST FUNCTION
# ============================================================

def run_oos_test(symbol: str, config: dict) -> dict:
    """
    Run final OOS test for a symbol.
    """
    max_depth = config['max_depth']
    
    # Load training data
    train_path = Path(f'/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI/data/train/{symbol}.csv')
    if not train_path.exists():
        print(f"  Training data not found: {train_path}")
        return None
    
    train_df = pd.read_csv(train_path, parse_dates=['date'])
    train_df = train_df.set_index('date').sort_index()
    train_df = train_df[(train_df.index >= TRAIN_START) & (train_df.index <= TRAIN_END)]
    
    # Load test data
    test_path = Path(f'/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI/data/test/{symbol}.csv')
    if not test_path.exists():
        print(f"  Test data not found: {test_path}")
        return None
    
    test_df = pd.read_csv(test_path, parse_dates=['date'])
    test_df = test_df.set_index('date').sort_index()
    test_df = test_df[(test_df.index >= TEST_START) & (test_df.index <= TEST_END)]
    
    print(f"  Training: {train_df.index[0].date()} to {train_df.index[-1].date()} ({len(train_df)} days)")
    print(f"  Testing:  {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} days)")
    
    # Compute features
    train_features = compute_minimal_features(train_df)
    test_features = compute_minimal_features(test_df)
    
    # Create labels for training
    train_labels, train_confidence = create_ensemble_labels(
        train_df.loc[train_features.index]
    )
    
    # Align training data
    min_len = min(len(train_features), len(train_labels))
    train_features = train_features.iloc[:min_len]
    train_labels = train_labels[:min_len]
    train_confidence = train_confidence[:min_len]
    
    # Sample weights (time decay + confidence)
    n_train = len(train_features)
    time_weights = np.linspace(0.5, 1.0, n_train)
    sample_weights = time_weights * (0.5 + 0.5 * train_confidence)
    sample_weights = sample_weights / sample_weights.mean()
    
    print(f"  Features: {train_features.shape[1]}")
    print(f"  Training samples: {n_train}")
    print(f"  Label distribution: {train_labels.mean():.1%} bullish")
    
    # Build model (exact same as validation)
    base_estimator = RandomForestClassifier(
        n_estimators=10,
        max_depth=max_depth,
        min_samples_leaf=20,
        max_features=0.5,
        random_state=42,
        n_jobs=-1
    )
    
    model = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=100,
        max_samples=0.8,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # Train on FULL training set
    print(f"  Training model (max_depth={max_depth})...")
    model.fit(train_features.values, train_labels, sample_weight=sample_weights)
    
    # Generate OOS predictions
    print(f"  Generating OOS predictions...")
    oos_predictions = model.predict(test_features.values)
    oos_proba = model.predict_proba(test_features.values)[:, 1]
    
    # Calculate returns
    test_close = test_df.loc[test_features.index, 'close']
    daily_returns = test_close.pct_change().fillna(0)
    
    # Strategy returns (long when bullish, flat otherwise)
    positions = pd.Series(oos_predictions, index=test_features.index)
    
    # Calculate position changes for transaction costs
    position_changes = positions.diff().abs().fillna(0)
    transaction_costs = position_changes * TRANSACTION_COST
    
    # Strategy returns
    strategy_returns = positions.shift(1).fillna(0) * daily_returns - transaction_costs
    strategy_returns = strategy_returns.iloc[1:]  # Remove first NaN
    
    # Metrics
    n_trades = int(position_changes.sum() / 2)  # Round trips
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-10) * np.sqrt(252)
    total_return = (1 + strategy_returns).prod() - 1
    
    # Drawdown
    cumulative = (1 + strategy_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Win rate
    winning_days = (strategy_returns > 0).sum()
    trading_days = (positions.shift(1).fillna(0) != 0).sum()
    win_rate = winning_days / max(trading_days, 1)
    
    # Buy & hold comparison
    bh_returns = daily_returns.iloc[1:]
    bh_sharpe = bh_returns.mean() / (bh_returns.std() + 1e-10) * np.sqrt(252)
    bh_total = (1 + bh_returns).prod() - 1
    
    return {
        'symbol': symbol,
        'max_depth': max_depth,
        'sharpe': sharpe,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_trades': n_trades,
        'strategy_returns': strategy_returns,
        'bh_sharpe': bh_sharpe,
        'bh_return': bh_total,
        'train_samples': n_train,
        'test_samples': len(test_features),
    }


def main():
    print("=" * 60)
    print("FINAL OUT-OF-SAMPLE TEST (2023-2026)")
    print("=" * 60)
    print()
    print("This data has NEVER been touched during development.")
    print("Testing on completely unseen market conditions.")
    print()
    print(f"Train Period: {TRAIN_START} to {TRAIN_END}")
    print(f"Test Period:  {TEST_START} to {TEST_END}")
    print("=" * 60)
    print()
    
    results = []
    all_returns = []
    
    for symbol, config in OOS_CONFIG.items():
        print(f"{'=' * 60}")
        print(f"{symbol}")
        print(f"{'=' * 60}")
        
        result = run_oos_test(symbol, config)
        
        if result is None:
            print(f"  SKIPPED - Data not available")
            continue
        
        results.append(result)
        all_returns.append(result['strategy_returns'])
        
        # Determine status
        if result['sharpe'] >= 0.5:
            status = "✅ EXCELLENT - Production Ready"
        elif result['sharpe'] >= 0.3:
            status = "✅ GOOD - Tradeable"
        elif result['sharpe'] >= 0.1:
            status = "⚠️ MARGINAL - Needs improvement"
        else:
            status = "❌ FAILED - Don't deploy"
        
        print()
        print(f"  OOS RESULTS:")
        print(f"    Sharpe Ratio:     {result['sharpe']:+.2f}")
        print(f"    Total Return:     {result['total_return']:+.1%}")
        print(f"    Max Drawdown:     {result['max_drawdown']:.1%}")
        print(f"    Win Rate:         {result['win_rate']:.1%}")
        print(f"    # Trades:         {result['n_trades']}")
        print()
        print(f"  BUY & HOLD COMPARISON:")
        print(f"    B&H Sharpe:       {result['bh_sharpe']:+.2f}")
        print(f"    B&H Return:       {result['bh_return']:+.1%}")
        print()
        print(f"  STATUS: {status}")
        print()
    
    # Portfolio summary
    if len(results) >= 2:
        print("=" * 60)
        print("PORTFOLIO SUMMARY (Equal Weight)")
        print("=" * 60)
        
        # Combine returns (equal weight)
        combined = pd.concat(all_returns, axis=1).mean(axis=1)
        
        portfolio_sharpe = combined.mean() / (combined.std() + 1e-10) * np.sqrt(252)
        portfolio_return = (1 + combined).prod() - 1
        
        # Portfolio drawdown
        cum = (1 + combined).cumprod()
        roll_max = cum.expanding().max()
        dd = (cum - roll_max) / roll_max
        portfolio_maxdd = dd.min()
        
        # Average metrics
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        
        print()
        print(f"  Combined Sharpe:    {portfolio_sharpe:+.2f}")
        print(f"  Combined Return:    {portfolio_return:+.1%}")
        print(f"  Combined MaxDD:     {portfolio_maxdd:.1%}")
        print()
        print(f"  Avg Individual Sharpe: {avg_sharpe:+.2f}")
        print()
        
        # Final verdict
        if portfolio_sharpe >= 0.5:
            verdict = "✅ PRODUCTION READY"
            recommendation = "System is validated for live trading."
        elif portfolio_sharpe >= 0.3:
            verdict = "✅ CAUTIOUSLY TRADEABLE"
            recommendation = "Consider smaller position sizes initially."
        elif portfolio_sharpe >= 0.1:
            verdict = "⚠️ NEEDS IMPROVEMENT"
            recommendation = "More research needed before deployment."
        else:
            verdict = "❌ DO NOT DEPLOY"
            recommendation = "Strategy did not survive OOS test."
        
        print(f"  FINAL VERDICT: {verdict}")
        print(f"  RECOMMENDATION: {recommendation}")
        print()
    
    print("=" * 60)
    print("INDIVIDUAL SYMBOL SUMMARY")
    print("=" * 60)
    print()
    print(f"{'Symbol':<8} | {'WF Sharpe':>10} | {'OOS Sharpe':>10} | {'Decay':>8} | Status")
    print("-" * 60)
    
    wf_sharpes = {'AAPL': 0.92, 'MSFT': 0.71, 'NVDA': 0.92}  # From validation
    
    for r in results:
        wf = wf_sharpes.get(r['symbol'], 0)
        decay = (wf - r['sharpe']) / max(wf, 0.01) * 100
        
        if r['sharpe'] >= 0.3:
            status = "✅ PASS"
        elif r['sharpe'] >= 0.1:
            status = "⚠️ MARGINAL"
        else:
            status = "❌ FAIL"
        
        print(f"{r['symbol']:<8} | {wf:>+10.2f} | {r['sharpe']:>+10.2f} | {decay:>7.0f}% | {status}")
    
    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
