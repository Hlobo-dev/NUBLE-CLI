#!/usr/bin/env python3
"""
REAL Baseline Comparison Test

This tests the ACTUAL NUBLE ML strategy against buy-and-hold SPY.
Uses REAL market data, REAL signal generation, REAL validation.

NO MOCKS. NO SYNTHETIC DATA. NO FAKE STRATEGIES.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def load_real_data(symbol: str, period: str = 'test') -> pd.DataFrame:
    """Load REAL market data from data directory."""
    if period == 'test':
        path = PROJECT_ROOT / 'data' / 'test' / f'{symbol}.csv'
    else:
        path = PROJECT_ROOT / 'data' / 'train' / f'{symbol}.csv'
    
    if not path.exists():
        raise FileNotFoundError(f"Real data not found: {path}")
    
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    
    # Ensure we have required columns
    if 'close' not in df.columns:
        if 'Close' in df.columns:
            df = df.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 
                                    'Open': 'open', 'Volume': 'volume'})
    
    print(f"  Loaded {symbol}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


def run_real_ml_strategy(symbol: str, data: pd.DataFrame) -> pd.Series:
    """
    Run the ACTUAL NUBLE ML strategy on real data.
    
    This uses your REAL EnhancedSignalGenerator.
    """
    # Import directly from the signals module to avoid loading all dependencies
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "enhanced_signals",
        str(PROJECT_ROOT / "src" / "institutional" / "signals" / "enhanced_signals.py")
    )
    signals_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(signals_module)
    EnhancedSignalGenerator = signals_module.EnhancedSignalGenerator
    
    signal_gen = EnhancedSignalGenerator()
    
    # Calculate returns
    returns = data['close'].pct_change()
    
    # Improved regime detection based on recent returns
    # FIXED: Original thresholds were too conservative and detected VOLATILE too often
    def detect_regime(recent_returns, lookback_60=None):
        if len(recent_returns) < 20:
            return 'SIDEWAYS'
        
        mean_ret = recent_returns.mean()
        vol = recent_returns.std()
        
        # Annualized metrics for better thresholds
        ann_return = mean_ret * 252  # Annualized return
        ann_vol = vol * np.sqrt(252)  # Annualized volatility
        
        # Use longer lookback for trend detection if available
        if lookback_60 is not None and len(lookback_60) >= 40:
            trend_return = lookback_60.mean() * 252
        else:
            trend_return = ann_return
        
        # FIXED thresholds:
        # - Only VOLATILE if vol > 35% annualized (was ~50% with 0.02 daily)
        # - BULL if >5% annualized return (was ~25% with 0.001 daily)
        # - BEAR if <-5% annualized return
        if ann_vol > 0.35:  # Very high volatility (>35% annual)
            return 'VOLATILE'
        elif trend_return > 0.05:  # >5% annual return = BULL
            return 'BULL'
        elif trend_return < -0.05:  # <-5% annual return = BEAR
            return 'BEAR'
        else:
            return 'SIDEWAYS'
    
    # Generate signals using YOUR signal generator
    signals = []
    lookback = 70  # Need enough history for signal generation
    
    print(f"  Generating signals for {len(data) - lookback} days...")
    
    for i in range(lookback, len(data)):
        window = data.iloc[i-lookback:i+1].copy()
        recent_returns = returns.iloc[i-20:i]
        lookback_60 = returns.iloc[max(0, i-60):i]  # Longer lookback for trend
        
        try:
            regime = detect_regime(recent_returns, lookback_60)
            
            signal = signal_gen.generate_signal(
                symbol=symbol,
                prices=window,
                sentiment=0.0,
                cross_asset_momentum=0.0,
                regime=regime
            )
            signals.append(signal.direction)
        except Exception as e:
            signals.append(0)  # No position on error
    
    # Pad beginning with zeros
    signals = [0] * lookback + signals
    signals_series = pd.Series(signals, index=data.index)
    
    # Calculate strategy returns (signal from yesterday * return today)
    # This avoids lookahead bias
    strategy_returns = signals_series.shift(1) * returns
    strategy_returns = strategy_returns.dropna()
    
    return strategy_returns


def calculate_metrics(returns: pd.Series, name: str) -> dict:
    """Calculate performance metrics."""
    if len(returns) == 0:
        return {'name': name, 'error': 'No returns'}
    
    # Remove any infinities or NaNs
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(returns) == 0:
        return {'name': name, 'error': 'No valid returns'}
    
    total_return = (1 + returns).prod() - 1
    
    # Annualize
    years = len(returns) / 252
    if years > 0:
        annual_return = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1
    else:
        annual_return = total_return
    
    volatility = returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).mean()
    
    # Sortino (downside deviation)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = annual_return / downside_vol if downside_vol > 0 else 0
    
    return {
        'name': name,
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'num_days': len(returns)
    }


def print_metrics(metrics: dict):
    """Print metrics nicely."""
    if 'error' in metrics:
        print(f"\n  {metrics['name']}: ERROR - {metrics['error']}")
        return
        
    print(f"\n  {metrics['name']}:")
    print(f"    Total Return:  {metrics['total_return']:+.1%}")
    print(f"    Annual Return: {metrics['annual_return']:+.1%}")
    print(f"    Volatility:    {metrics['volatility']:.1%}")
    print(f"    Sharpe Ratio:  {metrics['sharpe']:.2f}")
    print(f"    Sortino Ratio: {metrics['sortino']:.2f}")
    print(f"    Max Drawdown:  {metrics['max_drawdown']:.1%}")
    print(f"    Win Rate:      {metrics['win_rate']:.1%}")
    print(f"    Trading Days:  {metrics['num_days']}")


def test_single_symbol(symbol: str) -> dict:
    """Test NUBLE vs buy-and-hold for a single symbol."""
    print(f"\n{'='*60}")
    print(f"TESTING: {symbol}")
    print(f"{'='*60}")
    
    # Load REAL data
    try:
        data = load_real_data(symbol, 'test')
    except FileNotFoundError as e:
        print(f"  ❌ Data not found: {e}")
        return {'symbol': symbol, 'error': 'Data not found'}
    
    if len(data) < 100:
        print(f"  ❌ Insufficient data: {len(data)} rows")
        return {'symbol': symbol, 'error': 'Insufficient data'}
    
    # Calculate buy-and-hold returns
    bh_returns = data['close'].pct_change().dropna()
    
    # Run REAL ML strategy
    print(f"  Running ML strategy...")
    try:
        strategy_returns = run_real_ml_strategy(symbol, data)
    except Exception as e:
        print(f"  ❌ Strategy failed: {e}")
        import traceback
        traceback.print_exc()
        return {'symbol': symbol, 'error': str(e)}
    
    # Align returns
    common_idx = bh_returns.index.intersection(strategy_returns.index)
    if len(common_idx) == 0:
        return {'symbol': symbol, 'error': 'No overlapping dates'}
    
    bh_aligned = bh_returns.loc[common_idx]
    strat_aligned = strategy_returns.loc[common_idx]
    
    # Calculate metrics
    bh_metrics = calculate_metrics(bh_aligned, f"{symbol} Buy-and-Hold")
    strat_metrics = calculate_metrics(strat_aligned, f"{symbol} NUBLE ML")
    
    print_metrics(bh_metrics)
    print_metrics(strat_metrics)
    
    # Comparison
    if 'error' in bh_metrics or 'error' in strat_metrics:
        return {'symbol': symbol, 'error': 'Metrics calculation failed'}
    
    alpha = strat_metrics['total_return'] - bh_metrics['total_return']
    sharpe_diff = strat_metrics['sharpe'] - bh_metrics['sharpe']
    
    print(f"\n  COMPARISON:")
    print(f"    Alpha (excess return): {alpha:+.1%}")
    print(f"    Sharpe difference:     {sharpe_diff:+.2f}")
    
    if alpha > 0:
        print(f"    ✅ NUBLE OUTPERFORMS by {alpha:.1%}")
        outperforms = True
    else:
        print(f"    ❌ NUBLE UNDERPERFORMS by {abs(alpha):.1%}")
        outperforms = False
    
    return {
        'symbol': symbol,
        'strategy': strat_metrics,
        'benchmark': bh_metrics,
        'alpha': alpha,
        'sharpe_diff': sharpe_diff,
        'outperforms': outperforms
    }


def test_portfolio() -> dict:
    """Test equal-weight portfolio of validated symbols vs SPY."""
    print(f"\n{'='*70}")
    print(f"PORTFOLIO TEST: NUBLE vs SPY")
    print(f"{'='*70}")
    
    # Validated symbols from your institutional audit
    symbols = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'TSLA', 'AMZN']
    
    # Load SPY as benchmark
    try:
        spy_data = load_real_data('SPY', 'test')
        spy_returns = spy_data['close'].pct_change().dropna()
    except Exception as e:
        print(f"  ❌ Could not load SPY: {e}")
        return {'error': str(e)}
    
    # Run strategy on each symbol
    all_strategy_returns = []
    successful_symbols = []
    
    for symbol in symbols:
        try:
            data = load_real_data(symbol, 'test')
            strat_returns = run_real_ml_strategy(symbol, data)
            all_strategy_returns.append(strat_returns)
            successful_symbols.append(symbol)
            print(f"  ✓ {symbol}: {len(strat_returns)} days")
        except Exception as e:
            print(f"  ✗ {symbol}: {e}")
    
    if not all_strategy_returns:
        return {'error': 'No strategies ran successfully'}
    
    # Equal-weight portfolio
    portfolio_df = pd.concat(all_strategy_returns, axis=1)
    portfolio_df.columns = successful_symbols
    portfolio_df = portfolio_df.dropna()
    portfolio_returns = portfolio_df.mean(axis=1)
    
    # Align with SPY
    common_idx = spy_returns.index.intersection(portfolio_returns.index)
    if len(common_idx) == 0:
        return {'error': 'No overlapping dates with SPY'}
    
    spy_aligned = spy_returns.loc[common_idx]
    port_aligned = portfolio_returns.loc[common_idx]
    
    # Calculate metrics
    spy_metrics = calculate_metrics(spy_aligned, "SPY Buy-and-Hold")
    port_metrics = calculate_metrics(port_aligned, "NUBLE Portfolio")
    
    print_metrics(spy_metrics)
    print_metrics(port_metrics)
    
    # Alpha calculation
    if 'error' in spy_metrics or 'error' in port_metrics:
        return {'error': 'Metrics calculation failed'}
    
    alpha = port_metrics['total_return'] - spy_metrics['total_return']
    sharpe_diff = port_metrics['sharpe'] - spy_metrics['sharpe']
    
    print(f"\n  PORTFOLIO vs SPY:")
    print(f"    Alpha:       {alpha:+.1%}")
    print(f"    Sharpe diff: {sharpe_diff:+.2f}")
    
    if alpha > 0:
        print(f"\n    ✅ NUBLE PORTFOLIO BEATS SPY by {alpha:.1%}")
    else:
        print(f"\n    ❌ NUBLE PORTFOLIO LOSES TO SPY by {abs(alpha):.1%}")
    
    return {
        'portfolio': port_metrics,
        'benchmark': spy_metrics,
        'alpha': alpha,
        'sharpe_diff': sharpe_diff,
        'symbols_tested': successful_symbols
    }


def run_full_baseline_test():
    """Run complete baseline comparison."""
    print("="*70)
    print("NUBLE vs BUY-AND-HOLD: REAL DATA, REAL STRATEGY")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Using: EnhancedSignalGenerator from src/institutional/signals/")
    print("Data: Real market data from data/test/")
    print("="*70)
    
    # Test individual symbols - your 3 core validated symbols
    symbols_to_test = ['AAPL', 'MSFT', 'NVDA']
    individual_results = []
    
    for symbol in symbols_to_test:
        result = test_single_symbol(symbol)
        individual_results.append(result)
    
    # Test portfolio
    portfolio_result = test_portfolio()
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    valid_results = [r for r in individual_results if 'error' not in r]
    outperformers = [r for r in valid_results if r.get('outperforms', False)]
    
    print(f"\nIndividual Symbols ({len(valid_results)}/{len(individual_results)} tested):")
    print(f"  Outperformed: {len(outperformers)}/{len(valid_results)}")
    
    print(f"\n{'Symbol':<8} | {'Alpha':>10} | {'Sharpe Δ':>10} | Status")
    print("-"*50)
    
    for r in individual_results:
        if 'error' in r:
            print(f"{r['symbol']:<8} | {'N/A':>10} | {'N/A':>10} | ❌ {r['error'][:20]}")
        else:
            status = "✅ OUTPERFORM" if r['outperforms'] else "❌ UNDERPERFORM"
            print(f"{r['symbol']:<8} | {r['alpha']:>+9.1%} | {r['sharpe_diff']:>+10.2f} | {status}")
    
    if 'error' not in portfolio_result:
        print(f"\nPortfolio vs SPY:")
        print(f"  NUBLE Return: {portfolio_result['portfolio']['total_return']:+.1%}")
        print(f"  SPY Return:      {portfolio_result['benchmark']['total_return']:+.1%}")
        print(f"  Alpha:           {portfolio_result['alpha']:+.1%}")
        print(f"  Sharpe diff:     {portfolio_result['sharpe_diff']:+.2f}")
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if 'error' in portfolio_result:
        print(f"❌ COULD NOT DETERMINE - Portfolio test failed: {portfolio_result['error']}")
        verdict = "ERROR"
    else:
        portfolio_alpha = portfolio_result.get('alpha', 0)
        portfolio_sharpe = portfolio_result.get('sharpe_diff', 0)
        portfolio_sharpe_abs = portfolio_result['portfolio']['sharpe']
        
        print(f"\nPortfolio Alpha: {portfolio_alpha:+.1%}")
        print(f"Portfolio Sharpe: {portfolio_sharpe_abs:.2f}")
        print(f"Sharpe vs SPY: {portfolio_sharpe:+.2f}")
        
        if portfolio_alpha > 0.05 and portfolio_sharpe_abs > 0.3:
            print("\n✅ VERDICT: NUBLE HAS REAL EDGE OVER BUY-AND-HOLD")
            print("   Alpha > 5% AND Sharpe > 0.3")
            verdict = "PASS"
        elif portfolio_alpha > 0:
            print("\n⚠️ VERDICT: NUBLE MARGINALLY BETTER THAN BUY-AND-HOLD")
            print("   Positive alpha but below 5%")
            verdict = "MARGINAL"
        else:
            print("\n❌ VERDICT: NUBLE DOES NOT BEAT BUY-AND-HOLD")
            print("   Strategy underperformed passive investing")
            verdict = "FAIL"
    
    print("\n" + "="*70)
    
    return {
        'verdict': verdict,
        'individual_results': individual_results,
        'portfolio_result': portfolio_result
    }


if __name__ == "__main__":
    results = run_full_baseline_test()
