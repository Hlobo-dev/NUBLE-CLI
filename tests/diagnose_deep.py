#!/usr/bin/env python3
"""
Deep diagnosis of signal performance.
"""
import sys
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

# Import directly from the signals module to avoid loading all dependencies
PROJECT_ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location(
    "enhanced_signals",
    str(PROJECT_ROOT / "src" / "institutional" / "signals" / "enhanced_signals.py")
)
signals_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(signals_module)
EnhancedSignalGenerator = signals_module.EnhancedSignalGenerator

# Load AAPL data
data = pd.read_csv('data/test/AAPL.csv', index_col=0, parse_dates=True)

# Rename columns if needed
if 'close' not in data.columns:
    data = data.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 
                                'Open': 'open', 'Volume': 'volume'})

print(f'Data: {len(data)} rows, {data.index[0].date()} to {data.index[-1].date()}')

returns = data['close'].pct_change()

# Detect regime like the test does
def detect_regime(recent_returns, lookback_60=None):
    if len(recent_returns) < 20:
        return 'SIDEWAYS'
    
    mean_ret = recent_returns.mean()
    vol = recent_returns.std()
    
    ann_return = mean_ret * 252
    ann_vol = vol * np.sqrt(252)
    
    if lookback_60 is not None and len(lookback_60) >= 40:
        trend_return = lookback_60.mean() * 252
    else:
        trend_return = ann_return
    
    if ann_vol > 0.35:
        return 'VOLATILE'
    elif trend_return > 0.05:
        return 'BULL'
    elif trend_return < -0.05:
        return 'BEAR'
    else:
        return 'SIDEWAYS'

# Count regimes
regime_counts = {'BULL': 0, 'BEAR': 0, 'SIDEWAYS': 0, 'VOLATILE': 0}

gen = EnhancedSignalGenerator()
signals = []
regimes_used = []
lookback = 70

for i in range(lookback, len(data)):
    recent_returns = returns.iloc[i-20:i]
    lookback_60 = returns.iloc[max(0, i-60):i]
    
    regime = detect_regime(recent_returns, lookback_60)
    regime_counts[regime] += 1
    regimes_used.append(regime)
    
    window = data.iloc[i-lookback:i+1]
    sig = gen.generate_signal('AAPL', window, sentiment=0.0, regime=regime)
    signals.append(sig.direction)

total = len(signals)
print("\nRegime Distribution:")
for r, cnt in regime_counts.items():
    print(f"  {r}: {cnt} ({100*cnt/total:.1f}%)")

# Now check win rate per regime
signals_series = pd.Series(signals, index=data.index[lookback:])
returns_aligned = returns.iloc[lookback:]

# Strategy returns: signal from yesterday * return today
strategy_returns = signals_series.shift(1) * returns_aligned
strategy_returns = strategy_returns.dropna()

# Overall win rate
correct = (strategy_returns > 0).sum()
total = len(strategy_returns[strategy_returns != 0])  # Exclude neutral days
print(f"\nOverall Win Rate: {100*correct/total:.1f}% ({correct}/{total})")

# Check signal direction vs actual next-day return
regimes_arr = pd.Series(regimes_used, index=data.index[lookback:])
for regime in ['BULL', 'BEAR', 'SIDEWAYS', 'VOLATILE']:
    mask = regimes_arr == regime
    regime_signals = signals_series[mask]
    regime_returns = returns_aligned[mask]
    
    # Strategy returns for this regime
    strat_ret = regime_signals.shift(1) * regime_returns
    strat_ret = strat_ret.dropna()
    
    if len(strat_ret[strat_ret != 0]) > 0:
        correct = (strat_ret > 0).sum()
        total_active = len(strat_ret[strat_ret != 0])
        total_return = (1 + strat_ret).prod() - 1
        
        # Buy-hold for same period
        bh_return = (1 + regime_returns.iloc[1:]).prod() - 1
        
        print(f"\n{regime}:")
        print(f"  Days: {len(regime_signals)}")
        print(f"  Win Rate: {100*correct/total_active:.1f}% ({correct}/{total_active})")
        print(f"  Strategy Return: {100*total_return:.1f}%")
        print(f"  Buy-Hold Return: {100*bh_return:.1f}%")
        
        # Signal distribution in this regime
        buys = (regime_signals == 1).sum()
        sells = (regime_signals == -1).sum()
        neutrals = (regime_signals == 0).sum()
        print(f"  Signals: BUY {buys} ({100*buys/len(regime_signals):.1f}%), SELL {sells} ({100*sells/len(regime_signals):.1f}%), NEUTRAL {neutrals} ({100*neutrals/len(regime_signals):.1f}%)")
