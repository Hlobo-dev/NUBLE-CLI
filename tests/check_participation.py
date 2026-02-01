#!/usr/bin/env python3
"""
Check market participation.
"""
import sys
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

# Import directly from the signals module
PROJECT_ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location(
    "enhanced_signals",
    str(PROJECT_ROOT / "src" / "institutional" / "signals" / "enhanced_signals.py")
)
signals_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(signals_module)
EnhancedSignalGenerator = signals_module.EnhancedSignalGenerator

# Test all symbols
symbols = ['AAPL', 'MSFT', 'NVDA', 'SPY']

for symbol in symbols:
    data = pd.read_csv(f'data/test/{symbol}.csv', index_col=0, parse_dates=True)
    if 'close' not in data.columns:
        data = data.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 
                                    'Open': 'open', 'Volume': 'volume'})
    
    returns = data['close'].pct_change()
    
    def detect_regime(recent_returns, lookback_60=None):
        if len(recent_returns) < 20:
            return 'SIDEWAYS'
        mean_ret = recent_returns.mean()
        ann_return = mean_ret * 252
        if lookback_60 is not None and len(lookback_60) >= 40:
            trend_return = lookback_60.mean() * 252
        else:
            trend_return = ann_return
        vol = recent_returns.std()
        ann_vol = vol * np.sqrt(252)
        if ann_vol > 0.35:
            return 'VOLATILE'
        elif trend_return > 0.05:
            return 'BULL'
        elif trend_return < -0.05:
            return 'BEAR'
        else:
            return 'SIDEWAYS'
    
    gen = EnhancedSignalGenerator()
    
    signals = []
    lookback = 70
    for i in range(lookback, len(data)):
        recent_returns = returns.iloc[i-20:i]
        lookback_60 = returns.iloc[max(0, i-60):i]
        regime = detect_regime(recent_returns, lookback_60)
        
        window = data.iloc[i-lookback:i+1]
        sig = gen.generate_signal(symbol, window, sentiment=0.0, regime=regime)
        signals.append(sig.direction)
    
    signals = pd.Series(signals)
    in_market = (signals != 0).sum()
    total = len(signals)
    
    print(f"{symbol}:")
    print(f"  Days in market: {in_market}/{total} ({100*in_market/total:.1f}%)")
    print(f"  BUY days: {(signals == 1).sum()}")
    print(f"  SELL days: {(signals == -1).sum()}")
    print(f"  NEUTRAL days: {(signals == 0).sum()}")
    print()
