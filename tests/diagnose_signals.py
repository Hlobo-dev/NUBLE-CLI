#!/usr/bin/env python3
"""
Diagnose why signals are failing.
"""
import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from institutional.signals.enhanced_signals import EnhancedSignalGenerator

# Load AAPL data
data = pd.read_csv('data/test/AAPL.csv', index_col=0, parse_dates=True)

# Rename columns if needed
if 'close' not in data.columns:
    data = data.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 
                                'Open': 'open', 'Volume': 'volume'})

print(f'Data: {len(data)} rows, {data.index[0].date()} to {data.index[-1].date()}')

# Test signal generator
gen = EnhancedSignalGenerator()

# Test at different points with BULL regime
test_indices = [100, 200, 300, 400, 500]

print('\nSignal Analysis with BULL regime:')
print('-' * 80)

signal_counts = {'buy': 0, 'sell': 0, 'neutral': 0}
all_signals = []

# Run through all data
for idx in range(70, len(data)):
    window = data.iloc[idx-70:idx+1]
    sig = gen.generate_signal('AAPL', window, sentiment=0.0, regime='BULL')
    all_signals.append(sig.direction)
    
    if sig.direction > 0:
        signal_counts['buy'] += 1
    elif sig.direction < 0:
        signal_counts['sell'] += 1
    else:
        signal_counts['neutral'] += 1

total = len(all_signals)
print(f"\nSignal Distribution:")
print(f"  BUY:     {signal_counts['buy']:4d} ({100*signal_counts['buy']/total:.1f}%)")
print(f"  NEUTRAL: {signal_counts['neutral']:4d} ({100*signal_counts['neutral']/total:.1f}%)")
print(f"  SELL:    {signal_counts['sell']:4d} ({100*signal_counts['sell']/total:.1f}%)")

# Sample specific signals
print("\nSample signals at key points:")
for idx in test_indices:
    window = data.iloc[idx-70:idx+1]
    sig = gen.generate_signal('AAPL', window, sentiment=0.0, regime='BULL')
    
    print(f"\nDate: {window.index[-1].date()}")
    print(f"  Direction: {sig.direction}, Strength: {sig.strength.name}")
    print(f"  Short Mom: {sig.short_term_signal:.3f}")
    print(f"  Medium Mom: {sig.medium_term_signal:.3f}")
    print(f"  Long Mom: {sig.long_term_signal:.3f}")
    print(f"  Mean Rev Short: {sig.short_term_mr:.3f}")
    print(f"  Mean Rev Med: {sig.medium_term_mr:.3f}")
    print(f"  Confidence: {sig.confidence:.3f}")

# Actual price change
print("\nActual price performance:")
first_price = data['close'].iloc[70]
last_price = data['close'].iloc[-1]
total_return = (last_price / first_price - 1) * 100
print(f"  Start: ${first_price:.2f}")
print(f"  End:   ${last_price:.2f}")
print(f"  Total Return: {total_return:+.1f}%")
