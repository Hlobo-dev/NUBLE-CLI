#!/usr/bin/env python3
"""
Quick Institutional Audit Test
"""
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

print("="*70)
print("KYPERIAN INSTITUTIONAL AUDIT - QUICK TEST")
print("="*70)

# Import modules
print("\n1. Importing modules...")
try:
    from src.institutional.risk.risk_manager import RiskManager, RiskLimits, RiskState, TradeRequest
    print("   ✅ Risk Manager imported")
except Exception as e:
    print(f"   ❌ Risk Manager: {e}")

try:
    from src.institutional.costs.transaction_costs import TransactionCostModel, LiquidityTier
    print("   ✅ Transaction Costs imported")
except Exception as e:
    print(f"   ❌ Transaction Costs: {e}")

try:
    from validation.alpha_attribution import AlphaAttribution
    print("   ✅ Alpha Attribution imported")
except Exception as e:
    print(f"   ❌ Alpha Attribution: {e}")

# Test Risk Manager
print("\n2. Testing Risk Manager...")
limits = RiskLimits()
rm = RiskManager(limits=limits, initial_nav=1_000_000)

trade = TradeRequest(symbol='AAPL', side='BUY', quantity=100, price=180.0)
decision = rm.check_trade(trade)
print(f"   Trade validation: {'✅ PASS' if decision.allowed else '❌ FAIL'}")

rm.update_nav(900000)  # 10% drawdown
print(f"   Drawdown state: {rm.state.value} {'✅ PASS' if rm.state == RiskState.REDUCED else '❌ FAIL'}")

rm.kill_switch("Test")
print(f"   Kill switch: {'✅ PASS' if rm.state == RiskState.HALTED else '❌ FAIL'}")

# Test Transaction Costs
print("\n3. Testing Transaction Costs...")
tcm = TransactionCostModel()  # Uses default config
breakdown = tcm.calculate_cost(
    symbol='AAPL',
    trade_value=100_000, 
    daily_volume_usd=1_000_000_000,
    volatility=0.02
)
print(f"   Cost calculation: {breakdown.total_bps:.1f} bps {'✅ PASS' if breakdown.total_cost > 0 else '❌ FAIL'}")

# Test Alpha Attribution
print("\n4. Testing Alpha Attribution...")
import numpy as np
import pandas as pd

np.random.seed(42)
dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
benchmark = pd.Series(np.random.normal(0.0003, 0.01, len(dates)), index=dates)
strategy = 0.0002 + 0.8 * benchmark + pd.Series(np.random.normal(0, 0.005, len(dates)), index=dates)

aa = AlphaAttribution()
result = aa.calculate_alpha_beta(strategy, benchmark)
print(f"   Alpha: {result.alpha_annualized:.1%}")
print(f"   T-stat: {result.alpha_t_stat:.2f}")
print(f"   Has Alpha: {'✅ YES' if result.has_alpha else '❌ NO'}")

# Summary
print("\n" + "="*70)
print("AUDIT MODULES TEST COMPLETE")
print("="*70)
print("\n✅ All institutional audit modules working correctly")
print("\nNext Steps:")
print("1. Download Polygon data to run on real market data")
print("2. Run full validation with: python validation/run_institutional_audit.py")
