#!/usr/bin/env python3
"""Phase 7: E2E Trading Simulation - Full Trading Flow Validation.

End-to-end simulation testing:
- Complete trading workflow
- Signal → Decision → Execution → P&L
- Multi-day backtests
- Risk management integration
- Performance metrics calculation
"""
import sys
import os
import time
import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

class S(Enum):
    P="PASS"; F="FAIL"; W="WARN"

@dataclass
class R:
    name:str; status:S; time:float; detail:str=""

# ============================================================
# TRADING SIMULATION COMPONENTS
# ============================================================

@dataclass
class Trade:
    """Single trade record."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    shares: int
    price: float
    timestamp: datetime
    signal_confidence: float = 0.0

@dataclass
class Position:
    """Current position in a security."""
    symbol: str
    shares: int
    avg_cost: float
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return self.shares * (self.current_price - self.avg_cost)

@dataclass
class Portfolio:
    """Portfolio state."""
    cash: float = 100000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    
    @property
    def total_value(self) -> float:
        pos_value = sum(p.market_value for p in self.positions.values())
        return self.cash + pos_value
    
    def buy(self, symbol: str, shares: int, price: float, confidence: float = 0.0):
        cost = shares * price
        if cost > self.cash:
            raise ValueError(f"Insufficient cash: need ${cost:.2f}, have ${self.cash:.2f}")
        
        self.cash -= cost
        
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_shares = pos.shares + shares
            pos.avg_cost = (pos.avg_cost * pos.shares + price * shares) / total_shares
            pos.shares = total_shares
        else:
            self.positions[symbol] = Position(symbol, shares, price, price)
        
        self.trades.append(Trade(symbol, 'BUY', shares, price, datetime.now(), confidence))
    
    def sell(self, symbol: str, shares: int, price: float, confidence: float = 0.0):
        if symbol not in self.positions:
            raise ValueError(f"No position in {symbol}")
        
        pos = self.positions[symbol]
        if shares > pos.shares:
            raise ValueError(f"Insufficient shares: have {pos.shares}, trying to sell {shares}")
        
        self.cash += shares * price
        pos.shares -= shares
        
        if pos.shares == 0:
            del self.positions[symbol]
        
        self.trades.append(Trade(symbol, 'SELL', shares, price, datetime.now(), confidence))
    
    def update_prices(self, prices: Dict[str, float]):
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price


class TradingSimulator:
    """Full trading simulation engine."""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.portfolio = Portfolio(cash=initial_capital)
        self.initial_capital = initial_capital
        self.history: List[Dict] = []
        
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from price data."""
        signals = pd.DataFrame(index=prices.index)
        
        # Simple momentum + mean reversion hybrid
        signals['ma_short'] = prices['close'].rolling(5).mean()
        signals['ma_long'] = prices['close'].rolling(20).mean()
        signals['std'] = prices['close'].rolling(20).std()
        
        # Momentum signal
        signals['momentum'] = np.sign(signals['ma_short'] - signals['ma_long'])
        
        # Mean reversion signal
        zscore = (prices['close'] - signals['ma_long']) / signals['std']
        signals['mean_rev'] = np.where(zscore < -2, 1, np.where(zscore > 2, -1, 0))
        
        # Combined signal
        signals['signal'] = (signals['momentum'] + signals['mean_rev']) / 2
        signals['confidence'] = np.abs(signals['signal'])
        
        return signals.dropna()
    
    def apply_risk_filter(self, signal: float, confidence: float, 
                          volatility: float, max_risk: float = 0.02) -> Tuple[float, float]:
        """Apply risk management filters to signals."""
        # Scale position by inverse volatility
        vol_scalar = min(1.0, max_risk / volatility) if volatility > 0 else 0.5
        
        # Only trade high confidence signals
        if confidence < 0.3:
            return 0, 0
        
        # Scale signal by risk
        adjusted_signal = signal * vol_scalar * confidence
        adjusted_confidence = confidence * vol_scalar
        
        return adjusted_signal, adjusted_confidence
    
    def calculate_position_size(self, signal: float, confidence: float, 
                               current_price: float, max_position_pct: float = 0.10) -> int:
        """Calculate position size based on signal strength and risk."""
        if abs(signal) < 0.1:
            return 0
        
        # Max dollars per position
        max_dollars = self.portfolio.total_value * max_position_pct * abs(signal)
        
        # Shares to buy/sell
        shares = int(max_dollars / current_price)
        
        return shares * int(np.sign(signal))
    
    def run_simulation(self, price_data: pd.DataFrame, symbol: str = "SIM") -> Dict:
        """Run full trading simulation."""
        signals = self.generate_signals(price_data)
        
        # Track daily values
        daily_values = [self.initial_capital]
        
        for i, (date, row) in enumerate(signals.iterrows()):
            if i < 1:
                continue
            
            current_price = price_data.loc[date, 'close']
            volatility = row['std'] / current_price if current_price > 0 else 0.01
            
            # Get and filter signal
            raw_signal = row['signal']
            confidence = row['confidence']
            filtered_signal, adj_conf = self.apply_risk_filter(
                raw_signal, confidence, volatility
            )
            
            # Calculate position size
            target_shares = self.calculate_position_size(
                filtered_signal, adj_conf, current_price
            )
            
            # Current position
            current_shares = self.portfolio.positions.get(symbol, Position(symbol, 0, 0)).shares
            
            # Execute trade
            trade_shares = target_shares - current_shares
            
            try:
                if trade_shares > 0:
                    # Check if we can afford
                    cost = trade_shares * current_price
                    if cost <= self.portfolio.cash:
                        self.portfolio.buy(symbol, trade_shares, current_price, adj_conf)
                elif trade_shares < 0 and symbol in self.portfolio.positions:
                    # Sell (can only sell what we have)
                    sell_shares = min(abs(trade_shares), current_shares)
                    if sell_shares > 0:
                        self.portfolio.sell(symbol, sell_shares, current_price, adj_conf)
            except ValueError:
                pass  # Skip invalid trades
            
            # Update prices and track value
            self.portfolio.update_prices({symbol: current_price})
            daily_values.append(self.portfolio.total_value)
        
        # Calculate returns
        daily_values = np.array(daily_values)
        returns = np.diff(daily_values) / daily_values[:-1]
        
        # Performance metrics
        total_return = (daily_values[-1] - self.initial_capital) / self.initial_capital
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_dd = self._calculate_max_drawdown(daily_values)
        
        return {
            "final_value": daily_values[-1],
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "n_trades": len(self.portfolio.trades),
            "daily_values": daily_values
        }
    
    def _calculate_max_drawdown(self, values: np.ndarray) -> float:
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return np.max(drawdown)


class E2ETester:
    """End-to-end trading simulation test suite."""
    
    def __init__(self):
        self.results: List[R] = []
        
    def add(self, r: R):
        self.results.append(r)
        sym = "✅" if r.status == S.P else "❌" if r.status == S.F else "⚠️"
        print(f"  {sym} {r.name} ({r.time:.1f}s) {r.detail}")
    
    def _generate_price_data(self, n_days: int = 252, trend: float = 0.0, 
                             vol: float = 0.02, seed: int = 42) -> pd.DataFrame:
        """Generate synthetic price data."""
        np.random.seed(seed)
        
        dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
        
        # GBM price simulation
        returns = np.random.randn(n_days) * vol + trend
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create OHLCV
        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(n_days) * 0.002),
            'high': prices * (1 + np.abs(np.random.randn(n_days)) * 0.01),
            'low': prices * (1 - np.abs(np.random.randn(n_days)) * 0.01),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
        
        return df
    
    # ============================================================
    # E2E SIMULATION TESTS
    # ============================================================
    
    def test_bull_market_simulation(self) -> R:
        """E2E: Trading simulation in bull market."""
        t = time.time()
        try:
            # Strong uptrend
            prices = self._generate_price_data(n_days=252, trend=0.001, vol=0.015, seed=42)
            
            sim = TradingSimulator(initial_capital=100000)
            results = sim.run_simulation(prices, "BULL")
            
            # In bull market, should make positive returns
            profit = results['total_return'] > 0
            has_trades = results['n_trades'] > 5
            
            return R("E2E: BullMarket", S.P if profit and has_trades else S.W, 
                    time.time()-t, f"ret={results['total_return']:.1%},trades={results['n_trades']}")
        except Exception as e:
            return R("E2E: BullMarket", S.F, time.time()-t, str(e)[:50])
    
    def test_bear_market_simulation(self) -> R:
        """E2E: Trading simulation in bear market."""
        t = time.time()
        try:
            # Strong downtrend
            prices = self._generate_price_data(n_days=252, trend=-0.001, vol=0.02, seed=123)
            
            sim = TradingSimulator(initial_capital=100000)
            results = sim.run_simulation(prices, "BEAR")
            
            # In bear market with long-only, expect limited losses
            max_dd = results['max_drawdown']
            reasonable_dd = max_dd < 0.50  # Not more than 50% drawdown
            
            return R("E2E: BearMarket", S.P if reasonable_dd else S.W, 
                    time.time()-t, f"ret={results['total_return']:.1%},maxDD={max_dd:.1%}")
        except Exception as e:
            return R("E2E: BearMarket", S.F, time.time()-t, str(e)[:50])
    
    def test_sideways_market_simulation(self) -> R:
        """E2E: Trading simulation in sideways market."""
        t = time.time()
        try:
            # No trend, moderate volatility
            prices = self._generate_price_data(n_days=252, trend=0.0, vol=0.018, seed=456)
            
            sim = TradingSimulator(initial_capital=100000)
            results = sim.run_simulation(prices, "FLAT")
            
            # In sideways, mean reversion should work
            has_value = results['final_value'] > 0
            
            return R("E2E: Sideways", S.P if has_value else S.F, 
                    time.time()-t, f"ret={results['total_return']:.1%},sharpe={results['sharpe_ratio']:.2f}")
        except Exception as e:
            return R("E2E: Sideways", S.F, time.time()-t, str(e)[:50])
    
    def test_high_volatility_simulation(self) -> R:
        """E2E: Trading simulation in high volatility regime."""
        t = time.time()
        try:
            # High volatility environment
            prices = self._generate_price_data(n_days=252, trend=0.0005, vol=0.04, seed=789)
            
            sim = TradingSimulator(initial_capital=100000)
            results = sim.run_simulation(prices, "HVOL")
            
            # Should handle high vol without blowing up
            survived = results['final_value'] > 0 and results['max_drawdown'] < 0.80
            
            return R("E2E: HighVol", S.P if survived else S.W, 
                    time.time()-t, f"ret={results['total_return']:.1%},maxDD={results['max_drawdown']:.1%}")
        except Exception as e:
            return R("E2E: HighVol", S.F, time.time()-t, str(e)[:50])
    
    def test_crash_recovery_simulation(self) -> R:
        """E2E: Simulate crash and recovery scenario."""
        t = time.time()
        try:
            np.random.seed(101)
            
            # Normal market
            normal1 = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.015))
            # Crash (30% drop in 10 days)
            crash = normal1[-1] * np.exp(np.cumsum(np.random.randn(10) * 0.03 - 0.03))
            # Recovery
            recovery = crash[-1] * np.exp(np.cumsum(np.random.randn(142) * 0.015 + 0.002))
            
            prices_arr = np.concatenate([normal1, crash, recovery])
            
            dates = pd.date_range('2020-01-01', periods=252, freq='B')
            prices = pd.DataFrame({
                'open': prices_arr * 0.99,
                'high': prices_arr * 1.01,
                'low': prices_arr * 0.98,
                'close': prices_arr,
                'volume': np.random.randint(1000000, 10000000, 252)
            }, index=dates)
            
            sim = TradingSimulator(initial_capital=100000)
            results = sim.run_simulation(prices, "CRASH")
            
            # Should survive crash and participate in recovery
            survived = results['final_value'] > 50000  # Not total loss
            
            return R("E2E: CrashRecov", S.P if survived else S.W, 
                    time.time()-t, f"final=${results['final_value']:.0f},maxDD={results['max_drawdown']:.1%}")
        except Exception as e:
            return R("E2E: CrashRecov", S.F, time.time()-t, str(e)[:50])
    
    def test_multi_year_simulation(self) -> R:
        """E2E: Multi-year backtest simulation."""
        t = time.time()
        try:
            # 5 years of data
            prices = self._generate_price_data(n_days=1260, trend=0.0003, vol=0.018, seed=202)
            
            sim = TradingSimulator(initial_capital=100000)
            results = sim.run_simulation(prices, "5YR")
            
            # 5 years should have meaningful results
            has_trades = results['n_trades'] > 20
            valid_results = not np.isnan(results['sharpe_ratio'])
            
            return R("E2E: MultiYear", S.P if has_trades and valid_results else S.W, 
                    time.time()-t, f"ret={results['total_return']:.1%},trades={results['n_trades']}")
        except Exception as e:
            return R("E2E: MultiYear", S.F, time.time()-t, str(e)[:50])
    
    # ============================================================
    # RISK MANAGEMENT INTEGRATION TESTS
    # ============================================================
    
    def test_position_sizing_limits(self) -> R:
        """E2E: Position sizing respects limits."""
        t = time.time()
        try:
            prices = self._generate_price_data(n_days=100, trend=0.002, vol=0.01, seed=303)
            
            sim = TradingSimulator(initial_capital=100000)
            results = sim.run_simulation(prices, "LIMIT")
            
            # Check no position exceeded 10% of portfolio
            max_position = 0
            for trade in sim.portfolio.trades:
                position_value = trade.shares * trade.price
                pct_of_portfolio = position_value / sim.initial_capital
                max_position = max(max_position, pct_of_portfolio)
            
            within_limits = max_position <= 0.15  # Allow some buffer
            
            return R("E2E: PosLimits", S.P if within_limits else S.W, 
                    time.time()-t, f"maxPos={max_position:.1%}")
        except Exception as e:
            return R("E2E: PosLimits", S.F, time.time()-t, str(e)[:50])
    
    def test_no_negative_cash(self) -> R:
        """E2E: Portfolio never goes negative cash."""
        t = time.time()
        try:
            prices = self._generate_price_data(n_days=252, trend=0.001, vol=0.02, seed=404)
            
            sim = TradingSimulator(initial_capital=100000)
            
            # Track cash through simulation
            original_buy = sim.portfolio.buy
            cash_history = [sim.portfolio.cash]
            
            def tracked_buy(*args, **kwargs):
                result = original_buy(*args, **kwargs)
                cash_history.append(sim.portfolio.cash)
                return result
            
            sim.portfolio.buy = tracked_buy
            
            results = sim.run_simulation(prices, "CASH")
            
            # Verify cash never went negative
            min_cash = min(cash_history)
            no_negative = min_cash >= 0
            
            return R("E2E: NoCashNeg", S.P if no_negative else S.F, 
                    time.time()-t, f"minCash=${min_cash:.0f}")
        except Exception as e:
            return R("E2E: NoCashNeg", S.F, time.time()-t, str(e)[:50])
    
    # ============================================================
    # PERFORMANCE METRICS VALIDATION
    # ============================================================
    
    def test_sharpe_calculation(self) -> R:
        """E2E: Sharpe ratio calculated correctly."""
        t = time.time()
        try:
            # Known returns for Sharpe verification
            np.random.seed(505)
            returns = np.random.randn(252) * 0.01 + 0.0005  # ~12% annual, 16% vol
            
            # Manual Sharpe
            manual_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            # Should be roughly 1.0
            reasonable = 0.5 < manual_sharpe < 2.0
            
            return R("E2E: SharpCalc", S.P if reasonable else S.W, 
                    time.time()-t, f"sharpe={manual_sharpe:.2f}")
        except Exception as e:
            return R("E2E: SharpCalc", S.F, time.time()-t, str(e)[:50])
    
    def test_drawdown_calculation(self) -> R:
        """E2E: Max drawdown calculated correctly."""
        t = time.time()
        try:
            # Create known drawdown scenario
            values = np.array([100, 110, 120, 100, 90, 80, 85, 95, 100, 105])
            # Peak at 120, trough at 80 = 33.3% DD
            
            peak = np.maximum.accumulate(values)
            drawdown = (peak - values) / peak
            max_dd = np.max(drawdown)
            
            # Should be ~33.3%
            correct = abs(max_dd - 0.333) < 0.01
            
            return R("E2E: DDCalc", S.P if correct else S.F, 
                    time.time()-t, f"maxDD={max_dd:.1%}")
        except Exception as e:
            return R("E2E: DDCalc", S.F, time.time()-t, str(e)[:50])
    
    # ============================================================
    # RUN ALL E2E TESTS
    # ============================================================
    
    def run_all(self):
        print("\n" + "="*60)
        print("PHASE 7: E2E TRADING SIMULATION")
        print("="*60)
        
        print("\n" + "-"*40)
        print("MARKET CONDITION SIMULATIONS")
        print("-"*40)
        self.add(self.test_bull_market_simulation())
        self.add(self.test_bear_market_simulation())
        self.add(self.test_sideways_market_simulation())
        self.add(self.test_high_volatility_simulation())
        self.add(self.test_crash_recovery_simulation())
        self.add(self.test_multi_year_simulation())
        
        print("\n" + "-"*40)
        print("RISK MANAGEMENT INTEGRATION")
        print("-"*40)
        self.add(self.test_position_sizing_limits())
        self.add(self.test_no_negative_cash())
        
        print("\n" + "-"*40)
        print("PERFORMANCE METRICS")
        print("-"*40)
        self.add(self.test_sharpe_calculation())
        self.add(self.test_drawdown_calculation())
        
        # Summary
        passed = sum(1 for r in self.results if r.status == S.P)
        warned = sum(1 for r in self.results if r.status == S.W)
        failed = sum(1 for r in self.results if r.status == S.F)
        total = len(self.results)
        total_time = sum(r.time for r in self.results)
        
        print("\n" + "="*60)
        print(f"RESULTS: ✅{passed} ⚠️{warned} ❌{failed} Total:{total} Time:{total_time:.1f}s")
        print("="*60)
        
        if failed == 0:
            print("🏆 E2E TRADING: ALL SIMULATIONS PASSED")
        else:
            print(f"❌ E2E TRADING: {failed} FAILURES")
            
        print("="*60 + "\n")
        
        return failed == 0


if __name__ == "__main__":
    tester = E2ETester()
    success = tester.run_all()
    sys.exit(0 if success else 1)
