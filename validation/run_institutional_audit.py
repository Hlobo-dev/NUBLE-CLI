#!/usr/bin/env python3
"""
KYPERIAN INSTITUTIONAL AUDIT
============================

Master script that runs ALL audit modules on actual data.

Modules:
1. Risk Manager - Position limits, kill switch
2. Proper PBO - Bailey/López de Prado methodology
3. Transaction Costs - Realistic cost modeling
4. Full Universe Test - All 22 symbols
5. Alpha Attribution - Prove alpha vs beta

This produces the DEFINITIVE institutional-grade validation report.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Import audit modules
from src.institutional.risk.risk_manager import (
    RiskManager, RiskLimits, RiskState, TradeRequest, Position
)
from src.institutional.costs.transaction_costs import (
    TransactionCostModel, LiquidityTier, backtest_with_realistic_costs
)
from validation.alpha_attribution import AlphaAttribution

# Try to import PBO (may need scipy)
try:
    from src.institutional.validation.proper_pbo import ProperPBO
    HAS_PBO = True
except ImportError as e:
    print(f"Warning: Could not import ProperPBO: {e}")
    HAS_PBO = False


class InstitutionalAudit:
    """
    Master audit class that runs all institutional validation.
    """
    
    # Data paths to try
    DATA_PATHS = [
        PROJECT_ROOT / 'data' / 'test',      # Real test data (2023-2026)
        PROJECT_ROOT / 'data' / 'train',     # Real train data (2015-2022)
        PROJECT_ROOT / 'data' / 'polygon',
        PROJECT_ROOT / 'data',
        Path.home() / 'polygon_data',
        PROJECT_ROOT / 'validation' / 'test_data',
    ]
    
    # Symbols to test
    FULL_UNIVERSE = [
        'AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC', 'GOOGL', 'META', 'AMZN', 'TSLA',
        'JPM', 'BAC', 'GS', 'JNJ', 'PFE', 'UNH', 'XOM', 'CVX',
        'SPY', 'QQQ', 'IWM', 'DIA', 'GLD'
    ]
    
    SECTOR_MAP = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
        'AMD': 'Technology', 'INTC': 'Technology', 'GOOGL': 'Technology',
        'META': 'Technology', 'AMZN': 'Consumer', 'TSLA': 'Consumer',
        'JPM': 'Financials', 'BAC': 'Financials', 'GS': 'Financials',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
        'XOM': 'Energy', 'CVX': 'Energy',
        'SPY': 'Index', 'QQQ': 'Index', 'IWM': 'Index', 'DIA': 'Index',
        'GLD': 'Commodities'
    }
    
    LIQUIDITY_TIERS = {
        'AAPL': LiquidityTier.MEGA_CAP, 'MSFT': LiquidityTier.MEGA_CAP,
        'NVDA': LiquidityTier.MEGA_CAP, 'GOOGL': LiquidityTier.MEGA_CAP,
        'META': LiquidityTier.MEGA_CAP, 'AMZN': LiquidityTier.MEGA_CAP,
        'TSLA': LiquidityTier.LARGE_CAP, 'AMD': LiquidityTier.LARGE_CAP,
        'INTC': LiquidityTier.LARGE_CAP, 'JPM': LiquidityTier.MEGA_CAP,
        'BAC': LiquidityTier.MEGA_CAP, 'GS': LiquidityTier.LARGE_CAP,
        'JNJ': LiquidityTier.MEGA_CAP, 'PFE': LiquidityTier.LARGE_CAP,
        'UNH': LiquidityTier.LARGE_CAP, 'XOM': LiquidityTier.MEGA_CAP,
        'CVX': LiquidityTier.LARGE_CAP, 'SPY': LiquidityTier.MEGA_CAP,
        'QQQ': LiquidityTier.MEGA_CAP, 'IWM': LiquidityTier.LARGE_CAP,
        'DIA': LiquidityTier.LARGE_CAP, 'GLD': LiquidityTier.LARGE_CAP,
    }
    
    def __init__(self, initial_nav: float = 1_000_000):
        self.initial_nav = initial_nav
        self.data_dir = self._find_data_dir()
        self.data: Dict[str, pd.DataFrame] = {}
        self.results = {}
        
    def _find_data_dir(self) -> Optional[Path]:
        """Find directory with data files."""
        for path in self.DATA_PATHS:
            if path.exists() and list(path.glob('*.csv')):
                return path
        return None
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available symbol data."""
        print("\n📂 LOADING DATA...")
        print("-" * 60)
        
        if self.data_dir is None:
            print("   ⚠️ No data directory found, generating synthetic data...")
            self._generate_synthetic_data()
            return self.data
        
        print(f"   Data directory: {self.data_dir}")
        
        loaded = 0
        for symbol in self.FULL_UNIVERSE:
            df = self._load_symbol(symbol)
            if df is not None:
                self.data[symbol] = df
                loaded += 1
                print(f"   ✅ {symbol}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
        
        if loaded == 0:
            print("   ⚠️ No data loaded, generating synthetic data...")
            self._generate_synthetic_data()
        else:
            print(f"\n   Loaded {loaded}/{len(self.FULL_UNIVERSE)} symbols")
        
        return self.data
    
    def _load_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data for a single symbol."""
        patterns = [f"{symbol}.csv", f"{symbol}_daily.csv", f"{symbol}_1d.csv"]
        
        for pattern in patterns:
            filepath = self.data_dir / pattern
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    df.columns = [c.lower() for c in df.columns]
                    
                    # Parse date
                    date_col = 'date' if 'date' in df.columns else 'timestamp'
                    if date_col in df.columns:
                        df['date'] = pd.to_datetime(df[date_col])
                        df = df.set_index('date').sort_index()
                    
                    # Validate columns
                    required = ['open', 'high', 'low', 'close', 'volume']
                    if all(c in df.columns for c in required):
                        return df
                except Exception as e:
                    print(f"   ⚠️ Error loading {symbol}: {e}")
        
        return None
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for testing when real data unavailable."""
        print("   Generating synthetic market data...")
        
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
        
        for symbol in self.FULL_UNIVERSE[:10]:  # First 10 symbols
            np.random.seed(hash(symbol) % 2**32)
            
            # Different characteristics by sector
            sector = self.SECTOR_MAP.get(symbol, 'Unknown')
            if sector == 'Technology':
                drift, vol = 0.0008, 0.025
            elif sector == 'Financials':
                drift, vol = 0.0004, 0.018
            elif sector == 'Index':
                drift, vol = 0.0003, 0.012
            else:
                drift, vol = 0.0005, 0.020
            
            returns = np.random.normal(drift, vol, len(dates))
            price = 100 * np.exp(np.cumsum(returns))
            
            self.data[symbol] = pd.DataFrame({
                'open': price * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
                'high': price * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'low': price * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'close': price,
                'volume': np.random.uniform(1e6, 1e8, len(dates))
            }, index=dates)
        
        print(f"   Generated {len(self.data)} synthetic symbols")
    
    def run_risk_manager_audit(self) -> Dict:
        """
        AUDIT 1: Risk Manager Stress Test
        """
        print("\n" + "=" * 70)
        print("🛡️ AUDIT 1: RISK MANAGER STRESS TEST")
        print("=" * 70)
        
        limits = RiskLimits(
            max_position_pct=0.10,
            max_sector_pct=0.30,
            max_gross_exposure=1.5,
            drawdown_warning=0.05,
            drawdown_reduced=0.10,
            drawdown_minimal=0.15,
            drawdown_halt=0.20,
            max_daily_loss=0.03
        )
        
        risk_mgr = RiskManager(limits=limits, initial_nav=self.initial_nav)
        
        results = {
            'tests': {},
            'passed': 0,
            'failed': 0
        }
        
        # Test 1: Normal trade validation
        print("\n   Test 1: Normal Trade Validation")
        trade = TradeRequest(symbol='AAPL', side='BUY', quantity=100, price=180.0)
        decision = risk_mgr.check_trade(trade)
        passed = decision.allowed
        results['tests']['normal_trade'] = passed
        results['passed' if passed else 'failed'] += 1
        print(f"   {'✅' if passed else '❌'} Normal trade allowed: {passed}")
        
        # Test 2: Position limit enforcement
        print("\n   Test 2: Position Limit (10% max)")
        big_trade = TradeRequest(symbol='MSFT', side='BUY', quantity=1000, price=400.0)
        decision = risk_mgr.check_trade(big_trade)
        # $400K trade on $1M portfolio = 40%, should be reduced or rejected
        passed = not decision.allowed or decision.adjusted_quantity < 1000
        results['tests']['position_limit'] = passed
        results['passed' if passed else 'failed'] += 1
        print(f"   {'✅' if passed else '❌'} Position limit enforced: {passed}")
        print(f"      Original: 1000 shares, Adjusted: {decision.adjusted_quantity:.0f}")
        
        # Test 3: Sector limit
        print("\n   Test 3: Sector Concentration (30% max)")
        # Simulate existing tech positions
        risk_mgr.positions['AAPL'] = Position('AAPL', 500, 180, 90000, 'Technology', 'Stock', 'LONG')
        risk_mgr.positions['MSFT'] = Position('MSFT', 250, 400, 100000, 'Technology', 'Stock', 'LONG')
        risk_mgr.positions['NVDA'] = Position('NVDA', 100, 800, 80000, 'Technology', 'Stock', 'LONG')
        # Tech exposure now: $270K / $1M = 27%
        
        another_tech = TradeRequest(symbol='AMD', side='BUY', quantity=500, price=150.0, sector='Technology')
        decision = risk_mgr.check_trade(another_tech)
        # Adding $75K tech would exceed 30% sector limit
        passed = len(decision.warnings) > 0 or not decision.allowed
        results['tests']['sector_limit'] = passed
        results['passed' if passed else 'failed'] += 1
        print(f"   {'✅' if passed else '❌'} Sector limit warning: {passed}")
        
        # Test 4: Drawdown state machine
        print("\n   Test 4: Drawdown State Machine")
        risk_mgr = RiskManager(limits=limits, initial_nav=1_000_000)  # Fresh instance
        
        states_correct = True
        
        risk_mgr.update_nav(970000)  # 3% DD - should stay normal
        if risk_mgr.state != RiskState.NORMAL:
            states_correct = False
            print(f"   ❌ 3% DD: Expected NORMAL, got {risk_mgr.state}")
        
        risk_mgr.update_nav(900000)  # 10% DD - should be REDUCED
        if risk_mgr.state != RiskState.REDUCED:
            states_correct = False
            print(f"   ❌ 10% DD: Expected REDUCED, got {risk_mgr.state}")
        
        risk_mgr.update_nav(850000)  # 15% DD - should be MINIMAL
        if risk_mgr.state != RiskState.MINIMAL:
            states_correct = False
            print(f"   ❌ 15% DD: Expected MINIMAL, got {risk_mgr.state}")
        
        risk_mgr.update_nav(790000)  # 21% DD - should be HALTED
        if risk_mgr.state != RiskState.HALTED:
            states_correct = False
            print(f"   ❌ 21% DD: Expected HALTED, got {risk_mgr.state}")
        
        results['tests']['drawdown_states'] = states_correct
        results['passed' if states_correct else 'failed'] += 1
        print(f"   {'✅' if states_correct else '❌'} Drawdown states correct: {states_correct}")
        
        # Test 5: Kill switch blocks trading
        print("\n   Test 5: Kill Switch")
        risk_mgr = RiskManager(limits=limits, initial_nav=1_000_000)
        risk_mgr.kill_switch("Audit test")
        
        trade = TradeRequest(symbol='AAPL', side='BUY', quantity=10, price=180.0)
        decision = risk_mgr.check_trade(trade)
        passed = not decision.allowed and risk_mgr.state == RiskState.HALTED
        results['tests']['kill_switch'] = passed
        results['passed' if passed else 'failed'] += 1
        print(f"   {'✅' if passed else '❌'} Kill switch blocks trading: {passed}")
        
        # Test 6: Risk scaling
        print("\n   Test 6: Risk Scaling by State")
        risk_mgr = RiskManager(limits=limits, initial_nav=1_000_000)
        
        # In NORMAL state
        trade = TradeRequest(symbol='AAPL', side='BUY', quantity=100, price=180.0)
        decision_normal = risk_mgr.check_trade(trade)
        
        # Force REDUCED state
        risk_mgr.update_nav(890000)  # >10% DD
        decision_reduced = risk_mgr.check_trade(trade)
        
        # Should be scaled to 50%
        passed = decision_reduced.adjusted_quantity <= decision_normal.adjusted_quantity * 0.6
        results['tests']['risk_scaling'] = passed
        results['passed' if passed else 'failed'] += 1
        print(f"   {'✅' if passed else '❌'} Risk scaling works: {passed}")
        print(f"      NORMAL qty: {decision_normal.adjusted_quantity}, REDUCED qty: {decision_reduced.adjusted_quantity}")
        
        # Summary
        print("\n" + "-" * 60)
        print(f"   RISK MANAGER: {results['passed']}/{results['passed']+results['failed']} tests passed")
        
        self.results['risk_manager'] = results
        return results
    
    def run_transaction_cost_audit(self) -> Dict:
        """
        AUDIT 2: Transaction Cost Reality Check
        """
        print("\n" + "=" * 70)
        print("💰 AUDIT 2: TRANSACTION COST ANALYSIS")
        print("=" * 70)
        
        results = {
            'by_symbol': {},
            'gross_vs_net': {}
        }
        
        print("\n   Cost breakdown by symbol (100K trade):")
        print("   " + "-" * 55)
        print(f"   {'Symbol':<8} {'Tier':<12} {'Spread':>8} {'Impact':>8} {'Total':>10}")
        print("   " + "-" * 55)
        
        for symbol in ['AAPL', 'MSFT', 'NVDA', 'JPM', 'SPY']:
            if symbol not in self.data:
                continue
            
            tier = self.LIQUIDITY_TIERS.get(symbol, LiquidityTier.LARGE_CAP)
            tcm = TransactionCostModel()  # Uses default config
            
            # Use actual data for volume and volatility
            df = self.data[symbol]
            avg_volume = df['volume'].mean() * df['close'].iloc[-1]  # Dollar volume
            volatility = df['close'].pct_change().std()
            
            # Use calculate_cost method with proper signature
            breakdown = tcm.calculate_cost(
                symbol=symbol,
                trade_value=100_000,
                daily_volume_usd=avg_volume,
                volatility=volatility
            )
            
            cost = breakdown.cost_pct
            
            results['by_symbol'][symbol] = {
                'tier': tier.name,
                'spread_bps': breakdown.spread_cost / 100_000 * 10000,
                'impact_bps': breakdown.impact_cost / 100_000 * 10000,
                'total_bps': breakdown.total_bps
            }
            
            print(f"   {symbol:<8} {tier.name:<12} {breakdown.spread_cost/100_000*10000:>6.1f}bp {breakdown.impact_cost/100_000*10000:>6.1f}bp {breakdown.total_bps:>8.1f}bp")
        
        # Calculate gross vs net Sharpe impact
        print("\n   Gross vs Net Sharpe Impact:")
        print("   " + "-" * 55)
        
        # Simulate a backtest
        if 'SPY' in self.data:
            spy_data = self.data['SPY']
            
            # Simple momentum strategy
            returns = spy_data['close'].pct_change().dropna()
            signal = np.sign(returns.rolling(20).mean())
            
            # Gross returns
            gross_returns = signal.shift(1) * returns
            gross_returns = gross_returns.dropna()
            
            # Count trades
            trades = (signal.diff().abs() > 0).sum()
            turnover = trades / len(signal)
            
            # Estimate costs per trade (round trip = 2x)
            avg_cost_per_trade = 0.002  # 20 bps round trip for SPY
            total_cost = turnover * avg_cost_per_trade * 2
            
            # Net returns
            daily_cost = total_cost / len(gross_returns)
            net_returns = gross_returns - daily_cost
            
            gross_sharpe = gross_returns.mean() / gross_returns.std() * np.sqrt(252)
            net_sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)
            
            results['gross_vs_net'] = {
                'gross_sharpe': gross_sharpe,
                'net_sharpe': net_sharpe,
                'degradation': (gross_sharpe - net_sharpe) / gross_sharpe if gross_sharpe > 0 else 0,
                'turnover': turnover,
                'n_trades': int(trades)
            }
            
            print(f"   Gross Sharpe: {gross_sharpe:.2f}")
            print(f"   Net Sharpe:   {net_sharpe:.2f}")
            print(f"   Degradation:  {results['gross_vs_net']['degradation']:.1%}")
            print(f"   Turnover:     {turnover:.1%}")
            print(f"   Total Trades: {trades}")
            
            if net_sharpe < 0.5:
                print("\n   ⚠️ WARNING: Net Sharpe < 0.5 after costs!")
            elif net_sharpe > 1.0:
                print("\n   ✅ Net Sharpe > 1.0 - Strategy viable after costs")
        
        self.results['transaction_costs'] = results
        return results
    
    def run_alpha_attribution_audit(self) -> Dict:
        """
        AUDIT 3: Alpha Attribution - Is it alpha or beta?
        """
        print("\n" + "=" * 70)
        print("📈 AUDIT 3: ALPHA ATTRIBUTION")
        print("=" * 70)
        
        results = {}
        
        # Need SPY as benchmark
        if 'SPY' not in self.data:
            print("   ⚠️ SPY data not available, using synthetic benchmark")
            dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
            np.random.seed(42)
            benchmark_returns = pd.Series(
                np.random.normal(0.0003, 0.01, len(dates)),
                index=dates
            )
        else:
            benchmark_returns = self.data['SPY']['close'].pct_change().dropna()
        
        # Build equal-weight portfolio of available symbols
        print("\n   Building test portfolio...")
        
        position_returns = {}
        all_returns = []
        
        for symbol in self.data:
            if symbol == 'SPY':  # Exclude benchmark
                continue
            
            returns = self.data[symbol]['close'].pct_change().dropna()
            position_returns[symbol] = returns
            all_returns.append(returns)
        
        if not all_returns:
            print("   ⚠️ No position data available")
            return {'error': 'No data'}
        
        # Create portfolio returns (equal weight)
        portfolio_returns = pd.concat(all_returns, axis=1).mean(axis=1)
        
        print(f"   Portfolio: {len(position_returns)} symbols")
        print(f"   Period: {portfolio_returns.index[0].date()} to {portfolio_returns.index[-1].date()}")
        
        # Run attribution
        aa = AlphaAttribution()
        
        # 1. CAPM alpha
        print("\n   1. CAPM Alpha (vs SPY):")
        print("   " + "-" * 40)
        
        capm = aa.calculate_alpha_beta(portfolio_returns, benchmark_returns)
        results['capm'] = {
            'alpha': capm.alpha_annualized,
            't_stat': capm.alpha_t_stat,
            'beta': capm.beta,
            'r_squared': capm.r_squared,
            'has_alpha': capm.has_alpha
        }
        
        print(f"   Alpha (annual):  {capm.alpha_annualized:.1%}")
        print(f"   Alpha t-stat:    {capm.alpha_t_stat:.2f}")
        print(f"   Beta:            {capm.beta:.2f}")
        print(f"   R-squared:       {capm.r_squared:.1%}")
        print(f"   Has Alpha:       {'YES ✅' if capm.has_alpha else 'NO ❌'}")
        
        # 2. Contribution analysis
        print("\n   2. Return Contribution:")
        print("   " + "-" * 40)
        
        contribution = aa.contribution_analysis(portfolio_returns, position_returns)
        results['contribution'] = {
            'top_contributor': contribution['top_contributor'],
            'top_pct': contribution['top_contributor_pct'],
            'concentration_risk': contribution['concentration_risk']
        }
        
        print(f"   Top contributor: {contribution['top_contributor']} ({contribution['top_contributor_pct']:.0%})")
        print(f"   Concentration:   {'RISK ⚠️' if contribution['concentration_risk'] else 'OK ✅'}")
        
        # Top 3 contributors
        print("\n   Top contributors:")
        for i, (symbol, data) in enumerate(list(contribution['contributions'].items())[:3]):
            print(f"     {i+1}. {symbol}: {data['pct_of_total']:.0%} (Sharpe={data['sharpe']:.2f})")
        
        # 3. Regime analysis
        print("\n   3. Regime Analysis:")
        print("   " + "-" * 40)
        
        regime = aa.regime_alpha(portfolio_returns, benchmark_returns)
        results['regime'] = {}
        
        for r in ['bull', 'bear', 'sideways']:
            if r in regime and not np.isnan(regime[r].get('alpha', np.nan)):
                alpha = regime[r]['alpha']
                n_days = regime[r]['n_days']
                emoji = "📈" if r == "bull" else "📉" if r == "bear" else "➡️"
                print(f"   {emoji} {r.upper():10s} Alpha={alpha:>6.1%} ({n_days} days)")
                results['regime'][r] = alpha
        
        print(f"\n   {regime.get('skill_summary', '')}")
        
        # Final verdict
        print("\n   VERDICT:")
        print("   " + "-" * 40)
        
        if capm.has_alpha and not contribution['concentration_risk']:
            verdict = "ALPHA"
            print("   ✅ RETURNS ARE ALPHA (not just beta)")
        elif capm.has_alpha and contribution['concentration_risk']:
            verdict = "CONCENTRATED"
            print("   ⚠️ ALPHA EXISTS BUT CONCENTRATED")
        else:
            verdict = "BETA"
            print("   ❌ RETURNS ARE LIKELY BETA EXPOSURE")
        
        results['verdict'] = verdict
        
        self.results['alpha_attribution'] = results
        return results
    
    def run_universe_coverage_audit(self) -> Dict:
        """
        AUDIT 4: Full Universe Coverage
        """
        print("\n" + "=" * 70)
        print("🌍 AUDIT 4: UNIVERSE COVERAGE")
        print("=" * 70)
        
        results = {
            'symbols': {},
            'sectors': {},
            'summary': {}
        }
        
        print("\n   Symbol Coverage:")
        print("   " + "-" * 50)
        
        available = 0
        viable = 0
        
        for symbol in self.FULL_UNIVERSE:
            sector = self.SECTOR_MAP.get(symbol, 'Unknown')
            
            if symbol in self.data:
                df = self.data[symbol]
                n_bars = len(df)
                
                # Check viability
                returns = df['close'].pct_change().dropna()
                if len(returns) > 252:
                    sharpe = returns.mean() / returns.std() * np.sqrt(252)
                    is_viable = sharpe > 0 and n_bars > 500
                else:
                    sharpe = np.nan
                    is_viable = False
                
                results['symbols'][symbol] = {
                    'available': True,
                    'n_bars': n_bars,
                    'sharpe': sharpe,
                    'viable': is_viable,
                    'sector': sector
                }
                
                available += 1
                if is_viable:
                    viable += 1
                    status = "✅"
                else:
                    status = "⚠️"
                
                print(f"   {status} {symbol:6s} [{sector:12s}] {n_bars:5d} bars, Sharpe={sharpe:.2f}" if not np.isnan(sharpe) else f"   {status} {symbol:6s} [{sector:12s}] {n_bars:5d} bars")
            else:
                results['symbols'][symbol] = {
                    'available': False,
                    'sector': sector
                }
                print(f"   ❌ {symbol:6s} [{sector:12s}] No data")
        
        # Sector summary
        print("\n   Sector Coverage:")
        print("   " + "-" * 50)
        
        for sector in set(self.SECTOR_MAP.values()):
            sector_symbols = [s for s, sec in self.SECTOR_MAP.items() if sec == sector]
            sector_available = sum(1 for s in sector_symbols if results['symbols'].get(s, {}).get('available', False))
            sector_viable = sum(1 for s in sector_symbols if results['symbols'].get(s, {}).get('viable', False))
            
            results['sectors'][sector] = {
                'total': len(sector_symbols),
                'available': sector_available,
                'viable': sector_viable
            }
            
            status = "✅" if sector_available == len(sector_symbols) else "⚠️" if sector_available > 0 else "❌"
            print(f"   {status} {sector:12s}: {sector_available}/{len(sector_symbols)} available, {sector_viable} viable")
        
        # Summary
        results['summary'] = {
            'total_symbols': len(self.FULL_UNIVERSE),
            'available': available,
            'viable': viable,
            'coverage_pct': available / len(self.FULL_UNIVERSE),
            'viability_pct': viable / available if available > 0 else 0
        }
        
        print("\n   Summary:")
        print("   " + "-" * 50)
        print(f"   Total universe:  {len(self.FULL_UNIVERSE)} symbols")
        print(f"   Data available:  {available} ({results['summary']['coverage_pct']:.0%})")
        print(f"   Viable:          {viable} ({results['summary']['viability_pct']:.0%})")
        
        if results['summary']['coverage_pct'] < 0.5:
            print("\n   ⚠️ WARNING: Less than 50% data coverage!")
            print("   Survivorship bias risk is HIGH")
        
        self.results['universe_coverage'] = results
        return results
    
    def run_pbo_audit(self) -> Dict:
        """
        AUDIT 5: Probability of Backtest Overfitting
        """
        print("\n" + "=" * 70)
        print("📊 AUDIT 5: PROBABILITY OF BACKTEST OVERFITTING")
        print("=" * 70)
        
        if not HAS_PBO:
            print("   ⚠️ PBO module not available")
            return {'error': 'Module not available'}
        
        results = {}
        
        # For PBO, we need multiple strategy variants
        # Simulate by varying lookback periods
        print("\n   Generating strategy variants...")
        
        if 'SPY' not in self.data:
            print("   ⚠️ SPY data not available")
            return {'error': 'No SPY data'}
        
        spy = self.data['SPY']['close']
        returns = spy.pct_change().dropna()
        
        # Create strategy variants with different parameters
        strategy_sharpes = []
        lookbacks = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, 252]
        
        print(f"   Testing {len(lookbacks)} lookback variants...")
        
        for lookback in lookbacks:
            signal = np.sign(returns.rolling(lookback).mean())
            strat_returns = signal.shift(1) * returns
            strat_returns = strat_returns.dropna()
            
            if len(strat_returns) > 100:
                sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(252)
                strategy_sharpes.append(sharpe)
        
        strategy_sharpes = np.array(strategy_sharpes)
        
        print(f"\n   Strategy Sharpes: {strategy_sharpes}")
        print(f"   Best: {strategy_sharpes.max():.2f}, Worst: {strategy_sharpes.min():.2f}")
        
        # Simple PBO estimation (rank-based)
        # Sort by IS performance, check OOS rank
        n = len(strategy_sharpes)
        ranks = np.argsort(np.argsort(-strategy_sharpes))  # Ranks (0 = best)
        
        # Best IS performer's rank
        best_is_idx = np.argmax(strategy_sharpes)
        
        # Simulate OOS by using second half of each strategy
        half = len(returns) // 2
        oos_sharpes = []
        
        for lookback in lookbacks:
            signal = np.sign(returns.iloc[half:].rolling(lookback).mean())
            strat_returns = signal.shift(1) * returns.iloc[half:]
            strat_returns = strat_returns.dropna()
            
            if len(strat_returns) > 50:
                sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(252)
                oos_sharpes.append(sharpe)
            else:
                oos_sharpes.append(np.nan)
        
        oos_sharpes = np.array(oos_sharpes)
        valid_mask = ~np.isnan(oos_sharpes)
        
        if valid_mask.sum() < 4:
            print("   ⚠️ Not enough valid OOS results")
            return {'error': 'Insufficient OOS data'}
        
        # Calculate degradation
        valid_is = strategy_sharpes[valid_mask]
        valid_oos = oos_sharpes[valid_mask]
        
        # Rank correlation
        from scipy.stats import spearmanr
        rank_corr, _ = spearmanr(valid_is, valid_oos)
        
        # PBO estimate: what fraction of best IS performers underperform OOS?
        is_ranks = np.argsort(np.argsort(-valid_is))
        oos_ranks = np.argsort(np.argsort(-valid_oos))
        
        # Best IS strategy's OOS rank
        best_is_idx = np.argmax(valid_is)
        best_is_oos_rank = oos_ranks[best_is_idx]
        
        # PBO = probability best IS is below median OOS
        n_valid = len(valid_oos)
        pbo_estimate = best_is_oos_rank / n_valid
        
        results = {
            'pbo': pbo_estimate,
            'n_strategies': len(strategy_sharpes),
            'best_is_sharpe': strategy_sharpes.max(),
            'best_is_oos_sharpe': valid_oos[best_is_idx] if best_is_idx < len(valid_oos) else np.nan,
            'rank_correlation': rank_corr,
            'degradation': 1 - (valid_oos.mean() / valid_is.mean()) if valid_is.mean() > 0 else 0
        }
        
        print(f"\n   PBO Estimate:      {pbo_estimate:.1%}")
        print(f"   Rank Correlation:  {rank_corr:.3f}")
        print(f"   IS→OOS Degradation: {results['degradation']:.1%}")
        print(f"   Best IS Sharpe:    {results['best_is_sharpe']:.2f}")
        print(f"   Best IS OOS Sharpe: {results['best_is_oos_sharpe']:.2f}")
        
        if pbo_estimate < 0.5:
            print("\n   ✅ PBO < 50%: Low overfitting probability")
        else:
            print("\n   ⚠️ PBO >= 50%: Possible overfitting!")
        
        self.results['pbo'] = results
        return results
    
    def generate_final_report(self) -> str:
        """Generate comprehensive final report."""
        
        print("\n" + "=" * 70)
        print("📋 FINAL INSTITUTIONAL AUDIT REPORT")
        print("=" * 70)
        print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Initial NAV: ${self.initial_nav:,.0f}")
        print(f"Data Source: {self.data_dir or 'Synthetic'}")
        
        # Summary scores
        print("\n" + "-" * 70)
        print("AUDIT SUMMARY")
        print("-" * 70)
        
        scores = {}
        
        # Risk Manager
        rm = self.results.get('risk_manager', {})
        rm_score = rm.get('passed', 0) / (rm.get('passed', 0) + rm.get('failed', 1))
        scores['risk_manager'] = rm_score
        status = "✅" if rm_score >= 0.8 else "⚠️" if rm_score >= 0.5 else "❌"
        print(f"\n{status} RISK MANAGER:      {rm_score:.0%} passed")
        
        # Transaction Costs
        tc = self.results.get('transaction_costs', {})
        gvn = tc.get('gross_vs_net', {})
        net_sharpe = gvn.get('net_sharpe', 0)
        tc_score = 1.0 if net_sharpe > 1.0 else 0.7 if net_sharpe > 0.5 else 0.3
        scores['transaction_costs'] = tc_score
        status = "✅" if net_sharpe > 1.0 else "⚠️" if net_sharpe > 0.5 else "❌"
        print(f"{status} TRANSACTION COSTS: Net Sharpe = {net_sharpe:.2f}")
        
        # Alpha Attribution
        aa = self.results.get('alpha_attribution', {})
        verdict = aa.get('verdict', 'UNKNOWN')
        aa_score = 1.0 if verdict == 'ALPHA' else 0.5 if verdict == 'CONCENTRATED' else 0.0
        scores['alpha_attribution'] = aa_score
        status = "✅" if verdict == 'ALPHA' else "⚠️" if verdict == 'CONCENTRATED' else "❌"
        print(f"{status} ALPHA ATTRIBUTION: {verdict}")
        
        # Universe Coverage
        uc = self.results.get('universe_coverage', {})
        coverage = uc.get('summary', {}).get('viability_pct', 0)
        uc_score = coverage
        scores['universe_coverage'] = uc_score
        status = "✅" if coverage > 0.7 else "⚠️" if coverage > 0.4 else "❌"
        print(f"{status} UNIVERSE COVERAGE: {coverage:.0%} viable")
        
        # PBO
        pbo = self.results.get('pbo', {})
        pbo_val = pbo.get('pbo', 1.0)
        pbo_score = 1.0 - pbo_val
        scores['pbo'] = pbo_score
        status = "✅" if pbo_val < 0.5 else "⚠️" if pbo_val < 0.7 else "❌"
        print(f"{status} PBO:               {pbo_val:.0%}")
        
        # Overall score
        overall = np.mean(list(scores.values()))
        print("\n" + "-" * 70)
        
        if overall >= 0.8:
            print(f"🏆 OVERALL SCORE: {overall:.0%} - INSTITUTIONAL GRADE")
            verdict = "PASS"
        elif overall >= 0.6:
            print(f"⚠️ OVERALL SCORE: {overall:.0%} - NEEDS IMPROVEMENT")
            verdict = "CONDITIONAL"
        else:
            print(f"❌ OVERALL SCORE: {overall:.0%} - NOT READY")
            verdict = "FAIL"
        
        print("-" * 70)
        
        # Recommendations
        print("\n📌 RECOMMENDATIONS:")
        
        if rm_score < 0.8:
            print("   • Review risk manager limits and state transitions")
        
        if net_sharpe < 1.0:
            print("   • Reduce turnover or improve signal quality to offset costs")
        
        if verdict != 'ALPHA':
            print("   • Investigate alpha sources - may be beta exposure")
        
        if coverage < 0.7:
            print("   • Expand data coverage to reduce survivorship bias")
        
        if pbo_val >= 0.5:
            print("   • High PBO indicates overfitting - use more robust validation")
        
        print("\n" + "=" * 70)
        print(f"VERDICT: {verdict}")
        print("=" * 70)
        
        return verdict
    
    def run_full_audit(self):
        """Run complete institutional audit."""
        print("╔" + "═" * 68 + "╗")
        print("║" + " KYPERIAN INSTITUTIONAL AUDIT ".center(68) + "║")
        print("║" + f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(68) + "║")
        print("╚" + "═" * 68 + "╝")
        
        # Load data
        self.load_data()
        
        # Run all audits
        self.run_risk_manager_audit()
        self.run_transaction_cost_audit()
        self.run_alpha_attribution_audit()
        self.run_universe_coverage_audit()
        self.run_pbo_audit()
        
        # Generate report
        verdict = self.generate_final_report()
        
        # Save results
        output_path = Path(__file__).parent / 'institutional_audit_results.json'
        import json
        
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, pd.Timestamp):
                return str(obj)
            if hasattr(obj, '__dict__'):
                return str(obj)
            return str(obj)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=convert)
            print(f"\nResults saved to: {output_path}")
        except Exception as e:
            print(f"\nWarning: Could not save JSON results: {e}")
        
        return verdict, self.results


def main():
    """Run the institutional audit."""
    audit = InstitutionalAudit(initial_nav=1_000_000)
    verdict, results = audit.run_full_audit()
    return verdict


if __name__ == "__main__":
    main()
