#!/usr/bin/env python3
"""
Dynamic Beta Hedge Module

Maintains market-neutral exposure by dynamically shorting SPY.
Updates hedge ratio as portfolio beta changes.

Based on validated audit results:
- Current Beta: 1.20
- Target Beta: ~0 (market-neutral)
- Hedge Ratio: -1.20 (short 120% of portfolio in SPY)

This module will:
1. Calculate rolling portfolio beta
2. Determine optimal hedge ratio
3. Execute hedge trades when threshold breached
4. Track hedge effectiveness
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import statsmodels, fall back to numpy if not available
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("statsmodels not available, using numpy for regression")


class HedgeFrequency(Enum):
    """Hedge rebalance frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class HedgeConfig:
    """Beta hedge configuration."""
    target_beta: float = 0.0           # Target portfolio beta (0 = market-neutral)
    rebalance_threshold: float = 0.10  # Rebalance if beta drifts 10%
    lookback_days: int = 60            # Days for beta calculation
    hedge_instrument: str = 'SPY'      # Instrument to short
    max_hedge_ratio: float = 2.0       # Maximum hedge ratio
    min_hedge_ratio: float = 0.0       # Minimum hedge ratio
    update_frequency: HedgeFrequency = HedgeFrequency.DAILY
    transaction_cost_bps: float = 5.0  # Transaction costs for hedging


@dataclass
class HedgeState:
    """Current hedge state."""
    current_beta: float
    hedge_ratio: float
    hedge_notional: float
    last_updated: datetime
    needs_rebalance: bool
    tracking_error: float = 0.0
    cumulative_cost: float = 0.0


@dataclass
class BetaStats:
    """Beta calculation statistics."""
    beta: float
    alpha: float  # Annualized
    r_squared: float
    std_error: float
    t_stat: float
    p_value: float
    n_observations: int


@dataclass
class HedgeEffectiveness:
    """Hedge effectiveness analysis."""
    unhedged_beta: float
    unhedged_sharpe: float
    unhedged_volatility: float
    hedged_beta: float
    hedged_sharpe: float
    hedged_volatility: float
    beta_reduction: float
    sharpe_improvement: float
    volatility_reduction: float
    tracking_error: float


class DynamicBetaHedge:
    """
    Dynamic beta hedging system.
    
    Maintains market-neutral exposure by:
    1. Calculating rolling portfolio beta
    2. Determining optimal hedge ratio
    3. Executing hedge trades when threshold breached
    4. Tracking hedge effectiveness
    
    Example:
        hedger = DynamicBetaHedge(HedgeConfig(target_beta=0.0))
        
        # Update hedge
        result = hedger.update_hedge(
            portfolio_returns=my_returns,
            benchmark_returns=spy_returns,
            portfolio_value=100000
        )
        
        if result['action'] == 'REBALANCE':
            # Execute hedge trade
            trade = result['trade']
            print(f"Short ${trade['notional']:,.0f} of {trade['instrument']}")
    """
    
    def __init__(self, config: HedgeConfig = None):
        self.config = config or HedgeConfig()
        self.state: Optional[HedgeState] = None
        self.history: List[HedgeState] = []
        self.trades: List[Dict] = []
        
    def calculate_portfolio_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        lookback: int = None
    ) -> BetaStats:
        """
        Calculate portfolio beta using regression.
        
        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark (SPY) daily returns
            lookback: Number of days to use (default from config)
            
        Returns:
            BetaStats with beta, alpha, R², and statistics
        """
        lookback = lookback or self.config.lookback_days
        
        # Align and trim to lookback
        aligned = pd.concat(
            [portfolio_returns, benchmark_returns], 
            axis=1, 
            keys=['portfolio', 'benchmark']
        ).dropna()
        aligned = aligned.tail(lookback)
        
        if len(aligned) < 20:
            raise ValueError(f"Insufficient data: {len(aligned)} days (need at least 20)")
        
        port_ret = aligned['portfolio'].values
        bench_ret = aligned['benchmark'].values
        
        if HAS_STATSMODELS:
            # OLS regression with statsmodels
            X = sm.add_constant(bench_ret)
            model = sm.OLS(port_ret, X).fit()
            
            return BetaStats(
                beta=float(model.params[1]),
                alpha=float(model.params[0] * 252),  # Annualized
                r_squared=float(model.rsquared),
                std_error=float(model.bse[1]),
                t_stat=float(model.tvalues[1]),
                p_value=float(model.pvalues[1]),
                n_observations=len(aligned)
            )
        else:
            # Numpy fallback
            cov = np.cov(port_ret, bench_ret)
            beta = cov[0, 1] / cov[1, 1]
            alpha = np.mean(port_ret) - beta * np.mean(bench_ret)
            
            # R-squared
            predicted = alpha + beta * bench_ret
            ss_res = np.sum((port_ret - predicted) ** 2)
            ss_tot = np.sum((port_ret - np.mean(port_ret)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Approximate t-stat
            n = len(port_ret)
            se = np.sqrt(ss_res / (n - 2)) / np.sqrt(np.sum((bench_ret - np.mean(bench_ret)) ** 2))
            t_stat = beta / se if se > 0 else 0
            
            return BetaStats(
                beta=float(beta),
                alpha=float(alpha * 252),
                r_squared=float(r_squared),
                std_error=float(se),
                t_stat=float(t_stat),
                p_value=0.0,  # Would need scipy for this
                n_observations=len(aligned)
            )
    
    def calculate_hedge_ratio(
        self,
        current_beta: float,
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calculate optimal hedge ratio to achieve target beta.
        
        Hedge ratio = (current_beta - target_beta)
        
        Example:
        - Current beta: 1.20
        - Target beta: 0.00
        - Hedge ratio: 1.20 (short 120% of portfolio in SPY)
        
        Args:
            current_beta: Current portfolio beta
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary with hedge details
        """
        target = self.config.target_beta
        
        # Raw hedge ratio
        raw_ratio = current_beta - target
        
        # Clamp to limits
        clamped_ratio = np.clip(
            raw_ratio,
            self.config.min_hedge_ratio,
            self.config.max_hedge_ratio
        )
        
        # Calculate notional
        hedge_notional = portfolio_value * clamped_ratio
        
        return {
            'raw_ratio': raw_ratio,
            'clamped_ratio': clamped_ratio,
            'hedge_notional': hedge_notional,
            'hedge_instrument': self.config.hedge_instrument,
            'direction': 'SHORT' if clamped_ratio > 0 else 'LONG',
            'was_clamped': raw_ratio != clamped_ratio
        }
    
    def should_rebalance(self, current_beta: float) -> Tuple[bool, str]:
        """
        Check if hedge needs rebalancing.
        
        Returns:
            Tuple of (should_rebalance, reason)
        """
        if self.state is None:
            return True, "Initial hedge setup"
        
        # Check beta drift
        beta_drift = abs(current_beta - self.state.current_beta)
        drift_pct = beta_drift / max(abs(self.state.current_beta), 0.01)
        
        if drift_pct > self.config.rebalance_threshold:
            return True, f"Beta drifted {drift_pct:.1%} (threshold: {self.config.rebalance_threshold:.1%})"
        
        # Check time since last rebalance
        days_since = (datetime.now() - self.state.last_updated).days
        
        freq = self.config.update_frequency
        if freq == HedgeFrequency.DAILY and days_since >= 1:
            return True, f"Daily rebalance due ({days_since} days)"
        elif freq == HedgeFrequency.WEEKLY and days_since >= 7:
            return True, f"Weekly rebalance due ({days_since} days)"
        elif freq == HedgeFrequency.MONTHLY and days_since >= 30:
            return True, f"Monthly rebalance due ({days_since} days)"
        
        return False, "No rebalance needed"
    
    def update_hedge(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_value: float
    ) -> Dict:
        """
        Update hedge position.
        
        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary with hedge trade instructions
        """
        # Calculate current beta
        beta_stats = self.calculate_portfolio_beta(
            portfolio_returns, benchmark_returns
        )
        current_beta = beta_stats.beta
        
        # Check if rebalance needed
        should_rebal, reason = self.should_rebalance(current_beta)
        
        if not should_rebal:
            return {
                'action': 'HOLD',
                'reason': reason,
                'current_beta': current_beta,
                'current_state': self.state
            }
        
        # Calculate new hedge
        hedge = self.calculate_hedge_ratio(current_beta, portfolio_value)
        
        # Determine trade
        if self.state is None:
            trade_notional = hedge['hedge_notional']
            prev_notional = 0
        else:
            trade_notional = hedge['hedge_notional'] - self.state.hedge_notional
            prev_notional = self.state.hedge_notional
        
        # Calculate transaction cost
        transaction_cost = abs(trade_notional) * (self.config.transaction_cost_bps / 10000)
        cumulative_cost = (self.state.cumulative_cost if self.state else 0) + transaction_cost
        
        # Update state
        self.state = HedgeState(
            current_beta=current_beta,
            hedge_ratio=hedge['clamped_ratio'],
            hedge_notional=hedge['hedge_notional'],
            last_updated=datetime.now(),
            needs_rebalance=False,
            tracking_error=0.0,
            cumulative_cost=cumulative_cost
        )
        self.history.append(self.state)
        
        # Record trade
        trade_record = {
            'timestamp': datetime.now(),
            'instrument': self.config.hedge_instrument,
            'direction': 'SHORT' if trade_notional > 0 else 'COVER',
            'notional': abs(trade_notional),
            'total_hedge_notional': hedge['hedge_notional'],
            'transaction_cost': transaction_cost,
            'beta_before': self.history[-2].current_beta if len(self.history) > 1 else current_beta,
            'beta_after': current_beta
        }
        self.trades.append(trade_record)
        
        return {
            'action': 'REBALANCE',
            'reason': reason,
            'current_beta': current_beta,
            'target_beta': self.config.target_beta,
            'hedge_ratio': hedge['clamped_ratio'],
            'trade': {
                'instrument': self.config.hedge_instrument,
                'direction': 'SHORT' if trade_notional > 0 else 'COVER',
                'notional': abs(trade_notional),
                'total_hedge_notional': hedge['hedge_notional'],
                'previous_notional': prev_notional,
                'transaction_cost': transaction_cost
            },
            'beta_stats': {
                'beta': beta_stats.beta,
                'alpha': beta_stats.alpha,
                'r_squared': beta_stats.r_squared,
                't_stat': beta_stats.t_stat
            }
        }
    
    def calculate_hedged_returns(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        hedge_ratio: float = None
    ) -> pd.Series:
        """
        Calculate returns of hedged portfolio.
        
        Hedged return = Portfolio return - (hedge_ratio × Benchmark return)
        
        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns
            hedge_ratio: Override hedge ratio (default: current state)
            
        Returns:
            Series of hedged returns
        """
        if hedge_ratio is None:
            if self.state is None:
                raise ValueError("No hedge state and no hedge_ratio provided")
            hedge_ratio = self.state.hedge_ratio
        
        aligned = pd.concat(
            [portfolio_returns, benchmark_returns], 
            axis=1,
            keys=['portfolio', 'benchmark']
        ).dropna()
        
        hedged_returns = aligned['portfolio'] - hedge_ratio * aligned['benchmark']
        
        return hedged_returns
    
    def analyze_hedge_effectiveness(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> HedgeEffectiveness:
        """
        Analyze how effective the hedge has been.
        
        Compares:
        - Unhedged vs hedged beta
        - Unhedged vs hedged Sharpe ratio
        - Unhedged vs hedged volatility
        
        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns
            
        Returns:
            HedgeEffectiveness with comparison metrics
        """
        if self.state is None:
            raise ValueError("No hedge state to analyze")
        
        # Unhedged stats
        unhedged_beta_stats = self.calculate_portfolio_beta(
            portfolio_returns, benchmark_returns
        )
        unhedged_sharpe = (
            portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            if portfolio_returns.std() > 0 else 0
        )
        unhedged_vol = portfolio_returns.std() * np.sqrt(252)
        
        # Hedged stats
        hedged_returns = self.calculate_hedged_returns(
            portfolio_returns,
            benchmark_returns,
            self.state.hedge_ratio
        )
        
        hedged_beta_stats = self.calculate_portfolio_beta(
            hedged_returns, benchmark_returns
        )
        hedged_sharpe = (
            hedged_returns.mean() / hedged_returns.std() * np.sqrt(252)
            if hedged_returns.std() > 0 else 0
        )
        hedged_vol = hedged_returns.std() * np.sqrt(252)
        
        # Tracking error (std of hedged returns)
        tracking_error = hedged_returns.std() * np.sqrt(252)
        
        return HedgeEffectiveness(
            unhedged_beta=unhedged_beta_stats.beta,
            unhedged_sharpe=unhedged_sharpe,
            unhedged_volatility=unhedged_vol,
            hedged_beta=hedged_beta_stats.beta,
            hedged_sharpe=hedged_sharpe,
            hedged_volatility=hedged_vol,
            beta_reduction=unhedged_beta_stats.beta - hedged_beta_stats.beta,
            sharpe_improvement=hedged_sharpe - unhedged_sharpe,
            volatility_reduction=unhedged_vol - hedged_vol,
            tracking_error=tracking_error
        )
    
    def get_hedge_summary(self) -> Dict:
        """Get summary of current hedge state."""
        if self.state is None:
            return {'status': 'NO_HEDGE'}
        
        return {
            'status': 'ACTIVE',
            'current_beta': self.state.current_beta,
            'target_beta': self.config.target_beta,
            'hedge_ratio': self.state.hedge_ratio,
            'hedge_notional': self.state.hedge_notional,
            'hedge_instrument': self.config.hedge_instrument,
            'last_updated': self.state.last_updated.isoformat(),
            'cumulative_cost': self.state.cumulative_cost,
            'n_rebalances': len(self.trades),
            'history_length': len(self.history)
        }


class MultiAssetBetaHedge(DynamicBetaHedge):
    """
    Beta hedge for multi-asset portfolio.
    
    Calculates portfolio-weighted beta and hedges accordingly.
    Useful when portfolio contains multiple assets with different betas.
    """
    
    def __init__(self, config: HedgeConfig = None):
        super().__init__(config)
        self.asset_betas: Dict[str, float] = {}
    
    def calculate_portfolio_beta_weighted(
        self,
        asset_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        weights: Dict[str, float]
    ) -> Dict:
        """
        Calculate portfolio beta from individual asset betas.
        
        Portfolio beta = Σ(weight_i × beta_i)
        
        Args:
            asset_returns: DataFrame with column per asset
            benchmark_returns: Benchmark returns
            weights: Dictionary of asset weights
            
        Returns:
            Dictionary with portfolio beta and asset betas
        """
        asset_betas = {}
        
        for asset in asset_returns.columns:
            if asset in weights and weights[asset] != 0:
                aligned = pd.concat([
                    asset_returns[asset],
                    benchmark_returns
                ], axis=1).dropna()
                
                if len(aligned) >= 20:
                    try:
                        beta_stats = self.calculate_portfolio_beta(
                            aligned.iloc[:, 0],
                            aligned.iloc[:, 1]
                        )
                        asset_betas[asset] = beta_stats.beta
                    except Exception:
                        asset_betas[asset] = 1.0  # Default
                else:
                    asset_betas[asset] = 1.0  # Default
        
        # Weighted average
        portfolio_beta = sum(
            weights.get(asset, 0) * beta
            for asset, beta in asset_betas.items()
        )
        
        self.asset_betas = asset_betas
        
        return {
            'portfolio_beta': portfolio_beta,
            'asset_betas': asset_betas,
            'weights': weights,
            'weighted_contributions': {
                asset: weights.get(asset, 0) * beta
                for asset, beta in asset_betas.items()
            }
        }
    
    def identify_high_beta_positions(
        self,
        threshold: float = 1.5
    ) -> List[Tuple[str, float]]:
        """
        Identify positions with unusually high beta.
        
        Args:
            threshold: Beta threshold (default 1.5)
            
        Returns:
            List of (asset, beta) tuples for high-beta assets
        """
        return [
            (asset, beta) 
            for asset, beta in self.asset_betas.items()
            if beta > threshold
        ]


def run_beta_hedge_test():
    """
    Test the beta hedge module with real data.
    """
    print("="*60)
    print("BETA HEDGE MODULE TEST")
    print("="*60)
    
    # Check for real data
    from pathlib import Path
    data_dir = Path("/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI/data/test")
    
    spy_path = data_dir / "SPY.csv"
    aapl_path = data_dir / "AAPL.csv"
    
    if not spy_path.exists() or not aapl_path.exists():
        print("❌ Real data not found, using synthetic data")
        
        # Generate synthetic data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=500, freq='D')
        
        # SPY returns (market)
        spy_returns = pd.Series(
            np.random.normal(0.0004, 0.01, 500),
            index=dates,
            name='SPY'
        )
        
        # AAPL returns (beta ~ 1.2)
        noise = np.random.normal(0, 0.005, 500)
        aapl_returns = pd.Series(
            1.2 * spy_returns.values + noise + 0.0001,  # Beta=1.2, small alpha
            index=dates,
            name='AAPL'
        )
    else:
        print("✅ Using real data from data/test/")
        
        # Load real data
        spy_df = pd.read_csv(spy_path)
        aapl_df = pd.read_csv(aapl_path)
        
        # Calculate returns
        spy_returns = spy_df['close'].pct_change().dropna()
        spy_returns.index = range(len(spy_returns))
        
        aapl_returns = aapl_df['close'].pct_change().dropna()
        aapl_returns.index = range(len(aapl_returns))
        
        # Align
        min_len = min(len(spy_returns), len(aapl_returns))
        spy_returns = spy_returns.tail(min_len)
        aapl_returns = aapl_returns.tail(min_len)
    
    print(f"\nData: {len(spy_returns)} days of returns")
    
    # Initialize hedger
    config = HedgeConfig(
        target_beta=0.0,
        rebalance_threshold=0.10,
        lookback_days=60
    )
    hedger = DynamicBetaHedge(config)
    
    # 1. Calculate current beta
    print("\n" + "-"*40)
    print("1. BETA CALCULATION")
    print("-"*40)
    
    beta_stats = hedger.calculate_portfolio_beta(aapl_returns, spy_returns)
    print(f"   Beta: {beta_stats.beta:.3f}")
    print(f"   Alpha (ann.): {beta_stats.alpha:.2%}")
    print(f"   R-squared: {beta_stats.r_squared:.3f}")
    print(f"   T-statistic: {beta_stats.t_stat:.2f}")
    print(f"   Observations: {beta_stats.n_observations}")
    
    # 2. Calculate hedge
    print("\n" + "-"*40)
    print("2. HEDGE CALCULATION")
    print("-"*40)
    
    portfolio_value = 100000
    result = hedger.update_hedge(aapl_returns, spy_returns, portfolio_value)
    
    print(f"   Action: {result['action']}")
    print(f"   Reason: {result['reason']}")
    print(f"   Current Beta: {result['current_beta']:.3f}")
    print(f"   Target Beta: {result['target_beta']:.3f}")
    print(f"   Hedge Ratio: {result['hedge_ratio']:.3f}")
    
    if 'trade' in result:
        trade = result['trade']
        print(f"\n   📊 TRADE:")
        print(f"      Instrument: {trade['instrument']}")
        print(f"      Direction: {trade['direction']}")
        print(f"      Notional: ${trade['notional']:,.0f}")
        print(f"      Transaction Cost: ${trade['transaction_cost']:.2f}")
    
    # 3. Analyze effectiveness
    print("\n" + "-"*40)
    print("3. HEDGE EFFECTIVENESS")
    print("-"*40)
    
    effectiveness = hedger.analyze_hedge_effectiveness(aapl_returns, spy_returns)
    
    print(f"\n   UNHEDGED:")
    print(f"      Beta: {effectiveness.unhedged_beta:.3f}")
    print(f"      Sharpe: {effectiveness.unhedged_sharpe:.3f}")
    print(f"      Volatility: {effectiveness.unhedged_volatility:.2%}")
    
    print(f"\n   HEDGED:")
    print(f"      Beta: {effectiveness.hedged_beta:.3f}")
    print(f"      Sharpe: {effectiveness.hedged_sharpe:.3f}")
    print(f"      Volatility: {effectiveness.hedged_volatility:.2%}")
    
    print(f"\n   IMPROVEMENT:")
    print(f"      Beta Reduction: {effectiveness.beta_reduction:.3f}")
    print(f"      Sharpe Change: {effectiveness.sharpe_improvement:+.3f}")
    print(f"      Vol Reduction: {effectiveness.volatility_reduction:.2%}")
    print(f"      Tracking Error: {effectiveness.tracking_error:.2%}")
    
    # 4. Summary
    print("\n" + "-"*40)
    print("4. HEDGE SUMMARY")
    print("-"*40)
    
    summary = hedger.get_hedge_summary()
    print(f"   Status: {summary['status']}")
    print(f"   Current Beta: {summary['current_beta']:.3f}")
    print(f"   Target Beta: {summary['target_beta']:.3f}")
    print(f"   Hedge Notional: ${summary['hedge_notional']:,.0f}")
    print(f"   Instrument: {summary['hedge_instrument']}")
    
    # Verdict
    print("\n" + "="*60)
    if effectiveness.hedged_beta < 0.3:
        print("✅ BETA HEDGE MODULE: WORKING")
        print(f"   Successfully reduced beta from {effectiveness.unhedged_beta:.2f} to {effectiveness.hedged_beta:.2f}")
    else:
        print("⚠️ BETA HEDGE MODULE: NEEDS TUNING")
        print(f"   Beta still {effectiveness.hedged_beta:.2f} (target < 0.3)")
    print("="*60)
    
    return hedger, effectiveness


if __name__ == "__main__":
    run_beta_hedge_test()
