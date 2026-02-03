#!/usr/bin/env python3
"""
NUBLE Production Configuration

Based on validated audit results (February 1, 2026):
- Alpha: 13.8% (t=3.21)
- PBO: 25% (low overfitting)
- Sharpe: 0.41 (modest but real)
- Beta: 1.20 (needs hedging)

This configuration is designed for PAPER TRADING FIRST.
Do not use with real money until paper trading validates.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum


class DeploymentPhase(Enum):
    """Deployment phases."""
    PAPER = "paper"           # Paper trading only
    SMALL_LIVE = "small_live" # 10% of intended capital
    FULL_LIVE = "full_live"   # Full deployment


@dataclass
class RiskConfig:
    """Risk management configuration based on audit results."""
    
    # Position limits (validated in audit)
    max_position_pct: float = 0.10          # 10% max single position
    max_sector_pct: float = 0.30            # 30% max sector exposure
    max_correlated_pct: float = 0.40        # 40% max correlated positions
    
    # Exposure limits
    max_gross_exposure: float = 1.5         # 150% gross
    max_net_exposure: float = 0.5           # 50% net long (after beta hedge)
    
    # Drawdown limits (trigger state transitions)
    drawdown_warning: float = 0.05          # 5% DD → reduce new positions
    drawdown_reduced: float = 0.10          # 10% DD → 50% risk budget
    drawdown_minimal: float = 0.15          # 15% DD → 25% risk budget  
    drawdown_halt: float = 0.20             # 20% DD → HALT all trading
    
    # Daily limits
    max_daily_trades: int = 50              # Max trades per day
    max_daily_turnover: float = 0.30        # 30% of portfolio max
    max_daily_loss: float = 0.02            # 2% daily loss limit
    
    # Volatility targeting (based on audit Sharpe 0.41)
    target_volatility: float = 0.15         # 15% annualized vol
    max_volatility: float = 0.25            # 25% max before scaling


@dataclass
class BetaHedgeConfig:
    """Beta hedging configuration."""
    
    # Validated beta from audit
    target_beta: float = 0.0                # Target market-neutral
    current_beta: float = 1.20              # Measured beta
    
    # Hedge instrument
    hedge_symbol: str = "SPY"               # Use SPY for hedging
    
    # Rebalance frequency
    rebalance_days: int = 5                 # Weekly rebalance
    
    # Hedge ratio = -current_beta to achieve 0 beta
    @property
    def hedge_ratio(self) -> float:
        return -(self.current_beta - self.target_beta)
    
    def calculate_hedge(self, portfolio_value: float) -> float:
        """Calculate SPY position needed to hedge beta."""
        return portfolio_value * self.hedge_ratio


@dataclass
class TransactionCostConfig:
    """Transaction cost parameters based on audit."""
    
    # Validated costs by asset
    cost_by_tier: Dict[str, float] = field(default_factory=lambda: {
        'mega_cap': 0.0048,   # 48 bps (AAPL, MSFT, NVDA)
        'large_cap': 0.0075,  # 75 bps
        'mid_cap': 0.0120,    # 120 bps
        'small_cap': 0.0200,  # 200 bps
        'etf': 0.0030,        # 30 bps (SPY, QQQ)
    })
    
    # Slippage assumption
    slippage_bps: float = 10.0              # 10 bps slippage
    
    # Minimum trade size (to avoid commission dominance)
    min_trade_size: float = 1000.0          # $1000 minimum


@dataclass
class UniverseConfig:
    """Trading universe based on audit viability."""
    
    # Viable symbols from audit (100% pass rate)
    viable_symbols: List[str] = field(default_factory=lambda: [
        # Technology (primary alpha source)
        'AAPL', 'MSFT', 'NVDA', 'AMD', 'GOOGL',
        
        # Financials
        'JPM', 'BAC',
        
        # Consumer
        'AMZN', 'TSLA',
        
        # ETFs (for hedging and diversification)
        'SPY', 'QQQ', 'IWM',
        
        # Commodities
        'GLD', 'SLV',
        
        # Bonds
        'TLT', 'IEF'
    ])
    
    # Top contributors (monitor closely)
    watch_list: List[str] = field(default_factory=lambda: [
        'NVDA',  # 18% of returns - largest contributor
        'AMD',   # High volatility
        'TSLA',  # High volatility
    ])
    
    # Sector allocation targets
    sector_targets: Dict[str, float] = field(default_factory=lambda: {
        'Technology': 0.40,    # 40% max
        'Financials': 0.20,    # 20% max
        'Consumer': 0.15,      # 15% max
        'ETF': 0.15,           # 15% for hedging
        'Commodities': 0.10,   # 10% max
    })


@dataclass
class SignalConfig:
    """Signal generation configuration."""
    
    # Lookback periods (optimized during validation)
    momentum_lookback: int = 20             # 20-day momentum
    volatility_lookback: int = 20           # 20-day volatility
    
    # Signal thresholds
    entry_threshold: float = 0.5            # Signal > 0.5 to enter
    exit_threshold: float = 0.2             # Signal < 0.2 to exit
    
    # Position sizing
    use_kelly: bool = False                 # Kelly sizing (aggressive)
    use_equal_weight: bool = True           # Start with equal weight
    
    # Rebalance frequency
    rebalance_frequency: str = "daily"      # daily, weekly, monthly


@dataclass
class ProductionConfig:
    """Master production configuration."""
    
    # Deployment phase
    phase: DeploymentPhase = DeploymentPhase.PAPER
    
    # Capital allocation by phase
    capital_by_phase: Dict[str, float] = field(default_factory=lambda: {
        DeploymentPhase.PAPER.value: 100000,      # $100K paper
        DeploymentPhase.SMALL_LIVE.value: 10000,  # $10K live (10%)
        DeploymentPhase.FULL_LIVE.value: 100000,  # $100K full
    })
    
    # Component configs
    risk: RiskConfig = field(default_factory=RiskConfig)
    beta_hedge: BetaHedgeConfig = field(default_factory=BetaHedgeConfig)
    costs: TransactionCostConfig = field(default_factory=TransactionCostConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_alerts: bool = True
    alert_email: str = ""                   # Set your email
    
    # API keys (loaded from environment)
    polygon_api_key: str = ""               # Set from env
    broker_api_key: str = ""                # Set from env
    
    @property
    def capital(self) -> float:
        """Current capital based on phase."""
        return self.capital_by_phase[self.phase.value]
    
    def validate(self) -> bool:
        """Validate configuration."""
        errors = []
        
        if self.risk.max_position_pct > 0.15:
            errors.append("Position limit > 15% is too aggressive")
        
        if self.risk.max_daily_loss > 0.03:
            errors.append("Daily loss limit > 3% is too aggressive")
        
        if not self.universe.viable_symbols:
            errors.append("No viable symbols configured")
        
        if errors:
            print("Configuration errors:")
            for e in errors:
                print(f"  - {e}")
            return False
        
        return True


# Default configuration for paper trading
DEFAULT_CONFIG = ProductionConfig(
    phase=DeploymentPhase.PAPER,
)


def get_config(phase: str = "paper") -> ProductionConfig:
    """Get configuration for specified phase."""
    config = ProductionConfig()
    
    if phase == "paper":
        config.phase = DeploymentPhase.PAPER
    elif phase == "small_live":
        config.phase = DeploymentPhase.SMALL_LIVE
    elif phase == "full_live":
        config.phase = DeploymentPhase.FULL_LIVE
    else:
        raise ValueError(f"Unknown phase: {phase}")
    
    return config


if __name__ == "__main__":
    print("="*60)
    print("NUBLE Production Configuration")
    print("="*60)
    
    config = get_config("paper")
    
    print(f"\nPhase: {config.phase.value}")
    print(f"Capital: ${config.capital:,.0f}")
    
    print(f"\nRisk Limits:")
    print(f"  Max Position: {config.risk.max_position_pct:.0%}")
    print(f"  Max Sector: {config.risk.max_sector_pct:.0%}")
    print(f"  Daily Loss Limit: {config.risk.max_daily_loss:.0%}")
    print(f"  Drawdown Halt: {config.risk.drawdown_halt:.0%}")
    
    print(f"\nBeta Hedge:")
    print(f"  Current Beta: {config.beta_hedge.current_beta:.2f}")
    print(f"  Target Beta: {config.beta_hedge.target_beta:.2f}")
    print(f"  Hedge Ratio: {config.beta_hedge.hedge_ratio:.2f}")
    print(f"  For ${config.capital:,.0f} long: Short ${config.beta_hedge.calculate_hedge(config.capital):,.0f} SPY")
    
    print(f"\nUniverse: {len(config.universe.viable_symbols)} symbols")
    print(f"  {', '.join(config.universe.viable_symbols[:5])}...")
    
    print(f"\nValidation: {'✅ PASSED' if config.validate() else '❌ FAILED'}")
    print("="*60)
