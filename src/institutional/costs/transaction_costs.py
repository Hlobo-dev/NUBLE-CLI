"""
Realistic Transaction Cost Model

Components:
1. Spread cost (bid-ask)
2. Market impact (Almgren-Chriss square root model)
3. Slippage (execution vs decision price)
4. Borrow costs (for shorts)

Reference:
- Almgren, R., & Chriss, N. (2001). "Optimal execution of portfolio transactions."
- Kissell, R., & Glantz, M. (2003). "Optimal Trading Strategies."

This is what separates "gross Sharpe" from "net Sharpe" - the real test.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum


class LiquidityTier(Enum):
    """Liquidity classification for spread estimation."""
    MEGA_CAP = "mega_cap"       # AAPL, MSFT, GOOGL, AMZN, etc.
    LARGE_CAP = "large_cap"     # Large but not mega
    MID_CAP = "mid_cap"         # Mid-cap stocks
    SMALL_CAP = "small_cap"     # Small caps
    ETF_LIQUID = "etf_liquid"   # SPY, QQQ, IWM
    ETF_OTHER = "etf_other"     # Sector and specialty ETFs
    CRYPTO_MAJOR = "crypto_major"  # BTC, ETH
    CRYPTO_ALT = "crypto_alt"   # Other cryptos


class BorrowTier(Enum):
    """Stock borrow difficulty classification."""
    EASY_TO_BORROW = "easy"     # General collateral
    GENERAL = "general"         # Normal borrow
    HARD_TO_BORROW = "htb"      # Hard to borrow
    VERY_HARD = "very_hard"     # Very hard to borrow
    CRYPTO = "crypto"           # Crypto funding rate


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a trade."""
    symbol: str
    trade_value: float
    
    # Individual cost components
    spread_cost: float
    impact_cost: float
    slippage_cost: float
    borrow_cost: float
    
    # Totals
    total_cost: float
    total_bps: float
    cost_pct: float
    
    # Metadata
    liquidity_tier: str
    borrow_tier: str
    participation_rate: float
    
    def __str__(self) -> str:
        return f"""
Cost Breakdown for {self.symbol}:
================================
Trade Value: ${self.trade_value:,.2f}

Spread:    ${self.spread_cost:,.2f} ({self.spread_cost/self.trade_value*10000:.1f} bps)
Impact:    ${self.impact_cost:,.2f} ({self.impact_cost/self.trade_value*10000:.1f} bps)
Slippage:  ${self.slippage_cost:,.2f} ({self.slippage_cost/self.trade_value*10000:.1f} bps)
Borrow:    ${self.borrow_cost:,.2f} ({self.borrow_cost/self.trade_value*10000:.1f} bps)
-----------------------------------
TOTAL:     ${self.total_cost:,.2f} ({self.total_bps:.1f} bps)

Participation Rate: {self.participation_rate*100:.2f}%
"""


@dataclass
class TransactionCostConfig:
    """Configuration for transaction cost model."""
    
    # Spread models by liquidity tier (half-spread in decimal)
    spread_by_tier: Dict[str, float] = field(default_factory=lambda: {
        LiquidityTier.MEGA_CAP.value: 0.00005,     # 0.5 bps (1 bp spread)
        LiquidityTier.LARGE_CAP.value: 0.00015,    # 1.5 bps
        LiquidityTier.MID_CAP.value: 0.0004,       # 4 bps
        LiquidityTier.SMALL_CAP.value: 0.001,      # 10 bps
        LiquidityTier.ETF_LIQUID.value: 0.00005,   # 0.5 bps
        LiquidityTier.ETF_OTHER.value: 0.00025,    # 2.5 bps
        LiquidityTier.CRYPTO_MAJOR.value: 0.0005,  # 5 bps
        LiquidityTier.CRYPTO_ALT.value: 0.001,     # 10 bps
    })
    
    # Almgren-Chriss impact parameters
    impact_eta: float = 0.3           # Temporary impact coefficient
    impact_gamma: float = 0.5         # Permanent impact coefficient
    
    # Slippage parameters
    slippage_delay_days: float = 1.0  # Delay from signal to execution
    
    # Borrow rates (annualized)
    borrow_rates: Dict[str, float] = field(default_factory=lambda: {
        BorrowTier.EASY_TO_BORROW.value: 0.005,   # 0.5%
        BorrowTier.GENERAL.value: 0.02,           # 2%
        BorrowTier.HARD_TO_BORROW.value: 0.10,    # 10%
        BorrowTier.VERY_HARD.value: 0.50,         # 50%
        BorrowTier.CRYPTO.value: 0.08,            # 8% (funding rate proxy)
    })


class TransactionCostModel:
    """
    Institutional transaction cost model.
    
    Total cost = spread + impact + slippage + borrow
    
    This model is critical for converting "gross Sharpe" to "net Sharpe"
    which is what actually matters for deployment.
    """
    
    # Known mega-cap symbols
    MEGA_CAP = {
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
        'BRK.A', 'BRK.B', 'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD'
    }
    
    # Known liquid ETFs
    LIQUID_ETFS = {
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'EEM', 'EFA',
        'GLD', 'TLT', 'XLF', 'XLK', 'XLE', 'XLV'
    }
    
    # Easy to borrow symbols
    EASY_BORROW = {
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'JPM', 'BAC',
        'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT'
    }
    
    # Hard to borrow (meme stocks, heavily shorted)
    HARD_BORROW = {
        'GME', 'AMC', 'BBBY', 'KOSS', 'BB', 'NOK', 'CLOV'
    }
    
    def __init__(self, config: TransactionCostConfig = None):
        self.config = config or TransactionCostConfig()
    
    def calculate_cost(
        self,
        symbol: str,
        trade_value: float,
        daily_volume_usd: float,
        volatility: float,
        side: str = 'BUY',
        is_short: bool = False,
        holding_period_days: int = 10,
        liquidity_tier: Optional[str] = None
    ) -> CostBreakdown:
        """
        Calculate total transaction cost.
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol
        trade_value : float
            Dollar value of trade
        daily_volume_usd : float
            Average daily volume in USD
        volatility : float
            Daily volatility (std of returns)
        side : str
            'BUY' or 'SELL'
        is_short : bool
            True if short sale
        holding_period_days : int
            Expected holding period for borrow cost
        liquidity_tier : str, optional
            Override automatic tier detection
            
        Returns:
        --------
        CostBreakdown with all cost components
        """
        # Determine liquidity tier
        if liquidity_tier is None:
            liquidity_tier = self._classify_liquidity(symbol, daily_volume_usd)
        
        # Determine borrow tier
        borrow_tier = self._classify_borrow(symbol)
        
        # 1. Spread cost (half spread to cross)
        half_spread = self.config.spread_by_tier.get(
            liquidity_tier, 
            self.config.spread_by_tier[LiquidityTier.MID_CAP.value]
        )
        spread_cost = trade_value * half_spread
        
        # 2. Market impact (Almgren-Chriss square root model)
        # Impact = eta * sigma * sqrt(Q/V)
        if daily_volume_usd > 0:
            participation_rate = trade_value / daily_volume_usd
        else:
            participation_rate = 0.1  # Assume 10% if unknown
        
        participation_rate = min(participation_rate, 1.0)  # Cap at 100%
        
        impact_cost = (
            self.config.impact_eta * 
            volatility * 
            np.sqrt(participation_rate) * 
            trade_value
        )
        
        # 3. Slippage (execution delay)
        slippage_cost = (
            volatility * 
            np.sqrt(self.config.slippage_delay_days) * 
            trade_value * 
            0.5  # Assume 50% of volatility as slippage
        )
        
        # 4. Borrow cost (for shorts only)
        if is_short:
            annual_rate = self.config.borrow_rates.get(
                borrow_tier,
                self.config.borrow_rates[BorrowTier.GENERAL.value]
            )
            borrow_cost = trade_value * annual_rate * holding_period_days / 365
        else:
            borrow_cost = 0.0
        
        # Total
        total_cost = spread_cost + impact_cost + slippage_cost + borrow_cost
        
        return CostBreakdown(
            symbol=symbol,
            trade_value=trade_value,
            spread_cost=spread_cost,
            impact_cost=impact_cost,
            slippage_cost=slippage_cost,
            borrow_cost=borrow_cost,
            total_cost=total_cost,
            total_bps=total_cost / trade_value * 10000,
            cost_pct=total_cost / trade_value,
            liquidity_tier=liquidity_tier,
            borrow_tier=borrow_tier,
            participation_rate=participation_rate
        )
    
    def _classify_liquidity(self, symbol: str, daily_volume_usd: float) -> str:
        """Classify symbol by liquidity."""
        symbol = symbol.upper()
        
        # Check known categories first
        if symbol in self.MEGA_CAP:
            return LiquidityTier.MEGA_CAP.value
        
        if symbol in self.LIQUID_ETFS:
            return LiquidityTier.ETF_LIQUID.value
        
        # Check if crypto
        if symbol in ['BTC', 'ETH']:
            return LiquidityTier.CRYPTO_MAJOR.value
        if symbol in ['SOL', 'XRP', 'ADA', 'DOGE', 'DOT', 'AVAX', 'MATIC']:
            return LiquidityTier.CRYPTO_ALT.value
        
        # Check if ETF (3-4 letter, ends in common patterns)
        if len(symbol) <= 4 and symbol.endswith(('X', 'Y', 'G', 'V', 'W')):
            return LiquidityTier.ETF_OTHER.value
        
        # Classify by volume
        if daily_volume_usd > 500_000_000:  # $500M+
            return LiquidityTier.MEGA_CAP.value
        elif daily_volume_usd > 100_000_000:  # $100M+
            return LiquidityTier.LARGE_CAP.value
        elif daily_volume_usd > 20_000_000:   # $20M+
            return LiquidityTier.MID_CAP.value
        else:
            return LiquidityTier.SMALL_CAP.value
    
    def _classify_borrow(self, symbol: str) -> str:
        """Classify symbol by borrow difficulty."""
        symbol = symbol.upper()
        
        if symbol in self.EASY_BORROW:
            return BorrowTier.EASY_TO_BORROW.value
        
        if symbol in self.HARD_BORROW:
            return BorrowTier.HARD_TO_BORROW.value
        
        if symbol in ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE']:
            return BorrowTier.CRYPTO.value
        
        return BorrowTier.GENERAL.value


def backtest_with_realistic_costs(
    signals: pd.Series,
    prices: pd.DataFrame,
    volumes: pd.Series,
    symbol: str,
    cost_model: TransactionCostModel = None,
    initial_capital: float = 1_000_000
) -> Dict:
    """
    Backtest strategy with realistic transaction costs.
    
    This is the TRUE test - converting gross returns to net returns.
    
    Parameters:
    -----------
    signals : pd.Series
        Trading signals (-1, 0, +1)
    prices : pd.DataFrame
        OHLCV data
    volumes : pd.Series
        Daily volume
    symbol : str
        Ticker symbol
    cost_model : TransactionCostModel
        Cost model (uses default if None)
    initial_capital : float
        Starting capital
        
    Returns:
    --------
    dict with gross/net Sharpe comparison
    """
    if cost_model is None:
        cost_model = TransactionCostModel()
    
    # Align data
    common_idx = signals.index.intersection(prices.index)
    signals = signals.loc[common_idx]
    prices = prices.loc[common_idx]
    volumes = volumes.loc[common_idx] if volumes is not None else pd.Series(1e9, index=common_idx)
    
    # Calculate volatility
    volatility = prices['close'].pct_change().rolling(20).std()
    
    returns_gross = []
    returns_net = []
    costs_paid = []
    trades = 0
    
    position = 0
    capital = initial_capital
    
    for i in range(1, len(signals)):
        date = signals.index[i]
        signal = signals.iloc[i]
        close = prices['close'].iloc[i]
        vol = volumes.iloc[i]
        daily_vol = volatility.iloc[i]
        
        # Calculate return from position
        price_return = prices['close'].pct_change().iloc[i]
        gross_return = position * price_return
        
        # Check if we need to trade
        if signal != position:
            trade_size = abs(signal - position) * capital
            daily_volume_usd = vol * close
            
            cost = cost_model.calculate_cost(
                symbol=symbol,
                trade_value=trade_size,
                daily_volume_usd=daily_volume_usd,
                volatility=daily_vol if not np.isnan(daily_vol) else 0.02,
                side='BUY' if signal > position else 'SELL',
                is_short=signal < 0,
                holding_period_days=10
            )
            
            cost_return = cost.total_cost / capital
            costs_paid.append(cost.total_cost)
            trades += 1
        else:
            cost_return = 0.0
        
        net_return = gross_return - cost_return
        
        returns_gross.append(gross_return)
        returns_net.append(net_return)
        
        position = signal
    
    returns_gross = np.array(returns_gross)
    returns_net = np.array(returns_net)
    
    # Calculate Sharpe ratios
    gross_sharpe = (np.mean(returns_gross) / np.std(returns_gross) * np.sqrt(252)) if np.std(returns_gross) > 0 else 0
    net_sharpe = (np.mean(returns_net) / np.std(returns_net) * np.sqrt(252)) if np.std(returns_net) > 0 else 0
    
    # Total costs
    total_costs = sum(costs_paid)
    
    # Annualized cost drag
    years = len(returns_net) / 252
    annual_cost_drag = (total_costs / initial_capital) / years if years > 0 else 0
    
    # Average cost per trade
    avg_cost_per_trade = total_costs / trades if trades > 0 else 0
    avg_cost_bps = avg_cost_per_trade / (initial_capital / trades) * 10000 if trades > 0 else 0
    
    return {
        'gross_sharpe': gross_sharpe,
        'net_sharpe': net_sharpe,
        'sharpe_drag': gross_sharpe - net_sharpe,
        'gross_return': np.prod(1 + returns_gross) - 1,
        'net_return': np.prod(1 + returns_net) - 1,
        'total_costs': total_costs,
        'annual_cost_drag': annual_cost_drag,
        'n_trades': trades,
        'avg_cost_per_trade': avg_cost_per_trade,
        'avg_cost_bps': avg_cost_bps
    }


# Test the implementation
if __name__ == "__main__":
    print("Testing Transaction Cost Model...")
    print("="*60)
    
    model = TransactionCostModel()
    
    # Test different scenarios
    scenarios = [
        {'symbol': 'AAPL', 'trade_value': 100000, 'volume': 5e9, 'vol': 0.02, 'short': False},
        {'symbol': 'MSFT', 'trade_value': 100000, 'volume': 3e9, 'vol': 0.018, 'short': False},
        {'symbol': 'GME', 'trade_value': 50000, 'volume': 500e6, 'vol': 0.08, 'short': True},
        {'symbol': 'BTC', 'trade_value': 100000, 'volume': 30e9, 'vol': 0.03, 'short': False},
        {'symbol': 'SPY', 'trade_value': 500000, 'volume': 20e9, 'vol': 0.01, 'short': False},
    ]
    
    print(f"{'Symbol':<8} {'Trade Value':>12} {'Spread':>8} {'Impact':>8} {'Slip':>8} {'Borrow':>8} {'Total':>10}")
    print("-"*70)
    
    for s in scenarios:
        cost = model.calculate_cost(
            symbol=s['symbol'],
            trade_value=s['trade_value'],
            daily_volume_usd=s['volume'],
            volatility=s['vol'],
            is_short=s['short'],
            holding_period_days=10
        )
        
        print(f"{s['symbol']:<8} ${s['trade_value']:>10,.0f} {cost.spread_cost/s['trade_value']*10000:>7.1f}bp "
              f"{cost.impact_cost/s['trade_value']*10000:>7.1f}bp {cost.slippage_cost/s['trade_value']*10000:>7.1f}bp "
              f"{cost.borrow_cost/s['trade_value']*10000:>7.1f}bp {cost.total_bps:>9.1f}bp")
    
    print("\n" + "="*60)
    print("Key Insight: These costs turn gross Sharpe into net Sharpe!")
    print("A 2.0 gross Sharpe with 50bp round-trip costs becomes ~1.5 net Sharpe")
