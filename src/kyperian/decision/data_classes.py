"""
KYPERIAN Decision Engine V2 - Data Classes
===========================================

Structured data classes for all decision engine layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np


class TradeStrength(Enum):
    """Trading signal strength levels."""
    NO_TRADE = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4
    
    @property
    def position_multiplier(self) -> float:
        """Position size multiplier based on strength."""
        return {
            TradeStrength.NO_TRADE: 0.0,
            TradeStrength.WEAK: 0.25,
            TradeStrength.MODERATE: 0.50,
            TradeStrength.STRONG: 0.75,
            TradeStrength.VERY_STRONG: 1.0,
        }[self]


class Regime(Enum):
    """Market regime states."""
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    SIDEWAYS = "SIDEWAYS"
    BEAR = "BEAR"
    STRONG_BEAR = "STRONG_BEAR"
    HIGH_VOLATILITY = "HIGH_VOL"
    CRISIS = "CRISIS"
    
    @property
    def trend_alignment(self) -> float:
        """Returns alignment score: positive for bullish, negative for bearish."""
        return {
            Regime.STRONG_BULL: 1.0,
            Regime.BULL: 0.6,
            Regime.SIDEWAYS: 0.0,
            Regime.BEAR: -0.6,
            Regime.STRONG_BEAR: -1.0,
            Regime.HIGH_VOLATILITY: 0.0,
            Regime.CRISIS: -0.8,
        }[self]
    
    @property
    def risk_multiplier(self) -> float:
        """Risk reduction multiplier for position sizing."""
        return {
            Regime.STRONG_BULL: 1.0,
            Regime.BULL: 1.0,
            Regime.SIDEWAYS: 0.7,
            Regime.BEAR: 0.5,
            Regime.STRONG_BEAR: 0.3,
            Regime.HIGH_VOLATILITY: 0.4,
            Regime.CRISIS: 0.1,
        }[self]


class VolatilityState(Enum):
    """Volatility regime states."""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"
    
    @property
    def position_multiplier(self) -> float:
        return {
            VolatilityState.VERY_LOW: 1.2,
            VolatilityState.LOW: 1.1,
            VolatilityState.NORMAL: 1.0,
            VolatilityState.HIGH: 0.6,
            VolatilityState.EXTREME: 0.3,
        }[self]


@dataclass
class SignalLayerScore:
    """
    Layer 1: Technical Signal Scores (40% weight)
    
    Combines multiple technical signal sources:
    - LuxAlgo multi-timeframe signals
    - Momentum indicators (RSI, MACD, Stochastic)
    - Trend strength (ADX, MA alignment)
    - Support/Resistance levels
    """
    # LuxAlgo Signals (15% of total)
    luxalgo_weekly: float = 0.0      # -1 to +1
    luxalgo_daily: float = 0.0       # -1 to +1
    luxalgo_h4: float = 0.0          # -1 to +1
    luxalgo_alignment: float = 0.0   # 0 to 1 (how aligned are timeframes)
    luxalgo_confirmations: int = 0   # Number of confirmations
    
    # Momentum Signals (10% of total)
    rsi_score: float = 0.0           # -1 (oversold) to +1 (overbought)
    macd_score: float = 0.0          # -1 to +1
    stochastic_score: float = 0.0    # -1 to +1
    momentum_divergence: float = 0.0 # -1 (bearish div) to +1 (bullish div)
    
    # Trend Strength (10% of total)
    adx_value: float = 0.0           # 0 to 100
    trend_direction: float = 0.0     # -1 to +1
    ma_alignment: float = 0.0        # 0 to 1 (are MAs properly stacked)
    price_vs_ma200: float = 0.0      # -1 (below) to +1 (above)
    
    # Support/Resistance (5% of total)
    sr_position: float = 0.0         # -1 (at resistance) to +1 (at support)
    breakout_score: float = 0.0      # 0 to 1 (breakout confirmation)
    volume_confirmation: float = 0.0 # 0 to 1
    
    @property
    def luxalgo_score(self) -> float:
        """Combined LuxAlgo score."""
        # Weekly has highest weight (veto power)
        weekly_weight = 0.45
        daily_weight = 0.35
        h4_weight = 0.20
        
        weighted = (
            self.luxalgo_weekly * weekly_weight +
            self.luxalgo_daily * daily_weight +
            self.luxalgo_h4 * h4_weight
        )
        
        # Boost if aligned
        if self.luxalgo_alignment > 0.8:
            weighted *= 1.2
        
        return np.clip(weighted, -1, 1)
    
    @property
    def momentum_score(self) -> float:
        """Combined momentum score."""
        base = (
            self.rsi_score * 0.3 +
            self.macd_score * 0.35 +
            self.stochastic_score * 0.2 +
            self.momentum_divergence * 0.15
        )
        return np.clip(base, -1, 1)
    
    @property
    def trend_score(self) -> float:
        """Combined trend score."""
        # Strong trend (high ADX) amplifies direction
        trend_strength = min(1.0, self.adx_value / 40)  # Normalize ADX
        
        direction = (
            self.trend_direction * 0.4 +
            self.ma_alignment * self.trend_direction * 0.3 +
            self.price_vs_ma200 * 0.3
        )
        
        return np.clip(direction * (0.5 + 0.5 * trend_strength), -1, 1)
    
    @property
    def sr_score(self) -> float:
        """Support/Resistance score."""
        return (
            self.sr_position * 0.5 +
            self.breakout_score * 0.3 +
            self.volume_confirmation * 0.2
        )
    
    @property
    def combined(self) -> float:
        """Combined signal layer score (-1 to +1)."""
        # Weights within signal layer
        return (
            self.luxalgo_score * 0.375 +      # 15% / 40% = 37.5%
            self.momentum_score * 0.25 +       # 10% / 40% = 25%
            self.trend_score * 0.25 +          # 10% / 40% = 25%
            self.sr_score * 0.125              # 5% / 40% = 12.5%
        )
    
    @property
    def direction(self) -> int:
        """Primary direction: -1 (SELL), 0 (NEUTRAL), +1 (BUY)."""
        if self.combined > 0.15:
            return 1
        elif self.combined < -0.15:
            return -1
        return 0
    
    @property
    def confidence(self) -> float:
        """Confidence in the signal (0 to 1)."""
        return min(1.0, abs(self.combined) * 1.2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'luxalgo': {
                'weekly': round(self.luxalgo_weekly, 3),
                'daily': round(self.luxalgo_daily, 3),
                'h4': round(self.luxalgo_h4, 3),
                'alignment': round(self.luxalgo_alignment, 3),
                'combined': round(self.luxalgo_score, 3),
            },
            'momentum': {
                'rsi': round(self.rsi_score, 3),
                'macd': round(self.macd_score, 3),
                'stochastic': round(self.stochastic_score, 3),
                'divergence': round(self.momentum_divergence, 3),
                'combined': round(self.momentum_score, 3),
            },
            'trend': {
                'adx': round(self.adx_value, 1),
                'direction': round(self.trend_direction, 3),
                'ma_alignment': round(self.ma_alignment, 3),
                'vs_ma200': round(self.price_vs_ma200, 3),
                'combined': round(self.trend_score, 3),
            },
            'support_resistance': {
                'position': round(self.sr_position, 3),
                'breakout': round(self.breakout_score, 3),
                'volume': round(self.volume_confirmation, 3),
                'combined': round(self.sr_score, 3),
            },
            'combined_score': round(self.combined, 3),
            'direction': self.direction,
            'confidence': round(self.confidence, 3),
        }


@dataclass
class ContextLayerScore:
    """
    Layer 2: Market Context Scores (30% weight)
    
    Provides market context to validate signals:
    - Regime detection (Bull/Bear/Sideways)
    - Sentiment analysis (News, Social)
    - Volatility state
    - Macro environment
    """
    # Regime Detection (10% of total)
    regime: Regime = Regime.SIDEWAYS
    regime_confidence: float = 0.5
    regime_duration_days: int = 0
    trend_strength: float = 0.0      # 0 to 1
    
    # Sentiment Analysis (10% of total)
    news_sentiment: float = 0.0      # -1 to +1
    news_confidence: float = 0.0     # 0 to 1
    social_sentiment: float = 0.0    # -1 to +1
    social_volume: float = 0.0       # Relative volume (0 to 1)
    fear_greed_index: float = 50     # 0 (extreme fear) to 100 (extreme greed)
    
    # Volatility (5% of total)
    volatility_state: VolatilityState = VolatilityState.NORMAL
    volatility_percentile: float = 0.5  # 0 to 1
    vix_level: float = 20.0          # VIX value
    atr_percentile: float = 0.5      # ATR relative to history
    
    # Macro Environment (5% of total)
    dxy_trend: float = 0.0           # -1 to +1 (USD strength)
    rates_environment: float = 0.0   # -1 (dovish) to +1 (hawkish)
    spy_correlation: float = 0.0     # -1 to +1
    sector_momentum: float = 0.0     # -1 to +1
    
    @property
    def regime_score(self) -> float:
        """Regime contribution to context score."""
        alignment = self.regime.trend_alignment
        return alignment * self.regime_confidence * self.trend_strength
    
    @property
    def sentiment_score(self) -> float:
        """Combined sentiment score."""
        news = self.news_sentiment * self.news_confidence
        social = self.social_sentiment * min(1.0, self.social_volume * 2)
        
        # Fear/Greed normalization (-1 to +1)
        fg_normalized = (self.fear_greed_index - 50) / 50
        
        return (
            news * 0.4 +
            social * 0.3 +
            fg_normalized * 0.3
        )
    
    @property
    def volatility_score(self) -> float:
        """Volatility contribution (lower vol = higher score for trading)."""
        # Invert: low volatility is good for confidence
        vol_factor = 1 - self.volatility_percentile
        
        # Extreme volatility is always bad
        if self.volatility_state in [VolatilityState.EXTREME, VolatilityState.HIGH]:
            vol_factor *= 0.5
        
        return vol_factor
    
    @property
    def macro_score(self) -> float:
        """Macro environment score."""
        return (
            self.sector_momentum * 0.4 +
            self.spy_correlation * 0.3 +
            (-self.rates_environment) * 0.15 +  # Hawkish = negative
            (-self.dxy_trend) * 0.15            # Strong USD = negative for risk
        )
    
    @property
    def combined(self) -> float:
        """Combined context layer score (-1 to +1)."""
        # Apply regime risk multiplier
        base = (
            self.regime_score * 0.33 +
            self.sentiment_score * 0.33 +
            self.volatility_score * 0.17 +
            self.macro_score * 0.17
        )
        
        # Reduce score in dangerous regimes
        return base * self.regime.risk_multiplier
    
    @property
    def should_trade(self) -> bool:
        """Whether market context supports trading."""
        if self.regime in [Regime.CRISIS, Regime.HIGH_VOLATILITY]:
            return False
        if self.volatility_state == VolatilityState.EXTREME:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime': {
                'state': self.regime.value,
                'confidence': round(self.regime_confidence, 3),
                'duration_days': self.regime_duration_days,
                'trend_strength': round(self.trend_strength, 3),
                'score': round(self.regime_score, 3),
            },
            'sentiment': {
                'news': round(self.news_sentiment, 3),
                'news_confidence': round(self.news_confidence, 3),
                'social': round(self.social_sentiment, 3),
                'fear_greed': round(self.fear_greed_index, 1),
                'combined': round(self.sentiment_score, 3),
            },
            'volatility': {
                'state': self.volatility_state.value,
                'percentile': round(self.volatility_percentile, 3),
                'vix': round(self.vix_level, 1),
                'score': round(self.volatility_score, 3),
            },
            'macro': {
                'dxy_trend': round(self.dxy_trend, 3),
                'rates': round(self.rates_environment, 3),
                'spy_corr': round(self.spy_correlation, 3),
                'sector': round(self.sector_momentum, 3),
                'combined': round(self.macro_score, 3),
            },
            'combined_score': round(self.combined, 3),
            'should_trade': self.should_trade,
        }


@dataclass
class ValidationLayerScore:
    """
    Layer 3: Historical Validation Scores (20% weight)
    
    Validates decisions against historical performance:
    - Win rate of similar setups
    - Backtest confidence metrics
    - Pattern recognition matching
    """
    # Historical Win Rate (10% of total)
    win_rate: float = 0.5            # 0 to 1
    sample_size: int = 0             # Number of similar trades
    avg_profit_factor: float = 1.0   # Profit factor of similar trades
    avg_hold_time_hours: float = 0   # Average holding time
    
    # Backtest Metrics (5% of total)
    backtest_sharpe: float = 0.0     # Sharpe ratio
    backtest_sortino: float = 0.0    # Sortino ratio
    backtest_max_dd: float = 0.0     # Max drawdown
    pbo_probability: float = 0.5     # Probability of backtest overfitting
    
    # Pattern Matching (5% of total)
    pattern_match_score: float = 0.0  # 0 to 1
    similar_patterns_found: int = 0
    pattern_avg_return: float = 0.0   # Average return of matched patterns
    pattern_consistency: float = 0.0  # How consistent are pattern returns
    
    @property
    def win_rate_score(self) -> float:
        """Win rate contribution with sample size confidence."""
        # Penalize low sample sizes
        if self.sample_size < 10:
            confidence = self.sample_size / 10
        elif self.sample_size < 30:
            confidence = 0.7 + (self.sample_size - 10) / 66.67
        else:
            confidence = 1.0
        
        # Convert win rate to score (0.5 = neutral)
        win_normalized = (self.win_rate - 0.5) * 2  # -1 to +1
        
        return win_normalized * confidence
    
    @property
    def backtest_score(self) -> float:
        """Backtest metrics score."""
        # Normalize Sharpe (assume 2.0 is excellent)
        sharpe_norm = min(1.0, self.backtest_sharpe / 2.0)
        
        # Normalize Sortino (assume 3.0 is excellent)
        sortino_norm = min(1.0, self.backtest_sortino / 3.0)
        
        # Max DD penalty (assume 20% is bad)
        dd_penalty = max(0, 1 - self.backtest_max_dd / 0.2)
        
        # PBO penalty (high probability of overfitting is bad)
        pbo_factor = 1 - self.pbo_probability
        
        return (
            sharpe_norm * 0.35 +
            sortino_norm * 0.25 +
            dd_penalty * 0.2 +
            pbo_factor * 0.2
        )
    
    @property
    def pattern_score(self) -> float:
        """Pattern matching score."""
        if self.similar_patterns_found == 0:
            return 0.5  # Neutral if no patterns found
        
        # Pattern confidence
        pattern_conf = min(1.0, self.similar_patterns_found / 20)
        
        # Return score (positive return = positive score)
        return_score = np.tanh(self.pattern_avg_return * 10)  # Normalize to -1,+1
        
        return (
            self.pattern_match_score * 0.3 +
            return_score * pattern_conf * 0.4 +
            self.pattern_consistency * 0.3
        )
    
    @property
    def combined(self) -> float:
        """Combined validation layer score (-1 to +1)."""
        return (
            self.win_rate_score * 0.5 +
            self.backtest_score * 0.25 +
            self.pattern_score * 0.25
        )
    
    @property
    def is_validated(self) -> bool:
        """Whether the signal passes validation."""
        if self.sample_size < 5:
            return True  # Not enough data to invalidate
        return self.win_rate >= 0.45 and self.pbo_probability < 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'historical': {
                'win_rate': round(self.win_rate, 3),
                'sample_size': self.sample_size,
                'profit_factor': round(self.avg_profit_factor, 2),
                'avg_hold_hours': round(self.avg_hold_time_hours, 1),
                'score': round(self.win_rate_score, 3),
            },
            'backtest': {
                'sharpe': round(self.backtest_sharpe, 2),
                'sortino': round(self.backtest_sortino, 2),
                'max_dd': round(self.backtest_max_dd, 3),
                'pbo_probability': round(self.pbo_probability, 3),
                'score': round(self.backtest_score, 3),
            },
            'pattern': {
                'match_score': round(self.pattern_match_score, 3),
                'patterns_found': self.similar_patterns_found,
                'avg_return': round(self.pattern_avg_return, 4),
                'consistency': round(self.pattern_consistency, 3),
                'score': round(self.pattern_score, 3),
            },
            'combined_score': round(self.combined, 3),
            'is_validated': self.is_validated,
        }


@dataclass
class RiskLayerScore:
    """
    Layer 4: Risk Management (10% weight + VETO POWER)
    
    Final risk checks that can VETO any trade:
    - Position size limits
    - Portfolio drawdown
    - Correlation with existing positions
    - Liquidity checks
    - Time-based restrictions
    """
    # Position Limits
    current_position_pct: float = 0.0    # Current position in this asset
    max_position_pct: float = 10.0       # Maximum allowed
    position_check_passed: bool = True
    
    # Drawdown Limits
    portfolio_drawdown: float = 0.0      # Current portfolio drawdown
    max_drawdown_limit: float = 0.15     # Maximum allowed (15%)
    drawdown_check_passed: bool = True
    
    # Correlation
    max_correlation: float = 0.0         # Highest correlation with existing positions
    correlation_limit: float = 0.7       # Maximum allowed
    correlation_check_passed: bool = True
    
    # Liquidity
    avg_daily_volume: float = 0          # Average daily volume
    min_volume_required: float = 0       # Minimum required
    spread_pct: float = 0.0              # Bid-ask spread percentage
    liquidity_check_passed: bool = True
    
    # Time Restrictions
    market_hours: bool = True            # Is market open
    avoid_earnings: bool = False         # Earnings within 24h
    avoid_fomc: bool = False             # FOMC within 24h
    time_check_passed: bool = True
    
    # Additional Risk Factors
    sector_exposure_pct: float = 0.0     # Current exposure to this sector
    max_sector_exposure: float = 25.0    # Maximum sector exposure
    sector_check_passed: bool = True
    
    @property
    def all_checks_passed(self) -> bool:
        """Whether all risk checks pass."""
        return all([
            self.position_check_passed,
            self.drawdown_check_passed,
            self.correlation_check_passed,
            self.liquidity_check_passed,
            self.time_check_passed,
            self.sector_check_passed,
        ])
    
    @property
    def any_veto(self) -> bool:
        """Whether any check triggers a veto."""
        return not self.all_checks_passed
    
    @property
    def veto_reasons(self) -> List[str]:
        """List of veto reasons."""
        reasons = []
        if not self.position_check_passed:
            reasons.append(f"Position limit exceeded ({self.current_position_pct:.1f}% > {self.max_position_pct:.1f}%)")
        if not self.drawdown_check_passed:
            reasons.append(f"Drawdown limit reached ({self.portfolio_drawdown:.1%} > {self.max_drawdown_limit:.1%})")
        if not self.correlation_check_passed:
            reasons.append(f"High correlation with existing positions ({self.max_correlation:.2f} > {self.correlation_limit:.2f})")
        if not self.liquidity_check_passed:
            reasons.append(f"Insufficient liquidity (spread: {self.spread_pct:.2%})")
        if not self.time_check_passed:
            if not self.market_hours:
                reasons.append("Market closed")
            if self.avoid_earnings:
                reasons.append("Earnings announcement within 24h")
            if self.avoid_fomc:
                reasons.append("FOMC meeting within 24h")
        if not self.sector_check_passed:
            reasons.append(f"Sector exposure limit ({self.sector_exposure_pct:.1f}% > {self.max_sector_exposure:.1f}%)")
        return reasons
    
    @property
    def risk_score(self) -> float:
        """Risk score (0 = no risk, 1 = maximum risk)."""
        if self.any_veto:
            return 1.0
        
        # Calculate risk from each factor
        position_risk = self.current_position_pct / self.max_position_pct
        dd_risk = self.portfolio_drawdown / self.max_drawdown_limit
        corr_risk = self.max_correlation / self.correlation_limit
        sector_risk = self.sector_exposure_pct / self.max_sector_exposure
        
        return np.clip(
            max(position_risk, dd_risk, corr_risk, sector_risk),
            0, 1
        )
    
    @property
    def position_size_multiplier(self) -> float:
        """Multiplier to reduce position size based on risk."""
        if self.any_veto:
            return 0.0
        
        # Reduce position size as we approach limits
        margin = 1 - self.risk_score
        return max(0.25, margin)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'position': {
                'current_pct': round(self.current_position_pct, 2),
                'max_pct': round(self.max_position_pct, 2),
                'passed': self.position_check_passed,
            },
            'drawdown': {
                'current': round(self.portfolio_drawdown, 4),
                'limit': round(self.max_drawdown_limit, 4),
                'passed': self.drawdown_check_passed,
            },
            'correlation': {
                'max_corr': round(self.max_correlation, 3),
                'limit': round(self.correlation_limit, 3),
                'passed': self.correlation_check_passed,
            },
            'liquidity': {
                'volume': self.avg_daily_volume,
                'spread_pct': round(self.spread_pct, 4),
                'passed': self.liquidity_check_passed,
            },
            'time': {
                'market_hours': self.market_hours,
                'avoid_earnings': self.avoid_earnings,
                'avoid_fomc': self.avoid_fomc,
                'passed': self.time_check_passed,
            },
            'sector': {
                'exposure_pct': round(self.sector_exposure_pct, 2),
                'max_pct': round(self.max_sector_exposure, 2),
                'passed': self.sector_check_passed,
            },
            'all_passed': self.all_checks_passed,
            'veto_reasons': self.veto_reasons,
            'risk_score': round(self.risk_score, 3),
            'position_multiplier': round(self.position_size_multiplier, 3),
        }


@dataclass
class TradingDecision:
    """
    Final Trading Decision
    
    Complete decision with all layer scores, reasoning, and trade parameters.
    """
    # Core Decision
    symbol: str
    timestamp: datetime
    direction: int                    # -1 (SELL), 0 (NEUTRAL), +1 (BUY)
    strength: TradeStrength
    confidence: float                 # 0 to 100
    
    # Trade Parameters
    recommended_position_pct: float   # 0 to 10%
    entry_price: float
    stop_loss: float
    take_profit_1: float              # Conservative target
    take_profit_2: float              # Moderate target
    take_profit_3: float              # Aggressive target
    risk_reward_ratio: float
    max_holding_hours: float = 0      # Suggested max holding time
    
    # Layer Scores
    signal_layer: SignalLayerScore = field(default_factory=SignalLayerScore)
    context_layer: ContextLayerScore = field(default_factory=ContextLayerScore)
    validation_layer: ValidationLayerScore = field(default_factory=ValidationLayerScore)
    risk_layer: RiskLayerScore = field(default_factory=RiskLayerScore)
    
    # Metadata
    reasoning: List[str] = field(default_factory=list)
    data_points_used: int = 0
    processing_time_ms: float = 0
    
    @property
    def direction_str(self) -> str:
        if self.direction > 0:
            return "BUY"
        elif self.direction < 0:
            return "SELL"
        return "NEUTRAL"
    
    @property
    def should_trade(self) -> bool:
        return (
            self.strength != TradeStrength.NO_TRADE and
            not self.risk_layer.any_veto and
            self.context_layer.should_trade
        )
    
    @property
    def risk_amount(self) -> float:
        """Dollar risk per unit."""
        return abs(self.entry_price - self.stop_loss)
    
    @property
    def reward_amount(self) -> float:
        """Dollar reward per unit (to TP2)."""
        return abs(self.take_profit_2 - self.entry_price)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'direction': self.direction_str,
            'strength': self.strength.name,
            'confidence': round(self.confidence, 1),
            'should_trade': self.should_trade,
            'trade_parameters': {
                'position_size_pct': round(self.recommended_position_pct, 2),
                'entry': round(self.entry_price, 4),
                'stop_loss': round(self.stop_loss, 4),
                'targets': [
                    round(self.take_profit_1, 4),
                    round(self.take_profit_2, 4),
                    round(self.take_profit_3, 4),
                ],
                'risk_reward': round(self.risk_reward_ratio, 2),
                'max_holding_hours': round(self.max_holding_hours, 1),
            },
            'layers': {
                'signal': self.signal_layer.to_dict(),
                'context': self.context_layer.to_dict(),
                'validation': self.validation_layer.to_dict(),
                'risk': self.risk_layer.to_dict(),
            },
            'reasoning': self.reasoning,
            'data_points_used': self.data_points_used,
            'processing_time_ms': round(self.processing_time_ms, 2),
        }
    
    def to_summary(self) -> str:
        """Human-readable summary."""
        emoji = "🚀" if self.strength == TradeStrength.STRONG else "✅" if self.strength == TradeStrength.MODERATE else "⚠️" if self.strength == TradeStrength.WEAK else "⛔"
        
        lines = [
            f"{emoji} {self.symbol} | {self.direction_str} | {self.strength.name}",
            f"   Confidence: {self.confidence:.1f}% (based on {self.data_points_used} data points)",
            f"   Entry: ${self.entry_price:,.2f}",
            f"   Stop: ${self.stop_loss:,.2f} ({self._pct_diff(self.entry_price, self.stop_loss):.1f}%)",
            f"   Target: ${self.take_profit_2:,.2f} ({self._pct_diff(self.entry_price, self.take_profit_2):.1f}%)",
            f"   Position: {self.recommended_position_pct:.1f}% | R/R: {self.risk_reward_ratio:.1f}:1",
        ]
        
        if self.risk_layer.any_veto:
            lines.append(f"   ⛔ VETO: {', '.join(self.risk_layer.veto_reasons)}")
        
        return "\n".join(lines)
    
    def _pct_diff(self, a: float, b: float) -> float:
        if a == 0:
            return 0
        return ((b - a) / a) * 100
