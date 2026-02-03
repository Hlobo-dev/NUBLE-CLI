"""
NUBLE ELITE: Institutional Position Calculator

Calculates optimal position size using:
- Kelly Criterion (modified for trading)
- Multi-timeframe alignment score
- Signal freshness decay
- Regime adjustment
- Risk management caps

The philosophy:
- Size positions based on EDGE, not conviction
- Never risk more than 2% per trade
- Full position only on perfect setups
- Scale in, don't go all-in
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math

from .timeframe_manager import TimeframeSignal, Timeframe
from .veto_engine import VetoResult, VetoDecision

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """
    Calculated position size with full breakdown.
    """
    # Core sizing
    recommended_size: float          # As fraction of portfolio (0-1)
    dollar_amount: float             # Actual dollar amount
    shares: int                      # Number of shares/units
    
    # Risk levels
    stop_loss_price: float
    stop_loss_pct: float
    take_profit_prices: List[float]  # [TP1, TP2, TP3]
    take_profit_pcts: List[float]
    risk_reward_ratio: float
    
    # Confidence metrics
    confidence: float
    alignment_score: float
    kelly_fraction: float
    
    # Breakdown
    breakdown: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recommended_size": self.recommended_size,
            "dollar_amount": self.dollar_amount,
            "shares": self.shares,
            "stop_loss_price": self.stop_loss_price,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_prices": self.take_profit_prices,
            "take_profit_pcts": self.take_profit_pcts,
            "risk_reward_ratio": self.risk_reward_ratio,
            "confidence": self.confidence,
            "alignment_score": self.alignment_score,
            "kelly_fraction": self.kelly_fraction,
            "breakdown": self.breakdown,
        }


class PositionCalculator:
    """
    Institutional-grade position calculator.
    
    Uses modified Kelly Criterion with:
    - Multi-timeframe alignment
    - Signal strength weighting
    - Regime adjustment
    - Conservative caps
    """
    
    # Risk parameters
    MAX_RISK_PER_TRADE = 0.02       # 2% max risk per trade
    MAX_POSITION_SIZE = 0.10        # 10% max position size
    KELLY_FRACTION = 0.5            # Half-Kelly for safety
    
    # Historical performance estimates (conservative)
    BASE_WIN_RATE = 0.45            # Base win rate
    WIN_RATE_PER_ALIGNMENT = 0.05   # Win rate boost per aligned timeframe
    BASE_WIN_LOSS_RATIO = 1.5       # Base reward:risk
    
    # ATR multipliers for stops
    STOP_LOSS_ATR_MULT = 2.0
    TP1_ATR_MULT = 2.0              # 1:1 risk/reward
    TP2_ATR_MULT = 4.0              # 2:1 risk/reward
    TP3_ATR_MULT = 6.0              # 3:1 risk/reward
    
    def __init__(
        self,
        max_risk: float = 0.02,
        max_position: float = 0.10,
        kelly_fraction: float = 0.5
    ):
        """
        Initialize PositionCalculator.
        
        Args:
            max_risk: Maximum risk per trade as fraction
            max_position: Maximum position size as fraction
            kelly_fraction: Fraction of Kelly to use (0.5 = half-Kelly)
        """
        self.max_risk = max_risk
        self.max_position = max_position
        self.kelly_fraction = kelly_fraction
        
        logger.info(
            f"PositionCalculator initialized: max_risk={max_risk:.1%}, "
            f"max_position={max_position:.1%}, kelly={kelly_fraction}"
        )
    
    def calculate_kelly(
        self,
        win_rate: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calculate Kelly Criterion fraction.
        
        Kelly = (p * b - q) / b
        Where:
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = win/loss ratio
        
        Args:
            win_rate: Probability of winning (0-1)
            win_loss_ratio: Average win / average loss
            
        Returns:
            Optimal fraction to bet (can be negative if edge is negative)
        """
        q = 1 - win_rate
        kelly = (win_rate * win_loss_ratio - q) / win_loss_ratio
        return max(0, kelly)
    
    def calculate_alignment_score(
        self,
        weekly: Optional[TimeframeSignal],
        daily: Optional[TimeframeSignal],
        four_hour: Optional[TimeframeSignal],
        direction: int
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate alignment score and breakdown.
        
        Args:
            weekly: Weekly signal
            daily: Daily signal
            four_hour: 4H signal
            direction: Trading direction
            
        Returns:
            Tuple of (alignment_score, breakdown_dict)
        """
        score = 0.0
        breakdown = {
            "weekly": {"aligned": False, "contribution": 0},
            "daily": {"aligned": False, "contribution": 0},
            "4h": {"aligned": False, "contribution": 0},
        }
        
        # Weekly contribution (40%)
        if weekly and weekly.is_fresh:
            if weekly.direction == direction:
                contrib = 0.40 * weekly.freshness
                if weekly.is_strong:
                    contrib *= 1.2
                score += contrib
                breakdown["weekly"]["aligned"] = True
                breakdown["weekly"]["contribution"] = contrib
                breakdown["weekly"]["freshness"] = weekly.freshness
            elif weekly.direction != 0:
                # Weekly opposes - heavy penalty
                score -= 0.20
                breakdown["weekly"]["penalty"] = 0.20
        
        # Daily contribution (35%)
        if daily and daily.is_fresh:
            if daily.direction == direction:
                contrib = 0.35 * daily.freshness
                if daily.is_strong:
                    contrib *= 1.15
                # Bonus if aligned with weekly
                if weekly and weekly.direction == daily.direction:
                    contrib *= 1.2
                score += contrib
                breakdown["daily"]["aligned"] = True
                breakdown["daily"]["contribution"] = contrib
            elif daily.direction != 0:
                score -= 0.10
                breakdown["daily"]["penalty"] = 0.10
        
        # 4H contribution (25%)
        if four_hour and four_hour.is_fresh:
            if four_hour.direction == direction:
                contrib = 0.25 * four_hour.freshness
                if four_hour.is_strong:
                    contrib *= 1.15
                # Bonus for confirmations
                if four_hour.confirmations >= 8:
                    contrib *= 1.0 + (four_hour.confirmations - 8) * 0.03
                # Bonus if aligned with daily
                if daily and daily.direction == four_hour.direction:
                    contrib *= 1.15
                score += contrib
                breakdown["4h"]["aligned"] = True
                breakdown["4h"]["contribution"] = contrib
                breakdown["4h"]["confirmations"] = four_hour.confirmations
            elif four_hour.direction != 0:
                score -= 0.05
                breakdown["4h"]["penalty"] = 0.05
        
        # Normalize to 0-1 range
        alignment_score = max(0.0, min(1.0, score))
        breakdown["total_score"] = alignment_score
        
        return alignment_score, breakdown
    
    def estimate_win_rate(
        self,
        alignment_score: float,
        weekly: Optional[TimeframeSignal],
        daily: Optional[TimeframeSignal],
        four_hour: Optional[TimeframeSignal]
    ) -> float:
        """
        Estimate win rate based on signal alignment and strength.
        
        Conservative estimates based on institutional research.
        
        Args:
            alignment_score: Overall alignment score
            weekly/daily/four_hour: Signals
            
        Returns:
            Estimated win rate (0-1)
        """
        # Base win rate
        win_rate = self.BASE_WIN_RATE
        
        # Alignment boost
        win_rate += alignment_score * 0.15  # Up to 15% boost
        
        # Strong signal boost
        strong_count = sum(1 for s in [weekly, daily, four_hour] 
                         if s and s.is_strong)
        win_rate += strong_count * 0.02  # 2% per strong signal
        
        # Confirmation boost
        if four_hour and four_hour.confirmations >= 10:
            win_rate += 0.03
        
        # Trend strength boost
        if weekly and weekly.trend_strength > 70:
            win_rate += 0.02
        if daily and daily.trend_strength > 60:
            win_rate += 0.01
        
        # Cap win rate at realistic levels
        return min(0.65, max(0.35, win_rate))
    
    def calculate_stops(
        self,
        current_price: float,
        direction: int,
        atr: Optional[float] = None,
        four_hour: Optional[TimeframeSignal] = None
    ) -> Tuple[float, List[float], float, List[float]]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            current_price: Current asset price
            direction: Trade direction (1 = long, -1 = short)
            atr: Average True Range (optional, estimated if not provided)
            four_hour: 4H signal (for Smart Trail level)
            
        Returns:
            Tuple of (stop_loss, take_profits, stop_pct, tp_pcts)
        """
        # Estimate ATR if not provided (2% of price)
        if atr is None:
            atr = current_price * 0.02
        
        if direction > 0:  # Long
            stop_loss = current_price - (atr * self.STOP_LOSS_ATR_MULT)
            
            # Use Smart Trail as stop if available and tighter
            if four_hour and four_hour.smart_trail_level:
                if four_hour.smart_trail_level < current_price:
                    stop_loss = max(stop_loss, four_hour.smart_trail_level)
            
            take_profits = [
                current_price + (atr * self.TP1_ATR_MULT),
                current_price + (atr * self.TP2_ATR_MULT),
                current_price + (atr * self.TP3_ATR_MULT),
            ]
            
        else:  # Short
            stop_loss = current_price + (atr * self.STOP_LOSS_ATR_MULT)
            
            # Use Smart Trail as stop if available and tighter
            if four_hour and four_hour.smart_trail_level:
                if four_hour.smart_trail_level > current_price:
                    stop_loss = min(stop_loss, four_hour.smart_trail_level)
            
            take_profits = [
                current_price - (atr * self.TP1_ATR_MULT),
                current_price - (atr * self.TP2_ATR_MULT),
                current_price - (atr * self.TP3_ATR_MULT),
            ]
        
        # Calculate percentages
        stop_pct = abs(stop_loss - current_price) / current_price
        tp_pcts = [abs(tp - current_price) / current_price for tp in take_profits]
        
        return stop_loss, take_profits, stop_pct, tp_pcts
    
    def calculate_position(
        self,
        veto_result: VetoResult,
        current_price: float,
        portfolio_value: float,
        atr: Optional[float] = None,
        regime: str = "NORMAL"
    ) -> PositionSize:
        """
        Calculate optimal position size.
        
        Args:
            veto_result: Result from VetoEngine
            current_price: Current asset price
            portfolio_value: Total portfolio value
            atr: Average True Range (optional)
            regime: Market regime (BULL, BEAR, SIDEWAYS, VOLATILE)
            
        Returns:
            PositionSize with full breakdown
        """
        # If vetoed, return zero position
        if not veto_result.can_trade:
            return PositionSize(
                recommended_size=0.0,
                dollar_amount=0.0,
                shares=0,
                stop_loss_price=0.0,
                stop_loss_pct=0.0,
                take_profit_prices=[],
                take_profit_pcts=[],
                risk_reward_ratio=0.0,
                confidence=0.0,
                alignment_score=0.0,
                kelly_fraction=0.0,
                breakdown={"reason": veto_result.reason}
            )
        
        # Extract signals
        weekly = veto_result.weekly_signal
        daily = veto_result.daily_signal
        four_hour = veto_result.four_hour_signal
        direction = veto_result.direction
        
        # Calculate alignment score
        alignment_score, alignment_breakdown = self.calculate_alignment_score(
            weekly, daily, four_hour, direction
        )
        
        # Estimate win rate
        win_rate = self.estimate_win_rate(alignment_score, weekly, daily, four_hour)
        
        # Calculate stops and targets
        stop_loss, take_profits, stop_pct, tp_pcts = self.calculate_stops(
            current_price, direction, atr, four_hour
        )
        
        # Calculate risk/reward ratio (based on TP2)
        if len(tp_pcts) >= 2 and stop_pct > 0:
            risk_reward = tp_pcts[1] / stop_pct
        else:
            risk_reward = 2.0  # Default
        
        # Calculate Kelly fraction
        kelly = self.calculate_kelly(win_rate, risk_reward)
        
        # Apply half-Kelly for safety
        kelly_adjusted = kelly * self.kelly_fraction
        
        # Apply veto multiplier
        position_fraction = kelly_adjusted * veto_result.position_multiplier
        
        # Apply regime adjustment
        regime_mult = self._get_regime_multiplier(regime)
        position_fraction *= regime_mult
        
        # Cap at max position
        position_fraction = min(position_fraction, self.max_position)
        
        # Risk-based cap
        # Risk amount = position * stop_pct
        # Max risk = portfolio * max_risk
        # Therefore: position <= (portfolio * max_risk) / stop_pct
        if stop_pct > 0:
            risk_based_max = self.max_risk / stop_pct
            position_fraction = min(position_fraction, risk_based_max)
        
        # Calculate dollar amount and shares
        dollar_amount = portfolio_value * position_fraction
        shares = int(dollar_amount / current_price) if current_price > 0 else 0
        
        # Calculate confidence
        confidence = alignment_score * veto_result.position_multiplier
        
        # Build breakdown
        breakdown = {
            "alignment": alignment_breakdown,
            "win_rate_estimate": win_rate,
            "kelly_raw": kelly,
            "kelly_adjusted": kelly_adjusted,
            "veto_multiplier": veto_result.position_multiplier,
            "regime": regime,
            "regime_multiplier": regime_mult,
            "position_before_caps": kelly_adjusted * veto_result.position_multiplier * regime_mult,
            "max_position_cap": self.max_position,
            "risk_based_cap": risk_based_max if stop_pct > 0 else None,
            "final_position": position_fraction,
        }
        
        return PositionSize(
            recommended_size=position_fraction,
            dollar_amount=dollar_amount,
            shares=shares,
            stop_loss_price=stop_loss,
            stop_loss_pct=stop_pct,
            take_profit_prices=take_profits,
            take_profit_pcts=tp_pcts,
            risk_reward_ratio=risk_reward,
            confidence=confidence,
            alignment_score=alignment_score,
            kelly_fraction=kelly_adjusted,
            breakdown=breakdown,
        )
    
    def _get_regime_multiplier(self, regime: str) -> float:
        """Get position size multiplier based on market regime."""
        regime = regime.upper()
        
        multipliers = {
            "BULL": 1.1,       # Slightly larger in bull market
            "BEAR": 0.9,       # Slightly smaller in bear market
            "SIDEWAYS": 0.8,   # Smaller in sideways
            "VOLATILE": 0.6,   # Much smaller in volatile
            "NORMAL": 1.0,     # Normal sizing
        }
        
        return multipliers.get(regime, 1.0)
    
    def calculate_scaling_plan(
        self,
        position: PositionSize,
        max_adds: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Calculate a scaling plan for pyramiding.
        
        Args:
            position: Initial position size
            max_adds: Maximum number of add-ons
            
        Returns:
            List of scaling levels
        """
        if position.recommended_size == 0:
            return []
        
        plan = []
        
        # Initial entry
        plan.append({
            "level": 0,
            "action": "Initial Entry",
            "size_pct": 0.5,  # 50% of total position
            "size_of_total": position.recommended_size * 0.5,
            "condition": "All timeframes aligned",
        })
        
        # First add
        if max_adds >= 1 and len(position.take_profit_prices) > 0:
            plan.append({
                "level": 1,
                "action": "First Add",
                "size_pct": 0.3,  # 30% of total
                "size_of_total": position.recommended_size * 0.3,
                "trigger_price": position.take_profit_prices[0] * 0.5,  # 50% to TP1
                "condition": "Price moves 50% toward TP1 + 4H confirms direction",
            })
        
        # Second add
        if max_adds >= 2 and len(position.take_profit_prices) > 0:
            plan.append({
                "level": 2,
                "action": "Second Add",
                "size_pct": 0.2,  # 20% of total
                "size_of_total": position.recommended_size * 0.2,
                "trigger_price": position.take_profit_prices[0],  # At TP1
                "condition": "TP1 reached + all timeframes still aligned",
            })
        
        return plan


# Convenience function
def calculate_position(
    veto_result: VetoResult,
    current_price: float,
    portfolio_value: float = 100000,
    atr: Optional[float] = None,
    regime: str = "NORMAL"
) -> PositionSize:
    """
    Calculate optimal position size.
    
    Args:
        veto_result: Result from VetoEngine
        current_price: Current asset price
        portfolio_value: Total portfolio value
        atr: Average True Range
        regime: Market regime
        
    Returns:
        PositionSize with recommendations
    """
    calc = PositionCalculator()
    return calc.calculate_position(veto_result, current_price, portfolio_value, atr, regime)
