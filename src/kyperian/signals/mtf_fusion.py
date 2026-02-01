"""
KYPERIAN ELITE: Multi-Timeframe Signal Fusion Engine

The brain that combines all components:
- TimeframeManager: Stores and tracks signals
- VetoEngine: Applies institutional veto rules
- PositionCalculator: Calculates optimal sizing

This is the main interface for the multi-timeframe system.

Usage:
    engine = MTFFusionEngine()
    
    # Add signals from webhooks
    engine.add_signal(parse_mtf_webhook(payload))
    
    # Generate trading decision
    decision = engine.generate_decision("ETHUSD", current_price=2340.0)
    
    if decision.can_trade:
        print(f"Execute {decision.action} at {decision.entry_price}")
        print(f"Size: ${decision.position.dollar_amount:,.2f}")
        print(f"Stop: ${decision.position.stop_loss_price:,.2f}")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from .timeframe_manager import (
    TimeframeManager, TimeframeSignal, Timeframe,
    get_timeframe_manager, parse_mtf_webhook
)
from .veto_engine import VetoEngine, VetoResult, VetoDecision
from .position_calculator import PositionCalculator, PositionSize

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Trading signal strength levels."""
    NONE = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class TradingDecision:
    """
    Complete trading decision with full context.
    
    This is the output of the MTF fusion system.
    """
    # Core decision
    symbol: str
    timestamp: datetime
    can_trade: bool
    action: str                       # "BUY", "SELL", "HOLD"
    direction: int                    # -1, 0, 1
    
    # Strength and confidence
    strength: SignalStrength
    confidence: float                 # 0-1
    
    # Position sizing
    position: Optional[PositionSize]
    entry_price: Optional[float]
    
    # Veto information
    veto_result: VetoResult
    
    # Multi-timeframe breakdown
    weekly_summary: str
    daily_summary: str
    four_hour_summary: str
    hourly_summary: str
    
    # Reasoning
    reasoning: List[str]
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def strength_label(self) -> str:
        """Get human-readable strength label."""
        labels = {
            SignalStrength.NONE: "⚪ NO SIGNAL",
            SignalStrength.WEAK: "🟡 WEAK",
            SignalStrength.MODERATE: "🟠 MODERATE",
            SignalStrength.STRONG: "🟢 STRONG",
            SignalStrength.VERY_STRONG: "🟢🟢 VERY STRONG",
        }
        return labels.get(self.strength, "UNKNOWN")
    
    @property
    def action_label(self) -> str:
        """Get action with emoji."""
        if self.action == "BUY":
            return f"📈 {self.strength_label} BUY"
        elif self.action == "SELL":
            return f"📉 {self.strength_label} SELL"
        else:
            return "⏸️ HOLD"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "can_trade": self.can_trade,
            "action": self.action,
            "action_label": self.action_label,
            "direction": self.direction,
            "strength": self.strength.name,
            "strength_label": self.strength_label,
            "confidence": round(self.confidence, 3),
            "position": self.position.to_dict() if self.position else None,
            "entry_price": self.entry_price,
            "veto": self.veto_result.to_dict(),
            "timeframes": {
                "weekly": self.weekly_summary,
                "daily": self.daily_summary,
                "4h": self.four_hour_summary,
                "1h": self.hourly_summary,
            },
            "reasoning": self.reasoning,
            "generated_at": self.generated_at.isoformat(),
        }
    
    def __str__(self) -> str:
        """Human-readable string."""
        if not self.can_trade:
            return f"{self.symbol}: HOLD - {self.veto_result.reason}"
        
        pos_str = ""
        if self.position:
            pos_str = f" (${self.position.dollar_amount:,.0f})"
        
        return f"{self.symbol}: {self.action_label}{pos_str} @ ${self.entry_price:,.2f}" if self.entry_price else f"{self.symbol}: {self.action_label}"


class MTFFusionEngine:
    """
    Multi-Timeframe Fusion Engine.
    
    Combines:
    - TimeframeManager: Signal storage and tracking
    - VetoEngine: Institutional veto logic
    - PositionCalculator: Kelly-based sizing
    
    This is the main entry point for the MTF system.
    """
    
    def __init__(
        self,
        timeframe_manager: Optional[TimeframeManager] = None,
        portfolio_value: float = 100000,
        max_risk: float = 0.02,
        max_position: float = 0.10,
    ):
        """
        Initialize MTFFusionEngine.
        
        Args:
            timeframe_manager: Optional TimeframeManager (creates new if not provided)
            portfolio_value: Portfolio value for position sizing
            max_risk: Maximum risk per trade (default 2%)
            max_position: Maximum position size (default 10%)
        """
        self.tf_manager = timeframe_manager or get_timeframe_manager()
        self.veto_engine = VetoEngine(self.tf_manager)
        self.position_calc = PositionCalculator(
            max_risk=max_risk,
            max_position=max_position
        )
        self.portfolio_value = portfolio_value
        
        # Track decisions for learning
        self.decision_history: List[TradingDecision] = []
        
        logger.info(
            f"MTFFusionEngine initialized: portfolio=${portfolio_value:,.0f}, "
            f"max_risk={max_risk:.1%}, max_position={max_position:.1%}"
        )
    
    def add_signal(self, signal: TimeframeSignal) -> None:
        """
        Add a signal to the manager.
        
        Args:
            signal: TimeframeSignal to add
        """
        self.tf_manager.add_signal(signal)
        logger.info(f"Added signal: {signal}")
    
    def add_from_webhook(self, payload: Dict[str, Any]) -> TimeframeSignal:
        """
        Parse webhook payload and add signal.
        
        Args:
            payload: Webhook JSON payload
            
        Returns:
            Parsed TimeframeSignal
        """
        signal = parse_mtf_webhook(payload)
        self.add_signal(signal)
        return signal
    
    def get_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Get all current signals for a symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dictionary with all timeframe signals
        """
        signals = self.tf_manager.get_all_signals(symbol)
        return {
            tf.value: sig.to_dict() for tf, sig in signals.items()
        }
    
    def get_alignment(self, symbol: str) -> Dict[str, Any]:
        """
        Get alignment information for a symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Alignment dictionary
        """
        return self.tf_manager.get_alignment(symbol)
    
    def check_veto(self, symbol: str) -> VetoResult:
        """
        Check if trading is vetoed for a symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            VetoResult
        """
        return self.veto_engine.check_veto(symbol)
    
    def generate_decision(
        self,
        symbol: str,
        current_price: Optional[float] = None,
        atr: Optional[float] = None,
        regime: str = "NORMAL",
        portfolio_value: Optional[float] = None
    ) -> TradingDecision:
        """
        Generate a complete trading decision.
        
        This is the main method that combines all components.
        
        Args:
            symbol: Asset symbol
            current_price: Current price (required for position sizing)
            atr: Average True Range (optional, estimated if not provided)
            regime: Market regime
            portfolio_value: Portfolio value (uses default if not provided)
            
        Returns:
            TradingDecision with full analysis
        """
        symbol = symbol.upper()
        timestamp = datetime.now()
        reasoning = []
        
        # Use portfolio value
        port_value = portfolio_value or self.portfolio_value
        
        # Step 1: Get all signals
        weekly, daily, four_hour, hourly = self.tf_manager.get_cascade(symbol)
        
        # Build signal summaries
        weekly_summary = self._summarize_signal(weekly, "Weekly")
        daily_summary = self._summarize_signal(daily, "Daily")
        four_hour_summary = self._summarize_signal(four_hour, "4H")
        hourly_summary = self._summarize_signal(hourly, "1H")
        
        reasoning.append(f"📊 Weekly: {weekly_summary}")
        reasoning.append(f"📊 Daily: {daily_summary}")
        reasoning.append(f"📊 4H: {four_hour_summary}")
        if hourly:
            reasoning.append(f"📊 1H: {hourly_summary}")
        
        # Step 2: Apply veto logic
        veto_result = self.veto_engine.check_veto(symbol)
        reasoning.extend(veto_result.details)
        
        if not veto_result.can_trade:
            decision = TradingDecision(
                symbol=symbol,
                timestamp=timestamp,
                can_trade=False,
                action="HOLD",
                direction=0,
                strength=SignalStrength.NONE,
                confidence=0.0,
                position=None,
                entry_price=None,
                veto_result=veto_result,
                weekly_summary=weekly_summary,
                daily_summary=daily_summary,
                four_hour_summary=four_hour_summary,
                hourly_summary=hourly_summary,
                reasoning=reasoning,
            )
            self._track_decision(decision)
            return decision
        
        # Step 3: Calculate position size
        entry_price = current_price
        if entry_price is None and four_hour:
            entry_price = four_hour.price
        if entry_price is None and daily:
            entry_price = daily.price
        if entry_price is None:
            entry_price = 0.0
        
        position = self.position_calc.calculate_position(
            veto_result=veto_result,
            current_price=entry_price,
            portfolio_value=port_value,
            atr=atr,
            regime=regime
        )
        
        reasoning.append(f"💰 Position size: {position.recommended_size:.1%} (${position.dollar_amount:,.0f})")
        reasoning.append(f"🎯 Entry: ${entry_price:,.2f}")
        reasoning.append(f"🛑 Stop: ${position.stop_loss_price:,.2f} ({position.stop_loss_pct:.1%})")
        if position.take_profit_prices:
            reasoning.append(f"✅ TP1: ${position.take_profit_prices[0]:,.2f} ({position.take_profit_pcts[0]:.1%})")
            if len(position.take_profit_prices) > 1:
                reasoning.append(f"✅ TP2: ${position.take_profit_prices[1]:,.2f} ({position.take_profit_pcts[1]:.1%})")
        
        # Step 4: Determine action and strength
        action = "BUY" if veto_result.direction > 0 else "SELL" if veto_result.direction < 0 else "HOLD"
        strength = self._calculate_strength(position, veto_result)
        
        # Step 5: Build final decision
        decision = TradingDecision(
            symbol=symbol,
            timestamp=timestamp,
            can_trade=True,
            action=action,
            direction=veto_result.direction,
            strength=strength,
            confidence=position.confidence,
            position=position,
            entry_price=entry_price,
            veto_result=veto_result,
            weekly_summary=weekly_summary,
            daily_summary=daily_summary,
            four_hour_summary=four_hour_summary,
            hourly_summary=hourly_summary,
            reasoning=reasoning,
        )
        
        self._track_decision(decision)
        
        logger.info(f"Generated decision: {decision}")
        
        return decision
    
    def _summarize_signal(
        self, 
        signal: Optional[TimeframeSignal], 
        name: str
    ) -> str:
        """Create human-readable signal summary."""
        if signal is None:
            return f"No {name} signal"
        
        if signal.is_expired:
            return f"EXPIRED ({signal.action})"
        
        strength = "💪" if signal.is_strong else ""
        fresh = f"fresh={signal.freshness:.0%}"
        conf = f"conf={signal.confirmations}" if signal.confirmations > 1 else ""
        
        parts = [signal.action, strength, fresh, conf]
        return " ".join(p for p in parts if p)
    
    def _calculate_strength(
        self, 
        position: PositionSize, 
        veto: VetoResult
    ) -> SignalStrength:
        """Calculate signal strength from position and veto."""
        # Use confidence and position size
        score = position.confidence * 0.5 + position.recommended_size * 5
        
        # Boost for full alignment
        if veto.decision == VetoDecision.APPROVED:
            score += 0.5
        
        if score >= 1.5:
            return SignalStrength.VERY_STRONG
        elif score >= 1.0:
            return SignalStrength.STRONG
        elif score >= 0.6:
            return SignalStrength.MODERATE
        elif score >= 0.3:
            return SignalStrength.WEAK
        else:
            return SignalStrength.NONE
    
    def _track_decision(self, decision: TradingDecision) -> None:
        """Track decision for history."""
        self.decision_history.append(decision)
        
        # Keep last 1000 decisions
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all tracked symbols."""
        return {
            "symbols": self.tf_manager.get_status(),
            "total_signals": sum(
                len(self.tf_manager.get_all_signals(s)) 
                for s in self.tf_manager.get_tracked_symbols()
            ),
            "recent_decisions": len(self.decision_history),
            "portfolio_value": self.portfolio_value,
        }
    
    def cleanup(self) -> int:
        """Cleanup expired signals."""
        return self.tf_manager.cleanup_expired()


# Global instance
_mtf_engine: Optional[MTFFusionEngine] = None


def get_mtf_engine(portfolio_value: float = 100000) -> MTFFusionEngine:
    """Get the global MTFFusionEngine instance."""
    global _mtf_engine
    if _mtf_engine is None:
        _mtf_engine = MTFFusionEngine(portfolio_value=portfolio_value)
    return _mtf_engine


def generate_mtf_decision(
    symbol: str,
    current_price: Optional[float] = None,
    regime: str = "NORMAL"
) -> TradingDecision:
    """
    Generate a multi-timeframe trading decision.
    
    Convenience function using global engine.
    
    Args:
        symbol: Asset symbol
        current_price: Current price
        regime: Market regime
        
    Returns:
        TradingDecision
    """
    engine = get_mtf_engine()
    return engine.generate_decision(symbol, current_price, regime=regime)
