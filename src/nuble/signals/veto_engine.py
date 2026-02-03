"""
NUBLE ELITE: Institutional Veto Engine

Implements the institutional veto system where:
- Weekly signal has VETO POWER over all other timeframes
- Daily confirms direction
- 4H triggers entry
- 1H fine-tunes (optional)

The Golden Rules:
1. NEVER trade against the Weekly trend
2. Daily must align with Weekly or reduce size by 75%
3. 4H triggers entry only after Weekly + Daily alignment
4. Conflicting signals = NO TRADE
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

from .timeframe_manager import TimeframeSignal, Timeframe, TimeframeManager, get_timeframe_manager

logger = logging.getLogger(__name__)


class VetoDecision(Enum):
    """Possible veto decisions."""
    APPROVED = "approved"           # All systems go
    APPROVED_REDUCED = "approved_reduced"  # OK but reduce position
    WAITING = "waiting"             # Wait for alignment
    VETOED = "vetoed"               # Do not trade


@dataclass
class VetoResult:
    """
    Result of the veto check.
    
    Contains the decision, reasoning, and recommended position multiplier.
    """
    decision: VetoDecision
    can_trade: bool
    position_multiplier: float  # 0.0 to 1.0
    direction: int              # -1, 0, 1
    reason: str
    details: List[str]
    
    # Signal references
    weekly_signal: Optional[TimeframeSignal] = None
    daily_signal: Optional[TimeframeSignal] = None
    four_hour_signal: Optional[TimeframeSignal] = None
    hourly_signal: Optional[TimeframeSignal] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision": self.decision.value,
            "can_trade": self.can_trade,
            "position_multiplier": self.position_multiplier,
            "direction": self.direction,
            "reason": self.reason,
            "details": self.details,
            "weekly": self.weekly_signal.to_dict() if self.weekly_signal else None,
            "daily": self.daily_signal.to_dict() if self.daily_signal else None,
            "4h": self.four_hour_signal.to_dict() if self.four_hour_signal else None,
            "1h": self.hourly_signal.to_dict() if self.hourly_signal else None,
        }


class VetoEngine:
    """
    Institutional-grade veto engine for multi-timeframe trading.
    
    Implements strict hierarchy where higher timeframes have absolute
    veto power over lower timeframes.
    
    The core principle:
    - Weekly WRONG = Everything else is noise
    - Daily WRONG = 4H signals will fail
    - 4H WRONG = Bad entry, but recoverable
    """
    
    # Minimum freshness thresholds
    MIN_FRESHNESS_WEEKLY = 0.3    # At least 30% fresh
    MIN_FRESHNESS_DAILY = 0.4     # At least 40% fresh
    MIN_FRESHNESS_4H = 0.5        # At least 50% fresh
    MIN_FRESHNESS_1H = 0.5        # At least 50% fresh
    
    # Position multipliers for different scenarios
    MULT_FULL_ALIGNMENT = 1.0     # All timeframes aligned
    MULT_WEEKLY_DAILY_ONLY = 0.75 # Weekly + Daily, waiting for 4H
    MULT_WEEKLY_NEUTRAL = 0.50    # Weekly is neutral
    MULT_COUNTER_TREND = 0.25     # Counter-trend setup
    MULT_CONFLICT = 0.0           # Conflicting signals
    
    def __init__(self, timeframe_manager: Optional[TimeframeManager] = None):
        """
        Initialize VetoEngine.
        
        Args:
            timeframe_manager: Optional TimeframeManager instance
        """
        self.tf_manager = timeframe_manager or get_timeframe_manager()
        logger.info("VetoEngine initialized")
    
    def check_veto(
        self,
        symbol: str,
        weekly: Optional[TimeframeSignal] = None,
        daily: Optional[TimeframeSignal] = None,
        four_hour: Optional[TimeframeSignal] = None,
        hourly: Optional[TimeframeSignal] = None,
    ) -> VetoResult:
        """
        Check if a trade should be vetoed.
        
        Can accept signals directly or fetch from TimeframeManager.
        
        Args:
            symbol: Asset symbol
            weekly: Weekly signal (optional, fetched if not provided)
            daily: Daily signal (optional, fetched if not provided)
            four_hour: 4H signal (optional, fetched if not provided)
            hourly: 1H signal (optional, fetched if not provided)
            
        Returns:
            VetoResult with decision and details
        """
        symbol = symbol.upper()
        details = []
        
        # Fetch signals if not provided
        if weekly is None:
            weekly = self.tf_manager.get_signal(symbol, Timeframe.WEEKLY)
        if daily is None:
            daily = self.tf_manager.get_signal(symbol, Timeframe.DAILY)
        if four_hour is None:
            four_hour = self.tf_manager.get_signal(symbol, Timeframe.FOUR_HOUR)
        if hourly is None:
            hourly = self.tf_manager.get_signal(symbol, Timeframe.ONE_HOUR)
        
        # Log available signals
        available = []
        if weekly:
            available.append(f"1W:{weekly.action}")
        if daily:
            available.append(f"1D:{daily.action}")
        if four_hour:
            available.append(f"4H:{four_hour.action}")
        if hourly:
            available.append(f"1H:{hourly.action}")
        
        details.append(f"Available signals: {', '.join(available) if available else 'None'}")
        
        # ========== RULE 1: Must have at least one signal ==========
        if not any([weekly, daily, four_hour]):
            return VetoResult(
                decision=VetoDecision.WAITING,
                can_trade=False,
                position_multiplier=0.0,
                direction=0,
                reason="No signals available - waiting for data",
                details=details,
            )
        
        # ========== RULE 2: Weekly signal has VETO POWER ==========
        if weekly is not None:
            if not weekly.is_fresh:
                details.append(f"⚠️ Weekly signal is stale (freshness: {weekly.freshness:.0%})")
                # Stale weekly = reduce confidence
            else:
                details.append(f"✅ Weekly signal: {weekly.action} (fresh: {weekly.freshness:.0%})")
            
            # Weekly NEUTRAL = reduced trading
            if weekly.direction == 0:
                details.append("📊 Weekly is NEUTRAL - reduced position sizes allowed")
                return self._handle_weekly_neutral(
                    weekly, daily, four_hour, hourly, details
                )
            
            # Weekly has direction - this is the MASTER direction
            master_direction = weekly.direction
            details.append(f"🎯 Master direction from Weekly: {'LONG' if master_direction > 0 else 'SHORT'}")
            
        else:
            # No weekly signal - can still trade but with caution
            details.append("⚠️ No weekly signal - using daily as primary")
            
            if daily is not None and daily.is_fresh:
                master_direction = daily.direction
            elif four_hour is not None and four_hour.is_fresh:
                master_direction = four_hour.direction
            else:
                return VetoResult(
                    decision=VetoDecision.WAITING,
                    can_trade=False,
                    position_multiplier=0.0,
                    direction=0,
                    reason="No fresh higher timeframe signals",
                    details=details,
                )
        
        # ========== RULE 3: Daily must confirm or reduce size ==========
        if daily is not None:
            if not daily.is_fresh:
                details.append(f"⚠️ Daily signal is stale (freshness: {daily.freshness:.0%})")
            else:
                if daily.direction == master_direction:
                    details.append(f"✅ Daily CONFIRMS weekly direction: {daily.action}")
                elif daily.direction == 0:
                    details.append("📊 Daily is NEUTRAL")
                else:
                    details.append(f"⚠️ Daily CONFLICTS with weekly: {daily.action}")
        
        # ========== RULE 4: Check for conflicting signals ==========
        conflict_result = self._check_conflicts(
            weekly, daily, four_hour, master_direction, details
        )
        if conflict_result is not None:
            return conflict_result
        
        # ========== RULE 5: Determine position multiplier ==========
        return self._calculate_position(
            weekly, daily, four_hour, hourly, master_direction, details
        )
    
    def _handle_weekly_neutral(
        self,
        weekly: TimeframeSignal,
        daily: Optional[TimeframeSignal],
        four_hour: Optional[TimeframeSignal],
        hourly: Optional[TimeframeSignal],
        details: List[str]
    ) -> VetoResult:
        """Handle the case when weekly is neutral."""
        
        # Weekly neutral = can trade in either direction but with reduced size
        if daily is not None and daily.is_fresh and daily.direction != 0:
            direction = daily.direction
            details.append(f"Using daily direction: {'LONG' if direction > 0 else 'SHORT'}")
            
            # Check if 4H aligns with daily
            if four_hour is not None and four_hour.is_fresh:
                if four_hour.direction == direction:
                    details.append("✅ 4H aligns with daily")
                    return VetoResult(
                        decision=VetoDecision.APPROVED_REDUCED,
                        can_trade=True,
                        position_multiplier=self.MULT_WEEKLY_NEUTRAL,
                        direction=direction,
                        reason="Weekly neutral - trading with daily/4H alignment at 50% size",
                        details=details,
                        weekly_signal=weekly,
                        daily_signal=daily,
                        four_hour_signal=four_hour,
                        hourly_signal=hourly,
                    )
                else:
                    details.append("⚠️ 4H conflicts with daily")
                    return VetoResult(
                        decision=VetoDecision.WAITING,
                        can_trade=False,
                        position_multiplier=0.0,
                        direction=0,
                        reason="Waiting for 4H to align with daily",
                        details=details,
                        weekly_signal=weekly,
                        daily_signal=daily,
                        four_hour_signal=four_hour,
                        hourly_signal=hourly,
                    )
            else:
                # No 4H signal, trade with daily only
                return VetoResult(
                    decision=VetoDecision.APPROVED_REDUCED,
                    can_trade=True,
                    position_multiplier=self.MULT_WEEKLY_NEUTRAL * 0.75,  # 37.5%
                    direction=direction,
                    reason="Weekly neutral, daily has direction - reduced size",
                    details=details,
                    weekly_signal=weekly,
                    daily_signal=daily,
                    four_hour_signal=four_hour,
                    hourly_signal=hourly,
                )
        
        # No clear direction
        return VetoResult(
            decision=VetoDecision.WAITING,
            can_trade=False,
            position_multiplier=0.0,
            direction=0,
            reason="Weekly neutral and no clear daily direction",
            details=details,
            weekly_signal=weekly,
            daily_signal=daily,
            four_hour_signal=four_hour,
            hourly_signal=hourly,
        )
    
    def _check_conflicts(
        self,
        weekly: Optional[TimeframeSignal],
        daily: Optional[TimeframeSignal],
        four_hour: Optional[TimeframeSignal],
        master_direction: int,
        details: List[str]
    ) -> Optional[VetoResult]:
        """
        Check for conflicting signals that would veto the trade.
        
        Returns VetoResult if trade should be vetoed, None otherwise.
        """
        
        # NEVER trade against weekly
        if weekly and weekly.is_fresh and weekly.direction != 0:
            
            # Daily conflicts with weekly
            if daily and daily.is_fresh and daily.direction != 0:
                if daily.direction != weekly.direction:
                    
                    # 4H also conflicts = HARD VETO
                    if four_hour and four_hour.is_fresh and four_hour.direction != 0:
                        if four_hour.direction == daily.direction:
                            details.append("🚫 VETO: Daily AND 4H against weekly trend")
                            return VetoResult(
                                decision=VetoDecision.VETOED,
                                can_trade=False,
                                position_multiplier=0.0,
                                direction=0,
                                reason="NEVER trade against weekly trend when daily and 4H also oppose",
                                details=details,
                                weekly_signal=weekly,
                                daily_signal=daily,
                                four_hour_signal=four_hour,
                            )
                        elif four_hour.direction == weekly.direction:
                            # 4H agrees with weekly, daily is the odd one out
                            # This could be a pullback opportunity
                            details.append("📊 Daily opposes weekly but 4H aligns - possible pullback")
                            # Continue to position calculation
                    else:
                        # Daily opposes weekly, no 4H signal
                        details.append("⚠️ Daily opposes weekly - waiting for 4H confirmation")
                        return VetoResult(
                            decision=VetoDecision.WAITING,
                            can_trade=False,
                            position_multiplier=0.0,
                            direction=weekly.direction,
                            reason="Waiting for 4H to confirm weekly direction",
                            details=details,
                            weekly_signal=weekly,
                            daily_signal=daily,
                            four_hour_signal=four_hour,
                        )
            
            # 4H conflicts with weekly (no daily conflict)
            if four_hour and four_hour.is_fresh and four_hour.direction != 0:
                if four_hour.direction != weekly.direction:
                    if daily is None or not daily.is_fresh or daily.direction == weekly.direction:
                        # 4H opposes weekly, but daily aligns
                        details.append("⚠️ 4H opposes weekly - waiting for alignment")
                        return VetoResult(
                            decision=VetoDecision.WAITING,
                            can_trade=False,
                            position_multiplier=0.0,
                            direction=weekly.direction,
                            reason="Waiting for 4H to align with higher timeframes",
                            details=details,
                            weekly_signal=weekly,
                            daily_signal=daily,
                            four_hour_signal=four_hour,
                        )
        
        return None  # No veto
    
    def _calculate_position(
        self,
        weekly: Optional[TimeframeSignal],
        daily: Optional[TimeframeSignal],
        four_hour: Optional[TimeframeSignal],
        hourly: Optional[TimeframeSignal],
        direction: int,
        details: List[str]
    ) -> VetoResult:
        """Calculate the final position multiplier."""
        
        multiplier = 1.0
        alignment_count = 0
        total_count = 0
        
        # Check each timeframe alignment
        if weekly and weekly.is_fresh:
            total_count += 1
            if weekly.direction == direction:
                alignment_count += 1
                if weekly.is_strong:
                    multiplier *= 1.1  # Boost for strong weekly
            elif weekly.direction != 0:
                multiplier *= 0.5  # Penalty for weekly opposition
        
        if daily and daily.is_fresh:
            total_count += 1
            if daily.direction == direction:
                alignment_count += 1
                if daily.is_strong:
                    multiplier *= 1.05
            elif daily.direction != 0:
                multiplier *= 0.7
        
        if four_hour and four_hour.is_fresh:
            total_count += 1
            if four_hour.direction == direction:
                alignment_count += 1
                if four_hour.is_strong:
                    multiplier *= 1.05
            elif four_hour.direction != 0:
                multiplier *= 0.8
        
        # Calculate alignment ratio
        if total_count > 0:
            alignment_ratio = alignment_count / total_count
        else:
            alignment_ratio = 0.0
        
        # Determine decision and final multiplier
        if alignment_count == total_count and total_count >= 2:
            # Perfect alignment
            decision = VetoDecision.APPROVED
            final_multiplier = min(1.0, multiplier)
            reason = f"✅ Perfect alignment ({alignment_count}/{total_count} timeframes)"
            details.append(reason)
            
        elif alignment_ratio >= 0.66:
            # Good alignment
            decision = VetoDecision.APPROVED
            final_multiplier = min(0.85, multiplier * 0.85)
            reason = f"✅ Good alignment ({alignment_count}/{total_count} timeframes)"
            details.append(reason)
            
        elif alignment_ratio >= 0.5:
            # Partial alignment
            decision = VetoDecision.APPROVED_REDUCED
            final_multiplier = min(0.5, multiplier * 0.5)
            reason = f"📊 Partial alignment ({alignment_count}/{total_count}) - reduced size"
            details.append(reason)
            
        else:
            # Poor alignment
            decision = VetoDecision.WAITING
            final_multiplier = 0.0
            reason = f"⚠️ Poor alignment ({alignment_count}/{total_count}) - waiting"
            details.append(reason)
        
        # Apply confirmation bonus
        if four_hour and four_hour.confirmations >= 8:
            final_multiplier *= 1.1
            details.append(f"📈 Confirmation bonus: {four_hour.confirmations} confirmations")
        
        # Apply freshness penalty if applicable
        avg_freshness = 0.0
        fresh_count = 0
        for sig in [weekly, daily, four_hour]:
            if sig:
                avg_freshness += sig.freshness
                fresh_count += 1
        
        if fresh_count > 0:
            avg_freshness /= fresh_count
            if avg_freshness < 0.5:
                final_multiplier *= avg_freshness * 2  # Scale down for stale signals
                details.append(f"⏳ Freshness adjustment: {avg_freshness:.0%} average")
        
        # Cap final multiplier
        final_multiplier = max(0.0, min(1.0, final_multiplier))
        
        details.append(f"💰 Position multiplier: {final_multiplier:.0%}")
        
        return VetoResult(
            decision=decision,
            can_trade=decision in [VetoDecision.APPROVED, VetoDecision.APPROVED_REDUCED],
            position_multiplier=final_multiplier,
            direction=direction,
            reason=reason,
            details=details,
            weekly_signal=weekly,
            daily_signal=daily,
            four_hour_signal=four_hour,
            hourly_signal=hourly,
        )
    
    def get_trade_permission(self, symbol: str) -> Tuple[bool, str, float]:
        """
        Simple interface to check if trading is permitted.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Tuple of (can_trade, reason, position_multiplier)
        """
        result = self.check_veto(symbol)
        return result.can_trade, result.reason, result.position_multiplier


# Convenience function
def check_veto(symbol: str) -> VetoResult:
    """
    Quick check if a trade should be vetoed.
    
    Args:
        symbol: Asset symbol
        
    Returns:
        VetoResult with decision and details
    """
    engine = VetoEngine()
    return engine.check_veto(symbol)
