"""
Tier 2 Escalation Detector
===========================

Determines when a Tier 1 decision should be escalated to Tier 2.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .config import Tier2Config, EscalationReason, DEFAULT_CONFIG
from .schemas import Tier1DecisionPack


@dataclass
class EscalationResult:
    """Result of escalation detection."""
    should_escalate: bool
    reasons: List[str]
    priority: str  # "normal", "high", "critical"
    rationale: str


class EscalationDetector:
    """
    Detects when a decision should be escalated to Tier 2.
    
    Escalation triggers:
    1. Low confidence (< threshold)
    2. High confidence (> threshold, check overconfidence)
    3. High volatility (VIX or ATR)
    4. Signal conflicts
    5. Earnings proximity
    6. Regime transition signals
    7. Portfolio concentration concerns
    8. Data quality issues
    """
    
    def __init__(self, config: Tier2Config = None):
        self.config = config or DEFAULT_CONFIG
    
    def detect(
        self,
        tier1_pack: Tier1DecisionPack,
        portfolio_snapshot: Optional[Dict] = None,
    ) -> EscalationResult:
        """
        Detect if Tier 1 decision should be escalated.
        
        Args:
            tier1_pack: The Tier 1 decision pack
            portfolio_snapshot: Optional portfolio context
            
        Returns:
            EscalationResult with decision and reasons
        """
        reasons = []
        priority = "normal"
        
        # 1. Confidence checks
        if tier1_pack.confidence < self.config.min_confidence_for_escalation:
            reasons.append(EscalationReason.LOW_CONFIDENCE.value)
        
        if tier1_pack.confidence > self.config.max_confidence_for_escalation:
            reasons.append(EscalationReason.HIGH_CONFIDENCE_CHECK.value)
        
        # 2. Volatility checks
        if tier1_pack.vix >= self.config.escalation_volatility_threshold:
            reasons.append(EscalationReason.HIGH_VOLATILITY.value)
            if tier1_pack.vix >= 40:
                priority = "critical"
        
        if tier1_pack.atr_pct >= self.config.escalation_atr_pct_threshold:
            reasons.append(EscalationReason.HIGH_VOLATILITY.value)
        
        # 3. Signal conflict detection
        signal_conflict = self._check_signal_conflict(tier1_pack)
        if signal_conflict:
            reasons.append(EscalationReason.SIGNAL_CONFLICT.value)
            priority = max(priority, "high")
        
        # 4. Regime transition
        if self._check_regime_transition(tier1_pack):
            reasons.append(EscalationReason.REGIME_TRANSITION.value)
        
        # 5. Data quality
        if self._check_data_quality_issues(tier1_pack):
            reasons.append(EscalationReason.STALE_DATA.value)
        
        # 6. Sentiment conflict
        if self._check_sentiment_conflict(tier1_pack):
            reasons.append(EscalationReason.NEWS_SENTIMENT_CONFLICT.value)
        
        # 7. Portfolio checks (if available)
        if portfolio_snapshot:
            portfolio_reasons = self._check_portfolio(tier1_pack, portfolio_snapshot)
            reasons.extend(portfolio_reasons)
        
        # Deduplicate
        reasons = list(set(reasons))
        
        should_escalate = len(reasons) > 0
        
        rationale = self._build_rationale(reasons, tier1_pack)
        
        return EscalationResult(
            should_escalate=should_escalate,
            reasons=reasons,
            priority=priority,
            rationale=rationale,
        )
    
    def _check_signal_conflict(self, pack: Tier1DecisionPack) -> bool:
        """Check for multi-timeframe signal conflicts."""
        signals = []
        
        if pack.weekly_signal:
            action = pack.weekly_signal.get("action", "").upper()
            if action in ["BUY", "STRONG_BUY"]:
                signals.append(("weekly", 1))
            elif action in ["SELL", "STRONG_SELL"]:
                signals.append(("weekly", -1))
        
        if pack.daily_signal:
            action = pack.daily_signal.get("action", "").upper()
            if action in ["BUY", "STRONG_BUY"]:
                signals.append(("daily", 1))
            elif action in ["SELL", "STRONG_SELL"]:
                signals.append(("daily", -1))
        
        if pack.h4_signal:
            action = pack.h4_signal.get("action", "").upper()
            if action in ["BUY", "STRONG_BUY"]:
                signals.append(("h4", 1))
            elif action in ["SELL", "STRONG_SELL"]:
                signals.append(("h4", -1))
        
        if len(signals) < 2:
            return False
        
        # Check for conflicts (especially weekly vs others)
        directions = [s[1] for s in signals]
        if 1 in directions and -1 in directions:
            return True
        
        return False
    
    def _check_regime_transition(self, pack: Tier1DecisionPack) -> bool:
        """Check for regime transition signals."""
        # Low regime confidence suggests instability
        if pack.regime_confidence < 60:
            return True
        
        # VIX state changes
        if pack.vix_state in ["HIGH", "EXTREME"]:
            return True
        
        return False
    
    def _check_data_quality_issues(self, pack: Tier1DecisionPack) -> bool:
        """Check for data quality issues."""
        # Stale data
        if pack.data_age_seconds > 300:  # 5 minutes
            return True
        
        # Missing feeds
        if pack.missing_feeds and len(pack.missing_feeds) > 0:
            return True
        
        return False
    
    def _check_sentiment_conflict(self, pack: Tier1DecisionPack) -> bool:
        """Check if sentiment conflicts with technical direction."""
        # Bullish technicals + negative sentiment
        if pack.direction == "BULLISH" and pack.sentiment_score < -0.3:
            return True
        
        # Bearish technicals + positive sentiment
        if pack.direction == "BEARISH" and pack.sentiment_score > 0.3:
            return True
        
        return False
    
    def _check_portfolio(
        self,
        pack: Tier1DecisionPack,
        portfolio: Dict,
    ) -> List[str]:
        """Check portfolio-level concerns."""
        reasons = []
        
        # Position concentration
        if pack.current_position > 8.0:  # Already large position
            reasons.append(EscalationReason.PORTFOLIO_CONCENTRATION.value)
        
        # Sector exposure
        if pack.sector_exposure_pct > 25.0:
            reasons.append(EscalationReason.PORTFOLIO_CONCENTRATION.value)
        
        # Drawdown check
        drawdown = portfolio.get("current_drawdown_pct", 0)
        if drawdown > 10:
            reasons.append(EscalationReason.DRAWDOWN_ELEVATED.value)
        
        return reasons
    
    def _build_rationale(
        self,
        reasons: List[str],
        pack: Tier1DecisionPack,
    ) -> str:
        """Build human-readable escalation rationale."""
        if not reasons:
            return "No escalation needed"
        
        parts = [f"Escalating {pack.symbol} for:"]
        
        for reason in reasons:
            if reason == EscalationReason.LOW_CONFIDENCE.value:
                parts.append(f"- Low confidence ({pack.confidence:.0f}%)")
            elif reason == EscalationReason.HIGH_CONFIDENCE_CHECK.value:
                parts.append(f"- High confidence check ({pack.confidence:.0f}%)")
            elif reason == EscalationReason.HIGH_VOLATILITY.value:
                parts.append(f"- High volatility (VIX: {pack.vix:.1f}, ATR%: {pack.atr_pct:.2f}%)")
            elif reason == EscalationReason.SIGNAL_CONFLICT.value:
                parts.append("- Signal conflict across timeframes")
            elif reason == EscalationReason.REGIME_TRANSITION.value:
                parts.append(f"- Regime instability ({pack.regime}, conf: {pack.regime_confidence:.0f}%)")
            elif reason == EscalationReason.STALE_DATA.value:
                parts.append(f"- Data quality issues (age: {pack.data_age_seconds:.0f}s)")
            elif reason == EscalationReason.NEWS_SENTIMENT_CONFLICT.value:
                parts.append(f"- Sentiment conflict (score: {pack.sentiment_score:.2f})")
            else:
                parts.append(f"- {reason}")
        
        return " ".join(parts)
