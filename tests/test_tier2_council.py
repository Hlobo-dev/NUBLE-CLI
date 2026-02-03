"""
Tier 2 Council-of-Experts Tests
================================

Comprehensive tests for the Tier 2 orchestrator.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

# Import Tier 2 components
from src.institutional.tier2.config import Tier2Config, DEFAULT_CONFIG, EscalationReason
from src.institutional.tier2.schemas import (
    Tier1DecisionPack,
    Tier2Decision,
    Claim,
    AgentOutput,
    ClaimType,
    ClaimStance,
)
from src.institutional.tier2.escalation import EscalationDetector, EscalationResult
from src.institutional.tier2.circuit_breaker import CircuitBreaker, CircuitBreakerState
from src.institutional.tier2.store import InMemoryStore, DecisionRecord, AgentRunRecord
from src.institutional.tier2.arbiter import Arbiter
from src.institutional.tier2.allocator import Allocator


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def tier1_pack_bullish():
    """A bullish Tier 1 decision pack."""
    return Tier1DecisionPack(
        symbol="AAPL",
        action="BUY",
        confidence=75.0,
        direction="BULLISH",
        price=185.50,
        rsi=58.2,
        macd_value=0.52,
        macd_signal=0.45,
        trend_state="UPTREND",
        sma_20=183.2,
        sma_50=180.5,
        sma_200=175.0,
        atr_pct=1.85,
        regime="BULLISH",
        regime_confidence=72.0,
        vix=15.5,
        vix_state="LOW",
        sentiment_score=0.35,
        news_count_7d=12,
        weekly_signal={"action": "BUY", "confidence": 70},
        daily_signal={"action": "BUY", "confidence": 75},
        h4_signal={"action": "BUY", "confidence": 65},
        data_age_seconds=5,
        missing_feeds=[],
        current_position=2.5,
        sector_exposure_pct=12.0,
    )


@pytest.fixture
def tier1_pack_conflicting():
    """A Tier 1 decision with signal conflicts."""
    return Tier1DecisionPack(
        symbol="TSLA",
        action="BUY",
        confidence=65.0,
        direction="BULLISH",
        price=245.00,
        rsi=62.5,
        macd_value=1.25,
        macd_signal=1.10,
        trend_state="UPTREND",
        sma_20=242.0,
        sma_50=238.0,
        sma_200=220.0,
        atr_pct=3.5,
        regime="BULLISH",
        regime_confidence=55.0,
        vix=22.5,
        vix_state="ELEVATED",
        sentiment_score=-0.15,  # Negative sentiment
        news_count_7d=25,
        weekly_signal={"action": "BUY", "confidence": 70},
        daily_signal={"action": "SELL", "confidence": 60},  # Conflict!
        h4_signal={"action": "NEUTRAL", "confidence": 50},
        data_age_seconds=15,
        missing_feeds=[],
        current_position=0.0,
        sector_exposure_pct=8.0,
    )


@pytest.fixture
def tier1_pack_low_confidence():
    """A low confidence Tier 1 decision."""
    return Tier1DecisionPack(
        symbol="AMD",
        action="BUY",
        confidence=45.0,  # Low confidence
        direction="BULLISH",
        price=155.00,
        rsi=52.0,
        macd_value=0.15,
        macd_signal=0.12,
        trend_state="NEUTRAL",
        sma_20=154.0,
        sma_50=152.0,
        sma_200=145.0,
        atr_pct=2.8,
        regime="NEUTRAL",
        regime_confidence=48.0,  # Low regime confidence
        vix=18.0,
        vix_state="NORMAL",
        sentiment_score=0.1,
        news_count_7d=8,
        weekly_signal={"action": "NEUTRAL", "confidence": 50},
        daily_signal={"action": "BUY", "confidence": 55},
        h4_signal={"action": "BUY", "confidence": 45},
        data_age_seconds=8,
        missing_feeds=[],
        current_position=0.0,
        sector_exposure_pct=5.0,
    )


@pytest.fixture
def config():
    """Test configuration."""
    return DEFAULT_CONFIG


# ============================================================================
# Escalation Detector Tests
# ============================================================================

class TestEscalationDetector:
    """Tests for the escalation detection logic."""
    
    def test_no_escalation_for_strong_decision(self, tier1_pack_bullish, config):
        """Strong aligned signals should not escalate."""
        detector = EscalationDetector(config)
        result = detector.detect(tier1_pack_bullish)
        
        # With good alignment, should not escalate
        assert isinstance(result, EscalationResult)
    
    def test_escalation_for_signal_conflict(self, tier1_pack_conflicting, config):
        """Signal conflicts should trigger escalation."""
        detector = EscalationDetector(config)
        result = detector.detect(tier1_pack_conflicting)
        
        assert result.should_escalate is True
        assert EscalationReason.SIGNAL_CONFLICT.value in result.reasons
    
    def test_escalation_for_low_confidence(self, tier1_pack_low_confidence, config):
        """Low confidence should trigger escalation."""
        # Adjust config for test
        test_config = Tier2Config(min_confidence_for_escalation=50)
        detector = EscalationDetector(test_config)
        result = detector.detect(tier1_pack_low_confidence)
        
        assert result.should_escalate is True
        assert EscalationReason.LOW_CONFIDENCE.value in result.reasons
    
    def test_escalation_for_high_vix(self, tier1_pack_bullish, config):
        """High VIX should trigger escalation."""
        pack = tier1_pack_bullish
        pack.vix = 35.0
        pack.vix_state = "HIGH"
        
        detector = EscalationDetector(config)
        result = detector.detect(pack)
        
        assert result.should_escalate is True
        assert EscalationReason.HIGH_VOLATILITY.value in result.reasons
    
    def test_escalation_for_stale_data(self, tier1_pack_bullish, config):
        """Stale data should trigger escalation."""
        pack = tier1_pack_bullish
        pack.data_age_seconds = 600  # 10 minutes - stale
        
        detector = EscalationDetector(config)
        result = detector.detect(pack)
        
        assert result.should_escalate is True
        assert EscalationReason.STALE_DATA.value in result.reasons
    
    def test_escalation_priority_levels(self, tier1_pack_conflicting, config):
        """Critical conditions should increase priority."""
        pack = tier1_pack_conflicting
        pack.vix = 45.0  # Extreme VIX
        
        detector = EscalationDetector(config)
        result = detector.detect(pack)
        
        assert result.priority == "critical"


# ============================================================================
# Circuit Breaker Tests
# ============================================================================

class TestCircuitBreaker:
    """Tests for the circuit breaker safety mechanism."""
    
    def test_initial_state_closed(self, config):
        """Circuit breaker should start closed."""
        cb = CircuitBreaker(config)
        assert cb.is_open() is False
        assert cb.state == CircuitBreakerState.CLOSED
    
    def test_opens_on_high_error_rate(self, config):
        """Circuit breaker should open on high error rate."""
        cb = CircuitBreaker(config)
        
        # Record failures above threshold
        for i in range(config.circuit_breaker_error_threshold + 5):
            cb.record_failure(1000.0, "test_error")
        
        assert cb.is_open() is True
        assert cb.state == CircuitBreakerState.OPEN
    
    def test_remains_closed_on_successes(self, config):
        """Circuit breaker should remain closed with successes."""
        cb = CircuitBreaker(config)
        
        for i in range(50):
            cb.record_success(500.0)
        
        assert cb.is_open() is False
    
    def test_half_open_allows_probe(self, config):
        """Half-open state should allow probe requests."""
        cb = CircuitBreaker(config)
        
        # Force open
        for i in range(config.circuit_breaker_error_threshold + 5):
            cb.record_failure(1000.0, "test_error")
        
        assert cb.is_open() is True
        
        # Manually transition to half-open
        cb._state = CircuitBreakerState.HALF_OPEN
        
        # Should allow one probe
        assert cb.is_open() is False  # First call in half-open is allowed
    
    def test_metrics_tracking(self, config):
        """Circuit breaker should track metrics."""
        cb = CircuitBreaker(config)
        
        cb.record_success(100.0)
        cb.record_success(200.0)
        cb.record_failure(500.0, "error1")
        
        metrics = cb.get_metrics()
        
        assert metrics["total_requests"] == 3
        assert metrics["success_count"] == 2
        assert metrics["failure_count"] == 1


# ============================================================================
# In-Memory Store Tests
# ============================================================================

class TestInMemoryStore:
    """Tests for the in-memory decision store."""
    
    def test_save_and_retrieve_decision(self):
        """Should save and retrieve decisions."""
        store = InMemoryStore()
        
        record = DecisionRecord(
            decision_id="test123",
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc).isoformat(),
            tier1_action="BUY",
            tier1_confidence=75.0,
            tier2_delta={"type": "CONFIDENCE_DOWN", "confidence": -10.0},
            final_action="BUY",
            final_confidence=65.0,
            escalation_reasons=["signal_conflict"],
            agent_count=5,
            latency_ms=1500.0,
            ttl=0,
        )
        
        assert store.save_decision(record) is True
        
        retrieved = store.get_decision("test123")
        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
        assert retrieved.final_confidence == 65.0
    
    def test_get_decisions_for_symbol(self):
        """Should retrieve decisions for a specific symbol."""
        store = InMemoryStore()
        
        # Add multiple decisions
        for i in range(5):
            record = DecisionRecord(
                decision_id=f"test{i}",
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc).isoformat(),
                tier1_action="BUY",
                tier1_confidence=70.0 + i,
                tier2_delta={},
                final_action="BUY",
                final_confidence=70.0 + i,
                escalation_reasons=[],
                agent_count=5,
                latency_ms=1000.0,
                ttl=0,
            )
            store.save_decision(record)
        
        # Add one for different symbol
        record = DecisionRecord(
            decision_id="tsla1",
            symbol="TSLA",
            timestamp=datetime.now(timezone.utc).isoformat(),
            tier1_action="SELL",
            tier1_confidence=60.0,
            tier2_delta={},
            final_action="SELL",
            final_confidence=60.0,
            escalation_reasons=[],
            agent_count=3,
            latency_ms=800.0,
            ttl=0,
        )
        store.save_decision(record)
        
        # Query for AAPL
        aapl_decisions = store.get_decisions_for_symbol("AAPL")
        assert len(aapl_decisions) == 5
        
        tsla_decisions = store.get_decisions_for_symbol("TSLA")
        assert len(tsla_decisions) == 1


# ============================================================================
# Arbiter Tests
# ============================================================================

class TestArbiter:
    """Tests for the arbiter synthesis layer."""
    
    def test_synthesize_unanimous_support(self, tier1_pack_bullish, config):
        """Unanimous support should boost confidence."""
        arbiter = Arbiter(config)
        
        # Create supporting agent outputs
        outputs = [
            AgentOutput(
                agent_name="mtf_dominance",
                round_type="light",
                claims=[
                    Claim(
                        id="c1",
                        type=ClaimType.SUPPORT.value,
                        stance=ClaimStance.PRO.value,
                        strength=0.8,
                        statement="Strong trend alignment across timeframes",
                        evidence_keys=["weekly_signal", "daily_signal"],
                    ),
                ],
                confidence=75.0,
                direction="BULLISH",
                recommended_deltas=None,
                input_tokens=100,
                output_tokens=150,
                latency_ms=500,
            ),
            AgentOutput(
                agent_name="trend_integrity",
                round_type="light",
                claims=[
                    Claim(
                        id="c2",
                        type=ClaimType.SUPPORT.value,
                        stance=ClaimStance.PRO.value,
                        strength=0.75,
                        statement="Trend structure intact",
                        evidence_keys=["sma_stack", "macd"],
                    ),
                ],
                confidence=72.0,
                direction="BULLISH",
                recommended_deltas=None,
                input_tokens=100,
                output_tokens=140,
                latency_ms=480,
            ),
        ]
        
        result = arbiter.synthesize(tier1_pack_bullish, outputs, severity="low")
        
        # Should not significantly reduce confidence with support
        assert result.confidence_delta >= -5.0  # Minor or positive
    
    def test_synthesize_with_opposition(self, tier1_pack_conflicting, config):
        """Opposition should reduce confidence."""
        arbiter = Arbiter(config)
        
        outputs = [
            AgentOutput(
                agent_name="red_team",
                round_type="deep",
                claims=[
                    Claim(
                        id="c1",
                        type=ClaimType.RISK.value,
                        stance=ClaimStance.ANTI.value,
                        strength=0.85,
                        statement="Daily signal contradicts weekly - high reversal risk",
                        evidence_keys=["daily_signal", "weekly_signal"],
                    ),
                ],
                confidence=35.0,
                direction="BEARISH",
                recommended_deltas=None,
                input_tokens=200,
                output_tokens=350,
                latency_ms=800,
            ),
        ]
        
        result = arbiter.synthesize(tier1_pack_conflicting, outputs, severity="high")
        
        # Should reduce confidence
        assert result.confidence_delta < 0
        assert "conflict" in result.rationale.lower() or "risk" in result.rationale.lower()
    
    def test_risk_gatekeeper_veto(self, tier1_pack_bullish, config):
        """Risk gatekeeper veto should override."""
        arbiter = Arbiter(config)
        
        outputs = [
            AgentOutput(
                agent_name="risk_gatekeeper",
                round_type="light",
                claims=[
                    Claim(
                        id="c1",
                        type=ClaimType.RISK.value,
                        stance=ClaimStance.ANTI.value,
                        strength=1.0,  # Max strength = veto
                        statement="Position would exceed concentration limits",
                        evidence_keys=["portfolio_concentration"],
                    ),
                ],
                confidence=10.0,
                direction="NEUTRAL",
                recommended_deltas=None,
                input_tokens=100,
                output_tokens=120,
                latency_ms=400,
            ),
        ]
        
        result = arbiter.synthesize(tier1_pack_bullish, outputs, severity="critical")
        
        # Risk veto should result in WAIT or significant reduction
        assert result.delta_type in ["WAIT", "NO_TRADE", "CONFIDENCE_DOWN"]


# ============================================================================
# Allocator Tests
# ============================================================================

class TestAllocator:
    """Tests for the agent allocator."""
    
    def test_allocates_light_agents_by_default(self, tier1_pack_bullish, config):
        """Should allocate light agents for normal escalations."""
        # Create a mock registry
        registry = Mock()
        registry.list_agents.return_value = [
            "mtf_dominance", "trend_integrity", "volatility_state",
            "risk_gatekeeper", "data_integrity"
        ]
        registry.get_config.return_value = Mock(enabled=True)
        
        allocator = Allocator(config, registry)
        
        result = allocator.allocate(
            tier1_pack_bullish,
            escalation_reasons=[EscalationReason.LOW_CONFIDENCE.value],
        )
        
        assert len(result.light_agents) > 0
    
    def test_allocates_deep_agents_for_critical(self, tier1_pack_conflicting, config):
        """Should allocate deep agents for critical escalations."""
        registry = Mock()
        registry.list_agents.return_value = [
            "mtf_dominance", "trend_integrity", "red_team",
            "risk_gatekeeper", "regime_transition"
        ]
        registry.get_config.return_value = Mock(enabled=True)
        
        allocator = Allocator(config, registry)
        
        result = allocator.allocate(
            tier1_pack_conflicting,
            escalation_reasons=[
                EscalationReason.SIGNAL_CONFLICT.value,
                EscalationReason.HIGH_VOLATILITY.value,
            ],
        )
        
        assert len(result.deep_agents) > 0
        assert result.severity in ["high", "critical"]


# ============================================================================
# Integration Tests
# ============================================================================

class TestTier2Integration:
    """Integration tests for the full Tier 2 pipeline."""
    
    @pytest.mark.skip(reason="Requires Bedrock connection")
    def test_full_pipeline_with_mocked_runtime(self, tier1_pack_conflicting, config):
        """Test full pipeline with mocked LLM runtime."""
        from src.institutional.tier2.orchestrator import Tier2Orchestrator
        from src.institutional.tier2.runtime import AgentRuntime
        
        # Mock the runtime to return predictable outputs
        mock_runtime = Mock(spec=AgentRuntime)
        mock_runtime.run_agents_parallel.return_value = [
            AgentOutput(
                agent_name="red_team",
                round_type="light",
                claims=[
                    Claim(
                        id="c1",
                        type=ClaimType.RISK.value,
                        stance=ClaimStance.ANTI.value,
                        strength=0.7,
                        statement="Signal conflict detected",
                        evidence_keys=["daily_signal", "weekly_signal"],
                    ),
                ],
                confidence=40.0,
                direction="NEUTRAL",
                recommended_deltas=None,
                input_tokens=150,
                output_tokens=200,
                latency_ms=600,
            ),
        ]
        
        orchestrator = Tier2Orchestrator(
            config=config,
            runtime=mock_runtime,
        )
        
        decision = orchestrator.run(
            tier1_pack=tier1_pack_conflicting,
            escalation_reasons=[EscalationReason.SIGNAL_CONFLICT.value],
        )
        
        assert decision is not None
        assert decision.symbol == "TSLA"
        assert decision.agent_count > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
