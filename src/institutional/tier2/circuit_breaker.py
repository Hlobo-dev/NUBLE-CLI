"""
Tier 2 Circuit Breaker
=======================

Safety mechanism to prevent Tier 2 from blocking trading.

Triggers:
- Bedrock error rate spikes
- JSON valid rate < 70%
- Tier 2 p95 latency > threshold

Behavior:
- Auto-disable Tier 2 for cooldown window
- Fallback to Tier 1 with tier2_status=DISABLED
- Log incident reason + metrics snapshot
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Tripped, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class Incident:
    """Record of a circuit breaker incident."""
    timestamp: datetime
    reason: str
    metrics_snapshot: Dict[str, Any]
    state_change: str  # "opened" or "closed"


@dataclass
class CircuitMetrics:
    """Metrics tracked by the circuit breaker."""
    # Error tracking
    bedrock_errors: int = 0
    json_parse_errors: int = 0
    timeout_errors: int = 0
    
    # Success tracking
    successful_requests: int = 0
    total_requests: int = 0
    
    # Latency tracking (last N requests)
    latencies: List[float] = field(default_factory=list)
    
    # JSON validity
    json_valid_count: int = 0
    json_total_count: int = 0
    
    # Timestamps
    window_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    
    def record_success(self, latency_ms: float, json_valid: bool = True):
        """Record a successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.last_success = datetime.now(timezone.utc)
        
        # Track latency (keep last 100)
        self.latencies.append(latency_ms)
        if len(self.latencies) > 100:
            self.latencies.pop(0)
        
        # Track JSON validity
        self.json_total_count += 1
        if json_valid:
            self.json_valid_count += 1
    
    def record_error(self, error_type: str):
        """Record an error."""
        self.total_requests += 1
        self.last_failure = datetime.now(timezone.utc)
        
        if error_type == "bedrock":
            self.bedrock_errors += 1
        elif error_type == "json":
            self.json_parse_errors += 1
        elif error_type == "timeout":
            self.timeout_errors += 1
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_requests == 0:
            return 0.0
        total_errors = self.bedrock_errors + self.json_parse_errors + self.timeout_errors
        return total_errors / self.total_requests
    
    @property
    def json_valid_rate(self) -> float:
        """Calculate JSON validity rate."""
        if self.json_total_count == 0:
            return 1.0
        return self.json_valid_count / self.json_total_count
    
    @property
    def p95_latency(self) -> float:
        """Calculate p95 latency."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    def reset(self):
        """Reset metrics for new window."""
        self.bedrock_errors = 0
        self.json_parse_errors = 0
        self.timeout_errors = 0
        self.successful_requests = 0
        self.total_requests = 0
        self.latencies = []
        self.json_valid_count = 0
        self.json_total_count = 0
        self.window_start = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "bedrock_errors": self.bedrock_errors,
            "json_parse_errors": self.json_parse_errors,
            "timeout_errors": self.timeout_errors,
            "error_rate": round(self.error_rate, 3),
            "json_valid_rate": round(self.json_valid_rate, 3),
            "p95_latency_ms": round(self.p95_latency, 1),
            "window_start": self.window_start.isoformat(),
        }


class CircuitBreaker:
    """
    Circuit breaker for Tier 2 operations.
    
    Prevents cascading failures by:
    1. Monitoring error rates and latency
    2. Opening circuit when thresholds exceeded
    3. Allowing gradual recovery (half-open state)
    
    Usage:
        breaker = CircuitBreaker()
        
        if breaker.allow_request():
            try:
                result = tier2.process(request)
                breaker.record_success(latency_ms)
            except Exception as e:
                breaker.record_failure("bedrock")
                # Fallback to Tier 1
        else:
            # Circuit is open, skip Tier 2
            use_tier1_fallback()
    """
    
    def __init__(
        self,
        error_threshold: int = 5,
        error_rate_threshold: float = 0.3,
        json_valid_threshold: float = 0.7,
        latency_threshold_ms: float = 10000,
        cooldown_seconds: int = 300,
        half_open_requests: int = 3,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            error_threshold: Number of errors before tripping
            error_rate_threshold: Error rate (0-1) before tripping
            json_valid_threshold: Min JSON validity rate
            latency_threshold_ms: P95 latency threshold
            cooldown_seconds: Time to wait before half-open
            half_open_requests: Requests to test in half-open
        """
        self.error_threshold = error_threshold
        self.error_rate_threshold = error_rate_threshold
        self.json_valid_threshold = json_valid_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.cooldown_seconds = cooldown_seconds
        self.half_open_requests = half_open_requests
        
        # State
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self.opened_at: Optional[datetime] = None
        self.half_open_successes = 0
        
        # History
        self.incidents: List[Incident] = []
    
    def allow_request(self) -> bool:
        """
        Check if request should be allowed.
        
        Returns:
            True if request should proceed, False if circuit is open
        """
        now = datetime.now(timezone.utc)
        
        if self.state == CircuitState.CLOSED:
            return True
        
        elif self.state == CircuitState.OPEN:
            # Check if cooldown has passed
            if self.opened_at:
                elapsed = (now - self.opened_at).total_seconds()
                if elapsed >= self.cooldown_seconds:
                    self._transition_to_half_open()
                    return True
            return False
        
        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited requests to test recovery
            return True
        
        return False
    
    def record_success(self, latency_ms: float, json_valid: bool = True):
        """Record a successful request."""
        self.metrics.record_success(latency_ms, json_valid)
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.half_open_requests:
                self._close_circuit("Recovery successful after half-open test")
        
        # Check if we should trip based on latency
        if self.state == CircuitState.CLOSED:
            self._check_trip_conditions()
    
    def record_failure(self, error_type: str):
        """
        Record a failed request.
        
        Args:
            error_type: "bedrock", "json", or "timeout"
        """
        self.metrics.record_error(error_type)
        
        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open → back to open
            self._open_circuit(f"Failure during half-open: {error_type}")
        elif self.state == CircuitState.CLOSED:
            self._check_trip_conditions()
    
    def _check_trip_conditions(self):
        """Check if circuit should trip."""
        # Check absolute error count
        total_errors = (
            self.metrics.bedrock_errors + 
            self.metrics.json_parse_errors + 
            self.metrics.timeout_errors
        )
        if total_errors >= self.error_threshold:
            self._open_circuit(f"Error count {total_errors} >= {self.error_threshold}")
            return
        
        # Check error rate (only if enough samples)
        if self.metrics.total_requests >= 10:
            if self.metrics.error_rate >= self.error_rate_threshold:
                self._open_circuit(
                    f"Error rate {self.metrics.error_rate:.1%} >= {self.error_rate_threshold:.1%}"
                )
                return
        
        # Check JSON validity rate
        if self.metrics.json_total_count >= 5:
            if self.metrics.json_valid_rate < self.json_valid_threshold:
                self._open_circuit(
                    f"JSON valid rate {self.metrics.json_valid_rate:.1%} < {self.json_valid_threshold:.1%}"
                )
                return
        
        # Check latency
        if len(self.metrics.latencies) >= 5:
            if self.metrics.p95_latency > self.latency_threshold_ms:
                self._open_circuit(
                    f"P95 latency {self.metrics.p95_latency:.0f}ms > {self.latency_threshold_ms}ms"
                )
                return
    
    def _open_circuit(self, reason: str):
        """Open the circuit (trip)."""
        if self.state == CircuitState.OPEN:
            return  # Already open
        
        old_state = self.state
        self.state = CircuitState.OPEN
        self.opened_at = datetime.now(timezone.utc)
        
        # Record incident
        incident = Incident(
            timestamp=self.opened_at,
            reason=reason,
            metrics_snapshot=self.metrics.to_dict(),
            state_change="opened",
        )
        self.incidents.append(incident)
        
        logger.warning(
            f"Circuit breaker OPENED: {reason}. "
            f"Tier 2 disabled for {self.cooldown_seconds}s. "
            f"Metrics: {self.metrics.to_dict()}"
        )
    
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.half_open_successes = 0
        self.metrics.reset()
        
        logger.info("Circuit breaker transitioning to HALF_OPEN state")
    
    def _close_circuit(self, reason: str):
        """Close the circuit (recover)."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.opened_at = None
        self.half_open_successes = 0
        self.metrics.reset()
        
        # Record incident
        incident = Incident(
            timestamp=datetime.now(timezone.utc),
            reason=reason,
            metrics_snapshot={},
            state_change="closed",
        )
        self.incidents.append(incident)
        
        logger.info(f"Circuit breaker CLOSED: {reason}")
    
    def force_open(self, reason: str = "Manual trigger"):
        """Manually open the circuit."""
        self._open_circuit(f"MANUAL: {reason}")
    
    def force_close(self, reason: str = "Manual trigger"):
        """Manually close the circuit."""
        self._close_circuit(f"MANUAL: {reason}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        status = {
            "state": self.state.value,
            "metrics": self.metrics.to_dict(),
            "recent_incidents": len(self.incidents),
        }
        
        if self.state == CircuitState.OPEN and self.opened_at:
            elapsed = (datetime.now(timezone.utc) - self.opened_at).total_seconds()
            remaining = max(0, self.cooldown_seconds - elapsed)
            status["cooldown_remaining_seconds"] = round(remaining)
        
        if self.state == CircuitState.HALF_OPEN:
            status["half_open_successes"] = self.half_open_successes
            status["half_open_required"] = self.half_open_requests
        
        return status
    
    def get_recent_incidents(self, limit: int = 10) -> List[Dict]:
        """Get recent incidents."""
        recent = self.incidents[-limit:]
        return [
            {
                "timestamp": i.timestamp.isoformat(),
                "reason": i.reason,
                "state_change": i.state_change,
                "metrics": i.metrics_snapshot,
            }
            for i in recent
        ]
