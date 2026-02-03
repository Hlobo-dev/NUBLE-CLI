"""
NUBLE TIER 2 - Council-of-Experts Orchestrator (CEO)
========================================================

Governed Decision Quality Layer

Mission:
    Tier 2 does not replace Tier 1. It only runs on escalation, and its job
    is to produce auditable deltas to Tier 1's decision:
    - WAIT / NO_TRADE
    - confidence down
    - position cap down
    - timing delay
    - rarely confidence up (only if unanimous + risk OK)

Non-Negotiable Principles:
    1. Evidence-bounded: every claim references specific inputs
    2. Budgeted depth: everyone runs light; only a few run deep
    3. Risk is a gate, not a vote: Tier 2 cannot override hard risk constraints
    4. Deterministic fallbacks: Tier 2 can never break trading
    5. Self-calibration: static weights → learned weights over time

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         TIER 2 PIPELINE                             │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  ┌───────────────┐   ┌────────────────┐   ┌────────────────────┐   │
    │  │ TIER 1        │──►│ ESCALATION     │──►│ TIER 2 ORCHESTRATOR│   │
    │  │ DECISION      │   │ DETECTOR       │   │ (CEO)              │   │
    │  └───────────────┘   └────────────────┘   └─────────┬──────────┘   │
    │                                                       │              │
    │                  ┌────────────────────────────────────┤              │
    │                  │                                    │              │
    │  ┌───────────────▼──────────────┐    ┌───────────────▼──────────┐  │
    │  │ LAYER A: ALLOCATOR           │    │ LAYER B: EXPERT ROUND    │  │
    │  │ - Token budgets              │    │ - 150-350 tokens each    │  │
    │  │ - Agent selection            │    │ - JSON claims only       │  │
    │  │ - Timeout management         │    │ - 10-12 MVP agents       │  │
    │  └───────────────┬──────────────┘    └───────────────┬──────────┘  │
    │                  │                                    │              │
    │  ┌───────────────▼──────────────┐    ┌───────────────▼──────────┐  │
    │  │ LAYER C: DEEP DIVE           │    │ LAYER D: CROSS-EXAM     │  │
    │  │ - 1000-2500 tokens           │    │ (Optional, future)       │  │
    │  │ - 3-8 activated agents       │    │ - Debate resolution      │  │
    │  └───────────────┬──────────────┘    └───────────────┬──────────┘  │
    │                  │                                    │              │
    │                  └────────────────┬───────────────────┘              │
    │                                   ▼                                  │
    │                  ┌────────────────────────────────────┐             │
    │                  │ LAYER E: SYNTHESIS + GOVERNANCE    │             │
    │                  │ ┌──────────────┐  ┌─────────────┐  │             │
    │                  │ │ ARBITER      │  │ RISK GATE   │  │             │
    │                  │ │ (aggregate)  │  │ (VETO)      │  │             │
    │                  │ └──────────────┘  └─────────────┘  │             │
    │                  └───────────────┬────────────────────┘             │
    │                                  ▼                                   │
    │                  ┌────────────────────────────────────┐             │
    │                  │ TIER 1 + DELTAS = FINAL DECISION   │             │
    │                  └────────────────────────────────────┘             │
    └─────────────────────────────────────────────────────────────────────┘

Version: 1.0.0
Author: NUBLE Institutional
"""

from .config import Tier2Config, AgentConfig, ESCALATION_REASONS, EscalationReason
from .schemas import (
    Claim,
    AgentOutput,
    ClaimsGraph,
    Tier2Decision,
    Tier1DecisionPack,
    ClaimType,
    ClaimStance,
    Verdict,
    RecommendedDeltas,
    ArbiterOutput,
)
from .registry import AgentRegistry, get_default_registry, create_default_registry
from .agents import (
    BaseAgent,
    AgentContext,
    MTFDominanceAgent,
    TrendIntegrityAgent,
    ReversalPullbackAgent,
    VolatilityStateAgent,
    RegimeTransitionAgent,
    EventWindowAgent,
    RedTeamAgent,
    RiskGatekeeperAgent,
    DataIntegrityAgent,
    TimingAgent,
    ConcentrationAgent,
    LiquidityAgent,
    create_agent,
)
from .runtime import AgentRuntime
from .allocator import Allocator
from .arbiter import Arbiter
from .orchestrator import Tier2Orchestrator, create_orchestrator
from .circuit_breaker import CircuitBreaker
from .store import DecisionStore, DynamoDBStore, InMemoryStore
from .escalation import EscalationDetector, EscalationResult

__version__ = "1.0.0"
__all__ = [
    # Config
    "Tier2Config",
    "AgentConfig",
    "ESCALATION_REASONS",
    "EscalationReason",
    # Schemas
    "Claim",
    "AgentOutput",
    "ClaimsGraph",
    "Tier2Decision",
    "Tier1DecisionPack",
    "ClaimType",
    "ClaimStance",
    "Verdict",
    "RecommendedDeltas",
    "ArbiterOutput",
    # Registry
    "AgentRegistry",
    "get_default_registry",
    "create_default_registry",
    # Agents
    "BaseAgent",
    "AgentContext",
    "MTFDominanceAgent",
    "TrendIntegrityAgent",
    "ReversalPullbackAgent",
    "VolatilityStateAgent",
    "RegimeTransitionAgent",
    "EventWindowAgent",
    "RedTeamAgent",
    "RiskGatekeeperAgent",
    "DataIntegrityAgent",
    "TimingAgent",
    "ConcentrationAgent",
    "LiquidityAgent",
    "create_agent",
    # Runtime
    "AgentRuntime",
    # Core
    "Allocator",
    "Arbiter",
    "Tier2Orchestrator",
    "create_orchestrator",
    "CircuitBreaker",
    "DecisionStore",
    "DynamoDBStore",
    "InMemoryStore",
    "EscalationDetector",
    "EscalationResult",
]
