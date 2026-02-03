"""
Tier 2 Expert Agents Package
=============================

Modular agent architecture for independent development and optimization.

Structure:
    agents/
    ├── __init__.py           # This file - exports all agents
    ├── base.py               # BaseAgent, AgentContext, common utilities
    ├── registry.py           # Agent registry and factory
    ├── prompts/              # Versioned prompts per agent
    │   ├── mtf_dominance/
    │   │   ├── v1.0.yaml     # Prompt version 1.0
    │   │   └── v1.1.yaml     # Improved version
    │   └── ...
    ├── tests/                # Per-agent test suites
    │   ├── test_mtf_dominance.py
    │   └── ...
    ├── evals/                # Evaluation datasets
    │   ├── mtf_dominance_cases.json
    │   └── ...
    │
    ├── technical/            # Technical analysis agents
    │   ├── mtf_dominance.py
    │   ├── trend_integrity.py
    │   ├── reversal_pullback.py
    │   └── volatility_state.py
    │
    ├── market/               # Market context agents
    │   ├── regime_transition.py
    │   └── event_window.py
    │
    ├── risk/                 # Risk management agents
    │   ├── risk_gatekeeper.py
    │   ├── concentration.py
    │   └── liquidity.py
    │
    ├── quality/              # Quality & validation agents
    │   ├── data_integrity.py
    │   └── timing.py
    │
    └── adversarial/          # Adversarial agents
        └── red_team.py

Each agent file contains:
    - Agent class extending BaseAgent
    - Prompt templates (light + deep)
    - Evidence key mappings
    - Example test cases
    - Performance metrics hooks
"""

# Base classes
from .base import BaseAgent, AgentContext, AgentMetrics

# Technical agents
from .technical.mtf_dominance import MTFDominanceAgent
from .technical.trend_integrity import TrendIntegrityAgent
from .technical.reversal_pullback import ReversalPullbackAgent
from .technical.volatility_state import VolatilityStateAgent

# Market context agents
from .market.regime_transition import RegimeTransitionAgent
from .market.event_window import EventWindowAgent

# Risk agents
from .risk.risk_gatekeeper import RiskGatekeeperAgent
from .risk.concentration import ConcentrationAgent
from .risk.liquidity import LiquidityAgent

# Quality agents
from .quality.data_integrity import DataIntegrityAgent
from .quality.timing import TimingAgent

# Adversarial agents
from .adversarial.red_team import RedTeamAgent

# Registry
from .registry import AgentRegistry, create_agent, get_all_agents

__all__ = [
    # Base
    "BaseAgent",
    "AgentContext", 
    "AgentMetrics",
    
    # Technical
    "MTFDominanceAgent",
    "TrendIntegrityAgent",
    "ReversalPullbackAgent",
    "VolatilityStateAgent",
    
    # Market
    "RegimeTransitionAgent",
    "EventWindowAgent",
    
    # Risk
    "RiskGatekeeperAgent",
    "ConcentrationAgent",
    "LiquidityAgent",
    
    # Quality
    "DataIntegrityAgent",
    "TimingAgent",
    
    # Adversarial
    "RedTeamAgent",
    
    # Registry
    "AgentRegistry",
    "create_agent",
    "get_all_agents",
]

# Agent categories for documentation
AGENT_CATEGORIES = {
    "technical": [
        "mtf_dominance",
        "trend_integrity", 
        "reversal_pullback",
        "volatility_state",
    ],
    "market": [
        "regime_transition",
        "event_window",
    ],
    "risk": [
        "risk_gatekeeper",
        "concentration",
        "liquidity",
    ],
    "quality": [
        "data_integrity",
        "timing",
    ],
    "adversarial": [
        "red_team",
    ],
}
