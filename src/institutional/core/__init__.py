"""
Core module initialization.
Contains the orchestrator, intent engine, router, and synthesizers.
"""

from .orchestrator import Orchestrator, QueryResult, QueryStatus
from .intent_engine import IntentEngine, QueryIntent
from .router import DataRouter, RoutingResult
from .synthesizer import Synthesizer, SynthesisResult
from .claude_synthesizer import (
    ClaudeSynthesizer,
    ClaudeModel,
    ClaudeResponse,
    AnalysisContext,
    create_claude_synthesizer,
)

__all__ = [
    "Orchestrator",
    "QueryResult",
    "QueryStatus",
    "IntentEngine",
    "QueryIntent",
    "DataRouter",
    "RoutingResult",
    "Synthesizer",
    "SynthesisResult",
    # Claude
    "ClaudeSynthesizer",
    "ClaudeModel",
    "ClaudeResponse",
    "AnalysisContext",
    "create_claude_synthesizer",
]
