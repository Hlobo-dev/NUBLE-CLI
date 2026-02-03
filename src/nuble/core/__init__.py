"""
NUBLE Core - Unified Agent Orchestration System
================================================

This module provides the unified architecture that connects:
- UltimateDecisionEngine (28+ data points)
- ML Prediction Pipeline (46M+ parameters)
- Multi-Agent System (9 specialized agents)
- Lambda Real-Time Data (40+ endpoints)
- SEC Filings Analysis (TENK integration)

Architecture:
    User Query → UnifiedOrchestrator → {
        Fast Path (no LLM) → Direct data response
        Decision Path → UltimateDecisionEngine → Trade signals
        Research Path → Multi-Agent → Claude synthesis
    }
"""

from .unified_orchestrator import (
    UnifiedOrchestrator,
    OrchestratorConfig,
    QueryResult,
    ExecutionPath,
)

from .tools import (
    ToolRegistry,
    Tool,
    ToolResult,
    get_all_tools,
)

from .memory import (
    ConversationMemory,
    PredictionTracker,
    UserPreferences,
    MemoryManager,
)

# Alias for compatibility
MemoryStore = MemoryManager

from .tool_handlers import (
    ToolHandlers,
)

# Alias
handle_tool_call = ToolHandlers.dispatch

__all__ = [
    # Main orchestrator
    'UnifiedOrchestrator',
    'OrchestratorConfig',
    'QueryResult',
    'ExecutionPath',
    
    # Tools
    'ToolRegistry',
    'Tool',
    'ToolResult',
    'get_all_tools',
    
    # Memory
    'ConversationMemory',
    'PredictionTracker',
    'UserPreferences',
    'MemoryManager',
    'MemoryStore',  # Alias
    
    # Tool handlers
    'ToolHandlers',
    'handle_tool_call',
]
