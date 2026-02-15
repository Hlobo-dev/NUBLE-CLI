"""
NUBLE Core — DEPRECATED
========================

The original core orchestrator, tools, tool_handlers, and memory modules
were removed in v7 (Phase 2 cleanup). They were superseded by:

  - agents/orchestrator.py     → OrchestratorAgent (9-agent pipeline)
  - memory.py                  → MemoryManager (root-level)
  - decision/                  → UltimateDecisionEngine

These modules had zero runtime imports and ~3,100 lines of dead code.
"""
