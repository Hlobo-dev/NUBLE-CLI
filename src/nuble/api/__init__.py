#!/usr/bin/env python3
"""
NUBLE API Package

FastAPI backend for NUBLE.

Three API modules:
  - server.py        — Production API wrapping Manager (SmartRouter + APEX Dual-Brain)
  - intelligence.py  — System A+B intelligence endpoints (predictions, regime, top picks)
  - main.py          — Legacy API (direct OrchestratorAgent, kept for reference)

Use server.py for your frontend integration:
    uvicorn nuble.api.server:app --host 0.0.0.0 --port 8000

Intelligence endpoints are mounted at /api/intel/* automatically.
"""

# Import the new production server as the default
try:
    from .server import app
except ImportError:
    # Fallback to legacy if FastAPI not installed
    app = None

# Import intelligence router for standalone use
try:
    from .intelligence import router as intel_router
except ImportError:
    intel_router = None

__all__ = ['app', 'intel_router']
