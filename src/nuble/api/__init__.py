#!/usr/bin/env python3
"""
NUBLE API Package

FastAPI backend for NUBLE.

Two API modules exist:
  - server.py  — Production API wrapping Manager (SmartRouter + APEX Dual-Brain)
  - main.py    — Legacy API (direct OrchestratorAgent, kept for reference)

Use server.py for your frontend integration:
    uvicorn nuble.api.server:app --host 0.0.0.0 --port 8000
"""

# Import the new production server as the default
try:
    from .server import app
except ImportError:
    # Fallback to legacy if FastAPI not installed
    app = None

__all__ = ['app']
