"""
NUBLE Learning System

Continuous learning infrastructure for the signal fusion system.

Components:
- PredictionTracker: Track every prediction
- OutcomeResolver: Match predictions to outcomes
- AccuracyMonitor: Calculate accuracy by source
- WeightAdjuster: Dynamically adjust source weights
- LearningHub: Singleton coordinator (wired into Manager + Orchestrator)
- PredictionResolver: Background task that resolves predictions hourly
"""

from .prediction_tracker import PredictionTracker, Prediction
from .accuracy_monitor import AccuracyMonitor
from .weight_adjuster import WeightAdjuster
from .learning_hub import LearningHub
from .resolver import PredictionResolver

__all__ = [
    'PredictionTracker',
    'Prediction',
    'AccuracyMonitor',
    'WeightAdjuster',
    'LearningHub',
    'PredictionResolver',
]
