"""
NUBLE Learning System

Continuous learning infrastructure for the signal fusion system.

Components:
- PredictionTracker: Track every prediction
- OutcomeResolver: Match predictions to outcomes
- AccuracyMonitor: Calculate accuracy by source
- WeightAdjuster: Dynamically adjust source weights
"""

from .prediction_tracker import PredictionTracker, Prediction
from .accuracy_monitor import AccuracyMonitor
from .weight_adjuster import WeightAdjuster

__all__ = [
    'PredictionTracker',
    'Prediction',
    'AccuracyMonitor',
    'WeightAdjuster'
]
