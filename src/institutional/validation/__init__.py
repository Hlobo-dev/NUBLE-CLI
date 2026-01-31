"""
Model Validation Module
========================

The ONLY thing that matters: Do these models have predictive power?

This module provides rigorous out-of-sample validation for all ML models.
Without this, you have nothing.
"""

from .model_validator import (
    ModelValidator,
    ValidationResult,
    run_full_validation,
    run_validation_on_real_data,
)

__all__ = [
    'ModelValidator',
    'ValidationResult',
    'run_full_validation',
    'run_validation_on_real_data',
]
