"""
Meta-Labeling Module
=====================
Two-stage ML approach per Lopez de Prado (2018) "Advances in Financial Machine Learning"

The meta-labeling approach:
1. Primary model: Predicts direction (side of trade)
2. Secondary model (meta-labeler): Predicts whether to ACT on primary signal

Benefits:
- Separates alpha (direction) from bet sizing
- Reduces false positives dramatically
- Enables precision-focused trading
- Works with ANY primary model
"""

from .meta_labeler import MetaLabeler, MetaLabelResult, MetaLabelConfig

__all__ = [
    'MetaLabeler',
    'MetaLabelResult',
    'MetaLabelConfig',
]
