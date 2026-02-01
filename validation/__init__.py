"""
VALIDATION PACKAGE
==================
Rigorous ML trading strategy validation following Lopez de Prado methodology.

This package provides:
1. Data acquisition (Polygon.io paid API)
2. Lookahead bias audit
3. Walk-forward validation with purging/embargo
4. CPCV with PBO and Deflated Sharpe
5. Complete validation orchestration

Key insight: If Sharpe > 3.0, there are bugs.
Expected realistic Sharpe: 1.0 - 1.5
"""

from .config import CONFIG
from .data_downloader import PolygonDataDownloader
from .lookahead_audit import LookaheadAudit
from .walk_forward import (
    WalkForwardValidator,
    WalkForwardBacktest,
    PerformanceMetrics,
    WalkForwardSplit
)
from .cpcv import (
    CombinatorialPurgedCV,
    CPCVBacktest,
    ProbabilityOfBacktestOverfitting,
    DeflatedSharpeRatio
)
from .orchestrator import (
    ValidationOrchestrator,
    create_simple_feature_pipeline,
    create_simple_label_pipeline,
    create_simple_model_factory
)

__all__ = [
    # Config
    'CONFIG',
    
    # Data
    'PolygonDataDownloader',
    
    # Audit
    'LookaheadAudit',
    
    # Walk-Forward
    'WalkForwardValidator',
    'WalkForwardBacktest',
    'PerformanceMetrics',
    'WalkForwardSplit',
    
    # CPCV
    'CombinatorialPurgedCV',
    'CPCVBacktest',
    'ProbabilityOfBacktestOverfitting',
    'DeflatedSharpeRatio',
    
    # Orchestrator
    'ValidationOrchestrator',
    'create_simple_feature_pipeline',
    'create_simple_label_pipeline',
    'create_simple_model_factory'
]

# Version
__version__ = '1.0.0'
