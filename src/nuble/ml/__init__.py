"""
NUBLE ML — Institutional-Grade Machine Learning Pipeline
=========================================================

Feature engineering, model training, and prediction infrastructure
built on de Prado's Advances in Financial Machine Learning.
"""

from .features_v2 import (
    FeaturePipeline,
    FractionalDifferentiator,
    CyclicalEncoder,
    CrossAssetFeatures,
    TechnicalFeatures,
    build_features,
)

from .labeling import (
    VolatilityEstimator,
    TripleBarrierLabeler,
    SampleWeighter,
    MetaLabeler,
    create_labels,
    create_meta_labels,
    label_distribution_report,
)

from .trainer_v2 import (
    PurgedWalkForwardCV,
    FinancialMetrics,
    ModelTrainer,
    TrainingPipeline,
    TrainingResults,
)

from .predictor import (
    MLPredictor,
    get_predictor,
)

# Phase 1 upgrade: Universal Technical Model
try:
    from .universal_model import UniversalTechnicalModel
except ImportError:
    UniversalTechnicalModel = None

# Phase 2: Model lifecycle management
try:
    from .model_manager import ModelManager, get_model_manager
except ImportError:
    ModelManager = None
    get_model_manager = None

# Phase 3: WRDS institutional predictor (GKX walk-forward LightGBM)
try:
    from .wrds_predictor import WRDSPredictor, get_wrds_predictor
except ImportError:
    WRDSPredictor = None
    get_wrds_predictor = None

__all__ = [
    # Features (F1)
    "FeaturePipeline",
    "FractionalDifferentiator",
    "CyclicalEncoder",
    "CrossAssetFeatures",
    "TechnicalFeatures",
    "build_features",
    # Labeling (F2)
    "VolatilityEstimator",
    "TripleBarrierLabeler",
    "SampleWeighter",
    "MetaLabeler",
    "create_labels",
    "create_meta_labels",
    "label_distribution_report",
    # Training (F3)
    "PurgedWalkForwardCV",
    "FinancialMetrics",
    "ModelTrainer",
    "TrainingPipeline",
    "TrainingResults",
    # Predictor (F4)
    "MLPredictor",
    "get_predictor",
    # Universal model (Phase 1)
    "UniversalTechnicalModel",
    # Model manager (Phase 2)
    "ModelManager",
    "get_model_manager",
    # WRDS predictor (Phase 3)
    "WRDSPredictor",
    "get_wrds_predictor",
]
