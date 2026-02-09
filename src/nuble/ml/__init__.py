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
]
