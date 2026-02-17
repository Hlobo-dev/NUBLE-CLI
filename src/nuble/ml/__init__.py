"""
NUBLE ML — Institutional-Grade Machine Learning Pipeline
=========================================================

Feature engineering, model training, and prediction infrastructure
built on de Prado's Advances in Financial Machine Learning.
"""

try:
    from .features_v2 import (
        FeaturePipeline,
        FractionalDifferentiator,
        CyclicalEncoder,
        CrossAssetFeatures,
        TechnicalFeatures,
        build_features,
    )
except ImportError:
    FeaturePipeline = None
    FractionalDifferentiator = None
    CyclicalEncoder = None
    CrossAssetFeatures = None
    TechnicalFeatures = None
    build_features = None

try:
    from .labeling import (
        VolatilityEstimator,
        TripleBarrierLabeler,
        SampleWeighter,
        MetaLabeler,
        create_labels,
        create_meta_labels,
        label_distribution_report,
    )
except ImportError:
    VolatilityEstimator = None
    TripleBarrierLabeler = None
    SampleWeighter = None
    MetaLabeler = None
    create_labels = None
    create_meta_labels = None
    label_distribution_report = None

try:
    from .trainer_v2 import (
        PurgedWalkForwardCV,
        FinancialMetrics,
        ModelTrainer,
        TrainingPipeline,
        TrainingResults,
    )
except ImportError:
    PurgedWalkForwardCV = None
    FinancialMetrics = None
    ModelTrainer = None
    TrainingPipeline = None
    TrainingResults = None

try:
    from .predictor import (
        MLPredictor,
        get_predictor,
    )
except ImportError:
    MLPredictor = None
    get_predictor = None

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

# Phase 4: LivePredictor (Polygon live + WRDS-trained multi-tier ensemble)
try:
    from .live_predictor import LivePredictor, get_live_predictor
except ImportError:
    LivePredictor = None
    get_live_predictor = None

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
    # Predictor (F4) — DEPRECATED, use LivePredictor
    "MLPredictor",
    "get_predictor",
    # Universal model (Phase 1)
    "UniversalTechnicalModel",
    # Model manager (Phase 2)
    "ModelManager",
    "get_model_manager",
    # WRDS predictor (Phase 3) — historical fallback
    "WRDSPredictor",
    "get_wrds_predictor",
    # LivePredictor (Phase 4) — PRIMARY signal source
    "LivePredictor",
    "get_live_predictor",
]
