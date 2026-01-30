"""
Machine Learning & Deep Learning Module
=======================================

Institutional-grade ML/DL components for financial analysis:
- Transformer-based market prediction
- LSTM/GRU time series forecasting
- Attention mechanisms for feature importance
- Reinforcement learning for portfolio optimization
- Ensemble methods for robust predictions
"""

from .transformers import (
    MarketTransformer,
    TemporalFusionTransformer,
    AttentionLayer,
    PositionalEncoding,
)

from .lstm import (
    DeepLSTM,
    BidirectionalLSTM,
    StackedLSTM,
    AttentionLSTM,
)

from .ensemble import (
    EnsemblePredictor,
    StackingEnsemble,
    BoostingEnsemble,
    BaggingEnsemble,
)

from .regime import (
    MarketRegimeDetector,
    HMMRegimeModel,
    RegimeState,
)

from .features import (
    FeatureEngineer,
    TechnicalFeatureExtractor,
    FundamentalFeatureExtractor,
    SentimentFeatureExtractor,
)

from .training import (
    ModelTrainer,
    CrossValidator,
    HyperparameterOptimizer,
    WalkForwardValidator,
)

__all__ = [
    # Transformers
    "MarketTransformer",
    "TemporalFusionTransformer",
    "AttentionLayer",
    "PositionalEncoding",
    # LSTM
    "DeepLSTM",
    "BidirectionalLSTM",
    "StackedLSTM",
    "AttentionLSTM",
    # Ensemble
    "EnsemblePredictor",
    "StackingEnsemble",
    "BoostingEnsemble",
    "BaggingEnsemble",
    # Regime Detection
    "MarketRegimeDetector",
    "HMMRegimeModel",
    "RegimeState",
    # Features
    "FeatureEngineer",
    "TechnicalFeatureExtractor",
    "FundamentalFeatureExtractor",
    "SentimentFeatureExtractor",
    # Training
    "ModelTrainer",
    "CrossValidator",
    "HyperparameterOptimizer",
    "WalkForwardValidator",
]
