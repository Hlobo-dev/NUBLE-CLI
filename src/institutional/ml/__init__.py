"""
Machine Learning & Deep Learning Module
=======================================

Institutional-grade ML/DL components for financial analysis:
- PyTorch-based Transformer market prediction
- LSTM/GRU time series forecasting with attention
- Neural ensemble for robust predictions
- Market regime detection and change point analysis
- Real data pipelines with Polygon integration

Production Components (torch_models):
- FinancialLSTM, AttentionLSTM
- MarketTransformer, TemporalFusionTransformer  
- EnsembleNetwork, NeuralRegimeClassifier
- FinancialTrainer with walk-forward validation
"""

# Legacy imports (NumPy-based, being deprecated)
from .transformers import (
    MarketTransformer as LegacyMarketTransformer,
    TemporalFusionTransformer as LegacyTemporalFusionTransformer,
    AttentionLayer as LegacyAttentionLayer,
    PositionalEncoding as LegacyPositionalEncoding,
)

from .lstm import (
    DeepLSTM,
    BidirectionalLSTM,
    StackedLSTM,
    AttentionLSTM as LegacyAttentionLSTM,
)

from .ensemble import (
    EnsemblePredictor as LegacyEnsemblePredictor,
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
    TechnicalFeatureExtractor as LegacyTechnicalFeatureExtractor,
    FundamentalFeatureExtractor,
    SentimentFeatureExtractor,
)

from .training import (
    ModelTrainer,
    CrossValidator,
    HyperparameterOptimizer,
    WalkForwardValidator as LegacyWalkForwardValidator,
)

# Production PyTorch imports
from .torch_models import (
    # Base
    ModelConfig,
    TrainingMetrics,
    PredictionResult,
    BaseFinancialModel,
    get_device,
    # LSTM
    FinancialLSTM,
    AttentionLSTM,
    # Transformer
    MarketTransformer,
    TemporalFusionTransformer,
    # Ensemble
    EnsembleNetwork,
    EnsemblePrediction,
    UncertaintyEstimator,
    # Regime
    NeuralRegimeClassifier,
    NeuralHMM,
    ChangepointDetector as ChangePointDetector,
    VolatilityRegimeClassifier,
    RegimeState as NeuralRegimeState,
    RegimeDetection,
    # Training
    ModelTrainer as FinancialTrainer,
    TrainingConfig,
    WalkForwardValidator,
    PurgedKFold,
    # Data
    PolygonDataFetcher,
    TechnicalFeatureExtractor,
    FinancialDataset,
    FeatureConfig,
    OHLCVBar,
)

# Integration layer
from .integration import (
    MLPredictor,
    MLIntegration,
    PredictionOutput,
    get_predictor,
    predict,
)

__all__ = [
    # Configuration
    "ModelConfig",
    "TrainingMetrics", 
    "PredictionResult",
    "FeatureConfig",
    "TrainingConfig",
    
    # PyTorch Models (Production)
    "BaseFinancialModel",
    "FinancialLSTM",
    "AttentionLSTM",
    "MarketTransformer",
    "TemporalFusionTransformer",
    "EnsembleNetwork",
    "EnsemblePrediction",
    "UncertaintyEstimator",
    "NeuralRegimeClassifier",
    "NeuralHMM",
    "ChangePointDetector",
    "VolatilityRegimeClassifier",
    "NeuralRegimeState",
    "RegimeDetection",
    
    # Training & Validation
    "FinancialTrainer",
    "WalkForwardValidator",
    "PurgedKFold",
    
    # Data Pipeline
    "PolygonDataFetcher",
    "TechnicalFeatureExtractor",
    "FinancialDataset",
    "OHLCVBar",
    
    # Integration
    "MLPredictor",
    "MLIntegration",
    "PredictionOutput",
    "get_predictor",
    "predict",
    "get_device",
    
    # Legacy (NumPy-based)
    "DeepLSTM",
    "BidirectionalLSTM",
    "StackedLSTM",
    "LegacyAttentionLSTM",
    "LegacyMarketTransformer",
    "LegacyTemporalFusionTransformer",
    "LegacyEnsemblePredictor",
    "StackingEnsemble",
    "BoostingEnsemble",
    "BaggingEnsemble",
    "MarketRegimeDetector",
    "HMMRegimeModel",
    "RegimeState",
    "FeatureEngineer",
    "LegacyTechnicalFeatureExtractor",
    "FundamentalFeatureExtractor",
    "SentimentFeatureExtractor",
    "ModelTrainer",
    "CrossValidator",
    "HyperparameterOptimizer",
    "LegacyWalkForwardValidator",
]
