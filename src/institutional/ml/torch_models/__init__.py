"""
PyTorch-Based Neural Network Models for Financial Time Series
==============================================================

Production-grade deep learning models for institutional financial analysis:
- FinancialLSTM: Attention-augmented bidirectional LSTM
- MarketTransformer: Temporal Fusion Transformer for multi-horizon forecasting
- EnsembleNetwork: Neural ensemble with uncertainty quantification
- RegimeClassifier: HMM-inspired neural regime detection
- Training Pipeline: Walk-forward validation, mixed precision, checkpointing
- Data Pipeline: Real-time data from Polygon API with feature engineering

All models support:
- GPU acceleration (CUDA/MPS when available)
- Mixed precision training (AMP)
- Gradient checkpointing for memory efficiency
- Model checkpointing and versioning
- Uncertainty quantification via Monte Carlo Dropout
- Walk-forward and purged K-fold cross-validation
"""

from .base import (
    BaseFinancialModel,
    ModelConfig,
    PredictionResult,
    TrainingMetrics,
    PositionalEncoding,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    get_device
)

from .financial_lstm import FinancialLSTM, AttentionLSTM
from .market_transformer import MarketTransformer, TemporalFusionTransformer
from .ensemble_network import EnsembleNetwork, UncertaintyEstimator, EnsemblePrediction
from .regime_classifier import (
    NeuralRegimeClassifier,
    NeuralHMM,
    ChangepointDetector,
    VolatilityRegimeClassifier,
    RegimeState,
    RegimeDetection
)
from .trainer import (
    ModelTrainer,
    TrainingConfig,
    TrainingState,
    WalkForwardValidator,
    PurgedKFold,
    CosineWarmupScheduler,
    compute_financial_metrics
)
from .data_pipeline import (
    TechnicalFeatureExtractor,
    FeatureConfig,
    FinancialDataset,
    PolygonDataFetcher,
    OHLCVBar,
    create_training_data
)

__all__ = [
    # Base
    'BaseFinancialModel',
    'ModelConfig',
    'PredictionResult',
    'TrainingMetrics',
    'PositionalEncoding',
    'GatedResidualNetwork',
    'VariableSelectionNetwork',
    'get_device',
    
    # Models
    'FinancialLSTM',
    'AttentionLSTM',
    'MarketTransformer',
    'TemporalFusionTransformer',
    'EnsembleNetwork',
    'UncertaintyEstimator',
    'EnsemblePrediction',
    'NeuralRegimeClassifier',
    'NeuralHMM',
    'ChangepointDetector',
    'VolatilityRegimeClassifier',
    'RegimeState',
    'RegimeDetection',
    
    # Training
    'ModelTrainer',
    'TrainingConfig',
    'TrainingState',
    'WalkForwardValidator',
    'PurgedKFold',
    'CosineWarmupScheduler',
    'compute_financial_metrics',
    
    # Data
    'TechnicalFeatureExtractor',
    'FeatureConfig',
    'FinancialDataset',
    'PolygonDataFetcher',
    'OHLCVBar',
    'create_training_data',
]
