"""
Institutional Research Platform - Advanced Edition
====================================================

A comprehensive, institutional-grade financial data aggregation, analysis, 
and machine learning platform powered by state-of-the-art deep learning.

Features:
---------
CORE:
- Multi-provider data aggregation (Polygon.io/Massive.com, Alpha Vantage, Finnhub, SEC EDGAR)
- Natural language query understanding with LLM
- Intelligent query routing and caching

ANALYTICS:
- Technical analysis with 50+ indicators
- Pattern recognition (chart patterns, candlesticks)
- Sentiment analysis (news, social media)
- Anomaly detection with ML

MACHINE LEARNING (Advanced):
- Transformer models (Temporal Fusion Transformer, Financial Transformer)
- LSTM/GRU with attention mechanisms
- Ensemble methods (Stacking, Bagging, Boosting)
- Hidden Markov Model regime detection
- Change point detection
- Advanced feature engineering

MCP INTEGRATION:
- Massive.com MCP server for AI-powered market queries
- Real-time data streaming
- Options analytics

AGENTS:
- Multi-agent AI research system
- Specialized agents (Research, Trading, Risk, Portfolio)
- Autonomous market analysis

"""

__version__ = "2.0.0"
__author__ = "Institutional Research Team"

from .config import Config, load_config

# Core modules
from .core import (
    Orchestrator,
    QueryResult,
    QueryStatus,
    IntentEngine,
    QueryIntent,
    DataRouter,
    Synthesizer,
    SynthesisResult,
)

# Provider modules
from .providers import (
    BaseProvider,
    ProviderResponse,
    DataType,
    PolygonProvider,
    AlphaVantageProvider,
    FinnhubProvider,
    SECEdgarProvider,
)

# Analytics modules
from .analytics import (
    TechnicalAnalyzer,
    SentimentAnalyzer,
    PatternRecognizer,
    AnomalyDetector,
)

# Machine Learning modules - Production PyTorch models
from .ml import (
    # Configuration
    ModelConfig,
    TrainingMetrics,
    PredictionResult,
    FeatureConfig,
    
    # Production PyTorch Models
    BaseFinancialModel,
    FinancialLSTM,
    AttentionLSTM,
    MarketTransformer,
    TemporalFusionTransformer,
    EnsembleNetwork,
    EnsemblePrediction,
    NeuralRegimeClassifier,
    ChangePointDetector,
    
    # Training & Validation
    FinancialTrainer,
    WalkForwardValidator,
    
    # Data Pipeline
    PolygonDataFetcher,
    TechnicalFeatureExtractor,
    FinancialDataset,
    
    # Integration
    MLPredictor,
    MLIntegration,
    get_predictor,
    predict,
    get_device,
    
    # Legacy (for backward compatibility)
    DeepLSTM,
    BidirectionalLSTM,
    StackedLSTM,
    StackingEnsemble,
    BoostingEnsemble,
    BaggingEnsemble,
    MarketRegimeDetector,
    HMMRegimeModel,
    RegimeState,
    FeatureEngineer,
    FundamentalFeatureExtractor,
    SentimentFeatureExtractor,
    ModelTrainer,
    CrossValidator,
    HyperparameterOptimizer,
)

# MCP Integration
from .mcp import (
    MassiveMCPClient,
    MCPServerConfig,
    MCPToolCall,
    MassiveDataProvider,
)

# Agent System
from .agents import (
    BaseAgent,
    ResearchAgent,
    TradingAgent,
    NewsAgent,
    RiskAgent,
    AgentOrchestrator,
    AgentRole,
    AgentContext,
    FinancialAssistant,
)

# SEC Filings (TENK Integration)
from .filings import (
    FilingsDatabase,
    FilingChunk,
    FilingMetadata,
    FilingsSearch,
    SearchResult,
    FilingsLoader,
    LoadedFiling,
    FilingInfo,
    FilingForm,
    FilingsAnalyzer,
    AnalysisType,
    AnalysisResult,
    SECFilingsAgent,
    AgentMessage,
    AgentState,
    FilingsExporter,
    ExportResult,
)

# ============================================================================
# Phase 1+2: Lopez de Prado Institutional ML Components
# ============================================================================

# Triple Barrier Labeling (Phase 1.1)
try:
    from .labeling.triple_barrier import (
        TripleBarrierLabeler,
        TripleBarrierConfig,
        BarrierResult,
        BarrierType,
    )
except ImportError:
    TripleBarrierLabeler = None
    TripleBarrierConfig = None
    BarrierResult = None
    BarrierType = None

# Fractional Differentiation (Phase 1.2)
try:
    from .features.frac_diff import (
        FractionalDifferentiator,
        FracDiffConfig,
    )
except ImportError:
    FractionalDifferentiator = None
    FracDiffConfig = None

# HMM Regime Detection (Phase 2.2)
try:
    from .regime.hmm_detector import (
        HMMRegimeDetector,
        RegimeStatistics,
    )
except ImportError:
    HMMRegimeDetector = None
    RegimeStatistics = None

# Meta-Labeling (Phase 2.1)
try:
    from .models.meta.meta_labeler import (
        MetaLabeler,
        MetaLabelConfig,
        MetaLabelResult,
        MetaLabelPipeline,
    )
except ImportError:
    MetaLabeler = None
    MetaLabelConfig = None
    MetaLabelResult = None
    MetaLabelPipeline = None

# Complete ML Pipeline Integration
try:
    from .ml_pipeline import (
        InstitutionalMLPipeline,
        PipelineConfig,
        PipelineResult,
        create_institutional_pipeline,
    )
except ImportError:
    InstitutionalMLPipeline = None
    PipelineConfig = None
    PipelineResult = None
    create_institutional_pipeline = None

__all__ = [
    # Version
    "__version__",
    # Config
    "Config",
    "load_config",
    # Core
    "Orchestrator",
    "QueryResult",
    "QueryStatus",
    "IntentEngine",
    "QueryIntent",
    "DataRouter",
    "Synthesizer",
    "SynthesisResult",
    # Providers
    "BaseProvider",
    "ProviderResponse",
    "DataType",
    "PolygonProvider",
    "AlphaVantageProvider",
    "FinnhubProvider",
    "SECEdgarProvider",
    # Analytics
    "TechnicalAnalyzer",
    "SentimentAnalyzer",
    "PatternRecognizer",
    "AnomalyDetector",
    
    # Machine Learning - Configuration
    "ModelConfig",
    "TrainingMetrics",
    "PredictionResult",
    "FeatureConfig",
    
    # Machine Learning - Production PyTorch Models
    "BaseFinancialModel",
    "FinancialLSTM",
    "AttentionLSTM",
    "MarketTransformer",
    "TemporalFusionTransformer",
    "EnsembleNetwork",
    "EnsemblePrediction",
    "NeuralRegimeClassifier",
    "ChangePointDetector",
    
    # Machine Learning - Training & Data
    "FinancialTrainer",
    "WalkForwardValidator",
    "PolygonDataFetcher",
    "TechnicalFeatureExtractor",
    "FinancialDataset",
    
    # Machine Learning - Integration
    "MLPredictor",
    "MLIntegration",
    "get_predictor",
    "predict",
    "get_device",
    
    # Machine Learning - Legacy
    "DeepLSTM",
    "BidirectionalLSTM",
    "StackedLSTM",
    "StackingEnsemble",
    "BoostingEnsemble",
    "BaggingEnsemble",
    "MarketRegimeDetector",
    "HMMRegimeModel",
    "RegimeState",
    "FeatureEngineer",
    "FundamentalFeatureExtractor",
    "SentimentFeatureExtractor",
    "ModelTrainer",
    "CrossValidator",
    "HyperparameterOptimizer",
    
    # MCP
    "MassiveMCPClient",
    "MCPServerConfig",
    "MCPToolCall",
    "MassiveDataProvider",
    # Agents
    "BaseAgent",
    "ResearchAgent",
    "TradingAgent",
    "NewsAgent",
    "RiskAgent",
    "AgentOrchestrator",
    "AgentRole",
    "AgentContext",
    "FinancialAssistant",
    # SEC Filings (TENK)
    "FilingsDatabase",
    "FilingChunk",
    "FilingMetadata",
    "FilingsSearch",
    "SearchResult",
    "FilingsLoader",
    "LoadedFiling",
    "FilingInfo",
    "FilingForm",
    "FilingsAnalyzer",
    "AnalysisType",
    "AnalysisResult",
    "SECFilingsAgent",
    "AgentMessage",
    "AgentState",
    "FilingsExporter",
    "ExportResult",
    
    # Phase 1+2: Lopez de Prado Institutional ML
    # Triple Barrier Labeling
    "TripleBarrierLabeler",
    "TripleBarrierConfig",
    "BarrierResult",
    "BarrierType",
    # Fractional Differentiation
    "FractionalDifferentiator",
    "FracDiffConfig",
    # HMM Regime Detection
    "HMMRegimeDetector",
    "RegimeStatistics",
    # Meta-Labeling
    "MetaLabeler",
    "MetaLabelConfig",
    "MetaLabelResult",
    "MetaLabelPipeline",
    # Complete Pipeline
    "InstitutionalMLPipeline",
    "PipelineConfig",
    "PipelineResult",
    "create_institutional_pipeline",
]
