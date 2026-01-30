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

# Machine Learning modules
from .ml import (
    # Transformers
    MarketTransformer,
    TemporalFusionTransformer,
    AttentionLayer,
    PositionalEncoding,
    # LSTM/GRU
    DeepLSTM,
    BidirectionalLSTM,
    StackedLSTM,
    AttentionLSTM,
    # Ensemble
    EnsemblePredictor,
    StackingEnsemble,
    BoostingEnsemble,
    BaggingEnsemble,
    # Regime Detection
    MarketRegimeDetector,
    HMMRegimeModel,
    RegimeState,
    # Feature Engineering
    FeatureEngineer,
    TechnicalFeatureExtractor,
    FundamentalFeatureExtractor,
    SentimentFeatureExtractor,
    # Training
    ModelTrainer,
    CrossValidator,
    HyperparameterOptimizer,
    WalkForwardValidator,
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
    # Machine Learning - Transformers
    "MarketTransformer",
    "TemporalFusionTransformer",
    "AttentionLayer",
    "PositionalEncoding",
    # Machine Learning - LSTM/GRU
    "DeepLSTM",
    "BidirectionalLSTM",
    "StackedLSTM",
    "AttentionLSTM",
    # Machine Learning - Ensemble
    "EnsemblePredictor",
    "StackingEnsemble",
    "BoostingEnsemble",
    "BaggingEnsemble",
    # Machine Learning - Regime
    "MarketRegimeDetector",
    "HMMRegimeModel",
    "RegimeState",
    # Machine Learning - Features
    "FeatureEngineer",
    "TechnicalFeatureExtractor",
    "FundamentalFeatureExtractor",
    "SentimentFeatureExtractor",
    # Machine Learning - Training
    "ModelTrainer",
    "CrossValidator",
    "HyperparameterOptimizer",
    "WalkForwardValidator",
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
]
