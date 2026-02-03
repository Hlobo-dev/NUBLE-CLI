"""
NUBLE Signal Fusion System
=============================

Multi-source signal fusion for institutional-grade trading decisions.

Signal Sources:
1. Technical (LuxAlgo from TradingView)
2. ML (AFML pipeline)
3. Sentiment (FinBERT news analysis)
4. Regime (HMM market state)
5. Fundamental (Valuations)
6. On-chain (Crypto flows)

Multi-Timeframe Components:
1. TimeframeManager - Stores signals across timeframes
2. VetoEngine - Applies institutional veto rules
3. PositionCalculator - Kelly-based position sizing
4. MTFFusionEngine - Main brain combining all components

No single source dominates. Intelligence comes from fusion.
"""

from .luxalgo_webhook import (
    LuxAlgoSignal,
    LuxAlgoSignalType,
    LuxAlgoSignalStore,
    parse_luxalgo_webhook,
    get_signal_store
)

from .fusion_engine import (
    SignalFusionEngine,
    FusedSignal,
    FusedSignalStrength
)

from .base_source import (
    SignalSource,
    NormalizedSignal
)

# Multi-Timeframe System
from .timeframe_manager import (
    TimeframeManager,
    TimeframeSignal,
    Timeframe,
    get_timeframe_manager,
    parse_mtf_webhook
)

from .veto_engine import (
    VetoEngine,
    VetoResult,
    VetoDecision,
    check_veto
)

from .position_calculator import (
    PositionCalculator,
    PositionSize,
    calculate_position
)

from .mtf_fusion import (
    MTFFusionEngine,
    TradingDecision,
    SignalStrength,
    get_mtf_engine,
    generate_mtf_decision
)

__all__ = [
    # LuxAlgo
    'LuxAlgoSignal',
    'LuxAlgoSignalType',
    'LuxAlgoSignalStore',
    'parse_luxalgo_webhook',
    'get_signal_store',
    
    # Single-Source Fusion
    'SignalFusionEngine',
    'FusedSignal',
    'FusedSignalStrength',
    
    # Base
    'SignalSource',
    'NormalizedSignal',
    
    # Multi-Timeframe System
    'TimeframeManager',
    'TimeframeSignal',
    'Timeframe',
    'get_timeframe_manager',
    'parse_mtf_webhook',
    
    # Veto Engine
    'VetoEngine',
    'VetoResult',
    'VetoDecision',
    'check_veto',
    
    # Position Calculator
    'PositionCalculator',
    'PositionSize',
    'calculate_position',
    
    # MTF Fusion Engine
    'MTFFusionEngine',
    'TradingDecision',
    'SignalStrength',
    'get_mtf_engine',
    'generate_mtf_decision',
]
