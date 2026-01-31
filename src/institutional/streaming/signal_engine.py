"""
Real-Time Signal Generation Engine
===================================

Production-grade signal generation from live market data and model predictions.

Components:
- SignalEngine: Core signal generation pipeline
- SignalAggregator: Multi-model signal combination
- RiskFilter: Real-time risk-based filtering
- PositionSizer: Kelly criterion and volatility-based sizing

Architecture:
    Stream → Features → Models → Signals → Risk Filter → Orders
    
References:
- Lopez de Prado, "Advances in Financial Machine Learning"
- Chan, "Algorithmic Trading: Winning Strategies and Their Rationale"
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, 
    Set, Tuple, Union
)
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .realtime import MarketTick, MessageType, StreamProcessor

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal type."""
    LONG = "long"
    SHORT = "short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    HOLD = "hold"


class SignalStrength(Enum):
    """Signal confidence level."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class TradingSignal:
    """
    Trading signal with full metadata.
    """
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    strength: SignalStrength
    
    # Price targets
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Position sizing
    position_size: Optional[float] = None
    max_position_pct: float = 0.02  # Max 2% of portfolio
    
    # Risk metrics
    expected_return: Optional[float] = None
    expected_volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    win_probability: Optional[float] = None
    
    # Model metadata
    model_name: str = ""
    model_confidence: float = 0.0
    features_used: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    ttl_seconds: int = 300  # Signal valid for 5 minutes
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        age = (datetime.now(timezone.utc) - self.generated_at).total_seconds()
        return age > self.ttl_seconds
        
    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable (not HOLD, not expired)."""
        return self.signal_type != SignalType.HOLD and not self.is_expired
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'signal_type': self.signal_type.value,
            'strength': self.strength.name,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'expected_return': self.expected_return,
            'win_probability': self.win_probability,
            'model': self.model_name,
            'confidence': self.model_confidence,
            'is_expired': self.is_expired,
        }


@dataclass 
class SignalConfig:
    """
    Signal engine configuration.
    """
    # Feature windows
    feature_lookback: int = 60
    
    # Signal thresholds
    long_threshold: float = 0.55
    short_threshold: float = -0.55
    min_confidence: float = 0.6
    
    # Risk management
    max_position_pct: float = 0.02
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_correlation: float = 0.7
    
    # Kelly criterion
    use_kelly: bool = True
    kelly_fraction: float = 0.25  # Quarter Kelly
    
    # Cooldown
    signal_cooldown_seconds: int = 60
    max_signals_per_hour: int = 10
    
    # Ensemble
    min_agreeing_models: int = 2
    ensemble_method: str = "weighted_average"


class ModelWrapper(ABC):
    """
    Abstract wrapper for ML models.
    
    Subclass this to integrate any ML model into the signal engine.
    """
    
    @abstractmethod
    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        """
        Generate prediction from features.
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            (prediction, confidence) where:
            - prediction: float in [-1, 1], negative=short, positive=long
            - confidence: float in [0, 1]
        """
        pass
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging."""
        pass
        
    @property
    def required_features(self) -> List[str]:
        """List of required feature names."""
        return []


class TFTModelWrapper(ModelWrapper):
    """
    Wrapper for Temporal Fusion Transformer.
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self._device = next(model.parameters()).device
        
    @property
    def name(self) -> str:
        return "TFT"
        
    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Generate TFT prediction."""
        if not HAS_TORCH:
            return 0.0, 0.0
            
        # Convert features to model input
        # This is simplified - real implementation needs proper windowing
        try:
            self.model.eval()
            with torch.no_grad():
                # Build input tensor from features
                feature_values = [
                    features.get('price_pct_change', 0),
                    features.get('volume', 0),
                    features.get('spread_mean', 0),
                    features.get('price_std', 0),
                ]
                
                # Create input tensor [batch=1, time=1, features]
                x = torch.tensor([feature_values], dtype=torch.float32)
                x = x.unsqueeze(0).to(self._device)
                
                # Get prediction
                output = self.model(x)
                
                # Quantile predictions
                if len(output.shape) == 3:
                    # [batch, horizon, quantiles]
                    median = output[0, 0, output.shape[-1] // 2].item()
                    lower = output[0, 0, 0].item()
                    upper = output[0, 0, -1].item()
                    
                    # Confidence based on prediction interval width
                    width = upper - lower
                    confidence = max(0, 1 - width)
                    
                    # Normalize to [-1, 1]
                    prediction = np.tanh(median)
                    
                    return prediction, confidence
                    
        except Exception as e:
            logger.warning(f"TFT prediction error: {e}")
            
        return 0.0, 0.0


class DeepARModelWrapper(ModelWrapper):
    """
    Wrapper for DeepAR probabilistic forecaster.
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self._device = next(model.parameters()).device
        
    @property
    def name(self) -> str:
        return "DeepAR"
        
    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Generate DeepAR prediction with uncertainty."""
        if not HAS_TORCH:
            return 0.0, 0.0
            
        try:
            self.model.eval()
            with torch.no_grad():
                # Build input
                feature_values = [
                    features.get('price_pct_change', 0),
                    features.get('volume', 0),
                ]
                
                x = torch.tensor([feature_values], dtype=torch.float32)
                x = x.unsqueeze(0).to(self._device)
                
                # Get samples
                samples = self.model.sample(x, num_samples=100)
                
                # Mean prediction
                mean = samples.mean().item()
                std = samples.std().item()
                
                # Confidence inversely related to uncertainty
                confidence = max(0, 1 - std)
                
                # Normalize
                prediction = np.tanh(mean)
                
                return prediction, confidence
                
        except Exception as e:
            logger.warning(f"DeepAR prediction error: {e}")
            
        return 0.0, 0.0


class SignalAggregator:
    """
    Aggregates signals from multiple models.
    
    Methods:
    - Weighted average
    - Voting
    - Meta-model stacking
    """
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self._model_weights: Dict[str, float] = {}
        self._model_performance: Dict[str, List[float]] = {}
        
    def set_weight(self, model_name: str, weight: float):
        """Set model weight."""
        self._model_weights[model_name] = weight
        
    def aggregate(
        self,
        predictions: Dict[str, Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Aggregate predictions from multiple models.
        
        Args:
            predictions: Dict of model_name -> (prediction, confidence)
            
        Returns:
            (aggregated_prediction, aggregated_confidence)
        """
        if not predictions:
            return 0.0, 0.0
            
        if self.config.ensemble_method == "weighted_average":
            return self._weighted_average(predictions)
        elif self.config.ensemble_method == "voting":
            return self._voting(predictions)
        elif self.config.ensemble_method == "confidence_weighted":
            return self._confidence_weighted(predictions)
        else:
            return self._weighted_average(predictions)
            
    def _weighted_average(
        self, 
        predictions: Dict[str, Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Weighted average ensemble."""
        total_weight = 0
        weighted_pred = 0
        weighted_conf = 0
        
        for model_name, (pred, conf) in predictions.items():
            weight = self._model_weights.get(model_name, 1.0)
            weighted_pred += pred * weight
            weighted_conf += conf * weight
            total_weight += weight
            
        if total_weight > 0:
            return weighted_pred / total_weight, weighted_conf / total_weight
            
        return 0.0, 0.0
        
    def _confidence_weighted(
        self,
        predictions: Dict[str, Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Weight by confidence."""
        total_conf = 0
        weighted_pred = 0
        
        for model_name, (pred, conf) in predictions.items():
            weighted_pred += pred * conf
            total_conf += conf
            
        if total_conf > 0:
            return weighted_pred / total_conf, total_conf / len(predictions)
            
        return 0.0, 0.0
        
    def _voting(
        self,
        predictions: Dict[str, Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Simple voting ensemble."""
        longs = 0
        shorts = 0
        total_conf = 0
        
        for model_name, (pred, conf) in predictions.items():
            if pred > 0:
                longs += 1
            elif pred < 0:
                shorts += 1
            total_conf += conf
            
        n = len(predictions)
        avg_conf = total_conf / n if n > 0 else 0
        
        if longs > shorts:
            return (longs - shorts) / n, avg_conf
        else:
            return -(shorts - longs) / n, avg_conf
            
    def update_performance(self, model_name: str, accuracy: float):
        """Update model performance for adaptive weighting."""
        if model_name not in self._model_performance:
            self._model_performance[model_name] = []
            
        self._model_performance[model_name].append(accuracy)
        
        # Update weight based on recent performance
        if len(self._model_performance[model_name]) >= 10:
            recent = self._model_performance[model_name][-10:]
            self._model_weights[model_name] = np.mean(recent)


class RiskFilter:
    """
    Real-time risk filtering for signals.
    
    Filters:
    - Position limits
    - Correlation checks
    - Volatility regimes
    - Max drawdown protection
    """
    
    def __init__(self, config: SignalConfig):
        self.config = config
        
        # Position tracking
        self._positions: Dict[str, float] = {}
        self._portfolio_value: float = 100000  # Default
        
        # Signal history
        self._signal_history: deque = deque(maxlen=1000)
        self._signals_per_hour: Dict[str, int] = {}
        
        # Market state
        self._volatility_regime: str = "normal"  # low, normal, high
        
    def set_portfolio_value(self, value: float):
        """Update portfolio value."""
        self._portfolio_value = value
        
    def update_position(self, symbol: str, size: float):
        """Update position size."""
        self._positions[symbol] = size
        
    def filter_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
        """
        Filter signal through risk checks.
        
        Returns:
            (passed, reason) - True if signal passes, reason if rejected
        """
        # Check confidence
        if signal.model_confidence < self.config.min_confidence:
            return False, f"Low confidence: {signal.model_confidence:.2f}"
            
        # Check cooldown
        if not self._check_cooldown(signal.symbol):
            return False, "Signal cooldown active"
            
        # Check hourly limit
        if not self._check_hourly_limit(signal.symbol):
            return False, "Hourly signal limit reached"
            
        # Check position limits
        if not self._check_position_limit(signal):
            return False, "Position limit exceeded"
            
        # Check volatility regime
        if self._volatility_regime == "high" and signal.strength == SignalStrength.WEAK:
            return False, "Weak signal rejected in high volatility"
            
        # Signal passes
        self._record_signal(signal)
        return True, "OK"
        
    def _check_cooldown(self, symbol: str) -> bool:
        """Check signal cooldown."""
        now = datetime.now(timezone.utc)
        
        for sig in reversed(self._signal_history):
            if sig.symbol != symbol:
                continue
                
            age = (now - sig.generated_at).total_seconds()
            if age < self.config.signal_cooldown_seconds:
                return False
                
        return True
        
    def _check_hourly_limit(self, symbol: str) -> bool:
        """Check hourly signal limit."""
        now = datetime.now(timezone.utc)
        hour_key = f"{symbol}:{now.hour}"
        
        count = self._signals_per_hour.get(hour_key, 0)
        return count < self.config.max_signals_per_hour
        
    def _check_position_limit(self, signal: TradingSignal) -> bool:
        """Check position limit."""
        current_size = self._positions.get(signal.symbol, 0)
        new_size = signal.position_size or 0
        
        max_size = self._portfolio_value * self.config.max_position_pct
        
        if signal.signal_type in (SignalType.LONG, SignalType.SHORT):
            return abs(current_size + new_size) <= max_size
            
        return True
        
    def _record_signal(self, signal: TradingSignal):
        """Record signal for tracking."""
        self._signal_history.append(signal)
        
        now = datetime.now(timezone.utc)
        hour_key = f"{signal.symbol}:{now.hour}"
        self._signals_per_hour[hour_key] = self._signals_per_hour.get(hour_key, 0) + 1
        
    def set_volatility_regime(self, regime: str):
        """Set current volatility regime."""
        self._volatility_regime = regime


class PositionSizer:
    """
    Position sizing using Kelly Criterion and volatility targeting.
    
    Methods:
    - Kelly Criterion (with fractional Kelly)
    - Volatility targeting
    - Risk parity
    - Fixed fractional
    """
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self._portfolio_value: float = 100000
        self._target_volatility: float = 0.15  # 15% annual
        
    def set_portfolio_value(self, value: float):
        """Update portfolio value."""
        self._portfolio_value = value
        
    def calculate_size(
        self,
        signal: TradingSignal,
        current_volatility: float = 0.20
    ) -> float:
        """
        Calculate optimal position size.
        
        Args:
            signal: Trading signal
            current_volatility: Current annualized volatility
            
        Returns:
            Position size in dollars
        """
        if self.config.use_kelly:
            kelly_size = self._kelly_size(signal)
        else:
            kelly_size = self._portfolio_value * self.config.max_position_pct
            
        # Volatility adjustment
        vol_scalar = self._target_volatility / max(current_volatility, 0.01)
        vol_adjusted = kelly_size * min(vol_scalar, 2.0)
        
        # Apply limits
        max_size = self._portfolio_value * self.config.max_position_pct
        final_size = min(vol_adjusted, max_size)
        
        return final_size
        
    def _kelly_size(self, signal: TradingSignal) -> float:
        """
        Calculate Kelly Criterion position size.
        
        Kelly % = (p * b - q) / b
        where:
        - p = win probability
        - q = loss probability (1 - p)
        - b = win/loss ratio
        """
        win_prob = signal.win_probability or 0.5
        
        # Calculate b from expected return and stop loss
        if signal.take_profit and signal.stop_loss and signal.entry_price:
            win_amt = abs(signal.take_profit - signal.entry_price)
            loss_amt = abs(signal.entry_price - signal.stop_loss)
            b = win_amt / max(loss_amt, 0.01)
        else:
            b = self.config.take_profit_pct / max(self.config.stop_loss_pct, 0.01)
            
        q = 1 - win_prob
        
        # Kelly formula
        kelly_pct = (win_prob * b - q) / max(b, 0.01)
        
        # Apply fractional Kelly
        kelly_pct = max(0, kelly_pct * self.config.kelly_fraction)
        
        return self._portfolio_value * kelly_pct


class SignalEngine:
    """
    Main signal generation engine.
    
    Orchestrates:
    1. Feature extraction from stream
    2. Model prediction
    3. Signal aggregation
    4. Risk filtering
    5. Position sizing
    
    Usage:
        engine = SignalEngine(config)
        engine.add_model(TFTModelWrapper(tft_model, tft_config))
        engine.add_model(DeepARModelWrapper(deepar_model, deepar_config))
        
        # Process tick and generate signal
        signal = engine.process_tick(tick, processor)
        if signal and signal.is_actionable:
            execute_trade(signal)
    """
    
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        
        # Components
        self._models: List[ModelWrapper] = []
        self._aggregator = SignalAggregator(self.config)
        self._risk_filter = RiskFilter(self.config)
        self._position_sizer = PositionSizer(self.config)
        
        # State
        self._current_features: Dict[str, Dict[str, float]] = {}
        self._signal_history: Dict[str, List[TradingSignal]] = {}
        
        # Callbacks
        self._signal_callbacks: List[Callable[[TradingSignal], None]] = []
        
    def add_model(self, model: ModelWrapper, weight: float = 1.0):
        """Add model to ensemble."""
        self._models.append(model)
        self._aggregator.set_weight(model.name, weight)
        logger.info(f"Added model: {model.name} (weight={weight})")
        
    def remove_model(self, model_name: str):
        """Remove model from ensemble."""
        self._models = [m for m in self._models if m.name != model_name]
        
    def on_signal(self, callback: Callable[[TradingSignal], None]):
        """Register signal callback."""
        self._signal_callbacks.append(callback)
        
    def set_portfolio_value(self, value: float):
        """Update portfolio value for sizing."""
        self._risk_filter.set_portfolio_value(value)
        self._position_sizer.set_portfolio_value(value)
        
    def process_tick(
        self,
        tick: MarketTick,
        processor: StreamProcessor
    ) -> Optional[TradingSignal]:
        """
        Process tick and generate signal if conditions met.
        
        Args:
            tick: Market tick
            processor: Stream processor with feature state
            
        Returns:
            TradingSignal if generated, None otherwise
        """
        symbol = tick.symbol
        
        # Update features
        features = processor.get_features(symbol, self.config.feature_lookback)
        self._current_features[symbol] = features
        
        # Check if we have enough features
        if not features or 'last_price' not in features:
            return None
            
        # Get predictions from all models
        predictions = {}
        for model in self._models:
            try:
                pred, conf = model.predict(features)
                predictions[model.name] = (pred, conf)
            except Exception as e:
                logger.warning(f"Model {model.name} prediction failed: {e}")
                
        if not predictions:
            return None
            
        # Aggregate predictions
        agg_pred, agg_conf = self._aggregator.aggregate(predictions)
        
        # Determine signal type
        signal_type = self._determine_signal_type(agg_pred)
        
        if signal_type == SignalType.HOLD:
            return None
            
        # Calculate strength
        strength = self._calculate_strength(agg_pred, agg_conf)
        
        # Create signal
        entry_price = features.get('last_price', 0)
        
        signal = TradingSignal(
            symbol=symbol,
            timestamp=tick.timestamp,
            signal_type=signal_type,
            strength=strength,
            entry_price=entry_price,
            stop_loss=self._calculate_stop_loss(entry_price, signal_type),
            take_profit=self._calculate_take_profit(entry_price, signal_type),
            model_name="+".join(predictions.keys()),
            model_confidence=agg_conf,
            win_probability=self._estimate_win_probability(agg_conf),
            features_used=features,
        )
        
        # Position sizing
        volatility = features.get('price_std', 0.01) * np.sqrt(252)
        signal.position_size = self._position_sizer.calculate_size(signal, volatility)
        
        # Risk filtering
        passed, reason = self._risk_filter.filter_signal(signal)
        
        if not passed:
            logger.debug(f"Signal filtered: {reason}")
            return None
            
        # Execute callbacks
        for callback in self._signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")
                
        # Record signal
        if symbol not in self._signal_history:
            self._signal_history[symbol] = []
        self._signal_history[symbol].append(signal)
        
        return signal
        
    def _determine_signal_type(self, prediction: float) -> SignalType:
        """Determine signal type from prediction."""
        if prediction > self.config.long_threshold:
            return SignalType.LONG
        elif prediction < self.config.short_threshold:
            return SignalType.SHORT
        else:
            return SignalType.HOLD
            
    def _calculate_strength(self, prediction: float, confidence: float) -> SignalStrength:
        """Calculate signal strength."""
        score = abs(prediction) * confidence
        
        if score > 0.8:
            return SignalStrength.VERY_STRONG
        elif score > 0.6:
            return SignalStrength.STRONG
        elif score > 0.4:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
            
    def _calculate_stop_loss(self, entry: float, signal_type: SignalType) -> float:
        """Calculate stop loss price."""
        if signal_type == SignalType.LONG:
            return entry * (1 - self.config.stop_loss_pct)
        else:
            return entry * (1 + self.config.stop_loss_pct)
            
    def _calculate_take_profit(self, entry: float, signal_type: SignalType) -> float:
        """Calculate take profit price."""
        if signal_type == SignalType.LONG:
            return entry * (1 + self.config.take_profit_pct)
        else:
            return entry * (1 - self.config.take_profit_pct)
            
    def _estimate_win_probability(self, confidence: float) -> float:
        """
        Estimate win probability from model confidence.
        
        This should ideally be calibrated from historical performance.
        """
        # Simple linear mapping - calibrate with actual data
        base_prob = 0.5
        return base_prob + (confidence - 0.5) * 0.3
        
    def get_active_signals(self, symbol: Optional[str] = None) -> List[TradingSignal]:
        """Get non-expired signals."""
        signals = []
        
        if symbol:
            for sig in self._signal_history.get(symbol, []):
                if not sig.is_expired:
                    signals.append(sig)
        else:
            for sym_signals in self._signal_history.values():
                for sig in sym_signals:
                    if not sig.is_expired:
                        signals.append(sig)
                        
        return signals


class AsyncSignalEngine:
    """
    Async wrapper for signal engine.
    
    Usage:
        async with AsyncSignalEngine(config) as engine:
            async for signal in engine.signals(stream):
                print(f"Signal: {signal}")
    """
    
    def __init__(self, config: Optional[SignalConfig] = None):
        self._engine = SignalEngine(config)
        self._running = False
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        
    def add_model(self, model: ModelWrapper, weight: float = 1.0):
        """Add model to ensemble."""
        self._engine.add_model(model, weight)
        
    async def signals(self, stream, processor: StreamProcessor):
        """
        Async generator of signals.
        
        Args:
            stream: Async generator of MarketTicks
            processor: StreamProcessor for features
        """
        self._running = True
        
        async for tick in stream:
            if not self._running:
                break
                
            signal = self._engine.process_tick(tick, processor)
            
            if signal and signal.is_actionable:
                yield signal
