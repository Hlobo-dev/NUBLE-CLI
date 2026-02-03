"""
Signal Fusion Engine

The brain of NUBLE Elite - combines multiple signal sources
into optimal trading decisions.

Key innovations:
1. Dynamic weight adjustment based on recent accuracy
2. Agreement detection (sources agree = higher confidence)
3. Regime-adaptive fusion (different weights in bull vs bear)
4. Continuous learning from outcomes

Signal Sources:
- Technical (LuxAlgo): 50% base weight - proven, visual, real-time
- ML (AFML): 25% base weight - your trained models
- Sentiment (FinBERT): 10% base weight - news analysis
- Regime (HMM): 10% base weight - market state context
- Fundamental: 5% base weight - valuations (optional)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any
from enum import Enum
import logging
import numpy as np
import pandas as pd

from .luxalgo_webhook import (
    get_signal_store,
    LuxAlgoSignal,
    LuxAlgoSignalStore
)
from .base_source import SignalSource, NormalizedSignal

logger = logging.getLogger(__name__)


class FusedSignalStrength(Enum):
    """Strength of fused signal."""
    STRONG_BUY = 2
    BUY = 1
    WEAK_BUY = 0.5
    NEUTRAL = 0
    WEAK_SELL = -0.5
    SELL = -1
    STRONG_SELL = -2
    
    @property
    def label(self) -> str:
        labels = {
            2: "🟢🟢 STRONG BUY",
            1: "🟢 BUY",
            0.5: "🟢 WEAK BUY",
            0: "⚪ NEUTRAL",
            -0.5: "🔴 WEAK SELL",
            -1: "🔴 SELL",
            -2: "🔴🔴 STRONG SELL"
        }
        return labels.get(self.value, "⚪ UNKNOWN")


@dataclass
class FusedSignal:
    """
    A fused trading signal combining multiple sources.
    
    This is the final output of the fusion engine - a single
    decision that incorporates all available information.
    """
    symbol: str
    timestamp: datetime
    
    # Final decision
    direction: int              # -1, 0, 1
    strength: FusedSignalStrength
    confidence: float           # 0 to 1
    
    # Component signals (if available)
    luxalgo_signal: Optional[Dict] = None
    ml_signal: Optional[Dict] = None
    sentiment_signal: Optional[Dict] = None
    regime_signal: Optional[Dict] = None
    
    # Regime context
    regime: str = "UNKNOWN"
    
    # Position sizing
    recommended_size: float = 0.0    # 0 to 1
    
    # Risk levels
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    
    # Agreement metrics
    source_agreement: float = 0.0    # -1 to 1
    sources_bullish: int = 0
    sources_bearish: int = 0
    sources_neutral: int = 0
    
    # Reasoning
    reasoning: List[str] = field(default_factory=list)
    
    @property
    def risk_reward_ratio(self) -> float:
        if self.stop_loss_pct > 0:
            return self.take_profit_pct / self.stop_loss_pct
        return 0
    
    @property
    def is_actionable(self) -> bool:
        """Whether this signal should be acted on."""
        return (
            self.direction != 0 and 
            self.confidence > 0.4 and
            self.recommended_size > 0
        )
    
    @property
    def action_str(self) -> str:
        if self.direction > 0:
            return "BUY"
        elif self.direction < 0:
            return "SELL"
        return "HOLD"
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'direction': self.direction,
            'action': self.action_str,
            'strength': self.strength.value,
            'strength_label': self.strength.label,
            'confidence': round(self.confidence, 4),
            'is_actionable': self.is_actionable,
            
            # Component signals
            'luxalgo_signal': self.luxalgo_signal,
            'ml_signal': self.ml_signal,
            'sentiment_signal': self.sentiment_signal,
            
            # Context
            'regime': self.regime,
            
            # Position
            'recommended_size': round(self.recommended_size, 4),
            'stop_loss_pct': round(self.stop_loss_pct, 4),
            'take_profit_pct': round(self.take_profit_pct, 4),
            'risk_reward_ratio': round(self.risk_reward_ratio, 2),
            
            # Agreement
            'source_agreement': round(self.source_agreement, 4),
            'sources_bullish': self.sources_bullish,
            'sources_bearish': self.sources_bearish,
            
            # Reasoning
            'reasoning': self.reasoning
        }
    
    def __str__(self) -> str:
        return (
            f"{self.symbol}: {self.strength.label} "
            f"(conf={self.confidence:.0%}, size={self.recommended_size:.0%})"
        )


class SignalFusionEngine:
    """
    Multi-source signal fusion engine.
    
    Combines signals from multiple sources into trading decisions:
    1. LuxAlgo (TradingView) - Primary technical (50%)
    2. NUBLE ML - Secondary ML signals (25%)
    3. Sentiment - FinBERT news analysis (10%)
    4. Regime - HMM market state (10%)
    5. Fundamental - Optional valuations (5%)
    
    Key Rules:
    - LuxAlgo strong signal + ML agrees = STRONG signal, boost confidence
    - LuxAlgo signal + ML neutral = NORMAL signal
    - LuxAlgo signal + ML disagrees = WEAK signal, reduce size
    - No LuxAlgo signal = Use ML only (lower confidence)
    
    Example:
        engine = SignalFusionEngine()
        
        # Generate fused signal
        signal = engine.generate_fused_signal(
            symbol='ETHUSD',
            prices=eth_data,
            sentiment=0.3
        )
        
        if signal.is_actionable:
            print(f"Trade: {signal.action_str} {signal.symbol}")
            print(f"Size: {signal.recommended_size:.0%}")
    """
    
    def __init__(
        self,
        luxalgo_weight: float = 0.50,
        ml_weight: float = 0.25,
        sentiment_weight: float = 0.10,
        regime_weight: float = 0.10,
        fundamental_weight: float = 0.05,
        min_confidence_to_trade: float = 0.40,
        min_agreement_to_trade: float = 0.50
    ):
        """
        Initialize the fusion engine.
        
        Args:
            luxalgo_weight: Weight for LuxAlgo signals (default 50%)
            ml_weight: Weight for ML signals (default 25%)
            sentiment_weight: Weight for sentiment signals (default 10%)
            regime_weight: Weight for regime signals (default 10%)
            fundamental_weight: Weight for fundamental signals (default 5%)
            min_confidence_to_trade: Minimum confidence to generate trade
            min_agreement_to_trade: Minimum source agreement to generate trade
        """
        # Base weights (will be normalized)
        self.weights = {
            'luxalgo': luxalgo_weight,
            'ml': ml_weight,
            'sentiment': sentiment_weight,
            'regime': regime_weight,
            'fundamental': fundamental_weight
        }
        
        # Thresholds
        self.min_confidence = min_confidence_to_trade
        self.min_agreement = min_agreement_to_trade
        
        # Signal store
        self.signal_store = get_signal_store()
        
        # Accuracy tracking for dynamic weight adjustment
        self.source_accuracy: Dict[str, List[float]] = {
            k: [] for k in self.weights.keys()
        }
        
        # Dynamic weights (adjusted based on performance)
        self.dynamic_weights = self.weights.copy()
        
        # Prediction history for learning
        self.prediction_history: List[Dict] = []
        
        logger.info(
            f"SignalFusionEngine initialized with weights: "
            f"LuxAlgo={luxalgo_weight:.0%}, ML={ml_weight:.0%}, "
            f"Sentiment={sentiment_weight:.0%}, Regime={regime_weight:.0%}"
        )
    
    def generate_fused_signal(
        self,
        symbol: str,
        prices: pd.DataFrame = None,
        sentiment: float = None,
        regime: str = None,
        fundamental_score: float = None
    ) -> FusedSignal:
        """
        Generate optimal trading signal from all sources.
        
        Args:
            symbol: Asset symbol (e.g., 'ETHUSD', 'AAPL')
            prices: OHLCV DataFrame for ML signals
            sentiment: Pre-computed sentiment score (-1 to +1)
            regime: Pre-detected regime ('BULL', 'BEAR', 'SIDEWAYS')
            fundamental_score: Pre-computed fundamental score (-1 to +1)
            
        Returns:
            FusedSignal with combined decision
        """
        reasoning = []
        signals = {}  # source_name -> (direction, confidence)
        
        # 1. Get LuxAlgo signal (PRIMARY)
        luxalgo_data = self._get_luxalgo_signal(symbol)
        if luxalgo_data:
            signals['luxalgo'] = luxalgo_data
            consensus = luxalgo_data.get('consensus', {})
            reasoning.append(
                f"📊 LuxAlgo: {consensus.get('direction', 'NONE')} "
                f"({consensus.get('signal_count', 0)} signals, "
                f"{consensus.get('confidence', 0):.0%} confidence)"
            )
        else:
            reasoning.append("📊 LuxAlgo: No recent signals")
        
        # 2. Get ML signal (if prices available)
        ml_data = None
        if prices is not None and len(prices) > 60:
            ml_data = self._get_ml_signal(symbol, prices, regime)
            if ml_data:
                signals['ml'] = ml_data
                reasoning.append(
                    f"🤖 ML: {'BUY' if ml_data['direction'] > 0 else 'SELL' if ml_data['direction'] < 0 else 'NEUTRAL'} "
                    f"({ml_data['confidence']:.0%} confidence)"
                )
            else:
                reasoning.append("🤖 ML: No signal generated")
        else:
            reasoning.append("🤖 ML: No price data provided")
        
        # 3. Get sentiment signal
        sentiment_data = None
        if sentiment is not None:
            sentiment_data = {
                'direction': sentiment,
                'confidence': min(1.0, abs(sentiment) * 1.5)  # Scale confidence
            }
            signals['sentiment'] = sentiment_data
            reasoning.append(
                f"📰 Sentiment: {sentiment:+.2f} "
                f"({sentiment_data['confidence']:.0%} confidence)"
            )
        else:
            reasoning.append("📰 Sentiment: Not provided")
        
        # 4. Get regime signal
        if regime is None and prices is not None:
            regime = self._detect_regime(prices)
        regime = regime or "UNKNOWN"
        
        regime_bias = self._get_regime_bias(regime)
        signals['regime'] = {
            'direction': regime_bias,
            'confidence': 0.8  # High confidence in regime detection
        }
        reasoning.append(f"📈 Regime: {regime} (bias: {regime_bias:+.2f})")
        
        # 5. Get fundamental signal (optional)
        if fundamental_score is not None:
            signals['fundamental'] = {
                'direction': fundamental_score,
                'confidence': 0.6
            }
            reasoning.append(f"📋 Fundamental: {fundamental_score:+.2f}")
        
        # 6. Combine all signals
        combined_direction, combined_confidence = self._combine_signals(signals, regime)
        
        # 7. Calculate agreement
        agreement_score, bulls, bears, neutrals = self._calculate_agreement(signals)
        reasoning.append(
            f"🤝 Agreement: {agreement_score:.0%} "
            f"(📈{bulls} bullish, 📉{bears} bearish, ➖{neutrals} neutral)"
        )
        
        # 8. Adjust confidence based on agreement
        if agreement_score > 0.7:
            confidence_boost = 0.15
            reasoning.append("✅ High agreement - boosting confidence")
        elif agreement_score < 0.3:
            confidence_boost = -0.15
            reasoning.append("⚠️ Low agreement - reducing confidence")
        else:
            confidence_boost = 0
        
        final_confidence = min(1.0, max(0, combined_confidence + confidence_boost))
        
        # 9. Determine direction and strength
        direction = self._determine_direction(combined_direction)
        strength = self._score_to_strength(combined_direction, final_confidence)
        
        # 10. Calculate position size
        if direction == 0 or final_confidence < self.min_confidence:
            recommended_size = 0
            reasoning.append(
                f"🚫 No position: "
                f"{'signal too weak' if direction == 0 else f'confidence {final_confidence:.0%} below threshold'}"
            )
        else:
            recommended_size = self._calculate_position_size(
                combined_direction, final_confidence, regime, agreement_score
            )
            reasoning.append(f"💰 Position: {recommended_size:.0%} of capital")
        
        # 11. Calculate risk levels
        stop_loss_pct, take_profit_pct = self._calculate_risk_levels(
            direction, final_confidence, regime
        )
        
        # Build the fused signal
        return FusedSignal(
            symbol=symbol.upper(),
            timestamp=datetime.now(),
            direction=direction,
            strength=strength,
            confidence=final_confidence,
            luxalgo_signal=luxalgo_data.get('consensus') if luxalgo_data else None,
            ml_signal=ml_data,
            sentiment_signal=sentiment_data,
            regime_signal={'regime': regime, 'bias': regime_bias},
            regime=regime,
            recommended_size=recommended_size,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            source_agreement=agreement_score,
            sources_bullish=bulls,
            sources_bearish=bears,
            sources_neutral=neutrals,
            reasoning=reasoning
        )
    
    def _get_luxalgo_signal(self, symbol: str) -> Optional[Dict]:
        """Get LuxAlgo signal from store."""
        consensus = self.signal_store.get_signal_consensus(symbol.upper(), hours=24)
        
        if consensus['signal_count'] == 0:
            return None
        
        direction = 0
        if consensus['direction'] == 'BUY':
            direction = consensus['confidence']
        elif consensus['direction'] == 'SELL':
            direction = -consensus['confidence']
        
        return {
            'direction': direction,
            'confidence': consensus['confidence'],
            'consensus': consensus
        }
    
    def _get_ml_signal(
        self, 
        symbol: str, 
        prices: pd.DataFrame,
        regime: str = None
    ) -> Optional[Dict]:
        """Get signal from NUBLE ML system."""
        try:
            # Import your existing signal generator
            from institutional.signals.enhanced_signals import EnhancedSignalGenerator
            
            gen = EnhancedSignalGenerator()
            signal = gen.generate_signal(
                symbol=symbol,
                prices=prices,
                regime=regime or 'SIDEWAYS'
            )
            
            return {
                'direction': signal.direction * signal.confidence,
                'confidence': signal.confidence,
                'raw_signal': signal.to_dict()
            }
        except ImportError as e:
            logger.debug(f"ML signal generator not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"ML signal generation failed: {e}")
            return None
    
    def _detect_regime(self, prices: pd.DataFrame) -> str:
        """Detect market regime from prices."""
        if prices is None or len(prices) < 50:
            return "UNKNOWN"
        
        close = prices['close'] if 'close' in prices.columns else prices.iloc[:, 0]
        
        # Simple regime detection based on moving averages
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        price = close.iloc[-1]
        
        # Volatility
        returns = close.pct_change()
        vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        if vol > 0.35:
            return "VOLATILE"
        elif price > ma20 > ma50:
            return "BULL"
        elif price < ma20 < ma50:
            return "BEAR"
        else:
            return "SIDEWAYS"
    
    def _get_regime_bias(self, regime: str) -> float:
        """Get directional bias based on regime."""
        biases = {
            "BULL": 0.30,       # Bias long in bull market
            "BEAR": -0.15,     # Slight bias short in bear
            "SIDEWAYS": 0.10,  # Slight long bias (markets go up over time)
            "VOLATILE": 0.05,  # Very slight long bias
            "UNKNOWN": 0.0
        }
        return biases.get(regime.upper(), 0)
    
    def _combine_signals(
        self, 
        signals: Dict[str, Dict],
        regime: str
    ) -> Tuple[float, float]:
        """
        Combine all signals using weighted average.
        
        Returns:
            (combined_direction, combined_confidence)
        """
        if not signals:
            return 0, 0
        
        # Adjust weights for regime
        weights = self._adjust_weights_for_regime(regime)
        
        # Calculate weighted average
        weighted_direction = 0
        weighted_confidence = 0
        total_weight = 0
        
        for source, data in signals.items():
            if source not in weights:
                continue
            
            weight = weights[source]
            direction = data.get('direction', 0)
            confidence = data.get('confidence', 0.5)
            
            # Skip if no real signal
            if confidence == 0:
                continue
            
            weighted_direction += weight * direction * confidence
            weighted_confidence += weight * confidence
            total_weight += weight
        
        if total_weight == 0:
            return 0, 0
        
        # Normalize
        combined_direction = weighted_direction / total_weight
        combined_confidence = weighted_confidence / total_weight
        
        # Clip direction to [-1, 1]
        combined_direction = max(-1, min(1, combined_direction))
        
        return combined_direction, combined_confidence
    
    def _adjust_weights_for_regime(self, regime: str) -> Dict[str, float]:
        """Adjust source weights based on market regime."""
        weights = self.dynamic_weights.copy()
        
        if regime == "BULL":
            # In bull market: boost momentum/trend sources
            weights['luxalgo'] *= 1.2  # LuxAlgo is trend-following
            weights['ml'] *= 1.1
            weights['sentiment'] *= 0.9
        
        elif regime == "BEAR":
            # In bear market: be more defensive
            weights['luxalgo'] *= 1.1
            weights['ml'] *= 0.9
            weights['sentiment'] *= 1.2  # Sentiment matters more
        
        elif regime == "SIDEWAYS":
            # In sideways: mean reversion works
            weights['luxalgo'] *= 0.9
            weights['ml'] *= 1.2
            weights['sentiment'] *= 1.0
        
        elif regime == "VOLATILE":
            # In volatile: reduce all, be cautious
            weights['luxalgo'] *= 0.8
            weights['ml'] *= 0.7
            weights['sentiment'] *= 0.8
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _calculate_agreement(
        self, 
        signals: Dict[str, Dict]
    ) -> Tuple[float, int, int, int]:
        """
        Calculate how much sources agree.
        
        Returns:
            (agreement_score, bulls, bears, neutrals)
        """
        if not signals:
            return 0, 0, 0, 0
        
        bulls = 0
        bears = 0
        neutrals = 0
        
        for source, data in signals.items():
            direction = data.get('direction', 0)
            if direction > 0.1:
                bulls += 1
            elif direction < -0.1:
                bears += 1
            else:
                neutrals += 1
        
        total = bulls + bears + neutrals
        if total == 0:
            return 0, 0, 0, 0
        
        # Agreement = max(bulls, bears) / total
        # 1.0 = all agree, 0.5 = evenly split
        agreement = max(bulls, bears) / total
        
        return agreement, bulls, bears, neutrals
    
    def _determine_direction(self, combined_score: float) -> int:
        """Convert combined score to direction."""
        if combined_score > 0.08:
            return 1
        elif combined_score < -0.08:
            return -1
        return 0
    
    def _score_to_strength(
        self, 
        score: float, 
        confidence: float
    ) -> FusedSignalStrength:
        """Convert combined score to strength enum."""
        if score > 0.5 and confidence > 0.7:
            return FusedSignalStrength.STRONG_BUY
        elif score > 0.25:
            return FusedSignalStrength.BUY
        elif score > 0.08:
            return FusedSignalStrength.WEAK_BUY
        elif score < -0.5 and confidence > 0.7:
            return FusedSignalStrength.STRONG_SELL
        elif score < -0.25:
            return FusedSignalStrength.SELL
        elif score < -0.08:
            return FusedSignalStrength.WEAK_SELL
        else:
            return FusedSignalStrength.NEUTRAL
    
    def _calculate_position_size(
        self,
        score: float,
        confidence: float,
        regime: str,
        agreement: float
    ) -> float:
        """Calculate recommended position size."""
        # Base size from confidence (max 50%)
        base_size = confidence * 0.5
        
        # Adjust for signal strength
        strength_mult = min(1.5, 0.5 + abs(score))
        
        # Adjust for agreement
        if agreement > 0.7:
            agreement_mult = 1.2
        elif agreement < 0.4:
            agreement_mult = 0.7
        else:
            agreement_mult = 1.0
        
        # Adjust for regime
        regime_mult = {
            "BULL": 1.2,
            "BEAR": 0.7,
            "SIDEWAYS": 0.9,
            "VOLATILE": 0.6,
            "UNKNOWN": 0.8
        }.get(regime, 0.8)
        
        # Calculate final size
        size = base_size * strength_mult * agreement_mult * regime_mult
        
        # Cap at 100%
        return min(1.0, max(0, size))
    
    def _calculate_risk_levels(
        self,
        direction: int,
        confidence: float,
        regime: str
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        if direction == 0:
            return 0, 0
        
        # Base risk levels based on confidence
        if confidence > 0.7:
            stop_loss = 0.03  # 3% stop
            take_profit = 0.09  # 9% TP (3:1)
        elif confidence > 0.5:
            stop_loss = 0.025  # 2.5%
            take_profit = 0.05  # 5% (2:1)
        else:
            stop_loss = 0.02  # 2%
            take_profit = 0.03  # 3% (1.5:1)
        
        # Adjust for regime
        if regime == "VOLATILE":
            stop_loss *= 1.5  # Wider stops in volatile
            take_profit *= 1.3
        elif regime == "BULL" and direction == 1:
            take_profit *= 1.2  # Let winners run in bull
        
        return stop_loss, take_profit
    
    def update_source_accuracy(self, source: str, was_correct: bool):
        """Track accuracy of a source for dynamic weight adjustment."""
        if source not in self.source_accuracy:
            return
        
        self.source_accuracy[source].append(1.0 if was_correct else 0.0)
        
        # Keep only last 100
        if len(self.source_accuracy[source]) > 100:
            self.source_accuracy[source] = self.source_accuracy[source][-100:]
        
        # Update dynamic weights
        self._update_dynamic_weights()
    
    def _update_dynamic_weights(self):
        """Update weights based on recent accuracy."""
        for source in self.weights:
            history = self.source_accuracy.get(source, [])
            
            if len(history) < 10:
                # Not enough data, use base weight
                self.dynamic_weights[source] = self.weights[source]
                continue
            
            # Recent accuracy
            recent_accuracy = sum(history[-20:]) / len(history[-20:])
            
            # Adjust weight: better accuracy = higher weight
            # Accuracy of 0.5 = no change
            # Accuracy of 0.7 = 20% boost
            # Accuracy of 0.3 = 20% reduction
            adjustment = 1 + (recent_accuracy - 0.5) * 0.4
            
            self.dynamic_weights[source] = self.weights[source] * adjustment
        
        # Normalize
        total = sum(self.dynamic_weights.values())
        if total > 0:
            self.dynamic_weights = {
                k: v / total for k, v in self.dynamic_weights.items()
            }
    
    def get_source_stats(self) -> Dict:
        """Get statistics about each source."""
        stats = {}
        
        for source in self.weights:
            history = self.source_accuracy.get(source, [])
            
            stats[source] = {
                'base_weight': self.weights[source],
                'dynamic_weight': self.dynamic_weights.get(source, self.weights[source]),
                'predictions': len(history),
                'accuracy_all': sum(history) / len(history) if history else 0.5,
                'accuracy_recent': (
                    sum(history[-20:]) / len(history[-20:]) 
                    if len(history) >= 20 else 0.5
                )
            }
        
        return stats
    
    def log_prediction(
        self,
        symbol: str,
        signal: FusedSignal,
        outcome: str = None
    ):
        """Log a prediction for later analysis."""
        self.prediction_history.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'direction': signal.direction,
            'confidence': signal.confidence,
            'strength': signal.strength.value,
            'regime': signal.regime,
            'agreement': signal.source_agreement,
            'outcome': outcome  # Will be updated when known
        })
        
        # Keep only last 1000
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
