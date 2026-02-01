"""
Regime Signal Source - HMM

Uses Hidden Markov Model for market regime detection.
Provides regime context to the fusion engine.

Regimes:
- BULL: High return, moderate volatility → bullish bias
- BEAR: Negative return, high volatility → bearish/defensive bias
- SIDEWAYS: Low return, low volatility → mean reversion works
- VOLATILE: High volatility → smaller positions
"""

from datetime import datetime
from typing import Dict, Optional, Any
import logging
import pandas as pd
import numpy as np

from ..base_source import SignalSource, NormalizedSignal

logger = logging.getLogger(__name__)


class RegimeHMMSource(SignalSource):
    """
    Regime detection signal source using HMM.
    
    This source detects market regime and provides a directional
    bias based on the regime.
    
    The bias is used to:
    - Adjust signal weights in fusion
    - Modify position sizing
    - Filter trades in unfavorable regimes
    
    Regime Biases:
    - BULL: +0.30 (bullish bias)
    - BEAR: -0.15 (slight bearish bias)
    - SIDEWAYS: +0.10 (slight bullish, mean reversion)
    - VOLATILE: +0.05 (neutral-ish, smaller positions)
    
    Example:
        source = RegimeHMMSource()
        signal = source.generate_signal('SPY', prices_df)
        print(f"Regime: {signal.raw_data['regime']}")
    """
    
    name = "regime"
    base_weight = 0.10
    
    # Regime directional biases
    REGIME_BIASES = {
        'BULL': 0.30,
        'BEAR': -0.15,
        'SIDEWAYS': 0.10,
        'VOLATILE': 0.05,
        'UNKNOWN': 0.0
    }
    
    def __init__(
        self,
        weight: float = None,
        use_hmm: bool = True,
        n_regimes: int = 2,
        lookback: int = 252
    ):
        """
        Initialize regime source.
        
        Args:
            weight: Custom weight
            use_hmm: Use HMM detector (if False, use simple detection)
            n_regimes: Number of regimes for HMM
            lookback: Lookback period for regime detection
        """
        super().__init__(weight)
        self.use_hmm = use_hmm
        self.n_regimes = n_regimes
        self.lookback = lookback
        self._detector = None
    
    def _get_hmm_detector(self):
        """Lazy load HMM detector."""
        if self._detector is None and self.use_hmm:
            try:
                from institutional.regime.hmm_detector import HMMRegimeDetector
                self._detector = HMMRegimeDetector(
                    n_regimes=self.n_regimes
                )
            except ImportError as e:
                logger.debug(f"HMM detector not available: {e}")
                self.use_hmm = False
        return self._detector
    
    def generate_signal(
        self,
        symbol: str,
        data: Any = None,
        context: Dict = None
    ) -> Optional[NormalizedSignal]:
        """
        Generate regime signal.
        
        Args:
            symbol: Asset symbol
            data: Price DataFrame (OHLCV)
            context: Additional context (pre-detected regime, etc.)
            
        Returns:
            NormalizedSignal with regime information
        """
        context = context or {}
        
        # Option 1: Use pre-detected regime
        if 'regime' in context:
            regime = context['regime'].upper()
            confidence = context.get('regime_confidence', 0.8)
            
            direction = self.REGIME_BIASES.get(regime, 0)
            
            signal = NormalizedSignal(
                source_name=self.name,
                symbol=symbol,
                timestamp=datetime.now(),
                direction=direction,
                confidence=confidence,
                raw_data={'regime': regime, 'source': 'pre-detected'},
                reasoning=f"Regime: {regime} (bias: {direction:+.2f})"
            )
            self._last_signal = signal
            return signal
        
        # Option 2: Detect from price data
        if data is None or not isinstance(data, pd.DataFrame):
            return None
        
        if len(data) < 50:
            return None
        
        # Detect regime
        regime = self._detect_regime(data)
        direction = self.REGIME_BIASES.get(regime, 0)
        
        # Calculate volatility for confidence
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        returns = close.pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # Lower confidence in high volatility
        base_confidence = 0.8
        vol_adjustment = max(0, 0.3 - volatility) / 0.3 * 0.2  # Up to ±20%
        confidence = min(1.0, base_confidence + vol_adjustment)
        
        signal = NormalizedSignal(
            source_name=self.name,
            symbol=symbol,
            timestamp=datetime.now(),
            direction=direction,
            confidence=confidence,
            raw_data={
                'regime': regime,
                'volatility': round(volatility, 4),
                'source': 'detected'
            },
            reasoning=f"Regime: {regime} (vol={volatility:.0%}, bias={direction:+.2f})"
        )
        
        self._last_signal = signal
        return signal
    
    def _detect_regime(self, data: pd.DataFrame) -> str:
        """
        Detect market regime from price data.
        
        Uses HMM if available, otherwise simple moving average logic.
        """
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        returns = close.pct_change().dropna()
        
        # Try HMM first
        if self.use_hmm:
            detector = self._get_hmm_detector()
            if detector is not None:
                try:
                    # Fit and predict
                    if not hasattr(detector, 'is_fitted') or not detector.is_fitted:
                        detector.fit(returns)
                    
                    regimes = detector.predict(returns)
                    current_regime_idx = regimes.iloc[-1]
                    
                    # Map regime index to name
                    regime_map = detector.regime_names if hasattr(detector, 'regime_names') else {}
                    regime = regime_map.get(current_regime_idx, 'UNKNOWN')
                    
                    return regime.upper()
                except Exception as e:
                    logger.debug(f"HMM detection failed: {e}")
        
        # Fallback: Simple detection using moving averages
        return self._simple_regime_detection(close, returns)
    
    def _simple_regime_detection(
        self, 
        close: pd.Series, 
        returns: pd.Series
    ) -> str:
        """
        Simple regime detection using moving averages.
        
        Rules:
        - BULL: Price > MA20 > MA50
        - BEAR: Price < MA20 < MA50
        - VOLATILE: 20-day volatility > 35% annualized
        - SIDEWAYS: Everything else
        """
        if len(close) < 50:
            return 'UNKNOWN'
        
        price = close.iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        
        # Volatility check
        vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        if vol > 0.35:
            return 'VOLATILE'
        elif price > ma20 > ma50:
            return 'BULL'
        elif price < ma20 < ma50:
            return 'BEAR'
        else:
            return 'SIDEWAYS'
    
    def get_confidence(self) -> float:
        """Get confidence in regime detection."""
        # Regime detection is generally reliable
        base_confidence = 0.8
        
        if self.use_hmm and self._detector is not None:
            # Slightly higher for HMM
            base_confidence = 0.85
        
        return base_confidence
    
    def get_regime(self, data: pd.DataFrame) -> str:
        """Convenience method to just get regime string."""
        signal = self.generate_signal('TEMP', data)
        if signal:
            return signal.raw_data.get('regime', 'UNKNOWN')
        return 'UNKNOWN'
    
    def get_regime_stats(self, data: pd.DataFrame) -> Dict:
        """Get detailed regime statistics."""
        if data is None or len(data) < 50:
            return {'regime': 'UNKNOWN', 'stats': {}}
        
        close = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        returns = close.pct_change().dropna()
        
        # Calculate stats
        price = close.iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        regime = self._detect_regime(data)
        
        return {
            'regime': regime,
            'price': price,
            'ma20': ma20,
            'ma50': ma50,
            'price_vs_ma20': (price / ma20 - 1) * 100,  # % above/below
            'price_vs_ma50': (price / ma50 - 1) * 100,
            'volatility_annualized': vol,
            'bias': self.REGIME_BIASES.get(regime, 0),
            'recent_return_20d': returns.iloc[-20:].sum() if len(returns) >= 20 else 0
        }
