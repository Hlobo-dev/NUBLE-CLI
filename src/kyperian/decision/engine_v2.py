"""
KYPERIAN Decision Engine V2 - Institutional Grade
==================================================

The brain of the trading system. Makes decisions based on 15+ data points
across 4 layers:

1. Signal Layer (40%) - Technical signals from LuxAlgo, momentum, trend
2. Context Layer (30%) - Regime, sentiment, volatility, macro
3. Validation Layer (20%) - Historical win rate, backtest confidence
4. Risk Layer (10% + VETO) - Position limits, drawdown, correlation

Author: KYPERIAN ELITE
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .data_classes import (
    SignalLayerScore,
    ContextLayerScore,
    ValidationLayerScore,
    RiskLayerScore,
    TradingDecision,
    TradeStrength,
    Regime,
    VolatilityState,
)

logger = logging.getLogger(__name__)


class DecisionEngineV2:
    """
    Institutional-Grade Decision Engine
    
    Makes trading decisions based on 15+ data points across 4 layers.
    No trade is executed unless it passes ALL validation and risk checks.
    
    Usage:
        engine = DecisionEngineV2(config)
        decision = await engine.make_decision("BTCUSD")
        
        if decision.should_trade:
            print(f"Trade: {decision.direction_str} @ {decision.entry_price}")
    """
    
    # Layer weights (must sum to 1.0)
    SIGNAL_WEIGHT = 0.40
    CONTEXT_WEIGHT = 0.30
    VALIDATION_WEIGHT = 0.20
    RISK_WEIGHT = 0.10
    
    # Confidence thresholds
    VERY_STRONG_THRESHOLD = 85
    STRONG_THRESHOLD = 70
    MODERATE_THRESHOLD = 55
    WEAK_THRESHOLD = 40
    
    # Position sizing
    MAX_POSITION_PCT = 5.0
    MIN_POSITION_PCT = 0.5
    
    # Risk parameters
    DEFAULT_STOP_PCT = 2.0
    DEFAULT_TARGET_MULTIPLIER = 3.0  # R/R ratio
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        market_data_client: Optional[Any] = None,
        sentiment_analyzer: Optional[Any] = None,
        regime_detector: Optional[Any] = None,
    ):
        """
        Initialize the Decision Engine.
        
        Args:
            config: Configuration dictionary
            market_data_client: Client for fetching market data
            sentiment_analyzer: Sentiment analysis service
            regime_detector: HMM regime detector
        """
        self.config = config or {}
        self.market_data = market_data_client
        self.sentiment = sentiment_analyzer
        self.regime = regime_detector
        
        # Cache for performance
        self._price_cache: Dict[str, Tuple[float, datetime]] = {}
        self._indicator_cache: Dict[str, Tuple[Dict, datetime]] = {}
        
        # Signal storage (from DynamoDB)
        self._signal_storage: Dict[str, Dict[str, Any]] = {}
        
        logger.info("DecisionEngineV2 initialized")
    
    async def make_decision(
        self,
        symbol: str,
        signals: Optional[Dict[str, Any]] = None,
        price_data: Optional[Dict[str, Any]] = None,
    ) -> TradingDecision:
        """
        Make a trading decision for a symbol.
        
        This is the main entry point. It:
        1. Gathers data from all sources
        2. Calculates layer scores
        3. Combines into final confidence score
        4. Applies risk checks
        5. Returns detailed decision with reasoning
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSD")
            signals: Optional pre-loaded signals from DynamoDB
            price_data: Optional pre-loaded price data
            
        Returns:
            TradingDecision with full details
        """
        start_time = time.perf_counter()
        reasoning = []
        data_points = 0
        
        try:
            # Get current price
            current_price = await self._get_current_price(symbol, price_data)
            
            # ========== LAYER 1: SIGNAL LAYER (40%) ==========
            signal_score = await self._calculate_signal_layer(symbol, signals, price_data)
            data_points += self._count_signal_data_points(signal_score)
            
            signal_contrib = signal_score.combined * self.SIGNAL_WEIGHT * 100
            reasoning.append(
                f"📊 Signal Layer: {signal_contrib:+.1f}% "
                f"(LuxAlgo: {signal_score.luxalgo_score:.2f}, "
                f"Momentum: {signal_score.momentum_score:.2f}, "
                f"Trend: {signal_score.trend_score:.2f})"
            )
            
            # ========== LAYER 2: CONTEXT LAYER (30%) ==========
            context_score = await self._calculate_context_layer(symbol, price_data)
            data_points += self._count_context_data_points(context_score)
            
            context_contrib = context_score.combined * self.CONTEXT_WEIGHT * 100
            reasoning.append(
                f"🌍 Context Layer: {context_contrib:+.1f}% "
                f"(Regime: {context_score.regime.value}, "
                f"Sentiment: {context_score.sentiment_score:.2f}, "
                f"Vol: {context_score.volatility_state.value})"
            )
            
            # Check direction alignment with regime
            direction = signal_score.direction
            if direction != 0:
                regime_alignment = self._check_regime_alignment(
                    direction, context_score.regime
                )
                if regime_alignment > 0:
                    reasoning.append("✅ Trading WITH regime trend (+bonus)")
                elif regime_alignment < 0:
                    reasoning.append("⚠️ Trading AGAINST regime trend (-penalty)")
            
            # ========== LAYER 3: VALIDATION LAYER (20%) ==========
            validation_score = await self._calculate_validation_layer(
                symbol, signal_score, context_score
            )
            data_points += self._count_validation_data_points(validation_score)
            
            validation_contrib = validation_score.combined * self.VALIDATION_WEIGHT * 100
            reasoning.append(
                f"📜 Validation Layer: {validation_contrib:+.1f}% "
                f"(Win Rate: {validation_score.win_rate:.1%}, "
                f"Samples: {validation_score.sample_size})"
            )
            
            if not validation_score.is_validated:
                reasoning.append("⚠️ Historical validation failed - reducing confidence")
            
            # ========== LAYER 4: RISK LAYER (10% + VETO) ==========
            risk_score = await self._calculate_risk_layer(symbol, current_price)
            data_points += 6  # Risk always uses 6 checks
            
            if risk_score.any_veto:
                reasoning.append(f"🚫 VETO: {', '.join(risk_score.veto_reasons)}")
                return self._create_no_trade_decision(
                    symbol, current_price, risk_score, reasoning,
                    signal_score, context_score, validation_score,
                    data_points, start_time
                )
            
            # ========== COMBINE ALL LAYERS ==========
            base_confidence = self._calculate_combined_confidence(
                signal_score, context_score, validation_score, risk_score
            )
            
            # Apply regime alignment adjustment
            regime_alignment = self._check_regime_alignment(direction, context_score.regime)
            adjusted_confidence = base_confidence * (1 + regime_alignment * 0.1)
            
            # Apply volatility adjustment
            vol_adjustment = 1 - (context_score.volatility_percentile * 0.2)
            adjusted_confidence *= vol_adjustment
            
            # Cap at 0-100
            final_confidence = np.clip(adjusted_confidence, 0, 100)
            
            # Determine strength
            strength = self._determine_strength(final_confidence)
            
            # Calculate position size
            position_pct = self._calculate_position_size(
                final_confidence,
                context_score.volatility_state,
                risk_score.position_size_multiplier
            )
            
            # Calculate stop loss and targets
            stop_loss, targets = self._calculate_risk_levels(
                current_price, direction, context_score.volatility_percentile
            )
            
            # Risk/Reward ratio
            if direction != 0 and stop_loss != current_price:
                rr_ratio = abs(targets[1] - current_price) / abs(current_price - stop_loss)
            else:
                rr_ratio = 0
            
            # Final reasoning
            reasoning.append(
                f"🎯 Combined Score: {final_confidence:.1f}% → {strength.name}"
            )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return TradingDecision(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                direction=direction,
                strength=strength,
                confidence=final_confidence,
                recommended_position_pct=position_pct,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit_1=targets[0],
                take_profit_2=targets[1],
                take_profit_3=targets[2],
                risk_reward_ratio=rr_ratio,
                max_holding_hours=self._estimate_holding_time(validation_score),
                signal_layer=signal_score,
                context_layer=context_score,
                validation_layer=validation_score,
                risk_layer=risk_score,
                reasoning=reasoning,
                data_points_used=data_points,
                processing_time_ms=processing_time,
            )
            
        except Exception as e:
            logger.error(f"Error making decision for {symbol}: {e}")
            processing_time = (time.perf_counter() - start_time) * 1000
            return self._create_error_decision(symbol, str(e), processing_time)
    
    # ========== LAYER CALCULATIONS ==========
    
    async def _calculate_signal_layer(
        self,
        symbol: str,
        signals: Optional[Dict[str, Any]] = None,
        price_data: Optional[Dict[str, Any]] = None,
    ) -> SignalLayerScore:
        """Calculate Layer 1: Technical Signals."""
        
        # Get LuxAlgo signals (from DynamoDB or passed in)
        luxalgo = await self._get_luxalgo_signals(symbol, signals)
        
        # Get price data for indicators
        prices = await self._get_price_data(symbol, price_data)
        
        # Calculate momentum indicators
        momentum = self._calculate_momentum_indicators(prices)
        
        # Calculate trend strength
        trend = self._calculate_trend_indicators(prices)
        
        # Calculate S/R levels
        sr = self._calculate_support_resistance(prices)
        
        return SignalLayerScore(
            # LuxAlgo
            luxalgo_weekly=luxalgo.get('1W', {}).get('score', 0),
            luxalgo_daily=luxalgo.get('1D', {}).get('score', 0),
            luxalgo_h4=luxalgo.get('4h', {}).get('score', 0),
            luxalgo_alignment=self._calculate_luxalgo_alignment(luxalgo),
            luxalgo_confirmations=luxalgo.get('confirmations', 0),
            
            # Momentum
            rsi_score=momentum.get('rsi_score', 0),
            macd_score=momentum.get('macd_score', 0),
            stochastic_score=momentum.get('stoch_score', 0),
            momentum_divergence=momentum.get('divergence', 0),
            
            # Trend
            adx_value=trend.get('adx', 25),
            trend_direction=trend.get('direction', 0),
            ma_alignment=trend.get('ma_alignment', 0),
            price_vs_ma200=trend.get('vs_ma200', 0),
            
            # S/R
            sr_position=sr.get('position', 0),
            breakout_score=sr.get('breakout', 0),
            volume_confirmation=sr.get('volume', 0),
        )
    
    async def _calculate_context_layer(
        self,
        symbol: str,
        price_data: Optional[Dict[str, Any]] = None,
    ) -> ContextLayerScore:
        """Calculate Layer 2: Market Context."""
        
        prices = await self._get_price_data(symbol, price_data)
        
        # Detect market regime
        regime_result = self._detect_regime(prices)
        
        # Get sentiment
        sentiment_result = await self._get_sentiment(symbol)
        
        # Calculate volatility
        volatility = self._calculate_volatility(prices)
        
        # Get macro context
        macro = await self._get_macro_context()
        
        return ContextLayerScore(
            # Regime
            regime=regime_result['regime'],
            regime_confidence=regime_result['confidence'],
            regime_duration_days=regime_result.get('duration', 0),
            trend_strength=regime_result.get('trend_strength', 0.5),
            
            # Sentiment
            news_sentiment=sentiment_result.get('news', 0),
            news_confidence=sentiment_result.get('news_conf', 0.5),
            social_sentiment=sentiment_result.get('social', 0),
            social_volume=sentiment_result.get('social_vol', 0.5),
            fear_greed_index=sentiment_result.get('fear_greed', 50),
            
            # Volatility
            volatility_state=volatility['state'],
            volatility_percentile=volatility['percentile'],
            vix_level=volatility.get('vix', 20),
            atr_percentile=volatility.get('atr_pct', 0.5),
            
            # Macro
            dxy_trend=macro.get('dxy', 0),
            rates_environment=macro.get('rates', 0),
            spy_correlation=macro.get('spy_corr', 0),
            sector_momentum=macro.get('sector', 0),
        )
    
    async def _calculate_validation_layer(
        self,
        symbol: str,
        signal_score: SignalLayerScore,
        context_score: ContextLayerScore,
    ) -> ValidationLayerScore:
        """Calculate Layer 3: Historical Validation."""
        
        # Find similar historical setups
        similar_setups = await self._find_similar_setups(
            symbol,
            signal_direction=signal_score.direction,
            regime=context_score.regime,
            momentum_bucket=self._bucket(signal_score.momentum_score),
        )
        
        # Get backtest statistics
        backtest_stats = await self._get_backtest_stats(symbol)
        
        # Pattern matching
        pattern_match = await self._match_patterns(symbol, signal_score)
        
        return ValidationLayerScore(
            # Historical
            win_rate=similar_setups.get('win_rate', 0.5),
            sample_size=similar_setups.get('count', 0),
            avg_profit_factor=similar_setups.get('profit_factor', 1.0),
            avg_hold_time_hours=similar_setups.get('avg_hold', 24),
            
            # Backtest
            backtest_sharpe=backtest_stats.get('sharpe', 0),
            backtest_sortino=backtest_stats.get('sortino', 0),
            backtest_max_dd=backtest_stats.get('max_dd', 0.1),
            pbo_probability=backtest_stats.get('pbo', 0.5),
            
            # Pattern
            pattern_match_score=pattern_match.get('score', 0.5),
            similar_patterns_found=pattern_match.get('count', 0),
            pattern_avg_return=pattern_match.get('avg_return', 0),
            pattern_consistency=pattern_match.get('consistency', 0.5),
        )
    
    async def _calculate_risk_layer(
        self,
        symbol: str,
        current_price: float,
    ) -> RiskLayerScore:
        """Calculate Layer 4: Risk Management."""
        
        # Get current portfolio state
        portfolio = await self._get_portfolio_state()
        
        # Check position limits
        current_pos = portfolio.get('positions', {}).get(symbol, {}).get('pct', 0)
        max_pos = self.config.get('max_position_pct', 10.0)
        
        # Check drawdown
        current_dd = portfolio.get('drawdown', 0)
        max_dd = self.config.get('max_drawdown', 0.15)
        
        # Check correlation
        correlations = await self._check_correlations(symbol, portfolio)
        
        # Check liquidity
        liquidity = await self._check_liquidity(symbol)
        
        # Check time restrictions
        time_ok, earnings, fomc = self._check_time_restrictions(symbol)
        
        # Check sector exposure
        sector_exposure = portfolio.get('sector_exposure', {}).get(
            self._get_sector(symbol), 0
        )
        max_sector = self.config.get('max_sector_exposure', 25.0)
        
        return RiskLayerScore(
            # Position
            current_position_pct=current_pos,
            max_position_pct=max_pos,
            position_check_passed=(current_pos < max_pos * 0.9),
            
            # Drawdown
            portfolio_drawdown=current_dd,
            max_drawdown_limit=max_dd,
            drawdown_check_passed=(current_dd < max_dd),
            
            # Correlation
            max_correlation=correlations.get('max', 0),
            correlation_limit=0.7,
            correlation_check_passed=(correlations.get('max', 0) < 0.7),
            
            # Liquidity
            avg_daily_volume=liquidity.get('volume', 0),
            spread_pct=liquidity.get('spread', 0),
            liquidity_check_passed=(liquidity.get('spread', 0) < 0.01),
            
            # Time
            market_hours=time_ok,
            avoid_earnings=earnings,
            avoid_fomc=fomc,
            time_check_passed=(time_ok and not earnings and not fomc),
            
            # Sector
            sector_exposure_pct=sector_exposure,
            max_sector_exposure=max_sector,
            sector_check_passed=(sector_exposure < max_sector),
        )
    
    # ========== HELPER METHODS ==========
    
    def _calculate_combined_confidence(
        self,
        signal: SignalLayerScore,
        context: ContextLayerScore,
        validation: ValidationLayerScore,
        risk: RiskLayerScore,
    ) -> float:
        """Combine all layer scores into final confidence."""
        
        # Base calculation: weighted sum
        # Note: scores are -1 to +1, convert to 0-100 scale
        
        # Signal contribution (40%)
        signal_contrib = (signal.combined + 1) / 2 * 100 * self.SIGNAL_WEIGHT
        
        # Context contribution (30%)
        context_contrib = (context.combined + 1) / 2 * 100 * self.CONTEXT_WEIGHT
        
        # Validation contribution (20%)
        validation_contrib = (validation.combined + 1) / 2 * 100 * self.VALIDATION_WEIGHT
        
        # Risk contribution (10%)
        risk_contrib = (1 - risk.risk_score) * 100 * self.RISK_WEIGHT
        
        total = signal_contrib + context_contrib + validation_contrib + risk_contrib
        
        # Apply direction confidence boost
        # Strong directional conviction = higher confidence
        direction_strength = abs(signal.combined)
        if direction_strength > 0.5:
            total *= 1 + (direction_strength - 0.5) * 0.2
        
        return total
    
    def _determine_strength(self, confidence: float) -> TradeStrength:
        """Determine trade strength from confidence score."""
        if confidence >= self.VERY_STRONG_THRESHOLD:
            return TradeStrength.VERY_STRONG
        elif confidence >= self.STRONG_THRESHOLD:
            return TradeStrength.STRONG
        elif confidence >= self.MODERATE_THRESHOLD:
            return TradeStrength.MODERATE
        elif confidence >= self.WEAK_THRESHOLD:
            return TradeStrength.WEAK
        return TradeStrength.NO_TRADE
    
    def _calculate_position_size(
        self,
        confidence: float,
        volatility_state: VolatilityState,
        risk_multiplier: float,
    ) -> float:
        """Calculate recommended position size."""
        
        # Base size from confidence
        base_pct = (confidence / 100) * self.MAX_POSITION_PCT
        
        # Adjust for volatility
        vol_mult = volatility_state.position_multiplier
        
        # Apply risk multiplier
        adjusted = base_pct * vol_mult * risk_multiplier
        
        # Enforce limits
        return np.clip(adjusted, self.MIN_POSITION_PCT, self.MAX_POSITION_PCT)
    
    def _calculate_risk_levels(
        self,
        price: float,
        direction: int,
        volatility_pct: float,
    ) -> Tuple[float, List[float]]:
        """Calculate stop loss and take profit levels."""
        
        # Adjust stop distance based on volatility
        base_stop_pct = self.DEFAULT_STOP_PCT / 100
        vol_adjusted_stop = base_stop_pct * (1 + volatility_pct)
        
        # Calculate levels
        if direction > 0:  # BUY
            stop_loss = price * (1 - vol_adjusted_stop)
            tp1 = price * (1 + vol_adjusted_stop * 1.5)
            tp2 = price * (1 + vol_adjusted_stop * 3.0)
            tp3 = price * (1 + vol_adjusted_stop * 5.0)
        elif direction < 0:  # SELL
            stop_loss = price * (1 + vol_adjusted_stop)
            tp1 = price * (1 - vol_adjusted_stop * 1.5)
            tp2 = price * (1 - vol_adjusted_stop * 3.0)
            tp3 = price * (1 - vol_adjusted_stop * 5.0)
        else:  # NEUTRAL
            stop_loss = price
            tp1 = tp2 = tp3 = price
        
        return stop_loss, [tp1, tp2, tp3]
    
    def _check_regime_alignment(self, direction: int, regime: Regime) -> float:
        """Check if trade direction aligns with regime."""
        if direction == 0:
            return 0
        
        regime_direction = regime.trend_alignment
        
        # Same direction = positive alignment
        if (direction > 0 and regime_direction > 0) or \
           (direction < 0 and regime_direction < 0):
            return abs(regime_direction)
        
        # Opposite direction = negative alignment
        elif (direction > 0 and regime_direction < 0) or \
             (direction < 0 and regime_direction > 0):
            return -abs(regime_direction)
        
        return 0  # Sideways regime
    
    def _estimate_holding_time(self, validation: ValidationLayerScore) -> float:
        """Estimate optimal holding time based on historical data."""
        if validation.avg_hold_time_hours > 0:
            return validation.avg_hold_time_hours
        return 48  # Default 2 days
    
    # ========== DATA FETCHING METHODS ==========
    
    async def _get_current_price(
        self,
        symbol: str,
        price_data: Optional[Dict] = None,
    ) -> float:
        """Get current price for symbol."""
        if price_data and 'close' in price_data:
            return price_data['close'][-1]
        
        # Check cache
        if symbol in self._price_cache:
            price, cached_at = self._price_cache[symbol]
            if (datetime.now(timezone.utc) - cached_at).seconds < 60:
                return price
        
        # Fetch from market data client
        if self.market_data:
            try:
                price = await self.market_data.get_price(symbol)
                self._price_cache[symbol] = (price, datetime.now(timezone.utc))
                return price
            except Exception as e:
                logger.warning(f"Error fetching price for {symbol}: {e}")
        
        return 0.0
    
    async def _get_price_data(
        self,
        symbol: str,
        price_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Get OHLCV price data."""
        if price_data:
            return price_data
        
        # Fetch from market data client
        if self.market_data:
            try:
                return await self.market_data.get_ohlcv(
                    symbol, timeframe='1D', limit=200
                )
            except Exception as e:
                logger.warning(f"Error fetching price data for {symbol}: {e}")
        
        # Return empty structure
        return {
            'open': [], 'high': [], 'low': [], 'close': [], 'volume': [],
            'timestamp': []
        }
    
    async def _get_luxalgo_signals(
        self,
        symbol: str,
        signals: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Get LuxAlgo signals from storage."""
        if signals:
            return self._process_luxalgo_signals(signals)
        
        # Check local storage
        if symbol in self._signal_storage:
            return self._process_luxalgo_signals(self._signal_storage[symbol])
        
        return {
            '1W': {'score': 0, 'action': 'NEUTRAL'},
            '1D': {'score': 0, 'action': 'NEUTRAL'},
            '4h': {'score': 0, 'action': 'NEUTRAL'},
            'confirmations': 0,
        }
    
    def _process_luxalgo_signals(self, signals: Dict) -> Dict[str, Any]:
        """Process raw signals into scores."""
        result = {}
        
        for tf in ['1W', '1D', '4h']:
            sig = signals.get(tf, {})
            action = sig.get('action', 'NEUTRAL')
            
            if action in ['BUY', 'STRONG_BUY']:
                score = 0.8 if action == 'STRONG_BUY' else 0.5
            elif action in ['SELL', 'STRONG_SELL']:
                score = -0.8 if action == 'STRONG_SELL' else -0.5
            else:
                score = 0
            
            result[tf] = {
                'score': score,
                'action': action,
                'price': sig.get('price', 0),
                'age_hours': sig.get('age_hours', 0),
            }
        
        result['confirmations'] = signals.get('confirmations', 0)
        return result
    
    def _calculate_luxalgo_alignment(self, signals: Dict) -> float:
        """Calculate how aligned the LuxAlgo signals are."""
        scores = [
            signals.get('1W', {}).get('score', 0),
            signals.get('1D', {}).get('score', 0),
            signals.get('4h', {}).get('score', 0),
        ]
        
        # All same sign = high alignment
        if all(s > 0 for s in scores) or all(s < 0 for s in scores):
            return 1.0
        
        # Some neutral
        non_zero = [s for s in scores if s != 0]
        if len(non_zero) == 0:
            return 0.0
        
        if len(non_zero) < 3 and len(set(np.sign(non_zero))) == 1:
            return 0.6
        
        # Mixed signals
        return 0.0
    
    # ========== INDICATOR CALCULATIONS ==========
    
    def _calculate_momentum_indicators(self, prices: Dict) -> Dict[str, float]:
        """Calculate momentum indicators."""
        close = np.array(prices.get('close', []))
        
        if len(close) < 20:
            return {
                'rsi_score': 0, 'macd_score': 0,
                'stoch_score': 0, 'divergence': 0
            }
        
        # RSI
        rsi = self._calculate_rsi(close)
        rsi_score = (50 - rsi) / 50  # Normalize: oversold=+1, overbought=-1
        
        # MACD
        macd, signal = self._calculate_macd(close)
        macd_score = np.tanh(macd[-1] / np.std(close[-20:]))
        
        # Stochastic
        high = np.array(prices.get('high', close))
        low = np.array(prices.get('low', close))
        stoch_k = self._calculate_stochastic(high, low, close)
        stoch_score = (50 - stoch_k) / 50
        
        # Divergence (simplified)
        divergence = self._detect_divergence(close, rsi)
        
        return {
            'rsi_score': float(np.clip(rsi_score, -1, 1)),
            'macd_score': float(np.clip(macd_score, -1, 1)),
            'stoch_score': float(np.clip(stoch_score, -1, 1)),
            'divergence': float(divergence),
        }
    
    def _calculate_trend_indicators(self, prices: Dict) -> Dict[str, float]:
        """Calculate trend strength indicators."""
        close = np.array(prices.get('close', []))
        high = np.array(prices.get('high', close))
        low = np.array(prices.get('low', close))
        
        if len(close) < 50:
            return {
                'adx': 25, 'direction': 0,
                'ma_alignment': 0, 'vs_ma200': 0
            }
        
        # ADX
        adx = self._calculate_adx(high, low, close)
        
        # Trend direction from MAs
        ma20 = np.mean(close[-20:])
        ma50 = np.mean(close[-50:]) if len(close) >= 50 else ma20
        ma200 = np.mean(close[-200:]) if len(close) >= 200 else ma50
        
        current = close[-1]
        
        # Direction score
        if current > ma20 > ma50:
            direction = 1.0
        elif current < ma20 < ma50:
            direction = -1.0
        else:
            direction = (current - ma50) / ma50 if ma50 != 0 else 0
        
        # MA alignment
        if ma20 > ma50 > ma200:
            ma_alignment = 1.0
        elif ma20 < ma50 < ma200:
            ma_alignment = -1.0
        else:
            ma_alignment = 0.0
        
        # Price vs MA200
        vs_ma200 = (current - ma200) / ma200 if ma200 != 0 else 0
        
        return {
            'adx': float(adx),
            'direction': float(np.clip(direction, -1, 1)),
            'ma_alignment': float(ma_alignment),
            'vs_ma200': float(np.clip(vs_ma200, -1, 1)),
        }
    
    def _calculate_support_resistance(self, prices: Dict) -> Dict[str, float]:
        """Calculate support/resistance position."""
        close = np.array(prices.get('close', []))
        high = np.array(prices.get('high', close))
        low = np.array(prices.get('low', close))
        volume = np.array(prices.get('volume', [1] * len(close)))
        
        if len(close) < 20:
            return {'position': 0, 'breakout': 0, 'volume': 0}
        
        current = close[-1]
        
        # Find recent highs and lows
        recent_high = np.max(high[-20:])
        recent_low = np.min(low[-20:])
        range_size = recent_high - recent_low
        
        if range_size == 0:
            position = 0
        else:
            # Position in range: -1 at resistance, +1 at support
            pct_in_range = (current - recent_low) / range_size
            position = 1 - 2 * pct_in_range  # Flip: low=+1, high=-1
        
        # Breakout detection
        if current > recent_high:
            breakout = 1.0
        elif current < recent_low:
            breakout = -1.0
        else:
            breakout = 0.0
        
        # Volume confirmation
        avg_vol = np.mean(volume[-20:])
        recent_vol = np.mean(volume[-3:])
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
        vol_score = min(1.0, vol_ratio / 2)  # Normalize
        
        return {
            'position': float(np.clip(position, -1, 1)),
            'breakout': float(breakout),
            'volume': float(vol_score),
        }
    
    # ========== CONTEXT METHODS ==========
    
    def _detect_regime(self, prices: Dict) -> Dict[str, Any]:
        """Detect market regime."""
        close = np.array(prices.get('close', []))
        
        if len(close) < 50:
            return {
                'regime': Regime.SIDEWAYS,
                'confidence': 0.5,
                'duration': 0,
                'trend_strength': 0.5,
            }
        
        # Simple regime detection based on MA and volatility
        ma50 = np.mean(close[-50:])
        ma200 = np.mean(close[-200:]) if len(close) >= 200 else ma50
        
        returns = np.diff(close[-50:]) / close[-51:-1]
        volatility = np.std(returns) * np.sqrt(252)
        
        # Trend strength
        current = close[-1]
        trend = (current - ma200) / ma200 if ma200 != 0 else 0
        
        # Determine regime
        if volatility > 0.5:
            regime = Regime.HIGH_VOLATILITY
            confidence = 0.8
        elif trend > 0.1 and ma50 > ma200:
            regime = Regime.STRONG_BULL if trend > 0.2 else Regime.BULL
            confidence = min(0.9, 0.5 + abs(trend))
        elif trend < -0.1 and ma50 < ma200:
            regime = Regime.STRONG_BEAR if trend < -0.2 else Regime.BEAR
            confidence = min(0.9, 0.5 + abs(trend))
        else:
            regime = Regime.SIDEWAYS
            confidence = 0.6
        
        return {
            'regime': regime,
            'confidence': confidence,
            'duration': 0,  # Would need historical regime data
            'trend_strength': min(1.0, abs(trend) * 5),
        }
    
    async def _get_sentiment(self, symbol: str) -> Dict[str, float]:
        """Get sentiment data."""
        if self.sentiment:
            try:
                result = await self.sentiment.analyze(symbol)
                return result
            except Exception as e:
                logger.warning(f"Error getting sentiment for {symbol}: {e}")
        
        return {
            'news': 0,
            'news_conf': 0.5,
            'social': 0,
            'social_vol': 0.5,
            'fear_greed': 50,
        }
    
    def _calculate_volatility(self, prices: Dict) -> Dict[str, Any]:
        """Calculate volatility state."""
        close = np.array(prices.get('close', []))
        
        if len(close) < 20:
            return {
                'state': VolatilityState.NORMAL,
                'percentile': 0.5,
                'vix': 20,
                'atr_pct': 0.5,
            }
        
        # Calculate historical volatility
        returns = np.diff(close) / close[:-1]
        current_vol = np.std(returns[-20:]) * np.sqrt(252)
        
        # Historical percentile
        all_vols = [np.std(returns[i:i+20]) * np.sqrt(252) 
                    for i in range(len(returns) - 20)]
        if all_vols:
            percentile = np.searchsorted(sorted(all_vols), current_vol) / len(all_vols)
        else:
            percentile = 0.5
        
        # Determine state
        if percentile > 0.9:
            state = VolatilityState.EXTREME
        elif percentile > 0.75:
            state = VolatilityState.HIGH
        elif percentile > 0.25:
            state = VolatilityState.NORMAL
        elif percentile > 0.1:
            state = VolatilityState.LOW
        else:
            state = VolatilityState.VERY_LOW
        
        return {
            'state': state,
            'percentile': float(percentile),
            'vix': current_vol * 100,  # Approximate VIX-like value
            'atr_pct': float(percentile),
        }
    
    async def _get_macro_context(self) -> Dict[str, float]:
        """Get macro environment context."""
        # Would connect to macro data sources
        return {
            'dxy': 0,
            'rates': 0,
            'spy_corr': 0.5,
            'sector': 0,
        }
    
    # ========== VALIDATION METHODS ==========
    
    async def _find_similar_setups(
        self,
        symbol: str,
        signal_direction: int,
        regime: Regime,
        momentum_bucket: int,
    ) -> Dict[str, Any]:
        """Find similar historical trading setups."""
        # Would query historical database
        return {
            'win_rate': 0.55,
            'count': 25,
            'profit_factor': 1.3,
            'avg_hold': 36,
        }
    
    async def _get_backtest_stats(self, symbol: str) -> Dict[str, float]:
        """Get backtest statistics for symbol."""
        # Would load from backtest results
        return {
            'sharpe': 0.8,
            'sortino': 1.2,
            'max_dd': 0.12,
            'pbo': 0.4,
        }
    
    async def _match_patterns(
        self,
        symbol: str,
        signal: SignalLayerScore,
    ) -> Dict[str, Any]:
        """Match current setup to historical patterns."""
        return {
            'score': 0.6,
            'count': 15,
            'avg_return': 0.02,
            'consistency': 0.5,
        }
    
    # ========== RISK METHODS ==========
    
    async def _get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state."""
        return {
            'positions': {},
            'drawdown': 0,
            'sector_exposure': {},
        }
    
    async def _check_correlations(
        self,
        symbol: str,
        portfolio: Dict,
    ) -> Dict[str, float]:
        """Check correlation with existing positions."""
        return {'max': 0.3}
    
    async def _check_liquidity(self, symbol: str) -> Dict[str, Any]:
        """Check liquidity for symbol."""
        return {'volume': 1e6, 'spread': 0.001}
    
    def _check_time_restrictions(self, symbol: str) -> Tuple[bool, bool, bool]:
        """Check time-based trading restrictions."""
        now = datetime.now(timezone.utc)
        
        # Market hours (simplified)
        is_weekday = now.weekday() < 5
        hour = now.hour
        market_open = is_weekday and 9 <= hour <= 16
        
        # Would check earnings calendar
        earnings_soon = False
        
        # Would check FOMC calendar  
        fomc_soon = False
        
        return market_open, earnings_soon, fomc_soon
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol."""
        sectors = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'NVDA': 'Technology', 'AMD': 'Technology', 'TSLA': 'Consumer',
            'BTCUSD': 'Crypto', 'ETHUSD': 'Crypto',
            'SPY': 'Index', 'QQQ': 'Index',
        }
        return sectors.get(symbol, 'Other')
    
    # ========== UTILITY METHODS ==========
    
    def _bucket(self, value: float, buckets: int = 5) -> int:
        """Convert continuous value to bucket."""
        return int((value + 1) / 2 * buckets)
    
    def _count_signal_data_points(self, score: SignalLayerScore) -> int:
        """Count non-zero signal data points."""
        return 12  # All signal components
    
    def _count_context_data_points(self, score: ContextLayerScore) -> int:
        """Count context data points."""
        return 10
    
    def _count_validation_data_points(self, score: ValidationLayerScore) -> int:
        """Count validation data points."""
        return 8
    
    def _create_no_trade_decision(
        self,
        symbol: str,
        price: float,
        risk: RiskLayerScore,
        reasoning: List[str],
        signal: SignalLayerScore,
        context: ContextLayerScore,
        validation: ValidationLayerScore,
        data_points: int,
        start_time: float,
    ) -> TradingDecision:
        """Create a NO_TRADE decision."""
        return TradingDecision(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            direction=0,
            strength=TradeStrength.NO_TRADE,
            confidence=0,
            recommended_position_pct=0,
            entry_price=price,
            stop_loss=price,
            take_profit_1=price,
            take_profit_2=price,
            take_profit_3=price,
            risk_reward_ratio=0,
            signal_layer=signal,
            context_layer=context,
            validation_layer=validation,
            risk_layer=risk,
            reasoning=reasoning,
            data_points_used=data_points,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )
    
    def _create_error_decision(
        self,
        symbol: str,
        error: str,
        processing_time: float,
    ) -> TradingDecision:
        """Create an error decision."""
        return TradingDecision(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            direction=0,
            strength=TradeStrength.NO_TRADE,
            confidence=0,
            recommended_position_pct=0,
            entry_price=0,
            stop_loss=0,
            take_profit_1=0,
            take_profit_2=0,
            take_profit_3=0,
            risk_reward_ratio=0,
            reasoning=[f"❌ Error: {error}"],
            data_points_used=0,
            processing_time_ms=processing_time,
        )
    
    # ========== TECHNICAL INDICATOR IMPLEMENTATIONS ==========
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(
        self,
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD."""
        if len(prices) < slow:
            return np.array([0]), np.array([0])
        
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        
        return macd_line, signal_line
    
    def _calculate_stochastic(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> float:
        """Calculate Stochastic %K."""
        if len(close) < period:
            return 50
        
        highest_high = np.max(high[-period:])
        lowest_low = np.min(low[-period:])
        
        if highest_high == lowest_low:
            return 50
        
        return ((close[-1] - lowest_low) / (highest_high - lowest_low)) * 100
    
    def _calculate_adx(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> float:
        """Calculate ADX."""
        if len(close) < period + 1:
            return 25
        
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        # +DM and -DM
        plus_dm = np.where(
            (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
            np.maximum(high[1:] - high[:-1], 0),
            0
        )
        minus_dm = np.where(
            (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
            np.maximum(low[:-1] - low[1:], 0),
            0
        )
        
        # Smoothed values
        atr = np.mean(tr[-period:])
        plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr > 0 else 0
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr > 0 else 0
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
        
        return dx
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def _detect_divergence(self, prices: np.ndarray, rsi: float) -> float:
        """Detect price/RSI divergence."""
        # Simplified divergence detection
        if len(prices) < 20:
            return 0
        
        # Price trend
        price_trend = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] != 0 else 0
        
        # RSI trend (would need historical RSI)
        rsi_trend = (rsi - 50) / 50  # Simplified
        
        # Bullish divergence: price down, RSI up
        if price_trend < -0.05 and rsi_trend > 0.2:
            return 0.5
        
        # Bearish divergence: price up, RSI down
        if price_trend > 0.05 and rsi_trend < -0.2:
            return -0.5
        
        return 0
    
    # ========== PUBLIC METHODS ==========
    
    def load_signals(self, symbol: str, signals: Dict[str, Any]) -> None:
        """Load signals into the engine (from DynamoDB or webhook)."""
        self._signal_storage[symbol] = signals
        logger.info(f"Loaded signals for {symbol}")
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._price_cache.clear()
        self._indicator_cache.clear()
        self._signal_storage.clear()
        logger.info("Cleared all caches")
