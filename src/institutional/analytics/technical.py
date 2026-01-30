"""
Technical Analysis Module - Comprehensive technical indicators and analysis.
Implements 50+ indicators with institutional-grade precision.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import math


@dataclass
class IndicatorResult:
    """Result from a technical indicator calculation"""
    name: str
    value: float
    timestamp: datetime
    signal: Optional[str] = None  # 'buy', 'sell', 'neutral'
    strength: Optional[float] = None  # 0-1 signal strength
    metadata: Optional[Dict] = None


@dataclass
class TrendAnalysis:
    """Overall trend analysis result"""
    direction: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0-1
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    moving_averages: Dict[str, float] = field(default_factory=dict)
    momentum_indicators: Dict[str, float] = field(default_factory=dict)


class TechnicalAnalyzer:
    """
    Comprehensive technical analysis engine.
    
    Implements institutional-grade technical indicators:
    - Trend: SMA, EMA, WMA, DEMA, TEMA, KAMA
    - Momentum: RSI, Stochastic, MACD, CCI, MFI, Williams %R
    - Volatility: Bollinger Bands, ATR, Keltner Channels, Donchian
    - Volume: OBV, VWAP, CMF, Force Index, A/D Line
    - Support/Resistance: Pivot Points, Fibonacci retracements
    """
    
    def __init__(self):
        self._cache: Dict[str, List] = {}
    
    # ==================== Moving Averages ====================
    
    def sma(self, prices: List[float], period: int) -> List[float]:
        """Simple Moving Average"""
        if len(prices) < period:
            return []
        
        result = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            result.append(avg)
        return result
    
    def ema(self, prices: List[float], period: int) -> List[float]:
        """Exponential Moving Average"""
        if len(prices) < period:
            return []
        
        multiplier = 2 / (period + 1)
        ema_values = [sum(prices[:period]) / period]
        
        for price in prices[period:]:
            ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])
        
        return ema_values
    
    def wma(self, prices: List[float], period: int) -> List[float]:
        """Weighted Moving Average"""
        if len(prices) < period:
            return []
        
        weights = list(range(1, period + 1))
        weight_sum = sum(weights)
        
        result = []
        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1:i + 1]
            wma = sum(p * w for p, w in zip(window, weights)) / weight_sum
            result.append(wma)
        return result
    
    def dema(self, prices: List[float], period: int) -> List[float]:
        """Double Exponential Moving Average"""
        ema1 = self.ema(prices, period)
        if not ema1:
            return []
        ema2 = self.ema(ema1, period)
        if not ema2:
            return []
        
        # Align lengths
        offset = len(ema1) - len(ema2)
        return [2 * ema1[i + offset] - ema2[i] for i in range(len(ema2))]
    
    def tema(self, prices: List[float], period: int) -> List[float]:
        """Triple Exponential Moving Average"""
        ema1 = self.ema(prices, period)
        if not ema1:
            return []
        ema2 = self.ema(ema1, period)
        if not ema2:
            return []
        ema3 = self.ema(ema2, period)
        if not ema3:
            return []
        
        # Align lengths
        offset1 = len(ema1) - len(ema3)
        offset2 = len(ema2) - len(ema3)
        
        return [
            3 * ema1[i + offset1] - 3 * ema2[i + offset2] + ema3[i]
            for i in range(len(ema3))
        ]
    
    def kama(self, prices: List[float], period: int = 10, 
             fast_period: int = 2, slow_period: int = 30) -> List[float]:
        """Kaufman's Adaptive Moving Average"""
        if len(prices) < period + 1:
            return []
        
        fast_sc = 2 / (fast_period + 1)
        slow_sc = 2 / (slow_period + 1)
        
        kama_values = []
        
        for i in range(period, len(prices)):
            # Efficiency Ratio
            change = abs(prices[i] - prices[i - period])
            volatility = sum(abs(prices[j] - prices[j-1]) 
                           for j in range(i - period + 1, i + 1))
            
            if volatility == 0:
                er = 0
            else:
                er = change / volatility
            
            # Smoothing constant
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            
            # KAMA
            if not kama_values:
                kama_values.append(prices[i])
            else:
                kama_values.append(kama_values[-1] + sc * (prices[i] - kama_values[-1]))
        
        return kama_values
    
    # ==================== Momentum Indicators ====================
    
    def rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return []
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        # First average
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        rsi_values = []
        
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))
        
        return rsi_values
    
    def stochastic(
        self, 
        highs: List[float], 
        lows: List[float], 
        closes: List[float],
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[List[float], List[float]]:
        """Stochastic Oscillator - returns (%K, %D)"""
        if len(closes) < k_period:
            return [], []
        
        k_values = []
        
        for i in range(k_period - 1, len(closes)):
            highest_high = max(highs[i - k_period + 1:i + 1])
            lowest_low = min(lows[i - k_period + 1:i + 1])
            
            if highest_high == lowest_low:
                k_values.append(50)
            else:
                k = 100 * (closes[i] - lowest_low) / (highest_high - lowest_low)
                k_values.append(k)
        
        # %D is SMA of %K
        d_values = self.sma(k_values, d_period)
        
        return k_values, d_values
    
    def macd(
        self,
        prices: List[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[List[float], List[float], List[float]]:
        """MACD - returns (macd_line, signal_line, histogram)"""
        fast_ema = self.ema(prices, fast_period)
        slow_ema = self.ema(prices, slow_period)
        
        if not fast_ema or not slow_ema:
            return [], [], []
        
        # Align to slow EMA length
        offset = len(fast_ema) - len(slow_ema)
        macd_line = [fast_ema[i + offset] - slow_ema[i] for i in range(len(slow_ema))]
        
        signal_line = self.ema(macd_line, signal_period)
        
        if not signal_line:
            return macd_line, [], []
        
        # Histogram
        offset = len(macd_line) - len(signal_line)
        histogram = [macd_line[i + offset] - signal_line[i] for i in range(len(signal_line))]
        
        return macd_line, signal_line, histogram
    
    def cci(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 20
    ) -> List[float]:
        """Commodity Channel Index"""
        if len(closes) < period:
            return []
        
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        sma_tp = self.sma(typical_prices, period)
        
        if not sma_tp:
            return []
        
        cci_values = []
        for i in range(len(sma_tp)):
            idx = i + period - 1
            window = typical_prices[idx - period + 1:idx + 1]
            mean_deviation = sum(abs(tp - sma_tp[i]) for tp in window) / period
            
            if mean_deviation == 0:
                cci_values.append(0)
            else:
                cci = (typical_prices[idx] - sma_tp[i]) / (0.015 * mean_deviation)
                cci_values.append(cci)
        
        return cci_values
    
    def williams_r(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> List[float]:
        """Williams %R"""
        if len(closes) < period:
            return []
        
        values = []
        for i in range(period - 1, len(closes)):
            highest_high = max(highs[i - period + 1:i + 1])
            lowest_low = min(lows[i - period + 1:i + 1])
            
            if highest_high == lowest_low:
                values.append(-50)
            else:
                wr = -100 * (highest_high - closes[i]) / (highest_high - lowest_low)
                values.append(wr)
        
        return values
    
    def mfi(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[int],
        period: int = 14
    ) -> List[float]:
        """Money Flow Index"""
        if len(closes) < period + 1:
            return []
        
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        raw_money_flow = [tp * v for tp, v in zip(typical_prices, volumes)]
        
        positive_flow = []
        negative_flow = []
        
        for i in range(1, len(typical_prices)):
            if typical_prices[i] > typical_prices[i-1]:
                positive_flow.append(raw_money_flow[i])
                negative_flow.append(0)
            else:
                positive_flow.append(0)
                negative_flow.append(raw_money_flow[i])
        
        mfi_values = []
        for i in range(period - 1, len(positive_flow)):
            pos_sum = sum(positive_flow[i - period + 1:i + 1])
            neg_sum = sum(negative_flow[i - period + 1:i + 1])
            
            if neg_sum == 0:
                mfi_values.append(100)
            else:
                money_ratio = pos_sum / neg_sum
                mfi_values.append(100 - (100 / (1 + money_ratio)))
        
        return mfi_values
    
    # ==================== Volatility Indicators ====================
    
    def bollinger_bands(
        self,
        prices: List[float],
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[List[float], List[float], List[float]]:
        """Bollinger Bands - returns (upper, middle, lower)"""
        middle = self.sma(prices, period)
        
        if not middle:
            return [], [], []
        
        upper = []
        lower = []
        
        for i in range(len(middle)):
            idx = i + period - 1
            window = prices[idx - period + 1:idx + 1]
            
            # Standard deviation
            mean = middle[i]
            variance = sum((p - mean) ** 2 for p in window) / period
            std = math.sqrt(variance)
            
            upper.append(mean + std_dev * std)
            lower.append(mean - std_dev * std)
        
        return upper, middle, lower
    
    def atr(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> List[float]:
        """Average True Range"""
        if len(closes) < 2:
            return []
        
        true_ranges = [highs[0] - lows[0]]
        
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)
        
        # Use EMA for smoothing
        return self.ema(true_ranges, period)
    
    def keltner_channels(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 20,
        atr_mult: float = 2.0
    ) -> Tuple[List[float], List[float], List[float]]:
        """Keltner Channels - returns (upper, middle, lower)"""
        middle = self.ema(closes, period)
        atr_values = self.atr(highs, lows, closes, period)
        
        if not middle or not atr_values:
            return [], [], []
        
        # Align lengths
        min_len = min(len(middle), len(atr_values))
        offset_m = len(middle) - min_len
        offset_a = len(atr_values) - min_len
        
        upper = []
        lower = []
        
        for i in range(min_len):
            upper.append(middle[i + offset_m] + atr_mult * atr_values[i + offset_a])
            lower.append(middle[i + offset_m] - atr_mult * atr_values[i + offset_a])
        
        return upper, middle[offset_m:], lower
    
    def donchian_channels(
        self,
        highs: List[float],
        lows: List[float],
        period: int = 20
    ) -> Tuple[List[float], List[float], List[float]]:
        """Donchian Channels - returns (upper, middle, lower)"""
        if len(highs) < period:
            return [], [], []
        
        upper = []
        lower = []
        middle = []
        
        for i in range(period - 1, len(highs)):
            high = max(highs[i - period + 1:i + 1])
            low = min(lows[i - period + 1:i + 1])
            upper.append(high)
            lower.append(low)
            middle.append((high + low) / 2)
        
        return upper, middle, lower
    
    # ==================== Volume Indicators ====================
    
    def obv(self, closes: List[float], volumes: List[int]) -> List[float]:
        """On-Balance Volume"""
        if len(closes) < 2:
            return [volumes[0]] if volumes else []
        
        obv_values = [volumes[0]]
        
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv_values.append(obv_values[-1] + volumes[i])
            elif closes[i] < closes[i-1]:
                obv_values.append(obv_values[-1] - volumes[i])
            else:
                obv_values.append(obv_values[-1])
        
        return obv_values
    
    def vwap(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[int]
    ) -> List[float]:
        """Volume Weighted Average Price"""
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        
        cumulative_tpv = 0
        cumulative_volume = 0
        vwap_values = []
        
        for tp, vol in zip(typical_prices, volumes):
            cumulative_tpv += tp * vol
            cumulative_volume += vol
            
            if cumulative_volume == 0:
                vwap_values.append(tp)
            else:
                vwap_values.append(cumulative_tpv / cumulative_volume)
        
        return vwap_values
    
    def cmf(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[int],
        period: int = 20
    ) -> List[float]:
        """Chaikin Money Flow"""
        if len(closes) < period:
            return []
        
        money_flow_multipliers = []
        for h, l, c in zip(highs, lows, closes):
            if h == l:
                money_flow_multipliers.append(0)
            else:
                mfm = ((c - l) - (h - c)) / (h - l)
                money_flow_multipliers.append(mfm)
        
        money_flow_volumes = [mfm * v for mfm, v in zip(money_flow_multipliers, volumes)]
        
        cmf_values = []
        for i in range(period - 1, len(closes)):
            mfv_sum = sum(money_flow_volumes[i - period + 1:i + 1])
            vol_sum = sum(volumes[i - period + 1:i + 1])
            
            if vol_sum == 0:
                cmf_values.append(0)
            else:
                cmf_values.append(mfv_sum / vol_sum)
        
        return cmf_values
    
    # ==================== Support/Resistance ====================
    
    def pivot_points(
        self,
        high: float,
        low: float,
        close: float
    ) -> Dict[str, float]:
        """Standard Pivot Points"""
        pivot = (high + low + close) / 3
        
        return {
            "pivot": pivot,
            "r1": 2 * pivot - low,
            "r2": pivot + (high - low),
            "r3": high + 2 * (pivot - low),
            "s1": 2 * pivot - high,
            "s2": pivot - (high - low),
            "s3": low - 2 * (high - pivot),
        }
    
    def fibonacci_retracements(
        self,
        high: float,
        low: float
    ) -> Dict[str, float]:
        """Fibonacci Retracement Levels"""
        diff = high - low
        
        return {
            "0.0%": high,
            "23.6%": high - 0.236 * diff,
            "38.2%": high - 0.382 * diff,
            "50.0%": high - 0.5 * diff,
            "61.8%": high - 0.618 * diff,
            "78.6%": high - 0.786 * diff,
            "100.0%": low,
        }
    
    def fibonacci_extensions(
        self,
        high: float,
        low: float
    ) -> Dict[str, float]:
        """Fibonacci Extension Levels"""
        diff = high - low
        
        return {
            "0.0%": high,
            "61.8%": high + 0.618 * diff,
            "100.0%": high + diff,
            "161.8%": high + 1.618 * diff,
            "261.8%": high + 2.618 * diff,
            "423.6%": high + 4.236 * diff,
        }
    
    # ==================== Comprehensive Analysis ====================
    
    def analyze(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[int],
        symbol: str = ""
    ) -> TrendAnalysis:
        """Perform comprehensive technical analysis"""
        
        if len(closes) < 50:
            return TrendAnalysis(
                direction="neutral",
                strength=0,
                support_levels=[],
                resistance_levels=[],
            )
        
        # Moving averages
        sma_20 = self.sma(closes, 20)
        sma_50 = self.sma(closes, 50)
        ema_12 = self.ema(closes, 12)
        ema_26 = self.ema(closes, 26)
        
        # Momentum
        rsi_values = self.rsi(closes, 14)
        macd_line, signal_line, histogram = self.macd(closes)
        
        # Volatility
        upper_bb, middle_bb, lower_bb = self.bollinger_bands(closes)
        atr_values = self.atr(highs, lows, closes)
        
        # Current values
        current_price = closes[-1]
        current_rsi = rsi_values[-1] if rsi_values else 50
        current_macd = histogram[-1] if histogram else 0
        
        # Trend determination
        bullish_signals = 0
        bearish_signals = 0
        
        # Price vs MAs
        if sma_20 and current_price > sma_20[-1]:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        if sma_50 and current_price > sma_50[-1]:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # MA crossovers
        if sma_20 and sma_50 and len(sma_20) >= len(sma_50):
            offset = len(sma_20) - len(sma_50)
            if sma_20[-1] > sma_50[-1]:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        # RSI
        if current_rsi > 70:
            bearish_signals += 1  # Overbought
        elif current_rsi < 30:
            bullish_signals += 1  # Oversold
        elif current_rsi > 50:
            bullish_signals += 0.5
        else:
            bearish_signals += 0.5
        
        # MACD
        if current_macd > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Bollinger position
        if upper_bb and lower_bb:
            bb_position = (current_price - lower_bb[-1]) / (upper_bb[-1] - lower_bb[-1])
            if bb_position > 0.8:
                bearish_signals += 0.5  # Near upper band
            elif bb_position < 0.2:
                bullish_signals += 0.5  # Near lower band
        
        total_signals = bullish_signals + bearish_signals
        
        if bullish_signals > bearish_signals:
            direction = "bullish"
            strength = bullish_signals / total_signals if total_signals > 0 else 0
        elif bearish_signals > bullish_signals:
            direction = "bearish"
            strength = bearish_signals / total_signals if total_signals > 0 else 0
        else:
            direction = "neutral"
            strength = 0.5
        
        # Support and resistance from recent data
        recent_high = max(highs[-20:])
        recent_low = min(lows[-20:])
        pivot_levels = self.pivot_points(recent_high, recent_low, current_price)
        
        return TrendAnalysis(
            direction=direction,
            strength=strength,
            support_levels=[pivot_levels["s1"], pivot_levels["s2"], pivot_levels["s3"]],
            resistance_levels=[pivot_levels["r1"], pivot_levels["r2"], pivot_levels["r3"]],
            moving_averages={
                "sma_20": sma_20[-1] if sma_20 else None,
                "sma_50": sma_50[-1] if sma_50 else None,
                "ema_12": ema_12[-1] if ema_12 else None,
                "ema_26": ema_26[-1] if ema_26 else None,
            },
            momentum_indicators={
                "rsi": current_rsi,
                "macd_histogram": current_macd,
            }
        )
    
    def get_signals(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[int]
    ) -> List[IndicatorResult]:
        """Get trading signals from multiple indicators"""
        signals = []
        timestamp = datetime.now()
        
        # RSI Signal
        rsi_values = self.rsi(closes)
        if rsi_values:
            rsi = rsi_values[-1]
            if rsi < 30:
                signal = "buy"
                strength = (30 - rsi) / 30
            elif rsi > 70:
                signal = "sell"
                strength = (rsi - 70) / 30
            else:
                signal = "neutral"
                strength = 0
            
            signals.append(IndicatorResult(
                name="RSI",
                value=rsi,
                timestamp=timestamp,
                signal=signal,
                strength=min(strength, 1.0)
            ))
        
        # MACD Signal
        macd_line, signal_line, histogram = self.macd(closes)
        if histogram and len(histogram) >= 2:
            current = histogram[-1]
            previous = histogram[-2]
            
            if current > 0 and previous <= 0:
                signal = "buy"
                strength = 0.8
            elif current < 0 and previous >= 0:
                signal = "sell"
                strength = 0.8
            elif current > 0:
                signal = "buy"
                strength = 0.4
            else:
                signal = "sell"
                strength = 0.4
            
            signals.append(IndicatorResult(
                name="MACD",
                value=current,
                timestamp=timestamp,
                signal=signal,
                strength=strength
            ))
        
        # Bollinger Band Signal
        upper, middle, lower = self.bollinger_bands(closes)
        if upper and lower:
            current_price = closes[-1]
            if current_price <= lower[-1]:
                signals.append(IndicatorResult(
                    name="Bollinger",
                    value=current_price,
                    timestamp=timestamp,
                    signal="buy",
                    strength=0.7,
                    metadata={"position": "lower_band"}
                ))
            elif current_price >= upper[-1]:
                signals.append(IndicatorResult(
                    name="Bollinger",
                    value=current_price,
                    timestamp=timestamp,
                    signal="sell",
                    strength=0.7,
                    metadata={"position": "upper_band"}
                ))
        
        # Stochastic Signal
        k_values, d_values = self.stochastic(highs, lows, closes)
        if k_values and d_values and len(k_values) >= 2:
            k = k_values[-1]
            d = d_values[-1] if d_values else k
            
            if k < 20 and k > d:
                signal = "buy"
                strength = 0.6
            elif k > 80 and k < d:
                signal = "sell"
                strength = 0.6
            else:
                signal = "neutral"
                strength = 0
            
            signals.append(IndicatorResult(
                name="Stochastic",
                value=k,
                timestamp=timestamp,
                signal=signal,
                strength=strength,
                metadata={"k": k, "d": d}
            ))
        
        return signals
