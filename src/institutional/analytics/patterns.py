"""
Pattern Recognition Module - Chart pattern detection and analysis.
Implements classical patterns with optional CNN-based detection.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math


class PatternType(Enum):
    """Types of chart patterns"""
    # Reversal patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    
    # Continuation patterns
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    FLAG = "flag"
    PENNANT = "pennant"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    RECTANGLE = "rectangle"
    
    # Candlestick patterns
    DOJI = "doji"
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    HARAMI = "harami"
    SHOOTING_STAR = "shooting_star"
    HANGING_MAN = "hanging_man"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"


@dataclass
class PatternMatch:
    """Detected pattern match"""
    pattern_type: PatternType
    confidence: float  # 0 to 1
    start_index: int
    end_index: int
    direction: str  # 'bullish' or 'bearish'
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    description: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PatternRecognizer:
    """
    Chart pattern recognition engine.
    
    Features:
    - Classical chart patterns (H&S, triangles, etc.)
    - Candlestick pattern detection
    - Price target calculation
    - Pattern strength/confidence scoring
    
    For production CNN-based detection:
    - Install: pip install tensorflow or torch
    - Train CNN on labeled pattern images
    - Use image-based pattern recognition
    """
    
    def __init__(self, use_ml: bool = False):
        """
        Initialize pattern recognizer.
        
        Args:
            use_ml: If True, attempt to load CNN model for pattern detection
        """
        self.use_ml = use_ml
        self._model = None
        
        if use_ml:
            self._load_cnn_model()
    
    def _load_cnn_model(self):
        """Load CNN model for pattern recognition"""
        try:
            # This would load a pre-trained CNN model
            # Example using PyTorch:
            # self._model = torch.load('pattern_cnn.pth')
            print("CNN pattern detection not implemented. Using rule-based detection.")
            self.use_ml = False
        except Exception as e:
            print(f"Could not load CNN model: {e}")
            self.use_ml = False
    
    def _find_local_extrema(
        self,
        prices: List[float],
        window: int = 5
    ) -> Tuple[List[int], List[int]]:
        """Find local maxima and minima indices"""
        maxima = []
        minima = []
        
        for i in range(window, len(prices) - window):
            window_before = prices[i - window:i]
            window_after = prices[i + 1:i + window + 1]
            
            # Local maximum
            if prices[i] >= max(window_before) and prices[i] >= max(window_after):
                maxima.append(i)
            
            # Local minimum
            if prices[i] <= min(window_before) and prices[i] <= min(window_after):
                minima.append(i)
        
        return maxima, minima
    
    def _is_near(self, a: float, b: float, tolerance: float = 0.02) -> bool:
        """Check if two values are within tolerance of each other"""
        if b == 0:
            return abs(a) < tolerance
        return abs(a - b) / b < tolerance
    
    # ==================== Candlestick Patterns ====================
    
    def detect_doji(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        index: int
    ) -> Optional[PatternMatch]:
        """Detect Doji candlestick"""
        if index < 0 or index >= len(closes):
            return None
        
        o, h, l, c = opens[index], highs[index], lows[index], closes[index]
        body = abs(c - o)
        total_range = h - l
        
        if total_range == 0:
            return None
        
        body_ratio = body / total_range
        
        if body_ratio < 0.1:  # Body is less than 10% of range
            return PatternMatch(
                pattern_type=PatternType.DOJI,
                confidence=1 - body_ratio * 5,  # Higher confidence for smaller body
                start_index=index,
                end_index=index,
                direction="neutral",
                description="Doji indicates indecision, potential reversal"
            )
        
        return None
    
    def detect_hammer(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        index: int
    ) -> Optional[PatternMatch]:
        """Detect Hammer or Hanging Man candlestick"""
        if index < 1 or index >= len(closes):
            return None
        
        o, h, l, c = opens[index], highs[index], lows[index], closes[index]
        
        body = abs(c - o)
        total_range = h - l
        
        if total_range == 0:
            return None
        
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)
        
        # Hammer: small body at top, long lower shadow
        if lower_shadow > body * 2 and upper_shadow < body * 0.5:
            # Determine if in downtrend (hammer) or uptrend (hanging man)
            if closes[index-1] > closes[index]:
                return PatternMatch(
                    pattern_type=PatternType.HAMMER,
                    confidence=min(lower_shadow / body / 3, 0.9),
                    start_index=index,
                    end_index=index,
                    direction="bullish",
                    description="Hammer in downtrend suggests potential reversal up"
                )
            else:
                return PatternMatch(
                    pattern_type=PatternType.HANGING_MAN,
                    confidence=min(lower_shadow / body / 3, 0.9),
                    start_index=index,
                    end_index=index,
                    direction="bearish",
                    description="Hanging man in uptrend suggests potential reversal down"
                )
        
        return None
    
    def detect_engulfing(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        index: int
    ) -> Optional[PatternMatch]:
        """Detect Bullish or Bearish Engulfing pattern"""
        if index < 1 or index >= len(closes):
            return None
        
        # Previous candle
        prev_o, prev_c = opens[index-1], closes[index-1]
        prev_body_high = max(prev_o, prev_c)
        prev_body_low = min(prev_o, prev_c)
        
        # Current candle
        curr_o, curr_c = opens[index], closes[index]
        curr_body_high = max(curr_o, curr_c)
        curr_body_low = min(curr_o, curr_c)
        
        # Bullish engulfing: prev is bearish, current is bullish and engulfs
        if prev_c < prev_o and curr_c > curr_o:
            if curr_body_low <= prev_body_low and curr_body_high >= prev_body_high:
                return PatternMatch(
                    pattern_type=PatternType.ENGULFING_BULLISH,
                    confidence=0.8,
                    start_index=index-1,
                    end_index=index,
                    direction="bullish",
                    description="Bullish engulfing pattern - strong buy signal"
                )
        
        # Bearish engulfing: prev is bullish, current is bearish and engulfs
        if prev_c > prev_o and curr_c < curr_o:
            if curr_body_low <= prev_body_low and curr_body_high >= prev_body_high:
                return PatternMatch(
                    pattern_type=PatternType.ENGULFING_BEARISH,
                    confidence=0.8,
                    start_index=index-1,
                    end_index=index,
                    direction="bearish",
                    description="Bearish engulfing pattern - strong sell signal"
                )
        
        return None
    
    def detect_morning_evening_star(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        index: int
    ) -> Optional[PatternMatch]:
        """Detect Morning Star or Evening Star pattern"""
        if index < 2 or index >= len(closes):
            return None
        
        # Three candles
        c1_o, c1_c = opens[index-2], closes[index-2]
        c2_o, c2_c = opens[index-1], closes[index-1]
        c3_o, c3_c = opens[index], closes[index]
        
        c1_body = abs(c1_c - c1_o)
        c2_body = abs(c2_c - c2_o)
        c3_body = abs(c3_c - c3_o)
        
        # Morning star: big bearish, small body, big bullish
        if (c1_c < c1_o and  # First is bearish
            c2_body < c1_body * 0.3 and  # Second is small
            c3_c > c3_o and  # Third is bullish
            c3_c > (c1_o + c1_c) / 2):  # Third closes above midpoint of first
            
            return PatternMatch(
                pattern_type=PatternType.MORNING_STAR,
                confidence=0.85,
                start_index=index-2,
                end_index=index,
                direction="bullish",
                description="Morning star pattern - strong reversal signal"
            )
        
        # Evening star: big bullish, small body, big bearish
        if (c1_c > c1_o and  # First is bullish
            c2_body < c1_body * 0.3 and  # Second is small
            c3_c < c3_o and  # Third is bearish
            c3_c < (c1_o + c1_c) / 2):  # Third closes below midpoint of first
            
            return PatternMatch(
                pattern_type=PatternType.EVENING_STAR,
                confidence=0.85,
                start_index=index-2,
                end_index=index,
                direction="bearish",
                description="Evening star pattern - strong reversal signal"
            )
        
        return None
    
    # ==================== Chart Patterns ====================
    
    def detect_double_top_bottom(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        window: int = 20
    ) -> List[PatternMatch]:
        """Detect Double Top and Double Bottom patterns"""
        patterns = []
        
        maxima, minima = self._find_local_extrema(closes, window=5)
        
        # Double Top: Two similar highs with a valley between
        for i in range(len(maxima) - 1):
            for j in range(i + 1, len(maxima)):
                idx1, idx2 = maxima[i], maxima[j]
                
                if idx2 - idx1 < window // 2 or idx2 - idx1 > window * 3:
                    continue
                
                high1, high2 = closes[idx1], closes[idx2]
                
                if self._is_near(high1, high2, tolerance=0.03):
                    # Find valley between peaks
                    valley_idx = min(range(idx1, idx2), key=lambda x: closes[x])
                    valley = closes[valley_idx]
                    
                    # Neckline
                    neckline = valley
                    target = neckline - (high1 - neckline)
                    
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.DOUBLE_TOP,
                        confidence=0.75,
                        start_index=idx1,
                        end_index=idx2,
                        direction="bearish",
                        price_target=target,
                        stop_loss=max(high1, high2) * 1.02,
                        description=f"Double top at {high1:.2f}, neckline at {neckline:.2f}"
                    ))
        
        # Double Bottom: Two similar lows with a peak between
        for i in range(len(minima) - 1):
            for j in range(i + 1, len(minima)):
                idx1, idx2 = minima[i], minima[j]
                
                if idx2 - idx1 < window // 2 or idx2 - idx1 > window * 3:
                    continue
                
                low1, low2 = closes[idx1], closes[idx2]
                
                if self._is_near(low1, low2, tolerance=0.03):
                    # Find peak between troughs
                    peak_idx = max(range(idx1, idx2), key=lambda x: closes[x])
                    peak = closes[peak_idx]
                    
                    # Neckline
                    neckline = peak
                    target = neckline + (neckline - low1)
                    
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.DOUBLE_BOTTOM,
                        confidence=0.75,
                        start_index=idx1,
                        end_index=idx2,
                        direction="bullish",
                        price_target=target,
                        stop_loss=min(low1, low2) * 0.98,
                        description=f"Double bottom at {low1:.2f}, neckline at {neckline:.2f}"
                    ))
        
        return patterns
    
    def detect_head_and_shoulders(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        window: int = 20
    ) -> List[PatternMatch]:
        """Detect Head and Shoulders / Inverse H&S patterns"""
        patterns = []
        
        maxima, minima = self._find_local_extrema(closes, window=5)
        
        # Head and Shoulders: Left shoulder, head (highest), right shoulder
        for i in range(len(maxima) - 2):
            left_shoulder_idx = maxima[i]
            
            for j in range(i + 1, len(maxima) - 1):
                head_idx = maxima[j]
                
                for k in range(j + 1, len(maxima)):
                    right_shoulder_idx = maxima[k]
                    
                    left_shoulder = closes[left_shoulder_idx]
                    head = closes[head_idx]
                    right_shoulder = closes[right_shoulder_idx]
                    
                    # Head must be highest, shoulders should be similar
                    if (head > left_shoulder and 
                        head > right_shoulder and
                        self._is_near(left_shoulder, right_shoulder, tolerance=0.05)):
                        
                        # Find neckline (troughs between)
                        trough1_idx = min(range(left_shoulder_idx, head_idx), 
                                        key=lambda x: closes[x])
                        trough2_idx = min(range(head_idx, right_shoulder_idx),
                                        key=lambda x: closes[x])
                        
                        neckline = (closes[trough1_idx] + closes[trough2_idx]) / 2
                        target = neckline - (head - neckline)
                        
                        patterns.append(PatternMatch(
                            pattern_type=PatternType.HEAD_AND_SHOULDERS,
                            confidence=0.8,
                            start_index=left_shoulder_idx,
                            end_index=right_shoulder_idx,
                            direction="bearish",
                            price_target=target,
                            stop_loss=head * 1.02,
                            description=f"H&S pattern, neckline at {neckline:.2f}"
                        ))
        
        # Inverse Head and Shoulders
        for i in range(len(minima) - 2):
            left_shoulder_idx = minima[i]
            
            for j in range(i + 1, len(minima) - 1):
                head_idx = minima[j]
                
                for k in range(j + 1, len(minima)):
                    right_shoulder_idx = minima[k]
                    
                    left_shoulder = closes[left_shoulder_idx]
                    head = closes[head_idx]
                    right_shoulder = closes[right_shoulder_idx]
                    
                    # Head must be lowest, shoulders should be similar
                    if (head < left_shoulder and 
                        head < right_shoulder and
                        self._is_near(left_shoulder, right_shoulder, tolerance=0.05)):
                        
                        # Find neckline (peaks between)
                        peak1_idx = max(range(left_shoulder_idx, head_idx),
                                       key=lambda x: closes[x])
                        peak2_idx = max(range(head_idx, right_shoulder_idx),
                                       key=lambda x: closes[x])
                        
                        neckline = (closes[peak1_idx] + closes[peak2_idx]) / 2
                        target = neckline + (neckline - head)
                        
                        patterns.append(PatternMatch(
                            pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
                            confidence=0.8,
                            start_index=left_shoulder_idx,
                            end_index=right_shoulder_idx,
                            direction="bullish",
                            price_target=target,
                            stop_loss=head * 0.98,
                            description=f"Inverse H&S pattern, neckline at {neckline:.2f}"
                        ))
        
        return patterns
    
    def detect_triangles(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        min_points: int = 4
    ) -> List[PatternMatch]:
        """Detect Triangle patterns (ascending, descending, symmetrical)"""
        patterns = []
        
        maxima, minima = self._find_local_extrema(closes, window=3)
        
        if len(maxima) < 2 or len(minima) < 2:
            return patterns
        
        # Check last few points
        recent_maxima = maxima[-min_points:]
        recent_minima = minima[-min_points:]
        
        if len(recent_maxima) < 2 or len(recent_minima) < 2:
            return patterns
        
        # Calculate slopes
        high_values = [closes[i] for i in recent_maxima]
        low_values = [closes[i] for i in recent_minima]
        
        # Linear regression for trend
        def calculate_slope(values):
            n = len(values)
            if n < 2:
                return 0
            x_mean = (n - 1) / 2
            y_mean = sum(values) / n
            numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            return numerator / denominator if denominator != 0 else 0
        
        high_slope = calculate_slope(high_values)
        low_slope = calculate_slope(low_values)
        
        # Normalize slopes by average price
        avg_price = (sum(high_values) + sum(low_values)) / (len(high_values) + len(low_values))
        high_slope_norm = high_slope / avg_price
        low_slope_norm = low_slope / avg_price
        
        start_idx = min(recent_maxima[0], recent_minima[0])
        end_idx = max(recent_maxima[-1], recent_minima[-1])
        
        # Ascending triangle: flat highs, rising lows
        if abs(high_slope_norm) < 0.01 and low_slope_norm > 0.01:
            resistance = sum(high_values) / len(high_values)
            patterns.append(PatternMatch(
                pattern_type=PatternType.ASCENDING_TRIANGLE,
                confidence=0.7,
                start_index=start_idx,
                end_index=end_idx,
                direction="bullish",
                price_target=resistance + (resistance - min(low_values)),
                description=f"Ascending triangle, resistance at {resistance:.2f}"
            ))
        
        # Descending triangle: flat lows, falling highs
        elif abs(low_slope_norm) < 0.01 and high_slope_norm < -0.01:
            support = sum(low_values) / len(low_values)
            patterns.append(PatternMatch(
                pattern_type=PatternType.DESCENDING_TRIANGLE,
                confidence=0.7,
                start_index=start_idx,
                end_index=end_idx,
                direction="bearish",
                price_target=support - (max(high_values) - support),
                description=f"Descending triangle, support at {support:.2f}"
            ))
        
        # Symmetrical triangle: converging highs and lows
        elif high_slope_norm < -0.005 and low_slope_norm > 0.005:
            patterns.append(PatternMatch(
                pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
                confidence=0.65,
                start_index=start_idx,
                end_index=end_idx,
                direction="neutral",
                description="Symmetrical triangle - breakout direction determines trend"
            ))
        
        return patterns
    
    def scan_candlestick_patterns(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        lookback: int = 20
    ) -> List[PatternMatch]:
        """Scan for candlestick patterns in recent data"""
        patterns = []
        
        start = max(0, len(closes) - lookback)
        
        for i in range(start, len(closes)):
            # Check each pattern type
            doji = self.detect_doji(opens, highs, lows, closes, i)
            if doji:
                patterns.append(doji)
            
            hammer = self.detect_hammer(opens, highs, lows, closes, i)
            if hammer:
                patterns.append(hammer)
            
            engulfing = self.detect_engulfing(opens, highs, lows, closes, i)
            if engulfing:
                patterns.append(engulfing)
            
            star = self.detect_morning_evening_star(opens, highs, lows, closes, i)
            if star:
                patterns.append(star)
        
        return patterns
    
    def analyze(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: Optional[List[int]] = None
    ) -> Dict:
        """
        Perform comprehensive pattern analysis.
        
        Returns dict with detected patterns, signals, and recommendations.
        """
        all_patterns = []
        
        # Candlestick patterns
        candlestick_patterns = self.scan_candlestick_patterns(
            opens, highs, lows, closes, lookback=10
        )
        all_patterns.extend(candlestick_patterns)
        
        # Chart patterns
        double_patterns = self.detect_double_top_bottom(highs, lows, closes)
        all_patterns.extend(double_patterns)
        
        hs_patterns = self.detect_head_and_shoulders(highs, lows, closes)
        all_patterns.extend(hs_patterns)
        
        triangle_patterns = self.detect_triangles(highs, lows, closes)
        all_patterns.extend(triangle_patterns)
        
        # Sort by confidence
        all_patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        # Determine overall bias
        bullish_patterns = [p for p in all_patterns if p.direction == "bullish"]
        bearish_patterns = [p for p in all_patterns if p.direction == "bearish"]
        
        bullish_score = sum(p.confidence for p in bullish_patterns)
        bearish_score = sum(p.confidence for p in bearish_patterns)
        
        if bullish_score > bearish_score * 1.2:
            bias = "bullish"
        elif bearish_score > bullish_score * 1.2:
            bias = "bearish"
        else:
            bias = "neutral"
        
        return {
            "patterns": all_patterns,
            "pattern_count": len(all_patterns),
            "bullish_patterns": len(bullish_patterns),
            "bearish_patterns": len(bearish_patterns),
            "overall_bias": bias,
            "highest_confidence_pattern": all_patterns[0] if all_patterns else None,
            "timestamp": datetime.now().isoformat()
        }
