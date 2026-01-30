"""
Anomaly Detection Module - Detect unusual market activity.
Uses statistical methods with optional ML-based detection.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math


class AnomalyType(Enum):
    """Types of market anomalies"""
    VOLUME_SPIKE = "volume_spike"
    PRICE_SPIKE = "price_spike"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    GAP = "gap"
    UNUSUAL_OPTIONS = "unusual_options"
    DARK_POOL = "dark_pool"
    INSIDER_CLUSTER = "insider_cluster"
    INSTITUTIONAL_ACCUMULATION = "institutional_accumulation"
    CORRELATION_BREAK = "correlation_break"
    MOMENTUM_DIVERGENCE = "momentum_divergence"


@dataclass
class Anomaly:
    """Detected anomaly"""
    anomaly_type: AnomalyType
    severity: float  # 0 to 1 (1 = most severe)
    z_score: float  # Standard deviations from mean
    index: int  # Index in data array
    value: float  # The anomalous value
    expected_value: float  # Expected/normal value
    description: str
    timestamp: datetime = None
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass 
class AnomalyReport:
    """Comprehensive anomaly analysis report"""
    symbol: str
    anomalies: List[Anomaly]
    risk_score: float  # 0 to 1
    alert_level: str  # 'low', 'medium', 'high', 'critical'
    summary: str
    recommendations: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AnomalyDetector:
    """
    Market anomaly detection engine.
    
    Features:
    - Statistical outlier detection (z-score, IQR, MAD)
    - Volume anomalies
    - Price gap detection
    - Volatility breakouts
    - Options flow anomalies
    - Institutional activity patterns
    
    For production ML:
    - Isolation Forest for multi-dimensional anomalies
    - Autoencoders for complex pattern detection
    - LSTM for sequence-based anomaly detection
    """
    
    def __init__(
        self,
        z_threshold: float = 2.5,
        lookback_period: int = 20,
        use_ml: bool = False
    ):
        """
        Initialize anomaly detector.
        
        Args:
            z_threshold: Z-score threshold for outlier detection
            lookback_period: Period for calculating baselines
            use_ml: If True, use ML-based detection
        """
        self.z_threshold = z_threshold
        self.lookback_period = lookback_period
        self.use_ml = use_ml
        
        if use_ml:
            self._load_ml_models()
    
    def _load_ml_models(self):
        """Load ML models for anomaly detection"""
        try:
            # Example: Load Isolation Forest
            # from sklearn.ensemble import IsolationForest
            # self._isolation_forest = IsolationForest(contamination=0.1)
            print("ML anomaly detection not implemented. Using statistical methods.")
            self.use_ml = False
        except Exception as e:
            print(f"Could not load ML models: {e}")
            self.use_ml = False
    
    # ==================== Statistical Methods ====================
    
    def calculate_z_score(self, values: List[float], index: int) -> float:
        """Calculate z-score for a value at given index"""
        if index < self.lookback_period:
            return 0.0
        
        window = values[index - self.lookback_period:index]
        
        if len(window) < 2:
            return 0.0
        
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = math.sqrt(variance) if variance > 0 else 1e-10
        
        return (values[index] - mean) / std
    
    def calculate_mad(self, values: List[float], index: int) -> Tuple[float, float]:
        """
        Calculate Median Absolute Deviation score.
        More robust to outliers than z-score.
        
        Returns (mad_score, median)
        """
        if index < self.lookback_period:
            return 0.0, 0.0
        
        window = sorted(values[index - self.lookback_period:index])
        n = len(window)
        
        if n < 2:
            return 0.0, values[index]
        
        median = window[n // 2] if n % 2 else (window[n//2 - 1] + window[n//2]) / 2
        deviations = sorted(abs(x - median) for x in window)
        mad = deviations[len(deviations) // 2]
        
        if mad == 0:
            mad = 1e-10
        
        # Modified z-score using MAD
        mad_score = 0.6745 * (values[index] - median) / mad
        
        return mad_score, median
    
    def calculate_iqr_bounds(self, values: List[float]) -> Tuple[float, float]:
        """Calculate IQR-based outlier bounds"""
        if len(values) < 4:
            return float('-inf'), float('inf')
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        q1 = sorted_values[n // 4]
        q3 = sorted_values[3 * n // 4]
        iqr = q3 - q1
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        return lower, upper
    
    # ==================== Specific Anomaly Detection ====================
    
    def detect_volume_anomalies(
        self,
        volumes: List[int],
        multiplier_threshold: float = 3.0
    ) -> List[Anomaly]:
        """Detect unusual volume spikes"""
        anomalies = []
        
        if len(volumes) < self.lookback_period + 1:
            return anomalies
        
        for i in range(self.lookback_period, len(volumes)):
            window = volumes[i - self.lookback_period:i]
            avg_volume = sum(window) / len(window)
            current_volume = volumes[i]
            
            if avg_volume == 0:
                continue
            
            volume_ratio = current_volume / avg_volume
            z_score = self.calculate_z_score([float(v) for v in volumes], i)
            
            if volume_ratio >= multiplier_threshold or abs(z_score) >= self.z_threshold:
                severity = min(volume_ratio / 5, 1.0)  # Cap at 5x = max severity
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.VOLUME_SPIKE,
                    severity=severity,
                    z_score=z_score,
                    index=i,
                    value=current_volume,
                    expected_value=avg_volume,
                    description=f"Volume {volume_ratio:.1f}x average ({z_score:.1f} sigma)",
                    metadata={
                        "volume_ratio": volume_ratio,
                        "avg_volume": avg_volume
                    }
                ))
        
        return anomalies
    
    def detect_price_anomalies(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float]
    ) -> List[Anomaly]:
        """Detect unusual price movements"""
        anomalies = []
        
        if len(closes) < self.lookback_period + 1:
            return anomalies
        
        # Calculate returns
        returns = [0.0] + [
            (closes[i] - closes[i-1]) / closes[i-1] 
            for i in range(1, len(closes))
        ]
        
        for i in range(self.lookback_period, len(closes)):
            z_score = self.calculate_z_score(returns, i)
            
            if abs(z_score) >= self.z_threshold:
                return_pct = returns[i] * 100
                direction = "up" if returns[i] > 0 else "down"
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.PRICE_SPIKE,
                    severity=min(abs(z_score) / 5, 1.0),
                    z_score=z_score,
                    index=i,
                    value=closes[i],
                    expected_value=closes[i-1],
                    description=f"Price moved {direction} {abs(return_pct):.2f}% ({z_score:.1f} sigma)",
                    metadata={
                        "return_pct": return_pct,
                        "direction": direction
                    }
                ))
        
        return anomalies
    
    def detect_gaps(
        self,
        opens: List[float],
        closes: List[float],
        gap_threshold: float = 0.02
    ) -> List[Anomaly]:
        """Detect price gaps between sessions"""
        anomalies = []
        
        for i in range(1, len(opens)):
            prev_close = closes[i-1]
            current_open = opens[i]
            
            gap_pct = (current_open - prev_close) / prev_close
            
            if abs(gap_pct) >= gap_threshold:
                direction = "up" if gap_pct > 0 else "down"
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.GAP,
                    severity=min(abs(gap_pct) / 0.1, 1.0),  # 10% gap = max severity
                    z_score=0,  # Gaps don't use z-score
                    index=i,
                    value=current_open,
                    expected_value=prev_close,
                    description=f"Gap {direction} {abs(gap_pct*100):.2f}%",
                    metadata={
                        "gap_pct": gap_pct * 100,
                        "direction": direction,
                        "prev_close": prev_close,
                        "open": current_open
                    }
                ))
        
        return anomalies
    
    def detect_volatility_breakout(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        atr_multiplier: float = 2.0
    ) -> List[Anomaly]:
        """Detect volatility breakouts using ATR"""
        anomalies = []
        
        if len(closes) < self.lookback_period + 1:
            return anomalies
        
        # Calculate True Range
        true_ranges = [highs[0] - lows[0]]
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)
        
        # Calculate ATR
        for i in range(self.lookback_period, len(closes)):
            atr = sum(true_ranges[i-self.lookback_period:i]) / self.lookback_period
            current_range = highs[i] - lows[i]
            
            if atr == 0:
                continue
            
            range_ratio = current_range / atr
            
            if range_ratio >= atr_multiplier:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.VOLATILITY_BREAKOUT,
                    severity=min(range_ratio / 4, 1.0),
                    z_score=range_ratio,  # Using ratio as pseudo z-score
                    index=i,
                    value=current_range,
                    expected_value=atr,
                    description=f"Range {range_ratio:.1f}x ATR - volatility expansion",
                    metadata={
                        "range_ratio": range_ratio,
                        "atr": atr,
                        "current_range": current_range
                    }
                ))
        
        return anomalies
    
    def detect_momentum_divergence(
        self,
        closes: List[float],
        rsi_values: Optional[List[float]] = None,
        window: int = 5
    ) -> List[Anomaly]:
        """
        Detect divergence between price and momentum.
        Bullish divergence: price makes lower low, RSI makes higher low
        Bearish divergence: price makes higher high, RSI makes lower high
        """
        anomalies = []
        
        if rsi_values is None or len(rsi_values) < window * 2:
            return anomalies
        
        # Align arrays
        min_len = min(len(closes), len(rsi_values))
        offset = len(closes) - min_len
        
        for i in range(window, min_len - 1):
            idx = i + offset
            
            # Get recent highs/lows
            price_window = closes[idx-window:idx+1]
            rsi_window = rsi_values[i-window:i+1]
            
            current_price = closes[idx]
            prev_price_min = min(price_window[:-1])
            prev_price_max = max(price_window[:-1])
            
            current_rsi = rsi_values[i]
            prev_rsi_min = min(rsi_window[:-1])
            prev_rsi_max = max(rsi_window[:-1])
            
            # Bullish divergence
            if current_price < prev_price_min and current_rsi > prev_rsi_min:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.MOMENTUM_DIVERGENCE,
                    severity=0.7,
                    z_score=0,
                    index=idx,
                    value=current_rsi,
                    expected_value=prev_rsi_min,
                    description="Bullish divergence - price lower low, RSI higher low",
                    metadata={
                        "divergence_type": "bullish",
                        "price_trend": "lower_low",
                        "rsi_trend": "higher_low"
                    }
                ))
            
            # Bearish divergence
            if current_price > prev_price_max and current_rsi < prev_rsi_max:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.MOMENTUM_DIVERGENCE,
                    severity=0.7,
                    z_score=0,
                    index=idx,
                    value=current_rsi,
                    expected_value=prev_rsi_max,
                    description="Bearish divergence - price higher high, RSI lower high",
                    metadata={
                        "divergence_type": "bearish",
                        "price_trend": "higher_high",
                        "rsi_trend": "lower_high"
                    }
                ))
        
        return anomalies
    
    def detect_options_anomalies(
        self,
        options_data: List[Dict],
        volume_threshold: int = 10000,
        oi_threshold: float = 5.0
    ) -> List[Anomaly]:
        """
        Detect unusual options activity.
        
        Args:
            options_data: List of options contracts with volume, OI, etc.
            volume_threshold: Minimum volume to consider
            oi_threshold: Volume to OI ratio threshold
        """
        anomalies = []
        
        for i, option in enumerate(options_data):
            volume = option.get('volume', 0)
            open_interest = option.get('open_interest', 1)
            
            if volume < volume_threshold:
                continue
            
            vol_oi_ratio = volume / max(open_interest, 1)
            
            if vol_oi_ratio >= oi_threshold:
                strike = option.get('strike', 0)
                option_type = option.get('type', 'unknown')
                expiry = option.get('expiry', 'unknown')
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.UNUSUAL_OPTIONS,
                    severity=min(vol_oi_ratio / 10, 1.0),
                    z_score=vol_oi_ratio,
                    index=i,
                    value=volume,
                    expected_value=open_interest,
                    description=f"Unusual {option_type} ${strike} {expiry}: {volume:,} vol ({vol_oi_ratio:.1f}x OI)",
                    metadata={
                        "strike": strike,
                        "type": option_type,
                        "expiry": expiry,
                        "volume": volume,
                        "open_interest": open_interest,
                        "vol_oi_ratio": vol_oi_ratio
                    }
                ))
        
        return anomalies
    
    def detect_insider_clusters(
        self,
        insider_transactions: List[Dict],
        cluster_window_days: int = 14,
        min_cluster_size: int = 3
    ) -> List[Anomaly]:
        """Detect clusters of insider transactions"""
        anomalies = []
        
        if len(insider_transactions) < min_cluster_size:
            return anomalies
        
        # Group by transaction type and date proximity
        buys = [t for t in insider_transactions if t.get('type', '').lower() in ['buy', 'p']]
        sells = [t for t in insider_transactions if t.get('type', '').lower() in ['sell', 's']]
        
        for transactions, direction in [(buys, 'buy'), (sells, 'sell')]:
            if len(transactions) >= min_cluster_size:
                total_value = sum(t.get('value', 0) for t in transactions)
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.INSIDER_CLUSTER,
                    severity=min(len(transactions) / 10, 1.0),
                    z_score=len(transactions),
                    index=0,
                    value=len(transactions),
                    expected_value=1,
                    description=f"Cluster of {len(transactions)} insider {direction}s (${total_value:,.0f} total)",
                    metadata={
                        "direction": direction,
                        "transaction_count": len(transactions),
                        "total_value": total_value,
                        "insiders": [t.get('insider', 'Unknown') for t in transactions]
                    }
                ))
        
        return anomalies
    
    # ==================== Comprehensive Analysis ====================
    
    def analyze(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[int],
        symbol: str = "",
        options_data: Optional[List[Dict]] = None,
        insider_data: Optional[List[Dict]] = None
    ) -> AnomalyReport:
        """
        Perform comprehensive anomaly detection.
        
        Returns AnomalyReport with all detected anomalies and risk assessment.
        """
        all_anomalies = []
        
        # Price anomalies
        price_anomalies = self.detect_price_anomalies(closes, highs, lows)
        all_anomalies.extend(price_anomalies)
        
        # Volume anomalies
        volume_anomalies = self.detect_volume_anomalies(volumes)
        all_anomalies.extend(volume_anomalies)
        
        # Gap detection
        gap_anomalies = self.detect_gaps(opens, closes)
        all_anomalies.extend(gap_anomalies)
        
        # Volatility breakouts
        vol_anomalies = self.detect_volatility_breakout(highs, lows, closes)
        all_anomalies.extend(vol_anomalies)
        
        # Options anomalies
        if options_data:
            options_anomalies = self.detect_options_anomalies(options_data)
            all_anomalies.extend(options_anomalies)
        
        # Insider clusters
        if insider_data:
            insider_anomalies = self.detect_insider_clusters(insider_data)
            all_anomalies.extend(insider_anomalies)
        
        # Calculate overall risk score
        if not all_anomalies:
            risk_score = 0.0
            alert_level = "low"
        else:
            # Weight recent anomalies higher
            recent_cutoff = len(closes) - 5 if len(closes) > 5 else 0
            
            weighted_severity = 0
            for anomaly in all_anomalies:
                weight = 2.0 if anomaly.index >= recent_cutoff else 1.0
                weighted_severity += anomaly.severity * weight
            
            risk_score = min(weighted_severity / 5, 1.0)
            
            if risk_score >= 0.8:
                alert_level = "critical"
            elif risk_score >= 0.6:
                alert_level = "high"
            elif risk_score >= 0.3:
                alert_level = "medium"
            else:
                alert_level = "low"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_anomalies, alert_level)
        
        # Create summary
        summary = self._generate_summary(all_anomalies, risk_score, symbol)
        
        return AnomalyReport(
            symbol=symbol,
            anomalies=all_anomalies,
            risk_score=risk_score,
            alert_level=alert_level,
            summary=summary,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        anomalies: List[Anomaly],
        alert_level: str
    ) -> List[str]:
        """Generate actionable recommendations based on anomalies"""
        recommendations = []
        
        if alert_level == "critical":
            recommendations.append("⚠️ Critical alert level - immediate review recommended")
        
        # Check for specific anomaly types
        anomaly_types = set(a.anomaly_type for a in anomalies)
        
        if AnomalyType.VOLUME_SPIKE in anomaly_types:
            recommendations.append("Volume spike detected - check for news or institutional activity")
        
        if AnomalyType.GAP in anomaly_types:
            recommendations.append("Price gap detected - review overnight news and pre-market action")
        
        if AnomalyType.UNUSUAL_OPTIONS in anomaly_types:
            recommendations.append("Unusual options activity - potential smart money positioning")
        
        if AnomalyType.INSIDER_CLUSTER in anomaly_types:
            recommendations.append("Insider transaction cluster - review SEC Form 4 filings")
        
        if AnomalyType.VOLATILITY_BREAKOUT in anomaly_types:
            recommendations.append("Volatility expansion - consider adjusting position sizes")
        
        if AnomalyType.MOMENTUM_DIVERGENCE in anomaly_types:
            recommendations.append("Momentum divergence - potential trend reversal signal")
        
        if not recommendations:
            recommendations.append("No significant anomalies - normal market activity")
        
        return recommendations
    
    def _generate_summary(
        self,
        anomalies: List[Anomaly],
        risk_score: float,
        symbol: str
    ) -> str:
        """Generate text summary of anomaly analysis"""
        if not anomalies:
            return f"{symbol}: No significant anomalies detected. Market activity appears normal."
        
        # Count by type
        type_counts = {}
        for a in anomalies:
            type_name = a.anomaly_type.value.replace('_', ' ').title()
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        type_summary = ", ".join(f"{count} {name}" for name, count in type_counts.items())
        
        most_severe = max(anomalies, key=lambda a: a.severity)
        
        return (
            f"{symbol}: Detected {len(anomalies)} anomalies ({type_summary}). "
            f"Risk score: {risk_score:.2f}. "
            f"Most significant: {most_severe.description}"
        )
