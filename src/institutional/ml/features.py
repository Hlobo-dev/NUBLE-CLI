"""
Feature Engineering for Financial ML
=====================================

Comprehensive feature extraction:
- Technical indicators as features
- Fundamental data normalization
- Sentiment feature encoding
- Time-based features
- Cross-asset features
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum


@dataclass
class FeatureSet:
    """Container for extracted features"""
    features: np.ndarray
    feature_names: List[str]
    timestamps: Optional[List[datetime]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary mapping names to values"""
        return {name: self.features[:, i] for i, name in enumerate(self.feature_names)}


class FeatureEngineer:
    """
    Master feature engineering class combining all feature types.
    
    Provides a unified interface for extracting ML-ready features from:
    - Price/volume data
    - Fundamental data
    - Sentiment data
    - Macro/economic data
    """
    
    def __init__(
        self,
        include_technical: bool = True,
        include_fundamental: bool = True,
        include_sentiment: bool = True,
        include_time: bool = True,
        normalization: str = 'zscore',  # 'zscore', 'minmax', 'robust', 'none'
        lookback_periods: List[int] = None
    ):
        """
        Initialize feature engineer.
        
        Args:
            include_technical: Include technical features
            include_fundamental: Include fundamental features
            include_sentiment: Include sentiment features
            include_time: Include time-based features
            normalization: Normalization method
            lookback_periods: Custom lookback periods for features
        """
        self.include_technical = include_technical
        self.include_fundamental = include_fundamental
        self.include_sentiment = include_sentiment
        self.include_time = include_time
        self.normalization = normalization
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 100, 200]
        
        # Sub-extractors
        self.technical_extractor = TechnicalFeatureExtractor(self.lookback_periods)
        self.fundamental_extractor = FundamentalFeatureExtractor()
        self.sentiment_extractor = SentimentFeatureExtractor()
        
        # Fitted normalization parameters
        self.norm_params: Dict[str, Tuple[float, float]] = {}
    
    def extract(
        self,
        ohlcv: np.ndarray,
        fundamentals: Optional[Dict[str, Any]] = None,
        sentiment: Optional[Dict[str, Any]] = None,
        timestamps: Optional[List[datetime]] = None,
        fit_normalization: bool = True
    ) -> FeatureSet:
        """
        Extract all features from input data.
        
        Args:
            ohlcv: OHLCV data (n_samples, 5)
            fundamentals: Fundamental data dictionary
            sentiment: Sentiment data dictionary
            timestamps: List of timestamps
            fit_normalization: Whether to fit normalization parameters
            
        Returns:
            FeatureSet with all extracted features
        """
        all_features = []
        all_names = []
        
        # Technical features
        if self.include_technical:
            tech_features = self.technical_extractor.extract(ohlcv)
            all_features.append(tech_features.features)
            all_names.extend(tech_features.feature_names)
        
        # Fundamental features
        if self.include_fundamental and fundamentals:
            fund_features = self.fundamental_extractor.extract(fundamentals, len(ohlcv))
            all_features.append(fund_features.features)
            all_names.extend(fund_features.feature_names)
        
        # Sentiment features
        if self.include_sentiment and sentiment:
            sent_features = self.sentiment_extractor.extract(sentiment, len(ohlcv))
            all_features.append(sent_features.features)
            all_names.extend(sent_features.feature_names)
        
        # Time-based features
        if self.include_time and timestamps:
            time_features = self._extract_time_features(timestamps)
            all_features.append(time_features)
            all_names.extend([
                'day_of_week', 'day_of_month', 'month', 'quarter',
                'is_month_end', 'is_quarter_end', 'days_to_expiry'
            ])
        
        # Combine all features
        combined = np.concatenate(all_features, axis=1) if all_features else np.array([])
        
        # Normalize
        if self.normalization != 'none' and len(combined) > 0:
            combined = self._normalize(combined, all_names, fit=fit_normalization)
        
        return FeatureSet(
            features=combined,
            feature_names=all_names,
            timestamps=timestamps,
            metadata={
                'normalization': self.normalization,
                'lookback_periods': self.lookback_periods
            }
        )
    
    def _extract_time_features(self, timestamps: List[datetime]) -> np.ndarray:
        """Extract time-based features"""
        n = len(timestamps)
        features = np.zeros((n, 7))
        
        for i, ts in enumerate(timestamps):
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            elif isinstance(ts, date) and not isinstance(ts, datetime):
                ts = datetime.combine(ts, datetime.min.time())
            
            features[i, 0] = ts.weekday() / 6.0  # Day of week
            features[i, 1] = ts.day / 31.0  # Day of month
            features[i, 2] = ts.month / 12.0  # Month
            features[i, 3] = (ts.month - 1) // 3 / 3.0  # Quarter
            
            # Is month end (last 3 days)
            features[i, 4] = 1.0 if ts.day >= 28 else 0.0
            
            # Is quarter end
            features[i, 5] = 1.0 if ts.month in [3, 6, 9, 12] and ts.day >= 28 else 0.0
            
            # Days to options expiry (3rd Friday)
            features[i, 6] = self._days_to_expiry(ts) / 30.0
        
        return features
    
    def _days_to_expiry(self, dt: datetime) -> int:
        """Calculate days to next options expiry (3rd Friday)"""
        year, month = dt.year, dt.month
        
        # Find 3rd Friday of current month
        first_day = datetime(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)
        
        if dt.date() > third_friday.date():
            # Move to next month
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
            first_day = datetime(year, month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(weeks=2)
        
        return (third_friday.date() - dt.date()).days
    
    def _normalize(
        self,
        features: np.ndarray,
        names: List[str],
        fit: bool = True
    ) -> np.ndarray:
        """Apply normalization to features"""
        normalized = features.copy()
        
        for i, name in enumerate(names):
            col = features[:, i]
            valid_mask = ~np.isnan(col) & ~np.isinf(col)
            
            if not np.any(valid_mask):
                continue
            
            if fit or name not in self.norm_params:
                if self.normalization == 'zscore':
                    mean = np.mean(col[valid_mask])
                    std = np.std(col[valid_mask]) + 1e-8
                    self.norm_params[name] = (mean, std)
                    normalized[valid_mask, i] = (col[valid_mask] - mean) / std
                    
                elif self.normalization == 'minmax':
                    min_val = np.min(col[valid_mask])
                    max_val = np.max(col[valid_mask])
                    self.norm_params[name] = (min_val, max_val - min_val + 1e-8)
                    normalized[valid_mask, i] = (col[valid_mask] - min_val) / (max_val - min_val + 1e-8)
                    
                elif self.normalization == 'robust':
                    median = np.median(col[valid_mask])
                    iqr = np.percentile(col[valid_mask], 75) - np.percentile(col[valid_mask], 25) + 1e-8
                    self.norm_params[name] = (median, iqr)
                    normalized[valid_mask, i] = (col[valid_mask] - median) / iqr
            else:
                # Use fitted parameters
                param1, param2 = self.norm_params[name]
                if self.normalization == 'zscore':
                    normalized[valid_mask, i] = (col[valid_mask] - param1) / param2
                elif self.normalization == 'minmax':
                    normalized[valid_mask, i] = (col[valid_mask] - param1) / param2
                elif self.normalization == 'robust':
                    normalized[valid_mask, i] = (col[valid_mask] - param1) / param2
        
        return normalized


class TechnicalFeatureExtractor:
    """
    Extract technical indicator features for ML.
    
    Includes:
    - Price-based features (returns, ranges)
    - Momentum indicators
    - Trend indicators
    - Volatility features
    - Volume features
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        """Initialize with lookback periods"""
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 100]
    
    def extract(self, ohlcv: np.ndarray) -> FeatureSet:
        """
        Extract technical features from OHLCV data.
        
        Args:
            ohlcv: Array of shape (n, 5) with O, H, L, C, V
            
        Returns:
            FeatureSet with technical features
        """
        n = len(ohlcv)
        open_p, high, low, close, volume = ohlcv.T
        
        features = []
        names = []
        
        # Returns at multiple horizons
        for period in self.lookback_periods:
            ret = np.zeros(n)
            ret[period:] = (close[period:] - close[:-period]) / (close[:-period] + 1e-8)
            features.append(ret)
            names.append(f'return_{period}d')
        
        # Log returns
        log_ret = np.zeros(n)
        log_ret[1:] = np.log(close[1:] / (close[:-1] + 1e-8))
        features.append(log_ret)
        names.append('log_return_1d')
        
        # Volatility at multiple horizons
        for period in self.lookback_periods:
            vol = np.zeros(n)
            for i in range(period, n):
                vol[i] = np.std(log_ret[i-period:i])
            features.append(vol)
            names.append(f'volatility_{period}d')
        
        # Relative Strength Index (RSI)
        for period in [7, 14, 21]:
            rsi = self._compute_rsi(close, period)
            features.append(rsi)
            names.append(f'rsi_{period}')
        
        # MACD features
        macd, signal, hist = self._compute_macd(close)
        features.extend([macd, signal, hist])
        names.extend(['macd', 'macd_signal', 'macd_histogram'])
        
        # Bollinger Band position
        for period in [20, 50]:
            bb_pos = self._compute_bb_position(close, period)
            features.append(bb_pos)
            names.append(f'bb_position_{period}')
        
        # Average True Range (normalized)
        atr = self._compute_atr(high, low, close, 14)
        features.append(atr / (close + 1e-8))
        names.append('atr_normalized')
        
        # Price position in range
        for period in self.lookback_periods:
            pos = np.zeros(n)
            for i in range(period, n):
                range_high = np.max(high[i-period:i])
                range_low = np.min(low[i-period:i])
                pos[i] = (close[i] - range_low) / (range_high - range_low + 1e-8)
            features.append(pos)
            names.append(f'price_position_{period}d')
        
        # Volume features
        vol_sma = self._sma(volume, 20)
        features.append(volume / (vol_sma + 1e-8))  # Volume ratio
        names.append('volume_ratio')
        
        # On-Balance Volume trend
        obv = self._compute_obv(close, volume)
        obv_norm = (obv - self._sma(obv, 20)) / (np.std(obv) + 1e-8)
        features.append(obv_norm)
        names.append('obv_normalized')
        
        # Candlestick features
        body = close - open_p
        upper_wick = high - np.maximum(open_p, close)
        lower_wick = np.minimum(open_p, close) - low
        
        features.append(body / (high - low + 1e-8))  # Body ratio
        features.append(upper_wick / (high - low + 1e-8))  # Upper wick ratio
        features.append(lower_wick / (high - low + 1e-8))  # Lower wick ratio
        names.extend(['body_ratio', 'upper_wick_ratio', 'lower_wick_ratio'])
        
        # Trend features
        for period in [20, 50, 200]:
            sma = self._sma(close, period)
            features.append((close - sma) / (sma + 1e-8))
            names.append(f'distance_from_sma_{period}')
        
        # Moving average crossovers
        sma_20 = self._sma(close, 20)
        sma_50 = self._sma(close, 50)
        sma_200 = self._sma(close, 200)
        
        features.append((sma_20 - sma_50) / (sma_50 + 1e-8))
        features.append((sma_50 - sma_200) / (sma_200 + 1e-8))
        names.extend(['sma_20_50_diff', 'sma_50_200_diff'])
        
        return FeatureSet(
            features=np.column_stack(features),
            feature_names=names
        )
    
    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple moving average"""
        result = np.zeros_like(data)
        for i in range(period, len(data)):
            result[i] = np.mean(data[i-period:i])
        result[:period] = result[period] if len(data) > period else 0
        return result
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average"""
        result = np.zeros_like(data)
        alpha = 2.0 / (period + 1)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    def _compute_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute RSI"""
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = self._sma(gain, period)
        avg_loss = self._sma(loss, period)
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100  # Normalize to [0, 1]
    
    def _compute_macd(
        self,
        close: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute MACD"""
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        macd = ema_fast - ema_slow
        signal_line = self._ema(macd, signal)
        histogram = macd - signal_line
        
        # Normalize by price
        return macd / (close + 1e-8), signal_line / (close + 1e-8), histogram / (close + 1e-8)
    
    def _compute_bb_position(
        self,
        close: np.ndarray,
        period: int = 20,
        std_mult: float = 2.0
    ) -> np.ndarray:
        """Compute position within Bollinger Bands"""
        sma = self._sma(close, period)
        std = np.zeros_like(close)
        for i in range(period, len(close)):
            std[i] = np.std(close[i-period:i])
        std[:period] = std[period] if len(close) > period else 1
        
        upper = sma + std_mult * std
        lower = sma - std_mult * std
        
        position = (close - lower) / (upper - lower + 1e-8)
        return position
    
    def _compute_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Compute Average True Range"""
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        return self._sma(tr, period)
    
    def _compute_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Compute On-Balance Volume"""
        obv = np.zeros_like(close)
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        return obv


class FundamentalFeatureExtractor:
    """
    Extract and encode fundamental data as ML features.
    
    Handles:
    - Financial ratios
    - Growth metrics
    - Valuation multiples
    - Quality factors
    """
    
    def __init__(self):
        """Initialize fundamental extractor"""
        self.feature_names = [
            'pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda',
            'debt_to_equity', 'current_ratio', 'quick_ratio',
            'roe', 'roa', 'roic', 'gross_margin', 'operating_margin', 'net_margin',
            'revenue_growth', 'earnings_growth', 'fcf_growth',
            'dividend_yield', 'payout_ratio',
            'beta', 'market_cap_log',
            'earnings_surprise', 'revenue_surprise'
        ]
    
    def extract(
        self,
        fundamentals: Dict[str, Any],
        n_samples: int
    ) -> FeatureSet:
        """
        Extract fundamental features.
        
        Args:
            fundamentals: Dictionary of fundamental data
            n_samples: Number of samples (for broadcasting static data)
            
        Returns:
            FeatureSet with fundamental features
        """
        features = np.zeros((n_samples, len(self.feature_names)))
        
        for i, name in enumerate(self.feature_names):
            value = fundamentals.get(name, 0)
            
            # Handle special cases
            if name == 'market_cap_log':
                mc = fundamentals.get('market_cap', 1e9)
                value = np.log(mc + 1)
            elif name == 'pe_ratio':
                value = np.clip(value, -100, 500) if value else 0
            elif name == 'beta':
                value = np.clip(value, -2, 5) if value else 1
            
            # Convert to float and handle None
            try:
                value = float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                value = 0.0
            
            # Handle inf and nan
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            
            features[:, i] = value
        
        return FeatureSet(
            features=features,
            feature_names=self.feature_names.copy()
        )


class SentimentFeatureExtractor:
    """
    Extract sentiment-based features for ML.
    
    Encodes:
    - News sentiment scores
    - Social media metrics
    - Analyst ratings
    - Insider activity
    """
    
    def __init__(self):
        """Initialize sentiment extractor"""
        self.feature_names = [
            'news_sentiment_mean', 'news_sentiment_std', 'news_volume',
            'social_sentiment', 'social_volume', 'social_momentum',
            'analyst_rating', 'analyst_target_vs_price',
            'insider_buy_ratio', 'insider_activity_level',
            'short_interest_ratio', 'put_call_ratio'
        ]
    
    def extract(
        self,
        sentiment: Dict[str, Any],
        n_samples: int
    ) -> FeatureSet:
        """
        Extract sentiment features.
        
        Args:
            sentiment: Dictionary of sentiment data
            n_samples: Number of samples
            
        Returns:
            FeatureSet with sentiment features
        """
        features = np.zeros((n_samples, len(self.feature_names)))
        
        for i, name in enumerate(self.feature_names):
            value = sentiment.get(name, 0)
            
            try:
                value = float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                value = 0.0
            
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            
            features[:, i] = value
        
        return FeatureSet(
            features=features,
            feature_names=self.feature_names.copy()
        )
