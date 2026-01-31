"""
Financial Data Pipeline
========================

Production data pipeline for ML training:
- Real-time and historical data from Polygon API
- Feature engineering with technical indicators
- Data normalization and preprocessing
- Dataset classes for PyTorch
- Caching for efficiency
"""

import os
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class OHLCVBar:
    """Single OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trades: Optional[int] = None


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Technical indicator periods
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26, 50])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    
    # Additional features
    include_volume_features: bool = True
    include_volatility_features: bool = True
    include_momentum_features: bool = True
    include_trend_features: bool = True
    include_time_features: bool = True
    
    # Normalization
    normalize_method: str = 'zscore'  # 'zscore', 'minmax', 'robust'
    lookback_for_norm: int = 252  # 1 year for normalization stats


class TechnicalFeatureExtractor:
    """
    Extract technical analysis features from OHLCV data.
    
    Produces a comprehensive feature set for ML models.
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self._norm_stats: Dict[str, Tuple[float, float]] = {}
    
    def extract(
        self,
        ohlcv: np.ndarray,
        fit_normalization: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract all features from OHLCV data.
        
        Args:
            ohlcv: OHLCV data of shape (n_samples, 5) - [open, high, low, close, volume]
            fit_normalization: Whether to fit normalization parameters
            
        Returns:
            Tuple of (features array, feature names list)
        """
        open_prices = ohlcv[:, 0]
        high_prices = ohlcv[:, 1]
        low_prices = ohlcv[:, 2]
        close_prices = ohlcv[:, 3]
        volume = ohlcv[:, 4]
        
        features = []
        names = []
        
        # Price features (normalized)
        log_returns = np.diff(np.log(close_prices + 1e-10), prepend=np.nan)
        features.append(log_returns)
        names.append('log_return')
        
        # Intraday range
        intraday_range = (high_prices - low_prices) / (close_prices + 1e-10)
        features.append(intraday_range)
        names.append('intraday_range')
        
        # Gap
        gap = (open_prices - np.roll(close_prices, 1)) / (np.roll(close_prices, 1) + 1e-10)
        gap[0] = 0
        features.append(gap)
        names.append('gap')
        
        # Moving averages and their signals
        for period in self.config.sma_periods:
            sma = self._sma(close_prices, period)
            sma_ratio = close_prices / (sma + 1e-10) - 1
            features.append(sma_ratio)
            names.append(f'sma_{period}_ratio')
        
        for period in self.config.ema_periods:
            ema = self._ema(close_prices, period)
            ema_ratio = close_prices / (ema + 1e-10) - 1
            features.append(ema_ratio)
            names.append(f'ema_{period}_ratio')
        
        # RSI
        rsi = self._rsi(close_prices, self.config.rsi_period)
        features.append(rsi / 100)  # Normalize to 0-1
        names.append('rsi')
        
        # MACD
        macd, signal, hist = self._macd(
            close_prices,
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        )
        features.append(macd / (close_prices + 1e-10))
        features.append(signal / (close_prices + 1e-10))
        features.append(hist / (close_prices + 1e-10))
        names.extend(['macd', 'macd_signal', 'macd_hist'])
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(
            close_prices,
            self.config.bb_period,
            self.config.bb_std
        )
        bb_position = (close_prices - bb_lower) / (bb_upper - bb_lower + 1e-10)
        bb_width = (bb_upper - bb_lower) / (bb_middle + 1e-10)
        features.append(bb_position)
        features.append(bb_width)
        names.extend(['bb_position', 'bb_width'])
        
        # ATR
        atr = self._atr(high_prices, low_prices, close_prices, self.config.atr_period)
        atr_ratio = atr / (close_prices + 1e-10)
        features.append(atr_ratio)
        names.append('atr_ratio')
        
        if self.config.include_volume_features:
            # Volume features
            vol_sma = self._sma(volume.astype(float), 20)
            vol_ratio = volume / (vol_sma + 1e-10)
            features.append(np.log1p(vol_ratio))
            names.append('volume_ratio')
            
            # OBV
            obv = self._obv(close_prices, volume)
            obv_sma = self._sma(obv.astype(float), 20)
            obv_ratio = obv / (obv_sma + 1e-10) - 1
            features.append(obv_ratio)
            names.append('obv_ratio')
        
        if self.config.include_volatility_features:
            # Historical volatility
            for period in [5, 10, 20]:
                hv = self._historical_volatility(close_prices, period)
                features.append(hv)
                names.append(f'hvol_{period}')
            
            # Parkinson volatility
            parkinson = self._parkinson_volatility(high_prices, low_prices, 20)
            features.append(parkinson)
            names.append('parkinson_vol')
        
        if self.config.include_momentum_features:
            # Rate of change
            for period in [5, 10, 20]:
                roc = (close_prices / np.roll(close_prices, period) - 1)
                roc[:period] = 0
                features.append(roc)
                names.append(f'roc_{period}')
            
            # Stochastic
            k, d = self._stochastic(high_prices, low_prices, close_prices, 14, 3)
            features.append(k / 100)
            features.append(d / 100)
            names.extend(['stoch_k', 'stoch_d'])
        
        if self.config.include_trend_features:
            # ADX
            adx = self._adx(high_prices, low_prices, close_prices, 14)
            features.append(adx / 100)
            names.append('adx')
            
            # Trend strength (price above/below MAs)
            trend_score = sum(
                (close_prices > self._sma(close_prices, p)).astype(float)
                for p in self.config.sma_periods
            ) / len(self.config.sma_periods)
            features.append(trend_score)
            names.append('trend_score')
        
        # Stack all features
        feature_array = np.column_stack(features)
        
        # Handle NaN values
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize
        if fit_normalization:
            feature_array, self._norm_stats = self._normalize(
                feature_array,
                names,
                method=self.config.normalize_method
            )
        elif self._norm_stats:
            feature_array = self._apply_normalization(feature_array, names)
        
        return feature_array, names
    
    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average."""
        result = np.full_like(data, np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        result = np.zeros_like(data)
        multiplier = 2 / (period + 1)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = (data[i] - result[i-1]) * multiplier + result[i-1]
        return result
    
    def _rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Relative Strength Index."""
        deltas = np.diff(prices, prepend=prices[0])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = self._ema(gains, period)
        avg_loss = self._ema(losses, period)
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _macd(
        self,
        prices: np.ndarray,
        fast: int,
        slow: int,
        signal: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD indicator."""
        fast_ema = self._ema(prices, fast)
        slow_ema = self._ema(prices, slow)
        macd_line = fast_ema - slow_ema
        signal_line = self._ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _bollinger_bands(
        self,
        prices: np.ndarray,
        period: int,
        std_dev: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands."""
        middle = self._sma(prices, period)
        
        std = np.full_like(prices, np.nan)
        for i in range(period - 1, len(prices)):
            std[i] = np.std(prices[i - period + 1:i + 1])
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return upper, middle, lower
    
    def _atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Average True Range."""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - prev_close),
                np.abs(low - prev_close)
            )
        )
        
        return self._ema(tr, period)
    
    def _obv(self, prices: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """On Balance Volume."""
        direction = np.sign(np.diff(prices, prepend=prices[0]))
        obv = np.cumsum(direction * volume)
        return obv
    
    def _historical_volatility(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Historical volatility (annualized)."""
        log_returns = np.diff(np.log(prices + 1e-10), prepend=0)
        
        result = np.full_like(prices, np.nan)
        for i in range(period - 1, len(prices)):
            result[i] = np.std(log_returns[i - period + 1:i + 1]) * np.sqrt(252)
        
        return result
    
    def _parkinson_volatility(
        self,
        high: np.ndarray,
        low: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Parkinson volatility estimator."""
        log_hl = np.log(high / (low + 1e-10))
        factor = 1 / (4 * np.log(2))
        
        result = np.full_like(high, np.nan)
        for i in range(period - 1, len(high)):
            result[i] = np.sqrt(factor * np.mean(log_hl[i - period + 1:i + 1] ** 2)) * np.sqrt(252)
        
        return result
    
    def _stochastic(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int,
        d_period: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic oscillator."""
        k = np.full_like(close, np.nan)
        
        for i in range(k_period - 1, len(close)):
            highest = np.max(high[i - k_period + 1:i + 1])
            lowest = np.min(low[i - k_period + 1:i + 1])
            k[i] = 100 * (close[i] - lowest) / (highest - lowest + 1e-10)
        
        d = self._sma(k, d_period)
        
        return k, d
    
    def _adx(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Average Directional Index."""
        # True Range
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
        )
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = self._ema(tr, period)
        plus_di = 100 * self._ema(plus_dm, period) / (atr + 1e-10)
        minus_di = 100 * self._ema(minus_dm, period) / (atr + 1e-10)
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = self._ema(dx, period)
        
        return adx
    
    def _normalize(
        self,
        features: np.ndarray,
        names: List[str],
        method: str = 'zscore'
    ) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
        """Normalize features."""
        stats = {}
        normalized = features.copy()
        
        for i, name in enumerate(names):
            col = features[:, i]
            valid_mask = np.isfinite(col)
            
            if not np.any(valid_mask):
                stats[name] = (0.0, 1.0)
                continue
            
            valid_data = col[valid_mask]
            
            if method == 'zscore':
                mean = np.mean(valid_data)
                std = np.std(valid_data) + 1e-10
                normalized[:, i] = (col - mean) / std
                stats[name] = (mean, std)
            
            elif method == 'minmax':
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                range_val = max_val - min_val + 1e-10
                normalized[:, i] = (col - min_val) / range_val
                stats[name] = (min_val, range_val)
            
            elif method == 'robust':
                median = np.median(valid_data)
                q75, q25 = np.percentile(valid_data, [75, 25])
                iqr = q75 - q25 + 1e-10
                normalized[:, i] = (col - median) / iqr
                stats[name] = (median, iqr)
        
        return normalized, stats
    
    def _apply_normalization(
        self,
        features: np.ndarray,
        names: List[str]
    ) -> np.ndarray:
        """Apply saved normalization stats."""
        normalized = features.copy()
        
        for i, name in enumerate(names):
            if name in self._norm_stats:
                center, scale = self._norm_stats[name]
                normalized[:, i] = (features[:, i] - center) / scale
        
        return normalized


class FinancialDataset(Dataset):
    """
    PyTorch Dataset for financial time series.
    
    Creates sequences of features with targets for supervised learning.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 60,
        prediction_horizons: List[int] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            features: Feature array (n_samples, n_features)
            targets: Target array (n_samples,) or (n_samples, n_horizons)
            sequence_length: Length of input sequences
            prediction_horizons: List of prediction horizons
            transform: Optional transform function
        """
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).float()
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons or [1]
        self.transform = transform
        
        # Valid indices (where we have enough history and future)
        max_horizon = max(self.prediction_horizons)
        self.valid_indices = list(range(
            sequence_length,
            len(features) - max_horizon
        ))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        actual_idx = self.valid_indices[idx]
        
        # Get sequence
        start_idx = actual_idx - self.sequence_length
        x = self.features[start_idx:actual_idx]
        
        # Get targets for all horizons
        if self.targets.dim() == 1:
            y = torch.stack([
                self.targets[actual_idx + h - 1]
                for h in self.prediction_horizons
            ])
        else:
            y = self.targets[actual_idx]
        
        if self.transform:
            x = self.transform(x)
        
        return {
            'features': x,
            'targets': y,
            'index': actual_idx
        }


class PolygonDataFetcher:
    """
    Fetch real market data from Polygon.io API.
    
    Provides historical OHLCV data for model training.
    """
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(
        self,
        api_key: str = None,
        cache_dir: str = './data_cache'
    ):
        """
        Initialize data fetcher.
        
        Args:
            api_key: Polygon API key (or from env POLYGON_API_KEY)
            cache_dir: Directory for caching data
        """
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon API key required")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Import requests here to avoid import at module level
        import requests
        self.session = requests.Session()
    
    def fetch_historical(
        self,
        symbol: str,
        start_date: Union[str, date],
        end_date: Union[str, date],
        timeframe: str = '1d',
        use_cache: bool = True
    ) -> List[OHLCVBar]:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe (1d, 1h, 15m, etc.)
            use_cache: Whether to use cached data
            
        Returns:
            List of OHLCVBar objects
        """
        # Convert dates
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        # Check cache
        cache_key = f"{symbol}_{start_date}_{end_date}_{timeframe}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if use_cache and cache_file.exists():
            logger.info(f"Loading cached data for {symbol}")
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            return [OHLCVBar(**bar) for bar in cached]
        
        # Fetch from API
        logger.info(f"Fetching {symbol} data from Polygon API")
        
        # Map timeframe to Polygon API format
        tf_map = {
            '1m': (1, 'minute'),
            '5m': (5, 'minute'),
            '15m': (15, 'minute'),
            '1h': (1, 'hour'),
            '1d': (1, 'day'),
            '1w': (1, 'week'),
        }
        
        multiplier, span = tf_map.get(timeframe, (1, 'day'))
        
        url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{span}/{start_date}/{end_date}"
        params = {
            'apiKey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'OK':
            raise ValueError(f"API error: {data.get('error', 'Unknown error')}")
        
        bars = []
        for bar in data.get('results', []):
            bars.append(OHLCVBar(
                timestamp=datetime.fromtimestamp(bar['t'] / 1000),
                open=bar['o'],
                high=bar['h'],
                low=bar['l'],
                close=bar['c'],
                volume=bar['v'],
                vwap=bar.get('vw'),
                trades=bar.get('n')
            ))
        
        # Cache the data
        if use_cache and bars:
            with open(cache_file, 'w') as f:
                json.dump(
                    [{'timestamp': bar.timestamp.isoformat(), **{k: v for k, v in bar.__dict__.items() if k != 'timestamp'}} for bar in bars],
                    f
                )
        
        logger.info(f"Fetched {len(bars)} bars for {symbol}")
        return bars
    
    def bars_to_array(self, bars: List[OHLCVBar]) -> np.ndarray:
        """Convert bars to numpy array."""
        return np.array([
            [bar.open, bar.high, bar.low, bar.close, bar.volume]
            for bar in bars
        ])


def create_training_data(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
    sequence_length: int = 60,
    prediction_horizons: List[int] = None,
    target_type: str = 'returns',  # 'returns', 'direction', 'price'
    test_ratio: float = 0.2,
    val_ratio: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create complete training data from scratch.
    
    Args:
        symbol: Stock ticker
        start_date: Start date (default: 5 years ago)
        end_date: End date (default: today)
        sequence_length: Input sequence length
        prediction_horizons: List of forecast horizons
        target_type: Type of prediction target
        test_ratio: Ratio for test set
        val_ratio: Ratio for validation set
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, metadata)
    """
    # Default dates
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=5*365)
    
    prediction_horizons = prediction_horizons or [1, 5, 10, 20]
    
    # Fetch data
    fetcher = PolygonDataFetcher()
    bars = fetcher.fetch_historical(symbol, start_date, end_date, '1d')
    ohlcv = fetcher.bars_to_array(bars)
    
    # Extract features
    feature_extractor = TechnicalFeatureExtractor()
    features, feature_names = feature_extractor.extract(ohlcv)
    
    # Create targets
    close_prices = ohlcv[:, 3]
    
    if target_type == 'returns':
        # Log returns for each horizon
        targets = np.zeros((len(close_prices), len(prediction_horizons)))
        for i, h in enumerate(prediction_horizons):
            future_prices = np.roll(close_prices, -h)
            targets[:, i] = np.log(future_prices / (close_prices + 1e-10))
            targets[-h:, i] = 0
    
    elif target_type == 'direction':
        # Direction for each horizon (up=1, down=0)
        targets = np.zeros((len(close_prices), len(prediction_horizons)))
        for i, h in enumerate(prediction_horizons):
            future_prices = np.roll(close_prices, -h)
            targets[:, i] = (future_prices > close_prices).astype(float)
            targets[-h:, i] = 0.5
    
    elif target_type == 'price':
        # Normalized price targets
        targets = np.zeros((len(close_prices), len(prediction_horizons)))
        for i, h in enumerate(prediction_horizons):
            future_prices = np.roll(close_prices, -h)
            targets[:, i] = (future_prices - close_prices) / (close_prices + 1e-10)
            targets[-h:, i] = 0
    
    # Split data (time-series aware - no shuffling)
    n_samples = len(features)
    test_start = int(n_samples * (1 - test_ratio))
    val_start = int(test_start * (1 - val_ratio / (1 - test_ratio)))
    
    train_features = features[:val_start]
    train_targets = targets[:val_start]
    
    val_features = features[val_start:test_start]
    val_targets = targets[val_start:test_start]
    
    test_features = features[test_start:]
    test_targets = targets[test_start:]
    
    # Create datasets
    train_dataset = FinancialDataset(
        train_features, train_targets, sequence_length, prediction_horizons
    )
    val_dataset = FinancialDataset(
        val_features, val_targets, sequence_length, prediction_horizons
    )
    test_dataset = FinancialDataset(
        test_features, test_targets, sequence_length, prediction_horizons
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    metadata = {
        'symbol': symbol,
        'start_date': str(start_date),
        'end_date': str(end_date),
        'n_samples': n_samples,
        'n_features': features.shape[1],
        'feature_names': feature_names,
        'prediction_horizons': prediction_horizons,
        'target_type': target_type,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset)
    }
    
    logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, metadata
