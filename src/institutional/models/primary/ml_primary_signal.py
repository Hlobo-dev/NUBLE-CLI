"""
ML-BASED PRIMARY SIGNAL GENERATOR
==================================
Replaces simple momentum with feature-rich ML classifier.

This is the critical component - the Meta-Labeler can only filter signals,
it cannot create edge from nothing. If the primary signal is garbage,
filtered garbage still loses money.

Key Principles (AFML):
1. NO lookahead bias in ANY feature
2. Features must be stationary or made stationary
3. Use sample weighting to reduce label overlap bias
4. Train with proper cross-validation (TimeSeriesSplit)

Target: Improve Sharpe from -0.12 to >0.5
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PrimarySignalConfig:
    """Configuration for ML primary signal generator."""
    model_type: str = 'random_forest'  # 'random_forest', 'gradient_boosting', 'xgboost'
    n_estimators: int = 200
    max_depth: int = 6
    min_samples_leaf: int = 50  # High to prevent overfitting
    learning_rate: float = 0.05  # For gradient boosting
    use_class_weights: bool = True
    confidence_threshold: float = 0.55  # Only trade when confidence > threshold
    cv_splits: int = 5  # For hyperparameter tuning
    random_state: int = 42


class MLPrimarySignal:
    """
    ML-based primary signal generator.
    
    Uses multiple technical features to predict Triple Barrier labels.
    Designed to be used with MetaLabeler for two-stage decision making.
    
    Features are carefully designed to avoid ANY lookahead bias:
    - All use only past data
    - No future returns in any calculation
    - Clear separation between feature time and label time
    """
    
    def __init__(self, config: Optional[PrimarySignalConfig] = None):
        self.config = config or PrimarySignalConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.is_fitted = False
        self.feature_importance: Dict[str, float] = {}
        self.validation_metrics: Dict[str, float] = {}
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features WITHOUT any lookahead bias.
        
        Every feature is calculated using ONLY past data available at that time.
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV data with columns: open, high, low, close, volume
            
        Returns:
        --------
        pd.DataFrame
            Features aligned with input index (NaN rows at start)
        """
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        returns = close.pct_change()
        
        # ============================================================
        # MOMENTUM FEATURES (multiple lookbacks for robustness)
        # ============================================================
        for period in [5, 10, 20, 40, 60]:
            features[f'momentum_{period}d'] = close.pct_change(period)
            
        # Rate of change
        for period in [10, 20]:
            features[f'roc_{period}d'] = (close - close.shift(period)) / close.shift(period)
            
        # ============================================================
        # TREND FEATURES (Moving Average based)
        # ============================================================
        ma_5 = close.rolling(5).mean()
        ma_10 = close.rolling(10).mean()
        ma_20 = close.rolling(20).mean()
        ma_50 = close.rolling(50).mean()
        ma_100 = close.rolling(100).mean()
        
        # Distance from MAs (normalized)
        features['dist_ma5'] = (close - ma_5) / close
        features['dist_ma10'] = (close - ma_10) / close
        features['dist_ma20'] = (close - ma_20) / close
        features['dist_ma50'] = (close - ma_50) / close
        features['dist_ma100'] = (close - ma_100) / close
        
        # MA crossover signals
        features['ma_5_10_cross'] = (ma_5 > ma_10).astype(int)
        features['ma_10_20_cross'] = (ma_10 > ma_20).astype(int)
        features['ma_20_50_cross'] = (ma_20 > ma_50).astype(int)
        features['ma_50_100_cross'] = (ma_50 > ma_100).astype(int)
        
        # MA slopes (trend strength)
        for period, ma in [(5, ma_5), (20, ma_20), (50, ma_50)]:
            features[f'ma{period}_slope'] = ma.pct_change(5)
        
        # ============================================================
        # RSI (Relative Strength Index)
        # ============================================================
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = self._compute_rsi(close, period)
            
        # RSI divergence (price vs RSI momentum)
        features['rsi_14_momentum'] = features['rsi_14'].diff(5)
        features['price_momentum'] = close.pct_change(5)
        features['rsi_divergence'] = features['price_momentum'] - features['rsi_14_momentum'] / 100
        
        # ============================================================
        # MACD (Moving Average Convergence Divergence)
        # ============================================================
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        
        features['macd'] = macd_line / close  # Normalize
        features['macd_signal'] = signal_line / close
        features['macd_histogram'] = macd_histogram / close
        features['macd_cross'] = (macd_line > signal_line).astype(int)
        features['macd_histogram_slope'] = macd_histogram.diff(3) / close
        
        # ============================================================
        # BOLLINGER BANDS
        # ============================================================
        bb_period = 20
        bb_std = 2
        bb_ma = close.rolling(bb_period).mean()
        bb_std_dev = close.rolling(bb_period).std()
        bb_upper = bb_ma + bb_std * bb_std_dev
        bb_lower = bb_ma - bb_std * bb_std_dev
        
        # Bollinger Band position (0-1, where 0.5 is at MA)
        bb_width = bb_upper - bb_lower
        features['bb_position'] = (close - bb_lower) / bb_width.replace(0, np.nan)
        features['bb_width'] = bb_width / close  # Normalized width
        features['bb_squeeze'] = bb_width / bb_width.rolling(20).mean()  # Squeeze indicator
        
        # ============================================================
        # VOLATILITY FEATURES
        # ============================================================
        for period in [5, 10, 20, 60]:
            features[f'volatility_{period}d'] = returns.rolling(period).std() * np.sqrt(252)
            
        # Volatility ratio (short vs long term)
        features['vol_ratio_5_20'] = features['volatility_5d'] / features['volatility_20d']
        features['vol_ratio_20_60'] = features['volatility_20d'] / features['volatility_60d']
        
        # Volatility momentum
        features['vol_momentum'] = features['volatility_20d'].pct_change(5)
        
        # ============================================================
        # ATR (Average True Range)
        # ============================================================
        for period in [7, 14, 21]:
            features[f'atr_{period}'] = self._compute_atr(high, low, close, period) / close
            
        # ATR ratio
        features['atr_ratio'] = features['atr_7'] / features['atr_21']
        
        # ============================================================
        # VOLUME FEATURES
        # ============================================================
        vol_ma_5 = volume.rolling(5).mean()
        vol_ma_20 = volume.rolling(20).mean()
        vol_ma_60 = volume.rolling(60).mean()
        
        features['volume_ratio_5'] = volume / vol_ma_5
        features['volume_ratio_20'] = volume / vol_ma_20
        features['volume_ratio_60'] = volume / vol_ma_60
        
        # Volume trend
        features['volume_momentum'] = volume.pct_change(5)
        features['volume_ma_cross'] = (vol_ma_5 > vol_ma_20).astype(int)
        
        # On-Balance Volume slope
        obv = (np.sign(returns) * volume).cumsum()
        features['obv_slope'] = obv.pct_change(10)
        
        # ============================================================
        # PRICE PATTERN FEATURES
        # ============================================================
        # Candlestick patterns (simplified)
        body = close - df['open']
        range_hl = high - low
        
        features['body_ratio'] = body / range_hl.replace(0, np.nan)
        features['upper_shadow'] = (high - pd.concat([close, df['open']], axis=1).max(axis=1)) / range_hl.replace(0, np.nan)
        features['lower_shadow'] = (pd.concat([close, df['open']], axis=1).min(axis=1) - low) / range_hl.replace(0, np.nan)
        
        # Higher highs / Lower lows
        features['higher_high'] = (high > high.shift(1)).astype(int)
        features['lower_low'] = (low < low.shift(1)).astype(int)
        
        # N-day high/low
        features['days_from_20d_high'] = close / close.rolling(20).max()
        features['days_from_20d_low'] = close / close.rolling(20).min()
        
        # ============================================================
        # MEAN REVERSION FEATURES
        # ============================================================
        # Z-score of price relative to rolling mean
        for period in [20, 50]:
            rolling_mean = close.rolling(period).mean()
            rolling_std = close.rolling(period).std()
            features[f'zscore_{period}d'] = (close - rolling_mean) / rolling_std.replace(0, np.nan)
            
        # Return z-score
        for period in [20, 60]:
            ret_mean = returns.rolling(period).mean()
            ret_std = returns.rolling(period).std()
            features[f'return_zscore_{period}d'] = (returns - ret_mean) / ret_std.replace(0, np.nan)
            
        # ============================================================
        # REGIME FEATURES (lagged to avoid lookahead)
        # ============================================================
        # These help the model adapt to different market conditions
        # Market regime proxies (all using past data only)
        
        # Trend strength (ADX-like)
        features['trend_strength'] = abs(close.pct_change(20)) / features['volatility_20d']
        
        # Market breadth proxy (using price behavior)
        features['up_days_ratio_20'] = (returns > 0).rolling(20).mean()
        features['up_days_ratio_60'] = (returns > 0).rolling(60).mean()
        
        # Consecutive up/down days
        features['consecutive_up'] = self._consecutive_days(returns > 0)
        features['consecutive_down'] = self._consecutive_days(returns < 0)
        
        # ============================================================
        # CLEAN UP
        # ============================================================
        # Replace infinities with NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Store feature columns
        self.feature_columns = list(features.columns)
        
        logger.info(f"Generated {len(self.feature_columns)} features")
        
        return features
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _compute_atr(self, high: pd.Series, low: pd.Series, 
                     close: pd.Series, period: int = 14) -> pd.Series:
        """Compute Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
    
    def _consecutive_days(self, condition: pd.Series) -> pd.Series:
        """Count consecutive days where condition is True."""
        # Create groups where condition changes
        groups = (~condition).cumsum()
        # Count within each group
        return condition.groupby(groups).cumcount() + 1
    
    def fit(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        sample_weights: Optional[pd.Series] = None
    ) -> 'MLPrimarySignal':
        """
        Train the primary signal model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV data
        labels : pd.Series
            Triple Barrier labels (-1, 0, 1)
        sample_weights : pd.Series, optional
            AFML sample weights to reduce overlap bias
            
        Returns:
        --------
        self
        """
        logger.info("="*60)
        logger.info("TRAINING ML PRIMARY SIGNAL MODEL")
        logger.info("="*60)
        
        # Generate features
        features = self.generate_features(df)
        
        # Align features and labels
        common_idx = features.dropna().index.intersection(labels.dropna().index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        
        logger.info(f"Training samples: {len(X)}")
        logger.info(f"Features: {len(X.columns)}")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")
        
        # Binarize labels (1 for positive return, 0 otherwise)
        y_binary = (y > 0).astype(int)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        # Prepare sample weights
        if sample_weights is not None:
            weights = sample_weights.reindex(X.index).fillna(1.0)
        else:
            weights = None
            
        # Cross-validation for model selection
        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)
        
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_cv_train = X_scaled.iloc[train_idx]
            y_cv_train = y_binary.iloc[train_idx]
            X_cv_val = X_scaled.iloc[val_idx]
            y_cv_val = y_binary.iloc[val_idx]
            
            # Train model
            model = self._create_model()
            
            if weights is not None:
                w_train = weights.iloc[train_idx].values
                model.fit(X_cv_train, y_cv_train, sample_weight=w_train)
            else:
                model.fit(X_cv_train, y_cv_train)
            
            # Validate
            val_pred = model.predict(X_cv_val)
            precision = precision_score(y_cv_val, val_pred, zero_division=0)
            recall = recall_score(y_cv_val, val_pred, zero_division=0)
            f1 = f1_score(y_cv_val, val_pred, zero_division=0)
            
            cv_scores.append({
                'fold': fold,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
        # Log CV results
        avg_precision = np.mean([s['precision'] for s in cv_scores])
        avg_recall = np.mean([s['recall'] for s in cv_scores])
        avg_f1 = np.mean([s['f1'] for s in cv_scores])
        
        logger.info(f"\nCV Results ({self.config.cv_splits} folds):")
        logger.info(f"  Precision: {avg_precision:.3f}")
        logger.info(f"  Recall: {avg_recall:.3f}")
        logger.info(f"  F1: {avg_f1:.3f}")
        
        self.validation_metrics = {
            'cv_precision': avg_precision,
            'cv_recall': avg_recall,
            'cv_f1': avg_f1
        }
        
        # Train final model on all data
        self.model = self._create_model()
        if weights is not None:
            self.model.fit(X_scaled, y_binary, sample_weight=weights.values)
        else:
            self.model.fit(X_scaled, y_binary)
            
        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            self.feature_importance = dict(zip(X.columns, importances))
            
            # Log top features
            sorted_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:15]
            
            logger.info("\nTop 15 Features:")
            for feat, imp in sorted_features:
                logger.info(f"  {feat}: {imp:.4f}")
        
        self.is_fitted = True
        logger.info("\nModel training complete!")
        
        return self
    
    def _create_model(self):
        """Create the ML model based on config."""
        if self.config.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                class_weight='balanced' if self.config.use_class_weights else None,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif self.config.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state
            )
        elif self.config.model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    min_child_weight=self.config.min_samples_leaf,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    n_jobs=-1
                )
            except ImportError:
                logger.warning("XGBoost not installed, falling back to RandomForest")
                return self._create_model_rf()
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate primary signals for new data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV data
            
        Returns:
        --------
        pd.Series
            Trading signals: -1 (short), 0 (no trade), 1 (long)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Generate features
        features = self.generate_features(df)
        
        # Handle missing features (use 0)
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0
                
        # Keep only training features in order
        X = features[self.feature_columns].fillna(0)
        
        # Scale
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            index=X.index,
            columns=X.columns
        )
        
        # Predict probabilities
        probas = self.model.predict_proba(X_scaled)
        
        # Handle case where only one class exists
        if probas.shape[1] == 1:
            prob_positive = probas[:, 0] if self.model.classes_[0] == 1 else 1 - probas[:, 0]
        else:
            prob_positive = probas[:, 1]
        
        # Convert to signals based on confidence threshold
        signals = pd.Series(0, index=X.index)
        signals[prob_positive > self.config.confidence_threshold] = 1
        signals[prob_positive < (1 - self.config.confidence_threshold)] = -1
        
        return signals
    
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get prediction probabilities.
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV data
            
        Returns:
        --------
        pd.DataFrame
            Probability for each class
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Generate features
        features = self.generate_features(df)
        
        # Handle missing features
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0
                
        X = features[self.feature_columns].fillna(0)
        
        # Scale
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            index=X.index,
            columns=X.columns
        )
        
        # Predict
        probas = self.model.predict_proba(X_scaled)
        
        return pd.DataFrame(
            probas,
            index=X.index,
            columns=['prob_negative', 'prob_positive']
        )
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance ranking."""
        if not self.feature_importance:
            raise ValueError("Model not fitted or doesn't support feature importance")
            
        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importance.items()
        ])
        
        return df.sort_values('importance', ascending=False).head(top_n)


class RegimeAdaptivePrimarySignal(MLPrimarySignal):
    """
    Regime-adaptive primary signal generator.
    
    Uses different models or parameters for different market regimes
    detected by HMM.
    """
    
    def __init__(self, config: Optional[PrimarySignalConfig] = None):
        super().__init__(config)
        self.regime_models: Dict[int, Any] = {}
        self.regime_scalers: Dict[int, StandardScaler] = {}
        self.hmm_detector = None
        
    def fit(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        sample_weights: Optional[pd.Series] = None
    ) -> 'RegimeAdaptivePrimarySignal':
        """Train separate models for each regime."""
        from src.institutional.regime.hmm_detector import HMMRegimeDetector
        
        logger.info("="*60)
        logger.info("TRAINING REGIME-ADAPTIVE PRIMARY SIGNAL")
        logger.info("="*60)
        
        # Generate features
        features = self.generate_features(df)
        
        # Detect regimes
        returns = df['close'].pct_change()
        self.hmm_detector = HMMRegimeDetector(n_regimes=2)
        self.hmm_detector.fit(returns.dropna().values.reshape(-1, 1))
        regimes = self.hmm_detector.predict(returns.dropna().values.reshape(-1, 1))
        regime_series = pd.Series(index=returns.dropna().index, data=regimes)
        
        # Align
        common_idx = features.dropna().index.intersection(labels.dropna().index)
        features_aligned = features.loc[common_idx]
        labels_aligned = labels.loc[common_idx]
        regimes_aligned = regime_series.reindex(common_idx)
        
        # Train regime-specific models
        for regime in regimes_aligned.unique():
            regime_mask = regimes_aligned == regime
            n_samples = regime_mask.sum()
            
            if n_samples < 100:
                logger.warning(f"Regime {regime}: Only {n_samples} samples, skipping")
                continue
                
            logger.info(f"\nTraining Regime {regime} model ({n_samples} samples)")
            
            X_regime = features_aligned[regime_mask]
            y_regime = labels_aligned[regime_mask]
            y_binary = (y_regime > 0).astype(int)
            
            # Scale
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_regime),
                index=X_regime.index,
                columns=X_regime.columns
            )
            
            # Train
            model = self._create_model()
            weights = sample_weights.reindex(X_regime.index).fillna(1.0) if sample_weights is not None else None
            
            if weights is not None:
                model.fit(X_scaled, y_binary, sample_weight=weights.values)
            else:
                model.fit(X_scaled, y_binary)
            
            self.regime_models[regime] = model
            self.regime_scalers[regime] = scaler
            
            logger.info(f"  Regime {regime} model trained")
        
        # Also train a fallback model on all data
        super().fit(df, labels, sample_weights)
        
        self.is_fitted = True
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict using regime-appropriate model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Detect current regimes
        returns = df['close'].pct_change()
        regimes = self.hmm_detector.predict(returns.dropna().values.reshape(-1, 1))
        regime_series = pd.Series(index=returns.dropna().index, data=regimes)
        
        # Generate features
        features = self.generate_features(df)
        
        # Predict regime by regime
        signals = pd.Series(0, index=features.index)
        
        for regime in self.regime_models.keys():
            regime_mask = regime_series.reindex(features.index) == regime
            
            if not regime_mask.any():
                continue
                
            X_regime = features[regime_mask]
            for col in self.feature_columns:
                if col not in X_regime.columns:
                    X_regime[col] = 0
            X_regime = X_regime[self.feature_columns].fillna(0)
            
            scaler = self.regime_scalers[regime]
            model = self.regime_models[regime]
            
            X_scaled = scaler.transform(X_regime)
            probas = model.predict_proba(X_scaled)
            
            if probas.shape[1] == 1:
                prob_positive = probas[:, 0]
            else:
                prob_positive = probas[:, 1]
            
            regime_signals = pd.Series(0, index=X_regime.index)
            regime_signals[prob_positive > self.config.confidence_threshold] = 1
            regime_signals[prob_positive < (1 - self.config.confidence_threshold)] = -1
            
            signals.loc[regime_mask] = regime_signals
        
        # Fill any missing with fallback model
        missing_mask = signals == 0
        if missing_mask.any():
            fallback_signals = super().predict(df)
            signals.loc[missing_mask] = fallback_signals.loc[missing_mask]
        
        return signals


if __name__ == "__main__":
    # Test the model
    import sys
    import os
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    sys.path.insert(0, _root)
    
    from src.institutional.labeling.triple_barrier import TripleBarrierLabeler
    
    # Load data
    df = pd.read_csv(os.path.join(_root, 'data', 'train', 'SPY.csv'),
                     index_col=0, parse_dates=True)
    
    print(f"Data: {len(df)} rows")
    
    # Generate labels
    labeler = TripleBarrierLabeler(pt_sl=(2.0, 1.0), max_holding_period=10)
    labels = labeler.get_labels(df['close'])
    
    # Train model
    model = MLPrimarySignal()
    model.fit(df, labels)
    
    # Predict
    signals = model.predict(df)
    
    print(f"\nSignal distribution:")
    print(signals.value_counts())
    
    # Feature importance
    print("\nTop features:")
    print(model.get_feature_importance(10))
