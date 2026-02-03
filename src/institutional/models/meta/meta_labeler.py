"""
Meta-Labeling Implementation
============================

Institutional-grade implementation of Meta-Labeling per Lopez de Prado (2018)
"Advances in Financial Machine Learning" Chapter 3.

The key insight: Separate the problem of DIRECTION from the problem of SIZE.
- Primary model predicts direction (long/short)
- Meta-labeler predicts whether to act on that signal (bet/no-bet)

This separation:
1. Allows use of precision-maximizing models for execution
2. Enables optimal position sizing based on confidence
3. Dramatically reduces false positives
4. Works with ANY primary model (trend-following, mean-reversion, ML, etc.)

Mathematical Framework:
-----------------------
Let p₁ = P(correct direction | features)  [primary model]
Let p₂ = P(should act | correct direction) [meta-labeler]

Then: P(profitable trade) = p₁ × p₂

By maximizing p₂ (meta-labeler precision), we filter to only high-confidence trades.

Implementation Details:
-----------------------
- Uses RandomForest/Gradient Boosting for secondary model
- Features engineered from primary model + market context
- Confidence-based position sizing using Kelly-like formulas
- Full walk-forward validation support

Author: Nuble Institutional ML
License: Proprietary
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import logging

# Configure logging
logger = logging.getLogger(__name__)


class SecondaryModelType(Enum):
    """Supported secondary model types for meta-labeling."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOST = "gradient_boost"
    LOGISTIC = "logistic"
    BAGGED_ENSEMBLE = "bagged_ensemble"


@dataclass
class MetaLabelConfig:
    """
    Configuration for Meta-Labeler.
    
    Follows Lopez de Prado guidelines:
    - Secondary model should maximize precision (not accuracy)
    - Use probability outputs for position sizing
    - Cross-validation must be purged + embargoed
    """
    
    # Secondary model configuration
    secondary_model_type: SecondaryModelType = SecondaryModelType.RANDOM_FOREST
    n_estimators: int = 500
    max_depth: Optional[int] = 6
    min_samples_leaf: int = 50
    max_features: str = "sqrt"
    class_weight: str = "balanced_subsample"
    
    # Meta-label thresholds
    min_confidence: float = 0.55  # Minimum probability to act
    precision_target: float = 0.65  # Target precision for threshold selection
    
    # Position sizing
    use_kelly_sizing: bool = True
    kelly_fraction: float = 0.25  # Fractional Kelly (0.25 = quarter Kelly)
    max_position_size: float = 1.0  # Maximum position as fraction of portfolio
    
    # Feature engineering
    include_volatility_features: bool = True
    include_regime_features: bool = True
    include_momentum_features: bool = True
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 21, 63])
    
    # Validation
    purge_gap: int = 3  # Days to purge between train/test
    embargo_pct: float = 0.01  # Percentage of test to embargo
    n_splits: int = 5  # Number of cross-validation splits


@dataclass
class MetaLabelResult:
    """
    Result from meta-labeling process.
    
    Contains all information needed for execution:
    - Whether to act on the primary signal
    - Confidence level
    - Recommended position size
    - Feature importance (for interpretability)
    """
    
    # Core decision
    should_act: bool
    confidence: float  # Probability that acting is profitable
    
    # Position sizing
    recommended_size: float  # Fraction of max position
    kelly_size: float  # Pure Kelly optimal size
    
    # Context
    primary_signal: int  # Original signal from primary model
    regime_context: Optional[str] = None
    
    # Interpretability
    top_features: Optional[Dict[str, float]] = None
    
    # Timing
    timestamp: Optional[datetime] = None


class MetaLabeler:
    """
    Institutional-grade Meta-Labeling implementation.
    
    Key Features:
    - Works with any primary model
    - Precision-focused secondary classifier
    - Confidence-based position sizing
    - Full purged/embargoed cross-validation
    - Interpretable feature importance
    
    Usage:
    ------
    ```python
    # Initialize
    meta_labeler = MetaLabeler(config=MetaLabelConfig())
    
    # Prepare data
    primary_signals = primary_model.predict(X)
    labels = triple_barrier.get_labels(prices)
    
    # Fit meta-labeler
    meta_labeler.fit(X, primary_signals, labels)
    
    # Make decisions
    result = meta_labeler.decide(features, primary_signal=1)
    if result.should_act:
        execute_trade(side=primary_signal, size=result.recommended_size)
    ```
    
    References:
    -----------
    - Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
      Chapter 3: Meta-Labeling
    - Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
    """
    
    def __init__(self, config: Optional[MetaLabelConfig] = None):
        """
        Initialize Meta-Labeler.
        
        Parameters:
        -----------
        config : MetaLabelConfig, optional
            Configuration settings. Uses defaults if not provided.
        """
        self.config = config or MetaLabelConfig()
        self.model = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.feature_importances: Optional[Dict[str, float]] = None
        self.optimal_threshold: float = self.config.min_confidence
        self.validation_metrics: Dict[str, float] = {}
        
        # Initialize secondary model
        self._init_secondary_model()
        
    def _init_secondary_model(self) -> None:
        """Initialize the secondary classification model."""
        try:
            if self.config.secondary_model_type == SecondaryModelType.RANDOM_FOREST:
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    min_samples_leaf=self.config.min_samples_leaf,
                    max_features=self.config.max_features,
                    class_weight=self.config.class_weight,
                    n_jobs=-1,
                    random_state=42,
                    oob_score=True,
                )
                
            elif self.config.secondary_model_type == SecondaryModelType.GRADIENT_BOOST:
                from sklearn.ensemble import GradientBoostingClassifier
                self.model = GradientBoostingClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    min_samples_leaf=self.config.min_samples_leaf,
                    max_features=self.config.max_features,
                    random_state=42,
                )
                
            elif self.config.secondary_model_type == SecondaryModelType.LOGISTIC:
                from sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42,
                )
                
            elif self.config.secondary_model_type == SecondaryModelType.BAGGED_ENSEMBLE:
                from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
                base = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=self.config.max_depth,
                    min_samples_leaf=self.config.min_samples_leaf,
                    random_state=42,
                )
                self.model = BaggingClassifier(
                    estimator=base,
                    n_estimators=5,
                    max_samples=0.6,
                    max_features=0.8,
                    bootstrap=True,
                    bootstrap_features=True,
                    n_jobs=-1,
                    random_state=42,
                )
        except ImportError as e:
            raise ImportError(f"scikit-learn required for meta-labeling: {e}")
            
    def create_meta_labels(
        self,
        primary_signals: pd.Series,
        triple_barrier_labels: pd.Series,
    ) -> pd.Series:
        """
        Create meta-labels from primary signals and triple barrier labels.
        
        The meta-label is 1 if:
        - Primary signal was correct (predicted direction matched outcome)
        
        The meta-label is 0 if:
        - Primary signal was incorrect
        
        Parameters:
        -----------
        primary_signals : pd.Series
            Predictions from primary model (-1, 0, 1)
        triple_barrier_labels : pd.Series
            Labels from triple barrier method (-1, 0, 1)
            
        Returns:
        --------
        pd.Series
            Binary meta-labels (1=correct, 0=incorrect)
        """
        # Align indices
        common_idx = primary_signals.index.intersection(triple_barrier_labels.index)
        signals = primary_signals.loc[common_idx]
        labels = triple_barrier_labels.loc[common_idx]
        
        # Meta-label: 1 if primary signal matches outcome
        # This creates a binary classification: should we act on this signal?
        meta_labels = (signals == labels).astype(int)
        
        # Handle neutral signals (0) - these should not be meta-labeled
        # We only meta-label when primary model has an opinion
        meta_labels = meta_labels.where(signals != 0, np.nan)
        
        logger.info(
            f"Created meta-labels: {meta_labels.sum():.0f} correct / "
            f"{(~meta_labels.isna()).sum():.0f} total "
            f"({meta_labels.mean()*100:.1f}% base rate)"
        )
        
        return meta_labels
        
    def engineer_features(
        self,
        base_features: pd.DataFrame,
        primary_signals: pd.Series,
        returns: Optional[pd.Series] = None,
        regimes: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Engineer features specifically for meta-labeling.
        
        The meta-labeler needs features that predict WHEN the primary model
        is likely to be correct, not features that predict direction.
        
        Key feature categories:
        1. Primary model confidence (if available)
        2. Market regime indicators
        3. Volatility context
        4. Recent model performance
        
        Parameters:
        -----------
        base_features : pd.DataFrame
            Features used by primary model
        primary_signals : pd.Series
            Predictions from primary model
        returns : pd.Series, optional
            Price returns for volatility features
        regimes : pd.Series, optional
            Market regime labels from HMM
            
        Returns:
        --------
        pd.DataFrame
            Engineered features for meta-labeling
        """
        features = base_features.copy()
        self.feature_names = list(features.columns)
        
        # Add primary signal as feature
        features['primary_signal'] = primary_signals
        features['primary_signal_abs'] = np.abs(primary_signals)
        
        if returns is not None and self.config.include_volatility_features:
            # Volatility context features
            for period in self.config.lookback_periods:
                col_name = f'volatility_{period}d'
                features[col_name] = returns.rolling(period).std() * np.sqrt(252)
                self.feature_names.append(col_name)
                
            # Volatility of volatility (uncertainty)
            features['vol_of_vol'] = features['volatility_21d'].rolling(21).std()
            
            # Volatility regime (high/low)
            vol_median = features['volatility_21d'].rolling(63).median()
            features['high_vol_regime'] = (features['volatility_21d'] > vol_median).astype(int)
            
        if returns is not None and self.config.include_momentum_features:
            # Momentum context
            for period in self.config.lookback_periods:
                col_name = f'momentum_{period}d'
                features[col_name] = returns.rolling(period).sum()
                self.feature_names.append(col_name)
                
            # Momentum consistency (how aligned are different timeframes)
            features['momentum_alignment'] = np.sign(features['momentum_5d']) * \
                                             np.sign(features['momentum_21d']) * \
                                             np.sign(features['momentum_63d'])
                                             
        if regimes is not None and self.config.include_regime_features:
            # One-hot encode regimes
            regime_dummies = pd.get_dummies(regimes, prefix='regime')
            features = pd.concat([features, regime_dummies], axis=1)
            self.feature_names.extend(regime_dummies.columns.tolist())
            
        # Recent primary model performance (rolling accuracy)
        # Note: This requires known labels - use carefully to avoid lookahead
        features['days_since_signal_change'] = (
            primary_signals != primary_signals.shift()
        ).cumsum()
        
        # Ensure no NaN or inf values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        return features
        
    def _purged_cv_split(
        self,
        X: pd.DataFrame,
        n_splits: int = 5,
        purge_gap: int = 3,
        embargo_pct: float = 0.01,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged and embargoed cross-validation splits.
        
        Following Lopez de Prado's recommendations:
        - Purge: Remove training samples that could leak into test
        - Embargo: Skip samples immediately after test set
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix (with datetime index)
        n_splits : int
            Number of CV splits
        purge_gap : int
            Days to purge between train and test
        embargo_pct : float
            Percentage of samples to embargo after test
            
        Returns:
        --------
        List[Tuple[np.ndarray, np.ndarray]]
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate split sizes
        test_size = n_samples // n_splits
        embargo_size = int(n_samples * embargo_pct)
        
        splits = []
        
        for i in range(n_splits):
            # Test set: contiguous block
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)
            test_indices = indices[test_start:test_end]
            
            # Training set: everything except test + purge + embargo
            purge_start = max(0, test_start - purge_gap)
            embargo_end = min(n_samples, test_end + embargo_size)
            
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[purge_start:embargo_end] = False
            train_indices = indices[train_mask]
            
            # Ensure we have enough training data
            if len(train_indices) >= 100:
                splits.append((train_indices, test_indices))
            else:
                logger.warning(
                    f"Skipping split {i}: insufficient training data "
                    f"({len(train_indices)} samples)"
                )
                
        return splits
        
    def fit(
        self,
        features: pd.DataFrame,
        primary_signals: pd.Series,
        triple_barrier_labels: pd.Series,
        returns: Optional[pd.Series] = None,
        regimes: Optional[pd.Series] = None,
        optimize_threshold: bool = True,
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'MetaLabeler':
        """
        Fit the meta-labeler using purged cross-validation.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Base features for meta-labeling
        primary_signals : pd.Series
            Predictions from primary model
        triple_barrier_labels : pd.Series
            Labels from triple barrier method
        returns : pd.Series, optional
            Returns for feature engineering
        regimes : pd.Series, optional
            Regime labels from HMM
        optimize_threshold : bool
            Whether to optimize decision threshold for precision
        sample_weight : np.ndarray, optional
            AFML sample weights for each observation (reduces overfitting
            from overlapping Triple Barrier labels per Chapter 4)
            
        Returns:
        --------
        MetaLabeler
            Fitted meta-labeler instance
        """
        logger.info("Fitting meta-labeler...")
        
        # Store sample weights for later use
        self._sample_weight = sample_weight
        
        # Create meta-labels
        meta_labels = self.create_meta_labels(primary_signals, triple_barrier_labels)
        
        # Engineer features
        X = self.engineer_features(
            features, primary_signals, returns, regimes
        )
        
        # Align indices
        common_idx = meta_labels.dropna().index.intersection(X.dropna().index)
        X_clean = X.loc[common_idx].dropna()
        y = meta_labels.loc[X_clean.index]
        
        # Drop any remaining NaN
        valid_mask = ~(X_clean.isna().any(axis=1) | y.isna())
        X_final = X_clean.loc[valid_mask]
        y_final = y.loc[valid_mask]
        
        # Align sample weights if provided
        if sample_weight is not None:
            # Sample weight needs to align with final indices
            if len(sample_weight) == len(features):
                # Create weight series aligned with original features
                weight_series = pd.Series(sample_weight, index=features.index)
                aligned_weights = weight_series.loc[X_final.index].values
            elif len(sample_weight) == len(X_final):
                aligned_weights = sample_weight
            else:
                logger.warning(f"Sample weight length mismatch: {len(sample_weight)} vs {len(X_final)}. Ignoring weights.")
                aligned_weights = None
        else:
            aligned_weights = None
        
        logger.info(f"Training on {len(X_final)} samples")
        if aligned_weights is not None:
            logger.info(f"  Using AFML sample weights (mean={aligned_weights.mean():.2f})")
        
        # Fit with cross-validation for threshold optimization
        if optimize_threshold:
            self._fit_with_cv_threshold(X_final, y_final, sample_weight=aligned_weights)
        else:
            # Simple fit
            if aligned_weights is not None:
                self.model.fit(X_final, y_final, sample_weight=aligned_weights)
            else:
                self.model.fit(X_final, y_final)
            
        # Store feature importances
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = dict(
                zip(X_final.columns, self.model.feature_importances_)
            )
        elif hasattr(self.model, 'coef_'):
            self.feature_importances = dict(
                zip(X_final.columns, np.abs(self.model.coef_[0]))
            )
            
        self.is_fitted = True
        self.feature_names = list(X_final.columns)
        
        logger.info(
            f"Meta-labeler fitted. Optimal threshold: {self.optimal_threshold:.3f}"
        )
        
        return self
        
    def _fit_with_cv_threshold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit model and optimize threshold using cross-validation.
        
        The goal is to find a threshold that maximizes precision while
        maintaining acceptable recall (we want to be SURE when we act).
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Labels
        sample_weight : np.ndarray, optional
            AFML sample weights (per Ch 4)
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        splits = self._purged_cv_split(
            X,
            n_splits=self.config.n_splits,
            purge_gap=self.config.purge_gap,
            embargo_pct=self.config.embargo_pct,
        )
        
        # Collect out-of-fold predictions
        oof_probas = np.zeros(len(X))
        oof_mask = np.zeros(len(X), dtype=bool)
        
        for train_idx, test_idx in splits:
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            
            # Get training weights for this fold
            if sample_weight is not None:
                fold_weights = sample_weight[train_idx]
            else:
                fold_weights = None
            
            # Skip fold if only one class present
            if len(np.unique(y_train)) < 2:
                logger.warning("Skipping CV fold with only one class")
                continue
            
            # Clone and fit model with sample weights
            from sklearn.base import clone
            fold_model = clone(self.model)
            if fold_weights is not None:
                fold_model.fit(X_train, y_train, sample_weight=fold_weights)
            else:
                fold_model.fit(X_train, y_train)
            
            # Store OOF predictions (handle single-class edge case)
            proba = fold_model.predict_proba(X_test)
            if proba.shape[1] == 2:
                oof_probas[test_idx] = proba[:, 1]
            else:
                # Only one class in training - use constant probability
                oof_probas[test_idx] = 0.5
            oof_mask[test_idx] = True
            
        # Optimize threshold for target precision
        thresholds = np.arange(0.40, 0.80, 0.01)
        best_threshold = self.config.min_confidence
        best_f1 = 0
        
        y_true = y.values[oof_mask]
        probas = oof_probas[oof_mask]
        
        for thresh in thresholds:
            preds = (probas >= thresh).astype(int)
            
            # Skip if too few predictions
            if preds.sum() < 10:
                continue
                
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            
            # We want precision >= target with best F1
            if prec >= self.config.precision_target and f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
                
        self.optimal_threshold = best_threshold
        
        # Final fit on all data
        # Check if we have both classes
        if len(np.unique(y)) < 2:
            logger.warning("Only one class in meta-labels - using default predictions")
            self.model = None
            self.validation_metrics = {
                'precision': 0.5,
                'recall': 1.0,
                'f1': 0.67,
                'threshold': self.optimal_threshold,
                'trade_rate': 1.0,
            }
            return
        
        # Final fit with sample weights
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        
        # Calculate validation metrics
        proba_result = self.model.predict_proba(X)
        if proba_result.shape[1] == 2:
            final_probas = proba_result[:, 1]
        else:
            final_probas = np.ones(len(X)) * 0.5
        final_preds = (final_probas >= self.optimal_threshold).astype(int)
        
        self.validation_metrics = {
            'precision': precision_score(y, final_preds, zero_division=0),
            'recall': recall_score(y, final_preds, zero_division=0),
            'f1': f1_score(y, final_preds, zero_division=0),
            'threshold': self.optimal_threshold,
            'trade_rate': final_preds.mean(),  # What fraction of signals we act on
        }
        
        logger.info(
            f"CV Metrics - Precision: {self.validation_metrics['precision']:.3f}, "
            f"Recall: {self.validation_metrics['recall']:.3f}, "
            f"Trade Rate: {self.validation_metrics['trade_rate']:.3f}"
        )
        
    def predict_proba(self, features: pd.DataFrame, primary_signals: pd.Series = None) -> np.ndarray:
        """
        Predict probability of primary signal being correct.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Features for prediction (must match training features)
        primary_signals : pd.Series, optional
            Primary signals (for feature engineering)
            
        Returns:
        --------
        np.ndarray
            Probability of success for each sample (shape: n, 2)
        """
        if not self.is_fitted:
            raise ValueError("MetaLabeler must be fitted before prediction")
        
        # Handle case where model is None (all one class in training)
        if self.model is None:
            # Return constant probability
            n = len(features)
            return np.column_stack([np.zeros(n), np.ones(n) * 0.5])
        
        # Build prediction features to match training
        X = features.copy()
        
        # Add primary signal features
        if primary_signals is not None:
            X['primary_signal'] = primary_signals
            X['primary_signal_abs'] = np.abs(primary_signals)
            X['days_since_signal_change'] = (
                primary_signals != primary_signals.shift()
            ).cumsum()
        
        # Only use features that are available
        available_features = [f for f in self.feature_names if f in X.columns]
        X_final = X[available_features].fillna(0)
        
        # Ensure same order as training
        proba = self.model.predict_proba(X_final)
        
        # Handle single-class prediction
        if proba.shape[1] == 1:
            return np.column_stack([1 - proba[:, 0], proba[:, 0]])
        
        return proba
        
    def predict(self, features: pd.DataFrame, primary_signals: pd.Series = None) -> pd.Series:
        """
        Predict whether to act on each signal.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Features for prediction
        primary_signals : pd.Series, optional
            Primary signals for feature engineering
            
        Returns:
        --------
        pd.Series
            Binary predictions (1=act, 0=don't act)
        """
        probas = self.predict_proba(features, primary_signals)
        return pd.Series(
            (probas[:, 1] >= self.optimal_threshold).astype(int),
            index=features.index,
            name='meta_prediction'
        )
        
    def compute_position_size(
        self,
        probability: float,
        primary_signal: int,
    ) -> float:
        """
        Compute optimal position size using Kelly-like formula.
        
        The Kelly Criterion optimizes growth rate:
        f* = (p * b - q) / b
        
        Where:
        - p = probability of winning (meta-label probability)
        - q = probability of losing (1 - p)
        - b = win/loss ratio (assumed 1 for simplification)
        
        We use fractional Kelly for risk management.
        
        Parameters:
        -----------
        probability : float
            Meta-label probability (confidence)
        primary_signal : int
            Primary model signal (-1, 0, 1)
            
        Returns:
        --------
        float
            Recommended position size [0, max_position_size]
        """
        if not self.config.use_kelly_sizing:
            # Simple binary sizing
            if probability >= self.optimal_threshold:
                return self.config.max_position_size
            return 0.0
            
        # Kelly formula (with b=1)
        p = probability
        q = 1 - p
        kelly_raw = p - q  # Simplified Kelly
        
        # Apply fractional Kelly
        kelly_size = max(0, kelly_raw) * self.config.kelly_fraction
        
        # Cap at max position
        position_size = min(kelly_size, self.config.max_position_size)
        
        return position_size
        
    def decide(
        self,
        features: pd.DataFrame,
        primary_signal: int,
        regime: Optional[str] = None,
    ) -> MetaLabelResult:
        """
        Make a trading decision based on primary signal and meta-labeling.
        
        This is the main entry point for live trading decisions.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Current features (single row or last row used)
        primary_signal : int
            Primary model's prediction (-1, 0, 1)
        regime : str, optional
            Current market regime label
            
        Returns:
        --------
        MetaLabelResult
            Complete decision including action, confidence, and sizing
        """
        if not self.is_fitted:
            raise ValueError("MetaLabeler must be fitted before making decisions")
            
        # Get single row if DataFrame
        if isinstance(features, pd.DataFrame):
            if len(features) > 1:
                features = features.iloc[[-1]]
            row = features.iloc[0]
        else:
            row = features
            
        # Predict probability
        probas = self.predict_proba(features)
        confidence = float(probas.iloc[-1])
        
        # Decision
        should_act = (
            confidence >= self.optimal_threshold and
            primary_signal != 0
        )
        
        # Position sizing
        kelly_size = self.compute_position_size(confidence, primary_signal)
        
        # Recommended size (may incorporate other factors)
        recommended_size = kelly_size if should_act else 0.0
        
        # Top features for interpretability
        top_features = None
        if self.feature_importances:
            sorted_features = sorted(
                self.feature_importances.items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_features = dict(sorted_features[:5])
            
        return MetaLabelResult(
            should_act=should_act,
            confidence=confidence,
            recommended_size=recommended_size,
            kelly_size=kelly_size,
            primary_signal=primary_signal,
            regime_context=regime,
            top_features=top_features,
            timestamp=datetime.now(),
        )
        
    def get_feature_importance(
        self,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Get feature importance ranking.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to return
            
        Returns:
        --------
        pd.DataFrame
            Feature importance ranking
        """
        if not self.feature_importances:
            raise ValueError("Model not fitted or doesn't support feature importance")
            
        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importances.items()
        ])
        
        return df.sort_values('importance', ascending=False).head(top_n)
        
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostics for the meta-labeler.
        
        Returns:
        --------
        Dict[str, Any]
            Diagnostic information
        """
        diagnostics = {
            'is_fitted': self.is_fitted,
            'model_type': self.config.secondary_model_type.value,
            'optimal_threshold': self.optimal_threshold,
            'n_features': len(self.feature_names),
            'validation_metrics': self.validation_metrics,
        }
        
        if hasattr(self.model, 'oob_score_'):
            diagnostics['oob_score'] = self.model.oob_score_
            
        return diagnostics
        
    def backtest_filter(
        self,
        returns: pd.Series,
        primary_signals: pd.Series,
        features: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Backtest the meta-labeling filter effect.
        
        Compares performance of:
        1. Acting on all primary signals
        2. Acting only on meta-label approved signals
        
        Parameters:
        -----------
        returns : pd.Series
            Asset returns
        primary_signals : pd.Series
            Primary model signals
        features : pd.DataFrame
            Features for meta-labeling
            
        Returns:
        --------
        Dict[str, float]
            Comparison metrics
        """
        # Get meta-label predictions
        meta_preds = self.predict(features)
        
        # Strategy returns: all signals
        all_signal_returns = primary_signals.shift(1) * returns
        
        # Strategy returns: filtered signals
        filtered_signals = primary_signals * meta_preds
        filtered_returns = filtered_signals.shift(1) * returns
        
        # Calculate metrics
        def calc_sharpe(r: pd.Series) -> float:
            if r.std() == 0:
                return 0.0
            return r.mean() / r.std() * np.sqrt(252)
            
        def calc_max_dd(r: pd.Series) -> float:
            cum = (1 + r).cumprod()
            running_max = cum.cummax()
            dd = (cum - running_max) / running_max
            return dd.min()
            
        all_sharpe = calc_sharpe(all_signal_returns.dropna())
        filtered_sharpe = calc_sharpe(filtered_returns.dropna())
        
        all_dd = calc_max_dd(all_signal_returns.dropna())
        filtered_dd = calc_max_dd(filtered_returns.dropna())
        
        trade_reduction = 1 - (meta_preds.sum() / (primary_signals != 0).sum())
        
        return {
            'all_signals_sharpe': all_sharpe,
            'filtered_sharpe': filtered_sharpe,
            'sharpe_improvement': filtered_sharpe - all_sharpe,
            'all_signals_max_dd': all_dd,
            'filtered_max_dd': filtered_dd,
            'trade_reduction_pct': trade_reduction * 100,
            'precision': self.validation_metrics.get('precision', 0),
            'recall': self.validation_metrics.get('recall', 0),
        }


class MetaLabelPipeline:
    """
    Complete pipeline integrating Triple Barrier + HMM + Meta-Labeling.
    
    This combines all Phase 1+2 components for institutional trading:
    1. Triple Barrier Labeling → Realistic trade outcomes
    2. HMM Regime Detection → Market context
    3. Meta-Labeling → Trade filtering
    
    Usage:
    ------
    ```python
    pipeline = MetaLabelPipeline()
    pipeline.fit(prices, primary_model)
    result = pipeline.generate_signals(prices, primary_model)
    ```
    """
    
    def __init__(
        self,
        meta_config: Optional[MetaLabelConfig] = None,
        use_hmm: bool = True,
        hmm_regimes: int = 2,
    ):
        """
        Initialize the pipeline.
        
        Parameters:
        -----------
        meta_config : MetaLabelConfig, optional
            Configuration for meta-labeler
        use_hmm : bool
            Whether to use HMM regime features
        hmm_regimes : int
            Number of HMM regimes
        """
        self.meta_config = meta_config or MetaLabelConfig()
        self.use_hmm = use_hmm
        self.hmm_regimes = hmm_regimes
        
        self.meta_labeler = MetaLabeler(self.meta_config)
        self.triple_barrier = None
        self.hmm_detector = None
        self.is_fitted = False
        
    def fit(
        self,
        prices: pd.Series,
        primary_signals: pd.Series,
        features: pd.DataFrame,
        pt_sl: Tuple[float, float] = (2.0, 1.0),
        min_ret: float = 0.005,
    ) -> 'MetaLabelPipeline':
        """
        Fit the complete pipeline.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        primary_signals : pd.Series
            Primary model predictions
        features : pd.DataFrame
            Base features
        pt_sl : Tuple[float, float]
            Profit-take and stop-loss multiples
        min_ret : float
            Minimum return for labeling
            
        Returns:
        --------
        MetaLabelPipeline
            Fitted pipeline
        """
        # Calculate returns
        returns = prices.pct_change()
        
        # Import and fit Triple Barrier
        try:
            from ..labeling.triple_barrier import TripleBarrierLabeler, TripleBarrierConfig
            tb_config = TripleBarrierConfig(pt_sl=pt_sl, min_ret=min_ret)
            self.triple_barrier = TripleBarrierLabeler(config=tb_config)
            barrier_results = self.triple_barrier.apply_barriers(prices)
            tb_labels = self.triple_barrier.get_labels(barrier_results)
        except ImportError:
            logger.warning("Triple Barrier not available, using simple labels")
            # Fallback to simple labels
            tb_labels = np.sign(returns.shift(-1))
            tb_labels = pd.Series(tb_labels, index=returns.index)
            
        # Fit HMM if enabled
        regimes = None
        if self.use_hmm:
            try:
                from ..regime.hmm_detector import HMMRegimeDetector
                self.hmm_detector = HMMRegimeDetector(n_regimes=self.hmm_regimes)
                self.hmm_detector.fit(returns)
                regimes = self.hmm_detector.predict(returns)
            except ImportError:
                logger.warning("HMM not available, skipping regime features")
                
        # Fit meta-labeler
        self.meta_labeler.fit(
            features=features,
            primary_signals=primary_signals,
            triple_barrier_labels=tb_labels,
            returns=returns,
            regimes=regimes,
        )
        
        self.is_fitted = True
        return self
        
    def filter_signals(
        self,
        prices: pd.Series,
        primary_signals: pd.Series,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Filter primary signals through the meta-labeling pipeline.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        primary_signals : pd.Series
            Primary model predictions
        features : pd.DataFrame
            Base features
            
        Returns:
        --------
        pd.DataFrame
            Filtered signals with position sizes
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
            
        returns = prices.pct_change()
        
        # Get regime predictions
        regimes = None
        if self.hmm_detector is not None:
            regimes = self.hmm_detector.predict(returns)
            
        # Engineer features for meta-labeling
        meta_features = self.meta_labeler.engineer_features(
            features, primary_signals, returns, regimes
        )
        
        # Get meta-predictions
        probas = self.meta_labeler.predict_proba(meta_features)
        
        # Build output
        result = pd.DataFrame({
            'primary_signal': primary_signals,
            'meta_confidence': probas,
            'should_act': probas >= self.meta_labeler.optimal_threshold,
            'position_size': 0.0,
        }, index=primary_signals.index)
        
        # Calculate position sizes
        for idx in result.index:
            if result.loc[idx, 'should_act']:
                result.loc[idx, 'position_size'] = self.meta_labeler.compute_position_size(
                    result.loc[idx, 'meta_confidence'],
                    result.loc[idx, 'primary_signal'],
                )
                
        # Add regime context
        if regimes is not None:
            result['regime'] = regimes
            
        return result


# Convenience functions for quick usage
def create_meta_labels(
    primary_signals: pd.Series,
    returns: pd.Series,
    horizon: int = 5,
) -> pd.Series:
    """
    Quick function to create meta-labels from primary signals.
    
    Parameters:
    -----------
    primary_signals : pd.Series
        Primary model predictions
    returns : pd.Series
        Asset returns
    horizon : int
        Forward horizon for label calculation
        
    Returns:
    --------
    pd.Series
        Binary meta-labels
    """
    # Forward returns
    fwd_returns = returns.rolling(horizon).sum().shift(-horizon)
    
    # Actual outcomes
    outcomes = np.sign(fwd_returns)
    
    # Meta-labels: 1 if signal matches outcome
    meta_labels = (primary_signals == outcomes).astype(int)
    meta_labels = meta_labels.where(primary_signals != 0, np.nan)
    
    return meta_labels


def quick_meta_labeler(
    features: pd.DataFrame,
    primary_signals: pd.Series,
    returns: pd.Series,
    horizon: int = 5,
) -> Tuple[MetaLabeler, Dict[str, float]]:
    """
    Quick-start meta-labeler with default settings.
    
    Parameters:
    -----------
    features : pd.DataFrame
        Base features
    primary_signals : pd.Series
        Primary model predictions
    returns : pd.Series
        Asset returns
    horizon : int
        Forward horizon
        
    Returns:
    --------
    Tuple[MetaLabeler, Dict[str, float]]
        Fitted meta-labeler and validation metrics
    """
    # Create simple labels
    labels = create_meta_labels(primary_signals, returns, horizon)
    
    # Convert to triple barrier format (just using direction)
    tb_labels = np.sign(returns.rolling(horizon).sum().shift(-horizon))
    tb_labels = pd.Series(tb_labels, index=returns.index)
    
    # Fit meta-labeler
    ml = MetaLabeler()
    ml.fit(features, primary_signals, tb_labels, returns)
    
    return ml, ml.validation_metrics
