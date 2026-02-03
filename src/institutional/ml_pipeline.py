"""
Phase 1+2 ML Integration Module
================================

This module provides the unified integration layer for all institutional ML
enhancements from Lopez de Prado (2018) "Advances in Financial Machine Learning".

Components Integrated:
- Triple Barrier Labeling (Phase 1.1)
- Fractional Differentiation (Phase 1.2)
- HMM Regime Detection (Phase 2.2)
- Meta-Labeling (Phase 2.1)

This creates a complete institutional-grade ML pipeline for systematic trading.

Author: Nuble Institutional ML
License: Proprietary
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import warnings

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class PipelineConfig:
    """
    Complete configuration for the ML pipeline.
    
    This consolidates all component configurations into one place.
    """
    
    # Triple Barrier settings
    barrier_pt_sl: Tuple[float, float] = (2.0, 1.0)  # Profit-take, stop-loss multiples
    barrier_vertical_days: int = 5
    barrier_min_ret: float = 0.005
    barrier_vol_lookback: int = 21
    
    # Fractional Differentiation settings
    frac_diff_threshold: float = 0.01  # Weight threshold
    frac_diff_adf_threshold: float = 0.05  # ADF significance level
    auto_tune_d: bool = True  # Auto-find optimal d
    
    # HMM settings
    hmm_n_regimes: int = 2  # Number of market regimes
    hmm_covariance_type: str = 'full'
    hmm_n_iter: int = 100
    
    # Meta-labeling settings
    meta_min_confidence: float = 0.55
    meta_precision_target: float = 0.65
    meta_kelly_fraction: float = 0.25
    meta_n_estimators: int = 500
    
    # General pipeline settings
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 21, 63])
    purge_gap: int = 3
    embargo_pct: float = 0.01
    n_cv_splits: int = 5


@dataclass
class PipelineResult:
    """
    Complete result from the ML pipeline.
    
    Contains all signals, labels, and metadata for trading execution.
    """
    
    # Final trading signals
    signals: pd.Series  # Filtered signals (-1, 0, 1)
    position_sizes: pd.Series  # Recommended sizes [0, 1]
    confidence: pd.Series  # Meta-label confidence
    
    # Labels and features
    triple_barrier_labels: pd.Series
    regime_labels: Optional[pd.Series]
    frac_diff_features: Optional[pd.DataFrame]
    
    # Metadata
    timestamp: datetime
    config: PipelineConfig
    
    # Performance metrics
    metrics: Dict[str, float]


class InstitutionalMLPipeline:
    """
    Complete Institutional ML Pipeline.
    
    This class integrates all Phase 1+2 components:
    1. Triple Barrier Labeling → Realistic trade outcomes
    2. Fractional Differentiation → Stationary features
    3. HMM Regime Detection → Market context
    4. Meta-Labeling → Trade filtering
    
    The pipeline follows Lopez de Prado methodology throughout:
    - No lookahead bias
    - Purged cross-validation
    - Proper label construction
    - Walk-forward validation ready
    
    Usage:
    ------
    ```python
    # Initialize
    pipeline = InstitutionalMLPipeline(config)
    
    # Fit on training data
    pipeline.fit(prices_train, features_train, primary_signals_train)
    
    # Generate signals for live trading
    result = pipeline.generate_signals(prices_live, features_live, primary_signals_live)
    
    # Execute trades
    for i, row in result.iterrows():
        if row['signal'] != 0 and row['position_size'] > 0:
            execute(side=row['signal'], size=row['position_size'])
    ```
    
    References:
    -----------
    - Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.
        
        Parameters:
        -----------
        config : PipelineConfig, optional
            Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or PipelineConfig()
        
        # Component instances
        self.triple_barrier = None
        self.frac_differ = None
        self.hmm_detector = None
        self.meta_labeler = None
        
        # State
        self.is_fitted = False
        self.fit_date: Optional[datetime] = None
        self.training_samples: int = 0
        
        # Initialize components
        self._init_components()
        
    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        # Import and initialize Triple Barrier
        try:
            from .labeling.triple_barrier import TripleBarrierLabeler, TripleBarrierConfig
            tb_config = TripleBarrierConfig(
                pt_sl=self.config.barrier_pt_sl,
                vertical_days=self.config.barrier_vertical_days,
                min_ret=self.config.barrier_min_ret,
                vol_lookback=self.config.barrier_vol_lookback,
            )
            self.triple_barrier = TripleBarrierLabeler(config=tb_config)
            logger.info("Triple Barrier Labeler initialized")
        except ImportError as e:
            logger.warning(f"Triple Barrier not available: {e}")
            
        # Import and initialize Fractional Differentiation
        try:
            from .features.frac_diff import FractionalDifferentiator, FracDiffConfig
            fd_config = FracDiffConfig(
                threshold=self.config.frac_diff_threshold,
                adf_threshold=self.config.frac_diff_adf_threshold,
            )
            self.frac_differ = FractionalDifferentiator(config=fd_config)
            logger.info("Fractional Differentiator initialized")
        except ImportError as e:
            logger.warning(f"Fractional Differentiation not available: {e}")
            
        # Import and initialize HMM
        try:
            from .regime.hmm_detector import HMMRegimeDetector
            self.hmm_detector = HMMRegimeDetector(
                n_regimes=self.config.hmm_n_regimes,
                covariance_type=self.config.hmm_covariance_type,
                n_iter=self.config.hmm_n_iter,
            )
            logger.info("HMM Regime Detector initialized")
        except ImportError as e:
            logger.warning(f"HMM not available: {e}")
            
        # Import and initialize Meta-Labeler
        try:
            from .models.meta.meta_labeler import MetaLabeler, MetaLabelConfig
            ml_config = MetaLabelConfig(
                min_confidence=self.config.meta_min_confidence,
                precision_target=self.config.meta_precision_target,
                kelly_fraction=self.config.meta_kelly_fraction,
                n_estimators=self.config.meta_n_estimators,
            )
            self.meta_labeler = MetaLabeler(config=ml_config)
            logger.info("Meta-Labeler initialized")
        except ImportError as e:
            logger.warning(f"Meta-Labeler not available: {e}")
            
    def fit(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        primary_signals: pd.Series,
        verbose: bool = True,
    ) -> 'InstitutionalMLPipeline':
        """
        Fit the complete pipeline on training data.
        
        This fits all components in the correct order:
        1. Apply fractional differentiation to features
        2. Fit HMM on returns
        3. Apply triple barrier labeling
        4. Fit meta-labeler
        
        Parameters:
        -----------
        prices : pd.Series
            Price series (Close prices)
        features : pd.DataFrame
            Raw features from primary model
        primary_signals : pd.Series
            Predictions from primary model
        verbose : bool
            Print progress information
            
        Returns:
        --------
        InstitutionalMLPipeline
            Fitted pipeline instance
        """
        if verbose:
            logger.info("=" * 60)
            logger.info("INSTITUTIONAL ML PIPELINE - TRAINING")
            logger.info("=" * 60)
            
        returns = prices.pct_change()
        
        # Step 1: Fractional Differentiation
        enhanced_features = features.copy()
        if self.frac_differ is not None:
            if verbose:
                logger.info("Step 1: Applying Fractional Differentiation...")
            try:
                # Apply to price-based features
                for col in enhanced_features.columns:
                    if 'price' in col.lower() or 'close' in col.lower():
                        enhanced_features[f'{col}_frac'] = self.frac_differ.frac_diff_ffd(
                            enhanced_features[col]
                        )
                if verbose:
                    logger.info(f"  → Added fractionally differenced features")
            except Exception as e:
                logger.warning(f"Fractional diff failed: {e}")
        else:
            if verbose:
                logger.info("Step 1: Fractional Differentiation skipped (not available)")
                
        # Step 2: HMM Regime Detection
        regimes = None
        if self.hmm_detector is not None:
            if verbose:
                logger.info("Step 2: Fitting HMM Regime Detector...")
            try:
                self.hmm_detector.fit(returns.dropna())
                regimes = self.hmm_detector.predict(returns)
                if verbose:
                    regime_counts = regimes.value_counts()
                    logger.info(f"  → Detected {len(regime_counts)} regimes")
                    for r, count in regime_counts.items():
                        pct = count / len(regimes) * 100
                        logger.info(f"     Regime {r}: {pct:.1f}% of time")
            except Exception as e:
                logger.warning(f"HMM fitting failed: {e}")
        else:
            if verbose:
                logger.info("Step 2: HMM skipped (not available)")
                
        # Step 3: Triple Barrier Labeling
        tb_labels = None
        if self.triple_barrier is not None:
            if verbose:
                logger.info("Step 3: Applying Triple Barrier Labeling...")
            try:
                barrier_results = self.triple_barrier.apply_barriers(prices)
                tb_labels = self.triple_barrier.get_labels(barrier_results)
                if verbose:
                    label_dist = tb_labels.value_counts(normalize=True)
                    logger.info(f"  → Label distribution:")
                    for label, pct in label_dist.items():
                        label_name = {1: 'Long', -1: 'Short', 0: 'Neutral'}[label]
                        logger.info(f"     {label_name}: {pct*100:.1f}%")
            except Exception as e:
                logger.warning(f"Triple Barrier failed: {e}")
                # Fallback to simple labels
                tb_labels = np.sign(returns.shift(-self.config.barrier_vertical_days))
                tb_labels = pd.Series(tb_labels, index=returns.index)
        else:
            if verbose:
                logger.info("Step 3: Triple Barrier skipped, using simple labels")
            tb_labels = np.sign(returns.shift(-self.config.barrier_vertical_days))
            tb_labels = pd.Series(tb_labels, index=returns.index)
            
        # Step 4: Meta-Labeling
        if self.meta_labeler is not None:
            if verbose:
                logger.info("Step 4: Fitting Meta-Labeler...")
            try:
                self.meta_labeler.fit(
                    features=enhanced_features,
                    primary_signals=primary_signals,
                    triple_barrier_labels=tb_labels,
                    returns=returns,
                    regimes=regimes,
                    optimize_threshold=True,
                )
                if verbose:
                    metrics = self.meta_labeler.validation_metrics
                    logger.info(f"  → Meta-labeler fitted:")
                    logger.info(f"     Precision: {metrics.get('precision', 0):.3f}")
                    logger.info(f"     Recall: {metrics.get('recall', 0):.3f}")
                    logger.info(f"     Optimal Threshold: {metrics.get('threshold', 0):.3f}")
                    logger.info(f"     Trade Rate: {metrics.get('trade_rate', 0)*100:.1f}%")
            except Exception as e:
                logger.warning(f"Meta-Labeler fitting failed: {e}")
        else:
            if verbose:
                logger.info("Step 4: Meta-Labeler skipped (not available)")
                
        # Store state
        self.is_fitted = True
        self.fit_date = datetime.now()
        self.training_samples = len(prices)
        
        if verbose:
            logger.info("=" * 60)
            logger.info(f"Pipeline fitted on {self.training_samples} samples")
            logger.info("=" * 60)
            
        return self
        
    def generate_signals(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        primary_signals: pd.Series,
    ) -> PipelineResult:
        """
        Generate filtered trading signals.
        
        This is the main entry point for live trading.
        
        Parameters:
        -----------
        prices : pd.Series
            Current price series
        features : pd.DataFrame
            Current features
        primary_signals : pd.Series
            Primary model predictions
            
        Returns:
        --------
        PipelineResult
            Complete trading signals and metadata
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before generating signals")
            
        returns = prices.pct_change()
        
        # Get regime predictions
        regimes = None
        if self.hmm_detector is not None:
            try:
                regimes = self.hmm_detector.predict(returns)
            except Exception as e:
                logger.warning(f"HMM prediction failed: {e}")
                
        # Get current triple barrier labels (for reference)
        tb_labels = None
        if self.triple_barrier is not None:
            try:
                barrier_results = self.triple_barrier.apply_barriers(prices)
                tb_labels = self.triple_barrier.get_labels(barrier_results)
            except Exception as e:
                logger.warning(f"Triple Barrier failed: {e}")
                
        # Apply fractional differentiation
        enhanced_features = features.copy()
        if self.frac_differ is not None:
            try:
                for col in enhanced_features.columns:
                    if 'price' in col.lower() or 'close' in col.lower():
                        enhanced_features[f'{col}_frac'] = self.frac_differ.frac_diff_ffd(
                            enhanced_features[col]
                        )
            except Exception as e:
                logger.warning(f"Fractional diff failed: {e}")
                
        # Initialize output
        signals = primary_signals.copy()
        position_sizes = pd.Series(0.0, index=primary_signals.index)
        confidence = pd.Series(0.5, index=primary_signals.index)
        
        # Apply meta-labeling filter
        if self.meta_labeler is not None and self.meta_labeler.is_fitted:
            try:
                # Engineer features
                meta_features = self.meta_labeler.engineer_features(
                    enhanced_features, primary_signals, returns, regimes
                )
                
                # Get predictions
                probas = self.meta_labeler.predict_proba(meta_features)
                should_act = probas >= self.meta_labeler.optimal_threshold
                
                # Update outputs
                confidence = probas
                signals = signals.where(should_act, 0)
                
                # Calculate position sizes
                for idx in position_sizes.index:
                    if should_act.get(idx, False):
                        position_sizes.loc[idx] = self.meta_labeler.compute_position_size(
                            probas.get(idx, 0.5),
                            primary_signals.get(idx, 0),
                        )
                        
            except Exception as e:
                logger.warning(f"Meta-labeling failed: {e}")
                position_sizes = pd.Series(1.0, index=primary_signals.index)
                
        else:
            # No meta-labeler - use full signals
            position_sizes = pd.Series(1.0, index=primary_signals.index)
            
        # Calculate performance metrics
        metrics = self._calculate_metrics(
            returns, primary_signals, signals, position_sizes
        )
        
        return PipelineResult(
            signals=signals,
            position_sizes=position_sizes,
            confidence=confidence,
            triple_barrier_labels=tb_labels,
            regime_labels=regimes,
            frac_diff_features=enhanced_features if self.frac_differ else None,
            timestamp=datetime.now(),
            config=self.config,
            metrics=metrics,
        )
        
    def _calculate_metrics(
        self,
        returns: pd.Series,
        raw_signals: pd.Series,
        filtered_signals: pd.Series,
        position_sizes: pd.Series,
    ) -> Dict[str, float]:
        """Calculate performance comparison metrics."""
        # Raw strategy returns
        raw_strat = raw_signals.shift(1) * returns
        
        # Filtered strategy returns  
        filtered_strat = filtered_signals.shift(1) * returns * position_sizes.shift(1)
        
        # Metrics
        def sharpe(r: pd.Series) -> float:
            r = r.dropna()
            if len(r) < 20 or r.std() == 0:
                return 0.0
            return r.mean() / r.std() * np.sqrt(252)
            
        def max_dd(r: pd.Series) -> float:
            r = r.dropna()
            if len(r) < 2:
                return 0.0
            cum = (1 + r).cumprod()
            running_max = cum.cummax()
            dd = (cum - running_max) / running_max
            return abs(dd.min())
            
        raw_sharpe = sharpe(raw_strat)
        filt_sharpe = sharpe(filtered_strat)
        
        trade_count_raw = (raw_signals != 0).sum()
        trade_count_filt = (filtered_signals != 0).sum()
        trade_reduction = 1 - (trade_count_filt / max(trade_count_raw, 1))
        
        return {
            'raw_sharpe': raw_sharpe,
            'filtered_sharpe': filt_sharpe,
            'sharpe_improvement': filt_sharpe - raw_sharpe,
            'raw_max_dd': max_dd(raw_strat),
            'filtered_max_dd': max_dd(filtered_strat),
            'trade_reduction_pct': trade_reduction * 100,
            'avg_position_size': position_sizes.mean(),
        }
        
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline diagnostics.
        
        Returns:
        --------
        Dict[str, Any]
            Diagnostic information for all components
        """
        diagnostics = {
            'is_fitted': self.is_fitted,
            'fit_date': self.fit_date.isoformat() if self.fit_date else None,
            'training_samples': self.training_samples,
            'config': {
                'barrier_pt_sl': self.config.barrier_pt_sl,
                'hmm_regimes': self.config.hmm_n_regimes,
                'meta_confidence': self.config.meta_min_confidence,
            },
            'components': {
                'triple_barrier': self.triple_barrier is not None,
                'frac_diff': self.frac_differ is not None,
                'hmm': self.hmm_detector is not None,
                'meta_labeler': self.meta_labeler is not None,
            }
        }
        
        if self.meta_labeler and self.meta_labeler.is_fitted:
            diagnostics['meta_labeler_metrics'] = self.meta_labeler.validation_metrics
            
        if self.hmm_detector and hasattr(self.hmm_detector, 'regime_statistics'):
            diagnostics['regime_stats'] = {
                str(k): v for k, v in self.hmm_detector.regime_statistics.items()
            }
            
        return diagnostics
        
    def save(self, path: str) -> None:
        """Save pipeline to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Pipeline saved to {path}")
        
    @classmethod
    def load(cls, path: str) -> 'InstitutionalMLPipeline':
        """Load pipeline from disk."""
        import pickle
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)
        logger.info(f"Pipeline loaded from {path}")
        return pipeline


# Convenience function
def create_institutional_pipeline(
    aggressive: bool = False,
    conservative: bool = False,
) -> InstitutionalMLPipeline:
    """
    Create a pre-configured institutional pipeline.
    
    Parameters:
    -----------
    aggressive : bool
        Use more aggressive settings (lower thresholds)
    conservative : bool
        Use more conservative settings (higher thresholds)
        
    Returns:
    --------
    InstitutionalMLPipeline
        Configured pipeline
    """
    if aggressive and conservative:
        raise ValueError("Cannot be both aggressive and conservative")
        
    config = PipelineConfig()
    
    if aggressive:
        config.meta_min_confidence = 0.50
        config.meta_precision_target = 0.55
        config.barrier_pt_sl = (3.0, 1.0)  # Larger profit targets
        config.meta_kelly_fraction = 0.35  # More aggressive sizing
        
    elif conservative:
        config.meta_min_confidence = 0.65
        config.meta_precision_target = 0.75
        config.barrier_pt_sl = (1.5, 1.0)  # Smaller profit targets
        config.meta_kelly_fraction = 0.15  # More conservative sizing
        
    return InstitutionalMLPipeline(config)
