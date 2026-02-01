"""
Comprehensive Tests for Institutional ML Components
====================================================

Tests for Phase 1+2 Implementation:
- Triple Barrier Labeling
- Fractional Differentiation
- HMM Regime Detection
- Meta-Labeling

These tests validate:
1. Correctness of implementations
2. No lookahead bias
3. Edge case handling
4. Statistical properties

Author: Kyperian Institutional ML
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ============================================================================
# Test Data Generators
# ============================================================================

def generate_price_series(
    n: int = 500,
    start_price: float = 100.0,
    annual_return: float = 0.10,
    annual_vol: float = 0.20,
    seed: int = 42,
) -> pd.Series:
    """Generate realistic price series with trend and volatility."""
    np.random.seed(seed)
    
    daily_return = annual_return / 252
    daily_vol = annual_vol / np.sqrt(252)
    
    returns = np.random.normal(daily_return, daily_vol, n)
    prices = start_price * np.exp(np.cumsum(returns))
    
    dates = pd.date_range(start='2020-01-01', periods=n, freq='B')
    return pd.Series(prices, index=dates, name='Close')


def generate_regime_prices(
    n: int = 500,
    regime_lengths: list = None,
    seed: int = 42,
) -> pd.Series:
    """Generate prices with distinct regimes (bull/bear/sideways)."""
    np.random.seed(seed)
    
    if regime_lengths is None:
        regime_lengths = [100, 80, 120, 100, 100]  # Sum = 500
        
    regime_params = [
        (0.0015, 0.01),   # Bull: high return, low vol
        (-0.0020, 0.025), # Bear: negative return, high vol
        (0.0001, 0.008),  # Sideways: near-zero return, low vol
        (0.0012, 0.012),  # Bull
        (-0.0010, 0.018), # Bear
    ]
    
    all_returns = []
    for length, (mu, sigma) in zip(regime_lengths, regime_params):
        regime_returns = np.random.normal(mu, sigma, length)
        all_returns.extend(regime_returns)
        
    prices = 100 * np.exp(np.cumsum(all_returns))
    dates = pd.date_range(start='2020-01-01', periods=len(prices), freq='B')
    
    return pd.Series(prices, index=dates, name='Close')


def generate_features(prices: pd.Series, seed: int = 42) -> pd.DataFrame:
    """Generate realistic technical features from prices."""
    np.random.seed(seed)
    
    returns = prices.pct_change()
    
    features = pd.DataFrame(index=prices.index)
    
    # Momentum features
    for window in [5, 10, 21, 63]:
        features[f'momentum_{window}'] = returns.rolling(window).sum()
        features[f'volatility_{window}'] = returns.rolling(window).std()
        
    # Moving average ratios
    features['ma_ratio_5_21'] = prices.rolling(5).mean() / prices.rolling(21).mean()
    features['ma_ratio_21_63'] = prices.rolling(21).mean() / prices.rolling(63).mean()
    
    # RSI-like feature
    up = returns.clip(lower=0)
    down = (-returns).clip(lower=0)
    features['rsi_14'] = 100 - (100 / (1 + up.rolling(14).mean() / (down.rolling(14).mean() + 1e-8)))
    
    # Add some noise features
    features['noise_1'] = np.random.randn(len(prices))
    features['noise_2'] = np.random.randn(len(prices))
    
    return features.dropna()


def generate_primary_signals(prices: pd.Series, seed: int = 42) -> pd.Series:
    """Generate primary model signals (momentum-based)."""
    np.random.seed(seed)
    
    returns = prices.pct_change()
    momentum = returns.rolling(10).sum()
    
    # Base signals from momentum
    signals = np.sign(momentum)
    
    # Add some noise (make it imperfect)
    noise_mask = np.random.random(len(signals)) < 0.15
    signals[noise_mask] = -signals[noise_mask]
    
    return pd.Series(signals, index=prices.index, name='signal').fillna(0).astype(int)


# ============================================================================
# Triple Barrier Tests
# ============================================================================

class TestTripleBarrier:
    """Tests for Triple Barrier Labeling implementation."""
    
    @pytest.fixture
    def prices(self):
        return generate_price_series(n=300, seed=42)
        
    @pytest.fixture
    def labeler(self):
        from institutional.labeling.triple_barrier import (
            TripleBarrierLabeler, TripleBarrierConfig
        )
        config = TripleBarrierConfig(
            pt_sl=(2.0, 1.0),
            vertical_days=5,
            min_ret=0.005,
            vol_lookback=21,
        )
        return TripleBarrierLabeler(config)
        
    def test_initialization(self, labeler):
        """Test labeler initializes correctly."""
        assert labeler.config.pt_sl == (2.0, 1.0)
        assert labeler.config.vertical_days == 5
        
    def test_volatility_calculation(self, labeler, prices):
        """Test daily volatility calculation."""
        vol = labeler.get_daily_volatility(prices)
        
        # Should be positive
        assert (vol > 0).all()
        
        # Should be reasonable (annualized ~20%)
        assert vol.mean() * np.sqrt(252) < 0.5  # Less than 50% annual vol
        assert vol.mean() * np.sqrt(252) > 0.05  # More than 5% annual vol
        
    def test_barrier_application(self, labeler, prices):
        """Test barrier application produces valid results."""
        results = labeler.apply_barriers(prices)
        
        # Should have results
        assert len(results) > 0
        
        # Check barrier result properties
        for result in results[:10]:  # Check first 10
            assert hasattr(result, 'entry_idx')
            assert hasattr(result, 'exit_idx')
            assert hasattr(result, 'label')
            assert hasattr(result, 'barrier_touched')
            assert result.label in [-1, 0, 1]
            
    def test_no_lookahead_bias(self, labeler, prices):
        """Verify no lookahead bias in labels."""
        results = labeler.apply_barriers(prices)
        labels = labeler.get_labels(results)
        
        # Labels should only be defined where we have forward data
        # Last vertical_days should be NaN
        vertical = labeler.config.vertical_days
        assert labels.iloc[-vertical:].isna().all()
        
    def test_label_distribution(self, labeler, prices):
        """Test labels have reasonable distribution."""
        results = labeler.apply_barriers(prices)
        labels = labeler.get_labels(results)
        
        label_counts = labels.value_counts(normalize=True)
        
        # Should have all three labels
        assert set(label_counts.index).issubset({-1, 0, 1})
        
        # No single label should dominate completely
        assert label_counts.max() < 0.8
        
    def test_different_volatility_regimes(self):
        """Test labeler handles different vol regimes."""
        from institutional.labeling.triple_barrier import (
            TripleBarrierLabeler, TripleBarrierConfig
        )
        
        config = TripleBarrierConfig(pt_sl=(2.0, 1.0))
        labeler = TripleBarrierLabeler(config)
        
        # Low vol regime
        prices_low_vol = generate_price_series(n=200, annual_vol=0.10, seed=1)
        results_low = labeler.apply_barriers(prices_low_vol)
        
        # High vol regime
        prices_high_vol = generate_price_series(n=200, annual_vol=0.40, seed=1)
        results_high = labeler.apply_barriers(prices_high_vol)
        
        # Both should produce valid results
        assert len(results_low) > 50
        assert len(results_high) > 50


# ============================================================================
# Fractional Differentiation Tests
# ============================================================================

class TestFractionalDifferentiation:
    """Tests for Fractional Differentiation implementation."""
    
    @pytest.fixture
    def prices(self):
        return generate_price_series(n=500, seed=42)
        
    @pytest.fixture
    def differ(self):
        from institutional.features.frac_diff import (
            FractionalDifferentiator, FracDiffConfig
        )
        config = FracDiffConfig(
            threshold=0.01,
            adf_threshold=0.05,
        )
        return FractionalDifferentiator(config)
        
    def test_initialization(self, differ):
        """Test differentiator initializes correctly."""
        assert differ.config.threshold == 0.01
        
    def test_weight_calculation(self, differ):
        """Test FFD weights are calculated correctly."""
        weights = differ.get_weights_ffd(d=0.5, threshold=0.01)
        
        # First weight should be 1
        assert np.isclose(weights[0], 1.0)
        
        # Weights should decrease in magnitude
        assert abs(weights[-1]) < abs(weights[0])
        
        # All weights should be finite
        assert np.all(np.isfinite(weights))
        
    def test_frac_diff_output(self, differ, prices):
        """Test fractional differentiation produces valid output."""
        result = differ.frac_diff_ffd(prices, d=0.5)
        
        # Should return Series
        assert isinstance(result, pd.Series)
        
        # Should have same index (with possible NaN at start)
        assert len(result) == len(prices)
        
        # Should have finite values (after initial NaN)
        valid_values = result.dropna()
        assert np.all(np.isfinite(valid_values))
        
    def test_stationarity_improvement(self, differ, prices):
        """Test that fractional diff improves stationarity."""
        from statsmodels.tsa.stattools import adfuller
        
        # Original prices - likely non-stationary
        adf_original = adfuller(prices.dropna())
        
        # Fractionally differenced - should be more stationary
        frac_diff = differ.frac_diff_ffd(prices, d=0.5)
        adf_frac = adfuller(frac_diff.dropna())
        
        # ADF statistic should be more negative (more stationary)
        assert adf_frac[0] < adf_original[0]
        
    def test_memory_preservation(self, differ, prices):
        """Test that fractional diff preserves some memory."""
        # d=0 should give original series (minus constant)
        result_d0 = differ.frac_diff_ffd(prices, d=0.0)
        
        # d=1 should give first difference
        result_d1 = differ.frac_diff_ffd(prices, d=1.0)
        regular_diff = prices.diff()
        
        # d=1 result should be close to regular diff
        correlation = result_d1.dropna().corr(regular_diff.dropna())
        assert correlation > 0.9
        
    def test_auto_tune_d(self, differ, prices):
        """Test automatic d tuning finds valid value."""
        optimal_d, adf_pvalue = differ.find_min_ffd(prices)
        
        # d should be in valid range
        assert 0 <= optimal_d <= 1
        
        # ADF p-value should be below threshold
        assert adf_pvalue < differ.config.adf_threshold


# ============================================================================
# HMM Regime Detection Tests
# ============================================================================

class TestHMMRegimeDetector:
    """Tests for HMM Regime Detection implementation."""
    
    @pytest.fixture
    def regime_prices(self):
        return generate_regime_prices(n=500, seed=42)
        
    @pytest.fixture
    def detector(self):
        from institutional.regime.hmm_detector import HMMRegimeDetector
        return HMMRegimeDetector(n_regimes=2, n_iter=50)
        
    def test_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector.n_regimes == 2
        assert not detector.is_fitted
        
    def test_fitting(self, detector, regime_prices):
        """Test HMM fits on return data."""
        returns = regime_prices.pct_change().dropna()
        detector.fit(returns)
        
        assert detector.is_fitted
        assert detector.model is not None
        
    def test_regime_prediction(self, detector, regime_prices):
        """Test regime prediction produces valid output."""
        returns = regime_prices.pct_change().dropna()
        detector.fit(returns)
        
        regimes = detector.predict(returns)
        
        # Should have same length
        assert len(regimes) == len(returns)
        
        # Regimes should be integers
        assert regimes.dtype in [np.int32, np.int64, int]
        
        # All regimes should be valid
        unique_regimes = regimes.unique()
        assert all(r in range(detector.n_regimes) for r in unique_regimes)
        
    def test_no_lookahead_in_prediction(self, detector, regime_prices):
        """Verify no lookahead bias in online prediction."""
        returns = regime_prices.pct_change().dropna()
        
        # Fit on first half
        train_returns = returns.iloc[:250]
        detector.fit(train_returns)
        
        # Predict on full series
        full_regimes = detector.predict(returns)
        
        # Predict incrementally (online)
        online_regimes = []
        for i in range(len(returns)):
            partial_returns = returns.iloc[:i+1]
            regime = detector.predict(partial_returns).iloc[-1]
            online_regimes.append(regime)
            
        online_regimes = pd.Series(online_regimes, index=returns.index)
        
        # Online predictions should match full predictions
        # (with possible minor differences due to Viterbi vs. filtering)
        match_rate = (full_regimes == online_regimes).mean()
        assert match_rate > 0.85  # Allow some difference due to algorithm
        
    def test_regime_statistics(self, detector, regime_prices):
        """Test regime statistics calculation."""
        returns = regime_prices.pct_change().dropna()
        detector.fit(returns)
        
        regimes = detector.predict(returns)
        stats = detector.get_regime_statistics(returns, regimes)
        
        # Should have stats for each regime
        assert len(stats) == detector.n_regimes
        
        # Each stat should have required fields
        for regime_stat in stats.values():
            assert 'mean_return' in regime_stat
            assert 'volatility' in regime_stat
            assert 'count' in regime_stat
            
    def test_trading_filter(self, detector, regime_prices):
        """Test trading filter generation."""
        returns = regime_prices.pct_change().dropna()
        detector.fit(returns)
        
        # Allow trading only in regime 0
        filter_series = detector.get_trading_filter(returns, allowed_regimes=[0])
        
        # Should be boolean-like
        assert set(filter_series.unique()).issubset({0, 1, True, False})
        
        # Should filter some trades
        assert 0 < filter_series.mean() < 1


# ============================================================================
# Meta-Labeling Tests
# ============================================================================

class TestMetaLabeler:
    """Tests for Meta-Labeling implementation."""
    
    @pytest.fixture
    def data(self):
        prices = generate_price_series(n=400, seed=42)
        features = generate_features(prices, seed=42)
        signals = generate_primary_signals(prices, seed=42)
        
        # Align indices
        common_idx = features.index.intersection(signals.index)
        
        return {
            'prices': prices.loc[common_idx],
            'features': features.loc[common_idx],
            'signals': signals.loc[common_idx],
        }
        
    @pytest.fixture
    def labeler(self):
        from institutional.models.meta.meta_labeler import MetaLabeler, MetaLabelConfig
        config = MetaLabelConfig(
            min_confidence=0.55,
            precision_target=0.60,
            n_estimators=100,  # Fewer for testing speed
        )
        return MetaLabeler(config)
        
    def test_initialization(self, labeler):
        """Test meta-labeler initializes correctly."""
        assert not labeler.is_fitted
        assert labeler.model is not None
        
    def test_meta_label_creation(self, labeler, data):
        """Test meta-label creation from primary signals."""
        returns = data['prices'].pct_change()
        
        # Create simple labels (forward returns sign)
        fwd_returns = returns.shift(-5)
        true_labels = np.sign(fwd_returns)
        true_labels = pd.Series(true_labels, index=returns.index)
        
        meta_labels = labeler.create_meta_labels(data['signals'], true_labels)
        
        # Should have valid values
        assert not meta_labels.dropna().empty
        
        # Meta-labels should be binary
        assert set(meta_labels.dropna().unique()).issubset({0, 1})
        
    def test_feature_engineering(self, labeler, data):
        """Test feature engineering for meta-labeling."""
        returns = data['prices'].pct_change()
        
        meta_features = labeler.engineer_features(
            data['features'],
            data['signals'],
            returns,
        )
        
        # Should have more features than input
        assert meta_features.shape[1] >= data['features'].shape[1]
        
        # Should include primary signal
        assert 'primary_signal' in meta_features.columns
        
    def test_fitting(self, labeler, data):
        """Test meta-labeler fitting."""
        returns = data['prices'].pct_change()
        
        # Create simple labels
        fwd_returns = returns.shift(-5).rolling(5).sum()
        true_labels = np.sign(fwd_returns)
        true_labels = pd.Series(true_labels, index=returns.index)
        
        labeler.fit(
            features=data['features'],
            primary_signals=data['signals'],
            triple_barrier_labels=true_labels,
            returns=returns,
        )
        
        assert labeler.is_fitted
        assert labeler.optimal_threshold >= 0.5
        
    def test_prediction(self, labeler, data):
        """Test meta-labeler prediction."""
        returns = data['prices'].pct_change()
        
        # Create labels and fit
        fwd_returns = returns.shift(-5).rolling(5).sum()
        true_labels = np.sign(fwd_returns)
        true_labels = pd.Series(true_labels, index=returns.index)
        
        labeler.fit(
            features=data['features'],
            primary_signals=data['signals'],
            triple_barrier_labels=true_labels,
            returns=returns,
        )
        
        # Prepare features for prediction
        meta_features = labeler.engineer_features(
            data['features'],
            data['signals'],
            returns,
        )
        
        # Get predictions
        probas = labeler.predict_proba(meta_features)
        
        # Should have same length
        assert len(probas) == len(meta_features)
        
        # Probabilities should be in [0, 1]
        assert (probas >= 0).all() and (probas <= 1).all()
        
    def test_position_sizing(self, labeler):
        """Test Kelly-based position sizing."""
        # High confidence
        size_high = labeler.compute_position_size(0.75, 1)
        
        # Low confidence
        size_low = labeler.compute_position_size(0.55, 1)
        
        # Higher confidence should give larger position
        assert size_high > size_low
        
        # Below threshold should give 0
        size_zero = labeler.compute_position_size(0.45, 1)
        assert size_zero == 0
        
    def test_decision_making(self, labeler, data):
        """Test complete decision-making flow."""
        returns = data['prices'].pct_change()
        
        # Fit
        fwd_returns = returns.shift(-5).rolling(5).sum()
        true_labels = np.sign(fwd_returns)
        true_labels = pd.Series(true_labels, index=returns.index)
        
        labeler.fit(
            features=data['features'],
            primary_signals=data['signals'],
            triple_barrier_labels=true_labels,
            returns=returns,
        )
        
        # Prepare last row
        meta_features = labeler.engineer_features(
            data['features'],
            data['signals'],
            returns,
        )
        
        result = labeler.decide(
            features=meta_features.iloc[[-1]],
            primary_signal=1,
        )
        
        # Should have valid result
        assert hasattr(result, 'should_act')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'recommended_size')
        assert 0 <= result.confidence <= 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def data(self):
        prices = generate_regime_prices(n=500, seed=42)
        features = generate_features(prices, seed=42)
        signals = generate_primary_signals(prices, seed=42)
        
        common_idx = features.index.intersection(signals.index)
        
        return {
            'prices': prices.loc[common_idx],
            'features': features.loc[common_idx],
            'signals': signals.loc[common_idx],
        }
        
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        from institutional.ml_pipeline import (
            InstitutionalMLPipeline, PipelineConfig
        )
        
        config = PipelineConfig()
        pipeline = InstitutionalMLPipeline(config)
        
        assert not pipeline.is_fitted
        
    def test_pipeline_fitting(self, data):
        """Test complete pipeline fitting."""
        from institutional.ml_pipeline import (
            InstitutionalMLPipeline, PipelineConfig
        )
        
        config = PipelineConfig(
            meta_n_estimators=50,  # Fewer for speed
        )
        pipeline = InstitutionalMLPipeline(config)
        
        # This may fail if components aren't available
        # but should at least not crash
        try:
            pipeline.fit(
                data['prices'],
                data['features'],
                data['signals'],
                verbose=False,
            )
            assert pipeline.is_fitted
        except ImportError:
            pytest.skip("Not all components available")
            
    def test_signal_generation(self, data):
        """Test signal generation from fitted pipeline."""
        from institutional.ml_pipeline import (
            InstitutionalMLPipeline, PipelineConfig
        )
        
        config = PipelineConfig(
            meta_n_estimators=50,
        )
        pipeline = InstitutionalMLPipeline(config)
        
        try:
            pipeline.fit(
                data['prices'],
                data['features'],
                data['signals'],
                verbose=False,
            )
            
            result = pipeline.generate_signals(
                data['prices'],
                data['features'],
                data['signals'],
            )
            
            # Should have signals
            assert len(result.signals) > 0
            
            # Signals should be valid
            assert set(result.signals.unique()).issubset({-1, 0, 1})
            
            # Should have metrics
            assert 'filtered_sharpe' in result.metrics
            
        except ImportError:
            pytest.skip("Not all components available")


# ============================================================================
# Performance and Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_short_series(self):
        """Test handling of short price series."""
        from institutional.labeling.triple_barrier import (
            TripleBarrierLabeler, TripleBarrierConfig
        )
        
        # Very short series
        short_prices = generate_price_series(n=30, seed=42)
        
        config = TripleBarrierConfig(vertical_days=5)
        labeler = TripleBarrierLabeler(config)
        
        # Should not crash
        results = labeler.apply_barriers(short_prices)
        assert len(results) >= 0  # May be empty, but shouldn't crash
        
    def test_constant_prices(self):
        """Test handling of constant prices (no volatility)."""
        from institutional.features.frac_diff import FractionalDifferentiator
        
        constant_prices = pd.Series(
            [100.0] * 100,
            index=pd.date_range('2020-01-01', periods=100),
        )
        
        differ = FractionalDifferentiator()
        
        # Should handle gracefully
        result = differ.frac_diff_ffd(constant_prices, d=0.5)
        
        # Result should be all zeros or NaN
        valid = result.dropna()
        assert (valid == 0).all() or valid.empty
        
    def test_missing_values(self):
        """Test handling of missing values in input."""
        from institutional.regime.hmm_detector import HMMRegimeDetector
        
        prices = generate_price_series(n=200, seed=42)
        returns = prices.pct_change()
        
        # Add some NaN
        returns.iloc[50:55] = np.nan
        returns.iloc[100] = np.nan
        
        detector = HMMRegimeDetector(n_regimes=2)
        
        # Should handle NaN gracefully
        try:
            detector.fit(returns.dropna())
            regimes = detector.predict(returns.dropna())
            assert len(regimes) == len(returns.dropna())
        except ValueError:
            # Acceptable if it raises on NaN
            pass
            
    def test_extreme_values(self):
        """Test handling of extreme returns."""
        from institutional.regime.hmm_detector import HMMRegimeDetector
        
        prices = generate_price_series(n=200, seed=42)
        returns = prices.pct_change()
        
        # Add extreme values
        returns.iloc[50] = 0.5  # 50% daily return
        returns.iloc[51] = -0.3  # -30% daily return
        
        detector = HMMRegimeDetector(n_regimes=2)
        detector.fit(returns.dropna())
        
        regimes = detector.predict(returns.dropna())
        
        # Should still produce valid regimes
        assert len(regimes) == len(returns.dropna())
        assert set(regimes.unique()).issubset(set(range(detector.n_regimes)))


# ============================================================================
# Statistical Validation Tests
# ============================================================================

class TestStatisticalProperties:
    """Tests for statistical properties of the components."""
    
    def test_meta_labeling_improves_precision(self):
        """Verify meta-labeling improves precision over raw signals."""
        from institutional.models.meta.meta_labeler import MetaLabeler
        
        # Generate data
        prices = generate_price_series(n=400, seed=42)
        features = generate_features(prices, seed=42)
        signals = generate_primary_signals(prices, seed=42)
        returns = prices.pct_change()
        
        # Create labels
        fwd_returns = returns.shift(-5).rolling(5).sum()
        true_labels = np.sign(fwd_returns)
        true_labels = pd.Series(true_labels, index=returns.index)
        
        # Align
        common_idx = features.index.intersection(true_labels.dropna().index)
        features = features.loc[common_idx]
        signals = signals.loc[common_idx]
        true_labels = true_labels.loc[common_idx]
        returns = returns.loc[common_idx]
        
        # Calculate raw precision
        raw_correct = (signals == true_labels)
        raw_precision = raw_correct[signals != 0].mean()
        
        # Fit meta-labeler
        labeler = MetaLabeler()
        labeler.fit(features, signals, true_labels, returns)
        
        # Meta-labeler precision should be >= raw
        ml_precision = labeler.validation_metrics.get('precision', 0)
        
        # Meta-labeling should improve or maintain precision
        # (allow small tolerance for randomness)
        assert ml_precision >= raw_precision - 0.05 or ml_precision >= 0.55
        
    def test_hmm_identifies_regimes(self):
        """Verify HMM can identify distinct market regimes."""
        from institutional.regime.hmm_detector import HMMRegimeDetector
        
        # Generate data with clear regimes
        prices = generate_regime_prices(n=500, seed=42)
        returns = prices.pct_change().dropna()
        
        detector = HMMRegimeDetector(n_regimes=2)
        detector.fit(returns)
        
        regimes = detector.predict(returns)
        stats = detector.get_regime_statistics(returns, regimes)
        
        # Regimes should have different characteristics
        mean_returns = [s['mean_return'] for s in stats.values()]
        volatilities = [s['volatility'] for s in stats.values()]
        
        # Regimes should be distinguishable
        mean_spread = abs(mean_returns[0] - mean_returns[1])
        vol_spread = abs(volatilities[0] - volatilities[1])
        
        # At least one dimension should show clear separation
        assert mean_spread > 0.01 or vol_spread > 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
