"""
PHASE 1+2 PIPELINE INTEGRATION WITH VALIDATION
================================================
This integrates the Triple Barrier, Fractional Diff, HMM, and Meta-Labeler
with the rigorous validation framework.

This is the COMPLETE validation script that will:
1. Download data from Polygon.io (your paid subscription)
2. Audit for lookahead bias
3. Run walk-forward validation
4. Run CPCV with PBO
5. Report realistic Sharpe (expected: 1.0-1.5)
"""

import sys
from pathlib import Path

# Add paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

import pandas as pd
import numpy as np
from typing import Dict, Any, Callable
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_phase12_feature_pipeline() -> Callable:
    """
    Create feature pipeline using Phase 1+2 components:
    - Fractional Differentiation for stationarity with memory
    - HMM Regime Detection for regime features
    """
    from src.institutional.features.frac_diff import FractionalDifferentiator
    from src.institutional.regime.hmm_detector import HMMRegimeDetector
    
    frac_differ = None
    hmm_detector = None
    
    def feature_pipeline(prices: pd.DataFrame) -> pd.DataFrame:
        nonlocal frac_differ, hmm_detector
        
        # Get close price
        close = prices['close'] if 'close' in prices.columns else prices.iloc[:, 0]
        
        # Initialize components if needed
        if frac_differ is None:
            frac_differ = FractionalDifferentiator(
                d=0.4,  # Fractional order
                threshold=1e-5
            )
        
        if hmm_detector is None:
            hmm_detector = HMMRegimeDetector(n_regimes=2)
        
        features = pd.DataFrame(index=prices.index)
        
        # 1. Fractionally Differenced Features
        try:
            price_df = prices[['close', 'high', 'low', 'volume']].copy()
            frac_features = frac_differ.fit_transform(price_df)
            
            if isinstance(frac_features, pd.DataFrame):
                for col in frac_features.columns:
                    features[f'frac_{col}'] = frac_features[col]
            else:
                features['frac_close'] = frac_features
        except Exception as e:
            logger.warning(f"Fractional diff failed: {e}")
            # Fallback: simple returns
            features['frac_close'] = close.pct_change()
        
        # 2. HMM Regime Features
        try:
            returns = close.pct_change().dropna()
            hmm_detector.fit(returns)
            regimes = hmm_detector.predict(returns)
            
            if hasattr(regimes, 'regime'):
                features['regime'] = regimes['regime']
            elif isinstance(regimes, pd.Series):
                features['regime'] = regimes
            else:
                features['regime'] = 0
        except Exception as e:
            logger.warning(f"HMM failed: {e}")
            features['regime'] = 0
        
        # 3. Technical Features (no lookahead)
        features['ret_1d'] = close.pct_change(1)
        features['ret_5d'] = close.pct_change(5)
        features['ret_20d'] = close.pct_change(20)
        
        # Rolling volatility (uses only past data)
        features['vol_20d'] = close.pct_change().rolling(20).std()
        features['vol_60d'] = close.pct_change().rolling(60).std()
        
        # Momentum
        features['mom_10d'] = close / close.shift(10) - 1
        features['mom_20d'] = close / close.shift(20) - 1
        features['mom_60d'] = close / close.shift(60) - 1
        
        # Mean reversion
        features['sma_ratio_20'] = close / close.rolling(20).mean() - 1
        features['sma_ratio_50'] = close / close.rolling(50).mean() - 1
        
        # RSI (14-day)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Band position
        ma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features['bb_position'] = (close - ma_20) / (2 * std_20 + 1e-10)
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        # Volume features
        if 'volume' in prices.columns:
            volume = prices['volume']
            features['vol_ratio'] = volume / volume.rolling(20).mean()
        
        # Drop NaN
        return features.dropna()
    
    return feature_pipeline


def create_phase12_label_pipeline() -> Callable:
    """
    Create label pipeline using Triple Barrier Method.
    
    This generates realistic trading labels based on:
    - Profit taking barrier (upper)
    - Stop loss barrier (lower)
    - Maximum holding period (vertical)
    """
    from src.institutional.labeling.triple_barrier import TripleBarrierLabeler
    
    labeler = None
    
    def label_pipeline(prices: pd.DataFrame) -> pd.Series:
        nonlocal labeler
        
        close = prices['close'] if 'close' in prices.columns else prices.iloc[:, 0]
        
        if labeler is None:
            labeler = TripleBarrierLabeler(
                pt_sl=(1.0, 1.0),  # Symmetric barriers
                max_holding_period=20,  # 20 days max
                vol_lookback=20,
                min_return=0.0
            )
        
        try:
            events = labeler.fit_transform(close)
            
            # Extract labels
            if hasattr(events, 'label'):
                labels = events['label']
            elif 'label' in events.columns:
                labels = events['label']
            elif 'bin' in events.columns:
                labels = events['bin']
            else:
                # Fallback
                labels = pd.Series(0, index=close.index)
        except Exception as e:
            logger.warning(f"Triple Barrier failed: {e}")
            # Fallback to simple forward returns
            fwd_ret = close.shift(-5) / close - 1
            labels = pd.Series(0, index=close.index)
            labels[fwd_ret > 0.01] = 1
            labels[fwd_ret < -0.01] = -1
        
        return labels.dropna()
    
    return label_pipeline


def create_phase12_model_factory() -> Callable:
    """
    Create model factory using Meta-Labeler.
    
    This is a two-stage model:
    1. Primary model generates direction signals
    2. Meta-model learns when primary is correct (bet sizing)
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    
    def model_factory():
        """Return a new model instance."""
        try:
            from src.institutional.models.meta.meta_labeler import MetaLabeler
            
            # Primary model for direction
            primary = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Meta model for bet sizing
            meta = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            
            return MetaLabeler(
                primary_model=primary,
                meta_model=meta
            )
        except ImportError:
            logger.warning("MetaLabeler not available, using RandomForest")
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
    
    return model_factory


def run_complete_validation():
    """
    Run the complete validation with Phase 1+2 integration.
    """
    from validation.orchestrator import ValidationOrchestrator
    
    logger.info("\n" + "="*70)
    logger.info("  INSTITUTIONAL ML VALIDATION  ")
    logger.info("  Phase 1+2 Integration with Rigorous Testing  ")
    logger.info("="*70)
    
    # Create pipelines
    logger.info("\nInitializing Phase 1+2 pipelines...")
    
    try:
        feature_pipeline = create_phase12_feature_pipeline()
        logger.info("✓ Feature pipeline (Frac Diff + HMM) ready")
    except ImportError as e:
        logger.warning(f"Phase 1+2 features unavailable: {e}")
        from validation.orchestrator import create_simple_feature_pipeline
        feature_pipeline = create_simple_feature_pipeline()
        logger.info("⚠ Using simple feature pipeline")
    
    try:
        label_pipeline = create_phase12_label_pipeline()
        logger.info("✓ Label pipeline (Triple Barrier) ready")
    except ImportError as e:
        logger.warning(f"Triple Barrier unavailable: {e}")
        from validation.orchestrator import create_simple_label_pipeline
        label_pipeline = create_simple_label_pipeline()
        logger.info("⚠ Using simple label pipeline")
    
    try:
        model_factory = create_phase12_model_factory()
        logger.info("✓ Model factory (Meta-Labeler) ready")
    except ImportError as e:
        logger.warning(f"MetaLabeler unavailable: {e}")
        from validation.orchestrator import create_simple_model_factory
        model_factory = create_simple_model_factory()
        logger.info("⚠ Using simple model factory")
    
    # Create orchestrator
    orchestrator = ValidationOrchestrator(
        feature_pipeline=feature_pipeline,
        label_pipeline=label_pipeline,
        model_factory=model_factory
    )
    
    # Run validation
    logger.info("\n" + "-"*70)
    logger.info("Starting validation pipeline...")
    logger.info("-"*70)
    
    results = orchestrator.run_complete_validation(
        download_data=True,      # Download from Polygon.io
        run_audit=True,          # Check for lookahead bias
        run_walk_forward=True,   # Walk-forward validation
        run_cpcv=True,           # CPCV with PBO
        run_final_test=False     # Don't touch test data yet!
    )
    
    # Print summary
    print_summary(results)
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print validation summary."""
    logger.info("\n" + "="*70)
    logger.info("  VALIDATION SUMMARY  ")
    logger.info("="*70)
    
    # Walk-forward results
    if "walk_forward" in results:
        wf = results["walk_forward"]
        logger.info("\nWalk-Forward Validation:")
        logger.info(f"  Symbols: {wf['n_symbols']}")
        logger.info(f"  Sharpe: {wf['sharpe_mean']:.2f} ± {wf['sharpe_std']:.2f}")
        logger.info(f"  Range: [{wf['sharpe_min']:.2f}, {wf['sharpe_max']:.2f}]")
    
    # CPCV results
    if "cpcv" in results:
        cpcv = results["cpcv"]
        logger.info("\nCPCV Validation:")
        logger.info(f"  Symbols: {cpcv['n_symbols']}")
        logger.info(f"  PBO Mean: {cpcv['pbo_mean']:.1%}")
        logger.info(f"  DSR Mean: {cpcv['dsr_mean']:.1%}")
    
    # Assessment
    logger.info("\n" + "-"*70)
    
    issues = []
    
    if "walk_forward" in results:
        if results["walk_forward"]["sharpe_mean"] > 3.0:
            issues.append("Sharpe > 3.0 detected - likely bugs")
    
    if "cpcv" in results:
        if results["cpcv"]["pbo_mean"] > 0.3:
            issues.append("PBO > 30% - possible overfitting")
    
    if issues:
        logger.warning("\n⚠️  ISSUES DETECTED:")
        for issue in issues:
            logger.warning(f"   - {issue}")
    else:
        logger.info("\n✓ Strategy appears to have realistic performance")
    
    logger.info("\n" + "="*70)
    logger.info("  EXPECTED REALISTIC SHARPE: 1.0 - 1.5  ")
    logger.info("  Renaissance Medallion (best ever): 2-3  ")
    logger.info("  Sharpe > 3.0 = BUGS  ")
    logger.info("="*70)


def run_quick_test():
    """
    Quick test with just one symbol.
    
    Use this to verify the pipeline works before full validation.
    """
    from validation.orchestrator import ValidationOrchestrator
    from validation.config import CONFIG
    
    logger.info("\n" + "="*60)
    logger.info("QUICK VALIDATION TEST")
    logger.info("="*60)
    
    # Use simple pipelines for quick test
    from validation.orchestrator import (
        create_simple_feature_pipeline,
        create_simple_label_pipeline,
        create_simple_model_factory
    )
    
    # Test with just SPY
    CONFIG.symbols = ["SPY"]
    
    orchestrator = ValidationOrchestrator(
        feature_pipeline=create_simple_feature_pipeline(),
        label_pipeline=create_simple_label_pipeline(),
        model_factory=create_simple_model_factory()
    )
    
    results = orchestrator.run_complete_validation(
        download_data=True,
        run_audit=False,
        run_walk_forward=True,
        run_cpcv=False,
        run_final_test=False
    )
    
    logger.info("\n✓ Quick test completed")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run validation")
    parser.add_argument("--quick", action="store_true", help="Quick test with one symbol")
    parser.add_argument("--full", action="store_true", help="Full validation")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        run_complete_validation()
