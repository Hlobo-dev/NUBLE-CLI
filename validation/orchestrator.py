"""
COMPLETE VALIDATION ORCHESTRATOR
=================================
This orchestrates the entire validation pipeline:

1. Data Acquisition (Polygon.io)
2. Lookahead Bias Audit
3. Walk-Forward Validation with Purging
4. CPCV with PBO and Deflated Sharpe
5. Final OOS Test (only once at the end)

Expected Results:
- Realistic Sharpe: 1.0 - 1.5
- PBO < 0.30
- DSR significant at 95%
- Consistent performance across splits

If Sharpe > 3.0 is detected, this indicates bugs.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from pathlib import Path
import json
import logging
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import CONFIG
from .data_downloader import PolygonDataDownloader
from .lookahead_audit import LookaheadAudit
from .walk_forward import WalkForwardValidator, WalkForwardBacktest, PerformanceMetrics
from .cpcv import CombinatorialPurgedCV, CPCVBacktest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationOrchestrator:
    """
    Complete validation orchestration.
    
    This runs the full validation pipeline ensuring
    institutional-grade rigor.
    """
    
    def __init__(
        self,
        feature_pipeline: Callable,
        label_pipeline: Callable,
        model_factory: Callable
    ):
        """
        Parameters:
        -----------
        feature_pipeline : Callable
            Function(prices) -> features DataFrame
        label_pipeline : Callable
            Function(prices) -> labels Series
        model_factory : Callable
            Function() -> model instance with fit/predict
        """
        self.feature_pipeline = feature_pipeline
        self.label_pipeline = label_pipeline
        self.model_factory = model_factory
        
        self.metrics = PerformanceMetrics(
            transaction_cost=CONFIG.transaction_cost,
            risk_free_rate=CONFIG.risk_free_rate
        )
        
        self.results = {}
    
    def download_data(self, force: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Step 1: Download data from Polygon.io.
        
        Parameters:
        -----------
        force : bool
            If True, re-download even if files exist
        
        Returns:
        --------
        Dict of symbol -> DataFrame
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 1: DATA ACQUISITION")
        logger.info("="*60)
        
        # Check if data already exists
        existing_files = list(CONFIG.historical_dir.glob("*.csv"))
        
        if existing_files and not force:
            logger.info(f"Found {len(existing_files)} existing data files")
            logger.info("Use force=True to re-download")
            
            # Load existing data
            data = {}
            for filepath in existing_files:
                symbol = filepath.stem
                data[symbol] = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            return data
        
        # Download from Polygon
        downloader = PolygonDataDownloader()
        data = downloader.download_all_symbols()
        
        # Split into train/test
        train_data, test_data = downloader.split_train_test(data)
        
        logger.info(f"\n✓ Downloaded {len(data)} symbols")
        logger.info(f"✓ Train: {len(train_data)} symbols ({CONFIG.train_start} to {CONFIG.train_end})")
        logger.info(f"✓ Test: {len(test_data)} symbols ({CONFIG.test_start} to {CONFIG.test_end})")
        logger.info("\n⚠️  TEST DATA MUST NOT BE TOUCHED UNTIL FINAL VALIDATION!")
        
        self.results["data_acquisition"] = {
            "n_symbols": len(data),
            "train_symbols": len(train_data),
            "test_symbols": len(test_data),
            "timestamp": datetime.now().isoformat()
        }
        
        return data
    
    def run_lookahead_audit(
        self,
        symbol: str = "SPY"
    ) -> Dict[str, Any]:
        """
        Step 2: Audit for lookahead bias.
        
        Parameters:
        -----------
        symbol : str
            Symbol to audit (should be representative)
        
        Returns:
        --------
        Dict with audit results
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 2: LOOKAHEAD BIAS AUDIT")
        logger.info("="*60)
        
        # Load training data only
        filepath = CONFIG.train_dir / f"{symbol}.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Training data not found: {filepath}")
        
        prices = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Auditing with {symbol}: {len(prices)} rows")
        
        # Create auditor
        auditor = LookaheadAudit()
        
        # Create pipeline components for auditing
        try:
            from src.institutional.labeling.triple_barrier import TripleBarrierLabeler
            from src.institutional.features.frac_diff import FractionalDifferentiator
            
            labeler = TripleBarrierLabeler()
            frac_differ = FractionalDifferentiator()
            
            results = auditor.run_full_audit(
                prices=prices,
                labeler=labeler,
                frac_differ=frac_differ
            )
            
        except ImportError as e:
            logger.warning(f"Could not import components: {e}")
            results = auditor.run_full_audit(prices=prices)
        
        self.results["lookahead_audit"] = results
        
        if not results["overall_passed"]:
            logger.error("\n⛔ LOOKAHEAD BIAS DETECTED!")
            logger.error("   Fix issues before continuing validation.")
            raise ValueError("Lookahead bias detected - cannot continue")
        
        return results
    
    def run_walk_forward(
        self,
        symbols: List[str] = None,
        max_symbols: int = 5
    ) -> Dict[str, Any]:
        """
        Step 3: Walk-forward validation with purging.
        
        Parameters:
        -----------
        symbols : List[str]
            Symbols to validate (defaults to first max_symbols from CONFIG)
        max_symbols : int
            Maximum symbols to validate (for speed)
        
        Returns:
        --------
        Dict with walk-forward results per symbol
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 3: WALK-FORWARD VALIDATION")
        logger.info("="*60)
        
        if symbols is None:
            symbols = CONFIG.symbols[:max_symbols]
        
        validator = WalkForwardValidator(
            train_size=CONFIG.walk_forward.train_size,
            test_size=CONFIG.walk_forward.test_size,
            purge_size=CONFIG.walk_forward.purge_size,
            embargo_size=CONFIG.walk_forward.embargo_size
        )
        
        all_results = {}
        all_sharpes = []
        
        for symbol in symbols:
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing {symbol}")
            logger.info(f"{'='*40}")
            
            try:
                # Load data
                filepath = CONFIG.train_dir / f"{symbol}.csv"
                if not filepath.exists():
                    logger.warning(f"Data not found for {symbol}")
                    continue
                
                prices = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                # Generate features and labels using provided pipelines
                X = self.feature_pipeline(prices)
                y = self.label_pipeline(prices)
                
                # Align data
                common_idx = X.index.intersection(y.index).intersection(prices.index)
                X = X.loc[common_idx]
                y = y.loc[common_idx]
                price_series = prices.loc[common_idx, 'close']
                
                if len(X) < 500:
                    logger.warning(f"{symbol}: Only {len(X)} rows after alignment - skipping")
                    continue
                
                # Run walk-forward
                backtest = WalkForwardBacktest(
                    model_factory=self.model_factory,
                    validator=validator,
                    metrics=self.metrics
                )
                
                result = backtest.run(X, y, price_series)
                all_results[symbol] = result
                all_sharpes.append(result["aggregate_metrics"]["sharpe_ratio"])
                
            except Exception as e:
                logger.error(f"{symbol}: Error - {e}")
                continue
        
        # Aggregate
        aggregate = {
            "n_symbols": len(all_results),
            "sharpe_mean": np.mean(all_sharpes) if all_sharpes else 0,
            "sharpe_std": np.std(all_sharpes) if all_sharpes else 0,
            "sharpe_min": np.min(all_sharpes) if all_sharpes else 0,
            "sharpe_max": np.max(all_sharpes) if all_sharpes else 0,
            "symbol_results": all_results
        }
        
        self.results["walk_forward"] = aggregate
        
        logger.info("\n" + "-"*60)
        logger.info("WALK-FORWARD AGGREGATE RESULTS")
        logger.info("-"*60)
        logger.info(f"Symbols validated: {aggregate['n_symbols']}")
        logger.info(f"Sharpe Mean: {aggregate['sharpe_mean']:.2f} ± {aggregate['sharpe_std']:.2f}")
        logger.info(f"Sharpe Range: [{aggregate['sharpe_min']:.2f}, {aggregate['sharpe_max']:.2f}]")
        
        if aggregate['sharpe_mean'] > 3.0:
            logger.warning("\n⚠️  SHARPE > 3.0 DETECTED - LIKELY BUGS!")
        
        return aggregate
    
    def run_cpcv(
        self,
        symbols: List[str] = None,
        max_symbols: int = 3
    ) -> Dict[str, Any]:
        """
        Step 4: CPCV with PBO and Deflated Sharpe.
        
        Parameters:
        -----------
        symbols : List[str]
            Symbols to validate
        max_symbols : int
            Maximum symbols (CPCV is computationally expensive)
        
        Returns:
        --------
        Dict with CPCV results
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 4: COMBINATORIAL PURGED CROSS-VALIDATION")
        logger.info("="*60)
        
        if symbols is None:
            symbols = CONFIG.symbols[:max_symbols]
        
        cpcv = CombinatorialPurgedCV(
            n_splits=CONFIG.cpcv.n_splits,
            n_test_groups=CONFIG.cpcv.n_test_groups,
            purge_size=CONFIG.cpcv.purge_size,
            embargo_size=CONFIG.cpcv.embargo_size
        )
        
        all_results = {}
        all_pbo = []
        all_dsr = []
        
        for symbol in symbols:
            logger.info(f"\n{'='*40}")
            logger.info(f"CPCV: {symbol}")
            logger.info(f"{'='*40}")
            
            try:
                # Load data
                filepath = CONFIG.train_dir / f"{symbol}.csv"
                if not filepath.exists():
                    continue
                
                prices = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                # Generate features and labels
                X = self.feature_pipeline(prices)
                y = self.label_pipeline(prices)
                
                # Align
                common_idx = X.index.intersection(y.index).intersection(prices.index)
                X = X.loc[common_idx]
                y = y.loc[common_idx]
                price_series = prices.loc[common_idx, 'close']
                
                if len(X) < 500:
                    logger.warning(f"{symbol}: Insufficient data for CPCV")
                    continue
                
                # Run CPCV
                backtest = CPCVBacktest(
                    model_factory=self.model_factory,
                    cpcv=cpcv,
                    transaction_cost=CONFIG.transaction_cost
                )
                
                result = backtest.run(X, y, price_series)
                all_results[symbol] = result
                all_pbo.append(result["pbo"]["pbo"])
                all_dsr.append(result["dsr"]["dsr"])
                
            except Exception as e:
                logger.error(f"{symbol}: Error - {e}")
                continue
        
        aggregate = {
            "n_symbols": len(all_results),
            "pbo_mean": np.mean(all_pbo) if all_pbo else 0,
            "pbo_max": np.max(all_pbo) if all_pbo else 0,
            "dsr_mean": np.mean(all_dsr) if all_dsr else 0,
            "symbol_results": all_results
        }
        
        self.results["cpcv"] = aggregate
        
        logger.info("\n" + "-"*60)
        logger.info("CPCV AGGREGATE RESULTS")
        logger.info("-"*60)
        logger.info(f"Symbols validated: {aggregate['n_symbols']}")
        logger.info(f"PBO Mean: {aggregate['pbo_mean']:.1%}")
        logger.info(f"DSR Mean: {aggregate['dsr_mean']:.1%}")
        
        if aggregate['pbo_mean'] > 0.3:
            logger.warning("\n⚠️  PBO > 30% - POSSIBLE OVERFITTING!")
        
        return aggregate
    
    def run_final_oos_test(
        self,
        symbols: List[str] = None
    ) -> Dict[str, Any]:
        """
        Step 5: Final Out-of-Sample Test (2023-2026).
        
        ⚠️  WARNING: This uses test data that was never seen before!
        ⚠️  Only run this ONCE at the very end of development!
        
        Parameters:
        -----------
        symbols : List[str]
            Symbols to test
        
        Returns:
        --------
        Dict with final OOS results
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 5: FINAL OUT-OF-SAMPLE TEST")
        logger.info("="*60)
        logger.warning("\n⚠️  THIS USES PREVIOUSLY UNSEEN TEST DATA!")
        logger.warning("⚠️  DO NOT RE-RUN OR ITERATE BASED ON THESE RESULTS!")
        
        if symbols is None:
            symbols = CONFIG.symbols
        
        all_results = {}
        all_sharpes = []
        all_returns = []
        
        for symbol in symbols:
            try:
                # Load training data to fit model
                train_filepath = CONFIG.train_dir / f"{symbol}.csv"
                test_filepath = CONFIG.test_dir / f"{symbol}.csv"
                
                if not train_filepath.exists() or not test_filepath.exists():
                    continue
                
                train_prices = pd.read_csv(train_filepath, index_col=0, parse_dates=True)
                test_prices = pd.read_csv(test_filepath, index_col=0, parse_dates=True)
                
                # Generate features and labels for training
                X_train = self.feature_pipeline(train_prices)
                y_train = self.label_pipeline(train_prices)
                
                # Align training data
                common_idx = X_train.index.intersection(y_train.index)
                X_train = X_train.loc[common_idx]
                y_train = y_train.loc[common_idx]
                
                if len(X_train) < 500:
                    continue
                
                # Train model on ALL training data
                model = self.model_factory()
                model.fit(X_train, y_train)
                
                # Generate features for test data
                X_test = self.feature_pipeline(test_prices)
                
                # Align test data
                common_idx = X_test.index.intersection(test_prices.index)
                X_test = X_test.loc[common_idx]
                price_series = test_prices.loc[common_idx, 'close']
                
                if len(X_test) < 50:
                    continue
                
                # Predict on test data
                predictions = model.predict(X_test)
                signals = pd.Series(predictions, index=X_test.index)
                
                # Calculate returns
                returns = self.metrics.calculate_returns(signals, price_series)
                metrics = self.metrics.calculate_all(returns)
                
                all_results[symbol] = metrics
                all_sharpes.append(metrics["sharpe_ratio"])
                all_returns.append(returns)
                
                logger.info(f"{symbol}: Sharpe={metrics['sharpe_ratio']:.2f}, "
                           f"Return={metrics['total_return']:.1%}")
                
            except Exception as e:
                logger.error(f"{symbol}: Error - {e}")
                continue
        
        # Aggregate
        if all_returns:
            combined_returns = pd.concat(all_returns)
            aggregate_metrics = self.metrics.calculate_all(combined_returns)
        else:
            aggregate_metrics = {}
        
        aggregate = {
            "n_symbols": len(all_results),
            "aggregate_metrics": aggregate_metrics,
            "sharpe_mean": np.mean(all_sharpes) if all_sharpes else 0,
            "sharpe_std": np.std(all_sharpes) if all_sharpes else 0,
            "symbol_results": all_results,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results["final_oos"] = aggregate
        
        logger.info("\n" + "="*60)
        logger.info("FINAL OUT-OF-SAMPLE RESULTS")
        logger.info("="*60)
        logger.info(f"Symbols tested: {aggregate['n_symbols']}")
        logger.info(f"Aggregate Sharpe: {aggregate_metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Aggregate Return: {aggregate_metrics.get('total_return', 0):.1%}")
        logger.info(f"Sharpe Mean: {aggregate['sharpe_mean']:.2f} ± {aggregate['sharpe_std']:.2f}")
        
        # Final assessment
        oos_sharpe = aggregate_metrics.get('sharpe_ratio', 0)
        
        if oos_sharpe > 3.0:
            logger.error("\n⛔ SHARPE > 3.0 IN OOS TEST - THIS IS UNREALISTIC!")
            logger.error("   There are likely bugs in the pipeline.")
        elif oos_sharpe > 2.0:
            logger.warning("\n⚠️  SHARPE > 2.0 - Review carefully")
            logger.warning("   Only elite quant funds achieve this.")
        elif oos_sharpe > 1.0:
            logger.info("\n✓ SHARPE IN REALISTIC RANGE (1.0-2.0)")
            logger.info("   This is consistent with good institutional strategies.")
        elif oos_sharpe > 0.5:
            logger.info("\n⚠️  SHARPE 0.5-1.0 - Marginal strategy")
            logger.info("   May not be profitable after all costs.")
        else:
            logger.warning("\n⚠️  SHARPE < 0.5 - Strategy may not be viable")
        
        return aggregate
    
    def run_complete_validation(
        self,
        download_data: bool = True,
        run_audit: bool = True,
        run_walk_forward: bool = True,
        run_cpcv: bool = True,
        run_final_test: bool = False  # Default False for safety
    ) -> Dict[str, Any]:
        """
        Run the complete validation pipeline.
        
        Parameters:
        -----------
        download_data : bool
            Whether to download data
        run_audit : bool
            Whether to run lookahead audit
        run_walk_forward : bool
            Whether to run walk-forward validation
        run_cpcv : bool
            Whether to run CPCV
        run_final_test : bool
            Whether to run final OOS test (defaults to False for safety)
        
        Returns:
        --------
        Dict with all validation results
        """
        logger.info("\n" + "="*70)
        logger.info("  COMPLETE VALIDATION PIPELINE  ")
        logger.info("  Following Lopez de Prado AFML Methodology  ")
        logger.info("="*70)
        
        start_time = datetime.now()
        
        # Step 1: Data
        if download_data:
            self.download_data()
        
        # Step 2: Audit
        if run_audit:
            self.run_lookahead_audit()
        
        # Step 3: Walk-Forward
        if run_walk_forward:
            self.run_walk_forward()
        
        # Step 4: CPCV
        if run_cpcv:
            self.run_cpcv()
        
        # Step 5: Final OOS (only if explicitly requested)
        if run_final_test:
            self.run_final_oos_test()
        else:
            logger.info("\n" + "-"*60)
            logger.info("ℹ️  Final OOS test skipped (run_final_test=False)")
            logger.info("   Only run final test ONCE at the very end!")
        
        # Summary
        duration = datetime.now() - start_time
        
        self.results["summary"] = {
            "duration_seconds": duration.total_seconds(),
            "timestamp": datetime.now().isoformat(),
            "steps_completed": {
                "data_acquisition": download_data,
                "lookahead_audit": run_audit,
                "walk_forward": run_walk_forward,
                "cpcv": run_cpcv,
                "final_oos": run_final_test
            }
        }
        
        # Save results
        results_path = CONFIG.results_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert non-serializable objects
        serializable_results = self._make_serializable(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"\n✓ Results saved to: {results_path}")
        logger.info(f"✓ Total duration: {duration}")
        
        return self.results
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj


def create_simple_feature_pipeline():
    """
    Create a simple feature pipeline for testing.
    
    This is a placeholder - replace with Phase 1+2 pipeline.
    """
    def feature_pipeline(prices: pd.DataFrame) -> pd.DataFrame:
        """Generate simple technical features."""
        close = prices['close'] if 'close' in prices.columns else prices.iloc[:, 0]
        
        features = pd.DataFrame(index=prices.index)
        
        # Returns
        features['ret_1d'] = close.pct_change(1)
        features['ret_5d'] = close.pct_change(5)
        features['ret_20d'] = close.pct_change(20)
        
        # Volatility
        features['vol_20d'] = close.pct_change().rolling(20).std()
        
        # Momentum
        features['mom_20d'] = close / close.shift(20) - 1
        
        # Mean reversion
        features['mean_rev'] = close / close.rolling(20).mean() - 1
        
        return features.dropna()
    
    return feature_pipeline


def create_simple_label_pipeline():
    """
    Create a simple label pipeline for testing.
    
    This is a placeholder - replace with Triple Barrier labels.
    """
    def label_pipeline(prices: pd.DataFrame) -> pd.Series:
        """Generate simple forward return labels."""
        close = prices['close'] if 'close' in prices.columns else prices.iloc[:, 0]
        
        # Forward 5-day return
        fwd_ret = close.shift(-5) / close - 1
        
        # Classify
        labels = pd.Series(0, index=prices.index)
        labels[fwd_ret > 0.01] = 1   # Long
        labels[fwd_ret < -0.01] = -1  # Short
        
        return labels.dropna()
    
    return label_pipeline


def create_simple_model_factory():
    """
    Create a simple model factory for testing.
    
    This is a placeholder - replace with Meta-Labeler + HMM model.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    def model_factory():
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    
    return model_factory


if __name__ == "__main__":
    logger.info("Validation Orchestrator")
    logger.info("="*60)
    logger.info("\nTo run validation:")
    logger.info("  from validation.orchestrator import ValidationOrchestrator")
    logger.info("  orchestrator = ValidationOrchestrator(features, labels, model)")
    logger.info("  results = orchestrator.run_complete_validation()")
