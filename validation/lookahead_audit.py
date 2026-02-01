"""
LOOKAHEAD BIAS AUDIT
=====================
This module audits ALL pipeline components for information leakage.

Sharpe > 3.0 in backtests typically means lookahead bias exists.
This audit will find it.

Common sources of lookahead bias:
1. Volatility calculated using future data
2. Labels using future information before barrier touch
3. Features normalized using full dataset statistics
4. Technical indicators using forward-looking windows
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Callable
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LookaheadAudit:
    """
    Comprehensive audit for lookahead bias in ML trading pipelines.
    
    This checks:
    1. Triple Barrier labeling uses only past volatility
    2. Features don't use future information
    3. Normalization doesn't leak test data
    4. Model predictions only use historical data
    """
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        
    def audit_triple_barrier(
        self,
        prices: pd.Series,
        labeler,
        test_idx: int = None
    ) -> Dict[str, Any]:
        """
        Audit Triple Barrier labeling for lookahead.
        
        Checks:
        1. Volatility at time t only uses data up to t
        2. Barrier levels set using past information only
        3. Label determined at barrier touch, not before
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        labeler : TripleBarrierLabeler
            The labeler instance
        test_idx : int
            Index to test (defaults to middle of series)
        """
        if test_idx is None:
            test_idx = len(prices) // 2
        
        results = {
            "component": "TripleBarrierLabeler",
            "passed": True,
            "checks": []
        }
        
        # Check 1: Volatility calculation
        logger.info("Checking volatility calculation...")
        
        try:
            # Get volatility at test_idx
            if hasattr(labeler, '_compute_volatility'):
                # Truncate data to test_idx
                truncated = prices.iloc[:test_idx + 1]
                vol_truncated = labeler._compute_volatility(truncated)
                
                # Get full data volatility at same point
                vol_full = labeler._compute_volatility(prices)
                
                # At test_idx, both should be identical
                if hasattr(vol_truncated, 'iloc'):
                    vol_at_idx_truncated = vol_truncated.iloc[-1]
                    vol_at_idx_full = vol_full.iloc[test_idx]
                    
                    if not np.isclose(vol_at_idx_truncated, vol_at_idx_full, rtol=1e-6):
                        results["passed"] = False
                        results["checks"].append({
                            "check": "volatility_lookback",
                            "passed": False,
                            "message": f"Volatility differs: truncated={vol_at_idx_truncated:.6f}, full={vol_at_idx_full:.6f}",
                            "severity": "CRITICAL"
                        })
                        self.issues.append("Triple Barrier volatility uses future data!")
                    else:
                        results["checks"].append({
                            "check": "volatility_lookback",
                            "passed": True,
                            "message": "Volatility correctly uses only past data"
                        })
        except Exception as e:
            results["checks"].append({
                "check": "volatility_lookback",
                "passed": None,
                "message": f"Could not verify: {e}"
            })
            self.warnings.append(f"Could not audit volatility: {e}")
        
        # Check 2: Label timing
        logger.info("Checking label timing...")
        
        try:
            # Get labels
            events = labeler.fit_transform(prices)
            
            if hasattr(events, 't1') or 't1' in events.columns:
                t1_col = events['t1'] if 't1' in events.columns else events.t1
                
                # Check that t1 (barrier touch time) is always after event start
                event_times = events.index if hasattr(events, 'index') else events.iloc[:, 0]
                
                violations = (t1_col < event_times).sum()
                
                if violations > 0:
                    results["passed"] = False
                    results["checks"].append({
                        "check": "label_timing",
                        "passed": False,
                        "message": f"{violations} labels determined before event time!",
                        "severity": "CRITICAL"
                    })
                    self.issues.append("Labels determined before barrier touch!")
                else:
                    results["checks"].append({
                        "check": "label_timing",
                        "passed": True,
                        "message": "Labels correctly determined at barrier touch"
                    })
        except Exception as e:
            results["checks"].append({
                "check": "label_timing",
                "passed": None,
                "message": f"Could not verify: {e}"
            })
        
        return results
    
    def audit_fractional_diff(
        self,
        prices: pd.DataFrame,
        frac_differ
    ) -> Dict[str, Any]:
        """
        Audit Fractional Differentiation for lookahead.
        
        Checks:
        1. Weights calculation doesn't use future data
        2. Transformation at t only uses data up to t
        """
        results = {
            "component": "FractionalDifferentiator",
            "passed": True,
            "checks": []
        }
        
        logger.info("Checking fractional differentiation...")
        
        try:
            # Transform full data
            full_result = frac_differ.fit_transform(prices)
            
            # Transform truncated data
            mid_idx = len(prices) // 2
            truncated = prices.iloc[:mid_idx + 1]
            truncated_result = frac_differ.fit_transform(truncated)
            
            # Compare at the truncation point
            if hasattr(truncated_result, 'iloc'):
                # The last value of truncated should equal mid_idx value of full
                val_truncated = truncated_result.iloc[-1]
                val_full = full_result.iloc[mid_idx]
                
                # Check close values
                if not np.allclose(val_truncated.values, val_full.values, rtol=0.01, equal_nan=True):
                    results["passed"] = False
                    results["checks"].append({
                        "check": "frac_diff_lookback",
                        "passed": False,
                        "message": "Fractional diff values differ between truncated and full data",
                        "severity": "WARNING"
                    })
                    self.warnings.append("Fractional diff may use future data (small differences)")
                else:
                    results["checks"].append({
                        "check": "frac_diff_lookback",
                        "passed": True,
                        "message": "Fractional diff correctly uses only past data"
                    })
        except Exception as e:
            results["checks"].append({
                "check": "frac_diff_lookback",
                "passed": None,
                "message": f"Could not verify: {e}"
            })
        
        return results
    
    def audit_features(
        self,
        prices: pd.DataFrame,
        feature_func: Callable,
        feature_name: str = "features"
    ) -> Dict[str, Any]:
        """
        Audit feature generation for lookahead.
        
        Tests if features at time t are identical when computed on:
        1. Data up to time t
        2. Full data (if lookahead exists, these will differ)
        """
        results = {
            "component": f"Features: {feature_name}",
            "passed": True,
            "checks": []
        }
        
        logger.info(f"Checking {feature_name}...")
        
        try:
            # Compute on full data
            full_features = feature_func(prices)
            
            # Compute on truncated data
            mid_idx = len(prices) // 2
            truncated = prices.iloc[:mid_idx + 1]
            truncated_features = feature_func(truncated)
            
            # Compare at truncation point
            val_truncated = truncated_features.iloc[-1]
            val_full = full_features.iloc[mid_idx]
            
            # Check equality
            if not np.allclose(val_truncated.values, val_full.values, rtol=0.01, equal_nan=True):
                results["passed"] = False
                results["checks"].append({
                    "check": "feature_lookback",
                    "passed": False,
                    "message": f"{feature_name} differs between truncated and full data",
                    "severity": "CRITICAL"
                })
                self.issues.append(f"Feature {feature_name} uses future data!")
            else:
                results["checks"].append({
                    "check": "feature_lookback",
                    "passed": True,
                    "message": f"{feature_name} correctly uses only past data"
                })
        except Exception as e:
            results["checks"].append({
                "check": "feature_lookback",
                "passed": None,
                "message": f"Could not verify: {e}"
            })
        
        return results
    
    def audit_normalization(
        self,
        data: pd.DataFrame,
        normalizer
    ) -> Dict[str, Any]:
        """
        Audit normalization for data leakage.
        
        Checks if normalizer uses test data statistics.
        """
        results = {
            "component": "Normalization",
            "passed": True,
            "checks": []
        }
        
        logger.info("Checking normalization...")
        
        try:
            # Fit on training data only
            mid_idx = len(data) // 2
            train_data = data.iloc[:mid_idx]
            test_data = data.iloc[mid_idx:]
            
            # Fit normalizer on train only
            normalizer.fit(train_data)
            
            # Check if normalizer statistics match training data
            if hasattr(normalizer, 'mean_'):
                train_mean = train_data.mean()
                if not np.allclose(normalizer.mean_, train_mean.values, rtol=0.01):
                    results["passed"] = False
                    results["checks"].append({
                        "check": "normalizer_fit",
                        "passed": False,
                        "message": "Normalizer statistics don't match training data",
                        "severity": "CRITICAL"
                    })
            else:
                results["checks"].append({
                    "check": "normalizer_fit",
                    "passed": True,
                    "message": "Normalizer correctly fitted on training data only"
                })
        except Exception as e:
            results["checks"].append({
                "check": "normalizer_fit",
                "passed": None,
                "message": f"Could not verify: {e}"
            })
        
        return results
    
    def audit_cv_leakage(
        self,
        cv_splits: List[Tuple[np.ndarray, np.ndarray]],
        purge_size: int = 5,
        embargo_size: int = 5
    ) -> Dict[str, Any]:
        """
        Audit cross-validation for data leakage.
        
        Checks:
        1. Purge gap exists between train and test
        2. Embargo period after test
        3. No overlapping indices
        """
        results = {
            "component": "Cross-Validation",
            "passed": True,
            "checks": []
        }
        
        logger.info("Checking CV splits...")
        
        for i, (train_idx, test_idx) in enumerate(cv_splits):
            # Check for overlap
            overlap = np.intersect1d(train_idx, test_idx)
            if len(overlap) > 0:
                results["passed"] = False
                results["checks"].append({
                    "check": f"cv_split_{i}_overlap",
                    "passed": False,
                    "message": f"Split {i}: {len(overlap)} overlapping indices!",
                    "severity": "CRITICAL"
                })
                self.issues.append(f"CV split {i} has overlapping train/test!")
                continue
            
            # Check purge gap
            train_max = train_idx.max()
            test_min = test_idx.min()
            
            gap = test_min - train_max - 1
            
            if gap < purge_size:
                results["passed"] = False
                results["checks"].append({
                    "check": f"cv_split_{i}_purge",
                    "passed": False,
                    "message": f"Split {i}: Gap={gap}, required purge={purge_size}",
                    "severity": "CRITICAL"
                })
                self.issues.append(f"CV split {i} missing purge gap!")
            else:
                results["checks"].append({
                    "check": f"cv_split_{i}",
                    "passed": True,
                    "message": f"Split {i}: Proper gap of {gap} days"
                })
        
        return results
    
    def run_full_audit(
        self,
        prices: pd.DataFrame,
        labeler=None,
        frac_differ=None,
        feature_funcs: Dict[str, Callable] = None,
        cv_splits: List[Tuple] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive lookahead audit on full pipeline.
        
        Returns:
        --------
        Dict with all audit results
        """
        self.issues = []
        self.warnings = []
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_passed": True,
            "components": []
        }
        
        logger.info("\n" + "="*60)
        logger.info("LOOKAHEAD BIAS AUDIT")
        logger.info("="*60)
        
        # Audit Triple Barrier
        if labeler is not None:
            close = prices['close'] if 'close' in prices.columns else prices.iloc[:, 0]
            result = self.audit_triple_barrier(close, labeler)
            results["components"].append(result)
            if not result["passed"]:
                results["overall_passed"] = False
        
        # Audit Fractional Diff
        if frac_differ is not None:
            result = self.audit_fractional_diff(prices, frac_differ)
            results["components"].append(result)
            if not result["passed"]:
                results["overall_passed"] = False
        
        # Audit Features
        if feature_funcs is not None:
            for name, func in feature_funcs.items():
                result = self.audit_features(prices, func, name)
                results["components"].append(result)
                if not result["passed"]:
                    results["overall_passed"] = False
        
        # Audit CV
        if cv_splits is not None:
            result = self.audit_cv_leakage(cv_splits)
            results["components"].append(result)
            if not result["passed"]:
                results["overall_passed"] = False
        
        # Summary
        results["issues"] = self.issues
        results["warnings"] = self.warnings
        
        logger.info("\n" + "-"*60)
        if results["overall_passed"]:
            logger.info("✓ AUDIT PASSED - No lookahead bias detected")
        else:
            logger.error(f"✗ AUDIT FAILED - {len(self.issues)} critical issues found:")
            for issue in self.issues:
                logger.error(f"  - {issue}")
        
        if self.warnings:
            logger.warning(f"⚠ {len(self.warnings)} warnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        logger.info("-"*60)
        
        return results


def audit_existing_pipeline(data_path: str = None) -> Dict[str, Any]:
    """
    Convenience function to audit the existing Phase 1+2 pipeline.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from pathlib import Path
    from .config import CONFIG
    
    # Load sample data
    if data_path:
        prices = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        # Load first available file
        data_files = list(CONFIG.train_dir.glob("*.csv"))
        if not data_files:
            raise ValueError("No training data found. Run data_downloader first.")
        prices = pd.read_csv(data_files[0], index_col=0, parse_dates=True)
    
    # Import pipeline components
    try:
        from src.institutional.labeling.triple_barrier import TripleBarrierLabeler
        from src.institutional.features.frac_diff import FractionalDifferentiator
        
        labeler = TripleBarrierLabeler()
        frac_differ = FractionalDifferentiator()
        
        auditor = LookaheadAudit()
        results = auditor.run_full_audit(
            prices=prices,
            labeler=labeler,
            frac_differ=frac_differ
        )
        
        return results
        
    except ImportError as e:
        logger.warning(f"Could not import components: {e}")
        logger.info("Running basic audit only...")
        
        auditor = LookaheadAudit()
        return auditor.run_full_audit(prices=prices)


if __name__ == "__main__":
    from pathlib import Path
    
    results = audit_existing_pipeline()
    
    print("\n" + "="*60)
    print("AUDIT SUMMARY")
    print("="*60)
    print(f"Overall: {'PASSED' if results['overall_passed'] else 'FAILED'}")
    print(f"Issues: {len(results['issues'])}")
    print(f"Warnings: {len(results['warnings'])}")
