"""
COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)
==============================================
With Probability of Backtest Overfitting (PBO) and Deflated Sharpe Ratio.

This is the most rigorous validation from AFML Ch.11-12.

CPCV Properties:
- Tests ALL possible train/test combinations
- Purges data between train and test
- Computes PBO to estimate overfitting probability

Key Metrics:
- PBO > 0.5 = Strategy is likely overfit
- PBO < 0.3 = Strategy has reasonable generalization
- Deflated Sharpe adjusts for multiple testing
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Callable, Optional
from itertools import combinations
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
import logging
from pathlib import Path

from .config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CPCVSplit:
    """One CPCV split."""
    train_groups: Tuple[int, ...]
    test_groups: Tuple[int, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation.
    
    This tests ALL combinations of train/test groups,
    providing a robust estimate of strategy performance
    and its distribution.
    
    Parameters:
    -----------
    n_splits : int
        Number of groups to split data into (default: 6)
    n_test_groups : int
        Number of groups to use for testing (default: 2)
    purge_size : int
        Days to purge between train and test
    embargo_size : int
        Days to embargo after test
    """
    
    def __init__(
        self,
        n_splits: int = 6,
        n_test_groups: int = 2,
        purge_size: int = 5,
        embargo_size: int = 5
    ):
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.purge_size = purge_size
        self.embargo_size = embargo_size
        
        # Calculate number of paths (combinations)
        self.n_paths = self._count_paths()
        
        logger.info(f"CPCV: {n_splits} groups, {n_test_groups} test groups")
        logger.info(f"      {self.n_paths} total backtest paths")
    
    def _count_paths(self) -> int:
        """Count number of backtest paths."""
        from math import comb
        return comb(self.n_splits, self.n_test_groups)
    
    def split(
        self,
        data: pd.DataFrame
    ) -> List[CPCVSplit]:
        """
        Generate all CPCV splits.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to split
        
        Returns:
        --------
        List of CPCVSplit objects
        """
        n = len(data)
        group_size = n // self.n_splits
        
        # Create groups
        groups = []
        for i in range(self.n_splits):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < self.n_splits - 1 else n
            groups.append(np.arange(start_idx, end_idx))
        
        # Generate all test group combinations
        test_combinations = list(combinations(range(self.n_splits), self.n_test_groups))
        
        splits = []
        
        for test_groups in test_combinations:
            train_groups = tuple(g for g in range(self.n_splits) if g not in test_groups)
            
            # Get indices
            train_indices = np.concatenate([groups[g] for g in train_groups])
            test_indices = np.concatenate([groups[g] for g in test_groups])
            
            # Apply purging
            train_indices, test_indices = self._apply_purging(
                train_indices, test_indices, groups, train_groups, test_groups
            )
            
            split = CPCVSplit(
                train_groups=train_groups,
                test_groups=test_groups,
                train_indices=train_indices,
                test_indices=test_indices
            )
            
            splits.append(split)
        
        logger.info(f"Generated {len(splits)} CPCV splits")
        
        return splits
    
    def _apply_purging(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        groups: List[np.ndarray],
        train_groups: Tuple[int, ...],
        test_groups: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply purging between adjacent train/test groups.
        
        Removes observations from training that are too close to test.
        """
        train_set = set(train_indices)
        
        for test_g in test_groups:
            test_start = groups[test_g][0]
            test_end = groups[test_g][-1]
            
            # Purge before test
            for i in range(max(0, test_start - self.purge_size), test_start):
                train_set.discard(i)
            
            # Embargo after test
            for i in range(test_end + 1, min(len(groups) * len(groups[0]), test_end + 1 + self.embargo_size)):
                train_set.discard(i)
        
        return np.array(sorted(train_set)), test_indices
    
    def get_sklearn_cv(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get splits in sklearn format.
        """
        splits = self.split(data)
        return [(s.train_indices, s.test_indices) for s in splits]


class ProbabilityOfBacktestOverfitting:
    """
    Calculate Probability of Backtest Overfitting (PBO).
    
    From Bailey et al. "The Probability of Backtest Overfitting" (2014)
    
    PBO measures the probability that the best in-sample strategy
    will underperform the median out-of-sample.
    
    Interpretation:
    - PBO > 0.5: Strategy is overfit
    - PBO < 0.3: Reasonable generalization
    - PBO < 0.1: Excellent generalization
    """
    
    def __init__(self):
        self.results = None
    
    def calculate(
        self,
        in_sample_returns: List[pd.Series],
        out_sample_returns: List[pd.Series]
    ) -> Dict[str, Any]:
        """
        Calculate PBO from IS and OOS returns.
        
        Parameters:
        -----------
        in_sample_returns : List[pd.Series]
            Returns for each strategy in-sample
        out_sample_returns : List[pd.Series]
            Returns for each strategy out-of-sample
        
        Returns:
        --------
        Dict with PBO and related statistics
        """
        # Calculate Sharpe for each strategy
        is_sharpes = [self._sharpe(r) for r in in_sample_returns]
        oos_sharpes = [self._sharpe(r) for r in out_sample_returns]
        
        n_strategies = len(is_sharpes)
        
        if n_strategies < 2:
            return {
                "pbo": 0.0,
                "message": "Need at least 2 strategies for PBO"
            }
        
        # Find best IS strategy
        best_is_idx = np.argmax(is_sharpes)
        best_is_sharpe = is_sharpes[best_is_idx]
        best_is_oos_sharpe = oos_sharpes[best_is_idx]
        
        # Calculate OOS rank of best IS strategy
        oos_rank = sum(1 for s in oos_sharpes if s > best_is_oos_sharpe)
        oos_percentile = oos_rank / n_strategies
        
        # PBO is probability that best IS underperforms median OOS
        # We estimate this from the distribution
        median_oos = np.median(oos_sharpes)
        
        # Count how many times best IS underperforms median OOS
        # In single calculation, this is binary
        underperforms = best_is_oos_sharpe < median_oos
        
        # For proper PBO, we need multiple trials (CPCV provides this)
        # Here we use rank-based approximation
        pbo = oos_percentile
        
        self.results = {
            "pbo": pbo,
            "n_strategies": n_strategies,
            "best_is_idx": best_is_idx,
            "best_is_sharpe": best_is_sharpe,
            "best_is_oos_sharpe": best_is_oos_sharpe,
            "oos_rank": oos_rank,
            "oos_median": median_oos,
            "underperforms_median": underperforms,
            "is_sharpes": is_sharpes,
            "oos_sharpes": oos_sharpes
        }
        
        return self.results
    
    def calculate_from_cpcv(
        self,
        split_results: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Calculate PBO from CPCV split results.
        
        This is the proper way to calculate PBO using CPCV paths.
        
        Parameters:
        -----------
        split_results : List[Dict]
            Results from each CPCV split, must contain 'sharpe_ratio'
        
        Returns:
        --------
        Dict with PBO
        """
        sharpes = np.array([r['sharpe_ratio'] for r in split_results])
        n = len(sharpes)
        
        if n < 2:
            return {"pbo": 0.0, "message": "Need at least 2 splits"}
        
        # For CPCV, PBO is the fraction of paths where OOS Sharpe < median
        median_sharpe = np.median(sharpes)
        n_below_median = np.sum(sharpes < median_sharpe)
        
        # The PBO from CPCV is related to consistency
        # If strategy is robust, OOS Sharpe should be consistent
        sharpe_mean = np.mean(sharpes)
        sharpe_std = np.std(sharpes)
        
        # Coefficient of variation as overfitting proxy
        cv = sharpe_std / abs(sharpe_mean) if sharpe_mean != 0 else np.inf
        
        # PBO approximation: high CV suggests overfitting
        # Map CV to probability (heuristic, not exact)
        pbo_approx = 1 - 1 / (1 + cv)
        
        # Also compute rank-based PBO
        # Sort by IS performance (first half of path) and check OOS (second half)
        # For simplicity, use variance-based approximation
        
        self.results = {
            "pbo": pbo_approx,
            "sharpe_mean": sharpe_mean,
            "sharpe_std": sharpe_std,
            "sharpe_cv": cv,
            "sharpe_median": median_sharpe,
            "n_splits": n,
            "n_below_median": n_below_median
        }
        
        return self.results
    
    def _sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        mean = returns.mean()
        std = returns.std()
        if std == 0:
            return 0.0
        return mean / std * np.sqrt(252)


class DeflatedSharpeRatio:
    """
    Calculate Deflated Sharpe Ratio.
    
    From Bailey & Lopez de Prado "The Deflated Sharpe Ratio" (2014)
    
    This adjusts Sharpe for:
    1. Number of strategies tested
    2. Skewness and kurtosis of returns
    3. Track record length
    
    DSR < 1 suggests the Sharpe is not statistically significant
    after accounting for multiple testing.
    """
    
    def calculate(
        self,
        observed_sharpe: float,
        n_trials: int,
        returns: pd.Series,
        sharpe0: float = 0.0  # Null hypothesis Sharpe
    ) -> Dict[str, Any]:
        """
        Calculate Deflated Sharpe Ratio.
        
        Parameters:
        -----------
        observed_sharpe : float
            The observed Sharpe ratio
        n_trials : int
            Number of strategies/configurations tested
        returns : pd.Series
            Return series for skewness/kurtosis calculation
        sharpe0 : float
            Null hypothesis Sharpe (typically 0)
        
        Returns:
        --------
        Dict with DSR and related statistics
        """
        T = len(returns)
        
        if T < 4:
            return {"dsr": 0.0, "message": "Need at least 4 observations"}
        
        # Calculate moments
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)  # Excess kurtosis
        
        # Expected maximum Sharpe from n trials under null
        # E[max(SR)] ≈ √(2 * ln(n)) for normal returns
        if n_trials > 1:
            expected_max = np.sqrt(2 * np.log(n_trials))
        else:
            expected_max = 0
        
        # Variance of Sharpe ratio estimator
        # Var(SR) ≈ (1 + 0.5*SR^2 - skew*SR + (kurt/4)*SR^2) / T
        var_sr = (1 + 0.5 * observed_sharpe**2 
                  - skew * observed_sharpe 
                  + (kurt / 4) * observed_sharpe**2) / T
        
        std_sr = np.sqrt(max(var_sr, 1e-10))
        
        # Deflated Sharpe Ratio
        # DSR = (SR - E[max(SR)]) / std(SR)
        # We want P(SR > SR0 | trials)
        
        # Adjusted Sharpe accounting for trials
        adjusted_sharpe = observed_sharpe - expected_max
        
        # DSR is the probability that observed Sharpe is significant
        # Using the distribution of maximum Sharpe
        dsr = stats.norm.cdf(adjusted_sharpe / std_sr)
        
        return {
            "dsr": dsr,
            "observed_sharpe": observed_sharpe,
            "expected_max_sharpe": expected_max,
            "adjusted_sharpe": adjusted_sharpe,
            "std_sharpe": std_sr,
            "n_trials": n_trials,
            "T": T,
            "skewness": skew,
            "kurtosis": kurt,
            "significant": dsr > 0.95  # 95% confidence
        }


class CPCVBacktest:
    """
    Complete CPCV backtesting framework.
    
    This runs the strategy through all CPCV paths and calculates:
    - Performance distribution
    - PBO (Probability of Backtest Overfitting)
    - Deflated Sharpe Ratio
    """
    
    def __init__(
        self,
        model_factory: Callable,
        cpcv: CombinatorialPurgedCV = None,
        transaction_cost: float = 0.001
    ):
        """
        Parameters:
        -----------
        model_factory : Callable
            Function that returns a new model instance
        cpcv : CombinatorialPurgedCV
            CPCV splitter (uses CONFIG defaults if None)
        transaction_cost : float
            Round-trip transaction cost
        """
        self.model_factory = model_factory
        
        if cpcv is None:
            self.cpcv = CombinatorialPurgedCV(
                n_splits=CONFIG.cpcv.n_splits,
                n_test_groups=CONFIG.cpcv.n_test_groups,
                purge_size=CONFIG.cpcv.purge_size,
                embargo_size=CONFIG.cpcv.embargo_size
            )
        else:
            self.cpcv = cpcv
        
        self.transaction_cost = transaction_cost
        self.pbo_calculator = ProbabilityOfBacktestOverfitting()
        self.dsr_calculator = DeflatedSharpeRatio()
    
    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prices: pd.Series,
        n_trials: int = 1  # Number of strategies tested
    ) -> Dict[str, Any]:
        """
        Run complete CPCV backtest.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Labels
        prices : pd.Series
            Prices for return calculation
        n_trials : int
            Number of strategies tested (for DSR)
        
        Returns:
        --------
        Dict with all metrics including PBO and DSR
        """
        logger.info("\n" + "="*60)
        logger.info("COMBINATORIAL PURGED CROSS-VALIDATION")
        logger.info("="*60)
        
        splits = self.cpcv.split(X)
        
        all_returns = []
        split_results = []
        
        for i, split in enumerate(splits):
            logger.info(f"\nPath {i+1}/{len(splits)}: "
                       f"Train groups {split.train_groups}, "
                       f"Test groups {split.test_groups}")
            
            # Get data
            X_train = X.iloc[split.train_indices]
            y_train = y.iloc[split.train_indices]
            X_test = X.iloc[split.test_indices]
            y_test = y.iloc[split.test_indices]
            prices_test = prices.iloc[split.test_indices]
            
            # Train and predict
            model = self.model_factory()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Calculate returns
            signals = pd.Series(predictions, index=X_test.index)
            returns = self._calculate_returns(signals, prices_test)
            all_returns.append(returns)
            
            # Metrics
            sharpe = self._sharpe(returns)
            split_results.append({
                "split_id": i,
                "train_groups": split.train_groups,
                "test_groups": split.test_groups,
                "sharpe_ratio": sharpe,
                "total_return": (1 + returns).prod() - 1,
                "n_trades": len(returns)
            })
            
            logger.info(f"  Sharpe: {sharpe:.2f}")
        
        # Aggregate
        combined_returns = pd.concat(all_returns)
        aggregate_sharpe = self._sharpe(combined_returns)
        
        # Calculate PBO
        pbo_result = self.pbo_calculator.calculate_from_cpcv(split_results)
        
        # Calculate DSR
        dsr_result = self.dsr_calculator.calculate(
            observed_sharpe=aggregate_sharpe,
            n_trials=n_trials,
            returns=combined_returns
        )
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "n_paths": len(splits),
            "aggregate_sharpe": aggregate_sharpe,
            "aggregate_return": (1 + combined_returns).prod() - 1,
            "split_results": split_results,
            "sharpe_distribution": {
                "mean": np.mean([r["sharpe_ratio"] for r in split_results]),
                "std": np.std([r["sharpe_ratio"] for r in split_results]),
                "min": np.min([r["sharpe_ratio"] for r in split_results]),
                "max": np.max([r["sharpe_ratio"] for r in split_results])
            },
            "pbo": pbo_result,
            "dsr": dsr_result,
            "all_returns": combined_returns
        }
        
        self._log_summary(results)
        
        return results
    
    def _calculate_returns(
        self,
        signals: pd.Series,
        prices: pd.Series
    ) -> pd.Series:
        """Calculate strategy returns with transaction costs."""
        price_returns = prices.pct_change()
        strategy_returns = signals.shift(1) * price_returns
        trades = signals.diff().abs()
        costs = trades * self.transaction_cost
        return (strategy_returns - costs).dropna()
    
    def _sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        mean = returns.mean()
        std = returns.std()
        if std == 0:
            return 0.0
        return mean / std * np.sqrt(252)
    
    def _log_summary(self, results: Dict[str, Any]):
        """Log CPCV summary."""
        logger.info("\n" + "-"*60)
        logger.info("CPCV VALIDATION SUMMARY")
        logger.info("-"*60)
        
        logger.info(f"Number of paths: {results['n_paths']}")
        
        dist = results["sharpe_distribution"]
        logger.info(f"\nSharpe Distribution:")
        logger.info(f"  Mean: {dist['mean']:.2f} ± {dist['std']:.2f}")
        logger.info(f"  Range: [{dist['min']:.2f}, {dist['max']:.2f}]")
        
        logger.info(f"\nAggregate Sharpe: {results['aggregate_sharpe']:.2f}")
        logger.info(f"Aggregate Return: {results['aggregate_return']:.1%}")
        
        pbo = results["pbo"]
        logger.info(f"\nProbability of Backtest Overfitting (PBO):")
        logger.info(f"  PBO: {pbo['pbo']:.1%}")
        
        if pbo["pbo"] > 0.5:
            logger.warning("  ⚠️ PBO > 50% - Strategy is likely OVERFIT")
        elif pbo["pbo"] > 0.3:
            logger.warning("  ⚠️ PBO > 30% - Review carefully")
        else:
            logger.info("  ✓ PBO < 30% - Reasonable generalization")
        
        dsr = results["dsr"]
        logger.info(f"\nDeflated Sharpe Ratio:")
        logger.info(f"  DSR: {dsr['dsr']:.1%}")
        logger.info(f"  Expected Max Sharpe: {dsr['expected_max_sharpe']:.2f}")
        logger.info(f"  Adjusted Sharpe: {dsr['adjusted_sharpe']:.2f}")
        
        if dsr["significant"]:
            logger.info("  ✓ Sharpe is statistically significant (95%)")
        else:
            logger.warning("  ⚠️ Sharpe NOT statistically significant")
        
        # Final assessment
        logger.info("\n" + "="*60)
        if (results["aggregate_sharpe"] > 3.0 or 
            pbo["pbo"] > 0.5 or 
            not dsr["significant"]):
            logger.warning("⚠️ STRATEGY NEEDS REVIEW")
            if results["aggregate_sharpe"] > 3.0:
                logger.warning("   - Sharpe > 3.0 suggests bugs/lookahead bias")
            if pbo["pbo"] > 0.5:
                logger.warning("   - High PBO suggests overfitting")
            if not dsr["significant"]:
                logger.warning("   - Sharpe not statistically significant")
        else:
            logger.info("✓ STRATEGY PASSES VALIDATION")
            logger.info(f"  Realistic Sharpe: {results['aggregate_sharpe']:.2f}")
            logger.info(f"  PBO: {pbo['pbo']:.1%}")
        logger.info("="*60)


if __name__ == "__main__":
    logger.info("CPCV Validation Module")
    logger.info("Use CPCVBacktest class with your model and data")
