"""
Proper Probability of Backtest Overfitting (PBO)

Reference: Bailey, Borwein, López de Prado, Zhu (2014)
"Pseudo-Mathematics and Financial Charlatanism"

The correct procedure:
1. Generate all CSCV (Combinatorially Symmetric Cross-Validation) paths
2. For each path, rank strategies by IS performance
3. Compute logit of rank for the best IS strategy's OOS performance
4. PBO = probability that logit < 0 (best IS underperforms median OOS)

This is the CORRECT implementation - not the simplified version.
"""

import numpy as np
from scipy import stats
from itertools import combinations
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PBOResult:
    """Complete PBO analysis result."""
    pbo: float                          # Probability of Backtest Overfitting
    pbo_ci_lower: float                 # 95% CI lower bound
    pbo_ci_upper: float                 # 95% CI upper bound
    degradation: float                  # IS to OOS Sharpe degradation
    rank_correlation: float             # Spearman IS vs OOS ranks
    n_paths: int                        # Number of CSCV paths
    n_strategies: int                   # Number of strategies tested
    logits: np.ndarray                  # Raw logit values
    is_overfit: bool                    # PBO > 0.5
    confidence_level: str               # 'low', 'moderate', 'high' overfitting
    
    def __str__(self) -> str:
        return f"""
PBO Analysis Results:
=====================
PBO: {self.pbo:.1%} [{self.pbo_ci_lower:.1%}, {self.pbo_ci_upper:.1%}]
Degradation: {self.degradation:.1%}
Rank Correlation: {self.rank_correlation:.3f}
Paths: {self.n_paths}, Strategies: {self.n_strategies}
Confidence Level: {self.confidence_level}
Is Overfit: {'YES ❌' if self.is_overfit else 'NO ✅'}
"""


class ProperPBO:
    """
    Correct implementation of Probability of Backtest Overfitting.
    
    Key difference from simplified version:
    - Uses rank-based logit transformation
    - Computes full distribution across all CSCV paths
    - Returns confidence intervals
    - Properly handles edge cases
    
    Reference:
    Bailey, D., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014).
    "Pseudo-Mathematics and Financial Charlatanism: The Effects of 
    Backtest Overfitting on Out-of-Sample Performance."
    Notices of the American Mathematical Society, 61(5), 458-471.
    """
    
    def __init__(
        self,
        n_splits: int = 6,
        n_test_groups: int = 2,
        bootstrap_samples: int = 1000
    ):
        """
        Initialize PBO calculator.
        
        Parameters:
        -----------
        n_splits : int
            Number of data splits for CSCV
        n_test_groups : int
            Number of groups to use for testing in each path
        bootstrap_samples : int
            Number of bootstrap samples for CI calculation
        """
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.bootstrap_samples = bootstrap_samples
    
    def calculate(
        self,
        strategy_returns: Dict[str, np.ndarray],
        split_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        data_length: Optional[int] = None
    ) -> PBOResult:
        """
        Calculate PBO using proper Bailey/LdP methodology.
        
        Parameters:
        -----------
        strategy_returns : dict
            Dictionary mapping strategy name to daily returns array
            All arrays must be same length
            
        split_indices : list, optional
            Pre-computed (train_indices, test_indices) tuples
            If None, will generate CSCV splits from data_length
            
        data_length : int, optional
            Length of data for generating splits (required if split_indices is None)
            
        Returns:
        --------
        PBOResult with complete analysis
        """
        strategy_names = list(strategy_returns.keys())
        n_strategies = len(strategy_names)
        
        if n_strategies < 2:
            raise ValueError("Need at least 2 strategies for PBO calculation")
        
        # Validate all arrays same length
        lengths = [len(r) for r in strategy_returns.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All strategy return arrays must be same length")
        
        data_len = lengths[0]
        
        # Generate CSCV splits if not provided
        if split_indices is None:
            if data_length is None:
                data_length = data_len
            split_indices = self._generate_cscv_splits(data_length)
        
        n_paths = len(split_indices)
        
        # For each CSCV path, compute IS and OOS Sharpe for each strategy
        is_sharpes = np.zeros((n_paths, n_strategies))
        oos_sharpes = np.zeros((n_paths, n_strategies))
        
        for path_idx, (train_idx, test_idx) in enumerate(split_indices):
            for strat_idx, name in enumerate(strategy_names):
                returns = strategy_returns[name]
                
                # Validate indices are within bounds
                train_idx = np.array(train_idx)
                test_idx = np.array(test_idx)
                train_idx = train_idx[train_idx < len(returns)]
                test_idx = test_idx[test_idx < len(returns)]
                
                if len(train_idx) < 20 or len(test_idx) < 10:
                    is_sharpes[path_idx, strat_idx] = np.nan
                    oos_sharpes[path_idx, strat_idx] = np.nan
                    continue
                
                is_returns = returns[train_idx]
                oos_returns = returns[test_idx]
                
                is_sharpes[path_idx, strat_idx] = self._sharpe(is_returns)
                oos_sharpes[path_idx, strat_idx] = self._sharpe(oos_returns)
        
        # Remove paths with NaN values
        valid_mask = ~np.isnan(is_sharpes).any(axis=1) & ~np.isnan(oos_sharpes).any(axis=1)
        is_sharpes = is_sharpes[valid_mask]
        oos_sharpes = oos_sharpes[valid_mask]
        n_paths = len(is_sharpes)
        
        if n_paths < 5:
            raise ValueError(f"Only {n_paths} valid paths. Need at least 5 for reliable PBO.")
        
        # For each path, find best IS strategy and compute logit of its OOS rank
        logits = []
        
        for path_idx in range(n_paths):
            # Get Sharpes for this path
            is_path = is_sharpes[path_idx]
            oos_path = oos_sharpes[path_idx]
            
            # Find best IS strategy
            best_is_idx = np.argmax(is_path)
            
            # Get OOS Sharpe of best IS strategy
            best_is_oos_sharpe = oos_path[best_is_idx]
            
            # Compute rank of best IS strategy in OOS
            # Rank 1 = worst, rank n = best
            oos_rank = 1 + np.sum(oos_path < best_is_oos_sharpe)
            
            # Normalized rank to (0, 1) with epsilon to avoid log(0)
            n = n_strategies
            normalized_rank = (oos_rank - 0.5) / n
            normalized_rank = np.clip(normalized_rank, 0.01, 0.99)
            
            # Logit transformation: log(p / (1-p))
            logit = np.log(normalized_rank / (1 - normalized_rank))
            logits.append(logit)
        
        logits = np.array(logits)
        
        # PBO = probability that logit < 0
        # (i.e., best IS strategy is in bottom half OOS)
        pbo = np.mean(logits < 0)
        
        # Bootstrap confidence interval
        bootstrap_pbos = []
        for _ in range(self.bootstrap_samples):
            sample = np.random.choice(logits, size=len(logits), replace=True)
            bootstrap_pbos.append(np.mean(sample < 0))
        
        pbo_ci = np.percentile(bootstrap_pbos, [2.5, 97.5])
        
        # Degradation: average IS Sharpe vs OOS Sharpe for best IS strategy
        degradations = []
        for path_idx in range(n_paths):
            best_is_idx = np.argmax(is_sharpes[path_idx])
            is_sharpe = is_sharpes[path_idx, best_is_idx]
            oos_sharpe = oos_sharpes[path_idx, best_is_idx]
            if is_sharpe > 0:
                degradations.append((is_sharpe - oos_sharpe) / is_sharpe)
        
        degradation = np.mean(degradations) if degradations else 0
        
        # Rank correlation (Spearman) across all paths
        all_is_ranks = []
        all_oos_ranks = []
        for path_idx in range(n_paths):
            is_ranks = stats.rankdata(is_sharpes[path_idx])
            oos_ranks = stats.rankdata(oos_sharpes[path_idx])
            all_is_ranks.extend(is_ranks)
            all_oos_ranks.extend(oos_ranks)
        
        rank_corr, _ = stats.spearmanr(all_is_ranks, all_oos_ranks)
        
        # Determine confidence level
        if pbo < 0.3:
            confidence_level = 'low'
        elif pbo < 0.5:
            confidence_level = 'moderate'
        else:
            confidence_level = 'high'
        
        return PBOResult(
            pbo=pbo,
            pbo_ci_lower=pbo_ci[0],
            pbo_ci_upper=pbo_ci[1],
            degradation=degradation,
            rank_correlation=rank_corr,
            n_paths=n_paths,
            n_strategies=n_strategies,
            logits=logits,
            is_overfit=pbo > 0.5,
            confidence_level=confidence_level
        )
    
    def _generate_cscv_splits(self, data_length: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate Combinatorially Symmetric Cross-Validation splits.
        
        CSCV generates all possible combinations of train/test splits
        where test set is made of n_test_groups out of n_splits groups.
        """
        # Create group indices
        split_size = data_length // self.n_splits
        groups = []
        
        for i in range(self.n_splits):
            start = i * split_size
            end = start + split_size if i < self.n_splits - 1 else data_length
            groups.append(np.arange(start, end))
        
        # Generate all combinations
        splits = []
        for test_combo in combinations(range(self.n_splits), self.n_test_groups):
            test_indices = np.concatenate([groups[i] for i in test_combo])
            train_indices = np.concatenate([groups[i] for i in range(self.n_splits) if i not in test_combo])
            splits.append((train_indices, test_indices))
        
        return splits
    
    def _sharpe(self, returns: np.ndarray, annual_factor: float = 252) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(returns) / std * np.sqrt(annual_factor)


def validate_purge_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    label_horizon: int,
    feature_lookback: int
) -> Dict[str, any]:
    """
    Validate that purge/embargo is sufficient to prevent leakage.
    
    Common leakage sources:
    1. Label horizon overlap: If label uses t+10, need purge >= 10
    2. Feature lookback overlap: If feature uses t-20, need embargo >= 20
    
    Parameters:
    -----------
    train_idx : array
        Training indices
    test_idx : array
        Test indices  
    label_horizon : int
        Days forward used in label (e.g., 10 for triple barrier)
    feature_lookback : int
        Max lookback in features (e.g., 60 for 60-day vol)
    
    Returns:
    --------
    dict with validation results
    """
    train_max = int(np.max(train_idx))
    test_min = int(np.min(test_idx))
    
    gap = test_min - train_max
    
    results = {
        'gap_days': gap,
        'label_horizon': label_horizon,
        'feature_lookback': feature_lookback,
        'required_purge': label_horizon,
        'required_embargo': feature_lookback,
        'sufficient_purge': gap >= label_horizon,
        'sufficient_embargo': gap >= feature_lookback,
        'leakage_detected': False,
        'leakage_type': None,
        'leakage_detail': None
    }
    
    # Check for label leakage
    if gap < label_horizon:
        results['leakage_detected'] = True
        results['leakage_type'] = 'label_horizon'
        results['leakage_detail'] = f"Gap ({gap}) < label horizon ({label_horizon}). Labels in test set may use training data."
    
    # Check for feature leakage  
    elif gap < feature_lookback:
        results['leakage_detected'] = True
        results['leakage_type'] = 'feature_lookback'
        results['leakage_detail'] = f"Gap ({gap}) < feature lookback ({feature_lookback}). Features in test set may use training data."
    
    return results


class CSCVGenerator:
    """
    Combinatorially Symmetric Cross-Validation Generator.
    
    Generates all possible train/test splits with proper purge and embargo.
    """
    
    def __init__(
        self,
        n_splits: int = 6,
        n_test_groups: int = 2,
        purge_days: int = 10,
        embargo_pct: float = 0.01
    ):
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.purge_days = purge_days
        self.embargo_pct = embargo_pct
    
    def generate(self, data_length: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate all CSCV splits with purge and embargo.
        
        Parameters:
        -----------
        data_length : int
            Length of the data
            
        Returns:
        --------
        List of (train_indices, test_indices) tuples
        """
        # Create groups
        split_size = data_length // self.n_splits
        groups = []
        
        for i in range(self.n_splits):
            start = i * split_size
            end = start + split_size if i < self.n_splits - 1 else data_length
            groups.append(np.arange(start, end))
        
        splits = []
        
        # Generate all combinations
        for test_combo in combinations(range(self.n_splits), self.n_test_groups):
            # Test indices
            test_indices = np.concatenate([groups[i] for i in test_combo])
            test_indices = np.sort(test_indices)
            
            # Train indices (before purge)
            train_groups = [i for i in range(self.n_splits) if i not in test_combo]
            train_indices = np.concatenate([groups[i] for i in train_groups])
            train_indices = np.sort(train_indices)
            
            # Apply purge: remove train indices within purge_days of test
            if len(test_indices) > 0 and len(train_indices) > 0:
                test_min = test_indices.min()
                test_max = test_indices.max()
                
                # Remove train indices too close to test
                purge_mask = (train_indices < test_min - self.purge_days) | \
                            (train_indices > test_max + self.purge_days)
                train_indices = train_indices[purge_mask]
            
            # Apply embargo: remove first embargo_pct of test set
            if len(test_indices) > 0:
                embargo_size = int(len(test_indices) * self.embargo_pct)
                if embargo_size > 0:
                    test_indices = test_indices[embargo_size:]
            
            if len(train_indices) > 100 and len(test_indices) > 50:
                splits.append((train_indices, test_indices))
        
        return splits


# Test the implementation
if __name__ == "__main__":
    print("Testing Proper PBO Implementation...")
    print("="*60)
    
    # Generate synthetic strategy returns
    np.random.seed(42)
    n_days = 1000
    n_strategies = 5
    
    strategies = {}
    for i in range(n_strategies):
        # Each strategy has slightly different characteristics
        alpha = 0.0001 * (i + 1)  # Increasing alpha
        noise = np.random.randn(n_days) * 0.02
        strategies[f'strategy_{i}'] = alpha + noise
    
    # Calculate PBO
    pbo_calc = ProperPBO(n_splits=6, n_test_groups=2)
    result = pbo_calc.calculate(strategies, data_length=n_days)
    
    print(result)
    
    # Test leakage validation
    print("\nLeakage Validation Test:")
    print("-"*40)
    
    train_idx = np.arange(0, 500)
    test_idx = np.arange(505, 700)  # 5 day gap
    
    validation = validate_purge_embargo(
        train_idx, test_idx,
        label_horizon=10,
        feature_lookback=60
    )
    
    print(f"Gap: {validation['gap_days']} days")
    print(f"Leakage detected: {validation['leakage_detected']}")
    if validation['leakage_detected']:
        print(f"Type: {validation['leakage_type']}")
        print(f"Detail: {validation['leakage_detail']}")
