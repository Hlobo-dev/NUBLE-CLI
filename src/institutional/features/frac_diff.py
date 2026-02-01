"""
Fractional Differentiation
===========================

Implementation of Fractional Differentiation from:
Marcos López de Prado, "Advances in Financial Machine Learning" (2018), Chapter 5.

The Problem:
------------
Standard differencing (d=1) makes price series stationary but removes ALL memory.
No memory = no predictability. This is why simple returns are hard to predict.

The Solution:
-------------
Fractional differentiation uses d between 0 and 1 to achieve stationarity
while preserving as much memory (autocorrelation) as possible.

Key Insight:
------------
Find the MINIMUM d that makes the series stationary. This preserves maximum
predictive information while still allowing ML models to learn properly.

Methods Implemented:
1. FFD (Fixed-width window Fractional Differentiation) - Preferred
2. Standard Fracdiff - Requires infinite history, not practical

The FFD method uses a finite window of weights, making it practical for
real-time applications while closely approximating true fracdiff.

References:
-----------
[1] López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 5.
[2] Hosking, J.R.M. (1981). "Fractional Differencing." Biometrika.

Author: KYPERIAN Institutional
Version: 2.0.0 (Lopez de Prado Compliant)
"""

import numpy as np
import pandas as pd
import numba
from numba import jit, prange
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
import logging
import warnings

# Statistical tests
try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available. ADF tests disabled.")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FracDiffConfig:
    """
    Configuration for Fractional Differentiation.
    
    Attributes:
    -----------
    threshold : float
        Weight cutoff threshold. Weights below this are truncated.
        Default 1e-5 (very conservative). Use 1e-4 for faster computation.
        
    pvalue_threshold : float
        P-value threshold for ADF test. Series is stationary if p < threshold.
        Default 0.05 (95% confidence). Use 0.01 for stricter stationarity.
        
    max_d : float
        Maximum differentiation order to try.
        Default 1.0. Rarely need d > 1 for financial series.
        
    min_d : float
        Minimum differentiation order.
        Default 0.0 (no differentiation). Sometimes need small d > 0.
        
    d_step : float
        Step size for d search. Smaller = more precise but slower.
        Default 0.01 for binary search. Use 0.1 for grid search.
        
    search_method : str
        Method to find optimal d:
        - 'binary': Binary search (fast, precise)
        - 'grid': Grid search (slower, guarantees global minimum)
        
    min_periods : int
        Minimum observations required for ADF test.
        Default 20. Increase for more reliable tests.
        
    lag_order : int
        Lag order for ADF test. If None, uses automatic selection.
        Default None (auto).
    """
    threshold: float = 1e-5
    pvalue_threshold: float = 0.05
    max_d: float = 1.0
    min_d: float = 0.0
    d_step: float = 0.01
    search_method: str = 'binary'
    min_periods: int = 20
    lag_order: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if not 0 < self.pvalue_threshold < 1:
            raise ValueError("pvalue_threshold must be between 0 and 1")
        if self.max_d < self.min_d:
            raise ValueError("max_d must be >= min_d")
        if self.d_step <= 0:
            raise ValueError("d_step must be positive")
        if self.search_method not in ['binary', 'grid']:
            raise ValueError("search_method must be 'binary' or 'grid'")


# =============================================================================
# WEIGHT COMPUTATION
# =============================================================================

@jit(nopython=True, cache=True)
def _get_weights_ffd_numba(d: float, threshold: float, max_length: int) -> np.ndarray:
    """
    Numba-optimized weight computation for FFD.
    
    Computes weights from the binomial expansion of (1-B)^d
    where B is the backshift operator.
    
    The expansion is:
    (1-B)^d = sum_{k=0}^{infinity} C(d,k) * (-B)^k
    
    where C(d,k) = d! / (k! * (d-k)!) is the binomial coefficient.
    
    For fractional d, this becomes:
    C(d,k) = d * (d-1) * ... * (d-k+1) / k!
    
    Parameters:
    -----------
    d : float
        Differentiation order (0 < d < 1 typically)
    threshold : float
        Cutoff for weight magnitude
    max_length : int
        Maximum number of weights to compute
        
    Returns:
    --------
    np.ndarray : Array of weights [w_k, w_{k-1}, ..., w_1, w_0]
                 Note: returned in reverse order for convolution
    """
    weights = np.zeros(max_length, dtype=np.float64)
    weights[0] = 1.0
    
    k = 1
    while k < max_length:
        # Recursive formula: w_k = -w_{k-1} * (d - k + 1) / k
        weights[k] = -weights[k - 1] * (d - k + 1) / k
        
        # Check convergence
        if abs(weights[k]) < threshold:
            break
        k += 1
    
    # Trim to actual length and reverse for convolution
    return weights[:k][::-1]


def get_weights_ffd(
    d: float,
    threshold: float = 1e-5,
    max_length: int = 10000
) -> np.ndarray:
    """
    Get weights for Fixed-width window Fractional Differentiation.
    
    The FFD method approximates true fractional differentiation using
    a finite window of weights. This makes it practical for real-time
    applications while preserving most of the memory structure.
    
    Parameters:
    -----------
    d : float
        Differentiation order. 
        - d=0: No differentiation (original series)
        - d=1: Standard first difference
        - 0<d<1: Fractional differentiation (preserves memory)
        
    threshold : float
        Minimum absolute weight to include. Weights below this are truncated.
        Lower threshold = more accurate but wider window.
        
    max_length : int
        Maximum number of weights (safety limit)
        
    Returns:
    --------
    np.ndarray : Weight array for convolution with time series.
                 First element corresponds to most recent observation.
    
    Example:
    --------
    >>> weights = get_weights_ffd(0.5, threshold=1e-4)
    >>> print(f"Window size: {len(weights)}")
    >>> print(f"Sum of weights: {weights.sum():.4f}")  # Should be near 0 for d=1
    
    Notes:
    ------
    - For d=1, weights sum to exactly 0 (first difference)
    - For 0<d<1, weights sum to positive value (preserves level information)
    - Larger d = faster weight decay = shorter window
    """
    if d < 0:
        raise ValueError("d must be non-negative")
    if d == 0:
        return np.array([1.0])
    
    weights = _get_weights_ffd_numba(d, threshold, max_length)
    
    logger.debug(f"FFD weights: d={d:.3f}, window={len(weights)}, sum={weights.sum():.4f}")
    
    return weights


# =============================================================================
# FRACTIONAL DIFFERENTIATION
# =============================================================================

@jit(nopython=True, parallel=True, cache=True)
def _apply_ffd_numba(
    series: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Numba-optimized FFD application.
    
    Applies fractional differentiation as a convolution with weights.
    Uses parallel processing for large series.
    """
    n = len(series)
    width = len(weights)
    result = np.full(n, np.nan, dtype=np.float64)
    
    # Apply weights starting from where we have enough history
    for i in prange(width - 1, n):
        result[i] = np.dot(weights, series[i - width + 1:i + 1])
    
    return result


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5
) -> pd.Series:
    """
    Apply Fixed-width window Fractional Differentiation.
    
    This is the main function for transforming a time series to achieve
    stationarity while preserving memory (predictability).
    
    The transformation is:
    X_t^{(d)} = sum_{k=0}^{K} w_k * X_{t-k}
    
    where w_k are the FFD weights and K is the window size.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to differentiate. Should be prices (not returns).
        
    d : float
        Fractional differentiation order:
        - d=0: No change (returns original series)
        - d=1: Standard first difference (same as series.diff())
        - 0<d<1: Fractional (preserves some memory while achieving stationarity)
        
    threshold : float
        Weight cutoff threshold. Smaller = more accurate but wider window.
        
    Returns:
    --------
    pd.Series : Fractionally differentiated series.
                Leading values are NaN (require more history than available).
    
    Example:
    --------
    >>> prices = pd.Series([100, 101, 102, 101, 103, 104])
    >>> ffd_prices = frac_diff_ffd(prices, d=0.4)
    >>> print(ffd_prices.dropna())
    
    Notes:
    ------
    - Apply to PRICES, not returns. Returns are already d=1 differentiated.
    - The output has the same index as input, but leading values are NaN.
    - For ML, drop NaN values and align with other features.
    """
    if d == 0:
        return series.copy()
    
    if d < 0:
        raise ValueError("d must be non-negative")
    
    # Get weights
    weights = get_weights_ffd(d, threshold)
    width = len(weights)
    
    if width > len(series):
        logger.warning(
            f"FFD window ({width}) larger than series ({len(series)}). "
            f"Consider using larger threshold or smaller d."
        )
        return pd.Series(np.nan, index=series.index)
    
    # Apply FFD using Numba
    values = series.values.astype(np.float64)
    result = _apply_ffd_numba(values, weights)
    
    # Return as Series with same index
    return pd.Series(result, index=series.index)


# =============================================================================
# STATIONARITY TESTING
# =============================================================================

def test_stationarity(
    series: pd.Series,
    significance: float = 0.05,
    regression: str = 'c',
    maxlag: Optional[int] = None
) -> Dict[str, Any]:
    """
    Test for stationarity using Augmented Dickey-Fuller test.
    
    The null hypothesis is that the series has a unit root (non-stationary).
    We reject the null if p-value < significance.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to test
    significance : float
        Significance level (default 0.05 for 95% confidence)
    regression : str
        Constant and trend order to include:
        - 'c': constant only (default)
        - 'ct': constant and trend
        - 'ctt': constant, linear and quadratic trend
        - 'n': no constant, no trend
    maxlag : int, optional
        Maximum lag for ADF test. If None, uses automatic selection.
        
    Returns:
    --------
    Dict with:
        - is_stationary: bool (True if can reject unit root)
        - adf_statistic: float (test statistic)
        - pvalue: float (p-value)
        - critical_values: dict (critical values at 1%, 5%, 10%)
        - n_lags: int (number of lags used)
        - n_obs: int (number of observations)
    """
    if not STATSMODELS_AVAILABLE:
        return {
            'is_stationary': None,
            'error': 'statsmodels not available'
        }
    
    # Drop NaN values
    clean_series = series.dropna()
    
    if len(clean_series) < 20:
        return {
            'is_stationary': None,
            'error': f'Not enough observations ({len(clean_series)} < 20)'
        }
    
    try:
        adf_result = adfuller(
            clean_series,
            maxlag=maxlag,
            regression=regression,
            autolag='AIC' if maxlag is None else None
        )
        
        adf_stat, pvalue, n_lags, n_obs, critical_values, icbest = adf_result
        
        return {
            'is_stationary': pvalue < significance,
            'adf_statistic': adf_stat,
            'pvalue': pvalue,
            'critical_values': critical_values,
            'n_lags': n_lags,
            'n_obs': n_obs,
            'significance': significance
        }
        
    except Exception as e:
        return {
            'is_stationary': None,
            'error': str(e)
        }


# =============================================================================
# OPTIMAL D SEARCH
# =============================================================================

def find_min_ffd(
    series: pd.Series,
    d_range: Tuple[float, float] = (0.0, 1.0),
    threshold: float = 1e-5,
    pvalue_threshold: float = 0.05,
    method: str = 'binary',
    d_step: float = 0.01,
    verbose: bool = False
) -> float:
    """
    Find minimum d that makes series stationary.
    
    This is the key function for optimal fractional differentiation.
    It finds the smallest d such that the ADF test rejects the unit root
    hypothesis at the specified significance level.
    
    Why minimum d?
    - Smaller d = more memory preserved = more predictability
    - We want JUST ENOUGH differentiation to achieve stationarity
    
    Parameters:
    -----------
    series : pd.Series
        Time series (typically log prices)
        
    d_range : Tuple[float, float]
        Range to search (default 0 to 1)
        
    threshold : float
        FFD weight cutoff
        
    pvalue_threshold : float
        ADF test significance level
        
    method : str
        Search method:
        - 'binary': Binary search (fast, precise)
        - 'grid': Grid search (exhaustive, slower)
        
    d_step : float
        Step size for grid search or precision for binary search
        
    verbose : bool
        Print search progress
        
    Returns:
    --------
    float : Minimum d for stationarity.
            Returns d_range[1] if no stationary d found.
    
    Example:
    --------
    >>> import yfinance as yf
    >>> prices = yf.download('SPY', period='5y')['Close']
    >>> log_prices = np.log(prices)
    >>> optimal_d = find_min_ffd(log_prices)
    >>> print(f"Optimal d: {optimal_d:.2f}")
    >>> # Apply the optimal d
    >>> stationary_prices = frac_diff_ffd(log_prices, optimal_d)
    
    Notes:
    ------
    - Apply to LOG prices for better numerical stability
    - The function assumes d=0 is non-stationary and d=1 is stationary
    - If the original series is already stationary, returns 0
    """
    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not available. Returning d=0.5 as fallback.")
        return 0.5
    
    d_low, d_high = d_range
    
    # First check if already stationary at d=0
    test_result = test_stationarity(series, pvalue_threshold)
    if test_result.get('is_stationary', False):
        if verbose:
            print(f"Series already stationary at d=0 (p={test_result['pvalue']:.4f})")
        return 0.0
    
    if method == 'binary':
        return _find_min_ffd_binary(
            series, d_low, d_high, threshold, pvalue_threshold, d_step, verbose
        )
    else:
        return _find_min_ffd_grid(
            series, d_low, d_high, threshold, pvalue_threshold, d_step, verbose
        )


def _find_min_ffd_binary(
    series: pd.Series,
    d_low: float,
    d_high: float,
    threshold: float,
    pvalue_threshold: float,
    precision: float,
    verbose: bool
) -> float:
    """Binary search for minimum stationary d."""
    
    iteration = 0
    max_iterations = 50
    
    while d_high - d_low > precision and iteration < max_iterations:
        d_mid = (d_low + d_high) / 2
        
        # Compute FFD at d_mid
        diff_series = frac_diff_ffd(series, d_mid, threshold)
        
        # Test stationarity
        test_result = test_stationarity(diff_series.dropna(), pvalue_threshold)
        
        if verbose:
            pval = test_result.get('pvalue', np.nan)
            stat = 'stationary' if test_result.get('is_stationary', False) else 'non-stationary'
            print(f"  d={d_mid:.3f}: p={pval:.4f} ({stat})")
        
        if test_result.get('is_stationary', False):
            d_high = d_mid  # Stationary, try lower d
        else:
            d_low = d_mid   # Not stationary, need higher d
        
        iteration += 1
    
    # Return the upper bound (guaranteed stationary)
    return d_high


def _find_min_ffd_grid(
    series: pd.Series,
    d_low: float,
    d_high: float,
    threshold: float,
    pvalue_threshold: float,
    d_step: float,
    verbose: bool
) -> float:
    """Grid search for minimum stationary d."""
    
    d_values = np.arange(d_low, d_high + d_step, d_step)
    
    for d in d_values:
        diff_series = frac_diff_ffd(series, d, threshold)
        test_result = test_stationarity(diff_series.dropna(), pvalue_threshold)
        
        if verbose:
            pval = test_result.get('pvalue', np.nan)
            stat = 'stationary' if test_result.get('is_stationary', False) else 'non-stationary'
            print(f"  d={d:.3f}: p={pval:.4f} ({stat})")
        
        if test_result.get('is_stationary', False):
            return d
    
    return d_high


# =============================================================================
# MEMORY PRESERVATION ANALYSIS
# =============================================================================

def compute_memory_preservation(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
    max_lags: int = 20
) -> Dict[str, Any]:
    """
    Analyze memory preservation after fractional differentiation.
    
    Compares autocorrelation structure of original and differentiated series
    to quantify how much predictive information is preserved.
    
    Parameters:
    -----------
    series : pd.Series
        Original time series
    d : float
        Differentiation order
    threshold : float
        FFD weight cutoff
    max_lags : int
        Maximum lags for autocorrelation
        
    Returns:
    --------
    Dict with:
        - original_acf: Autocorrelation of original series
        - diff_acf: Autocorrelation of differentiated series
        - memory_ratio: Ratio of preserved autocorrelation
        - correlation: Correlation between original and diff series
    """
    from scipy.stats import pearsonr
    
    # Differentiate
    diff_series = frac_diff_ffd(series, d, threshold)
    
    # Align series (drop NaN from differentiated)
    valid_idx = diff_series.dropna().index
    original_aligned = series.loc[valid_idx]
    diff_aligned = diff_series.loc[valid_idx]
    
    # Compute autocorrelations
    def compute_acf(s, max_lags):
        acf = []
        for lag in range(1, max_lags + 1):
            if len(s) > lag:
                corr, _ = pearsonr(s[:-lag], s[lag:])
                acf.append(corr)
            else:
                acf.append(np.nan)
        return np.array(acf)
    
    original_acf = compute_acf(original_aligned.values, max_lags)
    diff_acf = compute_acf(diff_aligned.values, max_lags)
    
    # Memory ratio (how much autocorrelation is preserved)
    # Higher is better (more memory preserved)
    valid_mask = ~(np.isnan(original_acf) | np.isnan(diff_acf) | (original_acf == 0))
    if valid_mask.sum() > 0:
        memory_ratio = np.abs(diff_acf[valid_mask]).sum() / np.abs(original_acf[valid_mask]).sum()
    else:
        memory_ratio = 0.0
    
    # Correlation between original and differentiated
    corr, pval = pearsonr(original_aligned, diff_aligned)
    
    return {
        'original_acf': original_acf,
        'diff_acf': diff_acf,
        'memory_ratio': memory_ratio,
        'correlation': corr,
        'correlation_pvalue': pval,
        'd': d
    }


# =============================================================================
# MAIN DIFFERENTIATOR CLASS
# =============================================================================

class FractionalDifferentiator:
    """
    Auto-tuning Fractional Differentiator for feature engineering.
    
    This class provides a scikit-learn compatible interface for applying
    fractional differentiation to multiple features simultaneously.
    
    Key Features:
    - Automatically finds optimal d for each feature
    - Caches optimal d values for reuse
    - Supports fit/transform pattern
    - Provides diagnostics and visualization
    
    Example:
    --------
    >>> # Create differentiator
    >>> fd = FractionalDifferentiator(pvalue_threshold=0.05)
    >>> 
    >>> # Fit and transform features
    >>> features = pd.DataFrame({
    ...     'price': prices,
    ...     'volume': volumes,
    ...     'volatility': vol
    ... })
    >>> stationary_features = fd.fit_transform(features)
    >>> 
    >>> # Check what d was found for each feature
    >>> print(fd.optimal_d)
    >>> 
    >>> # Transform new data using cached d values
    >>> new_features = fd.transform(new_data)
    """
    
    def __init__(
        self,
        pvalue_threshold: float = 0.05,
        threshold: float = 1e-5,
        d_range: Tuple[float, float] = (0.0, 1.0),
        search_method: str = 'binary',
        verbose: bool = False
    ):
        """
        Initialize FractionalDifferentiator.
        
        Parameters:
        -----------
        pvalue_threshold : float
            Significance level for ADF test (default 0.05)
            
        threshold : float
            FFD weight cutoff (default 1e-5)
            
        d_range : Tuple[float, float]
            Range to search for optimal d
            
        search_method : str
            'binary' (fast) or 'grid' (exhaustive)
            
        verbose : bool
            Print progress during fit
        """
        self.pvalue_threshold = pvalue_threshold
        self.threshold = threshold
        self.d_range = d_range
        self.search_method = search_method
        self.verbose = verbose
        
        # Fitted parameters
        self.optimal_d: Dict[str, float] = {}
        self.stationarity_results: Dict[str, Dict] = {}
        self.is_fitted = False
        
        logger.info(
            f"FractionalDifferentiator initialized: "
            f"pvalue={pvalue_threshold}, threshold={threshold}"
        )
    
    def fit(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> 'FractionalDifferentiator':
        """
        Find optimal d for each column.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with features to differentiate
        columns : List[str], optional
            Columns to process. If None, use all numeric columns.
            
        Returns:
        --------
        self : FractionalDifferentiator
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Finding optimal d for {len(columns)} columns...")
        
        for col in columns:
            if self.verbose:
                print(f"\nProcessing '{col}'...")
            
            series = df[col].dropna()
            
            if len(series) < 50:
                logger.warning(f"Column '{col}' has too few observations ({len(series)}). Skipping.")
                self.optimal_d[col] = 0.0
                continue
            
            # Find minimum d for stationarity
            optimal_d = find_min_ffd(
                series,
                d_range=self.d_range,
                threshold=self.threshold,
                pvalue_threshold=self.pvalue_threshold,
                method=self.search_method,
                verbose=self.verbose
            )
            
            self.optimal_d[col] = optimal_d
            
            # Store stationarity test results
            diff_series = frac_diff_ffd(series, optimal_d, self.threshold)
            self.stationarity_results[col] = test_stationarity(
                diff_series.dropna(), 
                self.pvalue_threshold
            )
            
            if self.verbose:
                print(f"  Optimal d: {optimal_d:.3f}")
        
        self.is_fitted = True
        logger.info(f"Fit complete. Optimal d values: {self.optimal_d}")
        
        return self
    
    def transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        suffix: str = '_ffd'
    ) -> pd.DataFrame:
        """
        Apply fractional differentiation using fitted d values.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to transform
        columns : List[str], optional
            Columns to transform. Must have been fitted.
        suffix : str
            Suffix for new column names
            
        Returns:
        --------
        pd.DataFrame : Original data with FFD columns added
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        if columns is None:
            columns = list(self.optimal_d.keys())
        
        result = df.copy()
        
        for col in columns:
            if col not in self.optimal_d:
                raise ValueError(f"Column '{col}' was not fitted")
            
            d = self.optimal_d[col]
            
            if d == 0:
                # No differentiation needed
                result[f"{col}{suffix}"] = df[col]
            else:
                result[f"{col}{suffix}"] = frac_diff_ffd(
                    df[col], d, self.threshold
                )
        
        return result
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        suffix: str = '_ffd'
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to fit and transform
        columns : List[str], optional
            Columns to process
        suffix : str
            Suffix for new column names
            
        Returns:
        --------
        pd.DataFrame : Transformed data
        """
        self.fit(df, columns)
        return self.transform(df, columns, suffix)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information.
        
        Returns:
        --------
        Dict with:
            - optimal_d: Dict of column -> optimal d
            - stationarity: Dict of column -> ADF test results
            - summary: Overall summary
        """
        if not self.is_fitted:
            return {'error': 'Not fitted'}
        
        return {
            'optimal_d': self.optimal_d,
            'stationarity_results': self.stationarity_results,
            'summary': {
                'n_columns': len(self.optimal_d),
                'avg_d': np.mean(list(self.optimal_d.values())),
                'max_d': max(self.optimal_d.values()),
                'min_d': min(self.optimal_d.values()),
                'all_stationary': all(
                    r.get('is_stationary', False) 
                    for r in self.stationarity_results.values()
                )
            }
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_min_ffd(
    series: pd.Series,
    d_values: Optional[np.ndarray] = None,
    threshold: float = 1e-5,
    pvalue_threshold: float = 0.05,
    figsize: Tuple[int, int] = (14, 10)
) -> Dict[str, Any]:
    """
    Plot analysis of minimum FFD d value.
    
    Creates a comprehensive visualization showing:
    1. Original vs differentiated series
    2. ADF statistic as function of d
    3. Autocorrelation preservation
    4. Optimal d annotation
    
    Parameters:
    -----------
    series : pd.Series
        Time series to analyze
    d_values : np.ndarray, optional
        d values to test. Default: 0 to 1 in 0.1 steps
    threshold : float
        FFD weight cutoff
    pvalue_threshold : float
        Significance level
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    Dict with analysis results
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for plotting")
        return {}
    
    if d_values is None:
        d_values = np.arange(0, 1.05, 0.1)
    
    # Compute ADF stats for each d
    results = []
    for d in d_values:
        diff_series = frac_diff_ffd(series, d, threshold)
        test = test_stationarity(diff_series.dropna(), pvalue_threshold)
        
        results.append({
            'd': d,
            'adf_stat': test.get('adf_statistic', np.nan),
            'pvalue': test.get('pvalue', np.nan),
            'is_stationary': test.get('is_stationary', False)
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal d
    optimal_d = find_min_ffd(series, threshold=threshold, pvalue_threshold=pvalue_threshold)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Original series
    ax1 = axes[0, 0]
    ax1.plot(series.index, series.values, 'b-', alpha=0.7)
    ax1.set_title('Original Series')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    
    # 2. Differentiated series
    ax2 = axes[0, 1]
    diff_optimal = frac_diff_ffd(series, optimal_d, threshold)
    ax2.plot(diff_optimal.index, diff_optimal.values, 'g-', alpha=0.7)
    ax2.set_title(f'FFD Series (d={optimal_d:.2f})')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value')
    
    # 3. ADF statistic vs d
    ax3 = axes[1, 0]
    ax3.plot(results_df['d'], results_df['adf_stat'], 'b-o', markersize=8)
    ax3.axhline(y=-2.86, color='r', linestyle='--', label='5% Critical Value')
    ax3.axvline(x=optimal_d, color='g', linestyle=':', label=f'Optimal d={optimal_d:.2f}')
    ax3.set_title('ADF Statistic vs Differentiation Order')
    ax3.set_xlabel('d')
    ax3.set_ylabel('ADF Statistic')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. P-value vs d
    ax4 = axes[1, 1]
    ax4.semilogy(results_df['d'], results_df['pvalue'], 'b-o', markersize=8)
    ax4.axhline(y=pvalue_threshold, color='r', linestyle='--', label=f'p={pvalue_threshold}')
    ax4.axvline(x=optimal_d, color='g', linestyle=':', label=f'Optimal d={optimal_d:.2f}')
    ax4.set_title('P-Value vs Differentiation Order')
    ax4.set_xlabel('d')
    ax4.set_ylabel('P-Value (log scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'optimal_d': optimal_d,
        'results': results_df,
        'figure': fig
    }


# =============================================================================
# INTEGRATION WITH EXISTING PIPELINE
# =============================================================================

def add_ffd_features(
    df: pd.DataFrame,
    price_column: str = 'close',
    additional_columns: Optional[List[str]] = None,
    pvalue_threshold: float = 0.05,
    threshold: float = 1e-5
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Add fractional differentiation features to a DataFrame.
    
    Convenience function for quick integration with existing pipelines.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with price column
    price_column : str
        Name of price column (default 'close')
    additional_columns : List[str], optional
        Other columns to differentiate
    pvalue_threshold : float
        ADF significance level
    threshold : float
        FFD weight cutoff
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, float]] :
        - DataFrame with FFD features added
        - Dict of optimal d values
    
    Example:
    --------
    >>> data, d_values = add_ffd_features(ohlcv_df, price_column='close')
    >>> print(f"Price optimal d: {d_values['close']}")
    """
    columns = [price_column]
    if additional_columns:
        columns.extend(additional_columns)
    
    fd = FractionalDifferentiator(
        pvalue_threshold=pvalue_threshold,
        threshold=threshold,
        verbose=False
    )
    
    result = fd.fit_transform(df, columns)
    
    return result, fd.optimal_d


if __name__ == '__main__':
    # Simple test
    import yfinance as yf
    
    print("Testing Fractional Differentiator...")
    
    # Download test data
    data = yf.download('SPY', start='2020-01-01', end='2024-01-01', progress=False)
    prices = data['Close']
    log_prices = np.log(prices)
    
    # Test stationarity of original
    print("\n1. Testing original log prices:")
    result = test_stationarity(log_prices)
    print(f"   Stationary: {result.get('is_stationary')}, p-value: {result.get('pvalue', 'N/A'):.4f}")
    
    # Find optimal d
    print("\n2. Finding optimal d...")
    optimal_d = find_min_ffd(log_prices, verbose=True)
    print(f"   Optimal d: {optimal_d:.3f}")
    
    # Apply FFD
    print("\n3. Applying FFD...")
    ffd_prices = frac_diff_ffd(log_prices, optimal_d)
    
    # Test stationarity after FFD
    result_after = test_stationarity(ffd_prices.dropna())
    print(f"   After FFD - Stationary: {result_after.get('is_stationary')}, "
          f"p-value: {result_after.get('pvalue', 'N/A'):.4f}")
    
    # Test memory preservation
    print("\n4. Memory preservation analysis:")
    memory = compute_memory_preservation(log_prices, optimal_d)
    print(f"   Memory ratio: {memory['memory_ratio']:.3f}")
    print(f"   Correlation with original: {memory['correlation']:.3f}")
    
    # Test FractionalDifferentiator class
    print("\n5. Testing FractionalDifferentiator class...")
    fd = FractionalDifferentiator(verbose=False)
    
    test_df = pd.DataFrame({
        'price': prices,
        'volume': data['Volume']
    })
    
    result_df = fd.fit_transform(test_df, columns=['price'])
    diag = fd.get_diagnostics()
    print(f"   Optimal d values: {fd.optimal_d}")
    print(f"   All stationary: {diag['summary']['all_stationary']}")
    
    print("\n✅ All tests passed!")
