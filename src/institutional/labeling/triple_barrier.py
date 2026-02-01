"""
Triple Barrier Labeling Method
==============================

Implementation of the Triple Barrier Method from:
Marcos López de Prado, "Advances in Financial Machine Learning" (2018), Chapter 3.

The Triple Barrier Method labels observations based on three conditions:
1. Upper barrier: Profit target (take profit)
2. Lower barrier: Stop loss
3. Vertical barrier: Maximum holding period (time expiration)

This creates realistic labels that simulate actual trading conditions,
unlike simplistic "next day return" labels.

Key Innovations:
- Dynamic barriers based on rolling volatility (adapts to market conditions)
- Vertical barrier prevents label leakage from distant future
- Side information for meta-labeling compatibility
- Vectorized implementation for performance

References:
-----------
[1] López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
[2] López de Prado, M. (2020). Machine Learning for Asset Managers. Cambridge.

Author: KYPERIAN Institutional
Version: 2.0.0 (Lopez de Prado Compliant)
"""

import numpy as np
import pandas as pd
import numba
from numba import jit, prange
from typing import Tuple, Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TripleBarrierConfig:
    """
    Configuration for Triple Barrier Labeling.
    
    All parameters are based on empirical research and Lopez de Prado's
    recommendations for different asset classes and timeframes.
    
    Attributes:
    -----------
    pt_sl : Tuple[float, float]
        (profit_take_multiplier, stop_loss_multiplier) relative to volatility.
        Default (1.0, 1.0) means symmetric barriers at 1x daily volatility.
        Research suggests pt_sl=(2.0, 1.0) for trend-following strategies.
        
    max_holding_period : int
        Maximum bars to hold position before timeout (vertical barrier).
        Default 10 days for daily data. Shorter for intraday.
        
    volatility_lookback : int
        Window for calculating rolling volatility.
        Default 20 days (~1 trading month). Min 10, max 100.
        
    min_return : float
        Minimum absolute return to assign non-zero label.
        Returns below this threshold get label 0 (neutral).
        Default 0.0 (no minimum). Set to 0.005 (0.5%) to filter noise.
        
    volatility_type : str
        Method for volatility estimation:
        - 'standard': Standard deviation of returns
        - 'parkinson': Parkinson (high-low) estimator
        - 'garman_klass': Garman-Klass OHLC estimator (most accurate)
        - 'yang_zhang': Yang-Zhang overnight/intraday estimator
        
    side : Optional[pd.Series]
        Side of the bet (-1 for short, +1 for long, 0 for no bet).
        If provided, barriers are one-sided based on bet direction.
        If None, symmetric barriers are used.
        
    num_threads : int
        Number of threads for parallel processing.
        Default 1 (single-threaded). Set to -1 for all cores.
    """
    pt_sl: Tuple[float, float] = (1.0, 1.0)
    max_holding_period: int = 10
    volatility_lookback: int = 20
    min_return: float = 0.0
    volatility_type: str = 'standard'
    side: Optional[pd.Series] = None
    num_threads: int = 1
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate pt_sl
        if not isinstance(self.pt_sl, (tuple, list)) or len(self.pt_sl) != 2:
            raise ValueError("pt_sl must be a tuple of (profit_take, stop_loss)")
        if self.pt_sl[0] <= 0 or self.pt_sl[1] <= 0:
            raise ValueError("pt_sl multipliers must be positive")
        
        # Validate max_holding_period
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be >= 1")
        if self.max_holding_period > 252:
            logger.warning("max_holding_period > 252 days may cause lookahead bias")
        
        # Validate volatility_lookback
        if self.volatility_lookback < 5:
            raise ValueError("volatility_lookback must be >= 5")
        if self.volatility_lookback > 100:
            logger.warning("volatility_lookback > 100 may be too smooth")
        
        # Validate volatility_type
        valid_vol_types = ['standard', 'parkinson', 'garman_klass', 'yang_zhang']
        if self.volatility_type not in valid_vol_types:
            raise ValueError(f"volatility_type must be one of {valid_vol_types}")


@dataclass
class BarrierEvent:
    """
    Result of applying triple barrier to a single event.
    
    Attributes:
    -----------
    t0 : pd.Timestamp
        Entry time (event start)
    t1 : pd.Timestamp
        Exit time (barrier touch)
    barrier_type : str
        Which barrier was touched: 'upper', 'lower', or 'vertical'
    ret : float
        Return at exit (signed)
    label : int
        Classification label: +1 (profit), -1 (loss), 0 (timeout/neutral)
    side : int
        Side of the bet: +1 (long), -1 (short), 0 (none)
    volatility : float
        Volatility at entry time
    upper_barrier : float
        Upper barrier level (price)
    lower_barrier : float
        Lower barrier level (price)
    holding_period : int
        Actual holding period in bars
    """
    t0: pd.Timestamp
    t1: pd.Timestamp
    barrier_type: str
    ret: float
    label: int
    side: int = 1
    volatility: float = 0.0
    upper_barrier: float = 0.0
    lower_barrier: float = 0.0
    holding_period: int = 0


# =============================================================================
# VOLATILITY ESTIMATORS
# =============================================================================

def get_daily_volatility(
    close: pd.Series,
    lookback: int = 20,
    method: str = 'standard'
) -> pd.Series:
    """
    Calculate daily volatility for dynamic barrier sizing.
    
    Multiple estimation methods available, each with different properties:
    - Standard: Simple but noisy
    - Parkinson: Uses high-low range, more efficient
    - Garman-Klass: Uses OHLC, most efficient for complete data
    - Yang-Zhang: Handles overnight gaps, best for stocks
    
    Parameters:
    -----------
    close : pd.Series
        Close prices (or OHLC DataFrame for advanced methods)
    lookback : int
        Rolling window size (default 20 = ~1 trading month)
    method : str
        Volatility estimation method
        
    Returns:
    --------
    pd.Series : Rolling volatility estimate (daily scale)
    
    References:
    -----------
    [1] Parkinson, M. (1980). "The Extreme Value Method for Estimating the
        Variance of the Rate of Return." Journal of Business.
    [2] Garman, M. & Klass, M. (1980). "On the Estimation of Security Price
        Volatilities from Historical Data." Journal of Business.
    """
    if method == 'standard':
        # Standard deviation of log returns
        returns = np.log(close / close.shift(1))
        volatility = returns.rolling(window=lookback, min_periods=max(5, lookback // 2)).std()
        
    elif method == 'parkinson':
        # Parkinson estimator using high-low range
        # Requires DataFrame with 'high' and 'low' columns
        if isinstance(close, pd.DataFrame):
            high = close['high']
            low = close['low']
        else:
            # Fallback to standard if only close provided
            logger.warning("Parkinson requires OHLC data, falling back to standard")
            return get_daily_volatility(close, lookback, 'standard')
        
        log_hl = np.log(high / low) ** 2
        factor = 1 / (4 * np.log(2))
        volatility = np.sqrt(factor * log_hl.rolling(window=lookback).mean())
        
    elif method == 'garman_klass':
        # Garman-Klass estimator using OHLC
        if isinstance(close, pd.DataFrame):
            o = close['open']
            h = close['high']
            l = close['low']
            c = close['close']
        else:
            logger.warning("Garman-Klass requires OHLC data, falling back to standard")
            return get_daily_volatility(close, lookback, 'standard')
        
        log_hl = np.log(h / l) ** 2
        log_co = np.log(c / o) ** 2
        
        gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        volatility = np.sqrt(gk_var.rolling(window=lookback).mean())
        
    elif method == 'yang_zhang':
        # Yang-Zhang estimator (handles overnight gaps)
        if isinstance(close, pd.DataFrame):
            o = close['open']
            h = close['high']
            l = close['low']
            c = close['close']
        else:
            logger.warning("Yang-Zhang requires OHLC data, falling back to standard")
            return get_daily_volatility(close, lookback, 'standard')
        
        # Overnight volatility
        log_oc = np.log(o / c.shift(1))
        overnight_var = log_oc.rolling(window=lookback).var()
        
        # Open-to-close volatility
        log_co = np.log(c / o)
        open_close_var = log_co.rolling(window=lookback).var()
        
        # Rogers-Satchell volatility
        log_ho = np.log(h / o)
        log_lo = np.log(l / o)
        log_hc = np.log(h / c)
        log_lc = np.log(l / c)
        rs_var = (log_ho * log_hc + log_lo * log_lc).rolling(window=lookback).mean()
        
        # Yang-Zhang combination
        k = 0.34 / (1.34 + (lookback + 1) / (lookback - 1))
        volatility = np.sqrt(overnight_var + k * open_close_var + (1 - k) * rs_var)
        
    else:
        raise ValueError(f"Unknown volatility method: {method}")
    
    return volatility


# =============================================================================
# NUMBA-OPTIMIZED BARRIER DETECTION
# =============================================================================

@jit(nopython=True, cache=False)
def _find_barrier_touch_numba(
    prices: np.ndarray,
    idx: int,
    upper_barrier: float,
    lower_barrier: float,
    max_period: int
) -> Tuple[int, float, int]:
    """
    Numba-optimized barrier touch detection.
    
    Parameters:
    -----------
    prices : np.ndarray
        Array of prices
    idx : int
        Starting index
    upper_barrier : float
        Upper barrier price level
    lower_barrier : float
        Lower barrier price level
    max_period : int
        Maximum holding period
        
    Returns:
    --------
    Tuple[int, float, int] : (exit_idx, return, barrier_type)
        barrier_type: 1 = upper, -1 = lower, 0 = vertical
    """
    entry_price = prices[idx]
    n = len(prices)
    
    # Check each bar until barrier touch or max period
    for i in range(1, min(max_period + 1, n - idx)):
        current_price = prices[idx + i]
        
        # Check upper barrier
        if upper_barrier > 0 and current_price >= upper_barrier:
            ret = (current_price - entry_price) / entry_price
            return idx + i, ret, 1  # Upper barrier
        
        # Check lower barrier
        if lower_barrier > 0 and current_price <= lower_barrier:
            ret = (current_price - entry_price) / entry_price
            return idx + i, ret, -1  # Lower barrier
    
    # Vertical barrier (timeout)
    exit_idx = min(idx + max_period, n - 1)
    exit_price = prices[exit_idx]
    ret = (exit_price - entry_price) / entry_price
    
    return exit_idx, ret, 0  # Vertical barrier


@jit(nopython=True, parallel=True, cache=True)
def _apply_barriers_parallel(
    prices: np.ndarray,
    volatilities: np.ndarray,
    event_indices: np.ndarray,
    pt_mult: float,
    sl_mult: float,
    max_period: int,
    sides: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel barrier application using Numba.
    
    Returns:
    --------
    Tuple of arrays: (exit_indices, returns, barrier_types, upper_barriers, lower_barriers)
    """
    n_events = len(event_indices)
    
    exit_indices = np.zeros(n_events, dtype=np.int64)
    returns = np.zeros(n_events, dtype=np.float64)
    barrier_types = np.zeros(n_events, dtype=np.int64)
    upper_barriers = np.zeros(n_events, dtype=np.float64)
    lower_barriers = np.zeros(n_events, dtype=np.float64)
    
    for i in prange(n_events):
        idx = event_indices[i]
        entry_price = prices[idx]
        vol = volatilities[idx]
        side = sides[i] if len(sides) > 0 else 1
        
        # Calculate barrier levels
        if side >= 0:  # Long position
            upper = entry_price * (1 + pt_mult * vol) if pt_mult > 0 else 0
            lower = entry_price * (1 - sl_mult * vol) if sl_mult > 0 else 0
        else:  # Short position
            upper = entry_price * (1 + sl_mult * vol) if sl_mult > 0 else 0
            lower = entry_price * (1 - pt_mult * vol) if pt_mult > 0 else 0
        
        upper_barriers[i] = upper
        lower_barriers[i] = lower
        
        # Find barrier touch
        exit_idx, ret, barrier_type = _find_barrier_touch_numba(
            prices, idx, upper, lower, max_period
        )
        
        exit_indices[i] = exit_idx
        returns[i] = ret
        barrier_types[i] = barrier_type
    
    return exit_indices, returns, barrier_types, upper_barriers, lower_barriers


# =============================================================================
# MAIN TRIPLE BARRIER LABELER CLASS
# =============================================================================

class TripleBarrierLabeler:
    """
    Triple Barrier Labeling per López de Prado (2018).
    
    The Triple Barrier Method creates labels that simulate realistic trade outcomes
    by defining three exit conditions:
    
    1. **Upper Barrier (Take Profit)**: Exit when price rises by a multiple of 
       volatility. This represents profit-taking in a winning trade.
       
    2. **Lower Barrier (Stop Loss)**: Exit when price falls by a multiple of
       volatility. This represents cutting losses in a losing trade.
       
    3. **Vertical Barrier (Time Expiration)**: Exit after a maximum holding
       period regardless of price. This represents opportunity cost.
    
    Labels:
    - +1: Profit (upper barrier touched first for long, lower for short)
    - -1: Loss (lower barrier touched first for long, upper for short)
    -  0: Timeout (vertical barrier touched, no clear outcome)
    
    Key Features:
    - Dynamic barriers based on rolling volatility
    - Supports both long and short positions
    - Numba-optimized for performance
    - Comprehensive validation and diagnostics
    
    Example:
    --------
    >>> labeler = TripleBarrierLabeler(
    ...     pt_sl=(2.0, 1.0),      # Asymmetric: 2x vol profit, 1x vol stop
    ...     max_holding_period=10,  # 10 days max
    ...     volatility_lookback=20  # 20-day rolling vol
    ... )
    >>> labels = labeler.get_labels(close_prices)
    >>> print(f"Label distribution: {labels.value_counts()}")
    
    Reference Implementation:
    -------------------------
    This follows the exact methodology from AFML Chapter 3, with additional
    optimizations for production use.
    """
    
    def __init__(
        self,
        pt_sl: Tuple[float, float] = (1.0, 1.0),
        max_holding_period: int = 10,
        volatility_lookback: int = 20,
        min_return: float = 0.0,
        volatility_type: str = 'standard',
        num_threads: int = 1
    ):
        """
        Initialize Triple Barrier Labeler.
        
        Parameters:
        -----------
        pt_sl : Tuple[float, float]
            (profit_take_mult, stop_loss_mult) relative to daily volatility.
            Example: (2.0, 1.0) = take profit at 2x vol, stop loss at 1x vol
            
        max_holding_period : int
            Maximum bars to hold before vertical barrier (timeout)
            
        volatility_lookback : int
            Rolling window for volatility calculation
            
        min_return : float
            Minimum absolute return for non-zero label
            
        volatility_type : str
            Volatility estimation method: 'standard', 'parkinson', 
            'garman_klass', 'yang_zhang'
            
        num_threads : int
            Threads for parallel processing (-1 = all cores)
        """
        self.config = TripleBarrierConfig(
            pt_sl=pt_sl,
            max_holding_period=max_holding_period,
            volatility_lookback=volatility_lookback,
            min_return=min_return,
            volatility_type=volatility_type,
            num_threads=num_threads
        )
        
        # Cache for intermediate results
        self._volatility_cache: Optional[pd.Series] = None
        self._events_cache: Optional[List[BarrierEvent]] = None
        
        logger.info(
            f"TripleBarrierLabeler initialized: pt_sl={pt_sl}, "
            f"max_hold={max_holding_period}, vol_lookback={volatility_lookback}"
        )
    
    def get_volatility(
        self, 
        close: Union[pd.Series, pd.DataFrame],
        recalculate: bool = False
    ) -> pd.Series:
        """
        Get rolling volatility estimate.
        
        Parameters:
        -----------
        close : pd.Series or pd.DataFrame
            Price data (Series for close, DataFrame for OHLC)
        recalculate : bool
            Force recalculation even if cached
            
        Returns:
        --------
        pd.Series : Daily volatility estimates
        """
        if self._volatility_cache is not None and not recalculate:
            return self._volatility_cache
        
        volatility = get_daily_volatility(
            close,
            lookback=self.config.volatility_lookback,
            method=self.config.volatility_type
        )
        
        self._volatility_cache = volatility
        return volatility
    
    def apply_barriers(
        self,
        close: pd.Series,
        events: Optional[pd.DatetimeIndex] = None,
        side: Optional[pd.Series] = None,
        volatility: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Apply triple barrier to each event.
        
        This is the core method that determines which barrier is touched first
        for each trading event.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices with DatetimeIndex
        events : pd.DatetimeIndex, optional
            Timestamps of trading events. If None, use all valid timestamps.
        side : pd.Series, optional
            Side of bet (+1 long, -1 short) for each event.
            If None, assume all long (+1).
        volatility : pd.Series, optional
            Pre-calculated volatility. If None, calculate from close.
            
        Returns:
        --------
        pd.DataFrame with columns:
            - t1: Exit timestamp
            - ret: Return at exit
            - label: +1 (profit), -1 (loss), 0 (timeout)
            - barrier_type: 'upper', 'lower', 'vertical'
            - holding_period: Bars held
            - upper_barrier: Upper barrier price
            - lower_barrier: Lower barrier price
            - volatility: Entry volatility
        """
        # Calculate or use provided volatility
        if volatility is None:
            volatility = self.get_volatility(close, recalculate=True)
        
        # Determine valid events (where we have volatility)
        if events is None:
            # Use all timestamps where we have valid volatility
            valid_mask = volatility.notna() & (volatility > 0)
            # Leave room for max_holding_period at the end
            valid_mask.iloc[-self.config.max_holding_period:] = False
            events = close.index[valid_mask]
        else:
            # Validate provided events
            events = pd.DatetimeIndex([e for e in events if e in close.index])
        
        if len(events) == 0:
            logger.warning("No valid events for barrier application")
            return pd.DataFrame()
        
        # Prepare side information
        if side is None:
            sides_arr = np.ones(len(events), dtype=np.float64)
        else:
            sides_arr = side.reindex(events).fillna(1).values.astype(np.float64)
        
        # Get numpy arrays for Numba
        prices = close.values.astype(np.float64)
        vols = volatility.values.astype(np.float64)
        
        # Map event timestamps to indices
        event_indices = np.array([close.index.get_loc(e) for e in events], dtype=np.int64)
        
        # Apply barriers using Numba-optimized function
        pt_mult, sl_mult = self.config.pt_sl
        
        exit_indices, returns, barrier_types, upper_barriers, lower_barriers = \
            _apply_barriers_parallel(
                prices, vols, event_indices,
                pt_mult, sl_mult,
                self.config.max_holding_period,
                sides_arr
            )
        
        # Build result DataFrame
        results = []
        for i, event_time in enumerate(events):
            entry_idx = event_indices[i]
            exit_idx = exit_indices[i]
            
            # Determine label based on barrier type and side
            barrier_type = barrier_types[i]
            ret = returns[i]
            side_val = sides_arr[i]
            
            if barrier_type == 1:  # Upper barrier
                barrier_name = 'upper'
                label = 1 if side_val >= 0 else -1  # Profit for long, loss for short
            elif barrier_type == -1:  # Lower barrier
                barrier_name = 'lower'
                label = -1 if side_val >= 0 else 1  # Loss for long, profit for short
            else:  # Vertical barrier
                barrier_name = 'vertical'
                # Label based on return at timeout
                if abs(ret) < self.config.min_return:
                    label = 0  # Neutral
                else:
                    label = 1 if ret * side_val > 0 else -1
            
            results.append({
                't0': event_time,
                't1': close.index[exit_idx],
                'ret': ret,
                'label': label,
                'barrier_type': barrier_name,
                'holding_period': exit_idx - entry_idx,
                'upper_barrier': upper_barriers[i],
                'lower_barrier': lower_barriers[i],
                'volatility': vols[entry_idx],
                'side': side_val
            })
        
        result_df = pd.DataFrame(results)
        result_df.set_index('t0', inplace=True)
        
        # Cache events for diagnostics
        self._events_cache = results
        
        logger.info(
            f"Applied barriers to {len(events)} events. "
            f"Label distribution: +1={sum(result_df['label']==1)}, "
            f"0={sum(result_df['label']==0)}, -1={sum(result_df['label']==-1)}"
        )
        
        return result_df
    
    def get_labels(
        self,
        close: pd.Series,
        events: Optional[pd.DatetimeIndex] = None,
        side: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Get labels for training.
        
        Main entry point for integration with ML pipelines.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices with DatetimeIndex
        events : pd.DatetimeIndex, optional
            Event timestamps. If None, label all valid timestamps.
        side : pd.Series, optional
            Side of bet for each event
            
        Returns:
        --------
        pd.Series : Labels aligned with close index (-1, 0, +1)
        """
        barrier_results = self.apply_barriers(close, events, side)
        
        if barrier_results.empty:
            return pd.Series(index=close.index, dtype=float).fillna(0)
        
        # Extract labels
        labels = barrier_results['label']
        
        # Reindex to match close index
        full_labels = pd.Series(index=close.index, dtype=float)
        full_labels.update(labels)
        
        return full_labels
    
    def get_sample_weights(
        self,
        close: pd.Series,
        events: Optional[pd.DatetimeIndex] = None,
        method: str = 'return_attribution'
    ) -> pd.Series:
        """
        Calculate sample weights for training.
        
        Weights account for overlapping labels (concurrent trades) and
        the uniqueness of each observation's information.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
        events : pd.DatetimeIndex, optional
            Event timestamps
        method : str
            Weighting method:
            - 'return_attribution': Weight by attributed return
            - 'uniqueness': Weight by label uniqueness (less overlap = higher weight)
            - 'time_decay': Weight by recency
            
        Returns:
        --------
        pd.Series : Sample weights
        """
        barrier_results = self.apply_barriers(close, events)
        
        if barrier_results.empty:
            return pd.Series(index=close.index, dtype=float).fillna(1.0)
        
        if method == 'return_attribution':
            # Weight by absolute return (more informative events)
            weights = barrier_results['ret'].abs()
            # Normalize
            weights = weights / weights.sum() * len(weights)
            
        elif method == 'uniqueness':
            # Count concurrent labels at each timestamp
            concurrency = pd.Series(0, index=close.index, dtype=float)
            
            for idx, row in barrier_results.iterrows():
                t0 = idx
                t1 = row['t1']
                # Add 1 to all timestamps in this trade's span
                concurrency.loc[t0:t1] += 1
            
            # Uniqueness = 1 / average concurrency during trade
            weights = pd.Series(index=barrier_results.index, dtype=float)
            for idx, row in barrier_results.iterrows():
                t0 = idx
                t1 = row['t1']
                avg_conc = concurrency.loc[t0:t1].mean()
                weights.loc[t0] = 1.0 / avg_conc if avg_conc > 0 else 1.0
            
            # Normalize
            weights = weights / weights.sum() * len(weights)
            
        elif method == 'time_decay':
            # More recent events get higher weight
            n = len(barrier_results)
            decay_factor = 0.95
            weights = pd.Series(
                [decay_factor ** (n - i - 1) for i in range(n)],
                index=barrier_results.index
            )
            weights = weights / weights.sum() * len(weights)
            
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        # Reindex to match close index
        full_weights = pd.Series(index=close.index, dtype=float)
        full_weights.update(weights)
        full_weights = full_weights.fillna(1.0)
        
        return full_weights
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about the labeling process.
        
        Returns:
        --------
        Dict with:
            - label_distribution: Count of each label
            - avg_holding_period: Average bars held
            - barrier_hit_rates: Rate of each barrier type
            - avg_return_by_label: Mean return for each label
            - volatility_stats: Statistics of entry volatility
        """
        if self._events_cache is None or len(self._events_cache) == 0:
            return {'error': 'No events cached. Run apply_barriers first.'}
        
        df = pd.DataFrame(self._events_cache)
        
        diagnostics = {
            'n_events': len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'label_balance': {
                'positive_pct': (df['label'] == 1).mean(),
                'neutral_pct': (df['label'] == 0).mean(),
                'negative_pct': (df['label'] == -1).mean()
            },
            'avg_holding_period': df['holding_period'].mean(),
            'median_holding_period': df['holding_period'].median(),
            'max_holding_period': df['holding_period'].max(),
            'barrier_hit_rates': df['barrier_type'].value_counts(normalize=True).to_dict(),
            'avg_return_by_label': df.groupby('label')['ret'].mean().to_dict(),
            'std_return_by_label': df.groupby('label')['ret'].std().to_dict(),
            'volatility_stats': {
                'mean': df['volatility'].mean(),
                'std': df['volatility'].std(),
                'min': df['volatility'].min(),
                'max': df['volatility'].max()
            },
            'config': {
                'pt_sl': self.config.pt_sl,
                'max_holding_period': self.config.max_holding_period,
                'volatility_lookback': self.config.volatility_lookback,
                'min_return': self.config.min_return,
                'volatility_type': self.config.volatility_type
            }
        }
        
        # Quality checks
        diagnostics['quality_checks'] = {
            'balanced_labels': max(diagnostics['label_balance'].values()) < 0.7,
            'reasonable_holding': diagnostics['avg_holding_period'] <= self.config.max_holding_period,
            'positive_expected_return_for_positive_label': diagnostics['avg_return_by_label'].get(1, 0) > 0,
            'negative_expected_return_for_negative_label': diagnostics['avg_return_by_label'].get(-1, 0) < 0,
        }
        
        return diagnostics
    
    def plot_barriers(
        self,
        close: pd.Series,
        event_idx: int = 0,
        figsize: Tuple[int, int] = (14, 7)
    ):
        """
        Visualize barriers for a specific event.
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
        event_idx : int
            Index of event to visualize
        figsize : Tuple[int, int]
            Figure size
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib required for plotting")
            return
        
        if self._events_cache is None or event_idx >= len(self._events_cache):
            logger.error("No cached events or invalid index")
            return
        
        event = self._events_cache[event_idx]
        t0 = event['t0']
        t1 = event['t1']
        
        # Get price range to plot
        start_idx = max(0, close.index.get_loc(t0) - 5)
        end_idx = min(len(close), close.index.get_loc(t1) + 5)
        plot_range = close.iloc[start_idx:end_idx]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot price
        ax.plot(plot_range.index, plot_range.values, 'b-', linewidth=2, label='Close')
        
        # Plot barriers
        entry_price = close.loc[t0]
        barrier_range = pd.date_range(t0, t1)
        
        ax.axhline(event['upper_barrier'], color='green', linestyle='--', 
                   label=f'Upper Barrier ({event["upper_barrier"]:.2f})')
        ax.axhline(event['lower_barrier'], color='red', linestyle='--',
                   label=f'Lower Barrier ({event["lower_barrier"]:.2f})')
        ax.axvline(t1, color='gray', linestyle=':', alpha=0.7,
                   label=f'Vertical Barrier (t1)')
        
        # Mark entry and exit
        ax.scatter([t0], [entry_price], color='blue', s=100, zorder=5, 
                   label=f'Entry ({entry_price:.2f})')
        ax.scatter([t1], [close.loc[t1]], color='purple', s=100, zorder=5,
                   label=f'Exit ({close.loc[t1]:.2f})')
        
        # Title with outcome
        label_map = {1: 'PROFIT (+1)', -1: 'LOSS (-1)', 0: 'TIMEOUT (0)'}
        ax.set_title(
            f"Triple Barrier Example\n"
            f"Barrier Hit: {event['barrier_type'].upper()} | "
            f"Label: {label_map[event['label']]} | "
            f"Return: {event['ret']*100:.2f}%"
        )
        
        ax.legend(loc='best')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        plt.tight_layout()
        plt.show()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def apply_triple_barrier(
    close: pd.Series,
    pt_sl: Tuple[float, float] = (1.0, 1.0),
    max_holding_period: int = 10,
    volatility_lookback: int = 20,
    events: Optional[pd.DatetimeIndex] = None,
    side: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Apply triple barrier method to price data.
    
    Convenience function that creates a TripleBarrierLabeler and applies it.
    
    Parameters:
    -----------
    close : pd.Series
        Close prices with DatetimeIndex
    pt_sl : Tuple[float, float]
        (profit_take_mult, stop_loss_mult) relative to volatility
    max_holding_period : int
        Maximum bars to hold
    volatility_lookback : int
        Rolling window for volatility
    events : pd.DatetimeIndex, optional
        Specific events to label
    side : pd.Series, optional
        Side of bet (+1 long, -1 short)
        
    Returns:
    --------
    pd.DataFrame with barrier results
    """
    labeler = TripleBarrierLabeler(
        pt_sl=pt_sl,
        max_holding_period=max_holding_period,
        volatility_lookback=volatility_lookback
    )
    
    return labeler.apply_barriers(close, events, side)


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def validate_triple_barrier(
    labeler: TripleBarrierLabeler,
    close: pd.Series,
    expected_label_balance: float = 0.7,
    expected_avg_return_positive: bool = True
) -> Dict[str, Any]:
    """
    Validate triple barrier labeler on data.
    
    Parameters:
    -----------
    labeler : TripleBarrierLabeler
        Configured labeler
    close : pd.Series
        Test price data
    expected_label_balance : float
        Maximum allowed single label frequency
    expected_avg_return_positive : bool
        Whether positive labels should have positive average return
        
    Returns:
    --------
    Dict with validation results
    """
    # Apply barriers
    labels = labeler.get_labels(close)
    diagnostics = labeler.get_diagnostics()
    
    # Validation checks
    checks = {
        'has_all_label_types': all(
            label in labels.unique() for label in [-1, 0, 1]
        ),
        'balanced_labels': max(diagnostics['label_balance'].values()) < expected_label_balance,
        'positive_return_for_positive_label': 
            diagnostics['avg_return_by_label'].get(1, 0) > 0 if expected_avg_return_positive else True,
        'negative_return_for_negative_label':
            diagnostics['avg_return_by_label'].get(-1, 0) < 0,
        'reasonable_holding_period':
            0 < diagnostics['avg_holding_period'] <= labeler.config.max_holding_period,
        'no_nan_labels': not labels.isna().any(),
    }
    
    return {
        'passed': all(checks.values()),
        'checks': checks,
        'diagnostics': diagnostics
    }


if __name__ == '__main__':
    # Simple test
    import yfinance as yf
    
    print("Testing Triple Barrier Labeler...")
    
    # Download test data
    data = yf.download('SPY', start='2020-01-01', end='2024-01-01', progress=False)
    close = data['Close']
    
    # Create labeler
    labeler = TripleBarrierLabeler(
        pt_sl=(2.0, 1.0),  # Asymmetric: 2x profit, 1x stop
        max_holding_period=10,
        volatility_lookback=20
    )
    
    # Get labels
    labels = labeler.get_labels(close)
    
    # Print diagnostics
    diag = labeler.get_diagnostics()
    print(f"\nLabel Distribution: {diag['label_distribution']}")
    print(f"Average Holding Period: {diag['avg_holding_period']:.1f} days")
    print(f"Barrier Hit Rates: {diag['barrier_hit_rates']}")
    print(f"Avg Return by Label: {diag['avg_return_by_label']}")
    
    # Validate
    validation = validate_triple_barrier(labeler, close)
    print(f"\nValidation Passed: {validation['passed']}")
    print(f"Checks: {validation['checks']}")
