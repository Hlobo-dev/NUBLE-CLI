"""
AFML SAMPLE WEIGHTING
======================
Implements Lopez de Prado's sample weighting from Chapter 4.

The Problem:
- Triple Barrier labels often overlap in time
- Overlapping labels cause data leakage
- Information is redundant (same info counted multiple times)
- This leads to overfitting

The Solution:
- Weight each sample by its "uniqueness"
- More unique samples get higher weights
- Overlapping samples get lower weights
- Use weights in model training

Methods:
1. Concurrent Labels (AFML Ch 4.5.1)
2. Uniqueness (AFML Ch 4.5.2)
3. Average Uniqueness (AFML Ch 4.5.3)
4. Sequential Bootstrap (AFML Ch 4.5.4)

Reference:
López de Prado, M. (2018). Advances in Financial Machine Learning. Chapter 4.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from numba import jit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AFMLSampleWeights:
    """
    AFML Sample Weighting for Triple Barrier Labels.
    
    Reduces overfitting from overlapping labels by computing sample weights
    based on uniqueness of information.
    
    Example:
    --------
    >>> from src.institutional.labeling.triple_barrier import TripleBarrierLabeler
    >>> 
    >>> labeler = TripleBarrierLabeler(pt_sl=(2.0, 1.0), max_holding_period=10)
    >>> labels = labeler.get_labels(close)
    >>> 
    >>> # Get sample weights
    >>> weights = AFMLSampleWeights()
    >>> sample_weights = weights.get_weights(
    ...     event_times=labels.index,
    ...     t1_times=labeler.get_t1()  # End times for each label
    ... )
    >>> 
    >>> # Use weights in training
    >>> model.fit(X_train, y_train, sample_weight=sample_weights)
    """
    
    def __init__(self, decay_factor: float = 1.0):
        """
        Initialize sample weighter.
        
        Parameters:
        -----------
        decay_factor : float
            Optional time decay (1.0 = no decay)
            Values < 1.0 give more weight to recent samples
        """
        self.decay_factor = decay_factor
        
    def get_concurrent_events(
        self,
        close_idx: pd.DatetimeIndex,
        t0: pd.Series,
        t1: pd.Series
    ) -> pd.Series:
        """
        Count concurrent events at each time point (AFML 4.5.1).
        
        Parameters:
        -----------
        close_idx : pd.DatetimeIndex
            Index of price series (time points to evaluate)
        t0 : pd.Series
            Start times of events
        t1 : pd.Series
            End times of events
            
        Returns:
        --------
        pd.Series
            Number of concurrent events at each time point
        """
        # Build matrix of event spans
        t0_vals = t0.values
        t1_vals = t1.dropna().values
        
        # For each time point, count how many events span it
        concurrent = pd.Series(0.0, index=close_idx)
        
        for i, (start, end) in enumerate(zip(t0_vals, t1_vals)):
            if pd.isna(end):
                continue
            # Events active from start to end
            mask = (close_idx >= start) & (close_idx <= end)
            concurrent[mask] += 1
        
        return concurrent
    
    def get_uniqueness(
        self,
        event_times: pd.DatetimeIndex,
        t1_times: pd.Series,
        close_idx: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Compute average uniqueness for each event (AFML 4.5.2).
        
        An event is unique if it doesn't overlap with other events.
        Uniqueness = 1 / (number of concurrent events).
        
        Parameters:
        -----------
        event_times : pd.DatetimeIndex
            Start times of events
        t1_times : pd.Series
            End times of events (indexed by start time)
        close_idx : pd.DatetimeIndex
            Full price index
            
        Returns:
        --------
        pd.Series
            Average uniqueness for each event (0 to 1)
        """
        # Get concurrent events
        concurrent = self.get_concurrent_events(
            close_idx,
            pd.Series(event_times),
            t1_times
        )
        
        # For each event, compute average uniqueness over its span
        uniqueness = pd.Series(index=event_times, dtype=float)
        
        for t0 in event_times:
            if t0 not in t1_times.index:
                uniqueness[t0] = 0
                continue
                
            t1 = t1_times[t0]
            if pd.isna(t1):
                uniqueness[t0] = 0
                continue
            
            # Get concurrent during this event
            mask = (close_idx >= t0) & (close_idx <= t1)
            if mask.sum() == 0:
                uniqueness[t0] = 0
                continue
                
            # Average uniqueness = 1 / concurrent
            avg_u = (1.0 / concurrent[mask]).mean()
            uniqueness[t0] = avg_u
        
        return uniqueness
    
    def get_sample_weights(
        self,
        event_times: pd.DatetimeIndex,
        t1_times: pd.Series,
        close_idx: pd.DatetimeIndex,
        returns: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Get final sample weights (AFML 4.5.3).
        
        Combines uniqueness with optional absolute returns weighting.
        
        Parameters:
        -----------
        event_times : pd.DatetimeIndex
            Start times of events
        t1_times : pd.Series
            End times of events
        close_idx : pd.DatetimeIndex
            Full price index
        returns : pd.Series, optional
            Event returns for return attribution
            
        Returns:
        --------
        pd.Series
            Sample weights (normalized to sum to number of samples)
        """
        logger.info("Computing AFML sample weights...")
        
        # Get uniqueness
        uniqueness = self.get_uniqueness(event_times, t1_times, close_idx)
        
        # Apply decay
        if self.decay_factor != 1.0:
            # More recent samples get higher weight
            days_from_end = (event_times.max() - event_times).days
            decay = np.power(self.decay_factor, days_from_end / 365)
            uniqueness = uniqueness * decay
        
        # Optional: weight by absolute returns
        if returns is not None:
            abs_returns = returns.abs().reindex(event_times).fillna(0)
            # Combine with uniqueness
            uniqueness = uniqueness * (1 + abs_returns)
        
        # Normalize to sum to number of samples
        total = uniqueness.sum()
        if total > 0:
            weights = uniqueness * len(event_times) / total
        else:
            weights = pd.Series(1.0, index=event_times)
        
        logger.info(f"  Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
        logger.info(f"  Weight mean: {weights.mean():.2f}")
        
        return weights


class SequentialBootstrap:
    """
    Sequential Bootstrap for model bagging (AFML 4.5.4).
    
    Standard bootstrap creates redundant samples when events overlap.
    Sequential bootstrap samples based on uniqueness, ensuring
    each sample adds new information.
    
    Use this for RandomForest or GradientBoosting with overlapping labels.
    """
    
    def __init__(self, n_samples: int = None):
        """
        Initialize sequential bootstrap.
        
        Parameters:
        -----------
        n_samples : int, optional
            Number of samples to draw (default = length of data)
        """
        self.n_samples = n_samples
        
    def get_ind_matrix(
        self,
        bar_idx: pd.DatetimeIndex,
        t1: pd.Series
    ) -> pd.DataFrame:
        """
        Build indicator matrix of which bars belong to which labels.
        
        Parameters:
        -----------
        bar_idx : pd.DatetimeIndex
            Index of price bars
        t1 : pd.Series
            End times indexed by start times
            
        Returns:
        --------
        pd.DataFrame
            Binary matrix (bars × labels)
        """
        ind_matrix = pd.DataFrame(
            0,
            index=bar_idx,
            columns=range(len(t1))
        )
        
        for i, (t0, t1_val) in enumerate(t1.items()):
            if pd.isna(t1_val):
                continue
            mask = (bar_idx >= t0) & (bar_idx <= t1_val)
            ind_matrix.loc[mask, i] = 1
        
        return ind_matrix
    
    def get_avg_uniqueness(self, ind_matrix: pd.DataFrame) -> pd.Series:
        """
        Compute average uniqueness from indicator matrix.
        """
        # Concurrent = sum of each row
        concurrent = ind_matrix.sum(axis=1)
        
        # Uniqueness for each label
        uniqueness = pd.Series(index=ind_matrix.columns, dtype=float)
        
        for col in ind_matrix.columns:
            label_mask = ind_matrix[col] == 1
            if label_mask.sum() == 0:
                uniqueness[col] = 0
            else:
                uniqueness[col] = (1.0 / concurrent[label_mask]).mean()
        
        return uniqueness
    
    def sample(
        self,
        bar_idx: pd.DatetimeIndex,
        t1: pd.Series,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Sequential bootstrap sampling.
        
        Parameters:
        -----------
        bar_idx : pd.DatetimeIndex
            Index of price bars
        t1 : pd.Series
            End times indexed by start times
        random_state : int
            Random seed
            
        Returns:
        --------
        np.ndarray
            Indices of sampled labels
        """
        np.random.seed(random_state)
        
        n_labels = len(t1)
        n_samples = self.n_samples or n_labels
        
        # Build indicator matrix
        ind_matrix = self.get_ind_matrix(bar_idx, t1)
        
        # Track concurrent at each step
        concurrent = pd.Series(0.0, index=bar_idx)
        
        # Sample sequentially
        sampled = []
        
        for _ in range(n_samples):
            # Compute probability for each label based on current concurrent
            probs = pd.Series(1.0, index=range(n_labels))
            
            for i in range(n_labels):
                label_mask = ind_matrix[i] == 1
                if label_mask.sum() == 0:
                    probs[i] = 0
                    continue
                # Weight by inverse of concurrent
                avg_concurrent = concurrent[label_mask].mean() + 1
                probs[i] = 1.0 / avg_concurrent
            
            # Normalize
            probs = probs / probs.sum()
            
            # Sample
            chosen = np.random.choice(n_labels, p=probs.values)
            sampled.append(chosen)
            
            # Update concurrent
            label_mask = ind_matrix[chosen] == 1
            concurrent[label_mask] += 1
        
        return np.array(sampled)


def demo_sample_weights():
    """Demonstrate AFML sample weighting."""
    import sys
    sys.path.insert(0, '/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI')
    
    from src.institutional.labeling.triple_barrier import TripleBarrierLabeler
    
    # Load data
    df = pd.read_csv(
        '/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI/data/train/SPY.csv',
        index_col=0, parse_dates=True
    )
    
    print("="*60)
    print("AFML SAMPLE WEIGHTS DEMONSTRATION")
    print("="*60)
    
    # Generate labels
    labeler = TripleBarrierLabeler(pt_sl=(2.0, 1.0), max_holding_period=10)
    labels = labeler.get_labels(df['close'])
    
    print(f"\nLabels: {len(labels)} events")
    print(f"Label distribution: {labels.value_counts().to_dict()}")
    
    # Get t1 (end times) - we need to access this from labeler
    # For now, create synthetic t1 based on holding period
    t1 = pd.Series(index=labels.index, dtype='datetime64[ns]')
    for i, idx in enumerate(labels.index):
        pos = df.index.get_loc(idx)
        end_pos = min(pos + 10, len(df) - 1)
        t1[idx] = df.index[end_pos]
    
    # Compute weights
    weighter = AFMLSampleWeights()
    weights = weighter.get_sample_weights(
        event_times=labels.index,
        t1_times=t1,
        close_idx=df.index
    )
    
    print(f"\nSample Weights:")
    print(f"  Min weight: {weights.min():.3f}")
    print(f"  Max weight: {weights.max():.3f}")
    print(f"  Mean weight: {weights.mean():.3f}")
    print(f"  Std weight: {weights.std():.3f}")
    
    # Show distribution
    print(f"\nWeight Distribution:")
    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 100]
    for i in range(len(bins) - 1):
        count = ((weights >= bins[i]) & (weights < bins[i+1])).sum()
        pct = count / len(weights) * 100
        print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count} ({pct:.1f}%)")
    
    # Compare weighted vs unweighted
    print(f"\nEffect of Weighting:")
    print(f"  Without weights: All samples contribute equally")
    print(f"  With weights: Unique samples contribute {weights.max()/weights.mean():.1f}x more")
    
    return weights


if __name__ == "__main__":
    demo_sample_weights()
