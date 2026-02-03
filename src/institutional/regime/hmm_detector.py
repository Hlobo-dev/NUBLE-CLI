"""
Hidden Markov Model Regime Detection
=====================================

Implementation of HMM-based market regime detection from:
Lopez de Prado, "Advances in Financial Machine Learning" (2018), Chapter 10.

Market regimes represent underlying states that generate observed returns:
- Bull Market: High mean return, moderate volatility
- Bear Market: Negative mean return, high volatility
- Sideways/Range: Low mean return, low volatility

HMMs learn these latent states from observable returns and can:
1. Classify current market regime
2. Predict regime transitions
3. Filter trades based on favorable regimes

Key Properties:
- Online prediction (no lookahead bias)
- Probabilistic regime assignment
- Transition probability estimation
- Automatic regime characterization

References:
-----------
[1] López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 10.
[2] Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of 
    Nonstationary Time Series." Econometrica.
[3] Kim, C.J. (1994). "Dynamic Linear Models with Markov-Switching." 
    Journal of Econometrics.

Author: NUBLE Institutional
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings

# Try to import hmmlearn
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logging.warning("hmmlearn not available. Install with: pip install hmmlearn")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION AND TYPES
# =============================================================================

class RegimeState(Enum):
    """Standard regime states."""
    BULL = 0       # High return, moderate vol
    BEAR = 1       # Negative return, high vol
    SIDEWAYS = 2   # Low return, low vol
    UNKNOWN = -1   # Undetermined


@dataclass
class RegimeConfig:
    """
    Configuration for HMM Regime Detection.
    
    Attributes:
    -----------
    n_regimes : int
        Number of hidden states (regimes). Default 2 (bull/bear).
        Use 3 for bull/bear/sideways.
        
    covariance_type : str
        Type of covariance matrix:
        - 'full': Each state has full covariance matrix
        - 'tied': All states share same covariance
        - 'diag': Diagonal covariance (features independent)
        - 'spherical': Single variance per state
        
    n_iter : int
        Maximum EM algorithm iterations. Default 100.
        
    tol : float
        Convergence tolerance for EM. Default 1e-4.
        
    random_state : int
        Random seed for reproducibility.
        
    lookback : int
        Minimum lookback for regime estimation.
        
    min_regime_duration : int
        Minimum bars in a regime before switching (reduces noise).
    """
    n_regimes: int = 2
    covariance_type: str = 'full'
    n_iter: int = 100
    tol: float = 1e-4
    random_state: int = 42
    lookback: int = 252  # 1 year for daily data
    min_regime_duration: int = 5
    
    def __post_init__(self):
        """Validate configuration."""
        if self.n_regimes < 2:
            raise ValueError("n_regimes must be >= 2")
        if self.n_regimes > 5:
            logger.warning("More than 5 regimes may lead to overfitting")
        
        valid_cov_types = ['full', 'tied', 'diag', 'spherical']
        if self.covariance_type not in valid_cov_types:
            raise ValueError(f"covariance_type must be one of {valid_cov_types}")


@dataclass
class RegimeStatistics:
    """Statistics for a single regime."""
    regime_id: int
    name: str
    mean_return: float      # Annualized mean return
    volatility: float       # Annualized volatility
    sharpe_ratio: float     # Regime Sharpe ratio
    frequency: float        # % of time in this regime
    avg_duration: float     # Average bars in regime
    n_occurrences: int      # Number of regime switches to this state


# =============================================================================
# HMM REGIME DETECTOR
# =============================================================================

class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection.
    
    Identifies latent market states (bull, bear, sideways) from observable
    returns. Used to filter trades in unfavorable regimes and adapt
    strategy parameters to current conditions.
    
    Key Features:
    - Online prediction without lookahead bias
    - Automatic regime characterization by Sharpe
    - Transition probability analysis
    - Regime filtering for meta-labeling
    
    Example:
    --------
    >>> detector = HMMRegimeDetector(n_regimes=2)
    >>> detector.fit(returns)
    >>> 
    >>> # Get current regime
    >>> current_regime = detector.predict(returns)
    >>> print(f"Current regime: {detector.regime_names[current_regime.iloc[-1]]}")
    >>> 
    >>> # Get trading filter (only trade in favorable regimes)
    >>> trade_filter = detector.get_trading_filter(returns, allowed_regimes=[0])
    >>> 
    >>> # Apply filter to signals
    >>> filtered_signals = signals * trade_filter
    
    Mathematical Background:
    ------------------------
    HMM assumes observations Y_t are generated by hidden states S_t:
    
    P(S_t | S_{t-1}) = A[S_{t-1}, S_t]  (transition probabilities)
    P(Y_t | S_t) = N(μ_{S_t}, σ_{S_t})  (emission probabilities)
    
    The Baum-Welch (EM) algorithm estimates A, μ, σ from data.
    The Viterbi algorithm finds most likely state sequence.
    Forward algorithm computes P(S_t | Y_1, ..., Y_t) for online prediction.
    """
    
    def __init__(
        self,
        n_regimes: int = 2,
        covariance_type: str = 'full',
        n_iter: int = 100,
        random_state: int = 42,
        min_regime_duration: int = 5
    ):
        """
        Initialize HMM Regime Detector.
        
        Parameters:
        -----------
        n_regimes : int
            Number of market regimes (default 2: bull/bear)
            
        covariance_type : str
            Covariance structure: 'full', 'tied', 'diag', 'spherical'
            
        n_iter : int
            Max EM iterations
            
        random_state : int
            Random seed
            
        min_regime_duration : int
            Minimum bars before regime switch
        """
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn required for HMM regime detection. "
                "Install with: pip install hmmlearn"
            )
        
        self.config = RegimeConfig(
            n_regimes=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            min_regime_duration=min_regime_duration
        )
        
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            tol=1e-4
        )
        
        # Fitted parameters
        self.is_fitted = False
        self.regime_stats: Dict[int, RegimeStatistics] = {}
        self.regime_order: np.ndarray = np.arange(n_regimes)  # Order by Sharpe
        self.regime_names: Dict[int, str] = {}
        self.transition_matrix: Optional[np.ndarray] = None
        
        # Training data for diagnostics
        self._train_returns: Optional[pd.Series] = None
        self._train_states: Optional[np.ndarray] = None
        
        logger.info(
            f"HMMRegimeDetector initialized: {n_regimes} regimes, "
            f"covariance={covariance_type}"
        )
    
    def fit(
        self,
        returns: pd.Series,
        additional_features: Optional[pd.DataFrame] = None
    ) -> 'HMMRegimeDetector':
        """
        Fit HMM on historical returns.
        
        Parameters:
        -----------
        returns : pd.Series
            Daily returns series (simple or log returns)
            
        additional_features : pd.DataFrame, optional
            Additional features for multi-variate HMM.
            Must be aligned with returns index.
            
        Returns:
        --------
        self : HMMRegimeDetector
        """
        # Prepare data
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 100:
            raise ValueError(f"Need at least 100 observations, got {len(clean_returns)}")
        
        # Build feature matrix
        if additional_features is not None:
            # Multi-variate HMM
            X = additional_features.reindex(clean_returns.index).dropna()
            X['returns'] = clean_returns
            X = X.dropna().values
        else:
            # Univariate HMM on returns
            X = clean_returns.values.reshape(-1, 1)
        
        # Fit HMM
        logger.info(f"Fitting HMM on {len(X)} observations...")
        self.model.fit(X)
        
        # Get state sequence
        states = self.model.predict(X)
        
        # Calculate regime statistics
        self._compute_regime_statistics(clean_returns, states)
        
        # Order regimes by Sharpe (best = 0)
        self._order_regimes_by_sharpe()
        
        # Name regimes
        self._name_regimes()
        
        # Store transition matrix
        self.transition_matrix = self.model.transmat_
        
        # Cache training data
        self._train_returns = clean_returns
        self._train_states = states
        
        self.is_fitted = True
        logger.info(f"HMM fit complete. Regimes: {self.regime_names}")
        
        return self
    
    def _compute_regime_statistics(
        self,
        returns: pd.Series,
        states: np.ndarray
    ):
        """Compute statistics for each regime."""
        for regime_id in range(self.config.n_regimes):
            mask = states == regime_id
            regime_returns = returns.iloc[mask]
            
            if len(regime_returns) < 2:
                logger.warning(f"Regime {regime_id} has too few observations")
                # Create placeholder stats for degenerate regimes
                self.regime_stats[regime_id] = RegimeStatistics(
                    regime_id=regime_id,
                    name=f"Regime_{regime_id}",
                    mean_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    frequency=0.0,
                    avg_duration=0.0,
                    n_occurrences=0
                )
                continue
            
            # Annualize metrics (assuming daily data)
            mean_return = regime_returns.mean() * 252
            volatility = regime_returns.std() * np.sqrt(252)
            sharpe = mean_return / volatility if volatility > 0 else 0
            
            # Frequency and duration
            frequency = mask.sum() / len(states)
            
            # Count regime switches - fixed indexing
            state_changes = np.diff(states)
            # Regime starts when previous state != current and current == regime_id
            regime_starts = np.where((state_changes != 0) & (states[1:] == regime_id))[0]
            n_occurrences = len(regime_starts) + (1 if states[0] == regime_id else 0)
            
            # Average duration
            avg_duration = mask.sum() / max(n_occurrences, 1)
            
            self.regime_stats[regime_id] = RegimeStatistics(
                regime_id=regime_id,
                name=f"Regime_{regime_id}",
                mean_return=mean_return,
                volatility=volatility,
                sharpe_ratio=sharpe,
                frequency=frequency,
                avg_duration=avg_duration,
                n_occurrences=n_occurrences
            )
    
    def _order_regimes_by_sharpe(self):
        """Order regimes so index 0 = best Sharpe."""
        sharpes = [self.regime_stats[i].sharpe_ratio for i in range(self.config.n_regimes)]
        self.regime_order = np.argsort(sharpes)[::-1]  # Descending order
    
    def _name_regimes(self):
        """Assign descriptive names to regimes."""
        n = self.config.n_regimes
        
        if n == 2:
            # Bull/Bear classification
            for ordered_idx, regime_id in enumerate(self.regime_order):
                if ordered_idx == 0:
                    self.regime_names[regime_id] = "BULL"
                else:
                    self.regime_names[regime_id] = "BEAR"
                self.regime_stats[regime_id].name = self.regime_names[regime_id]
        
        elif n == 3:
            # Bull/Sideways/Bear classification
            for ordered_idx, regime_id in enumerate(self.regime_order):
                if ordered_idx == 0:
                    self.regime_names[regime_id] = "BULL"
                elif ordered_idx == 1:
                    self.regime_names[regime_id] = "SIDEWAYS"
                else:
                    self.regime_names[regime_id] = "BEAR"
                self.regime_stats[regime_id].name = self.regime_names[regime_id]
        
        else:
            # Generic naming by Sharpe ranking
            for ordered_idx, regime_id in enumerate(self.regime_order):
                sharpe = self.regime_stats[regime_id].sharpe_ratio
                self.regime_names[regime_id] = f"REGIME_{ordered_idx+1} (SR={sharpe:.2f})"
                self.regime_stats[regime_id].name = self.regime_names[regime_id]
    
    def predict(
        self,
        returns: pd.Series,
        method: str = 'viterbi'
    ) -> pd.Series:
        """
        Predict regime for each observation.
        
        Parameters:
        -----------
        returns : pd.Series
            Returns series
            
        method : str
            Prediction method:
            - 'viterbi': Most likely state sequence (uses full sequence)
            - 'filter': Online filtering (no lookahead)
            
        Returns:
        --------
        pd.Series : Regime labels (indexed by Sharpe: 0 = best)
        
        Notes:
        ------
        For trading, use 'filter' method to avoid lookahead bias.
        'viterbi' is useful for backtesting analysis only.
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before predict()")
        
        clean_returns = returns.dropna()
        X = clean_returns.values.reshape(-1, 1)
        
        if method == 'viterbi':
            # Most likely state sequence (uses entire series)
            raw_states = self.model.predict(X)
        else:
            # Online filtering (no lookahead)
            raw_states = self._online_predict(X)
        
        # Map to ordered states (0 = best Sharpe)
        ordered_states = np.zeros_like(raw_states)
        for ordered_idx, regime_id in enumerate(self.regime_order):
            ordered_states[raw_states == regime_id] = ordered_idx
        
        return pd.Series(ordered_states, index=clean_returns.index, name='regime')
    
    def _online_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Online regime prediction without lookahead.
        
        Uses forward algorithm to compute P(S_t | Y_1, ..., Y_t).
        """
        n_samples = len(X)
        states = np.zeros(n_samples, dtype=np.int32)
        
        # Initial state probabilities
        log_startprob = np.log(self.model.startprob_ + 1e-10)
        log_transmat = np.log(self.model.transmat_ + 1e-10)
        
        # Forward pass
        alpha = log_startprob + self._compute_log_likelihood(X[0])
        states[0] = np.argmax(alpha)
        
        for t in range(1, n_samples):
            # Compute forward probability
            alpha_new = np.zeros(self.config.n_regimes)
            for j in range(self.config.n_regimes):
                alpha_new[j] = np.logaddexp.reduce(alpha + log_transmat[:, j])
            
            alpha = alpha_new + self._compute_log_likelihood(X[t])
            states[t] = np.argmax(alpha)
        
        return states
    
    def _compute_log_likelihood(self, obs: np.ndarray) -> np.ndarray:
        """Compute log emission probability for observation."""
        ll = np.zeros(self.config.n_regimes)
        
        for k in range(self.config.n_regimes):
            mean = self.model.means_[k]
            
            if self.config.covariance_type == 'full':
                cov = self.model.covars_[k]
            elif self.config.covariance_type == 'diag':
                cov = np.diag(self.model.covars_[k])
            elif self.config.covariance_type == 'spherical':
                cov = np.eye(len(mean)) * self.model.covars_[k]
            else:  # tied
                cov = self.model.covars_
            
            # Multivariate normal log probability
            diff = obs - mean
            try:
                inv_cov = np.linalg.inv(cov)
                det_cov = np.linalg.det(cov)
                ll[k] = -0.5 * (np.dot(diff, np.dot(inv_cov, diff)) + 
                               np.log(det_cov) + len(mean) * np.log(2 * np.pi))
            except np.linalg.LinAlgError:
                ll[k] = -np.inf
        
        return ll
    
    def predict_proba(
        self,
        returns: pd.Series
    ) -> pd.DataFrame:
        """
        Get probability of each regime.
        
        Parameters:
        -----------
        returns : pd.Series
            Returns series
            
        Returns:
        --------
        pd.DataFrame : Columns for each regime probability
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before predict_proba()")
        
        clean_returns = returns.dropna()
        X = clean_returns.values.reshape(-1, 1)
        
        # Get posterior probabilities
        proba = self.model.predict_proba(X)
        
        # Reorder columns by Sharpe
        proba_ordered = proba[:, self.regime_order]
        
        columns = [f'regime_{i}_prob' for i in range(self.config.n_regimes)]
        
        return pd.DataFrame(
            proba_ordered,
            index=clean_returns.index,
            columns=columns
        )
    
    def get_trading_filter(
        self,
        returns: pd.Series,
        allowed_regimes: List[int] = None,
        method: str = 'filter'
    ) -> pd.Series:
        """
        Create binary trading filter based on regime.
        
        Parameters:
        -----------
        returns : pd.Series
            Returns series
            
        allowed_regimes : List[int]
            Which regimes to allow trading.
            Default [0] = best regime only.
            Use [0, 1] to include top 2 regimes.
            
        method : str
            Prediction method ('filter' for online)
            
        Returns:
        --------
        pd.Series : 1 = trade allowed, 0 = no trade
        
        Example:
        --------
        >>> filter = detector.get_trading_filter(returns, allowed_regimes=[0])
        >>> # Apply to signals
        >>> filtered_signals = raw_signals * filter
        """
        if allowed_regimes is None:
            allowed_regimes = [0]  # Best regime only
        
        regimes = self.predict(returns, method=method)
        trading_filter = regimes.isin(allowed_regimes).astype(int)
        
        # Apply minimum duration filter
        if self.config.min_regime_duration > 1:
            trading_filter = self._apply_duration_filter(trading_filter)
        
        return trading_filter
    
    def _apply_duration_filter(self, filter_series: pd.Series) -> pd.Series:
        """Apply minimum duration requirement to filter."""
        min_dur = self.config.min_regime_duration
        result = filter_series.copy()
        
        # Find regime change points
        changes = filter_series.diff().fillna(0).abs()
        change_points = changes[changes > 0].index.tolist()
        
        # Add start point
        change_points = [filter_series.index[0]] + change_points
        
        # Check duration of each segment
        for i in range(len(change_points) - 1):
            start = change_points[i]
            end = change_points[i + 1]
            segment = filter_series.loc[start:end]
            
            if len(segment) < min_dur:
                # Too short, use previous value
                if i > 0:
                    result.loc[start:end] = result.loc[change_points[i - 1]]
        
        return result
    
    def get_regime_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for each regime.
        
        Returns:
        --------
        pd.DataFrame with regime statistics
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() first")
        
        rows = []
        for regime_id in range(self.config.n_regimes):
            stats = self.regime_stats[regime_id]
            ordered_idx = np.where(self.regime_order == regime_id)[0][0]
            
            rows.append({
                'regime_id': regime_id,
                'ordered_idx': ordered_idx,
                'name': stats.name,
                'mean_return': stats.mean_return,
                'volatility': stats.volatility,
                'sharpe_ratio': stats.sharpe_ratio,
                'frequency': stats.frequency,
                'avg_duration': stats.avg_duration,
                'n_occurrences': stats.n_occurrences
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('ordered_idx').reset_index(drop=True)
        
        return df
    
    def get_transition_probabilities(self) -> pd.DataFrame:
        """
        Get regime transition probability matrix.
        
        Returns:
        --------
        pd.DataFrame : Transition matrix (row = from, col = to)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() first")
        
        labels = [self.regime_names[i] for i in range(self.config.n_regimes)]
        
        return pd.DataFrame(
            self.transition_matrix,
            index=labels,
            columns=labels
        )
    
    def get_expected_regime_duration(self) -> Dict[int, float]:
        """
        Calculate expected duration in each regime.
        
        Based on transition matrix: E[duration] = 1 / (1 - P_ii)
        
        Returns:
        --------
        Dict[int, float] : Expected bars in each regime
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() first")
        
        durations = {}
        for i in range(self.config.n_regimes):
            p_stay = self.transition_matrix[i, i]
            expected_duration = 1 / (1 - p_stay) if p_stay < 1 else np.inf
            durations[i] = expected_duration
        
        return durations
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostics.
        
        Returns:
        --------
        Dict with model diagnostics
        """
        if not self.is_fitted:
            return {'error': 'Not fitted'}
        
        return {
            'n_regimes': self.config.n_regimes,
            'covariance_type': self.config.covariance_type,
            'regime_statistics': self.get_regime_statistics().to_dict('records'),
            'transition_matrix': self.transition_matrix.tolist(),
            'expected_durations': self.get_expected_regime_duration(),
            'regime_names': self.regime_names,
            'model_score': self.model.score(
                self._train_returns.values.reshape(-1, 1)
            ) if self._train_returns is not None else None
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_regime_features(
    returns: pd.Series,
    n_regimes: int = 2,
    lookback: int = 252
) -> pd.DataFrame:
    """
    Create regime-based features for ML models.
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns
    n_regimes : int
        Number of regimes
    lookback : int
        Training window for rolling HMM
        
    Returns:
    --------
    pd.DataFrame with columns:
        - regime: Current regime (0 = best)
        - regime_prob_0: Probability of regime 0
        - regime_prob_1: Probability of regime 1
        - regime_duration: Bars in current regime
        - regime_mean: Mean return of current regime
        - regime_vol: Volatility of current regime
    """
    detector = HMMRegimeDetector(n_regimes=n_regimes)
    
    # Fit on available data
    detector.fit(returns)
    
    # Get predictions
    regimes = detector.predict(returns, method='filter')
    proba = detector.predict_proba(returns)
    
    # Build features
    features = pd.DataFrame(index=returns.index)
    features['regime'] = regimes
    
    # Add probabilities
    for col in proba.columns:
        features[col] = proba[col]
    
    # Regime duration (bars since last switch)
    regime_changes = regimes.diff().fillna(0).abs()
    features['regime_duration'] = regime_changes.groupby(
        (regime_changes != 0).cumsum()
    ).cumcount() + 1
    
    # Regime statistics as features
    features['regime_mean'] = 0.0
    features['regime_vol'] = 0.0
    
    for regime_id in range(n_regimes):
        mask = regimes == regime_id
        if regime_id in detector.regime_stats:
            stats = detector.regime_stats[regime_id]
            features.loc[mask, 'regime_mean'] = stats.mean_return / 252  # Daily
            features.loc[mask, 'regime_vol'] = stats.volatility / np.sqrt(252)
    
    return features


def plot_regimes(
    returns: pd.Series,
    prices: Optional[pd.Series] = None,
    detector: Optional[HMMRegimeDetector] = None,
    n_regimes: int = 2,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Plot regime analysis visualization.
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns
    prices : pd.Series, optional
        Price series for overlay
    detector : HMMRegimeDetector, optional
        Pre-fitted detector
    n_regimes : int
        Number of regimes if detector not provided
    figsize : Tuple[int, int]
        Figure size
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.error("matplotlib required for plotting")
        return
    
    # Fit detector if not provided
    if detector is None:
        detector = HMMRegimeDetector(n_regimes=n_regimes)
        detector.fit(returns)
    
    # Get predictions
    regimes = detector.predict(returns, method='filter')
    
    # Use prices or cumulative returns
    if prices is None:
        prices = (1 + returns).cumprod()
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize, 
                             gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # 1. Price with regime shading
    ax1 = axes[0]
    ax1.plot(prices.index, prices.values, 'b-', linewidth=1, alpha=0.8)
    
    # Color regimes
    colors = ['green', 'red', 'yellow', 'orange', 'purple'][:n_regimes]
    
    for regime_id in range(n_regimes):
        mask = regimes == regime_id
        if mask.any():
            # Find contiguous regime periods
            regime_periods = mask.astype(int).diff().fillna(0)
            starts = mask.index[regime_periods == 1].tolist()
            ends = mask.index[regime_periods == -1].tolist()
            
            # Handle edge cases
            if mask.iloc[0]:
                starts = [mask.index[0]] + starts
            if mask.iloc[-1]:
                ends = ends + [mask.index[-1]]
            
            for start, end in zip(starts, ends):
                ax1.axvspan(start, end, alpha=0.3, 
                           color=colors[regime_id],
                           label=detector.regime_names.get(regime_id, f'Regime {regime_id}'))
    
    ax1.set_title('Price with Regime Shading')
    ax1.set_ylabel('Price')
    
    # Remove duplicate legend entries
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    # 2. Regime sequence
    ax2 = axes[1]
    ax2.plot(regimes.index, regimes.values, 'k-', linewidth=0.5)
    ax2.scatter(regimes.index, regimes.values, c=[colors[int(r)] for r in regimes.values],
               s=5, alpha=0.5)
    ax2.set_ylabel('Regime')
    ax2.set_title('Regime Sequence')
    ax2.set_yticks(range(n_regimes))
    ax2.set_yticklabels([detector.regime_names.get(i, f'R{i}') for i in range(n_regimes)])
    
    # 3. Regime probabilities
    ax3 = axes[2]
    proba = detector.predict_proba(returns)
    for i, col in enumerate(proba.columns):
        ax3.fill_between(proba.index, 0, proba[col].values, 
                        alpha=0.5, color=colors[i],
                        label=detector.regime_names.get(i, f'R{i}'))
    ax3.set_ylabel('Probability')
    ax3.set_xlabel('Date')
    ax3.set_title('Regime Probabilities')
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("REGIME STATISTICS")
    print("="*60)
    print(detector.get_regime_statistics().to_string(index=False))
    print("\nTransition Matrix:")
    print(detector.get_transition_probabilities().to_string())


if __name__ == '__main__':
    # Test HMM Regime Detector
    import yfinance as yf
    
    print("Testing HMM Regime Detector...")
    
    # Download test data
    data = yf.download('SPY', start='2015-01-01', end='2024-01-01', progress=False)
    returns = data['Close'].pct_change().dropna()
    prices = data['Close']
    
    # Create detector
    detector = HMMRegimeDetector(n_regimes=2)
    
    # Fit
    detector.fit(returns)
    
    # Print diagnostics
    print("\nRegime Statistics:")
    print(detector.get_regime_statistics())
    
    print("\nTransition Matrix:")
    print(detector.get_transition_probabilities())
    
    print("\nExpected Durations:")
    for regime_id, duration in detector.get_expected_regime_duration().items():
        print(f"  {detector.regime_names[regime_id]}: {duration:.1f} days")
    
    # Test trading filter
    trading_filter = detector.get_trading_filter(returns, allowed_regimes=[0])
    print(f"\nTrading filter: {trading_filter.sum()} / {len(trading_filter)} days allowed")
    
    # Compare filtered vs unfiltered returns
    filtered_returns = returns * trading_filter.shift(1).fillna(0)
    
    unfiltered_sharpe = returns.mean() / returns.std() * np.sqrt(252)
    filtered_sharpe = filtered_returns.mean() / filtered_returns.std() * np.sqrt(252)
    
    print(f"\nSharpe Comparison:")
    print(f"  Unfiltered: {unfiltered_sharpe:.2f}")
    print(f"  Filtered (best regime only): {filtered_sharpe:.2f}")
    
    print("\n✅ HMM Regime Detector test complete!")
