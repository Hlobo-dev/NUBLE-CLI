"""
Market Regime Detection
========================

Advanced regime detection using:
- Hidden Markov Models (HMM)
- Change point detection
- Volatility regime classification
- Trend/Range detection

Identifies market states: Bull, Bear, High Volatility, Low Volatility, Ranging
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class RegimeState(Enum):
    """Market regime states"""
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RANGING = "ranging"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


@dataclass
class RegimeDetection:
    """Result of regime detection"""
    current_regime: RegimeState
    regime_probability: float
    all_probabilities: Dict[RegimeState, float]
    regime_start_date: Optional[str] = None
    days_in_regime: int = 0
    transition_probabilities: Optional[Dict[str, float]] = None
    features_used: List[str] = field(default_factory=list)


class HMMRegimeModel:
    """
    Hidden Markov Model for market regime detection.
    
    States represent unobservable market regimes.
    Observations are market features (returns, volatility, etc.)
    
    Uses Baum-Welch algorithm for parameter estimation
    and Viterbi algorithm for state sequence inference.
    """
    
    def __init__(
        self,
        n_states: int = 3,
        n_features: int = 4,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-4
    ):
        """
        Initialize HMM.
        
        Args:
            n_states: Number of hidden states (regimes)
            n_features: Number of observation features
            max_iterations: Max EM iterations
            convergence_threshold: Convergence threshold for EM
        """
        self.n_states = n_states
        self.n_features = n_features
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Initialize parameters
        self._initialize_parameters()
        
        # State names
        self.state_names = [RegimeState.BULL, RegimeState.BEAR, RegimeState.RANGING][:n_states]
    
    def _initialize_parameters(self):
        """Initialize HMM parameters"""
        # Initial state probabilities (uniform)
        self.pi = np.ones(self.n_states) / self.n_states
        
        # Transition matrix (with slight persistence bias)
        self.A = np.ones((self.n_states, self.n_states)) * 0.1 / (self.n_states - 1)
        np.fill_diagonal(self.A, 0.9)  # States tend to persist
        
        # Emission parameters (Gaussian): means and covariances for each state
        self.means = np.zeros((self.n_states, self.n_features))
        self.covars = np.array([np.eye(self.n_features) for _ in range(self.n_states)])
        
        # Set distinct initial means for different regimes
        if self.n_states >= 2:
            self.means[0] = [0.001, 0.01, 0, 1]  # Bull: positive return, low vol
            self.means[1] = [-0.001, 0.02, 0, -1]  # Bear: negative return, high vol
        if self.n_states >= 3:
            self.means[2] = [0, 0.015, 0, 0]  # Ranging: neutral
    
    def _gaussian_pdf(
        self,
        x: np.ndarray,
        mean: np.ndarray,
        covar: np.ndarray
    ) -> float:
        """Multivariate Gaussian PDF"""
        k = len(mean)
        
        # Handle singular covariance
        try:
            covar_inv = np.linalg.inv(covar + 1e-6 * np.eye(k))
            det = np.linalg.det(covar + 1e-6 * np.eye(k))
        except np.linalg.LinAlgError:
            return 1e-10
        
        if det <= 0:
            det = 1e-10
        
        diff = x - mean
        exponent = -0.5 * np.dot(diff, np.dot(covar_inv, diff))
        
        return np.exp(exponent) / (np.sqrt((2 * np.pi) ** k * det) + 1e-10)
    
    def _compute_emission_probs(self, observations: np.ndarray) -> np.ndarray:
        """Compute emission probabilities for all observations and states"""
        T = len(observations)
        B = np.zeros((T, self.n_states))
        
        for t in range(T):
            for s in range(self.n_states):
                B[t, s] = self._gaussian_pdf(
                    observations[t],
                    self.means[s],
                    self.covars[s]
                )
        
        # Avoid zeros
        B = np.clip(B, 1e-10, None)
        
        return B
    
    def _forward_algorithm(
        self,
        observations: np.ndarray,
        emission_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward algorithm for computing P(O|λ).
        
        Returns:
            alpha: Forward probabilities
            scaling: Scaling factors for numerical stability
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        scaling = np.zeros(T)
        
        # Initialization
        alpha[0] = self.pi * emission_probs[0]
        scaling[0] = np.sum(alpha[0])
        if scaling[0] > 0:
            alpha[0] /= scaling[0]
        
        # Induction
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * emission_probs[t, j]
            
            scaling[t] = np.sum(alpha[t])
            if scaling[t] > 0:
                alpha[t] /= scaling[t]
        
        return alpha, scaling
    
    def _backward_algorithm(
        self,
        observations: np.ndarray,
        emission_probs: np.ndarray,
        scaling: np.ndarray
    ) -> np.ndarray:
        """Backward algorithm for Baum-Welch"""
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # Initialization
        beta[T-1] = 1
        
        # Induction
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.A[i, :] * emission_probs[t+1, :] * beta[t+1, :]
                )
            if scaling[t+1] > 0:
                beta[t] /= scaling[t+1]
        
        return beta
    
    def _viterbi_algorithm(
        self,
        observations: np.ndarray,
        emission_probs: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Viterbi algorithm for finding most likely state sequence.
        
        Returns:
            path: Most likely state sequence
            prob: Log probability of the path
        """
        T = len(observations)
        
        # Log probabilities for numerical stability
        log_A = np.log(self.A + 1e-10)
        log_pi = np.log(self.pi + 1e-10)
        log_B = np.log(emission_probs + 1e-10)
        
        # Viterbi variables
        viterbi = np.zeros((T, self.n_states))
        backpointer = np.zeros((T, self.n_states), dtype=int)
        
        # Initialization
        viterbi[0] = log_pi + log_B[0]
        
        # Recursion
        for t in range(1, T):
            for s in range(self.n_states):
                trans_probs = viterbi[t-1] + log_A[:, s]
                backpointer[t, s] = np.argmax(trans_probs)
                viterbi[t, s] = np.max(trans_probs) + log_B[t, s]
        
        # Termination
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(viterbi[T-1])
        prob = viterbi[T-1, path[T-1]]
        
        # Backtrack
        for t in range(T-2, -1, -1):
            path[t] = backpointer[t+1, path[t+1]]
        
        return path, prob
    
    def fit(self, observations: np.ndarray):
        """
        Fit HMM parameters using Baum-Welch algorithm.
        
        Args:
            observations: Observation sequence (T, n_features)
        """
        T = len(observations)
        
        for iteration in range(self.max_iterations):
            # E-step
            emission_probs = self._compute_emission_probs(observations)
            alpha, scaling = self._forward_algorithm(observations, emission_probs)
            beta = self._backward_algorithm(observations, emission_probs, scaling)
            
            # Compute gamma (state occupation probabilities)
            gamma = alpha * beta
            gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-10)
            
            # Compute xi (transition probabilities)
            xi = np.zeros((T-1, self.n_states, self.n_states))
            for t in range(T-1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = (
                            alpha[t, i] * 
                            self.A[i, j] * 
                            emission_probs[t+1, j] * 
                            beta[t+1, j]
                        )
                xi[t] /= (xi[t].sum() + 1e-10)
            
            # M-step
            old_A = self.A.copy()
            
            # Update initial probabilities
            self.pi = gamma[0]
            
            # Update transition matrix
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.A[i, j] = xi[:, i, j].sum() / (gamma[:-1, i].sum() + 1e-10)
            
            # Update emission parameters
            for s in range(self.n_states):
                weights = gamma[:, s]
                total_weight = weights.sum() + 1e-10
                
                # Update mean
                self.means[s] = np.sum(weights[:, np.newaxis] * observations, axis=0) / total_weight
                
                # Update covariance
                diff = observations - self.means[s]
                self.covars[s] = np.dot(weights * diff.T, diff) / total_weight
                # Add regularization
                self.covars[s] += 1e-4 * np.eye(self.n_features)
            
            # Check convergence
            if np.max(np.abs(self.A - old_A)) < self.convergence_threshold:
                break
    
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predict most likely state sequence"""
        emission_probs = self._compute_emission_probs(observations)
        path, _ = self._viterbi_algorithm(observations, emission_probs)
        return path
    
    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """Get state probabilities for each timestep"""
        emission_probs = self._compute_emission_probs(observations)
        alpha, _ = self._forward_algorithm(observations, emission_probs)
        beta = self._backward_algorithm(observations, emission_probs, np.ones(len(observations)))
        
        gamma = alpha * beta
        gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-10)
        
        return gamma
    
    def get_current_regime(
        self,
        observations: np.ndarray
    ) -> Tuple[RegimeState, float]:
        """Get current regime and its probability"""
        probs = self.predict_proba(observations)
        current_probs = probs[-1]
        
        best_state_idx = np.argmax(current_probs)
        best_prob = current_probs[best_state_idx]
        
        regime = self.state_names[best_state_idx] if best_state_idx < len(self.state_names) else RegimeState.UNKNOWN
        
        return regime, best_prob


class MarketRegimeDetector:
    """
    Comprehensive market regime detection system.
    
    Combines multiple approaches:
    - HMM for state modeling
    - Volatility clustering
    - Trend strength analysis
    - Change point detection
    """
    
    def __init__(
        self,
        use_hmm: bool = True,
        n_hmm_states: int = 4,
        volatility_window: int = 20,
        trend_window: int = 50
    ):
        """
        Initialize regime detector.
        
        Args:
            use_hmm: Whether to use HMM component
            n_hmm_states: Number of HMM states
            volatility_window: Window for volatility calculation
            trend_window: Window for trend calculation
        """
        self.use_hmm = use_hmm
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        
        if use_hmm:
            self.hmm = HMMRegimeModel(n_states=n_hmm_states, n_features=4)
        
        # Volatility thresholds (will be calibrated)
        self.vol_low_threshold = 0.01
        self.vol_high_threshold = 0.02
        
        # Trend thresholds
        self.trend_threshold = 0.3
    
    def _compute_features(self, prices: np.ndarray) -> np.ndarray:
        """Compute features for regime detection"""
        n = len(prices)
        
        # Returns
        returns = np.diff(np.log(prices + 1e-8))
        returns = np.concatenate([[0], returns])
        
        # Rolling volatility
        volatility = np.zeros(n)
        for i in range(self.volatility_window, n):
            volatility[i] = np.std(returns[i-self.volatility_window:i])
        volatility[:self.volatility_window] = volatility[self.volatility_window]
        
        # Trend strength (normalized price position in rolling window)
        trend = np.zeros(n)
        for i in range(self.trend_window, n):
            window = prices[i-self.trend_window:i+1]
            trend[i] = (prices[i] - np.min(window)) / (np.max(window) - np.min(window) + 1e-8) * 2 - 1
        trend[:self.trend_window] = 0
        
        # Momentum
        momentum = np.zeros(n)
        for i in range(20, n):
            momentum[i] = (prices[i] - prices[i-20]) / (prices[i-20] + 1e-8)
        
        features = np.column_stack([returns, volatility, momentum, trend])
        return features
    
    def _classify_volatility_regime(self, volatility: float) -> RegimeState:
        """Classify based on volatility level"""
        if volatility < self.vol_low_threshold:
            return RegimeState.LOW_VOLATILITY
        elif volatility > self.vol_high_threshold:
            return RegimeState.HIGH_VOLATILITY
        return RegimeState.RANGING
    
    def _classify_trend_regime(self, trend: float, returns: float) -> RegimeState:
        """Classify based on trend and returns"""
        if trend > self.trend_threshold and returns > 0:
            return RegimeState.BULL
        elif trend < -self.trend_threshold and returns < 0:
            return RegimeState.BEAR
        return RegimeState.RANGING
    
    def fit(self, prices: np.ndarray):
        """
        Fit the regime detector.
        
        Args:
            prices: Historical price series
        """
        features = self._compute_features(prices)
        
        # Calibrate thresholds
        volatility = features[:, 1]
        self.vol_low_threshold = np.percentile(volatility[volatility > 0], 25)
        self.vol_high_threshold = np.percentile(volatility[volatility > 0], 75)
        
        # Fit HMM if enabled
        if self.use_hmm:
            # Use features after warmup period
            warmup = max(self.volatility_window, self.trend_window)
            self.hmm.fit(features[warmup:])
    
    def detect(self, prices: np.ndarray) -> RegimeDetection:
        """
        Detect current market regime.
        
        Args:
            prices: Price series (including recent history)
            
        Returns:
            RegimeDetection with detailed regime information
        """
        features = self._compute_features(prices)
        current_features = features[-1]
        
        returns, volatility, momentum, trend = current_features
        
        # Multiple regime indicators
        vol_regime = self._classify_volatility_regime(volatility)
        trend_regime = self._classify_trend_regime(trend, returns)
        
        # Combine indicators
        regime_probs: Dict[RegimeState, float] = {state: 0.0 for state in RegimeState}
        
        # Weight from volatility analysis
        if vol_regime == RegimeState.HIGH_VOLATILITY:
            regime_probs[RegimeState.HIGH_VOLATILITY] += 0.3
            regime_probs[RegimeState.CRISIS] += 0.1 if returns < -0.02 else 0
        elif vol_regime == RegimeState.LOW_VOLATILITY:
            regime_probs[RegimeState.LOW_VOLATILITY] += 0.3
        else:
            regime_probs[RegimeState.RANGING] += 0.2
        
        # Weight from trend analysis
        if trend_regime == RegimeState.BULL:
            regime_probs[RegimeState.BULL] += 0.4
            regime_probs[RegimeState.RECOVERY] += 0.1 if volatility > self.vol_low_threshold else 0
        elif trend_regime == RegimeState.BEAR:
            regime_probs[RegimeState.BEAR] += 0.4
        else:
            regime_probs[RegimeState.RANGING] += 0.2
        
        # HMM contribution
        if self.use_hmm and len(prices) > max(self.volatility_window, self.trend_window):
            warmup = max(self.volatility_window, self.trend_window)
            hmm_regime, hmm_prob = self.hmm.get_current_regime(features[warmup:])
            regime_probs[hmm_regime] += 0.3 * hmm_prob
        
        # Normalize probabilities
        total = sum(regime_probs.values())
        if total > 0:
            regime_probs = {k: v/total for k, v in regime_probs.items()}
        
        # Get best regime
        current_regime = max(regime_probs, key=regime_probs.get)
        
        # Calculate days in regime (simplified)
        days_in_regime = 1  # Would need historical tracking
        
        # Transition probabilities from HMM
        transition_probs = None
        if self.use_hmm:
            current_state_idx = list(RegimeState).index(current_regime)
            if current_state_idx < len(self.hmm.A):
                transition_probs = {
                    self.hmm.state_names[i].value if i < len(self.hmm.state_names) else f"state_{i}": 
                    float(self.hmm.A[current_state_idx, i])
                    for i in range(len(self.hmm.A[current_state_idx]))
                }
        
        return RegimeDetection(
            current_regime=current_regime,
            regime_probability=regime_probs[current_regime],
            all_probabilities=regime_probs,
            days_in_regime=days_in_regime,
            transition_probabilities=transition_probs,
            features_used=['returns', 'volatility', 'momentum', 'trend']
        )
    
    def get_regime_history(self, prices: np.ndarray) -> List[RegimeState]:
        """Get regime classification for entire price history"""
        features = self._compute_features(prices)
        warmup = max(self.volatility_window, self.trend_window)
        
        if self.use_hmm and len(prices) > warmup:
            hmm_states = self.hmm.predict(features[warmup:])
            # Convert to RegimeState
            regimes = [RegimeState.UNKNOWN] * warmup
            regimes.extend([
                self.hmm.state_names[s] if s < len(self.hmm.state_names) else RegimeState.UNKNOWN
                for s in hmm_states
            ])
            return regimes
        
        # Fallback to rule-based
        regimes = []
        for i in range(len(prices)):
            if i < warmup:
                regimes.append(RegimeState.UNKNOWN)
            else:
                vol_regime = self._classify_volatility_regime(features[i, 1])
                trend_regime = self._classify_trend_regime(features[i, 3], features[i, 0])
                regimes.append(trend_regime if trend_regime != RegimeState.RANGING else vol_regime)
        
        return regimes
