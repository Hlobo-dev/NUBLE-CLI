"""
Neural Regime Classifier
=========================

Deep learning approach to market regime detection:
- Neural HMM for regime states (bull/bear/volatile/ranging)
- Changepoint detection with neural networks
- Volatility regime classification
- Regime-aware prediction adjustments

Combines classical HMM intuition with neural network flexibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from .base import (
    BaseFinancialModel,
    ModelConfig,
    PredictionResult,
    get_device
)


class RegimeState(Enum):
    """Market regime states."""
    BULL = 0
    BEAR = 1
    HIGH_VOLATILITY = 2
    LOW_VOLATILITY = 3
    RANGING = 4
    CRISIS = 5


@dataclass
class RegimeDetection:
    """Result of regime detection."""
    current_regime: RegimeState
    regime_probabilities: Dict[str, float]
    transition_matrix: np.ndarray
    regime_duration: int
    confidence: float
    changepoint_probability: float
    historical_regimes: List[int]
    features_importance: Optional[Dict[str, float]] = None


class NeuralHMM(nn.Module):
    """
    Neural Hidden Markov Model.
    
    Combines HMM structure with neural network emission and transition models.
    The emission probabilities are computed by a neural network instead of
    fixed Gaussian distributions.
    """
    
    def __init__(
        self,
        num_states: int,
        input_size: int,
        hidden_size: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_states = num_states
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Learnable initial state distribution
        self.initial_logits = nn.Parameter(torch.zeros(num_states))
        
        # Learnable transition matrix (as logits)
        # Initialize with strong diagonal (states tend to persist)
        init_trans = torch.eye(num_states) * 2.0
        self.transition_logits = nn.Parameter(init_trans)
        
        # Neural emission model - outputs log probabilities for each state
        self.emission_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_states)
        )
    
    @property
    def initial_probs(self) -> torch.Tensor:
        """Get initial state probabilities."""
        return F.softmax(self.initial_logits, dim=-1)
    
    @property
    def transition_matrix(self) -> torch.Tensor:
        """Get transition probability matrix."""
        return F.softmax(self.transition_logits, dim=-1)
    
    def emission_logprobs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log emission probabilities.
        
        Args:
            x: Input features (batch, seq_len, input_size)
            
        Returns:
            Log probabilities (batch, seq_len, num_states)
        """
        return F.log_softmax(self.emission_network(x), dim=-1)
    
    def forward_algorithm(
        self,
        emissions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward algorithm for computing state probabilities.
        
        Args:
            emissions: Log emission probabilities (batch, seq_len, num_states)
            
        Returns:
            Tuple of (alpha, log_likelihood)
            - alpha: Forward probabilities (batch, seq_len, num_states)
            - log_likelihood: Log likelihood of sequence (batch,)
        """
        batch_size, seq_len, _ = emissions.shape
        device = emissions.device
        
        # Initialize with initial probs + first emission
        log_init = torch.log(self.initial_probs + 1e-10).unsqueeze(0)
        alpha = log_init + emissions[:, 0, :]  # (batch, num_states)
        
        alphas = [alpha]
        
        # Forward pass
        log_trans = torch.log(self.transition_matrix + 1e-10)
        
        for t in range(1, seq_len):
            # alpha_t = emission_t * sum_j(alpha_{t-1,j} * trans_{j,i})
            # In log space: log_alpha_t = log_emission_t + logsumexp(log_alpha_{t-1} + log_trans)
            
            # Expand for broadcasting
            alpha_expanded = alpha.unsqueeze(-1)  # (batch, num_states, 1)
            trans_expanded = log_trans.unsqueeze(0)  # (1, num_states, num_states)
            
            # Sum over previous states
            alpha = emissions[:, t, :] + torch.logsumexp(
                alpha_expanded + trans_expanded, dim=1
            )
            alphas.append(alpha)
        
        # Stack all alphas
        alphas = torch.stack(alphas, dim=1)  # (batch, seq_len, num_states)
        
        # Log likelihood is logsumexp of final alpha
        log_likelihood = torch.logsumexp(alphas[:, -1, :], dim=-1)
        
        return alphas, log_likelihood
    
    def viterbi_decode(
        self,
        emissions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Viterbi algorithm for finding most likely state sequence.
        
        Args:
            emissions: Log emission probabilities (batch, seq_len, num_states)
            
        Returns:
            Tuple of (best_path, path_score)
        """
        batch_size, seq_len, _ = emissions.shape
        device = emissions.device
        
        log_trans = torch.log(self.transition_matrix + 1e-10)
        log_init = torch.log(self.initial_probs + 1e-10)
        
        # Initialize
        viterbi = log_init.unsqueeze(0) + emissions[:, 0, :]
        backpointers = []
        
        # Forward pass
        for t in range(1, seq_len):
            viterbi_expanded = viterbi.unsqueeze(-1)  # (batch, num_states, 1)
            trans_expanded = log_trans.unsqueeze(0)  # (1, num_states, num_states)
            
            # Best previous state for each current state
            scores = viterbi_expanded + trans_expanded
            best_scores, best_states = scores.max(dim=1)
            
            viterbi = best_scores + emissions[:, t, :]
            backpointers.append(best_states)
        
        # Backtrack
        best_path_scores, best_last_states = viterbi.max(dim=-1)
        
        best_paths = [best_last_states]
        for bp in reversed(backpointers):
            best_paths.append(
                bp.gather(1, best_paths[-1].unsqueeze(-1)).squeeze(-1)
            )
        
        best_paths = torch.stack(list(reversed(best_paths)), dim=1)
        
        return best_paths, best_path_scores
    
    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features (batch, seq_len, input_size)
            return_states: Whether to return decoded state sequence
            
        Returns:
            Dict with emissions, state probs, and optionally decoded states
        """
        # Compute emissions
        emissions = self.emission_logprobs(x)
        
        # Forward algorithm for state probabilities
        alphas, log_likelihood = self.forward_algorithm(emissions)
        
        # Convert to probabilities
        state_probs = F.softmax(alphas, dim=-1)
        
        result = {
            'emissions': emissions,
            'state_probs': state_probs,
            'log_likelihood': log_likelihood,
            'transition_matrix': self.transition_matrix.detach()
        }
        
        if return_states:
            best_path, path_score = self.viterbi_decode(emissions)
            result['best_path'] = best_path
            result['path_score'] = path_score
        
        return result


class ChangepointDetector(nn.Module):
    """
    Neural network for changepoint detection.
    
    Identifies structural breaks in time series using
    a sliding window approach with a neural classifier.
    """
    
    def __init__(
        self,
        input_size: int,
        window_size: int = 20,
        hidden_size: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.window_size = window_size
        
        # Process windows before and after potential changepoint
        self.encoder = nn.Sequential(
            nn.Linear(window_size * input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Compare before/after windows
        self.comparator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Detect changepoints in sequence.
        
        Args:
            x: Input sequence (batch, seq_len, input_size)
            threshold: Probability threshold for changepoint
            
        Returns:
            Dict with changepoint probabilities and locations
        """
        batch_size, seq_len, _ = x.shape
        
        if seq_len < 2 * self.window_size:
            # Sequence too short
            return {
                'changepoint_probs': torch.zeros(batch_size, seq_len, device=x.device),
                'changepoints': torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
            }
        
        changepoint_probs = []
        
        for t in range(self.window_size, seq_len - self.window_size):
            # Windows before and after
            before = x[:, t - self.window_size:t, :].reshape(batch_size, -1)
            after = x[:, t:t + self.window_size, :].reshape(batch_size, -1)
            
            # Encode windows
            before_enc = self.encoder(before)
            after_enc = self.encoder(after)
            
            # Compare
            combined = torch.cat([before_enc, after_enc], dim=-1)
            prob = self.comparator(combined).squeeze(-1)
            
            changepoint_probs.append(prob)
        
        # Pad with zeros for positions where we couldn't compute
        probs = torch.stack(changepoint_probs, dim=1)
        pad_before = torch.zeros(batch_size, self.window_size, device=x.device)
        pad_after = torch.zeros(batch_size, self.window_size, device=x.device)
        probs = torch.cat([pad_before, probs, pad_after], dim=1)
        
        return {
            'changepoint_probs': probs,
            'changepoints': probs > threshold
        }


class VolatilityRegimeClassifier(nn.Module):
    """
    Classify volatility regime (low/normal/high/crisis).
    
    Uses realized volatility and returns distribution
    characteristics to classify the current regime.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_regimes: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_regimes = num_regimes
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_regimes)
        )
        
        # Regime names
        self.regime_names = ['low_volatility', 'normal', 'high_volatility', 'crisis']
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Classify volatility regime.
        
        Args:
            x: Input features (batch, seq_len, input_size)
            return_features: Whether to return intermediate features
            
        Returns:
            Dict with regime probabilities and predicted regime
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Temporal modeling
        lstm_out, _ = self.lstm(features)
        
        # Use last position
        last_hidden = lstm_out[:, -1, :]
        
        # Classify
        logits = self.classifier(last_hidden)
        probs = F.softmax(logits, dim=-1)
        
        result = {
            'regime_logits': logits,
            'regime_probs': probs,
            'predicted_regime': probs.argmax(dim=-1),
            'regime_names': {
                i: name for i, name in enumerate(self.regime_names[:self.num_regimes])
            }
        }
        
        if return_features:
            result['features'] = last_hidden
        
        return result


class NeuralRegimeClassifier(BaseFinancialModel):
    """
    Complete neural regime detection system.
    
    Combines:
    - Neural HMM for hidden state inference
    - Changepoint detection
    - Volatility regime classification
    - Regime-aware predictions
    
    Features:
    - End-to-end trainable
    - Uncertainty quantification on regime predictions
    - Temporal consistency regularization
    - Regime transition forecasting
    """
    
    def __init__(
        self,
        config: ModelConfig,
        num_regimes: int = 6,
        changepoint_window: int = 20
    ):
        super().__init__(config)
        
        self.num_regimes = num_regimes
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU()
        )
        
        # Neural HMM for regime detection
        self.neural_hmm = NeuralHMM(
            num_states=num_regimes,
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )
        
        # Changepoint detector
        self.changepoint_detector = ChangepointDetector(
            input_size=config.hidden_size,
            window_size=changepoint_window,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )
        
        # Volatility regime classifier
        self.volatility_classifier = VolatilityRegimeClassifier(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_regimes=4,
            dropout=config.dropout
        )
        
        # Regime duration predictor
        self.duration_predictor = nn.Sequential(
            nn.Linear(config.hidden_size + num_regimes, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Softplus()  # Ensure positive duration
        )
        
        # Transition forecaster
        self.transition_forecaster = nn.Sequential(
            nn.Linear(config.hidden_size + num_regimes, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, num_regimes)  # Next regime probs
        )
        
        # Regime names
        self.regime_names = [
            'bull', 'bear', 'high_volatility', 
            'low_volatility', 'ranging', 'crisis'
        ][:num_regimes]
        
        self._init_weights()
        self.to_device()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = True
    ) -> Dict[str, Any]:
        """
        Complete regime detection forward pass.
        
        Args:
            x: Input features (batch, seq_len, input_size)
            return_all: Whether to return all intermediate results
            
        Returns:
            Comprehensive regime detection results
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        projected = self.input_proj(x)
        
        # Neural HMM
        hmm_result = self.neural_hmm(projected, return_states=True)
        
        # Changepoint detection
        changepoint_result = self.changepoint_detector(projected)
        
        # Volatility regime
        volatility_result = self.volatility_classifier(projected)
        
        # Current regime info
        current_regime_idx = hmm_result['best_path'][:, -1]
        current_regime_probs = hmm_result['state_probs'][:, -1, :]
        
        # Last hidden state
        last_hidden = projected[:, -1, :]
        
        # Combine for duration and transition prediction
        combined = torch.cat([last_hidden, current_regime_probs], dim=-1)
        
        # Predicted duration of current regime
        duration = self.duration_predictor(combined).squeeze(-1)
        
        # Next regime forecast
        next_regime_logits = self.transition_forecaster(combined)
        next_regime_probs = F.softmax(next_regime_logits, dim=-1)
        
        # Compute regime confidence
        confidence = current_regime_probs.max(dim=-1).values
        
        # Count current regime duration
        regime_duration = self._compute_regime_duration(hmm_result['best_path'])
        
        result = {
            'current_regime': {
                'index': current_regime_idx,
                'name': [self.regime_names[idx.item()] for idx in current_regime_idx],
                'probabilities': current_regime_probs,
                'confidence': confidence
            },
            'regime_sequence': hmm_result['best_path'],
            'regime_probabilities': hmm_result['state_probs'],
            'transition_matrix': hmm_result['transition_matrix'],
            'log_likelihood': hmm_result['log_likelihood'],
            'changepoints': {
                'probabilities': changepoint_result['changepoint_probs'],
                'detected': changepoint_result['changepoints']
            },
            'volatility_regime': {
                'index': volatility_result['predicted_regime'],
                'probabilities': volatility_result['regime_probs']
            },
            'duration': {
                'current': regime_duration,
                'predicted': duration
            },
            'next_regime': {
                'logits': next_regime_logits,
                'probabilities': next_regime_probs,
                'most_likely': next_regime_probs.argmax(dim=-1)
            },
            'regime_names': self.regime_names
        }
        
        return result
    
    def _compute_regime_duration(self, regime_sequence: torch.Tensor) -> torch.Tensor:
        """Compute how long we've been in the current regime."""
        batch_size, seq_len = regime_sequence.shape
        
        durations = []
        for b in range(batch_size):
            seq = regime_sequence[b]
            current = seq[-1]
            duration = 1
            for t in range(seq_len - 2, -1, -1):
                if seq[t] == current:
                    duration += 1
                else:
                    break
            durations.append(duration)
        
        return torch.tensor(durations, device=regime_sequence.device)
    
    def predict_regime(
        self,
        x: torch.Tensor,
        num_mc_samples: int = 50
    ) -> RegimeDetection:
        """
        Predict current regime with full uncertainty quantification.
        
        Args:
            x: Input features (batch, seq_len, input_size)
            num_mc_samples: Number of MC dropout samples
            
        Returns:
            RegimeDetection dataclass with all regime info
        """
        self.train()  # Enable dropout
        
        all_regime_probs = []
        all_changepoint_probs = []
        
        with torch.no_grad():
            for _ in range(num_mc_samples):
                output = self.forward(x)
                all_regime_probs.append(output['current_regime']['probabilities'])
                all_changepoint_probs.append(
                    output['changepoints']['probabilities'][:, -1]
                )
        
        self.eval()
        
        # Aggregate
        regime_probs = torch.stack(all_regime_probs)
        mean_probs = regime_probs.mean(dim=0)
        std_probs = regime_probs.std(dim=0)
        
        changepoint_probs = torch.stack(all_changepoint_probs)
        mean_changepoint = changepoint_probs.mean(dim=0)
        
        # Get final prediction
        with torch.no_grad():
            output = self.forward(x)
        
        # Create result (for first batch item)
        regime_idx = mean_probs[0].argmax().item()
        
        return RegimeDetection(
            current_regime=RegimeState(regime_idx),
            regime_probabilities={
                name: float(mean_probs[0, i])
                for i, name in enumerate(self.regime_names)
            },
            transition_matrix=output['transition_matrix'].cpu().numpy(),
            regime_duration=output['duration']['current'][0].item(),
            confidence=float(mean_probs[0].max()),
            changepoint_probability=float(mean_changepoint[0]),
            historical_regimes=output['regime_sequence'][0].cpu().numpy().tolist()
        )
    
    def compute_loss(
        self,
        x: torch.Tensor,
        regime_labels: Optional[torch.Tensor] = None,
        changepoint_labels: Optional[torch.Tensor] = None,
        temporal_consistency_weight: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            x: Input features
            regime_labels: Optional ground truth regimes (batch, seq_len)
            changepoint_labels: Optional changepoint indicators (batch, seq_len)
            temporal_consistency_weight: Weight for consistency regularization
            
        Returns:
            Dict with loss components
        """
        output = self.forward(x)
        
        losses = {}
        
        # Supervised regime loss
        if regime_labels is not None:
            # Cross entropy for regime classification
            regime_logits = output['regime_probabilities'].permute(0, 2, 1)  # (batch, states, seq)
            regime_loss = F.cross_entropy(regime_logits, regime_labels)
            losses['regime_loss'] = regime_loss
        
        # Supervised changepoint loss
        if changepoint_labels is not None:
            cp_probs = output['changepoints']['probabilities']
            cp_loss = F.binary_cross_entropy(cp_probs, changepoint_labels.float())
            losses['changepoint_loss'] = cp_loss
        
        # Unsupervised: negative log likelihood
        losses['nll_loss'] = -output['log_likelihood'].mean()
        
        # Temporal consistency: penalize rapid transitions
        state_probs = output['regime_probabilities']
        temporal_diff = (state_probs[:, 1:, :] - state_probs[:, :-1, :]).abs()
        losses['consistency_loss'] = temporal_consistency_weight * temporal_diff.mean()
        
        # Total loss
        losses['total_loss'] = sum(losses.values())
        
        return losses
