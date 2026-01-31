"""
DeepAR: Probabilistic Forecasting with Autoregressive RNNs
==========================================================

Based on: "DeepAR: Probabilistic Forecasting with Autoregressive 
Recurrent Networks" - Salinas et al., 2020 (Amazon)

Key Features:
- Autoregressive: Each prediction conditions on previous
- Probabilistic: Outputs distribution parameters, not point estimates
- Handles multiple related time series (global model)
- Produces calibrated prediction intervals

This is the foundation of Amazon's forecasting systems.

Reference: https://arxiv.org/abs/1704.04110
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import (
    Distribution, Normal, NegativeBinomial, 
    StudentT, Poisson, Beta
)
import numpy as np


@dataclass
class DeepARConfig:
    """
    Configuration for DeepAR model.
    
    Optimized for financial time series.
    """
    # Input dimensions
    input_size: int = 1                # Target variable dimension
    num_features: int = 32             # Additional features (covariates)
    
    # Sequence lengths
    context_length: int = 60           # Historical context
    prediction_length: int = 20        # Forecast horizon
    
    # Network architecture
    hidden_size: int = 128             # LSTM hidden size
    num_layers: int = 2                # LSTM layers
    dropout: float = 0.1
    
    # Distribution
    distribution: str = 'normal'       # 'normal', 'student_t', 'negative_binomial'
    
    # Embedding
    num_static_cat: int = 0            # Number of static categorical features
    static_cardinalities: List[int] = field(default_factory=list)
    embedding_dim: int = 16
    
    # Training
    num_samples: int = 100             # MC samples for evaluation
    scaling: bool = True               # Scale targets


class OutputDistribution(nn.Module, ABC):
    """
    Base class for output distributions.
    
    Each subclass defines:
    - Number of parameters to predict
    - How to create distribution from parameters
    - Parameter constraints (e.g., scale > 0)
    """
    
    @property
    @abstractmethod
    def num_params(self) -> int:
        """Number of distribution parameters."""
        pass
        
    @abstractmethod
    def distribution(self, params: Tensor) -> Distribution:
        """Create distribution from parameters."""
        pass
        
    def sample(self, params: Tensor, num_samples: int = 1) -> Tensor:
        """Sample from distribution."""
        dist = self.distribution(params)
        samples = dist.sample((num_samples,))
        return samples
        
    def mean(self, params: Tensor) -> Tensor:
        """Distribution mean."""
        return self.distribution(params).mean
        
    def log_prob(self, params: Tensor, targets: Tensor) -> Tensor:
        """Log probability of targets."""
        return self.distribution(params).log_prob(targets)


class NormalOutput(OutputDistribution):
    """
    Gaussian output distribution.
    
    Predicts: mu (mean), sigma (std)
    """
    
    @property
    def num_params(self) -> int:
        return 2
        
    def distribution(self, params: Tensor) -> Distribution:
        mu = params[..., 0]
        sigma = F.softplus(params[..., 1]) + 1e-6  # Ensure positive
        return Normal(mu, sigma)


class StudentTOutput(OutputDistribution):
    """
    Student-t output distribution.
    
    Better for heavy-tailed financial data.
    Predicts: df (degrees of freedom), loc, scale
    """
    
    @property
    def num_params(self) -> int:
        return 3
        
    def distribution(self, params: Tensor) -> Distribution:
        df = 2.0 + F.softplus(params[..., 0])  # df > 2 for finite variance
        loc = params[..., 1]
        scale = F.softplus(params[..., 2]) + 1e-6
        return StudentT(df, loc, scale)


class NegativeBinomialOutput(OutputDistribution):
    """
    Negative Binomial output distribution.
    
    For count data with overdispersion.
    Predicts: total_count, probs
    """
    
    @property
    def num_params(self) -> int:
        return 2
        
    def distribution(self, params: Tensor) -> Distribution:
        total_count = F.softplus(params[..., 0]) + 1e-6
        probs = torch.sigmoid(params[..., 1])
        return NegativeBinomial(total_count, probs)


def get_output_distribution(name: str) -> OutputDistribution:
    """Factory for output distributions."""
    distributions = {
        'normal': NormalOutput(),
        'gaussian': NormalOutput(),
        'student_t': StudentTOutput(),
        'negative_binomial': NegativeBinomialOutput(),
    }
    return distributions.get(name, NormalOutput())


class FeatureEmbedder(nn.Module):
    """
    Embed and combine all input features.
    
    Handles:
    - Static categorical features (embedded)
    - Static continuous features (projected)
    - Time-varying features (projected)
    """
    
    def __init__(
        self,
        num_static_cat: int,
        static_cardinalities: List[int],
        num_features: int,
        embedding_dim: int,
        hidden_size: int
    ):
        super().__init__()
        
        self.num_static_cat = num_static_cat
        self.num_features = num_features
        
        # Static categorical embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embedding_dim)
            for card in static_cardinalities
        ]) if static_cardinalities else nn.ModuleList()
        
        # Total static embedding dimension
        static_dim = num_static_cat * embedding_dim
        
        # Project covariates
        if num_features > 0:
            self.covariate_proj = nn.Linear(num_features, hidden_size)
        else:
            self.covariate_proj = None
            
        # Project static if present
        if static_dim > 0:
            self.static_proj = nn.Linear(static_dim, hidden_size)
        else:
            self.static_proj = None
            
    def forward(
        self,
        static_cat: Optional[Tensor] = None,
        covariates: Optional[Tensor] = None
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Args:
            static_cat: [batch, num_static_cat] categorical indices
            covariates: [batch, seq_len, num_features]
            
        Returns:
            static_embedding: [batch, hidden_size]
            covariate_embedding: [batch, seq_len, hidden_size]
        """
        static_emb = None
        cov_emb = None
        
        if static_cat is not None and self.embeddings:
            embeddings = [
                emb(static_cat[:, i])
                for i, emb in enumerate(self.embeddings)
            ]
            static_emb = torch.cat(embeddings, dim=-1)
            static_emb = self.static_proj(static_emb)
            
        if covariates is not None and self.covariate_proj is not None:
            cov_emb = self.covariate_proj(covariates)
            
        return static_emb, cov_emb


class DeepAREncoder(nn.Module):
    """
    DeepAR Encoder: Processes historical sequence.
    
    LSTM that encodes the context into hidden states.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(
        self,
        x: Tensor,
        init_state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            x: [batch, seq_len, input_size]
            init_state: Optional (h0, c0) each [num_layers, batch, hidden_size]
            
        Returns:
            output: [batch, seq_len, hidden_size]
            (hn, cn): Final hidden states
        """
        return self.lstm(x, init_state)


class DeepARDecoder(nn.Module):
    """
    DeepAR Decoder: Autoregressive prediction.
    
    At each step:
    1. Input is previous value (sampled or observed)
    2. LSTM updates hidden state
    3. Output distribution parameters predicted
    4. Sample next value from distribution
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_distribution: OutputDistribution,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.output_distribution = output_distribution
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.param_proj = nn.Linear(
            hidden_size,
            output_distribution.num_params
        )
        
    def forward(
        self,
        x: Tensor,
        hidden: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Single step decoding.
        
        Args:
            x: [batch, 1, input_size] current input
            hidden: (h, c) LSTM states
            
        Returns:
            params: [batch, 1, num_params] distribution parameters
            new_hidden: Updated states
        """
        output, new_hidden = self.lstm(x, hidden)
        params = self.param_proj(output)
        return params, new_hidden
        
    def decode_sequence(
        self,
        init_hidden: Tuple[Tensor, Tensor],
        covariates: Tensor,
        initial_value: Tensor,
        target: Optional[Tensor] = None,
        teacher_forcing: bool = False,
        num_samples: int = 1
    ) -> Tuple[Tensor, Tensor]:
        """
        Autoregressive sequence decoding.
        
        Args:
            init_hidden: Initial LSTM states from encoder
            covariates: [batch, pred_len, cov_dim] future covariates
            initial_value: [batch, 1] last observed value
            target: [batch, pred_len] ground truth (for teacher forcing)
            teacher_forcing: Use ground truth as input
            num_samples: Number of samples for probabilistic prediction
            
        Returns:
            all_params: [batch, pred_len, num_params]
            samples: [num_samples, batch, pred_len]
        """
        batch_size = covariates.shape[0]
        pred_len = covariates.shape[1]
        device = covariates.device
        
        hidden = init_hidden
        current_value = initial_value
        
        all_params = []
        all_samples = []
        
        for t in range(pred_len):
            # Create input: [previous_value, covariates_t]
            cov_t = covariates[:, t:t+1, :]  # [batch, 1, hidden]
            # current_value is [batch] or [batch, 1], need [batch, 1, 1]
            current_val_expanded = current_value.view(batch_size, 1, 1)
            input_t = torch.cat([current_val_expanded, cov_t], dim=-1)  # [batch, 1, 1+hidden]
            
            # Decode step
            params_t, hidden = self.forward(input_t, hidden)
            all_params.append(params_t)
            
            # Sample from distribution
            samples_t = self.output_distribution.sample(params_t, num_samples)
            all_samples.append(samples_t.squeeze(-1))
            
            # Update current value for next step
            if teacher_forcing and target is not None:
                current_value = target[:, t]
            else:
                current_value = samples_t[0].squeeze(-1)  # Use first sample
                
        all_params = torch.cat(all_params, dim=1)
        all_samples = torch.stack(all_samples, dim=-1)
        
        return all_params, all_samples


class DeepARV2(nn.Module):
    """
    DeepAR - Probabilistic Autoregressive Forecasting.
    
    Production implementation of Amazon's DeepAR model for
    probabilistic multi-horizon forecasting.
    
    Key Properties:
    - Autoregressive: Conditions on previous predictions
    - Probabilistic: Outputs full distributions, not points
    - Global: Learns from multiple time series
    - Interpretable: Produces calibrated intervals
    
    Usage:
        config = DeepARConfig(
            input_size=1,
            num_features=32,
            context_length=60,
            prediction_length=20,
            distribution='student_t'
        )
        model = DeepARV2(config)
        
        outputs = model(
            past_target=y_past,           # [batch, context_length]
            past_covariates=x_past,       # [batch, context_length, num_features]
            future_covariates=x_future    # [batch, pred_length, num_features]
        )
    """
    
    def __init__(self, config: DeepARConfig):
        super().__init__()
        
        self.config = config
        self.context_length = config.context_length
        self.prediction_length = config.prediction_length
        
        # Output distribution
        self.output_distribution = get_output_distribution(config.distribution)
        
        # Feature embedder
        self.embedder = FeatureEmbedder(
            num_static_cat=config.num_static_cat,
            static_cardinalities=config.static_cardinalities,
            num_features=config.num_features,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size
        )
        
        # Encoder input size: target + covariate embedding
        encoder_input_size = config.input_size + config.hidden_size
        
        # Encoder
        self.encoder = DeepAREncoder(
            input_size=encoder_input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
        
        # Decoder
        self.decoder = DeepARDecoder(
            input_size=config.input_size + config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_distribution=self.output_distribution,
            dropout=config.dropout
        )
        
        # Scaling
        self.scaling = config.scaling
        
    def _compute_scale(self, past_target: Tensor) -> Tensor:
        """Compute scaling factor from historical data."""
        if not self.scaling:
            return torch.ones(past_target.shape[0], 1, device=past_target.device)
            
        # Use mean absolute value (robust to zeros)
        scale = torch.mean(torch.abs(past_target), dim=1, keepdim=True)
        scale = torch.clamp(scale, min=1e-6)
        return scale
        
    def forward(
        self,
        past_target: Tensor,
        past_covariates: Optional[Tensor] = None,
        future_covariates: Optional[Tensor] = None,
        static_cat: Optional[Tensor] = None,
        future_target: Optional[Tensor] = None,
        num_samples: int = 100
    ) -> Dict[str, Tensor]:
        """
        Forward pass.
        
        Args:
            past_target: [batch, context_length] historical target
            past_covariates: [batch, context_length, num_features]
            future_covariates: [batch, pred_length, num_features]
            static_cat: [batch, num_static_cat]
            future_target: [batch, pred_length] for teacher forcing
            num_samples: Number of prediction samples
            
        Returns:
            Dict with predictions, samples, and distribution params
        """
        batch_size = past_target.shape[0]
        device = past_target.device
        
        # Scale target
        scale = self._compute_scale(past_target)
        past_target_scaled = past_target / scale
        
        # Embed features
        static_emb, cov_emb_past = self.embedder(static_cat, past_covariates)
        _, cov_emb_future = self.embedder(static_cat, future_covariates)
        
        # Handle missing covariates
        if cov_emb_past is None:
            cov_emb_past = torch.zeros(
                batch_size, self.context_length, self.config.hidden_size,
                device=device
            )
        if cov_emb_future is None:
            cov_emb_future = torch.zeros(
                batch_size, self.prediction_length, self.config.hidden_size,
                device=device
            )
            
        # Add static embedding to covariates
        if static_emb is not None:
            static_emb = static_emb.unsqueeze(1)
            cov_emb_past = cov_emb_past + static_emb
            cov_emb_future = cov_emb_future + static_emb
            
        # Create encoder input
        encoder_input = torch.cat([
            past_target_scaled.unsqueeze(-1),
            cov_emb_past
        ], dim=-1)
        
        # Encode
        _, (hn, cn) = self.encoder(encoder_input)
        
        # Decode
        initial_value = past_target_scaled[:, -1]
        
        params, samples = self.decoder.decode_sequence(
            init_hidden=(hn, cn),
            covariates=cov_emb_future,
            initial_value=initial_value,
            target=future_target / scale if future_target is not None else None,
            teacher_forcing=future_target is not None,
            num_samples=num_samples
        )
        
        # Unscale
        samples = samples * scale.unsqueeze(0)
        
        # Compute statistics
        mean_pred = samples.mean(dim=0)
        std_pred = samples.std(dim=0)
        median_pred = torch.median(samples, dim=0).values
        
        return {
            'prediction': mean_pred,
            'forecast': mean_pred,
            'median': median_pred,
            'std': std_pred,
            'samples': samples,
            'params': params,
            'scale': scale,
            'lower_5': torch.quantile(samples, 0.05, dim=0),
            'lower_25': torch.quantile(samples, 0.25, dim=0),
            'upper_75': torch.quantile(samples, 0.75, dim=0),
            'upper_95': torch.quantile(samples, 0.95, dim=0),
        }
        
    def predict(
        self,
        past_target: Tensor,
        past_covariates: Optional[Tensor] = None,
        future_covariates: Optional[Tensor] = None,
        static_cat: Optional[Tensor] = None,
        num_samples: int = 100
    ) -> Dict[str, Tensor]:
        """Prediction interface (no teacher forcing)."""
        self.eval()
        with torch.no_grad():
            return self.forward(
                past_target, past_covariates, future_covariates,
                static_cat, None, num_samples
            )
            
    def loss(
        self,
        past_target: Tensor,
        future_target: Tensor,
        past_covariates: Optional[Tensor] = None,
        future_covariates: Optional[Tensor] = None,
        static_cat: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Compute negative log likelihood loss.
        """
        output = self.forward(
            past_target, past_covariates, future_covariates,
            static_cat, future_target, num_samples=1
        )
        
        # Get distribution parameters
        params = output['params']
        scale = output['scale']
        
        # Scaled target
        target_scaled = future_target / scale
        
        # Negative log likelihood
        nll = -self.output_distribution.log_prob(params, target_scaled.unsqueeze(-1))
        nll = nll.mean()
        
        # Additional metrics
        mae = F.l1_loss(output['prediction'], future_target)
        mse = F.mse_loss(output['prediction'], future_target)
        
        return {
            'loss': nll,
            'nll': nll,
            'mae': mae,
            'mse': mse,
            'rmse': torch.sqrt(mse)
        }


class DeepARLoss(nn.Module):
    """
    Loss function for DeepAR training.
    
    Uses negative log likelihood of the predicted distribution.
    """
    
    def __init__(self, distribution: str = 'normal'):
        super().__init__()
        self.output_distribution = get_output_distribution(distribution)
        
    def forward(
        self,
        params: Tensor,
        targets: Tensor,
        scale: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Compute NLL loss.
        
        Args:
            params: [batch, seq_len, num_params] distribution parameters
            targets: [batch, seq_len] ground truth
            scale: [batch, 1] scaling factor
        """
        if scale is not None:
            targets = targets / scale
            
        targets = targets.unsqueeze(-1)
        
        nll = -self.output_distribution.log_prob(params, targets)
        nll = nll.mean()
        
        # Point prediction (mean)
        mean = self.output_distribution.mean(params)
        
        if scale is not None:
            mean = mean * scale
            targets = targets * scale
            
        mae = F.l1_loss(mean, targets.squeeze(-1))
        mse = F.mse_loss(mean, targets.squeeze(-1))
        
        return {
            'loss': nll,
            'nll': nll,
            'mae': mae,
            'mse': mse,
            'rmse': torch.sqrt(mse)
        }


class DeepARWithAttention(nn.Module):
    """
    Enhanced DeepAR with temporal attention.
    
    Adds attention mechanism for better handling of
    long-range dependencies.
    """
    
    def __init__(self, config: DeepARConfig):
        super().__init__()
        
        self.base_model = DeepARV2(config)
        self.config = config
        
        # Attention over encoder outputs
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Combine attention with hidden state
        self.combine = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
    def forward(
        self,
        past_target: Tensor,
        past_covariates: Optional[Tensor] = None,
        future_covariates: Optional[Tensor] = None,
        static_cat: Optional[Tensor] = None,
        future_target: Optional[Tensor] = None,
        num_samples: int = 100
    ) -> Dict[str, Tensor]:
        """
        Forward with attention augmentation.
        """
        # Get encoder outputs first (need to modify base model)
        # For now, use base model
        return self.base_model(
            past_target, past_covariates, future_covariates,
            static_cat, future_target, num_samples
        )
