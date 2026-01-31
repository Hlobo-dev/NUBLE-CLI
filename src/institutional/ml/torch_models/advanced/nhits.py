"""
N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting
======================================================================

Based on: "N-HiTS: Neural Hierarchical Interpolation for Time Series 
Forecasting" - Challu et al., 2022 (Nixtla)

Key Innovation: Multi-rate signal sampling with hierarchical interpolation
enabling efficient long-horizon forecasting with significantly fewer 
parameters than N-BEATS.

Key Features:
- Multi-Rate Sampling: Different blocks focus on different frequencies
- Hierarchical Interpolation: Downsample -> Predict -> Upsample
- MaxPool Stacks: Efficient multi-scale representation
- Much faster training than N-BEATS for long horizons

Reference: https://arxiv.org/abs/2201.12886
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


@dataclass
class NHiTSConfig:
    """
    Configuration for N-HiTS model.
    
    Key hyperparameters control the multi-rate sampling hierarchy.
    """
    # Input/output lengths
    context_length: int = 60          # Lookback window
    prediction_length: int = 20       # Forecast horizon
    
    # Stack configuration
    num_stacks: int = 3               # Number of hierarchical stacks
    num_blocks_per_stack: int = 1     # Blocks within each stack
    
    # Network dimensions
    hidden_size: int = 512            # FC layer width
    num_layers: int = 2               # FC layers per block
    
    # Hierarchical interpolation
    pooling_kernel_sizes: List[int] = field(default_factory=lambda: [16, 8, 1])
    interpolation_modes: List[str] = field(default_factory=lambda: ['linear', 'linear', 'linear'])
    
    # Expressiveness parameters
    n_freq_downsample: List[int] = field(default_factory=lambda: [168, 24, 1])  # Output downsampling
    
    # Training
    dropout: float = 0.0
    activation: str = 'relu'
    
    # Output
    n_harmonics: int = 0              # 0 for direct forecast, >0 for spectral
    
    def __post_init__(self):
        assert len(self.pooling_kernel_sizes) == self.num_stacks
        assert len(self.n_freq_downsample) == self.num_stacks


class NHiTSBlock(nn.Module):
    """
    N-HiTS Block with hierarchical interpolation.
    
    Key difference from N-BEATS:
    1. MaxPool on input for multi-rate sampling
    2. Predict at lower resolution
    3. Interpolate to full resolution
    
    This dramatically reduces computational cost for long horizons.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        pooling_kernel_size: int,
        n_freq_downsample: int,
        backcast_length: int,
        forecast_length: int,
        interpolation_mode: str = 'linear',
        dropout: float = 0.0,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.pooling_kernel_size = pooling_kernel_size
        self.n_freq_downsample = n_freq_downsample
        self.interpolation_mode = interpolation_mode
        
        # Effective input size after pooling
        self.pooled_input_size = int(np.ceil(input_size / pooling_kernel_size))
        
        # Output sizes (downsampled)
        self.backcast_out_size = int(np.ceil(backcast_length / n_freq_downsample))
        self.forecast_out_size = int(np.ceil(forecast_length / n_freq_downsample))
        
        # MaxPool for input downsampling
        self.pooling = nn.MaxPool1d(
            kernel_size=pooling_kernel_size,
            stride=pooling_kernel_size,
            ceil_mode=True
        )
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # FC stack
        layers = []
        for i in range(num_layers):
            in_features = self.pooled_input_size if i == 0 else hidden_size
            layers.extend([
                nn.Linear(in_features, hidden_size),
                self.activation,
                nn.Dropout(dropout)
            ])
        self.fc_stack = nn.Sequential(*layers)
        
        # Theta parameter generators (at downsampled resolution)
        self.theta_b = nn.Linear(hidden_size, self.backcast_out_size)
        self.theta_f = nn.Linear(hidden_size, self.forecast_out_size)
        
    def _interpolate(self, x: Tensor, target_size: int) -> Tensor:
        """Interpolate from downsampled to full resolution."""
        if x.shape[-1] == target_size:
            return x
            
        # Add dimension for interpolation: [batch, 1, length]
        x = x.unsqueeze(1)
        x = F.interpolate(
            x,
            size=target_size,
            mode=self.interpolation_mode,
            align_corners=False if self.interpolation_mode != 'nearest' else None
        )
        return x.squeeze(1)
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with hierarchical interpolation.
        
        Args:
            x: [batch, backcast_length] input
            
        Returns:
            backcast: [batch, backcast_length]
            forecast: [batch, forecast_length]
        """
        # MaxPool downsampling
        x_pooled = self.pooling(x.unsqueeze(1)).squeeze(1)
        
        # FC stack
        hidden = self.fc_stack(x_pooled)
        
        # Generate downsampled outputs
        theta_b = self.theta_b(hidden)
        theta_f = self.theta_f(hidden)
        
        # Interpolate to full resolution
        backcast = self._interpolate(theta_b, self.backcast_length)
        forecast = self._interpolate(theta_f, self.forecast_length)
        
        return backcast, forecast


class NHiTSStack(nn.Module):
    """
    N-HiTS Stack - Collection of blocks at same resolution.
    
    Uses doubly residual connections like N-BEATS.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_blocks: int,
        pooling_kernel_size: int,
        n_freq_downsample: int,
        backcast_length: int,
        forecast_length: int,
        interpolation_mode: str = 'linear',
        dropout: float = 0.0,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            NHiTSBlock(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                pooling_kernel_size=pooling_kernel_size,
                n_freq_downsample=n_freq_downsample,
                backcast_length=backcast_length,
                forecast_length=forecast_length,
                interpolation_mode=interpolation_mode,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_blocks)
        ])
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through stack.
        
        Args:
            x: [batch, backcast_length]
            
        Returns:
            residual: [batch, backcast_length]
            forecast: [batch, forecast_length]
        """
        forecast = None
        residual = x
        
        for block in self.blocks:
            block_backcast, block_forecast = block(residual)
            
            # Doubly residual
            residual = residual - block_backcast
            
            if forecast is None:
                forecast = block_forecast
            else:
                forecast = forecast + block_forecast
                
        return residual, forecast


class NHiTSV2(nn.Module):
    """
    N-HiTS Production Model.
    
    Hierarchical multi-rate sampling for efficient long-horizon forecasting.
    
    Architecture:
    1. Multiple stacks with different pooling rates
    2. Each stack captures different frequency components
    3. Hierarchical interpolation for efficiency
    4. Doubly residual connections
    
    Usage:
        config = NHiTSConfig(
            context_length=168,      # 1 week hourly
            prediction_length=24,     # 1 day ahead
            pooling_kernel_sizes=[16, 8, 1],
            n_freq_downsample=[168, 24, 1]
        )
        model = NHiTSV2(config)
        
        outputs = model(x)  # x: [batch, context_length]
    """
    
    def __init__(self, config: NHiTSConfig):
        super().__init__()
        
        self.config = config
        self.backcast_length = config.context_length
        self.forecast_length = config.prediction_length
        
        # Build hierarchical stacks
        self.stacks = nn.ModuleList([
            NHiTSStack(
                input_size=config.context_length,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                num_blocks=config.num_blocks_per_stack,
                pooling_kernel_size=config.pooling_kernel_sizes[i],
                n_freq_downsample=config.n_freq_downsample[i],
                backcast_length=config.context_length,
                forecast_length=config.prediction_length,
                interpolation_mode=config.interpolation_modes[i],
                dropout=config.dropout,
                activation=config.activation
            )
            for i in range(config.num_stacks)
        ])
        
    def forward(
        self,
        x: Tensor,
        return_decomposition: bool = False
    ) -> Dict[str, Tensor]:
        """
        Forward pass.
        
        Args:
            x: [batch, context_length] or [batch, context_length, 1]
            return_decomposition: Return per-stack forecasts
            
        Returns:
            Dict with forecast and optional decomposition
        """
        if x.dim() == 3:
            x = x.squeeze(-1)
            
        residual = x
        forecast = torch.zeros(
            x.shape[0], self.forecast_length, device=x.device
        )
        
        stack_forecasts = []
        
        for stack in self.stacks:
            residual, stack_forecast = stack(residual)
            forecast = forecast + stack_forecast
            stack_forecasts.append(stack_forecast)
            
        result = {
            'forecast': forecast,
            'backcast_residual': residual
        }
        
        if return_decomposition:
            result['stack_forecasts'] = torch.stack(stack_forecasts, dim=1)
            # Label by frequency
            result['low_freq'] = stack_forecasts[0] if len(stack_forecasts) > 0 else None
            result['mid_freq'] = stack_forecasts[1] if len(stack_forecasts) > 1 else None
            result['high_freq'] = stack_forecasts[-1] if len(stack_forecasts) > 2 else None
            
        return result
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_stacks': len(self.stacks),
            'context_length': self.config.context_length,
            'prediction_length': self.config.prediction_length
        }


class NHiTSWithExogenous(nn.Module):
    """
    N-HiTS with Exogenous Variables Support.
    
    Extended N-HiTS that can incorporate:
    - Static covariates (e.g., stock sector)
    - Time-varying known covariates (e.g., day of week)
    """
    
    def __init__(
        self,
        config: NHiTSConfig,
        num_static_features: int = 0,
        num_future_features: int = 0,
        static_embedding_dim: int = 16
    ):
        super().__init__()
        
        self.base_model = NHiTSV2(config)
        self.config = config
        
        # Static feature processing
        if num_static_features > 0:
            self.static_encoder = nn.Sequential(
                nn.Linear(num_static_features, static_embedding_dim),
                nn.ReLU(),
                nn.Linear(static_embedding_dim, config.hidden_size)
            )
            self.static_projector = nn.Linear(
                config.hidden_size, config.context_length
            )
        else:
            self.static_encoder = None
            
        # Future features processing
        if num_future_features > 0:
            self.future_encoder = nn.Linear(
                num_future_features, 1
            )
        else:
            self.future_encoder = None
            
    def forward(
        self,
        x: Tensor,
        static_features: Optional[Tensor] = None,
        future_features: Optional[Tensor] = None,
        return_decomposition: bool = False
    ) -> Dict[str, Tensor]:
        """
        Forward pass with exogenous variables.
        
        Args:
            x: [batch, context_length]
            static_features: [batch, num_static_features]
            future_features: [batch, prediction_length, num_future_features]
        """
        if x.dim() == 3:
            x = x.squeeze(-1)
            
        # Add static feature influence
        if static_features is not None and self.static_encoder is not None:
            static_encoded = self.static_encoder(static_features)
            static_influence = self.static_projector(static_encoded)
            x = x + static_influence
            
        # Base model forecast
        result = self.base_model(x, return_decomposition)
        
        # Add future feature influence
        if future_features is not None and self.future_encoder is not None:
            future_influence = self.future_encoder(future_features).squeeze(-1)
            result['forecast'] = result['forecast'] + future_influence
            
        return result


class NHiTSProbabilistic(nn.Module):
    """
    Probabilistic N-HiTS with Quantile Outputs.
    
    Produces calibrated prediction intervals instead of point forecasts.
    """
    
    def __init__(
        self,
        config: NHiTSConfig,
        num_quantiles: int = 7,
        quantiles: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.base_model = NHiTSV2(config)
        self.config = config
        
        self.num_quantiles = num_quantiles
        self.quantiles = quantiles or [i / (num_quantiles + 1) for i in range(1, num_quantiles + 1)]
        
        # Quantile projection
        self.quantile_proj = nn.Sequential(
            nn.Linear(config.prediction_length, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.prediction_length * num_quantiles)
        )
        
    def forward(
        self,
        x: Tensor,
        return_decomposition: bool = False
    ) -> Dict[str, Tensor]:
        """
        Forward pass with quantile outputs.
        
        Returns:
            Dict with:
                - quantiles: [batch, pred_len, num_quantiles]
                - median: [batch, pred_len]
                - forecast: [batch, pred_len] (same as median)
        """
        base_result = self.base_model(x, return_decomposition)
        
        # Generate quantiles
        batch_size = base_result['forecast'].shape[0]
        quantile_raw = self.quantile_proj(base_result['forecast'])
        quantiles = quantile_raw.view(
            batch_size, self.config.prediction_length, self.num_quantiles
        )
        
        # Sort for monotonicity
        quantiles, _ = torch.sort(quantiles, dim=-1)
        
        # Extract key quantiles
        median_idx = self.num_quantiles // 2
        
        result = {
            'quantiles': quantiles,
            'median': quantiles[..., median_idx],
            'forecast': quantiles[..., median_idx],  # Point forecast = median
            'lower': quantiles[..., 0],
            'upper': quantiles[..., -1],
            'quantile_levels': torch.tensor(self.quantiles, device=x.device),
            'backcast_residual': base_result['backcast_residual']
        }
        
        if return_decomposition:
            result.update({
                k: v for k, v in base_result.items()
                if k.startswith('stack') or k.endswith('freq')
            })
            
        return result


def nhits_quantile_loss(
    predictions: Tensor,
    targets: Tensor,
    quantiles: List[float]
) -> Tensor:
    """
    Quantile loss for probabilistic N-HiTS.
    
    Args:
        predictions: [batch, seq_len, num_quantiles]
        targets: [batch, seq_len]
        quantiles: List of quantile levels
        
    Returns:
        Scalar loss
    """
    targets = targets.unsqueeze(-1)
    quantiles_tensor = torch.tensor(quantiles, device=predictions.device)
    
    errors = targets - predictions
    
    loss = torch.max(
        quantiles_tensor * errors,
        (quantiles_tensor - 1) * errors
    )
    
    return loss.mean()


class NHiTSLoss(nn.Module):
    """Combined loss for N-HiTS training."""
    
    def __init__(
        self,
        loss_type: str = 'mae',
        quantiles: Optional[List[float]] = None
    ):
        super().__init__()
        self.loss_type = loss_type
        self.quantiles = quantiles
        
    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        quantile_predictions: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Compute loss.
        
        Args:
            predictions: Point forecasts [batch, forecast_length]
            targets: Ground truth [batch, forecast_length]
            quantile_predictions: Optional quantile forecasts
        """
        # Point forecast loss
        if self.loss_type == 'mae':
            point_loss = F.l1_loss(predictions, targets)
        elif self.loss_type == 'mse':
            point_loss = F.mse_loss(predictions, targets)
        else:
            point_loss = F.l1_loss(predictions, targets)
            
        # Quantile loss if applicable
        if quantile_predictions is not None and self.quantiles is not None:
            q_loss = nhits_quantile_loss(quantile_predictions, targets, self.quantiles)
            total_loss = 0.5 * point_loss + 0.5 * q_loss
        else:
            q_loss = torch.tensor(0.0)
            total_loss = point_loss
            
        return {
            'loss': total_loss,
            'point_loss': point_loss,
            'quantile_loss': q_loss,
            'mae': F.l1_loss(predictions, targets),
            'mse': F.mse_loss(predictions, targets),
            'rmse': torch.sqrt(F.mse_loss(predictions, targets))
        }
