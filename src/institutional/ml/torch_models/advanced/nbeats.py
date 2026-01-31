"""
N-BEATS: Neural Basis Expansion Analysis for Time Series Forecasting
=====================================================================

Based on: "N-BEATS: Neural basis expansion analysis for interpretable 
time series forecasting" - Oreshkin et al., 2020 (ServiceNow/Element AI)

This implementation includes both Generic and Interpretable architectures
with financial-specific enhancements.

Key Features:
- Doubly Residual Stacking: Block outputs predict AND subtract
- Generic Architecture: Fully learned basis functions  
- Interpretable Architecture: Trend/Seasonality decomposition
- Ensemble via Stacking: Multiple stack types combined

Reference: https://arxiv.org/abs/1905.10437
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Literal
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


@dataclass
class NBeatsConfig:
    """
    Configuration for N-BEATS model.
    
    Defaults are optimized for financial time series.
    """
    # Input/output lengths
    context_length: int = 60          # Lookback window
    prediction_length: int = 20       # Forecast horizon
    
    # Generic stack configuration
    num_generic_stacks: int = 2       # Number of generic stacks
    num_blocks_per_generic_stack: int = 3
    generic_hidden_size: int = 512
    generic_theta_size: int = 64      # Expansion coefficients
    generic_num_layers: int = 4
    
    # Interpretable stack configuration
    num_trend_stacks: int = 1
    num_seasonality_stacks: int = 1
    num_blocks_per_trend_stack: int = 3
    num_blocks_per_seasonality_stack: int = 3
    trend_hidden_size: int = 256
    seasonality_hidden_size: int = 2048
    trend_polynomial_degree: int = 3  # Polynomial basis for trend
    num_harmonics: int = 2            # Fourier basis for seasonality
    
    # Training
    dropout: float = 0.1
    share_weights: bool = False       # Share weights within stacks
    
    # Architecture type
    architecture: Literal['generic', 'interpretable', 'ensemble'] = 'ensemble'
    
    # Loss
    loss_type: Literal['mae', 'mse', 'smape', 'mape'] = 'mae'


class NBEATSBlock(nn.Module, ABC):
    """
    Base N-BEATS Block.
    
    Each block:
    1. Takes backcast input x
    2. Produces theta parameters via FC layers
    3. Generates backcast (past reconstruction) and forecast (future prediction)
    
    The "doubly residual" principle:
    - Forecast is added to stack output
    - Backcast is subtracted from input for next block
    """
    
    def __init__(
        self,
        input_size: int,
        theta_size: int,
        hidden_size: int,
        num_layers: int,
        backcast_length: int,
        forecast_length: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.theta_size = theta_size
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        
        # Fully connected stack
        layers = []
        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_size
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.fc_stack = nn.Sequential(*layers)
        
        # Theta parameter generators
        self.theta_b = nn.Linear(hidden_size, theta_size, bias=False)
        self.theta_f = nn.Linear(hidden_size, theta_size, bias=False)
        
    @abstractmethod
    def get_basis(self, theta: Tensor, length: int) -> Tensor:
        """Generate basis expansion from theta coefficients."""
        pass
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the block.
        
        Args:
            x: [batch, backcast_length] input sequence
            
        Returns:
            backcast: [batch, backcast_length] reconstruction
            forecast: [batch, forecast_length] prediction
        """
        # FC stack
        hidden = self.fc_stack(x)
        
        # Generate theta coefficients
        theta_b = self.theta_b(hidden)
        theta_f = self.theta_f(hidden)
        
        # Basis expansion
        backcast = self.get_basis(theta_b, self.backcast_length)
        forecast = self.get_basis(theta_f, self.forecast_length)
        
        return backcast, forecast


class GenericBlock(NBEATSBlock):
    """
    Generic N-BEATS Block with fully learned basis.
    
    The basis functions are learned via linear projections,
    giving maximum flexibility but less interpretability.
    """
    
    def __init__(
        self,
        input_size: int,
        theta_size: int,
        hidden_size: int,
        num_layers: int,
        backcast_length: int,
        forecast_length: int,
        dropout: float = 0.1
    ):
        super().__init__(
            input_size, theta_size, hidden_size, num_layers,
            backcast_length, forecast_length, dropout
        )
        
        # Learned basis projections
        self.backcast_basis = nn.Linear(theta_size, backcast_length, bias=False)
        self.forecast_basis = nn.Linear(theta_size, forecast_length, bias=False)
        
    def get_basis(self, theta: Tensor, length: int) -> Tensor:
        if length == self.backcast_length:
            return self.backcast_basis(theta)
        else:
            return self.forecast_basis(theta)


class TrendBlock(NBEATSBlock):
    """
    Trend Block with polynomial basis.
    
    Uses polynomial basis functions for smooth trend modeling:
    t^0, t^1, t^2, ..., t^p (polynomial degree p)
    
    This provides interpretable trend extraction.
    """
    
    def __init__(
        self,
        input_size: int,
        theta_size: int,  # = polynomial_degree + 1
        hidden_size: int,
        num_layers: int,
        backcast_length: int,
        forecast_length: int,
        dropout: float = 0.1
    ):
        # Theta size must match polynomial degree + 1
        super().__init__(
            input_size, theta_size, hidden_size, num_layers,
            backcast_length, forecast_length, dropout
        )
        
        self.polynomial_degree = theta_size - 1
        
        # Pre-compute polynomial basis matrices
        self.register_buffer(
            'backcast_basis_matrix',
            self._create_polynomial_basis(backcast_length, theta_size)
        )
        self.register_buffer(
            'forecast_basis_matrix', 
            self._create_polynomial_basis(forecast_length, theta_size)
        )
        
    def _create_polynomial_basis(self, length: int, degree: int) -> Tensor:
        """Create polynomial basis matrix T: [length, degree]"""
        t = torch.arange(length, dtype=torch.float32) / length
        basis = torch.stack([t ** i for i in range(degree)], dim=1)
        return basis
        
    def get_basis(self, theta: Tensor, length: int) -> Tensor:
        """
        Polynomial basis expansion: y = T @ theta
        
        Args:
            theta: [batch, degree] coefficients
            length: output length
            
        Returns:
            [batch, length] polynomial trend
        """
        if length == self.backcast_length:
            basis = self.backcast_basis_matrix
        else:
            basis = self.forecast_basis_matrix
            
        # [batch, degree] @ [degree, length] -> [batch, length]
        return torch.matmul(theta, basis.T)


class SeasonalityBlock(NBEATSBlock):
    """
    Seasonality Block with Fourier basis.
    
    Uses Fourier basis functions for periodic pattern modeling:
    cos(2πkt/T), sin(2πkt/T) for k = 1, ..., K (harmonics)
    
    This provides interpretable seasonality extraction.
    """
    
    def __init__(
        self,
        input_size: int,
        theta_size: int,  # = 2 * num_harmonics
        hidden_size: int,
        num_layers: int,
        backcast_length: int,
        forecast_length: int,
        num_harmonics: int,
        dropout: float = 0.1
    ):
        # Theta size = 2 * num_harmonics (sin + cos coefficients)
        super().__init__(
            input_size, 2 * num_harmonics, hidden_size, num_layers,
            backcast_length, forecast_length, dropout
        )
        
        self.num_harmonics = num_harmonics
        
        # Pre-compute Fourier basis matrices
        self.register_buffer(
            'backcast_basis_matrix',
            self._create_fourier_basis(backcast_length, num_harmonics)
        )
        self.register_buffer(
            'forecast_basis_matrix',
            self._create_fourier_basis(forecast_length, num_harmonics)
        )
        
    def _create_fourier_basis(self, length: int, num_harmonics: int) -> Tensor:
        """Create Fourier basis matrix S: [length, 2*num_harmonics]"""
        t = torch.arange(length, dtype=torch.float32) / length
        
        basis_funcs = []
        for k in range(1, num_harmonics + 1):
            basis_funcs.append(torch.cos(2 * math.pi * k * t))
            basis_funcs.append(torch.sin(2 * math.pi * k * t))
            
        basis = torch.stack(basis_funcs, dim=1)
        return basis
        
    def get_basis(self, theta: Tensor, length: int) -> Tensor:
        """
        Fourier basis expansion: y = S @ theta
        
        Args:
            theta: [batch, 2*num_harmonics] coefficients
            length: output length
            
        Returns:
            [batch, length] seasonal component
        """
        if length == self.backcast_length:
            basis = self.backcast_basis_matrix
        else:
            basis = self.forecast_basis_matrix
            
        return torch.matmul(theta, basis.T)


class NBEATSStack(nn.Module):
    """
    N-BEATS Stack - Collection of blocks with residual connections.
    
    Each stack:
    1. Processes input through sequence of blocks
    2. Accumulates forecasts from all blocks
    3. Subtracts backcasts from input (doubly residual)
    """
    
    def __init__(
        self,
        blocks: nn.ModuleList,
        share_weights: bool = False
    ):
        super().__init__()
        
        if share_weights:
            # All blocks share the first block's weights
            self.blocks = nn.ModuleList([blocks[0]] * len(blocks))
        else:
            self.blocks = blocks
            
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the stack.
        
        Args:
            x: [batch, backcast_length] input
            
        Returns:
            backcast: [batch, backcast_length] final residual
            forecast: [batch, forecast_length] accumulated forecast
        """
        forecast = None
        
        for block in self.blocks:
            block_backcast, block_forecast = block(x)
            
            # Subtract backcast from input (doubly residual)
            x = x - block_backcast
            
            # Accumulate forecasts
            if forecast is None:
                forecast = block_forecast
            else:
                forecast = forecast + block_forecast
                
        return x, forecast


class NBeatsGeneric(nn.Module):
    """
    N-BEATS Generic Architecture.
    
    Pure generic stacks with fully learned basis functions.
    Maximum flexibility, suitable for complex patterns.
    """
    
    def __init__(self, config: NBeatsConfig):
        super().__init__()
        
        self.config = config
        self.backcast_length = config.context_length
        self.forecast_length = config.prediction_length
        
        # Build stacks
        stacks = []
        for _ in range(config.num_generic_stacks):
            blocks = nn.ModuleList([
                GenericBlock(
                    input_size=config.context_length,
                    theta_size=config.generic_theta_size,
                    hidden_size=config.generic_hidden_size,
                    num_layers=config.generic_num_layers,
                    backcast_length=config.context_length,
                    forecast_length=config.prediction_length,
                    dropout=config.dropout
                )
                for _ in range(config.num_blocks_per_generic_stack)
            ])
            stacks.append(NBEATSStack(blocks, config.share_weights))
            
        self.stacks = nn.ModuleList(stacks)
        
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
        # Handle multi-feature input
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
            
        return result


class NBeatsInterpretable(nn.Module):
    """
    N-BEATS Interpretable Architecture.
    
    Specialized stacks for trend and seasonality with
    constrained basis functions that provide interpretability.
    """
    
    def __init__(self, config: NBeatsConfig):
        super().__init__()
        
        self.config = config
        self.backcast_length = config.context_length
        self.forecast_length = config.prediction_length
        
        # Build trend stacks
        trend_stacks = []
        for _ in range(config.num_trend_stacks):
            blocks = nn.ModuleList([
                TrendBlock(
                    input_size=config.context_length,
                    theta_size=config.trend_polynomial_degree + 1,
                    hidden_size=config.trend_hidden_size,
                    num_layers=4,
                    backcast_length=config.context_length,
                    forecast_length=config.prediction_length,
                    dropout=config.dropout
                )
                for _ in range(config.num_blocks_per_trend_stack)
            ])
            trend_stacks.append(NBEATSStack(blocks, config.share_weights))
            
        self.trend_stacks = nn.ModuleList(trend_stacks)
        
        # Build seasonality stacks
        seasonality_stacks = []
        for _ in range(config.num_seasonality_stacks):
            blocks = nn.ModuleList([
                SeasonalityBlock(
                    input_size=config.context_length,
                    theta_size=2 * config.num_harmonics,
                    hidden_size=config.seasonality_hidden_size,
                    num_layers=4,
                    backcast_length=config.context_length,
                    forecast_length=config.prediction_length,
                    num_harmonics=config.num_harmonics,
                    dropout=config.dropout
                )
                for _ in range(config.num_blocks_per_seasonality_stack)
            ])
            seasonality_stacks.append(NBEATSStack(blocks, config.share_weights))
            
        self.seasonality_stacks = nn.ModuleList(seasonality_stacks)
        
    def forward(
        self,
        x: Tensor,
        return_decomposition: bool = True
    ) -> Dict[str, Tensor]:
        """
        Forward pass with interpretable decomposition.
        
        Args:
            x: [batch, context_length] or [batch, context_length, 1]
            return_decomposition: Return trend/seasonality components
            
        Returns:
            Dict with forecast, trend, seasonality components
        """
        if x.dim() == 3:
            x = x.squeeze(-1)
            
        residual = x
        
        # Trend extraction
        trend_forecast = torch.zeros(
            x.shape[0], self.forecast_length, device=x.device
        )
        trend_backcast = torch.zeros(
            x.shape[0], self.backcast_length, device=x.device
        )
        
        for stack in self.trend_stacks:
            residual, stack_forecast = stack(residual)
            trend_forecast = trend_forecast + stack_forecast
            trend_backcast = trend_backcast + (x - residual)
            
        # Seasonality extraction
        seasonality_forecast = torch.zeros(
            x.shape[0], self.forecast_length, device=x.device
        )
        seasonality_backcast = torch.zeros(
            x.shape[0], self.backcast_length, device=x.device
        )
        
        for stack in self.seasonality_stacks:
            residual, stack_forecast = stack(residual)
            seasonality_forecast = seasonality_forecast + stack_forecast
            seasonality_backcast = seasonality_backcast + (x - trend_backcast - residual)
            
        # Combined forecast
        forecast = trend_forecast + seasonality_forecast
        
        result = {
            'forecast': forecast,
            'backcast_residual': residual
        }
        
        if return_decomposition:
            result['trend'] = trend_forecast
            result['seasonality'] = seasonality_forecast
            result['trend_backcast'] = trend_backcast
            result['seasonality_backcast'] = seasonality_backcast
            
        return result


class NBeatsEnsemble(nn.Module):
    """
    N-BEATS Ensemble Architecture.
    
    Combines Generic and Interpretable stacks for best of both:
    - Interpretable stacks capture trend and seasonality
    - Generic stacks capture residual complex patterns
    
    This is the recommended architecture for production use.
    """
    
    def __init__(self, config: NBeatsConfig):
        super().__init__()
        
        self.config = config
        
        # Interpretable component
        self.interpretable = NBeatsInterpretable(config)
        
        # Generic component for residuals
        self.generic = NBeatsGeneric(config)
        
        # Learnable ensemble weights
        self.ensemble_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(
        self,
        x: Tensor,
        return_decomposition: bool = True
    ) -> Dict[str, Tensor]:
        """
        Forward pass through ensemble.
        
        Args:
            x: [batch, context_length] or [batch, context_length, 1]
            return_decomposition: Return component forecasts
            
        Returns:
            Dict with combined forecast and components
        """
        if x.dim() == 3:
            x = x.squeeze(-1)
            
        # Interpretable decomposition
        interp_out = self.interpretable(x, return_decomposition=True)
        
        # Generic on residual
        generic_out = self.generic(interp_out['backcast_residual'])
        
        # Ensemble combination
        weight = torch.sigmoid(self.ensemble_weight)
        forecast = (
            weight * (interp_out['trend'] + interp_out['seasonality']) +
            (1 - weight) * generic_out['forecast']
        )
        
        result = {
            'forecast': forecast,
            'backcast_residual': generic_out['backcast_residual']
        }
        
        if return_decomposition:
            result['trend'] = interp_out['trend']
            result['seasonality'] = interp_out['seasonality']
            result['generic'] = generic_out['forecast']
            result['ensemble_weight'] = weight
            
        return result


class NBeatsV2(nn.Module):
    """
    N-BEATS Production Model - Unified Interface.
    
    Factory that creates appropriate N-BEATS variant based on config.
    Includes financial-specific enhancements:
    - Multi-variate support
    - Probabilistic outputs
    - Volatility-aware weighting
    
    Usage:
        config = NBeatsConfig(
            context_length=60,
            prediction_length=20,
            architecture='ensemble'
        )
        model = NBeatsV2(config)
        
        outputs = model(x)  # x: [batch, context_length]
    """
    
    def __init__(self, config: NBeatsConfig):
        super().__init__()
        
        self.config = config
        
        # Build appropriate architecture
        if config.architecture == 'generic':
            self.model = NBeatsGeneric(config)
        elif config.architecture == 'interpretable':
            self.model = NBeatsInterpretable(config)
        else:  # ensemble
            self.model = NBeatsEnsemble(config)
            
        # Optional multi-variate support
        self.input_projection = None
        self.output_projection = None
        
        # Probabilistic output (optional)
        self.probabilistic = False
        
    def enable_multivariate(self, num_features: int):
        """Enable multi-variate input processing."""
        self.input_projection = nn.Linear(
            num_features * self.config.context_length,
            self.config.context_length
        )
        self.output_projection = nn.Linear(
            self.config.prediction_length,
            num_features * self.config.prediction_length
        )
        self.num_features = num_features
        
    def enable_probabilistic(self, num_quantiles: int = 7):
        """Enable probabilistic output."""
        self.probabilistic = True
        self.quantile_proj = nn.Linear(
            self.config.prediction_length,
            self.config.prediction_length * num_quantiles
        )
        self.num_quantiles = num_quantiles
        self.quantiles = [i / (num_quantiles + 1) for i in range(1, num_quantiles + 1)]
        
    def forward(
        self,
        x: Tensor,
        return_decomposition: bool = True
    ) -> Dict[str, Tensor]:
        """
        Forward pass.
        
        Args:
            x: [batch, context_length] univariate or
               [batch, context_length, num_features] multivariate
               
        Returns:
            Dict with forecast and optional decomposition
        """
        # Handle multivariate
        original_shape = x.shape
        if x.dim() == 3 and self.input_projection is not None:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
            x = self.input_projection(x)
        elif x.dim() == 3:
            x = x.mean(dim=-1)  # Simple aggregation if no projection
            
        # Main model
        result = self.model(x, return_decomposition)
        
        # Handle multivariate output
        if self.output_projection is not None:
            batch_size = result['forecast'].shape[0]
            result['forecast'] = self.output_projection(result['forecast'])
            result['forecast'] = result['forecast'].view(
                batch_size, self.config.prediction_length, self.num_features
            )
            
        # Handle probabilistic output
        if self.probabilistic:
            batch_size = result['forecast'].shape[0]
            quantile_preds = self.quantile_proj(result['forecast'].view(batch_size, -1))
            result['quantiles'] = quantile_preds.view(
                batch_size, self.config.prediction_length, self.num_quantiles
            )
            result['quantiles'], _ = torch.sort(result['quantiles'], dim=-1)
            
        return result
    
    def predict_with_uncertainty(
        self,
        x: Tensor,
        num_samples: int = 100
    ) -> Dict[str, Tensor]:
        """
        Generate predictions with Monte Carlo uncertainty.
        """
        self.train()  # Enable dropout
        
        samples = []
        for _ in range(num_samples):
            with torch.no_grad():
                out = self.forward(x, return_decomposition=False)
                samples.append(out['forecast'])
                
        self.eval()
        
        samples = torch.stack(samples, dim=0)
        
        return {
            'mean': samples.mean(dim=0),
            'std': samples.std(dim=0),
            'median': torch.median(samples, dim=0).values,
            'lower_5': torch.quantile(samples, 0.05, dim=0),
            'upper_95': torch.quantile(samples, 0.95, dim=0),
            'samples': samples
        }


class NBeatsLoss(nn.Module):
    """
    Loss function for N-BEATS training.
    
    Supports multiple loss types as used in the paper.
    """
    
    def __init__(self, loss_type: str = 'mae'):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Compute loss.
        
        Args:
            predictions: [batch, forecast_length] or [batch, forecast_length, features]
            targets: Same shape as predictions
            mask: Optional mask for missing values
        """
        if mask is not None:
            predictions = predictions * mask
            targets = targets * mask
            
        if self.loss_type == 'mae':
            loss = F.l1_loss(predictions, targets)
        elif self.loss_type == 'mse':
            loss = F.mse_loss(predictions, targets)
        elif self.loss_type == 'smape':
            # Symmetric Mean Absolute Percentage Error
            denominator = torch.abs(predictions) + torch.abs(targets) + 1e-8
            loss = torch.mean(200.0 * torch.abs(predictions - targets) / denominator)
        elif self.loss_type == 'mape':
            # Mean Absolute Percentage Error
            loss = torch.mean(torch.abs((predictions - targets) / (targets + 1e-8)))
        else:
            loss = F.l1_loss(predictions, targets)
            
        # Additional metrics
        mae = F.l1_loss(predictions, targets)
        mse = F.mse_loss(predictions, targets)
        
        return {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'rmse': torch.sqrt(mse)
        }
