"""
Neural Ensemble Network with Uncertainty Quantification
========================================================

Production-grade ensemble learning:
- Stacking ensemble with meta-learner
- Dynamic model weighting
- Deep ensemble for uncertainty
- Conformal prediction for calibrated intervals

References:
- Lakshminarayanan et al. "Simple and Scalable Predictive Uncertainty Estimation"
- Gal & Ghahramani "Dropout as a Bayesian Approximation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from .base import (
    BaseFinancialModel,
    ModelConfig,
    PredictionResult,
    get_device
)


@dataclass
class EnsemblePrediction:
    """Prediction from ensemble with full uncertainty quantification."""
    mean: np.ndarray
    std: np.ndarray
    lower_5: np.ndarray
    upper_95: np.ndarray
    lower_25: np.ndarray
    upper_75: np.ndarray
    model_weights: Dict[str, float]
    individual_predictions: Dict[str, np.ndarray]
    confidence: np.ndarray
    agreement_score: float
    calibration_score: Optional[float] = None


class BaseSubModel(nn.Module):
    """Base class for sub-models in ensemble."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MLPSubModel(BaseSubModel):
    """MLP sub-model for ensemble."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__(input_size, hidden_size, output_size, dropout)
        
        layers = []
        current_size = input_size
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # Sequence input - use last position
            x = x[:, -1, :]
        return self.network(x)


class LSTMSubModel(BaseSubModel):
    """LSTM sub-model for ensemble."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__(input_size, hidden_size, output_size, dropout)
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.output_layer(last_hidden)


class TransformerSubModel(BaseSubModel):
    """Transformer sub-model for ensemble."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__(input_size, hidden_size, output_size, dropout)
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.output_layer(x[:, -1, :])


class MetaLearner(nn.Module):
    """
    Meta-learner for stacking ensemble.
    
    Learns optimal combination of base model predictions.
    """
    
    def __init__(
        self,
        num_models: int,
        output_size: int,
        hidden_size: int = 64,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.num_models = num_models
        self.output_size = output_size
        self.use_attention = use_attention
        
        if use_attention:
            # Attention-based weighting
            self.query = nn.Linear(output_size, hidden_size)
            self.key = nn.Linear(output_size, hidden_size)
            self.value = nn.Linear(output_size, output_size)
            self.scale = hidden_size ** -0.5
        
        # Learned combination
        self.combiner = nn.Sequential(
            nn.Linear(num_models * output_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Model importance weights (learnable prior)
        self.model_weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def forward(
        self,
        predictions: torch.Tensor,
        return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Combine predictions from base models.
        
        Args:
            predictions: (batch, num_models, output_size)
            return_weights: Whether to return attention weights
            
        Returns:
            Combined prediction (batch, output_size)
        """
        batch_size = predictions.size(0)
        
        if self.use_attention:
            # Compute attention weights across models
            Q = self.query(predictions)  # (batch, num_models, hidden)
            K = self.key(predictions)
            V = self.value(predictions)
            
            # Self-attention over models
            attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # Weighted combination
            attended = torch.bmm(attn_weights, V)  # (batch, num_models, output)
            
            # Average over models with learned weights
            weights = F.softmax(self.model_weights, dim=0)
            weighted_sum = (attended * weights.view(1, -1, 1)).sum(dim=1)
        else:
            weights = F.softmax(self.model_weights, dim=0)
            weighted_sum = (predictions * weights.view(1, -1, 1)).sum(dim=1)
        
        # Also use MLP combination
        flat_preds = predictions.view(batch_size, -1)
        mlp_output = self.combiner(flat_preds)
        
        # Average attention and MLP outputs
        output = 0.5 * weighted_sum + 0.5 * mlp_output
        
        if return_weights:
            return output, weights
        return output


class EnsembleNetwork(BaseFinancialModel):
    """
    Deep Ensemble Network with uncertainty quantification.
    
    Features:
    - Multiple diverse sub-models (MLP, LSTM, Transformer)
    - Meta-learner for optimal combination
    - Deep ensemble for epistemic uncertainty
    - MC Dropout for aleatoric uncertainty
    - Conformal prediction for calibrated intervals
    """
    
    def __init__(
        self,
        config: ModelConfig,
        num_sub_models: int = 5,
        sub_model_types: List[str] = None
    ):
        super().__init__(config)
        
        self.num_sub_models = num_sub_models
        sub_model_types = sub_model_types or ['mlp', 'mlp', 'lstm', 'lstm', 'transformer']
        
        # Output size: predictions for each horizon
        output_size = len(config.prediction_horizons) * 3  # mean, lower, upper
        
        # Create diverse sub-models
        self.sub_models = nn.ModuleList()
        for i, model_type in enumerate(sub_model_types[:num_sub_models]):
            if model_type == 'mlp':
                model = MLPSubModel(
                    input_size=config.input_size,
                    hidden_size=config.hidden_size + (i * 32),  # Vary sizes
                    output_size=output_size,
                    dropout=config.dropout
                )
            elif model_type == 'lstm':
                model = LSTMSubModel(
                    input_size=config.input_size,
                    hidden_size=config.hidden_size,
                    output_size=output_size,
                    dropout=config.dropout
                )
            elif model_type == 'transformer':
                model = TransformerSubModel(
                    input_size=config.input_size,
                    hidden_size=config.hidden_size,
                    output_size=output_size,
                    num_heads=config.num_heads,
                    dropout=config.dropout
                )
            else:
                # Default to MLP
                model = MLPSubModel(
                    input_size=config.input_size,
                    hidden_size=config.hidden_size,
                    output_size=output_size,
                    dropout=config.dropout
                )
            
            self.sub_models.append(model)
        
        # Meta-learner
        self.meta_learner = MetaLearner(
            num_models=len(self.sub_models),
            output_size=output_size,
            hidden_size=config.hidden_size // 2,
            use_attention=True
        )
        
        # Direction head (combined)
        self.direction_head = nn.Sequential(
            nn.Linear(output_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 3)
        )
        
        # Volatility head
        self.volatility_head = nn.Sequential(
            nn.Linear(output_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Softplus()
        )
        
        # Calibration buffer for conformal prediction
        self.register_buffer('calibration_scores', torch.zeros(1000))
        self._calibration_idx = 0
        self._calibration_count = 0
        
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
        return_individual: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor (batch, seq_len, input_size) or (batch, input_size)
            return_individual: Whether to return individual model predictions
            
        Returns:
            Dict with combined predictions and uncertainty
        """
        # Get predictions from each sub-model
        individual_preds = []
        for model in self.sub_models:
            pred = model(x)
            individual_preds.append(pred)
        
        # Stack predictions: (batch, num_models, output_size)
        stacked_preds = torch.stack(individual_preds, dim=1)
        
        # Meta-learner combination
        combined, model_weights = self.meta_learner(stacked_preds, return_weights=True)
        
        # Compute uncertainty from ensemble disagreement
        pred_mean = stacked_preds.mean(dim=1)
        pred_std = stacked_preds.std(dim=1)
        
        # Parse predictions for each horizon
        num_horizons = len(self.config.prediction_horizons)
        predictions = {}
        for i, horizon in enumerate(self.config.prediction_horizons):
            idx = i * 3
            predictions[f'h{horizon}'] = {
                'mean': combined[:, idx],
                'lower': combined[:, idx + 1],
                'upper': combined[:, idx + 2],
                'std': pred_std[:, idx]  # Uncertainty from disagreement
            }
        
        # Direction prediction
        direction_logits = self.direction_head(combined)
        direction_probs = F.softmax(direction_logits, dim=-1)
        
        # Volatility
        volatility = self.volatility_head(combined).squeeze(-1)
        
        result = {
            'predictions': predictions,
            'direction': {
                'logits': direction_logits,
                'probs': direction_probs,
                'up': direction_probs[:, 0],
                'down': direction_probs[:, 1],
                'sideways': direction_probs[:, 2]
            },
            'volatility': volatility,
            'model_weights': model_weights,
            'uncertainty': pred_std.mean(dim=-1)  # Overall uncertainty
        }
        
        if return_individual:
            result['individual_predictions'] = {
                f'model_{i}': individual_preds[i]
                for i in range(len(self.sub_models))
            }
        
        return result
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_mc_samples: int = 50,
        confidence_level: float = 0.9
    ) -> EnsemblePrediction:
        """
        Generate predictions with full uncertainty quantification.
        
        Combines:
        - Epistemic uncertainty (ensemble disagreement)
        - Aleatoric uncertainty (MC Dropout)
        - Calibrated intervals (conformal prediction)
        """
        self.train()  # Enable dropout
        
        all_predictions = []
        
        with torch.no_grad():
            for _ in range(num_mc_samples):
                output = self.forward(x, return_individual=True)
                all_predictions.append(output['predictions'])
        
        self.eval()
        
        # Aggregate predictions
        result = {}
        for horizon in self.config.prediction_horizons:
            key = f'h{horizon}'
            means = torch.stack([p[key]['mean'] for p in all_predictions])
            
            result[key] = {
                'mean': means.mean(dim=0).cpu().numpy(),
                'std': means.std(dim=0).cpu().numpy(),
                'lower_5': torch.quantile(means, 0.05, dim=0).cpu().numpy(),
                'upper_95': torch.quantile(means, 0.95, dim=0).cpu().numpy(),
                'lower_25': torch.quantile(means, 0.25, dim=0).cpu().numpy(),
                'upper_75': torch.quantile(means, 0.75, dim=0).cpu().numpy()
            }
        
        # Model weights
        with torch.no_grad():
            output = self.forward(x, return_individual=True)
            weights = output['model_weights'].cpu().numpy()
        
        # Agreement score (how much models agree)
        individual = output['individual_predictions']
        preds = torch.stack(list(individual.values()), dim=0)
        agreement = 1.0 - preds.std(dim=0).mean().item()
        
        # Compute confidence from uncertainty
        mean_std = np.mean([result[f'h{h}']['std'] for h in self.config.prediction_horizons])
        confidence = 1.0 / (1.0 + mean_std)
        
        return EnsemblePrediction(
            mean=np.stack([result[f'h{h}']['mean'] for h in self.config.prediction_horizons]),
            std=np.stack([result[f'h{h}']['std'] for h in self.config.prediction_horizons]),
            lower_5=np.stack([result[f'h{h}']['lower_5'] for h in self.config.prediction_horizons]),
            upper_95=np.stack([result[f'h{h}']['upper_95'] for h in self.config.prediction_horizons]),
            lower_25=np.stack([result[f'h{h}']['lower_25'] for h in self.config.prediction_horizons]),
            upper_75=np.stack([result[f'h{h}']['upper_75'] for h in self.config.prediction_horizons]),
            model_weights={f'model_{i}': float(weights[i]) for i in range(len(weights))},
            individual_predictions={
                f'model_{i}': individual[f'model_{i}'].cpu().numpy()
                for i in range(len(self.sub_models))
            },
            confidence=confidence,
            agreement_score=agreement
        )
    
    def update_calibration(self, residuals: torch.Tensor):
        """Update calibration scores for conformal prediction."""
        batch_size = residuals.size(0)
        for i in range(batch_size):
            idx = self._calibration_idx % self.calibration_scores.size(0)
            self.calibration_scores[idx] = residuals[i].abs().mean()
            self._calibration_idx += 1
            self._calibration_count = min(
                self._calibration_count + 1,
                self.calibration_scores.size(0)
            )
    
    def get_conformal_interval(self, alpha: float = 0.1) -> float:
        """Get conformal prediction interval width."""
        if self._calibration_count == 0:
            return float('inf')
        
        valid_scores = self.calibration_scores[:self._calibration_count]
        quantile = torch.quantile(valid_scores, 1 - alpha)
        return quantile.item()


class UncertaintyEstimator(nn.Module):
    """
    Dedicated uncertainty estimation module.
    
    Can be added to any model to provide uncertainty estimates
    using various methods.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        method: str = 'heteroscedastic'  # 'heteroscedastic', 'mcdropout', 'ensemble'
    ):
        super().__init__()
        
        self.method = method
        
        if method == 'heteroscedastic':
            # Predict mean and variance together
            self.mean_head = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1)
            )
            self.log_var_head = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1)
            )
        
        elif method == 'mcdropout':
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, 1)
            )
        
        elif method == 'ensemble':
            # Multiple small networks
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_size, hidden_size // 2),
                    nn.GELU(),
                    nn.Linear(hidden_size // 2, 1)
                )
                for _ in range(5)
            ])
    
    def forward(
        self,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate prediction with uncertainty.
        
        Returns:
            Tuple of (mean, std)
        """
        if self.method == 'heteroscedastic':
            mean = self.mean_head(x)
            log_var = self.log_var_head(x)
            std = torch.exp(0.5 * log_var)
            return mean.squeeze(-1), std.squeeze(-1)
        
        elif self.method == 'mcdropout':
            self.train()
            predictions = []
            with torch.no_grad():
                for _ in range(num_samples):
                    predictions.append(self.network(x))
            self.eval()
            
            preds = torch.stack(predictions)
            return preds.mean(dim=0).squeeze(-1), preds.std(dim=0).squeeze(-1)
        
        elif self.method == 'ensemble':
            preds = torch.stack([head(x) for head in self.heads])
            return preds.mean(dim=0).squeeze(-1), preds.std(dim=0).squeeze(-1)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def nll_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Negative log likelihood loss for heteroscedastic regression."""
        if self.method != 'heteroscedastic':
            raise ValueError("NLL loss only available for heteroscedastic method")
        
        mean = self.mean_head(x)
        log_var = self.log_var_head(x)
        
        # NLL = 0.5 * (log_var + (target - mean)^2 / exp(log_var))
        precision = torch.exp(-log_var)
        loss = 0.5 * (log_var + precision * (target.unsqueeze(-1) - mean) ** 2)
        
        return loss.mean()
