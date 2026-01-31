"""
Base Classes for Financial Neural Network Models
=================================================

Provides common functionality:
- Device management (CPU/GPU/MPS)
- Model serialization and checkpointing
- Training utilities
- Inference with uncertainty estimation
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for financial neural network models."""
    
    # Model architecture
    input_size: int = 64
    hidden_size: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.2
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0
    
    # Sequence
    sequence_length: int = 60
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    
    # Regularization
    use_layer_norm: bool = True
    use_residual: bool = True
    label_smoothing: float = 0.0
    
    # Uncertainty
    mc_dropout_samples: int = 100
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ModelConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass 
class TrainingMetrics:
    """Metrics collected during training."""
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PredictionResult:
    """Result from model inference."""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_device(device_str: str = "auto") -> torch.device:
    """Get the best available device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


class BaseFinancialModel(nn.Module, ABC):
    """
    Abstract base class for all financial neural network models.
    
    Provides:
    - Device management
    - Checkpointing
    - Training loop utilities
    - Inference with uncertainty estimation
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._device = get_device(config.device)
        self._training_history: List[TrainingMetrics] = []
        self._best_val_loss = float('inf')
        self._epochs_without_improvement = 0
        
    @property
    def device(self) -> torch.device:
        return self._device
    
    def to_device(self) -> 'BaseFinancialModel':
        """Move model to configured device."""
        return self.to(self._device)
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses."""
        pass
    
    def predict(
        self,
        x: Union[np.ndarray, torch.Tensor],
        return_uncertainty: bool = True
    ) -> PredictionResult:
        """
        Generate predictions with optional uncertainty estimation.
        
        Uses Monte Carlo Dropout for uncertainty quantification.
        """
        self.eval()
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        x = x.to(self._device)
        
        with torch.no_grad():
            if return_uncertainty and self.config.dropout > 0:
                # Monte Carlo Dropout for uncertainty
                predictions = self._mc_dropout_inference(x)
                mean_pred = predictions.mean(dim=0)
                std_pred = predictions.std(dim=0)
                
                return PredictionResult(
                    predictions=mean_pred.cpu().numpy(),
                    uncertainty=std_pred.cpu().numpy(),
                    confidence=(1 / (1 + std_pred)).cpu().numpy(),
                    metadata={'mc_samples': self.config.mc_dropout_samples}
                )
            else:
                output = self.forward(x)
                return PredictionResult(
                    predictions=output.cpu().numpy()
                )
    
    def _mc_dropout_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Run multiple forward passes with dropout for uncertainty."""
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(self.config.mc_dropout_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)
        
        self.eval()
        return torch.stack(predictions)
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metadata: Optional[Dict] = None
    ):
        """Save model checkpoint with all training state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'training_history': [asdict(m) for m in self._training_history],
            'best_val_loss': self._best_val_loss,
            'epochs_without_improvement': self._epochs_without_improvement,
            'timestamp': datetime.now().isoformat(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict:
        """Load model checkpoint."""
        path = Path(path)
        
        checkpoint = torch.load(path, map_location=self._device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self._training_history = [
            TrainingMetrics(**m) for m in checkpoint.get('training_history', [])
        ]
        self._best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self._epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint.get('metadata', {})
    
    def get_training_history(self) -> List[TrainingMetrics]:
        """Get training history."""
        return self._training_history
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """Get model summary."""
        total_params = self.count_parameters()
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total_params - trainable
        
        return (
            f"Model: {self.__class__.__name__}\n"
            f"Device: {self._device}\n"
            f"Total Parameters: {total_params:,}\n"
            f"Trainable: {trainable:,}\n"
            f"Non-trainable: {non_trainable:,}\n"
            f"Config: {self.config}"
        )


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    Supports both sinusoidal (fixed) and learnable encodings.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        learnable: bool = False
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.learnable = learnable
        
        if learnable:
            self.encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        else:
            # Sinusoidal encoding
            position = torch.arange(max_len).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            )
            
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('encoding', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.size(1)
        x = x + self.encoding[:, :seq_len, :]
        return self.dropout(x)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) for Temporal Fusion Transformer.
    
    Provides adaptive depth by learning to skip layers when not needed.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        
        # Main layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Context projection if provided
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.context_fc = None
        
        # Gating
        self.gate = nn.Linear(hidden_size, output_size)
        
        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection if sizes differ
        if input_size != output_size:
            self.residual_fc = nn.Linear(input_size, output_size)
        else:
            self.residual_fc = None
        
        # ELU activation
        self.elu = nn.ELU()
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optional context."""
        # Main transformation
        hidden = self.fc1(x)
        
        # Add context if provided
        if context is not None and self.context_fc is not None:
            hidden = hidden + self.context_fc(context)
        
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        
        # Gating mechanism
        gate = torch.sigmoid(self.gate(self.elu(self.fc1(x))))
        gated = gate * hidden
        
        # Residual connection
        if self.residual_fc is not None:
            residual = self.residual_fc(x)
        else:
            residual = x
        
        return self.layer_norm(gated + residual)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network for feature importance.
    
    Learns which input features are most relevant for prediction.
    """
    
    def __init__(
        self,
        input_size: int,
        num_inputs: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        
        # GRN for each input
        self.grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
                context_size=context_size
            )
            for _ in range(num_inputs)
        ])
        
        # Softmax weights for variable selection
        self.weight_grn = GatedResidualNetwork(
            input_size=input_size * num_inputs,
            hidden_size=hidden_size,
            output_size=num_inputs,
            dropout=dropout,
            context_size=context_size
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, num_inputs, input_size)
            context: Optional context tensor
            
        Returns:
            Tuple of (output, variable_weights)
        """
        batch_size = x.size(0)
        
        # Process each input through its GRN
        processed = []
        for i, grn in enumerate(self.grns):
            processed.append(grn(x[:, i, :], context))
        
        processed = torch.stack(processed, dim=1)  # (batch, num_inputs, hidden)
        
        # Compute variable weights
        flat_x = x.view(batch_size, -1)
        weights = self.weight_grn(flat_x, context)
        weights = torch.softmax(weights, dim=-1)  # (batch, num_inputs)
        
        # Weighted sum
        weights_expanded = weights.unsqueeze(-1)  # (batch, num_inputs, 1)
        output = (processed * weights_expanded).sum(dim=1)  # (batch, hidden)
        
        return output, weights
