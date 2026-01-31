"""
Financial LSTM with Attention
==============================

Production-grade LSTM architecture for financial time series:
- Bidirectional LSTM layers with residual connections
- Multi-head self-attention mechanism
- Layer normalization for training stability
- Monte Carlo Dropout for uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .base import (
    BaseFinancialModel,
    ModelConfig,
    PredictionResult,
    PositionalEncoding
)


class AttentionLayer(nn.Module):
    """
    Multi-head self-attention layer optimized for financial sequences.
    
    Features:
    - Scaled dot-product attention
    - Causal masking option for autoregressive prediction
    - Dropout for regularization
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = True
    ):
        super().__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Causal mask
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply custom mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.out_proj(context)
        
        if return_attention:
            return output, attn_weights
        return output, None


class LSTMBlock(nn.Module):
    """
    Single LSTM block with residual connection and layer normalization.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bidirectional: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output size depends on bidirectionality
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        
        # Residual projection if sizes differ
        if input_size != lstm_output_size:
            self.residual_proj = nn.Linear(input_size, lstm_output_size)
        else:
            self.residual_proj = None
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with residual connection."""
        
        # LSTM forward
        output, (h_n, c_n) = self.lstm(x, hidden)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x
        
        output = self.layer_norm(output + residual)
        output = self.dropout(output)
        
        return output, (h_n, c_n)


class FinancialLSTM(BaseFinancialModel):
    """
    Production-grade Financial LSTM for time series prediction.
    
    Architecture:
    - Multiple bidirectional LSTM layers with residual connections
    - Layer normalization for training stability
    - Multi-head self-attention for capturing long-range dependencies
    - Multi-horizon output heads
    
    Features:
    - GPU acceleration
    - Mixed precision training support
    - Monte Carlo Dropout for uncertainty quantification
    - Attention visualization for interpretability
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        hidden_size = config.hidden_size
        bidirectional = True
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Input projection
        self.input_proj = nn.Linear(config.input_size, hidden_size)
        
        # Stacked LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(config.num_layers):
            input_dim = hidden_size if i == 0 else lstm_output_size
            self.lstm_layers.append(
                LSTMBlock(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    bidirectional=bidirectional,
                    dropout=config.dropout
                )
            )
        
        # Self-attention
        self.attention = AttentionLayer(
            hidden_size=lstm_output_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            causal=True
        )
        self.attention_norm = nn.LayerNorm(lstm_output_size)
        
        # Output heads for multi-horizon prediction
        self.output_heads = nn.ModuleDict()
        for horizon in config.prediction_horizons:
            self.output_heads[f'h{horizon}'] = nn.Sequential(
                nn.Linear(lstm_output_size, hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_size, 3)  # mean, lower, upper bounds
            )
        
        # Direction prediction head (up/down/sideways)
        self.direction_head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, 3)  # 3 classes
        )
        
        # Volatility prediction head
        self.volatility_head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()  # Ensure positive volatility
        )
        
        # Initialize weights
        self._init_weights()
        
        # Move to device
        self.to_device()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM uses orthogonal initialization
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional list of hidden states for each LSTM layer
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
            - predictions: Dict of horizon -> (mean, lower, upper)
            - direction: Direction probabilities
            - volatility: Predicted volatility
            - hidden_states: Updated hidden states
            - attention_weights: (optional) Attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)
        
        # LSTM layers
        all_hidden = []
        for i, lstm_layer in enumerate(self.lstm_layers):
            h = hidden[i] if hidden is not None else None
            x, h_new = lstm_layer(x, h)
            all_hidden.append(h_new)
        
        # Self-attention
        attn_output, attn_weights = self.attention(x, return_attention=return_attention)
        x = self.attention_norm(x + attn_output)
        
        # Use last position for prediction
        last_hidden = x[:, -1, :]  # (batch, lstm_output_size)
        
        # Multi-horizon predictions
        predictions = {}
        for horizon_key, head in self.output_heads.items():
            output = head(last_hidden)
            predictions[horizon_key] = {
                'mean': output[:, 0],
                'lower': output[:, 1],
                'upper': output[:, 2]
            }
        
        # Direction prediction
        direction_logits = self.direction_head(last_hidden)
        direction_probs = F.softmax(direction_logits, dim=-1)
        
        # Volatility prediction
        volatility = self.volatility_head(last_hidden).squeeze(-1)
        
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
            'hidden_states': all_hidden,
            'last_hidden': last_hidden
        }
        
        if return_attention:
            result['attention_weights'] = attn_weights
        
        return result
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Predict with Monte Carlo Dropout uncertainty estimation.
        
        Args:
            x: Input tensor
            num_samples: Number of MC samples
            
        Returns:
            Predictions with uncertainty estimates
        """
        self.train()  # Enable dropout
        
        all_predictions = {f'h{h}': [] for h in self.config.prediction_horizons}
        all_directions = []
        all_volatilities = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                output = self.forward(x)
                
                for h in self.config.prediction_horizons:
                    all_predictions[f'h{h}'].append(
                        output['predictions'][f'h{h}']['mean']
                    )
                
                all_directions.append(output['direction']['probs'])
                all_volatilities.append(output['volatility'])
        
        self.eval()
        
        # Compute statistics
        result = {'predictions': {}}
        for h in self.config.prediction_horizons:
            preds = torch.stack(all_predictions[f'h{h}'], dim=0)
            result['predictions'][f'h{h}'] = {
                'mean': preds.mean(dim=0),
                'std': preds.std(dim=0),
                'lower_5': torch.quantile(preds, 0.05, dim=0),
                'upper_95': torch.quantile(preds, 0.95, dim=0)
            }
        
        directions = torch.stack(all_directions, dim=0)
        result['direction'] = {
            'mean': directions.mean(dim=0),
            'std': directions.std(dim=0)
        }
        
        volatilities = torch.stack(all_volatilities, dim=0)
        result['volatility'] = {
            'mean': volatilities.mean(dim=0),
            'std': volatilities.std(dim=0)
        }
        
        return result


class AttentionLSTM(FinancialLSTM):
    """
    LSTM with enhanced temporal attention mechanism.
    
    Adds temporal attention that learns to focus on important
    time steps in the input sequence.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        lstm_output_size = config.hidden_size * 2  # Bidirectional
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(lstm_output_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1, bias=False)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward with temporal attention pooling."""
        
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)
        
        # LSTM layers
        all_hidden = []
        for i, lstm_layer in enumerate(self.lstm_layers):
            h = hidden[i] if hidden is not None else None
            x, h_new = lstm_layer(x, h)
            all_hidden.append(h_new)
        
        # Temporal attention scores
        attn_scores = self.temporal_attention(x).squeeze(-1)  # (batch, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, seq_len)
        
        # Weighted sum over time
        context = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, seq_len)
            x  # (batch, seq_len, hidden)
        ).squeeze(1)  # (batch, hidden)
        
        # Self-attention on sequence
        attn_output, self_attn_weights = self.attention(x, return_attention=return_attention)
        x = self.attention_norm(x + attn_output)
        
        # Combine temporal context with last hidden
        last_hidden = x[:, -1, :]
        combined = last_hidden + context  # Residual connection
        
        # Multi-horizon predictions
        predictions = {}
        for horizon_key, head in self.output_heads.items():
            output = head(combined)
            predictions[horizon_key] = {
                'mean': output[:, 0],
                'lower': output[:, 1],
                'upper': output[:, 2]
            }
        
        # Direction and volatility
        direction_logits = self.direction_head(combined)
        direction_probs = F.softmax(direction_logits, dim=-1)
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
            'hidden_states': all_hidden,
            'last_hidden': combined,
            'temporal_attention': attn_weights
        }
        
        if return_attention:
            result['attention_weights'] = self_attn_weights
        
        return result
