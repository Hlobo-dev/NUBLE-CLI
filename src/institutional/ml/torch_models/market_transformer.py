"""
Market Transformer and Temporal Fusion Transformer
===================================================

State-of-the-art transformer architectures for financial forecasting:
- MarketTransformer: Custom transformer for OHLCV data
- TemporalFusionTransformer: Multi-horizon forecasting with interpretability

Features:
- Multi-horizon predictions
- Interpretable attention for feature importance
- Quantile regression for uncertainty
- Static and temporal covariate handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import math

from .base import (
    BaseFinancialModel,
    ModelConfig,
    PositionalEncoding,
    GatedResidualNetwork,
    VariableSelectionNetwork
)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with interpretable weights.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.out_linear(context)
        
        return output, attn_weights


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm and gated connections.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Gating for residual
        self.gate1 = nn.Linear(d_model, d_model)
        self.gate2 = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with pre-norm and gating."""
        
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, attn_weights = self.attention(normed, normed, normed, mask)
        
        # Gated residual
        gate = torch.sigmoid(self.gate1(x))
        x = x + gate * attn_out
        
        # Feed-forward with pre-norm
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        
        # Gated residual
        gate = torch.sigmoid(self.gate2(x))
        x = x + gate * ff_out
        
        return x, attn_weights


class MarketTransformer(BaseFinancialModel):
    """
    Custom Transformer optimized for financial market prediction.
    
    Features:
    - Specialized input embedding for OHLCV + indicators
    - Multi-scale temporal attention
    - Interpretable attention weights
    - Multi-horizon predictions with uncertainty
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        d_model = config.hidden_size
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(config.input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=config.sequence_length * 2,
            dropout=config.dropout,
            learnable=True
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=config.num_heads,
                d_ff=d_model * 4,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Prediction heads
        self.prediction_heads = nn.ModuleDict()
        for horizon in config.prediction_horizons:
            self.prediction_heads[f'h{horizon}'] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(d_model // 2, 5)  # mean, std, q10, q50, q90
            )
        
        # Direction head
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model // 2, 3)  # up, down, sideways
        )
        
        # Volatility regime head
        self.volatility_regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model // 2, 3)  # low, normal, high
        )
        
        self._init_weights()
        self.to_device()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            return_attention: Whether to return attention weights
            
        Returns:
            Dict with predictions, direction, volatility regime
        """
        batch_size, seq_len, _ = x.shape
        
        # Embed input
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Causal mask
        mask = self._create_causal_mask(seq_len, x.device)
        
        # Transformer blocks
        all_attention = []
        for block in self.transformer_blocks:
            x, attn = block(x, mask)
            if return_attention:
                all_attention.append(attn)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Use last position
        last_hidden = x[:, -1, :]
        
        # Predictions
        predictions = {}
        for h in self.config.prediction_horizons:
            out = self.prediction_heads[f'h{h}'](last_hidden)
            predictions[f'h{h}'] = {
                'mean': out[:, 0],
                'std': F.softplus(out[:, 1]),  # Ensure positive
                'q10': out[:, 2],
                'q50': out[:, 3],
                'q90': out[:, 4]
            }
        
        # Direction
        direction_logits = self.direction_head(last_hidden)
        direction_probs = F.softmax(direction_logits, dim=-1)
        
        # Volatility regime
        vol_regime_logits = self.volatility_regime_head(last_hidden)
        vol_regime_probs = F.softmax(vol_regime_logits, dim=-1)
        
        result = {
            'predictions': predictions,
            'direction': {
                'logits': direction_logits,
                'probs': direction_probs,
                'up': direction_probs[:, 0],
                'down': direction_probs[:, 1],
                'sideways': direction_probs[:, 2]
            },
            'volatility_regime': {
                'logits': vol_regime_logits,
                'probs': vol_regime_probs,
                'low': vol_regime_probs[:, 0],
                'normal': vol_regime_probs[:, 1],
                'high': vol_regime_probs[:, 2]
            },
            'last_hidden': last_hidden
        }
        
        if return_attention:
            result['attention_weights'] = all_attention
        
        return result


class TemporalFusionTransformer(BaseFinancialModel):
    """
    Temporal Fusion Transformer for multi-horizon forecasting.
    
    Based on the paper "Temporal Fusion Transformers for Interpretable 
    Multi-horizon Time Series Forecasting" (Lim et al., 2019).
    
    Features:
    - Variable selection for interpretability
    - Static covariate encoding
    - Temporal self-attention
    - Multi-horizon quantile forecasts
    - Gated residual connections throughout
    """
    
    def __init__(
        self,
        config: ModelConfig,
        num_static_features: int = 0,
        num_temporal_features: int = 64,
        quantiles: List[float] = None
    ):
        super().__init__(config)
        
        self.num_static = num_static_features
        self.num_temporal = num_temporal_features
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        
        d_model = config.hidden_size
        
        # Static variable selection (if static features exist)
        if num_static_features > 0:
            self.static_vsn = VariableSelectionNetwork(
                input_size=1,
                num_inputs=num_static_features,
                hidden_size=d_model,
                dropout=config.dropout
            )
            self.static_context_grn = GatedResidualNetwork(
                input_size=d_model,
                hidden_size=d_model,
                output_size=d_model,
                dropout=config.dropout
            )
        else:
            self.static_vsn = None
            self.static_context_grn = None
        
        # Temporal variable selection
        self.temporal_vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=num_temporal_features,
            hidden_size=d_model,
            dropout=config.dropout,
            context_size=d_model if num_static_features > 0 else None
        )
        
        # LSTM encoder for local processing
        self.lstm_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True,
            bidirectional=False
        )
        self.lstm_encoder_norm = nn.LayerNorm(d_model)
        
        # Self-attention for long-range dependencies
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        self.attention_norm = nn.LayerNorm(d_model)
        
        # Post-attention GRN
        self.post_attention_grn = GatedResidualNetwork(
            input_size=d_model,
            hidden_size=d_model,
            output_size=d_model,
            dropout=config.dropout
        )
        
        # Output layers for each horizon and quantile
        self.output_heads = nn.ModuleDict()
        for horizon in config.prediction_horizons:
            self.output_heads[f'h{horizon}'] = nn.Linear(
                d_model, len(self.quantiles)
            )
        
        # Feature importance tracking
        self.register_buffer(
            'feature_importance',
            torch.zeros(num_temporal_features)
        )
        self._importance_count = 0
        
        self._init_weights()
        self.to_device()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(
        self,
        temporal_inputs: torch.Tensor,
        static_inputs: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        update_importance: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            temporal_inputs: (batch, seq_len, num_temporal_features)
            static_inputs: (batch, num_static_features) or None
            return_attention: Whether to return attention weights
            update_importance: Whether to update feature importance
            
        Returns:
            Dict with quantile predictions and interpretability info
        """
        batch_size, seq_len, num_features = temporal_inputs.shape
        
        # Reshape for variable selection: (batch, seq_len, num_features, 1)
        temporal_reshaped = temporal_inputs.unsqueeze(-1)
        
        # Static context
        if static_inputs is not None and self.static_vsn is not None:
            static_reshaped = static_inputs.unsqueeze(-1)
            static_embedding, static_weights = self.static_vsn(static_reshaped)
            static_context = self.static_context_grn(static_embedding)
        else:
            static_context = None
            static_weights = None
        
        # Temporal variable selection at each time step
        temporal_embeddings = []
        all_temporal_weights = []
        
        for t in range(seq_len):
            temporal_t = temporal_reshaped[:, t, :, :]  # (batch, num_features, 1)
            embedding, weights = self.temporal_vsn(temporal_t, static_context)
            temporal_embeddings.append(embedding)
            all_temporal_weights.append(weights)
        
        # Stack temporal embeddings: (batch, seq_len, d_model)
        temporal_embeddings = torch.stack(temporal_embeddings, dim=1)
        temporal_weights = torch.stack(all_temporal_weights, dim=1)  # (batch, seq_len, num_features)
        
        # Update feature importance
        if update_importance and self.training:
            avg_weights = temporal_weights.mean(dim=(0, 1))
            self.feature_importance = (
                self._importance_count * self.feature_importance + avg_weights
            ) / (self._importance_count + 1)
            self._importance_count += 1
        
        # LSTM encoder
        lstm_out, _ = self.lstm_encoder(temporal_embeddings)
        lstm_out = self.lstm_encoder_norm(temporal_embeddings + lstm_out)
        
        # Self-attention with causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=temporal_inputs.device))
        attn_out, attn_weights = self.self_attention(lstm_out, lstm_out, lstm_out, mask)
        attn_out = self.attention_norm(lstm_out + attn_out)
        
        # Post-attention processing
        output = self.post_attention_grn(attn_out)
        
        # Use last position for predictions
        last_hidden = output[:, -1, :]
        
        # Quantile predictions for each horizon
        predictions = {}
        for horizon in self.config.prediction_horizons:
            quantile_preds = self.output_heads[f'h{horizon}'](last_hidden)
            predictions[f'h{horizon}'] = {
                f'q{int(q*100)}': quantile_preds[:, i]
                for i, q in enumerate(self.quantiles)
            }
            # Add mean as average of quantiles
            predictions[f'h{horizon}']['mean'] = quantile_preds.mean(dim=-1)
        
        result = {
            'predictions': predictions,
            'feature_importance': {
                'temporal': self.feature_importance.cpu().numpy(),
                'current': temporal_weights[:, -1, :].cpu().numpy()
            },
            'last_hidden': last_hidden
        }
        
        if static_weights is not None:
            result['feature_importance']['static'] = static_weights.cpu().numpy()
        
        if return_attention:
            result['attention_weights'] = attn_weights
        
        return result
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get accumulated feature importance scores."""
        return {
            'temporal': self.feature_importance.cpu().numpy(),
            'count': self._importance_count
        }
    
    def reset_feature_importance(self):
        """Reset feature importance tracking."""
        self.feature_importance.zero_()
        self._importance_count = 0
