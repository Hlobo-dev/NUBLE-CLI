"""
Informer: Efficient Transformer for Long Sequence Time-Series Forecasting
=========================================================================

Based on: "Informer: Beyond Efficient Transformer for Long Sequence 
Time-Series Forecasting" - Zhou et al., 2021 (AAAI 2021 Best Paper)

Key Innovations:
- ProbSparse Self-Attention: O(L log L) complexity vs O(L²)
- Self-Attention Distilling: Halves input each layer
- Generative Style Decoder: One-shot multi-step prediction

This enables efficient handling of very long sequences (1000+) that
would be intractable with standard Transformers.

Reference: https://arxiv.org/abs/2012.07436
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


@dataclass
class InformerConfig:
    """
    Configuration for Informer model.
    
    Optimized defaults for financial time series.
    """
    # Input dimensions
    input_size: int = 32               # Number of input features
    output_size: int = 1               # Number of output features
    
    # Sequence lengths
    context_length: int = 96           # Encoder sequence length
    label_length: int = 48             # Start token length for decoder
    prediction_length: int = 24        # Prediction horizon
    
    # Model dimensions
    d_model: int = 512                 # Model dimension
    n_heads: int = 8                   # Number of attention heads
    e_layers: int = 3                  # Encoder layers
    d_layers: int = 2                  # Decoder layers
    d_ff: int = 2048                   # Feed-forward dimension
    
    # ProbSparse attention
    factor: int = 5                    # ProbSparse factor (top-k queries)
    
    # Regularization
    dropout: float = 0.1
    
    # Embedding
    embed_type: str = 'timeF'          # 'fixed', 'learned', 'timeF'
    freq: str = 'h'                    # Time frequency
    
    # Distilling
    distil: bool = True                # Use distilling in encoder
    
    # Output
    output_attention: bool = False


class TriangularCausalMask:
    """Causal mask for decoder self-attention."""
    
    def __init__(self, B: int, L: int, device: torch.device):
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        self._mask = mask.unsqueeze(0).expand(B, -1, -1)
        
    @property
    def mask(self) -> Tensor:
        return self._mask


class ProbMask:
    """Mask for ProbSparse attention."""
    
    def __init__(self, B: int, H: int, L: int, index: Tensor, scores: Tensor, device: torch.device):
        # Create mask where only top-k queries are active
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device)
        _mask[index.reshape(-1), :] = False
        self._mask = _mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        
    @property
    def mask(self) -> Tensor:
        return self._mask


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """Convolutional token embedding."""
    
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')
        
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq_len, features]
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        x = self.tokenConv(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, d_model]
        return x


class TemporalEmbedding(nn.Module):
    """Learnable temporal embeddings for time features."""
    
    def __init__(self, d_model: int, embed_type: str = 'timeF', freq: str = 'h'):
        super().__init__()
        
        # Embedding dimensions based on time feature cardinality
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        
        if freq == 't':  # minutely
            self.minute_embed = nn.Embedding(minute_size, d_model)
        self.hour_embed = nn.Embedding(hour_size, d_model)
        self.weekday_embed = nn.Embedding(weekday_size, d_model)
        self.day_embed = nn.Embedding(day_size, d_model)
        self.month_embed = nn.Embedding(month_size, d_model)
        
        self.freq = freq
        
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq_len, num_time_features]
        # Expected features: [month, day, weekday, hour, (minute)]
        
        month = x[..., 0].long()
        day = x[..., 1].long()
        weekday = x[..., 2].long()
        hour = x[..., 3].long()
        
        embed = (
            self.month_embed(month) +
            self.day_embed(day) +
            self.weekday_embed(weekday) +
            self.hour_embed(hour)
        )
        
        if self.freq == 't' and x.shape[-1] > 4:
            minute = x[..., 4].long()
            embed = embed + self.minute_embed(minute)
            
        return embed


class DataEmbedding(nn.Module):
    """
    Complete data embedding: token + position + temporal.
    """
    
    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = 'timeF',
        freq: str = 'h',
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.temporal_embedding = TemporalEmbedding(d_model, embed_type, freq)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, x_mark: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: [batch, seq_len, features] value data
            x_mark: [batch, seq_len, time_features] temporal markers
        """
        # First embed values, then add position
        value_emb = self.value_embedding(x)
        embedded = self.position_embedding(value_emb)
        
        if x_mark is not None:
            embedded = embedded + self.temporal_embedding(x_mark)
            
        return self.dropout(embedded)


class ProbSparseAttention(nn.Module):
    """
    ProbSparse Self-Attention Mechanism.
    
    Key Innovation: Instead of computing attention for all queries,
    identify the "dominant" queries using sampling, then compute
    attention only for those. O(L log L) complexity.
    
    The idea: Most queries produce nearly uniform attention distributions.
    Only a few "active" queries have sparse, peaky attention. We can
    approximate by only computing attention for those.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        factor: int = 5,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.d_k = d_model // n_heads
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(attention_dropout)
        
    def _prob_QK(self, Q: Tensor, K: Tensor, sample_k: int, n_top: int) -> Tuple[Tensor, Tensor]:
        """
        Compute sparsity measurement M and find top-k queries.
        
        Args:
            Q: [batch, heads, seq_len, d_k]
            K: [batch, heads, seq_len, d_k]
            sample_k: Number of keys to sample
            n_top: Number of top queries to keep
        """
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape
        
        # Sample keys uniformly
        K_sample = K[:, :, torch.randint(L_K, (sample_k,)), :]  # [B, H, sample_k, D]
        
        # Compute Q @ K^T for sampled keys
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1))  # [B, H, L_Q, sample_k]
        
        # Find sparsity measurement M(q_i)
        # M(q_i) = max(q_i * k_j) - mean(q_i * k_j)
        # High M means query has focused attention (sparse)
        M = Q_K_sample.max(dim=-1).values - Q_K_sample.mean(dim=-1)  # [B, H, L_Q]
        
        # Select top-n_top queries with highest sparsity
        M_top = M.topk(n_top, sorted=False).indices  # [B, H, n_top]
        
        return M_top, M
        
    def _get_initial_context(self, V: Tensor, L_Q: int) -> Tensor:
        """Get initial context by averaging values."""
        B, H, L_V, D = V.shape
        
        # Use mean of all values as default context
        V_mean = V.mean(dim=2, keepdim=True)  # [B, H, 1, D]
        return V_mean.expand(-1, -1, L_Q, -1)  # [B, H, L_Q, D]
        
    def _update_context(
        self,
        context: Tensor,
        V: Tensor,
        scores: Tensor,
        index: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Update context at selected indices."""
        B, H, L_V, D = V.shape
        
        # Compute attention for selected queries
        attn = self.dropout(F.softmax(scores, dim=-1))
        
        # Update context at selected positions
        batch_idx = torch.arange(B, device=index.device)[:, None, None].expand(-1, H, index.shape[-1])
        head_idx = torch.arange(H, device=index.device)[None, :, None].expand(B, -1, index.shape[-1])
        
        context[batch_idx, head_idx, index, :] = torch.matmul(attn, V)
        
        return context, attn
        
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        ProbSparse attention forward pass.
        
        Args:
            queries, keys, values: [batch, seq_len, d_model]
            attn_mask: Optional mask
            
        Returns:
            output: [batch, seq_len, d_model]
            attention: Attention weights (for top queries only)
        """
        B, L_Q, _ = queries.shape
        _, L_K, _ = keys.shape
        
        # Project and reshape
        Q = self.query_proj(queries).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key_proj(keys).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value_proj(values).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        
        # Number of queries to actively compute
        U = self.factor * int(np.ceil(np.log(L_K + 1)))  # Sample keys
        u = self.factor * int(np.ceil(np.log(L_Q + 1)))  # Top queries
        
        U = min(U, L_K)
        u = min(u, L_Q)
        
        # Find top queries
        scores_top_idx, M = self._prob_QK(Q, K, U, u)
        
        # Scale
        scale = 1.0 / math.sqrt(self.d_k)
        
        # Initialize context with mean of values
        context = self._get_initial_context(V, L_Q)
        
        # Compute attention only for top queries
        # Gather Q at top indices
        batch_idx = torch.arange(B, device=Q.device)[:, None, None, None].expand(-1, self.n_heads, u, self.d_k)
        head_idx = torch.arange(self.n_heads, device=Q.device)[None, :, None, None].expand(B, -1, u, self.d_k)
        query_idx = scores_top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
        
        Q_selected = Q[batch_idx, head_idx, query_idx, torch.arange(self.d_k, device=Q.device)]
        
        # Compute attention scores
        scores = torch.matmul(Q_selected, K.transpose(-2, -1)) * scale
        
        # Simplified mask handling - if mask provided and compatible, apply it
        if attn_mask is not None:
            if attn_mask.dim() == 4 and attn_mask.shape[2] >= u:
                scores = scores.masked_fill(attn_mask[:, :, :u, :L_K], float('-inf'))
            elif attn_mask.dim() == 3:
                # Broadcast mask over heads
                scores = scores.masked_fill(attn_mask[:, :u, :L_K].unsqueeze(1), float('-inf'))
            
        # Update context at selected positions
        context, attn = self._update_context(context, V, scores, scores_top_idx)
        
        # Reshape output
        context = context.transpose(1, 2).contiguous().view(B, L_Q, -1)
        output = self.out_proj(context)
        
        return output, attn


class AttentionLayer(nn.Module):
    """
    Wrapper for attention with pre-norm and residual.
    """
    
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = attention
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Pre-norm attention with residual.
        """
        queries_norm = self.norm(queries)
        attn_out, attn_weights = self.attention(queries_norm, keys, values, attn_mask)
        return queries + self.dropout(attn_out), attn_weights


class ConvLayer(nn.Module):
    """
    Distilling layer: Conv + MaxPool to halve sequence length.
    
    This is key to Informer's efficiency - each encoder layer
    reduces sequence length by half, creating a pyramid.
    """
    
    def __init__(self, c_in: int):
        super().__init__()
        
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq_len, d_model]
        x = x.permute(0, 2, 1)  # [batch, d_model, seq_len]
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len//2, d_model]
        return x


class EncoderLayer(nn.Module):
    """
    Informer Encoder Layer.
    
    Attention -> Feed Forward (with residuals and norms)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        factor: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = AttentionLayer(
            ProbSparseAttention(d_model, n_heads, factor, dropout),
            d_model,
            dropout
        )
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        # Self attention
        attn_out, attn_weights = self.attention(x, x, x, attn_mask)
        
        # Feed forward with residual
        ff_out = self.ff(self.norm2(attn_out))
        output = attn_out + self.dropout(ff_out)
        
        return output, attn_weights


class Encoder(nn.Module):
    """
    Informer Encoder with optional distilling.
    
    Stacks encoder layers with conv distilling layers between them
    to progressively reduce sequence length.
    """
    
    def __init__(
        self,
        layers: nn.ModuleList,
        conv_layers: Optional[nn.ModuleList] = None,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.layers = layers
        self.conv_layers = conv_layers
        self.norm = norm_layer
        
    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, List[Tensor]]:
        attns = []
        
        if self.conv_layers is not None:
            for layer, conv in zip(self.layers, self.conv_layers):
                x, attn = layer(x, attn_mask)
                x = conv(x)
                attns.append(attn)
                
            # Last layer without conv
            x, attn = self.layers[-1](x, attn_mask)
            attns.append(attn)
        else:
            for layer in self.layers:
                x, attn = layer(x, attn_mask)
                attns.append(attn)
                
        if self.norm is not None:
            x = self.norm(x)
            
        return x, attns


class DecoderLayer(nn.Module):
    """
    Informer Decoder Layer.
    
    Self Attention -> Cross Attention -> Feed Forward
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        factor: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self attention (causal)
        self.self_attention = AttentionLayer(
            ProbSparseAttention(d_model, n_heads, factor, dropout),
            d_model,
            dropout
        )
        
        # Cross attention (full, not sparse)
        self.cross_attention = AttentionLayer(
            ProbSparseAttention(d_model, n_heads, factor, dropout),
            d_model,
            dropout
        )
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: Tensor,
        cross: Tensor,
        self_mask: Optional[Tensor] = None,
        cross_mask: Optional[Tensor] = None
    ) -> Tensor:
        # Self attention
        x, _ = self.self_attention(x, x, x, self_mask)
        
        # Cross attention
        x, _ = self.cross_attention(x, cross, cross, cross_mask)
        
        # Feed forward
        ff_out = self.ff(self.norm3(x))
        output = x + self.dropout(ff_out)
        
        return output


class Decoder(nn.Module):
    """Informer Decoder."""
    
    def __init__(
        self,
        layers: nn.ModuleList,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.layers = layers
        self.norm = norm_layer
        
    def forward(
        self,
        x: Tensor,
        cross: Tensor,
        self_mask: Optional[Tensor] = None,
        cross_mask: Optional[Tensor] = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, cross, self_mask, cross_mask)
            
        if self.norm is not None:
            x = self.norm(x)
            
        return x


class InformerV2(nn.Module):
    """
    Informer - Efficient Transformer for Long Sequence Forecasting.
    
    State-of-the-art model for long-horizon time series prediction
    with O(L log L) complexity instead of O(L²).
    
    Key Components:
    1. ProbSparse Attention: Efficient attention mechanism
    2. Distilling: Progressive sequence reduction
    3. Generative Decoder: Direct multi-step prediction
    
    Usage:
        config = InformerConfig(
            input_size=32,
            context_length=96,
            prediction_length=24
        )
        model = InformerV2(config)
        
        # x_enc: [batch, context_length, input_size]
        # x_mark_enc: [batch, context_length, time_features]
        # x_dec: [batch, label_length + pred_length, input_size]
        # x_mark_dec: [batch, label_length + pred_length, time_features]
        
        outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    """
    
    def __init__(self, config: InformerConfig):
        super().__init__()
        
        self.config = config
        self.pred_len = config.prediction_length
        self.label_len = config.label_length
        
        # Embedding layers
        self.enc_embedding = DataEmbedding(
            config.input_size, config.d_model,
            config.embed_type, config.freq, config.dropout
        )
        self.dec_embedding = DataEmbedding(
            config.input_size, config.d_model,
            config.embed_type, config.freq, config.dropout
        )
        
        # Encoder
        encoder_layers = nn.ModuleList([
            EncoderLayer(
                config.d_model, config.n_heads, config.d_ff,
                config.factor, config.dropout
            )
            for _ in range(config.e_layers)
        ])
        
        conv_layers = None
        if config.distil:
            conv_layers = nn.ModuleList([
                ConvLayer(config.d_model)
                for _ in range(config.e_layers - 1)
            ])
            
        self.encoder = Encoder(
            encoder_layers,
            conv_layers,
            nn.LayerNorm(config.d_model)
        )
        
        # Decoder
        decoder_layers = nn.ModuleList([
            DecoderLayer(
                config.d_model, config.n_heads, config.d_ff,
                config.factor, config.dropout
            )
            for _ in range(config.d_layers)
        ])
        
        self.decoder = Decoder(
            decoder_layers,
            nn.LayerNorm(config.d_model)
        )
        
        # Output projection
        self.projection = nn.Linear(config.d_model, config.output_size)
        
    def forward(
        self,
        x_enc: Tensor,
        x_mark_enc: Optional[Tensor] = None,
        x_dec: Optional[Tensor] = None,
        x_mark_dec: Optional[Tensor] = None,
        enc_self_mask: Optional[Tensor] = None,
        dec_self_mask: Optional[Tensor] = None,
        dec_enc_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Forward pass.
        
        Args:
            x_enc: Encoder input [batch, context_length, input_size]
            x_mark_enc: Encoder time features [batch, context_length, time_features]
            x_dec: Decoder input [batch, label_length + pred_length, input_size]
            x_mark_dec: Decoder time features
            *_mask: Optional attention masks
            
        Returns:
            Dict with predictions and optional attention weights
        """
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, enc_self_mask)
        
        # Decoder
        if x_dec is None:
            # Create decoder input: last label_len of encoder + zeros for prediction
            batch_size = x_enc.shape[0]
            x_dec = torch.zeros(
                batch_size,
                self.label_len + self.pred_len,
                self.config.input_size,
                device=x_enc.device
            )
            x_dec[:, :self.label_len, :] = x_enc[:, -self.label_len:, :]
            
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        # Create causal mask for decoder
        if dec_self_mask is None:
            dec_self_mask = TriangularCausalMask(
                dec_out.shape[0], dec_out.shape[1], dec_out.device
            ).mask
            
        dec_out = self.decoder(dec_out, enc_out, dec_self_mask, dec_enc_mask)
        
        # Project to output
        output = self.projection(dec_out)
        
        # Return only prediction part
        prediction = output[:, -self.pred_len:, :]
        
        result = {
            'prediction': prediction.squeeze(-1) if self.config.output_size == 1 else prediction,
            'forecast': prediction.squeeze(-1) if self.config.output_size == 1 else prediction,
        }
        
        if self.config.output_attention:
            result['encoder_attention'] = attns
            
        return result
    
    def predict(
        self,
        x_enc: Tensor,
        x_mark_enc: Optional[Tensor] = None,
        x_mark_dec: Optional[Tensor] = None
    ) -> Tensor:
        """
        Simple prediction interface.
        
        Creates decoder input automatically and returns forecast.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x_enc, x_mark_enc, None, x_mark_dec)
        return output['forecast']


class InformerLoss(nn.Module):
    """Loss function for Informer."""
    
    def __init__(self, loss_type: str = 'mse'):
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
            predictions: [batch, pred_length, output_size]
            targets: Same shape as predictions
            mask: Optional mask for missing values
        """
        if predictions.dim() == 2:
            predictions = predictions.unsqueeze(-1)
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)
            
        if mask is not None:
            predictions = predictions * mask
            targets = targets * mask
            
        if self.loss_type == 'mse':
            loss = F.mse_loss(predictions, targets)
        elif self.loss_type == 'mae':
            loss = F.l1_loss(predictions, targets)
        else:
            loss = F.mse_loss(predictions, targets)
            
        mae = F.l1_loss(predictions, targets)
        mse = F.mse_loss(predictions, targets)
        
        return {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'rmse': torch.sqrt(mse)
        }
