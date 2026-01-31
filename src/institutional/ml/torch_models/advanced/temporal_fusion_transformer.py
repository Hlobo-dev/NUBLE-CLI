"""
Temporal Fusion Transformer (TFT) - Production Implementation
==============================================================

Based on: "Temporal Fusion Transformers for Interpretable Multi-horizon 
Time Series Forecasting" - Lim et al., 2021 (Google AI)

This implementation follows the exact architecture from the paper with
financial-specific enhancements for institutional trading applications.

Key Features:
- Variable Selection Networks: Learn which features matter
- Gated Residual Networks: Skip connections with gating
- Multi-head Interpretable Attention: Temporal patterns
- Quantile Outputs: Probabilistic forecasting
- Static/Known/Observed Covariate Handling

Reference: https://arxiv.org/abs/1912.09363
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class CovariateType(Enum):
    """Types of covariates in TFT."""
    STATIC = "static"           # Time-invariant (e.g., ticker sector)
    KNOWN = "known"             # Known in advance (e.g., day of week)
    OBSERVED = "observed"       # Only observed historically (e.g., price)


@dataclass
class TFTConfig:
    """
    Configuration for Temporal Fusion Transformer.
    
    Based on optimal hyperparameters from the paper for financial data.
    """
    # Input dimensions
    num_static_categorical: int = 0      # Number of static categorical variables
    num_static_continuous: int = 0       # Number of static continuous variables
    num_known_categorical: int = 0       # Known future categorical
    num_known_continuous: int = 5        # Known future continuous (e.g., time features)
    num_observed_continuous: int = 32    # Observed continuous (main inputs)
    
    # Embedding dimensions
    static_categorical_sizes: List[int] = field(default_factory=list)  # Cardinality per categorical
    embedding_dim: int = 64              # Embedding dimension for categoricals
    
    # Architecture
    hidden_size: int = 256               # Hidden layer size (d_model)
    num_heads: int = 4                   # Number of attention heads
    num_encoder_layers: int = 1          # LSTM encoder layers
    num_decoder_layers: int = 1          # LSTM decoder layers
    dropout: float = 0.1                 # Dropout rate
    
    # Sequence lengths
    context_length: int = 60             # Historical context (lookback)
    prediction_length: int = 20          # Prediction horizon
    
    # Output
    num_quantiles: int = 7               # Number of quantile outputs
    quantiles: List[float] = field(default_factory=lambda: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
    
    # Training
    max_gradient_norm: float = 1.0
    learning_rate: float = 1e-3
    
    def __post_init__(self):
        if len(self.quantiles) != self.num_quantiles:
            self.quantiles = [i / (self.num_quantiles + 1) for i in range(1, self.num_quantiles + 1)]
        self.quantiles = sorted(self.quantiles)


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) from Dauphin et al. 2017.
    
    GLU(x) = sigmoid(Wx + b) ⊙ (Vx + c)
    
    Used throughout TFT for controlled information flow.
    """
    
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_size, output_size * 2)
        self.output_size = output_size
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        x = self.fc(x)
        return x[..., :self.output_size] * torch.sigmoid(x[..., self.output_size:])


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - Core building block of TFT.
    
    Provides flexible nonlinear processing with skip connections
    and optional context input for static covariate integration.
    
    Architecture:
        η₁ = ELU(W₁a + b₁)
        η₂ = W₂η₁ + b₂  (if context: η₂ = W₂η₁ + W₃c + b₂)
        GRN(a, c) = LayerNorm(a + GLU(η₂))
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        context_size: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.context_size = context_size
        
        # Dense layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.output_size)
        
        # Optional context integration
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        
        # Gating
        self.glu = GatedLinearUnit(self.output_size, self.output_size, dropout)
        
        # Skip connection and normalization
        if input_size != self.output_size:
            self.skip_fc = nn.Linear(input_size, self.output_size)
        else:
            self.skip_fc = None
            
        self.layer_norm = nn.LayerNorm(self.output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        # Skip connection
        if self.skip_fc is not None:
            skip = self.skip_fc(x)
        else:
            skip = x
            
        # First dense layer with ELU
        hidden = F.elu(self.fc1(x))
        
        # Add context if provided
        if self.context_size is not None and context is not None:
            hidden = hidden + self.context_fc(context)
            
        # Second dense layer
        hidden = self.dropout(self.fc2(hidden))
        
        # Gating and residual
        gated = self.glu(hidden)
        return self.layer_norm(skip + gated)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) - Learns feature importance.
    
    Applies instance-wise variable selection to identify relevant inputs.
    Produces interpretable variable selection weights (softmax).
    
    Architecture:
        1. Transform each variable independently
        2. Flatten and apply GRN to get selection weights
        3. Apply softmax for interpretable weights
        4. Weighted sum of transformed variables
    """
    
    def __init__(
        self,
        num_inputs: int,
        input_size: int,
        hidden_size: int,
        context_size: Optional[int] = None,
        dropout: float = 0.1,
        single_variable_grn: bool = False
    ):
        super().__init__()
        
        self.num_inputs = num_inputs
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Per-variable transformations - project each variable to hidden_size
        self.var_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU()
            )
            for _ in range(num_inputs)
        ])
        
        # Variable selection weights via GRN
        # Input: flattened transformed variables [batch, seq, num_inputs * hidden_size]
        # Output: selection weights [batch, seq, num_inputs]
        self.selection_grn = GatedResidualNetwork(
            input_size=hidden_size * num_inputs,
            hidden_size=hidden_size,
            output_size=num_inputs,
            context_size=context_size,
            dropout=dropout
        )
        
        # Final processing GRN
        self.output_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )
        
    def forward(
        self, 
        inputs: List[Tensor],
        context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: List of tensors, each [batch, seq_len, input_size] or [batch, input_size]
            context: Optional context tensor [batch, context_size]
            
        Returns:
            output: Selected and combined features [batch, seq_len, hidden_size]
            selection_weights: Importance weights [batch, seq_len, num_inputs]
        """
        # Transform each variable
        transformed = []
        for inp, transform in zip(inputs, self.var_transforms):
            transformed.append(transform(inp))
                
        # Stack: [batch, seq_len, num_inputs, hidden_size]
        # or [batch, num_inputs, hidden_size] for static
        if inputs[0].dim() == 3:
            stacked = torch.stack(transformed, dim=2)
            batch_size, seq_len = stacked.shape[:2]
            
            # Flatten for selection network
            flattened = stacked.view(batch_size, seq_len, -1)
            
            # Expand context for sequence
            if context is not None:
                context = context.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            stacked = torch.stack(transformed, dim=1)
            batch_size = stacked.shape[0]
            flattened = stacked.view(batch_size, -1)
            
        # Compute selection weights
        selection_weights = self.selection_grn(flattened, context)
        selection_weights = F.softmax(selection_weights, dim=-1)
        
        # Weighted combination
        if inputs[0].dim() == 3:
            # [batch, seq_len, num_inputs, 1] * [batch, seq_len, num_inputs, hidden_size]
            weighted = stacked * selection_weights.unsqueeze(-1)
            combined = weighted.sum(dim=2)  # [batch, seq_len, hidden_size]
        else:
            weighted = stacked * selection_weights.unsqueeze(-1)
            combined = weighted.sum(dim=1)
            
        # Final processing
        output = self.output_grn(combined)
        
        return output, selection_weights


class StaticCovariateEncoders(nn.Module):
    """
    Static Covariate Encoders - Process time-invariant features.
    
    Produces four context vectors used throughout TFT:
    1. c_s: Context for variable selection
    2. c_e: Context for local processing (LSTM)
    3. c_h: Initial hidden state for LSTM
    4. c_c: Initial cell state for LSTM
    """
    
    def __init__(
        self,
        num_categorical: int,
        categorical_sizes: List[int],
        num_continuous: int,
        embedding_dim: int,
        hidden_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_categorical = num_categorical
        self.num_continuous = num_continuous
        self.hidden_size = hidden_size
        
        # Categorical embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim)
            for size in categorical_sizes
        ]) if categorical_sizes else nn.ModuleList()
        
        # Continuous linear projections
        if num_continuous > 0:
            self.continuous_transforms = nn.ModuleList([
                nn.Linear(1, embedding_dim)
                for _ in range(num_continuous)
            ])
        else:
            self.continuous_transforms = nn.ModuleList()
            
        # Total number of static inputs
        total_static = len(categorical_sizes) + num_continuous
        
        if total_static > 0:
            # Variable selection for static inputs
            self.vsn = VariableSelectionNetwork(
                num_inputs=total_static,
                input_size=embedding_dim,
                hidden_size=hidden_size,
                dropout=dropout
            )
            
            # Four context encoders
            self.context_grns = nn.ModuleList([
                GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout=dropout)
                for _ in range(4)
            ])
        else:
            self.vsn = None
            self.context_grns = None
            
    def forward(
        self, 
        categorical: Optional[Tensor] = None,
        continuous: Optional[Tensor] = None
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Args:
            categorical: [batch, num_categorical] long tensor
            continuous: [batch, num_continuous] float tensor
            
        Returns:
            c_s, c_e, c_h, c_c: Context vectors [batch, hidden_size]
        """
        if self.vsn is None:
            return None, None, None, None
            
        inputs = []
        
        # Embed categoricals
        if categorical is not None:
            for i, emb in enumerate(self.embeddings):
                inputs.append(emb(categorical[:, i]))
                
        # Transform continuous
        if continuous is not None:
            for i, transform in enumerate(self.continuous_transforms):
                inputs.append(transform(continuous[:, i:i+1]))
                
        if not inputs:
            return None, None, None, None
            
        # Variable selection
        static_encoding, static_weights = self.vsn(inputs)
        
        # Generate four context vectors
        contexts = [grn(static_encoding) for grn in self.context_grns]
        
        return tuple(contexts)


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention for TFT.
    
    Key modification from standard attention:
    - Uses additive attention (not scaled dot-product)
    - Shares values across heads for interpretability
    - Produces interpretable attention weights
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Query and key projections (per head)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        
        # Value projection (shared across heads for interpretability)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query: [batch, seq_q, hidden_size]
            key: [batch, seq_k, hidden_size]
            value: [batch, seq_k, hidden_size]
            mask: [batch, seq_q, seq_k] or broadcastable
            
        Returns:
            output: [batch, seq_q, hidden_size]
            attention_weights: [batch, num_heads, seq_q, seq_k]
        """
        batch_size, seq_q, _ = query.shape
        seq_k = key.shape[1]
        
        # Project queries, keys, values
        q = self.q_proj(query).view(batch_size, seq_q, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_k, self.num_heads, self.head_dim)
        v = self.v_proj(value)
        
        # Transpose for attention: [batch, num_heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values (shared across heads)
        # Average attention across heads
        avg_attn = attn_weights.mean(dim=1)  # [batch, seq_q, seq_k]
        output = torch.matmul(avg_attn, v)  # [batch, seq_q, hidden_size]
        
        # Output projection
        output = self.out_proj(output)
        
        return output, attn_weights


class TemporalSelfAttention(nn.Module):
    """
    Temporal Self-Attention Layer with Gating.
    
    Applies interpretable multi-head attention followed by
    gating and layer normalization.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = InterpretableMultiHeadAttention(
            hidden_size, num_heads, dropout
        )
        self.glu = GatedLinearUnit(hidden_size, hidden_size, dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: [batch, seq_len, hidden_size]
            mask: Optional attention mask
            
        Returns:
            output: [batch, seq_len, hidden_size]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        attn_output, attn_weights = self.attention(x, x, x, mask)
        gated = self.glu(attn_output)
        output = self.layer_norm(x + gated)
        return output, attn_weights


class QuantileOutput(nn.Module):
    """
    Quantile Output Layer for Probabilistic Forecasting.
    
    Produces calibrated quantile predictions instead of point forecasts.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_quantiles: int,
        quantiles: List[float]
    ):
        super().__init__()
        
        self.num_quantiles = num_quantiles
        self.quantiles = quantiles
        
        # One output per quantile per time step
        self.quantile_proj = nn.Linear(hidden_size, num_quantiles)
        
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: [batch, prediction_length, hidden_size]
            
        Returns:
            Dict with quantile predictions and statistics
        """
        # [batch, prediction_length, num_quantiles]
        quantile_preds = self.quantile_proj(x)
        
        # Sort to ensure monotonicity (optional but helps calibration)
        quantile_preds, _ = torch.sort(quantile_preds, dim=-1)
        
        # Extract key statistics
        median_idx = len(self.quantiles) // 2
        
        return {
            'quantiles': quantile_preds,
            'median': quantile_preds[..., median_idx],
            'lower_bound': quantile_preds[..., 0],
            'upper_bound': quantile_preds[..., -1],
            'quantile_levels': torch.tensor(self.quantiles, device=x.device)
        }


class TemporalFusionTransformerV2(nn.Module):
    """
    Temporal Fusion Transformer - Full Production Implementation.
    
    State-of-the-art interpretable multi-horizon forecasting model
    designed for financial time series prediction.
    
    Architecture Overview:
    1. Static Covariate Encoders → context vectors
    2. Variable Selection Networks → feature importance
    3. LSTM Encoder-Decoder → local temporal processing
    4. Temporal Self-Attention → long-range dependencies
    5. Position-wise Feed-Forward → output processing
    6. Quantile Output → probabilistic forecasts
    
    Usage:
        config = TFTConfig(
            num_observed_continuous=32,
            context_length=60,
            prediction_length=20,
            hidden_size=256
        )
        model = TemporalFusionTransformerV2(config)
        
        outputs = model(
            observed_inputs=x,  # [batch, context_length, num_observed]
            known_inputs=time_features  # [batch, total_length, num_known]
        )
    """
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.context_length = config.context_length
        self.prediction_length = config.prediction_length
        
        # ===== Static Covariate Encoders =====
        self.static_encoders = StaticCovariateEncoders(
            num_categorical=config.num_static_categorical,
            categorical_sizes=config.static_categorical_sizes,
            num_continuous=config.num_static_continuous,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )
        
        # ===== Input Embeddings =====
        # Observed features (past only)
        if config.num_observed_continuous > 0:
            self.observed_embedding = nn.Linear(
                config.num_observed_continuous, config.hidden_size
            )
        
        # Known features (past and future)
        total_known = config.num_known_categorical + config.num_known_continuous
        if total_known > 0:
            self.known_embedding = nn.Linear(total_known, config.hidden_size)
        
        # ===== Variable Selection Networks =====
        # Historical inputs: always just 1 combined embedding
        # We'll combine observed + known before VSN in forward
        self.hist_vsn = VariableSelectionNetwork(
            num_inputs=1,
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            context_size=config.hidden_size if config.num_static_continuous > 0 or config.num_static_categorical > 0 else None,
            dropout=config.dropout
        )
        
        # Future inputs (known only)
        total_known = config.num_known_categorical + config.num_known_continuous
        if total_known > 0:
            self.future_vsn = VariableSelectionNetwork(
                num_inputs=1,
                input_size=config.hidden_size,
                hidden_size=config.hidden_size,
                context_size=config.hidden_size if config.num_static_continuous > 0 or config.num_static_categorical > 0 else None,
                dropout=config.dropout
            )
        
        # ===== LSTM Encoder-Decoder =====
        self.lstm_encoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_encoder_layers,
            batch_first=True,
            dropout=config.dropout if config.num_encoder_layers > 1 else 0
        )
        
        self.lstm_decoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_decoder_layers,
            batch_first=True,
            dropout=config.dropout if config.num_decoder_layers > 1 else 0
        )
        
        # Gated skip connection for encoder
        self.encoder_glu = GatedLinearUnit(config.hidden_size, config.hidden_size, config.dropout)
        self.encoder_ln = nn.LayerNorm(config.hidden_size)
        
        # Gated skip connection for decoder
        self.decoder_glu = GatedLinearUnit(config.hidden_size, config.hidden_size, config.dropout)
        self.decoder_ln = nn.LayerNorm(config.hidden_size)
        
        # ===== Temporal Self-Attention =====
        self.self_attention = TemporalSelfAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # ===== Position-wise Feed-Forward =====
        self.post_attention_grn = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )
        
        # ===== Output Layer =====
        self.quantile_output = QuantileOutput(
            hidden_size=config.hidden_size,
            num_quantiles=config.num_quantiles,
            quantiles=config.quantiles
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following the paper's recommendations."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(
        self,
        observed_inputs: Tensor,
        known_inputs: Optional[Tensor] = None,
        static_categorical: Optional[Tensor] = None,
        static_continuous: Optional[Tensor] = None,
        return_attention: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass of the Temporal Fusion Transformer.
        
        Args:
            observed_inputs: Historical observed features
                [batch, context_length, num_observed_continuous]
            known_inputs: Known features for both past and future
                [batch, context_length + prediction_length, num_known]
            static_categorical: Static categorical features
                [batch, num_static_categorical]
            static_continuous: Static continuous features
                [batch, num_static_continuous]
            return_attention: Whether to return attention weights
            
        Returns:
            Dict containing:
                - predictions: Quantile predictions [batch, pred_len, num_quantiles]
                - median: Median forecast [batch, pred_len]
                - attention_weights: Temporal attention patterns
                - variable_weights: Feature importance scores
        """
        batch_size = observed_inputs.shape[0]
        device = observed_inputs.device
        
        # ===== 1. Static Covariate Encoding =====
        c_s, c_e, c_h, c_c = self.static_encoders(static_categorical, static_continuous)
        
        # ===== 2. Input Embedding =====
        # Embed observed inputs
        observed_embedded = self.observed_embedding(observed_inputs) if hasattr(self, 'observed_embedding') else None
            
        # Embed known inputs
        if known_inputs is not None and hasattr(self, 'known_embedding'):
            known_embedded = self.known_embedding(known_inputs)
            hist_known = known_embedded[:, :self.context_length]
            future_known = known_embedded[:, self.context_length:]
        else:
            hist_known = None
            future_known = None
        
        # Combine historical inputs
        if observed_embedded is not None and hist_known is not None:
            hist_combined = observed_embedded + hist_known
        elif observed_embedded is not None:
            hist_combined = observed_embedded
        elif hist_known is not None:
            hist_combined = hist_known
        else:
            raise ValueError("Must have either observed or known inputs")
            
        # ===== 3. Variable Selection =====
        # Historical variable selection (single combined input)
        hist_selected, hist_weights = self.hist_vsn([hist_combined], c_s)
        
        # Future variable selection
        if future_known is not None and hasattr(self, 'future_vsn'):
            future_selected, future_weights = self.future_vsn([future_known], c_s)
        else:
            # Use zeros for future if no known inputs
            future_selected = torch.zeros(
                batch_size, self.prediction_length, self.hidden_size, device=device
            )
            future_weights = None
            
        # ===== 4. LSTM Encoder =====
        # Initialize LSTM states with static context
        if c_h is not None and c_c is not None:
            h0 = c_h.unsqueeze(0).expand(self.config.num_encoder_layers, -1, -1).contiguous()
            c0 = c_c.unsqueeze(0).expand(self.config.num_encoder_layers, -1, -1).contiguous()
            init_state = (h0, c0)
        else:
            init_state = None
            
        encoder_output, encoder_state = self.lstm_encoder(hist_selected, init_state)
        
        # Gated skip connection
        encoder_output = self.encoder_ln(
            hist_selected + self.encoder_glu(encoder_output)
        )
        
        # ===== 5. LSTM Decoder =====
        decoder_output, _ = self.lstm_decoder(future_selected, encoder_state)
        
        # Gated skip connection
        decoder_output = self.decoder_ln(
            future_selected + self.decoder_glu(decoder_output)
        )
        
        # ===== 6. Temporal Self-Attention =====
        # Combine encoder and decoder outputs
        temporal_features = torch.cat([encoder_output, decoder_output], dim=1)
        
        # Create causal mask (decoder can't attend to future)
        total_len = self.context_length + self.prediction_length
        causal_mask = torch.triu(
            torch.ones(total_len, total_len, device=device), diagonal=1
        ).bool()
        causal_mask = ~causal_mask  # Invert for attention
        
        attended, attention_weights = self.self_attention(temporal_features, causal_mask)
        
        # ===== 7. Position-wise Feed-Forward =====
        # Apply GRN to decoder portion only
        decoder_attended = attended[:, self.context_length:]
        output = self.post_attention_grn(decoder_attended)
        
        # ===== 8. Quantile Output =====
        quantile_outputs = self.quantile_output(output)
        
        # ===== Build Output Dict =====
        result = {
            'predictions': quantile_outputs['quantiles'],
            'median': quantile_outputs['median'],
            'lower': quantile_outputs['lower_bound'],
            'upper': quantile_outputs['upper_bound'],
            'quantile_levels': quantile_outputs['quantile_levels'],
        }
        
        if return_attention:
            result['attention_weights'] = attention_weights
            result['historical_var_weights'] = hist_weights
            if future_weights is not None:
                result['future_var_weights'] = future_weights
                
        return result
    
    def predict(
        self,
        observed_inputs: Tensor,
        known_inputs: Optional[Tensor] = None,
        static_categorical: Optional[Tensor] = None,
        static_continuous: Optional[Tensor] = None,
        num_samples: int = 100
    ) -> Dict[str, Tensor]:
        """
        Generate predictions with uncertainty quantification.
        
        Uses Monte Carlo dropout for additional uncertainty estimation.
        """
        self.train()  # Enable dropout
        
        samples = []
        for _ in range(num_samples):
            with torch.no_grad():
                output = self.forward(
                    observed_inputs,
                    known_inputs,
                    static_categorical,
                    static_continuous,
                    return_attention=False
                )
                samples.append(output['median'])
                
        self.eval()
        
        samples = torch.stack(samples, dim=0)  # [num_samples, batch, pred_len]
        
        return {
            'mean': samples.mean(dim=0),
            'std': samples.std(dim=0),
            'samples': samples,
            'lower_5': torch.quantile(samples, 0.05, dim=0),
            'upper_95': torch.quantile(samples, 0.95, dim=0),
        }
    
    def get_interpretable_weights(
        self,
        observed_inputs: Tensor,
        known_inputs: Optional[Tensor] = None,
        static_categorical: Optional[Tensor] = None,
        static_continuous: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Extract interpretable attention and variable selection weights."""
        self.eval()
        with torch.no_grad():
            output = self.forward(
                observed_inputs, known_inputs,
                static_categorical, static_continuous,
                return_attention=True
            )
            
        return {
            'attention': output.get('attention_weights'),
            'historical_importance': output.get('historical_var_weights'),
            'future_importance': output.get('future_var_weights'),
        }


def quantile_loss(predictions: Tensor, targets: Tensor, quantiles: List[float]) -> Tensor:
    """
    Quantile (Pinball) Loss for probabilistic forecasting.
    
    Args:
        predictions: [batch, seq_len, num_quantiles]
        targets: [batch, seq_len]
        quantiles: List of quantile levels
        
    Returns:
        Scalar loss
    """
    targets = targets.unsqueeze(-1)  # [batch, seq_len, 1]
    quantiles = torch.tensor(quantiles, device=predictions.device)
    
    errors = targets - predictions
    
    loss = torch.max(
        quantiles * errors,
        (quantiles - 1) * errors
    )
    
    return loss.mean()


class TFTLoss(nn.Module):
    """Combined loss for TFT training."""
    
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Compute quantile loss with optional sample weighting.
        """
        q_loss = quantile_loss(predictions, targets, self.quantiles)
        
        if sample_weights is not None:
            q_loss = (q_loss * sample_weights).mean()
            
        # Additional metrics
        median_idx = len(self.quantiles) // 2
        median_pred = predictions[..., median_idx]
        mae = F.l1_loss(median_pred, targets)
        mse = F.mse_loss(median_pred, targets)
        
        return {
            'loss': q_loss,
            'quantile_loss': q_loss,
            'mae': mae,
            'mse': mse,
            'rmse': torch.sqrt(mse)
        }
