"""
LSTM/GRU Models for Financial Time Series
==========================================

Deep recurrent neural networks for sequence modeling:
- Stacked LSTM with attention
- Bidirectional LSTM for pattern recognition
- Attention-augmented LSTM for long-range dependencies
- GRU variants for efficiency
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class LSTMCell:
    """
    Single LSTM cell implementation.
    
    LSTM equations:
    i_t = σ(W_ii x_t + b_ii + W_hi h_{t-1} + b_hi)
    f_t = σ(W_if x_t + b_if + W_hf h_{t-1} + b_hf)
    g_t = tanh(W_ig x_t + b_ig + W_hg h_{t-1} + b_hg)
    o_t = σ(W_io x_t + b_io + W_ho h_{t-1} + b_ho)
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    h_t = o_t ⊙ tanh(c_t)
    """
    input_size: int
    hidden_size: int
    
    def __post_init__(self):
        # Initialize weights using Xavier initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize all LSTM weights"""
        limit_i = np.sqrt(6.0 / (self.input_size + self.hidden_size * 4))
        limit_h = np.sqrt(6.0 / (self.hidden_size + self.hidden_size * 4))
        
        # Input weights (i, f, g, o gates)
        self.W_i = np.random.uniform(-limit_i, limit_i, (self.input_size, self.hidden_size * 4))
        self.b_i = np.zeros(self.hidden_size * 4)
        
        # Hidden weights
        self.W_h = np.random.uniform(-limit_h, limit_h, (self.hidden_size, self.hidden_size * 4))
        self.b_h = np.zeros(self.hidden_size * 4)
        
        # Forget gate bias initialization (start with high forget bias)
        self.b_i[self.hidden_size:2*self.hidden_size] = 1.0
        self.b_h[self.hidden_size:2*self.hidden_size] = 1.0
    
    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single step forward pass.
        
        Args:
            x: Input at current timestep (batch, input_size)
            h_prev: Previous hidden state (batch, hidden_size)
            c_prev: Previous cell state (batch, hidden_size)
            
        Returns:
            h: New hidden state
            c: New cell state
        """
        # Compute all gates at once
        gates = np.matmul(x, self.W_i) + self.b_i + np.matmul(h_prev, self.W_h) + self.b_h
        
        # Split into individual gates
        i = self._sigmoid(gates[:, :self.hidden_size])
        f = self._sigmoid(gates[:, self.hidden_size:2*self.hidden_size])
        g = np.tanh(gates[:, 2*self.hidden_size:3*self.hidden_size])
        o = self._sigmoid(gates[:, 3*self.hidden_size:])
        
        # Update cell state
        c = f * c_prev + i * g
        
        # Compute hidden state
        h = o * np.tanh(c)
        
        return h, c
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid"""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)), 
                       np.exp(x) / (1 + np.exp(x)))


@dataclass
class GRUCell:
    """
    Gated Recurrent Unit cell.
    
    Simpler than LSTM with comparable performance on many tasks.
    
    GRU equations:
    r_t = σ(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)
    z_t = σ(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)
    n_t = tanh(W_in x_t + b_in + r_t ⊙ (W_hn h_{t-1} + b_hn))
    h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
    """
    input_size: int
    hidden_size: int
    
    def __post_init__(self):
        limit_i = np.sqrt(6.0 / (self.input_size + self.hidden_size * 3))
        limit_h = np.sqrt(6.0 / (self.hidden_size + self.hidden_size * 3))
        
        self.W_i = np.random.uniform(-limit_i, limit_i, (self.input_size, self.hidden_size * 3))
        self.b_i = np.zeros(self.hidden_size * 3)
        self.W_h = np.random.uniform(-limit_h, limit_h, (self.hidden_size, self.hidden_size * 3))
        self.b_h = np.zeros(self.hidden_size * 3)
    
    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray
    ) -> np.ndarray:
        """Single step forward pass"""
        # Compute r and z gates
        gates_i = np.matmul(x, self.W_i[:, :2*self.hidden_size]) + self.b_i[:2*self.hidden_size]
        gates_h = np.matmul(h_prev, self.W_h[:, :2*self.hidden_size]) + self.b_h[:2*self.hidden_size]
        gates = gates_i + gates_h
        
        r = self._sigmoid(gates[:, :self.hidden_size])
        z = self._sigmoid(gates[:, self.hidden_size:])
        
        # Compute n
        n_i = np.matmul(x, self.W_i[:, 2*self.hidden_size:]) + self.b_i[2*self.hidden_size:]
        n_h = np.matmul(r * h_prev, self.W_h[:, 2*self.hidden_size:]) + self.b_h[2*self.hidden_size:]
        n = np.tanh(n_i + n_h)
        
        # Update hidden state
        h = (1 - z) * n + z * h_prev
        
        return h
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class DeepLSTM:
    """
    Deep stacked LSTM network for financial time series.
    
    Features:
    - Multiple LSTM layers with residual connections
    - Dropout between layers
    - Layer normalization for training stability
    - Optional peephole connections
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_residual: bool = True
    ):
        """
        Initialize Deep LSTM.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden state dimension
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability between layers
            use_residual: Whether to use residual connections
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        
        # Create LSTM layers
        self.layers = []
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(LSTMCell(layer_input_size, hidden_size))
        
        # Projection for residual connections if input size differs
        if use_residual and input_size != hidden_size:
            limit = np.sqrt(6.0 / (input_size + hidden_size))
            self.input_projection = np.random.uniform(-limit, limit, (input_size, hidden_size))
        else:
            self.input_projection = None
        
        # Output projection
        self.output_projection = self._init_weights((hidden_size, 1))
    
    def _init_weights(self, shape: Tuple[int, int]) -> np.ndarray:
        limit = np.sqrt(6.0 / sum(shape))
        return np.random.uniform(-limit, limit, shape)
    
    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + eps)
    
    def _apply_dropout(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply dropout during training"""
        if not training or self.dropout == 0:
            return x
        mask = np.random.binomial(1, 1 - self.dropout, x.shape)
        return x * mask / (1 - self.dropout)
    
    def forward(
        self,
        x: np.ndarray,
        initial_states: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        training: bool = False
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Forward pass through all LSTM layers.
        
        Args:
            x: Input sequence (batch, seq_len, input_size)
            initial_states: List of (h, c) tuples for each layer
            training: Whether in training mode (affects dropout)
            
        Returns:
            outputs: Output sequence (batch, seq_len, hidden_size)
            final_states: Final (h, c) for each layer
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize states if not provided
        if initial_states is None:
            initial_states = [
                (np.zeros((batch_size, self.hidden_size)),
                 np.zeros((batch_size, self.hidden_size)))
                for _ in range(self.num_layers)
            ]
        
        # Project input for residual connections
        if self.input_projection is not None:
            x_proj = np.matmul(x, self.input_projection)
        else:
            x_proj = x
        
        current_input = x
        all_outputs = []
        final_states = []
        
        for layer_idx, layer in enumerate(self.layers):
            h, c = initial_states[layer_idx]
            layer_outputs = []
            
            # Process each timestep
            for t in range(seq_len):
                h, c = layer.forward(current_input[:, t, :], h, c)
                layer_outputs.append(h)
            
            # Stack outputs
            layer_output = np.stack(layer_outputs, axis=1)  # (batch, seq_len, hidden)
            
            # Apply residual connection (except for first layer if sizes differ)
            if self.use_residual and layer_idx > 0:
                layer_output = layer_output + current_input
            elif self.use_residual and layer_idx == 0 and self.input_projection is not None:
                layer_output = layer_output + x_proj
            
            # Layer normalization
            layer_output = self._layer_norm(layer_output)
            
            # Dropout between layers
            if layer_idx < self.num_layers - 1:
                layer_output = self._apply_dropout(layer_output, training)
            
            current_input = layer_output
            final_states.append((h, c))
        
        return current_input, final_states
    
    def predict(
        self,
        x: np.ndarray,
        horizons: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions from input sequence.
        
        Args:
            x: Input sequence (batch, seq_len, input_size)
            horizons: Forecast horizons
            
        Returns:
            Dictionary of predictions for each horizon
        """
        # Forward pass
        outputs, _ = self.forward(x, training=False)
        
        # Use last output for prediction
        last_output = outputs[:, -1, :]  # (batch, hidden_size)
        
        # Generate prediction (returns prediction)
        prediction = np.matmul(last_output, self.output_projection).squeeze(-1)
        
        # For multi-horizon, scale the base prediction
        # (In practice, separate heads would be used)
        predictions = {}
        for h in horizons:
            # Simple scaling heuristic for different horizons
            scale = np.sqrt(h)
            predictions[f'{h}d'] = prediction * scale
        
        return predictions


class BidirectionalLSTM:
    """
    Bidirectional LSTM for pattern recognition.
    
    Processes sequence in both directions for better pattern detection.
    Useful for detecting chart patterns, support/resistance levels.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        merge_mode: str = 'concat'  # 'concat', 'sum', 'mul', 'avg'
    ):
        """
        Initialize Bidirectional LSTM.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state size per direction
            num_layers: Number of BiLSTM layers
            merge_mode: How to combine forward/backward outputs
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.merge_mode = merge_mode
        
        # Forward and backward LSTM layers
        self.forward_layers = []
        self.backward_layers = []
        
        for i in range(num_layers):
            if i == 0:
                layer_input = input_size
            elif merge_mode == 'concat':
                layer_input = hidden_size * 2
            else:
                layer_input = hidden_size
            
            self.forward_layers.append(LSTMCell(layer_input, hidden_size))
            self.backward_layers.append(LSTMCell(layer_input, hidden_size))
        
        # Output size depends on merge mode
        self.output_size = hidden_size * 2 if merge_mode == 'concat' else hidden_size
    
    def _process_direction(
        self,
        x: np.ndarray,
        layer: LSTMCell,
        reverse: bool = False
    ) -> np.ndarray:
        """Process sequence in one direction"""
        batch_size, seq_len, _ = x.shape
        
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        
        outputs = []
        indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        
        for t in indices:
            h, c = layer.forward(x[:, t, :], h, c)
            outputs.append(h)
        
        if reverse:
            outputs = outputs[::-1]
        
        return np.stack(outputs, axis=1)
    
    def _merge_outputs(
        self,
        forward: np.ndarray,
        backward: np.ndarray
    ) -> np.ndarray:
        """Merge forward and backward outputs"""
        if self.merge_mode == 'concat':
            return np.concatenate([forward, backward], axis=-1)
        elif self.merge_mode == 'sum':
            return forward + backward
        elif self.merge_mode == 'mul':
            return forward * backward
        elif self.merge_mode == 'avg':
            return (forward + backward) / 2
        else:
            raise ValueError(f"Unknown merge mode: {self.merge_mode}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through BiLSTM.
        
        Args:
            x: Input (batch, seq_len, input_size)
            
        Returns:
            output: Bidirectional output (batch, seq_len, output_size)
        """
        current_input = x
        
        for i in range(self.num_layers):
            forward_out = self._process_direction(
                current_input, self.forward_layers[i], reverse=False
            )
            backward_out = self._process_direction(
                current_input, self.backward_layers[i], reverse=True
            )
            
            current_input = self._merge_outputs(forward_out, backward_out)
        
        return current_input
    
    def detect_patterns(
        self,
        x: np.ndarray,
        pattern_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Detect patterns in the sequence using bidirectional context.
        
        Returns:
            Pattern detection results with confidence scores
        """
        # Get bidirectional representations
        bi_output = self.forward(x)  # (batch, seq_len, output_size)
        
        # Compute pattern scores at each position
        # Using simple heuristics based on representation similarity
        batch_size, seq_len, _ = bi_output.shape
        
        patterns = []
        for b in range(batch_size):
            sample_patterns = []
            
            # Look for symmetric patterns (head & shoulders, double tops)
            for center in range(10, seq_len - 10):
                left = bi_output[b, center-10:center, :]
                right = bi_output[b, center:center+10, :]
                
                # Compute symmetry score
                right_flipped = right[::-1, :]
                similarity = np.mean(np.abs(left - right_flipped))
                symmetry_score = 1.0 / (1.0 + similarity)
                
                if symmetry_score > pattern_threshold:
                    sample_patterns.append({
                        'type': 'symmetric_pattern',
                        'position': center,
                        'confidence': float(symmetry_score)
                    })
            
            patterns.append(sample_patterns)
        
        return {'patterns': patterns}


class StackedLSTM:
    """Alias for DeepLSTM with additional features"""
    
    def __init__(self, *args, **kwargs):
        self._base = DeepLSTM(*args, **kwargs)
    
    def forward(self, x, **kwargs):
        return self._base.forward(x, **kwargs)
    
    def predict(self, x, **kwargs):
        return self._base.predict(x, **kwargs)


class AttentionLSTM:
    """
    LSTM with Bahdanau-style attention for financial time series.
    
    Combines LSTM sequence modeling with attention mechanism to:
    - Focus on important time steps
    - Provide interpretable importance weights
    - Handle variable-length dependencies
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        attention_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize Attention LSTM.
        
        Args:
            input_size: Input feature dimension
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            attention_dim: Attention mechanism dimension
            dropout: Dropout rate
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        
        # Base LSTM
        self.lstm = DeepLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_residual=True
        )
        
        # Attention weights
        self.W_query = self._init_weights((hidden_size, attention_dim))
        self.W_key = self._init_weights((hidden_size, attention_dim))
        self.W_value = self._init_weights((hidden_size, hidden_size))
        self.v_attention = self._init_weights((attention_dim, 1))
        
        # Output projection
        self.output_projection = self._init_weights((hidden_size * 2, 1))
    
    def _init_weights(self, shape: Tuple[int, int]) -> np.ndarray:
        limit = np.sqrt(6.0 / sum(shape))
        return np.random.uniform(-limit, limit, shape)
    
    def _compute_attention(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Bahdanau attention.
        
        Args:
            query: Query vector (batch, hidden_size)
            keys: Key vectors (batch, seq_len, hidden_size)
            values: Value vectors (batch, seq_len, hidden_size)
            
        Returns:
            context: Attended context vector
            attention_weights: Attention distribution over sequence
        """
        batch_size, seq_len, _ = keys.shape
        
        # Project query and keys
        query_proj = np.matmul(query, self.W_query)  # (batch, attention_dim)
        keys_proj = np.matmul(keys, self.W_key)  # (batch, seq_len, attention_dim)
        
        # Expand query for broadcasting
        query_proj = query_proj[:, np.newaxis, :]  # (batch, 1, attention_dim)
        
        # Compute attention scores
        scores = np.tanh(query_proj + keys_proj)  # (batch, seq_len, attention_dim)
        scores = np.matmul(scores, self.v_attention).squeeze(-1)  # (batch, seq_len)
        
        # Softmax to get attention weights
        attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
        
        # Project values
        values_proj = np.matmul(values, self.W_value)
        
        # Compute context vector
        context = np.sum(attention_weights[:, :, np.newaxis] * values_proj, axis=1)
        
        return context, attention_weights
    
    def forward(
        self,
        x: np.ndarray,
        return_attention: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forward pass with attention.
        
        Args:
            x: Input sequence (batch, seq_len, input_size)
            return_attention: Whether to return attention weights
            
        Returns:
            output: Final output (batch, 1) or prediction
            attention_weights: (optional) Attention distribution
        """
        # Get LSTM outputs
        lstm_outputs, final_states = self.lstm.forward(x, training=False)
        
        # Use final hidden state as query
        query = final_states[-1][0]  # Last layer's hidden state
        
        # Compute attention over LSTM outputs
        context, attention_weights = self._compute_attention(
            query, lstm_outputs, lstm_outputs
        )
        
        # Concatenate context with final hidden state
        combined = np.concatenate([query, context], axis=-1)
        
        # Output projection
        output = np.matmul(combined, self.output_projection).squeeze(-1)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def predict_with_interpretation(
        self,
        x: np.ndarray,
        timestamps: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate predictions with attention-based interpretation.
        
        Args:
            x: Input sequence
            timestamps: Optional list of timestamps for each position
            
        Returns:
            Dictionary with predictions and attention analysis
        """
        output, attention = self.forward(x, return_attention=True)
        
        batch_size, seq_len = attention.shape
        
        # Find most attended positions
        top_k = min(5, seq_len)
        top_indices = np.argsort(attention, axis=-1)[:, -top_k:][:, ::-1]
        
        result = {
            'prediction': output,
            'attention_weights': attention,
            'top_attended_positions': top_indices,
            'attention_concentration': float(np.max(attention, axis=-1).mean()),  # How focused
            'attention_entropy': float(-np.sum(attention * np.log(attention + 1e-8), axis=-1).mean())
        }
        
        if timestamps:
            result['top_attended_timestamps'] = [
                [timestamps[i] for i in indices] 
                for indices in top_indices
            ]
        
        return result
