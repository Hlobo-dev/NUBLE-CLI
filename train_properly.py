#!/usr/bin/env python3
"""
KYPERIAN Proper Training Script

This script fixes the overfitting issues by:
1. Using proper regularization (L2, dropout, early stopping)
2. Using directional loss (not just MSE)
3. Early stopping based on validation performance
4. Starting with simple models before complex ones

Key Insight:
    Perfect training loss (0.000000) = guaranteed overfitting
    Good training loss = ~0.0004-0.001 (matching noise level)

Usage:
    python train_properly.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

# Import scipy if available
try:
    from scipy.stats import spearmanr, ttest_1samp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# PROPER LOSS FUNCTIONS
# =============================================================================

class DirectionalLoss(nn.Module):
    """
    Loss that heavily penalizes wrong direction predictions.
    
    In trading, direction matters more than magnitude.
    Getting the direction right with wrong magnitude = profit
    Getting the direction wrong = loss
    """
    
    def __init__(self, direction_weight: float = 3.0, mse_weight: float = 1.0):
        super().__init__()
        self.direction_weight = direction_weight
        self.mse_weight = mse_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # MSE component
        mse = F.mse_loss(predictions, targets, reduction='none')
        
        # Direction component
        pred_sign = torch.sign(predictions)
        target_sign = torch.sign(targets)
        wrong_direction = (pred_sign != target_sign).float()
        
        # Penalize wrong direction more heavily
        weights = 1.0 + wrong_direction * (self.direction_weight - 1.0)
        
        return (weights * mse).mean()


class ICLoss(nn.Module):
    """
    Loss based on Information Coefficient (rank correlation).
    
    Directly optimizes what we care about: ranking correctly.
    """
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Soft ranking using sigmoid
        pred_ranks = torch.sigmoid(predictions * 10)
        target_ranks = torch.sigmoid(targets * 10)
        
        # Correlation loss
        pred_centered = pred_ranks - pred_ranks.mean()
        target_centered = target_ranks - target_ranks.mean()
        
        numerator = (pred_centered * target_centered).sum()
        denominator = (pred_centered.norm() * target_centered.norm() + 1e-8)
        
        correlation = numerator / denominator
        
        return -correlation


class CombinedLoss(nn.Module):
    """Combined loss for financial prediction."""
    
    def __init__(self, mse_weight=0.3, direction_weight=0.5, ic_weight=0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.direction_weight = direction_weight
        self.ic_weight = ic_weight
        self.direction_loss = DirectionalLoss()
        self.ic_loss = ICLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(predictions, targets)
        direction = self.direction_loss(predictions, targets)
        ic = self.ic_loss(predictions, targets)
        
        return self.mse_weight * mse + self.direction_weight * direction + self.ic_weight * ic


# =============================================================================
# SIMPLE BUT EFFECTIVE MODELS
# =============================================================================

class SimpleLinear(nn.Module):
    """Linear model - your baseline. If this doesn't work, nothing will."""
    
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.linear(x).squeeze(-1)


class RegularizedMLP(nn.Module):
    """
    MLP with proper regularization.
    
    Key differences from naive MLP:
    - Dropout between layers
    - Batch normalization
    - Smaller hidden sizes
    - Skip connections (residual learning)
    """
    
    def __init__(self, input_size: int, hidden_size: int = 32, dropout: float = 0.3):
        super().__init__()
        
        self.input_bn = nn.BatchNorm1d(input_size)
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        # Skip connection (linear baseline)
        self.skip = nn.Linear(input_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, -1, :]
        
        # Skip connection
        skip = self.skip(x)
        
        # Main path
        x = self.input_bn(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        # Combine with skip (residual)
        return (x + skip).squeeze(-1)


class RegularizedLSTM(nn.Module):
    """
    LSTM with proper regularization.
    
    Key changes from naive LSTM:
    - Much smaller hidden size
    - Dropout
    - Layer normalization
    - Attention over sequence
    """
    
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Attention over sequence
        self.attention = nn.Linear(hidden_size, 1)
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Project input
        x = self.input_proj(x)
        x = self.ln1(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.ln2(lstm_out)
        
        # Attention pooling
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)
        
        context = self.dropout(context)
        
        return self.fc(context).squeeze(-1)


# =============================================================================
# TRAINING WITH EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        
        return self.should_stop
    
    def restore_best(self, model: nn.Module):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def train_model(
    model: nn.Module,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    patience: int = 20,
    batch_size: int = 64,
    loss_type: str = 'combined',
    device: str = 'auto'
) -> Dict:
    """
    Train model with proper regularization and early stopping.
    """
    # Device selection
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else \
                 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    model = model.to(device)
    train_X = train_X.to(device)
    train_y = train_y.to(device)
    val_X = val_X.to(device)
    val_y = val_y.to(device)
    
    # Loss function
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'directional':
        criterion = DirectionalLoss()
    elif loss_type == 'ic':
        criterion = ICLoss()
    else:
        criterion = CombinedLoss()
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_dir_acc': []}
    
    n_batches = (len(train_X) + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        # Shuffle training data
        perm = torch.randperm(len(train_X), device=device)
        train_X_shuffled = train_X[perm]
        train_y_shuffled = train_y[perm]
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(train_X))
            
            batch_X = train_X_shuffled[start_idx:end_idx]
            batch_y = train_y_shuffled[start_idx:end_idx]
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X)
            val_loss = criterion(val_pred, val_y).item()
            
            # Directional accuracy on validation
            val_dir_acc = (torch.sign(val_pred) == torch.sign(val_y)).float().mean().item()
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_dir_acc'].append(val_dir_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"   Early stopping at epoch {epoch+1}")
            break
        
        # Logging
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}: Train={avg_train_loss:.6f}, "
                  f"Val={val_loss:.6f}, Dir Acc={val_dir_acc:.2%}")
    
    # Restore best model
    early_stopping.restore_best(model)
    model = model.to(device)
    
    return history


# =============================================================================
# VALIDATION
# =============================================================================

@dataclass
class ValidationResult:
    model_name: str
    directional_accuracy: float
    correlation: float
    ic_mean: float
    ic_ir: float
    sharpe_ratio: float
    max_drawdown: float
    t_statistic: float
    p_value: float
    is_significant: bool
    total_return: float
    
    def __str__(self):
        status = "✅ SIGNIFICANT" if self.is_significant else "❌ NOT SIGNIFICANT"
        grade = self._get_grade()
        
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║  {self.model_name:^62} ║
║  Grade: {grade}                                                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Directional Accuracy: {self.directional_accuracy:>6.2%}  (random = 50%)            ║
║  Correlation:          {self.correlation:>6.3f}                              ║
║  IC Mean:              {self.ic_mean:>6.4f}                             ║
║  IC IR:                {self.ic_ir:>6.2f}                               ║
╠══════════════════════════════════════════════════════════════════╣
║  Sharpe Ratio:         {self.sharpe_ratio:>6.2f}                               ║
║  Max Drawdown:         {self.max_drawdown:>6.2%}                             ║
║  Total Return:         {self.total_return:>6.2%}                             ║
╠══════════════════════════════════════════════════════════════════╣
║  t-statistic:          {self.t_statistic:>6.2f}                               ║
║  p-value:              {self.p_value:>6.4f}                             ║
║  Status:               {status:>24}       ║
╚══════════════════════════════════════════════════════════════════╝
"""
    
    def _get_grade(self) -> str:
        if self.sharpe_ratio > 2.0 and self.is_significant:
            return "A"
        elif self.sharpe_ratio > 1.0 and self.is_significant:
            return "B"
        elif self.sharpe_ratio > 0.5 and self.directional_accuracy > 0.52:
            return "C"
        elif self.sharpe_ratio > 0 and self.directional_accuracy > 0.50:
            return "D"
        else:
            return "F"


def validate_model(
    model: nn.Module,
    test_X: torch.Tensor,
    test_y: torch.Tensor,
    model_name: str,
    transaction_cost_bps: float = 10.0,
    device: str = 'cpu'
) -> ValidationResult:
    """Validate model on test set."""
    
    model = model.to(device)
    test_X = test_X.to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(test_X).cpu().numpy()
    
    actuals = test_y.cpu().numpy() if isinstance(test_y, torch.Tensor) else test_y
    
    # Ensure same length
    min_len = min(len(predictions), len(actuals))
    predictions = predictions[:min_len]
    actuals = actuals[:min_len]
    
    # Metrics
    dir_acc = np.mean(np.sign(predictions) == np.sign(actuals))
    
    if np.std(predictions) > 0 and np.std(actuals) > 0:
        corr = np.corrcoef(predictions, actuals)[0, 1]
    else:
        corr = 0
    
    # IC calculation
    ic_vals = []
    window = 20
    if HAS_SCIPY:
        for i in range(0, len(predictions) - window, window // 4):
            ic, _ = spearmanr(predictions[i:i+window], actuals[i:i+window])
            if not np.isnan(ic):
                ic_vals.append(ic)
    
    ic_mean = np.mean(ic_vals) if ic_vals else 0
    ic_std = np.std(ic_vals) if len(ic_vals) > 1 else 1
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    
    # Trading simulation
    cost = transaction_cost_bps / 10000
    positions = np.sign(predictions)
    costs = np.abs(np.diff(positions, prepend=0)) * cost
    returns = positions * actuals - costs
    
    # Sharpe
    if np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Max drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / (running_max + 1e-8)
    max_dd = drawdown.min()
    
    # Total return
    total_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0
    
    # Statistical significance
    if HAS_SCIPY and len(returns) > 1:
        t_stat, p_value = ttest_1samp(returns, 0)
    else:
        t_stat, p_value = 0, 1
    
    return ValidationResult(
        model_name=model_name,
        directional_accuracy=dir_acc,
        correlation=corr if not np.isnan(corr) else 0,
        ic_mean=ic_mean,
        ic_ir=ic_ir,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        t_statistic=t_stat if not np.isnan(t_stat) else 0,
        p_value=p_value if not np.isnan(p_value) else 1,
        is_significant=p_value < 0.05 and sharpe > 0,
        total_return=total_return
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("  PROPER MODEL TRAINING WITH REGULARIZATION")
    print("="*70)
    
    # Device
    device = 'mps' if torch.backends.mps.is_available() else \
             'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Using device: {device}")
    
    # Generate data with KNOWN signal
    print("\n📊 Generating synthetic data with known signal...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    dates = pd.date_range('2019-01-01', '2024-12-31', freq='B')
    n = len(dates)
    n_features = 20  # Fewer features to reduce overfitting
    
    # Generate features with fixed seed for reproducibility
    np.random.seed(42)
    features = np.random.randn(n, n_features)
    
    # CRITICAL: The TRUE weights are known - test if model can recover them
    true_weights = np.zeros(n_features)
    true_weights[0] = 0.4  # Feature 0 has strong signal
    true_weights[1] = 0.25  # Feature 1 has medium signal  
    true_weights[2] = 0.15  # Feature 2 has weak signal
    
    # The TRUE signal is a linear combination
    signal = features @ true_weights
    signal = signal / signal.std()  # Normalize to unit variance
    
    # Generate noise with DIFFERENT seed to ensure independence
    np.random.seed(123)
    noise = np.random.randn(n)
    
    # Signal to Noise Ratio: 30% signal, 70% noise
    signal_strength = 0.3
    
    # Forward-looking: signal[t] predicts returns[t+1]
    returns = np.zeros(n)
    returns[1:] = signal_strength * signal[:-1] * 0.02 + (1 - signal_strength) * noise[1:] * 0.02
    
    # Reset seed for PyTorch
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(f"   Signal strength: {signal_strength:.0%}")
    print(f"   Data points: {n}")
    print(f"   Features: {n_features}")
    print(f"   Target std: {returns.std():.4f}")
    
    # Verify signal across ALL data
    full_corr = np.corrcoef(features[:-1,0], returns[1:])[0,1]
    print(f"   Correlation(feature_0[t], returns[t+1]) FULL: {full_corr:.3f}")
    
    # Split data
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    # Verify signal is consistent across splits
    train_corr = np.corrcoef(features[:train_end-1, 0], returns[1:train_end])[0,1]
    val_corr = np.corrcoef(features[train_end:val_end-1, 0], returns[train_end+1:val_end])[0,1]
    test_corr = np.corrcoef(features[val_end:-1, 0], returns[val_end+1:])[0,1]
    print(f"   Correlation in Train: {train_corr:.3f}")
    print(f"   Correlation in Val:   {val_corr:.3f}")
    print(f"   Correlation in Test:  {test_corr:.3f}")
    
    # CRITICAL: Align data so features[t] predicts returns[t+1]
    # X = features[0:n-1], Y = returns[1:n]
    # Then split this aligned data
    X_aligned = features[:-1]  # features 0 to n-2
    y_aligned = returns[1:]    # returns 1 to n-1
    
    train_end_aligned = train_end - 1
    val_end_aligned = val_end - 1
    
    train_X = torch.tensor(X_aligned[:train_end_aligned], dtype=torch.float32)
    train_y = torch.tensor(y_aligned[:train_end_aligned], dtype=torch.float32)
    
    val_X = torch.tensor(X_aligned[train_end_aligned:val_end_aligned], dtype=torch.float32)
    val_y = torch.tensor(y_aligned[train_end_aligned:val_end_aligned], dtype=torch.float32)
    
    test_X = torch.tensor(X_aligned[val_end_aligned:], dtype=torch.float32)
    test_y = torch.tensor(y_aligned[val_end_aligned:], dtype=torch.float32)
    
    print(f"\n   Train: {len(train_X)} samples")
    print(f"   Val:   {len(val_X)} samples")
    print(f"   Test:  {len(test_X)} samples")
    
    # Verify alignment
    train_corr_check = np.corrcoef(train_X[:, 0].numpy(), train_y.numpy())[0,1]
    test_corr_check = np.corrcoef(test_X[:, 0].numpy(), test_y.numpy())[0,1]
    print(f"   Aligned correlation (train): {train_corr_check:.3f}")
    print(f"   Aligned correlation (test):  {test_corr_check:.3f}")
    
    results = []
    
    # ==========================================================================
    # Model 0: SKLEARN LINEAR REGRESSION (GOLD STANDARD)
    # ==========================================================================
    print("\n" + "="*70)
    print("  Model 0: SKLEARN OLS (Gold Standard)")
    print("="*70)
    
    from sklearn.linear_model import LinearRegression
    
    # Train on features[t] -> returns[t+1]
    X_train_np = features[:train_end-1]
    y_train_np = returns[1:train_end]
    X_test_np = features[val_end:-1]
    y_test_np = returns[val_end+1:]
    
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train_np, y_train_np)
    
    predictions = sklearn_model.predict(X_test_np)
    dir_acc = np.mean(np.sign(predictions) == np.sign(y_test_np))
    positions = np.sign(predictions)
    costs = np.abs(np.diff(positions, prepend=0)) * 0.001
    strat_returns = positions * y_test_np - costs
    sharpe = np.mean(strat_returns) / np.std(strat_returns) * np.sqrt(252)
    
    if HAS_SCIPY:
        t_stat, p_val = ttest_1samp(strat_returns, 0)
    else:
        t_stat, p_val = 0, 1
    
    print(f"   Dir Acc: {dir_acc:.2%}, Sharpe: {sharpe:.2f}, p-value: {p_val:.4f}")
    print(f"   ✅ This is the benchmark - neural nets should match or beat this!")
    
    # ==========================================================================
    # Model 1: Simple Linear (MSE LOSS - simpler is better)
    # ==========================================================================
    print("\n" + "="*70)
    print("  Model 1: SIMPLE LINEAR (MSE Loss)")
    print("="*70)
    
    model = SimpleLinear(n_features)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Use SIMPLE MSE, lower learning rate, less regularization
    history = train_model(
        model, train_X, train_y, val_X, val_y,
        epochs=500, lr=0.001, weight_decay=0.01,
        patience=50, loss_type='mse', device=device
    )
    
    result = validate_model(model, test_X, test_y, "Linear (MSE)", device=device)
    print(result)
    results.append(result)
    
    # ==========================================================================
    # Model 2: Regularized MLP
    # ==========================================================================
    print("\n" + "="*70)
    print("  Model 2: REGULARIZED MLP")
    print("="*70)
    
    model = RegularizedMLP(n_features, hidden_size=32, dropout=0.3)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    history = train_model(
        model, train_X, train_y, val_X, val_y,
        epochs=200, lr=0.001, weight_decay=0.01,
        patience=30, loss_type='combined', device=device
    )
    
    result = validate_model(model, test_X, test_y, "Regularized MLP", device=device)
    print(result)
    results.append(result)
    
    # ==========================================================================
    # Model 3: Regularized LSTM
    # ==========================================================================
    print("\n" + "="*70)
    print("  Model 3: REGULARIZED LSTM")
    print("="*70)
    
    # Reshape for LSTM (add sequence dimension)
    seq_len = 10
    
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len + 1):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len-1])
        return torch.stack(X_seq), torch.stack(y_seq)
    
    train_X_seq, train_y_seq = create_sequences(train_X, train_y, seq_len)
    val_X_seq, val_y_seq = create_sequences(val_X, val_y, seq_len)
    test_X_seq, test_y_seq = create_sequences(test_X, test_y, seq_len)
    
    model = RegularizedLSTM(n_features, hidden_size=32, dropout=0.3)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    history = train_model(
        model, train_X_seq, train_y_seq, val_X_seq, val_y_seq,
        epochs=200, lr=0.001, weight_decay=0.01,
        patience=30, loss_type='combined', device=device
    )
    
    result = validate_model(model, test_X_seq, test_y_seq, "Regularized LSTM", device=device)
    print(result)
    results.append(result)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"  {'Model':<20} {'Dir Acc':>10} {'Sharpe':>10} {'IC IR':>10} {'Status':>15}")
    print("  " + "-"*65)
    
    for r in results:
        status = "✅ SIG" if r.is_significant else "❌ NOSIG"
        print(f"  {r.model_name:<20} {r.directional_accuracy:>10.2%} {r.sharpe_ratio:>10.2f} {r.ic_ir:>10.2f} {status:>15}")
    
    # Interpretation
    print("\n" + "="*70)
    print("  INTERPRETATION")
    print("="*70)
    
    best = max(results, key=lambda x: x.sharpe_ratio)
    
    if best.sharpe_ratio > 1.0 and best.is_significant:
        print(f"""
  ✅ SUCCESS! {best.model_name} shows significant alpha.
  
  Next steps:
  1. Test on REAL market data (not synthetic)
  2. If it still works, add complexity gradually
  3. Implement walk-forward retraining
""")
    elif best.directional_accuracy > 0.52:
        print(f"""
  🟡 PARTIAL SUCCESS: {best.model_name} shows some signal.
  
  The model is better than random but not statistically significant.
  This could mean:
  1. Need more data
  2. Signal is there but weak
  3. Need different features
""")
    else:
        print(f"""
  ❌ MODELS FAILED TO LEARN THE SIGNAL
  
  Even with known signal in synthetic data, models couldn't learn it.
  This indicates fundamental issues with:
  1. Model architecture
  2. Training process  
  3. Feature representation
  
  Recommendation: Start with even simpler models and verify training works.
""")
    
    print("\n✓ Training complete!")
    
    return results


if __name__ == "__main__":
    main()
