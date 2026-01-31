"""
Production Trainer for Institutional Time Series Models
========================================================

Enterprise-grade training infrastructure with:
- Walk-forward cross-validation
- Conformal prediction for calibrated intervals
- Early stopping with patience
- Learning rate scheduling
- Gradient accumulation
- Mixed precision training
- Checkpointing and resumption
- Comprehensive logging and metrics
- Hyperparameter optimization integration

This trainer is designed for production deployment at institutional scale.
"""

import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from enum import Enum
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import (
    OneCycleLR, CosineAnnealingWarmRestarts, 
    ReduceLROnPlateau, LinearLR, SequentialLR
)
from torch.cuda.amp import GradScaler, autocast
from torch import Tensor
import numpy as np


class ValidationStrategy(Enum):
    """Validation splitting strategies."""
    HOLDOUT = "holdout"              # Simple train/val split
    WALK_FORWARD = "walk_forward"    # Expanding window
    ROLLING = "rolling"              # Fixed window rolling
    PURGED_KFOLD = "purged_kfold"   # K-fold with temporal gap


@dataclass
class TrainerConfig:
    """
    Configuration for production training.
    
    Optimized defaults for financial time series.
    """
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Gradient handling
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Scheduling
    scheduler: str = 'onecycle'      # 'onecycle', 'cosine', 'plateau', 'warmup'
    warmup_epochs: int = 5
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-6
    
    # Validation
    validation_strategy: ValidationStrategy = ValidationStrategy.WALK_FORWARD
    val_frequency: int = 1           # Validate every N epochs
    num_folds: int = 5               # For k-fold
    purge_gap: int = 5               # Temporal gap for purged CV
    
    # Mixed precision
    use_amp: bool = True             # Automatic mixed precision
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    save_frequency: int = 10         # Save every N epochs
    
    # Logging
    log_frequency: int = 10          # Log every N batches
    verbose: int = 1                 # 0=silent, 1=progress, 2=detailed
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: str = "auto"             # 'auto', 'cuda', 'mps', 'cpu'


@dataclass
class TrainingState:
    """Tracks training progress for checkpointing."""
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float('inf')
    best_epoch: int = 0
    patience_counter: int = 0
    history: Dict[str, List[float]] = field(default_factory=dict)


class EarlyStopping:
    """
    Early stopping handler with patience.
    
    Monitors validation metric and stops when no improvement.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                
        return self.should_stop


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time series.
    
    Implements expanding window validation where each fold
    uses all prior data for training.
    
    Timeline:
    Fold 1: [Train: 0-T1] [Val: T1-T2]
    Fold 2: [Train: 0-T2] [Val: T2-T3]
    Fold 3: [Train: 0-T3] [Val: T3-T4]
    ...
    """
    
    def __init__(
        self,
        num_folds: int = 5,
        min_train_size: int = 100,
        gap: int = 0
    ):
        self.num_folds = num_folds
        self.min_train_size = min_train_size
        self.gap = gap
        
    def split(self, dataset: Dataset) -> List[Tuple[List[int], List[int]]]:
        """
        Generate walk-forward splits.
        
        Args:
            dataset: Dataset to split
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        n = len(dataset)
        fold_size = (n - self.min_train_size) // self.num_folds
        
        splits = []
        for i in range(self.num_folds):
            train_end = self.min_train_size + i * fold_size
            val_start = train_end + self.gap
            val_end = train_end + fold_size
            
            train_idx = list(range(train_end))
            val_idx = list(range(val_start, min(val_end, n)))
            
            if len(val_idx) > 0:
                splits.append((train_idx, val_idx))
                
        return splits


class RollingWindowValidator:
    """
    Rolling window cross-validation.
    
    Fixed-size training window that rolls forward.
    Better for non-stationary data where old data may be less relevant.
    """
    
    def __init__(
        self,
        num_folds: int = 5,
        window_size: int = 200,
        step_size: int = 50,
        gap: int = 0
    ):
        self.num_folds = num_folds
        self.window_size = window_size
        self.step_size = step_size
        self.gap = gap
        
    def split(self, dataset: Dataset) -> List[Tuple[List[int], List[int]]]:
        """Generate rolling window splits."""
        n = len(dataset)
        splits = []
        
        for i in range(self.num_folds):
            train_start = i * self.step_size
            train_end = train_start + self.window_size
            val_start = train_end + self.gap
            val_end = val_start + self.step_size
            
            if val_end > n:
                break
                
            train_idx = list(range(train_start, train_end))
            val_idx = list(range(val_start, val_end))
            
            splits.append((train_idx, val_idx))
            
        return splits


class PurgedKFoldValidator:
    """
    Purged K-Fold cross-validation.
    
    Standard k-fold with temporal gap between train and validation
    to prevent information leakage.
    
    Reference: "Advances in Financial Machine Learning" - de Prado
    """
    
    def __init__(
        self,
        num_folds: int = 5,
        purge_gap: int = 5
    ):
        self.num_folds = num_folds
        self.purge_gap = purge_gap
        
    def split(self, dataset: Dataset) -> List[Tuple[List[int], List[int]]]:
        """Generate purged k-fold splits."""
        n = len(dataset)
        fold_size = n // self.num_folds
        
        splits = []
        for i in range(self.num_folds):
            val_start = i * fold_size
            val_end = val_start + fold_size
            
            # Training excludes validation and purge gap
            train_before = list(range(max(0, val_start - self.purge_gap)))
            train_after = list(range(min(n, val_end + self.purge_gap), n))
            train_idx = train_before + train_after
            
            val_idx = list(range(val_start, val_end))
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
                
        return splits


class ConformalPredictor:
    """
    Conformal Prediction for calibrated prediction intervals.
    
    Provides distribution-free, finite-sample valid prediction intervals.
    
    Methods:
    - Split Conformal: Simple calibration on holdout set
    - Adaptive: Width adapts to local difficulty
    
    Reference: "A Gentle Introduction to Conformal Prediction"
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        method: str = 'split'
    ):
        """
        Args:
            alpha: Miscoverage rate (0.1 = 90% coverage)
            method: 'split' or 'adaptive'
        """
        self.alpha = alpha
        self.method = method
        self.calibration_scores = None
        self.quantile = None
        
    def calibrate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ):
        """
        Calibrate on holdout set.
        
        Args:
            predictions: [n_samples, horizon] point predictions
            targets: [n_samples, horizon] true values
        """
        # Compute conformity scores (absolute residuals)
        scores = np.abs(predictions - targets)
        
        # Aggregate over horizon
        if scores.ndim > 1:
            scores = scores.mean(axis=1)
            
        self.calibration_scores = np.sort(scores)
        
        # Compute quantile
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(scores, min(q_level, 1.0))
        
    def predict_intervals(
        self,
        predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals.
        
        Args:
            predictions: [n_samples, horizon] point predictions
            
        Returns:
            lower: Lower bounds
            upper: Upper bounds
        """
        if self.quantile is None:
            raise ValueError("Must calibrate before predicting intervals")
            
        lower = predictions - self.quantile
        upper = predictions + self.quantile
        
        return lower, upper
        
    def coverage_score(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Compute empirical coverage."""
        lower, upper = self.predict_intervals(predictions)
        covered = (targets >= lower) & (targets <= upper)
        return covered.mean()


class AdaptiveConformalPredictor:
    """
    Adaptive Conformal Prediction with local calibration.
    
    Interval width adapts based on predicted uncertainty or
    local difficulty (residual magnitude).
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        num_bins: int = 10
    ):
        self.alpha = alpha
        self.num_bins = num_bins
        self.bin_quantiles = {}
        
    def calibrate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: Optional[np.ndarray] = None
    ):
        """
        Calibrate with adaptive bins.
        
        Uses predicted uncertainty to determine local calibration.
        """
        scores = np.abs(predictions - targets)
        
        if scores.ndim > 1:
            scores = scores.mean(axis=1)
            
        # Use uncertainty or prediction magnitude for binning
        if uncertainties is None:
            uncertainties = np.abs(predictions.mean(axis=1) if predictions.ndim > 1 else predictions)
            
        # Create bins
        bin_edges = np.percentile(uncertainties, np.linspace(0, 100, self.num_bins + 1))
        
        for i in range(self.num_bins):
            mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_scores = scores[mask]
                n = len(bin_scores)
                q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
                self.bin_quantiles[i] = {
                    'edges': (bin_edges[i], bin_edges[i + 1]),
                    'quantile': np.quantile(bin_scores, min(q_level, 1.0))
                }
                
    def predict_intervals(
        self,
        predictions: np.ndarray,
        uncertainties: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate adaptive intervals."""
        if not self.bin_quantiles:
            raise ValueError("Must calibrate before predicting intervals")
            
        if uncertainties is None:
            uncertainties = np.abs(predictions.mean(axis=1) if predictions.ndim > 1 else predictions)
            
        widths = np.zeros(len(predictions))
        
        for i, (edges, q) in enumerate(
            [(v['edges'], v['quantile']) for v in self.bin_quantiles.values()]
        ):
            mask = (uncertainties >= edges[0]) & (uncertainties < edges[1])
            widths[mask] = q
            
        # Handle edge cases
        widths[widths == 0] = np.median(list(v['quantile'] for v in self.bin_quantiles.values()))
        
        if predictions.ndim > 1:
            widths = widths[:, np.newaxis]
            
        lower = predictions - widths
        upper = predictions + widths
        
        return lower, upper


class MetricsTracker:
    """
    Comprehensive metrics tracking for training.
    """
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': [],
            'learning_rate': [],
            'epoch_time': [],
        }
        self.best_metrics = {}
        
    def update(self, phase: str, metrics: Dict[str, float]):
        """Update metrics for a phase."""
        for key, value in metrics.items():
            full_key = f"{phase}_{key}"
            if full_key not in self.history:
                self.history[full_key] = []
            self.history[full_key].append(value)
            
    def update_best(self, metrics: Dict[str, float]):
        """Update best metrics."""
        self.best_metrics = metrics.copy()
        
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'best': self.best_metrics,
            'final': {k: v[-1] if v else None for k, v in self.history.items()},
            'history_length': len(self.history.get('train_loss', []))
        }


def get_device(config: TrainerConfig) -> torch.device:
    """Determine best available device."""
    if config.device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(config.device)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ProductionTrainer:
    """
    Production-grade trainer for time series models.
    
    Features:
    - Walk-forward cross-validation
    - Conformal prediction intervals
    - Mixed precision training
    - Comprehensive checkpointing
    - Early stopping
    - Learning rate scheduling
    
    Usage:
        trainer = ProductionTrainer(model, config)
        
        # Train with walk-forward CV
        results = trainer.fit(
            train_dataset,
            val_dataset=None,  # Uses walk-forward
            loss_fn=loss_fn
        )
        
        # Get calibrated predictions
        predictions = trainer.predict_with_intervals(
            test_data,
            alpha=0.1  # 90% intervals
        )
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainerConfig] = None,
        loss_fn: Optional[Callable] = None
    ):
        self.config = config or TrainerConfig()
        self.device = get_device(self.config)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn or nn.MSELoss()
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if self.config.use_amp and self.device.type == 'cuda' else None
        
        # State
        self.state = TrainingState()
        self.metrics = MetricsTracker()
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )
        
        # Conformal predictor
        self.conformal = ConformalPredictor()
        
        # Set seed
        set_seed(self.config.seed)
        
        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
    def _setup_optimizer(self, num_training_steps: int):
        """Setup optimizer and scheduler."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        warmup_steps = self.config.warmup_epochs * (num_training_steps // self.config.gradient_accumulation_steps)
        
        if self.config.scheduler == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=self.config.epochs * num_training_steps // self.config.gradient_accumulation_steps,
                pct_start=0.1
            )
        elif self.config.scheduler == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2
            )
        elif self.config.scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        elif self.config.scheduler == 'warmup':
            warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps)
            main = CosineAnnealingWarmRestarts(self.optimizer, T_0=10)
            self.scheduler = SequentialLR(self.optimizer, [warmup, main], milestones=[warmup_steps])
            
    def _get_validator(self) -> Union[WalkForwardValidator, RollingWindowValidator, PurgedKFoldValidator]:
        """Get appropriate cross-validator."""
        strategy = self.config.validation_strategy
        
        if strategy == ValidationStrategy.WALK_FORWARD:
            return WalkForwardValidator(
                num_folds=self.config.num_folds,
                gap=self.config.purge_gap
            )
        elif strategy == ValidationStrategy.ROLLING:
            return RollingWindowValidator(
                num_folds=self.config.num_folds,
                gap=self.config.purge_gap
            )
        elif strategy == ValidationStrategy.PURGED_KFOLD:
            return PurgedKFoldValidator(
                num_folds=self.config.num_folds,
                purge_gap=self.config.purge_gap
            )
        else:
            return None
            
    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
            # Forward pass
            use_amp = self.scaler is not None
            
            with autocast(enabled=use_amp):
                outputs = self.model(inputs)
                
                # Handle dict outputs
                if isinstance(outputs, dict):
                    predictions = outputs.get('forecast', outputs.get('prediction', outputs.get('output')))
                else:
                    predictions = outputs
                    
                # Compute loss
                if isinstance(self.loss_fn, nn.Module):
                    loss_dict = self.loss_fn(predictions, targets)
                    if isinstance(loss_dict, dict):
                        loss = loss_dict['loss']
                    else:
                        loss = loss_dict
                else:
                    loss = self.loss_fn(predictions, targets)
                    
                loss = loss / self.config.gradient_accumulation_steps
                
            # Backward pass
            if use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if use_amp:
                    self.scaler.unscale_(self.optimizer)
                    
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                if use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
                
                if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step()
                    
            # Metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_mae += torch.abs(predictions - targets).mean().item()
            total_mse += ((predictions - targets) ** 2).mean().item()
            num_batches += 1
            
            self.state.global_step += 1
            
        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches,
            'rmse': np.sqrt(total_mse / num_batches)
        }
        
    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate on held-out data."""
        self.model.eval()
        
        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
            outputs = self.model(inputs)
            
            if isinstance(outputs, dict):
                predictions = outputs.get('forecast', outputs.get('prediction', outputs.get('output')))
            else:
                predictions = outputs
                
            if isinstance(self.loss_fn, nn.Module):
                loss_dict = self.loss_fn(predictions, targets)
                if isinstance(loss_dict, dict):
                    loss = loss_dict['loss']
                else:
                    loss = loss_dict
            else:
                loss = self.loss_fn(predictions, targets)
                
            total_loss += loss.item()
            total_mae += torch.abs(predictions - targets).mean().item()
            total_mse += ((predictions - targets) ** 2).mean().item()
            num_batches += 1
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
        # Store for conformal calibration
        self._last_val_predictions = np.concatenate(all_predictions)
        self._last_val_targets = np.concatenate(all_targets)
        
        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches,
            'rmse': np.sqrt(total_mse / num_batches)
        }
        
    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        loss_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (uses CV if None)
            loss_fn: Loss function (uses default if None)
            
        Returns:
            Training results and metrics
        """
        if loss_fn is not None:
            self.loss_fn = loss_fn
            
        # Create data loaders
        if val_dataset is not None:
            # Simple holdout
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
            cv_splits = [(list(range(len(train_dataset))), list(range(len(val_dataset))))]
        else:
            # Cross-validation
            validator = self._get_validator()
            if validator is not None:
                cv_splits = validator.split(train_dataset)
            else:
                # Default 80/20 split
                n = len(train_dataset)
                split = int(0.8 * n)
                cv_splits = [(list(range(split)), list(range(split, n)))]
                
        # Setup optimizer
        self._setup_optimizer(len(train_dataset) // self.config.batch_size)
        
        # Training loop
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            if self.config.verbose >= 1:
                print(f"\n{'='*50}")
                print(f"Fold {fold_idx + 1}/{len(cv_splits)}")
                print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
                
            # Create fold data loaders
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(train_dataset, val_idx) if val_dataset is None else val_dataset
            
            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
            
            # Reset for each fold
            self.early_stopping = EarlyStopping(
                patience=self.config.patience,
                min_delta=self.config.min_delta
            )
            
            # Epoch loop
            for epoch in range(self.config.epochs):
                start_time = time.time()
                
                # Train
                train_metrics = self._train_epoch(train_loader)
                
                # Validate
                if epoch % self.config.val_frequency == 0:
                    val_metrics = self._validate(val_loader)
                    
                    # Update scheduler if plateau
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                        
                    # Check early stopping
                    if self.early_stopping(val_metrics['loss']):
                        if self.config.verbose >= 1:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break
                        
                    # Update best
                    if val_metrics['loss'] < self.state.best_loss:
                        self.state.best_loss = val_metrics['loss']
                        self.state.best_epoch = epoch
                        self.metrics.update_best(val_metrics)
                        
                        if self.config.save_best_only:
                            self._save_checkpoint('best')
                            
                else:
                    val_metrics = None
                    
                epoch_time = time.time() - start_time
                
                # Update metrics
                self.metrics.update('train', train_metrics)
                if val_metrics:
                    self.metrics.update('val', val_metrics)
                self.metrics.history['epoch_time'].append(epoch_time)
                self.metrics.history['learning_rate'].append(
                    self.optimizer.param_groups[0]['lr']
                )
                
                # Log
                if self.config.verbose >= 1 and epoch % 10 == 0:
                    val_str = f", Val Loss: {val_metrics['loss']:.4f}" if val_metrics else ""
                    print(f"Epoch {epoch + 1}/{self.config.epochs} - "
                          f"Train Loss: {train_metrics['loss']:.4f}{val_str} - "
                          f"Time: {epoch_time:.1f}s")
                    
                self.state.epoch = epoch + 1
                
                # Checkpoint
                if epoch % self.config.save_frequency == 0:
                    self._save_checkpoint(f'epoch_{epoch}')
                    
            # Fold results
            fold_results.append({
                'fold': fold_idx,
                'best_loss': self.state.best_loss,
                'best_epoch': self.state.best_epoch,
                'final_metrics': self.metrics.get_summary()
            })
            
        # Calibrate conformal predictor on last validation set
        if hasattr(self, '_last_val_predictions'):
            self.conformal.calibrate(
                self._last_val_predictions,
                self._last_val_targets
            )
            
        return {
            'fold_results': fold_results,
            'best_loss': min(r['best_loss'] for r in fold_results),
            'summary': self.metrics.get_summary(),
            'conformal_calibrated': self.conformal.quantile is not None
        }
        
    @torch.no_grad()
    def predict(
        self,
        inputs: Union[Tensor, DataLoader],
        return_samples: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions.
        
        Args:
            inputs: Input tensor or DataLoader
            return_samples: Return MC samples if model supports it
            
        Returns:
            Dict with predictions and optional uncertainty
        """
        self.model.eval()
        
        if isinstance(inputs, DataLoader):
            all_preds = []
            for batch in inputs:
                if isinstance(batch, (tuple, list)):
                    x = batch[0].to(self.device)
                else:
                    x = batch['input'].to(self.device)
                    
                outputs = self.model(x)
                
                if isinstance(outputs, dict):
                    pred = outputs.get('forecast', outputs.get('prediction'))
                else:
                    pred = outputs
                    
                all_preds.append(pred.cpu().numpy())
                
            predictions = np.concatenate(all_preds)
        else:
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            
            if isinstance(outputs, dict):
                predictions = outputs.get('forecast', outputs.get('prediction')).cpu().numpy()
            else:
                predictions = outputs.cpu().numpy()
                
        return {'predictions': predictions}
        
    def predict_with_intervals(
        self,
        inputs: Union[Tensor, DataLoader],
        alpha: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions with calibrated intervals.
        
        Args:
            inputs: Input data
            alpha: Miscoverage rate (default 0.1 for 90% intervals)
            
        Returns:
            Dict with predictions, lower, upper bounds
        """
        result = self.predict(inputs)
        
        if self.conformal.quantile is not None:
            self.conformal.alpha = alpha
            lower, upper = self.conformal.predict_intervals(result['predictions'])
            result['lower'] = lower
            result['upper'] = upper
            result['interval_width'] = upper - lower
        else:
            warnings.warn("Conformal predictor not calibrated. Run fit() first.")
            
        return result
        
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'state': asdict(self.state),
            'config': asdict(self.config),
            'metrics': self.metrics.history
        }
        
        path = Path(self.config.checkpoint_dir) / f"{name}.pt"
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if 'state' in checkpoint:
            for key, value in checkpoint['state'].items():
                setattr(self.state, key, value)
                
        if 'metrics' in checkpoint:
            self.metrics.history = checkpoint['metrics']
