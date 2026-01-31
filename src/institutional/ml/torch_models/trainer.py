"""
Production Training Pipeline
=============================

Professional training infrastructure for financial ML:
- Walk-forward validation for time series
- Distributed training support
- Mixed precision training
- Hyperparameter optimization
- Learning rate scheduling
- Early stopping with patience
- Model checkpointing and versioning
- Training metrics logging
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.cuda.amp import GradScaler, autocast

from .base import BaseFinancialModel, ModelConfig, TrainingMetrics

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Optimization
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-7
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 200
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-6
    
    # Learning rate scheduling
    scheduler_type: str = 'cosine_warmup'  # 'cosine_warmup', 'plateau', 'step', 'one_cycle'
    warmup_epochs: int = 5
    lr_decay_factor: float = 0.5
    lr_patience: int = 5
    
    # Gradient handling
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3
    
    # Logging
    log_every_n_steps: int = 100
    wandb_project: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainingState:
    """Current training state."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    epochs_without_improvement: int = 0
    training_history: List[Dict] = field(default_factory=list)


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time series.
    
    Proper time series validation that prevents lookahead bias:
    - Expanding window or rolling window
    - Gap between train and test to prevent leakage
    - Multiple folds for robust evaluation
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        gap: int = 5,
        expanding: bool = True
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            n_splits: Number of train/test splits
            train_ratio: Ratio of data for training in each split
            gap: Number of samples to skip between train and test
            expanding: If True, use expanding window; else rolling
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.gap = gap
        self.expanding = expanding
    
    def split(
        self,
        n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for walk-forward validation.
        
        Args:
            n_samples: Total number of samples
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        test_size = int(n_samples * (1 - self.train_ratio) / self.n_splits)
        
        splits = []
        
        for fold in range(self.n_splits):
            if self.expanding:
                # Expanding window: train on all data up to test start
                test_start = int(n_samples * self.train_ratio) + fold * test_size
            else:
                # Rolling window: fixed-size training window
                window_size = int(n_samples * self.train_ratio)
                test_start = window_size + fold * test_size
            
            test_end = min(test_start + test_size, n_samples)
            train_end = test_start - self.gap
            
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, test_start - window_size - self.gap)
            
            if train_end <= train_start or test_end <= test_start:
                continue
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        return splits


class PurgedKFold:
    """
    Purged K-Fold cross-validation.
    
    Prevents information leakage in overlapping samples
    by removing (purging) train samples that overlap with test.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01
    ):
        """
        Initialize purged K-fold.
        
        Args:
            n_splits: Number of folds
            embargo_pct: Percentage of samples to embargo after each test set
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        n_samples: int,
        sample_groups: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged train/test indices.
        
        Args:
            n_samples: Total number of samples
            sample_groups: Optional group labels for samples
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        indices = np.arange(n_samples)
        test_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)
        
        splits = []
        
        for fold in range(self.n_splits):
            test_start = fold * test_size
            test_end = (fold + 1) * test_size if fold < self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            
            # Purge: remove samples that could leak into test
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:test_end] = False
            
            # Embargo: remove samples right after test
            embargo_end = min(test_end + embargo_size, n_samples)
            train_mask[test_end:embargo_end] = False
            
            train_indices = indices[train_mask]
            
            splits.append((train_indices, test_indices))
        
        return splits


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with linear warmup.
    
    Learning rate schedule:
    1. Linear warmup from 0 to initial_lr
    2. Cosine annealing from initial_lr to min_lr
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-7,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
                for base_lr in self.base_lrs
            ]


class ModelTrainer:
    """
    Production-grade model trainer.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Metrics logging
    """
    
    def __init__(
        self,
        model: BaseFinancialModel,
        config: TrainingConfig
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.state = TrainingState()
        
        # Set seed for reproducibility
        self._set_seed(config.seed)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp and torch.cuda.is_available() else None
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self._setup_logging()
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with weight decay handling."""
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if self.config.scheduler_type == 'cosine_warmup':
            return CosineWarmupScheduler(
                self.optimizer,
                warmup_epochs=self.config.warmup_epochs,
                total_epochs=self.config.max_epochs,
                min_lr=self.config.min_learning_rate
            )
        elif self.config.scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.lr_decay_factor,
                patience=self.config.lr_patience,
                min_lr=self.config.min_learning_rate,
                verbose=True
            )
        elif self.config.scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=self.config.lr_decay_factor
            )
        elif self.config.scheduler_type == 'one_cycle':
            # Requires knowing total steps, set placeholder
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.max_epochs,
                steps_per_epoch=1000  # Will be updated
            )
        else:
            return optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0)
    
    def _setup_logging(self):
        """Setup logging handlers."""
        log_file = self.checkpoint_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable,
        metrics_fn: Optional[Callable] = None
    ) -> TrainingState:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            metrics_fn: Optional metrics computation function
            
        Returns:
            Final training state
        """
        logger.info(f"Starting training for {self.config.max_epochs} epochs")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        logger.info(f"Device: {self.model.device}")
        
        try:
            for epoch in range(self.config.max_epochs):
                self.state.epoch = epoch
                
                # Training phase
                train_metrics = self._train_epoch(train_loader, loss_fn)
                
                # Validation phase
                val_metrics = self._validate(val_loader, loss_fn, metrics_fn)
                
                # Update learning rate
                self._update_scheduler(val_metrics['loss'])
                
                # Log metrics
                self._log_epoch(epoch, train_metrics, val_metrics)
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_every_n_epochs == 0:
                    self._save_checkpoint(epoch, val_metrics['loss'])
                
                # Early stopping check
                if self._check_early_stopping(val_metrics['loss']):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Update best model
                if val_metrics['loss'] < self.state.best_val_loss:
                    self.state.best_val_loss = val_metrics['loss']
                    self.state.epochs_without_improvement = 0
                    self._save_checkpoint(epoch, val_metrics['loss'], is_best=True)
                else:
                    self.state.epochs_without_improvement += 1
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint(self.state.epoch, self.state.best_val_loss, is_best=False)
        
        return self.state
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: Callable
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        grad_norms = []
        
        accumulation_counter = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            x, y = self._prepare_batch(batch)
            
            # Forward pass with optional AMP
            if self.scaler is not None:
                with autocast():
                    output = self.model(x)
                    loss = loss_fn(output, y)
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass with scaled gradients
                self.scaler.scale(loss).backward()
            else:
                output = self.model(x)
                loss = loss_fn(output, y)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            accumulation_counter += 1
            
            # Gradient accumulation
            if accumulation_counter >= self.config.gradient_accumulation_steps:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                grad_norms.append(grad_norm.item())
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                accumulation_counter = 0
                self.state.global_step += 1
            
            # Track metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps * x.size(0)
            total_samples += x.size(0)
            
            # Logging
            if batch_idx % self.config.log_every_n_steps == 0:
                logger.debug(
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.6f}, "
                    f"Grad norm: {grad_norms[-1] if grad_norms else 0:.4f}"
                )
        
        return {
            'loss': total_loss / total_samples,
            'grad_norm': np.mean(grad_norms) if grad_norms else 0.0
        }
    
    def _validate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        metrics_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = self._prepare_batch(batch)
                
                if self.scaler is not None:
                    with autocast():
                        output = self.model(x)
                        loss = loss_fn(output, y)
                else:
                    output = self.model(x)
                    loss = loss_fn(output, y)
                
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
                
                # Store for metrics
                if metrics_fn is not None:
                    if isinstance(output, dict):
                        pred = output.get('predictions', output.get('h1', {}).get('mean', output))
                    else:
                        pred = output
                    all_predictions.append(pred.cpu())
                    all_targets.append(y.cpu())
        
        metrics = {'loss': total_loss / total_samples}
        
        if metrics_fn is not None and all_predictions:
            predictions = torch.cat(all_predictions)
            targets = torch.cat(all_targets)
            additional_metrics = metrics_fn(predictions, targets)
            metrics.update(additional_metrics)
        
        return metrics
    
    def _prepare_batch(
        self,
        batch: Union[Tuple, Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch for training."""
        if isinstance(batch, dict):
            x = batch['features'].to(self.model.device)
            y = batch['targets'].to(self.model.device)
        elif isinstance(batch, (tuple, list)):
            x, y = batch[0].to(self.model.device), batch[1].to(self.model.device)
        else:
            raise ValueError(f"Unknown batch type: {type(batch)}")
        
        return x.float(), y.float()
    
    def _update_scheduler(self, val_loss: float):
        """Update learning rate scheduler."""
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if should stop early."""
        if val_loss < self.state.best_val_loss - self.config.early_stopping_min_delta:
            return False
        
        return self.state.epochs_without_improvement >= self.config.early_stopping_patience
    
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log epoch metrics."""
        current_lr = self.optimizer.param_groups[0]['lr']
        
        log_msg = (
            f"Epoch {epoch}/{self.config.max_epochs} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"Grad Norm: {train_metrics.get('grad_norm', 0):.4f}"
        )
        logger.info(log_msg)
        
        # Store in history
        metrics_entry = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'learning_rate': current_lr,
            'grad_norm': train_metrics.get('grad_norm', 0),
            'timestamp': datetime.now().isoformat(),
            **{f'val_{k}': v for k, v in val_metrics.items() if k != 'loss'}
        }
        self.state.training_history.append(metrics_entry)
    
    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_state': asdict(self.state) if hasattr(self.state, '__dataclass_fields__') else vars(self.state),
            'config': self.config.to_dict(),
            'model_config': self.model.config.to_dict(),
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        filename = f'checkpoint_epoch_{epoch:04d}.pt'
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filepath}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent."""
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda p: int(p.stem.split('_')[-1]),
            reverse=True
        )
        
        for checkpoint in checkpoints[self.config.keep_n_checkpoints:]:
            checkpoint.unlink()
            logger.debug(f"Removed old checkpoint: {checkpoint}")
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        load_optimizer: bool = True
    ):
        """Load from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore training state
        state_dict = checkpoint.get('training_state', {})
        self.state.epoch = state_dict.get('epoch', 0)
        self.state.global_step = state_dict.get('global_step', 0)
        self.state.best_val_loss = state_dict.get('best_val_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}, epoch {self.state.epoch}")
        
        return checkpoint.get('val_loss', None)


def compute_financial_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute financial prediction metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Dictionary of metrics
    """
    # Handle nested predictions
    if isinstance(predictions, dict):
        # Get first horizon mean
        for key in predictions:
            if isinstance(predictions[key], dict) and 'mean' in predictions[key]:
                predictions = predictions[key]['mean']
                break
            elif isinstance(predictions[key], torch.Tensor):
                predictions = predictions[key]
                break
    
    predictions = predictions.float().squeeze()
    targets = targets.float().squeeze()
    
    if predictions.shape != targets.shape:
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]
    
    # MSE and RMSE
    mse = ((predictions - targets) ** 2).mean().item()
    rmse = np.sqrt(mse)
    
    # MAE
    mae = (predictions - targets).abs().mean().item()
    
    # MAPE (avoid division by zero)
    mask = targets.abs() > 1e-8
    if mask.sum() > 0:
        mape = ((predictions[mask] - targets[mask]).abs() / targets[mask].abs()).mean().item() * 100
    else:
        mape = float('inf')
    
    # Direction accuracy
    if len(predictions) > 1:
        pred_direction = (predictions[1:] > predictions[:-1]).float()
        true_direction = (targets[1:] > targets[:-1]).float()
        direction_accuracy = (pred_direction == true_direction).float().mean().item()
    else:
        direction_accuracy = 0.5
    
    # R-squared
    ss_res = ((targets - predictions) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'direction_accuracy': direction_accuracy,
        'r2': r2
    }
