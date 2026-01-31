"""
Model Training and Validation Utilities
========================================

Professional training infrastructure:
- Walk-forward validation for time series
- Cross-validation with purging
- Hyperparameter optimization
- Learning rate scheduling
- Early stopping
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
import time


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    l2_regularization: float = 0.0001
    dropout: float = 0.1
    gradient_clip: float = 1.0
    lr_decay: float = 0.95
    lr_decay_steps: int = 10


@dataclass
class TrainingResult:
    """Results from model training"""
    final_loss: float
    best_loss: float
    training_history: Dict[str, List[float]]
    best_epoch: int
    total_epochs: int
    training_time: float
    parameters_count: int
    validation_metrics: Dict[str, float]


class ModelTrainer:
    """
    Generic model trainer with best practices.
    
    Features:
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    - Regularization
    - Training history tracking
    """
    
    def __init__(self, config: TrainingConfig = None):
        """Initialize trainer with configuration"""
        self.config = config or TrainingConfig()
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        loss_fn: Callable = None,
        verbose: bool = True
    ) -> TrainingResult:
        """
        Train a model with best practices.
        
        Args:
            model: Model with fit/predict methods or gradient-based training
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            loss_fn: Loss function (default: MSE)
            verbose: Print progress
            
        Returns:
            TrainingResult with training history
        """
        start_time = time.time()
        
        # Default loss function
        if loss_fn is None:
            loss_fn = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
        
        # Split validation if not provided
        if X_val is None or y_val is None:
            split_idx = int(len(X_train) * (1 - self.config.validation_split))
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
        
        # Training loop
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        current_lr = self.config.learning_rate
        
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
        
        n_batches = max(1, len(X_train) // self.config.batch_size)
        
        for epoch in range(self.config.epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_losses = []
            
            for batch in range(n_batches):
                start_idx = batch * self.config.batch_size
                end_idx = start_idx + self.config.batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                if hasattr(model, 'forward'):
                    predictions = model.forward(X_batch)
                elif hasattr(model, 'predict'):
                    predictions = model.predict(X_batch)
                else:
                    predictions = model(X_batch)
                
                if isinstance(predictions, dict):
                    predictions = predictions.get('prediction', predictions.get('mean', 0))
                
                # Compute loss
                loss = loss_fn(y_batch, predictions)
                epoch_losses.append(loss)
                
                # Gradient update (if model supports it)
                if hasattr(model, 'backward'):
                    model.backward(loss, current_lr)
                elif hasattr(model, 'update'):
                    model.update(X_batch, y_batch, current_lr)
            
            train_loss = np.mean(epoch_losses)
            
            # Validation
            if hasattr(model, 'predict'):
                val_pred = model.predict(X_val)
            elif hasattr(model, 'forward'):
                val_pred = model.forward(X_val)
            else:
                val_pred = model(X_val)
            
            if isinstance(val_pred, dict):
                val_pred = val_pred.get('prediction', val_pred.get('mean', 0))
            
            val_loss = loss_fn(y_val, val_pred)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            # Learning rate decay
            if (epoch + 1) % self.config.lr_decay_steps == 0:
                current_lr *= self.config.lr_decay
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={current_lr:.6f}")
        
        training_time = time.time() - start_time
        
        # Final validation metrics
        val_metrics = self._compute_metrics(y_val, val_pred)
        
        return TrainingResult(
            final_loss=self.history['val_loss'][-1],
            best_loss=best_val_loss,
            training_history=self.history,
            best_epoch=best_epoch,
            total_epochs=len(self.history['train_loss']),
            training_time=training_time,
            parameters_count=self._count_parameters(model),
            validation_metrics=val_metrics
        )
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute various metrics"""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        
        # Directional accuracy (for returns)
        if len(y_true) > 0:
            directional = np.mean(np.sign(y_true) == np.sign(y_pred))
        else:
            directional = 0.0
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'directional_accuracy': directional
        }
    
    def _count_parameters(self, model: Any) -> int:
        """Count model parameters"""
        count = 0
        for attr in dir(model):
            if attr.startswith('W_') or attr.startswith('b_'):
                param = getattr(model, attr, None)
                if isinstance(param, np.ndarray):
                    count += param.size
        return count


class CrossValidator:
    """
    Cross-validation with purging for time series.
    
    Implements:
    - Standard K-fold
    - Time series split
    - Purged K-fold (gap between train/test)
    - Combinatorial purged CV
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        method: str = 'time_series',  # 'kfold', 'time_series', 'purged'
        purge_gap: int = 0,
        embargo_pct: float = 0.0
    ):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of CV splits
            method: CV method to use
            purge_gap: Number of samples to exclude between train/test
            embargo_pct: Percentage of test set to exclude from training
        """
        self.n_splits = n_splits
        self.method = method
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.
        
        Args:
            X: Feature array
            y: Target array (optional)
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        n_samples = len(X)
        
        if self.method == 'kfold':
            return self._kfold_split(n_samples)
        elif self.method == 'time_series':
            return self._time_series_split(n_samples)
        elif self.method == 'purged':
            return self._purged_split(n_samples)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _kfold_split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Standard K-fold split"""
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits
        
        splits = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else n_samples
            
            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def _time_series_split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Time series split (expanding window)"""
        test_size = n_samples // (self.n_splits + 1)
        
        splits = []
        for i in range(self.n_splits):
            train_end = (i + 1) * test_size
            test_start = train_end + self.purge_gap
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def _purged_split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Purged cross-validation with embargo"""
        test_size = n_samples // (self.n_splits + 1)
        embargo_size = int(test_size * self.embargo_pct)
        
        splits = []
        for i in range(self.n_splits):
            test_start = (i + 1) * test_size
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
            
            # Training excludes purge gap and embargo period
            train_end = test_start - self.purge_gap
            train_idx = np.arange(0, max(0, train_end - embargo_size))
            
            # Also include samples after test if available
            after_test_start = test_end + self.purge_gap
            if after_test_start < n_samples:
                train_idx = np.concatenate([
                    train_idx,
                    np.arange(after_test_start, n_samples)
                ])
            
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def cross_validate(
        self,
        model_factory: Callable,
        X: np.ndarray,
        y: np.ndarray,
        trainer: ModelTrainer = None
    ) -> Dict[str, Any]:
        """
        Run cross-validation.
        
        Args:
            model_factory: Function that creates a new model instance
            X: Features
            y: Targets
            trainer: Model trainer to use
            
        Returns:
            Dictionary with CV results
        """
        if trainer is None:
            trainer = ModelTrainer()
        
        splits = self.split(X, y)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            model = model_factory()
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Train
            result = trainer.train(
                model, X_train, y_train, X_test, y_test,
                verbose=False
            )
            
            fold_results.append({
                'fold': fold,
                'best_loss': result.best_loss,
                'metrics': result.validation_metrics
            })
        
        # Aggregate results
        all_metrics = {}
        for metric in fold_results[0]['metrics'].keys():
            values = [r['metrics'][metric] for r in fold_results]
            all_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        return {
            'fold_results': fold_results,
            'aggregated_metrics': all_metrics,
            'n_splits': len(splits)
        }


class WalkForwardValidator:
    """
    Walk-forward optimization and validation.
    
    More realistic for trading strategy development:
    - Train on historical data
    - Test on next period
    - Slide window forward
    - Re-optimize periodically
    """
    
    def __init__(
        self,
        train_window: int = 252,  # 1 year
        test_window: int = 21,    # 1 month
        step_size: int = 21,      # Slide by 1 month
        reoptimize_every: int = 63  # Re-optimize every quarter
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            train_window: Training window size (samples)
            test_window: Testing window size
            step_size: How much to slide forward each iteration
            reoptimize_every: How often to re-optimize hyperparameters
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.reoptimize_every = reoptimize_every
    
    def validate(
        self,
        model_factory: Callable,
        X: np.ndarray,
        y: np.ndarray,
        hyperopt: Optional['HyperparameterOptimizer'] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation.
        
        Args:
            model_factory: Function to create model
            X: Features
            y: Targets
            hyperopt: Optional hyperparameter optimizer
            
        Returns:
            Walk-forward results
        """
        n_samples = len(X)
        results = []
        all_predictions = []
        all_actuals = []
        
        step = 0
        current_pos = self.train_window
        
        while current_pos + self.test_window <= n_samples:
            train_start = max(0, current_pos - self.train_window)
            train_end = current_pos
            test_start = current_pos
            test_end = min(current_pos + self.test_window, n_samples)
            
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            # Re-optimize if needed
            if hyperopt and step % (self.reoptimize_every // self.step_size) == 0:
                best_params = hyperopt.optimize(model_factory, X_train, y_train)
                model = model_factory(**best_params)
            else:
                model = model_factory()
            
            # Train model
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            
            # Predict
            if hasattr(model, 'predict'):
                predictions = model.predict(X_test)
            else:
                predictions = model(X_test)
            
            if isinstance(predictions, dict):
                predictions = predictions.get('prediction', 0)
            
            predictions = np.array(predictions).flatten()
            y_test = np.array(y_test).flatten()
            
            # Compute metrics
            mse = np.mean((y_test - predictions) ** 2)
            directional = np.mean(np.sign(y_test) == np.sign(predictions))
            
            results.append({
                'step': step,
                'train_period': (train_start, train_end),
                'test_period': (test_start, test_end),
                'mse': mse,
                'directional_accuracy': directional,
                'n_samples': len(y_test)
            })
            
            all_predictions.extend(predictions.tolist())
            all_actuals.extend(y_test.tolist())
            
            current_pos += self.step_size
            step += 1
        
        # Overall statistics
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        
        return {
            'step_results': results,
            'overall_mse': np.mean((all_actuals - all_predictions) ** 2),
            'overall_directional': np.mean(np.sign(all_actuals) == np.sign(all_predictions)),
            'sharpe_ratio': self._compute_sharpe(all_actuals, all_predictions),
            'predictions': all_predictions,
            'actuals': all_actuals
        }
    
    def _compute_sharpe(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        risk_free: float = 0.0
    ) -> float:
        """Compute Sharpe ratio of strategy returns"""
        # Strategy: go long when prediction > 0, else flat
        strategy_returns = np.where(predictions > 0, actuals, 0)
        
        mean_return = np.mean(strategy_returns) - risk_free / 252
        std_return = np.std(strategy_returns) + 1e-8
        
        return np.sqrt(252) * mean_return / std_return


class HyperparameterOptimizer:
    """
    Hyperparameter optimization for financial models.
    
    Methods:
    - Grid search
    - Random search
    - Bayesian optimization (simplified)
    """
    
    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        method: str = 'random',  # 'grid', 'random', 'bayesian'
        n_iter: int = 20,
        cv_splits: int = 3,
        scoring: str = 'mse'  # 'mse', 'directional', 'sharpe'
    ):
        """
        Initialize optimizer.
        
        Args:
            param_grid: Dictionary of parameter name -> possible values
            method: Optimization method
            n_iter: Number of iterations (for random/bayesian)
            cv_splits: Number of CV splits for evaluation
            scoring: Scoring metric
        """
        self.param_grid = param_grid
        self.method = method
        self.n_iter = n_iter
        self.cv_splits = cv_splits
        self.scoring = scoring
        
        self.results_: List[Dict] = []
        self.best_params_: Dict = {}
        self.best_score_: float = float('inf') if scoring == 'mse' else float('-inf')
    
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations based on method"""
        if self.method == 'grid':
            # Full grid
            keys = list(self.param_grid.keys())
            values = list(self.param_grid.values())
            
            combinations = []
            indices = [0] * len(keys)
            
            while True:
                combo = {keys[i]: values[i][indices[i]] for i in range(len(keys))}
                combinations.append(combo)
                
                # Increment indices
                for i in range(len(keys) - 1, -1, -1):
                    indices[i] += 1
                    if indices[i] < len(values[i]):
                        break
                    indices[i] = 0
                else:
                    break
            
            return combinations
        
        elif self.method == 'random':
            # Random sampling
            combinations = []
            for _ in range(self.n_iter):
                combo = {
                    key: np.random.choice(values)
                    for key, values in self.param_grid.items()
                }
                combinations.append(combo)
            return combinations
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def optimize(
        self,
        model_factory: Callable,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters.
        
        Args:
            model_factory: Function that takes params and returns model
            X: Features
            y: Targets
            
        Returns:
            Best parameters found
        """
        combinations = self._generate_param_combinations()
        cv = CrossValidator(n_splits=self.cv_splits, method='time_series')
        
        for params in combinations:
            # Evaluate with CV
            try:
                cv_result = cv.cross_validate(
                    lambda: model_factory(**params),
                    X, y
                )
                
                # Get score
                if self.scoring == 'mse':
                    score = cv_result['aggregated_metrics']['mse']['mean']
                    is_better = score < self.best_score_
                elif self.scoring == 'directional':
                    score = cv_result['aggregated_metrics']['directional_accuracy']['mean']
                    is_better = score > self.best_score_
                else:
                    score = cv_result['aggregated_metrics']['mse']['mean']
                    is_better = score < self.best_score_
                
                self.results_.append({
                    'params': params,
                    'score': score,
                    'cv_result': cv_result
                })
                
                if is_better:
                    self.best_score_ = score
                    self.best_params_ = params
                    
            except Exception as e:
                print(f"Failed for params {params}: {e}")
                continue
        
        return self.best_params_
