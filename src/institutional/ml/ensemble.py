"""
Ensemble Models for Financial Prediction
=========================================

Combines multiple models for robust predictions:
- Stacking ensemble with meta-learner
- Boosting for sequential error correction
- Bagging for variance reduction
- Dynamic weighting based on recent performance
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum


class ModelType(Enum):
    """Types of models in ensemble"""
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    LINEAR = "linear"
    TECHNICAL = "technical"  # Rule-based technical analysis


@dataclass
class ModelPrediction:
    """Prediction from a single model"""
    model_name: str
    model_type: ModelType
    prediction: float
    confidence: float = 1.0
    uncertainty: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsemblePrediction:
    """Combined prediction from ensemble"""
    mean: float
    median: float
    std: float
    lower_bound: float  # 10th percentile
    upper_bound: float  # 90th percentile
    model_weights: Dict[str, float]
    individual_predictions: List[ModelPrediction]
    confidence: float
    agreement_score: float  # How much models agree


class BaseModel(ABC):
    """Abstract base model for ensemble"""
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        pass
    
    @abstractmethod
    def get_confidence(self, x: np.ndarray) -> np.ndarray:
        """Get prediction confidence"""
        pass


class EnsemblePredictor:
    """
    Advanced ensemble predictor combining multiple models.
    
    Features:
    - Dynamic model weighting based on recent performance
    - Uncertainty quantification via disagreement
    - Adaptive ensemble selection
    - Online learning for weight updates
    """
    
    def __init__(
        self,
        models: Optional[Dict[str, Any]] = None,
        weighting_strategy: str = 'performance',  # 'equal', 'performance', 'uncertainty'
        lookback_for_weights: int = 20,
        min_models: int = 2
    ):
        """
        Initialize ensemble.
        
        Args:
            models: Dictionary of model_name -> model object
            weighting_strategy: How to weight model predictions
            lookback_for_weights: Number of past predictions for performance weighting
            min_models: Minimum models required for prediction
        """
        self.models = models or {}
        self.weighting_strategy = weighting_strategy
        self.lookback_for_weights = lookback_for_weights
        self.min_models = min_models
        
        # Track model performance
        self.model_errors: Dict[str, List[float]] = {name: [] for name in self.models}
        self.model_weights: Dict[str, float] = {name: 1.0/len(self.models) if self.models else 0 
                                                  for name in self.models}
    
    def add_model(self, name: str, model: Any, model_type: ModelType = ModelType.LINEAR):
        """Add a model to the ensemble"""
        self.models[name] = {'model': model, 'type': model_type}
        self.model_errors[name] = []
        self._update_equal_weights()
    
    def _update_equal_weights(self):
        """Reset to equal weights"""
        n = len(self.models)
        self.model_weights = {name: 1.0/n for name in self.models}
    
    def _update_performance_weights(self):
        """Update weights based on recent performance (inverse error)"""
        # Compute mean recent error for each model
        mean_errors = {}
        for name, errors in self.model_errors.items():
            if len(errors) >= 3:
                recent = errors[-self.lookback_for_weights:]
                mean_errors[name] = np.mean(np.abs(recent)) + 1e-6
            else:
                mean_errors[name] = 1.0  # Default
        
        # Inverse error weighting
        inverse_errors = {name: 1.0/err for name, err in mean_errors.items()}
        total = sum(inverse_errors.values())
        
        self.model_weights = {name: inv/total for name, inv in inverse_errors.items()}
    
    def _update_uncertainty_weights(self, uncertainties: Dict[str, float]):
        """Update weights based on model uncertainty (lower = better)"""
        inverse_unc = {name: 1.0/(unc + 1e-6) for name, unc in uncertainties.items()}
        total = sum(inverse_unc.values())
        self.model_weights = {name: inv/total for name, inv in inverse_unc.items()}
    
    def predict(
        self,
        x: np.ndarray,
        return_individual: bool = True
    ) -> EnsemblePrediction:
        """
        Generate ensemble prediction.
        
        Args:
            x: Input data
            return_individual: Whether to return individual model predictions
            
        Returns:
            EnsemblePrediction with combined results
        """
        if len(self.models) < self.min_models:
            raise ValueError(f"Need at least {self.min_models} models")
        
        # Collect predictions from all models
        predictions = []
        uncertainties = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            model_type = model_info['type']
            
            try:
                # Get prediction
                if hasattr(model, 'predict'):
                    pred = model.predict(x)
                    if isinstance(pred, dict):
                        pred_value = pred.get('prediction', pred.get('mean', 0))
                        unc = pred.get('uncertainty', pred.get('std', 0.1))
                    elif isinstance(pred, np.ndarray):
                        pred_value = float(np.mean(pred))
                        unc = float(np.std(pred)) if len(pred) > 1 else 0.1
                    else:
                        pred_value = float(pred)
                        unc = 0.1
                else:
                    # Assume callable
                    pred_value = float(model(x))
                    unc = 0.1
                
                predictions.append(ModelPrediction(
                    model_name=name,
                    model_type=model_type,
                    prediction=pred_value,
                    confidence=1.0 / (1.0 + unc),
                    uncertainty=unc
                ))
                uncertainties[name] = unc
                
            except Exception as e:
                print(f"Model {name} failed: {e}")
                continue
        
        if len(predictions) < self.min_models:
            raise ValueError(f"Only {len(predictions)} models succeeded")
        
        # Update weights based on strategy
        if self.weighting_strategy == 'performance':
            self._update_performance_weights()
        elif self.weighting_strategy == 'uncertainty':
            self._update_uncertainty_weights(uncertainties)
        # else: keep equal weights
        
        # Compute weighted ensemble prediction
        pred_values = np.array([p.prediction for p in predictions])
        weights = np.array([self.model_weights.get(p.model_name, 1.0/len(predictions)) 
                           for p in predictions])
        weights = weights / weights.sum()  # Normalize
        
        weighted_mean = np.sum(pred_values * weights)
        weighted_std = np.sqrt(np.sum(weights * (pred_values - weighted_mean)**2))
        
        # Compute percentiles for confidence intervals
        sorted_preds = np.sort(pred_values)
        n = len(sorted_preds)
        lower_idx = max(0, int(0.1 * n))
        upper_idx = min(n - 1, int(0.9 * n))
        
        # Agreement score: how much models agree (inverse of spread)
        spread = np.max(pred_values) - np.min(pred_values)
        agreement = 1.0 / (1.0 + spread / (np.abs(weighted_mean) + 1e-6))
        
        # Overall confidence
        avg_confidence = np.mean([p.confidence for p in predictions])
        confidence = avg_confidence * agreement
        
        return EnsemblePrediction(
            mean=weighted_mean,
            median=np.median(pred_values),
            std=weighted_std,
            lower_bound=sorted_preds[lower_idx],
            upper_bound=sorted_preds[upper_idx],
            model_weights={p.model_name: weights[i] for i, p in enumerate(predictions)},
            individual_predictions=predictions if return_individual else [],
            confidence=confidence,
            agreement_score=agreement
        )
    
    def update_with_actual(self, actual: float, predictions: EnsemblePrediction):
        """Update model errors with actual value for future weighting"""
        for pred in predictions.individual_predictions:
            error = actual - pred.prediction
            if pred.model_name in self.model_errors:
                self.model_errors[pred.model_name].append(error)
                # Keep only recent errors
                if len(self.model_errors[pred.model_name]) > self.lookback_for_weights * 2:
                    self.model_errors[pred.model_name] = \
                        self.model_errors[pred.model_name][-self.lookback_for_weights:]


class StackingEnsemble:
    """
    Stacking ensemble with meta-learner.
    
    Level 1: Base models make predictions
    Level 2: Meta-model learns to combine base predictions
    
    This is more sophisticated than simple averaging as it learns
    optimal combinations based on training data.
    """
    
    def __init__(
        self,
        base_models: List[Any],
        meta_model: Optional[Any] = None,
        use_cross_validation: bool = True,
        n_folds: int = 5
    ):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of base models
            meta_model: Meta-learner (defaults to Ridge regression)
            use_cross_validation: Use CV for meta-features
            n_folds: Number of CV folds
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.use_cross_validation = use_cross_validation
        self.n_folds = n_folds
        
        # If no meta-model, use simple linear combination
        if meta_model is None:
            self.meta_weights = None  # Will be learned
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the stacking ensemble.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
        """
        n_samples = X.shape[0]
        n_base = len(self.base_models)
        
        # Generate meta-features via cross-validation or direct prediction
        if self.use_cross_validation and n_samples >= self.n_folds * 2:
            meta_features = self._generate_cv_meta_features(X, y)
        else:
            # Train base models and predict
            meta_features = np.zeros((n_samples, n_base))
            for i, model in enumerate(self.base_models):
                if hasattr(model, 'fit'):
                    model.fit(X, y)
                if hasattr(model, 'predict'):
                    meta_features[:, i] = model.predict(X).flatten()
                else:
                    meta_features[:, i] = np.array([model(x) for x in X])
        
        # Fit meta-model
        if self.meta_model is not None:
            if hasattr(self.meta_model, 'fit'):
                self.meta_model.fit(meta_features, y)
        else:
            # Simple linear regression for meta-weights
            self.meta_weights = self._fit_linear_meta(meta_features, y)
    
    def _generate_cv_meta_features(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Generate meta-features using cross-validation"""
        n_samples = X.shape[0]
        n_base = len(self.base_models)
        meta_features = np.zeros((n_samples, n_base))
        
        # Create fold indices
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        fold_size = n_samples // self.n_folds
        
        for fold in range(self.n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < self.n_folds - 1 else n_samples
            
            val_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val = X[val_idx]
            
            for i, model in enumerate(self.base_models):
                # Clone and train on fold
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                if hasattr(model, 'predict'):
                    meta_features[val_idx, i] = model.predict(X_val).flatten()
                else:
                    meta_features[val_idx, i] = np.array([model(x) for x in X_val])
        
        # Re-train base models on full data
        for model in self.base_models:
            if hasattr(model, 'fit'):
                model.fit(X, y)
        
        return meta_features
    
    def _fit_linear_meta(
        self,
        meta_features: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Fit linear meta-weights using least squares with regularization"""
        # Ridge regression: (X'X + λI)^{-1} X'y
        X = meta_features
        lambda_reg = 0.01 * len(y)
        
        XtX = np.matmul(X.T, X) + lambda_reg * np.eye(X.shape[1])
        Xty = np.matmul(X.T, y)
        
        weights = np.linalg.solve(XtX, Xty)
        
        # Normalize to sum to 1
        weights = np.clip(weights, 0, None)  # Non-negative
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        return weights
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate stacked predictions"""
        n_base = len(self.base_models)
        
        # Get base model predictions
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        n_samples = X.shape[0]
        meta_features = np.zeros((n_samples, n_base))
        
        for i, model in enumerate(self.base_models):
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                if isinstance(pred, np.ndarray):
                    meta_features[:, i] = pred.flatten()
                else:
                    meta_features[:, i] = float(pred)
            else:
                meta_features[:, i] = np.array([model(x) for x in X])
        
        # Apply meta-model
        if self.meta_model is not None:
            if hasattr(self.meta_model, 'predict'):
                return self.meta_model.predict(meta_features)
            else:
                return self.meta_model(meta_features)
        else:
            # Use learned linear weights
            return np.matmul(meta_features, self.meta_weights)


class BoostingEnsemble:
    """
    Gradient boosting-style ensemble for financial prediction.
    
    Each subsequent model learns to correct the errors of previous models.
    Particularly effective for capturing complex non-linear patterns.
    """
    
    def __init__(
        self,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 5,
        subsample: float = 0.8
    ):
        """
        Initialize boosting ensemble.
        
        Args:
            n_estimators: Number of boosting stages
            learning_rate: Shrinkage parameter
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            subsample: Fraction of samples for each tree
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        
        self.estimators: List[Dict] = []
        self.initial_prediction = 0.0
    
    def _build_tree(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        depth: int = 0
    ) -> Dict:
        """Build a simple decision tree for residuals"""
        n_samples, n_features = X.shape
        
        # Terminal conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            np.std(residuals) < 1e-6):
            return {'type': 'leaf', 'value': np.mean(residuals)}
        
        # Find best split
        best_gain = 0
        best_split = None
        
        for feature in range(n_features):
            values = X[:, feature]
            thresholds = np.percentile(values, [25, 50, 75])
            
            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                
                if left_mask.sum() < 2 or right_mask.sum() < 2:
                    continue
                
                # Compute variance reduction
                left_var = np.var(residuals[left_mask]) * left_mask.sum()
                right_var = np.var(residuals[right_mask]) * right_mask.sum()
                gain = np.var(residuals) * n_samples - left_var - right_var
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature': feature,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }
        
        if best_split is None:
            return {'type': 'leaf', 'value': np.mean(residuals)}
        
        # Recursively build children
        return {
            'type': 'split',
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': self._build_tree(
                X[best_split['left_mask']], 
                residuals[best_split['left_mask']], 
                depth + 1
            ),
            'right': self._build_tree(
                X[best_split['right_mask']], 
                residuals[best_split['right_mask']], 
                depth + 1
            )
        }
    
    def _predict_tree(self, tree: Dict, x: np.ndarray) -> float:
        """Predict using a single tree"""
        if tree['type'] == 'leaf':
            return tree['value']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_tree(tree['left'], x)
        else:
            return self._predict_tree(tree['right'], x)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the boosting ensemble.
        
        Args:
            X: Training features
            y: Training targets
        """
        self.initial_prediction = np.mean(y)
        current_predictions = np.full(len(y), self.initial_prediction)
        
        self.estimators = []
        
        for i in range(self.n_estimators):
            # Compute residuals
            residuals = y - current_predictions
            
            # Subsample
            if self.subsample < 1.0:
                mask = np.random.rand(len(y)) < self.subsample
                X_sub, residuals_sub = X[mask], residuals[mask]
            else:
                X_sub, residuals_sub = X, residuals
            
            # Fit tree to residuals
            tree = self._build_tree(X_sub, residuals_sub)
            self.estimators.append(tree)
            
            # Update predictions
            for j in range(len(y)):
                current_predictions[j] += self.learning_rate * self._predict_tree(tree, X[j])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate boosted predictions"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        predictions = np.full(len(X), self.initial_prediction)
        
        for tree in self.estimators:
            for i in range(len(X)):
                predictions[i] += self.learning_rate * self._predict_tree(tree, X[i])
        
        return predictions


class BaggingEnsemble:
    """
    Bagging (Bootstrap Aggregating) ensemble.
    
    Reduces variance by training on different bootstrap samples.
    Effective for unstable models that are sensitive to data variations.
    """
    
    def __init__(
        self,
        base_model_factory: Callable,
        n_estimators: int = 10,
        max_samples: float = 0.8,
        max_features: float = 1.0,
        bootstrap: bool = True
    ):
        """
        Initialize bagging ensemble.
        
        Args:
            base_model_factory: Function that creates a new base model
            n_estimators: Number of base models
            max_samples: Fraction of samples per model
            max_features: Fraction of features per model
            bootstrap: Whether to sample with replacement
        """
        self.base_model_factory = base_model_factory
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        
        self.estimators: List[Tuple[Any, np.ndarray]] = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the bagging ensemble"""
        n_samples, n_features = X.shape
        sample_size = int(n_samples * self.max_samples)
        feature_size = int(n_features * self.max_features)
        
        self.estimators = []
        
        for i in range(self.n_estimators):
            # Sample indices
            if self.bootstrap:
                sample_idx = np.random.choice(n_samples, sample_size, replace=True)
            else:
                sample_idx = np.random.choice(n_samples, sample_size, replace=False)
            
            # Feature indices
            feature_idx = np.random.choice(n_features, feature_size, replace=False)
            feature_idx = np.sort(feature_idx)
            
            # Create and fit model
            model = self.base_model_factory()
            X_sub = X[sample_idx][:, feature_idx]
            y_sub = y[sample_idx]
            
            if hasattr(model, 'fit'):
                model.fit(X_sub, y_sub)
            
            self.estimators.append((model, feature_idx))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate bagged predictions"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        all_predictions = []
        
        for model, feature_idx in self.estimators:
            X_sub = X[:, feature_idx]
            if hasattr(model, 'predict'):
                pred = model.predict(X_sub)
            else:
                pred = np.array([model(x) for x in X_sub])
            all_predictions.append(pred.flatten())
        
        # Average predictions
        return np.mean(all_predictions, axis=0)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimate from ensemble variance"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        all_predictions = []
        
        for model, feature_idx in self.estimators:
            X_sub = X[:, feature_idx]
            if hasattr(model, 'predict'):
                pred = model.predict(X_sub)
            else:
                pred = np.array([model(x) for x in X_sub])
            all_predictions.append(pred.flatten())
        
        predictions = np.array(all_predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        return mean, std
