"""
Pre-Trained Model Registry
===========================

Manages loading, saving, and serving pre-trained models.
Provides instant predictions without needing to train.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a pre-trained model."""
    symbol: str
    model_type: str
    timeframe: str
    trained_at: str
    training_samples: int
    features: List[str]
    
    # Performance metrics (from validation)
    validation_sharpe: float
    validation_return: float
    directional_accuracy: float
    p_value: float
    
    # Walk-forward metrics
    wf_sharpe: Optional[float] = None
    wf_return: Optional[float] = None
    wf_periods: Optional[int] = None
    
    # Model config
    hidden_dims: List[int] = None
    learning_rate: float = 0.001
    epochs_trained: int = 100
    
    # Validation grade
    grade: str = "F"
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]
        self._compute_grade()
    
    def _compute_grade(self):
        """
        Institutional-grade strategy assessment.
        
        Based on:
        - Hedge fund benchmark: Sharpe > 1.0 is "good"
        - Statistical significance: p < 0.05 required for any passing grade
        - Directional edge: >52% needed to overcome transaction costs
        """
        sharpe = self.wf_sharpe if self.wf_sharpe else self.validation_sharpe
        p_value = self.p_value
        
        # Must be statistically significant to pass
        if p_value > 0.10:
            self.grade = "F"  # Not significant, could be random
            self.confidence = 0.30
            return
        
        if p_value > 0.05:
            # Marginal significance - cap at C
            if sharpe >= 0.5:
                self.grade = "C"
                self.confidence = 0.55
            else:
                self.grade = "D"
                self.confidence = 0.40
            return
        
        # Statistically significant (p < 0.05)
        # Grade based on Sharpe
        if sharpe >= 2.5:
            self.grade = "A+"  # Exceptional - Renaissance/Citadel territory
            self.confidence = 0.98
        elif sharpe >= 2.0:
            self.grade = "A"   # Excellent - Top-tier hedge fund
            self.confidence = 0.95
        elif sharpe >= 1.5:
            self.grade = "A-"  # Very good - Strong institutional quality
            self.confidence = 0.90
        elif sharpe >= 1.0:
            self.grade = "B+"  # Good - Solid hedge fund strategy
            self.confidence = 0.85
        elif sharpe >= 0.75:
            self.grade = "B"   # Above average - Tradeable with size limits
            self.confidence = 0.80
        elif sharpe >= 0.5:
            self.grade = "B-"  # Decent - Worth trading small
            self.confidence = 0.70
        elif sharpe >= 0.25:
            self.grade = "C+"  # Marginal - Needs improvement
            self.confidence = 0.60
        elif sharpe > 0:
            self.grade = "C"   # Weak positive - Research further
            self.confidence = 0.50
        else:
            self.grade = "D"   # Negative Sharpe - Don't trade
            self.confidence = 0.35


class PreTrainedModelRegistry:
    """
    Registry for pre-trained models.
    
    Provides:
    - Auto-loading of models on startup
    - Model versioning and metadata
    - Performance tracking
    - Easy model addition/removal
    """
    
    # Default models directory
    MODELS_DIR = Path(__file__).parent.parent / "models" / "pretrained"
    
    # Validated alpha sources from walk-forward testing
    VALIDATED_SYMBOLS = {
        'SLV': {'sharpe': 0.94, 'timeframe': '1d'},
        'TSLA': {'sharpe': 0.91, 'timeframe': '1d'},
        'AMD': {'sharpe': 0.76, 'timeframe': '1d'},
        'XLK': {'sharpe': 0.92, 'timeframe': '1d'},
    }
    
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir) if models_dir else self.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self._loaded_models: Dict[str, Any] = {}
        self._model_metadata: Dict[str, ModelMetadata] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load registry index from disk."""
        registry_file = self.models_dir / "registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    for symbol, meta_dict in data.get('models', {}).items():
                        self._model_metadata[symbol] = ModelMetadata(**meta_dict)
                logger.info(f"Loaded registry with {len(self._model_metadata)} models")
            except Exception as e:
                logger.warning(f"Could not load registry: {e}")
    
    def _save_registry(self):
        """Save registry index to disk."""
        registry_file = self.models_dir / "registry.json"
        
        data = {
            'version': '2.0.0',
            'updated_at': datetime.now().isoformat(),
            'models': {
                symbol: asdict(meta) 
                for symbol, meta in self._model_metadata.items()
            }
        }
        
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    @property
    def available_models(self) -> List[str]:
        """Get list of available pre-trained models."""
        return list(self._model_metadata.keys())
    
    def get_metadata(self, symbol: str) -> Optional[ModelMetadata]:
        """Get metadata for a model."""
        return self._model_metadata.get(symbol.upper())
    
    def is_available(self, symbol: str) -> bool:
        """Check if a pre-trained model exists for symbol."""
        symbol = symbol.upper()
        
        # Check if model file exists
        model_file = self.models_dir / f"{symbol}_model.pt"
        return model_file.exists() and symbol in self._model_metadata
    
    def load_model(self, symbol: str) -> Optional[Any]:
        """
        Load a pre-trained model.
        
        Returns the PyTorch model or None if not available.
        """
        symbol = symbol.upper()
        
        # Check cache first
        if symbol in self._loaded_models:
            return self._loaded_models[symbol]
        
        model_file = self.models_dir / f"{symbol}_model.pt"
        if not model_file.exists():
            logger.warning(f"No pre-trained model for {symbol}")
            return None
        
        try:
            import torch
            
            # Load model state
            state = torch.load(model_file, map_location='cpu', weights_only=False)
            
            # Reconstruct model architecture
            meta = self._model_metadata.get(symbol)
            if meta:
                from .torch_models.ensemble_network import MLPSubModel
                
                input_dim = state.get('input_dim', len(meta.features))
                hidden_size = meta.hidden_dims[0] if meta.hidden_dims else 256
                
                model = MLPSubModel(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    output_size=1,
                    num_layers=len(meta.hidden_dims) if meta.hidden_dims else 3
                )
                model.load_state_dict(state['model_state'])
                model.eval()
                
                self._loaded_models[symbol] = {
                    'model': model,
                    'metadata': meta,
                    'scaler': state.get('scaler'),
                    'feature_names': state.get('feature_names', meta.features)
                }
                
                logger.info(f"Loaded pre-trained model for {symbol} (Grade: {meta.grade})")
                return self._loaded_models[symbol]
            
        except Exception as e:
            logger.error(f"Failed to load model for {symbol}: {e}")
            return None
    
    def save_model(
        self,
        symbol: str,
        model: Any,
        metadata: ModelMetadata,
        scaler: Any = None,
        feature_names: List[str] = None
    ) -> bool:
        """
        Save a trained model to the registry.
        
        Args:
            symbol: Stock symbol
            model: PyTorch model
            metadata: Model metadata with metrics
            scaler: Feature scaler (if any)
            feature_names: List of feature names
            
        Returns:
            True if saved successfully
        """
        symbol = symbol.upper()
        model_file = self.models_dir / f"{symbol}_model.pt"
        
        try:
            import torch
            
            state = {
                'model_state': model.state_dict(),
                'input_dim': model.input_dim if hasattr(model, 'input_dim') else None,
                'scaler': scaler,
                'feature_names': feature_names or metadata.features,
                'saved_at': datetime.now().isoformat()
            }
            
            torch.save(state, model_file)
            
            # Update registry
            self._model_metadata[symbol] = metadata
            self._save_registry()
            
            logger.info(f"Saved pre-trained model for {symbol} (Grade: {metadata.grade})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model for {symbol}: {e}")
            return False
    
    def predict(self, symbol: str, features: Any) -> Optional[Dict[str, Any]]:
        """
        Make prediction using pre-trained model.
        
        Args:
            symbol: Stock symbol
            features: Input features (numpy array or tensor)
            
        Returns:
            Prediction dict with direction, confidence, expected_return
        """
        symbol = symbol.upper()
        
        loaded = self.load_model(symbol)
        if not loaded:
            return None
        
        try:
            import torch
            import numpy as np
            
            model = loaded['model']
            scaler = loaded.get('scaler')
            metadata = loaded['metadata']
            
            # Prepare features
            if isinstance(features, np.ndarray):
                if scaler:
                    features = scaler.transform(features.reshape(1, -1))
                features = torch.tensor(features, dtype=torch.float32)
            
            # Make prediction
            model.eval()
            with torch.no_grad():
                output = model(features)
                prediction = output.item()
            
            # Interpret prediction
            direction = 'up' if prediction > 0 else 'down'
            confidence = min(abs(prediction) * 0.5 + 0.5, 0.95)  # Scale to confidence
            
            return {
                'symbol': symbol,
                'direction': direction,
                'confidence': confidence,
                'expected_return': prediction,
                'model_type': metadata.model_type,
                'model_grade': metadata.grade,
                'sharpe_ratio': metadata.wf_sharpe or metadata.validation_sharpe,
                'p_value': metadata.p_value
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return None
    
    def get_best_models(self, min_grade: str = "C") -> List[Tuple[str, ModelMetadata]]:
        """Get models meeting minimum grade threshold."""
        grade_order = ['A+', 'A', 'B', 'C', 'D', 'F']
        min_idx = grade_order.index(min_grade)
        
        results = []
        for symbol, meta in self._model_metadata.items():
            if grade_order.index(meta.grade) <= min_idx:
                results.append((symbol, meta))
        
        # Sort by grade
        results.sort(key=lambda x: grade_order.index(x[1].grade))
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get registry status."""
        models = self.available_models
        
        grade_counts = {}
        for meta in self._model_metadata.values():
            grade_counts[meta.grade] = grade_counts.get(meta.grade, 0) + 1
        
        return {
            'total_models': len(models),
            'models': models,
            'grade_distribution': grade_counts,
            'validated_symbols': list(self.VALIDATED_SYMBOLS.keys()),
            'models_dir': str(self.models_dir)
        }


# Global registry instance
_registry: Optional[PreTrainedModelRegistry] = None


def get_registry() -> PreTrainedModelRegistry:
    """Get the global model registry."""
    global _registry
    if _registry is None:
        _registry = PreTrainedModelRegistry()
    return _registry
