"""
ML Integration Layer for KYPERIAN
==================================

Connects PyTorch ML models to the main KYPERIAN system:
- Model management and inference
- Real-time prediction pipeline
- Integration with orchestrator
- Caching and optimization
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

import torch

from .torch_models import (
    ModelConfig,
    FinancialLSTM,
    AttentionLSTM,
    MarketTransformer,
    TemporalFusionTransformer,
    EnsembleNetwork,
    NeuralRegimeClassifier,
    TechnicalFeatureExtractor,
    FeatureConfig,
    PolygonDataFetcher,
    get_device
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionOutput:
    """Standardized prediction output."""
    symbol: str
    timestamp: datetime
    
    # Price predictions
    predictions: Dict[str, Dict[str, float]]  # horizon -> {mean, lower, upper}
    
    # Direction
    direction: str  # 'bullish', 'bearish', 'neutral'
    direction_confidence: float
    
    # Regime
    current_regime: str
    regime_confidence: float
    
    # Uncertainty
    uncertainty: float
    
    # Metadata
    model_used: str
    feature_count: int
    sequence_length: int


class MLPredictor:
    """
    Production ML predictor for KYPERIAN.
    
    Manages multiple models and provides unified prediction interface.
    """
    
    DEFAULT_MODEL_DIR = Path.home() / '.kyperian' / 'models'
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = 'auto',
        enable_cache: bool = True
    ):
        """
        Initialize ML predictor.
        
        Args:
            model_dir: Directory containing trained models
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            enable_cache: Whether to cache predictions
        """
        self.model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = get_device(device)
        self.enable_cache = enable_cache
        
        # Model registry
        self.models: Dict[str, torch.nn.Module] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        
        # Feature extractor
        self.feature_extractor = TechnicalFeatureExtractor()
        
        # Data fetcher
        self._data_fetcher = None
        
        # Prediction cache
        self._cache: Dict[str, Tuple[datetime, PredictionOutput]] = {}
        self._cache_ttl = timedelta(minutes=5)
        
        # Load available models
        self._load_models()
    
    @property
    def data_fetcher(self) -> PolygonDataFetcher:
        """Lazy-load data fetcher."""
        if self._data_fetcher is None:
            api_key = os.getenv('POLYGON_API_KEY')
            if api_key:
                self._data_fetcher = PolygonDataFetcher(api_key)
            else:
                raise ValueError("POLYGON_API_KEY not set")
        return self._data_fetcher
    
    def _load_models(self):
        """Load all available trained models."""
        model_files = list(self.model_dir.glob('*.pt'))
        
        for model_file in model_files:
            try:
                self._load_single_model(model_file)
                logger.info(f"Loaded model: {model_file.stem}")
            except Exception as e:
                logger.warning(f"Failed to load {model_file}: {e}")
    
    def _load_single_model(self, path: Path):
        """Load a single model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        config_dict = checkpoint.get('model_config', checkpoint.get('config', {}))
        config = ModelConfig.from_dict(config_dict)
        
        # Determine model type from filename or checkpoint
        model_name = path.stem
        model_type = checkpoint.get('model_type', model_name.split('_')[0])
        
        # Create model instance
        if 'lstm' in model_type.lower():
            model = FinancialLSTM(config)
        elif 'transformer' in model_type.lower():
            model = MarketTransformer(config)
        elif 'ensemble' in model_type.lower():
            model = EnsembleNetwork(config)
        elif 'regime' in model_type.lower():
            model = NeuralRegimeClassifier(config)
        else:
            model = FinancialLSTM(config)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.models[model_name] = model
        self.model_configs[model_name] = config
    
    def create_default_models(self):
        """Create default model configurations (untrained)."""
        default_config = ModelConfig(
            input_size=64,
            hidden_size=256,
            num_layers=3,
            num_heads=8,
            dropout=0.2,
            sequence_length=60,
            prediction_horizons=[1, 5, 10, 20]
        )
        
        # LSTM model
        lstm = FinancialLSTM(default_config)
        lstm.to(self.device)
        self.models['lstm'] = lstm
        self.model_configs['lstm'] = default_config
        
        # Transformer model
        transformer = MarketTransformer(default_config)
        transformer.to(self.device)
        self.models['transformer'] = transformer
        self.model_configs['transformer'] = default_config
        
        # Ensemble model
        ensemble = EnsembleNetwork(default_config)
        ensemble.to(self.device)
        self.models['ensemble'] = ensemble
        self.model_configs['ensemble'] = default_config
        
        # Regime classifier
        regime_config = ModelConfig(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        regime = NeuralRegimeClassifier(regime_config, num_regimes=6)
        regime.to(self.device)
        self.models['regime'] = regime
        self.model_configs['regime'] = regime_config
        
        logger.info("Created default models (untrained)")
    
    def predict(
        self,
        symbol: str,
        model_name: str = 'ensemble',
        use_cache: bool = True
    ) -> PredictionOutput:
        """
        Generate predictions for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            model_name: Which model to use
            use_cache: Whether to use cached predictions
            
        Returns:
            PredictionOutput with all predictions
        """
        # Check cache
        cache_key = f"{symbol}_{model_name}"
        if use_cache and self.enable_cache and cache_key in self._cache:
            cached_time, cached_output = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                logger.debug(f"Returning cached prediction for {symbol}")
                return cached_output
        
        # Get model
        if model_name not in self.models:
            if not self.models:
                self.create_default_models()
            model_name = list(self.models.keys())[0]
        
        model = self.models[model_name]
        config = self.model_configs[model_name]
        
        # Fetch recent data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=config.sequence_length + 50)
        
        try:
            bars = self.data_fetcher.fetch_historical(
                symbol, start_date, end_date, '1d', use_cache=True
            )
            ohlcv = self.data_fetcher.bars_to_array(bars)
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise
        
        # Extract features
        features, feature_names = self.feature_extractor.extract(ohlcv)
        
        # Prepare input tensor
        seq_len = min(config.sequence_length, len(features))
        x = torch.from_numpy(features[-seq_len:]).float().unsqueeze(0)
        x = x.to(self.device)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        # Parse predictions
        predictions = {}
        if 'predictions' in output:
            for horizon_key, pred_dict in output['predictions'].items():
                horizon = horizon_key.replace('h', '') if horizon_key.startswith('h') else horizon_key
                predictions[f'{horizon}d'] = {
                    'mean': float(pred_dict['mean'].cpu().numpy()),
                    'lower': float(pred_dict.get('lower', pred_dict.get('q10', pred_dict['mean'] - 0.02)).cpu().numpy()),
                    'upper': float(pred_dict.get('upper', pred_dict.get('q90', pred_dict['mean'] + 0.02)).cpu().numpy())
                }
        
        # Parse direction
        direction_probs = output.get('direction', {}).get('probs', torch.tensor([0.33, 0.33, 0.34]))
        if isinstance(direction_probs, torch.Tensor):
            direction_probs = direction_probs.cpu().numpy().flatten()
        
        direction_idx = np.argmax(direction_probs)
        direction_map = {0: 'bullish', 1: 'bearish', 2: 'neutral'}
        direction = direction_map.get(direction_idx, 'neutral')
        direction_confidence = float(direction_probs[direction_idx])
        
        # Get regime if available
        current_regime = 'unknown'
        regime_confidence = 0.5
        
        if 'regime' in self.models:
            regime_model = self.models['regime']
            with torch.no_grad():
                regime_output = regime_model(x)
            current_regime = regime_output['current_regime']['name'][0]
            regime_confidence = float(regime_output['current_regime']['confidence'].cpu().numpy())
        elif 'volatility_regime' in output:
            vol_probs = output['volatility_regime']['probs'].cpu().numpy().flatten()
            vol_idx = np.argmax(vol_probs)
            regime_map = {0: 'low_volatility', 1: 'normal', 2: 'high_volatility'}
            current_regime = regime_map.get(vol_idx, 'normal')
            regime_confidence = float(vol_probs[vol_idx])
        
        # Get uncertainty
        uncertainty = float(output.get('uncertainty', torch.tensor(0.1)).cpu().numpy())
        if 'predictions' in output and predictions:
            first_horizon = list(predictions.keys())[0]
            uncertainty = (predictions[first_horizon]['upper'] - predictions[first_horizon]['lower']) / 2
        
        # Build output
        result = PredictionOutput(
            symbol=symbol,
            timestamp=datetime.now(),
            predictions=predictions,
            direction=direction,
            direction_confidence=direction_confidence,
            current_regime=current_regime,
            regime_confidence=regime_confidence,
            uncertainty=uncertainty,
            model_used=model_name,
            feature_count=len(feature_names),
            sequence_length=seq_len
        )
        
        # Cache
        if self.enable_cache:
            self._cache[cache_key] = (datetime.now(), result)
        
        return result
    
    def predict_multiple(
        self,
        symbols: List[str],
        model_name: str = 'ensemble'
    ) -> Dict[str, PredictionOutput]:
        """Predict for multiple symbols."""
        return {
            symbol: self.predict(symbol, model_name)
            for symbol in symbols
        }
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded models."""
        info = {}
        for name, model in self.models.items():
            config = self.model_configs.get(name, ModelConfig())
            info[name] = {
                'type': model.__class__.__name__,
                'parameters': model.count_parameters() if hasattr(model, 'count_parameters') else sum(p.numel() for p in model.parameters()),
                'device': str(next(model.parameters()).device),
                'config': config.to_dict() if hasattr(config, 'to_dict') else str(config)
            }
        return info


class MLIntegration:
    """
    Integration layer for connecting ML to KYPERIAN orchestrator.
    """
    
    def __init__(self, predictor: Optional[MLPredictor] = None):
        """Initialize integration."""
        self.predictor = predictor or MLPredictor()
    
    def enhance_query_response(
        self,
        symbol: str,
        base_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance a query response with ML predictions.
        
        Args:
            symbol: Stock ticker
            base_response: Base response from data providers
            
        Returns:
            Enhanced response with ML predictions
        """
        try:
            prediction = self.predictor.predict(symbol)
            
            base_response['ml_predictions'] = {
                'price_forecasts': prediction.predictions,
                'direction': {
                    'prediction': prediction.direction,
                    'confidence': prediction.direction_confidence
                },
                'regime': {
                    'current': prediction.current_regime,
                    'confidence': prediction.regime_confidence
                },
                'uncertainty': prediction.uncertainty,
                'model': prediction.model_used,
                'timestamp': prediction.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.warning(f"ML prediction failed for {symbol}: {e}")
            base_response['ml_predictions'] = {
                'error': str(e),
                'available': False
            }
        
        return base_response
    
    def format_prediction_for_display(
        self,
        prediction: PredictionOutput
    ) -> str:
        """Format prediction for CLI display."""
        lines = [
            f"\n📊 **ML Analysis for {prediction.symbol}**",
            f"",
            f"**Direction Forecast:** {prediction.direction.upper()} "
            f"(confidence: {prediction.direction_confidence:.1%})",
            f"",
            f"**Price Predictions:**"
        ]
        
        for horizon, values in prediction.predictions.items():
            pct_change = values['mean'] * 100
            lines.append(
                f"  • {horizon}: {pct_change:+.2f}% "
                f"[{values['lower']*100:+.2f}% to {values['upper']*100:+.2f}%]"
            )
        
        lines.extend([
            f"",
            f"**Market Regime:** {prediction.current_regime} "
            f"(confidence: {prediction.regime_confidence:.1%})",
            f"",
            f"**Uncertainty:** {prediction.uncertainty:.4f}",
            f"_Model: {prediction.model_used}_"
        ])
        
        return "\n".join(lines)


# Global instance for easy access
_predictor_instance: Optional[MLPredictor] = None


def get_predictor() -> MLPredictor:
    """Get or create global predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = MLPredictor()
    return _predictor_instance


def predict(symbol: str, model: str = 'ensemble') -> PredictionOutput:
    """Convenience function for quick predictions."""
    return get_predictor().predict(symbol, model)
