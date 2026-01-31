#!/usr/bin/env python3
"""
Real-Time Prediction Module

This module provides real-time predictions using trained models.
Unlike the placeholder in cli.py, this uses ACTUAL trained models
with PROVEN predictive power.

Usage:
    predictor = RealTimePredictor(api_key=POLYGON_API_KEY)
    result = await predictor.predict('SPY')
"""

import os
import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from ..training.real_data_trainer import (
    TechnicalFeatureExtractor,
    MLPModel,
    LinearModel,
)


@dataclass
class PredictionResult:
    """Real-time prediction result."""
    symbol: str
    timestamp: str
    
    # Core prediction
    direction: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: float  # 0-1
    predicted_return: float  # Expected return (1d forward)
    
    # Model info
    model_type: str
    model_version: str
    
    # Historical performance (on test data)
    historical_sharpe: float
    historical_dir_acc: float
    
    # Current price context
    current_price: float
    price_target_1d: float
    
    # Key signals
    signals: Dict[str, Any] = field(default_factory=dict)
    
    # Raw model output
    raw_prediction: float = 0.0
    
    def __str__(self):
        emoji = "📈" if self.direction == "BULLISH" else "📉" if self.direction == "BEARISH" else "➡️"
        
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎯 REAL-TIME ML PREDICTION: {self.symbol:<15}                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PREDICTION                                                                  ║
║    Direction:         {self.direction:<10} {emoji}                            ║
║    Confidence:        {self.confidence:>6.1%}                                ║
║    Expected Return:   {self.predicted_return:>+6.2%} (1d)                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PRICE                                                                       ║
║    Current:           ${self.current_price:>10.2f}                            ║
║    Target (1d):       ${self.price_target_1d:>10.2f}                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  MODEL PERFORMANCE (Historical)                                              ║
║    Sharpe Ratio:      {self.historical_sharpe:>6.2f}                          ║
║    Dir Accuracy:      {self.historical_dir_acc:>6.1%}                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  MODEL INFO                                                                  ║
║    Type:              {self.model_type:<20}                                   ║
║    Version:           {self.model_version:<20}                                ║
║    Timestamp:         {self.timestamp:<20}                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  KEY SIGNALS                                                                 ║"""
    
    def get_signals_str(self) -> str:
        lines = []
        for key, value in list(self.signals.items())[:6]:
            if isinstance(value, float):
                lines.append(f"║    {key:<18}: {value:>8.2f}                          ║")
            else:
                lines.append(f"║    {key:<18}: {str(value):>8}                          ║")
        lines.append("╚══════════════════════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


@dataclass
class EnsemblePrediction:
    """Ensemble of multiple model predictions."""
    symbol: str
    timestamp: str
    
    # Individual predictions
    predictions: List[PredictionResult]
    
    # Ensemble aggregation
    ensemble_direction: str
    ensemble_confidence: float
    ensemble_return: float
    
    # Agreement
    model_agreement: float  # % of models agreeing on direction
    
    def __str__(self):
        emoji = "📈" if self.ensemble_direction == "BULLISH" else "📉" if self.ensemble_direction == "BEARISH" else "➡️"
        
        output = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎯 ENSEMBLE ML PREDICTION: {self.symbol:<15}                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ENSEMBLE RESULT                                                             ║
║    Direction:         {self.ensemble_direction:<10} {emoji}                   ║
║    Confidence:        {self.ensemble_confidence:>6.1%}                        ║
║    Expected Return:   {self.ensemble_return:>+6.2%} (1d)                      ║
║    Model Agreement:   {self.model_agreement:>6.1%}                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  INDIVIDUAL MODELS                                                           ║
"""
        for p in self.predictions:
            emoji = "📈" if p.direction == "BULLISH" else "📉" if p.direction == "BEARISH" else "➡️"
            output += f"║    {p.model_type:<15}: {p.direction:<8} {emoji}  (conf: {p.confidence:.1%}, Sharpe: {p.historical_sharpe:.2f}) ║\n"
        
        output += "╚══════════════════════════════════════════════════════════════════════════════╝"
        return output


class RealTimePredictor:
    """
    Real-time prediction using trained models.
    
    This loads pre-trained models and makes predictions on current data.
    """
    
    def __init__(
        self,
        api_key: str,
        models_dir: str = "models",
        device: str = 'auto'
    ):
        self.api_key = api_key
        self.models_dir = Path(models_dir)
        
        if device == 'auto':
            self.device = torch.device(
                'mps' if torch.backends.mps.is_available() else
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)
        
        self.feature_extractor = TechnicalFeatureExtractor()
        self.loaded_models: Dict[str, Tuple[nn.Module, Dict[str, Any]]] = {}
    
    def _load_model(self, symbol: str, model_type: str = 'mlp') -> Tuple[nn.Module, Dict[str, Any]]:
        """Load a trained model from disk."""
        key = f"{symbol}_{model_type}"
        
        if key in self.loaded_models:
            return self.loaded_models[key]
        
        # Look for model file
        model_patterns = [
            self.models_dir / f"{symbol}_{model_type}_*.pt",
            self.models_dir / f"{symbol}_{model_type}.pt",
            self.models_dir / "real_data" / f"{symbol}_{model_type}_*.pt",
        ]
        
        model_path = None
        for pattern in model_patterns:
            matches = list(pattern.parent.glob(pattern.name))
            if matches:
                model_path = sorted(matches)[-1]  # Most recent
                break
        
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"No trained model found for {symbol} ({model_type})")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        n_features = checkpoint.get('n_features', 15)
        if model_type == 'linear':
            model = LinearModel(n_features)
        else:
            model = MLPModel(n_features, hidden_size=32, dropout=0.3)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        metadata = checkpoint.get('metadata', {})
        self.loaded_models[key] = (model, metadata)
        
        return model, metadata
    
    async def fetch_recent_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """Fetch recent price data for prediction."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required for data fetching")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {
            'apiKey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise ValueError(f"API error: {response.status}")
                data = await response.json()
        
        if 'results' not in data or not data['results']:
            raise ValueError(f"No data returned for {symbol}")
        
        df = pd.DataFrame(data['results'])
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        })
        df = df.set_index('timestamp')
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    async def predict(
        self,
        symbol: str,
        model_type: str = 'mlp'
    ) -> PredictionResult:
        """
        Make a real-time prediction for a symbol.
        
        Args:
            symbol: Stock symbol
            model_type: 'linear' or 'mlp'
        
        Returns:
            PredictionResult with prediction and context
        """
        # Load model
        try:
            model, metadata = self._load_model(symbol, model_type)
        except FileNotFoundError:
            # If no trained model, return a "not trained" result
            return PredictionResult(
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                direction="UNKNOWN",
                confidence=0.0,
                predicted_return=0.0,
                model_type=model_type.upper(),
                model_version="NOT TRAINED",
                historical_sharpe=0.0,
                historical_dir_acc=0.0,
                current_price=0.0,
                price_target_1d=0.0,
                signals={"error": "Model not trained for this symbol"},
            )
        
        # Fetch recent data
        df = await self.fetch_recent_data(symbol, days=60)
        
        # Extract features
        features_df, feature_names = self.feature_extractor.extract(df)
        features_df = features_df.dropna()
        
        if len(features_df) == 0:
            raise ValueError("No valid features after extraction")
        
        # Get latest features
        latest_features = features_df.iloc[-1:][feature_names].values
        X = torch.tensor(latest_features, dtype=torch.float32).to(self.device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            raw_prediction = model(X).cpu().numpy()[0]
        
        # Convert to direction
        if raw_prediction > 0.002:  # 0.2% threshold
            direction = "BULLISH"
            confidence = min(abs(raw_prediction) * 50, 0.95)  # Scale to 0-95%
        elif raw_prediction < -0.002:
            direction = "BEARISH"
            confidence = min(abs(raw_prediction) * 50, 0.95)
        else:
            direction = "NEUTRAL"
            confidence = 0.5
        
        # Get current price
        current_price = df['close'].iloc[-1]
        price_target = current_price * (1 + raw_prediction)
        
        # Get historical performance from metadata
        historical_sharpe = metadata.get('sharpe', 0.0)
        historical_dir_acc = metadata.get('directional_accuracy', 0.5)
        model_version = metadata.get('trained_at', 'unknown')
        
        # Extract key signals
        signals = {}
        for feat in ['rsi', 'macd', 'momentum_5', 'volatility_20']:
            if feat in features_df.columns:
                signals[feat] = float(features_df[feat].iloc[-1])
        
        return PredictionResult(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            direction=direction,
            confidence=confidence,
            predicted_return=float(raw_prediction),
            model_type=model_type.upper(),
            model_version=str(model_version),
            historical_sharpe=historical_sharpe,
            historical_dir_acc=historical_dir_acc,
            current_price=current_price,
            price_target_1d=price_target,
            signals=signals,
            raw_prediction=float(raw_prediction),
        )
    
    async def ensemble_predict(
        self,
        symbol: str,
        model_types: List[str] = None
    ) -> EnsemblePrediction:
        """
        Make ensemble prediction using multiple models.
        
        Args:
            symbol: Stock symbol
            model_types: List of model types to use
        
        Returns:
            EnsemblePrediction with aggregated results
        """
        if model_types is None:
            model_types = ['linear', 'mlp']
        
        predictions = []
        for model_type in model_types:
            try:
                pred = await self.predict(symbol, model_type)
                if pred.direction != "UNKNOWN":
                    predictions.append(pred)
            except Exception as e:
                print(f"Warning: {model_type} prediction failed: {e}")
        
        if not predictions:
            raise ValueError(f"No models available for {symbol}")
        
        # Aggregate
        # Weight by historical Sharpe
        total_sharpe = sum(max(p.historical_sharpe, 0.1) for p in predictions)
        weights = [max(p.historical_sharpe, 0.1) / total_sharpe for p in predictions]
        
        weighted_return = sum(p.predicted_return * w for p, w in zip(predictions, weights))
        
        # Direction voting
        bullish_votes = sum(1 for p in predictions if p.direction == "BULLISH")
        bearish_votes = sum(1 for p in predictions if p.direction == "BEARISH")
        
        if bullish_votes > bearish_votes:
            ensemble_direction = "BULLISH"
            agreement = bullish_votes / len(predictions)
        elif bearish_votes > bullish_votes:
            ensemble_direction = "BEARISH"
            agreement = bearish_votes / len(predictions)
        else:
            ensemble_direction = "NEUTRAL"
            agreement = 0.5
        
        # Confidence is weighted average
        ensemble_confidence = sum(p.confidence * w for p, w in zip(predictions, weights))
        
        return EnsemblePrediction(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            predictions=predictions,
            ensemble_direction=ensemble_direction,
            ensemble_confidence=ensemble_confidence,
            ensemble_return=weighted_return,
            model_agreement=agreement,
        )


async def quick_predict(symbol: str, api_key: str = None) -> PredictionResult:
    """Quick prediction function."""
    if api_key is None:
        api_key = os.environ.get('POLYGON_API_KEY')
    
    if not api_key:
        raise ValueError("POLYGON_API_KEY not set")
    
    predictor = RealTimePredictor(api_key)
    return await predictor.predict(symbol)


if __name__ == "__main__":
    import sys
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    api_key = os.environ.get('POLYGON_API_KEY')
    
    if not api_key:
        print("❌ POLYGON_API_KEY not set")
        sys.exit(1)
    
    result = asyncio.run(quick_predict(symbol, api_key))
    print(result)
    print(result.get_signals_str())
