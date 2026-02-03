#!/usr/bin/env python3
"""
Real Data Training Module for NUBLE

This module trains and validates models on REAL market data from Polygon.io.

Key differences from synthetic data:
1. Real markets have ~5% signal (vs 30% in synthetic)
2. Expected Sharpe: 0.5-1.5 (vs 3+ in synthetic)
3. Non-stationary: patterns change over time
4. Noise is correlated, fat-tailed, regime-dependent

Usage:
    trainer = RealDataTrainer(api_key=POLYGON_API_KEY)
    result = await trainer.train_and_validate('SPY')
"""

import os
import sys
import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

# Import scipy if available
try:
    from scipy.stats import spearmanr, ttest_1samp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Import aiohttp for async requests
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrainingResult:
    """Result of training and validation on real data."""
    symbol: str
    model_name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    
    # Training metrics
    train_samples: int
    val_samples: int
    test_samples: int
    epochs_trained: int
    final_train_loss: float
    final_val_loss: float
    
    # Test metrics
    directional_accuracy: float
    correlation: float
    ic_mean: float
    ic_ir: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    total_return: float
    annual_return: float
    
    # Statistical significance
    t_statistic: float
    p_value: float
    is_significant: bool
    
    # Model info
    model_params: int
    model_path: Optional[str] = None
    
    # Feature importance (if available)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    def __str__(self):
        grade = self._get_grade()
        status = "✅ SIGNIFICANT" if self.is_significant else "❌ NOT SIGNIFICANT"
        
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  REAL DATA TRAINING RESULT                                                   ║
║  Symbol: {self.symbol:<10}  Model: {self.model_name:<20}                    ║
║  Grade: {grade}                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DATA                                                                        ║
║    Train: {self.train_start} to {self.train_end} ({self.train_samples:,} samples)
║    Test:  {self.test_start} to {self.test_end} ({self.test_samples:,} samples)
╠══════════════════════════════════════════════════════════════════════════════╣
║  TRAINING                                                                    ║
║    Epochs:         {self.epochs_trained:>6}                                  ║
║    Final Loss:     {self.final_val_loss:>10.6f}                              ║
║    Parameters:     {self.model_params:>10,}                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PREDICTION QUALITY                                                          ║
║    Directional Accuracy: {self.directional_accuracy:>6.2%}  (random = 50%)   ║
║    Correlation:          {self.correlation:>6.3f}                            ║
║    IC Mean:              {self.ic_mean:>6.4f}                                ║
║    IC IR:                {self.ic_ir:>6.2f}                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TRADING PERFORMANCE                                                         ║
║    Sharpe Ratio:         {self.sharpe_ratio:>6.2f}  (target: >0.5)           ║
║    Sortino Ratio:        {self.sortino_ratio:>6.2f}                          ║
║    Max Drawdown:         {self.max_drawdown:>6.2%}                           ║
║    Total Return:         {self.total_return:>6.2%}                           ║
║    Annual Return:        {self.annual_return:>6.2%}                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  STATISTICAL SIGNIFICANCE                                                    ║
║    t-statistic:          {self.t_statistic:>6.2f}                            ║
║    p-value:              {self.p_value:>6.4f}                                ║
║    Status:               {status:>24}                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    
    def _get_grade(self) -> str:
        # Adjusted for REAL data expectations (lower bar than synthetic)
        if self.sharpe_ratio > 1.5 and self.is_significant:
            return "A"
        elif self.sharpe_ratio > 0.8 and self.is_significant:
            return "B"
        elif self.sharpe_ratio > 0.5 and self.directional_accuracy > 0.52:
            return "C"
        elif self.sharpe_ratio > 0 and self.directional_accuracy > 0.50:
            return "D"
        else:
            return "F"
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'model_name': self.model_name,
            'train_start': self.train_start,
            'train_end': self.train_end,
            'test_start': self.test_start,
            'test_end': self.test_end,
            'train_samples': self.train_samples,
            'val_samples': self.val_samples,
            'test_samples': self.test_samples,
            'epochs_trained': self.epochs_trained,
            'final_train_loss': self.final_train_loss,
            'final_val_loss': self.final_val_loss,
            'directional_accuracy': self.directional_accuracy,
            'correlation': self.correlation,
            'ic_mean': self.ic_mean,
            'ic_ir': self.ic_ir,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            't_statistic': self.t_statistic,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'model_params': self.model_params,
            'model_path': self.model_path,
            'feature_importance': self.feature_importance,
        }


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class TechnicalFeatureExtractor:
    """
    Extract technical features from OHLCV data.
    
    Features include:
    - Returns (various lookbacks)
    - Volatility (various windows)
    - Moving averages ratios
    - RSI, MACD
    - Volume features
    """
    
    def __init__(self, windows: List[int] = [5, 10, 20, 60]):
        self.windows = windows
    
    def extract(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract features from OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        
        Returns:
            DataFrame with features, list of feature names
        """
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Returns
        for w in self.windows:
            features[f'ret_{w}d'] = close.pct_change(w)
        
        # Log returns
        log_close = np.log(close)
        for w in self.windows:
            features[f'logret_{w}d'] = log_close.diff(w)
        
        # Volatility
        daily_ret = close.pct_change()
        for w in self.windows:
            features[f'vol_{w}d'] = daily_ret.rolling(w).std()
        
        # Moving average ratios
        for w in self.windows:
            ma = close.rolling(w).mean()
            features[f'ma_ratio_{w}d'] = close / ma - 1
        
        # RSI
        for w in [14, 28]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
            rs = gain / (loss + 1e-8)
            features[f'rsi_{w}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        features['macd'] = macd / close  # Normalize
        features['macd_signal'] = signal / close
        features['macd_hist'] = (macd - signal) / close
        
        # Bollinger Bands
        for w in [20]:
            ma = close.rolling(w).mean()
            std = close.rolling(w).std()
            features[f'bb_upper_{w}'] = (close - (ma + 2*std)) / close
            features[f'bb_lower_{w}'] = (close - (ma - 2*std)) / close
            features[f'bb_width_{w}'] = (4 * std) / ma
        
        # ATR (Average True Range)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        for w in [14]:
            features[f'atr_{w}'] = tr.rolling(w).mean() / close
        
        # Volume features
        vol_ma20 = volume.rolling(20).mean()
        features['volume_ratio'] = volume / (vol_ma20 + 1)
        features['volume_change'] = volume.pct_change()
        
        # Price position within range
        for w in self.windows:
            highest = high.rolling(w).max()
            lowest = low.rolling(w).min()
            features[f'price_position_{w}d'] = (close - lowest) / (highest - lowest + 1e-8)
        
        # Momentum
        for w in self.windows:
            features[f'momentum_{w}d'] = close / close.shift(w) - 1
        
        feature_names = features.columns.tolist()
        
        return features, feature_names


# =============================================================================
# MODELS
# =============================================================================

class LinearModel(nn.Module):
    """Simple linear model - baseline."""
    
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.linear(x).squeeze(-1)


class MLPModel(nn.Module):
    """MLP with proper regularization."""
    
    def __init__(self, input_size: int, hidden_size: int = 32, dropout: float = 0.3):
        super().__init__()
        
        self.bn_input = nn.BatchNorm1d(input_size)
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        # Skip connection
        self.skip = nn.Linear(input_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, -1, :]
        
        skip = self.skip(x)
        
        x = self.bn_input(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return (x + skip).squeeze(-1)


# =============================================================================
# TRAINING
# =============================================================================

class EarlyStopping:
    """Early stopping with model checkpointing."""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_state = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
    
    def restore_best(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


class RealDataTrainer:
    """
    Train models on real market data from Polygon.io.
    """
    
    def __init__(
        self,
        api_key: str,
        models_dir: str = "models",
        results_dir: str = "training_results",
        device: str = 'auto'
    ):
        self.api_key = api_key
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        if device == 'auto':
            self.device = torch.device(
                'mps' if torch.backends.mps.is_available() else
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)
        
        self.feature_extractor = TechnicalFeatureExtractor()
    
    async def fetch_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Polygon."""
        
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required for async data fetching")
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'apiKey': self.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Polygon API error: {response.status}")
                
                data = await response.json()
        
        if 'results' not in data or not data['results']:
            raise ValueError(f"No data returned for {symbol}")
        
        df = pd.DataFrame(data['results'])
        df['date'] = pd.to_datetime(df['t'], unit='ms')
        df = df.set_index('date')
        df = df.rename(columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        })
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2
    ) -> Tuple[Dict[str, torch.Tensor], List[str], pd.DatetimeIndex]:
        """
        Prepare training data with proper alignment.
        
        CRITICAL: Features[t] predict returns[t+1]
        """
        # Extract features
        features_df, feature_names = self.feature_extractor.extract(df)
        
        # Calculate forward returns (what we're predicting)
        forward_returns = df['close'].pct_change().shift(-1)  # return from t to t+1
        
        # Combine and drop NaN
        combined = pd.concat([features_df, forward_returns.rename('target')], axis=1)
        combined = combined.dropna()
        
        X = combined[feature_names].values
        y = combined['target'].values
        dates = combined.index
        
        # Split
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        data = {
            'train_X': torch.tensor(X[:train_end], dtype=torch.float32),
            'train_y': torch.tensor(y[:train_end], dtype=torch.float32),
            'val_X': torch.tensor(X[train_end:val_end], dtype=torch.float32),
            'val_y': torch.tensor(y[train_end:val_end], dtype=torch.float32),
            'test_X': torch.tensor(X[val_end:], dtype=torch.float32),
            'test_y': torch.tensor(y[val_end:], dtype=torch.float32),
        }
        
        return data, feature_names, dates
    
    def train_model(
        self,
        model: nn.Module,
        data: Dict[str, torch.Tensor],
        epochs: int = 300,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        patience: int = 30,
        batch_size: int = 64
    ) -> Tuple[nn.Module, Dict]:
        """Train model with early stopping."""
        
        model = model.to(self.device)
        
        train_X = data['train_X'].to(self.device)
        train_y = data['train_y'].to(self.device)
        val_X = data['val_X'].to(self.device)
        val_y = data['val_y'].to(self.device)
        
        # Combined loss
        def combined_loss(pred, target):
            mse = F.mse_loss(pred, target)
            
            # Directional penalty
            pred_sign = torch.sign(pred)
            target_sign = torch.sign(target)
            wrong_dir = (pred_sign != target_sign).float()
            dir_penalty = (wrong_dir * (pred - target) ** 2).mean()
            
            return 0.5 * mse + 0.5 * dir_penalty
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        early_stopping = EarlyStopping(patience=patience)
        
        history = {'train_loss': [], 'val_loss': [], 'val_dir_acc': []}
        
        n_batches = (len(train_X) + batch_size - 1) // batch_size
        epochs_trained = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            perm = torch.randperm(len(train_X), device=self.device)
            train_X_shuffled = train_X[perm]
            train_y_shuffled = train_y[perm]
            
            train_losses = []
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(train_X))
                
                batch_X = train_X_shuffled[start:end]
                batch_y = train_y_shuffled[start:end]
                
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = combined_loss(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(val_X)
                val_loss = combined_loss(val_pred, val_y).item()
                val_dir_acc = (torch.sign(val_pred) == torch.sign(val_y)).float().mean().item()
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_dir_acc'].append(val_dir_acc)
            
            scheduler.step(val_loss)
            epochs_trained = epoch + 1
            
            if early_stopping(val_loss, model):
                break
            
            if (epoch + 1) % 50 == 0:
                print(f"   Epoch {epoch+1}: Train={avg_train_loss:.6f}, Val={val_loss:.6f}, Dir Acc={val_dir_acc:.2%}")
        
        early_stopping.restore_best(model)
        model = model.to(self.device)
        
        history['epochs_trained'] = epochs_trained
        history['final_train_loss'] = history['train_loss'][-1] if history['train_loss'] else 0
        history['final_val_loss'] = early_stopping.best_loss or val_loss
        
        return model, history
    
    def evaluate(
        self,
        model: nn.Module,
        data: Dict[str, torch.Tensor],
        transaction_cost_bps: float = 10.0
    ) -> Dict:
        """Evaluate model on test set."""
        
        model = model.to(self.device)
        model.eval()
        
        test_X = data['test_X'].to(self.device)
        test_y = data['test_y'].cpu().numpy()
        
        with torch.no_grad():
            predictions = model(test_X).cpu().numpy()
        
        # Metrics
        metrics = {}
        
        # Directional accuracy
        metrics['directional_accuracy'] = np.mean(np.sign(predictions) == np.sign(test_y))
        
        # Correlation
        if np.std(predictions) > 0 and np.std(test_y) > 0:
            metrics['correlation'] = np.corrcoef(predictions, test_y)[0, 1]
        else:
            metrics['correlation'] = 0
        
        # IC
        ic_vals = []
        window = 20
        if HAS_SCIPY:
            for i in range(0, len(predictions) - window, window // 4):
                ic, _ = spearmanr(predictions[i:i+window], test_y[i:i+window])
                if not np.isnan(ic):
                    ic_vals.append(ic)
        
        metrics['ic_mean'] = np.mean(ic_vals) if ic_vals else 0
        metrics['ic_std'] = np.std(ic_vals) if len(ic_vals) > 1 else 1
        metrics['ic_ir'] = metrics['ic_mean'] / metrics['ic_std'] if metrics['ic_std'] > 0 else 0
        
        # Trading simulation
        cost = transaction_cost_bps / 10000
        positions = np.sign(predictions)
        costs = np.abs(np.diff(positions, prepend=0)) * cost
        returns = positions * test_y - costs
        
        # Sharpe
        if np.std(returns) > 0:
            metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0
        
        # Sortino
        downside = returns[returns < 0]
        if len(downside) > 0:
            downside_std = np.std(downside)
            metrics['sortino_ratio'] = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        else:
            metrics['sortino_ratio'] = metrics['sharpe_ratio']
        
        # Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-8)
        metrics['max_drawdown'] = drawdown.min()
        
        # Returns
        metrics['total_return'] = cumulative[-1] - 1 if len(cumulative) > 0 else 0
        n_years = len(test_y) / 252
        metrics['annual_return'] = (1 + metrics['total_return']) ** (1/n_years) - 1 if n_years > 0 else 0
        
        # Statistical significance
        if HAS_SCIPY and len(returns) > 1:
            t_stat, p_value = ttest_1samp(returns, 0)
            metrics['t_statistic'] = t_stat if not np.isnan(t_stat) else 0
            metrics['p_value'] = p_value if not np.isnan(p_value) else 1
        else:
            metrics['t_statistic'] = 0
            metrics['p_value'] = 1
        
        metrics['is_significant'] = metrics['p_value'] < 0.05 and metrics['sharpe_ratio'] > 0
        
        return metrics
    
    async def train_and_validate(
        self,
        symbol: str,
        model_type: str = 'mlp',
        start_date: str = None,
        end_date: str = None,
        save_model: bool = True
    ) -> TrainingResult:
        """
        Full pipeline: fetch data, train, validate, save.
        """
        print(f"\n{'='*70}")
        print(f"  TRAINING ON REAL DATA: {symbol}")
        print(f"{'='*70}")
        
        # Default dates
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        
        # Fetch data
        print(f"\n📊 Fetching {symbol} data from {start_date} to {end_date}...")
        df = await self.fetch_data(symbol, start_date, end_date)
        print(f"   Received {len(df):,} bars")
        
        # Prepare data
        print("\n🔧 Extracting features...")
        data, feature_names, dates = self.prepare_data(df)
        n_features = len(feature_names)
        print(f"   Features: {n_features}")
        print(f"   Train: {len(data['train_X']):,}, Val: {len(data['val_X']):,}, Test: {len(data['test_X']):,}")
        
        # Create model
        print(f"\n🧠 Creating {model_type} model...")
        if model_type == 'linear':
            model = LinearModel(n_features)
        else:
            model = MLPModel(n_features, hidden_size=64, dropout=0.3)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {n_params:,}")
        
        # Train
        print("\n🏋️ Training...")
        model, history = self.train_model(model, data)
        print(f"   Epochs: {history['epochs_trained']}")
        print(f"   Final val loss: {history['final_val_loss']:.6f}")
        
        # Evaluate
        print("\n📈 Evaluating on test set...")
        metrics = self.evaluate(model, data)
        
        # Get date ranges
        n = len(dates)
        train_end_idx = int(n * 0.6)
        val_end_idx = int(n * 0.8)
        
        train_start = dates[0].strftime('%Y-%m-%d')
        train_end = dates[train_end_idx-1].strftime('%Y-%m-%d')
        test_start = dates[val_end_idx].strftime('%Y-%m-%d')
        test_end = dates[-1].strftime('%Y-%m-%d')
        
        # Save model
        model_path = None
        if save_model:
            model_path = str(self.models_dir / f"{model_type}_{symbol}_{datetime.now().strftime('%Y%m%d')}.pt")
            torch.save({
                'model_state': model.state_dict(),
                'model_type': model_type,
                'n_features': n_features,
                'feature_names': feature_names,
                'symbol': symbol,
                'train_end': train_end,
            }, model_path)
            print(f"\n💾 Model saved: {model_path}")
        
        # Create result
        result = TrainingResult(
            symbol=symbol,
            model_name=model_type.upper(),
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_samples=len(data['train_X']),
            val_samples=len(data['val_X']),
            test_samples=len(data['test_X']),
            epochs_trained=history['epochs_trained'],
            final_train_loss=history['final_train_loss'],
            final_val_loss=history['final_val_loss'],
            directional_accuracy=metrics['directional_accuracy'],
            correlation=metrics['correlation'],
            ic_mean=metrics['ic_mean'],
            ic_ir=metrics['ic_ir'],
            sharpe_ratio=metrics['sharpe_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            max_drawdown=metrics['max_drawdown'],
            total_return=metrics['total_return'],
            annual_return=metrics['annual_return'],
            t_statistic=metrics['t_statistic'],
            p_value=metrics['p_value'],
            is_significant=metrics['is_significant'],
            model_params=n_params,
            model_path=model_path,
        )
        
        # Save result
        result_path = self.results_dir / f"result_{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        print(result)
        
        return result


async def train_on_real_data(
    symbols: List[str] = ['SPY'],
    model_type: str = 'mlp',
    api_key: str = None
) -> List[TrainingResult]:
    """
    Convenience function to train on multiple symbols.
    """
    if api_key is None:
        api_key = os.environ.get('POLYGON_API_KEY')
    
    if not api_key:
        raise ValueError("POLYGON_API_KEY not set")
    
    trainer = RealDataTrainer(api_key=api_key)
    
    results = []
    for symbol in symbols:
        try:
            result = await trainer.train_and_validate(symbol, model_type)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed for {symbol}: {e}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train on real market data")
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol to train on')
    parser.add_argument('--model', type=str, default='mlp', choices=['linear', 'mlp'])
    parser.add_argument('--api-key', type=str, default=None)
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('POLYGON_API_KEY')
    
    if not api_key:
        print("❌ POLYGON_API_KEY not set")
        print("   Set it with: export POLYGON_API_KEY='your-key'")
        sys.exit(1)
    
    asyncio.run(train_on_real_data([args.symbol], args.model, api_key))
