#!/usr/bin/env python3
"""
NUBLE Model Training Script
==============================

Production training script for financial ML models.
Fetches real market data, trains models, and saves checkpoints.

Usage:
    python -m src.institutional.ml.train_script --symbols AAPL MSFT GOOGL --epochs 100
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.institutional.ml.torch_models import (
    ModelConfig,
    FinancialLSTM,
    AttentionLSTM,
    MarketTransformer,
    TemporalFusionTransformer,
    EnsembleNetwork,
    NeuralRegimeClassifier,
    PolygonDataFetcher,
    TechnicalFeatureExtractor,
    FeatureConfig,
    FinancialTrainer,
    WalkForwardValidator,
    get_device
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_training_data(
    symbols: List[str],
    years: int = 5,
    api_key: Optional[str] = None
) -> dict:
    """
    Fetch historical data for multiple symbols.
    
    Args:
        symbols: List of ticker symbols
        years: Years of history to fetch
        api_key: Polygon API key
        
    Returns:
        Dict of symbol -> OHLCV arrays
    """
    api_key = api_key or os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError("POLYGON_API_KEY not set")
    
    fetcher = PolygonDataFetcher(api_key)
    extractor = TechnicalFeatureExtractor()
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=years * 365)
    
    data = {}
    
    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}...")
        try:
            bars = fetcher.fetch_historical(
                symbol, start_date, end_date, '1d', use_cache=True
            )
            ohlcv = fetcher.bars_to_array(bars)
            features, feature_names = extractor.extract(ohlcv)
            
            # Compute returns as targets
            close_prices = ohlcv[:, 3]  # Close prices
            returns = np.diff(close_prices) / close_prices[:-1]
            
            # Multi-horizon returns
            horizons = [1, 5, 10, 20]
            targets = {}
            for h in horizons:
                future_returns = np.zeros(len(returns))
                for i in range(len(returns) - h):
                    future_returns[i] = (close_prices[i + h] - close_prices[i]) / close_prices[i]
                targets[h] = future_returns[:-h] if h > 0 else future_returns
            
            # Align features and targets
            min_len = min(len(features) - 1, min(len(t) for t in targets.values()))
            features = features[:min_len]
            targets = {h: t[:min_len] for h, t in targets.items()}
            
            data[symbol] = {
                'features': features,
                'feature_names': feature_names,
                'targets': targets,
                'ohlcv': ohlcv[:min_len]
            }
            
            logger.info(f"  {symbol}: {len(features)} samples, {features.shape[1]} features")
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
    
    return data


def create_sequences(
    features: np.ndarray,
    targets: dict,
    sequence_length: int = 60
) -> tuple:
    """
    Create sequences for LSTM/Transformer training.
    
    Args:
        features: Feature array (N, F)
        targets: Dict of horizon -> target array
        sequence_length: Length of input sequences
        
    Returns:
        X, y tensors
    """
    n_samples = len(features) - sequence_length
    n_features = features.shape[1]
    n_horizons = len(targets)
    
    X = np.zeros((n_samples, sequence_length, n_features))
    y = np.zeros((n_samples, n_horizons))
    
    horizons = sorted(targets.keys())
    
    for i in range(n_samples):
        X[i] = features[i:i + sequence_length]
        for j, h in enumerate(horizons):
            if i + sequence_length < len(targets[h]):
                y[i, j] = targets[h][i + sequence_length]
    
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()


def train_model(
    model: torch.nn.Module,
    train_data: dict,
    config: ModelConfig,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    save_dir: Path = None,
    model_name: str = 'model'
) -> dict:
    """
    Train a single model.
    
    Args:
        model: PyTorch model
        train_data: Training data dict
        config: Model config
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        save_dir: Directory to save checkpoints
        model_name: Name for the model
        
    Returns:
        Training metrics
    """
    save_dir = save_dir or Path.home() / '.nuble' / 'models'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = next(model.parameters()).device
    
    # Combine all symbol data
    all_X, all_y = [], []
    for symbol, data in train_data.items():
        X, y = create_sequences(
            data['features'],
            data['targets'],
            config.sequence_length
        )
        all_X.append(X)
        all_y.append(y)
    
    X = torch.cat(all_X, dim=0)
    y = torch.cat(all_y, dim=0)
    
    logger.info(f"Training data: {X.shape[0]} sequences")
    
    # Train/val split (use last 20% as validation)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Loss function
    criterion = torch.nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            output = model(batch_x)
            
            # Extract predictions
            if 'predictions' in output:
                preds = []
                for h in sorted(output['predictions'].keys()):
                    preds.append(output['predictions'][h]['mean'])
                pred = torch.cat(preds, dim=-1)
            else:
                pred = output.get('output', batch_x.mean(dim=1))
            
            loss = criterion(pred, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                output = model(batch_x)
                
                if 'predictions' in output:
                    preds = []
                    for h in sorted(output['predictions'].keys()):
                        preds.append(output['predictions'][h]['mean'])
                    pred = torch.cat(preds, dim=-1)
                else:
                    pred = output.get('output', batch_x.mean(dim=1))
                
                loss = criterion(pred, batch_y)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_config': config.to_dict(),
                'model_type': model.__class__.__name__,
                'epoch': epoch,
                'val_loss': val_loss,
                'history': history
            }
            torch.save(checkpoint, save_dir / f'{model_name}_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save final model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': config.to_dict(),
        'model_type': model.__class__.__name__,
        'epoch': epochs,
        'val_loss': val_loss,
        'history': history
    }
    torch.save(checkpoint, save_dir / f'{model_name}_final.pt')
    
    logger.info(f"Model saved to {save_dir / model_name}_*.pt")
    
    return {
        'best_val_loss': best_val_loss,
        'final_val_loss': val_loss,
        'epochs_trained': len(history['train_loss']),
        'history': history
    }


def main():
    parser = argparse.ArgumentParser(description='Train NUBLE ML models')
    parser.add_argument(
        '--symbols', nargs='+', 
        default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
        help='Symbols to train on'
    )
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length')
    parser.add_argument('--years', type=int, default=5, help='Years of historical data')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['lstm', 'transformer', 'ensemble', 'regime', 'all'],
                       help='Model to train')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device('auto')
    logger.info(f"Using device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path.home() / '.nuble' / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch data
    logger.info("Fetching training data...")
    train_data = fetch_training_data(args.symbols, args.years)
    
    if not train_data:
        logger.error("No training data available")
        return
    
    # Get feature count from first symbol
    first_symbol = list(train_data.keys())[0]
    n_features = train_data[first_symbol]['features'].shape[1]
    
    # Base config
    config = ModelConfig(
        input_size=n_features,
        hidden_size=256,
        num_layers=3,
        num_heads=8,
        dropout=0.2,
        sequence_length=args.sequence_length,
        prediction_horizons=[1, 5, 10, 20]
    )
    
    results = {}
    
    # Train LSTM
    if args.model in ['lstm', 'all']:
        logger.info("\n" + "="*50)
        logger.info("Training LSTM Model")
        logger.info("="*50)
        
        lstm = FinancialLSTM(config).to(device)
        results['lstm'] = train_model(
            lstm, train_data, config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_dir=output_dir,
            model_name='lstm'
        )
    
    # Train Transformer
    if args.model in ['transformer', 'all']:
        logger.info("\n" + "="*50)
        logger.info("Training Transformer Model")
        logger.info("="*50)
        
        transformer = MarketTransformer(config).to(device)
        results['transformer'] = train_model(
            transformer, train_data, config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_dir=output_dir,
            model_name='transformer'
        )
    
    # Train Ensemble
    if args.model in ['ensemble', 'all']:
        logger.info("\n" + "="*50)
        logger.info("Training Ensemble Model")
        logger.info("="*50)
        
        ensemble = EnsembleNetwork(config).to(device)
        results['ensemble'] = train_model(
            ensemble, train_data, config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_dir=output_dir,
            model_name='ensemble'
        )
    
    # Train Regime Classifier
    if args.model in ['regime', 'all']:
        logger.info("\n" + "="*50)
        logger.info("Training Regime Classifier")
        logger.info("="*50)
        
        regime_config = ModelConfig(
            input_size=n_features,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            sequence_length=args.sequence_length
        )
        
        regime = NeuralRegimeClassifier(regime_config, num_regimes=6).to(device)
        results['regime'] = train_model(
            regime, train_data, regime_config,
            epochs=args.epochs // 2,  # Regime classifier converges faster
            batch_size=args.batch_size,
            lr=args.lr * 2,
            save_dir=output_dir,
            model_name='regime'
        )
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("Training Summary")
    logger.info("="*50)
    
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  Best Val Loss: {metrics['best_val_loss']:.6f}")
        logger.info(f"  Final Val Loss: {metrics['final_val_loss']:.6f}")
        logger.info(f"  Epochs Trained: {metrics['epochs_trained']}")
    
    logger.info(f"\nModels saved to: {output_dir}")


if __name__ == '__main__':
    main()
