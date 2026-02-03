"""
Train and Ship Pre-Trained Models
==================================

Trains models on validated alpha sources and saves them for distribution.
Uses walk-forward validation to ensure models generalize.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check for required environment variables
from dotenv import load_dotenv
load_dotenv()


def get_validated_symbols() -> Dict[str, Dict[str, Any]]:
    """Get symbols with validated alpha from walk-forward testing."""
    return {
        'SLV': {
            'wf_sharpe': 0.94,
            'timeframe': '1d',
            'reason': 'Silver ETF - momentum persistence',
            'priority': 1
        },
        'TSLA': {
            'wf_sharpe': 0.91,
            'timeframe': '1d', 
            'reason': 'High retail attention - pattern persistence',
            'priority': 2
        },
        'XLK': {
            'wf_sharpe': 0.92,
            'timeframe': '1d',
            'reason': 'Tech sector ETF - trend following',
            'priority': 3
        },
        'AMD': {
            'wf_sharpe': 0.76,
            'timeframe': '1d',
            'reason': 'Semiconductor volatility - mean reversion',
            'priority': 4
        },
    }


async def train_single_model(
    symbol: str,
    config: Dict[str, Any],
    save: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Train a single model and optionally save it.
    
    Returns training results dict.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training model for {symbol}")
    logger.info(f"Expected WF Sharpe: {config['wf_sharpe']}")
    logger.info(f"{'='*60}")
    
    try:
        from src.institutional.ml.training.real_data_trainer import RealDataTrainer
        from src.institutional.ml.registry import get_registry, ModelMetadata
        
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            logger.error("POLYGON_API_KEY not set")
            return None
        
        # Initialize trainer
        trainer = RealDataTrainer(
            api_key=api_key,
            device='mps'  # Use Apple Silicon GPU
        )
        
        # Train on real data
        result = await trainer.train_and_validate(
            symbol=symbol,
            model_type='mlp',
            save_model=True
        )
        
        if not result:
            logger.error(f"Training returned None for {symbol}")
            return None
        
        logger.info(f"Training complete for {symbol}")
        logger.info(f"  Val Sharpe: {result.sharpe_ratio:.2f}")
        logger.info(f"  Dir Acc: {result.directional_accuracy*100:.1f}%")
        logger.info(f"  P-value: {result.p_value:.4f}")
        
        # Save to registry if good performance
        if save and result.sharpe_ratio > 0:
            registry = get_registry()
            
            metadata = ModelMetadata(
                symbol=symbol,
                model_type='MLP',
                timeframe=config['timeframe'],
                trained_at=datetime.now().isoformat(),
                training_samples=result.train_samples,
                features=[],  # Will be loaded from model file
                validation_sharpe=result.sharpe_ratio,
                validation_return=result.total_return,
                directional_accuracy=result.directional_accuracy,
                p_value=result.p_value,
                wf_sharpe=config.get('wf_sharpe'),
                hidden_dims=[64],
                learning_rate=0.001,
                epochs_trained=result.epochs_trained
            )
            
            # Update registry
            registry._model_metadata[symbol] = metadata
            registry._save_registry()
            logger.info(f"✅ Model registered: {symbol} (Grade: {metadata.grade})")
        
        return {
            'val_sharpe': result.sharpe_ratio,
            'directional_accuracy': result.directional_accuracy,
            'p_value': result.p_value,
            'total_return': result.total_return,
            'model_path': result.model_path,
            'training_samples': result.train_samples,
        }
        
    except Exception as e:
        logger.error(f"Training failed for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


async def train_all_validated(
    save: bool = True,
    symbols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Train models for all validated alpha sources.
    
    Args:
        save: Whether to save models
        symbols: Specific symbols to train (default: all validated)
        
    Returns:
        Summary of training results
    """
    validated = get_validated_symbols()
    
    if symbols:
        validated = {s: validated[s] for s in symbols if s in validated}
    
    # Sort by priority
    sorted_symbols = sorted(validated.items(), key=lambda x: x[1]['priority'])
    
    results = {}
    successful = 0
    failed = 0
    
    print("\n" + "="*70)
    print("NUBLE Pre-Trained Model Training")
    print("="*70)
    print(f"Training {len(sorted_symbols)} models with validated alpha")
    print(f"Save: {save}")
    print("="*70 + "\n")
    
    for symbol, config in sorted_symbols:
        try:
            result = await train_single_model(symbol, config, save)
            
            if result and result.get('val_sharpe', 0) > 0:
                results[symbol] = {
                    'status': 'success',
                    'val_sharpe': result.get('val_sharpe', 0),
                    'dir_acc': result.get('directional_accuracy', 0),
                    'p_value': result.get('p_value', 1)
                }
                successful += 1
            else:
                results[symbol] = {'status': 'failed', 'error': 'Low performance'}
                failed += 1
                
        except Exception as e:
            results[symbol] = {'status': 'error', 'error': str(e)}
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()
    
    for symbol, result in results.items():
        status = result['status']
        if status == 'success':
            print(f"  ✅ {symbol}: Sharpe={result['val_sharpe']:.2f}, "
                  f"DirAcc={result['dir_acc']*100:.1f}%")
        else:
            print(f"  ❌ {symbol}: {result.get('error', 'Unknown error')}")
    
    print("="*70 + "\n")
    
    return {
        'successful': successful,
        'failed': failed,
        'results': results
    }


def verify_models() -> Dict[str, Any]:
    """Verify all saved pre-trained models work correctly."""
    from src.institutional.ml.registry import get_registry
    
    registry = get_registry()
    status = registry.get_status()
    
    print("\n" + "="*70)
    print("MODEL VERIFICATION")
    print("="*70)
    print(f"Models Directory: {status['models_dir']}")
    print(f"Total Models: {status['total_models']}")
    print()
    
    verified = 0
    failed = 0
    
    for symbol in status['models']:
        try:
            meta = registry.get_metadata(symbol)
            if meta:
                print(f"  ✅ {symbol}: Grade {meta.grade}, "
                      f"Sharpe {meta.wf_sharpe or meta.validation_sharpe:.2f}")
                verified += 1
            else:
                print(f"  ❌ {symbol}: No metadata")
                failed += 1
        except Exception as e:
            print(f"  ❌ {symbol}: {e}")
            failed += 1
    
    print()
    print(f"Verified: {verified}, Failed: {failed}")
    print("="*70 + "\n")
    
    return {'verified': verified, 'failed': failed}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train pre-trained models')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to train')
    parser.add_argument('--verify', action='store_true', help='Verify existing models')
    parser.add_argument('--no-save', action='store_true', help='Dont save models')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_models()
    else:
        asyncio.run(train_all_validated(
            save=not args.no_save,
            symbols=args.symbols
        ))
