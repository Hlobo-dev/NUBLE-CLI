#!/usr/bin/env python3
"""
Test the REAL data training pipeline.

This script will:
1. Fetch real market data from Polygon
2. Train a model with proper regularization
3. Validate on out-of-sample data
4. Report REAL performance metrics

The moment of truth: Does our model have predictive power on REAL data?
"""

import os
import sys
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def main():
    # Import after path setup
    from institutional.ml.training import RealDataTrainer, train_on_real_data
    
    api_key = os.environ.get('POLYGON_API_KEY', 'JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY')
    
    print("\n" + "="*70)
    print("  KYPERIAN ML - REAL DATA VALIDATION")
    print("="*70)
    print("\nThis is the moment of truth.")
    print("Training on REAL market data and validating on OUT-OF-SAMPLE data.\n")
    
    trainer = RealDataTrainer(api_key=api_key)
    
    # Test on SPY (most liquid, most efficient market = hardest to predict)
    symbols = ['SPY']
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  TRAINING: {symbol}")
        print(f"{'='*60}")
        
        try:
            result = await trainer.train_and_validate(
                symbol=symbol,
                start_date='2020-01-01',
                end_date='2024-01-01',
                model_type='mlp',
            )
            
            print("\n" + "="*60)
            print("  RESULT SUMMARY")
            print("="*60)
            print(f"  Symbol:              {result.symbol}")
            print(f"  Model:               {result.model_name}")
            print(f"  Training samples:    {result.train_samples}")
            print(f"  Validation samples:  {result.val_samples}")
            print(f"  Test samples:        {result.test_samples}")
            print()
            print(f"  📊 Test Directional Accuracy: {result.test_directional_accuracy:.2%}")
            print(f"  📈 Test Sharpe Ratio:         {result.test_sharpe:.2f}")
            print(f"  📉 Test Max Drawdown:         {result.test_max_drawdown:.2%}")
            print(f"  💰 Test Total Return:         {result.test_total_return:.2%}")
            print()
            
            # Interpret results
            if result.test_sharpe > 0.5 and result.test_directional_accuracy > 0.52:
                print("  ✅ PROMISING! Model shows some predictive power.")
                print("     Consider walk-forward validation for robustness.")
            elif result.test_sharpe > 0:
                print("  ⚠️  MARGINAL. Some edge, but not statistically robust.")
                print("     May be noise. Need more testing.")
            else:
                print("  ❌ NO EDGE. Model does not beat random on real data.")
                print("     This is expected - markets are efficient.")
            
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("  TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
