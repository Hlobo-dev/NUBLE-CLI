#!/usr/bin/env python3
"""
Multi-Symbol Validation with Financial Losses

This tests our models on multiple symbols with:
1. Proper directional loss function (not MSE)
2. Different asset classes
3. Longer prediction horizons

The goal: Find if ANY alpha exists, not just SPY.
"""

import os
import sys
import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class DirectionalLoss(nn.Module):
    """Loss that penalizes wrong direction more heavily."""
    def __init__(self, direction_weight: float = 3.0):
        super().__init__()
        self.direction_weight = direction_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Base MSE
        mse = (predictions - targets) ** 2
        
        # Direction penalty
        pred_sign = torch.sign(predictions)
        tgt_sign = torch.sign(targets)
        wrong_dir = (pred_sign != tgt_sign).float()
        
        # Higher weight when direction is wrong
        weights = 1.0 + wrong_dir * (self.direction_weight - 1)
        
        return (mse * weights).mean()


async def test_symbol(symbol: str, api_key: str, horizon: int = 1) -> dict:
    """Test a single symbol with directional loss."""
    try:
        import aiohttp
    except ImportError:
        raise ImportError("aiohttp required")
    
    print(f"\n--- Testing {symbol} (horizon={horizon}d) ---")
    
    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    params = {'apiKey': api_key, 'adjusted': 'true', 'sort': 'asc'}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                print(f"  ❌ API error: {response.status}")
                return None
            data = await response.json()
    
    if 'results' not in data or len(data['results']) < 200:
        print(f"  ❌ Insufficient data")
        return None
    
    df = pd.DataFrame(data['results'])
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']]
    
    # Simple features (to avoid overfitting)
    df['ret_1'] = df['close'].pct_change()
    df['ret_5'] = df['close'].pct_change(5)
    df['vol_20'] = df['ret_1'].rolling(20).std()
    df['mom_10'] = df['close'] / df['close'].shift(10) - 1
    df['rsi'] = 100 - (100 / (1 + df['ret_1'].rolling(14).apply(
        lambda x: x[x > 0].sum() / (-x[x < 0].sum() + 1e-8))))
    
    # Target: forward return
    df['target'] = df['close'].pct_change(horizon).shift(-horizon)
    
    df = df.dropna()
    
    if len(df) < 200:
        print(f"  ❌ Insufficient data after cleaning")
        return None
    
    # Split: 60% train, 20% val, 20% test
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    features = ['ret_1', 'ret_5', 'vol_20', 'mom_10', 'rsi']
    
    train_X = torch.tensor(df[features].iloc[:train_end].values, dtype=torch.float32)
    train_y = torch.tensor(df['target'].iloc[:train_end].values, dtype=torch.float32).unsqueeze(1)
    val_X = torch.tensor(df[features].iloc[train_end:val_end].values, dtype=torch.float32)
    val_y = torch.tensor(df['target'].iloc[train_end:val_end].values, dtype=torch.float32).unsqueeze(1)
    test_X = torch.tensor(df[features].iloc[val_end:].values, dtype=torch.float32)
    test_y = df['target'].iloc[val_end:].values
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(len(features), 16),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(16, 1)
    )
    
    # Train with directional loss
    loss_fn = DirectionalLoss(direction_weight=3.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    best_state = None
    patience = 20
    no_improve = 0
    
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        pred = model(train_X)
        loss = loss_fn(pred, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X)
            val_loss = loss_fn(val_pred, val_y)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            break
    
    # Load best
    model.load_state_dict(best_state)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        predictions = model(test_X).numpy().flatten()
    
    # Metrics
    dir_acc = np.mean(np.sign(predictions) == np.sign(test_y))
    
    positions = np.sign(predictions)
    costs = np.abs(np.diff(positions, prepend=0)) * 0.001
    returns = positions * test_y - costs
    
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 / horizon)
    total_return = np.prod(1 + returns) - 1
    
    # Status
    status = "✅" if sharpe > 0.5 and dir_acc > 0.52 else "⚠️" if sharpe > 0 else "❌"
    
    print(f"  {status} Dir Acc: {dir_acc:.2%}, Sharpe: {sharpe:.2f}, Return: {total_return:.2%}")
    
    return {
        'symbol': symbol,
        'horizon': horizon,
        'dir_acc': dir_acc,
        'sharpe': sharpe,
        'total_return': total_return,
        'n_test': len(test_y)
    }


async def main():
    api_key = os.environ.get('POLYGON_API_KEY', 'JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY')
    
    print("\n" + "="*70)
    print("  MULTI-SYMBOL ALPHA HUNTING")
    print("="*70)
    print("\nTesting multiple symbols with directional loss...\n")
    
    # Test multiple symbols across asset classes
    symbols = [
        # Major indices/ETFs
        'SPY', 'QQQ', 'IWM',
        # Sectors
        'XLF', 'XLK', 'XLE',
        # High vol
        'TSLA', 'NVDA', 'AMD',
        # Commodities
        'GLD', 'SLV', 'USO',
    ]
    
    horizons = [1, 5]  # 1-day and 5-day
    
    results = []
    for horizon in horizons:
        print(f"\n{'='*50}")
        print(f"  HORIZON: {horizon} days")
        print(f"{'='*50}")
        
        for symbol in symbols:
            try:
                result = await test_symbol(symbol, api_key, horizon)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"  ❌ {symbol}: {e}")
    
    # Summary
    if results:
        print("\n" + "="*70)
        print("  SUMMARY")
        print("="*70)
        
        df_results = pd.DataFrame(results)
        
        # Best results
        profitable = df_results[df_results['sharpe'] > 0].sort_values('sharpe', ascending=False)
        
        if len(profitable) > 0:
            print("\n✅ PROFITABLE (Sharpe > 0):")
            for _, row in profitable.head(5).iterrows():
                print(f"   {row['symbol']:5s} ({row['horizon']}d): Sharpe={row['sharpe']:.2f}, Dir={row['dir_acc']:.2%}")
        else:
            print("\n❌ NO profitable strategies found.")
        
        # Overall stats
        print(f"\n📊 Overall Statistics:")
        print(f"   Strategies tested: {len(results)}")
        print(f"   Profitable: {len(profitable)} ({len(profitable)/len(results):.1%})")
        print(f"   Avg Sharpe: {df_results['sharpe'].mean():.2f}")
        print(f"   Best Sharpe: {df_results['sharpe'].max():.2f}")
        print(f"   Avg Dir Acc: {df_results['dir_acc'].mean():.2%}")
    
    print("\n" + "="*70)
    print("  COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
