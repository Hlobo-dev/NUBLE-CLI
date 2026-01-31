#!/usr/bin/env python3
"""
Walk-Forward Validation on Best Performers

This tests whether our top performers (GLD, SLV, AMD, TSLA) maintain
edge over time with monthly retraining.
"""

import os
import sys
import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class DirectionalLoss(nn.Module):
    """Loss that penalizes wrong direction more heavily."""
    def __init__(self, direction_weight: float = 3.0):
        super().__init__()
        self.direction_weight = direction_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = (predictions - targets) ** 2
        pred_sign = torch.sign(predictions)
        tgt_sign = torch.sign(targets)
        wrong_dir = (pred_sign != tgt_sign).float()
        weights = 1.0 + wrong_dir * (self.direction_weight - 1)
        return (mse * weights).mean()


async def fetch_data(symbol: str, api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch price data."""
    import aiohttp
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    params = {'apiKey': api_key, 'adjusted': 'true', 'sort': 'asc'}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                raise ValueError(f"API error: {response.status}")
            data = await response.json()
    
    if 'results' not in data or not data['results']:
        raise ValueError(f"No data for {symbol}")
    
    df = pd.DataFrame(data['results'])
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']]
    
    return df


def prepare_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Create features."""
    df = df.copy()
    df['ret_1'] = df['close'].pct_change()
    df['ret_5'] = df['close'].pct_change(5)
    df['vol_20'] = df['ret_1'].rolling(20).std()
    df['mom_10'] = df['close'] / df['close'].shift(10) - 1
    df['rsi'] = 100 - (100 / (1 + df['ret_1'].rolling(14).apply(
        lambda x: x[x > 0].sum() / (-x[x < 0].sum() + 1e-8))))
    df['target'] = df['close'].pct_change(horizon).shift(-horizon)
    return df.dropna()


def train_model(train_X, train_y, val_X, val_y, features, epochs=100, patience=15):
    """Train a simple model."""
    model = nn.Sequential(
        nn.Linear(len(features), 16),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(16, 1)
    )
    
    loss_fn = DirectionalLoss(direction_weight=3.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    
    best_loss = float('inf')
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_X)
        loss = loss_fn(pred, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_X), val_y)
        
        if val_loss < best_loss:
            best_loss = val_loss.item()
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            break
    
    model.load_state_dict(best_state)
    return model


async def walk_forward_validate(symbol: str, horizon: int, api_key: str, 
                                 train_months: int = 24, test_months: int = 1):
    """Run walk-forward validation."""
    
    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD: {symbol} ({horizon}d horizon)")
    print(f"  Train: {train_months}m, Test: {test_months}m")
    print(f"{'='*60}")
    
    # Fetch all data
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=train_months + 24)  # Extra for test periods
    
    df = await fetch_data(symbol, api_key, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    df = prepare_features(df, horizon)
    
    features = ['ret_1', 'ret_5', 'vol_20', 'mom_10', 'rsi']
    
    # Determine periods
    periods = []
    current_test_start = start_date + relativedelta(months=train_months)
    
    while current_test_start + relativedelta(months=test_months) <= end_date:
        periods.append({
            'train_start': current_test_start - relativedelta(months=train_months),
            'train_end': current_test_start,
            'test_start': current_test_start,
            'test_end': current_test_start + relativedelta(months=test_months),
        })
        current_test_start += relativedelta(months=test_months)
    
    print(f"\n  {len(periods)} test periods identified")
    
    # Run each period
    period_results = []
    all_returns = []
    
    for i, period in enumerate(periods):
        train_mask = (df.index >= period['train_start']) & (df.index < period['train_end'])
        test_mask = (df.index >= period['test_start']) & (df.index < period['test_end'])
        
        train_data = df[train_mask]
        test_data = df[test_mask]
        
        if len(train_data) < 100 or len(test_data) < 5:
            continue
        
        # Split train into train/val
        split = int(len(train_data) * 0.8)
        
        train_X = torch.tensor(train_data[features].iloc[:split].values, dtype=torch.float32)
        train_y = torch.tensor(train_data['target'].iloc[:split].values, dtype=torch.float32).unsqueeze(1)
        val_X = torch.tensor(train_data[features].iloc[split:].values, dtype=torch.float32)
        val_y = torch.tensor(train_data['target'].iloc[split:].values, dtype=torch.float32).unsqueeze(1)
        test_X = torch.tensor(test_data[features].values, dtype=torch.float32)
        test_y = test_data['target'].values
        
        # Train
        model = train_model(train_X, train_y, val_X, val_y, features)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            predictions = model(test_X).numpy().flatten()
        
        # Calculate metrics
        dir_acc = np.mean(np.sign(predictions) == np.sign(test_y))
        positions = np.sign(predictions)
        costs = np.abs(np.diff(positions, prepend=0)) * 0.001
        returns = positions * test_y - costs
        all_returns.extend(returns)
        
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 / horizon)
        
        period_results.append({
            'period': f"{period['test_start'].strftime('%Y-%m')}",
            'dir_acc': dir_acc,
            'sharpe': sharpe,
            'n_trades': len(test_y)
        })
        
        status = "✅" if sharpe > 0 else "❌"
        print(f"  {status} {period['test_start'].strftime('%Y-%m')}: Sharpe={sharpe:.2f}, Dir={dir_acc:.2%}")
    
    # Summary
    if period_results:
        sharpes = [p['sharpe'] for p in period_results]
        dir_accs = [p['dir_acc'] for p in period_results]
        
        all_returns = np.array(all_returns)
        overall_sharpe = np.mean(all_returns) / (np.std(all_returns) + 1e-8) * np.sqrt(252 / horizon)
        
        print(f"\n  {'='*50}")
        print(f"  SUMMARY")
        print(f"  {'='*50}")
        print(f"  Total periods:        {len(period_results)}")
        print(f"  Profitable periods:   {sum(s > 0 for s in sharpes)} ({sum(s > 0 for s in sharpes)/len(sharpes):.1%})")
        print(f"  Avg period Sharpe:    {np.mean(sharpes):.2f}")
        print(f"  Overall Sharpe:       {overall_sharpe:.2f}")
        print(f"  Avg Dir Accuracy:     {np.mean(dir_accs):.2%}")
        print(f"  Sharpe Std:           {np.std(sharpes):.2f}")
        
        return {
            'symbol': symbol,
            'horizon': horizon,
            'n_periods': len(period_results),
            'avg_sharpe': np.mean(sharpes),
            'overall_sharpe': overall_sharpe,
            'avg_dir_acc': np.mean(dir_accs),
            'pct_profitable': sum(s > 0 for s in sharpes) / len(sharpes)
        }
    
    return None


async def main():
    api_key = os.environ.get('POLYGON_API_KEY', 'JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY')
    
    print("\n" + "="*70)
    print("  WALK-FORWARD VALIDATION - TOP PERFORMERS")
    print("="*70)
    
    # Test best performers from multi-symbol validation
    tests = [
        ('GLD', 5),   # Best: Sharpe 2.88
        ('SLV', 1),   # Second: Sharpe 2.31
        ('AMD', 5),   # Third: Sharpe 1.29
        ('TSLA', 1),  # Fourth: Sharpe 1.27
    ]
    
    results = []
    for symbol, horizon in tests:
        try:
            result = await walk_forward_validate(symbol, horizon, api_key)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ {symbol}: {e}")
    
    # Final summary
    print("\n" + "="*70)
    print("  FINAL WALK-FORWARD RESULTS")
    print("="*70)
    
    if results:
        print("\n  Symbol   Horizon   Periods   Avg Sharpe   Overall Sharpe   Dir Acc")
        print("  " + "-"*70)
        for r in results:
            status = "✅" if r['overall_sharpe'] > 0.5 else "⚠️" if r['overall_sharpe'] > 0 else "❌"
            print(f"  {status} {r['symbol']:5s}    {r['horizon']}d        {r['n_periods']:2d}        {r['avg_sharpe']:>6.2f}          {r['overall_sharpe']:>6.2f}       {r['avg_dir_acc']:.1%}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(main())
