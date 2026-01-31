#!/usr/bin/env python3
"""
Walk-Forward Retraining Module

Markets change. A model trained on 2019-2022 data may not work in 2024.
This module implements walk-forward validation/retraining:

1. Train on window [t-N, t]
2. Validate on [t, t+M]
3. Move window forward, repeat
4. Track performance over time

This is how professional quant funds operate.

Usage:
    trainer = WalkForwardTrainer(api_key=POLYGON_API_KEY)
    result = await trainer.walk_forward('SPY', train_months=24, test_months=1)
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
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

try:
    from scipy.stats import spearmanr, ttest_1samp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .real_data_trainer import (
    RealDataTrainer,
    TechnicalFeatureExtractor,
    MLPModel,
    LinearModel,
    EarlyStopping,
)


@dataclass
class WalkForwardPeriodResult:
    """Result for a single walk-forward period."""
    period_start: str
    period_end: str
    train_start: str
    train_end: str
    
    directional_accuracy: float
    sharpe_ratio: float
    total_return: float
    n_trades: int


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation result."""
    symbol: str
    model_type: str
    train_window_months: int
    test_window_months: int
    
    # Overall metrics
    periods: List[WalkForwardPeriodResult]
    overall_sharpe: float
    overall_return: float
    overall_dir_acc: float
    
    # Stability
    sharpe_std: float
    min_sharpe: float
    max_sharpe: float
    pct_profitable_periods: float
    
    # Time range
    full_start: str
    full_end: str
    
    def __str__(self):
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  WALK-FORWARD VALIDATION RESULT                                              ║
║  Symbol: {self.symbol:<10}  Model: {self.model_type:<15}                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CONFIGURATION                                                               ║
║    Train Window:   {self.train_window_months:>3} months                      ║
║    Test Window:    {self.test_window_months:>3} months                       ║
║    Total Periods:  {len(self.periods):>3}                                    ║
║    Date Range:     {self.full_start} to {self.full_end}                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  OVERALL PERFORMANCE                                                         ║
║    Avg Sharpe:          {self.overall_sharpe:>6.2f}                          ║
║    Total Return:        {self.overall_return:>6.2%}                          ║
║    Avg Dir Accuracy:    {self.overall_dir_acc:>6.2%}                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  STABILITY                                                                   ║
║    Sharpe Std:          {self.sharpe_std:>6.2f}                              ║
║    Min Sharpe:          {self.min_sharpe:>6.2f}                              ║
║    Max Sharpe:          {self.max_sharpe:>6.2f}                              ║
║    Profitable Periods:  {self.pct_profitable_periods:>6.2%}                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PERIOD BREAKDOWN                                                            ║
"""
    
    def print_periods(self):
        """Print per-period results."""
        print("  Period                 Sharpe    Dir Acc    Return")
        print("  " + "-" * 55)
        for p in self.periods:
            print(f"  {p.period_start} - {p.period_end}   {p.sharpe_ratio:>6.2f}   {p.directional_accuracy:>6.2%}   {p.total_return:>6.2%}")


class WalkForwardTrainer:
    """
    Walk-forward validation and retraining.
    
    This simulates what you would do in production:
    - Retrain monthly (or weekly) on recent data
    - Validate on the next period
    - Track whether the model maintains edge over time
    """
    
    def __init__(
        self,
        api_key: str,
        models_dir: str = "models/walk_forward",
        results_dir: str = "training_results/walk_forward",
        device: str = 'auto'
    ):
        self.api_key = api_key
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if device == 'auto':
            self.device = torch.device(
                'mps' if torch.backends.mps.is_available() else
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)
        
        self.feature_extractor = TechnicalFeatureExtractor()
        self.base_trainer = RealDataTrainer(api_key, models_dir, results_dir, device)
    
    async def walk_forward(
        self,
        symbol: str,
        train_months: int = 24,
        test_months: int = 1,
        model_type: str = 'mlp',
        start_date: str = None,
        end_date: str = None,
        verbose: bool = True
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.
        
        Args:
            symbol: Symbol to validate
            train_months: Months of training data per window
            test_months: Months of test data per window
            model_type: 'linear' or 'mlp'
            start_date: Start of walk-forward (earliest train date)
            end_date: End of walk-forward (latest test date)
            verbose: Print progress
        
        Returns:
            WalkForwardResult with all period results
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"  WALK-FORWARD VALIDATION: {symbol}")
            print(f"  Train Window: {train_months} months, Test Window: {test_months} months")
            print(f"{'='*70}")
        
        # Default dates
        if end_date is None:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_date is None:
            # Start with enough data for first train window + some test windows
            start_date = end_date - relativedelta(months=train_months + 12*2)  # 2 years of testing
        else:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Fetch all data upfront
        if verbose:
            print(f"\n📊 Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        
        df = await self.base_trainer.fetch_data(
            symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if verbose:
            print(f"   Received {len(df):,} bars")
        
        # Extract features once
        features_df, feature_names = self.feature_extractor.extract(df)
        forward_returns = df['close'].pct_change().shift(-1)
        
        combined = pd.concat([features_df, forward_returns.rename('target')], axis=1)
        combined = combined.dropna()
        
        # Determine walk-forward periods
        periods = []
        current_test_start = start_date + relativedelta(months=train_months)
        
        while current_test_start + relativedelta(months=test_months) <= end_date:
            train_start = current_test_start - relativedelta(months=train_months)
            test_end = current_test_start + relativedelta(months=test_months)
            
            periods.append({
                'train_start': train_start,
                'train_end': current_test_start,
                'test_start': current_test_start,
                'test_end': test_end,
            })
            
            current_test_start = test_end  # Non-overlapping test periods
        
        if verbose:
            print(f"\n📅 {len(periods)} walk-forward periods identified")
        
        # Run each period
        period_results = []
        all_returns = []
        
        for i, period in enumerate(periods):
            if verbose:
                print(f"\n--- Period {i+1}/{len(periods)}: Train [{period['train_start'].strftime('%Y-%m')} - {period['train_end'].strftime('%Y-%m')}], Test [{period['test_start'].strftime('%Y-%m')} - {period['test_end'].strftime('%Y-%m')}]")
            
            # Get data for this period
            train_mask = (combined.index >= period['train_start']) & (combined.index < period['train_end'])
            test_mask = (combined.index >= period['test_start']) & (combined.index < period['test_end'])
            
            train_data = combined[train_mask]
            test_data = combined[test_mask]
            
            if len(train_data) < 100 or len(test_data) < 10:
                if verbose:
                    print(f"   Skipping: insufficient data (train={len(train_data)}, test={len(test_data)})")
                continue
            
            # Prepare tensors
            train_X = torch.tensor(train_data[feature_names].values, dtype=torch.float32)
            train_y = torch.tensor(train_data['target'].values, dtype=torch.float32)
            test_X = torch.tensor(test_data[feature_names].values, dtype=torch.float32)
            test_y = test_data['target'].values
            
            # Split train into train/val (80/20)
            split = int(len(train_X) * 0.8)
            data = {
                'train_X': train_X[:split],
                'train_y': train_y[:split],
                'val_X': train_X[split:],
                'val_y': train_y[split:],
                'test_X': test_X,
                'test_y': torch.tensor(test_y, dtype=torch.float32),
            }
            
            # Create and train model
            n_features = len(feature_names)
            if model_type == 'linear':
                model = LinearModel(n_features)
            else:
                model = MLPModel(n_features, hidden_size=32, dropout=0.3)
            
            model, history = self.base_trainer.train_model(
                model, data, epochs=100, patience=15
            )
            
            # Evaluate on test
            model.eval()
            with torch.no_grad():
                predictions = model(data['test_X'].to(self.device)).cpu().numpy()
            
            # Calculate metrics
            dir_acc = np.mean(np.sign(predictions) == np.sign(test_y))
            
            positions = np.sign(predictions)
            costs = np.abs(np.diff(positions, prepend=0)) * 0.001
            returns = positions * test_y - costs
            all_returns.extend(returns)
            
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            total_return = np.prod(1 + returns) - 1
            n_trades = np.sum(np.abs(np.diff(positions, prepend=0)) > 0)
            
            period_result = WalkForwardPeriodResult(
                period_start=period['test_start'].strftime('%Y-%m-%d'),
                period_end=period['test_end'].strftime('%Y-%m-%d'),
                train_start=period['train_start'].strftime('%Y-%m-%d'),
                train_end=period['train_end'].strftime('%Y-%m-%d'),
                directional_accuracy=dir_acc,
                sharpe_ratio=sharpe,
                total_return=total_return,
                n_trades=int(n_trades),
            )
            period_results.append(period_result)
            
            if verbose:
                status = "✅" if sharpe > 0 else "❌"
                print(f"   {status} Sharpe: {sharpe:.2f}, Dir Acc: {dir_acc:.2%}, Return: {total_return:.2%}")
        
        # Calculate overall metrics
        if not period_results:
            raise ValueError("No valid periods")
        
        all_returns = np.array(all_returns)
        sharpes = [p.sharpe_ratio for p in period_results]
        returns = [p.total_return for p in period_results]
        dir_accs = [p.directional_accuracy for p in period_results]
        
        overall_sharpe = np.mean(sharpes)
        overall_return = np.prod([1 + r for r in returns]) - 1
        overall_dir_acc = np.mean(dir_accs)
        
        result = WalkForwardResult(
            symbol=symbol,
            model_type=model_type.upper(),
            train_window_months=train_months,
            test_window_months=test_months,
            periods=period_results,
            overall_sharpe=overall_sharpe,
            overall_return=overall_return,
            overall_dir_acc=overall_dir_acc,
            sharpe_std=np.std(sharpes),
            min_sharpe=np.min(sharpes),
            max_sharpe=np.max(sharpes),
            pct_profitable_periods=np.mean([s > 0 for s in sharpes]),
            full_start=period_results[0].train_start,
            full_end=period_results[-1].period_end,
        )
        
        if verbose:
            print(result)
            result.print_periods()
        
        # Save result
        result_path = self.results_dir / f"wf_{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_path, 'w') as f:
            json.dump({
                'symbol': result.symbol,
                'model_type': result.model_type,
                'train_window_months': result.train_window_months,
                'test_window_months': result.test_window_months,
                'overall_sharpe': result.overall_sharpe,
                'overall_return': result.overall_return,
                'overall_dir_acc': result.overall_dir_acc,
                'sharpe_std': result.sharpe_std,
                'min_sharpe': result.min_sharpe,
                'max_sharpe': result.max_sharpe,
                'pct_profitable_periods': result.pct_profitable_periods,
                'periods': [
                    {
                        'period_start': p.period_start,
                        'period_end': p.period_end,
                        'sharpe_ratio': p.sharpe_ratio,
                        'directional_accuracy': p.directional_accuracy,
                        'total_return': p.total_return,
                    }
                    for p in result.periods
                ]
            }, f, indent=2)
        
        return result


async def run_walk_forward(
    symbol: str = 'SPY',
    train_months: int = 24,
    test_months: int = 1,
    model_type: str = 'mlp',
    api_key: str = None
) -> WalkForwardResult:
    """Convenience function for walk-forward validation."""
    
    if api_key is None:
        api_key = os.environ.get('POLYGON_API_KEY')
    
    if not api_key:
        raise ValueError("POLYGON_API_KEY not set")
    
    trainer = WalkForwardTrainer(api_key=api_key)
    return await trainer.walk_forward(symbol, train_months, test_months, model_type)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument('--symbol', type=str, default='SPY')
    parser.add_argument('--train-months', type=int, default=24)
    parser.add_argument('--test-months', type=int, default=1)
    parser.add_argument('--model', type=str, default='mlp', choices=['linear', 'mlp'])
    
    args = parser.parse_args()
    
    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        print("❌ POLYGON_API_KEY not set")
        sys.exit(1)
    
    asyncio.run(run_walk_forward(args.symbol, args.train_months, args.test_months, args.model, api_key))
