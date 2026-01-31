"""
Model Validator - The ONLY Thing That Matters
==============================================

Do these models have predictive power?

This is the MOST CRITICAL module in the entire codebase.
Without evidence of statistically significant alpha on out-of-sample data,
everything else is pointless. You're just building a fancy car that doesn't run.

References:
- Lopez de Prado, "Advances in Financial Machine Learning" (Ch. 11-12)
- Harvey et al., "...and the Cross-Section of Expected Returns" (multiple testing)
- Bailey & Lopez de Prado, "The Deflated Sharpe Ratio"
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.stats import spearmanr, ttest_1samp, pearsonr

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Results from rigorous out-of-sample validation.
    
    This is the proof that a model works (or doesn't).
    """
    model_name: str
    train_period: Tuple[str, str]
    test_period: Tuple[str, str]
    n_train_samples: int
    n_test_samples: int
    
    # Prediction Quality
    directional_accuracy: float  # % of times sign(pred) == sign(actual)
    correlation: float           # Pearson correlation pred vs actual
    ic_mean: float              # Mean Information Coefficient (rank corr)
    ic_std: float               # IC standard deviation
    ic_ir: float                # IC Information Ratio = ic_mean / ic_std
    hit_rate: float             # % of predictions with correct direction
    
    # Trading Metrics (if we traded on predictions)
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    
    # Statistical Significance
    t_statistic: float
    p_value: float
    is_significant: bool  # p < 0.05 AND sharpe > 0
    
    # Robustness Checks
    sharpe_by_year: Dict[int, float] = field(default_factory=dict)
    sharpe_by_regime: Dict[str, float] = field(default_factory=dict)
    
    # Timestamp
    validated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'train_period': self.train_period,
            'test_period': self.test_period,
            'n_train_samples': self.n_train_samples,
            'n_test_samples': self.n_test_samples,
            'directional_accuracy': self.directional_accuracy,
            'correlation': self.correlation,
            'ic_mean': self.ic_mean,
            'ic_std': self.ic_std,
            'ic_ir': self.ic_ir,
            'hit_rate': self.hit_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'annualized_volatility': self.annualized_volatility,
            't_statistic': self.t_statistic,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'sharpe_by_year': self.sharpe_by_year,
            'sharpe_by_regime': self.sharpe_by_regime,
            'validated_at': self.validated_at,
        }
    
    @property
    def grade(self) -> str:
        """Letter grade for model performance."""
        if not self.is_significant:
            return "F"
        if self.sharpe_ratio >= 2.0 and self.ic_ir >= 0.5:
            return "A"
        if self.sharpe_ratio >= 1.5 and self.ic_ir >= 0.3:
            return "B"
        if self.sharpe_ratio >= 1.0 and self.ic_ir >= 0.2:
            return "C"
        if self.sharpe_ratio >= 0.5:
            return "D"
        return "F"
    
    def __str__(self) -> str:
        """Pretty print validation results."""
        status = "✅ SIGNIFICANT" if self.is_significant else "❌ NOT SIGNIFICANT"
        grade = self.grade
        
        # Format sharpe by year
        yearly = ""
        for year, sharpe in sorted(self.sharpe_by_year.items()):
            yearly += f"\n║    {year}: {sharpe:>6.2f}"
        
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MODEL VALIDATION REPORT                                                     ║
║  Model: {self.model_name:<66} ║
║  Grade: {grade:<66} ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DATA SPLIT                                                                  ║
║    Train: {self.train_period[0]} to {self.train_period[1]} ({self.n_train_samples:,} samples)             
║    Test:  {self.test_period[0]} to {self.test_period[1]} ({self.n_test_samples:,} samples)             
╠══════════════════════════════════════════════════════════════════════════════╣
║  PREDICTION QUALITY                                                          ║
║    Directional Accuracy: {self.directional_accuracy:>8.2%}                                       ║
║    Hit Rate:             {self.hit_rate:>8.2%}                                       ║
║    Correlation:          {self.correlation:>8.4f}                                       ║
║    IC Mean:              {self.ic_mean:>8.4f}                                       ║
║    IC Std:               {self.ic_std:>8.4f}                                       ║
║    IC IR:                {self.ic_ir:>8.2f}                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TRADING PERFORMANCE (simulated, 10bps costs)                                ║
║    Sharpe Ratio:         {self.sharpe_ratio:>8.2f}                                         ║
║    Sortino Ratio:        {self.sortino_ratio:>8.2f}                                         ║
║    Calmar Ratio:         {self.calmar_ratio:>8.2f}                                         ║
║    Max Drawdown:         {self.max_drawdown:>8.2%}                                       ║
║    Win Rate:             {self.win_rate:>8.2%}                                       ║
║    Profit Factor:        {self.profit_factor:>8.2f}                                         ║
║    Total Return:         {self.total_return:>8.2%}                                       ║
║    Annual Return:        {self.annualized_return:>8.2%}                                       ║
║    Annual Volatility:    {self.annualized_volatility:>8.2%}                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  STATISTICAL SIGNIFICANCE                                                    ║
║    t-statistic:          {self.t_statistic:>8.2f}                                         ║
║    p-value:              {self.p_value:>8.4f}                                       ║
║    Status:               {status:>30}               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ROBUSTNESS (Sharpe by Year)                                                 ║{yearly if yearly else chr(10) + '║    No yearly breakdown available                                             ║'}
╠══════════════════════════════════════════════════════════════════════════════╣
║  ROBUSTNESS (Sharpe by Regime)                                               ║
║    High Volatility:      {self.sharpe_by_regime.get('high_vol', 0):>8.2f}                                         ║
║    Low Volatility:       {self.sharpe_by_regime.get('low_vol', 0):>8.2f}                                         ║
║    Trending:             {self.sharpe_by_regime.get('trending', 0):>8.2f}                                         ║
║    Mean Reverting:       {self.sharpe_by_regime.get('mean_revert', 0):>8.2f}                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


class ModelValidator:
    """
    Rigorous out-of-sample validation for financial ML models.
    
    This is the ONLY thing that matters. Without this, you have nothing.
    
    Features:
    - Walk-forward validation (no lookahead bias)
    - Transaction cost modeling
    - Statistical significance testing
    - Regime-based robustness checks
    - Yearly performance breakdown
    
    Reference: Lopez de Prado, "Advances in Financial Machine Learning"
    """
    
    def __init__(
        self,
        transaction_cost_bps: float = 10.0,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        results_dir: str = "validation_results",
        periods_per_year: int = 252
    ):
        """
        Initialize validator.
        
        Args:
            transaction_cost_bps: Transaction cost in basis points (10 = 0.1%)
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation (test = 1 - train - val)
            results_dir: Directory to save results
            periods_per_year: Trading periods per year (252 for daily)
        """
        self.transaction_cost = transaction_cost_bps / 10000
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.periods_per_year = periods_per_year
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        self._all_results: List[ValidationResult] = []
    
    def validate_model(
        self,
        model: nn.Module,
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'returns',
        model_name: str = 'unnamed',
        sequence_length: int = 60,
        device: str = 'auto'
    ) -> ValidationResult:
        """
        Run comprehensive out-of-sample validation.
        
        Args:
            model: Trained PyTorch model
            data: DataFrame with features and target, DatetimeIndex required
            feature_cols: List of feature column names
            target_col: Name of target column (forward returns)
            model_name: Name for reporting
            sequence_length: Input sequence length for time series models
            device: Device for inference ('auto', 'cpu', 'cuda', 'mps')
        
        Returns:
            ValidationResult with all metrics
        """
        print(f"\n{'='*70}")
        print(f"  VALIDATING: {model_name}")
        print(f"{'='*70}")
        
        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        model = model.to(device)
        model.eval()
        
        # 1. Split data (walk-forward style - no shuffling!)
        n = len(data)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        print(f"\n📊 Data Split:")
        print(f"   Train: {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data):,} samples)")
        print(f"   Val:   {val_data.index[0].date()} to {val_data.index[-1].date()} ({len(val_data):,} samples)")
        print(f"   Test:  {test_data.index[0].date()} to {test_data.index[-1].date()} ({len(test_data):,} samples)")
        
        if len(test_data) < 60:
            raise ValueError(f"Not enough test data: {len(test_data)} samples")
        
        # 2. Generate predictions on TEST set only
        print(f"\n🧠 Generating predictions...")
        predictions = self._generate_predictions(
            model=model,
            data=test_data,
            feature_cols=feature_cols,
            sequence_length=sequence_length,
            device=device
        )
        
        # Align actuals with predictions
        actuals = test_data[target_col].values[sequence_length-1:]
        dates = test_data.index[sequence_length-1:]
        
        # Ensure same length
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]
        dates = dates[:min_len]
        
        print(f"   Generated {len(predictions):,} predictions")
        
        # 3. Calculate all metrics
        print(f"\n📈 Calculating metrics...")
        result = self._calculate_all_metrics(
            predictions=predictions,
            actuals=actuals,
            dates=dates,
            model_name=model_name,
            train_period=(str(train_data.index[0].date()), str(train_data.index[-1].date())),
            test_period=(str(test_data.index[0].date()), str(test_data.index[-1].date())),
            n_train=len(train_data),
            n_test=len(test_data)
        )
        
        # 4. Save results
        self._save_results(result, predictions, actuals, dates)
        self._all_results.append(result)
        
        print(f"\n✅ Validation complete for {model_name}")
        
        return result
    
    def _generate_predictions(
        self,
        model: nn.Module,
        data: pd.DataFrame,
        feature_cols: List[str],
        sequence_length: int,
        device: str
    ) -> np.ndarray:
        """Generate predictions using the model."""
        
        features = data[feature_cols].values
        predictions = []
        
        with torch.no_grad():
            for i in range(sequence_length - 1, len(features)):
                # Get sequence
                seq = features[i - sequence_length + 1:i + 1]
                
                # Convert to tensor
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Handle different input formats
                try:
                    # Try standard forward
                    output = model(x)
                except Exception:
                    # Try with flattened input for simpler models
                    try:
                        x_flat = x.view(1, -1)
                        output = model(x_flat)
                    except Exception:
                        # Try with just last timestep
                        x_last = x[:, -1, :]
                        output = model(x_last)
                
                # Extract prediction from various output formats
                pred = self._extract_prediction(output)
                predictions.append(pred)
        
        return np.array(predictions)
    
    def _extract_prediction(self, output: Any) -> float:
        """Extract scalar prediction from model output."""
        
        if isinstance(output, dict):
            # Handle dict outputs (TFT, DeepAR, etc.)
            for key in ['prediction', 'mean', 'forecast', 'predictions', 'output']:
                if key in output:
                    val = output[key]
                    if isinstance(val, torch.Tensor):
                        return val.cpu().numpy().flatten()[0]
                    return float(val)
            # Fall back to first value
            val = list(output.values())[0]
            if isinstance(val, torch.Tensor):
                return val.cpu().numpy().flatten()[0]
            return float(val)
        
        elif isinstance(output, tuple):
            val = output[0]
            if isinstance(val, torch.Tensor):
                return val.cpu().numpy().flatten()[0]
            return float(val)
        
        elif isinstance(output, torch.Tensor):
            return output.cpu().numpy().flatten()[0]
        
        else:
            return float(output)
    
    def _calculate_all_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: pd.DatetimeIndex,
        model_name: str,
        train_period: Tuple[str, str],
        test_period: Tuple[str, str],
        n_train: int,
        n_test: int
    ) -> ValidationResult:
        """Calculate comprehensive validation metrics."""
        
        # Clean data
        mask = ~(np.isnan(predictions) | np.isnan(actuals) | np.isinf(predictions) | np.isinf(actuals))
        predictions = predictions[mask]
        actuals = actuals[mask]
        dates = dates[mask]
        
        if len(predictions) < 20:
            raise ValueError(f"Not enough valid predictions: {len(predictions)}")
        
        # ============ Prediction Quality ============
        
        # Directional accuracy
        directional_accuracy = np.mean(np.sign(predictions) == np.sign(actuals))
        
        # Hit rate (slightly different - prediction != 0)
        nonzero_mask = predictions != 0
        if nonzero_mask.sum() > 0:
            hit_rate = np.mean(np.sign(predictions[nonzero_mask]) == np.sign(actuals[nonzero_mask]))
        else:
            hit_rate = 0.5
        
        # Correlation
        correlation, _ = pearsonr(predictions, actuals)
        
        if np.isnan(correlation):
            correlation = 0.0
        
        # Information Coefficient (rolling rank correlation)
        ic_values = []
        window = 20
        for i in range(window, len(predictions)):
            chunk_pred = predictions[i-window:i]
            chunk_actual = actuals[i-window:i]
            ic, _ = spearmanr(chunk_pred, chunk_actual)
            if not np.isnan(ic):
                ic_values.append(ic)
        
        ic_mean = np.mean(ic_values) if ic_values else 0
        ic_std = np.std(ic_values) if ic_values else 1
        ic_ir = ic_mean / ic_std if ic_std > 0.001 else 0
        
        # ============ Trading Performance ============
        
        # Position sizing: sign of prediction
        positions = np.sign(predictions)
        
        # Transaction costs on position changes
        position_changes = np.abs(np.diff(positions, prepend=0))
        costs = position_changes * self.transaction_cost
        
        # Strategy returns
        strategy_returns = positions * actuals - costs
        
        # Performance metrics
        sharpe = self._sharpe_ratio(strategy_returns)
        sortino = self._sortino_ratio(strategy_returns)
        max_dd = self._max_drawdown(strategy_returns)
        calmar = sharpe / abs(max_dd) if abs(max_dd) > 0.001 else 0
        
        # Win rate and profit factor
        winning = strategy_returns[strategy_returns > 0]
        losing = strategy_returns[strategy_returns < 0]
        win_rate = len(winning) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        gross_profit = winning.sum() if len(winning) > 0 else 0
        gross_loss = abs(losing.sum()) if len(losing) > 0 else 0.001
        profit_factor = gross_profit / gross_loss if gross_loss > 0.001 else 0
        
        # Returns
        total_return = np.prod(1 + strategy_returns) - 1
        n_years = len(strategy_returns) / self.periods_per_year
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        annualized_vol = np.std(strategy_returns) * np.sqrt(self.periods_per_year)
        
        # ============ Statistical Significance ============
        
        t_stat, p_value = ttest_1samp(strategy_returns, 0)
        
        is_significant = p_value < 0.05 and sharpe > 0
        
        # ============ Robustness Checks ============
        
        # Sharpe by year
        df = pd.DataFrame({'returns': strategy_returns}, index=dates)
        sharpe_by_year = {}
        for year in df.index.year.unique():
            year_rets = df[df.index.year == year]['returns'].values
            if len(year_rets) >= 20:
                sharpe_by_year[int(year)] = self._sharpe_ratio(year_rets)
        
        # Sharpe by regime
        rolling_vol = pd.Series(actuals).rolling(20).std()
        rolling_return = pd.Series(actuals).rolling(20).mean()
        
        vol_median = rolling_vol.median()
        ret_median = rolling_return.median()
        
        high_vol = (rolling_vol > vol_median).values
        low_vol = (rolling_vol <= vol_median).values
        trending = (rolling_return.abs() > ret_median).values
        mean_revert = (rolling_return.abs() <= ret_median).values
        
        sharpe_by_regime = {
            'high_vol': self._sharpe_ratio(strategy_returns[high_vol]) if high_vol.sum() >= 20 else 0,
            'low_vol': self._sharpe_ratio(strategy_returns[low_vol]) if low_vol.sum() >= 20 else 0,
            'trending': self._sharpe_ratio(strategy_returns[trending]) if trending.sum() >= 20 else 0,
            'mean_revert': self._sharpe_ratio(strategy_returns[mean_revert]) if mean_revert.sum() >= 20 else 0,
        }
        
        return ValidationResult(
            model_name=model_name,
            train_period=train_period,
            test_period=test_period,
            n_train_samples=n_train,
            n_test_samples=n_test,
            directional_accuracy=directional_accuracy,
            correlation=correlation,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            hit_rate=hit_rate,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_vol,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            sharpe_by_year=sharpe_by_year,
            sharpe_by_regime=sharpe_by_regime
        )
    
    def _sharpe_ratio(self, returns: np.ndarray) -> float:
        """Annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0
        std = np.std(returns)
        if std < 1e-10:
            return 0
        return np.mean(returns) / std * np.sqrt(self.periods_per_year)
    
    def _sortino_ratio(self, returns: np.ndarray) -> float:
        """Annualized Sortino ratio."""
        if len(returns) < 2:
            return 0
        downside = returns[returns < 0]
        if len(downside) < 2:
            return self._sharpe_ratio(returns)
        downside_std = np.std(downside)
        if downside_std < 1e-10:
            return 0
        return np.mean(returns) / downside_std * np.sqrt(self.periods_per_year)
    
    def _max_drawdown(self, returns: np.ndarray) -> float:
        """Maximum drawdown from peak."""
        if len(returns) < 2:
            return 0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _save_results(
        self,
        result: ValidationResult,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: pd.DatetimeIndex
    ):
        """Save validation results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = result.model_name.replace(' ', '_').replace('/', '_')
        
        # Save summary JSON
        summary_path = self.results_dir / f"{safe_name}_{timestamp}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        # Save predictions CSV
        pred_df = pd.DataFrame({
            'date': dates,
            'prediction': predictions,
            'actual': actuals,
            'position': np.sign(predictions),
            'strategy_return': np.sign(predictions) * actuals
        })
        pred_path = self.results_dir / f"{safe_name}_{timestamp}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        
        print(f"\n📁 Results saved:")
        print(f"   {summary_path}")
        print(f"   {pred_path}")
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all validation results."""
        if not self._all_results:
            return pd.DataFrame()
        
        rows = []
        for r in self._all_results:
            rows.append({
                'Model': r.model_name,
                'Sharpe': r.sharpe_ratio,
                'Sortino': r.sortino_ratio,
                'MaxDD': r.max_drawdown,
                'IC_IR': r.ic_ir,
                'DirAcc': r.directional_accuracy,
                'WinRate': r.win_rate,
                'p-value': r.p_value,
                'Significant': r.is_significant,
                'Grade': r.grade
            })
        
        return pd.DataFrame(rows)
    
    def print_comparison(self):
        """Print comparison of all validated models."""
        if not self._all_results:
            print("No validation results to compare.")
            return
        
        print("\n" + "="*90)
        print("  MODEL COMPARISON SUMMARY")
        print("="*90)
        print(f"{'Model':<25} {'Sharpe':>8} {'IC IR':>8} {'Dir Acc':>8} {'MaxDD':>8} {'p-value':>8} {'Status':>12}")
        print("-"*90)
        
        for r in sorted(self._all_results, key=lambda x: x.sharpe_ratio, reverse=True):
            status = "✅ SIG" if r.is_significant else "❌ NOSIG"
            print(f"{r.model_name:<25} {r.sharpe_ratio:>8.2f} {r.ic_ir:>8.2f} {r.directional_accuracy:>8.2%} {r.max_drawdown:>8.2%} {r.p_value:>8.4f} {status:>12}")
        
        print("="*90)
        
        # Best model
        best = max(self._all_results, key=lambda x: x.sharpe_ratio if x.is_significant else -999)
        if best.is_significant:
            print(f"\n🏆 Best Model: {best.model_name} (Sharpe: {best.sharpe_ratio:.2f}, Grade: {best.grade})")
        else:
            print(f"\n⚠️  No models showed statistically significant alpha!")


def train_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray = None, y_val: np.ndarray = None,
                epochs: int = 50, lr: float = 0.001, sequence_length: int = 60,
                patience: int = 15, verbose: bool = True) -> nn.Module:
    """
    Train a model with early stopping to prevent overfitting.
    
    Args:
        model: PyTorch model to train
        X_train: Training features [samples, features]
        y_train: Training targets [samples]
        X_val: Validation features (for early stopping)
        y_val: Validation targets
        epochs: Maximum number of training epochs
        lr: Learning rate
        sequence_length: Sequence length for sequential models
        patience: Early stopping patience
        verbose: Print training progress
    
    Returns:
        Trained model
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    # Create training sequences
    X_seq = []
    y_seq = []
    for i in range(sequence_length, len(X_train)):
        X_seq.append(X_train[i-sequence_length:i])
        y_seq.append(y_train[i])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Create validation tensors if provided
    if X_val is not None and y_val is not None:
        X_val_seq = []
        y_val_seq = []
        for i in range(sequence_length, len(X_val)):
            X_val_seq.append(X_val[i-sequence_length:i])
            y_val_seq.append(y_val[i])
        X_val_tensor = torch.tensor(np.array(X_val_seq), dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(np.array(y_val_seq), dtype=torch.float32).unsqueeze(1).to(device)
    
    # Create dataset with dropout for regularization
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            
            try:
                output = model(batch_X)
                if output.dim() > 1:
                    output = output[:, -1] if output.shape[1] > 1 else output.squeeze(-1)
                if output.dim() == 1:
                    output = output.unsqueeze(1)
                
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_train_loss += loss.item()
            except Exception as e:
                # Try flattened input
                try:
                    batch_X_flat = batch_X.view(batch_X.size(0), -1)
                    output = model(batch_X_flat)
                    if output.dim() == 1:
                        output = output.unsqueeze(1)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    total_train_loss += loss.item()
                except:
                    continue
        
        avg_train_loss = total_train_loss / len(loader)
        
        # Validation
        model.eval()
        if X_val is not None and len(X_val_seq) > 0:
            with torch.no_grad():
                try:
                    val_output = model(X_val_tensor)
                    if val_output.dim() > 1:
                        val_output = val_output[:, -1] if val_output.shape[1] > 1 else val_output.squeeze(-1)
                    if val_output.dim() == 1:
                        val_output = val_output.unsqueeze(1)
                    val_loss = criterion(val_output, y_val_tensor).item()
                except:
                    X_val_flat = X_val_tensor.view(X_val_tensor.size(0), -1)
                    val_output = model(X_val_flat)
                    if val_output.dim() == 1:
                        val_output = val_output.unsqueeze(1)
                    val_loss = criterion(val_output, y_val_tensor).item()
            
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"   Early stopping at epoch {epoch+1}")
                break
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Train: {avg_train_loss:.6f}, Val: {val_loss:.6f}")
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.6f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    return model


def run_full_validation(train_epochs: int = 100):
    """
    Run validation on all models with synthetic data.
    
    This demonstrates the COMPLETE pipeline:
    1. Generate synthetic data with known signal
    2. TRAIN models on training data
    3. Validate on HELD-OUT test data
    4. Report if models learned the signal
    
    Args:
        train_epochs: Number of epochs to train each model
    """
    print("\n" + "="*70)
    print("  KYPERIAN MODEL VALIDATION SUITE")
    print("  Proving Models Work (or Don't)")
    print("="*70)
    
    # Create validator
    validator = ModelValidator(
        transaction_cost_bps=10,
        results_dir="validation_results"
    )
    
    # Generate synthetic data WITH KNOWN PREDICTIVE SIGNAL
    print("\n⚠️  Using SYNTHETIC data with KNOWN PREDICTIVE signal!")
    print("   Features at time t predict returns at time t+1")
    print("   If models work, they MUST find this signal.")
    
    np.random.seed(42)
    dates = pd.date_range('2019-01-01', '2024-12-31', freq='B')
    n = len(dates)
    
    # Generate features
    features = np.random.randn(n, 35)
    
    # Generate FORWARD returns - features at t predict return at t+1
    # This is the CORRECT structure for testing predictive power
    signal = (0.05 * features[:, 0] +   # 5% weight on feature 0
              0.03 * features[:, 1] +   # 3% weight on feature 1
              0.02 * features[:, 2])    # 2% weight on feature 2
    noise = np.random.randn(n) * 0.01   # 1% daily noise (SNR = ~10:1)
    
    # SHIFT signal forward - features[t] -> returns[t+1]
    forward_returns = np.zeros(n)
    forward_returns[1:] = signal[:-1] + noise[1:]  # returns[t+1] = f(features[t]) + noise
    
    feature_cols = [f'feature_{i}' for i in range(35)]
    data = pd.DataFrame(features, columns=feature_cols, index=dates)
    data['returns'] = forward_returns
    
    # Calculate theoretical max Sharpe (if we knew the true signal)
    true_signal_returns = forward_returns[1:]  # Skip first NaN
    true_sharpe = np.sqrt(252) * np.abs(true_signal_returns.mean()) / true_signal_returns.std()
    print(f"\n📊 Data: {len(data):,} samples from {dates[0].date()} to {dates[-1].date()}")
    print(f"🎯 Theoretical Max Sharpe (if perfect signal recovery): {true_sharpe:.2f}")
    print(f"   Signal-to-Noise Ratio: ~10:1 (very learnable)")
    
    # Define models to test
    models_to_test = []
    
    # Simple baseline model (should work if signal is learnable)
    class SimpleLinear(nn.Module):
        """Baseline: Simple linear model"""
        def __init__(self, input_size=35):
            super().__init__()
            self.fc = nn.Linear(input_size, 1)
        
        def forward(self, x):
            if x.dim() == 3:
                x = x[:, -1, :]  # Take last timestep
            return self.fc(x)
    
    class SimpleMLP(nn.Module):
        """Baseline: 2-layer MLP"""
        def __init__(self, input_size=35):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            if x.dim() == 3:
                x = x[:, -1, :]
            return self.fc(x)
    
    class SimpleLSTM(nn.Module):
        """Baseline: Simple LSTM"""
        def __init__(self, input_size=35, hidden_size=64):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(0)
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
    
    models_to_test = [
        ('LinearBaseline', SimpleLinear(35)),
        ('MLP', SimpleMLP(35)),
        ('LSTM', SimpleLSTM(35)),
    ]
    
    # Try to add advanced models
    try:
        from ..ml.torch_models.advanced import (
            NBeatsV2, NBeatsConfig,
        )
        # N-BEATS expects univariate time series, not multi-feature
        # models_to_test.append(('N-BEATS', NBeatsV2(NBeatsConfig(context_length=60, prediction_length=1))))
        print("\n💡 Note: N-BEATS/N-HiTS are univariate models, using MLP/LSTM for multi-feature")
    except ImportError as e:
        print(f"\n⚠️  Could not import advanced models: {e}")
    
    # Create proper train/val splits that don't overlap with test
    # Validator uses 70/15/15 split, so we train on first 70%, validate on next 15%
    train_size = int(len(data) * 0.70)
    val_size = int(len(data) * 0.15)
    
    X_full = data[feature_cols].values
    y_full = data['returns'].values
    
    X_train = X_full[:train_size]
    y_train = y_full[:train_size]
    X_val = X_full[train_size:train_size + val_size]
    y_val = y_full[train_size:train_size + val_size]
    
    print(f"\n📊 Data Splits:")
    print(f"   Train: {train_size} samples (epochs 1-{train_size})")
    print(f"   Val:   {val_size} samples (for early stopping)")
    print(f"   Test:  {len(data) - train_size - val_size} samples (HELD OUT)")
    
    # Train and validate each model
    results = []
    
    for model_name, model in models_to_test:
        print(f"\n{'='*70}")
        print(f"  TRAINING & VALIDATING: {model_name}")
        print(f"{'='*70}")
        
        try:
            # Train the model with early stopping
            print(f"\n🏋️  Training {model_name} (max {train_epochs} epochs, early stopping)...")
            trained_model = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=train_epochs,
                lr=0.001,
                sequence_length=60,
                patience=15,
                verbose=True
            )
            
            # Validate on held-out data
            result = validator.validate_model(
                model=trained_model,
                data=data,
                feature_cols=feature_cols,
                target_col='returns',
                model_name=model_name,
                sequence_length=60
            )
            print(result)
            results.append(result)
            
        except Exception as e:
            print(f"\n❌ {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison
    if results:
        validator.print_comparison()
        
        # Additional analysis
        print("\n" + "="*70)
        print("  VALIDATION INTERPRETATION")
        print("="*70)
        
        sig_models = [r for r in results if r.is_significant]
        if sig_models:
            print(f"\n✅ {len(sig_models)}/{len(results)} models showed SIGNIFICANT alpha!")
            best = max(sig_models, key=lambda x: x.sharpe_ratio)
            print(f"🏆 Best: {best.model_name} (Sharpe: {best.sharpe_ratio:.2f})")
            print(f"\n   This proves the model architecture CAN learn predictive signals.")
            print(f"   Next: Train on REAL market data and validate.")
        else:
            print(f"\n❌ NO models showed significant alpha on synthetic data!")
            print(f"   This is a CRITICAL issue:")
            print(f"   - The synthetic data has a KNOWN signal (SNR ~10:1)")
            print(f"   - If models can't learn this, they won't work on real data")
            print(f"   - Check: model architecture, training process, hyperparameters")
    
    return validator


async def run_validation_on_real_data(
    symbols: List[str] = ['SPY'],
    lookback_days: int = 1500
):
    """
    Run validation with real market data from Polygon.
    
    Args:
        symbols: List of symbols to validate on
        lookback_days: Days of historical data to fetch
    """
    import os
    
    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        print("❌ POLYGON_API_KEY not set!")
        print("   Set it with: export POLYGON_API_KEY='your-key'")
        return
    
    print("\n" + "="*70)
    print("  KYPERIAN MODEL VALIDATION - REAL DATA")
    print("="*70)
    
    try:
        from ..ml.torch_models import PolygonDataFetcher, TechnicalFeatureExtractor
    except ImportError:
        print("❌ Could not import data pipeline modules")
        return
    
    # Fetch data
    print(f"\n📊 Fetching {lookback_days} days of data for {symbols}...")
    fetcher = PolygonDataFetcher(api_key)
    extractor = TechnicalFeatureExtractor()
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"  Symbol: {symbol}")
        print(f"{'='*50}")
        
        # Get historical data
        bars = await fetcher.fetch(symbol, lookback_days=lookback_days)
        
        if not bars:
            print(f"   ❌ No data for {symbol}")
            continue
        
        # Convert to DataFrame
        ohlcv = np.array([[b.open, b.high, b.low, b.close, b.volume] for b in bars])
        dates = pd.DatetimeIndex([b.timestamp for b in bars])
        
        # Extract features
        features, feature_names = extractor.extract(ohlcv)
        
        # Calculate forward returns (target)
        closes = ohlcv[:, 3]
        forward_returns = np.diff(np.log(closes), prepend=np.nan)
        forward_returns = np.roll(forward_returns, -1)  # Shift forward
        forward_returns[-1] = 0
        
        # Create DataFrame
        # Align lengths
        min_len = min(len(features), len(dates))
        data = pd.DataFrame(
            features[:min_len],
            columns=feature_names,
            index=dates[:min_len]
        )
        data['returns'] = forward_returns[:min_len]
        
        # Drop NaN rows
        data = data.dropna()
        
        print(f"   ✅ {len(data):,} samples with {len(feature_names)} features")
        
        # Now validate models
        validator = ModelValidator(transaction_cost_bps=10)
        
        try:
            from ..ml.torch_models.advanced import (
                NBeatsV2, NBeatsConfig,
                NHiTSV2, NHiTSConfig,
            )
            
            models = [
                ('N-BEATS', NBeatsV2(NBeatsConfig(context_length=60, prediction_length=1))),
                ('N-HiTS', NHiTSV2(NHiTSConfig(context_length=60, prediction_length=1))),
            ]
            
            for model_name, model in models:
                try:
                    result = validator.validate_model(
                        model=model,
                        data=data,
                        feature_cols=feature_names,
                        target_col='returns',
                        model_name=f"{model_name}_{symbol}"
                    )
                    print(result)
                except Exception as e:
                    print(f"   ❌ {model_name} failed: {e}")
        
        except ImportError as e:
            print(f"   ❌ Could not import models: {e}")
        
        validator.print_comparison()


if __name__ == "__main__":
    # Run with synthetic data by default
    run_full_validation()
