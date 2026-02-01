"""
WALK-FORWARD VALIDATION WITH PURGING AND EMBARGO
==================================================
Implementation following Lopez de Prado's AFML methodology.

This is the GOLD STANDARD for time-series ML validation:
- Expanding or rolling window training
- Purge gap to prevent label leakage
- Embargo period to prevent information leakage
- Realistic transaction costs

A Sharpe > 3.0 with this validation indicates bugs.
Expected realistic Sharpe: 1.0 - 1.5
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

from .config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """One split in walk-forward validation."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_indices: np.ndarray
    test_indices: np.ndarray
    purge_indices: np.ndarray
    embargo_indices: np.ndarray


class WalkForwardValidator:
    """
    Walk-forward validation with purging and embargo.
    
    This implements the correct methodology from AFML Ch.7.
    
    Parameters:
    -----------
    train_size : int
        Number of days for training (e.g., 756 = ~3 years)
    test_size : int
        Number of days for testing (e.g., 63 = ~3 months)
    purge_size : int
        Days to purge between train and test (default: 5)
    embargo_size : int
        Days to embargo after test (default: 5)
    expanding : bool
        If True, training window expands. If False, rolling window.
    """
    
    def __init__(
        self,
        train_size: int = 756,
        test_size: int = 63,
        purge_size: int = 5,
        embargo_size: int = 5,
        expanding: bool = True
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.purge_size = purge_size
        self.embargo_size = embargo_size
        self.expanding = expanding
    
    def split(
        self,
        data: pd.DataFrame,
        dates: pd.DatetimeIndex = None
    ) -> List[WalkForwardSplit]:
        """
        Generate walk-forward splits with purging and embargo.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to split
        dates : pd.DatetimeIndex, optional
            Date index (uses data.index if not provided)
        
        Returns:
        --------
        List of WalkForwardSplit objects
        """
        if dates is None:
            dates = data.index
        
        n = len(dates)
        splits = []
        
        # Calculate number of splits possible
        first_test_start = self.train_size + self.purge_size
        step = self.test_size + self.purge_size + self.embargo_size
        
        current_pos = first_test_start
        split_num = 0
        
        while current_pos + self.test_size <= n:
            # Training indices
            if self.expanding:
                train_start_idx = 0
            else:
                train_start_idx = max(0, current_pos - self.purge_size - self.train_size)
            
            train_end_idx = current_pos - self.purge_size
            
            # Purge indices
            purge_start_idx = train_end_idx
            purge_end_idx = current_pos
            
            # Test indices
            test_start_idx = current_pos
            test_end_idx = min(current_pos + self.test_size, n)
            
            # Embargo indices
            embargo_start_idx = test_end_idx
            embargo_end_idx = min(test_end_idx + self.embargo_size, n)
            
            split = WalkForwardSplit(
                train_start=dates[train_start_idx],
                train_end=dates[train_end_idx - 1],
                test_start=dates[test_start_idx],
                test_end=dates[test_end_idx - 1],
                train_indices=np.arange(train_start_idx, train_end_idx),
                test_indices=np.arange(test_start_idx, test_end_idx),
                purge_indices=np.arange(purge_start_idx, purge_end_idx),
                embargo_indices=np.arange(embargo_start_idx, embargo_end_idx)
            )
            
            splits.append(split)
            
            logger.debug(
                f"Split {split_num}: Train [{train_start_idx}:{train_end_idx}], "
                f"Purge [{purge_start_idx}:{purge_end_idx}], "
                f"Test [{test_start_idx}:{test_end_idx}], "
                f"Embargo [{embargo_start_idx}:{embargo_end_idx}]"
            )
            
            current_pos = test_end_idx + self.embargo_size
            split_num += 1
        
        logger.info(f"Generated {len(splits)} walk-forward splits")
        logger.info(f"  Training size: {self.train_size} days ({'expanding' if self.expanding else 'rolling'})")
        logger.info(f"  Test size: {self.test_size} days")
        logger.info(f"  Purge gap: {self.purge_size} days")
        logger.info(f"  Embargo: {self.embargo_size} days")
        
        return splits
    
    def get_sklearn_cv(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get splits in sklearn cross-validation format.
        
        Returns:
        --------
        List of (train_indices, test_indices) tuples
        """
        splits = self.split(data)
        return [(s.train_indices, s.test_indices) for s in splits]


class PerformanceMetrics:
    """
    Calculate realistic performance metrics with transaction costs.
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,  # 0.1% round-trip
        risk_free_rate: float = 0.02  # 2% annual
    ):
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(
        self,
        signals: pd.Series,
        prices: pd.Series,
        include_costs: bool = True
    ) -> pd.Series:
        """
        Calculate strategy returns with transaction costs.
        
        Parameters:
        -----------
        signals : pd.Series
            Position signals (-1, 0, 1)
        prices : pd.Series
            Price series
        include_costs : bool
            Whether to include transaction costs
        
        Returns:
        --------
        pd.Series of strategy returns
        """
        # Calculate price returns
        price_returns = prices.pct_change()
        
        # Calculate strategy returns
        strategy_returns = signals.shift(1) * price_returns
        
        # Calculate transaction costs
        if include_costs:
            trades = signals.diff().abs()
            costs = trades * self.transaction_cost
            strategy_returns = strategy_returns - costs
        
        return strategy_returns.dropna()
    
    def sharpe_ratio(
        self,
        returns: pd.Series,
        annualize: bool = True
    ) -> float:
        """
        Calculate Sharpe Ratio.
        
        Parameters:
        -----------
        returns : pd.Series
            Strategy returns
        annualize : bool
            Whether to annualize (assumes 252 trading days)
        
        Returns:
        --------
        float : Sharpe Ratio
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        # Daily Sharpe
        daily_rf = self.risk_free_rate / 252
        daily_sharpe = (mean_return - daily_rf) / std_return
        
        if annualize:
            return daily_sharpe * np.sqrt(252)
        return daily_sharpe
    
    def sortino_ratio(
        self,
        returns: pd.Series,
        annualize: bool = True
    ) -> float:
        """
        Calculate Sortino Ratio (downside risk only).
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = returns.mean()
        downside = returns[returns < 0]
        
        if len(downside) == 0:
            return np.inf
        
        downside_std = downside.std()
        
        if downside_std == 0:
            return np.inf
        
        daily_rf = self.risk_free_rate / 252
        daily_sortino = (mean_return - daily_rf) / downside_std
        
        if annualize:
            return daily_sortino * np.sqrt(252)
        return daily_sortino
    
    def max_drawdown(
        self,
        returns: pd.Series
    ) -> float:
        """
        Calculate Maximum Drawdown.
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calmar_ratio(
        self,
        returns: pd.Series
    ) -> float:
        """
        Calculate Calmar Ratio (return / max drawdown).
        """
        mdd = abs(self.max_drawdown(returns))
        if mdd == 0:
            return 0.0
        
        annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        return annual_return / mdd
    
    def win_rate(
        self,
        returns: pd.Series
    ) -> float:
        """
        Calculate win rate (% of positive returns).
        """
        return (returns > 0).mean()
    
    def profit_factor(
        self,
        returns: pd.Series
    ) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        """
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        
        if gross_loss == 0:
            return np.inf
        
        return gross_profit / gross_loss
    
    def calculate_all(
        self,
        returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate all metrics.
        """
        return {
            "sharpe_ratio": self.sharpe_ratio(returns),
            "sortino_ratio": self.sortino_ratio(returns),
            "max_drawdown": self.max_drawdown(returns),
            "calmar_ratio": self.calmar_ratio(returns),
            "win_rate": self.win_rate(returns),
            "profit_factor": self.profit_factor(returns),
            "total_return": (1 + returns).prod() - 1,
            "annual_return": (1 + returns).prod() ** (252 / len(returns)) - 1 if len(returns) > 0 else 0,
            "volatility": returns.std() * np.sqrt(252),
            "n_trades": len(returns)
        }


class WalkForwardBacktest:
    """
    Complete walk-forward backtesting framework.
    
    This orchestrates:
    1. Data splitting with purging/embargo
    2. Model training on each split
    3. Out-of-sample prediction
    4. Performance calculation with transaction costs
    """
    
    def __init__(
        self,
        model_factory: Callable,
        validator: WalkForwardValidator = None,
        metrics: PerformanceMetrics = None
    ):
        """
        Parameters:
        -----------
        model_factory : Callable
            Function that returns a new model instance
        validator : WalkForwardValidator
            Walk-forward validator (uses CONFIG defaults if None)
        metrics : PerformanceMetrics
            Performance calculator (uses defaults if None)
        """
        self.model_factory = model_factory
        
        if validator is None:
            self.validator = WalkForwardValidator(
                train_size=CONFIG.walk_forward.train_size,
                test_size=CONFIG.walk_forward.test_size,
                purge_size=CONFIG.walk_forward.purge_size,
                embargo_size=CONFIG.walk_forward.embargo_size
            )
        else:
            self.validator = validator
        
        if metrics is None:
            self.metrics = PerformanceMetrics(
                transaction_cost=CONFIG.transaction_cost,
                risk_free_rate=CONFIG.risk_free_rate
            )
        else:
            self.metrics = metrics
        
        self.results = []
    
    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prices: pd.Series
    ) -> Dict[str, Any]:
        """
        Run complete walk-forward backtest.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features (must be aligned with y and prices)
        y : pd.Series
            Labels
        prices : pd.Series
            Prices for return calculation
        
        Returns:
        --------
        Dict with performance metrics per split and aggregate
        """
        logger.info("\n" + "="*60)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info("="*60)
        
        splits = self.validator.split(X)
        
        all_returns = []
        split_results = []
        
        for i, split in enumerate(splits):
            logger.info(f"\nSplit {i+1}/{len(splits)}")
            logger.info(f"  Train: {split.train_start.date()} to {split.train_end.date()} ({len(split.train_indices)} days)")
            logger.info(f"  Test:  {split.test_start.date()} to {split.test_end.date()} ({len(split.test_indices)} days)")
            
            # Get train data
            X_train = X.iloc[split.train_indices]
            y_train = y.iloc[split.train_indices]
            
            # Get test data
            X_test = X.iloc[split.test_indices]
            y_test = y.iloc[split.test_indices]
            prices_test = prices.iloc[split.test_indices]
            
            # Train model
            model = self.model_factory()
            model.fit(X_train, y_train)
            
            # Predict
            predictions = model.predict(X_test)
            signals = pd.Series(predictions, index=X_test.index)
            
            # Calculate returns
            returns = self.metrics.calculate_returns(signals, prices_test)
            all_returns.append(returns)
            
            # Calculate split metrics
            split_metrics = self.metrics.calculate_all(returns)
            split_metrics["split_id"] = i
            split_metrics["train_start"] = split.train_start
            split_metrics["train_end"] = split.train_end
            split_metrics["test_start"] = split.test_start
            split_metrics["test_end"] = split.test_end
            
            split_results.append(split_metrics)
            
            logger.info(f"  Sharpe: {split_metrics['sharpe_ratio']:.2f}")
            logger.info(f"  Return: {split_metrics['total_return']:.1%}")
            logger.info(f"  Max DD: {split_metrics['max_drawdown']:.1%}")
        
        # Aggregate results
        combined_returns = pd.concat(all_returns)
        aggregate_metrics = self.metrics.calculate_all(combined_returns)
        
        # Calculate Sharpe distribution
        sharpes = [r["sharpe_ratio"] for r in split_results]
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "n_splits": len(splits),
            "aggregate_metrics": aggregate_metrics,
            "split_metrics": split_results,
            "sharpe_mean": np.mean(sharpes),
            "sharpe_std": np.std(sharpes),
            "sharpe_min": np.min(sharpes),
            "sharpe_max": np.max(sharpes),
            "all_returns": combined_returns
        }
        
        self._log_summary(results)
        
        return results
    
    def _log_summary(self, results: Dict[str, Any]):
        """Log validation summary."""
        logger.info("\n" + "-"*60)
        logger.info("WALK-FORWARD VALIDATION SUMMARY")
        logger.info("-"*60)
        
        agg = results["aggregate_metrics"]
        
        logger.info(f"Number of splits: {results['n_splits']}")
        logger.info(f"\nAggregate Performance:")
        logger.info(f"  Sharpe Ratio:    {agg['sharpe_ratio']:.2f}")
        logger.info(f"  Sortino Ratio:   {agg['sortino_ratio']:.2f}")
        logger.info(f"  Total Return:    {agg['total_return']:.1%}")
        logger.info(f"  Annual Return:   {agg['annual_return']:.1%}")
        logger.info(f"  Volatility:      {agg['volatility']:.1%}")
        logger.info(f"  Max Drawdown:    {agg['max_drawdown']:.1%}")
        logger.info(f"  Calmar Ratio:    {agg['calmar_ratio']:.2f}")
        logger.info(f"  Win Rate:        {agg['win_rate']:.1%}")
        logger.info(f"  Profit Factor:   {agg['profit_factor']:.2f}")
        
        logger.info(f"\nSharpe Distribution Across Splits:")
        logger.info(f"  Mean: {results['sharpe_mean']:.2f} ± {results['sharpe_std']:.2f}")
        logger.info(f"  Range: [{results['sharpe_min']:.2f}, {results['sharpe_max']:.2f}]")
        
        # Validation check
        if agg['sharpe_ratio'] > 3.0:
            logger.warning("\n⚠️  SHARPE > 3.0 DETECTED!")
            logger.warning("    This likely indicates bugs or lookahead bias.")
            logger.warning("    Expected realistic Sharpe: 1.0 - 1.5")
        elif agg['sharpe_ratio'] > 2.0:
            logger.warning("\n⚠️  SHARPE > 2.0 - Review carefully")
            logger.warning("    Only the best quant funds achieve this consistently.")
        elif agg['sharpe_ratio'] > 1.0:
            logger.info("\n✓ Sharpe in realistic range (1.0-2.0)")
        else:
            logger.info("\n⚠️  Sharpe < 1.0 - Strategy may not be profitable after costs")
        
        logger.info("-"*60)


def run_walk_forward_validation(
    symbol: str,
    model_factory: Callable,
    data_dir: Path = None
) -> Dict[str, Any]:
    """
    Convenience function to run walk-forward validation on a symbol.
    
    Parameters:
    -----------
    symbol : str
        Ticker symbol
    model_factory : Callable
        Function returning model instance
    data_dir : Path
        Directory with training data
    
    Returns:
    --------
    Dict with validation results
    """
    data_dir = data_dir or CONFIG.train_dir
    
    # Load data
    filepath = data_dir / f"{symbol}.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Data not found: {filepath}")
    
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # TODO: Generate features and labels using Phase 1+2 pipeline
    # For now, placeholder
    
    logger.info(f"Loaded {symbol}: {len(data)} rows")
    
    # This would integrate with the Phase 1+2 pipeline
    # For actual use, you'd need to:
    # 1. Apply FractionalDifferentiator to get features
    # 2. Apply TripleBarrierLabeler to get labels
    # 3. Apply HMM to get regime features
    # 4. Run walk-forward validation
    
    raise NotImplementedError(
        "This function needs integration with Phase 1+2 pipeline. "
        "Use WalkForwardBacktest class directly with your features/labels."
    )


if __name__ == "__main__":
    # Example usage
    logger.info("Walk-Forward Validation Module")
    logger.info("Use WalkForwardBacktest class with your model and data")
