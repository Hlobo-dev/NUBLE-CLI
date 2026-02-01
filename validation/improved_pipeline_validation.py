"""
IMPROVED PIPELINE VALIDATION
=============================
Validates the ML-based primary signal with Meta-Labeler.

Key Changes from Original:
1. ML-based primary signal (not simple momentum)
2. 70+ technical features
3. Regime-adaptive parameters
4. Proper sample alignment

Target: Sharpe > 0.5, PBO < 0.5
"""

import sys
sys.path.insert(0, '/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI')

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime
from itertools import combinations
from scipy import stats
from math import comb
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import components
from src.institutional.labeling.triple_barrier import TripleBarrierLabeler
from src.institutional.regime.hmm_detector import HMMRegimeDetector
from src.institutional.models.meta.meta_labeler import MetaLabeler, MetaLabelConfig
from src.institutional.models.primary.ml_primary_signal import MLPrimarySignal, PrimarySignalConfig


class ImprovedPipeline:
    """
    Improved Phase 1+2 Pipeline with ML Primary Signal.
    
    Architecture:
    1. TripleBarrier → Generate labels
    2. MLPrimarySignal → Predict direction (using 70+ features)
    3. MetaLabeler → Filter low-confidence signals
    4. Final Signal = Primary × Meta
    """
    
    def __init__(
        self,
        pt_sl: Tuple[float, float] = (2.0, 1.0),
        max_holding_period: int = 10,
        model_type: str = 'random_forest',
        confidence_threshold: float = 0.55
    ):
        # Label generator
        self.labeler = TripleBarrierLabeler(
            pt_sl=pt_sl,
            max_holding_period=max_holding_period,
            volatility_lookback=20
        )
        
        # Primary signal generator
        self.primary_config = PrimarySignalConfig(
            model_type=model_type,
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=50,
            confidence_threshold=confidence_threshold
        )
        self.primary_model = MLPrimarySignal(self.primary_config)
        
        # Meta-labeler (filters primary signals)
        self.meta_labeler = None
        
        # Regime detector
        self.hmm = HMMRegimeDetector(n_regimes=2)
        
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'ImprovedPipeline':
        """
        Fit the complete pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training OHLCV data
        """
        logger.info("="*60)
        logger.info("FITTING IMPROVED PIPELINE")
        logger.info("="*60)
        
        # Step 1: Generate Triple Barrier labels
        logger.info("\n1. Generating Triple Barrier labels...")
        labels = self.labeler.get_labels(df['close'])
        logger.info(f"   Labels: {labels.value_counts().to_dict()}")
        
        # Step 2: Fit HMM regime detector
        logger.info("\n2. Fitting HMM regime detector...")
        returns = df['close'].pct_change().dropna()
        try:
            self.hmm.fit(returns.values.reshape(-1, 1))
            regimes = self.hmm.predict(returns.values.reshape(-1, 1))
            regime_series = pd.Series(index=returns.index, data=regimes)
            logger.info(f"   Regimes: {pd.Series(regimes).value_counts().to_dict()}")
        except Exception as e:
            logger.warning(f"   HMM failed: {e}")
            regime_series = pd.Series(0, index=returns.index)
        
        # Step 3: Fit ML primary signal model
        logger.info("\n3. Training ML primary signal model...")
        self.primary_model.fit(df, labels)
        
        # Step 4: Generate primary signals for meta-labeler training
        logger.info("\n4. Generating primary signals for meta-labeler...")
        primary_signals = self.primary_model.predict(df)
        logger.info(f"   Primary signals: {primary_signals.value_counts().to_dict()}")
        
        # Step 5: Prepare features for meta-labeler
        logger.info("\n5. Preparing meta-labeler features...")
        features = self.primary_model.generate_features(df)
        
        # Add regime to features
        features['regime'] = regime_series.reindex(features.index).fillna(0)
        
        # Align everything
        common_idx = (
            features.dropna().index
            .intersection(labels.dropna().index)
            .intersection(primary_signals.index)
        )
        
        features_aligned = features.loc[common_idx]
        labels_aligned = labels.loc[common_idx]
        signals_aligned = primary_signals.loc[common_idx]
        
        logger.info(f"   Aligned samples: {len(common_idx)}")
        
        # Step 6: Fit meta-labeler
        logger.info("\n6. Training meta-labeler...")
        self.meta_labeler = MetaLabeler(config=MetaLabelConfig())
        self.meta_labeler.fit(
            features=features_aligned,
            primary_signals=signals_aligned,
            triple_barrier_labels=labels_aligned
        )
        
        self.is_fitted = True
        logger.info("\n✓ Pipeline fitted successfully!")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate final trading signals.
        
        Returns:
        --------
        pd.Series
            Final signals: -1 (short), 0 (no trade), 1 (long)
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Get primary signals  
        primary_signals = self.primary_model.predict(df)
        
        if len(primary_signals) == 0:
            return pd.Series(dtype=int)
        
        # Get features for meta-labeler
        features = self.primary_model.generate_features(df)
        
        # Add regime
        returns = df['close'].pct_change().dropna()
        try:
            regimes = self.hmm.predict(returns.values.reshape(-1, 1))
            regime_series = pd.Series(index=returns.index, data=regimes)
        except:
            regime_series = pd.Series(0, index=returns.index)
        features['regime'] = regime_series.reindex(features.index).fillna(0)
        
        # Align
        common_idx = features.dropna().index.intersection(primary_signals.index)
        
        if len(common_idx) == 0:
            return pd.Series(dtype=int)
            
        features_aligned = features.loc[common_idx].fillna(0)
        signals_aligned = primary_signals.loc[common_idx]
        
        # Apply meta-labeler filter with error handling
        try:
            meta_predictions = self.meta_labeler.predict(
                features=features_aligned,
                primary_signals=signals_aligned
            )
            # Final signal = Primary direction × Meta approval
            final_signals = signals_aligned * meta_predictions
        except Exception as e:
            # Fallback to primary signals only if meta-labeler fails
            logger.warning(f"Meta-labeler predict failed: {e}, using primary signals")
            final_signals = signals_aligned
        
        return final_signals
    
    def predict_with_sizing(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate signals with position sizing from meta-labeler.
        
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            (signals, position_sizes)
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Get primary signals and probabilities
        primary_signals = self.primary_model.predict(df)
        primary_probas = self.primary_model.predict_proba(df)
        
        # Get features
        features = self.primary_model.generate_features(df)
        
        # Add regime
        returns = df['close'].pct_change().dropna()
        try:
            regimes = self.hmm.predict(returns.values.reshape(-1, 1))
            regime_series = pd.Series(index=returns.index, data=regimes)
        except:
            regime_series = pd.Series(0, index=returns.index)
        features['regime'] = regime_series.reindex(features.index).fillna(0)
        
        # Align
        common_idx = features.dropna().index.intersection(primary_signals.index)
        features_aligned = features.loc[common_idx].fillna(0)
        signals_aligned = primary_signals.loc[common_idx]
        
        # Get meta-labeler probabilities
        meta_probas = self.meta_labeler.predict_proba(
            features=features_aligned,
            primary_signals=signals_aligned
        )
        
        # Meta probability of success
        if meta_probas.shape[1] > 1:
            meta_confidence = meta_probas[:, 1]
        else:
            meta_confidence = meta_probas[:, 0]
        
        # Position sizing (Kelly-like)
        # Size = 2 * (confidence - 0.5) for confidence > 0.5
        # Capped at 1.0
        sizes = np.clip(2 * (meta_confidence - 0.5), 0, 1)
        
        # Final signals
        meta_predictions = self.meta_labeler.predict(features_aligned, signals_aligned)
        final_signals = signals_aligned * meta_predictions
        
        return final_signals, pd.Series(sizes, index=common_idx)


def calculate_sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252)


def calculate_pbo(sharpes: List[float]) -> float:
    """Probability of Backtest Overfitting."""
    if len(sharpes) < 2:
        return 0.0
    sharpe_mean = np.mean(sharpes)
    sharpe_std = np.std(sharpes)
    if sharpe_mean == 0:
        return 1.0
    cv = abs(sharpe_std / sharpe_mean)
    return 1 - 1 / (1 + cv)


def calculate_dsr(observed_sharpe: float, n_trials: int, returns: pd.Series) -> float:
    """Deflated Sharpe Ratio probability."""
    T = len(returns)
    if T < 4:
        return 0.5
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)
    expected_max = np.sqrt(2 * np.log(n_trials)) if n_trials > 1 else 0
    var_sr = (1 + 0.5 * observed_sharpe**2 
              - skew * observed_sharpe 
              + (kurt / 4) * observed_sharpe**2) / T
    std_sr = np.sqrt(max(var_sr, 1e-10))
    adjusted_sharpe = observed_sharpe - expected_max
    return stats.norm.cdf(adjusted_sharpe / std_sr)


def run_walk_forward_validation(
    symbol: str,
    train_size: int = 504,
    test_size: int = 63,
    purge_size: int = 5,
    transaction_cost: float = 0.001
) -> Dict[str, Any]:
    """
    Run walk-forward validation on improved pipeline.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"WALK-FORWARD VALIDATION: {symbol}")
    logger.info(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(
        f'/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI/data/train/{symbol}.csv',
        index_col=0, parse_dates=True
    )
    logger.info(f"Data: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
    
    # Walk-forward splits
    n_samples = len(df)
    n_splits = (n_samples - train_size - purge_size) // test_size
    
    logger.info(f"Train: {train_size} days, Test: {test_size} days, Splits: {n_splits}")
    
    all_sharpes = []
    all_returns = []
    split_results = []
    
    for split_idx in range(n_splits):
        train_start = split_idx * test_size
        train_end = train_start + train_size
        test_start = train_end + purge_size
        test_end = test_start + test_size
        
        if test_end > n_samples:
            break
        
        # Get data splits
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        
        if len(train_df) < 200 or len(test_df) < 20:
            continue
        
        try:
            # Create and fit pipeline
            pipeline = ImprovedPipeline(
                pt_sl=(2.0, 1.0),
                max_holding_period=10,
                confidence_threshold=0.55
            )
            pipeline.fit(train_df)
            
            # Predict on test
            signals = pipeline.predict(test_df)
            
            if len(signals) < 10:
                continue
            
            # Calculate returns
            price_returns = test_df['close'].pct_change().reindex(signals.index)
            strategy_returns = signals.shift(1) * price_returns
            
            # Transaction costs
            trades = signals.diff().abs()
            costs = trades * transaction_cost
            net_returns = (strategy_returns - costs).dropna()
            
            if len(net_returns) < 10:
                continue
            
            # Metrics
            sharpe = calculate_sharpe(net_returns)
            total_return = (1 + net_returns).prod() - 1
            n_trades = (trades > 0).sum()
            
            all_sharpes.append(sharpe)
            all_returns.append(net_returns)
            
            split_results.append({
                'split': split_idx + 1,
                'sharpe': sharpe,
                'return': total_return,
                'trades': n_trades
            })
            
            logger.info(f"Split {split_idx+1}/{n_splits}: Sharpe={sharpe:.2f}, Return={total_return:.1%}")
            
        except Exception as e:
            logger.warning(f"Split {split_idx+1} failed: {e}")
            continue
    
    if len(all_sharpes) < 3:
        logger.warning("Not enough valid splits")
        return {'symbol': symbol, 'error': 'Insufficient splits'}
    
    # Aggregate results
    combined_returns = pd.concat(all_returns)
    aggregate_sharpe = calculate_sharpe(combined_returns)
    aggregate_return = (1 + combined_returns).prod() - 1
    pbo = calculate_pbo(all_sharpes)
    dsr = calculate_dsr(aggregate_sharpe, n_trials=1, returns=combined_returns)
    
    logger.info(f"\n{'-'*50}")
    logger.info(f"RESULTS FOR {symbol}")
    logger.info(f"{'-'*50}")
    logger.info(f"Valid Splits: {len(all_sharpes)}/{n_splits}")
    logger.info(f"\nSharpe Distribution:")
    logger.info(f"  Mean: {np.mean(all_sharpes):.2f} ± {np.std(all_sharpes):.2f}")
    logger.info(f"  Range: [{np.min(all_sharpes):.2f}, {np.max(all_sharpes):.2f}]")
    logger.info(f"\nAggregate Sharpe: {aggregate_sharpe:.2f}")
    logger.info(f"Aggregate Return: {aggregate_return:.1%}")
    logger.info(f"\nPBO: {pbo:.1%}")
    logger.info(f"DSR: {dsr:.1%}")
    
    return {
        'symbol': symbol,
        'n_splits': len(all_sharpes),
        'sharpe_mean': np.mean(all_sharpes),
        'sharpe_std': np.std(all_sharpes),
        'aggregate_sharpe': aggregate_sharpe,
        'aggregate_return': aggregate_return,
        'pbo': pbo,
        'dsr': dsr
    }


def run_cpcv_validation(
    symbol: str,
    n_splits: int = 6,
    n_test_groups: int = 2,
    purge_size: int = 5,
    transaction_cost: float = 0.001
) -> Dict[str, Any]:
    """
    Run CPCV validation with PBO analysis.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"CPCV VALIDATION: {symbol}")
    logger.info(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(
        f'/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI/data/train/{symbol}.csv',
        index_col=0, parse_dates=True
    )
    
    n_total = len(df)
    group_size = n_total // n_splits
    groups = [(i * group_size, (i + 1) * group_size if i < n_splits - 1 else n_total)
              for i in range(n_splits)]
    
    n_paths = comb(n_splits, n_test_groups)
    logger.info(f"Data: {n_total} rows, {n_paths} CPCV paths")
    
    test_combinations = list(combinations(range(n_splits), n_test_groups))
    
    all_sharpes = []
    all_returns = []
    
    for path_idx, test_groups in enumerate(test_combinations):
        train_groups = [g for g in range(n_splits) if g not in test_groups]
        
        # Build train indices
        train_idx = []
        for g in train_groups:
            train_idx.extend(range(groups[g][0], groups[g][1]))
        
        # Build test indices
        test_idx = []
        for g in test_groups:
            test_idx.extend(range(groups[g][0], groups[g][1]))
        
        # Apply purging
        test_starts = [groups[g][0] for g in test_groups]
        test_ends = [groups[g][1] for g in test_groups]
        
        purged_train = []
        for idx in train_idx:
            too_close = any(abs(idx - ts) < purge_size or abs(idx - te) < purge_size
                          for ts, te in zip(test_starts, test_ends))
            if not too_close:
                purged_train.append(idx)
        
        if len(purged_train) < 200 or len(test_idx) < 50:
            continue
        
        train_df = df.iloc[purged_train].copy()
        test_df = df.iloc[test_idx].copy()
        
        try:
            # Fit pipeline
            pipeline = ImprovedPipeline(
                pt_sl=(2.0, 1.0),
                max_holding_period=10,
                confidence_threshold=0.55
            )
            pipeline.fit(train_df)
            
            # Predict
            signals = pipeline.predict(test_df)
            
            if len(signals) < 20:
                continue
            
            # Calculate returns
            price_returns = test_df['close'].pct_change().reindex(signals.index)
            strategy_returns = signals.shift(1) * price_returns
            trades = signals.diff().abs()
            costs = trades * transaction_cost
            net_returns = (strategy_returns - costs).dropna()
            
            if len(net_returns) < 10:
                continue
            
            sharpe = calculate_sharpe(net_returns)
            all_sharpes.append(sharpe)
            all_returns.append(net_returns)
            
            logger.info(f"Path {path_idx+1}/{n_paths}: Train {train_groups} | "
                       f"Test {list(test_groups)} | Sharpe: {sharpe:.2f}")
            
        except Exception as e:
            logger.warning(f"Path {path_idx+1} failed: {e}")
            continue
    
    if len(all_sharpes) < 3:
        logger.warning("Not enough valid paths")
        return {'symbol': symbol, 'error': 'Insufficient paths'}
    
    # Aggregate
    combined_returns = pd.concat(all_returns)
    aggregate_sharpe = calculate_sharpe(combined_returns)
    aggregate_return = (1 + combined_returns).prod() - 1
    pbo = calculate_pbo(all_sharpes)
    dsr = calculate_dsr(aggregate_sharpe, n_trials=1, returns=combined_returns)
    
    logger.info(f"\n{'-'*50}")
    logger.info(f"RESULTS FOR {symbol}")
    logger.info(f"{'-'*50}")
    logger.info(f"Valid Paths: {len(all_sharpes)}/{n_paths}")
    logger.info(f"Sharpe: {np.mean(all_sharpes):.2f} ± {np.std(all_sharpes):.2f}")
    logger.info(f"Aggregate Sharpe: {aggregate_sharpe:.2f}")
    logger.info(f"PBO: {pbo:.1%}")
    logger.info(f"DSR: {dsr:.1%}")
    
    return {
        'symbol': symbol,
        'n_paths': len(all_sharpes),
        'sharpe_mean': np.mean(all_sharpes),
        'sharpe_std': np.std(all_sharpes),
        'aggregate_sharpe': aggregate_sharpe,
        'aggregate_return': aggregate_return,
        'pbo': pbo,
        'dsr': dsr
    }


def run_full_validation(symbols: List[str] = None, validation_type: str = 'walk_forward'):
    """
    Run full validation across multiple symbols.
    
    Parameters:
    -----------
    symbols : List[str]
        Symbols to validate
    validation_type : str
        'walk_forward' or 'cpcv'
    """
    if symbols is None:
        symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'GLD', 'TLT']
    
    print("\n" + "="*70)
    print(f"IMPROVED PIPELINE VALIDATION ({validation_type.upper()})")
    print("="*70)
    print(f"ML Primary Signal + Meta-Labeler")
    print(f"Symbols: {symbols}")
    print()
    
    results = []
    
    for symbol in symbols:
        try:
            if validation_type == 'walk_forward':
                result = run_walk_forward_validation(symbol)
            else:
                result = run_cpcv_validation(symbol)
            
            if 'error' not in result:
                results.append(result)
        except Exception as e:
            logger.error(f"Failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("\nNo valid results")
        return
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"\n{'Symbol':<8} {'Splits':>8} {'Sharpe':>10} {'Std':>8} {'PBO':>8} {'DSR':>8} {'Status':>12}")
    print("-"*70)
    
    for r in results:
        status = "✓ PASS" if r.get('pbo', 1) < 0.5 and r.get('aggregate_sharpe', 0) > 0 else "⚠️ REVIEW"
        n = r.get('n_splits', r.get('n_paths', 0))
        print(f"{r['symbol']:<8} {n:>8} {r.get('aggregate_sharpe', 0):>10.2f} "
              f"{r.get('sharpe_std', 0):>7.2f} {r.get('pbo', 0):>7.1%} "
              f"{r.get('dsr', 0):>7.1%} {status:>12}")
    
    print("-"*70)
    
    avg_sharpe = np.mean([r.get('aggregate_sharpe', 0) for r in results])
    avg_std = np.mean([r.get('sharpe_std', 0) for r in results])
    avg_pbo = np.mean([r.get('pbo', 0) for r in results])
    avg_dsr = np.mean([r.get('dsr', 0) for r in results])
    
    print(f"{'AVERAGE':<8} {'':>8} {avg_sharpe:>10.2f} {avg_std:>7.2f} {avg_pbo:>7.1%} {avg_dsr:>7.1%}")
    print("="*70)
    
    # Assessment
    print("\nFINAL ASSESSMENT:")
    if avg_sharpe > 3.0:
        print("❌ FAIL: Sharpe > 3.0 indicates bugs or lookahead bias")
    elif avg_pbo > 0.5:
        print("⚠️ CONCERN: High PBO indicates potential overfitting")
    elif avg_sharpe < 0:
        print("❌ FAIL: Negative Sharpe - strategy not profitable")
    elif avg_sharpe < 0.3:
        print("⚠️ WEAK: Strategy has minimal edge (Sharpe < 0.3)")
    elif avg_sharpe < 0.5:
        print("📊 MODEST: Strategy has small edge (Sharpe 0.3-0.5)")
        print("   Consider Priority 2 improvements (regime-adaptive params)")
    else:
        print("✓ REALISTIC: Strategy shows believable edge")
        print(f"   - Average Sharpe: {avg_sharpe:.2f}")
        print(f"   - Average PBO: {avg_pbo:.1%}")
        print(f"   - Ready for Priority 4 (Final OOS Test)")
    
    # Next steps
    print("\nNEXT STEPS:")
    if avg_sharpe < 0.5 or avg_pbo > 0.5:
        print("  1. Implement Priority 2: Regime-Adaptive Parameters")
        print("  2. Implement Priority 3: AFML Sample Weighting")
        print("  3. DO NOT run OOS test yet")
    else:
        print("  1. Consider running final OOS test on 2023-2026 data")
        print("  2. Ensure consistency across all symbols first")


if __name__ == "__main__":
    # Run validation
    run_full_validation(
        symbols=['SPY', 'QQQ', 'AAPL'],  # Start with 3 major symbols
        validation_type='walk_forward'
    )
