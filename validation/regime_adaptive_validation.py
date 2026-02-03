"""
REGIME-ADAPTIVE PIPELINE
=========================
Different strategy parameters for different market regimes.

Key Insight: A strategy that works in bull markets often fails in bear markets.
Solution: Detect regime and adapt strategy accordingly.

Regime Detection: HMM with 2-3 states
- State 0: Low volatility / Bull
- State 1: High volatility / Bear

Adaptive Parameters:
- Bull regime: Aggressive (wider profit target, more trades)
- Bear regime: Conservative (tighter stops, fewer trades, or sit out)
"""

import sys
sys.path.insert(0, '/Users/humbertolobo/Desktop/bolt.new-main/NUBLE-CLI')

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from src.institutional.labeling.triple_barrier import TripleBarrierLabeler
from src.institutional.regime.hmm_detector import HMMRegimeDetector
from src.institutional.models.meta.meta_labeler import MetaLabeler, MetaLabelConfig
from src.institutional.models.primary.ml_primary_signal import MLPrimarySignal, PrimarySignalConfig


@dataclass 
class RegimeConfig:
    """Configuration for a specific regime."""
    pt_sl: Tuple[float, float]  # Profit target / stop loss multipliers
    max_holding_period: int
    confidence_threshold: float
    trade_fraction: float  # What fraction of signals to act on
    description: str


class RegimeAdaptivePipeline:
    """
    Regime-Adaptive Trading Pipeline.
    
    Uses HMM to detect market regime, then applies regime-specific
    parameters for Triple Barrier labeling and signal generation.
    
    Regimes (based on volatility):
    - Regime 0 (Low Vol): Bull market - aggressive parameters
    - Regime 1 (High Vol): Bear/volatile market - conservative parameters
    """
    
    def __init__(self):
        # Regime detector
        self.hmm = HMMRegimeDetector(n_regimes=2)
        
        # Regime-specific configurations
        self.regime_configs: Dict[int, RegimeConfig] = {
            0: RegimeConfig(  # Low volatility / Bull
                pt_sl=(2.5, 1.0),  # Wider profit target
                max_holding_period=15,
                confidence_threshold=0.52,  # More aggressive
                trade_fraction=1.0,
                description="Bull (Low Vol)"
            ),
            1: RegimeConfig(  # High volatility / Bear
                pt_sl=(1.5, 1.5),  # Tighter, symmetric
                max_holding_period=7,
                confidence_threshold=0.60,  # More selective
                trade_fraction=0.5,  # Trade less often
                description="Bear (High Vol)"
            )
        }
        
        # Models per regime
        self.primary_models: Dict[int, MLPrimarySignal] = {}
        self.meta_labelers: Dict[int, MetaLabeler] = {}
        self.labelers: Dict[int, TripleBarrierLabeler] = {}
        
        # Fallback model (trained on all data)
        self.fallback_primary = None
        self.fallback_meta = None
        
        self.is_fitted = False
        self.regime_stats = {}
        
    def fit(self, df: pd.DataFrame) -> 'RegimeAdaptivePipeline':
        """
        Train regime-adaptive pipeline.
        
        1. Detect regimes in training data
        2. Split data by regime
        3. Train separate models for each regime
        """
        logger.info("="*60)
        logger.info("FITTING REGIME-ADAPTIVE PIPELINE")
        logger.info("="*60)
        
        # Step 1: Detect regimes using pd.Series (HMM expects Series)
        logger.info("\n1. Detecting regimes...")
        returns = df['close'].pct_change().dropna()  # Keep as Series
        
        try:
            self.hmm.fit(returns)  # Pass Series, not numpy array
            regime_series = self.hmm.predict(returns, method='filter')  # Get Series back
            
            # Analyze regimes
            for regime in regime_series.unique():
                mask = regime_series == regime
                regime_returns = returns[mask]
                vol = regime_returns.std() * np.sqrt(252)
                mean_ret = regime_returns.mean() * 252
                n_days = mask.sum()
                
                self.regime_stats[regime] = {
                    'n_days': n_days,
                    'volatility': vol,
                    'annual_return': mean_ret
                }
                
                if regime in self.regime_configs:
                    desc = self.regime_configs[regime].description
                else:
                    desc = f"Regime {regime}"
                
                logger.info(f"   Regime {regime} ({desc}): "
                           f"{n_days} days, Vol={vol:.1%}, Return={mean_ret:.1%}")
        except Exception as e:
            logger.warning(f"   HMM failed: {e}, using single regime")
            regime_series = pd.Series(0, index=returns.index)
            self.regime_stats[0] = {'n_days': len(returns), 'volatility': 0.2, 'annual_return': 0}
        
        # Step 2: Train models for each regime
        for regime, config in self.regime_configs.items():
            logger.info(f"\n2. Training Regime {regime} ({config.description})...")
            
            # Get data for this regime
            regime_mask = regime_series == regime
            regime_dates = regime_series[regime_mask].index
            
            if len(regime_dates) < 200:
                logger.warning(f"   Insufficient data for regime {regime} ({len(regime_dates)} days)")
                continue
            
            # Get full df rows for these dates (need OHLCV)
            regime_df = df.loc[df.index.isin(regime_dates)].copy()
            
            if len(regime_df) < 100:
                logger.warning(f"   Insufficient aligned data for regime {regime}")
                continue
            
            # Create regime-specific labeler
            labeler = TripleBarrierLabeler(
                pt_sl=config.pt_sl,
                max_holding_period=config.max_holding_period,
                volatility_lookback=20
            )
            self.labelers[regime] = labeler
            
            # Generate labels
            labels = labeler.get_labels(regime_df['close'])
            logger.info(f"   Labels: {labels.value_counts().to_dict()}")
            
            # Train primary model
            primary_config = PrimarySignalConfig(
                model_type='random_forest',
                n_estimators=150,
                max_depth=5,
                min_samples_leaf=30,
                confidence_threshold=config.confidence_threshold
            )
            primary = MLPrimarySignal(primary_config)
            
            try:
                primary.fit(regime_df, labels)
                self.primary_models[regime] = primary
                logger.info(f"   Primary model trained")
            except Exception as e:
                logger.warning(f"   Primary model failed: {e}")
                continue
            
            # Generate features and primary signals for meta-labeler
            features = primary.generate_features(regime_df)
            primary_signals = primary.predict(regime_df)
            
            # Align
            common_idx = (
                features.dropna().index
                .intersection(labels.dropna().index)
                .intersection(primary_signals.index)
            )
            
            if len(common_idx) < 100:
                logger.warning(f"   Insufficient aligned samples for meta-labeler")
                continue
            
            # Train meta-labeler
            try:
                meta = MetaLabeler(config=MetaLabelConfig())
                meta.fit(
                    features=features.loc[common_idx].fillna(0),
                    primary_signals=primary_signals.loc[common_idx],
                    triple_barrier_labels=labels.loc[common_idx]
                )
                self.meta_labelers[regime] = meta
                logger.info(f"   Meta-labeler trained")
            except Exception as e:
                logger.warning(f"   Meta-labeler failed: {e}")
        
        # Step 3: Train fallback model on all data
        logger.info("\n3. Training fallback model (all data)...")
        fallback_labeler = TripleBarrierLabeler(pt_sl=(2.0, 1.0), max_holding_period=10)
        fallback_labels = fallback_labeler.get_labels(df['close'])
        
        fallback_config = PrimarySignalConfig(
            model_type='random_forest',
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=50,
            confidence_threshold=0.55
        )
        self.fallback_primary = MLPrimarySignal(fallback_config)
        self.fallback_primary.fit(df, fallback_labels)
        
        fallback_features = self.fallback_primary.generate_features(df)
        fallback_signals = self.fallback_primary.predict(df)
        
        common_idx = (
            fallback_features.dropna().index
            .intersection(fallback_labels.dropna().index)
            .intersection(fallback_signals.index)
        )
        
        self.fallback_meta = MetaLabeler(config=MetaLabelConfig())
        self.fallback_meta.fit(
            features=fallback_features.loc[common_idx].fillna(0),
            primary_signals=fallback_signals.loc[common_idx],
            triple_barrier_labels=fallback_labels.loc[common_idx]
        )
        
        self.is_fitted = True
        logger.info("\n✓ Regime-adaptive pipeline fitted!")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals using regime-appropriate models.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Detect regimes using pd.Series (HMM expects Series)
        returns = df['close'].pct_change().dropna()
        
        try:
            regime_series = self.hmm.predict(returns, method='filter')  # Returns pd.Series
        except:
            regime_series = pd.Series(0, index=returns.index)
        
        # Generate signals for each regime
        all_signals = pd.Series(0, index=df.index)
        
        for regime in self.primary_models.keys():
            if regime not in self.meta_labelers:
                continue
                
            regime_mask = regime_series.reindex(df.index) == regime
            regime_dates = regime_mask[regime_mask].index
            
            if len(regime_dates) < 5:
                continue
            
            regime_df = df.loc[regime_dates].copy()
            
            try:
                # Get primary signals
                primary = self.primary_models[regime]
                primary_signals = primary.predict(regime_df)
                
                # Get features
                features = primary.generate_features(regime_df)
                
                # Align
                common_idx = features.dropna().index.intersection(primary_signals.index)
                
                if len(common_idx) < 2:
                    continue
                
                # Apply meta-labeler
                meta = self.meta_labelers[regime]
                meta_predictions = meta.predict(
                    features=features.loc[common_idx].fillna(0),
                    primary_signals=primary_signals.loc[common_idx]
                )
                
                # Final signals
                regime_signals = primary_signals.loc[common_idx] * meta_predictions
                
                # Apply trade fraction (reduce trading in high-vol regime)
                config = self.regime_configs[regime]
                if config.trade_fraction < 1.0:
                    # Randomly reduce trades
                    np.random.seed(42)
                    keep_mask = np.random.random(len(regime_signals)) < config.trade_fraction
                    regime_signals = regime_signals * keep_mask.astype(int)
                
                all_signals.loc[common_idx] = regime_signals
                
            except Exception as e:
                logger.warning(f"Regime {regime} predict failed: {e}")
                continue
        
        # Fill any missing with fallback
        missing_mask = all_signals == 0
        if missing_mask.any() and self.fallback_primary is not None:
            try:
                missing_df = df.loc[missing_mask]
                if len(missing_df) > 10:
                    fallback_signals = self.fallback_primary.predict(missing_df)
                    features = self.fallback_primary.generate_features(missing_df)
                    common_idx = features.dropna().index.intersection(fallback_signals.index)
                    
                    if len(common_idx) > 0:
                        meta_pred = self.fallback_meta.predict(
                            features=features.loc[common_idx].fillna(0),
                            primary_signals=fallback_signals.loc[common_idx]
                        )
                        final_fallback = fallback_signals.loc[common_idx] * meta_pred
                        all_signals.loc[common_idx] = final_fallback
            except:
                pass
        
        return all_signals


def calculate_sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252)


def validate_regime_adaptive(symbol: str) -> Dict[str, Any]:
    """Validate regime-adaptive pipeline on one symbol."""
    logger.info(f"\n{'='*60}")
    logger.info(f"REGIME-ADAPTIVE VALIDATION: {symbol}")
    logger.info(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(
        f'/Users/humbertolobo/Desktop/bolt.new-main/NUBLE-CLI/data/train/{symbol}.csv',
        index_col=0, parse_dates=True
    )
    logger.info(f"Data: {len(df)} rows")
    
    # 70/30 split
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Fit regime-adaptive pipeline
    pipeline = RegimeAdaptivePipeline()
    pipeline.fit(train_df)
    
    # Predict on test
    logger.info("\nPredicting on test data...")
    signals = pipeline.predict(test_df)
    logger.info(f"Signals: {signals.value_counts().to_dict()}")
    
    if len(signals[signals != 0]) < 10:
        logger.warning("Too few signals generated")
        return {'symbol': symbol, 'error': 'Insufficient signals'}
    
    # Backtest
    price_returns = test_df['close'].pct_change().reindex(signals.index)
    strategy_returns = signals.shift(1) * price_returns
    
    # Transaction costs
    trades = signals.diff().abs()
    costs = trades * 0.001
    net_returns = (strategy_returns - costs).dropna()
    
    if len(net_returns) < 10:
        return {'symbol': symbol, 'error': 'Insufficient returns'}
    
    sharpe = calculate_sharpe(net_returns)
    total_return = (1 + net_returns).prod() - 1
    buy_hold = (1 + price_returns.dropna()).prod() - 1
    n_trades = (trades > 0).sum()
    
    # Compare to naive
    naive_momentum = test_df['close'].pct_change(20).reindex(signals.index)
    naive_signals = pd.Series(0, index=signals.index)
    naive_signals[naive_momentum > 0.02] = 1
    naive_signals[naive_momentum < -0.02] = -1
    naive_returns = naive_signals.shift(1) * price_returns
    naive_net = (naive_returns - naive_signals.diff().abs() * 0.001).dropna()
    naive_sharpe = calculate_sharpe(naive_net)
    
    logger.info(f"\n--- RESULTS ---")
    logger.info(f"Strategy Sharpe: {sharpe:.2f}")
    logger.info(f"Strategy Return: {total_return:.1%}")
    logger.info(f"Buy & Hold: {buy_hold:.1%}")
    logger.info(f"Naive Sharpe: {naive_sharpe:.2f}")
    logger.info(f"Improvement: {sharpe - naive_sharpe:+.2f}")
    
    return {
        'symbol': symbol,
        'sharpe': sharpe,
        'return': total_return,
        'buy_hold': buy_hold,
        'naive_sharpe': naive_sharpe,
        'improvement': sharpe - naive_sharpe,
        'trades': n_trades
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("REGIME-ADAPTIVE PIPELINE VALIDATION")
    print("="*70)
    
    symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA']
    results = []
    
    for symbol in symbols:
        try:
            result = validate_regime_adaptive(symbol)
            if 'error' not in result:
                results.append(result)
        except Exception as e:
            print(f"\nError for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\n{'Symbol':<8} {'Regime Sharpe':>14} {'Naive':>10} {'Improvement':>14}")
        print("-"*50)
        
        for r in results:
            print(f"{r['symbol']:<8} {r['sharpe']:>14.2f} {r['naive_sharpe']:>10.2f} {r['improvement']:>+14.2f}")
        
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        avg_naive = np.mean([r['naive_sharpe'] for r in results])
        avg_improve = np.mean([r['improvement'] for r in results])
        
        print("-"*50)
        print(f"{'AVERAGE':<8} {avg_sharpe:>14.2f} {avg_naive:>10.2f} {avg_improve:>+14.2f}")
        print("="*70)
        
        if avg_sharpe > 0.5:
            print("\n✓ Regime-adaptive pipeline shows promising edge!")
            print("  Consider running CPCV validation for robustness")
        elif avg_sharpe > 0:
            print("\n📊 Regime-adaptive pipeline shows modest positive edge")
            print("  Consider adding AFML sample weighting (Priority 3)")
        else:
            print("\n⚠️ Still not showing consistent edge")
            print("  May need fundamental data or alternative signals")
