"""
PRIORITY 3: AFML SAMPLE-WEIGHTED VALIDATION
============================================

This implements proper sample weighting from AFML Chapter 4 to reduce overfitting
from overlapping Triple Barrier labels.

Key Changes from Previous Pipeline:
1. Compute sample weights based on label uniqueness
2. Pass weights to meta-labeler training
3. Compare weighted vs unweighted performance

Target: Reduce PBO from 74% to less than 50%
"""

import sys
sys.path.insert(0, '/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI')

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Our modules
from src.institutional.labeling.triple_barrier import TripleBarrierLabeler
from src.institutional.regime.hmm_detector import HMMRegimeDetector
from src.institutional.models.meta.meta_labeler import MetaLabeler, MetaLabelConfig
from src.institutional.models.primary.ml_primary_signal import MLPrimarySignal, PrimarySignalConfig
from src.institutional.validation.sample_weights import AFMLSampleWeights

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass  
class WeightedPipelineConfig:
    """Configuration for sample-weighted pipeline."""
    pt_sl: Tuple[float, float] = (2.0, 1.0)
    max_holding_period: int = 10
    meta_threshold: float = 0.55
    time_decay: float = 1.0
    train_size: int = 504
    test_size: int = 63
    purge: int = 5
    embargo: int = 5
    transaction_cost: float = 0.001


class SampleWeightedPipeline:
    """ML Pipeline with AFML Sample Weighting."""
    
    def __init__(self, config: WeightedPipelineConfig):
        self.config = config
        self.weighter = AFMLSampleWeights(decay_factor=config.time_decay)
        
    def compute_label_end_times(self, df, labels, max_holding):
        """Estimate end times for Triple Barrier labels."""
        t1 = pd.Series(index=labels.index, dtype='datetime64[ns]')
        for idx in labels.index:
            pos = df.index.get_loc(idx)
            end_pos = min(pos + max_holding, len(df) - 1)
            t1[idx] = df.index[end_pos]
        return t1
    
    def walk_forward_weighted(self, df, use_sample_weights=True):
        """Walk-forward validation with sample-weighted training."""
        cfg = self.config
        total_periods = len(df)
        
        results = {
            'trade_returns': [],
            'daily_returns': [],
            'sample_weights_used': []
        }
        
        labeler = TripleBarrierLabeler(pt_sl=cfg.pt_sl, max_holding_period=cfg.max_holding_period)
        primary_config = PrimarySignalConfig(n_estimators=100, max_depth=4, min_samples_leaf=30)
        
        start_idx = 0
        folds_processed = 0
        
        while True:
            train_end = start_idx + cfg.train_size
            test_start = train_end + cfg.purge
            test_end = test_start + cfg.test_size
            
            if test_end > total_periods:
                break
            
            train_df = df.iloc[start_idx:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            if len(train_df) < 200 or len(test_df) < 20:
                start_idx += cfg.test_size
                continue
            
            # Generate labels
            train_labels = labeler.get_labels(train_df['close'])
            train_labels = train_labels[train_labels.isin([-1, 0, 1])].dropna()
            
            if len(train_labels) < 50:
                start_idx += cfg.test_size
                continue
            
            # Compute sample weights
            if use_sample_weights:
                t1 = self.compute_label_end_times(train_df, train_labels, cfg.max_holding_period)
                weights = self.weighter.get_sample_weights(
                    event_times=train_labels.index,
                    t1_times=t1,
                    close_idx=train_df.index
                )
                results['sample_weights_used'].append(weights.mean())
            else:
                weights = pd.Series(1.0, index=train_labels.index)
            
            # Train primary model
            primary_model = MLPrimarySignal(primary_config)
            features_df = primary_model.generate_features(train_df)
            train_feature_idx = train_labels.index.intersection(features_df.index)
            
            if len(train_feature_idx) < 50:
                start_idx += cfg.test_size
                continue
            
            aligned_labels = train_labels.loc[train_feature_idx]
            aligned_weights = weights.loc[train_feature_idx]
            
            primary_model.fit(
                train_df.loc[:train_feature_idx[-1]],
                aligned_labels,
                sample_weights=aligned_weights
            )
            
            # Get primary signals
            train_features = features_df.loc[train_feature_idx]
            train_proba = primary_model.predict_proba(train_df.loc[:train_feature_idx[-1]])
            
            primary_signals = pd.Series(0, index=train_feature_idx)
            for idx in train_feature_idx:
                if idx in train_proba.index:
                    prob_up = train_proba.loc[idx, 1] if 1 in train_proba.columns else 0.5
                    if prob_up > 0.55:
                        primary_signals[idx] = 1
                    elif prob_up < 0.45:
                        primary_signals[idx] = -1
            
            # Train meta-labeler
            meta_labeler = MetaLabeler(config=MetaLabelConfig())
            try:
                meta_labeler.fit(
                    features=train_features.fillna(0),
                    primary_signals=primary_signals,
                    triple_barrier_labels=aligned_labels,
                    sample_weight=aligned_weights.values if use_sample_weights else None
                )
            except Exception as e:
                start_idx += cfg.test_size
                continue
            
            # Test phase
            test_features_df = primary_model.generate_features(test_df)
            if len(test_features_df) < 10:
                start_idx += cfg.test_size
                continue
            
            test_proba = primary_model.predict_proba(test_df)
            
            test_primary_signals = pd.Series(0, index=test_features_df.index)
            for idx in test_features_df.index:
                if idx in test_proba.index:
                    prob_up = test_proba.loc[idx, 1] if 1 in test_proba.columns else 0.5
                    if prob_up > 0.55:
                        test_primary_signals[idx] = 1
                    elif prob_up < 0.45:
                        test_primary_signals[idx] = -1
            
            meta_proba = meta_labeler.predict_proba(test_features_df.fillna(0), test_primary_signals)
            
            positions = pd.Series(0, index=test_df.index)
            for i, idx in enumerate(test_features_df.index):
                if i >= len(meta_proba):
                    continue
                signal = test_primary_signals[idx]
                confidence = meta_proba[i, 1] if meta_proba.ndim > 1 else meta_proba[i]
                if signal != 0 and confidence > cfg.meta_threshold:
                    positions[idx] = signal
            
            positions = positions.replace(0, np.nan).ffill().fillna(0)
            
            test_returns = test_df['close'].pct_change().fillna(0)
            strategy_returns = positions.shift(1).fillna(0) * test_returns
            
            position_changes = positions.diff().abs()
            costs = position_changes * cfg.transaction_cost
            strategy_returns = strategy_returns - costs
            
            results['daily_returns'].extend(strategy_returns.values)
            
            for ret in strategy_returns[position_changes > 0]:
                if ret != 0:
                    results['trade_returns'].append(ret)
            
            folds_processed += 1
            start_idx += cfg.test_size
        
        daily_returns = np.array(results['daily_returns'])
        
        if len(daily_returns) == 0:
            return {'total_return': 0, 'sharpe': 0, 'trades': 0, 'win_rate': 0, 'avg_weight': 0}
        
        total_return = (1 + daily_returns).prod() - 1
        sharpe = daily_returns.mean() / (daily_returns.std() + 1e-8) * np.sqrt(252)
        
        trades = len(results['trade_returns'])
        win_rate = sum(1 for r in results['trade_returns'] if r > 0) / max(trades, 1)
        avg_weight = np.mean(results['sample_weights_used']) if results['sample_weights_used'] else 0
        
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'trades': trades,
            'win_rate': win_rate,
            'avg_weight': avg_weight,
            'folds': folds_processed
        }


def run_weighted_validation():
    """Run validation comparing weighted vs unweighted."""
    
    print("="*70)
    print("PRIORITY 3: AFML SAMPLE-WEIGHTED VALIDATION")
    print("="*70)
    print("\nGoal: Reduce overfitting from overlapping Triple Barrier labels")
    print("Method: Weight samples by uniqueness (AFML Chapter 4)")
    print("="*70)
    
    symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'MSFT', 'NVDA']
    
    results_weighted = {}
    results_unweighted = {}
    
    for symbol in symbols:
        try:
            df = pd.read_csv(
                f'/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI/data/train/{symbol}.csv',
                index_col=0, parse_dates=True
            )
            
            if len(df) < 600:
                print(f"\n{symbol}: Insufficient data ({len(df)} rows)")
                continue
            
            print(f"\n{'='*60}")
            print(f"SYMBOL: {symbol} ({len(df)} days)")
            print("="*60)
            
            config = WeightedPipelineConfig(
                pt_sl=(2.0, 1.0),
                max_holding_period=10,
                meta_threshold=0.55,
                time_decay=0.95
            )
            
            pipeline = SampleWeightedPipeline(config)
            
            # Weighted
            print("\n[1] WEIGHTED PIPELINE (AFML Sample Weights)")
            result_w = pipeline.walk_forward_weighted(df, use_sample_weights=True)
            results_weighted[symbol] = result_w
            
            print(f"  Sharpe: {result_w['sharpe']:.2f}")
            print(f"  Return: {result_w['total_return']*100:.1f}%")
            print(f"  Trades: {result_w['trades']}")
            print(f"  Win Rate: {result_w['win_rate']*100:.1f}%")
            
            # Unweighted
            print("\n[2] UNWEIGHTED PIPELINE (baseline)")
            result_uw = pipeline.walk_forward_weighted(df, use_sample_weights=False)
            results_unweighted[symbol] = result_uw
            
            print(f"  Sharpe: {result_uw['sharpe']:.2f}")
            print(f"  Return: {result_uw['total_return']*100:.1f}%")
            print(f"  Trades: {result_uw['trades']}")
            
            # Comparison
            improvement = result_w['sharpe'] - result_uw['sharpe']
            print(f"\n[3] IMPROVEMENT: {improvement:+.2f}")
            
        except Exception as e:
            print(f"\n{symbol}: Error - {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: WEIGHTED vs UNWEIGHTED")
    print("="*70)
    
    print(f"\n{'Symbol':<8} {'Weighted':<12} {'Unweighted':<12} {'Change':<10}")
    print("-"*50)
    
    total_w, total_uw = 0, 0
    count = 0
    
    for symbol in symbols:
        if symbol in results_weighted and symbol in results_unweighted:
            w = results_weighted[symbol]['sharpe']
            uw = results_unweighted[symbol]['sharpe']
            diff = w - uw
            
            print(f"{symbol:<8} {w:+.2f}        {uw:+.2f}        {diff:+.2f}")
            
            total_w += w
            total_uw += uw
            count += 1
    
    if count > 0:
        avg_w = total_w / count
        avg_uw = total_uw / count
        
        print("-"*50)
        print(f"{'AVERAGE':<8} {avg_w:+.2f}        {avg_uw:+.2f}        {avg_w - avg_uw:+.2f}")
        
        print("\n" + "="*70)
        print("CONCLUSION")
        print("="*70)
        
        if avg_w > avg_uw:
            print(f"AFML Sample Weighting IMPROVED average Sharpe by {avg_w - avg_uw:.2f}")
        else:
            print(f"Sample weighting did not improve this configuration")
        
        print(f"\nWeighted Average Sharpe: {avg_w:.2f}")
        print(f"Unweighted Average Sharpe: {avg_uw:.2f}")
    
    return results_weighted, results_unweighted


if __name__ == "__main__":
    run_weighted_validation()
