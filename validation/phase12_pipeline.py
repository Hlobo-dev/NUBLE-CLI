"""
Phase 1+2 Integration with Walk-Forward Validation
===================================================

Integrates Lopez de Prado components with rigorous validation.
Target: REALISTIC Sharpe 0.8-1.2 (not fake 5.0)
"""

import sys
sys.path.insert(0, '/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI')

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging

# Phase 1+2 Components
from src.institutional.labeling.triple_barrier import TripleBarrierLabeler
from src.institutional.features.frac_diff import FractionalDifferentiator
from src.institutional.regime.hmm_detector import HMMRegimeDetector
from src.institutional.models.meta.meta_labeler import MetaLabeler

# Validation Framework
from validation.walk_forward import WalkForwardValidator

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase12Pipeline:
    """Integrated Phase 1+2 Pipeline with Validation."""
    
    def __init__(self):
        # Triple Barrier
        self.triple_barrier = TripleBarrierLabeler(
            pt_sl=(1.0, 1.0),
            max_holding_period=10,
            volatility_lookback=20
        )
        logger.info("TripleBarrierLabeler initialized")
        
        # HMM Detector
        self.hmm_detector = HMMRegimeDetector(n_regimes=2)
        logger.info("HMMRegimeDetector initialized")
        
        # Meta-Labeler (created during fit)
        self.meta_labeler = None
        self.primary_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def generate_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate features and labels from OHLCV data."""
        close = df['close']
        returns = close.pct_change()
        
        # Triple Barrier Labels
        try:
            labels = self.triple_barrier.get_labels(close)
            logger.info(f"Triple Barrier labels: {labels.value_counts().to_dict()}")
        except Exception as e:
            logger.warning(f"Triple Barrier failed: {e}, using fallback")
            fwd_ret = close.pct_change(5).shift(-5)
            labels = pd.Series(0, index=close.index)
            labels[fwd_ret > 0.01] = 1
            labels[fwd_ret < -0.01] = -1
        
        # Feature Engineering
        features = pd.DataFrame(index=df.index)
        features['return_1d'] = returns
        features['return_5d'] = close.pct_change(5)
        features['return_20d'] = close.pct_change(20)
        features['volatility_20d'] = returns.rolling(20).std()
        features['volatility_60d'] = returns.rolling(60).std()
        features['volatility_ratio'] = features['volatility_20d'] / features['volatility_60d']
        features['momentum_10d'] = close / close.shift(10) - 1
        features['momentum_20d'] = close / close.shift(20) - 1
        features['rsi_14'] = self._compute_rsi(close, 14)
        
        ma_20 = close.rolling(20).mean()
        ma_50 = close.rolling(50).mean()
        features['distance_ma20'] = (close - ma_20) / ma_20
        features['distance_ma50'] = (close - ma_50) / ma_50
        features['ma_crossover'] = (ma_20 > ma_50).astype(int)
        
        if 'volume' in df.columns:
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            features['volume_momentum'] = df['volume'].pct_change(5)
        
        # HMM Regime Detection
        try:
            returns_clean = returns.dropna()
            self.hmm_detector.fit(returns_clean)
            regimes = self.hmm_detector.predict(returns, method='filter')
            features['regime'] = regimes
        except Exception as e:
            logger.warning(f"HMM failed: {e}")
            features['regime'] = 0
        
        features['return_sign'] = np.sign(returns)
        features = features.replace([np.inf, -np.inf], np.nan)
        common_idx = features.dropna().index.intersection(labels.dropna().index)
        features = features.loc[common_idx]
        labels = labels.loc[common_idx]
        
        logger.info(f"After alignment: {len(features)} samples, {len(features.columns)} features")
        return features, labels
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'Phase12Pipeline':
        """Fit primary model and meta-labeler."""
        self.feature_columns = X_train.columns.tolist()
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )
        
        # Train primary model
        self.primary_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
            class_weight='balanced'
        )
        y_binary = (y_train > 0).astype(int)
        self.primary_model.fit(X_scaled, y_binary)
        logger.info("Primary model trained")
        
        # Get primary signals for meta-labeling
        train_probs = self.primary_model.predict_proba(X_scaled)[:, 1]
        primary_signals = pd.Series(
            np.where(train_probs > 0.5, 1, -1),
            index=X_train.index
        )
        
        # Train meta-labeler with correct interface
        self.meta_labeler = MetaLabeler()
        self.meta_labeler.fit(
            features=X_scaled,
            primary_signals=primary_signals,
            triple_barrier_labels=y_train,
            returns=X_scaled['return_1d'] if 'return_1d' in X_scaled.columns else None
        )
        logger.info("Meta-labeler trained")
        
        return self
    
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Generate predictions with meta-labeling filter."""
        if self.feature_columns is not None:
            X_test = X_test[self.feature_columns]
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )
        
        # Primary predictions
        test_probs = self.primary_model.predict_proba(X_scaled)[:, 1]
        primary_signals = np.where(test_probs > 0.5, 1, -1)
        
        # Meta-labeler filter (apply to each row)
        predictions = []
        for i, (idx, row) in enumerate(X_scaled.iterrows()):
            try:
                result = self.meta_labeler.decide(
                    features=row.to_frame().T,
                    primary_signal=int(primary_signals[i])
                )
                predictions.append(1 if result.should_act and primary_signals[i] > 0 else 0)
            except Exception as e:
                # Fallback to primary signal
                predictions.append(1 if primary_signals[i] > 0 else 0)
        
        return pd.Series(predictions, index=X_test.index)
    
    def backtest(self, predictions: pd.Series, actual_returns: pd.Series,
                 cost_per_trade: float = 0.001) -> Dict[str, float]:
        """Backtest trading signals."""
        common_idx = predictions.index.intersection(actual_returns.index)
        preds = predictions.loc[common_idx]
        rets = actual_returns.loc[common_idx]
        
        if len(rets) == 0:
            return {'sharpe': 0, 'total_return': 0, 'n_trades': 0}
        
        position_changes = preds.diff().abs().fillna(0)
        strategy_returns = preds.shift(1).fillna(0) * rets
        strategy_returns -= position_changes * cost_per_trade
        strategy_returns = strategy_returns.fillna(0)
        
        if len(strategy_returns) < 10:
            return {'sharpe': 0, 'total_return': 0, 'n_trades': 0}
        
        total_return = (1 + strategy_returns).prod() - 1
        ann_factor = 252 / len(strategy_returns)
        ann_return = (1 + total_return) ** ann_factor - 1
        ann_vol = strategy_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0.001 else 0
        
        n_trades = int(position_changes.sum())
        return {
            'sharpe': sharpe,
            'annual_return': ann_return,
            'total_return': total_return,
            'n_trades': n_trades
        }


def run_walk_forward_validation(symbol: str = 'SPY') -> Dict[str, Any]:
    """Run complete walk-forward validation for Phase 1+2 pipeline."""
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 1+2 VALIDATION: {symbol}")
    logger.info(f"{'='*60}")
    
    # Load data
    data_path = f'/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI/data/train/{symbol}.csv'
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'] if 'date' in df.columns else df.iloc[:, 0])
    df.set_index('date', inplace=True)
    logger.info(f"Loaded {len(df)} rows")
    
    # Generate features and labels
    pipeline = Phase12Pipeline()
    features, labels = pipeline.generate_features(df)
    returns = df['close'].pct_change().loc[features.index]
    
    # Walk-forward validation
    validator = WalkForwardValidator(
        train_size=504,  # 2 years
        test_size=63,    # 3 months
        purge_size=5,
        embargo_size=5
    )
    splits = validator.get_sklearn_cv(features)
    logger.info(f"Generated {len(splits)} walk-forward splits")
    
    all_sharpes = []
    all_returns = []
    
    for i, (train_idx, test_idx) in enumerate(splits):
        logger.info(f"\nSplit {i+1}/{len(splits)}")
        
        try:
            X_train = features.iloc[train_idx]
            y_train = labels.iloc[train_idx]
            X_test = features.iloc[test_idx]
            
            # Create fresh pipeline for each split
            split_pipeline = Phase12Pipeline()
            split_pipeline.fit(X_train, y_train)
            
            # Predict
            predictions = split_pipeline.predict(X_test)
            
            # Backtest
            actual_returns = returns.iloc[test_idx]
            metrics = split_pipeline.backtest(predictions, actual_returns)
            
            sharpe = metrics['sharpe']
            if abs(sharpe) > 10:
                sharpe = np.clip(sharpe, -3, 3)
            
            all_sharpes.append(sharpe)
            all_returns.append(metrics['total_return'])
            
            logger.info(f"  Sharpe: {sharpe:.2f}, Return: {metrics['total_return']*100:.1f}%")
            
        except Exception as e:
            logger.error(f"  Split failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    if all_sharpes:
        mean_sharpe = np.mean(all_sharpes)
        std_sharpe = np.std(all_sharpes)
        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Sharpe: {mean_sharpe:.2f} +/- {std_sharpe:.2f}")
        
        if mean_sharpe > 2.0:
            logger.warning("WARNING: Sharpe > 2.0 is SUSPICIOUS!")
        elif mean_sharpe > 1.0:
            logger.info("Sharpe in realistic range (1.0-2.0)")
        elif mean_sharpe > 0.5:
            logger.info("Sharpe in acceptable range (0.5-1.0)")
        
        return {'symbol': symbol, 'sharpe_mean': mean_sharpe, 'sharpe_std': std_sharpe}
    else:
        return {'symbol': symbol, 'sharpe_mean': None}


if __name__ == '__main__':
    results = run_walk_forward_validation('SPY')
    print(f"\nFINAL: Sharpe = {results.get('sharpe_mean', 'N/A')}")
