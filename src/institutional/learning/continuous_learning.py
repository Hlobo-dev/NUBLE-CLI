#!/usr/bin/env python3
"""
Continuous Learning Engine

Features:
1. Daily prediction tracking
2. Model drift detection
3. Automated retraining triggers
4. A/B testing of model updates
5. Performance monitoring dashboard

This ensures the system improves over time and doesn't decay.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any
import json
from pathlib import Path
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DriftType(Enum):
    """Types of model drift."""
    ACCURACY_DROP = "accuracy_drop"
    SHARPE_DEGRADATION = "sharpe_degradation"
    CALIBRATION_DRIFT = "calibration_drift"
    DISTRIBUTION_SHIFT = "distribution_shift"
    CONCEPT_DRIFT = "concept_drift"


@dataclass
class PredictionRecord:
    """Record of a single prediction for tracking."""
    timestamp: datetime
    symbol: str
    predicted_direction: int        # -1, 0, +1
    predicted_confidence: float     # 0 to 1
    predicted_magnitude: float      # Expected return magnitude
    model_version: str = "v1"
    regime: str = "unknown"
    
    # Outcome (filled later)
    actual_return: Optional[float] = None
    correct: Optional[bool] = None
    profit_loss: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'predicted_direction': int(self.predicted_direction),
            'predicted_confidence': float(self.predicted_confidence),
            'predicted_magnitude': float(self.predicted_magnitude),
            'model_version': self.model_version,
            'regime': self.regime,
            'actual_return': float(self.actual_return) if self.actual_return is not None else None,
            'correct': bool(self.correct) if self.correct is not None else None,
            'profit_loss': float(self.profit_loss) if self.profit_loss is not None else None
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'PredictionRecord':
        return cls(
            timestamp=datetime.fromisoformat(d['timestamp']),
            symbol=d['symbol'],
            predicted_direction=d['predicted_direction'],
            predicted_confidence=d['predicted_confidence'],
            predicted_magnitude=d.get('predicted_magnitude', 0.0),
            model_version=d.get('model_version', 'v1'),
            regime=d.get('regime', 'unknown'),
            actual_return=d.get('actual_return'),
            correct=d.get('correct'),
            profit_loss=d.get('profit_loss')
        )


@dataclass
class ModelPerformance:
    """Performance metrics for a model version."""
    model_version: str
    start_date: datetime
    end_date: datetime
    n_predictions: int
    
    # Core metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Trading metrics
    sharpe: float
    total_return: float
    max_drawdown: float
    hit_rate: float
    profit_factor: float
    
    # Calibration
    avg_confidence: float
    calibration_error: float
    
    # Comparison to baseline
    accuracy_vs_baseline: float
    sharpe_vs_baseline: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class DriftAlert:
    """Alert for model drift detection."""
    timestamp: datetime
    alert_type: DriftType
    severity: AlertSeverity
    metric_name: str
    expected_value: float
    actual_value: float
    deviation_pct: float
    lookback_days: int
    recommendation: str
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'metric_name': self.metric_name,
            'expected_value': self.expected_value,
            'actual_value': self.actual_value,
            'deviation_pct': self.deviation_pct,
            'lookback_days': self.lookback_days,
            'recommendation': self.recommendation
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'DriftAlert':
        return cls(
            timestamp=datetime.fromisoformat(d['timestamp']),
            alert_type=DriftType(d['alert_type']),
            severity=AlertSeverity(d['severity']),
            metric_name=d['metric_name'],
            expected_value=d['expected_value'],
            actual_value=d['actual_value'],
            deviation_pct=d['deviation_pct'],
            lookback_days=d.get('lookback_days', 30),
            recommendation=d['recommendation']
        )


class ContinuousLearningEngine:
    """
    Continuous learning system for model improvement.
    
    Workflow:
    1. Track all predictions and outcomes
    2. Monitor for performance degradation
    3. Detect distribution shifts
    4. Trigger retraining when thresholds breached
    5. A/B test new models before deployment
    
    Example:
        engine = ContinuousLearningEngine(
            baseline_accuracy=0.54,
            baseline_sharpe=0.41
        )
        
        # Record prediction
        engine.record_prediction(
            symbol='AAPL',
            direction=1,
            confidence=0.65
        )
        
        # Later, record outcome
        engine.record_outcome(
            symbol='AAPL',
            prediction_time=pred.timestamp,
            actual_return=0.02
        )
        
        # Check for drift
        alerts = engine.check_for_drift()
        if alerts:
            print(f"⚠️ {len(alerts)} drift alerts!")
    """
    
    def __init__(
        self,
        storage_path: str = "learning_data",
        baseline_accuracy: float = 0.54,
        baseline_sharpe: float = 0.41,
        drift_threshold: float = 0.10,
        min_samples_for_evaluation: int = 50,
        alert_callback: Optional[Callable[[DriftAlert], None]] = None
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Baselines from validation
        self.baseline_accuracy = baseline_accuracy
        self.baseline_sharpe = baseline_sharpe
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples_for_evaluation
        self.alert_callback = alert_callback
        
        # State
        self.predictions: List[PredictionRecord] = []
        self.alerts: List[DriftAlert] = []
        self.model_versions: Dict[str, ModelPerformance] = {}
        self.current_model_version = "v1"
        
        # Rolling metrics
        self._rolling_accuracy: List[float] = []
        self._rolling_sharpe: List[float] = []
        
        # Load history
        self._load_state()
        
        logger.info(
            f"ContinuousLearningEngine initialized: "
            f"{len(self.predictions)} predictions, "
            f"{len(self.alerts)} alerts"
        )
    
    def record_prediction(
        self,
        symbol: str,
        direction: int,
        confidence: float,
        magnitude: float = 0.0,
        model_version: str = None,
        regime: str = "unknown"
    ) -> PredictionRecord:
        """
        Record a new prediction.
        
        Args:
            symbol: Stock symbol
            direction: Predicted direction (-1, 0, +1)
            confidence: Prediction confidence (0-1)
            magnitude: Expected return magnitude
            model_version: Model version (default: current)
            regime: Market regime
            
        Returns:
            PredictionRecord that was created
        """
        record = PredictionRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            predicted_direction=direction,
            predicted_confidence=confidence,
            predicted_magnitude=magnitude,
            model_version=model_version or self.current_model_version,
            regime=regime
        )
        
        self.predictions.append(record)
        self._save_state()
        
        logger.debug(
            f"Recorded prediction: {symbol} → {direction} "
            f"(conf: {confidence:.2f})"
        )
        
        return record
    
    def record_outcome(
        self,
        symbol: str,
        prediction_time: datetime,
        actual_return: float,
        time_tolerance_hours: int = 24
    ) -> Optional[PredictionRecord]:
        """
        Record the actual outcome for a prediction.
        
        Args:
            symbol: Stock symbol
            prediction_time: Time of original prediction
            actual_return: Actual return that occurred
            time_tolerance_hours: How close in time to match
            
        Returns:
            Updated PredictionRecord or None if not found
        """
        tolerance = timedelta(hours=time_tolerance_hours)
        
        # Find matching prediction
        for pred in reversed(self.predictions):
            if (pred.symbol == symbol and 
                abs(pred.timestamp - prediction_time) < tolerance and
                pred.actual_return is None):
                
                pred.actual_return = actual_return
                pred.correct = self._is_correct(
                    pred.predicted_direction,
                    actual_return
                )
                pred.profit_loss = actual_return * pred.predicted_direction
                
                self._save_state()
                
                logger.info(
                    f"Recorded outcome: {symbol} "
                    f"actual={actual_return:.4f} "
                    f"correct={pred.correct}"
                )
                
                return pred
        
        logger.warning(
            f"No matching prediction found for {symbol} "
            f"at {prediction_time}"
        )
        return None
    
    def _is_correct(self, predicted_direction: int, actual_return: float) -> bool:
        """Determine if prediction was correct."""
        if predicted_direction == 0:
            # Neutral prediction correct if return was small
            return abs(actual_return) < 0.01
        elif predicted_direction > 0:
            return actual_return > 0
        else:
            return actual_return < 0
    
    def check_for_drift(self, lookback_days: int = 30) -> List[DriftAlert]:
        """
        Check for model drift.
        
        Monitors:
        1. Accuracy degradation
        2. Sharpe ratio degradation
        3. Confidence calibration
        4. Distribution shifts
        
        Args:
            lookback_days: How many days to look back
            
        Returns:
            List of new DriftAlerts
        """
        new_alerts = []
        
        # Get recent predictions with outcomes
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent = [
            p for p in self.predictions 
            if p.actual_return is not None and p.timestamp > cutoff
        ]
        
        if len(recent) < self.min_samples:
            logger.info(
                f"Not enough samples for drift detection "
                f"({len(recent)} < {self.min_samples})"
            )
            return []
        
        # 1. Check accuracy
        accuracy = sum(1 for p in recent if p.correct) / len(recent)
        accuracy_drift = (self.baseline_accuracy - accuracy) / self.baseline_accuracy
        
        if accuracy_drift > self.drift_threshold:
            alert = DriftAlert(
                timestamp=datetime.now(),
                alert_type=DriftType.ACCURACY_DROP,
                severity=AlertSeverity.CRITICAL if accuracy_drift > 0.20 else AlertSeverity.WARNING,
                metric_name='accuracy',
                expected_value=self.baseline_accuracy,
                actual_value=accuracy,
                deviation_pct=accuracy_drift,
                lookback_days=lookback_days,
                recommendation='Consider retraining model with recent data'
            )
            new_alerts.append(alert)
            logger.warning(
                f"DRIFT ALERT: Accuracy dropped {accuracy_drift:.1%} "
                f"({accuracy:.1%} vs baseline {self.baseline_accuracy:.1%})"
            )
        
        # 2. Check Sharpe degradation
        returns = [
            p.actual_return * p.predicted_direction 
            for p in recent
        ]
        
        if len(returns) > 20:
            daily_returns = np.array(returns)
            sharpe = (
                np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                if np.std(daily_returns) > 0 else 0
            )
            sharpe_drift = (self.baseline_sharpe - sharpe) / max(self.baseline_sharpe, 0.01)
            
            if sharpe_drift > self.drift_threshold:
                alert = DriftAlert(
                    timestamp=datetime.now(),
                    alert_type=DriftType.SHARPE_DEGRADATION,
                    severity=AlertSeverity.CRITICAL if sharpe_drift > 0.30 else AlertSeverity.WARNING,
                    metric_name='sharpe_ratio',
                    expected_value=self.baseline_sharpe,
                    actual_value=sharpe,
                    deviation_pct=sharpe_drift,
                    lookback_days=lookback_days,
                    recommendation='Strategy performance degrading, investigate market regime'
                )
                new_alerts.append(alert)
                logger.warning(
                    f"DRIFT ALERT: Sharpe dropped {sharpe_drift:.1%} "
                    f"({sharpe:.2f} vs baseline {self.baseline_sharpe:.2f})"
                )
        
        # 3. Check calibration
        calibration = self._check_calibration(recent)
        if calibration['error'] > 0.15:
            alert = DriftAlert(
                timestamp=datetime.now(),
                alert_type=DriftType.CALIBRATION_DRIFT,
                severity=AlertSeverity.WARNING,
                metric_name='calibration_error',
                expected_value=0.05,
                actual_value=calibration['error'],
                deviation_pct=calibration['error'] / 0.05,
                lookback_days=lookback_days,
                recommendation='Model confidence scores need recalibration'
            )
            new_alerts.append(alert)
            logger.warning(
                f"DRIFT ALERT: Calibration error {calibration['error']:.1%}"
            )
        
        # 4. Check for distribution shift in predictions
        dist_shift = self._check_distribution_shift(recent, lookback_days)
        if dist_shift['shifted']:
            alert = DriftAlert(
                timestamp=datetime.now(),
                alert_type=DriftType.DISTRIBUTION_SHIFT,
                severity=AlertSeverity.WARNING,
                metric_name='prediction_distribution',
                expected_value=0.0,
                actual_value=dist_shift['shift_magnitude'],
                deviation_pct=dist_shift['shift_magnitude'],
                lookback_days=lookback_days,
                recommendation=dist_shift['message']
            )
            new_alerts.append(alert)
        
        # Store and callback
        self.alerts.extend(new_alerts)
        self._save_state()
        
        for alert in new_alerts:
            if self.alert_callback:
                self.alert_callback(alert)
        
        return new_alerts
    
    def _check_calibration(
        self, 
        predictions: List[PredictionRecord]
    ) -> Dict[str, Any]:
        """
        Check if confidence scores are well-calibrated.
        
        A well-calibrated model should have:
        - 60% confidence → 60% accuracy
        - 80% confidence → 80% accuracy
        """
        bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        calibration_errors = []
        bin_details = []
        
        for low, high in bins:
            bin_preds = [
                p for p in predictions 
                if low <= p.predicted_confidence < high
            ]
            
            if len(bin_preds) >= 5:
                expected = (low + high) / 2
                actual = sum(1 for p in bin_preds if p.correct) / len(bin_preds)
                error = abs(expected - actual)
                calibration_errors.append(error)
                bin_details.append({
                    'bin': f"{low:.1f}-{high:.1f}",
                    'n': len(bin_preds),
                    'expected': expected,
                    'actual': actual,
                    'error': error
                })
        
        return {
            'error': np.mean(calibration_errors) if calibration_errors else 0,
            'n_bins': len(calibration_errors),
            'details': bin_details
        }
    
    def _check_distribution_shift(
        self,
        recent: List[PredictionRecord],
        lookback_days: int
    ) -> Dict[str, Any]:
        """Check for shifts in prediction distribution."""
        if len(recent) < 20:
            return {'shifted': False, 'shift_magnitude': 0, 'message': ''}
        
        # Split into first half and second half
        mid = len(recent) // 2
        first_half = recent[:mid]
        second_half = recent[mid:]
        
        # Compare long/short ratios
        first_long_pct = sum(1 for p in first_half if p.predicted_direction > 0) / len(first_half)
        second_long_pct = sum(1 for p in second_half if p.predicted_direction > 0) / len(second_half)
        
        shift = abs(first_long_pct - second_long_pct)
        
        if shift > 0.20:
            direction = "more bullish" if second_long_pct > first_long_pct else "more bearish"
            return {
                'shifted': True,
                'shift_magnitude': shift,
                'message': f"Model becoming {direction} ({shift:.1%} shift in long signals)"
            }
        
        return {'shifted': False, 'shift_magnitude': shift, 'message': ''}
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Determine if model should be retrained.
        
        Returns:
            (should_retrain, reason)
        """
        recent_alerts = [
            a for a in self.alerts 
            if a.timestamp > datetime.now() - timedelta(days=7)
        ]
        
        critical_alerts = [
            a for a in recent_alerts 
            if a.severity == AlertSeverity.CRITICAL
        ]
        
        if len(critical_alerts) >= 2:
            return True, f"{len(critical_alerts)} critical alerts in past week"
        
        if len(recent_alerts) >= 5:
            return True, f"{len(recent_alerts)} total alerts in past week"
        
        return False, "No retrain needed"
    
    def get_performance_report(self, days: int = 30) -> Dict:
        """
        Generate comprehensive performance report.
        
        Args:
            days: Lookback period
            
        Returns:
            Dictionary with performance metrics
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent = [
            p for p in self.predictions 
            if p.actual_return is not None and p.timestamp > cutoff
        ]
        
        if not recent:
            return {'error': 'No predictions with outcomes', 'n_predictions': 0}
        
        # Core metrics
        accuracy = sum(1 for p in recent if p.correct) / len(recent)
        
        # Trading metrics
        returns = [p.actual_return * p.predicted_direction for p in recent]
        returns_arr = np.array(returns)
        
        sharpe = (
            np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252)
            if np.std(returns_arr) > 0 else 0
        )
        
        total_return = (1 + returns_arr).prod() - 1
        
        # Drawdown
        cumulative = (1 + returns_arr).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_drawdown = drawdowns.max()
        
        # Profit factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Hit rate (% of winning trades)
        winning = sum(1 for r in returns if r > 0)
        hit_rate = winning / len(returns)
        
        # By symbol
        by_symbol = {}
        for symbol in set(p.symbol for p in recent):
            sym_preds = [p for p in recent if p.symbol == symbol]
            sym_returns = [p.actual_return * p.predicted_direction for p in sym_preds]
            by_symbol[symbol] = {
                'n_predictions': len(sym_preds),
                'accuracy': sum(1 for p in sym_preds if p.correct) / len(sym_preds),
                'avg_confidence': np.mean([p.predicted_confidence for p in sym_preds]),
                'total_pnl': sum(sym_returns),
                'hit_rate': sum(1 for r in sym_returns if r > 0) / len(sym_returns)
            }
        
        # By regime
        by_regime = {}
        for regime in set(p.regime for p in recent):
            reg_preds = [p for p in recent if p.regime == regime]
            if len(reg_preds) >= 5:
                by_regime[regime] = {
                    'n_predictions': len(reg_preds),
                    'accuracy': sum(1 for p in reg_preds if p.correct) / len(reg_preds)
                }
        
        return {
            'period_days': days,
            'n_predictions': len(recent),
            
            # Core metrics
            'accuracy': accuracy,
            'baseline_accuracy': self.baseline_accuracy,
            'accuracy_vs_baseline': accuracy - self.baseline_accuracy,
            
            # Trading metrics
            'sharpe': sharpe,
            'baseline_sharpe': self.baseline_sharpe,
            'sharpe_vs_baseline': sharpe - self.baseline_sharpe,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate,
            'profit_factor': profit_factor,
            
            # Calibration
            'avg_confidence': np.mean([p.predicted_confidence for p in recent]),
            'calibration': self._check_calibration(recent),
            
            # Breakdown
            'by_symbol': by_symbol,
            'by_regime': by_regime,
            
            # Alerts
            'recent_alerts': len([a for a in self.alerts if a.timestamp > cutoff])
        }
    
    def get_summary(self) -> Dict:
        """Get quick summary of engine state."""
        total = len(self.predictions)
        with_outcomes = len([p for p in self.predictions if p.actual_return is not None])
        recent_alerts = len([
            a for a in self.alerts 
            if a.timestamp > datetime.now() - timedelta(days=7)
        ])
        
        should_retrain, retrain_reason = self.should_retrain()
        
        return {
            'total_predictions': total,
            'predictions_with_outcomes': with_outcomes,
            'outcome_rate': with_outcomes / total if total > 0 else 0,
            'current_model_version': self.current_model_version,
            'total_alerts': len(self.alerts),
            'recent_alerts': recent_alerts,
            'should_retrain': should_retrain,
            'retrain_reason': retrain_reason
        }
    
    def _save_state(self):
        """Save state to disk."""
        state = {
            'predictions': [p.to_dict() for p in self.predictions[-10000:]],
            'alerts': [a.to_dict() for a in self.alerts[-1000:]],
            'current_model_version': self.current_model_version,
            'baseline_accuracy': self.baseline_accuracy,
            'baseline_sharpe': self.baseline_sharpe,
            'last_saved': datetime.now().isoformat()
        }
        
        with open(self.storage_path / 'learning_state.json', 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load state from disk."""
        state_file = self.storage_path / 'learning_state.json'
        
        if not state_file.exists():
            logger.info("No existing state found, starting fresh")
            return
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.predictions = [
                PredictionRecord.from_dict(p)
                for p in state.get('predictions', [])
            ]
            
            self.alerts = [
                DriftAlert.from_dict(a)
                for a in state.get('alerts', [])
            ]
            
            self.current_model_version = state.get('current_model_version', 'v1')
            
            logger.info(
                f"Loaded state: {len(self.predictions)} predictions, "
                f"{len(self.alerts)} alerts"
            )
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")


class AutoRetrainer:
    """
    Automated model retraining system.
    
    Triggered by ContinuousLearningEngine when drift detected.
    Implements A/B testing of new models before deployment.
    """
    
    def __init__(
        self,
        learning_engine: ContinuousLearningEngine,
        validation_threshold: float = 0.95,  # New model must be 95% as good
        ab_test_size: float = 0.10           # 10% of traffic for new model
    ):
        self.engine = learning_engine
        self.validation_threshold = validation_threshold
        self.ab_test_size = ab_test_size
        self.candidate_models: Dict[str, Any] = {}
        self.ab_tests: Dict[str, Dict] = {}
    
    def check_and_retrain(self, train_func: Callable) -> Dict:
        """
        Check if retraining needed and execute if so.
        
        Args:
            train_func: Function that trains a new model
            
        Returns:
            Dictionary with action taken
        """
        should_retrain, reason = self.engine.should_retrain()
        
        if not should_retrain:
            return {'action': 'none', 'reason': reason}
        
        logger.info(f"Retraining triggered: {reason}")
        
        try:
            # Train new model
            new_model = train_func()
            
            # Generate new version
            new_version = f"v{len(self.engine.model_versions) + 2}"
            
            # Start A/B test
            self.start_ab_test(new_version, new_model)
            
            return {
                'action': 'ab_test_started',
                'reason': reason,
                'new_version': new_version
            }
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return {
                'action': 'failed',
                'reason': str(e)
            }
    
    def start_ab_test(self, version: str, model: Any):
        """Start A/B test for new model."""
        self.candidate_models[version] = model
        self.ab_tests[version] = {
            'start_time': datetime.now(),
            'predictions': [],
            'outcomes': [],
            'traffic_pct': self.ab_test_size
        }
        logger.info(f"A/B test started for {version} ({self.ab_test_size:.0%} traffic)")
    
    def should_use_candidate(self, version: str) -> bool:
        """Determine if we should use candidate model for this prediction."""
        if version not in self.ab_tests:
            return False
        
        return np.random.random() < self.ab_tests[version]['traffic_pct']
    
    def evaluate_ab_test(self, version: str) -> Dict:
        """Evaluate A/B test results."""
        if version not in self.ab_tests:
            return {'error': 'No A/B test found'}
        
        test = self.ab_tests[version]
        
        if len(test['outcomes']) < 50:
            return {
                'status': 'insufficient_data',
                'n_outcomes': len(test['outcomes'])
            }
        
        # Calculate metrics for candidate
        accuracy = sum(1 for o in test['outcomes'] if o['correct']) / len(test['outcomes'])
        
        # Compare to baseline
        baseline = self.engine.baseline_accuracy
        ratio = accuracy / baseline
        
        if ratio >= self.validation_threshold:
            return {
                'status': 'passed',
                'accuracy': accuracy,
                'baseline': baseline,
                'ratio': ratio,
                'recommendation': 'Deploy new model'
            }
        else:
            return {
                'status': 'failed',
                'accuracy': accuracy,
                'baseline': baseline,
                'ratio': ratio,
                'recommendation': 'Keep current model'
            }
    
    def promote_candidate(self, version: str):
        """Promote candidate model to production."""
        if version in self.candidate_models:
            self.engine.current_model_version = version
            del self.ab_tests[version]
            logger.info(f"Promoted {version} to production")


def run_learning_test():
    """Test the continuous learning engine."""
    print("="*60)
    print("CONTINUOUS LEARNING ENGINE TEST")
    print("="*60)
    
    import tempfile
    import shutil
    
    # Create temp directory for test
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize engine
        engine = ContinuousLearningEngine(
            storage_path=temp_dir,
            baseline_accuracy=0.54,
            baseline_sharpe=0.41,
            min_samples_for_evaluation=20
        )
        
        print(f"\nBaseline Accuracy: {engine.baseline_accuracy:.1%}")
        print(f"Baseline Sharpe: {engine.baseline_sharpe:.2f}")
        
        # Simulate predictions
        print("\n" + "-"*40)
        print("1. SIMULATING PREDICTIONS")
        print("-"*40)
        
        np.random.seed(42)
        symbols = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMD']
        
        for i in range(100):
            symbol = np.random.choice(symbols)
            direction = np.random.choice([-1, 0, 1], p=[0.3, 0.2, 0.5])
            confidence = np.random.uniform(0.5, 0.9)
            
            pred = engine.record_prediction(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                regime='BULL' if i < 50 else 'BEAR'
            )
            
            # Simulate outcome (with some alpha)
            if direction != 0:
                # Good model: 55% accuracy
                correct = np.random.random() < 0.55
                actual_return = np.random.uniform(0.01, 0.05) if correct else -np.random.uniform(0.01, 0.03)
                if direction < 0:
                    actual_return = -actual_return
            else:
                actual_return = np.random.uniform(-0.005, 0.005)
            
            engine.record_outcome(
                symbol=symbol,
                prediction_time=pred.timestamp,
                actual_return=actual_return
            )
        
        print(f"Recorded {len(engine.predictions)} predictions with outcomes")
        
        # Check for drift
        print("\n" + "-"*40)
        print("2. DRIFT DETECTION")
        print("-"*40)
        
        alerts = engine.check_for_drift(lookback_days=30)
        
        if alerts:
            print(f"\n⚠️ {len(alerts)} drift alerts detected:")
            for alert in alerts:
                print(f"   [{alert.severity.value.upper()}] {alert.alert_type.value}")
                print(f"       Expected: {alert.expected_value:.3f}")
                print(f"       Actual: {alert.actual_value:.3f}")
                print(f"       Deviation: {alert.deviation_pct:.1%}")
        else:
            print("\n✅ No significant drift detected")
        
        # Performance report
        print("\n" + "-"*40)
        print("3. PERFORMANCE REPORT")
        print("-"*40)
        
        report = engine.get_performance_report(days=30)
        
        print(f"\n📊 Overall Metrics:")
        print(f"   Predictions: {report['n_predictions']}")
        print(f"   Accuracy: {report['accuracy']:.1%} (vs baseline {report['baseline_accuracy']:.1%})")
        print(f"   Sharpe: {report['sharpe']:.2f} (vs baseline {report['baseline_sharpe']:.2f})")
        print(f"   Hit Rate: {report['hit_rate']:.1%}")
        print(f"   Profit Factor: {report['profit_factor']:.2f}")
        print(f"   Max Drawdown: {report['max_drawdown']:.1%}")
        
        print(f"\n📈 By Symbol:")
        for symbol, stats in report['by_symbol'].items():
            print(f"   {symbol}: {stats['accuracy']:.1%} accuracy, {stats['n_predictions']} preds")
        
        # Summary
        print("\n" + "-"*40)
        print("4. ENGINE SUMMARY")
        print("-"*40)
        
        summary = engine.get_summary()
        print(f"\n   Total predictions: {summary['total_predictions']}")
        print(f"   With outcomes: {summary['predictions_with_outcomes']}")
        print(f"   Model version: {summary['current_model_version']}")
        print(f"   Recent alerts: {summary['recent_alerts']}")
        print(f"   Should retrain: {summary['should_retrain']} ({summary['retrain_reason']})")
        
        print("\n" + "="*60)
        print("✅ CONTINUOUS LEARNING ENGINE: WORKING")
        print("="*60)
        
        return engine
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_learning_test()
