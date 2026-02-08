"""
LearningHub — Singleton that coordinates all learning components.
Imported by Manager and Orchestrator to track predictions and adjust weights.

Adapts the existing PredictionTracker, AccuracyMonitor, and WeightAdjuster
to a simple interface that the runtime system can call without knowing
the internal learning architecture.
"""
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
import logging
import uuid

logger = logging.getLogger(__name__)


class LearningHub:
    """Thread-safe singleton coordinating prediction tracking, accuracy monitoring, and weight adjustment."""

    _instance: Optional['LearningHub'] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Storage directory
        self.data_dir = Path.home() / '.nuble' / 'learning'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Import existing learning components
        from nuble.learning.prediction_tracker import PredictionTracker
        from nuble.learning.accuracy_monitor import AccuracyMonitor
        from nuble.learning.weight_adjuster import WeightAdjuster

        self.tracker = PredictionTracker(
            storage_path=str(self.data_dir / 'predictions.json')
        )
        self.monitor = AccuracyMonitor()

        # WeightAdjuster needs base_weights in its constructor
        default_weights = {
            'technical_luxalgo': 0.20,
            'technical_classic': 0.03,
            'ml_ensemble': 0.12,
            'sentiment_finbert': 0.08,
            'sentiment_news': 0.08,
            'regime_hmm': 0.07,
            'macro_context': 0.05,
            'fundamental': 0.05,
        }
        self.adjuster = WeightAdjuster(base_weights=default_weights)

        # Current learned weights (loaded from disk or defaults)
        self._weights_path = self.data_dir / 'learned_weights.json'
        self.signal_weights = self._load_weights()

        # Thread safety for weight updates
        self._weights_lock = threading.Lock()

        # In-memory store for raw prediction records (keyed by id)
        # These are dicts that don't go through the FusedSignal-based tracker
        self._raw_predictions: Dict[str, Dict[str, Any]] = {}
        self._raw_predictions_path = self.data_dir / 'raw_predictions.json'
        self._load_raw_predictions()

        logger.info(
            f"LearningHub initialized — storage: {self.data_dir}, "
            f"{len(self._raw_predictions)} raw predictions loaded"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Weight persistence
    # ──────────────────────────────────────────────────────────────────────

    def _load_weights(self) -> Dict[str, float]:
        """Load learned weights from disk, or return defaults."""
        if self._weights_path.exists():
            try:
                with open(self._weights_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'technical_luxalgo': 0.20,
            'technical_classic': 0.03,
            'ml_ensemble': 0.12,
            'sentiment_finbert': 0.08,
            'sentiment_news': 0.08,
            'regime_hmm': 0.07,
            'macro_context': 0.05,
            'fundamental': 0.05,
        }

    def _save_weights(self):
        """Persist current weights to disk."""
        try:
            with open(self._weights_path, 'w') as f:
                json.dump(self.signal_weights, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save weights: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Raw prediction store (dict-based, doesn't require FusedSignal)
    # ──────────────────────────────────────────────────────────────────────

    def _load_raw_predictions(self):
        """Load raw predictions from disk."""
        if self._raw_predictions_path.exists():
            try:
                with open(self._raw_predictions_path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for rec in data:
                        pid = rec.get('id', str(uuid.uuid4())[:12])
                        self._raw_predictions[pid] = rec
                elif isinstance(data, dict):
                    self._raw_predictions = data
            except Exception as e:
                logger.warning(f"Failed to load raw predictions: {e}")

    def _save_raw_predictions(self):
        """Persist raw predictions to disk."""
        try:
            with open(self._raw_predictions_path, 'w') as f:
                json.dump(list(self._raw_predictions.values()), f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save raw predictions: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Public API — called by Manager and Orchestrator
    # ──────────────────────────────────────────────────────────────────────

    def record_prediction(
        self,
        symbol: str,
        direction: str,           # "BULLISH", "BEARISH", "NEUTRAL"
        confidence: float,
        price_at_prediction: float,
        source: str,              # "decision_engine", "ml_ensemble", "apex_full"
        signal_snapshot: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a prediction. Returns prediction_id for later resolution.

        This bypasses the FusedSignal-based PredictionTracker.log_prediction()
        (which requires a FusedSignal object we don't have at runtime) and
        stores a plain dict that we can resolve later.
        """
        prediction_id = f"{symbol}_{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"

        record = {
            'id': prediction_id,
            'symbol': symbol,
            'direction': direction,
            'confidence': confidence,
            'price_at_prediction': price_at_prediction,
            'source': source,
            'signal_snapshot': signal_snapshot,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'resolved': False,
            'outcome': None,
            'outcome_price': None,
            'actual_return': None,
        }

        self._raw_predictions[prediction_id] = record
        self._save_raw_predictions()

        logger.info(
            f"Recorded prediction {prediction_id}: "
            f"{symbol} {direction} (conf={confidence:.0%}, "
            f"price={price_at_prediction:.2f}, source={source})"
        )
        return prediction_id

    def resolve_predictions(self, symbol: str, current_price: float):
        """
        Called periodically (e.g., hourly) to resolve outstanding predictions.
        Checks predictions from 1d, 5d, 20d ago and scores them.
        Updates accuracy monitor and triggers weight adjustment.
        """
        now = datetime.now()
        resolved_count = 0

        for pid, pred in list(self._raw_predictions.items()):
            if pred.get('resolved'):
                continue
            if pred.get('symbol') != symbol:
                continue

            try:
                pred_time = datetime.fromisoformat(pred['timestamp'])
            except (ValueError, KeyError):
                continue

            age_hours = (now - pred_time).total_seconds() / 3600

            # Resolve at 24h, 120h (5d), or 480h (20d)
            if age_hours < 24:
                continue

            entry_price = pred.get('price_at_prediction', 0)
            if entry_price <= 0:
                continue

            actual_return = (current_price - entry_price) / entry_price
            predicted_direction = pred.get('direction', 'NEUTRAL')

            # Determine if prediction was correct
            if predicted_direction in ('BULLISH', 'BUY', 'LONG'):
                was_correct = actual_return > 0
            elif predicted_direction in ('BEARISH', 'SELL', 'SHORT'):
                was_correct = actual_return < 0
            elif predicted_direction in ('NEUTRAL', 'HOLD'):
                was_correct = abs(actual_return) < 0.02  # <2% move = neutral correct
            else:
                was_correct = False

            # Update the record
            pred['resolved'] = True
            pred['outcome'] = 'CORRECT' if was_correct else 'INCORRECT'
            pred['outcome_price'] = current_price
            pred['actual_return'] = actual_return
            pred['resolved_at'] = now.isoformat()

            # Feed into AccuracyMonitor
            source = pred.get('source', 'unknown')
            regime = pred.get('signal_snapshot', {}).get('regime', 'UNKNOWN')
            self.monitor.record_outcome(
                source=source,
                was_correct=was_correct,
                symbol=symbol,
                regime=regime,
            )

            # Feed into WeightAdjuster
            self.adjuster.record_outcome(
                source=source,
                was_correct=was_correct,
                symbol=symbol,
                regime=regime,
            )

            resolved_count += 1

        if resolved_count > 0:
            self._save_raw_predictions()

            # Update learned weights from adjuster
            new_weights = self.adjuster.get_weights()
            if new_weights:
                with self._weights_lock:
                    self.signal_weights = new_weights
                    self._save_weights()

            logger.info(
                f"Resolved {resolved_count} predictions for {symbol} "
                f"at price ${current_price:.2f}"
            )

    def get_unresolved(self) -> List[Dict[str, Any]]:
        """Get all unresolved predictions (for the resolver background task)."""
        return [
            p for p in self._raw_predictions.values()
            if not p.get('resolved')
        ]

    def get_weights(self) -> Dict[str, float]:
        """Get current learned signal weights. Thread-safe."""
        with self._weights_lock:
            return dict(self.signal_weights)

    def get_accuracy_report(self) -> Dict[str, Any]:
        """Get current accuracy statistics from AccuracyMonitor."""
        return self.monitor.get_all_accuracies()

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction tracking statistics."""
        total = len(self._raw_predictions)
        resolved = sum(1 for p in self._raw_predictions.values() if p.get('resolved'))
        correct = sum(
            1 for p in self._raw_predictions.values()
            if p.get('outcome') == 'CORRECT'
        )

        # Also pull from the formal PredictionTracker if it has data
        tracker_stats = self.tracker.get_accuracy_stats()

        return {
            'raw_predictions': {
                'total': total,
                'resolved': resolved,
                'pending': total - resolved,
                'correct': correct,
                'accuracy': correct / resolved if resolved > 0 else 0.0,
            },
            'formal_tracker': tracker_stats,
        }
