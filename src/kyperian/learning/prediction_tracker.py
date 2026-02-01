"""
Prediction Tracker

Tracks every prediction made by the fusion system for later analysis
and continuous learning.

Features:
- Log every signal with timestamp and context
- Store component signal contributions
- Match predictions to outcomes when known
- Export for analysis
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import logging
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class PredictionOutcome(Enum):
    """Outcome of a prediction."""
    PENDING = "pending"
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"  # Moved in right direction but didn't hit target
    EXPIRED = "expired"  # No clear resolution


@dataclass
class Prediction:
    """
    A single prediction from the fusion system.
    
    Tracks:
    - What was predicted
    - Component signals that contributed
    - Outcome when resolved
    - Context at time of prediction
    """
    prediction_id: str
    timestamp: datetime
    symbol: str
    
    # What was predicted
    direction: int              # -1, 0, 1
    confidence: float           # 0 to 1
    recommended_size: float     # 0 to 1
    
    # Component signals
    component_signals: Dict[str, Dict] = field(default_factory=dict)
    
    # Context
    regime: str = "UNKNOWN"
    price_at_prediction: float = 0
    
    # Risk levels
    stop_loss_pct: float = 0
    take_profit_pct: float = 0
    
    # Resolution
    outcome: PredictionOutcome = PredictionOutcome.PENDING
    outcome_timestamp: Optional[datetime] = None
    outcome_price: Optional[float] = None
    outcome_return: Optional[float] = None
    outcome_notes: str = ""
    
    # Which sources were correct
    source_outcomes: Dict[str, bool] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'prediction_id': self.prediction_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'direction': self.direction,
            'confidence': self.confidence,
            'recommended_size': self.recommended_size,
            'component_signals': self.component_signals,
            'regime': self.regime,
            'price_at_prediction': self.price_at_prediction,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'outcome': self.outcome.value,
            'outcome_timestamp': self.outcome_timestamp.isoformat() if self.outcome_timestamp else None,
            'outcome_price': self.outcome_price,
            'outcome_return': self.outcome_return,
            'source_outcomes': self.source_outcomes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Prediction':
        """Create prediction from dict."""
        return cls(
            prediction_id=data['prediction_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            symbol=data['symbol'],
            direction=data['direction'],
            confidence=data['confidence'],
            recommended_size=data.get('recommended_size', 0),
            component_signals=data.get('component_signals', {}),
            regime=data.get('regime', 'UNKNOWN'),
            price_at_prediction=data.get('price_at_prediction', 0),
            stop_loss_pct=data.get('stop_loss_pct', 0),
            take_profit_pct=data.get('take_profit_pct', 0),
            outcome=PredictionOutcome(data.get('outcome', 'pending')),
            outcome_timestamp=datetime.fromisoformat(data['outcome_timestamp']) if data.get('outcome_timestamp') else None,
            outcome_price=data.get('outcome_price'),
            outcome_return=data.get('outcome_return'),
            source_outcomes=data.get('source_outcomes', {})
        )
    
    @property
    def is_resolved(self) -> bool:
        return self.outcome != PredictionOutcome.PENDING
    
    @property
    def was_correct(self) -> bool:
        return self.outcome == PredictionOutcome.CORRECT
    
    @property
    def action_str(self) -> str:
        return {1: 'BUY', -1: 'SELL', 0: 'HOLD'}.get(self.direction, 'HOLD')


class PredictionTracker:
    """
    Tracks predictions for learning and analysis.
    
    Features:
    - Log predictions from fusion engine
    - Match predictions to outcomes
    - Query by symbol, timeframe, source
    - Export for analysis
    
    Example:
        tracker = PredictionTracker()
        
        # Log a prediction
        pred_id = tracker.log_prediction(fused_signal, price=2340.50)
        
        # Later, resolve with outcome
        tracker.resolve_prediction(pred_id, outcome_price=2400.00)
        
        # Get accuracy stats
        stats = tracker.get_accuracy_stats()
    """
    
    def __init__(
        self,
        storage_path: str = None,
        max_predictions: int = 10000
    ):
        """
        Initialize tracker.
        
        Args:
            storage_path: Path to store predictions (JSON)
            max_predictions: Maximum predictions to keep in memory
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_predictions = max_predictions
        
        self.predictions: List[Prediction] = []
        self.predictions_by_symbol: Dict[str, List[Prediction]] = {}
        self.predictions_by_id: Dict[str, Prediction] = {}
        
        # Load existing predictions
        if self.storage_path and self.storage_path.exists():
            self._load_predictions()
    
    def log_prediction(
        self,
        signal,  # FusedSignal
        price: float = None,
        notes: str = ""
    ) -> str:
        """
        Log a new prediction.
        
        Args:
            signal: FusedSignal from fusion engine
            price: Current price at prediction time
            notes: Optional notes
            
        Returns:
            prediction_id for later resolution
        """
        prediction_id = str(uuid.uuid4())[:12]
        
        # Extract component signals
        component_signals = {}
        
        if signal.luxalgo_signal:
            component_signals['luxalgo'] = signal.luxalgo_signal
        if signal.ml_signal:
            component_signals['ml'] = signal.ml_signal
        if signal.sentiment_signal:
            component_signals['sentiment'] = signal.sentiment_signal
        
        prediction = Prediction(
            prediction_id=prediction_id,
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            direction=signal.direction,
            confidence=signal.confidence,
            recommended_size=signal.recommended_size,
            component_signals=component_signals,
            regime=signal.regime,
            price_at_prediction=price or 0,
            stop_loss_pct=signal.stop_loss_pct,
            take_profit_pct=signal.take_profit_pct
        )
        
        # Store
        self._add_prediction(prediction)
        
        logger.info(
            f"Logged prediction {prediction_id}: "
            f"{signal.symbol} {prediction.action_str} "
            f"(conf={signal.confidence:.0%})"
        )
        
        return prediction_id
    
    def resolve_prediction(
        self,
        prediction_id: str,
        outcome_price: float,
        outcome_timestamp: datetime = None
    ) -> Optional[Prediction]:
        """
        Resolve a prediction with its outcome.
        
        Args:
            prediction_id: ID of prediction to resolve
            outcome_price: Price at resolution
            outcome_timestamp: When resolved (default now)
            
        Returns:
            Updated Prediction
        """
        prediction = self.predictions_by_id.get(prediction_id)
        if not prediction:
            logger.warning(f"Prediction {prediction_id} not found")
            return None
        
        if prediction.is_resolved:
            logger.warning(f"Prediction {prediction_id} already resolved")
            return prediction
        
        prediction.outcome_price = outcome_price
        prediction.outcome_timestamp = outcome_timestamp or datetime.now()
        
        # Calculate return
        if prediction.price_at_prediction > 0:
            price_change = outcome_price - prediction.price_at_prediction
            prediction.outcome_return = price_change / prediction.price_at_prediction
        else:
            prediction.outcome_return = 0
        
        # Determine outcome
        prediction.outcome = self._determine_outcome(prediction)
        
        # Determine source outcomes
        prediction.source_outcomes = self._determine_source_outcomes(prediction)
        
        logger.info(
            f"Resolved prediction {prediction_id}: "
            f"{prediction.outcome.value} "
            f"(return={prediction.outcome_return:.2%})"
        )
        
        # Save
        self._save_predictions()
        
        return prediction
    
    def _determine_outcome(self, prediction: Prediction) -> PredictionOutcome:
        """Determine if prediction was correct."""
        if prediction.outcome_return is None:
            return PredictionOutcome.PENDING
        
        ret = prediction.outcome_return
        
        if prediction.direction == 0:
            # HOLD prediction - correct if small move
            if abs(ret) < 0.02:
                return PredictionOutcome.CORRECT
            else:
                return PredictionOutcome.INCORRECT
        
        elif prediction.direction == 1:  # BUY
            if ret >= prediction.take_profit_pct:
                return PredictionOutcome.CORRECT
            elif ret <= -prediction.stop_loss_pct:
                return PredictionOutcome.INCORRECT
            elif ret > 0:
                return PredictionOutcome.PARTIAL
            else:
                return PredictionOutcome.INCORRECT
        
        else:  # SELL (direction == -1)
            if ret <= -prediction.take_profit_pct:
                return PredictionOutcome.CORRECT
            elif ret >= prediction.stop_loss_pct:
                return PredictionOutcome.INCORRECT
            elif ret < 0:
                return PredictionOutcome.PARTIAL
            else:
                return PredictionOutcome.INCORRECT
    
    def _determine_source_outcomes(
        self, 
        prediction: Prediction
    ) -> Dict[str, bool]:
        """Determine which sources were correct."""
        outcomes = {}
        
        # Actual direction
        if prediction.outcome_return is None:
            return outcomes
        
        actual_dir = 1 if prediction.outcome_return > 0 else -1 if prediction.outcome_return < 0 else 0
        
        for source, data in prediction.component_signals.items():
            source_dir = data.get('direction', 0)
            
            # Source is correct if it predicted the right direction
            if source_dir > 0.1:
                outcomes[source] = actual_dir == 1
            elif source_dir < -0.1:
                outcomes[source] = actual_dir == -1
            else:
                outcomes[source] = abs(prediction.outcome_return) < 0.02
        
        return outcomes
    
    def _add_prediction(self, prediction: Prediction):
        """Add prediction to storage."""
        self.predictions.append(prediction)
        self.predictions_by_id[prediction.prediction_id] = prediction
        
        if prediction.symbol not in self.predictions_by_symbol:
            self.predictions_by_symbol[prediction.symbol] = []
        self.predictions_by_symbol[prediction.symbol].append(prediction)
        
        # Trim if needed
        if len(self.predictions) > self.max_predictions:
            removed = self.predictions.pop(0)
            del self.predictions_by_id[removed.prediction_id]
            if removed.symbol in self.predictions_by_symbol:
                self.predictions_by_symbol[removed.symbol] = [
                    p for p in self.predictions_by_symbol[removed.symbol]
                    if p.prediction_id != removed.prediction_id
                ]
        
        # Auto-save
        self._save_predictions()
    
    def get_pending_predictions(self, symbol: str = None) -> List[Prediction]:
        """Get predictions awaiting resolution."""
        if symbol:
            preds = self.predictions_by_symbol.get(symbol, [])
        else:
            preds = self.predictions
        
        return [p for p in preds if not p.is_resolved]
    
    def get_resolved_predictions(
        self, 
        symbol: str = None,
        last_n: int = None
    ) -> List[Prediction]:
        """Get resolved predictions."""
        if symbol:
            preds = self.predictions_by_symbol.get(symbol, [])
        else:
            preds = self.predictions
        
        resolved = [p for p in preds if p.is_resolved]
        
        if last_n:
            resolved = resolved[-last_n:]
        
        return resolved
    
    def get_accuracy_stats(
        self,
        symbol: str = None,
        last_n: int = None
    ) -> Dict:
        """
        Get accuracy statistics.
        
        Returns:
            Dict with accuracy metrics by source and overall
        """
        resolved = self.get_resolved_predictions(symbol, last_n)
        
        if not resolved:
            return {
                'total_predictions': 0,
                'resolved_predictions': 0,
                'overall_accuracy': 0,
                'source_accuracy': {}
            }
        
        # Overall accuracy
        correct = sum(1 for p in resolved if p.was_correct)
        overall_accuracy = correct / len(resolved)
        
        # Accuracy by source
        source_accuracy = {}
        source_totals = {}
        
        for pred in resolved:
            for source, was_correct in pred.source_outcomes.items():
                if source not in source_totals:
                    source_totals[source] = {'correct': 0, 'total': 0}
                
                source_totals[source]['total'] += 1
                if was_correct:
                    source_totals[source]['correct'] += 1
        
        for source, totals in source_totals.items():
            if totals['total'] > 0:
                source_accuracy[source] = {
                    'accuracy': totals['correct'] / totals['total'],
                    'correct': totals['correct'],
                    'total': totals['total']
                }
        
        # Accuracy by regime
        regime_accuracy = {}
        for pred in resolved:
            regime = pred.regime
            if regime not in regime_accuracy:
                regime_accuracy[regime] = {'correct': 0, 'total': 0}
            
            regime_accuracy[regime]['total'] += 1
            if pred.was_correct:
                regime_accuracy[regime]['correct'] += 1
        
        for regime in regime_accuracy:
            total = regime_accuracy[regime]['total']
            if total > 0:
                regime_accuracy[regime]['accuracy'] = (
                    regime_accuracy[regime]['correct'] / total
                )
        
        return {
            'total_predictions': len(self.predictions),
            'resolved_predictions': len(resolved),
            'pending_predictions': len(self.predictions) - len(resolved),
            'overall_accuracy': overall_accuracy,
            'correct_predictions': correct,
            'source_accuracy': source_accuracy,
            'regime_accuracy': regime_accuracy
        }
    
    def _save_predictions(self):
        """Save predictions to storage."""
        if not self.storage_path:
            return
        
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = [p.to_dict() for p in self.predictions]
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save predictions: {e}")
    
    def _load_predictions(self):
        """Load predictions from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for pred_data in data:
                pred = Prediction.from_dict(pred_data)
                self.predictions.append(pred)
                self.predictions_by_id[pred.prediction_id] = pred
                
                if pred.symbol not in self.predictions_by_symbol:
                    self.predictions_by_symbol[pred.symbol] = []
                self.predictions_by_symbol[pred.symbol].append(pred)
            
            logger.info(f"Loaded {len(self.predictions)} predictions")
        except Exception as e:
            logger.warning(f"Failed to load predictions: {e}")
    
    def export_for_analysis(self) -> List[Dict]:
        """Export all predictions for external analysis."""
        return [p.to_dict() for p in self.predictions]
