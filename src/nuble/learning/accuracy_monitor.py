"""
Accuracy Monitor

Monitors accuracy of signal sources and the overall system.
Provides insights for weight adjustment and system improvement.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class AccuracySnapshot:
    """Point-in-time accuracy measurement."""
    timestamp: datetime
    source: str
    accuracy: float
    predictions: int
    regime: str = "ALL"


class AccuracyMonitor:
    """
    Monitors and tracks accuracy metrics over time.
    
    Features:
    - Track accuracy by source, symbol, and regime
    - Calculate rolling accuracy windows
    - Detect accuracy degradation
    - Generate insights for improvement
    
    Example:
        monitor = AccuracyMonitor()
        
        # Record outcomes
        monitor.record_outcome('luxalgo', True)
        monitor.record_outcome('ml', False)
        
        # Get current accuracy
        acc = monitor.get_accuracy('luxalgo')
        
        # Check for degradation
        if monitor.is_degrading('luxalgo'):
            print("LuxAlgo accuracy is declining!")
    """
    
    def __init__(
        self,
        rolling_window: int = 100,
        degradation_threshold: float = 0.1
    ):
        """
        Initialize monitor.
        
        Args:
            rolling_window: Number of predictions for rolling accuracy
            degradation_threshold: Accuracy drop to trigger alert
        """
        self.rolling_window = rolling_window
        self.degradation_threshold = degradation_threshold
        
        # Outcome history by source
        self.outcomes: Dict[str, List[bool]] = defaultdict(list)
        
        # Outcome history by source + symbol
        self.outcomes_by_symbol: Dict[str, Dict[str, List[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Outcome history by source + regime
        self.outcomes_by_regime: Dict[str, Dict[str, List[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Accuracy snapshots for trend analysis
        self.accuracy_history: Dict[str, List[AccuracySnapshot]] = defaultdict(list)
    
    def record_outcome(
        self,
        source: str,
        was_correct: bool,
        symbol: str = None,
        regime: str = None
    ):
        """
        Record a prediction outcome.
        
        Args:
            source: Signal source name
            was_correct: Whether prediction was correct
            symbol: Asset symbol
            regime: Market regime at prediction time
        """
        # Record overall
        self.outcomes[source].append(was_correct)
        
        # Trim to rolling window
        if len(self.outcomes[source]) > self.rolling_window * 2:
            self.outcomes[source] = self.outcomes[source][-self.rolling_window:]
        
        # Record by symbol
        if symbol:
            self.outcomes_by_symbol[source][symbol].append(was_correct)
            if len(self.outcomes_by_symbol[source][symbol]) > self.rolling_window:
                self.outcomes_by_symbol[source][symbol] = (
                    self.outcomes_by_symbol[source][symbol][-self.rolling_window:]
                )
        
        # Record by regime
        if regime:
            self.outcomes_by_regime[source][regime].append(was_correct)
            if len(self.outcomes_by_regime[source][regime]) > self.rolling_window:
                self.outcomes_by_regime[source][regime] = (
                    self.outcomes_by_regime[source][regime][-self.rolling_window:]
                )
        
        # Take snapshot periodically
        if len(self.outcomes[source]) % 10 == 0:
            self._take_snapshot(source, regime)
    
    def _take_snapshot(self, source: str, regime: str = None):
        """Take accuracy snapshot for trend analysis."""
        accuracy = self.get_accuracy(source)
        
        snapshot = AccuracySnapshot(
            timestamp=datetime.now(),
            source=source,
            accuracy=accuracy,
            predictions=len(self.outcomes[source]),
            regime=regime or "ALL"
        )
        
        self.accuracy_history[source].append(snapshot)
        
        # Keep last 100 snapshots
        if len(self.accuracy_history[source]) > 100:
            self.accuracy_history[source] = self.accuracy_history[source][-100:]
    
    def get_accuracy(
        self,
        source: str,
        window: int = None
    ) -> float:
        """
        Get current accuracy for a source.
        
        Args:
            source: Signal source name
            window: Number of recent predictions (default: rolling_window)
            
        Returns:
            Accuracy as float (0 to 1)
        """
        window = window or self.rolling_window
        
        history = self.outcomes.get(source, [])
        if not history:
            return 0.5  # No data = neutral
        
        recent = history[-window:]
        return sum(recent) / len(recent)
    
    def get_accuracy_by_symbol(
        self,
        source: str,
        symbol: str,
        window: int = None
    ) -> float:
        """Get accuracy for a specific symbol."""
        window = window or self.rolling_window
        
        history = self.outcomes_by_symbol.get(source, {}).get(symbol, [])
        if not history:
            return 0.5
        
        recent = history[-window:]
        return sum(recent) / len(recent)
    
    def get_accuracy_by_regime(
        self,
        source: str,
        regime: str,
        window: int = None
    ) -> float:
        """Get accuracy for a specific regime."""
        window = window or self.rolling_window
        
        history = self.outcomes_by_regime.get(source, {}).get(regime, [])
        if not history:
            return 0.5
        
        recent = history[-window:]
        return sum(recent) / len(recent)
    
    def is_degrading(self, source: str) -> bool:
        """
        Check if source accuracy is degrading.
        
        Compares recent accuracy to historical average.
        """
        history = self.outcomes.get(source, [])
        if len(history) < 50:
            return False
        
        # Recent accuracy (last 20)
        recent_acc = sum(history[-20:]) / 20
        
        # Historical accuracy (before recent)
        historical = history[:-20]
        historical_acc = sum(historical) / len(historical)
        
        # Degrading if recent is significantly worse
        return recent_acc < historical_acc - self.degradation_threshold
    
    def get_trend(self, source: str) -> str:
        """
        Get accuracy trend for a source.
        
        Returns:
            'improving', 'stable', 'degrading'
        """
        snapshots = self.accuracy_history.get(source, [])
        if len(snapshots) < 5:
            return 'stable'
        
        recent = snapshots[-5:]
        
        # Calculate trend
        first_half = sum(s.accuracy for s in recent[:2]) / 2
        second_half = sum(s.accuracy for s in recent[-2:]) / 2
        
        diff = second_half - first_half
        
        if diff > 0.05:
            return 'improving'
        elif diff < -0.05:
            return 'degrading'
        else:
            return 'stable'
    
    def get_all_accuracies(self) -> Dict[str, Dict]:
        """Get accuracy summary for all sources."""
        result = {}
        
        for source in self.outcomes:
            result[source] = {
                'accuracy': self.get_accuracy(source),
                'predictions': len(self.outcomes[source]),
                'trend': self.get_trend(source),
                'is_degrading': self.is_degrading(source)
            }
        
        return result
    
    def get_best_source_for_regime(self, regime: str) -> Optional[str]:
        """
        Get the best performing source for a regime.
        
        Returns:
            Source name with highest accuracy for the regime
        """
        best_source = None
        best_accuracy = 0
        
        for source in self.outcomes_by_regime:
            if regime in self.outcomes_by_regime[source]:
                acc = self.get_accuracy_by_regime(source, regime)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_source = source
        
        return best_source
    
    def get_insights(self) -> List[str]:
        """
        Generate insights from accuracy data.
        
        Returns:
            List of insight strings
        """
        insights = []
        
        all_acc = self.get_all_accuracies()
        
        # Best and worst sources
        if all_acc:
            sorted_sources = sorted(
                all_acc.items(),
                key=lambda x: x[1]['accuracy'],
                reverse=True
            )
            
            best = sorted_sources[0]
            worst = sorted_sources[-1]
            
            if best[1]['accuracy'] > 0.6:
                insights.append(
                    f"📈 {best[0]} is the most accurate source "
                    f"({best[1]['accuracy']:.0%} accuracy)"
                )
            
            if worst[1]['accuracy'] < 0.4:
                insights.append(
                    f"⚠️ {worst[0]} has low accuracy "
                    f"({worst[1]['accuracy']:.0%}), consider reducing weight"
                )
        
        # Degrading sources
        for source, data in all_acc.items():
            if data['is_degrading']:
                insights.append(
                    f"📉 {source} accuracy is declining - investigate"
                )
        
        # Regime-specific insights
        for regime in ['BULL', 'BEAR', 'SIDEWAYS', 'VOLATILE']:
            best = self.get_best_source_for_regime(regime)
            if best:
                acc = self.get_accuracy_by_regime(best, regime)
                if acc > 0.6:
                    insights.append(
                        f"💡 {best} performs best in {regime} markets ({acc:.0%})"
                    )
        
        return insights
