"""
Weight Adjuster

Dynamically adjusts signal source weights based on performance.
Implements continuous learning for the fusion engine.
"""

from typing import Dict, List, Optional
import logging
from datetime import datetime

from .accuracy_monitor import AccuracyMonitor

logger = logging.getLogger(__name__)


class WeightAdjuster:
    """
    Dynamically adjusts signal source weights.
    
    Based on:
    - Recent accuracy
    - Performance by regime
    - Trend (improving/degrading)
    
    Constraints:
    - Weights always sum to 1
    - Minimum weight per source (don't disable completely)
    - Maximum weight per source (don't over-rely)
    - Smooth adjustments (no sudden changes)
    
    Example:
        adjuster = WeightAdjuster(base_weights={
            'luxalgo': 0.50,
            'ml': 0.25,
            'sentiment': 0.10,
            'regime': 0.10,
            'fundamental': 0.05
        })
        
        # Record outcomes
        adjuster.record_outcome('luxalgo', True)
        
        # Get adjusted weights
        weights = adjuster.get_weights()
        
        # Get regime-specific weights
        regime_weights = adjuster.get_weights_for_regime('BULL')
    """
    
    def __init__(
        self,
        base_weights: Dict[str, float],
        min_weight: float = 0.05,
        max_weight: float = 0.60,
        adjustment_rate: float = 0.1,
        rolling_window: int = 100
    ):
        """
        Initialize weight adjuster.
        
        Args:
            base_weights: Base weights for each source
            min_weight: Minimum weight per source
            max_weight: Maximum weight per source
            adjustment_rate: How fast to adjust weights (0-1)
            rolling_window: Window for accuracy calculation
        """
        self.base_weights = base_weights.copy()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.adjustment_rate = adjustment_rate
        
        # Normalize base weights
        total = sum(self.base_weights.values())
        if total > 0:
            self.base_weights = {
                k: v / total for k, v in self.base_weights.items()
            }
        
        # Current weights
        self.current_weights = self.base_weights.copy()
        
        # Regime-specific weight adjustments
        self.regime_adjustments: Dict[str, Dict[str, float]] = {}
        
        # Accuracy monitor
        self.monitor = AccuracyMonitor(rolling_window=rolling_window)
        
        # Adjustment history
        self.adjustment_history: List[Dict] = []
    
    def record_outcome(
        self,
        source: str,
        was_correct: bool,
        symbol: str = None,
        regime: str = None
    ):
        """
        Record a prediction outcome.
        
        This feeds into the accuracy monitor and triggers
        weight adjustments.
        """
        self.monitor.record_outcome(source, was_correct, symbol, regime)
        
        # Trigger weight update periodically
        outcomes = self.monitor.outcomes.get(source, [])
        if len(outcomes) % 10 == 0 and len(outcomes) >= 20:
            self._update_weights()
    
    def _update_weights(self):
        """Update weights based on recent accuracy."""
        all_accuracies = self.monitor.get_all_accuracies()
        
        if not all_accuracies:
            return
        
        # Calculate adjustment for each source
        adjustments = {}
        
        for source, data in all_accuracies.items():
            accuracy = data['accuracy']
            trend = data['trend']
            
            # Target: sources with >50% accuracy get weight boost
            # Sources with <50% accuracy get weight reduction
            
            # Base adjustment from accuracy deviation
            # Accuracy of 0.5 = no change, 0.7 = +20%, 0.3 = -20%
            accuracy_adj = (accuracy - 0.5) * 2 * self.adjustment_rate
            
            # Trend adjustment
            if trend == 'improving':
                trend_adj = 0.02
            elif trend == 'degrading':
                trend_adj = -0.03
            else:
                trend_adj = 0
            
            adjustments[source] = accuracy_adj + trend_adj
        
        # Apply adjustments
        new_weights = {}
        for source, base in self.base_weights.items():
            adj = adjustments.get(source, 0)
            new_weight = base * (1 + adj)
            
            # Apply constraints
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            new_weights[source] = new_weight
        
        # Normalize to sum to 1
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}
        
        # Smooth transition (don't change too fast)
        for source in new_weights:
            old = self.current_weights.get(source, new_weights[source])
            new_weights[source] = old * 0.7 + new_weights[source] * 0.3
        
        # Log adjustment
        self.adjustment_history.append({
            'timestamp': datetime.now().isoformat(),
            'old_weights': self.current_weights.copy(),
            'new_weights': new_weights.copy(),
            'accuracies': all_accuracies
        })
        
        # Keep only last 50 adjustments
        if len(self.adjustment_history) > 50:
            self.adjustment_history = self.adjustment_history[-50:]
        
        self.current_weights = new_weights
        
        logger.info(f"Updated weights: {self._format_weights(new_weights)}")
    
    def _format_weights(self, weights: Dict[str, float]) -> str:
        """Format weights for logging."""
        parts = [f"{k}={v:.0%}" for k, v in weights.items()]
        return ", ".join(parts)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current adjusted weights."""
        return self.current_weights.copy()
    
    def get_weights_for_regime(self, regime: str) -> Dict[str, float]:
        """
        Get regime-specific weights.
        
        Adjusts weights based on source performance in this regime.
        """
        weights = self.current_weights.copy()
        
        # Check if we have regime-specific performance data
        regime_adjustments = {}
        
        for source in weights:
            accuracy = self.monitor.get_accuracy_by_regime(source, regime)
            
            # Only adjust if we have enough data
            history = self.monitor.outcomes_by_regime.get(source, {}).get(regime, [])
            if len(history) < 10:
                continue
            
            # Adjust based on regime performance
            if accuracy > 0.6:
                regime_adjustments[source] = 0.15  # Boost good performers
            elif accuracy < 0.4:
                regime_adjustments[source] = -0.15  # Reduce poor performers
        
        # Apply regime adjustments
        for source, adj in regime_adjustments.items():
            weights[source] = weights.get(source, 0) * (1 + adj)
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def reset_to_base(self):
        """Reset weights to base values."""
        self.current_weights = self.base_weights.copy()
        logger.info("Reset weights to base values")
    
    def get_weight_history(self) -> List[Dict]:
        """Get history of weight adjustments."""
        return self.adjustment_history
    
    def get_status(self) -> Dict:
        """Get current status of weight adjuster."""
        return {
            'base_weights': self.base_weights,
            'current_weights': self.current_weights,
            'adjustments_made': len(self.adjustment_history),
            'source_accuracies': self.monitor.get_all_accuracies(),
            'insights': self.monitor.get_insights()
        }
    
    def suggest_weights(self) -> Dict[str, float]:
        """
        Suggest optimal weights based on all data.
        
        More aggressive than current weights - for manual review.
        """
        all_accuracies = self.monitor.get_all_accuracies()
        
        if not all_accuracies:
            return self.base_weights.copy()
        
        # Weight sources proportionally to accuracy
        suggested = {}
        
        for source, data in all_accuracies.items():
            accuracy = data['accuracy']
            
            # Square the accuracy to amplify differences
            weight = accuracy ** 2
            suggested[source] = weight
        
        # Add sources without data at base weight
        for source, base in self.base_weights.items():
            if source not in suggested:
                suggested[source] = base
        
        # Normalize
        total = sum(suggested.values())
        if total > 0:
            suggested = {k: v / total for k, v in suggested.items()}
        
        # Apply constraints
        for source in suggested:
            suggested[source] = max(self.min_weight, 
                                   min(self.max_weight, suggested[source]))
        
        # Re-normalize
        total = sum(suggested.values())
        if total > 0:
            suggested = {k: v / total for k, v in suggested.items()}
        
        return suggested
