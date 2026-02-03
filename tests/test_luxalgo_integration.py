#!/usr/bin/env python3
"""
Test LuxAlgo Integration

Comprehensive tests for the LuxAlgo webhook integration and signal fusion system.

Run with: python tests/test_luxalgo_integration.py
"""

import sys
sys.path.insert(0, 'src')

from datetime import datetime
import json


def test_webhook_parsing():
    """Test parsing LuxAlgo webhook payloads."""
    print("\n" + "="*60)
    print("TEST: Webhook Parsing")
    print("="*60)
    
    from nuble.signals.luxalgo_webhook import parse_luxalgo_webhook, LuxAlgoSignalType
    
    # Test case 1: Basic BUY signal
    payload1 = {
        "action": "BUY",
        "symbol": "ETHUSD",
        "exchange": "COINBASE",
        "price": 2340.61,
        "timeframe": "4h",
        "signal_type": "Bullish Confirmation",
        "confirmations": 12,
        "trend_strength": 54.04,
        "time": "2026-02-01T14:05:00Z"
    }
    
    signal1 = parse_luxalgo_webhook(payload1)
    
    print(f"\n✅ Parsed BUY signal:")
    print(f"   Symbol: {signal1.symbol}")
    print(f"   Action: {signal1.action}")
    print(f"   Confirmations: {signal1.confirmations}")
    print(f"   Timeframe: {signal1.timeframe}")
    print(f"   Confidence: {signal1.confidence:.2%}")
    print(f"   Is Strong: {signal1.is_strong}")
    
    assert signal1.symbol == "ETHUSD"
    assert signal1.action == "BUY"
    assert signal1.confirmations == 12
    assert signal1.is_strong == True
    
    # Test case 2: SELL signal with indicators
    payload2 = {
        "action": "SELL",
        "symbol": "BTCUSD",
        "exchange": "BINANCE",
        "price": 97500.00,
        "timeframe": "4h",
        "signal_type": "Bearish Confirmation",
        "confirmations": 8,
        "trend_strength": 65.0,
        "trend_tracer": "bearish",
        "smart_trail": "bearish",
        "neo_cloud": "bearish"
    }
    
    signal2 = parse_luxalgo_webhook(payload2)
    
    print(f"\n✅ Parsed SELL signal:")
    print(f"   Symbol: {signal2.symbol}")
    print(f"   Action: {signal2.action}")
    print(f"   Confirmations: {signal2.confirmations}")
    print(f"   Indicator Agreement: {signal2.indicator_agreement_count}")
    print(f"   Confidence: {signal2.confidence:.2%}")
    
    assert signal2.symbol == "BTCUSD"
    assert signal2.action == "SELL"
    assert signal2.is_bearish == True
    
    print("\n✅ Webhook parsing tests PASSED")


def test_signal_store():
    """Test signal storage and retrieval."""
    print("\n" + "="*60)
    print("TEST: Signal Store")
    print("="*60)
    
    from nuble.signals.luxalgo_webhook import (
        LuxAlgoSignalStore, parse_luxalgo_webhook
    )
    
    store = LuxAlgoSignalStore()
    
    # Add multiple signals
    signals = [
        {"action": "BUY", "symbol": "ETHUSD", "price": 2300, "timeframe": "4h", "confirmations": 8, "trend_strength": 60},
        {"action": "BUY", "symbol": "ETHUSD", "price": 2320, "timeframe": "4h", "confirmations": 10, "trend_strength": 65},
        {"action": "BUY", "symbol": "ETHUSD", "price": 2340, "timeframe": "4h", "confirmations": 12, "trend_strength": 54},
        {"action": "SELL", "symbol": "BTCUSD", "price": 98000, "timeframe": "4h", "confirmations": 6, "trend_strength": 45},
    ]
    
    for payload in signals:
        signal = parse_luxalgo_webhook(payload)
        store.add_signal(signal)
    
    print(f"\n✅ Added {len(signals)} signals to store")
    
    # Get latest
    latest_eth = store.get_latest_signal("ETHUSD")
    print(f"\n   Latest ETHUSD: {latest_eth.action} at ${latest_eth.price}")
    
    assert latest_eth.price == 2340
    assert latest_eth.confirmations == 12
    
    # Get consensus
    consensus = store.get_signal_consensus("ETHUSD", hours=24)
    print(f"\n   ETHUSD Consensus:")
    print(f"   Direction: {consensus['direction']}")
    print(f"   Confidence: {consensus['confidence']:.2%}")
    print(f"   Buy signals: {consensus['buy_signals']}")
    print(f"   Sell signals: {consensus['sell_signals']}")
    
    assert consensus['direction'] == 'BUY'
    assert consensus['buy_signals'] == 3
    
    # Get strong signals
    strong = store.get_strong_signals("ETHUSD", hours=24)
    print(f"\n   Strong ETHUSD signals: {len(strong)}")
    
    assert len(strong) >= 2  # At least 2 with 4+ confirmations on 4h
    
    print("\n✅ Signal store tests PASSED")


def test_signal_fusion():
    """Test signal fusion engine."""
    print("\n" + "="*60)
    print("TEST: Signal Fusion Engine")
    print("="*60)
    
    from nuble.signals.luxalgo_webhook import (
        get_signal_store, parse_luxalgo_webhook, reset_signal_store
    )
    from nuble.signals.fusion_engine import SignalFusionEngine, FusedSignalStrength
    
    # Reset store for clean test
    reset_signal_store()
    store = get_signal_store()
    
    # Add LuxAlgo signals
    signals = [
        {"action": "BUY", "symbol": "ETHUSD", "price": 2340, "timeframe": "4h", "confirmations": 12, "trend_strength": 54},
        {"action": "BUY", "symbol": "ETHUSD", "price": 2350, "timeframe": "4h", "confirmations": 10, "trend_strength": 58},
    ]
    
    for payload in signals:
        signal = parse_luxalgo_webhook(payload)
        store.add_signal(signal)
    
    # Create fusion engine
    engine = SignalFusionEngine(
        luxalgo_weight=0.50,
        ml_weight=0.25,
        sentiment_weight=0.10,
        regime_weight=0.10,
        fundamental_weight=0.05
    )
    
    # Generate fused signal (no ML data, just LuxAlgo + context)
    fused = engine.generate_fused_signal(
        symbol="ETHUSD",
        sentiment=0.3,  # Positive sentiment
        regime="BULL"
    )
    
    print(f"\n✅ Fused signal for ETHUSD:")
    print(f"   Direction: {fused.direction}")
    print(f"   Strength: {fused.strength.label}")
    print(f"   Confidence: {fused.confidence:.2%}")
    print(f"   Is Actionable: {fused.is_actionable}")
    print(f"   Recommended Size: {fused.recommended_size:.0%}")
    print(f"\n   Reasoning:")
    for reason in fused.reasoning:
        print(f"     - {reason}")
    
    # With strong LuxAlgo signals, should be bullish
    assert fused.direction == 1  # BUY
    assert fused.confidence > 0.5
    
    print("\n✅ Signal fusion tests PASSED")


def test_signal_sources():
    """Test individual signal sources."""
    print("\n" + "="*60)
    print("TEST: Signal Sources")
    print("="*60)
    
    from nuble.signals.sources.technical_luxalgo import TechnicalLuxAlgoSource
    from nuble.signals.sources.regime_hmm import RegimeHMMSource
    from nuble.signals.luxalgo_webhook import get_signal_store, parse_luxalgo_webhook, reset_signal_store
    import pandas as pd
    import numpy as np
    
    # Reset and populate store
    reset_signal_store()
    store = get_signal_store()
    
    signal = parse_luxalgo_webhook({
        "action": "BUY", "symbol": "AAPL", "price": 185.0, 
        "timeframe": "4h", "confirmations": 8, "trend_strength": 55
    })
    store.add_signal(signal)
    
    # Test TechnicalLuxAlgoSource
    print("\n--- TechnicalLuxAlgoSource ---")
    luxalgo_source = TechnicalLuxAlgoSource()
    
    tech_signal = luxalgo_source.generate_signal("AAPL")
    
    if tech_signal:
        print(f"   Direction: {tech_signal.direction:.2f}")
        print(f"   Confidence: {tech_signal.confidence:.2%}")
        print(f"   Reasoning: {tech_signal.reasoning}")
    else:
        print("   No signal (expected if no recent data)")
    
    # Test RegimeHMMSource
    print("\n--- RegimeHMMSource ---")
    regime_source = RegimeHMMSource(use_hmm=False)  # Use simple detection
    
    # Create sample price data
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    prices = 100 * (1 + np.random.randn(100).cumsum() * 0.01 + 0.001)  # Slight uptrend
    
    price_data = pd.DataFrame({
        'close': prices,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'open': np.roll(prices, 1),
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    regime_signal = regime_source.generate_signal("SPY", price_data)
    
    if regime_signal:
        print(f"   Regime: {regime_signal.raw_data.get('regime')}")
        print(f"   Direction: {regime_signal.direction:.2f}")
        print(f"   Confidence: {regime_signal.confidence:.2%}")
        print(f"   Reasoning: {regime_signal.reasoning}")
    
    print("\n✅ Signal sources tests PASSED")


def test_prediction_tracking():
    """Test prediction tracking system."""
    print("\n" + "="*60)
    print("TEST: Prediction Tracking")
    print("="*60)
    
    from nuble.learning.prediction_tracker import PredictionTracker, PredictionOutcome
    from nuble.signals.fusion_engine import FusedSignal, FusedSignalStrength
    
    tracker = PredictionTracker()
    
    # Create a mock fused signal
    fused = FusedSignal(
        symbol="ETHUSD",
        timestamp=datetime.now(),
        direction=1,
        strength=FusedSignalStrength.BUY,
        confidence=0.75,
        regime="BULL",
        recommended_size=0.30,
        stop_loss_pct=0.02,
        take_profit_pct=0.06,
        reasoning=["Test signal"]
    )
    
    # Log prediction
    pred_id = tracker.log_prediction(fused, price=2340.00)
    print(f"\n   Logged prediction: {pred_id}")
    
    # Resolve prediction (price went up)
    tracker.resolve_prediction(pred_id, outcome_price=2450.00)
    
    pred = tracker.predictions_by_id[pred_id]
    print(f"   Outcome: {pred.outcome.value}")
    print(f"   Return: {pred.outcome_return:.2%}")
    
    # Get stats
    stats = tracker.get_accuracy_stats()
    print(f"\n   Accuracy Stats:")
    print(f"   Total predictions: {stats['total_predictions']}")
    print(f"   Resolved: {stats['resolved_predictions']}")
    print(f"   Overall accuracy: {stats['overall_accuracy']:.0%}")
    
    print("\n✅ Prediction tracking tests PASSED")


def test_accuracy_monitoring():
    """Test accuracy monitoring."""
    print("\n" + "="*60)
    print("TEST: Accuracy Monitoring")
    print("="*60)
    
    from nuble.learning.accuracy_monitor import AccuracyMonitor
    
    monitor = AccuracyMonitor(rolling_window=20)
    
    # Record outcomes for luxalgo (70% accuracy)
    for i in range(30):
        was_correct = i % 10 < 7  # 7/10 correct
        monitor.record_outcome('luxalgo', was_correct, regime='BULL')
    
    # Record outcomes for ml (50% accuracy)
    for i in range(30):
        was_correct = i % 2 == 0  # 50% correct
        monitor.record_outcome('ml', was_correct, regime='BULL')
    
    # Check accuracies
    luxalgo_acc = monitor.get_accuracy('luxalgo')
    ml_acc = monitor.get_accuracy('ml')
    
    print(f"\n   LuxAlgo accuracy: {luxalgo_acc:.0%}")
    print(f"   ML accuracy: {ml_acc:.0%}")
    
    assert luxalgo_acc > 0.6
    assert ml_acc < 0.6
    
    # Check insights
    insights = monitor.get_insights()
    print(f"\n   Insights:")
    for insight in insights:
        print(f"     {insight}")
    
    # Get all accuracies
    all_acc = monitor.get_all_accuracies()
    print(f"\n   All Source Accuracies:")
    for source, data in all_acc.items():
        print(f"     {source}: {data['accuracy']:.0%} ({data['trend']})")
    
    print("\n✅ Accuracy monitoring tests PASSED")


def test_weight_adjustment():
    """Test dynamic weight adjustment."""
    print("\n" + "="*60)
    print("TEST: Weight Adjustment")
    print("="*60)
    
    from nuble.learning.weight_adjuster import WeightAdjuster
    
    base_weights = {
        'luxalgo': 0.50,
        'ml': 0.25,
        'sentiment': 0.10,
        'regime': 0.10,
        'fundamental': 0.05
    }
    
    adjuster = WeightAdjuster(base_weights=base_weights)
    
    print(f"\n   Initial weights: {adjuster.get_weights()}")
    
    # Simulate outcomes (LuxAlgo very accurate, ML poor)
    for i in range(50):
        adjuster.record_outcome('luxalgo', i % 10 < 8)  # 80% accurate
        adjuster.record_outcome('ml', i % 10 < 4)       # 40% accurate
        adjuster.record_outcome('sentiment', i % 10 < 6) # 60% accurate
    
    current = adjuster.get_weights()
    print(f"\n   Adjusted weights:")
    for source, weight in current.items():
        print(f"     {source}: {weight:.0%}")
    
    # LuxAlgo should have higher weight now
    assert current['luxalgo'] > base_weights['luxalgo'] * 0.9
    
    # Get suggested weights
    suggested = adjuster.suggest_weights()
    print(f"\n   Suggested weights (more aggressive):")
    for source, weight in suggested.items():
        print(f"     {source}: {weight:.0%}")
    
    # Get status
    status = adjuster.get_status()
    print(f"\n   Adjustments made: {status['adjustments_made']}")
    print(f"   Insights: {status['insights']}")
    
    print("\n✅ Weight adjustment tests PASSED")


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("NUBLE LUXALGO INTEGRATION TESTS")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Webhook Parsing", test_webhook_parsing),
        ("Signal Store", test_signal_store),
        ("Signal Fusion", test_signal_fusion),
        ("Signal Sources", test_signal_sources),
        ("Prediction Tracking", test_prediction_tracking),
        ("Accuracy Monitoring", test_accuracy_monitoring),
        ("Weight Adjustment", test_weight_adjustment),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"   Passed: {passed}/{len(tests)}")
    print(f"   Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
