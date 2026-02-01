#!/usr/bin/env python3
"""
KYPERIAN ELITE: Multi-Timeframe System Tests

Tests the complete institutional multi-timeframe signal system:
1. TimeframeManager - Signal storage and freshness
2. VetoEngine - Institutional veto rules
3. PositionCalculator - Kelly-based sizing
4. MTFFusionEngine - Complete decision generation
"""

import sys
import os
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kyperian.signals import (
    TimeframeManager,
    TimeframeSignal,
    Timeframe,
    VetoEngine,
    VetoResult,
    VetoDecision,
    PositionCalculator,
    PositionSize,
    MTFFusionEngine,
    TradingDecision,
    SignalStrength,
)


def test_timeframe_enum():
    """Test Timeframe enum properties."""
    print("\n" + "="*60)
    print("TEST: Timeframe Enum")
    print("="*60)
    
    # Test weights
    assert Timeframe.WEEKLY.weight == 0.40, "Weekly should be 40%"
    assert Timeframe.DAILY.weight == 0.35, "Daily should be 35%"
    assert Timeframe.FOUR_HOUR.weight == 0.25, "4H should be 25%"
    assert Timeframe.ONE_HOUR.weight == 0.00, "1H should be 0% (fine-tuning only)"
    
    # Test max age
    assert Timeframe.WEEKLY.max_age_hours == 168, "Weekly valid for 7 days"
    assert Timeframe.DAILY.max_age_hours == 24, "Daily valid for 24 hours"
    assert Timeframe.FOUR_HOUR.max_age_hours == 8, "4H valid for 8 hours"
    assert Timeframe.ONE_HOUR.max_age_hours == 2, "1H valid for 2 hours"
    
    # Test parsing
    assert Timeframe.from_string("1W") == Timeframe.WEEKLY
    assert Timeframe.from_string("1D") == Timeframe.DAILY
    assert Timeframe.from_string("4H") == Timeframe.FOUR_HOUR
    assert Timeframe.from_string("240") == Timeframe.FOUR_HOUR  # TradingView format
    
    print("✅ Timeframe enum properties PASSED")
    
    # Print summary
    print("\n   Timeframe Weights:")
    for tf in Timeframe:
        print(f"   {tf.value}: {tf.weight:.0%} weight, valid {tf.max_age_hours}h")
    
    return True


def test_timeframe_signal():
    """Test TimeframeSignal creation and properties."""
    print("\n" + "="*60)
    print("TEST: TimeframeSignal")
    print("="*60)
    
    # Create a fresh signal
    signal = TimeframeSignal(
        symbol="ETHUSD",
        timeframe=Timeframe.FOUR_HOUR,
        timestamp=datetime.now(),
        direction=1,
        action="BUY",
        strength="strong",
        confirmations=10,
        trend_strength=72.5,
        smart_trail_sentiment="bullish",
        neo_cloud_sentiment="bullish",
        price=2340.61,
    )
    
    # Test freshness
    assert signal.freshness > 0.9, "Fresh signal should have high freshness"
    assert signal.is_fresh, "Signal should be fresh"
    assert not signal.is_expired, "Signal should not be expired"
    
    # Test strength
    assert signal.is_strong, "10 confirmations should be strong"
    assert signal.is_bullish, "Direction 1 should be bullish"
    
    # Test weighted direction
    weighted = signal.weighted_direction
    assert weighted > 0, "Bullish signal should have positive weighted direction"
    assert weighted <= 1.0, "Weighted direction should be capped at 1.0"
    
    print(f"✅ Created signal: {signal}")
    print(f"   Freshness: {signal.freshness:.0%}")
    print(f"   Weighted direction: {weighted:.3f}")
    print(f"   Confidence: {signal.confidence:.0%}")
    
    # Test expired signal
    old_signal = TimeframeSignal(
        symbol="ETHUSD",
        timeframe=Timeframe.FOUR_HOUR,
        timestamp=datetime.now() - timedelta(hours=10),
        direction=1,
        action="BUY",
    )
    
    assert old_signal.is_expired, "10h old 4H signal should be expired"
    assert old_signal.freshness == 0, "Expired signal should have 0 freshness"
    
    print(f"✅ Old signal correctly identified as expired")
    
    # Test parsing from webhook
    webhook_payload = {
        "action": "SELL",
        "symbol": "BTCUSD",
        "timeframe": "1D",
        "price": 42500.0,
        "confirmations": 8,
        "strength": "strong",
        "trend_strength": 65,
        "smart_trail": "bearish",
    }
    
    parsed = TimeframeSignal.from_webhook(webhook_payload)
    assert parsed.symbol == "BTCUSD"
    assert parsed.timeframe == Timeframe.DAILY
    assert parsed.direction == -1
    assert parsed.is_bearish
    
    print(f"✅ Parsed webhook signal: {parsed}")
    
    return True


def test_timeframe_manager():
    """Test TimeframeManager signal storage and retrieval."""
    print("\n" + "="*60)
    print("TEST: TimeframeManager")
    print("="*60)
    
    manager = TimeframeManager()
    
    # Add signals for ETHUSD
    weekly_signal = TimeframeSignal(
        symbol="ETHUSD",
        timeframe=Timeframe.WEEKLY,
        timestamp=datetime.now(),
        direction=1,
        action="BUY",
        strength="strong",
        confirmations=12,
    )
    
    daily_signal = TimeframeSignal(
        symbol="ETHUSD",
        timeframe=Timeframe.DAILY,
        timestamp=datetime.now(),
        direction=1,
        action="BUY",
        confirmations=10,
    )
    
    four_hour_signal = TimeframeSignal(
        symbol="ETHUSD",
        timeframe=Timeframe.FOUR_HOUR,
        timestamp=datetime.now(),
        direction=1,
        action="BUY",
        confirmations=8,
    )
    
    manager.add_signal(weekly_signal)
    manager.add_signal(daily_signal)
    manager.add_signal(four_hour_signal)
    
    print(f"✅ Added 3 signals for ETHUSD")
    
    # Test retrieval
    weekly, daily, four_hour, hourly = manager.get_cascade("ETHUSD")
    
    assert weekly is not None, "Weekly should be retrieved"
    assert daily is not None, "Daily should be retrieved"
    assert four_hour is not None, "4H should be retrieved"
    assert hourly is None, "1H should be None (not added)"
    
    print(f"   Weekly: {weekly.action}")
    print(f"   Daily: {daily.action}")
    print(f"   4H: {four_hour.action}")
    
    # Test alignment
    alignment = manager.get_alignment("ETHUSD")
    assert alignment["aligned"], "All bullish signals should be aligned"
    assert alignment["alignment_score"] == 1.0, "Perfect alignment should be 1.0"
    
    print(f"✅ Alignment: {alignment['alignment_score']:.0%} - {alignment['reason']}")
    
    # Test with conflicting signal
    conflicting_4h = TimeframeSignal(
        symbol="ETHUSD",
        timeframe=Timeframe.FOUR_HOUR,
        timestamp=datetime.now(),
        direction=-1,
        action="SELL",
        confirmations=6,
    )
    
    manager.add_signal(conflicting_4h)
    
    alignment2 = manager.get_alignment("ETHUSD")
    assert not alignment2["aligned"], "Conflicting signal should break alignment"
    
    print(f"✅ Conflicting signal detected: {alignment2['reason']}")
    
    # Test status
    status = manager.get_status()
    assert "ETHUSD" in status, "ETHUSD should be in status"
    
    print(f"✅ TimeframeManager tests PASSED")
    
    return True


def test_veto_engine():
    """Test VetoEngine institutional rules."""
    print("\n" + "="*60)
    print("TEST: VetoEngine")
    print("="*60)
    
    manager = TimeframeManager()
    veto_engine = VetoEngine(manager)
    
    # Scenario 1: Perfect alignment - should approve
    weekly = TimeframeSignal(
        symbol="AAPL",
        timeframe=Timeframe.WEEKLY,
        timestamp=datetime.now(),
        direction=1,
        action="BUY",
        strength="strong",
    )
    daily = TimeframeSignal(
        symbol="AAPL",
        timeframe=Timeframe.DAILY,
        timestamp=datetime.now(),
        direction=1,
        action="BUY",
    )
    four_hour = TimeframeSignal(
        symbol="AAPL",
        timeframe=Timeframe.FOUR_HOUR,
        timestamp=datetime.now(),
        direction=1,
        action="BUY",
        confirmations=10,
    )
    
    manager.add_signal(weekly)
    manager.add_signal(daily)
    manager.add_signal(four_hour)
    
    result = veto_engine.check_veto("AAPL")
    
    assert result.can_trade, "Perfect alignment should allow trade"
    assert result.decision in [VetoDecision.APPROVED, VetoDecision.APPROVED_REDUCED]
    assert result.position_multiplier > 0.8, "Should have high position multiplier"
    
    print(f"✅ Scenario 1 (Perfect alignment): {result.decision.value}")
    print(f"   Position multiplier: {result.position_multiplier:.0%}")
    print(f"   Reason: {result.reason}")
    
    # Scenario 2: Against weekly trend - should VETO
    manager2 = TimeframeManager()
    veto2 = VetoEngine(manager2)
    
    weekly_bull = TimeframeSignal(
        symbol="TSLA",
        timeframe=Timeframe.WEEKLY,
        timestamp=datetime.now(),
        direction=1,
        action="BUY",
    )
    daily_bear = TimeframeSignal(
        symbol="TSLA",
        timeframe=Timeframe.DAILY,
        timestamp=datetime.now(),
        direction=-1,
        action="SELL",
    )
    four_hour_bear = TimeframeSignal(
        symbol="TSLA",
        timeframe=Timeframe.FOUR_HOUR,
        timestamp=datetime.now(),
        direction=-1,
        action="SELL",
    )
    
    manager2.add_signal(weekly_bull)
    manager2.add_signal(daily_bear)
    manager2.add_signal(four_hour_bear)
    
    result2 = veto2.check_veto("TSLA")
    
    assert not result2.can_trade, "Against weekly should be vetoed"
    assert result2.decision == VetoDecision.VETOED
    
    print(f"\n✅ Scenario 2 (Against weekly): {result2.decision.value}")
    print(f"   Reason: {result2.reason}")
    
    # Scenario 3: Weekly neutral - should allow reduced position
    manager3 = TimeframeManager()
    veto3 = VetoEngine(manager3)
    
    weekly_neutral = TimeframeSignal(
        symbol="SPY",
        timeframe=Timeframe.WEEKLY,
        timestamp=datetime.now(),
        direction=0,
        action="NEUTRAL",
    )
    daily_bull = TimeframeSignal(
        symbol="SPY",
        timeframe=Timeframe.DAILY,
        timestamp=datetime.now(),
        direction=1,
        action="BUY",
    )
    four_hour_bull = TimeframeSignal(
        symbol="SPY",
        timeframe=Timeframe.FOUR_HOUR,
        timestamp=datetime.now(),
        direction=1,
        action="BUY",
    )
    
    manager3.add_signal(weekly_neutral)
    manager3.add_signal(daily_bull)
    manager3.add_signal(four_hour_bull)
    
    result3 = veto3.check_veto("SPY")
    
    assert result3.can_trade, "Weekly neutral should allow trading"
    assert result3.position_multiplier <= 0.5, "Weekly neutral should reduce size"
    
    print(f"\n✅ Scenario 3 (Weekly neutral): {result3.decision.value}")
    print(f"   Position multiplier: {result3.position_multiplier:.0%}")
    
    print(f"\n✅ VetoEngine tests PASSED")
    
    return True


def test_position_calculator():
    """Test PositionCalculator Kelly-based sizing."""
    print("\n" + "="*60)
    print("TEST: PositionCalculator")
    print("="*60)
    
    calc = PositionCalculator(
        max_risk=0.02,      # 2% max risk
        max_position=0.10,  # 10% max position
        kelly_fraction=0.5  # Half-Kelly
    )
    
    # Test Kelly calculation
    kelly = calc.calculate_kelly(win_rate=0.55, win_loss_ratio=2.0)
    print(f"   Kelly (55% win, 2:1 R/R): {kelly:.2%}")
    assert kelly > 0, "Positive edge should give positive Kelly"
    
    # Test with perfect veto result
    veto_result = VetoResult(
        decision=VetoDecision.APPROVED,
        can_trade=True,
        position_multiplier=1.0,
        direction=1,
        reason="Perfect alignment",
        details=[],
        weekly_signal=TimeframeSignal(
            symbol="ETHUSD",
            timeframe=Timeframe.WEEKLY,
            timestamp=datetime.now(),
            direction=1,
            action="BUY",
            strength="strong",
        ),
        daily_signal=TimeframeSignal(
            symbol="ETHUSD",
            timeframe=Timeframe.DAILY,
            timestamp=datetime.now(),
            direction=1,
            action="BUY",
        ),
        four_hour_signal=TimeframeSignal(
            symbol="ETHUSD",
            timeframe=Timeframe.FOUR_HOUR,
            timestamp=datetime.now(),
            direction=1,
            action="BUY",
            confirmations=10,
        ),
    )
    
    position = calc.calculate_position(
        veto_result=veto_result,
        current_price=2340.0,
        portfolio_value=100000,
        regime="BULL"
    )
    
    print(f"\n✅ Position calculated:")
    print(f"   Size: {position.recommended_size:.1%} (${position.dollar_amount:,.0f})")
    print(f"   Shares: {position.shares}")
    print(f"   Stop Loss: ${position.stop_loss_price:,.2f} ({position.stop_loss_pct:.1%})")
    print(f"   TP1: ${position.take_profit_prices[0]:,.2f} ({position.take_profit_pcts[0]:.1%})")
    print(f"   TP2: ${position.take_profit_prices[1]:,.2f} ({position.take_profit_pcts[1]:.1%})")
    print(f"   R/R: {position.risk_reward_ratio:.1f}")
    print(f"   Confidence: {position.confidence:.0%}")
    print(f"   Kelly: {position.kelly_fraction:.2%}")
    
    assert position.recommended_size > 0, "Should have positive position"
    assert position.recommended_size <= 0.10, "Should not exceed max position"
    assert position.stop_loss_price < 2340.0, "Long stop should be below price"
    assert position.take_profit_prices[0] > 2340.0, "Long TP should be above price"
    
    # Test with vetoed result
    vetoed_result = VetoResult(
        decision=VetoDecision.VETOED,
        can_trade=False,
        position_multiplier=0.0,
        direction=0,
        reason="Against weekly",
        details=[],
    )
    
    no_position = calc.calculate_position(
        veto_result=vetoed_result,
        current_price=2340.0,
        portfolio_value=100000,
    )
    
    assert no_position.recommended_size == 0, "Vetoed should have zero position"
    assert no_position.dollar_amount == 0, "Vetoed should have zero dollars"
    
    print(f"\n✅ Vetoed position: {no_position.recommended_size:.0%}")
    
    # Test scaling plan
    scale_plan = calc.calculate_scaling_plan(position, max_adds=2)
    
    print(f"\n✅ Scaling plan:")
    for level in scale_plan:
        print(f"   Level {level['level']}: {level['action']} - {level['size_pct']:.0%}")
    
    print(f"\n✅ PositionCalculator tests PASSED")
    
    return True


def test_mtf_fusion_engine():
    """Test the complete MTFFusionEngine."""
    print("\n" + "="*60)
    print("TEST: MTFFusionEngine")
    print("="*60)
    
    engine = MTFFusionEngine(portfolio_value=100000)
    
    # Add signals via webhook payloads
    weekly_payload = {
        "action": "BUY",
        "symbol": "ETHUSD",
        "timeframe": "1W",
        "price": 2300.0,
        "confirmations": 12,
        "strength": "strong",
        "trend_strength": 75,
    }
    
    daily_payload = {
        "action": "BUY",
        "symbol": "ETHUSD",
        "timeframe": "1D",
        "price": 2320.0,
        "confirmations": 10,
    }
    
    four_hour_payload = {
        "action": "BUY",
        "symbol": "ETHUSD",
        "timeframe": "4H",
        "price": 2340.0,
        "confirmations": 9,
        "smart_trail": "bullish",
        "neo_cloud": "bullish",
    }
    
    engine.add_from_webhook(weekly_payload)
    engine.add_from_webhook(daily_payload)
    engine.add_from_webhook(four_hour_payload)
    
    print(f"✅ Added 3 signals from webhooks")
    
    # Get signals
    signals = engine.get_signals("ETHUSD")
    print(f"   Signals: {list(signals.keys())}")
    
    # Check alignment
    alignment = engine.get_alignment("ETHUSD")
    print(f"   Alignment: {alignment['alignment_score']:.0%} - {alignment['reason']}")
    
    # Generate decision
    decision = engine.generate_decision(
        symbol="ETHUSD",
        current_price=2340.0,
        regime="BULL"
    )
    
    print(f"\n✅ Trading Decision: {decision}")
    print(f"   Action: {decision.action_label}")
    print(f"   Can Trade: {decision.can_trade}")
    print(f"   Strength: {decision.strength.name}")
    print(f"   Confidence: {decision.confidence:.0%}")
    
    if decision.position:
        print(f"   Position: ${decision.position.dollar_amount:,.0f}")
        print(f"   Stop: ${decision.position.stop_loss_price:,.2f}")
        print(f"   TP1: ${decision.position.take_profit_prices[0]:,.2f}")
    
    print(f"\n   Reasoning:")
    for reason in decision.reasoning[:5]:
        print(f"     - {reason}")
    
    assert decision.can_trade, "Aligned signals should allow trade"
    assert decision.action == "BUY", "Should be BUY"
    assert decision.position.recommended_size > 0, "Should have position"
    
    # Test status
    status = engine.get_status()
    print(f"\n   Status: {len(status['symbols'])} symbols tracked")
    
    # Convert to dict
    decision_dict = decision.to_dict()
    assert "symbol" in decision_dict
    assert "can_trade" in decision_dict
    assert "position" in decision_dict
    
    print(f"\n✅ MTFFusionEngine tests PASSED")
    
    return True


def test_full_integration():
    """Test full integration scenario."""
    print("\n" + "="*60)
    print("TEST: Full Integration")
    print("="*60)
    
    engine = MTFFusionEngine(portfolio_value=250000)
    
    # Simulate receiving webhooks over time
    print("   Simulating webhook stream...")
    
    # Week 1: Weekly bullish signal
    engine.add_from_webhook({
        "action": "BUY",
        "symbol": "BTCUSD",
        "timeframe": "1W",
        "price": 42000,
        "confirmations": 12,
        "strength": "strong",
        "trend_strength": 80,
    })
    
    decision1 = engine.generate_decision("BTCUSD", current_price=42000)
    print(f"   After Weekly only: {decision1.action} (can_trade={decision1.can_trade})")
    
    # Daily confirms
    engine.add_from_webhook({
        "action": "BUY",
        "symbol": "BTCUSD",
        "timeframe": "1D",
        "price": 42500,
        "confirmations": 10,
    })
    
    decision2 = engine.generate_decision("BTCUSD", current_price=42500)
    print(f"   After Daily: {decision2.action} (can_trade={decision2.can_trade})")
    
    # 4H triggers
    engine.add_from_webhook({
        "action": "BUY",
        "symbol": "BTCUSD",
        "timeframe": "4h",
        "price": 42800,
        "confirmations": 9,
    })
    
    decision3 = engine.generate_decision("BTCUSD", current_price=42800)
    print(f"   After 4H: {decision3.action} (can_trade={decision3.can_trade})")
    
    if decision3.can_trade:
        print(f"\n   ✅ TRADE TRIGGERED!")
        print(f"   Entry: ${decision3.entry_price:,.2f}")
        print(f"   Size: ${decision3.position.dollar_amount:,.2f}")
        print(f"   Stop: ${decision3.position.stop_loss_price:,.2f}")
        print(f"   TP1: ${decision3.position.take_profit_prices[0]:,.2f}")
    
    # Now simulate 4H flip to sell (against weekly)
    print("\n   Simulating 4H flip to SELL...")
    
    engine.add_from_webhook({
        "action": "SELL",
        "symbol": "BTCUSD",
        "timeframe": "4h",
        "price": 42200,
        "confirmations": 6,
    })
    
    decision4 = engine.generate_decision("BTCUSD", current_price=42200)
    print(f"   After 4H SELL: {decision4.action} (can_trade={decision4.can_trade})")
    
    if not decision4.can_trade:
        print(f"   ✅ Correctly waiting for alignment")
        print(f"   Reason: {decision4.veto_result.reason}")
    
    print(f"\n✅ Full Integration tests PASSED")
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("KYPERIAN ELITE: MULTI-TIMEFRAME SYSTEM TESTS")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Timeframe Enum", test_timeframe_enum),
        ("TimeframeSignal", test_timeframe_signal),
        ("TimeframeManager", test_timeframe_manager),
        ("VetoEngine", test_veto_engine),
        ("PositionCalculator", test_position_calculator),
        ("MTFFusionEngine", test_mtf_fusion_engine),
        ("Full Integration", test_full_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    
    for name, p, err in results:
        status = "✅ PASSED" if p else f"❌ FAILED: {err}"
        print(f"   {name}: {status}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
