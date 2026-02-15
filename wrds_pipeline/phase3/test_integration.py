"""
End-to-end integration test for the WRDS ML pipeline.

Verifies:
1. Step 6 outputs exist (model, predictions, metrics)
2. Step 7 outputs exist (backtest results)
3. WRDSAdvisor works with real data
4. WRDSPredictor integrates with APEX
5. Full pipeline: ticker → prediction → signal
"""

import os
import sys
import json

sys.path.insert(0, "/Users/humbertolobo/Desktop/NUBLE-CLI")
sys.path.insert(0, "/Users/humbertolobo/Desktop/NUBLE-CLI/wrds_pipeline/phase3")

DATA_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/data/wrds"
RESULTS_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/wrds_pipeline/phase3/results"


def test_step6_outputs():
    """Verify Step 6 produced all expected outputs."""
    print("\n📋 TEST 1: Step 6 Outputs")
    files = {
        "Model": os.path.join(DATA_DIR, "lgb_latest_model.txt"),
        "Predictions": os.path.join(DATA_DIR, "lgb_predictions.parquet"),
        "Metrics": os.path.join(RESULTS_DIR, "walk_forward_metrics.csv"),
        "Feature Importance": os.path.join(RESULTS_DIR, "feature_importance.csv"),
        "Model Summary": os.path.join(RESULTS_DIR, "model_summary.json"),
    }

    all_ok = True
    for name, path in files.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "✅" if exists and size > 0 else "❌"
        if not exists or size == 0:
            all_ok = False
        print(f"  {status} {name}: {path.split('/')[-1]} ({size:,} bytes)")

    if all_ok:
        # Read model summary
        with open(files["Model Summary"]) as f:
            summary = json.load(f)
        print(f"  📊 Overall IC: {summary.get('overall_ic', 'N/A')}")
        print(f"  📊 IC IR: {summary.get('ic_ir', 'N/A')}")
        print(f"  📊 Features: {summary.get('n_features', 'N/A')}")
        print(f"  📊 Top 5 features: {summary.get('top_10_features', [])[:5]}")

    return all_ok


def test_step7_outputs():
    """Verify Step 7 produced backtest results."""
    print("\n📋 TEST 2: Step 7 Outputs")
    files = {
        "Monthly Results": os.path.join(RESULTS_DIR, "backtest_monthly.csv"),
        "Subperiods": os.path.join(RESULTS_DIR, "backtest_subperiods.csv"),
        "Summary": os.path.join(RESULTS_DIR, "backtest_summary.json"),
    }

    all_ok = True
    for name, path in files.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "✅" if exists and size > 0 else "❌"
        if not exists or size == 0:
            all_ok = False
        print(f"  {status} {name}: {path.split('/')[-1]} ({size:,} bytes)")

    if all_ok:
        with open(files["Summary"]) as f:
            summary = json.load(f)
        print(f"  📊 Sharpe (net): {summary.get('sharpe_net', 'N/A')}")
        print(f"  📊 Ann Return (net): {summary.get('ann_return_net_pct', 'N/A')}%")
        print(f"  📊 Max Drawdown: {summary.get('max_drawdown_pct', 'N/A')}%")
        print(f"  📊 Hit Rate: {summary.get('hit_rate_pct', 'N/A')}%")

    return all_ok


def test_wrds_predictor():
    """Test WRDSPredictor with real data."""
    print("\n📋 TEST 3: WRDS Predictor")
    try:
        from src.nuble.ml.wrds_predictor import WRDSPredictor

        predictor = WRDSPredictor()
        print(f"  ✅ Predictor initialized: ready={predictor.is_ready}")

        if not predictor.is_ready:
            print(f"  ⚠️ Predictor not ready — model may not be trained yet")
            return False

        # Get model info
        info = predictor.get_model_info()
        print(f"  📊 Model info: {json.dumps(info, indent=2, default=str)}")

        # Test predictions for known stocks
        test_tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
        for ticker in test_tickers:
            result = predictor.predict(ticker)
            if "error" in result:
                print(f"  ⚠️ {ticker}: {result['error']}")
            else:
                print(f"  ✅ {ticker}: {result['direction']} "
                      f"(D{result['decile_rank']}, {result['confidence']:.0%}) "
                      f"pred={result['predicted_return_pct']:+.2f}%")

        # Test top picks
        top = predictor.get_top_picks(10)
        if len(top) > 0:
            print(f"\n  📊 Top 10 Picks:")
            for _, row in top.iterrows():
                ticker = row.get("ticker_name", str(row.get("permno", "?")))
                pred = row.get("predicted_return", 0)
                print(f"    {ticker:<8} pred={pred*100:+.2f}%")
        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wrds_advisor():
    """Test WRDSAdvisor data access layer."""
    print("\n📋 TEST 4: WRDS Advisor")
    try:
        from step8_wrds_advisor import WRDSAdvisor

        advisor = WRDSAdvisor()

        # Market overview
        overview = advisor.get_market_overview()
        print(f"  ✅ Market Overview: {overview.get('n_stocks', 0)} stocks, "
              f"as_of={overview.get('as_of', 'N/A')}")

        # Macro dashboard
        macro = advisor.get_macro_dashboard()
        print(f"  ✅ Macro: regime={macro.get('regime', 'N/A')}, "
              f"yield_curve={macro.get('yield_curve', 'N/A')}")

        # Stock profile
        profile = advisor.get_stock_profile("AAPL")
        if "error" not in profile:
            print(f"  ✅ AAPL Profile: permno={profile.get('permno')}, "
                  f"ml_signal={profile.get('ml_signal', 'N/A')}")
        else:
            print(f"  ⚠️ AAPL Profile: {profile['error']}")

        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_apex_integration():
    """Test APEX integration bridge."""
    print("\n📋 TEST 5: APEX Integration")
    try:
        from step9_apex_integration import APEXIntegration

        bridge = APEXIntegration()

        # Test each agent type
        agent_types = ["fundamental_analyst", "technical_analyst", "risk_analyst",
                        "macro_analyst", "ml_predictor"]
        for agent in agent_types:
            context = bridge.get_wrds_context_for_agent(agent, "AAPL")
            n_keys = sum(1 for k, v in context.items() if v and not isinstance(v, str))
            print(f"  ✅ {agent}: {n_keys} data sections")

        # Investment memo
        memo = bridge.generate_investment_memo("AAPL")
        print(f"  ✅ Investment memo: {len(memo)} chars")

        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    print("=" * 70)
    print("PHASE 3 — END-TO-END INTEGRATION TEST")
    print("=" * 70)

    results = {
        "Step 6 Outputs": test_step6_outputs(),
        "Step 7 Outputs": test_step7_outputs(),
        "WRDS Predictor": test_wrds_predictor(),
        "WRDS Advisor": test_wrds_advisor(),
        "APEX Integration": test_apex_integration(),
    }

    print(f"\n{'=' * 70}")
    print("INTEGRATION TEST RESULTS")
    print(f"{'=' * 70}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Score: {passed}/{total}")
    if passed == total:
        print(f"  🏆 ALL TESTS PASSED — Pipeline fully operational!")
    elif passed >= 3:
        print(f"  📊 Mostly working — some components need attention")
    else:
        print(f"  ⚠️ Multiple failures — check Step 6 completion")


if __name__ == "__main__":
    main()
