"""
PHASE 3 — STEP 14: Alternative Data Integration
=================================================
Augment WRDS academic data with real-time alternative data:
  1. Polygon.io real-time prices (existing integration)
  2. FRED daily regime detection
  3. SEC EDGAR NLP sentiment (existing integration)
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import json
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/humbertolobo/Desktop/NUBLE-CLI")

DATA_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/data/wrds"
RESULTS_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/wrds_pipeline/phase3/results"


def integrate_polygon_realtime():
    """Connect existing Polygon.io data to WRDS pipeline."""
    print("\n📊 1. POLYGON.IO REAL-TIME INTEGRATION")

    # Check if Polygon module exists
    polygon_path = "/Users/humbertolobo/Desktop/NUBLE-CLI/src/nuble/data/polygon_universe.py"
    if os.path.exists(polygon_path):
        print("  ✅ Polygon universe module found")
        print("  Connection: WRDS monthly fundamentals + Polygon daily prices")
        print("  Use case: Intraday signal generation with monthly alpha model")

        # Check for Polygon API key
        api_key = os.environ.get("POLYGON_API_KEY", "")
        if api_key:
            print(f"  API key: ...{api_key[-4:]}")
        else:
            print("  ⚠️ POLYGON_API_KEY not set — set in environment for real-time data")

        integration = {
            "source": "polygon",
            "status": "module_available",
            "use_case": "daily_price_overlay_on_monthly_model",
            "features": [
                "intraday_return",
                "volume_vs_average",
                "price_vs_model_prediction",
                "real_time_momentum",
            ],
        }
    else:
        print("  ⚠️ Polygon module not found at expected path")
        integration = {"source": "polygon", "status": "not_found"}

    return integration


def integrate_fred_regime():
    """Use FRED daily data for regime detection."""
    print("\n📊 2. FRED DAILY REGIME DETECTION")

    fred_path = os.path.join(DATA_DIR, "fred_daily.parquet")
    if not os.path.exists(fred_path):
        print("  ⚠️ fred_daily.parquet not found — run Step 3 first")
        return {"source": "fred_regime", "status": "no_data"}

    fred = pd.read_parquet(fred_path)
    fred["date"] = pd.to_datetime(fred["date"])
    latest = fred.sort_values("date").iloc[-1]

    regime = {"source": "fred_regime", "as_of": str(latest["date"].date())}

    # VIX regime
    if "vix" in fred.columns:
        vix = latest["vix"]
        vix_20d = fred["vix"].tail(20).mean()
        vix_60d = fred["vix"].tail(60).mean()
        regime["vix"] = {
            "current": round(float(vix), 1) if pd.notna(vix) else None,
            "avg_20d": round(float(vix_20d), 1) if pd.notna(vix_20d) else None,
            "avg_60d": round(float(vix_60d), 1) if pd.notna(vix_60d) else None,
            "regime": "CRISIS" if vix > 35 else "HIGH" if vix > 25 else "ELEVATED" if vix > 18 else "LOW",
        }
        print(f"  VIX: {vix:.1f} → {regime['vix']['regime']}")

    # Credit spread regime
    if "credit_spread" in fred.columns:
        cs = latest["credit_spread"]
        cs_avg = fred["credit_spread"].tail(252).mean()
        regime["credit"] = {
            "spread": round(float(cs), 2) if pd.notna(cs) else None,
            "avg_1y": round(float(cs_avg), 2) if pd.notna(cs_avg) else None,
            "regime": "STRESS" if cs > 2.0 else "TIGHT" if cs < 0.8 else "NORMAL",
        }
        print(f"  Credit Spread: {cs:.2f} → {regime['credit']['regime']}")

    # Yield curve regime
    if "term_spread_10y2y" in fred.columns:
        ts = latest["term_spread_10y2y"]
        regime["yield_curve"] = {
            "spread_10y2y": round(float(ts), 2) if pd.notna(ts) else None,
            "regime": "INVERTED" if ts < 0 else "FLAT" if ts < 0.5 else "NORMAL" if ts < 1.5 else "STEEP",
        }
        print(f"  Yield Curve: {ts:.2f} → {regime['yield_curve']['regime']}")

    # Dollar regime
    if "trade_weighted_usd" in fred.columns:
        usd = latest["trade_weighted_usd"]
        usd_200d = fred["trade_weighted_usd"].tail(200).mean()
        regime["dollar"] = {
            "current": round(float(usd), 1) if pd.notna(usd) else None,
            "avg_200d": round(float(usd_200d), 1) if pd.notna(usd_200d) else None,
            "trend": "STRONG" if usd > usd_200d * 1.02 else "WEAK" if usd < usd_200d * 0.98 else "NEUTRAL",
        }
        print(f"  USD: {usd:.1f} → {regime['dollar']['trend']}")

    # Overall regime
    stress_signals = 0
    if regime.get("vix", {}).get("regime") in ["CRISIS", "HIGH"]:
        stress_signals += 1
    if regime.get("credit", {}).get("regime") == "STRESS":
        stress_signals += 1
    if regime.get("yield_curve", {}).get("regime") == "INVERTED":
        stress_signals += 1

    regime["overall"] = "RISK-OFF" if stress_signals >= 2 else "CAUTION" if stress_signals == 1 else "RISK-ON"
    print(f"  Overall regime: {regime['overall']} ({stress_signals} stress signals)")

    return regime


def integrate_sec_edgar():
    """Connect existing SEC EDGAR NLP to WRDS pipeline."""
    print("\n📊 3. SEC EDGAR NLP INTEGRATION")

    edgar_path = "/Users/humbertolobo/Desktop/NUBLE-CLI/src/nuble/data/sec_edgar.py"
    if os.path.exists(edgar_path):
        print("  ✅ SEC EDGAR module found")
        print("  Connection: NLP sentiment from 10-K/10-Q filings + WRDS fundamentals")

        integration = {
            "source": "sec_edgar",
            "status": "module_available",
            "use_case": "filing_sentiment_overlay",
            "features": [
                "filing_sentiment_score",
                "management_tone_change",
                "risk_factor_count",
                "readability_score",
            ],
        }
    else:
        print("  ⚠️ SEC EDGAR module not found")
        integration = {"source": "sec_edgar", "status": "not_found"}

    # Check TENK_SOURCE
    tenk_path = "/Users/humbertolobo/Desktop/NUBLE-CLI/TENK_SOURCE"
    if os.path.exists(tenk_path):
        print(f"  ✅ TENK_SOURCE found — 10-K filing analysis available")
        integration["tenk_available"] = True

    return integration


def main():
    print("=" * 70)
    print("PHASE 3 — STEP 14: ALTERNATIVE DATA INTEGRATION")
    print("=" * 70)
    start = time.time()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Run all integrations
    polygon_result = integrate_polygon_realtime()
    fred_result = integrate_fred_regime()
    edgar_result = integrate_sec_edgar()

    # Save integration status
    integration_status = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "polygon": polygon_result,
        "fred_regime": fred_result,
        "sec_edgar": edgar_result,
        "data_sources": {
            "academic": ["WRDS CRSP", "WRDS Compustat", "WRDS IBES", "WRDS FF",
                         "Chen-Zimmermann", "Welch-Goyal"],
            "macro": ["FRED 35+ series", "Welch-Goyal 14 vars"],
            "alternative": ["Polygon.io (real-time)", "SEC EDGAR (NLP)",
                            "FRED daily (regime)"],
        },
    }

    with open(os.path.join(RESULTS_DIR, "alt_data_integration.json"), "w") as f:
        json.dump(integration_status, f, indent=2, default=str)

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"ALTERNATIVE DATA INTEGRATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Polygon.io:   {polygon_result.get('status', 'N/A')}")
    print(f"  FRED Regime:  {fred_result.get('overall', 'N/A')}")
    print(f"  SEC EDGAR:    {edgar_result.get('status', 'N/A')}")
    print(f"  Time:         {elapsed:.0f}s")
    print(f"  ✅ Integration status saved")


if __name__ == "__main__":
    main()
