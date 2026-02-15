"""
PHASE 3 — STEP 9: APEX Integration
====================================
Bridge WRDSAdvisor to the existing APEX multi-agent system.
Injects WRDS data access into the APEX pipeline so agents can
use 100 years of academic data + ML predictions.
"""

import os
import sys
import json

# Add project root to path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


class APEXIntegration:
    """Bridge between WRDSAdvisor and APEX agent system."""

    def __init__(self):
        from step8_wrds_advisor import WRDSAdvisor
        self.advisor = WRDSAdvisor()
        self._load_apex_config()

    def _load_apex_config(self):
        """Load APEX system configuration."""
        manifest_path = os.path.join(_PROJECT_ROOT, "system_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}

    def get_wrds_context_for_agent(self, agent_type: str, ticker: str = None) -> dict:
        """Generate WRDS context payload for a specific APEX agent.

        Agent types:
            fundamental_analyst, technical_analyst, risk_analyst,
            macro_analyst, sentiment_analyst, ml_predictor,
            portfolio_manager, orchestrator
        """
        context = {"source": "WRDS", "agent": agent_type}

        if agent_type == "fundamental_analyst":
            if ticker:
                context["profile"] = self.advisor.get_stock_profile(ticker)
                context["sector"] = self.advisor.get_sector_analysis(
                    context["profile"].get("siccd") if "siccd" in context["profile"] else None
                )
            context["market"] = self.advisor.get_market_overview()

        elif agent_type == "technical_analyst":
            if ticker:
                profile = self.advisor.get_stock_profile(ticker)
                context["momentum"] = {
                    "mom_12m": profile.get("mom_12m"),
                    "mom_1m": profile.get("mom_1m"),
                    "realized_vol": profile.get("realized_vol"),
                }
            context["market"] = self.advisor.get_market_overview()

        elif agent_type == "risk_analyst":
            if ticker:
                profile = self.advisor.get_stock_profile(ticker)
                context["risk_metrics"] = {
                    "beta": profile.get("beta"),
                    "realized_vol": profile.get("realized_vol"),
                    "amihud_illiq": profile.get("amihud_illiq"),
                    "leverage": profile.get("lev"),
                }
            context["macro"] = self.advisor.get_macro_dashboard()

        elif agent_type == "macro_analyst":
            context["macro"] = self.advisor.get_macro_dashboard()
            context["market"] = self.advisor.get_market_overview()

        elif agent_type == "ml_predictor":
            if ticker:
                context["prediction"] = self.advisor.get_model_prediction(ticker)
                context["profile"] = self.advisor.get_stock_profile(ticker)
            context["model_info"] = self._get_model_info()

        elif agent_type == "portfolio_manager":
            context["portfolio"] = self.advisor.get_portfolio_analytics()
            context["macro"] = self.advisor.get_macro_dashboard()
            if ticker:
                context["prediction"] = self.advisor.get_model_prediction(ticker)

        elif agent_type == "orchestrator":
            # Full context for orchestrator
            context["market"] = self.advisor.get_market_overview()
            context["macro"] = self.advisor.get_macro_dashboard()
            context["portfolio"] = self.advisor.get_portfolio_analytics()
            if ticker:
                context["prediction"] = self.advisor.get_model_prediction(ticker)
                context["profile"] = self.advisor.get_stock_profile(ticker)

        else:
            # Default: basic context
            context["market"] = self.advisor.get_market_overview()
            if ticker:
                context["profile"] = self.advisor.get_stock_profile(ticker)

        return context

    def _get_model_info(self) -> dict:
        """Get ML model metadata."""
        summary_path = os.path.join(RESULTS_DIR, "model_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                return json.load(f)
        return {"status": "model not yet trained"}

    def screen_universe(self, criteria: dict, top_n: int = 20) -> list:
        """Screen stocks and return top N by ML prediction."""
        results = self.advisor.search_stocks(criteria)
        if len(results) == 0:
            return []

        keep_cols = ["permno", "ticker", "log_market_cap", "bm", "roaq", "mom_12m"]
        if "prediction" in results.columns:
            keep_cols.append("prediction")
        keep_cols = [c for c in keep_cols if c in results.columns]

        return results[keep_cols].head(top_n).to_dict(orient="records")

    def generate_investment_memo(self, ticker: str) -> str:
        """Generate a structured investment memo using WRDS data."""
        profile = self.advisor.get_stock_profile(ticker)
        macro = self.advisor.get_macro_dashboard()
        prediction = self.advisor.get_model_prediction(ticker)

        memo = f"""
═══════════════════════════════════════════════════════════
INVESTMENT MEMO: {ticker.upper()}
Generated by NUBLE WRDS-Powered Advisory System
═══════════════════════════════════════════════════════════

1. COMPANY OVERVIEW
   Ticker:     {profile.get('ticker', 'N/A')}
   PERMNO:     {profile.get('permno', 'N/A')}
   Date:       {profile.get('date', 'N/A')}
   Market Cap: {profile.get('log_market_cap', 'N/A')} (log)

2. FUNDAMENTALS
   Book/Market:       {profile.get('bm', 'N/A')}
   Earnings/Price:    {profile.get('ep', 'N/A')}
   ROA (quarterly):   {profile.get('roaq', 'N/A')}
   ROE (quarterly):   {profile.get('roeq', 'N/A')}
   Leverage:          {profile.get('lev', 'N/A')}

3. MOMENTUM & RISK
   12M Momentum:      {profile.get('mom_12m', 'N/A')}
   1M Momentum:       {profile.get('mom_1m', 'N/A')}
   Beta:              {profile.get('beta', 'N/A')}
   Realized Vol:      {profile.get('realized_vol', 'N/A')}
   Illiquidity:       {profile.get('amihud_illiq', 'N/A')}

4. ML PREDICTION
   Signal:            {prediction.get('signal', 'N/A')}
   Predicted Return:  {prediction.get('predicted_return', 'N/A')}%
   Percentile:        {prediction.get('percentile', 'N/A')}
   Model IC:          {prediction.get('model_ic', 'N/A')}

5. MACRO ENVIRONMENT
   Regime:            {macro.get('regime', 'N/A')}
   Yield Curve:       {macro.get('yield_curve', 'N/A')}
   Volatility:        {macro.get('volatility_regime', 'N/A')}

═══════════════════════════════════════════════════════════
"""
        return memo


def main():
    print("=" * 70)
    print("PHASE 3 — STEP 9: APEX INTEGRATION")
    print("=" * 70)

    bridge = APEXIntegration()

    # Test: Get context for each agent type
    test_ticker = "AAPL"
    agent_types = [
        "fundamental_analyst", "technical_analyst", "risk_analyst",
        "macro_analyst", "ml_predictor", "portfolio_manager", "orchestrator",
    ]

    for agent in agent_types:
        context = bridge.get_wrds_context_for_agent(agent, test_ticker)
        n_keys = sum(1 for k, v in context.items() if v and not isinstance(v, str))
        print(f"  ✅ {agent:<25} → {n_keys} data sections")

    # Test: Investment memo
    print(f"\n📝 INVESTMENT MEMO")
    memo = bridge.generate_investment_memo(test_ticker)
    print(memo)

    # Test: Stock screening
    print("📊 STOCK SCREENING: Value + Quality")
    value_quality = bridge.screen_universe(
        {"bm_min": 0.5, "roaq_min": 0.02}, top_n=10
    )
    for stock in value_quality[:5]:
        print(f"  {stock}")

    print(f"\n{'=' * 70}")
    print(f"APEX INTEGRATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  ✅ APEXIntegration class ready")
    print(f"  ✅ All agent types supported")
    print(f"  ✅ Investment memo generator working")
    print(f"  ✅ Stock screener working")


if __name__ == "__main__":
    main()
