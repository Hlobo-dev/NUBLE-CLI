"""
PHASE 3 — STEP 8: WRDSAdvisor — Data Access Layer
===================================================
The brain of the financial advisor. Provides structured data access
methods that APEX agents can call for any stock, sector, or macro query.

Methods:
  get_stock_profile(ticker)     → company fundamentals + ML prediction
  get_market_overview()         → broad market snapshot
  get_macro_dashboard()         → macro regime + indicators
  get_model_prediction(ticker)  → ML return forecast with confidence
  search_stocks(criteria)       → screen by fundamentals
  get_portfolio_analytics()     → current model portfolio stats
  get_historical_comparison()   → compare to similar periods
  get_sector_analysis(sector)   → sector-level aggregates
"""

import pandas as pd
import numpy as np
import os
import json
import warnings

warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


class WRDSAdvisor:
    """Data access layer for WRDS-powered financial advisory."""

    def __init__(self):
        self._panel = None
        self._predictions = None
        self._macro = None
        self._fred = None
        self._model_summary = None
        self._backtest = None
        self._ticker_to_permno = None
        self._permno_to_ticker = None
        print("WRDSAdvisor initializing...")
        self._load_data()
        print(f"WRDSAdvisor ready: {len(self._panel):,} stock-months, "
              f"{len(self._predictions):,} predictions, "
              f"{len(self._ticker_to_permno):,} ticker mappings")

    def _load_data(self):
        """Load all available data."""
        # ── Ticker ↔ Permno mapping (CRITICAL for real-time lookups) ──
        ticker_map_path = os.path.join(DATA_DIR, "ticker_permno_map.parquet")
        permno_map_path = os.path.join(DATA_DIR, "permno_ticker_map.parquet")
        if os.path.exists(ticker_map_path):
            tm = pd.read_parquet(ticker_map_path)
            self._ticker_to_permno = dict(zip(tm["ticker"].str.upper(), tm["permno"].astype(int)))
        else:
            # Fallback: build from crsp_monthly
            crsp_path = os.path.join(DATA_DIR, "crsp_monthly.parquet")
            if os.path.exists(crsp_path):
                crsp = pd.read_parquet(crsp_path, columns=["permno", "ticker", "date"])
                crsp = crsp.dropna(subset=["permno", "ticker"])
                crsp["permno"] = crsp["permno"].astype(int)
                latest = crsp.sort_values("date").drop_duplicates("ticker", keep="last")
                self._ticker_to_permno = dict(zip(latest["ticker"].str.upper(), latest["permno"]))
            else:
                self._ticker_to_permno = {}

        if os.path.exists(permno_map_path):
            pm = pd.read_parquet(permno_map_path)
            self._permno_to_ticker = dict(zip(pm["permno"].astype(int), pm["ticker"]))
        else:
            self._permno_to_ticker = {v: k for k, v in self._ticker_to_permno.items()}

        # ── Panel data (prefer enriched gkx_panel, fallback to training_panel) ──
        panel_path = os.path.join(DATA_DIR, "gkx_panel.parquet")
        if not os.path.exists(panel_path):
            panel_path = os.path.join(DATA_DIR, "training_panel.parquet")
        if os.path.exists(panel_path):
            # Only load columns we need for serving (not all 539)
            essential_cols = [
                "permno", "date", "fwd_ret_1m",
                "log_market_cap", "market_cap", "bm", "ep", "cashpr", "dy", "lev",
                "roaq", "roeq", "mom_12m", "mom_1m", "mom_6m", "beta",
                "realized_vol", "amihud_illiq", "turnover", "turnover_6m",
                "siccd", "ret_crsp",
                # Level 3 features (key ones)
                "revenue_growth_yoy", "revenue_acceleration", "operating_leverage",
                "gross_margin_trend", "operating_margin_trend",
                "total_accruals", "piotroski_f_score", "altman_z_score",
                "beneish_m_score", "earnings_persistence",
                "eps_revision_1m", "sue", "insider_buy_ratio",
            ]
            try:
                import pyarrow.parquet as pq
                available_cols = pq.read_schema(panel_path).names
                load_cols = [c for c in essential_cols if c in available_cols]
                self._panel = pd.read_parquet(panel_path, columns=load_cols)
            except Exception:
                self._panel = pd.read_parquet(panel_path)
            self._panel["date"] = pd.to_datetime(self._panel["date"])
            print(f"  Panel: {os.path.basename(panel_path)} → {len(self._panel):,} rows × {len(self._panel.columns)} cols")
        else:
            self._panel = pd.DataFrame()

        # ── ML predictions (prefer ensemble, fallback chain) ──
        pred_priority = [
            "ensemble_predictions.parquet",
            "production_predictions.parquet",
            "final_predictions.parquet",
            "lgb_predictions.parquet",
        ]
        self._predictions = pd.DataFrame()
        for pred_file in pred_priority:
            pred_path = os.path.join(DATA_DIR, pred_file)
            if os.path.exists(pred_path):
                self._predictions = pd.read_parquet(pred_path)
                self._predictions["date"] = pd.to_datetime(self._predictions["date"])
                print(f"  Predictions: {pred_file} → {len(self._predictions):,} rows")
                break

        # Macro data
        macro_path = os.path.join(DATA_DIR, "macro_predictors.parquet")
        if os.path.exists(macro_path):
            self._macro = pd.read_parquet(macro_path)
            self._macro["date"] = pd.to_datetime(self._macro["date"])
        else:
            self._macro = pd.DataFrame()

        # FRED daily
        fred_path = os.path.join(DATA_DIR, "fred_daily.parquet")
        if os.path.exists(fred_path):
            self._fred = pd.read_parquet(fred_path)
            self._fred["date"] = pd.to_datetime(self._fred["date"])
        else:
            self._fred = pd.DataFrame()

        # Model summary
        summary_path = os.path.join(RESULTS_DIR, "model_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                self._model_summary = json.load(f)

        # Backtest summary
        bt_path = os.path.join(RESULTS_DIR, "backtest_summary.json")
        if os.path.exists(bt_path):
            with open(bt_path) as f:
                self._backtest = json.load(f)

    def _resolve_permno(self, ticker: str) -> int:
        """Resolve a ticker symbol to its CRSP permno."""
        t = ticker.strip().upper()
        return self._ticker_to_permno.get(t)

    def _resolve_ticker(self, permno: int) -> str:
        """Resolve a CRSP permno to its ticker symbol."""
        return self._permno_to_ticker.get(int(permno), str(permno))

    def get_stock_profile(self, ticker: str) -> dict:
        """Get comprehensive stock profile with fundamentals + ML prediction."""
        if len(self._panel) == 0:
            return {"error": "No panel data available"}

        # Resolve ticker → permno via mapping
        permno = self._resolve_permno(ticker)
        if permno is None:
            return {"error": f"Ticker {ticker} not found in permno mapping"}

        stock = self._panel[self._panel["permno"] == permno]
        if len(stock) == 0:
            return {"error": f"Ticker {ticker} (permno={permno}) not found in panel"}

        latest = stock.sort_values("date").iloc[-1]

        profile = {
            "ticker": ticker.upper(),
            "date": str(latest["date"].date()),
            "permno": int(permno),
        }

        # Fundamentals
        fundamental_keys = [
            "log_market_cap", "bm", "ep", "cashpr", "dy", "lev",
            "roaq", "roeq", "mom_12m", "mom_1m", "beta",
            "realized_vol", "amihud_illiq", "turnover",
        ]
        for key in fundamental_keys:
            if key in latest.index and pd.notna(latest[key]):
                profile[key] = round(float(latest[key]), 4)

        # ML prediction
        if len(self._predictions) > 0 and "permno" in self._predictions.columns:
            preds = self._predictions[self._predictions["permno"] == profile.get("permno", -1)]
            if len(preds) > 0:
                latest_pred = preds.sort_values("date").iloc[-1]
                profile["ml_prediction"] = round(float(latest_pred["prediction"]), 4)
                profile["ml_date"] = str(latest_pred["date"].date())

                # Rank within latest month
                month_preds = self._predictions[self._predictions["date"] == latest_pred["date"]]
                rank = (month_preds["prediction"] <= latest_pred["prediction"]).mean()
                profile["ml_percentile"] = round(float(rank * 100), 1)
                profile["ml_signal"] = "BUY" if rank > 0.8 else "SELL" if rank < 0.2 else "HOLD"

        return profile

    def get_market_overview(self) -> dict:
        """Get current market overview."""
        overview = {"as_of": None}

        if len(self._panel) > 0:
            latest_date = self._panel["date"].max()
            overview["as_of"] = str(latest_date.date())
            month_data = self._panel[self._panel["date"] == latest_date]

            overview["n_stocks"] = int(len(month_data))

            if "ret" in month_data.columns:
                overview["avg_return"] = round(float(month_data["ret"].mean()) * 100, 2)
                overview["median_return"] = round(float(month_data["ret"].median()) * 100, 2)
                overview["return_dispersion"] = round(float(month_data["ret"].std()) * 100, 2)

            if "log_market_cap" in month_data.columns:
                overview["median_market_cap_log"] = round(float(month_data["log_market_cap"].median()), 2)

        # Macro snapshot
        if len(self._fred) > 0:
            latest_fred = self._fred.sort_values("date").iloc[-1]
            overview["macro"] = {}
            for col in ["vix", "fed_funds_rate", "treasury_10y", "credit_spread", "wti_crude"]:
                if col in latest_fred.index and pd.notna(latest_fred[col]):
                    overview["macro"][col] = round(float(latest_fred[col]), 2)

        return overview

    def get_macro_dashboard(self) -> dict:
        """Get macro regime and key indicators."""
        dashboard = {"regime": "unknown", "indicators": {}}

        if len(self._macro) == 0:
            return dashboard

        latest = self._macro.sort_values("date").iloc[-1]
        dashboard["as_of"] = str(latest["date"].date()) if "date" in latest.index else None

        # Classify regime
        indicators = {}
        for col in self._macro.columns:
            if col != "date" and pd.notna(latest[col]):
                indicators[col] = round(float(latest[col]), 4)

        dashboard["indicators"] = indicators

        # Regime classification
        if "tms" in indicators:
            if indicators["tms"] < 0:
                dashboard["yield_curve"] = "INVERTED (recession signal)"
            elif indicators["tms"] < 0.5:
                dashboard["yield_curve"] = "FLAT (late cycle)"
            else:
                dashboard["yield_curve"] = "NORMAL (expansion)"

        if "nber_recession" in indicators:
            dashboard["regime"] = "RECESSION" if indicators["nber_recession"] == 1 else "EXPANSION"

        if "vix" in indicators:
            vix = indicators["vix"]
            if vix > 30:
                dashboard["volatility_regime"] = "HIGH STRESS"
            elif vix > 20:
                dashboard["volatility_regime"] = "ELEVATED"
            else:
                dashboard["volatility_regime"] = "LOW/NORMAL"

        return dashboard

    def get_model_prediction(self, ticker: str = None, permno: int = None) -> dict:
        """Get ML model prediction for a stock."""
        if len(self._predictions) == 0:
            return {"error": "No predictions available — run Step 6b"}

        latest_date = self._predictions["date"].max()
        month_preds = self._predictions[self._predictions["date"] == latest_date]

        # Resolve ticker → permno via mapping
        if ticker and permno is None:
            permno = self._resolve_permno(ticker)

        if permno is None:
            return {"error": f"Ticker '{ticker}' not found in permno mapping"}

        stock_pred = month_preds[month_preds["permno"] == permno]
        if len(stock_pred) == 0:
            return {"error": f"No prediction for permno {permno}"}

        pred_value = float(stock_pred.iloc[0]["prediction"])
        rank = (month_preds["prediction"] <= pred_value).mean()

        result = {
            "permno": permno,
            "date": str(latest_date.date()),
            "predicted_return": round(pred_value * 100, 2),
            "percentile": round(float(rank * 100), 1),
            "signal": "STRONG BUY" if rank > 0.9 else "BUY" if rank > 0.7 else
                      "HOLD" if rank > 0.3 else "SELL" if rank > 0.1 else "STRONG SELL",
            "n_stocks_ranked": len(month_preds),
        }

        if self._model_summary:
            result["model_ic"] = self._model_summary.get("overall_ic")
            result["model_sharpe"] = self._backtest.get("sharpe_net") if self._backtest else None

        return result

    def search_stocks(self, criteria: dict) -> pd.DataFrame:
        """Screen stocks by criteria.

        Example criteria:
            {"log_market_cap_min": 8, "bm_min": 0.5, "mom_12m_min": 0.1}
        """
        if len(self._panel) == 0:
            return pd.DataFrame()

        latest_date = self._panel["date"].max()
        universe = self._panel[self._panel["date"] == latest_date].copy()

        for key, value in criteria.items():
            col = key.replace("_min", "").replace("_max", "")
            if col not in universe.columns:
                continue
            if key.endswith("_min"):
                universe = universe[universe[col] >= value]
            elif key.endswith("_max"):
                universe = universe[universe[col] <= value]

        # Add ML prediction if available
        if len(self._predictions) > 0:
            month_preds = self._predictions[self._predictions["date"] == latest_date]
            if len(month_preds) > 0:
                universe = universe.merge(
                    month_preds[["permno", "prediction"]],
                    on="permno", how="left"
                )
                universe = universe.sort_values("prediction", ascending=False)

        # Add ticker names from mapping
        universe["ticker"] = universe["permno"].map(self._permno_to_ticker)

        return universe

    def get_portfolio_analytics(self) -> dict:
        """Get current model portfolio statistics."""
        if self._backtest is None:
            return {"error": "No backtest results — run Step 7"}

        analytics = {
            "strategy": self._backtest.get("strategy"),
            "period": self._backtest.get("period"),
            "sharpe_net": self._backtest.get("sharpe_net"),
            "ann_return_net_pct": self._backtest.get("ann_return_net_pct"),
            "max_drawdown_pct": self._backtest.get("max_drawdown_pct"),
            "hit_rate_pct": self._backtest.get("hit_rate_pct"),
        }

        # Current portfolio composition (latest month)
        if len(self._predictions) > 0:
            latest_date = self._predictions["date"].max()
            month_preds = self._predictions[self._predictions["date"] == latest_date]
            decile = pd.qcut(month_preds["prediction"], 10, labels=False, duplicates="drop")
            long_permnos = month_preds[decile == decile.max()]["permno"].tolist()
            short_permnos = month_preds[decile == decile.min()]["permno"].tolist()

            analytics["current_date"] = str(latest_date.date())
            analytics["n_long"] = len(long_permnos)
            analytics["n_short"] = len(short_permnos)

            # Map permnos to tickers
            analytics["top_longs"] = [self._resolve_ticker(p) for p in long_permnos[:10]]
            analytics["top_shorts"] = [self._resolve_ticker(p) for p in short_permnos[:10]]

        return analytics

    def get_sector_analysis(self, siccd: int = None) -> dict:
        """Get sector-level analysis."""
        if len(self._panel) == 0:
            return {"error": "No panel data"}

        latest_date = self._panel["date"].max()
        universe = self._panel[self._panel["date"] == latest_date]

        if "siccd" not in universe.columns:
            return {"error": "No SIC codes available"}

        # 1-digit SIC sectors
        universe = universe.copy()
        universe["sector"] = (universe["siccd"] // 1000).astype(int)
        sector_names = {
            0: "Agriculture", 1: "Mining", 2: "Manufacturing (light)",
            3: "Manufacturing (heavy)", 4: "Transportation", 5: "Retail",
            6: "Finance", 7: "Services", 8: "Healthcare/Education", 9: "Government",
        }

        sectors = []
        for sector, group in universe.groupby("sector"):
            info = {
                "sector_code": int(sector),
                "sector_name": sector_names.get(int(sector), "Other"),
                "n_stocks": len(group),
            }
            for metric in ["ret", "log_market_cap", "bm", "mom_12m", "roaq"]:
                if metric in group.columns:
                    info[f"avg_{metric}"] = round(float(group[metric].mean()), 4)
            sectors.append(info)

        result = {"date": str(latest_date.date()), "sectors": sectors}

        if siccd is not None:
            target_sector = siccd // 1000
            result["focus_sector"] = sector_names.get(target_sector, f"SIC {siccd}")
            focus = universe[universe["sector"] == target_sector]
            if len(focus) > 0:
                result["focus_n_stocks"] = len(focus)
                top_stocks = focus.nlargest(
                    min(10, len(focus)),
                    "log_market_cap" if "log_market_cap" in focus.columns else focus.columns[0]
                )
                result["focus_top_stocks"] = [
                    self._resolve_ticker(int(row["permno"])) for _, row in top_stocks.iterrows()
                ]

        return result


def main():
    print("=" * 70)
    print("PHASE 3 — STEP 8: WRDS ADVISOR (DATA ACCESS LAYER)")
    print("=" * 70)

    advisor = WRDSAdvisor()

    # Demo: Market Overview
    print("\n📊 MARKET OVERVIEW")
    overview = advisor.get_market_overview()
    for k, v in overview.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    # Demo: Macro Dashboard
    print("\n📊 MACRO DASHBOARD")
    macro = advisor.get_macro_dashboard()
    print(f"  Regime: {macro.get('regime')}")
    print(f"  Yield curve: {macro.get('yield_curve', 'N/A')}")
    print(f"  Volatility: {macro.get('volatility_regime', 'N/A')}")

    # Demo: Stock Profile (if we have ticker data)
    print("\n📊 STOCK PROFILE (AAPL)")
    profile = advisor.get_stock_profile("AAPL")
    for k, v in profile.items():
        print(f"  {k}: {v}")

    # Demo: Portfolio Analytics
    print("\n📊 PORTFOLIO ANALYTICS")
    analytics = advisor.get_portfolio_analytics()
    for k, v in analytics.items():
        if not isinstance(v, list):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v[:5]}...")

    print(f"\n{'=' * 70}")
    print(f"WRDS ADVISOR READY")
    print(f"{'=' * 70}")
    print(f"  ✅ Available methods:")
    print(f"    advisor.get_stock_profile('AAPL')")
    print(f"    advisor.get_market_overview()")
    print(f"    advisor.get_macro_dashboard()")
    print(f"    advisor.get_model_prediction('AAPL')")
    print(f"    advisor.search_stocks({{'bm_min': 0.5}})")
    print(f"    advisor.get_portfolio_analytics()")
    print(f"    advisor.get_sector_analysis()")


if __name__ == "__main__":
    main()
