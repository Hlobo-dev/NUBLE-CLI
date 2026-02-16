#!/usr/bin/env python3
"""
NUBLE Tool Executor v2 — ROKET-powered Claude ↔ Tools loop
=============================================================
This module receives tool_use requests from Claude and dispatches them to
the ROKET API endpoints (direct function calls, no HTTP overhead).

Two integration patterns supported:

Pattern A — Direct (recommended for standalone frontends):
    Your frontend calls Claude with tools → Claude returns tool_use →
    your frontend calls the matching /api/roket/* endpoint → feeds result
    back to Claude → Claude generates final answer.

Pattern B — Server-side dispatch (this module):
    Your frontend sends the raw user message → this endpoint handles the
    full Claude ↔ tools loop internally → returns the final answer.

Usage (Pattern B):
    POST /api/intel/chat-with-tools
    {
        "message": "Should I buy NVDA?",
        "conversation_id": "optional"
    }

    This will:
    1. Send message to Claude Opus with ROKET tools schema
    2. If Claude calls a tool, execute it against the intelligence stack
    3. Feed tool result back to Claude
    4. Return Claude's final synthesized answer
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/intel", tags=["Intelligence Chat"])


class ChatWithToolsRequest(BaseModel):
    """Request for tool-augmented chat."""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = None
    max_tool_rounds: int = Field(3, description="Max tool call rounds before forcing a response")


class ChatWithToolsResponse(BaseModel):
    """Response from tool-augmented chat."""
    message: str
    tools_used: List[str] = []
    tool_results: List[Dict[str, Any]] = []
    execution_time_seconds: float


# ── Tool dispatch map ───────────────────────────────────────────────────

def _dispatch_tool(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool call locally (no HTTP, direct function call).
    This is faster than routing through HTTP when running server-side.
    Supports all ROKET tools + legacy tool names for backward compatibility.
    """
    try:
        # ── ROKET prediction tools ──────────────────────────────────
        if tool_name in ("roket_predict", "get_stock_prediction"):
            ticker = tool_input.get('ticker', '').upper().strip()
            if not ticker:
                return {'error': 'Ticker symbol required'}
            from ..ml.live_predictor import get_live_predictor
            lp = get_live_predictor()
            result = lp.predict(ticker)
            # Surface ticker-not-found errors clearly
            if result.get("error") and result.get("confidence", 1) == 0.0:
                return {'ticker': ticker, 'error': result['error']}
            return result

        elif tool_name in ("roket_analyze",):
            # Delegate to the full ROKET analyze endpoint which has all engines wired in
            # (UDE, TradeSetup, SignalFusion, Veto, PositionCalculator)
            import httpx
            ticker = tool_input['ticker'].upper().strip()
            try:
                # In server.py mode, ROKET is mounted at /api/roket/*
                base = os.environ.get("ROKET_BASE_URL", "http://localhost:8000")
                url = f"{base}/api/roket/analyze/{ticker}"
                with httpx.Client(timeout=120.0) as client:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        return resp.json()
                    else:
                        logger.warning(f"ROKET analyze returned {resp.status_code}, falling back to local")
            except Exception as e:
                logger.debug(f"ROKET HTTP analyze failed ({e}), falling back to local dispatch")

            # Fallback: local dispatch (without advanced engines)
            from ..ml.live_predictor import get_live_predictor
            from ..ml.wrds_predictor import get_wrds_predictor
            import pandas as pd
            lp = get_live_predictor()
            wp = get_wrds_predictor()
            wp._ensure_loaded()
            try:
                prediction = lp.predict(ticker)
            except Exception:
                prediction = wp.predict(ticker)
            # Extract feature groups from panel
            fundamentals, earnings_data, risk_data, insider_data, inst_data = {}, {}, {}, {}, {}
            permno = wp._resolve_permno(ticker)
            if permno is not None and wp._panel is not None:
                stock_data = wp._panel[wp._panel["permno"] == permno].sort_values("date")
                if len(stock_data) > 0:
                    row = stock_data.iloc[-1]
                    for col in ["bm", "pe_inc", "pe_exi", "pe_op_basic", "ptb", "pcf",
                                "ps", "divyield", "capei", "roe", "roa", "roce",
                                "gpm", "npm", "opmad", "cfm", "fcf_ocf",
                                "curr_ratio", "de_ratio", "debt_at",
                                "piotroski_f_score", "beneish_m_score",
                                "altman_z_score", "ohlson_o_score",
                                "revenue_growth_yoy", "rd_sale", "rd_intensity",
                                "ep", "sp", "cfp", "dp", "gma", "opa", "operprof",
                                "cash", "cashdebt", "cashpr", "lev", "sgr", "agr"]:
                        if col in row.index and pd.notna(row[col]):
                            fundamentals[col] = round(float(row[col]), 6)
                    for col in ["sue_ibes", "earnings_persistence", "earnings_smoothness",
                                "beat_miss_streak", "eps_dispersion", "eps_revision_1m",
                                "eps_revision_3m", "revision_breadth",
                                "accrual", "total_accruals", "accruals_to_cash_flow",
                                "sue", "re", "nincr", "roaq", "roavol",
                                "e2p", "earnings_yield"]:
                        if col in row.index and pd.notna(row[col]):
                            earnings_data[col] = round(float(row[col]), 6)
                    for col in ["beta_mkt", "beta_smb", "beta_hml", "r_squared", "alpha",
                                "realized_vol", "idio_vol", "total_vol",
                                "down_vol", "up_vol", "vol_3m", "vol_6m",
                                "return_skewness", "return_kurtosis", "svar",
                                "mom_1m", "mom_3m", "mom_6m", "mom_12m",
                                "str_reversal", "turnover",
                                "beta", "betasq", "idiovol", "retvol", "maxret"]:
                        if col in row.index and pd.notna(row[col]):
                            risk_data[col] = round(float(row[col]), 6)
                    for col in ["insider_buy_ratio_6m", "insider_ceo_buy",
                                "insider_cluster_buy", "num_analysts",
                                "analyst_dispersion", "analyst_revision",
                                "nanalyst", "fgr5yr", "disp", "sfe"]:
                        if col in row.index and pd.notna(row[col]):
                            insider_data[col] = round(float(row[col]), 6)
                    for col in ["inst_ownership_change", "inst_hhi", "inst_breadth",
                                "market_cap", "log_market_cap", "turnover",
                                "sp500_member", "n_months", "rd_sale", "rd_intensity",
                                "orgcap", "herf", "dolvol", "turn", "mvel1", "age"]:
                        if col in row.index and pd.notna(row[col]):
                            inst_data[col] = round(float(row[col]), 6)
            regime = wp.get_market_regime()
            return {
                'ticker': ticker, 'prediction': prediction,
                'fundamentals': fundamentals, 'earnings': earnings_data,
                'risk': risk_data, 'insider': insider_data,
                'institutional': inst_data, 'regime': regime,
                'note': 'fallback_local_dispatch_no_advanced_engines',
            }

        elif tool_name in ("roket_fundamentals",):
            from ..ml.wrds_predictor import get_wrds_predictor
            import pandas as pd
            ticker = tool_input['ticker'].upper()
            wp = get_wrds_predictor()
            wp._ensure_loaded()
            permno = wp._resolve_permno(ticker)
            if permno is None:
                return {'ticker': ticker, 'error': f'Ticker {ticker} not found'}
            stock_data = wp._panel[wp._panel["permno"] == permno].sort_values("date")
            if len(stock_data) == 0:
                return {'ticker': ticker, 'error': f'No data for {ticker}'}
            row = stock_data.iloc[-1]
            fundamentals = {}
            for col in ["bm", "pe_inc", "pe_exi", "pe_op_basic", "pe_op_dil",
                        "ptb", "pcf", "ps", "divyield", "capei", "peg_trailing",
                        "roe", "roa", "roce", "gpm", "npm", "opmad", "opmbd",
                        "cfm", "fcf_ocf", "fcf_to_revenue",
                        "curr_ratio", "quick_ratio", "de_ratio", "debt_at",
                        "piotroski_f_score", "beneish_m_score",
                        "altman_z_score", "ohlson_o_score", "montier_c_score",
                        "revenue_growth_yoy", "rd_sale", "rd_intensity",
                        # Legacy names
                        "ep", "sp", "cfp", "dp", "gma", "opa", "operprof",
                        "cash", "cashdebt", "cashpr", "pctacc", "acc",
                        "absacc", "lev", "depr", "sgr", "agr", "rd_mve"]:
                if col in row.index and pd.notna(row[col]):
                    fundamentals[col] = round(float(row[col]), 6)
            return {'ticker': ticker, 'fundamentals': fundamentals,
                    'date': str(row["date"].date()) if pd.notna(row.get("date")) else None}

        elif tool_name in ("roket_earnings",):
            from ..ml.wrds_predictor import get_wrds_predictor
            import pandas as pd
            ticker = tool_input['ticker'].upper()
            wp = get_wrds_predictor()
            wp._ensure_loaded()
            permno = wp._resolve_permno(ticker)
            if permno is None:
                return {'ticker': ticker, 'error': f'Ticker {ticker} not found'}
            stock_data = wp._panel[wp._panel["permno"] == permno].sort_values("date")
            if len(stock_data) == 0:
                return {'ticker': ticker, 'error': f'No data for {ticker}'}
            row = stock_data.iloc[-1]
            earnings = {}
            for col in ["sue_ibes", "earnings_persistence", "earnings_smoothness",
                        "beat_miss_streak", "eps_dispersion", "eps_dispersion_trend",
                        "eps_revision_1m", "eps_revision_3m", "eps_estimate_momentum",
                        "revision_breadth", "accrual", "total_accruals",
                        "accruals_to_cash_flow", "accruals_vs_industry",
                        "roe", "roa", "roce", "gpm", "npm", "pcf",
                        # Legacy names
                        "ep", "e2p", "earnings_yield", "sue", "re", "gma",
                        "opa", "nincr", "roaq", "roavol", "operprof", "ps",
                        "cash", "cashdebt", "cashpr", "pctacc", "acc",
                        "absacc", "stdacc", "cfp", "cf2p"]:
                if col in row.index and pd.notna(row[col]):
                    earnings[col] = round(float(row[col]), 6)
            return {'ticker': ticker, 'earnings': earnings,
                    'date': str(row["date"].date()) if pd.notna(row.get("date")) else None}

        elif tool_name in ("roket_risk",):
            from ..ml.wrds_predictor import get_wrds_predictor
            import pandas as pd
            ticker = tool_input['ticker'].upper()
            wp = get_wrds_predictor()
            wp._ensure_loaded()
            permno = wp._resolve_permno(ticker)
            if permno is None:
                return {'ticker': ticker, 'error': f'Ticker {ticker} not found'}
            stock_data = wp._panel[wp._panel["permno"] == permno].sort_values("date")
            if len(stock_data) == 0:
                return {'ticker': ticker, 'error': f'No data for {ticker}'}
            row = stock_data.iloc[-1]
            risk = {}
            for col in ["beta_mkt", "beta_smb", "beta_hml", "beta_umd",
                        "beta_cma", "beta_rmw", "r_squared", "alpha",
                        "realized_vol", "idio_vol", "total_vol",
                        "down_vol", "up_vol", "vol_3m", "vol_6m", "vol_12m",
                        "return_skewness", "return_kurtosis", "svar",
                        "mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12_2",
                        "str_reversal", "ltr", "turnover",
                        "beta", "betasq", "beta_dimson", "idiovol", "retvol",
                        "std_dolvol", "std_turn", "maxret", "zerotrade", "ill",
                        "garch_vol", "coskew", "iskew", "mom_36m"]:
                if col in row.index and pd.notna(row[col]):
                    risk[col] = round(float(row[col]), 6)
            return {'ticker': ticker, 'risk': risk,
                    'date': str(row["date"].date()) if pd.notna(row.get("date")) else None}

        elif tool_name in ("roket_insider",):
            from ..ml.wrds_predictor import get_wrds_predictor
            import pandas as pd
            ticker = tool_input['ticker'].upper()
            wp = get_wrds_predictor()
            wp._ensure_loaded()
            permno = wp._resolve_permno(ticker)
            if permno is None:
                return {'ticker': ticker, 'error': f'Ticker {ticker} not found'}
            stock_data = wp._panel[wp._panel["permno"] == permno].sort_values("date")
            if len(stock_data) == 0:
                return {'ticker': ticker, 'error': f'No data for {ticker}'}
            row = stock_data.iloc[-1]
            insider = {}
            for col in ["insider_buy_ratio_6m", "insider_ceo_buy", "insider_cluster_buy",
                        "num_analysts", "num_analysts_fy1", "analyst_dispersion",
                        "analyst_revision", "eps_estimate_momentum",
                        "nanalyst", "fgr5yr", "disp", "sfe", "securedind", "convind", "ms"]:
                if col in row.index and pd.notna(row[col]):
                    insider[col] = round(float(row[col]), 6)
            return {'ticker': ticker, 'insider': insider,
                    'date': str(row["date"].date()) if pd.notna(row.get("date")) else None}

        elif tool_name in ("roket_institutional",):
            from ..ml.wrds_predictor import get_wrds_predictor
            import pandas as pd
            ticker = tool_input['ticker'].upper()
            wp = get_wrds_predictor()
            wp._ensure_loaded()
            permno = wp._resolve_permno(ticker)
            if permno is None:
                return {'ticker': ticker, 'error': f'Ticker {ticker} not found'}
            stock_data = wp._panel[wp._panel["permno"] == permno].sort_values("date")
            if len(stock_data) == 0:
                return {'ticker': ticker, 'error': f'No data for {ticker}'}
            row = stock_data.iloc[-1]
            inst = {}
            for col in ["inst_ownership_change", "inst_hhi", "inst_breadth",
                        "market_cap", "log_market_cap", "mktcap",
                        "turnover", "turnover_3m", "turnover_6m",
                        "sp500_member", "n_months",
                        "rd_sale", "rd_intensity", "rd_intensity_trend",
                        "capex_intensity", "revenue_share",
                        "orgcap", "herf", "dolvol", "turn", "std_turn",
                        "mve0", "mvel1", "age", "sp", "rd_mve"]:
                if col in row.index and pd.notna(row[col]):
                    inst[col] = round(float(row[col]), 6)
            return {'ticker': ticker, 'institutional': inst,
                    'date': str(row["date"].date()) if pd.notna(row.get("date")) else None}

        # ── Regime ──────────────────────────────────────────────────
        elif tool_name in ("roket_regime", "get_market_regime"):
            from ..ml.wrds_predictor import get_wrds_predictor
            wp = get_wrds_predictor()
            return wp.get_market_regime()

        # ── Screener ────────────────────────────────────────────────
        elif tool_name in ("roket_screener",):
            from ..ml.wrds_predictor import get_wrds_predictor
            wp = get_wrds_predictor()
            all_stocks = wp.get_top_picks(n=5000)
            filtered = all_stocks
            if tool_input.get('tier'):
                filtered = [s for s in filtered if s.get('tier') == tool_input['tier']]
            if tool_input.get('signal'):
                sig = tool_input['signal'].upper()
                filtered = [s for s in filtered if s.get('signal') == sig]
            if tool_input.get('min_decile'):
                filtered = [s for s in filtered if s.get('decile', 0) >= tool_input['min_decile']]
            limit = tool_input.get('limit', 20)
            return {'count': len(filtered[:limit]), 'total_matched': len(filtered),
                    'stocks': filtered[:limit]}

        # ── Universe ────────────────────────────────────────────────
        elif tool_name in ("roket_universe",):
            from ..ml.wrds_predictor import get_wrds_predictor
            wp = get_wrds_predictor()
            tier = tool_input.get('tier')
            limit = tool_input.get('limit', 50)
            if tier:
                results = wp.get_tier_predictions(tier)
            else:
                results = wp.get_top_picks(n=limit)
            return {'count': len(results[:limit]), 'stocks': results[:limit]}

        # ── News (StockNews API) ────────────────────────────────────
        elif tool_name in ("roket_news",):
            import asyncio
            ticker = tool_input.get('ticker', '').upper().strip()
            if not ticker:
                return {'error': 'Ticker symbol required'}
            from ..news.client import StockNewsClient
            client = StockNewsClient()
            try:
                summary = asyncio.get_event_loop().run_until_complete(
                    client.get_news_summary(ticker)
                )
            except RuntimeError:
                # Already in async context — use await
                loop = asyncio.new_event_loop()
                try:
                    summary = loop.run_until_complete(client.get_news_summary(ticker))
                finally:
                    loop.close()
            finally:
                try:
                    asyncio.get_event_loop().run_until_complete(client.close())
                except Exception:
                    pass
            return summary

        # ── Real-time Snapshot (DataAggregator) ─────────────────────
        elif tool_name in ("roket_snapshot",):
            import asyncio
            ticker = tool_input.get('ticker', '').upper().strip()
            if not ticker:
                return {'error': 'Ticker symbol required'}
            from ..data.aggregator import DataAggregator
            agg = DataAggregator()

            async def _get_snap():
                await agg.initialize()
                return await agg.get_snapshot(ticker)

            try:
                snapshot = asyncio.get_event_loop().run_until_complete(_get_snap())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    snapshot = loop.run_until_complete(_get_snap())
                finally:
                    loop.close()
            return snapshot.to_dict()

        # ── SEC Quality (SEC EDGAR XBRL) ────────────────────────────
        elif tool_name in ("roket_sec_quality",):
            ticker = tool_input.get('ticker', '').upper().strip()
            if not ticker:
                return {'error': 'Ticker symbol required'}
            from ..data.sec_edgar import SECEdgarXBRL
            edgar = SECEdgarXBRL()
            quality = edgar.get_quality_score(ticker)
            ratios = edgar.get_fundamental_ratios(ticker)
            if quality is None:
                return {'ticker': ticker, 'error': f'No SEC filings found for {ticker}'}
            return {
                'ticker': ticker,
                'quality_score': quality.get('score'),
                'grade': quality.get('grade'),
                'components': quality.get('components', {}),
                'fundamental_ratios': ratios if ratios else {},
                'source': 'SEC EDGAR XBRL (live)',
            }

        # ── Macro Environment (FRED) ───────────────────────────────
        elif tool_name in ("roket_macro",):
            from ..data.fred_macro import FREDMacroData
            fred = FREDMacroData()
            current = fred.get_current()
            if current is None:
                return {'status': 'unavailable', 'reason': 'FRED_API_KEY not set'}
            return {'status': 'ok', **current}

        # ── Lambda Decision Engine ──────────────────────────────────
        elif tool_name in ("roket_lambda",):
            ticker = tool_input.get('ticker', '').upper().strip()
            if not ticker:
                return {'error': 'Ticker symbol required'}
            from ..lambda_client import NubleLambdaClient
            client = NubleLambdaClient()
            analysis = client.get_analysis(ticker)
            if analysis is None:
                return {'ticker': ticker, 'error': f'Lambda API returned no data for {ticker}'}
            return {
                'ticker': analysis.symbol,
                'action': analysis.action,
                'direction': analysis.direction,
                'strength': analysis.strength,
                'score': analysis.score,
                'confidence': analysis.confidence,
                'should_trade': analysis.should_trade,
                'veto': analysis.veto,
                'veto_reason': analysis.veto_reason,
                'current_price': analysis.current_price,
                'regime': analysis.regime.value if hasattr(analysis.regime, 'value') else str(analysis.regime),
                'technicals': {
                    'rsi': analysis.technicals.rsi,
                    'macd': analysis.technicals.macd,
                    'trend_state': analysis.technicals.trend_state,
                    'technical_score': analysis.technicals.technical_score,
                },
                'intelligence': {
                    'sentiment_score': analysis.intelligence.sentiment_score,
                    'sentiment_label': analysis.intelligence.sentiment_label,
                    'news_count_7d': analysis.intelligence.news_count_7d,
                    'headlines': analysis.intelligence.headlines[:5],
                    'vix_value': analysis.intelligence.vix_value,
                },
                'luxalgo': {
                    'weekly': analysis.luxalgo_weekly_action,
                    'daily': analysis.luxalgo_daily_action,
                    'h4': analysis.luxalgo_h4_action,
                    'aligned': analysis.luxalgo_aligned,
                },
                'analysis_summary': analysis.analysis_summary,
            }

        # ── Compare (side-by-side) ──────────────────────────────────
        elif tool_name in ("roket_compare",):
            tickers_raw = tool_input.get('tickers', '')
            if isinstance(tickers_raw, list):
                ticker_list = [t.upper().strip() for t in tickers_raw if t.strip()]
            else:
                ticker_list = [t.upper().strip() for t in str(tickers_raw).split(',') if t.strip()]
            if len(ticker_list) < 2:
                return {'error': 'Provide at least 2 tickers'}
            if len(ticker_list) > 5:
                return {'error': 'Maximum 5 tickers'}
            from ..ml.live_predictor import get_live_predictor
            from ..ml.wrds_predictor import get_wrds_predictor
            import pandas as pd
            lp = get_live_predictor()
            wp = get_wrds_predictor()
            wp._ensure_loaded()
            comparisons = []
            for ticker in ticker_list:
                entry = {'ticker': ticker}
                try:
                    pred = lp.predict(ticker)
                except Exception:
                    try:
                        pred = wp.predict(ticker)
                    except Exception:
                        pred = {'error': f'Not found: {ticker}'}
                entry['prediction'] = pred
                permno = wp._resolve_permno(ticker)
                if permno is not None and wp._panel is not None:
                    stock_data = wp._panel[wp._panel["permno"] == permno].sort_values("date")
                    if len(stock_data) > 0:
                        row = stock_data.iloc[-1]
                        fundamentals = {}
                        for col in ["ep", "bm", "roe", "roa", "gma", "sgr", "agr", "lev"]:
                            if col in row.index and pd.notna(row[col]):
                                fundamentals[col] = round(float(row[col]), 6)
                        entry['fundamentals'] = fundamentals
                        risk = {}
                        for col in ["beta", "idiovol", "retvol", "mom_1m", "mom_6m", "mom_12m"]:
                            if col in row.index and pd.notna(row[col]):
                                risk[col] = round(float(row[col]), 6)
                        entry['risk'] = risk
                comparisons.append(entry)
            regime = wp.get_market_regime()
            return {'count': len(comparisons), 'tickers': ticker_list,
                    'comparisons': comparisons, 'regime': regime}

        # ── Position Sizing (Kelly Criterion) ───────────────────────
        elif tool_name in ("roket_position_size",):
            ticker = tool_input.get('ticker', '').upper().strip()
            if not ticker:
                return {'error': 'Ticker symbol required'}
            portfolio_value = float(tool_input.get('portfolio_value', 100000))
            risk_per_trade = float(tool_input.get('risk_per_trade', 0.02))
            from ..ml.live_predictor import get_live_predictor
            from ..ml.wrds_predictor import get_wrds_predictor
            import pandas as pd
            lp = get_live_predictor()
            wp = get_wrds_predictor()
            wp._ensure_loaded()
            try:
                prediction = lp.predict(ticker)
            except Exception:
                prediction = wp.predict(ticker)
            if prediction.get("error") and prediction.get("confidence", 1) == 0.0:
                return {'ticker': ticker, 'error': prediction['error']}
            current_price = prediction.get("live_price", prediction.get("current_price", 0))
            # ATR from panel + fallback price
            atr = None
            permno = wp._resolve_permno(ticker)
            if permno is not None and wp._panel is not None:
                stock_data = wp._panel[wp._panel["permno"] == permno].sort_values("date")
                if len(stock_data) > 0:
                    row = stock_data.iloc[-1]
                    # Fallback price from panel
                    if current_price == 0:
                        for pc in ["prc", "close", "last_price"]:
                            if pc in row.index and pd.notna(row[pc]):
                                current_price = abs(float(row[pc]))
                                break
                        if current_price == 0 and "mvel1" in row.index and pd.notna(row["mvel1"]):
                            current_price = float(row["mvel1"])
                    if "atr_14" in row.index and pd.notna(row["atr_14"]):
                        atr = float(row["atr_14"])
                    elif "retvol" in row.index and pd.notna(row["retvol"]):
                        atr = float(row["retvol"]) * current_price if current_price > 0 else None
            # Try Polygon for live price if still 0
            if current_price == 0:
                try:
                    import requests as req
                    polygon_key = os.environ.get("POLYGON_API_KEY", "JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY")
                    snap_url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}?apiKey={polygon_key}"
                    snap_r = req.get(snap_url, timeout=5)
                    if snap_r.status_code == 200:
                        snap_data = snap_r.json()
                        day_data = snap_data.get("ticker", {}).get("day", {})
                        prev_data = snap_data.get("ticker", {}).get("prevDay", {})
                        current_price = day_data.get("c", 0) or prev_data.get("c", 0)
                except Exception:
                    pass
            if atr is None or atr <= 0:
                retvol = prediction.get("retvol", 0.02)
                atr = retvol * current_price * 1.5 if current_price > 0 else 1.0
            signal = prediction.get("signal", "HOLD")
            confidence = prediction.get("confidence", 0.5)
            signal_win_rates = {"STRONG_BUY": 0.60, "BUY": 0.55, "HOLD": 0.45, "SELL": 0.40, "STRONG_SELL": 0.35}
            win_rate = signal_win_rates.get(signal, 0.45) * (0.8 + 0.4 * confidence)
            win_rate = min(max(win_rate, 0.30), 0.70)
            q = 1 - win_rate
            kelly_raw = max(0, (win_rate * 1.5 - q) / 1.5)
            kelly_half = kelly_raw * 0.5
            max_risk_dollars = portfolio_value * risk_per_trade
            stop_distance = atr * 2.0
            shares_from_risk = int(max_risk_dollars / stop_distance) if stop_distance > 0 else 0
            kelly_dollars = portfolio_value * kelly_half
            max_pos_dollars = portfolio_value * 0.10
            shares_from_kelly = int(kelly_dollars / current_price) if current_price > 0 else 0
            shares_from_cap = int(max_pos_dollars / current_price) if current_price > 0 else 0
            recommended_shares = max(0, min(shares_from_risk, shares_from_kelly, shares_from_cap))
            dollar_amount = recommended_shares * current_price
            is_long = signal in ("STRONG_BUY", "BUY", "HOLD")
            if is_long:
                stop_loss = current_price - stop_distance
                tp1 = current_price + stop_distance
                tp2 = current_price + stop_distance * 2
                tp3 = current_price + stop_distance * 3
            else:
                stop_loss = current_price + stop_distance
                tp1 = current_price - stop_distance
                tp2 = current_price - stop_distance * 2
                tp3 = current_price - stop_distance * 3
            return {
                'ticker': ticker, 'current_price': round(current_price, 2),
                'signal': signal, 'confidence': round(confidence, 4),
                'position_sizing': {
                    'recommended_shares': recommended_shares,
                    'dollar_amount': round(dollar_amount, 2),
                    'position_pct': round(dollar_amount / portfolio_value * 100, 2) if portfolio_value > 0 else 0,
                    'kelly_fraction': round(kelly_half, 4), 'kelly_raw': round(kelly_raw, 4),
                },
                'risk_management': {
                    'stop_loss_price': round(stop_loss, 2),
                    'take_profit_1': round(tp1, 2), 'take_profit_2': round(tp2, 2), 'take_profit_3': round(tp3, 2),
                    'risk_reward_ratio': 1.5,
                    'max_loss_dollars': round(recommended_shares * stop_distance, 2),
                    'atr_14': round(atr, 4),
                },
                'parameters': {
                    'portfolio_value': portfolio_value, 'risk_per_trade': risk_per_trade,
                    'win_rate_estimate': round(win_rate, 4), 'direction': 'LONG' if is_long else 'SHORT',
                },
            }

        # ── Legacy tools (backward compat) ──────────────────────────
        elif tool_name == "get_batch_predictions":
            from ..ml.live_predictor import get_live_predictor
            lp = get_live_predictor()
            return {'predictions': lp.predict_batch(tool_input['tickers'])}

        elif tool_name == "get_top_picks":
            from ..ml.live_predictor import get_live_predictor
            lp = get_live_predictor()
            n = tool_input.get('n', 10)
            tier = tool_input.get('tier')
            picks = lp.get_live_top_picks(n=n, tier=tier)
            return {'picks': picks, 'count': len(picks)}

        elif tool_name == "analyze_portfolio":
            from ..ml.live_predictor import get_live_predictor
            lp = get_live_predictor()
            holdings = tool_input['holdings']
            results = []
            for ticker in holdings:
                try:
                    pred = lp.predict(ticker.upper())
                    pred['weight'] = holdings[ticker]
                    results.append(pred)
                except Exception as e:
                    results.append({'ticker': ticker, 'error': str(e)})
            from ..ml.wrds_predictor import get_wrds_predictor
            regime = get_wrds_predictor().get_market_regime()
            return {'holdings': results, 'regime': regime}

        elif tool_name == "get_tier_info":
            from ..ml.wrds_predictor import get_wrds_predictor, TIER_CONFIG
            wrds = get_wrds_predictor()
            wrds._ensure_loaded()
            ticker = tool_input['ticker'].upper()
            pred = wrds.predict(ticker)
            tier = pred.get('tier', 'small')
            tc = TIER_CONFIG.get(tier, {})
            return {
                'ticker': ticker, 'tier': tier,
                'label': tc.get('label', ''), 'ic': tc.get('ic', 0),
                'weight': tc.get('weight', 0), 'strategy': tc.get('strategy', ''),
                'market_cap_millions': pred.get('market_cap_millions', 0),
            }

        elif tool_name == "get_system_status":
            from ..data.data_service import get_data_service
            ds = get_data_service()
            return ds.get_status()

        else:
            return {'error': f'Unknown tool: {tool_name}'}

    except Exception as e:
        logger.error(f"Tool dispatch failed for {tool_name}: {e}", exc_info=True)
        return {'error': str(e)}


# ── Tool definitions (same as intelligence.py/tools-schema) ─────────────

def _get_tools_for_claude() -> List[Dict[str, Any]]:
    """
    Get all 17 ROKET tools in Anthropic API format.
    These are the canonical tool definitions sent to Claude.
    """
    return [
        {
            "name": "roket_predict",
            "description": (
                "Get an institutional-grade ML prediction for a single stock. "
                "Returns composite score (70% fundamental + 30% timing), signal "
                "(strong_buy / buy / hold / sell / strong_sell), confidence, tier, "
                "top feature drivers. The model is a multi-tier LightGBM ensemble "
                "trained on 3.76M observations of 539 academic features from WRDS/GKX. "
                "Use for any question about a specific stock's outlook."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL, TSLA, SPY)"}
                },
                "required": ["ticker"]
            },
        },
        {
            "name": "roket_analyze",
            "description": (
                "Full deep-dive analysis on a single stock. Returns prediction + "
                "fundamentals + earnings + risk + insider + institutional + regime "
                "in one call. Use when the user says 'analyze AAPL' or 'tell me everything about TSLA'. "
                "This is the most comprehensive WRDS-based tool — prefer it for detailed single-stock questions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"]
            },
        },
        {
            "name": "roket_fundamentals",
            "description": (
                "Get fundamental valuation factors for a stock from the WRDS/GKX panel. "
                "Returns metrics like E/P, B/M, S/P, CF/P, D/P, ROE, ROA, gross margin, "
                "operating profitability, leverage, depreciation, sales growth, asset growth, "
                "R&D intensity. Use for valuation questions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"]
            },
        },
        {
            "name": "roket_earnings",
            "description": (
                "Get earnings quality and surprise metrics for a stock. "
                "Returns SUE (standardized unexpected earnings), earnings yield, "
                "ROE, ROA, accruals, cash flow ratios, analyst dispersion, "
                "forecast growth. Use for earnings-related questions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"]
            },
        },
        {
            "name": "roket_risk",
            "description": (
                "Get risk and volatility metrics for a stock. Returns beta, "
                "idiosyncratic volatility, return volatility, max daily return, "
                "momentum factors (1m, 6m, 12m, 36m), realized vol, GARCH vol, "
                "coskewness, illiquidity. Use for risk assessment questions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"]
            },
        },
        {
            "name": "roket_insider",
            "description": (
                "Get insider and analyst sentiment metrics for a stock. "
                "Returns analyst count, forecast growth, dispersion, "
                "standardized forecast error, secured/convertible indicators. "
                "Use for questions about insider activity or analyst consensus."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"]
            },
        },
        {
            "name": "roket_institutional",
            "description": (
                "Get institutional ownership and market structure metrics. "
                "Returns organizational capital, Herfindahl index, dollar volume, "
                "turnover, market cap, firm age, S&P membership, R&D intensity. "
                "Use for questions about institutional flows or market microstructure."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"]
            },
        },
        {
            "name": "roket_regime",
            "description": (
                "Detect current market regime using a Hidden Markov Model trained "
                "on 420 months of macro data. Returns regime state (bull/neutral/crisis), "
                "state probabilities, transition matrix, VIX-based exposure adjustment. "
                "Use for any question about overall market conditions or macro environment."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "roket_screener",
            "description": (
                "Screen the 20,723-ticker universe with filters. "
                "Filter by tier (mega/large/mid/small), signal (strong_buy/buy/hold/sell/strong_sell), "
                "minimum decile ranking. Returns ranked list of matching stocks with predictions. "
                "Use when user asks to screen or filter stocks."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "tier": {
                        "type": "string",
                        "enum": ["mega", "large", "mid", "small"],
                        "description": "Filter by market cap tier"
                    },
                    "signal": {
                        "type": "string",
                        "enum": ["strong_buy", "buy", "hold", "sell", "strong_sell"],
                        "description": "Filter by signal"
                    },
                    "min_decile": {
                        "type": "integer",
                        "description": "Minimum decile ranking (1-10, 10=top)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 20)"
                    }
                },
            },
        },
        {
            "name": "roket_universe",
            "description": (
                "Browse the ranked stock universe. Returns top stocks ranked by "
                "ML composite score. Can filter by tier. Use when user asks 'what are "
                "your top picks' or 'show me the best stocks' or 'what's in the universe'."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "tier": {
                        "type": "string",
                        "enum": ["mega", "large", "mid", "small"],
                        "description": "Filter by tier"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 50)"
                    }
                },
            },
        },
        # ── NEW TOOLS (Full codebase leverage) ───────────────────
        {
            "name": "roket_news",
            "description": (
                "Get REAL-TIME news and sentiment for a stock from the StockNews API. "
                "Returns today's news articles, 7-day daily sentiment scores (-1.5 to +1.5), "
                "and trending headlines. Use when the user asks about news, sentiment, "
                "what's happening with a stock, or recent catalysts."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"]
            },
        },
        {
            "name": "roket_snapshot",
            "description": (
                "Get a LIVE real-time market snapshot from Polygon.io. "
                "Returns current price, bid/ask, volume, technicals (SMA 20/50/200, RSI 14, MACD, "
                "Bollinger Bands, ATR), options flow (put/call ratio, unusual activity, IV), "
                "news sentiment, market regime, and LuxAlgo multi-timeframe signals. "
                "Use for real-time market data, technical analysis, or options flow questions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"]
            },
        },
        {
            "name": "roket_sec_quality",
            "description": (
                "Get LIVE fundamental quality score from SEC EDGAR XBRL filings. "
                "Returns a composite quality score (0-100), letter grade (A/B/C/D/F), "
                "and 40 GKX fundamental ratios computed from the company's actual SEC filings "
                "(book-to-market, E/P, CF/P, ROE, ROA, debt/equity, sales growth, accruals, etc.). "
                "This is LIVE data directly from SEC EDGAR, not the historical WRDS panel. "
                "Use for questions about a company's fundamental quality or financial health."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"]
            },
        },
        {
            "name": "roket_macro",
            "description": (
                "Get the current macroeconomic environment from FRED. "
                "Returns Treasury yields (3M, 10Y), credit spreads (BBB, AAA), "
                "breakeven inflation, industrial production, unemployment, Fed Funds rate, "
                "plus derived regime indicators: yield curve state (inverted/flat/normal/steep), "
                "credit cycle (tightening/easing/stable), monetary policy stance (hawkish/dovish/neutral). "
                "Use for macro environment questions, yield curve analysis, or when placing stock analysis in macro context."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "roket_lambda",
            "description": (
                "Get a LIVE composite decision from the production Lambda Decision Engine. "
                "Aggregates data from Polygon.io (price/technicals), StockNews API (sentiment/headlines), "
                "CryptoNews API (for crypto), LuxAlgo multi-timeframe signals, and HMM regime detection "
                "into a single action (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL) with score (0-100). "
                "This is the MOST POWERFUL real-time tool — it combines ALL live data sources. "
                "Use for the most authoritative live trading decision on any symbol."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock or crypto ticker symbol"}
                },
                "required": ["ticker"]
            },
        },
        {
            "name": "roket_compare",
            "description": (
                "Side-by-side comparison of 2-5 stocks. Returns ML prediction, "
                "key fundamentals (E/P, B/M, ROE, ROA, growth, leverage), "
                "and risk metrics (beta, volatility, momentum) for each stock, "
                "plus the current market regime. Use when user asks to compare stocks, "
                "says 'AAPL vs MSFT', or wants to choose between options."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "string",
                        "description": "Comma-separated ticker symbols (e.g. 'AAPL,MSFT,GOOGL')"
                    }
                },
                "required": ["tickers"]
            },
        },
        {
            "name": "roket_position_size",
            "description": (
                "Calculate optimal position size using modified Kelly Criterion. "
                "Returns recommended shares, dollar amount, stop-loss price, "
                "three take-profit levels (1:1, 2:1, 3:1 R:R), risk/reward ratio, "
                "and max loss in dollars. Uses the ML prediction confidence to estimate "
                "win rate, ATR for stop distance, with 2% max risk and 10% max position caps. "
                "Use when user asks 'how much should I buy', 'what's my position size', "
                "or 'where do I put my stop loss'."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "portfolio_value": {
                        "type": "number",
                        "description": "Total portfolio value in dollars (default 100000)"
                    },
                    "risk_per_trade": {
                        "type": "number",
                        "description": "Max risk per trade as fraction (default 0.02 = 2%)"
                    }
                },
                "required": ["ticker"]
            },
        },
    ]


# ── Server-side Claude ↔ Tools loop ─────────────────────────────────────

@router.post("/chat-with-tools", response_model=ChatWithToolsResponse)
async def chat_with_tools(request: ChatWithToolsRequest):
    """
    Full Claude ↔ Tools loop server-side.

    1. Sends user message to Claude Opus with tool definitions
    2. If Claude returns tool_use, executes the tool and feeds result back
    3. Repeats until Claude returns a text response (max N rounds)
    4. Returns the final synthesized answer

    This is Pattern B — your frontend just sends the message and gets back
    a rich, data-backed answer. No tool handling needed in the frontend.
    """
    t0 = time.time()

    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not set")

    try:
        import anthropic
    except ImportError:
        raise HTTPException(status_code=503, detail="anthropic package not installed")

    client = anthropic.Anthropic(api_key=api_key)
    tools = _get_tools_for_claude()
    tools_used = []
    tool_results_log = []

    system_prompt = (
        "You are ROKET — the Robust Quantitative Knowledge Engine for Trading.\n\n"
        "You are the conversational interface to NUBLE's institutional-grade ML system. "
        "You have 17 tools at your disposal that give you access to:\n\n"
        "SYSTEM A — Historical ML Intelligence (WRDS/GKX):\n"
        "• A multi-tier LightGBM ensemble trained on 3.76M observations (539 GKX features)\n"
        "• 20,723-ticker universe with per-tier models (mega/large/mid/small)\n"
        "• HMM-based market regime detection (bull/neutral/crisis, 420 months macro)\n"
        "• Deep fundamental, earnings, risk, insider, and institutional metrics\n\n"
        "SYSTEM B — Real-Time Live Intelligence:\n"
        "• StockNews API: Live news articles, sentiment scores (-1.5 to +1.5), trending headlines\n"
        "• Polygon.io: Real-time quotes, technicals (SMA, RSI, MACD, Bollinger, ATR), options flow\n"
        "• SEC EDGAR XBRL: Live fundamental ratios, quality scores (A-F) from actual filings\n"
        "• FRED: Macro data — yield curve, credit spreads, monetary policy stance\n"
        "• Lambda Decision Engine: Production composite decisions aggregating ALL live sources\n"
        "• Kelly Criterion: Position sizing with stop-loss and take-profit levels\n\n"
        "TOOL SELECTION RULES:\n"
        "1. ALWAYS use tools to back up your analysis. Never guess when you can look it up.\n"
        "2. For single stock questions → call roket_predict or roket_analyze.\n"
        "3. For market conditions → call roket_regime and/or roket_macro.\n"
        "4. For 'best stocks' / 'top picks' → call roket_universe or roket_screener.\n"
        "5. For deep dives → call roket_analyze (WRDS data) AND roket_lambda (live data).\n"
        "6. For news/sentiment/catalysts → call roket_news.\n"
        "7. For real-time price/technicals/options → call roket_snapshot.\n"
        "8. For fundamental quality from SEC filings → call roket_sec_quality.\n"
        "9. For comparing stocks → call roket_compare.\n"
        "10. For position sizing / stop loss / risk management → call roket_position_size.\n"
        "11. For the most authoritative LIVE trading decision → call roket_lambda.\n"
        "12. COMBINE System A + System B tools for the most complete analysis.\n\n"
        "RESPONSE FORMAT:\n"
        "• Lead with the ML signal and key metrics\n"
        "• Combine historical (WRDS) and live (Lambda/news/technicals) perspectives\n"
        "• Place analysis in regime and macro context\n"
        "• Give clear, actionable recommendation with caveats\n"
        "• Include position sizing guidance when appropriate\n"
        "• Be confident but honest about limitations (feature coverage, data recency)\n\n"
        "You speak with institutional authority. Your data comes from WRDS academic-grade "
        "sources, SEC EDGAR filings, and real-time market data feeds."
    )

    messages = [{"role": "user", "content": request.message}]

    for round_num in range(request.max_tool_rounds + 1):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            # Process all tool calls in this response
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            # Add assistant response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool and build results
            tool_result_content = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block.name
                tool_input = tool_block.input
                tools_used.append(tool_name)

                logger.info(f"Tool call [{round_num}]: {tool_name}({json.dumps(tool_input)[:100]})")

                result = _dispatch_tool(tool_name, tool_input)
                tool_results_log.append({
                    'tool': tool_name,
                    'input': tool_input,
                    'result_preview': str(result)[:500],
                })

                tool_result_content.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": json.dumps(result, default=str),
                })

            messages.append({"role": "user", "content": tool_result_content})

        else:
            # Claude returned a text response — we're done
            final_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    final_text += block.text

            return ChatWithToolsResponse(
                message=final_text,
                tools_used=tools_used,
                tool_results=tool_results_log,
                execution_time_seconds=round(time.time() - t0, 2),
            )

    # Max rounds reached — return whatever we have
    return ChatWithToolsResponse(
        message="Analysis incomplete — max tool rounds reached. Please try a more specific question.",
        tools_used=tools_used,
        tool_results=tool_results_log,
        execution_time_seconds=round(time.time() - t0, 2),
    )
