"""
ROKET API — NUBLE REST API for Frontend Integration
=====================================================
Lightweight FastAPI server exposing all NUBLE intelligence via REST.
Designed for the frontend (Prompt 3) to consume.

Endpoints:
    GET  /api/health              → System health + data freshness
    GET  /api/predict/{ticker}    → ML prediction (live Polygon → WRDS fallback)
    GET  /api/universe            → All stocks with predictions
    GET  /api/regime              → Current macro regime (HMM)
    GET  /api/fundamentals/{t}    → GKX fundamental features
    GET  /api/earnings/{ticker}   → Earnings-related features
    GET  /api/risk/{ticker}       → Risk profile (volatility, beta, drawdown)
    GET  /api/insider/{ticker}    → Insider activity signals
    GET  /api/institutional/{t}   → Institutional flow signals
    POST /api/analyze             → Batch portfolio analysis
    POST /api/screener            → Custom screening with filters
    GET  /api/top-picks           → Top N stock picks
    GET  /api/tier/{tier}         → Tier-specific predictions
    GET  /api/model-info          → Model metadata

Mounting:
    This router is automatically included in the main server.py app.
    Or run standalone: uvicorn nuble.api.roket:app --port 8001
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn[standard]")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# Lazy singletons (avoid heavy imports at module level)
# ─────────────────────────────────────────────────────────────

_ds = None
_lp = None
_wp = None
_regime = None


def _get_ds():
    global _ds
    if _ds is None:
        from ..data.data_service import get_data_service
        _ds = get_data_service()
    return _ds


def _get_lp():
    global _lp
    if _lp is None:
        from ..ml.live_predictor import get_live_predictor
        _lp = get_live_predictor()
    return _lp


def _get_wp():
    global _wp
    if _wp is None:
        from ..ml.wrds_predictor import get_wrds_predictor
        _wp = get_wrds_predictor()
    return _wp


def _get_regime():
    global _regime
    if _regime is None:
        try:
            from ..ml.hmm_regime import get_regime_detector
            _regime = get_regime_detector()
        except Exception as e:
            logger.warning(f"HMM regime detector unavailable: {e}")
            _regime = False
    return _regime if _regime is not False else None


# ─────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────

class ScreenerRequest(BaseModel):
    min_score: Optional[float] = Field(None, description="Minimum raw score")
    max_score: Optional[float] = Field(None, description="Maximum raw score")
    tiers: Optional[List[str]] = Field(None, description="Filter by tiers: mega, large, mid, small")
    signals: Optional[List[str]] = Field(None, description="Filter by signals: STRONG_BUY, BUY, etc.")
    min_market_cap: Optional[float] = Field(None, description="Minimum market cap in millions")
    max_market_cap: Optional[float] = Field(None, description="Maximum market cap in millions")
    limit: int = Field(50, ge=1, le=500, description="Max results")
    sort_by: str = Field("raw_score", description="Sort field")
    sort_desc: bool = Field(True, description="Descending sort")


class AnalyzeRequest(BaseModel):
    holdings: Dict[str, float] = Field(
        ..., description="Ticker → weight map, e.g. {'AAPL': 0.25, 'GOOGL': 0.20}"
    )


# ─────────────────────────────────────────────────────────────
# FastAPI App (standalone or mounted as sub-app)
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="NUBLE ROKET API",
    description="Real-time ML intelligence for equities. "
                "Live Polygon data → production LightGBM → regime-aware signals.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Mount LuxAlgo webhook router (TradingView → signal store)
# ─────────────────────────────────────────────────────────────
try:
    from .luxalgo_api import create_luxalgo_router
    _luxalgo_router = create_luxalgo_router()
    app.include_router(_luxalgo_router)
    logger.info("✅ LuxAlgo webhook router mounted (POST /webhooks/luxalgo, GET /signals/*)")
except Exception as e:
    logger.warning(f"LuxAlgo webhook router not mounted: {e}")

# ─────────────────────────────────────────────────────────────
# Mount Tool Bridge router (Anthropic/Bedrock format converter)
# ─────────────────────────────────────────────────────────────
try:
    from .nuble_tool_bridge import router as tool_bridge_router
    app.include_router(tool_bridge_router)
    logger.info("✅ Tool Bridge router mounted (GET /api/tools/, POST /api/tools/execute)")
except Exception as e:
    logger.warning(f"Tool Bridge router not mounted: {e}")


# Also expose as an APIRouter for mounting into the main server
from fastapi import APIRouter
router = APIRouter(prefix="/api/roket", tags=["ROKET"])


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/api/health")
@router.get("/health")
async def health():
    """System health + data freshness."""
    t0 = time.time()
    ds = _get_ds()
    status = ds.get_status()

    # Data freshness
    freshness = {}
    try:
        import pandas as pd
        panel_path = ds.data_dir / "gkx_panel.parquet"
        if panel_path.exists():
            dates = pd.read_parquet(panel_path, columns=["date"])
            max_date = pd.to_datetime(dates["date"]).max()
            staleness = (datetime.now() - max_date.to_pydatetime().replace(tzinfo=None)).days
            freshness["gkx_panel"] = {
                "latest_date": str(max_date.date()),
                "staleness_days": staleness,
                "size_mb": round(panel_path.stat().st_size / 1e6, 1),
            }
    except Exception:
        pass

    # Component readiness
    components = {}

    # LightGBM availability (root cause of most failures)
    try:
        import lightgbm as lgb
        components["lightgbm"] = {"available": True, "version": lgb.__version__}
    except ImportError as e:
        components["lightgbm"] = {"available": False, "error": str(e)}

    try:
        wp = _get_wp()
        wp._ensure_loaded()
        components["wrds_predictor"] = {
            "ready": wp.is_ready,
            "tickers": len(wp._ticker_to_permno),
            "models": list(wp._models.keys()),
        }
    except Exception as e:
        components["wrds_predictor"] = {"ready": False, "error": str(e)}

    try:
        lp = _get_lp()
        lp._load_production_models()
        components["live_predictor"] = {
            "ready": True,
            "polygon_engine": lp._polygon_engine is not None and lp._polygon_engine is not False,
            "production_models": len(lp._production_models),
            "production_tiers": list(lp._production_models.keys()),
        }
    except Exception as e:
        components["live_predictor"] = {"ready": False, "error": str(e)}

    try:
        det = _get_regime()
        if det is not None:
            if not det._ready:
                det.train()
            components["hmm_regime"] = {
                "ready": det._ready,
            }
        else:
            components["hmm_regime"] = {"ready": False, "error": "detector unavailable"}
    except Exception as e:
        components["hmm_regime"] = {"ready": False, "error": str(e)}

    return {
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "uptime_check_ms": round((time.time() - t0) * 1000, 1),
        "data": status,
        "data_freshness": freshness,
        "components": components,
    }


@app.get("/api/predict/{ticker}")
@router.get("/predict/{ticker}")
async def predict(ticker: str, source: str = Query("auto", enum=["auto", "live", "wrds"])):
    """
    ML prediction for a ticker.
    source=auto → tries live Polygon first, falls back to WRDS historical.
    source=live → Polygon only.
    source=wrds → WRDS historical only.
    """
    t0 = time.time()
    ticker = ticker.upper().strip()

    # Validate ticker is not empty
    if not ticker or not ticker.strip():
        raise HTTPException(status_code=400, detail="Ticker symbol required")

    try:
        if source == "wrds":
            result = _get_wp().predict(ticker)
        elif source == "live":
            result = _get_lp().predict(ticker)
        else:
            # Auto: try live first, fall back to WRDS
            try:
                result = _get_lp().predict(ticker)
            except Exception:
                result = _get_wp().predict(ticker)

        # If the predictor returned an error (e.g. ticker not found), surface as 404
        if result.get("error") and result.get("confidence", 1) == 0.0:
            raise HTTPException(status_code=404, detail=result["error"])

        result["execution_time_ms"] = round((time.time() - t0) * 1000, 1)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/universe")
@router.get("/universe")
async def universe(
    tier: Optional[str] = Query(None, enum=["mega", "large", "mid", "small"]),
    limit: int = Query(100, ge=1, le=5000),
):
    """All stocks with predictions, optionally filtered by tier."""
    t0 = time.time()
    try:
        wp = _get_wp()
        if tier:
            results = wp.get_tier_predictions(tier)
        else:
            results = wp.get_top_picks(n=limit)

        return {
            "count": len(results),
            "tier_filter": tier,
            "stocks": results[:limit],
            "execution_time_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/regime")
@router.get("/regime")
async def regime():
    """Current macro regime (HMM + rule-based fallback)."""
    t0 = time.time()
    try:
        wp = _get_wp()
        result = wp.get_market_regime()
        result["execution_time_ms"] = round((time.time() - t0) * 1000, 1)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fundamentals/{ticker}")
@router.get("/fundamentals/{ticker}")
async def fundamentals(ticker: str):
    """GKX fundamental features for a ticker from the latest panel snapshot."""
    t0 = time.time()
    ticker = ticker.upper().strip()

    try:
        wp = _get_wp()
        wp._ensure_loaded()

        if not wp._ready or wp._panel is None:
            raise HTTPException(status_code=503, detail="WRDS data not loaded")

        permno = wp._resolve_permno(ticker)
        if permno is None:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")

        stock_data = wp._panel[wp._panel["permno"] == permno].sort_values("date")
        if len(stock_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        row = stock_data.iloc[-1]
        import pandas as pd

        # Extract all non-null features
        features = {}
        for col in row.index:
            val = row[col]
            if pd.notna(val):
                try:
                    features[col] = round(float(val), 6)
                except (ValueError, TypeError):
                    features[col] = str(val)

        return {
            "ticker": ticker,
            "permno": int(permno),
            "date": str(row["date"].date()) if pd.notna(row.get("date")) else None,
            "feature_count": len(features),
            "features": features,
            "execution_time_ms": round((time.time() - t0) * 1000, 1),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/earnings/{ticker}")
@router.get("/earnings/{ticker}")
async def earnings(ticker: str):
    """Earnings-related features from GKX panel."""
    t0 = time.time()
    ticker = ticker.upper().strip()

    try:
        wp = _get_wp()
        wp._ensure_loaded()
        if not wp._ready:
            raise HTTPException(status_code=503, detail="WRDS data not loaded")

        permno = wp._resolve_permno(ticker)
        if permno is None:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")

        stock_data = wp._panel[wp._panel["permno"] == permno].sort_values("date")
        if len(stock_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        row = stock_data.iloc[-1]
        import pandas as pd

        earnings_cols = [
            # Earnings surprise & quality
            "sue_ibes", "earnings_persistence", "earnings_smoothness",
            "beat_miss_streak", "eps_dispersion", "eps_dispersion_trend",
            "eps_revision_1m", "eps_revision_3m", "eps_estimate_momentum",
            "revision_breadth",
            # Accruals & earnings quality
            "accrual", "total_accruals", "non_current_accruals",
            "working_capital_accruals", "accruals_to_cash_flow",
            "accruals_vs_industry",
            # Profitability
            "roe", "roa", "roce", "gpm", "npm", "opmad", "opmbd",
            "cfm", "fcf_ocf", "fcf_to_revenue", "fcf_to_revenue_trend",
            # Valuation (earnings-related)
            "pe_inc", "pe_exi", "pe_op_basic", "pe_op_dil", "pcf",
            # Legacy names (backward compat)
            "ep", "e2p", "earnings_yield", "sue", "re", "gma", "opa",
            "nincr", "roaq", "roavol", "operprof", "cash", "cashdebt",
            "cashpr", "pctacc", "acc", "absacc", "stdacc", "cfp", "cf2p",
        ]

        earnings_data = {}
        for col in earnings_cols:
            if col in row.index and pd.notna(row[col]):
                earnings_data[col] = round(float(row[col]), 6)

        return {
            "ticker": ticker,
            "permno": int(permno),
            "date": str(row["date"].date()) if pd.notna(row.get("date")) else None,
            "earnings_features": earnings_data,
            "feature_count": len(earnings_data),
            "execution_time_ms": round((time.time() - t0) * 1000, 1),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk/{ticker}")
@router.get("/risk/{ticker}")
async def risk_profile(ticker: str):
    """Risk profile: volatility, beta, drawdown, tail risk."""
    t0 = time.time()
    ticker = ticker.upper().strip()

    try:
        wp = _get_wp()
        wp._ensure_loaded()
        if not wp._ready:
            raise HTTPException(status_code=503, detail="WRDS data not loaded")

        permno = wp._resolve_permno(ticker)
        if permno is None:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")

        stock_data = wp._panel[wp._panel["permno"] == permno].sort_values("date")
        if len(stock_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        row = stock_data.iloc[-1]
        import pandas as pd
        import numpy as np

        risk_cols = [
            # Beta & factor exposures
            "beta_mkt", "beta_smb", "beta_hml", "beta_umd",
            "beta_cma", "beta_rmw", "r_squared", "alpha",
            # Volatility
            "realized_vol", "idio_vol", "total_vol",
            "down_vol", "up_vol", "vol_3m", "vol_6m", "vol_12m",
            "return_skewness", "return_kurtosis", "svar",
            # Momentum
            "mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12_2",
            "str_reversal", "ltr",
            # Turnover & liquidity
            "turnover", "turnover_3m", "turnover_6m",
            # Legacy names (backward compat)
            "beta", "betasq", "beta_dimson", "idiovol", "retvol",
            "std_dolvol", "std_turn", "maxret", "zerotrade", "ill",
            "garch_vol", "coskew", "iskew", "mom_36m",
        ]

        risk_data = {}
        for col in risk_cols:
            if col in row.index and pd.notna(row[col]):
                risk_data[col] = round(float(row[col]), 6)

        # Compute log_market_cap tier
        lmc = row.get("log_market_cap", np.nan)
        tier = wp._classify_tier(float(lmc)) if pd.notna(lmc) else "unknown"

        return {
            "ticker": ticker,
            "permno": int(permno),
            "tier": tier,
            "date": str(row["date"].date()) if pd.notna(row.get("date")) else None,
            "risk_metrics": risk_data,
            "feature_count": len(risk_data),
            "execution_time_ms": round((time.time() - t0) * 1000, 1),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/insider/{ticker}")
@router.get("/insider/{ticker}")
async def insider_activity(ticker: str):
    """Insider activity signals from GKX panel."""
    t0 = time.time()
    ticker = ticker.upper().strip()

    try:
        wp = _get_wp()
        wp._ensure_loaded()
        if not wp._ready:
            raise HTTPException(status_code=503, detail="WRDS data not loaded")

        permno = wp._resolve_permno(ticker)
        if permno is None:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")

        stock_data = wp._panel[wp._panel["permno"] == permno].sort_values("date")
        if len(stock_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        row = stock_data.iloc[-1]
        import pandas as pd

        insider_cols = [
            # Insider transactions
            "insider_buy_ratio_6m", "insider_ceo_buy", "insider_cluster_buy",
            # Analyst consensus
            "num_analysts", "num_analysts_fy1", "analyst_dispersion",
            "analyst_revision", "eps_estimate_momentum",
            # Legacy names (backward compat)
            "nanalyst", "fgr5yr", "disp", "sfe",
            "securedind", "convind", "ms",
        ]

        insider_data = {}
        for col in insider_cols:
            if col in row.index and pd.notna(row[col]):
                insider_data[col] = round(float(row[col]), 6)

        return {
            "ticker": ticker,
            "permno": int(permno),
            "date": str(row["date"].date()) if pd.notna(row.get("date")) else None,
            "insider_signals": insider_data,
            "feature_count": len(insider_data),
            "execution_time_ms": round((time.time() - t0) * 1000, 1),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/institutional/{ticker}")
@router.get("/institutional/{ticker}")
async def institutional_flows(ticker: str):
    """Institutional ownership and flow signals."""
    t0 = time.time()
    ticker = ticker.upper().strip()

    try:
        wp = _get_wp()
        wp._ensure_loaded()
        if not wp._ready:
            raise HTTPException(status_code=503, detail="WRDS data not loaded")

        permno = wp._resolve_permno(ticker)
        if permno is None:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")

        stock_data = wp._panel[wp._panel["permno"] == permno].sort_values("date")
        if len(stock_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        row = stock_data.iloc[-1]
        import pandas as pd

        inst_cols = [
            # Institutional ownership
            "inst_ownership_change", "inst_hhi", "inst_breadth",
            # Market structure
            "market_cap", "log_market_cap", "mktcap",
            "turnover", "turnover_3m", "turnover_6m",
            "sp500_member", "n_months",
            # R&D & investment
            "rd_sale", "rd_intensity", "rd_intensity_trend",
            "capex_intensity", "capex_intensity_trend",
            "capex_to_depreciation",
            # Industry position
            "revenue_share", "revenue_share_trend",
            # Legacy names (backward compat)
            "orgcap", "herf", "dolvol", "turn", "std_turn",
            "mve0", "mvel1", "age", "sp", "rd_mve",
        ]

        inst_data = {}
        for col in inst_cols:
            if col in row.index and pd.notna(row[col]):
                inst_data[col] = round(float(row[col]), 6)

        return {
            "ticker": ticker,
            "permno": int(permno),
            "date": str(row["date"].date()) if pd.notna(row.get("date")) else None,
            "institutional_signals": inst_data,
            "feature_count": len(inst_data),
            "execution_time_ms": round((time.time() - t0) * 1000, 1),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analyze/{ticker}")
@router.get("/analyze/{ticker}")
async def analyze_single(ticker: str):
    """
    Comprehensive single-stock analysis — combines prediction, fundamentals,
    earnings, risk, insider, institutional, and regime into one response.
    This is the "give me everything" endpoint.
    """
    t0 = time.time()
    ticker = ticker.upper().strip()

    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol required")

    try:
        import pandas as pd
        import numpy as np

        wp = _get_wp()
        wp._ensure_loaded()
        if not wp._ready:
            raise HTTPException(status_code=503, detail="WRDS data not loaded")

        # 1. ML Prediction (live → WRDS fallback)
        try:
            prediction = _get_lp().predict(ticker)
        except Exception:
            prediction = wp.predict(ticker)

        # If ticker not found, return 404
        if prediction.get("error") and prediction.get("confidence", 1) == 0.0:
            raise HTTPException(status_code=404, detail=prediction["error"])

        # 2. Resolve permno and get latest row
        permno = wp._resolve_permno(ticker)
        row = None
        if permno is not None and wp._panel is not None:
            stock_data = wp._panel[wp._panel["permno"] == permno].sort_values("date")
            if len(stock_data) > 0:
                row = stock_data.iloc[-1]

        # 3. Extract feature groups
        fundamentals = {}
        earnings = {}
        risk_metrics = {}
        insider = {}
        institutional = {}

        if row is not None:
            # Fundamentals (valuation + profitability + balance sheet)
            for col in ["bm", "pe_inc", "pe_exi", "pe_op_basic", "pe_op_dil",
                        "ptb", "pcf", "ps", "divyield", "capei", "peg_trailing",
                        "roe", "roa", "roce", "gpm", "npm", "opmad", "opmbd",
                        "cfm", "fcf_ocf", "fcf_to_revenue",
                        "curr_ratio", "quick_ratio", "de_ratio", "debt_at",
                        "debt_ebitda", "net_debt_to_ebitda", "capital_ratio",
                        "piotroski_f_score", "beneish_m_score",
                        "altman_z_score", "ohlson_o_score", "montier_c_score",
                        "operating_leverage", "sga_efficiency",
                        "revenue_growth_yoy", "revenue_growth_qoq",
                        "revenue_cagr_2yr", "revenue_acceleration",
                        "rd_sale", "rd_intensity", "capex_intensity",
                        # Legacy names
                        "ep", "sp", "cfp", "dp", "gma",
                        "opa", "operprof", "cash", "cashdebt", "cashpr",
                        "pctacc", "acc", "absacc", "lev", "depr", "sgr",
                        "agr", "grltnoa", "chcsho", "hire",
                        "rd_mve", "accruals_quality"]:
                if col in row.index and pd.notna(row[col]):
                    fundamentals[col] = round(float(row[col]), 6)

            # Earnings
            for col in ["sue_ibes", "earnings_persistence", "earnings_smoothness",
                        "beat_miss_streak", "eps_dispersion", "eps_dispersion_trend",
                        "eps_revision_1m", "eps_revision_3m", "eps_estimate_momentum",
                        "revision_breadth", "accrual", "total_accruals",
                        "non_current_accruals", "working_capital_accruals",
                        "accruals_to_cash_flow", "accruals_vs_industry",
                        "pe_inc", "pe_exi", "pcf",
                        # Legacy names
                        "sue", "re", "nincr", "roaq", "roavol", "stdacc",
                        "cfp", "cf2p", "e2p", "earnings_yield", "ep",
                        "nanalyst", "fgr5yr", "disp", "sfe"]:
                if col in row.index and pd.notna(row[col]):
                    earnings[col] = round(float(row[col]), 6)

            # Risk
            for col in ["beta_mkt", "beta_smb", "beta_hml", "beta_umd",
                        "beta_cma", "beta_rmw", "r_squared", "alpha",
                        "realized_vol", "idio_vol", "total_vol",
                        "down_vol", "up_vol", "vol_3m", "vol_6m", "vol_12m",
                        "return_skewness", "return_kurtosis", "svar",
                        "mom_1m", "mom_3m", "mom_6m", "mom_12m", "mom_12_2",
                        "str_reversal", "ltr", "turnover",
                        # Legacy names
                        "beta", "betasq", "beta_dimson", "idiovol", "retvol",
                        "std_dolvol", "std_turn", "maxret", "zerotrade", "ill",
                        "garch_vol", "coskew", "iskew", "mom_36m"]:
                if col in row.index and pd.notna(row[col]):
                    risk_metrics[col] = round(float(row[col]), 6)

            # Insider / Analyst
            for col in ["insider_buy_ratio_6m", "insider_ceo_buy",
                        "insider_cluster_buy", "num_analysts", "num_analysts_fy1",
                        "analyst_dispersion", "analyst_revision",
                        # Legacy names
                        "nanalyst", "fgr5yr", "disp", "sfe",
                        "securedind", "convind", "ms"]:
                if col in row.index and pd.notna(row[col]):
                    insider[col] = round(float(row[col]), 6)

            # Institutional / Market Structure
            for col in ["inst_ownership_change", "inst_hhi", "inst_breadth",
                        "market_cap", "log_market_cap", "mktcap",
                        "turnover", "turnover_3m", "turnover_6m",
                        "sp500_member", "n_months",
                        "rd_sale", "rd_intensity",
                        "capex_intensity", "revenue_share",
                        # Legacy names
                        "orgcap", "herf", "dolvol", "turn", "std_turn",
                        "mve0", "mvel1", "age", "sp", "rd_mve"]:
                if col in row.index and pd.notna(row[col]):
                    institutional[col] = round(float(row[col]), 6)

        # 4. Regime
        regime = wp.get_market_regime()

        # ── System B: Live Intelligence (best-effort, non-blocking) ──
        live_intelligence = {}

        # Live price + technicals from Polygon
        try:
            from ..data.aggregator import DataAggregator
            agg = DataAggregator()
            await agg.initialize()
            snapshot = await agg.get_snapshot(ticker)
            snap_dict = snapshot.to_dict()
            live_intelligence["live_price"] = snap_dict.get("price", {}).get("current")
            live_intelligence["technicals"] = snap_dict.get("technical", {})
            live_intelligence["options"] = snap_dict.get("options", {})
        except Exception as e:
            logger.debug(f"Snapshot unavailable for {ticker}: {e}")

        # SEC EDGAR quality score
        try:
            from ..data.sec_edgar import SECEdgarXBRL
            edgar = SECEdgarXBRL()
            quality = edgar.get_quality_score(ticker)
            if quality and quality.get("score") is not None:
                live_intelligence["sec_quality"] = {
                    "score": quality.get("score"),
                    "grade": quality.get("grade"),
                    "components": quality.get("components", {}),
                }
        except Exception as e:
            logger.debug(f"SEC quality unavailable for {ticker}: {e}")

        # Lambda Decision Engine + LuxAlgo extraction
        try:
            from ..lambda_client import NubleLambdaClient
            client = NubleLambdaClient()
            analysis = client.get_analysis(ticker)
            if analysis:
                live_intelligence["lambda_decision"] = {
                    "action": analysis.action,
                    "score": analysis.score,
                    "confidence": analysis.confidence,
                    "should_trade": analysis.should_trade,
                    "veto": analysis.veto,
                    "veto_reason": analysis.veto_reason,
                    "regime": analysis.regime.value if hasattr(analysis.regime, 'value') else str(analysis.regime),
                }
                # ── Extract LuxAlgo multi-timeframe signals ──
                live_intelligence["luxalgo"] = {
                    "weekly": analysis.luxalgo_weekly_action or "N/A",
                    "daily": analysis.luxalgo_daily_action or "N/A",
                    "h4": analysis.luxalgo_h4_action or "N/A",
                    "direction": analysis.luxalgo_direction or "",
                    "aligned": analysis.luxalgo_aligned,
                    "score": analysis.luxalgo_score,
                    "valid_count": analysis.luxalgo_valid_count,
                }
                # ── Extract full technical snapshot from Lambda ──
                if analysis.technicals:
                    t = analysis.technicals
                    live_intelligence["lambda_technicals"] = {
                        "rsi": t.rsi, "rsi_signal": t.rsi_signal,
                        "rsi_divergence": t.rsi_divergence,
                        "macd": t.macd, "macd_signal": t.macd_signal,
                        "macd_histogram": t.macd_histogram,
                        "macd_bullish": t.macd_bullish,
                        "macd_momentum": t.macd_momentum,
                        "sma_20": t.sma_20, "sma_50": t.sma_50,
                        "sma_200": t.sma_200, "trend_state": t.trend_state,
                        "atr": t.atr, "atr_percent": t.atr_percent,
                        "volatility_regime": t.volatility_regime,
                        "momentum_1d": t.momentum_1d,
                        "momentum_5d": t.momentum_5d,
                        "momentum_20d": t.momentum_20d,
                        "technical_score": t.technical_score,
                        "technical_confidence": t.technical_confidence,
                    }
                # ── Extract intelligence snapshot ──
                if analysis.intelligence:
                    i = analysis.intelligence
                    live_intelligence["lambda_intelligence"] = {
                        "sentiment_score": i.sentiment_score,
                        "sentiment_label": i.sentiment_label,
                        "news_count_7d": i.news_count_7d,
                        "news_count_24h": i.news_count_24h,
                        "is_trending": i.is_trending,
                        "trending_rank": i.trending_rank,
                        "upgrades": i.upgrades,
                        "downgrades": i.downgrades,
                        "whale_activity": i.whale_activity,
                        "vix_value": i.vix_value,
                        "vix_state": i.vix_state,
                        "intelligence_score": i.intelligence_score,
                        "intelligence_confidence": i.intelligence_confidence,
                    }
                live_intelligence["analysis_summary"] = analysis.analysis_summary
                live_intelligence["data_points_used"] = analysis.data_points_used
        except Exception as e:
            logger.debug(f"Lambda unavailable for {ticker}: {e}")

        # ── UltimateDecisionEngine — 28+ data point institutional decision ──
        ultimate_decision = None
        try:
            import asyncio as _aio
            from ..decision.ultimate_engine import UltimateDecisionEngine

            def _sanitize(obj):
                """Recursively convert numpy types to native Python for JSON."""
                if isinstance(obj, dict):
                    return {k: _sanitize(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [_sanitize(v) for v in obj]
                elif isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj)
                elif isinstance(obj, (np.bool_,)):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            ude = UltimateDecisionEngine()
            await _aio.wait_for(ude.initialize(), timeout=5.0)
            decision = await _aio.wait_for(ude.make_decision(ticker), timeout=15.0)
            if decision:
                ultimate_decision = _sanitize(decision.to_dict())
        except Exception as e:
            logger.debug(f"UltimateDecisionEngine unavailable for {ticker}: {e}")

        # ── TradeSetupCalculator — entry/stop/targets ──
        trade_setup = None
        try:
            from ..decision.trade_setup import TradeSetupCalculator
            # Need OHLC data — use snapshot or Polygon
            price_data = live_intelligence.get("technicals", {})
            current_price = live_intelligence.get("live_price")
            if current_price and current_price > 0:
                # Try to get historical bars from Polygon for ATR
                import httpx
                polygon_key = os.environ.get("POLYGON_API_KEY", "JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY")
                if polygon_key:
                    from datetime import datetime as dt, timedelta
                    end_date = dt.now().strftime("%Y-%m-%d")
                    start_date = (dt.now() - timedelta(days=120)).strftime("%Y-%m-%d")
                    bars_url = (
                        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day"
                        f"/{start_date}/{end_date}?adjusted=true&sort=asc&limit=100"
                        f"&apiKey={polygon_key}"
                    )
                    async with httpx.AsyncClient(timeout=10.0) as hclient:
                        resp = await hclient.get(bars_url)
                        if resp.status_code == 200:
                            bars_data = resp.json().get("results", [])
                            if len(bars_data) >= 20:
                                closes = np.array([b["c"] for b in bars_data])
                                highs = np.array([b["h"] for b in bars_data])
                                lows = np.array([b["l"] for b in bars_data])
                                # Determine conviction from ML prediction
                                ml_score = abs(prediction.get("composite_score", 0))
                                if ml_score > 0.05:
                                    conviction = "high"
                                elif ml_score > 0.02:
                                    conviction = "moderate"
                                else:
                                    conviction = "low"
                                direction = "LONG" if prediction.get("signal") == "BUY" else "SHORT"
                                calc = TradeSetupCalculator()
                                setup = calc.compute(
                                    direction=direction,
                                    conviction=conviction,
                                    current_price=current_price,
                                    closes=closes,
                                    highs=highs,
                                    lows=lows,
                                )
                                if setup:
                                    trade_setup = {
                                        "direction": setup.direction,
                                        "entry_price": round(setup.entry_price, 2),
                                        "stop_loss": round(setup.stop_loss, 2),
                                        "stop_distance_pct": round(setup.stop_distance_pct, 2),
                                        "stop_basis": setup.stop_basis,
                                        "tp1": round(setup.tp1, 2),
                                        "tp2": round(setup.tp2, 2),
                                        "tp3": round(setup.tp3, 2),
                                        "position_size_pct": round(setup.position_size_pct, 2),
                                        "position_size_basis": setup.position_size_basis,
                                        "risk_per_trade_pct": round(setup.risk_per_trade_pct, 2),
                                        "atr_14": round(setup.atr_14, 2),
                                        "atr_pct": round(setup.atr_pct, 2),
                                        "keltner_upper": round(setup.keltner_upper, 2),
                                        "keltner_lower": round(setup.keltner_lower, 2),
                                        "keltner_mid": round(setup.keltner_mid, 2),
                                        "risk_reward_tp1": round(setup.risk_reward_tp1, 2),
                                        "risk_reward_tp2": round(setup.risk_reward_tp2, 2),
                                        "risk_reward_tp3": round(setup.risk_reward_tp3, 2),
                                        "conviction": setup.conviction,
                                        "annualized_volatility": round(setup.annualized_volatility, 4),
                                        "notes": setup.notes,
                                    }
        except Exception as e:
            logger.debug(f"TradeSetup unavailable for {ticker}: {e}")

        # ── SignalFusionEngine — multi-source signal fusion ──
        signal_fusion = None
        try:
            from ..signals.fusion_engine import SignalFusionEngine
            fusion = SignalFusionEngine()
            # Build inputs from what we already have
            sentiment_score = None
            if live_intelligence.get("lambda_intelligence"):
                s = live_intelligence["lambda_intelligence"].get("sentiment_score")
                if s is not None:
                    sentiment_score = float(s)
            regime_label = None
            if isinstance(regime, dict) and regime.get("regime"):
                regime_label = regime["regime"]
            elif isinstance(regime, str):
                regime_label = regime
            # ML prediction as fundamental_score (-1 to +1 scale)
            fund_score = prediction.get("composite_score")
            if fund_score is not None:
                fund_score = max(-1.0, min(1.0, float(fund_score) * 10))  # scale to [-1,1]

            fused = fusion.generate_fused_signal(
                symbol=ticker,
                prices=None,  # We don't have a full DataFrame here; fusion uses LuxAlgo store
                sentiment=sentiment_score,
                regime=regime_label,
                fundamental_score=fund_score,
            )
            if fused:
                signal_fusion = fused.to_dict()
        except Exception as e:
            logger.debug(f"SignalFusionEngine unavailable for {ticker}: {e}")

        # ── VetoEngine — multi-timeframe institutional veto ──
        veto_check = None
        try:
            from ..signals.veto_engine import VetoEngine, VetoDecision
            from ..signals.timeframe_manager import TimeframeSignal, Timeframe
            from datetime import datetime as _dt

            luxalgo_data = live_intelligence.get("luxalgo", {})
            # Build TimeframeSignals from Lambda LuxAlgo extraction (if available)
            tf_signals = {}
            for tf_key, tf_enum in [("weekly", Timeframe.WEEKLY), ("daily", Timeframe.DAILY), ("h4", Timeframe.FOUR_HOUR)]:
                action_str = luxalgo_data.get(tf_key, "N/A")
                if action_str and action_str not in ("N/A", "NEUTRAL", ""):
                    direction = 1 if action_str.upper() == "BUY" else (-1 if action_str.upper() == "SELL" else 0)
                    tf_signals[tf_key] = TimeframeSignal(
                        symbol=ticker,
                        timeframe=tf_enum,
                        timestamp=_dt.now(),
                        direction=direction,
                        action=action_str.upper(),
                        price=live_intelligence.get("live_price", 0) or 0,
                    )

            if tf_signals:
                veto = VetoEngine()
                veto_result = veto.check_veto(
                    symbol=ticker,
                    weekly=tf_signals.get("weekly"),
                    daily=tf_signals.get("daily"),
                    four_hour=tf_signals.get("h4"),
                )
                if veto_result:
                    veto_check = veto_result.to_dict()
        except Exception as e:
            logger.debug(f"VetoEngine unavailable for {ticker}: {e}")

        # ── PositionCalculator — institutional-grade position sizing ──
        position_calc = None
        try:
            from ..signals.position_calculator import PositionCalculator

            current_price = live_intelligence.get("live_price", 0) or 0
            if current_price > 0 and veto_check and 'veto_result' in dir():
                pc = PositionCalculator()
                portfolio_value = 100000  # Default portfolio value
                atr_val = trade_setup.get("atr_14", 0) if trade_setup else 0
                regime_label_pc = None
                if isinstance(regime, dict) and regime.get("regime"):
                    regime_label_pc = regime["regime"]
                elif isinstance(regime, str):
                    regime_label_pc = regime

                pos_result = pc.calculate_position(
                    veto_result=veto_result,
                    current_price=current_price,
                    portfolio_value=portfolio_value,
                    atr=atr_val if atr_val > 0 else None,
                    regime=regime_label_pc or "NORMAL",
                )
                if pos_result:
                    position_calc = pos_result.to_dict()
        except Exception as e:
            logger.debug(f"PositionCalculator unavailable for {ticker}: {e}")

        return {
            "ticker": ticker,
            "prediction": prediction,
            "fundamentals": fundamentals,
            "earnings": earnings,
            "risk": risk_metrics,
            "insider": insider,
            "institutional": institutional,
            "regime": regime,
            "live_intelligence": live_intelligence,
            "ultimate_decision": ultimate_decision,
            "trade_setup": trade_setup,
            "signal_fusion": signal_fusion,
            "veto_check": veto_check,
            "position_sizing": position_calc,
            "execution_time_ms": round((time.time() - t0) * 1000, 1),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
@router.post("/analyze")
async def analyze_portfolio(request: AnalyzeRequest):
    """Batch portfolio analysis — predict all holdings, compute composite."""
    t0 = time.time()

    try:
        lp = _get_lp()
        wp = _get_wp()

        holdings = request.holdings
        total_weight = sum(holdings.values())
        norm_weights = {t: w / total_weight for t, w in holdings.items()} if total_weight > 0 else {
            t: 1 / len(holdings) for t in holdings
        }

        results = []
        portfolio_score = 0.0
        tier_allocation = {}
        signal_dist = {}

        for ticker, weight in holdings.items():
            nw = norm_weights[ticker]
            try:
                pred = lp.predict(ticker.upper())
                pred["weight"] = round(weight, 4)
                pred["normalized_weight"] = round(nw, 4)
                results.append(pred)

                score = pred.get("composite_score", pred.get("raw_score", 0))
                portfolio_score += score * nw

                tier = pred.get("tier", "small")
                tier_allocation[tier] = tier_allocation.get(tier, 0) + nw

                signal = pred.get("signal", "HOLD")
                signal_dist[signal] = signal_dist.get(signal, 0) + 1

            except Exception as e:
                results.append({
                    "ticker": ticker.upper(),
                    "error": str(e),
                    "weight": round(weight, 4),
                })

        # Regime
        regime_data = wp.get_market_regime()

        return {
            "holdings": results,
            "portfolio_score": round(portfolio_score, 6),
            "regime": regime_data,
            "tier_allocation": {k: round(v, 4) for k, v in tier_allocation.items()},
            "signal_distribution": signal_dist,
            "execution_time_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/screener")
@router.post("/screener")
async def screener(request: ScreenerRequest):
    """Custom stock screening with filters."""
    t0 = time.time()

    try:
        wp = _get_wp()
        # Get full universe
        all_stocks = wp.get_top_picks(n=5000)

        filtered = all_stocks

        if request.tiers:
            filtered = [s for s in filtered if s.get("tier") in request.tiers]

        if request.signals:
            filtered = [s for s in filtered if s.get("signal") in request.signals]

        if request.min_score is not None:
            filtered = [s for s in filtered if s.get("raw_score", 0) >= request.min_score]

        if request.max_score is not None:
            filtered = [s for s in filtered if s.get("raw_score", 0) <= request.max_score]

        if request.min_market_cap is not None:
            filtered = [s for s in filtered
                        if s.get("market_cap_millions", 0) >= request.min_market_cap]

        if request.max_market_cap is not None:
            filtered = [s for s in filtered
                        if s.get("market_cap_millions", float("inf")) <= request.max_market_cap]

        # Sort
        reverse = request.sort_desc
        filtered.sort(key=lambda x: x.get(request.sort_by, 0), reverse=reverse)

        return {
            "count": len(filtered[:request.limit]),
            "total_matched": len(filtered),
            "filters_applied": {
                k: v for k, v in request.model_dump().items()
                if v is not None and k not in ("limit", "sort_by", "sort_desc")
            },
            "stocks": filtered[:request.limit],
            "execution_time_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/top-picks")
@router.get("/top-picks")
async def top_picks(
    n: int = Query(20, ge=1, le=200),
    tier: Optional[str] = Query(None, enum=["mega", "large", "mid", "small"]),
    live: bool = Query(False, description="Re-score with live Polygon data"),
):
    """Top N stock picks, optionally with live re-scoring."""
    t0 = time.time()
    try:
        if live:
            lp = _get_lp()
            results = lp.get_live_top_picks(n=n, tier=tier)
        else:
            wp = _get_wp()
            results = wp.get_top_picks(n=n, tier=tier)

        return {
            "count": len(results),
            "tier_filter": tier,
            "live_scored": live,
            "picks": results,
            "execution_time_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tier/{tier}")
@router.get("/tier/{tier}")
async def tier_predictions(tier: str):
    """Get all predictions for a specific market cap tier."""
    t0 = time.time()
    tier = tier.lower()
    if tier not in ("mega", "large", "mid", "small"):
        raise HTTPException(status_code=400, detail="Invalid tier. Use: mega, large, mid, small")

    try:
        wp = _get_wp()
        results = wp.get_tier_predictions(tier)
        return {
            "tier": tier,
            "count": len(results),
            "stocks": results,
            "execution_time_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-info")
@router.get("/model-info")
async def model_info():
    """Model metadata: tiers, features, performance metrics."""
    t0 = time.time()
    try:
        wp = _get_wp()
        info = wp.get_model_info()

        # Add DataService status
        ds = _get_ds()
        info["data_service"] = ds.get_status()
        info["execution_time_ms"] = round((time.time() - t0) * 1000, 1)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# NEW ENDPOINTS — Full codebase leverage
# ─────────────────────────────────────────────────────────────

@app.get("/api/news/{ticker}")
@router.get("/news/{ticker}")
async def news_summary(ticker: str):
    """
    Real-time news + sentiment for a ticker via StockNews API.
    Returns recent articles, 7-day sentiment stats, and trending headlines.
    """
    t0 = time.time()
    ticker = ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol required")

    try:
        from ..news.client import StockNewsClient
        client = StockNewsClient()
        try:
            import asyncio
            summary = await client.get_news_summary(ticker)
            return {
                "ticker": ticker,
                **summary,
                "execution_time_ms": round((time.time() - t0) * 1000, 1),
            }
        finally:
            await client.close()
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"News client dependency missing: {e}")
    except Exception as e:
        logger.error(f"News fetch failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/snapshot/{ticker}")
@router.get("/snapshot/{ticker}")
async def realtime_snapshot(ticker: str):
    """
    Real-time market snapshot: price, technicals (SMA, RSI, MACD, Bollinger, ATR),
    options flow (P/C ratio, unusual activity, IV), sentiment, regime, LuxAlgo signals.
    Aggregates from Polygon, DynamoDB, FinBERT, HMM in parallel.
    """
    t0 = time.time()
    ticker = ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol required")

    try:
        from ..data.aggregator import DataAggregator
        agg = DataAggregator()
        await agg.initialize()
        snapshot = await agg.get_snapshot(ticker)
        result = snapshot.to_dict()
        result["execution_time_ms"] = round((time.time() - t0) * 1000, 1)
        return result
    except Exception as e:
        logger.error(f"Snapshot failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sec-quality/{ticker}")
@router.get("/sec-quality/{ticker}")
async def sec_quality(ticker: str):
    """
    SEC EDGAR fundamental quality assessment.
    Returns 40 GKX fundamental ratios computed from live XBRL filings,
    plus a composite quality score (0-100) and letter grade (A-F).
    """
    t0 = time.time()
    ticker = ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol required")

    try:
        from ..data.sec_edgar import SECEdgarXBRL
        edgar = SECEdgarXBRL()

        # Get quality score (includes ratios)
        quality = edgar.get_quality_score(ticker)
        if quality is None:
            raise HTTPException(status_code=404, detail=f"No SEC filings found for {ticker}")

        # Also get raw fundamental ratios
        ratios = edgar.get_fundamental_ratios(ticker)

        return {
            "ticker": ticker,
            "quality_score": quality.get("score"),
            "grade": quality.get("grade"),
            "components": quality.get("components", {}),
            "fundamental_ratios": ratios if ratios else {},
            "source": "SEC EDGAR XBRL (live)",
            "execution_time_ms": round((time.time() - t0) * 1000, 1),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SEC quality failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/macro")
@router.get("/macro")
async def macro_environment():
    """
    FRED macro environment: Treasury yields, credit spreads, inflation,
    employment, Fed Funds rate, plus derived regime indicators
    (yield curve state, credit cycle, monetary policy stance).
    """
    t0 = time.time()
    try:
        from ..data.fred_macro import FREDMacroData
        fred = FREDMacroData()
        current = fred.get_current()

        if current is None:
            return {
                "status": "unavailable",
                "reason": "FRED_API_KEY not set — macro data requires a free FRED API key",
                "execution_time_ms": round((time.time() - t0) * 1000, 1),
            }

        return {
            "status": "ok",
            **current,
            "source": "Federal Reserve Economic Data (FRED)",
            "execution_time_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        logger.error(f"Macro data failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/lambda/{ticker}")
@router.get("/lambda/{ticker}")
async def lambda_analysis(ticker: str):
    """
    Production Lambda Decision Engine analysis.
    Aggregates Polygon + StockNews + CryptoNews + LuxAlgo + regime detection
    into a single composite decision with score (0-100), action, and confidence.
    """
    t0 = time.time()
    ticker = ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol required")

    try:
        from ..lambda_client import NubleLambdaClient
        client = NubleLambdaClient()
        analysis = client.get_analysis(ticker)

        if analysis is None:
            raise HTTPException(status_code=502, detail=f"Lambda API returned no data for {ticker}")

        # Convert dataclass to dict
        result = {
            "ticker": analysis.symbol,
            "action": analysis.action,
            "direction": analysis.direction,
            "strength": analysis.strength,
            "score": analysis.score,
            "confidence": analysis.confidence,
            "should_trade": analysis.should_trade,
            "veto": analysis.veto,
            "veto_reason": analysis.veto_reason,
            "current_price": analysis.current_price,
            "change_percent": analysis.change_percent,
            "regime": analysis.regime.value if hasattr(analysis.regime, 'value') else str(analysis.regime),
            "regime_confidence": analysis.regime_confidence,
            "technicals": {
                "rsi": analysis.technicals.rsi,
                "rsi_signal": analysis.technicals.rsi_signal,
                "macd": analysis.technicals.macd,
                "macd_bullish": analysis.technicals.macd_bullish,
                "trend_state": analysis.technicals.trend_state,
                "volatility_regime": analysis.technicals.volatility_regime,
                "momentum_1d": analysis.technicals.momentum_1d,
                "momentum_5d": analysis.technicals.momentum_5d,
                "momentum_20d": analysis.technicals.momentum_20d,
                "technical_score": analysis.technicals.technical_score,
            },
            "intelligence": {
                "sentiment_score": analysis.intelligence.sentiment_score,
                "sentiment_label": analysis.intelligence.sentiment_label,
                "news_count_7d": analysis.intelligence.news_count_7d,
                "news_count_24h": analysis.intelligence.news_count_24h,
                "is_trending": analysis.intelligence.is_trending,
                "headlines": analysis.intelligence.headlines[:5],
                "vix_value": analysis.intelligence.vix_value,
                "vix_state": analysis.intelligence.vix_state,
                "intelligence_score": analysis.intelligence.intelligence_score,
            },
            "luxalgo": {
                "weekly": analysis.luxalgo_weekly_action,
                "daily": analysis.luxalgo_daily_action,
                "h4": analysis.luxalgo_h4_action,
                "direction": analysis.luxalgo_direction,
                "aligned": analysis.luxalgo_aligned,
                "score": analysis.luxalgo_score,
            },
            "summaries": {
                "stocknews": analysis.stocknews_summary,
                "cryptonews": analysis.cryptonews_summary,
                "analysis": analysis.analysis_summary,
            },
            "data_points_used": analysis.data_points_used,
            "api_latency_ms": analysis.api_latency_ms,
            "execution_time_ms": round((time.time() - t0) * 1000, 1),
        }
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lambda analysis failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/compare")
@router.get("/compare")
async def compare_stocks(
    tickers: str = Query(..., description="Comma-separated tickers, e.g. AAPL,MSFT,GOOGL"),
):
    """
    Side-by-side comparison of 2-5 stocks.
    Returns ML prediction, key fundamentals, risk metrics, and regime for each.
    """
    t0 = time.time()
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    if len(ticker_list) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 tickers (comma-separated)")
    if len(ticker_list) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 tickers for comparison")

    try:
        import pandas as pd
        wp = _get_wp()
        wp._ensure_loaded()
        lp = _get_lp()

        comparisons = []
        for ticker in ticker_list:
            entry = {"ticker": ticker}

            # Prediction
            try:
                pred = lp.predict(ticker)
            except Exception:
                try:
                    pred = wp.predict(ticker)
                except Exception:
                    pred = {"error": f"Not found: {ticker}"}
            entry["prediction"] = pred

            # Key fundamentals + risk from panel
            permno = wp._resolve_permno(ticker)
            if permno is not None and wp._panel is not None:
                stock_data = wp._panel[wp._panel["permno"] == permno].sort_values("date")
                if len(stock_data) > 0:
                    row = stock_data.iloc[-1]
                    key_fundamentals = {}
                    for col in ["ep", "bm", "roe", "roa", "gma", "sgr", "agr", "lev"]:
                        if col in row.index and pd.notna(row[col]):
                            key_fundamentals[col] = round(float(row[col]), 6)
                    entry["fundamentals"] = key_fundamentals

                    key_risk = {}
                    for col in ["beta", "idiovol", "retvol", "mom_1m", "mom_6m", "mom_12m"]:
                        if col in row.index and pd.notna(row[col]):
                            key_risk[col] = round(float(row[col]), 6)
                    entry["risk"] = key_risk

            comparisons.append(entry)

        # Regime context
        regime = wp.get_market_regime()

        return {
            "count": len(comparisons),
            "tickers": ticker_list,
            "comparisons": comparisons,
            "regime": regime,
            "execution_time_ms": round((time.time() - t0) * 1000, 1),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compare failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class PositionSizeRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    portfolio_value: float = Field(100000, description="Total portfolio value in dollars")
    risk_per_trade: float = Field(0.02, ge=0.001, le=0.05, description="Max risk per trade (fraction)")


@app.post("/api/position-size")
@router.post("/position-size")
async def position_size(request: PositionSizeRequest):
    """
    Kelly Criterion position sizing with stop-loss and take-profit levels.
    Returns recommended shares, dollar amount, stop/TP prices, risk/reward ratio.
    """
    t0 = time.time()
    ticker = request.ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol required")

    try:
        import pandas as pd
        lp = _get_lp()
        wp = _get_wp()
        wp._ensure_loaded()

        # Get prediction for confidence/signal
        try:
            prediction = lp.predict(ticker)
        except Exception:
            prediction = wp.predict(ticker)

        if prediction.get("error") and prediction.get("confidence", 1) == 0.0:
            raise HTTPException(status_code=404, detail=prediction["error"])

        # Resolve current price from multiple sources
        current_price = prediction.get("live_price", prediction.get("current_price", 0))

        # Get ATR and fallback price from panel
        atr = None
        permno = wp._resolve_permno(ticker)
        if permno is not None and wp._panel is not None:
            stock_data = wp._panel[wp._panel["permno"] == permno].sort_values("date")
            if len(stock_data) > 0:
                row = stock_data.iloc[-1]
                # Fallback price from panel (market_cap / shares or prc column)
                if current_price == 0:
                    for price_col in ["prc", "close", "last_price"]:
                        if price_col in row.index and pd.notna(row[price_col]):
                            current_price = abs(float(row[price_col]))
                            break
                    # Estimate from market cap if available
                    if current_price == 0 and "mvel1" in row.index and pd.notna(row["mvel1"]):
                        current_price = float(row["mvel1"])  # mvel1 is in millions, rough proxy
                if "atr_14" in row.index and pd.notna(row["atr_14"]):
                    atr = float(row["atr_14"])
                elif "retvol" in row.index and pd.notna(row["retvol"]):
                    # Estimate ATR from return volatility
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

        # If no ATR, estimate from volatility
        if atr is None or atr <= 0:
            retvol = prediction.get("retvol", 0.02)
            atr = retvol * current_price * 1.5 if current_price > 0 else 1.0

        # Kelly calculation
        signal = prediction.get("signal", "HOLD")
        confidence = prediction.get("confidence", 0.5)

        # Map signal to win rate estimate
        signal_win_rates = {
            "STRONG_BUY": 0.60, "BUY": 0.55, "HOLD": 0.45,
            "SELL": 0.40, "STRONG_SELL": 0.35,
        }
        win_rate = signal_win_rates.get(signal, 0.45) * (0.8 + 0.4 * confidence)
        win_rate = min(max(win_rate, 0.30), 0.70)

        win_loss_ratio = 1.5  # Target 1.5:1 R:R

        # Kelly fraction
        q = 1 - win_rate
        kelly_raw = (win_rate * win_loss_ratio - q) / win_loss_ratio
        kelly_raw = max(0, kelly_raw)
        kelly_half = kelly_raw * 0.5  # Half-Kelly for safety

        # Position sizing
        max_risk_dollars = request.portfolio_value * request.risk_per_trade
        stop_distance = atr * 2.0  # 2 ATR stop
        risk_per_share = stop_distance

        if risk_per_share > 0:
            shares_from_risk = int(max_risk_dollars / risk_per_share)
        else:
            shares_from_risk = 0

        kelly_dollars = request.portfolio_value * kelly_half
        max_position_dollars = request.portfolio_value * 0.10  # 10% cap

        if current_price > 0:
            shares_from_kelly = int(kelly_dollars / current_price)
            shares_from_cap = int(max_position_dollars / current_price)
        else:
            shares_from_kelly = 0
            shares_from_cap = 0

        recommended_shares = min(shares_from_risk, shares_from_kelly, shares_from_cap)
        recommended_shares = max(recommended_shares, 0)
        dollar_amount = recommended_shares * current_price

        # Stop loss and take profit
        is_long = signal in ("STRONG_BUY", "BUY", "HOLD")
        if is_long:
            stop_loss = current_price - stop_distance
            tp1 = current_price + (stop_distance * 1.0)  # 1:1
            tp2 = current_price + (stop_distance * 2.0)  # 2:1
            tp3 = current_price + (stop_distance * 3.0)  # 3:1
        else:
            stop_loss = current_price + stop_distance
            tp1 = current_price - (stop_distance * 1.0)
            tp2 = current_price - (stop_distance * 2.0)
            tp3 = current_price - (stop_distance * 3.0)

        stop_loss_pct = abs(stop_loss - current_price) / current_price if current_price > 0 else 0

        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "signal": signal,
            "confidence": round(confidence, 4),
            "position_sizing": {
                "recommended_shares": recommended_shares,
                "dollar_amount": round(dollar_amount, 2),
                "position_pct": round(dollar_amount / request.portfolio_value * 100, 2) if request.portfolio_value > 0 else 0,
                "kelly_fraction": round(kelly_half, 4),
                "kelly_raw": round(kelly_raw, 4),
            },
            "risk_management": {
                "stop_loss_price": round(stop_loss, 2),
                "stop_loss_pct": round(stop_loss_pct * 100, 2),
                "take_profit_1": round(tp1, 2),
                "take_profit_2": round(tp2, 2),
                "take_profit_3": round(tp3, 2),
                "risk_reward_ratio": round(1.5, 2),
                "max_loss_dollars": round(recommended_shares * stop_distance, 2),
                "atr_14": round(atr, 4),
            },
            "parameters": {
                "portfolio_value": request.portfolio_value,
                "risk_per_trade": request.risk_per_trade,
                "win_rate_estimate": round(win_rate, 4),
                "direction": "LONG" if is_long else "SHORT",
            },
            "prediction": prediction,
            "execution_time_ms": round((time.time() - t0) * 1000, 1),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Position sizing failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# CLI runner (standalone mode)
# ─────────────────────────────────────────────────────────────

def run():
    """Start ROKET API server standalone."""
    import uvicorn
    host = os.environ.get("ROKET_HOST", "0.0.0.0")
    port = int(os.environ.get("ROKET_PORT", "8001"))
    uvicorn.run("nuble.api.roket:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    run()
