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
        from nuble.data.data_service import get_data_service
        _ds = get_data_service()
    return _ds


def _get_lp():
    global _lp
    if _lp is None:
        from nuble.ml.live_predictor import get_live_predictor
        _lp = get_live_predictor()
    return _lp


def _get_wp():
    global _wp
    if _wp is None:
        from nuble.ml.wrds_predictor import get_wrds_predictor
        _wp = get_wrds_predictor()
    return _wp


def _get_regime():
    global _regime
    if _regime is None:
        try:
            from nuble.ml.hmm_regime import get_regime_detector
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
        components["live_predictor"] = {
            "polygon_engine": lp._polygon_engine is not None and lp._polygon_engine is not False,
            "production_models": len(lp._production_models),
        }
    except Exception as e:
        components["live_predictor"] = {"ready": False, "error": str(e)}

    det = _get_regime()
    components["hmm_regime"] = {"ready": det is not None and det._ready} if det else {"ready": False}

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

        result["execution_time_ms"] = round((time.time() - t0) * 1000, 1)
        return result

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
            "ep", "e2p", "earnings_yield", "sue", "re",
            "roe", "roa", "gma", "opa", "nincr",
            "roaq", "roavol", "operprof", "ps",
            "cash", "cashdebt", "cashpr", "pctacc",
            "acc", "absacc", "stdacc", "cfp", "cf2p",
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
            "beta", "betasq", "beta_dimson",
            "idiovol", "retvol", "std_dolvol", "std_turn",
            "maxret", "zerotrade", "ill",
            "realized_vol", "garch_vol",
            "coskew", "iskew",
            "mom_1m", "mom_6m", "mom_12m", "mom_36m",
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
            "orgcap", "herf", "dolvol", "turn",
            "std_turn", "mve0", "mvel1",
            "age", "sp", "rd_sale", "rd_mve",
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
