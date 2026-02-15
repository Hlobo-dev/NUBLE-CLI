"""
PHASE 3 — STEP 12: Online Prediction API (FastAPI)
====================================================
Real-time prediction serving endpoints.
Endpoints:
  GET  /predict/{ticker}   → ML return forecast
  GET  /profile/{ticker}   → Full company profile
  GET  /market             → Market overview
  GET  /macro              → Macro dashboard
  POST /screen             → Stock screening
  GET  /portfolio          → Model portfolio analytics
  GET  /health             → System health check
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List

app = FastAPI(
    title="NUBLE WRDS Prediction API",
    description="Institutional-grade ML predictions powered by 100 years of WRDS data",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded advisor
_advisor = None


def get_advisor():
    global _advisor
    if _advisor is None:
        from step8_wrds_advisor import WRDSAdvisor
        _advisor = WRDSAdvisor()
    return _advisor


# ── Request/Response Models ──────────────────────────────────────────────

class ScreenRequest(BaseModel):
    bm_min: Optional[float] = None
    bm_max: Optional[float] = None
    log_market_cap_min: Optional[float] = None
    log_market_cap_max: Optional[float] = None
    roaq_min: Optional[float] = None
    mom_12m_min: Optional[float] = None
    mom_12m_max: Optional[float] = None
    top_n: int = 20


class PredictionResponse(BaseModel):
    ticker: str
    permno: Optional[int] = None
    date: Optional[str] = None
    predicted_return_pct: Optional[float] = None
    percentile: Optional[float] = None
    signal: Optional[str] = None
    model_ic: Optional[float] = None


class ProfileResponse(BaseModel):
    ticker: str
    data: Dict


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "NUBLE WRDS Prediction API",
        "version": "3.0.0",
        "endpoints": [
            "/predict/{ticker}",
            "/profile/{ticker}",
            "/market",
            "/macro",
            "/screen",
            "/portfolio",
            "/health",
        ],
    }


@app.get("/health")
async def health():
    """System health check."""
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")
    checks = {}

    for fname in ["gkx_panel.parquet", "ensemble_predictions.parquet",
                   "ticker_permno_map.parquet", "macro_predictors.parquet"]:
        fpath = os.path.join(DATA_DIR, fname)
        checks[fname] = {
            "exists": os.path.exists(fpath),
            "size_mb": round(os.path.getsize(fpath) / 1e6, 1) if os.path.exists(fpath) else 0,
        }

    return {
        "status": "healthy" if all(c["exists"] for c in checks.values()) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "data_files": checks,
    }


@app.get("/predict/{ticker}")
async def predict(ticker: str):
    """Get ML prediction for a stock."""
    advisor = get_advisor()
    result = advisor.get_model_prediction(ticker)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return {
        "ticker": ticker.upper(),
        "permno": result.get("permno"),
        "date": result.get("date"),
        "predicted_return_pct": result.get("predicted_return"),
        "percentile": result.get("percentile"),
        "signal": result.get("signal"),
        "n_stocks_ranked": result.get("n_stocks_ranked"),
        "model_ic": result.get("model_ic"),
        "model_sharpe": result.get("model_sharpe"),
    }


@app.get("/profile/{ticker}")
async def profile(ticker: str):
    """Get full company profile with fundamentals + ML prediction."""
    advisor = get_advisor()
    result = advisor.get_stock_profile(ticker)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return {"ticker": ticker.upper(), "data": result}


@app.get("/market")
async def market():
    """Get market overview."""
    advisor = get_advisor()
    return advisor.get_market_overview()


@app.get("/macro")
async def macro():
    """Get macro dashboard."""
    advisor = get_advisor()
    return advisor.get_macro_dashboard()


@app.post("/screen")
async def screen(request: ScreenRequest):
    """Screen stocks by criteria."""
    advisor = get_advisor()
    criteria = {}
    if request.bm_min is not None:
        criteria["bm_min"] = request.bm_min
    if request.bm_max is not None:
        criteria["bm_max"] = request.bm_max
    if request.log_market_cap_min is not None:
        criteria["log_market_cap_min"] = request.log_market_cap_min
    if request.log_market_cap_max is not None:
        criteria["log_market_cap_max"] = request.log_market_cap_max
    if request.roaq_min is not None:
        criteria["roaq_min"] = request.roaq_min
    if request.mom_12m_min is not None:
        criteria["mom_12m_min"] = request.mom_12m_min
    if request.mom_12m_max is not None:
        criteria["mom_12m_max"] = request.mom_12m_max

    results = advisor.search_stocks(criteria)
    if len(results) == 0:
        return {"count": 0, "stocks": []}

    keep_cols = [c for c in ["permno", "ticker", "log_market_cap", "bm",
                             "roaq", "mom_12m", "prediction"] if c in results.columns]
    top = results[keep_cols].head(request.top_n)

    return {
        "count": len(results),
        "showing": len(top),
        "stocks": top.to_dict(orient="records"),
    }


@app.get("/portfolio")
async def portfolio():
    """Get model portfolio analytics."""
    advisor = get_advisor()
    return advisor.get_portfolio_analytics()


@app.get("/sectors")
async def sectors(siccd: Optional[int] = Query(None)):
    """Get sector analysis."""
    advisor = get_advisor()
    return advisor.get_sector_analysis(siccd)


def main():
    """Run the prediction API server."""
    import uvicorn

    print("=" * 70)
    print("PHASE 3 — STEP 12: ONLINE PREDICTION API")
    print("=" * 70)
    print("Starting server on http://localhost:8100")
    print("Docs at http://localhost:8100/docs")

    uvicorn.run(app, host="0.0.0.0", port=8100)


if __name__ == "__main__":
    main()
