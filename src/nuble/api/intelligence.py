#!/usr/bin/env python3
"""
NUBLE Intelligence API — System A+B Endpoints
===============================================
Exposes the full quantitative intelligence stack as structured REST endpoints
that the frontend (Claude Opus tool_use) can call directly.

Endpoints:
    GET  /api/intel/predict/{ticker}        → LivePredictor single-stock
    POST /api/intel/predict/batch           → LivePredictor multi-stock
    GET  /api/intel/regime                  → HMM regime detection
    GET  /api/intel/top-picks               → Top N ranked by composite score
    GET  /api/intel/system-status           → Full system health + data freshness
    GET  /api/intel/tier-info/{ticker}      → Which tier, model details
    GET  /api/intel/universe/stats          → Universe coverage stats
    POST /api/intel/portfolio/analyze       → Portfolio-level analysis
    GET  /api/intel/tools-schema            → OpenAPI-compatible tool definitions
                                              for Claude function calling

Architecture:
    Frontend Chat (Claude Opus 4)
        ↓ tool_use / function_calling
    These endpoints
        ↓
    LivePredictor → PolygonFeatureEngine → per-tier LightGBM
    WRDSPredictor → GKX Panel (3.76M rows × 539 features)
    HMMRegimeDetector → 3-state macro regime
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Pydantic Response Models — structured JSON for frontend consumption
# ═══════════════════════════════════════════════════════════════════════


class PredictionResponse(BaseModel):
    """Single-stock prediction result."""
    ticker: str
    tier: str = Field(description="mega / large / mid / small")
    signal: str = Field(description="STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL")
    composite_score: float
    fundamental_score: float
    timing_score: float
    confidence: float
    decile: str
    data_source: str = Field(description="live_polygon / wrds_historical / wrds_historical_fallback")
    feature_coverage: str
    feature_coverage_pct: float
    market_cap_millions: float = 0
    sector: Optional[str] = None
    top_drivers: List[Dict[str, Any]] = []
    macro_regime: Optional[Dict[str, Any]] = None
    historical_score: float = 0
    score_drift: float = 0
    timestamp: str = ""


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    tickers: List[str] = Field(..., description="List of ticker symbols", max_length=50)


class BatchPredictionResponse(BaseModel):
    """Batch prediction results."""
    predictions: List[Dict[str, Any]]
    total: int
    successes: int
    errors: int
    execution_time_seconds: float


class RegimeResponse(BaseModel):
    """Market regime detection result."""
    state: str = Field(description="bull / neutral / crisis")
    state_id: int
    probabilities: Dict[str, float]
    features: Dict[str, float] = {}
    confidence: float
    vix_exposure: float = Field(description="Portfolio exposure scaling factor: 1.0 (bull), 0.8 (neutral), 0.3 (crisis)")
    method: Optional[str] = None
    timestamp: str = ""


class TopPicksResponse(BaseModel):
    """Top stock picks ranked by composite score."""
    picks: List[Dict[str, Any]]
    count: int
    regime: Dict[str, Any]
    execution_time_seconds: float


class TierInfoResponse(BaseModel):
    """Tier classification and model info for a ticker."""
    ticker: str
    tier: str
    tier_label: str
    market_cap_millions: float = 0
    log_market_cap: float = 0
    model_features: int = 0
    model_ic: float = 0
    model_weight: float = 0
    strategy: str = ""


class SystemStatusResponse(BaseModel):
    """Full system health and data freshness."""
    status: str
    components: Dict[str, Dict[str, Any]]
    models: Dict[str, Dict[str, Any]]
    data_freshness: Dict[str, Any]
    timestamp: str


class UniverseStatsResponse(BaseModel):
    """Universe coverage statistics."""
    total_tickers: int
    tiers: Dict[str, Dict[str, Any]]
    panel_rows: int = 0
    panel_columns: int = 0
    panel_end_date: str = ""
    feature_coverage_live: Dict[str, Any] = {}


class PortfolioAnalyzeRequest(BaseModel):
    """Request for portfolio analysis."""
    holdings: Dict[str, float] = Field(
        ...,
        description="Ticker → weight (or shares). E.g. {'AAPL': 0.25, 'GOOGL': 0.20}"
    )


class PortfolioAnalyzeResponse(BaseModel):
    """Portfolio-level analysis."""
    holdings: List[Dict[str, Any]]
    portfolio_score: float
    regime: Dict[str, Any]
    tier_allocation: Dict[str, float]
    signal_distribution: Dict[str, int]
    execution_time_seconds: float


class ToolDefinition(BaseModel):
    """Claude-compatible tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════
# Lazy singletons — only instantiate on first request
# ═══════════════════════════════════════════════════════════════════════

_live_predictor = None
_wrds_predictor = None
_regime_detector = None


def _get_live():
    global _live_predictor
    if _live_predictor is None:
        from ..ml.live_predictor import get_live_predictor
        _live_predictor = get_live_predictor()
    return _live_predictor


def _get_wrds():
    global _wrds_predictor
    if _wrds_predictor is None:
        from ..ml.wrds_predictor import get_wrds_predictor
        _wrds_predictor = get_wrds_predictor()
    return _wrds_predictor


def _get_regime():
    global _regime_detector
    if _regime_detector is None:
        from ..ml.hmm_regime import get_regime_detector
        _regime_detector = get_regime_detector()
    return _regime_detector


# ═══════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════

router = APIRouter(prefix="/api/intel", tags=["Intelligence"])


# ── Single-Stock Prediction ─────────────────────────────────────────────

@router.get("/predict/{ticker}", response_model=PredictionResponse)
async def predict_ticker(ticker: str):
    """
    Get ML prediction for a single stock ticker.

    Uses LivePredictor:
      1. Fetches real-time features from Polygon
      2. Routes to correct per-tier LightGBM model (mega/large/mid/small)
      3. Blends 70% fundamental + 30% timing signal
      4. Returns composite score, signal, confidence, top drivers

    Falls back to WRDS historical data when Polygon is unavailable.
    """
    t0 = time.time()
    try:
        lp = _get_live()
        result = lp.predict(ticker.upper())

        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])

        return PredictionResponse(
            ticker=result.get('ticker', ticker.upper()),
            tier=result.get('tier', 'unknown'),
            signal=result.get('signal', 'HOLD'),
            composite_score=result.get('composite_score', 0),
            fundamental_score=result.get('fundamental_score', 0),
            timing_score=result.get('timing_score', 0),
            confidence=result.get('confidence', 0),
            decile=result.get('decile', 'D5'),
            data_source=result.get('data_source', 'unknown'),
            feature_coverage=result.get('feature_coverage', '0/0 (0%)'),
            feature_coverage_pct=result.get('feature_coverage_pct', 0),
            market_cap_millions=result.get('market_cap_millions', 0),
            sector=result.get('sector'),
            top_drivers=result.get('top_drivers', []),
            macro_regime=result.get('macro_regime'),
            historical_score=result.get('historical_score', 0),
            score_drift=result.get('score_drift', 0),
            timestamp=result.get('timestamp', datetime.now().isoformat()),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Batch Prediction ────────────────────────────────────────────────────

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Get ML predictions for multiple tickers.

    Rate-limited for Polygon free tier (5 req/min).
    Max 50 tickers per request.
    """
    t0 = time.time()
    lp = _get_live()

    predictions = []
    errors = 0
    for ticker in request.tickers:
        try:
            result = lp.predict(ticker.upper())
            predictions.append(result)
        except Exception as e:
            predictions.append({'ticker': ticker.upper(), 'error': str(e)})
            errors += 1

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(request.tickers),
        successes=len(request.tickers) - errors,
        errors=errors,
        execution_time_seconds=round(time.time() - t0, 2),
    )


# ── Market Regime ────────────────────────────────────────────────────────

@router.get("/regime", response_model=RegimeResponse)
async def market_regime():
    """
    Detect current market regime using HMM (Hidden Markov Model).

    3 states:
      - bull:    Low VIX, positive term spread, strong returns → exposure 1.0
      - neutral: Average conditions → exposure 0.8
      - crisis:  High VIX, flat/inverted curve → exposure 0.3

    HMM trained on 420 months of macro data (VIX, term spread, credit spread,
    realized vol, momentum).
    """
    try:
        detector = _get_regime()
        regime = detector.detect_regime()
        vix_exposure = detector.get_vix_exposure()

        return RegimeResponse(
            state=regime.get('state', 'neutral'),
            state_id=regime.get('state_id', 1),
            probabilities=regime.get('probabilities', {}),
            features=regime.get('features', {}),
            confidence=regime.get('confidence', 0.5),
            vix_exposure=vix_exposure,
            method=regime.get('method'),
            timestamp=regime.get('timestamp', datetime.now().isoformat()),
        )
    except Exception as e:
        logger.error(f"Regime detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Top Picks ────────────────────────────────────────────────────────────

@router.get("/top-picks", response_model=TopPicksResponse)
async def top_picks(
    n: int = Query(10, ge=1, le=50, description="Number of picks"),
    tier: Optional[str] = Query(None, description="Filter by tier: mega / large / mid / small"),
):
    """
    Get top N stock picks ranked by composite score.

    Uses WRDS historical rankings as candidates, then re-scores with
    live Polygon data for the top candidates.
    """
    t0 = time.time()
    try:
        lp = _get_live()
        picks = lp.get_live_top_picks(n=n, tier=tier)

        detector = _get_regime()
        regime = detector.detect_regime()

        return TopPicksResponse(
            picks=picks,
            count=len(picks),
            regime=regime,
            execution_time_seconds=round(time.time() - t0, 2),
        )
    except Exception as e:
        logger.error(f"Top picks failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Tier Info ────────────────────────────────────────────────────────────

@router.get("/tier-info/{ticker}", response_model=TierInfoResponse)
async def tier_info(ticker: str):
    """
    Get tier classification and model metadata for a ticker.

    Shows which tier model (mega/large/mid/small) handles this stock,
    the number of features, historical IC, ensemble weight, and strategy.
    """
    try:
        from ..ml.wrds_predictor import TIER_CONFIG
        wrds = _get_wrds()
        wrds._ensure_loaded()

        # Use predict() to get full tier info (handles permno lookup, panel search)
        pred = wrds.predict(ticker.upper())

        if 'error' in pred and 'not found' in pred.get('error', '').lower():
            raise HTTPException(status_code=404, detail=pred['error'])

        tier = pred.get('tier', 'small')
        tc = TIER_CONFIG.get(tier, {})
        model = wrds._models.get(tier)
        n_features = len(model.feature_name()) if model else 0
        lmc = pred.get('log_market_cap', 0)
        mcap = pred.get('market_cap_millions', 0)

        return TierInfoResponse(
            ticker=ticker.upper(),
            tier=tier,
            tier_label=tc.get('label', tier),
            market_cap_millions=mcap,
            log_market_cap=round(lmc, 2) if lmc else 0,
            model_features=n_features,
            model_ic=tc.get('ic', 0),
            model_weight=tc.get('weight', 0),
            strategy=tc.get('strategy', ''),
        )
    except Exception as e:
        logger.error(f"Tier info failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Universe Stats ───────────────────────────────────────────────────────

@router.get("/universe/stats", response_model=UniverseStatsResponse)
async def universe_stats():
    """
    Get statistics about the stock universe: total tickers, per-tier counts,
    panel dimensions, data freshness, and live feature coverage.
    """
    try:
        wrds = _get_wrds()
        wrds._ensure_loaded()

        total_tickers = len(wrds._ticker_to_permno) if wrds._ticker_to_permno else 0

        # Tier breakdown — we can't easily get market caps from the ticker map
        # (it's just ticker→permno), so report model-level info instead
        tiers = {}
        from ..ml.wrds_predictor import TIER_CONFIG
        for tier_name in ['mega', 'large', 'mid', 'small']:
            tc = TIER_CONFIG.get(tier_name, {})
            model = wrds._models.get(tier_name)
            tiers[tier_name] = {
                'label': tc.get('label', tier_name),
                'ic': tc.get('ic', 0),
                'weight': tc.get('weight', 0),
                'model_loaded': model is not None,
                'features': len(model.feature_name()) if model else 0,
            }

        # Panel info
        panel_rows = 0
        panel_cols = 0
        panel_end = ""
        try:
            import pandas as pd
            from ..data.data_service import get_data_service
            _ds = get_data_service()
            panel_path = str(_ds.data_dir / "gkx_panel.parquet")
            if os.path.exists(panel_path):
                pf = pd.read_parquet(panel_path, columns=['date'])
                panel_rows = len(pf)
                panel_end = str(pf['date'].max())
                # Get column count from metadata
                import pyarrow.parquet as pq
                panel_cols = len(pq.read_schema(panel_path).names)
        except Exception:
            pass

        return UniverseStatsResponse(
            total_tickers=total_tickers,
            tiers=tiers,
            panel_rows=panel_rows,
            panel_columns=panel_cols,
            panel_end_date=panel_end,
        )
    except Exception as e:
        logger.error(f"Universe stats failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Portfolio Analysis ───────────────────────────────────────────────────

@router.post("/portfolio/analyze", response_model=PortfolioAnalyzeResponse)
async def portfolio_analyze(request: PortfolioAnalyzeRequest):
    """
    Analyze a portfolio of holdings against the ML models and regime.

    Input: { "AAPL": 0.25, "GOOGL": 0.20, "NVDA": 0.15, ... }
    Output: Per-holding predictions + portfolio-level composite score,
            tier allocation, signal distribution, regime context.
    """
    t0 = time.time()
    try:
        lp = _get_live()
        detector = _get_regime()

        tickers = list(request.holdings.keys())
        weights = list(request.holdings.values())
        total_weight = sum(weights)

        # Normalize weights
        if total_weight > 0:
            norm_weights = [w / total_weight for w in weights]
        else:
            norm_weights = [1.0 / len(tickers)] * len(tickers)

        # Predict each holding
        holdings = []
        portfolio_score = 0.0
        tier_allocation = {'mega': 0, 'large': 0, 'mid': 0, 'small': 0}
        signal_dist = {}

        for ticker, weight, norm_w in zip(tickers, weights, norm_weights):
            try:
                pred = lp.predict(ticker.upper())
                pred['weight'] = round(weight, 4)
                pred['normalized_weight'] = round(norm_w, 4)
                holdings.append(pred)

                # Accumulate
                score = pred.get('composite_score', 0)
                portfolio_score += score * norm_w

                tier = pred.get('tier', 'small')
                tier_allocation[tier] = tier_allocation.get(tier, 0) + norm_w

                signal = pred.get('signal', 'HOLD')
                signal_dist[signal] = signal_dist.get(signal, 0) + 1

            except Exception as e:
                holdings.append({
                    'ticker': ticker.upper(),
                    'error': str(e),
                    'weight': round(weight, 4),
                    'normalized_weight': round(norm_w, 4),
                })

        regime = detector.detect_regime()

        # Round tier allocation
        tier_allocation = {k: round(v, 4) for k, v in tier_allocation.items()}

        return PortfolioAnalyzeResponse(
            holdings=holdings,
            portfolio_score=round(portfolio_score, 6),
            regime=regime,
            tier_allocation=tier_allocation,
            signal_distribution=signal_dist,
            execution_time_seconds=round(time.time() - t0, 2),
        )
    except Exception as e:
        logger.error(f"Portfolio analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── System Status ────────────────────────────────────────────────────────

@router.get("/system-status", response_model=SystemStatusResponse)
async def system_status():
    """
    Full system health check: models loaded, data freshness, component status.
    """
    try:
        components = {}

        # LivePredictor
        try:
            lp = _get_live()
            engine = lp._get_polygon_engine()
            components['live_predictor'] = {
                'available': True,
                'polygon_engine': engine is not None,
                'timing_model': lp._timing_model is not None and lp._timing_model is not False,
            }
        except Exception as e:
            components['live_predictor'] = {'available': False, 'error': str(e)}

        # WRDSPredictor
        try:
            wrds = _get_wrds()
            wrds._ensure_loaded()
            components['wrds_predictor'] = {
                'available': True,
                'is_ready': wrds.is_ready,
                'tickers': len(wrds._ticker_to_permno) if wrds._ticker_to_permno else 0,
                'models_loaded': list(wrds._models.keys()) if wrds._models else [],
            }
        except Exception as e:
            components['wrds_predictor'] = {'available': False, 'error': str(e)}

        # HMM Regime
        try:
            det = _get_regime()
            components['hmm_regime'] = {
                'available': True,
                'ready': det._ready,
                'states': list(det._state_labels.values()) if det._state_labels else [],
            }
        except Exception as e:
            components['hmm_regime'] = {'available': False, 'error': str(e)}

        # Models
        models = {}
        from ..data.data_service import get_data_service
        _ds = get_data_service()

        # Research models
        lgb_dir = str(_ds.models_dir / "lightgbm")
        if os.path.isdir(lgb_dir):
            for fname in os.listdir(lgb_dir):
                fpath = os.path.join(lgb_dir, fname)
                if fname.endswith('.txt'):
                    models[fname] = {
                        'path': fpath,
                        'type': 'research',
                        'size_mb': round(os.path.getsize(fpath) / 1e6, 2),
                        'modified': datetime.fromtimestamp(
                            os.path.getmtime(fpath)
                        ).isoformat(),
                    }

        # Production models
        prod_dir = str(_ds.models_dir / "production")
        if os.path.isdir(prod_dir):
            for fname in os.listdir(prod_dir):
                fpath = os.path.join(prod_dir, fname)
                if fname.endswith('.txt'):
                    models[fname] = {
                        'path': fpath,
                        'type': 'production',
                        'size_mb': round(os.path.getsize(fpath) / 1e6, 2),
                        'modified': datetime.fromtimestamp(
                            os.path.getmtime(fpath)
                        ).isoformat(),
                    }

        # Regime model
        regime_path = str(_ds.models_dir / "regime" / "hmm_regime_model.pkl")
        if os.path.exists(regime_path):
            models['hmm_regime_model.pkl'] = {
                'path': regime_path,
                'size_mb': round(os.path.getsize(regime_path) / 1e6, 2),
                'modified': datetime.fromtimestamp(
                    os.path.getmtime(regime_path)
                ).isoformat(),
            }

        # Data freshness
        data_freshness = {}
        panel_path = str(_ds.data_dir / "gkx_panel.parquet")
        if os.path.exists(panel_path):
            data_freshness['gkx_panel'] = {
                'exists': True,
                'size_mb': round(os.path.getsize(panel_path) / 1e6, 2),
                'modified': datetime.fromtimestamp(
                    os.path.getmtime(panel_path)
                ).isoformat(),
            }
            try:
                import pandas as pd
                dates = pd.read_parquet(panel_path, columns=['date'])
                max_date = pd.to_datetime(dates['date']).max()
                staleness_days = (datetime.now() - max_date.to_pydatetime().replace(tzinfo=None)).days
                data_freshness['gkx_panel']['latest_date'] = str(max_date.date())
                data_freshness['gkx_panel']['staleness_days'] = staleness_days
            except Exception:
                pass

        ticker_path = str(_ds.data_dir / "ticker_permno_map.parquet")
        if os.path.exists(ticker_path):
            data_freshness['ticker_permno_map'] = {
                'exists': True,
                'size_mb': round(os.path.getsize(ticker_path) / 1e6, 2),
            }

        return SystemStatusResponse(
            status='operational',
            components=components,
            models=models,
            data_freshness=data_freshness,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"System status failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Claude Tool Definitions ─────────────────────────────────────────────

@router.get("/tools-schema", response_model=List[ToolDefinition])
async def tools_schema():
    """
    Returns Claude-compatible tool definitions for all intelligence endpoints.

    Your frontend Claude Opus instance can use these as tools via the
    Anthropic tool_use API. Each tool maps to one of the endpoints above.

    Example usage in your frontend:
        tools = requests.get("http://your-api/api/intel/tools-schema").json()
        response = anthropic.messages.create(
            model="claude-opus-4-20250514",
            tools=tools,
            messages=[{"role": "user", "content": "Should I buy NVDA?"}]
        )
    """
    return [
        ToolDefinition(
            name="get_stock_prediction",
            description=(
                "Get a machine learning prediction for a stock ticker. "
                "Returns composite score (70% fundamental from WRDS-trained LightGBM + "
                "30% timing), signal (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL), confidence, "
                "tier classification, feature coverage, and top prediction drivers. "
                "Use this whenever the user asks about a specific stock."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. AAPL, NVDA, TSLA)"
                    }
                },
                "required": ["ticker"]
            },
        ),
        ToolDefinition(
            name="get_batch_predictions",
            description=(
                "Get ML predictions for multiple stock tickers at once. "
                "Use this when the user asks about comparing multiple stocks "
                "or analyzing a watchlist. Max 50 tickers per request."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ticker symbols",
                        "maxItems": 50
                    }
                },
                "required": ["tickers"]
            },
        ),
        ToolDefinition(
            name="get_market_regime",
            description=(
                "Detect the current market regime using a Hidden Markov Model trained "
                "on 420 months of macro data (VIX, term spread, credit spread, volatility, "
                "momentum). Returns bull/neutral/crisis state with probabilities and "
                "a VIX exposure scaling factor. Use this when the user asks about market "
                "conditions, risk environment, or whether it's a good time to invest."
            ),
            input_schema={
                "type": "object",
                "properties": {},
            },
        ),
        ToolDefinition(
            name="get_top_picks",
            description=(
                "Get the top N stock picks ranked by composite ML score. "
                "Can filter by tier (mega/large/mid/small). Use this when the user "
                "asks 'what should I buy?' or 'best stocks right now' or 'top picks'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of picks (1-50, default 10)",
                        "default": 10
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["mega", "large", "mid", "small"],
                        "description": "Optional tier filter"
                    }
                },
            },
        ),
        ToolDefinition(
            name="analyze_portfolio",
            description=(
                "Analyze a portfolio of stock holdings against the ML models. "
                "Input is a dict of ticker → weight. Returns per-holding predictions, "
                "portfolio composite score, tier allocation, signal distribution, and "
                "current market regime. Use this when the user shares their portfolio "
                "or asks about portfolio risk/optimization."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "holdings": {
                        "type": "object",
                        "description": "Ticker → weight mapping. E.g. {\"AAPL\": 0.25, \"GOOGL\": 0.20}",
                        "additionalProperties": {"type": "number"}
                    }
                },
                "required": ["holdings"]
            },
        ),
        ToolDefinition(
            name="get_tier_info",
            description=(
                "Get which tier model (mega/large/mid/small) handles a specific stock, "
                "along with model IC, features count, ensemble weight, and strategy. "
                "Use this when the user asks about why a stock is classified a certain way."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    }
                },
                "required": ["ticker"]
            },
        ),
        ToolDefinition(
            name="get_system_status",
            description=(
                "Get full system health: which ML models are loaded, data freshness "
                "(how stale is the WRDS panel), component availability. Use this when "
                "the user asks about system capabilities or data quality."
            ),
            input_schema={
                "type": "object",
                "properties": {},
            },
        ),
    ]
