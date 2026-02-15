# NUBLE Intelligence API — Frontend Integration Guide

## Overview

The NUBLE Intelligence API exposes the full quantitative stack (System A+B) as structured REST endpoints. Your frontend Claude Opus instance can use these as **tools** via the Anthropic `tool_use` API, transforming it from "Claude answering from general knowledge" into "Claude with institutional-grade ML predictions."

## Architecture

```
┌──────────────────────────────────────────────────┐
│              Frontend (Chat UI)                   │
│         Claude Opus 4 with tool_use              │
└──────────┬─────────────────────┬─────────────────┘
           │                     │
    Pattern A                Pattern B
    (recommended)            (simpler)
           │                     │
           ▼                     ▼
┌────────────────────┐  ┌───────────────────────┐
│  Individual calls  │  │  /api/intel/           │
│  to /api/intel/*   │  │  chat-with-tools      │
│  endpoints         │  │  (server-side loop)   │
└────────┬───────────┘  └───────────┬───────────┘
         │                          │
         ▼                          ▼
┌──────────────────────────────────────────────────┐
│              NUBLE Backend (FastAPI)              │
│                                                   │
│  LivePredictor ─── PolygonFeatureEngine           │
│       │              │                            │
│  LightGBM (4 tier models) ←── GKX Panel (3.76M)  │
│       │                                           │
│  HMMRegimeDetector ── 3-state market regime       │
│       │                                           │
│  WRDSPredictor ── 20,723 tickers                  │
└──────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Start the API

```bash
cd /path/to/NUBLE-CLI
pip install -e ".[all]"
uvicorn nuble.api.server:app --host 0.0.0.0 --port 8000
```

The API auto-docs are at `http://localhost:8000/docs`

### 2. Pattern A — Frontend Tool Calling (Recommended)

Your frontend fetches the tool definitions, passes them to Claude, and handles tool_use responses:

```typescript
// 1. Get tool definitions from NUBLE
const toolsRes = await fetch('http://your-api:8000/api/intel/tools-schema');
const nublTools = await toolsRes.json();

// 2. Convert to Anthropic tool format
const tools = nublTools.map(t => ({
  name: t.name,
  description: t.description,
  input_schema: t.input_schema
}));

// 3. Send to Claude with tools
const response = await anthropic.messages.create({
  model: 'claude-opus-4-20250514',
  max_tokens: 4096,
  system: NUBLE_SYSTEM_PROMPT,  // see below
  tools: tools,
  messages: [{ role: 'user', content: userMessage }]
});

// 4. Handle tool_use responses
if (response.stop_reason === 'tool_use') {
  for (const block of response.content) {
    if (block.type === 'tool_use') {
      // Route to the correct NUBLE endpoint
      const result = await executeNubleTool(block.name, block.input);
      // Feed result back to Claude
      // ... (standard Anthropic tool_use loop)
    }
  }
}
```

**Tool → Endpoint mapping:**

| Tool Name | HTTP Method | Endpoint |
|-----------|-------------|----------|
| `get_stock_prediction` | GET | `/api/intel/predict/{ticker}` |
| `get_batch_predictions` | POST | `/api/intel/predict/batch` |
| `get_market_regime` | GET | `/api/intel/regime` |
| `get_top_picks` | GET | `/api/intel/top-picks?n=10&tier=mega` |
| `analyze_portfolio` | POST | `/api/intel/portfolio/analyze` |
| `get_tier_info` | GET | `/api/intel/tier-info/{ticker}` |
| `get_system_status` | GET | `/api/intel/system-status` |

### 3. Pattern B — Server-Side Loop (Simpler)

Just POST the user message and get back a fully synthesized answer:

```typescript
const res = await fetch('http://your-api:8000/api/intel/chat-with-tools', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'Should I buy NVDA?',
    conversation_id: 'conv_123'
  })
});
const data = await res.json();
// data.message     → Claude's analysis backed by ML data
// data.tools_used  → ['get_stock_prediction', 'get_market_regime']
// data.tool_results → [{ tool: '...', result_preview: '...' }]
```

---

## System Prompt for Claude

Use this system prompt for your frontend Claude instance to get the best results:

```
You are NUBLE, an institutional-grade AI financial advisor powered by
quantitative intelligence. You have access to:

• Multi-tier LightGBM ensemble trained on 3.76M observations of 539
  academic features (GKX/WRDS dataset covering 1957-2024)
• 4 per-tier models: Mega-cap (IC=0.029), Large-cap (IC=0.046),
  Mid-cap (IC=0.084), Small-cap (IC=0.129)
• HMM-based market regime detection (bull/neutral/crisis)
• Live feature computation from Polygon market data
• 20,723-ticker universe with per-tier routing

INSTRUCTIONS:
- ALWAYS use get_stock_prediction when discussing any specific stock
- ALWAYS use get_market_regime when discussing market conditions
- ALWAYS use get_top_picks when the user asks for recommendations
- ALWAYS use analyze_portfolio when the user shares holdings
- Lead with the ML signal, then explain the drivers
- Be honest about feature coverage limitations
- Put analysis in regime context
- Never make up numbers — use your tools
```

---

## Endpoint Reference

### GET `/api/intel/predict/{ticker}`

Single-stock ML prediction.

**Response:**
```json
{
  "ticker": "NVDA",
  "tier": "mega",
  "signal": "HOLD",
  "composite_score": -0.000761,
  "fundamental_score": -0.037745,
  "timing_score": 0.085536,
  "confidence": 0.0011,
  "decile": "D5",
  "data_source": "live_polygon",
  "feature_coverage": "12/51 (23.5%)",
  "feature_coverage_pct": 23.5,
  "market_cap_millions": 3200000.0,
  "sector": 45,
  "top_drivers": [
    {"feature": "mom_12m", "importance_pct": 8.2, "live_value": 0.42, "available": true},
    {"feature": "sue_ibes", "importance_pct": 6.1, "live_value": null, "available": false}
  ],
  "macro_regime": {"state": "neutral", "vix_exposure": 0.8},
  "historical_score": 0.012,
  "score_drift": -0.013
}
```

### GET `/api/intel/regime`

Market regime detection.

**Response:**
```json
{
  "state": "neutral",
  "state_id": 1,
  "probabilities": {"bull": 0.0, "neutral": 1.0, "crisis": 0.0},
  "features": {"vix": 17.35, "term_spread_10y2y": 0.42, "corp_spread_bbb": 1.2},
  "confidence": 1.0,
  "vix_exposure": 0.8
}
```

### GET `/api/intel/top-picks?n=10&tier=mega`

Top stock picks.

**Response:**
```json
{
  "picks": [
    {"ticker": "SNFCA", "composite_score": 0.0105, "signal": "BUY", "tier": "small"},
    {"ticker": "PROV", "composite_score": 0.0086, "signal": "BUY", "tier": "small"}
  ],
  "count": 10,
  "regime": {"state": "neutral"},
  "execution_time_seconds": 4.2
}
```

### POST `/api/intel/portfolio/analyze`

**Request:**
```json
{
  "holdings": {"AAPL": 0.25, "GOOGL": 0.20, "NVDA": 0.15, "JPM": 0.15, "XOM": 0.25}
}
```

**Response:**
```json
{
  "holdings": [...],
  "portfolio_score": 0.0023,
  "regime": {"state": "neutral", "vix_exposure": 0.8},
  "tier_allocation": {"mega": 0.85, "large": 0.15, "mid": 0, "small": 0},
  "signal_distribution": {"HOLD": 4, "BUY": 1}
}
```

### GET `/api/intel/system-status`

System health check. Returns model status, data freshness, component availability.

### GET `/api/intel/tools-schema`

Returns Claude-compatible tool definitions that your frontend can pass directly to the Anthropic API.

---

## Existing Chat Endpoints (Still Available)

The original chat endpoints at `/api/chat` (SSE streaming) and `/api/chat/sync` still work. They use the full Manager pipeline (SmartRouter + APEX 9-agent orchestrator + Claude synthesis). The intelligence endpoints are **in addition** to these — they expose the ML layer directly.

| Endpoint | Purpose |
|----------|---------|
| `POST /api/chat` | Full APEX pipeline, SSE streaming |
| `POST /api/chat/sync` | Full APEX pipeline, blocking |
| `WS /ws/chat` | WebSocket real-time chat |
| `GET /api/quote/{symbol}` | Quick price quote |
| `GET /api/lambda/{symbol}` | Lambda Decision Engine |
| `GET /api/intel/predict/{ticker}` | **NEW** ML prediction |
| `GET /api/intel/regime` | **NEW** HMM regime |
| `GET /api/intel/top-picks` | **NEW** Top picks |
| `POST /api/intel/chat-with-tools` | **NEW** Claude + Tools loop |

---

## Data Freshness Notes

- **GKX Panel**: Ends December 2024 (~410 days stale). Historical model accuracy unaffected, but the model hasn't seen recent market events.
- **Polygon Live**: Free tier, 23.5% feature coverage for mega-caps. Model runs with imputed zeros for missing features.
- **HMM Regime**: Trained on 420 months post-1990. Uses latest panel data for regime detection (also stale).
- **Ticker Map**: 20,723 tickers from the panel. New IPOs may not be in the map.
