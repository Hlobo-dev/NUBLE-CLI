# NUBLE v6.0.0 "APEX PREDATOR" — Master System Documentation

> **Generated**: Exhaustive five-pass file-by-file audit of every source file in the codebase  
> **Scope**: ~226 files, 118,000+ lines of production code (nuble core + institutional pro + TENK RAG + infrastructure + scripts)  
> **Audit Depth**: Every method, every data flow, every integration point documented  
> **Purpose**: The definitive technical reference for NUBLE — Fifth Pass Deep Audit Edition

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Package Entry Points](#3-package-entry-points)
4. [CLI Layer (Interactive Shell)](#4-cli-layer-interactive-shell)
5. [Manager (Conversation Engine)](#5-manager-conversation-engine)
6. [Agent Layer (LLM Planning)](#6-agent-layer-llm-planning)
7. [Smart Router](#7-smart-router)
8. [Unified Services Layer](#8-unified-services-layer)
9. [ROKET REST API (System B Frontend)](#9-roket-rest-api-system-b-frontend)
10. [Lambda Decision Engine Client](#10-lambda-decision-engine-client)
11. [Ultimate Decision Engine (28+ Data Points)](#11-ultimate-decision-engine-28-data-points)
12. [Signal Pipeline](#12-signal-pipeline)
13. [Multi-Agent Orchestrator (APEX Tier)](#13-multi-agent-orchestrator-apex-tier)
14. [Specialist Agents — Deep Dive](#14-specialist-agents--deep-dive)
15. [SharedDataLayer — Async Data Cache](#15-shareddatalayer--async-data-cache)
16. [Elite API Server (SSE + WebSocket)](#16-elite-api-server-sse--websocket)
17. [Tool Executor (Claude ↔ Tools Loop)](#17-tool-executor-claude--tools-loop)
18. [Intelligence API (System A+B Endpoints)](#18-intelligence-api-system-ab-endpoints)
19. [LuxAlgo Webhook API](#19-luxalgo-webhook-api)
20. [ML Pipeline (System B)](#20-ml-pipeline-system-b)
21. [Data Layer](#21-data-layer)
22. [Decision Layer](#22-decision-layer)
23. [Learning System — Deep Dive](#23-learning-system--deep-dive)
24. [News & Sentiment Pipeline](#24-news--sentiment-pipeline)
25. [Memory & Cache](#25-memory--cache)
26. [LLM Wrapper](#26-llm-wrapper)
27. [Helpers & Utilities](#27-helpers--utilities)
28. [AWS Cloud Infrastructure](#28-aws-cloud-infrastructure)
29. [Configuration & Dependencies](#29-configuration--dependencies)
30. [Production Configuration & Deployment Scripts](#30-production-configuration--deployment-scripts)
31. [Asset Detection & Crypto Analysis](#31-asset-detection--crypto-analysis)
32. [Institutional Pro Module (`src/institutional/`)](#32-institutional-pro-module-srcinstitutional)
33. [TENK SEC Filing RAG System (`TENK_SOURCE/`)](#33-tenk-sec-filing-rag-system-tenk_source)
34. [Data Sources Summary](#34-data-sources-summary)
35. [API Endpoint Reference](#35-api-endpoint-reference)
36. [File-by-File Index](#36-file-by-file-index)

---

## 1. System Overview

NUBLE is an institutional-grade AI investment research platform built in Python. It operates as a **dual-brain** system:

### System A: CLI + Conversational AI
- **Rich terminal interface** with gradient ASCII art banner
- **Claude Sonnet 4** (or GPT-4.1 fallback) for planning and answer generation
- **9 specialist AI agents** running in parallel (APEX tier)
- **Lambda Decision Engine** providing real-time data from 40+ API endpoints
- **Conversation memory** with auto-compaction at 40 messages

### System B: ROKET REST API + ML Pipeline
- **FastAPI server** (20 REST endpoints) serving ML predictions
- **4 LightGBM models** trained on WRDS/GKX panel (522 features, 20,723 tickers)
- **Live Polygon.io** feature engine computing 600+ WRDS-compatible features
- **HMM regime detection** (3 states: BULL/SIDEWAYS/BEAR)

### Bridge: Both systems converge through
- **UltimateDecisionEngine** (28+ weighted data points with risk veto)
- **SignalFusionEngine** (multi-source signal blending)
- **VetoEngine** (institutional multi-timeframe hierarchy)
- **TradeSetupCalculator** (ATR-based entries/stops/targets)
- **PositionCalculator** (fractional Kelly criterion sizing)

---

## 2. Architecture Diagram

```
USER INPUT
    │
    ├── CLI (cli.py)
    │       │
    │       ├── Quick Commands (/status, /luxalgo, /tenk, /model-health, /lambda)
    │       │       → Direct data display, no LLM
    │       │
    │       └── Normal Query → Manager (manager.py)
    │               │
    │               ├── SmartRouter (router.py) → Fast Path?
    │               │       │                        │
    │               │       │ YES                    │ NO
    │               │       ↓                        ↓
    │               │   Direct Response       Full LLM Planning
    │               │   (Quote/Tech/ML)            │
    │               │                              ├── Agent.run() → Claude plans research steps
    │               │                              ├── Agent.action() → Lambda data + Claude analysis
    │               │                              ├── APEX Orchestrator (background thread)
    │               │                              │       ├── 9 Specialist Agents (parallel)
    │               │                              │       ├── UltimateDecisionEngine
    │               │                              │       ├── ML Predictor
    │               │                              │       └── EnrichmentEngine
    │               │                              └── Agent.answer() → Claude synthesizes + streams
    │               │
    │               └── Learning System records predictions
    │
    ├── ROKET API (roket.py) — FastAPI @ port 8000/8001
    │       │
    │       ├── GET /api/predict/{ticker} → LivePredictor → LightGBM
    │       ├── GET /api/analyze/{ticker} → All engines combined (15 sections)
    │       ├── GET /api/universe         → 20,723 stocks with predictions
    │       ├── GET /api/regime           → HMM macro regime
    │       ├── GET /api/fundamentals/{t} → 522 GKX features
    │       ├── GET /api/earnings/{t}     → Earnings features
    │       ├── GET /api/risk/{t}         → Risk profile
    │       ├── GET /api/insider/{t}      → Insider signals
    │       ├── GET /api/institutional/{t}→ Institutional flows
    │       ├── POST /api/analyze         → Portfolio batch analysis
    │       ├── POST /api/screener        → Custom stock screening
    │       ├── GET /api/top-picks        → Top N picks
    │       ├── GET /api/tier/{tier}      → Tier-specific predictions
    │       ├── GET /api/model-info       → Model metadata
    │       ├── GET /api/news/{t}         → StockNews API
    │       ├── GET /api/snapshot/{t}     → Real-time multi-source snapshot
    │       ├── GET /api/sec-quality/{t}  → SEC EDGAR quality score
    │       ├── GET /api/macro            → FRED macro environment
    │       ├── GET /api/lambda/{t}       → Lambda decision pass-through
    │       ├── GET /api/compare          → Side-by-side comparison
    │       ├── POST /api/position-size   → Kelly criterion position sizing
    │       └── POST /webhooks/luxalgo    → TradingView webhook receiver
    │
    └── Lambda API (AWS)
            │
            └── https://9vyvetp9c7.execute-api.us-east-1.amazonaws.com/production
                    ├── GET /check/{symbol}  → Full decision engine
                    ├── GET /health          → Health check
                    └── GET /signals/{sym}   → Trading signals
```

---

## 3. Package Entry Points

### `src/nuble/__init__.py` (29 lines)
- **Version**: `6.0.0` ("APEX PREDATOR Edition")
- Loads `.env` from multiple search paths: `./`, `../`, project root, `~/.nuble/`
- Exports: `console` (Rich Console), `__version__`

### `src/nuble/__main__.py` (6 lines)
- Entry point: `python -m nuble` → calls `nuble.cli.main()`

### CLI command: `nuble` (via pyproject.toml `[project.scripts]`)

---

## 4. CLI Layer (Interactive Shell)

### `src/nuble/cli.py` (512 lines)

The terminal user interface built with Rich library.

#### Startup Sequence:
1. Display gradient ASCII art banner (NUBLE logo with rainbow colors)
2. Check Lambda Decision Engine connection → show ✓ or ⚠
3. Check APEX Dual-Brain Fusion availability → show ✓ or ⚠
4. Display 9 usage tips (quick quote, crypto, ML, technical, research, lambda, luxalgo, SEC, help)

#### Quick Commands (handled before LLM, no conversation context):

| Command | Behavior |
|---------|----------|
| `/status` | Rich table: Services health, Lambda status, LuxAlgo signals, TENK RAG status, ML Model health with backtest metrics (IC, Sharpe, drawdown, decile monotonicity, deployability verdict) |
| `/version` | Shows "v6.0.0 — APEX PREDATOR Edition" |
| `/help` | Displays command reference |
| `/clear` | Clears conversation history |
| `/quit` | Exits |
| `/luxalgo SYMBOL` | Fetches LuxAlgo multi-TF signals from Lambda: alignment, direction, weekly/daily/4H actions, score, valid count |
| `/tenk SYMBOL` | TENK SEC Filing RAG: searches filings via `FundamentalAnalystAgent._get_tenk_filing_insights()`, shows risk factors, revenue breakdown, management outlook, competitive position with DuckDB + sentence-transformers (384-dim embeddings) |
| `/model-health` | Displays ML backtest walk-forward results, IC, Sharpe, max drawdown, decile monotonicity, deployability verdict |
| `/lambda SYMBOL` | Direct Lambda API test: shows action, score, confidence, price, RSI, MACD, VIX, StockNews, CryptoNews, LuxAlgo signals |
| `/lambda` (no args) | Shows Lambda health/connection status |

#### Query Examples (auto-detected by SmartRouter):
- `AAPL` → Fast path: instant quote from Polygon
- `predict TSLA` → Fast path: ML prediction
- `RSI for AMD` → Fast path: technical indicators
- `Should I buy NVDA?` → Full APEX: 9 agents + decision engine + ML
- `Why is BTC down?` → Full path: crypto analysis with whale/regulatory data

#### Conversation Management:
- `MAX_CONVERSATION_MESSAGES = 40`
- Auto-compaction via `agent.compact()` when exceeded (keeps last 10 as fallback)
- Cleanup on exit: suppresses asyncio `Unclosed client session` warnings

---

## 5. Manager (Conversation Engine)

### `src/nuble/manager.py` (1,498 lines)

The **central orchestration hub** that coordinates all subsystems.

#### Initialization (`__init__`):
Initializes (in order, all with graceful fallbacks):
1. **LLM Agent** (Claude Sonnet 4 / GPT-4.1)
2. **ML Predictor v2** (F4 pipeline, preferred) → v1 legacy fallback
3. **Model Manager** (freshness tracking)
4. **UltimateDecisionEngine** (28+ data points)
5. **UnifiedServices** + **SmartRouter** (fast path handling)
6. **LearningHub** (prediction tracking + weight adjustment)
7. **OrchestratorAgent** (APEX tier, 9 agents + DecisionEngine + ML)
   - Config: `max_parallel_agents=5`, `default_timeout=25s`, `use_opus=True`

#### Core Properties:
- `services` → `UnifiedServices`
- `router` → `SmartRouter`
- `decision_engine` → `UltimateDecisionEngine`
- `ml_predictor` → v2 preferred, v1 fallback
- `orchestrator` → `OrchestratorAgent`

#### ML Prediction Flow (`get_ml_prediction(symbol)`):
1. Try v2 predictor (per-ticker MLP models trained on OHLCV)
2. Fetch 120 days OHLCV from Polygon
3. Fallback to v1 institutional ML
4. Returns formatted direction, confidence, model type, top features

#### Fast Path Handling (`_handle_fast_path(routed)`):
When SmartRouter detects high-confidence simple queries:

| Intent | Handler | How It Works |
|--------|---------|-------------|
| QUOTE | `_fast_quote()` | Polygon `/prev` endpoint → price, change%, volume, range |
| PREDICTION | `_fast_prediction()` | Direct model registry lookup → grade, Sharpe, accuracy. Falls back to `_fallback_prediction()` which computes RSI/SMA/momentum from Polygon bars |
| TECHNICAL | `_fast_technical()` | 90 days from Polygon → RSI(14), MACD, SMA 20/50, Rich table with signals |
| PATTERN | `_fast_patterns()` | Defers to full LLM (needs deep analysis) |
| SENTIMENT | `_fast_sentiment()` | Defers to full LLM (needs news/social data) |
| FILINGS_SEARCH | `_fast_filings_search()` | Links to SEC EDGAR |
| DECISION | `_fast_decision()` | UltimateDecisionEngine → action, confidence, risk score, entry/stop/TP, score breakdown |

#### Crypto Mapping (single source of truth):
```python
CRYPTO_TICKERS = {
    'BTC': 'X:BTCUSD', 'ETH': 'X:ETHUSD', 'SOL': 'X:SOLUSD',
    'XRP': 'X:XRPUSD', 'ADA': 'X:ADAUSD', 'DOT': 'X:DOTUSD',
    'DOGE': 'X:DOGEUSD', 'AVAX': 'X:AVAXUSD', 'LINK': 'X:LINKUSD',
    'MATIC': 'X:MATICUSD', 'LTC': 'X:LTCUSD', 'UNI': 'X:UNIUSD',
    'ATOM': 'X:ATOMUSD', 'SHIB': 'X:SHIBUSD', 'NEAR': 'X:NEARUSD',
    'ARB': 'X:ARBUSD', 'OP': 'X:OPUSD',
    ...
}
```

#### Full Path Processing (`process_prompt(prompt, conversation)`):
1. **Model freshness check** (once per session)
2. **API key validation** (ANTHROPIC_API_KEY or OPENAI_API_KEY)
3. **Fast path attempt** → if SmartRouter says `fast_path=True` and `confidence >= 0.8`, handle directly
4. **APEX launch** → Orchestrator starts on **background thread** BEFORE planning
5. **LLM Planning** → `agent.run()` returns JSON array of research steps
6. **Step Execution** → For each step:
   - Display in Rich planning panel
   - `execute_plan()` runs `agent.action()` on a thread with progressive timeout messages
   - Lambda data injected automatically
   - `agent.summarize()` condenses results
7. **APEX Collection** → Wait up to 20s for Orchestrator results
8. **APEX Injection** → Orchestrator's full analysis injected as conversation context
9. **Answer Streaming** → `agent.answer()` streams markdown, displayed in Rich panel
10. **Token Tracking** → Claude API usage, cost estimate displayed in footer
11. **Learning** → If research/prediction query, `LearningHub.record_prediction()` called
12. **Cleanup** → Remove raw data messages from conversation to save tokens

#### APEX Dual-Brain Fusion:
- **Zero added latency**: Orchestrator runs on background thread while Manager's own planning executes
- **Parallel intelligence**: 9 agents + DecisionEngine + ML run concurrently with sequential planning
- **Result injection**: Orchestrator output injected before `agent.answer()` so Claude synthesizes BOTH paths
- **Panel title**: Shows "Answer — APEX Synthesis" when APEX data available

---

## 6. Agent Layer (LLM Planning)

### `src/nuble/agents/__init__.py` (51 lines)
**Module Export Map — Orchestrator + 9 Specialized Agents**

Exports 12 symbols organized into 3 groups:
1. **Base**: `SpecializedAgent` (abstract), `AgentType`, `AgentTask`, `AgentResult`, `TaskPriority`
2. **Orchestrator**: `OrchestratorAgent`, `OrchestratorConfig`, `ConversationContext`
3. **9 Specialized Agents**: `MarketAnalystAgent`, `QuantAnalystAgent`, `NewsAnalystAgent`, `FundamentalAnalystAgent`, `MacroAnalystAgent`, `RiskManagerAgent`, `PortfolioOptimizerAgent`, `CryptoSpecialistAgent`, `EducatorAgent`

### `src/nuble/agent/agent.py` (241 lines)

The LLM interface for planning, executing, and answering.

#### Methods:

| Method | Purpose |
|--------|---------|
| `run(messages)` | Returns JSON array of research steps (plan) |
| `action(question, title, description)` | Executes a single research step — fetches Lambda data, gets Claude analysis |
| `summarize(messages)` | Condenses conversation to <50 word summary |
| `answer(question, messages)` | Streams final answer with real-time Lambda data injection |
| `compact(messages)` | Compacts conversation to key bullet points (max 25) |
| `get_realtime_analysis(text)` | Extracts symbols → fetches Lambda analysis → formats for prompt injection |

#### Lambda Integration:
- `get_realtime_analysis()` extracts symbols using `extract_symbols()` from `lambda_client.py`
- Fetches up to 3 symbols' analyses
- Injects formatted institutional-grade context (from `format_analysis_for_context()`) into every action and answer

#### Polygon Fallback:
When Lambda is unavailable, `_get_market_data()` directly hits:
- `https://api.polygon.io/v2/aggs/ticker/{ticker}/prev` → OHLCV
- `https://api.polygon.io/v3/reference/tickers/{ticker}` → company info, market cap
- `https://api.polygon.io/v2/reference/news?ticker={ticker}` → recent news

### `src/nuble/agent/prompts.py` (231 lines)

All system prompts used by the Agent:

| Prompt | Purpose | Key Details |
|--------|---------|-------------|
| `agent_prompt` | Planning — decompose queries into research steps | Returns JSON array of `{title, description}` |
| `answer_prompt` | Final answer generation | 7-section structure: Direct Answer, Key Data, Decision Engine Verdict, Technical, Sentiment, Risk, Actionable Insight. APEX synthesis protocol for dual-brain convergence |
| `action_prompt` | Individual research step execution | Full data source documentation (Polygon, StockNews, CryptoNews, LuxAlgo, TENK) in formatted tables |
| `summary_prompt` | Conversation summarization | <50 words |
| `compact_prompt` | Conversation compaction | Max 25 key bullet points |
| `ml_prediction_prompt` | ML prediction format template | Direction, price table, regime, uncertainty |

---

## 7. Smart Router

### `src/nuble/router.py` (383 lines)

Regex-based intent detection to route queries without LLM.

#### Query Intents:

| Intent | Fast Path | Example |
|--------|-----------|---------|
| `QUOTE` | ✅ Yes | "AAPL", "TSLA price" |
| `TECHNICAL` | ✅ Yes | "RSI for AMD", "MACD TSLA" |
| `PREDICTION` | ✅ Yes | "predict AAPL", "ML forecast" |
| `PATTERN` | ✅ Yes | "patterns for SPY" |
| `SENTIMENT` | ✅ Yes | "sentiment for NVDA" |
| `FILINGS_SEARCH` | ✅ Yes | "risk factors AAPL" |
| `FILINGS_ANALYSIS` | ❌ No | "analyze AAPL 10-K" |
| `RESEARCH` | ❌ No | "Should I buy TSLA?" |
| `COMPARISON` | ❌ No | "AAPL vs MSFT" |
| `GENERAL` | ❌ No | Anything else |

#### Priority Order (checked first = wins):
1. RESEARCH (before PREDICTION to catch "should I buy" queries)
2. PREDICTION
3. TECHNICAL
4. PATTERN
5. SENTIMENT
6. FILINGS
7. COMPARISON
8. QUOTE (catches bare tickers)
9. GENERAL (default)

#### Symbol Extraction:
- `$AAPL` format (dollar prefix)
- Known tickers from `COMMON_TICKERS` set (120+ symbols)
- Context words: `{TICKER} stock/price/shares/quote`

---

## 8. Unified Services Layer

### `src/nuble/services.py` (950 lines)

Bridges three subsystems: `nuble/` (CLI), `institutional/` (Pro), `TENK_SOURCE/` (SEC).

#### Service Types:
| Service | Init Method | Sources |
|---------|-------------|---------|
| MARKET_DATA | `_init_market_data()` | Polygon, Alpha Vantage, Finnhub via Orchestrator |
| TECHNICAL | Always available | 50+ indicators (RSI, MACD, Bollinger, SMA, EMA, ATR) |
| FILINGS | `_init_filings()` | SEC EDGAR via `FilingsAnalyzer` + `FilingsSearch` with DuckDB |
| ML_PREDICTION | `_init_ml()` | Pre-trained model registry → RealTimePredictor fallback |
| SENTIMENT | Always available | Lexicon-based (FinBERT not loaded by default) |
| PATTERNS | Always available | Classical pattern recognition |

#### Key Methods:
- `get_quote(symbol)` → cached 60s, via institutional Orchestrator → Polygon
- `get_historical(symbol, days)` → OHLCV bars
- `get_technical_indicators(symbol, indicators)` → RSI, MACD, Bollinger, SMA, EMA, ATR
- `search_filings(query, symbol, form_types)` → DuckDB semantic search
- `analyze_filing(symbol, analysis_type, form)` → Claude-powered filing analysis
- `get_prediction(symbol, model_type)` → Pre-trained registry → RealTimePredictor
- `get_ensemble_prediction(symbol)` → Multi-model ensemble
- `detect_patterns(symbol)` → Classical chart patterns
- `get_sentiment(symbol)` → News sentiment aggregation

---

## 9. ROKET REST API (System B Frontend)

### `src/nuble/api/roket.py` (1,705 lines)

The FastAPI server exposing ALL NUBLE intelligence via REST. This is the primary API for frontend consumption.

#### Startup:
- Creates `FastAPI` app with CORS (`allow_origins=["*"]`)
- Mounts **LuxAlgo webhook router** (`POST /webhooks/luxalgo`, `GET /signals/*`)
- Also creates `APIRouter` prefix `/api/roket` for mounting in `server.py`
- Lazy-loads: `DataService`, `LivePredictor`, `WRDSPredictor`, `HMMRegimeDetector`

#### Complete Endpoint Reference:

| Method | Path | Description | Data Sources |
|--------|------|-------------|-------------|
| GET | `/api/health` | System health + component readiness + data freshness | DataService, WRDSPredictor, LivePredictor, HMM |
| GET | `/api/predict/{ticker}` | ML prediction (auto/live/wrds source selection) | LivePredictor → WRDSPredictor fallback |
| GET | `/api/universe` | All stocks with predictions (up to 5000) | WRDSPredictor |
| GET | `/api/regime` | Current macro regime | WRDSPredictor + HMM |
| GET | `/api/fundamentals/{ticker}` | All GKX features (522 columns) | WRDS panel |
| GET | `/api/earnings/{ticker}` | Earnings features (SUE, persistence, smoothness, accruals, etc.) | WRDS panel |
| GET | `/api/risk/{ticker}` | Risk profile (betas, vol, momentum, turnover) | WRDS panel |
| GET | `/api/insider/{ticker}` | Insider activity + analyst consensus | WRDS panel |
| GET | `/api/institutional/{ticker}` | Institutional ownership + market structure | WRDS panel |
| GET | `/api/analyze/{ticker}` | **COMPREHENSIVE** — combines ALL of the above + live intelligence + UDE + trade setup + signal fusion + veto + position sizing | ALL sources |
| POST | `/api/analyze` | Portfolio batch analysis (predict all holdings) | LivePredictor + WRDSPredictor |
| POST | `/api/screener` | Custom stock screening (score, tier, signal, market cap filters) | WRDSPredictor |
| GET | `/api/top-picks` | Top N stock picks (optional live re-scoring) | LivePredictor or WRDSPredictor |
| GET | `/api/tier/{tier}` | Tier-specific predictions (mega/large/mid/small) | WRDSPredictor |
| GET | `/api/model-info` | Model metadata + DataService status | WRDSPredictor |
| GET | `/api/news/{ticker}` | Real-time news + sentiment | StockNewsClient |
| GET | `/api/snapshot/{ticker}` | Real-time market snapshot (price, technicals, options, sentiment, regime, LuxAlgo) | DataAggregator |
| GET | `/api/sec-quality/{ticker}` | SEC EDGAR fundamental quality (40 ratios + composite score A-F) | SECEdgarXBRL |
| GET | `/api/macro` | FRED macro environment (yields, spreads, inflation, employment) | FREDMacroData |
| GET | `/api/lambda/{ticker}` | Lambda Decision Engine pass-through | NubleLambdaClient |
| GET | `/api/compare` | Side-by-side comparison (2-5 stocks) | LivePredictor + WRDSPredictor |
| POST | `/api/position-size` | Kelly criterion position sizing with stop/TP levels | LivePredictor + WRDSPredictor + Polygon |
| POST | `/webhooks/luxalgo` | TradingView LuxAlgo webhook receiver | TradingView |
| GET | `/signals/{symbol}` | Get stored LuxAlgo signals | DynamoDB/memory |

#### The `/api/analyze/{ticker}` Mega-Endpoint:
This is the "give me everything" endpoint. It returns 15 sections:
1. **prediction** → ML signal (live → WRDS fallback)
2. **fundamentals** → 50+ valuation/profitability/balance sheet features
3. **earnings** → SUE, persistence, accruals, P/E variants
4. **risk** → Betas (6 factors), volatility (15 metrics), momentum
5. **insider** → Buy ratio, CEO buy, cluster buy, analyst dispersion
6. **institutional** → Ownership change, HHI, breadth, R&D, capex
7. **regime** → HMM macro regime
8. **live_intelligence** → Polygon snapshot + SEC quality + Lambda decision + LuxAlgo + technicals
9. **ultimate_decision** → UDE 28+ data points (async, 15s timeout)
10. **trade_setup** → ATR-based entry/stop/TP with Keltner channels
11. **signal_fusion** → Multi-source signal blending
12. **veto_check** → Multi-timeframe institutional veto
13. **position_sizing** → Kelly criterion sizing
14. **execution_time_ms** → Total latency

#### Running Standalone:
```bash
PYTHONPATH="$PWD/src:$PWD" TOKENIZERS_PARALLELISM=false \
python3 -m uvicorn nuble.api.roket:app --host 0.0.0.0 --port 8000 --log-level warning
```

---

## 10. Lambda Decision Engine Client

### `src/nuble/lambda_client.py` (1,114 lines)

Production-grade client for the AWS Lambda Decision Engine.

#### Production Endpoint:
```
https://9vyvetp9c7.execute-api.us-east-1.amazonaws.com/production
```

#### Data Sources Aggregated by Lambda:
```
├── Polygon.io (Real-time): Price, OHLCV, technicals (RSI, MACD, ATR, Bollinger, SMA stack)
├── StockNews API (24 Endpoints): Sentiment, analyst ratings, earnings, SEC filings, events, trending
├── CryptoNews API (17 Endpoints): Crypto sentiment, whale tracking, institutional flows, regulatory
└── Derived Intelligence: Regime detection, cross-asset correlation, volatility classification
```

#### Key Data Structures:

**`LambdaAnalysis`** — The primary response dataclass:
- Core: `symbol`, `action` (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL), `score` (0-100), `confidence`
- Price: `current_price`, `change_percent`, `volume`
- Regime: `MarketRegime` enum (BULL/BEAR/VOLATILE/RANGING/CRISIS)
- Technicals: `TechnicalSnapshot` (RSI + signal + divergence, MACD + histogram + momentum, SMA 20/50/200, ATR + volatility regime, momentum 1/5/20D)
- Intelligence: `IntelligenceSnapshot` (sentiment score/label, news counts, analyst upgrades/downgrades, trending, whale activity, VIX)
- LuxAlgo: `luxalgo_weekly_action`, `luxalgo_daily_action`, `luxalgo_h4_action`, `luxalgo_aligned`, `luxalgo_score`, `luxalgo_valid_count`
- Veto: `veto` (bool), `veto_reason`

**`NubleLambdaClient`** — Production client:
- Connection pooling (10 connections)
- Retry with exponential backoff (3 retries, 1.5x backoff)
- Circuit breaker pattern (opens after 5 failures, resets after 60s)
- Health check caching (30s TTL)
- User-Agent: `NUBLE-CLI/6.0.0-APEX`

#### Key Functions:
- `get_lambda_client()` → Singleton instance
- `analyze_symbol(symbol)` → Quick analysis shortcut
- `format_analysis_for_context(analysis)` → Markdown for LLM prompt injection
- `extract_symbols(text)` → Regex symbol extraction from natural language
- `is_crypto(symbol)` → Check if crypto

#### LuxAlgo Signal Validation:
Lambda validates signal freshness — only actions with `valid=True` are shown. Stale signals (>162 hours old) are suppressed to avoid misleading data.

---

## 11. Ultimate Decision Engine (28+ Data Points)

### `src/nuble/decision/ultimate_engine.py` (1,663 lines)

Institutional-grade trading decision engine integrating ALL available resources.

#### Weighted Layer Architecture:
```
├── TECHNICAL SIGNALS (35%)
│   ├── LuxAlgo Multi-Timeframe (12%)
│   ├── ML Signal Generator - AFML (10%)
│   ├── Deep Learning LSTM/Transformer (8%)
│   └── Classic TA (RSI, MACD, BB) (5%)
├── INTELLIGENCE LAYER (30%)
│   ├── FinBERT Sentiment (8%)
│   ├── News Analysis (8%)
│   ├── HMM Regime Detection (7%)
│   └── Claude Reasoning (7%)
├── MARKET STRUCTURE (20%)
│   ├── Options Flow (6%)
│   ├── Order Flow / Dark Pool (5%)
│   ├── Macro Context (DXY, VIX) (5%)
│   └── On-Chain Crypto (4%)
└── VALIDATION (15%)
    ├── Historical Win Rate (6%)
    ├── Backtest Validation (5%)
    └── Pattern Similarity (4%)
```

#### Risk Layer (Veto Power):
- Max Position Check
- Drawdown Limit
- Correlation Check
- Liquidity Check
- News Blackout
- Earnings Window
- Conflicting Signals
- Regime Unfavorable
- Volatility Spike
- Stale Data

#### Output (`UltimateDecision`):
- `direction`: `TradeDirection` (LONG/SHORT/NEUTRAL)
- `strength`: `TradeStrength` (STRONG/MODERATE/WEAK/NO_TRADE)
- `confidence`: 0-1 scale
- `trade_setup`: `TradeSetup` with entry, stop_loss, targets, position_pct, risk_reward
- `risk_checks`: List of `RiskCheck` with pass/fail and veto flags
- Layer scores with component breakdowns
- Full reasoning chain

---

## 12. Signal Pipeline

### `src/nuble/signals/__init__.py` (114 lines)
**Module Export Map — 6 Signal Sources + MTF System**

Documents 6 signal source types and exports 25 symbols organized into 6 groups:
1. **LuxAlgo**: `LuxAlgoSignal`, `LuxAlgoSignalType`, `LuxAlgoSignalStore`, `parse_luxalgo_webhook`, `get_signal_store`
2. **Single-Source Fusion**: `SignalFusionEngine`, `FusedSignal`, `FusedSignalStrength`
3. **Base**: `SignalSource`, `NormalizedSignal` (abstract interface)
4. **Multi-Timeframe**: `TimeframeManager`, `TimeframeSignal`, `Timeframe`, `get_timeframe_manager`, `parse_mtf_webhook`
5. **Veto Engine**: `VetoEngine`, `VetoResult`, `VetoDecision`, `check_veto`
6. **Position Calculator**: `PositionCalculator`, `PositionSize`, `calculate_position`
7. **MTF Fusion Engine**: `MTFFusionEngine`, `TradingDecision`, `SignalStrength`, `get_mtf_engine`, `generate_mtf_decision`

### Signal Flow:
```
TradingView → LuxAlgo Webhook → Signal Store → Fusion Engine → Veto Engine → Position Calculator
                                                      ↑
                                              ML AFML + FinBERT + HMM
```

### `src/nuble/signals/luxalgo_webhook.py` (604 lines)
**LuxAlgo Signal Receiver & Store**

- Receives webhooks from TradingView LuxAlgo alerts
- Parses signal types: Buy/Sell Confirmations (1-12), Trend Tracer, Smart Trail, Neo Cloud, Trend Catcher
- `LuxAlgoSignal` dataclass: signal_id, timestamp, symbol, exchange, timeframe, signal_type, action, price, confirmations, trend_strength
- `LuxAlgoSignalStore`: In-memory signal store with DynamoDB backing
- Timeframe multipliers: 1m=0.40, 5m=0.50, 1h=0.80, 4h=0.95, 1D/1W/1M=1.00
- Strong signal: ≥4 confirmations on 4h+ timeframe

### `src/nuble/signals/fusion_engine.py` (793 lines)
**Signal Fusion Engine — Multi-Source Blending**

Base Weights:
- Technical (LuxAlgo): **50%**
- ML (AFML): **25%**
- Sentiment (FinBERT): **10%**
- Regime (HMM): **10%**
- Fundamental: **5%**

Key Rules:
- LuxAlgo strong + ML agrees → STRONG signal, boost confidence
- LuxAlgo + ML neutral → NORMAL signal
- LuxAlgo + ML disagrees → WEAK signal, reduce size
- No LuxAlgo → ML only, lower confidence

Output: `FusedSignal` with direction (-1/0/+1), strength (STRONG_BUY to STRONG_SELL), confidence, recommended_size, stop_loss_pct, take_profit_pct, source agreement metrics, reasoning

### `src/nuble/signals/veto_engine.py` (520 lines)
**Institutional Multi-Timeframe Veto**

Golden Rules:
1. **NEVER** trade against the Weekly trend
2. Daily must align with Weekly or reduce size by 75%
3. 4H triggers entry only after Weekly + Daily alignment
4. Conflicting signals = NO TRADE

Position Multipliers:
- Full alignment: 1.0
- Weekly + Daily only: 0.75
- Weekly neutral: 0.50
- Counter-trend: 0.25
- Conflict: 0.0

Freshness Thresholds: Weekly ≥ 30%, Daily ≥ 40%, 4H ≥ 50%

Output: `VetoResult` with decision (APPROVED/APPROVED_REDUCED/WAITING/VETOED), position_multiplier, direction, reason, details

### `src/nuble/signals/position_calculator.py` (544 lines)
**Institutional Position Calculator — Modified Kelly Criterion**

Philosophy: Size positions based on EDGE, not conviction. Never risk more than 2% per trade. Full position only on perfect setups. Scale in, don't go all-in.

`PositionSize` dataclass:
- `recommended_size` (0-1 fraction), `dollar_amount`, `shares`
- `stop_loss_price`, `stop_loss_pct`, `take_profit_prices` [TP1/TP2/TP3], `risk_reward_ratio`
- `confidence`, `alignment_score`, `kelly_fraction`
- Full `breakdown` dict with all calculation components

`PositionCalculator` class:
- **Risk parameters**: MAX_RISK_PER_TRADE = 2%, MAX_POSITION_SIZE = 10%, KELLY_FRACTION = 0.5 (half-Kelly)
- **ATR multipliers**: Stop = 2.0×ATR, TP1 = 2.0×ATR (1:1 R/R), TP2 = 4.0×ATR (2:1), TP3 = 6.0×ATR (3:1)

Key Methods:
- `calculate_kelly(win_rate, win_loss_ratio)` → Kelly = (p×b − q) / b, clamped ≥ 0
- `calculate_alignment_score(weekly, daily, four_hour, direction)` → 0-1 score with breakdown:
  - Weekly: 40% contribution, 20% penalty if opposed, 1.2× boost if strong
  - Daily: 35% contribution, 1.15× boost if strong, 1.2× bonus if aligned with weekly
  - 4H: 25% contribution, confirmations ≥8 boost (+3% per), 1.15× bonus if aligned with daily
- `estimate_win_rate(alignment_score, signals)` → Conservative estimate: base 45% + up to 15% from alignment + 2% per strong signal + 3% for 10+ confirmations. Capped at 35%-65%
- `calculate_stops(price, direction, atr)` → Uses Smart Trail as stop if tighter than ATR-based stop
- `calculate_position(veto_result, price, portfolio_value, atr, regime)`:
  1. Kelly fraction → half-Kelly for safety
  2. × veto multiplier (0-1)
  3. × regime multiplier (BULL=1.1, BEAR=0.9, SIDEWAYS=0.8, VOLATILE=0.6)
  4. Cap at MAX_POSITION_SIZE
  5. Risk-based cap: position ≤ max_risk / stop_pct
  6. Calculate dollar amount and shares
- `calculate_scaling_plan(position, max_adds=2)` → Pyramiding: 50% initial + 30% at 50%-to-TP1 + 20% at TP1

### `src/nuble/signals/timeframe_manager.py` (765 lines)
**Multi-Timeframe Signal Manager — Storage, Freshness, Alignment**

`Timeframe` enum with embedded properties:
| Timeframe | Weight | Max Age | Decay Start | LuxAlgo Sensitivity |
|-----------|--------|---------|-------------|---------------------|
| WEEKLY (1W) | 40% | 168h (7d) | 126h | 22 |
| DAILY (1D) | 35% | 24h | 18h | 15 |
| FOUR_HOUR (4h) | 25% | 8h | 6h | 11 |
| ONE_HOUR (1h) | 0% (tuning only) | 2h | 1.5h | 9 |

Supports parsing: "1W", "W", "WEEKLY", "240" (TradingView minutes format), etc.

`TimeframeSignal` dataclass:
- Core: symbol, timeframe, timestamp, direction (-1/0/+1), action, strength, confirmations (1-12)
- LuxAlgo indicators: trend_strength (0-100), smart_trail_sentiment/level, neo_cloud_sentiment, reversal zones, ml_classification
- Computed properties:
  - `freshness` → Decay curve: 1.0→0.5 linear over 75% of max_age, then 0.5→0.0 accelerated
  - `is_fresh` → freshness ≥ 30%
  - `is_strong` → strength=="strong" OR confirmations ≥ 8
  - `weighted_direction` → direction × freshness × strength bonus × confirmation bonus × trend strength bonus
  - `confidence` → 0-100% from freshness (50%) + strength (15%) + confirmations (25%) + indicator agreement (10%)
- `from_webhook(payload)` → Robust webhook parsing supporting multiple payload formats, timestamp formats, nested structures

`TimeframeManager` class:
- Thread-safe with `threading.Lock()`, per-symbol signal storage
- `add_signal(signal)` → Store current, move old to history (max 100 per symbol/timeframe)
- `get_cascade(symbol)` → Tuple of (weekly, daily, 4h, 1h) signals
- `get_alignment(symbol)` → Weighted alignment score with weekly veto: perfect=1.0, conflicting=0.3, counter-weekly=×0.25 penalty
- `get_status()` → All tracked symbols with alignment
- `cleanup_expired()` → Remove expired signals, move to history
- `to_json()` → Full serialization of all current signals
- Global singleton: `get_timeframe_manager()`

### `src/nuble/signals/mtf_fusion.py` (477 lines)
**MTFFusionEngine — Complete Trading Decision Generator**

The brain that combines TimeframeManager + VetoEngine + PositionCalculator into a single `TradingDecision`.

`TradingDecision` dataclass:
- `can_trade`, `action` (BUY/SELL/HOLD), `direction`, `strength` (NONE→VERY_STRONG), `confidence`
- `position` (PositionSize), `entry_price`, `veto_result`
- Per-timeframe summaries (weekly/daily/4h/1h)
- `reasoning` list, `action_label` with emoji (📈 🟢 STRONG BUY)

`MTFFusionEngine`:
- Init: `portfolio_value=100K`, `max_risk=2%`, `max_position=10%`
- `generate_decision(symbol, current_price, atr, regime, portfolio_value)`:
  1. Get cascade signals (weekly/daily/4h/1h) from TimeframeManager
  2. Apply VetoEngine → if vetoed, return HOLD with reason
  3. Calculate PositionSize via PositionCalculator (Kelly + alignment + regime)
  4. Determine action (BUY/SELL) from veto direction
  5. Calculate strength from confidence × position size
  6. Track decision in history (last 1,000)
- `add_signal(signal)` → Add to TimeframeManager
- `add_from_webhook(payload)` → Parse and add
- Global singleton: `get_mtf_engine(portfolio_value)`
- Convenience: `generate_mtf_decision(symbol, price, regime)` one-liner

### `src/nuble/signals/base_source.py` (251 lines)
**Base Signal Source — Abstract Base for All Signal Sources**

`SignalDirection` enum: STRONG_BUY (+2), BUY (+1), NEUTRAL (0), SELL (-1), STRONG_SELL (-2)

`NormalizedSignal` dataclass:
- All signals normalized to [-1, +1] scale for fusion
- `direction` (continuous -1 to +1), `confidence` (0-1)
- Properties: `is_bullish` (>0.1), `is_bearish` (<-0.1), `strength` (|direction|), `weighted_signal` (direction × confidence)

`SignalSource` ABC:
- Must implement: `generate_signal(symbol, data, context)` → NormalizedSignal, `get_confidence()` → 0-1
- Accuracy tracking: `update_accuracy(was_correct)`, `get_recent_accuracy(lookback=20)` with rolling window of 100
- Dynamic weights: `get_weight()`, `set_weight()`, enable/disable
- Default `base_weight = 0.15`

`CompositeSignalSource`:
- Combines multiple sub-sources with weighted average
- `generate_signal()` → Weighted average of all enabled sub-source directions and confidences

### Signal Sources (`src/nuble/signals/sources/`)

| File | Lines | Source Name | Base Weight | Description |
|------|-------|------------|-------------|-------------|
| `ml_afml.py` | 195 | "ml" | 25% | Uses AFML `EnhancedSignalGenerator` — multi-timeframe momentum, regime-adaptive, mean reversion. Lazy-loads from `institutional.signals.enhanced_signals`. Needs ≥70 rows OHLCV. |
| `regime_hmm.py` | 281 | "regime" | 10% | Wraps `HMMRegimeDetector` — 3-state regime classification. Returns +0.3 for BULL, 0 for NEUTRAL, -0.5 for CRISIS regime direction. Confidence from HMM state probability. |
| `sentiment_finbert.py` | 236 | "sentiment" | 10% | Wraps news sentiment pipeline. Fetches StockNews headlines → FinBERT analysis → aggregated score (-1 to +1). Uses SharedDataLayer for caching. |
| `technical_luxalgo.py` | 174 | "luxalgo" | 50% | Wraps LuxAlgoSignalStore. Highest-timeframe signal wins. Confidence from freshness × confirmations. Direction from most recent strong signal. |

---

## 13. Multi-Agent Orchestrator (APEX Tier)

### `src/nuble/agents/orchestrator.py` (1,733 lines)

The master brain of the APEX system, powered by Claude Sonnet 4.

#### Architecture Flow:
1. **Intent Understanding** — Parse user query
2. **Decision Engine Check** — UltimateDecisionEngine for trading decisions
3. **Lambda API Data Fetch** — Real-time market intelligence
4. **Task Planning** — Decompose into agent tasks
5. **Parallel Execution** — Run agents concurrently (max 5 parallel)
6. **Enrichment** — EnrichmentEngine adds statistical context
7. **Trade Setup** — TradeSetupCalculator computes entry/stop/targets
8. **Learning Context** — LearningHub provides track record
9. **Result Synthesis** — Combine all outputs
10. **Response Generation** — Claude Sonnet 4 final synthesis

#### Integration Points:
- **UltimateDecisionEngine**: 28+ data points, weighted scoring, risk veto
- **LivePredictor**: Polygon live → LightGBM → composite score
- **WRDSPredictor**: WRDS historical fallback
- **Lambda Client**: 40+ real-time API endpoints
- **EnrichmentEngine**: Percentile ranks, z-scores, anomaly detection
- **TradeSetupCalculator**: ATR-based entry/stop/TP
- **LearningHub**: Prediction tracking + weight adjustment

#### Config (`OrchestratorConfig`):
```python
use_opus: bool = True               # Use Opus for orchestration
max_parallel_agents: int = 5        # Max concurrent agents
default_timeout: int = 30           # Agent timeout
enable_decision_engine: bool = True # UDE
enable_ml_predictor: bool = True    # ML Predictor
```

#### Deep Dive — `process()` Full Pipeline (1,733 lines):

The `process()` method implements the complete APEX pipeline in this exact order:

1. **Symbol Extraction** — Parse symbols from user query + conversation context
2. **Lambda Data Fetch** — `NubleLambdaClient.analyze(symbol)` for each symbol (up to 3)
3. **ML Predictor Chain** — LivePredictor → WRDSPredictor → v2 (DEPRECATED) → v1 (DEPRECATED)
4. **UDE Decision** — If trading query, `UltimateDecisionEngine.make_decision()` with 28+ data points
5. **Planning** — Claude Opus `_plan_execution()` decomposes query into agent tasks (JSON)
   - `_build_planning_prompt()` includes: portfolio, risk tolerance, recent symbols, all 9 agent descriptions, LuxAlgo/TENK availability notes
   - `_simple_plan()` is rule-based fallback (keyword matching) if Claude planning fails
6. **SharedDataLayer Prefetch** — `await shared.prefetch(symbols, agent_types)` fires ALL needed API calls in parallel BEFORE agents run
7. **Task Execution** — `_execute_tasks()` with topological sort for dependencies, `asyncio.gather()` with timeout per agent
8. **Enrichment** — `EnrichmentEngine` adds percentile ranks, z-scores, anomaly detection to agent outputs
9. **Trade Setup** — `TradeSetupCalculator` computes BOTH LONG and SHORT setups (ATR-based entry/stop/TP)
10. **Learning Feedback** — `LearningHub.get_learning_context(symbol)` injects track record
11. **Response Generation** — Claude Sonnet 4 synthesizes ALL outputs into final markdown
12. **Learning Recording** — `LearningHub.record_prediction()` with `signal_snapshot` containing:
    - `technical_score`, `intelligence_score`, `risk_score`
    - All ML predictions, agents used, decision engine verdict
13. **SharedDataLayer Cleanup** — `shared.cleanup()` releases async resources
14. **Agent Output Tracking** — `_last_agent_outputs` stored for API progress events

#### Planning Prompt Architecture (`_build_planning_prompt()`):

The planning prompt sent to Claude Opus includes:
- **User Profile**: Portfolio holdings, risk tolerance
- **Market Context**: Recent symbols discussed, ML predictions, Lambda data
- **Agent Registry**: Full descriptions of all 9 agents with their capabilities and data sources
- **Orchestration Rules**: Max parallel agents, timeout constraints, dependency ordering
- **Special Capabilities Notes**: LuxAlgo availability, TENK RAG availability, FRED macro availability

#### SharedDataLayer Integration:
```python
# Before agents run — prefetch ALL data in parallel
await shared.prefetch(symbols, ['market_analyst', 'fundamental_analyst', 'risk_manager', ...])
# This fires ~25-80 parallel HTTP calls via aiohttp, completing in 1-3 seconds
# After this, ALL agents read from cache with ZERO additional HTTP calls
```

### `src/nuble/agents/base.py` (186 lines)
Base class for all agents. Lazy-loads Anthropic. Uses `claude-sonnet-4-20250514`.

### 9 Specialist Agents (in `src/nuble/agents/`):

| Agent | File | Lines | Expertise | Data Sources |
|-------|------|-------|-----------|-------------|
| MarketAnalyst | `market_analyst.py` | 1,159 | Real-time price action, technicals, volume analysis | Polygon, StockNews |
| NewsAnalyst | `news_analyst.py` | 868 | News sentiment, breaking news, event detection | StockNews PRO, CryptoNews PRO, Polygon, Alternative.me |
| RiskManager | `risk_manager.py` | 570 | Portfolio risk, drawdown limits, correlation | Polygon, VIX, Alternative.me FGI, StockNews |
| FundamentalAnalyst | `fundamental_analyst.py` | 843 | SEC filings (TENK RAG), valuations, earnings | Polygon Financials, SEC EDGAR XBRL, TENK RAG |
| QuantAnalyst | `quant_analyst.py` | 682 | ML signals, factor models, backtesting | LivePredictor, HMM Regime, Polygon |
| MacroAnalyst | `macro_analyst.py` | 652 | FRED data, yields, inflation, sector rotation | Polygon (21 ETFs), FRED, Alternative.me, StockNews |
| PortfolioOptimizer | `portfolio_optimizer.py` | 468 | Position sizing, asset allocation, rebalancing | Polygon Historical/SMA/Dividends/Price |
| CryptoSpecialist | `crypto_specialist.py` | 654 | Whale tracking, DeFi, regulatory, on-chain | Polygon, CoinGecko, Alternative.me, CryptoNews |
| Educator | `educator.py` | 160 | Explains concepts, teaches strategies | Claude Sonnet 4 |

---

## 14. Specialist Agents — Deep Dive

### `src/nuble/agents/market_analyst.py` (1,159 lines)
**MarketAnalystAgent — Real-Time Price Action Expert**

Data Sources:
- **Polygon.io**: Quotes, 90-day historical OHLCV, server-side RSI/SMA/EMA/MACD
- **StockNews API**: Sentiment analysis (/stat for 7-day quantitative), analyst ratings, earnings calendar

Core Method — `_analyze_symbol(symbol)`:
1. Get real-time quote via SharedDataLayer → Polygon `/v2/aggs/ticker/{symbol}/prev`
2. Fetch 90-day historical OHLCV from Polygon
3. Calculate 50+ technical indicators:
   - RSI(14), MACD(12,26,9), Bollinger Bands(20,2), Stochastic(14,3)
   - SMA(20/50/200), EMA(12/26), ATR(14), OBV, VWAP
   - Rate of Change, Williams %R, CCI, ADX
4. Detect candlestick patterns (hammer, engulfing, doji, morning/evening star, etc.)
5. Calculate support/resistance levels from price pivots
6. Trend analysis (SMA crossovers, ADX strength, momentum)
7. **StockNews sentiment**: Aggregates 7-day sentiment stats + article-level sentiment scoring
8. **Analyst ratings**: Latest upgrades/downgrades with price targets
9. **Earnings calendar**: Upcoming earnings dates for event risk

Sentiment Integration — `_get_stocknews_sentiment()`:
- Uses SharedDataLayer first (cache hit), falls back to direct HTTP
- StockNews `/stat` endpoint: `positive/negative/neutral` counts over 7 days
- Computes normalized sentiment score (-1.0 to +1.0)

### `src/nuble/agents/news_analyst.py` (868 lines)
**NewsAnalystAgent — ELITE Sentiment & Event Expert (FULL PREMIUM)**

This is the most data-rich agent in the system. It calls ALL premium endpoints:

**StockNews PRO Endpoints Used (10 endpoints)**:
1. `/api/v1` — Ticker news with sentiment labels and rank scores
2. `/api/v1/stat` — Quantitative sentiment analysis over time (7d/30d)
3. `/api/v1/top-mention` — Most mentioned stocks with sentiment
4. `/api/v1/events` — Breaking news events with event IDs
5. `/api/v1/trending-headlines` — Top trending headlines
6. `/api/v1/earnings-calendar` — Upcoming earnings dates
7. `/api/v1/ratings` — Analyst upgrades/downgrades with price targets
8. `/api/v1/sundown-digest` — Daily market summary
9. `/api/v1/category?section=general` — General market news (Fed, CPI, etc.)
10. `/api/v1?topic=earnings` — Topic-filtered news

**CryptoNews PRO Endpoints Used (7 endpoints)**:
1. `/api/v1` — Ticker news with crypto sentiment
2. `/api/v1/stat` — Crypto sentiment over time
3. `/api/v1/top-mention` — Most mentioned coins
4. `/api/v1/events` — Breaking crypto events
5. `/api/v1/trending-headlines` — Top crypto headlines
6. `/api/v1/category?section=general` — General crypto market news
7. `/api/v1/sundown-digest` — Daily crypto summary

**Additional Data Sources**:
- Alternative.me Fear & Greed Index (7-day trend)
- Polygon VIX — Volatility context for sentiment calibration
- Polygon News — Rich article data with publisher, tickers, description

**Output Structure**: Returns `symbol_news` (per-ticker with articles, sentiment, quantitative_sentiment, events) + `market_sentiment` + `trending_headlines` + `top_mentioned_tickers` + `breaking_events` + `general_market_news` + `earnings_calendar` + `analyst_ratings` + `sundown_digest` + crypto-specific premium data

### `src/nuble/agents/risk_manager.py` (570 lines)
**RiskManagerAgent — Portfolio Risk & Volatility Expert**

6 Real Data Sources:
1. Polygon Historical → actual returns for VaR calculation
2. Polygon VIX → volatility regime monitoring
3. Polygon Sectors → correlation with sector indices
4. Polygon News → event risk detection
5. Alternative.me → Fear & Greed Index
6. StockNews → negative news / risk events

Risk Calculations (all from real data):
- **VaR (Value at Risk)**: Historical Simulation at 95% and 99% confidence + 10-day VaR
- **CVaR (Expected Shortfall)**: Mean of losses beyond VaR threshold
- **Parametric VaR**: Monte Carlo simulation with fitted normal distribution
- **Correlation Analysis**: Correlation matrix from actual Polygon returns vs SPY/QQQ
- **Drawdown Analysis**: Maximum drawdown from actual price history
- **Volatility Regime**: VIX-based (CRISIS >35, HIGH_VOL >25, NORMAL >20, LOW_VOL >15, VERY_LOW <15)

### `src/nuble/agents/fundamental_analyst.py` (843 lines)
**FundamentalAnalystAgent — Valuation & Financial Statement Expert**

8 Data Sources:
1. Polygon Financials API — `/vX/reference/financials` (income, balance, cashflow)
2. Polygon Company Details — `/v3/reference/tickers/{ticker}`
3. Polygon Dividends — `/v3/reference/dividends`
4. Polygon News — `/v2/reference/news`
5. Polygon Previous Close — OHLCV
6. Polygon SMA — Server-side moving averages
7. StockNews API — Supplementary sentiment
8. **TENK SEC Filing RAG** — DuckDB + sentence-transformers (384-dim embeddings)

Financial Statement Parsing (from Polygon vX/reference/financials):
- **Profitability**: gross_margin, operating_margin, net_margin, ROE, ROA, ROIC
- **Income Statement**: revenue, gross_profit, operating_income, net_income, EPS (basic/diluted)
- **Balance Sheet**: total_assets, total_liabilities, equity, cash, current_ratio, quick_ratio, D/E, net_debt
- **Cash Flow**: operating_cf, investing_cf, financing_cf, FCF, FCF margin, capex
- **TTM Calculations**: Trailing twelve months from last 4 quarterly reports

TENK RAG Integration:
- Loads SEC 10-K/10-Q filings into DuckDB
- Sentence-transformers create 384-dimensional embeddings
- Semantic search finds relevant sections (risk factors, revenue breakdown, management outlook)
- Claude analyzes matched filing sections

### `src/nuble/agents/quant_analyst.py` (682 lines)
**QuantAnalystAgent — ML Signals & Factor Models**

Production ML Integration:
- **LivePredictor**: Lazy-loaded production ML models (LSTM 3.2M params, Transformer 5.8M params, Ensemble)
- **HMM Regime Detector**: 3-state regime classification from GKX panel
- **Factor Exposures**: Computed from real Polygon data

Signal Generation — `_generate_signals(symbol)`:
1. Try real ML predictor (`predictor.predict(symbol)`) → direction, confidence, predictions, regime
2. Fallback: Technical analysis from Polygon indicators (RSI, SMA, MACD, EMA)

News Sentiment as Factor — `_get_news_sentiment_factor()`:
- StockNews `/stat` for quantitative 7-day sentiment score
- StockNews ticker news with sentiment label aggregation
- Earnings calendar check (upcoming earnings = risk penalty -0.1)
- Returns normalized factor score (-1.0 to +1.0)

Factor Analysis — `_analyze_factors()`:
- Value: P/E ratio from Polygon financials
- Momentum: Multi-period returns (7d, 30d, 90d)
- Quality: Margin stability, earnings consistency
- Size: Market cap decile
- Volatility: Realized vol vs sector

### `src/nuble/agents/macro_analyst.py` (652 lines)
**MacroAnalystAgent — ELITE Macroeconomic Analysis**

10 Real Data Sources:
1. Polygon VIX — Volatility regime
2. Polygon 11 Sector ETFs (XLK, XLV, XLF, XLE, XLY, XLP, XLI, XLB, XLU, XLRE, XLC) — Sector rotation
3. Polygon Treasury proxies (SHY=1-3yr, IEF=7-10yr, TLT=20+yr) — Yield curve shape
4. Polygon Dollar (UUP) — Currency risk
5. Polygon Commodities (GLD, USO) — Inflation/safe-haven demand
6. Polygon Market Indices (SPY, QQQ, IWM, DIA) — Breadth
7. Polygon News — Macro headlines & event risk
8. Polygon SMA (50-day for all ETFs) — Sector trend scoring
9. Alternative.me Fear & Greed — Sentiment overlay
10. **FRED Macro Data** (Phase 1 upgrade) — Real treasury yields, credit spreads, industrial production

Analysis Modules:
- `_analyze_volatility()` → VIX regime (CRISIS/HIGH_VOL/NORMAL/LOW_VOL/VERY_LOW)
- `_get_sector_performance()` → All 11 sectors with SMA50 trend, risk-on/risk-off rotation signal, sectors above SMA50 count
- `_analyze_market_breadth()` → SPY/QQQ/IWM/DIA advancing/declining, breadth signal (STRONG_BULLISH to STRONG_BEARISH)
- `_analyze_yield_curve()` → SHY/IEF/TLT proxy analysis, curve steepness, inversion detection
- `_analyze_dollar()` → UUP dollar strength, impact on equities
- `_analyze_commodities()` → GLD (gold) and USO (oil) as inflation/recession signals
- `_get_macro_news()` → Macro-relevant Polygon news headlines
- `_get_sentiment()` → Alternative.me Fear & Greed (7-day values)
- **FRED Integration**: `FREDMacroData.get_current()` → real term spread, credit spread, monetary policy regime, yield curve regime

### `src/nuble/agents/portfolio_optimizer.py` (468 lines)
**PortfolioOptimizerAgent — ELITE Allocation Expert**

5 Real Data Sources:
1. Polygon Historical — Actual returns, volatility, correlation, Sharpe, beta
2. Polygon SMA — Trend overlay (above/below SMA50 for momentum tilt)
3. Polygon Dividends — Dividend yield, frequency, growth rate
4. Polygon News — Sentiment-adjusted allocation
5. Polygon Previous Close — Live valuation & drift detection

Optimization Methods:
- `_analyze_current()` → Current holdings with 30d performance, annualized vol, concentration (HHI index)
- `_optimize()` → Two optimization approaches:
  - **Inverse-Volatility Risk Parity**: Weight proportional to 1/volatility
  - **Max-Sharpe Heuristic**: Tilt towards higher Sharpe ratio assets
- `_trend_overlay()` → SMA50 trend for each symbol → OVERWEIGHT/UNDERWEIGHT/NEUTRAL allocation tilt
- `_dividend_analysis()` → Annual yield, quarterly amounts, dividend growth rate
- `_recommend_trades()` → Specific rebalancing trade suggestions
- `_project_metrics()` → Expected portfolio return, vol, Sharpe after rebalancing

### `src/nuble/agents/crypto_specialist.py` (654 lines)
**CryptoSpecialistAgent — ELITE Crypto & DeFi Expert**

6 Real Data Sources:
1. **Polygon.io** — Price, volume, VWAP for crypto pairs (X:BTCUSD, X:ETHUSD, etc.)
2. **Polygon Indicators** — Server-side RSI, SMA, MACD for crypto tickers
3. **CoinGecko /coins/{id}** — ATH, ATL, supply, market cap rank, price changes (24h/7d/30d/1y)
4. **CoinGecko /global** — Total market cap, BTC/ETH dominance, total volume
5. **Alternative.me** — Crypto Fear & Greed Index (7-day trend)
6. **CryptoNews API** — Crypto-specific news with sentiment labels

Crypto Ticker Mapping:
```python
CRYPTO_TICKERS = {
    'BTC': 'X:BTCUSD', 'ETH': 'X:ETHUSD', 'SOL': 'X:SOLUSD',
    'XRP': 'X:XRPUSD', 'ADA': 'X:ADAUSD', 'DOGE': 'X:DOGEUSD',
    'AVAX': 'X:AVAXUSD', 'DOT': 'X:DOTUSD', 'MATIC': 'X:MATICUSD',
    'LINK': 'X:LINKUSD', 'UNI': 'X:UNIUSD', 'AAVE': 'X:AAVEUSD',
    'ATOM': 'X:ATOMUSD', 'NEAR': 'X:NEARUSD',
}
```

Analysis Modules:
- `_get_market_overview()` → CoinGecko global: total market cap ($T), BTC/ETH dominance, volume, FGI
- `_get_price_data()` → Polygon live OHLCV + change% for each crypto
- `_get_coin_fundamentals()` → CoinGecko: market cap rank, circulating/total/max supply, ATH/ATL, multi-period changes
- `_get_crypto_technicals()` → Polygon server-side RSI/SMA/MACD + historical returns analysis
- `_get_crypto_sentiment()` → CryptoNews + Alternative.me Fear & Greed
- `_get_defi_overview()` → CoinGecko DeFi TVL data

### `src/nuble/agents/educator.py` (160 lines)
**EducatorAgent — Financial Education Expert**

Uses Claude Sonnet 4 to generate educational content:
- `_generate_explanation(query)` → Structured JSON with `simple`, `detailed`, `example`, `misconceptions`
- `_get_related_topics(query)` → Topic-mapped related subjects (e.g., RSI → MACD, Stochastic, Divergence)
- `_get_resources(query)` → Learning resources at beginner/intermediate/advanced levels
- `_suggest_next_steps(query)` → Actionable next learning steps

Fallback: When Claude is unavailable, returns template-based explanations.

---

## 15. SharedDataLayer — Async Data Cache

### `src/nuble/agents/shared_data.py` (505 lines)
**The Performance Multiplier — Async-Native Shared Cache**

This is the critical optimization that makes 9 agents running in parallel performant. Without it, 9 agents × 10-15 API calls each = 90-135 sequential HTTP calls. With SharedDataLayer, this becomes ~25 unique parallel calls.

#### Architecture:
```
Agent 1 (MarketAnalyst)  ──┐
Agent 2 (RiskManager)    ──┤                    ┌── Polygon.io
Agent 3 (NewsAnalyst)    ──┼── SharedDataLayer ──┼── StockNews API
Agent 4 (Quant)          ──┤    (aiohttp)       ├── CryptoNews API
Agent 5 (Fundamental)    ──┤    per-key Lock    ├── CoinGecko
Agent 6 (Macro)          ──┤    TTL cache       ├── Alternative.me
Agent 7 (Portfolio)      ──┤                    └── (all via aiohttp)
Agent 8 (Crypto)         ──┤
Agent 9 (Educator)       ──┘
```

#### Connection Pool:
```python
aiohttp.TCPConnector(limit=20, limit_per_host=10)
timeout = aiohttp.ClientTimeout(total=15)
```

#### Per-Key Lock Deduplication:
If two agents request the same data simultaneously, only ONE HTTP call is made:
```python
async def _fetch_with_cache(self, key, fetcher, ttl):
    if key in self._cache and not expired:
        return cached_value  # Instant
    
    async with self._locks[key]:  # Per-key lock
        if key in self._cache and not expired:
            return cached_value  # Second checker won race
        result = await fetcher()  # Only ONE HTTP call
        self._cache[key] = CacheEntry(result, ttl)
        return result
```

#### TTL by Data Type:
| Data Type | TTL | Rationale |
|-----------|-----|-----------|
| Quotes | 30s | Fresh price data needed |
| Historical OHLCV | 300s | Doesn't change intraday |
| Company Info | 3600s | Rarely changes |
| News | 120s | New articles frequently |
| Financials | 3600s | Quarterly updates |
| Fear & Greed | 300s | Updates daily |
| Crypto Sentiment Stats | 300s | Updates hourly |

#### Complete Endpoint Inventory (35+ endpoints):

**Polygon Endpoints:**
- `get_quote(symbol)` → `/v2/aggs/ticker/{symbol}/prev`
- `get_historical(symbol, days)` → `/v2/aggs/ticker/{symbol}/range/1/day/...`
- `get_sma(symbol, window)` → `/v1/indicators/sma/{symbol}`
- `get_rsi(symbol)` → `/v1/indicators/rsi/{symbol}`
- `get_ema(symbol, window)` → `/v1/indicators/ema/{symbol}`
- `get_macd(symbol)` → `/v1/indicators/macd/{symbol}`
- `get_company_info(symbol)` → `/v3/reference/tickers/{symbol}`
- `get_financials(symbol, type, limit)` → `/vX/reference/financials`
- `get_dividends(symbol)` → `/v3/reference/dividends`
- `get_polygon_news(symbol)` → `/v2/reference/news`

**StockNews Endpoints (11):**
- `get_stocknews(symbol)` — Ticker news with sentiment
- `get_stocknews_negative(symbol)` — Negative sentiment articles
- `get_stocknews_stat(symbol, period)` — Quantitative sentiment stats
- `get_stocknews_ratings()` — Analyst ratings
- `get_stocknews_earnings()` — Earnings calendar
- `get_stocknews_events()` — Breaking events
- `get_stocknews_trending()` — Trending headlines
- `get_stocknews_top_mentioned()` — Most mentioned tickers
- `get_stocknews_sundown()` — Daily digest
- `get_stocknews_category(section)` — Category news
- `get_stocknews_topic(symbol, topic, items)` — Topic-filtered news

**CryptoNews Endpoints (7):**
- `get_cryptonews(symbol)` — Crypto ticker news
- `get_cryptonews_stat(symbol, period)` — Crypto sentiment stats
- `get_cryptonews_trending()` — Trending crypto headlines
- `get_cryptonews_top_mentioned()` — Most mentioned coins
- `get_cryptonews_events()` — Crypto events
- `get_cryptonews_category(section)` — Category crypto news
- `get_cryptonews_sundown()` — Crypto daily digest

**Market-Wide Endpoints:**
- `get_fear_greed()` → Alternative.me Fear & Greed Index
- `get_vix()` → VIX via Polygon
- `get_coingecko_global()` → CoinGecko global crypto data
- `get_coingecko_coin(coin_id)` → CoinGecko coin details
- `get_coingecko_defi()` → CoinGecko DeFi overview

#### Prefetch Strategy — `prefetch(symbols, agent_types)`:
Fires ALL needed requests in a single `asyncio.gather()`:
- **Always**: quote + 90d historical for each symbol + VIX + Fear & Greed
- **Per symbol**: RSI + SMA(20/50/200) + EMA(12/26) + MACD (7 technical indicator calls)
- **If StockNews agents**: 8 additional calls per symbol + 4 market-wide calls
- **If fundamental**: company_info + financials(quarterly,5) + financials(annual,2) + dividends + news (5 calls per symbol)
- **If macro**: 21 ETF quotes + 21 ETF SMA50 calls (42 total for sector/bond/commodity/dollar analysis)
- **If crypto**: CoinGecko global + DeFi + 7 CryptoNews calls + per-symbol crypto news/stats

Stats tracking: `_fetch_count`, `_cache_hit_count`, `hit_rate`

---

## 16. Elite API Server (SSE + WebSocket)

### `src/nuble/api/server.py` (1,144 lines)
**Production API Server with Real-Time Streaming**

This is the full production server that mounts all API routers and provides SSE streaming and WebSocket support for the frontend.

#### Server Architecture:
```
FastAPI App (server.py)
├── GET  /api/health              → System health
├── GET  /api/status              → Component status
├── POST /api/chat (SSE STREAM)   → Main chat endpoint
├── POST /api/chat/sync           → Synchronous chat
├── GET  /api/quote/{symbol}      → Quick quote
├── GET  /api/lambda/{symbol}     → Lambda pass-through
├── GET  /api/luxalgo/{symbol}    → LuxAlgo signals
├── WS   /ws/chat                 → WebSocket chat
├── GET  /api/learning/stats      → Learning statistics
├── GET  /api/learning/predictions → All predictions
├── DEL  /api/conversation/{id}   → Delete conversation
├── Mount: intel_router           → /api/intel/* (intelligence.py)
├── Mount: roket_router           → /api/roket/* (roket.py)
├── Mount: tool_router            → /api/intel/chat-with-tools
└── Mount: luxalgo_router         → /webhooks/luxalgo
```

#### SSE Streaming — `POST /api/chat` (Main Endpoint):
This implements Server-Sent Events (SSE) for real-time token-by-token streaming:

**Event Flow**:
```
start     → {"conversation_id": "...", "timestamp": "..."}
quote     → {"symbol": "AAPL", "price": 234.56, ...}
progress  → {"stage": "agent_name", "status": "running", "detail": "..."}
token     → {"token": "The", "full_response": "The"}
response  → {"response": "full markdown response", "metadata": {...}}
done      → {"conversation_id": "...", "agents_used": [...]}
error     → {"error": "message"}
```

**Frontend Integration** (example from docstring):
```javascript
const response = await fetch('/api/chat', {method: 'POST', body: JSON.stringify({message, conversation_id})});
const reader = response.body.getReader();
// Parse SSE events: event type is in 'event:' line, data in 'data:' line
```

#### `_EliteManagerWrapper` — Thread-Safe Manager Access:
The Manager class is not async-native, so the wrapper:
1. Runs `Manager.process_prompt()` on a background thread
2. Monkey-patches `agent.answer()` to intercept token streaming
3. Captures tokens via callback → pushes to async queue → SSE emits
4. Suppresses Rich console output during server mode

```python
class _EliteManagerWrapper:
    def __init__(self):
        self._manager = None       # Lazy-loaded Manager
        self._lock = threading.Lock()
    
    async def process_message(self, message, conversation_id, on_token=None):
        # Monkey-patch answer() → capture tokens
        original_answer = manager.agent.answer
        manager.agent.answer = _patched_answer_with_callback(original_answer, on_token)
        # Run on thread pool
        result = await asyncio.to_thread(manager.process_prompt, message, conversation)
```

#### `ConversationStore` — Server-Side Conversation Management:
- Max 1,000 conversations in memory
- Max 40 messages per conversation
- Auto-cleanup: oldest conversations evicted when limit reached
- Thread-safe access

#### `ChatMetadata` — Response Metadata Extraction:
Parses the response text to extract structured metadata:
- `symbols`: Detected stock/crypto symbols
- `path`: "fast" or "apex"
- `verdict`: Trading verdict (if mentioned)
- `score`, `confidence`, `price`
- `agents_used`: List of agents that contributed
- `decision_engine`: Whether UDE was used
- `ml_prediction`: Whether ML prediction was included
- `luxalgo`: Whether LuxAlgo signals were included

#### WebSocket — `WS /ws/chat`:
Bidirectional WebSocket for real-time chat:
- Receives JSON: `{"message": "...", "conversation_id": "..."}`
- Sends JSON: `{"type": "token|response|error", "data": "..."}`
- Supports reconnection with conversation_id persistence

#### Startup Tasks:
- **Learning Resolver**: `PredictionResolver` started as background task (runs every 3600s)
- Fetches current prices from Polygon for all unresolved predictions
- Calls `LearningHub.resolve_predictions()` to score past predictions

#### APEX Monitoring Thread:
Runs on server startup, monitors:
- System health components
- Memory usage
- Active conversation count
- Background task status

---

## 17. Tool Executor (Claude ↔ Tools Loop)

### `src/nuble/api/tool_executor.py` (1,138 lines)
**Server-Side Claude Function Calling with Direct Tool Dispatch**

This implements **Pattern B** (server-side Claude ↔ Tools loop) as opposed to Pattern A (frontend direct API calls).

#### Architecture:
```
Frontend POST /api/intel/chat-with-tools
    ↓
Claude Sonnet 4 (with 17 tool definitions)
    ↓ tool_use response
_dispatch_tool(tool_name, tool_input)
    ↓ direct function call (NO HTTP overhead)
Tool Result → back to Claude → next tool or final response
    ↓ (max 3 rounds)
Final Answer → Frontend
```

#### Router: `APIRouter(prefix="/api/intel")`, endpoint `POST /api/intel/chat-with-tools`

#### `ChatWithToolsRequest`:
```python
message: str              # User message
conversation_id: str      # Conversation tracking
max_tool_rounds: int = 3  # Max tool call iterations
```

#### `_dispatch_tool()` — Direct Function Calls (17+ tools):

All tools are dispatched as **direct function calls** (no HTTP overhead):

| Tool Name | Implementation | Data Returned |
|-----------|---------------|---------------|
| `roket_predict` / `get_stock_prediction` | `LivePredictor.predict(ticker)` | Composite score, signal, confidence, tier, feature coverage |
| `roket_analyze` | httpx to `/api/roket/analyze/{ticker}` with local fallback | All 15 analysis sections |
| `roket_fundamentals` | WRDS panel → 40+ valuation/profitability columns | Valuation ratios, margins, returns |
| `roket_earnings` | WRDS panel → 30+ earnings columns | SUE, persistence, accruals, P/E variants |
| `roket_risk` | WRDS panel → 40+ risk/volatility columns | Betas, vol, momentum, turnover |
| `roket_insider` | WRDS panel → 15+ insider/analyst columns | Buy ratio, CEO buy, analyst dispersion |
| `roket_institutional` | WRDS panel → 20+ institutional columns | Ownership change, HHI, breadth |
| `roket_regime` | `WRDSPredictor.get_market_regime()` | HMM state, probabilities, VIX exposure |
| `roket_screener` | Filter 20,723 universe | By tier/signal/decile/score range |
| `roket_universe` | Top N ranked stocks | With predictions and scores |
| `roket_news` | `StockNewsClient.get_ticker_news()` (async) | Articles with sentiment labels |
| `roket_snapshot` | `DataAggregator.get_snapshot()` | Real-time multi-source snapshot |
| `roket_sec_quality` | `SECEdgarXBRL.get_quality_score()` | 40 ratios + composite grade (A-F) |
| `roket_macro` | `FREDMacroData.get_current()` | Yields, spreads, regimes |
| `roket_lambda` | `NubleLambdaClient.analyze()` | Full Lambda decision |
| `roket_compare` | Side-by-side (2-5 stocks) | Comparative predictions |
| `roket_position_size` | Kelly criterion + ATR stops | Size, stop, TP levels |

Legacy tools (backward compatible):
- `get_batch_predictions`, `get_top_picks`, `analyze_portfolio`, `get_tier_info`, `get_system_status`

#### `_get_tools_for_claude()` — Tool Definitions:
Returns all 17 tool definitions in **Anthropic API format** with detailed descriptions and JSON schemas. Each tool has:
- `name`: Tool identifier
- `description`: Detailed description of what the tool returns
- `input_schema`: JSON Schema with required/optional parameters

#### `roket_analyze` HTTP→Local Fallback:
```python
# Try HTTP first (for when ROKET server is running separately)
async with httpx.AsyncClient() as client:
    resp = await client.get(f"http://localhost:8000/api/roket/analyze/{ticker}")

# If HTTP fails, extract data directly from WRDS panel
panel_row = predictor.get_latest_features(ticker)
# Extract all sections: prediction, fundamentals, earnings, risk, insider, institutional
```

---

## 18. Intelligence API (System A+B Endpoints)

### `src/nuble/api/intelligence.py` (807 lines)
**Structured REST Endpoints for Claude Function Calling**

#### Router: `APIRouter(prefix="/api/intel")`

#### Endpoints:
| Method | Path | Handler | Response Model |
|--------|------|---------|---------------|
| GET | `/api/intel/predict/{ticker}` | LivePredictor single-stock | `PredictionResponse` |
| POST | `/api/intel/predict/batch` | LivePredictor multi-stock (up to 50) | `BatchPredictionResponse` |
| GET | `/api/intel/regime` | HMM regime detection | `RegimeResponse` |
| GET | `/api/intel/top-picks` | Top N ranked by composite score | `TopPicksResponse` |
| GET | `/api/intel/system-status` | Full system health + data freshness | `SystemStatusResponse` |
| GET | `/api/intel/tier-info/{ticker}` | Tier classification + model details | `TierInfoResponse` |
| GET | `/api/intel/universe/stats` | Universe coverage statistics | `UniverseStatsResponse` |
| POST | `/api/intel/portfolio/analyze` | Portfolio-level analysis | `PortfolioAnalyzeResponse` |
| GET | `/api/intel/tools-schema` | OpenAPI tool definitions for Claude | `List[ToolDefinition]` |

#### Lazy Singletons:
- `_get_live()` → LivePredictor (lazy loaded on first request)
- `_get_wrds()` → WRDSPredictor (lazy loaded on first request)
- `_get_regime()` → HMMRegimeDetector (lazy loaded on first request)

#### Pydantic Response Models:
Every response is fully typed with Pydantic:
- `PredictionResponse`: ticker, tier, signal, composite_score, fundamental_score, timing_score, confidence, decile, data_source, feature_coverage, top_drivers, macro_regime
- `RegimeResponse`: state (bull/neutral/crisis), state_id, probabilities, confidence, vix_exposure
- `TopPicksResponse`: picks array, count, regime, execution_time_seconds

---

## 19. LuxAlgo Webhook API

### `src/nuble/api/__init__.py` (39 lines)
**Module Export Map — 3 API Servers with Graceful Fallback**

Architecture: 4 API modules, try/except lazy imports for optional FastAPI dependency:
1. `server.py` → `app` (Production API — **default**, imported first)
2. `intelligence.py` → `intel_router` (System A+B intelligence endpoints)
3. `roket.py` → `roket_router` (Data Access Layer REST)
4. `main.py` → Legacy API (direct OrchestratorAgent, kept for reference)

If FastAPI is not installed, `app` gracefully falls back to `None`. Intelligence and ROKET routers mounted at `/api/intel/*` and `/api/roket/*` automatically.

### `src/nuble/api/luxalgo_api.py` (412 lines)
**TradingView LuxAlgo Signal Receiver**

#### Router Endpoints:
| Method | Path | Description |
|--------|------|-------------|
| POST | `/webhooks/luxalgo` | Receive TradingView LuxAlgo alert webhook |
| GET | `/signals/{symbol}` | Get recent signals (filterable by hours, min_confirmations) |
| GET | `/signals/{symbol}/latest` | Get the latest signal only |
| GET | `/signals/consensus/{symbol}` | Get signal consensus across timeframes |
| GET | `/signals/status` | Status of all tracked symbols |

#### TradingView Alert Configuration:
```json
{
    "action": "BUY",
    "symbol": "{{ticker}}",
    "exchange": "{{exchange}}",
    "price": {{close}},
    "timeframe": "{{interval}}",
    "signal_type": "Bullish Confirmation",
    "confirmations": 12,
    "trend_strength": 54.04,
    "trend_tracer": "bullish",
    "smart_trail": "bullish",
    "neo_cloud": "bullish",
    "time": "{{time}}"
}
```

#### Webhook Processing:
1. Parse raw JSON body from TradingView
2. `parse_luxalgo_webhook(body)` creates `LuxAlgoSignal` object
3. Store in `LuxAlgoSignalStore` (in-memory + DynamoDB)
4. Process signal in background (callbacks, strong signal logging)
5. Strong signals (≥4 confirmations on 4h+) logged prominently

### Multi-Timeframe API — `src/nuble/api/mtf_api.py` (331 lines)
**Institutional Multi-Timeframe Signal System REST API**

Separate from the LuxAlgo webhook — this provides endpoints for the full MTF fusion pipeline:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/mtf/webhook` | Receive signal from any timeframe (TradingView alerts) |
| GET | `/mtf/decision/{symbol}` | Get complete MTF trading decision with position sizing |
| GET | `/mtf/signals/{symbol}` | Get all current signals across timeframes |
| GET | `/mtf/alignment/{symbol}` | Check timeframe alignment score and direction |
| GET | `/mtf/veto/{symbol}` | Check if veto engine would block a trade |
| GET | `/mtf/status` | System status for all tracked symbols |
| POST | `/mtf/cleanup` | Remove expired signals |
| GET | `/mtf/config` | Current MTF configuration (weights, expiry, sensitivity) |

`MTFWebhookPayload` Pydantic model: action, symbol, timeframe, price, plus optional LuxAlgo fields (exchange, signal_type, strength, confirmations, trend_strength, smart_trail_sentiment, neo_cloud_sentiment, reversal_zone, ml_classification).

`DecisionResponse` includes: can_trade, action, direction, strength, confidence, position_size_pct, position_dollars, entry_price, stop_loss, take_profit_1/2, reasoning, timeframes breakdown.

### Standalone API Entry Point — `src/nuble/api/main.py` (483 lines)
**Alternative FastAPI Application with User Profiles + Memory**

A factory-pattern (`create_app()`) standalone API entry point that integrates the Orchestrator directly with persistent memory:

Endpoints:
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System health + uptime + agents count + memory stats |
| POST | `/chat` | Main chat — routes through OrchestratorAgent.process() |
| POST | `/chat/stream` | SSE streaming chat with agent progress events |
| WS | `/ws/chat` | WebSocket bidirectional chat |
| GET | `/quick/{symbol}` | Quick symbol lookup (market analyst only) |
| GET | `/agents` | List all available agents |
| POST | `/users/profile` | Create/update user profile |
| GET | `/users/{user_id}/profile` | Get user profile |
| GET | `/users/{user_id}/conversations` | Get conversation history (limit 100) |
| GET | `/users/{user_id}/predictions` | Get prediction accuracy stats |
| POST | `/feedback` | Submit conversation feedback (1-5 rating) |

Integrates: `OrchestratorAgent`, `MemoryManager`, LuxAlgo routes, MTF routes. User context from memory is injected into orchestrator calls.

### Pattern A Tool Dispatcher — `src/nuble/api/roket_tools.py` (170 lines)
**Frontend-Side Claude ↔ ROKET HTTP Dispatcher**

This is the **Pattern A** counterpart to `tool_executor.py` (Pattern B). When the frontend handles the Claude ↔ Tools loop itself, it uses this module to dispatch tool calls via HTTP to the ROKET API.

`TOOL_ROUTES` mapping (17 tools → HTTP method + path):
- 11 single-ticker GET tools: predict, analyze, fundamentals, earnings, risk, insider, institutional, news, snapshot, sec_quality, lambda
- 2 no-input GET tools: regime, macro
- 4 parameterized tools: screener (POST), universe (GET), compare (GET), position_size (POST)

Special handling: Screener tier/signal singulars → plurals list conversion, compare tickers list → comma-separated string.

`load_tool_definitions()` loads canonical tool definitions from `roket_tool_definitions.json` (253 lines) for Claude function calling schema.

---

## 20. ML Pipeline (System B)

### `src/nuble/ml/__init__.py` (104 lines)
**ML Module Evolution — Phase 1→4 (F1→F4) Pipeline Architecture**

Documents the full evolution of the ML subsystem across 4 development phases:

| Phase | Stage | Components | Status |
|-------|-------|-----------|--------|
| Phase 1 | F1 (Features) | `FeaturePipeline`, `FractionalDifferentiator` | Active |
| Phase 2 | F2 (Labeling) | `TripleBarrierLabeler`, `SampleWeighter`, `MetaLabeler` | Active |
| Phase 3 | F3 (Training) | `PurgedWalkForwardCV`, `TrainingPipeline` | Active |
| Phase 4 | F4 (Prediction) | `LivePredictor` (PRIMARY), `MLPredictor` (DEPRECATED) | Active |

Additional lazy imports (try/except for optional dependencies):
- Phase 1: `UniversalTechnicalModel` — universal technical timing model
- Phase 2: `ModelManager` — model lifecycle management
- Phase 3: `WRDSPredictor` — historical WRDS-based prediction ("historical fallback")
- Phase 4: `LivePredictor` — production Polygon-based prediction ("PRIMARY signal source")

Deprecation: `MLPredictor` → use `LivePredictor` instead (backward-compatible re-export)

`__all__` exports 18 symbols covering the complete F1→F4 pipeline.

### Core ML Files:

#### `src/nuble/ml/wrds_predictor.py` (769 lines)
**Multi-Tier LightGBM Ensemble — The Production Brain**

Tier Configuration:
| Tier | Market Cap | IC | Weight | Strategy |
|------|-----------|-----|--------|----------|
| Mega | >$10B | 0.027 | 14.8% | raw |
| Large | $2-10B | 0.019 | 10.5% | hedged |
| Mid | $500M-2B | 0.038 | 21.3% | raw |
| Small | <$500M | 0.096 | 53.4% | vix_scaled |

Data: GKX panel (15GB parquet), 522 features, 20,723 tickers, latest=2024-12-31

Methods:
- `predict(ticker)` → Routes to correct tier model by market cap
- `get_top_picks(n)` → Top N picks across all tiers
- `get_tier_predictions(tier)` → All predictions for a tier
- `get_market_regime()` → HMM + rule-based regime detection

#### `src/nuble/ml/live_predictor.py` (493 lines)
**Live Predictions: Polygon + Trained Models**

Scoring: `composite = 0.70 × fundamental_score + 0.30 × timing_score`
- `fundamental_score` = LightGBM(live features from PolygonFeatureEngine)
- `timing_score` = universal_technical_model (daily timing signal)

Model Hierarchy:
1. Production models (221 Polygon-available features → 100% coverage)
2. Research models (all GKX features → 23% coverage live)

Signal Thresholds (decile-based):
- D10 (≥0.9): STRONG_BUY
- D8-D9 (0.7-0.9): BUY
- D5-D6 (0.4-0.6): HOLD
- D2-D3 (0.1-0.3): SELL
- D1 (<0.1): STRONG_SELL

#### `src/nuble/ml/hmm_regime.py` (336 lines)
**HMM Regime Detector — 3 States from Real Macro Data**

Training Data: GKX panel macro features aggregated monthly (1990+):
- `vix` — VIX implied volatility
- `term_spread_10y2y` — Yield curve slope
- `corp_spread_bbb` — BBB credit risk  
- `realized_vol` — Market-wide realized volatility
- `mom_1m` — Market momentum

Model: `GaussianHMM(n_components=3, covariance_type='diag', n_iter=500)` from hmmlearn
- Training: Standardized features → fit on 300+ months of data (1990+)
- State labeling: Post-hoc by VIX means (lowest VIX mean → BULL, highest → CRISIS, other → NEUTRAL)
- Model persistence: Saved as pickle at `models/regime/hmm_regime_model.pkl`
- Fallback: If hmmlearn not installed, uses rule-based VIX thresholds

Detection: `detect_regime()` → Returns `{state, state_id, probabilities, confidence, vix_exposure}`
- VIX exposure scaling: BULL=1.0 (full position), NEUTRAL=0.8, CRISIS=0.3 (minimal exposure)

#### `src/nuble/ml/predictor.py` (401 lines)
**MLPredictor — Production Prediction Interface (F4 Pipeline)**

Thread-safe, lazy-loading prediction wrapper. Zero-crash guarantee — every public method returns a safe default on error.

Model Priority Chain:
1. **CrossSectionalModel** (Phase 6 — best) — works for ANY stock, regression on excess returns
2. **UniversalTechnicalModel** (Phase 1 fallback) — LightGBM classification on 45 technical features
3. **Per-ticker MLP models** (legacy) — per-symbol models at `models/{SYMBOL}_{timestamp}/`

Features:
- **Auto-reload**: Detects newer model directories on disk and hot-swaps (naming convention: `{SYMBOL}_{timestamp}/`)
- **Thread-safe**: All mutable state guarded by `threading.Lock()`
- **Lazy loading**: Models loaded on first call, not at import time
- **SHAP explanations**: Every prediction returns top-5 feature attributions
- Singleton: `get_predictor(model_dir)` returns global instance

Output dict: `{symbol, prediction (0=DOWN/1=NEUTRAL/2=UP), direction, confidence, probabilities, explanation {top_features, base_value}, model_info}`

#### `src/nuble/ml/features_v2.py` (1,075 lines)
**Institutional-Grade Feature Engineering — de Prado AFML Methodology**

Key innovations over v1:
1. **Fractional Differentiation** (de Prado Ch.5): Finds minimum `d ∈ [0,1]` via ADF test that achieves stationarity while preserving maximum price memory. Fixed-Width Window method with threshold truncation
2. **Cyclical calendar encoding**: sin/cos transforms for day-of-week and month (preserves adjacency: Sunday↔Monday, December↔January)
3. **Cross-asset context**: SPY, VIX, sector ETFs — stocks don't move in isolation
4. **Wilder's smoothing**: Proper RSI/ATR using EWM alpha=1/14 (old code used simple rolling mean — a known bug)
5. **Hurst exponent**: Detects trending (H>0.5) vs mean-reverting (H<0.5) regimes
6. **Train/test isolation**: Features computed INSIDE CV folds — no lookahead bias

`FractionalDifferentiator` class:
- `_get_weights(d, threshold, max_len)` — de Prado Snippet 5.1 fractional differencing weights
- `fit_transform(series)` — finds optimal `d` that passes ADF test at 5% significance
- Typical `d` values: 0.3–0.5 (preserves memory while achieving stationarity)

`FeaturePipeline` class:
- `build_features(df)` → Full feature DataFrame with momentum, volatility, mean-reversion, microstructure, regime, calendar features
- Cross-asset features computed from SPY/VIX if available

#### `src/nuble/ml/labeling.py` (968 lines)
**Triple Barrier Labeling — de Prado Chapters 3-4**

Replaces naive binary (up/down next-day) labels with institutional-grade labeling:

`VolatilityEstimator`:
- `daily_volatility(close, span=21)` — EWMA standard deviation of log returns
- `parkinson_volatility(high, low, span=21)` — 5× more efficient range-based estimator

`TripleBarrierLabeler`:
Three barriers touched first determines the label:
- **Take-Profit** barrier: `price × (1 + tp_mult × daily_vol)` → label = +1
- **Stop-Loss** barrier: `price × (1 - sl_mult × daily_vol)` → label = -1
- **Time Expiry** barrier: after `max_holding_period` days → label = sign(return)
- Barriers are **volatility-scaled**: wider in high-vol (VIX=30), narrower in calm (VIX=12)

`SampleWeighter` (de Prado Ch.4):
- **Uniqueness weights**: For overlapping labels, computes average uniqueness per sample
- **Time decay**: Exponential decay `e^{-decay × age}` for older samples
- Prevents inflated CV metrics from correlated adjacent labels

`MetaLabeler`:
- Two-stage framework: primary model picks direction, secondary model sizes the bet
- `create_meta_labels(df, primary_predictions)` → meta-labels for bet sizing

Convenience functions: `create_labels(df, tp, sl, holding)`, `label_distribution_report(labels)`

#### `src/nuble/ml/trainer_v2.py` (1,468 lines)
**Institutional Model Training Pipeline — PurgedWalkForwardCV + SHAP + Calibration**

`PurgedWalkForwardCV` (de Prado Ch.7):
- Walk-forward with **purging**: removes training samples whose label period overlaps test
- **Embargo**: Additional buffer of `embargo_pct` at train/test boundary
- Expanding or sliding window (configurable)
- `n_splits=5`, `min_train_size=252` (~1 year), `max_holding_period=10`

`FinancialMetrics`:
- **IC** (Information Coefficient): Spearman rank correlation of predictions vs returns
- **Rank IC**: Rank-order correlation
- **IR** (Information Ratio): IC / std(IC)
- **Profit Factor**: Gross profits / gross losses
- Note: accuracy is IRRELEVANT in finance — IC is the correct metric

`TrainingPipeline`:
- Pipeline: `build_features()` → `create_labels()` → `PurgedWalkForwardCV` → LightGBM training
- **Early stopping**: Validation carved from END of training data (chronological, not random)
- **SHAP per-prediction**: Explains WHY each trade was triggered (top-5 features)
- **Isotonic calibration**: `IsotonicRegression` maps raw LightGBM probabilities to empirical frequencies for correct Kelly sizing
- Model persistence: `model.txt` + `metadata.json` + `features.json` + `calibrator.pkl`
- Methods: `run(df)`, `predict_latest(df)`, `save(path)`, `load(path)`

#### `src/nuble/ml/cross_sectional_model.py` (931 lines)
**Cross-Sectional Regression — Gu-Kelly-Xiu (2020) Framework**

THIS REPLACES classification with regression. Classification: P(direction) → loses magnitude info. Regression: E(return) → preserves magnitude → proper ranking.

Architecture:
1. Features: 112+ from `UniversalFeatureEngine`
2. **Cross-sectional rank normalization**: percentile ranks across ALL stocks per date
3. Target: Forward N-day **excess** log return (excess = stock return − universe median)
4. Model: LightGBM REGRESSION with **Huber loss** (robust to fat-tailed outliers)
5. Evaluation: **Spearman IC** (rank correlation between predictions and actual returns)

Key Methods:
- `predict(symbol, df)` → `{symbol, prediction, direction, confidence, probabilities, model_info}`
- `train(polygon_data, forward_days=21, min_stocks_per_date=100)` → Full training run
- `is_ready()` → Check if trained model is loaded
- Thread-safe with `threading.Lock()`
- Model files: `cross_sectional_model.txt`, `cross_sectional_metadata.json`, `cross_sectional_features.json`

#### `src/nuble/ml/universal_model.py` (873 lines)
**Universal Technical Model — ONE Model for ALL Stocks**

Replaces 5 per-ticker models with a single LightGBM trained on ALL stocks simultaneously.

`compute_universal_features(df)` → ~45 stock-agnostic features:
- **Momentum**: RSI(14) with Wilder's smoothing, ROC(5/21), multi-horizon log returns
- **Volatility**: Realized vol(21), ATR ratio (normalized), Garman-Klass, vol-of-vol(63), vol regime
- **Mean Reversion**: Z-score(20), Bollinger %B, distance from 52-week high/low
- **Microstructure**: Volume ratio, range %
- **Regime**: Hurst exponent, autocorrelation, ADX
- **Calendar**: Day-of-week sin/cos, month sin/cos
- **Fractional diff of close**: de Prado stationarity transform

`UniversalTechnicalModel` class:
- `train(polygon_data, forward_days, min_stocks)` → LightGBM on grouped bars
- `predict(symbol, df)` → Classification (UP/DOWN/NEUTRAL)
- Model files: `universal_technical_model.txt`, `universal_metadata.json`

#### `src/nuble/ml/universal_features.py` (1,141 lines)
**UniversalFeatureEngine — 120+ Features in 6 Groups**

Every feature is STOCK-AGNOSTIC: comparable across any stock at any price level.

| Group | Prefix | Count | Examples |
|-------|--------|-------|---------|
| Momentum | `mom_` | 25 | Multi-horizon returns (1d-252d), RSI, ROC, MACD, acceleration, momentum regime |
| Volatility | `vol_` | 20 | Realized vol (5/10/21/63), ATR%, Garman-Klass, Parkinson, vol-of-vol, term structure, tail ratio |
| Volume | `vlm_` | 18 | Volume ratio (5d/21d/63d), VWAP deviation, OBV trend, price-volume correlation, volume regime |
| Technical | `tech_` | 25 | RSI(14), Stochastic, MACD histogram, Bollinger %B/bandwidth, ADX, CCI, Williams %R, Ichimoku |
| Microstructure | `micro_` | 12 | Intraday range, body ratio, wick ratio, serial correlation, overnight gap, spread estimate |
| Context | `ctx_` | 12 | Calendar sin/cos, streak counter, relative strength vs SMA, drawdown, rally |

Design principles: No lookahead bias, all features use float32, handles NaN/inf/zero-division. Safe helpers: `_safe_div()`, `_safe_series_div()`, `_rolling_zscore()`.

#### `src/nuble/ml/model_manager.py` (423 lines)
**Model Lifecycle Manager — Freshness, Health, Background Retraining**

Rules:
- Universal model: stale after **7 days** → triggers retrain
- Per-ticker models: stale after 7 days (legacy)
- Health check: verify model files exist, are loadable, pass quality gates

`check_health()` returns:
- `universal`: exists, age_days, fresh, n_training_samples, test_ic, test_accuracy, n_features, quality_gates_passed, model_size_kb, backtest metrics
- `per_ticker`: {SYMBOL: {exists, age_days, fresh}}
- `overall`: {healthy, stale_count, missing_count, retraining_in_progress}

`trigger_background_retrain()`: Non-blocking retrain via `threading.Thread` — fetches Polygon data, runs `UniversalTechnicalModel.train()`, logs results. `is_retraining()` for status check.

`get_status_for_cli()`: Formatted Rich-compatible string for `/model-health` CLI command. Includes IC, Sharpe, drawdown, deployability verdict.

#### `src/nuble/ml/train_all.py` (333 lines)
**Batch Training Script — All Tiers**

CLI: `python -m nuble.ml.train_all [--symbols SPY AAPL] [--tp 2.5] [--sl 1.5] [--holding 15] [--folds 7] [--binary]`

Default symbols: SPY, AAPL, TSLA, AMD, QQQ. Fetches ~3 years OHLCV from Polygon. Runs `TrainingPipeline.run()` for each symbol. Falls back to cached data or synthetic stub if API unavailable.

#### `src/nuble/ml/validate_pipeline.py` (324 lines)
**F4 End-to-End Validation Suite — 8 Tests**

| Test | What It Validates |
|------|------------------|
| 1. Synthetic round-trip | Generate data → train → predict → verify dict shape |
| 2. SHAP explanations | top_features present with direction + value |
| 3. Predictor lazy-load | save model → create MLPredictor → predict successfully |
| 4. Hot-swap detection | save two models → verify MLPredictor uses newer one |
| 5. Empty/short data | predict with <10 rows → confidence=0, no crash |
| 6. Batch prediction | predict_batch for 3 symbols → all return dicts |
| 7. Graceful degradation | MLPredictor with bad model_dir → confidence=0 |
| 8. Direction mapping | prediction values 0/1/2 map to DOWN/NEUTRAL/UP |

#### `src/nuble/ml/backtest/walk_forward.py` (1,477 lines)
**Walk-Forward Backtest — Institutional Validation (Gold Standard)**

Expanding-window walk-forward: "If I had trained at time T and traded at T+1, what would returns be?"

Methodology:
1. **Expanding window training**: Min 120 days (~6 months), retrain every 21 days (~monthly), 10-day purge gap
2. **Per-date scoring**: Score ALL stocks in active universe, rank by predicted LONG probability
3. **Decile portfolios**: Form long/short portfolios (equal-weight within each decile)
4. **Portfolio returns**: Long-short = avg(top decile) − avg(bottom decile)

`BacktestResults` container: daily_results, window_results, predictions DataFrame, daily portfolios, stock histories cache. `summary()` aggregates IC, long-short return, hit rate, decile spread across all windows.

`WalkForwardBacktest` class: `run(polygon_data, feature_engine, ...)` → `BacktestResults`. Supports expanding vs sliding window, configurable rebalance frequency, parallel date scoring.

#### `src/nuble/ml/backtest/signal_analysis.py` (676 lines)
**Signal Quality & Characteristics Analysis**

Answers 5 critical pre-deployment questions:

| Analysis | Method | What It Measures |
|----------|--------|-----------------|
| Signal Decay | `signal_decay()` | IC at horizons [1,2,3,5,10,21] days → optimal_horizon, half_life_days |
| Factor Exposure | `factor_exposure()` | Is the model rediscovering known factors (value, momentum, quality)? |
| Turnover | `portfolio_turnover()` | How much does the portfolio change each rebalance? |
| Concentration | `portfolio_concentration()` | Does the model spread bets or concentrate (HHI index)? |
| Regime Dependence | `regime_analysis()` | Does the model work in all market conditions (bull/bear/vol)? |

Uses preloaded stock histories (from backtest run) to avoid redundant data loading.

---

## 21. Data Layer

### `src/nuble/data/__init__.py` (37 lines)
**Module Export Map — 5 Data Services with Optional Dependencies**

All imports wrapped in try/except for graceful degradation when dependencies are missing:
1. `DataService`, `get_data_service` — from `data_service` (unified data access)
2. `S3DataManager`, `get_data_manager` — from `s3_data_manager` (S3 transparent fallback)
3. `PolygonUniverseData` — from `polygon_universe` (full US stock universe)
4. `SECEdgarXBRL` — from `sec_edgar` (SEC EDGAR XBRL parser)
5. `FREDMacroData` — from `fred_macro` (FRED macro data pipeline)

### `src/nuble/data/data_service.py` (410 lines)
**Unified Data Access — Single Interface for All Data**

Singleton-pattern data service. Every module resolves paths through this one class.

**Project Root Auto-Detection** (`_find_project_root()`):
- Walks up 3 parent directories from `__file__`, checks for `pyproject.toml` or `.git`
- Docker fallback: `/app` if it exists
- Lambda fallback: `/var/task` if it exists
- Ultimate fallback: current working directory

**S3 Lazy Loading** (thread-safe):
- `_s3` property initializes `S3DataManager` on first access
- Double-checked locking via `threading.Lock()` for thread safety
- Falls back to local-only mode if S3 init fails

**Data Loading Hierarchy** (`load_parquet(filename, columns, filters)`):
1. Try local file at `data_dir / filename`
2. If missing → download from S3 via `S3DataManager.download_file()`
3. If S3 fails → try alternate local paths (models/, training_results/)
- Supports PyArrow **column projection** (read only needed columns) and **predicate pushdown** filters for memory efficiency

**Ticker ↔ PERMNO Mapping**:
- `ticker_to_permno(ticker)` / `permno_to_ticker(permno)`
- Lazy-loaded from `msenames_mapping.parquet` with thread-safe lock
- Cached in `_ticker_map` / `_permno_map` dicts after first load

Key Paths:
- `data_dir` → `data/wrds/` (GKX panel, ticker maps)
- `models_dir` → `models/` (LightGBM, HMM, production, universal)
- `project_root` → Auto-detected (works in Docker, EC2, local)

Methods:
- `load_parquet(filename, columns, filters)` → Load parquet files with projection/pushdown
- `get_model_path(relative)` → Get model file path
- `load_json(filename, root)` → Load JSON files
- `ticker_to_permno(ticker)` → Ticker → PERMNO mapping
- `permno_to_ticker(permno)` → PERMNO → Ticker mapping

### `src/nuble/data/polygon_feature_engine.py` (1,762 lines)
**Polygon Real-Time Feature Engine v2 — 600+ WRDS-Compatible Features**

Computes WRDS-compatible features from Polygon live data. Feature names match GKX panel columns EXACTLY so trained LightGBM models can score them directly.

`PolygonFeatureEngine` class (1 main entry point, 43 methods):
- `__init__(api_key)` — Polygon REST session, in-memory cache, Fama-French factor cache
- `compute_features(ticker)` → dict of 600+ float features (NaN/inf stripped)

**14-Step Feature Pipeline** (`compute_features`):
1. `_fetch_daily_ohlcv(ticker, days=800)` — 800 days of bars from Polygon, 1hr cache TTL
2. `_fetch_reference_data(ticker)` — Company details (market cap, shares, SIC), 24hr cache
3. `_fetch_financials(ticker)` — Up to 12 quarterly statements from Polygon vX, smart fallback chain for missing income_statement data
4. `_fetch_macro_data()` — 35+ FRED series via REST API, 24hr cache
5. `_compute_momentum(df)` → 9 features: `mom_1m/3m/6m/12m/36m`, `mom_12_2`, `str_reversal`, `ltr`, `svar`, `ret_crsp`
6. `_compute_liquidity(df, ref)` → 6 features: `turnover/3m/6m`, `amihud_illiq`, `zero_vol_days`, `dollar_volume`
7. `_compute_volatility(df)` → 15 features: `realized_vol/3m/6m/12m`, `up_vol`, `down_vol`, `return_skewness/kurtosis`, `idio_vol`, `max/min_daily_ret`, `intraday_range`
8. `_compute_size(df, ref)` → 14 features: `market_cap`, `log_market_cap`, lags, trends
9. `_compute_technicals(df)` → 3 features: `rsi_14`, `macd_signal`, `high_52w_pct`
10. `_compute_risk_factors(df)` → 8 features via Fama-French 6-factor regression: `beta_mkt`, `beta_smb/hml/rmw/cma/umd`, `alpha`, `r_squared`. Downloads Ken French daily factor data (auto-cached) + momentum factor
11. `_compute_valuation(df, financials, ref)` → 20+ ratios: `pe/ps/pcf/ptb/bm/capei/evm/peg`, earnings yield, CAPE variants, `ep`, `cfp`, `dp`, `bm_ia`
12. `_compute_profitability(financials)` → 25+ margins/returns: `roa/roe/roce/roic`, `gpm/opm/npm`, `aftret_eq/invcapd/equity`, `pretret_noa/earnat`, `gprof`, `operprof`
13. `_compute_financial_quality(financials)` → 50+ quality metrics:
    - **Accruals**: `total_accruals`, `accruals_to_cash_flow`, `working_capital_accruals`, `non_current_accruals`, `net_operating_assets`
    - **Revenue quality**: `revenue_growth_qoq/yoy`, `revenue_acceleration`, `revenue_cagr_2yr`, `revenue_share`
    - **Margin trends**: `gross_margin_trend/vol`, `operating_margin_trend`, `net_margin_trend`, `margin_divergence`, `cf_earnings_divergence`
    - **Earnings quality**: `earnings_persistence` (autocorrelation of quarterly NI), `earnings_smoothness` (NI vol / CF vol)
    - **Piotroski F-Score** (9 points): profitability (4: ROA>0, OCF>0, ΔROA>0, OCF>NI) + leverage (3: Δleverage<0, Δcurrent_ratio>0, no dilution) + efficiency (2: Δgross_margin>0, Δasset_turnover>0)
    - **Montier C-Score** (accounting manipulation warning)
    - **SUE** (Standardized Unexpected Earnings): `(EPS_current - EPS_4q_ago) / σ(EPS)`
14. `_compute_leverage(financials)` → 30+: `de_ratio`, `debt_at/ebitda`, `curr/quick_ratio`, `interest_coverage`, `cash_ratio`
15. `_compute_efficiency(financials)` → 20+: `at_turn`, `inv_turn`, `dso/dio/dpo/ccc`, `capex_intensity`, `rd_intensity`
16. `_compute_growth(financials)` → 10+: multi-quarter revenue/earnings growth, CAGR
17. `_compute_bankruptcy_scores(financials, df, ref)` → `altman_z_score`, `ohlson_o_score`
18. `_compute_beneish(financials)` → **Beneish M-Score** (8 variables detecting earnings manipulation): DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA → composite M-score (<-2.22 = likely manipulator)
19. `_compute_cash_flow_quality(financials)` → `fcf_margin`, `ocf_to_net_income`, `capex_to_depreciation`, `fcf_yield`
20. `_compute_lags(features)` → Adds `_lag1`, `_lag3`, `_trend3` for key features
21. `_compute_industry(ref)` → `ff49_industry` dummies + `gsector` from SIC→GICS mapping + Fama-French industry classification
22. `_compute_sp500(ticker)` → `sp500_member` (0/1) + lags + trends (checks against hardcoded 120+ constituent set)
23. `_compute_macro_transforms(macro)` → YoY/MoM transforms from FRED data
24. `_compute_interactions(features, macro)` → 400+ `ix_*` columns = characteristic × macro variable interactions

**FRED Macro Data** (`_fetch_macro_data()`) — 35+ economic series:
| Category | Series | Feature Names |
|----------|--------|--------------|
| Volatility | VIXCLS | `vix` |
| Treasury | DTB3, DGS3MO, DGS2/5/10/30, GS10 | `tbl`, `tbill_3m`, `treasury_2y/5y/10y/30y`, `lty` |
| Spreads | T10Y2Y, T10Y3M, BAAFFM, BAMLH0A0HYM2 | `term_spread_*`, `corp_spread_bbb/hy` |
| Inflation | T10YIE, CPIAUCSL, CPILFESL, PCEPI, PCEPILFE | `breakeven_10y`, `cpi`, `core_cpi`, `pce_deflator`, `core_pce` |
| Employment | PAYEMS, MANEMP, UNRATE, LNS14000006, ICSA | `nonfarm_payrolls`, `unemployment_rate`, `initial_claims` |
| Housing | HOUST, PERMIT, CSUSHPINSA | `housing_starts`, `building_permits`, `case_shiller_hpi` |
| Consumer | UMCSENT, RSXFS, TOTALSL | `consumer_sentiment`, `retail_sales`, `consumer_credit` |
| Industry | DGORDER, BUSLOANS, M2SL, TCU, USSLIND | `durable_goods_orders`, `m2_money_supply`, `capacity_utilization`, `leading_index` |
| Commodities | DTWEXBGS, DCOILWTICO | `trade_weighted_usd`, `wti_crude` |

Derived: `yield_curve_10y2y`, `yield_curve_10y3m`, `infl` (CPI YoY), `cpi_yoy`

**Fama-French Factor Loading** (`_fetch_ff_factors()`):
Downloads daily Fama-French 5-Factor + Momentum data from Ken French Data Library (Dartmouth). Used for computing stock-level factor betas via 252-day rolling OLS regression (Mkt-RF, SMB, HML, RMW, CMA, UMD).

`get_feature_coverage_report(ticker)` → Reports computed feature count, coverage percentage, and group breakdown for a given ticker.

### `src/nuble/data/aggregator.py` (655 lines)
**Data Aggregator — Multi-Source Parallel Fetch**

> **Note**: `polygon_feature_engine_v1_backup.py` (826 lines) is the pre-refactor snapshot of the feature engine. Retained for rollback capability.

Pulls from ALL sources in parallel:
- Polygon.io: quotes, OHLCV, options, news
- DynamoDB: LuxAlgo signals
- FinBERT: sentiment
- HMM: regime detection

**`MarketSnapshot` dataclass** — Complete market state for a single ticker:
- `price`: current, bid, ask, volume_24h, change_pct
- `technical`: SMA 20/50/200, RSI 14, MACD (dict with value/signal/histogram), Bollinger (dict with upper/middle/lower), ATR 14
- `options`: put_call_ratio, unusual_calls, unusual_puts, max_pain, implied_volatility
- `sentiment`: news_sentiment (float score), news_count, social_sentiment
- `regime`: state (string), probability, volatility (from HMM)
- `luxalgo`: weekly_action, daily_action, four_hour_action, aligned (bool)
- `to_dict()` → JSON-serializable snapshot

**`DataAggregator` class** — Async multi-source aggregator:
- `initialize()` — async setup of all data source connections
- 6 parallel enrichment methods: `_get_price_data()`, `_get_technical_data()`, `_get_options_data()`, `_get_sentiment_data()`, `_get_regime_data()`, `_get_luxalgo_signals()`
- `get_snapshot(symbol)` → `MarketSnapshot` (results cached)
- `get_multi_snapshot(symbols)` → parallel `asyncio.gather()` across symbols
- `clear_cache()` → reset cached snapshots

**`RealTimeDataStream` class** — Polygon WebSocket streaming:
- Imports `PolygonStream` from `institutional.streaming.realtime`
- Async context manager: `async with RealTimeDataStream() as stream:`
- `subscribe(symbols: List[str])` → subscribe to real-time trade/quote feeds
- `async for tick in stream.ticks()` → async generator yielding tick-by-tick data
- Graceful connection management with auto-reconnect

Convenience functions: `get_snapshot(symbol)` and `get_all_snapshots()` (default symbols list)

### Other Data Files:
| File | Lines | Purpose |
|------|-------|---------|
| `fred_macro.py` | 306 | FRED macro data pipeline — see below |
| `sec_edgar.py` | 766 | SEC EDGAR XBRL parser — see below |
| `s3_data_manager.py` | 713 | S3 transparent fallback — see below |
| `polygon_universe.py` | 561 | Polygon universe data — see below |

#### `src/nuble/data/fred_macro.py` (306 lines)
**FRED Macro Data Pipeline — Institutional-Grade Macro Intelligence**

Provides the 8 Gu-Kelly-Xiu macro variables plus derived regime indicators. **Gracefully degrades** if `FRED_API_KEY` is not set.

FRED Series (8 indicators):
| Series ID | Name | Frequency |
|-----------|------|-----------|
| DGS3MO | 3-Month Treasury Yield | Daily |
| DGS10 | 10-Year Treasury Yield | Daily |
| BAMLC0A4CBBB | BBB Corporate Bond Yield | Daily |
| BAMLC0A1CAAA | AAA Corporate Bond Yield | Daily |
| T10YIE | 10-Year Breakeven Inflation | Daily |
| INDPRO | Industrial Production Index | Monthly |
| UNRATE | Unemployment Rate | Monthly |
| FEDFUNDS | Effective Federal Funds Rate | Monthly |

Derived Indicators:
- `term_spread` = 10Y - 3M yield (yield curve slope)
- `credit_spread` = BBB - AAA yield (credit risk)
- `industrial_production_yoy` = 12-month % change
- `real_rate` = 10Y yield - breakeven inflation

Regime Classification:
- **Yield Curve**: Normal (ts > 1.0), Flat (0-1.0), Inverted (< 0)
- **Credit Cycle**: Tight (cs < 1.0), Normal (1.0-2.0), Widening (2.0-3.0), Stress (> 3.0)
- **Monetary Policy**: Accommodative (< 2%), Neutral (2-4%), Restrictive (> 4%)

Storage: Cached as Parquet at `~/.nuble/macro_data.parquet`, refreshed via `refresh()` method using `fredapi` library.

#### `src/nuble/data/sec_edgar.py` (766 lines)
**SEC EDGAR XBRL Parser — 24 Accounting Items + 40 Fundamental Ratios**

Data Source: SEC EDGAR Company Facts API (FREE, no API key needed, rate limit 10 req/s).

The Hard Problem Solved: XBRL concept names vary across companies. "Revenue" might be filed as `RevenueFromContractWithCustomerExcludingAssessedTax`, `Revenues`, `SalesRevenueNet`, etc. This module handles ALL variants with an **ordered fallback chain** of 24 concept maps:

Key Concepts with Fallback Chains:
- `revenue` → 5 XBRL concept variants
- `net_income` → 3 variants
- `total_assets` → 1 variant (standardized)
- `stockholders_equity` → 2 variants
- `operating_cash_flow` → 1 variant
- `capex` → 2 variants
- Plus: cost_of_revenue, gross_profit, operating_income, current_assets, cash, total_liabilities, current_liabilities, long_term_debt, short_term_debt, PPE, inventories, depreciation, R&D, interest_expense, dividends_per_share, EPS, income_tax, accounts_receivable

40 Gu-Kelly-Xiu Fundamental Ratios: Computed from extracted XBRL data including profitability ratios, leverage ratios, efficiency ratios, valuation multiples, growth rates.

Quality Score: Composite score (0-100) graded A through F.

CIK Mapping: Downloads ticker → CIK mapping from SEC, cached for 7 days.

#### `src/nuble/data/s3_data_manager.py` (713 lines)
**Enterprise S3 Data Access Layer**

Architecture:
```
Your Code → S3DataManager → Local Disk (data/wrds/)
                          → S3 Bucket (nuble-data-warehouse)
```

Features:
- **Transparent access**: `load_parquet()` checks local first, falls back to S3
- **Lazy S3 client**: boto3 only loaded on first S3 access (thread-safe)
- **Column selection**: Only load needed columns for memory efficiency
- **Local caching**: Downloaded S3 files cached to disk
- **Bidirectional sync**: `push_to_s3()`, `pull_from_s3()`, `sync()`

S3 Key Prefixes:
- `raw/wrds/` — Raw WRDS data files
- `features/` — Computed feature panels (gkx_panel.parquet, training_panel.parquet)
- `predictions/` — Prediction outputs by tier
- `hedging/` — Hedging strategies
- `models/` — Trained model files
- `metadata/` — Ticker mappings, manifests

#### `src/nuble/data/polygon_universe.py` (561 lines)
**Universal Data Fetcher — Entire US Stock Universe**

Fetches daily OHLCV data for ALL US stocks using Polygon's grouped daily bars endpoint: **one API call per date returns ALL stocks**.

Key Insight: To build 2 years of history = 500 trading days × 1 API call = 500 calls total.

Storage: Monthly Parquet files at `~/.nuble/universe_data/`, each with columns: `[ticker, date, open, high, low, close, volume, vwap, transactions]`.

`PolygonUniverseData` class (18 methods):
- `backfill(start_date, end_date, callback)` → Full historical download with checkpoint/resume
- `incremental_update(callback)` → Update from last saved date to today
- `get_stock_history(ticker, start_date, end_date)` → Extract single-stock OHLCV from universe files
- `get_active_universe(date, min_price, min_volume)` → All tickers active on a given date with filters
- `get_universe_stats()` → Total tickers, date range, file count, disk usage
- `_fetch_grouped_daily(date)` → Single Polygon API call for all US stocks on one date
- `_save_monthly_parquet(year, month, data)` → Parquet with Snappy compression
- `_load_monthly_parquet(year, month)` → Load with optional column projection
- `_is_market_holiday(date)` → Checks against hardcoded US market holidays (2023-2026)
- `_get_checkpoint()` / `_save_checkpoint(date)` → Checkpoint/resume for interrupted downloads
- `_filter_quality(df)` → Removes sub-$1 stocks, low-volume, non-standard tickers

Features:
- US market holiday calendar (2023-2026)
- Checkpoint/resume for interrupted downloads
- Data quality filtering: removes sub-$1 stocks
- Progress tracking via callback
- ~500MB for 2 years (Parquet compression)

---

## 22. Decision Layer

### `src/nuble/decision/__init__.py` (54 lines)
**Module Export Map — Engine V2 + Enrichment + Trade Setup**

Exports 15 symbols organized into 4 groups:
1. **Engine V2**: `DecisionEngineV2`, `TradingDecision`, `TradeStrength`
2. **Data Classes**: `SignalLayerScore`, `ContextLayerScore`, `ValidationLayerScore`, `RiskLayerScore`, `Regime`
3. **Enrichment**: `EnrichmentEngine`, `EnrichedIntelligence`, `EnrichedMetric`, `Anomaly`, `Divergence`, `ConflictV2`, `ConsensusMetrics`
4. **Trade Setup**: `TradeSetupCalculator`, `TradeSetupResult`

### `src/nuble/decision/ultimate_engine.py` (1,663 lines)
See Section 11 above.

### `src/nuble/decision/trade_setup.py` (360 lines)
**Pure Mathematical Risk Management**

Math used:
- ATR-based stop placement (adapts to actual volatility)
- Keltner Channel for dynamic support/resistance
- Fractional Kelly Criterion for position sizing
- Volatility-scaled take profits at defined R-multiples (1.5R, 2.5R, 4.0R)

Output: `TradeSetupResult` with entry, stop_loss (with distance % and basis), TP1/TP2/TP3 (with R-multiples), position_size_pct, risk_per_trade_pct, ATR, Keltner bands, annualized volatility, notes

### `src/nuble/decision/enrichment_engine.py` (1,432 lines)
**Statistical Enrichment — Institutional-Grade Context**

Does NOT make trading decisions. Transforms raw data into statistically enriched intelligence for Claude consumption.

**6 Dataclasses** (structured output types):

`EnrichedMetric` — Single metric with statistical context:
- name, value, unit, percentile_90d, z_score_20d, rate_of_change_5d
- min_90d, max_90d, mean_90d
- is_anomaly (|z_score| > 2.0), anomaly_description

`Divergence` — Cross-source disagreement:
- source_a, source_b, description
- severity: strong / moderate / weak
- implication (text explanation)

`Anomaly` — Statistical outlier:
- metric_name, current_value, z_score
- description, historical_context

`ConflictV2` — Multi-source conflict:
- sources (list), description, severity (critical / moderate / minor)
- agreement_score (-1.0 to +1.0), resolution_guidance

`ConsensusMetrics` — Cross-source agreement:
- direction_agreement (0.0–1.0), weighted_direction_agreement
- bullish_sources, bearish_sources, neutral_sources (lists)
- dominant_direction, conviction_level (high >80%, moderate 60–80%, low 40–60%, divided <40%)

`EnrichedIntelligence` — Complete enriched output:
- symbol, timestamp, price/volume as `EnrichedMetric`
- technicals, sentiment, fundamentals, macro, risk (dicts of `EnrichedMetric`)
- luxalgo, ml_predictions (dicts)
- anomalies (`List[Anomaly]`), divergences (`List[Divergence]`), conflicts (`List[ConflictV2]`)
- consensus (`ConsensusMetrics`), data_coverage (dict)
- intelligence_brief (formatted markdown string for Claude)

**14-Step `enrich()` Async Pipeline**:
1. `_enrich_price_volume()` → current price, volume, change as `EnrichedMetric`
2. `_enrich_technicals()` → RSI, MACD, SMA, Bollinger, ATR with percentiles/z-scores
3. `_enrich_sentiment()` → news sentiment score, news count, social sentiment
4. `_enrich_fundamentals()` → P/E, P/B, dividend yield, earnings quality
5. `_enrich_macro()` → VIX, yield curve, credit spreads, inflation
6. `_enrich_risk()` → beta, correlation, drawdown, volatility regime
7. Preserve LuxAlgo signals (pass-through from aggregator)
8. Preserve ML predictions (pass-through from aggregator)
8.5. Add v2 ML prediction (LivePredictor integration)
9. `_detect_anomalies()` → flag any metric with |z_score_20d| > 2.0
10. `_detect_divergences()` → find disagreements: price↔sentiment, technicals↔fundamentals, momentum↔trend
11. `_detect_conflicts()` → identify conflicting source signals, compute agreement_score
12. `_compute_consensus()` → aggregate all source directions into `ConsensusMetrics`
13. `_compute_data_coverage()` → % of data sources reporting (completeness metric)
14. `_build_intelligence_brief()` → generates formatted markdown summary for Claude with key metrics, anomalies, divergences, and consensus

**26+ Methods** including: `_enrich_single_metric()`, `_compute_rsi_series()`, `_compute_macd_histogram_series()`, `_normalize_direction()`, `_extract_agent_directions()`, `_get_historical_values()`, `_severity_label()`

### `src/nuble/decision/engine_v2.py` (1,276 lines)
**Decision Engine V2 — Predecessor to UDE**

4-Layer Architecture (weights sum to 1.0):
- **Signal Layer (40%)**: LuxAlgo multi-timeframe, momentum (RSI, MACD, Stochastic), trend (SMA crossovers)
- **Context Layer (30%)**: HMM regime, FinBERT sentiment, VIX volatility state, macro context
- **Validation Layer (20%)**: Historical win rate, backtest confidence
- **Risk Layer (10% + VETO POWER)**: Position limits, drawdown, correlation, liquidity

`DecisionEngineV2` class (30+ methods):
- `make_decision(symbol, data)` → Main entry, orchestrates all 4 layers → `TradingDecision`
- **Signal Layer**: `_calculate_signal_score()`, `_calculate_luxalgo_score()`, `_calculate_momentum_score()`, `_calculate_trend_score()`, `_calculate_ml_score()`
- **Context Layer**: `_calculate_context_score()`, `_calculate_regime_score()`, `_calculate_sentiment_score()`, `_calculate_volatility_score()`, `_calculate_macro_score()`
- **Validation Layer**: `_calculate_validation_score()`, `_calculate_win_rate()`, `_calculate_backtest_confidence()`
- **Risk Layer**: `_calculate_risk_score()`, `_check_position_limits()`, `_check_drawdown()`, `_check_correlation()`, `_check_liquidity()`
- **Technical helpers**: `_compute_rsi()`, `_compute_macd()`, `_compute_stochastic()`, `_compute_sma_crossover()`, `_compute_momentum_regime()`, `_compute_trend_strength()`, `_compute_support_resistance()`
- **Position sizing**: `_calculate_position_size()` (0.5%–5.0% scaled by strength × regime multiplier)

Confidence Thresholds:
- VERY_STRONG ≥ 85, STRONG ≥ 70, MODERATE ≥ 55, WEAK ≥ 40

Position Sizing: 0.5% to 5.0% of portfolio, scaled by signal strength and regime risk multiplier.

### `src/nuble/decision/data_classes.py` (771 lines)
**Structured Data Classes for Decision Engine**

Key Enums:
- `TradeStrength`: NO_TRADE=0, WEAK=1 (25% size), MODERATE=2 (50%), STRONG=3 (75%), VERY_STRONG=4 (100%)
- `Regime`: STRONG_BULL (1.0 risk mult), BULL (1.0), SIDEWAYS (0.7), BEAR (0.5), STRONG_BEAR (0.3), HIGH_VOL (0.4), CRISIS (0.1)
- `VolatilityState`: VERY_LOW (1.2x size), LOW (1.1x), NORMAL (1.0x), HIGH (0.6x), EXTREME (0.3x)

Key Dataclasses:
- `SignalLayerScore`: LuxAlgo, momentum, trend, ML, combined scores (40% weight)
- `ContextLayerScore`: Regime, sentiment, volatility, macro, combined scores (30% weight)
- `ValidationLayerScore`: Historical win rate, backtest confidence (20% weight)
- `RiskLayerScore`: Position limits, drawdown, correlation, veto flags (10% weight + veto power)
- `TradingDecision`: Final output with direction, strength, confidence, trade_setup, risk_checks, reasoning

---

## 23. Learning System — Deep Dive

### `src/nuble/learning/` (5 files)

A **self-improving closed-loop system** that tracks predictions, evaluates outcomes, and dynamically adjusts signal weights. This is a core differentiator — the system literally gets smarter over time.

#### Architecture Flow:
```
Prediction Made (by Orchestrator)
    ↓
LearningHub.record_prediction()     → saves to raw_predictions.json
    ↓ (hourly background task)
PredictionResolver._resolve_loop()  → fetches current prices from Polygon
    ↓
LearningHub.resolve_predictions()   → compares entry vs current price
    ↓
AccuracyMonitor.record_outcome()    → tracks accuracy by source, symbol, regime
    ↓
WeightAdjuster.record_outcome()     → adjusts signal weights
    ↓
LearningHub._save_weights()         → persists to learned_weights.json
    ↓ (next prediction)
UDE/FusionEngine reads new weights  → improved future predictions
```

#### `learning_hub.py` (313 lines)
**Singleton Coordinator** — Thread-safe, stores in `~/.nuble/learning/`

Coordinates:
- **PredictionTracker**: Records every prediction (symbol, direction, confidence, price, timestamp)
- **AccuracyMonitor**: Evaluates prediction accuracy against outcomes
- **WeightAdjuster**: Adjusts signal weights based on accuracy history

Default weights:
```python
{
    'technical_luxalgo': 0.20,
    'technical_classic': 0.03,
    'ml_ensemble': 0.12,
    'sentiment_finbert': 0.08,
    'sentiment_news': 0.08,
    'regime_hmm': 0.07,
    'macro_context': 0.05,
    'fundamental': 0.05,
}
```

Public API:
- `record_prediction(symbol, direction, confidence, price, source, signal_snapshot, metadata)` → Returns prediction_id
  - `signal_snapshot` includes: technical_score, intelligence_score, risk_score, ML predictions, agents_used
  - Stored as plain dict (bypasses FusedSignal requirement)
- `resolve_predictions(symbol, current_price)` → Resolves at 24h, 120h (5d), or 480h (20d)
  - BULLISH correct if actual_return > 0
  - BEARISH correct if actual_return < 0
  - NEUTRAL correct if |actual_return| < 2%
  - Feeds outcomes to AccuracyMonitor and WeightAdjuster
  - Triggers weight recalculation and persistence
- `get_unresolved()` → All pending predictions (for resolver task)
- `get_weights()` → Thread-safe current weights
- `get_accuracy_report()` → Full accuracy statistics
- `get_prediction_stats()` → Total/resolved/pending/correct/accuracy from both raw and formal trackers

Storage:
- `~/.nuble/learning/predictions.json` — Structured Prediction objects
- `~/.nuble/learning/raw_predictions.json` — Raw prediction dicts (lightweight)
- `~/.nuble/learning/learned_weights.json` — Current learned signal weights

#### `prediction_tracker.py` (507 lines)
**Formal Prediction Logging with Full Context**

`Prediction` dataclass:
- `prediction_id`, `timestamp`, `symbol`
- `direction` (-1/0/+1), `confidence` (0-1), `recommended_size` (0-1)
- `component_signals` — Dict of each source's contribution
- `regime` — Market regime at prediction time
- `price_at_prediction`, `stop_loss_pct`, `take_profit_pct`
- `outcome` — PredictionOutcome enum (PENDING/CORRECT/INCORRECT/PARTIAL/EXPIRED)
- `source_outcomes` — Which individual sources were correct

`PredictionTracker` class:
- `log_prediction(signal, price)` → Creates Prediction from FusedSignal, stores in memory + disk
- `resolve_prediction(prediction_id, outcome_price)` → Scores prediction, checks per-source accuracy
- `get_accuracy_stats()` → Overall and per-source accuracy metrics
- Max predictions: 10,000 in memory, persisted to JSON

#### `accuracy_monitor.py` (331 lines)
**Rolling Accuracy Tracking with Degradation Detection**

Tracks accuracy at 3 granularities:
1. **By source** — Overall accuracy per signal source (rolling window of 100)
2. **By source + symbol** — Source accuracy per specific stock
3. **By source + regime** — Source accuracy per market regime

Key Methods:
- `record_outcome(source, was_correct, symbol, regime)` → Records and trims to rolling window
- `get_accuracy(source, window)` → Current accuracy (0-1), defaults to 0.5 if no data
- `get_accuracy_by_symbol(source, symbol)` → Symbol-specific accuracy
- `get_accuracy_by_regime(source, regime)` → Regime-specific accuracy
- `is_degrading(source)` → Compares recent accuracy vs historical average (threshold: 10% drop)
- `get_all_accuracies()` → Full report with accuracy, predictions count, trend (improving/stable/degrading)

Snapshot System: Every 10 outcomes, takes an `AccuracySnapshot` for trend analysis (keeps last 100).

#### `weight_adjuster.py` (286 lines)
**Dynamic Weight Optimization Based on Performance**

Constraints:
- Min weight per source: **5%** (never fully disable a source)
- Max weight per source: **60%** (never over-rely on one source)
- Adjustment rate: **10%** per update cycle
- Smooth transition: 70% old weight + 30% new weight (prevents sudden swings)

Weight Update Algorithm (`_update_weights()`):
1. Get accuracy for each source from AccuracyMonitor
2. Calculate accuracy adjustment: `(accuracy - 0.5) × 2 × 0.1`
   - Sources with >50% accuracy get weight boost
   - Sources with <50% accuracy get weight reduction
3. Apply trend adjustment: improving +2%, degrading -3%
4. Apply to base weights: `new = base × (1 + adjustment)`
5. Clamp to [min_weight, max_weight]
6. Normalize to sum to 1.0
7. Smooth: `final = 0.7 × old + 0.3 × new`

Regime-Specific Weights: `get_weights_for_regime(regime)` adjusts weights based on which sources perform better in specific market regimes.

#### `resolver.py` (100 lines)
**Background Prediction Resolution Task**

`PredictionResolver` class:
- Started by server.py on API server startup
- Runs as `asyncio.Task` every **3,600 seconds** (1 hour)
- For each unresolved prediction:
  1. Fetch current price from Polygon (`/v2/aggs/ticker/{symbol}/prev`)
  2. Call `LearningHub.resolve_predictions(symbol, price)`
- Uses `asyncio.get_event_loop().run_in_executor()` for sync Polygon calls
- Graceful error handling: individual symbol failures don't stop resolution loop

---

## 24. News & Sentiment Pipeline

### `src/nuble/news/` (6 files)

| File | Purpose |
|------|---------|
| `client.py` (421 lines) | StockNews API client — 24 endpoints. API key: `zzad9pmlwttixx0fnsenstctzgdk7ysx0ctkgrk0`. Async via aiohttp. Methods: `get_ticker_news()`, `get_sentiment_stats()`, `get_trending()`, `get_news_summary()` |
| `crypto_client.py` (441 lines) | CryptoNews API client — see below |
| `coindesk_client.py` (877 lines) | CoinDesk Premium Data API — see below |
| `sentiment.py` (439 lines) | FinBERT sentiment analyzer — see below |
| `integrator.py` (541 lines) | ML + News signal integrator — see below |
| `pipeline.py` (454 lines) | Real-time news processing pipeline — see below |

### `src/nuble/news/crypto_client.py` (441 lines)
**CryptoNews API Client — Async Crypto Intelligence**

Base URL: `https://cryptonews-api.com/api/v1`  
API Key: `fci3fvhrbxocelhel4ddc7zbmgsxnq1zmwrkxgq2`

Uses aiohttp for async requests. Lazy session creation.

Key Methods:
- `get_ticker_news(tickers, items, page, date_range, source)` → News for specific crypto tickers
- `get_sentiment_stats(ticker, date_range)` → Quantitative sentiment statistics
- `get_trending_headlines(items)` → Trending crypto headlines
- `get_top_mentioned(items)` → Most mentioned coins
- `get_events(items)` → Breaking crypto events
- `get_category_news(section, items)` → General/DeFi/NFT/regulation news
- `get_sundown_digest(items)` → Daily crypto summary

### `src/nuble/news/coindesk_client.py` (877 lines)
**CoinDesk Premium Data API — OHLCV+, Streaming, Indices**

API Key: `78b5a8d834762d6baf867a6a465d8bbf401fbee0bbe4384940572b9cb1404f6c`  
Uses CryptoCompare infrastructure.

Data Types:
- **IndexType**: CADLI (Real-Time Adaptive Methodology), CCIX (Direct Trading Methodology)
- **TimeUnit**: MINUTE, HOUR, DAY
- **OHLCVBar** dataclass: timestamp, open, high, low, close, volume, quote_volume, total_updates

Key Methods:
- `get_ohlcv(coin, currency, time_unit, limit, end_ts)` → Historical OHLCV+ data
- `get_daily(coin, limit)` → Daily candles
- `get_hourly(coin, limit)` → Hourly candles
- `get_index(coin, index_type)` → CADLI/CCIX index values
- WebSocket streaming support for real-time data

### `src/nuble/news/sentiment.py` (439 lines)
**FinBERT Financial Sentiment Analyzer**

Model: `ProsusAI/finbert` — fine-tuned for financial text.

Architecture:
- Auto-detects device: CUDA → MPS → CPU
- Lazy model loading (loads on first use)
- LRU cache (1000 entries) for repeated texts
- Batch inference (batch_size=16)

`SentimentResult` dataclass: text, label (POSITIVE/NEGATIVE/NEUTRAL), score (0-1 confidence), normalized_score (-1 to +1)

Key Methods:
- `analyze(text)` → Single text analysis → `SentimentResult`
- `analyze_batch(texts)` → Batch analysis → `List[SentimentResult]`
- `get_aggregate_sentiment(texts)` → Average sentiment with confidence

### `src/nuble/news/pipeline.py` (454 lines)
**Real-Time News Processing Pipeline**

Combines StockNews fetching with FinBERT sentiment analysis in a continuous pipeline.

`NewsSignal` dataclass: timestamp, symbol, sentiment_score (-1 to +1), confidence, headline, source, article_count, actionable (bool), signal_type (BULLISH/BEARISH/NEUTRAL)

`NewsPipeline` class:
- Monitors configurable symbol watchlist at `poll_interval` (default 60s)
- Deduplicates articles by tracking `seen_articles` set
- Maintains latest signal per symbol + signal history
- Background async loop: `start()` → `_poll_loop()` → `_process_symbol()` → FinBERT analysis
- `get_signal(symbol)` → Latest signal
- `stream_signals(symbols)` → Async generator for streaming signals
- Optional disk persistence for signal history

### `src/nuble/news/integrator.py` (541 lines)
**ML + News Signal Integrator**

Combines ML trading signals with real-time news sentiment for enhanced prediction accuracy.

`CombinedSignal` dataclass: ml_signal + news_signal → combined_signal with agreement tracking

Integration Modes:
1. **CONFIRMATION**: News confirms ML signal → `confirmation_boost` (+15% confidence)
2. **CONTRADICTION**: News contradicts ML → `contradiction_penalty` (-25% confidence)
3. **NEWS_OVERRIDE**: Extreme news (>80% threshold) overrides ML signal entirely

Default Weights: `ml_weight=0.7`, `news_weight=0.3`

`NewsSignalIntegrator` class:
- `get_signal(symbol, features)` → `CombinedSignal` combining ML prediction + news pipeline
- Actionable threshold: combined confidence > 0.3 for trade signals
- Signal agreement tracking (ML and news agree on direction)

---

## 25. Memory & Cache

### `src/nuble/memory/memory_manager.py` (642 lines)
**Persistent Memory Layer — SQLite-Based**

Uses SQLite for lightweight, file-based storage of user profiles, conversations, predictions, and feedback.

Data Classes:
- **UserProfile**: user_id, name, risk_tolerance (conservative/moderate/aggressive), portfolio (Dict[str, float]), watchlist, preferences
- **Conversation**: conversation_id, user_id, messages (with role/content/timestamp/metadata), context, timestamps
- **Prediction**: prediction_id, user_id, symbol, prediction_type (PRICE/DIRECTION/EVENT), predicted_value, actual_value, confidence, horizon_days, was_correct

`MemoryManager` class:
- `__init__(db_path)` → Default: `~/.nuble/memory.db`
- `save_user_profile(profile)` → Upsert profile
- `get_user_profile(user_id)` → Load profile
- `save_conversation(conversation)` → Upsert conversation
- `get_conversation(conversation_id)` → Load conversation
- `save_prediction(prediction)` → Track prediction
- `resolve_prediction(prediction_id, actual_value, was_correct)` → Score prediction

SQLite Schema:
- `user_profiles` table — JSON blob for flexible schema
- `conversations` table — JSON blob for messages
- `predictions` table — Structured columns for querying

### `src/nuble/cache/redis_cache.py` (500 lines)
**Ultra-Low Latency Redis Cache Layer**

Target: <1ms cache operations. Graceful degradation if Redis unavailable.

`CacheConfig`:
```python
host: "localhost", port: 6379, max_connections: 20
signal_ttl_weekly: 604800 (7 days)
signal_ttl_daily: 86400 (24 hours)
signal_ttl_4h: 28800 (8 hours)
signal_ttl_1h: 7200 (2 hours)
decision_ttl: 300 (5 minutes)
alignment_ttl: 60 (1 minute)
```

`RedisCache` class (async):
- Connection pooling via `redis.asyncio.ConnectionPool`
- Key prefixes: `nuble:signal`, `nuble:decision`, `nuble:alignment`, `nuble:veto`
- `cache_signal(symbol, timeframe, source, data)` → SET with timeframe-appropriate TTL
- `get_signal(symbol, timeframe, source)` → GET with auto-deserialization
- `cache_decision(symbol, decision_data)` → SET with 5-min TTL
- `get_decision(symbol)` → GET cached decision
- `get_all_signals(symbol)` → SCAN for all timeframes/sources
- `get_signal_freshness(symbol)` → Check age of cached signals

Integration: Used by production deployment (ElastiCache Redis 7.1 on AWS).

---

## 26. LLM Wrapper

### `src/nuble/llm.py` (242 lines)

Dual-LLM support with token tracking.

#### Provider Priority:
1. **Claude (Anthropic)** — Preferred. Model: `claude-sonnet-4-20250514`
2. **GPT-4.1 (OpenAI)** — Fallback

#### Token Tracking (`TokenTracker` singleton):
- Tracks: input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, total_requests
- Cost estimate: Claude Sonnet 4 pricing ($3/M input, $15/M output)
- Session duration tracking

#### Methods:
- `prompt(messages, model, requires_json)` → Single response (with JSON extraction)
- `prompt_stream(messages, model)` → Streaming response (generator)

#### Anthropic Message Conversion:
- System/developer messages → Anthropic `system` parameter
- Ensures messages alternate user/assistant (merges consecutive same-role messages)

#### JSON Retry:
`@retry_json_decode(max_retries=3)` decorator on `prompt()` — retries on JSON parse failure

---

## 27. Helpers & Utilities

### `src/nuble/helpers.py` (245 lines)

| Function | Purpose |
|----------|---------|
| `TokenCounter` | Counts tokens using tiktoken (o200k_base encoding) |
| `get_timeout_message(elapsed)` | Progressive waiting messages (32 unique messages like "Cogitating...", "Synthesizing...", etc.) |
| `show_help(console)` | Display CLI help |
| `handle_command(prompt, conversation, agent, console)` | Route CLI commands (/help, /clear, /compact, /key, /exit, /feed) |
| `get_api_key()` / `set_api_key(key)` | Manage API keys in `~/.nuble/config.json` |
| `load_config()` / `save_config(config)` | JSON config file management |
| `handle_feed_command(console)` | Show example queries grouped by category |

---

## 28. AWS Cloud Infrastructure

### Overview:
Full production-grade AWS architecture for NUBLE deployment.

### `Dockerfile` (109 lines)
Multi-stage build optimized for ECS Fargate:
- **Stage 1 (Builder)**: Install all Python deps including torch (CPU), transformers, hmmlearn, LightGBM
- **Stage 2 (Production)**: Minimal image with awscli for S3 data sync
- Non-root user (`nuble`)
- Health check: `curl http://localhost:8000/api/health` (180s start period)
- CMD: `gunicorn nuble.api.roket:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --preload`
- PYTHONPATH: `/app/src:/app`

### `infrastructure/aws/cloudformation/master-stack.yaml` (~1,135 lines)
Complete CloudFormation template:
- **VPC**: 2 public + 2 private subnets, NAT Gateway, 6 VPC endpoints
- **S3**: 2 buckets (data + models) with lifecycle policies
- **Secrets Manager**: API keys (ANTHROPIC, POLYGON, STOCKNEWS)
- **DynamoDB**: 3 tables (signals, decisions, predictions) — PAY_PER_REQUEST, PITR, TTL
- **ElastiCache Redis 7.1**: r7g.large prod / t4g.micro dev, multi-AZ
- **EventBridge**: Scheduled tasks
- **ECR**: Container registry
- **ECS Fargate**: 2048 CPU / 8192 MB / 50GB ephemeral storage
- **ALB**: HTTPS + TLS 1.3
- **Auto-scaling**: 2→20 instances (CPU 65%, Memory 75%, 200 req)
- **CloudWatch**: Dashboard + 5 alarms
- **SNS**: Email alerts
- **Service Discovery**: Private DNS
- **IAM**: Least-privilege roles

### `infrastructure/aws/docker-entrypoint.sh`
1. Fetch secrets from Secrets Manager → set as env vars
2. S3 WRDS data sync → `/app/data/wrds/`
3. S3 model sync → `/app/models/`
4. Print environment summary
5. exec gunicorn

### `infrastructure/aws/deploy-production.sh`
Deployment script with modes: `--infra-only`, `--app-only`, `--upload-data`, `--status`, `--destroy`

### `.github/workflows/deploy.yml`
CI/CD pipeline:
1. Test (pytest + cfn-lint)
2. Build + push to ECR
3. Deploy CloudFormation
4. Update ECS
5. Health check

### `docker-compose.yml`
Local development: API + Redis with hot reload and volume mounts

---

## 29. Configuration & Dependencies

### `src/nuble/WRDS` (1,561 lines — no .py extension)
**WRDS Institutional Data Pipeline — Complete Implementation Guide**

This is a comprehensive engineering specification document (not executable Python) that defines the WRDS data pipeline architecture. It contains:

- **AWS RDS PostgreSQL connection details**: `trading-data-db.ca90y4g2mxtw.us-east-1.rds.amazonaws.com`, PostgreSQL 17, `trading_data` database
- **Existing data**: `stock_prices` table with 44,341,679 records, 20,388 unique tickers, date range 1990–2024
- **WRDS access credentials**: SSH to `wrds-cloud.wharton.upenn.edu:9737`
- **Step-by-step pipeline**: 11 implementation steps for downloading WRDS datasets and loading into RDS
- **Dataset specifications**: CRSP daily, Compustat fundamentals, IBES analyst estimates, 13F institutional holdings, option metrics, mutual fund flows
- **Feature engineering spec**: 522 Gu-Kelly-Xiu features with exact SQL/Python computation formulas

### `src/nuble/core/__init__.py` (13 lines)
**Deprecated Core Module**

Documents that the original core orchestrator, tools, tool_handlers, and memory modules (~3,100 lines of dead code) were removed in v7 Phase 2 cleanup. Superseded by:
- `agents/orchestrator.py` → OrchestratorAgent
- `memory/memory_manager.py` → MemoryManager
- `decision/` → UltimateDecisionEngine

### `pyproject.toml` (76 lines)
```
name = "nuble"
version = "1.0.0"
requires-python = ">=3.8"
license = "GPL-3.0"

dependencies = [
    rich, inquirer, tiktoken, requests, openai, anthropic,
    python-dotenv, numpy>=1.24, pandas>=2.0, aiohttp,
    fastapi, uvicorn[standard]
]

[optional-dependencies]
ml = [lightgbm>=4.0, shap>=0.42, scikit-learn>=1.3, scipy>=1.11,
      joblib>=1.3, statsmodels>=0.14, pyarrow>=14.0, torch>=2.0]
data = [fredapi>=0.5, boto3>=1.28]
production = [nuble[ml,data], gunicorn>=21.0, ...]
```

### API Keys (stored in `.env`):
| Key | Service | Value |
|-----|---------|-------|
| `POLYGON_API_KEY` | Polygon.io | `JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY` |
| `ANTHROPIC_API_KEY` | Claude API | User-provided |
| `OPENAI_API_KEY` | OpenAI API | User-provided (fallback) |
| StockNews API Key | StockNews | `zzad9pmlwttixx0fnsenstctzgdk7ysx0ctkgrk0` (hardcoded in client.py) |

### Config Storage:
- `~/.nuble/config.json` → API keys, user settings
- `~/.nuble/learning/` → Prediction tracking data, learned weights
- `data/wrds/` → GKX panel parquet (15GB), ticker maps
- `models/` → LightGBM models, HMM models, universal models
- `models/production/` → Production models with registry

---

## 30. Production Configuration & Deployment Scripts

### `config/production_config.py` (268 lines)
**Paper → Small Live → Full Live Deployment Configuration**

Based on validated audit results (February 1, 2026): Alpha 13.8% (t=3.21), PBO 25% (low overfitting), Sharpe 0.41, Beta 1.20.

**`DeploymentPhase` enum**: PAPER ($100K) → SMALL_LIVE ($10K, 10%) → FULL_LIVE ($100K)

**`RiskConfig`** dataclass (validated limits):
- Max position: 10%, Max sector: 30%, Max correlated: 40%
- Gross exposure: 150%, Net exposure: 50% (after beta hedge)
- Drawdown ladder: 5% warning → 10% reduced (50% risk) → 15% minimal (25% risk) → 20% HALT
- Daily limits: 50 trades, 30% turnover, 2% daily loss
- Target vol: 15%, Max vol: 25%

**`BetaHedgeConfig`**: Target beta 0.0 (market-neutral), current beta 1.20, hedge instrument SPY, weekly rebalance. `hedge_ratio = -1.20` → short $120K SPY per $100K long.

**`TransactionCostConfig`**: Mega-cap 48bps, large 75bps, mid 120bps, small 200bps, ETF 30bps. Slippage: 10bps. Min trade: $1,000.

**`UniverseConfig`**: 17 viable symbols (AAPL, MSFT, NVDA, AMD, GOOGL, JPM, BAC, AMZN, TSLA, SPY, QQQ, IWM, GLD, SLV, TLT, IEF). Watch list: NVDA (18% of returns), AMD, TSLA.

**`SignalConfig`**: 20-day momentum/volatility lookback, entry threshold 0.5, exit 0.2, equal-weight default (Kelly optional), daily rebalance.

### Root-Level Scripts

| Script | Lines | Purpose |
|--------|-------|---------|
| `run_production.py` | 274 | Production launch script — configures logging, starts ROKET API with uvicorn, health checks |
| `deploy_webhooks.py` | 531 | Deploys TradingView webhook infrastructure — tests connectivity, configures LuxAlgo alerts |
| `migrate_to_nuble.py` | 464 | Migration script from legacy codebase → NUBLE v6 structure |
| `patch_server.py` | 100 | Hot-patches for production server fixes |
| `test_connections.py` | 369 | Validates ALL external API connections (Polygon, StockNews, CryptoNews, FRED, Lambda, SEC) |
| `test_elite_system.py` | 467 | End-to-end system test — agents, orchestrator, ML, decision engine |
| `setup.py` | 22 | Package setup (references pyproject.toml) |
| `system_manifest.json` | 169 | Production certification manifest — validation phases 1-6 all PASSED, 80/80 tests |

---

## 31. Asset Detection & Crypto Analysis

### `src/nuble/assets/detector.py` (261 lines)
**Multi-Asset Class Detection Engine**

`AssetClass` enum: STOCK, ETF, CRYPTO, COMMODITY, FOREX, UNKNOWN

`AssetDetector` class with comprehensive lookup tables:
- **CRYPTO_SYMBOLS**: 55+ crypto symbols (BTC, ETH, SOL, XRP, ADA, DOT, DOGE, AVAX, LINK, MATIC, UNI, AAVE, ATOM, NEAR, FTM, APE, OP, ARB, SUI, SEI, TIA, JUP, WIF, PEPE + stablecoins)
- **ETF_SYMBOLS**: 80+ ETFs (index: SPY/QQQ/IWM/DIA/VTI/VOO, sector: XLK-XLRE, thematic: ARKK/SMH/SOXX/IBB, leveraged: TQQQ/SQQQ, commodity: GLD/SLV/USO, volatility: VXX/UVXY, bond: HYG/JNK/EMB, international: FXI/INDA/EWJ)
- **COMMODITY_ETFS**: 18 commodity-tracking ETFs
- **FOREX_CURRENCIES**: EUR, GBP, USD, JPY, CHF, AUD, NZD, CAD

Detection priority: Crypto (including X: prefix and pair formats) → Forex → Commodity ETF → ETF → Stock (default)

### `src/nuble/assets/crypto_analyzer.py` (345 lines)
**Unified Crypto Analysis Engine**

Combines CryptoNews + CoinDesk + FinBERT for comprehensive crypto analysis.

`CryptoSignal` enum: STRONG_LONG (+2), LONG (+1), NEUTRAL (0), SHORT (-1), STRONG_SHORT (-2)

`CryptoAnalysis` dataclass: symbol, current_price, price_change_24h, news_sentiment (-1 to +1), sentiment_confidence, news_count, signal, signal_strength, news_headlines, analysis_notes

`CryptoAnalyzer` class:
- Integrates: CryptoNewsClient + CoinDeskClient + FinBERT (optional)
- Signal thresholds: strong > 0.6, weak > 0.2
- Symbol mapping: BTC→["BTC","Bitcoin","BTCUSD"], ETH→["ETH","Ethereum","ETHUSD"], etc.

### `src/nuble/assets/base.py` (114 lines)
Base asset class with symbol normalization and validation utilities.

---

## 32. Institutional Pro Module (`src/institutional/`)

### Overview
The `src/institutional/` directory is a **massive 57,565-line** professional-grade trading system. It represents the "Institutional Pro" tier with advanced capabilities beyond the core NUBLE system.

### Module Architecture:

```
src/institutional/ (57,565 lines total)
├── core/           — Orchestrator, Router, Intent Engine, Claude Synthesizer
├── agents/         — Agent registry and base classes
├── tier2/          — Tier 2 agent system (specialized sub-agents)
│   ├── agents/
│   │   ├── technical/   — Trend integrity, reversal/pullback, volatility state, MTF dominance
│   │   ├── risk/        — Concentration, liquidity, risk gatekeeper
│   │   ├── quality/     — Timing, data integrity
│   │   ├── market/      — Regime transition, event window
│   │   └── adversarial/ — Red team agent
│   ├── orchestrator.py  — Tier 2 orchestrator
│   ├── arbiter.py       — Decision arbiter
│   ├── circuit_breaker.py — Circuit breaker pattern
│   └── escalation.py   — Agent escalation logic
├── ml/             — Deep learning models (see below)
│   ├── torch_models/   — PyTorch neural networks
│   │   ├── financial_lstm.py    (509 lines) — Financial LSTM
│   │   ├── market_transformer.py (556 lines) — Market Transformer
│   │   ├── ensemble_network.py  (634 lines) — Ensemble Network
│   │   ├── regime_classifier.py (737 lines) — HMM Regime Classifier
│   │   ├── data_pipeline.py     (782 lines) — Data pipeline
│   │   ├── trainer.py           (787 lines) — Model trainer
│   │   └── advanced/
│   │       ├── temporal_fusion_transformer.py (959 lines) — TFT
│   │       ├── informer.py                   (846 lines) — Informer
│   │       ├── nbeats.py                     (820 lines) — N-BEATS
│   │       ├── nhits.py                      (591 lines) — N-HiTS
│   │       ├── deepar.py                     (730 lines) — DeepAR
│   │       └── production_trainer.py         (1,035 lines) — Prod trainer
│   ├── features.py         (590 lines) — Feature engineering
│   ├── ensemble.py         (674 lines) — Multi-model ensemble
│   ├── lstm.py              (661 lines) — LSTM implementation
│   ├── transformers.py      (796 lines) — Transformer models
│   ├── regime.py            (547 lines) — Regime detection
│   ├── training.py          (710 lines) — Training pipeline
│   ├── losses/financial_losses.py (310 lines) — Custom financial losses
│   └── training/
│       ├── walk_forward.py       (419 lines) — Walk-forward validation
│       ├── real_data_trainer.py  (826 lines) — Real data training
│       └── train_pretrained.py   (280 lines) — Pre-trained model fine-tuning
├── providers/      — Data provider integrations
│   ├── polygon.py           (531 lines) — Polygon.io provider
│   ├── alpha_vantage.py     (687 lines) — Alpha Vantage provider
│   ├── finnhub.py           (848 lines) — Finnhub provider
│   └── sec_edgar.py         (516 lines) — SEC EDGAR provider
├── filings/        — SEC filing analysis
│   ├── database.py          (472 lines) — DuckDB filing storage
│   ├── analyzer.py          (499 lines) — Claude-powered filing analysis
│   ├── loader.py            (385 lines) — Filing data loader
│   ├── search.py            (317 lines) — Semantic search
│   ├── export.py            (621 lines) — Export functionality
│   └── agent.py             (378 lines) — Filing agent
├── analytics/      — Analysis tools
│   ├── technical.py         (807 lines) — 50+ technical indicators
│   ├── anomaly.py           (651 lines) — Anomaly detection
│   ├── patterns.py          (663 lines) — Chart pattern recognition
│   └── sentiment.py         (473 lines) — Sentiment analysis
├── signals/enhanced_signals.py (877 lines) — Enhanced signal generation
├── streaming/
│   ├── realtime.py          (998 lines) — Real-time data streaming
│   └── signal_engine.py     (870 lines) — Streaming signal engine
├── regime/hmm_detector.py   (929 lines) — Advanced HMM regime detection
├── learning/continuous_learning.py (984 lines) — Continuous learning system
├── hedging/beta_hedge.py    (733 lines) — Beta hedging strategies
├── risk/risk_manager.py     (658 lines) — Risk management
├── backtesting/engine.py    (1,168 lines) — Full backtesting engine
├── labeling/triple_barrier.py (1,031 lines) — Triple barrier labeling
├── features/frac_diff.py    (1,118 lines) — Fractional differentiation
├── validation/
│   ├── model_validator.py   (1,116 lines) — Model validation
│   ├── proper_pbo.py        (469 lines) — Probability of Backtest Overfitting
│   └── sample_weights.py    (435 lines) — Sample weight computation
├── models/
│   ├── primary/ml_primary_signal.py (749 lines) — Primary ML signal
│   └── meta/meta_labeler.py (1,241 lines) — Meta-labeling system
├── costs/transaction_costs.py (461 lines) — Transaction cost modeling
├── sec/                     (994 lines) — SEC data processing
├── mcp/                     (710 lines) — Model Context Protocol server
├── trading/paper_trader.py  (507 lines) — Paper trading simulator
├── cli.py                   (1,009 lines) — Institutional CLI interface
├── config.py                (329 lines) — Institutional configuration
├── ml_pipeline.py           (607 lines) — ML pipeline orchestration
└── examples.py              (245 lines) — Usage examples
```

### Key Highlights of Institutional Module:

**Deep Learning Models** (12+ neural network architectures):
- Financial LSTM, Market Transformer, Ensemble Network, Regime Classifier
- Advanced: Temporal Fusion Transformer (959 lines), Informer (846), N-BEATS (820), N-HiTS (591), DeepAR (730)
- Production trainer (1,035 lines) with walk-forward validation

**Tier 2 Agent System** (specialized sub-agents):
- 5 categories: Technical (4 agents), Risk (3 agents), Quality (2 agents), Market (2 agents), Adversarial (1 red team agent)
- Circuit breaker pattern for fault tolerance
- Escalation logic for agent handoff
- Decision arbiter for conflict resolution

**Multi-Provider Data** (4 providers):
- Polygon.io (531 lines), Alpha Vantage (687 lines), Finnhub (848 lines), SEC EDGAR (516 lines)

**Backtesting & Validation**:
- Full backtesting engine (1,168 lines)
- Walk-forward cross-validation
- Probability of Backtest Overfitting (PBO)
- Triple barrier labeling (1,031 lines)
- Fractional differentiation (1,118 lines)

---

## 33. TENK SEC Filing RAG System (`TENK_SOURCE/`)

### Overview
A standalone SEC 10-K/10-Q filing analysis system integrated into NUBLE's FundamentalAnalystAgent.

### Files:

| File | Lines | Purpose |
|------|-------|---------|
| `src/db.py` | 157 | DuckDB database — sentence-level storage with 384-dim embeddings |
| `src/agent.py` | 158 | Claude-powered filing analysis agent |
| `src/tools.py` | 191 | Tool definitions for Claude function calling |
| `src/terminal.py` | 159 | Interactive terminal interface |
| `src/export.py` | 190 | Export filing data (JSON, CSV, markdown) |
| `src/utils.py` | 122 | Utilities (text chunking, embedding generation) |
| `src/prompts.py` | 60 | System prompts for filing analysis |
| `src/cli.py` | 16 | CLI entry point |
| `db/` | — | DuckDB database files for filings |
| `static/` | — | Static assets |

### Integration with NUBLE:
- `FundamentalAnalystAgent` imports TENK's `FilingsAnalyzer` + `FilingsSearch`
- Semantic search: user query → sentence-transformers → cosine similarity → top-K filing sections
- Claude analyzes matched filing text with financial context
- Supports risk factors, revenue breakdown, management outlook, competitive analysis

---

## 34. Data Sources Summary

| Source | Type | Access | What It Provides |
|--------|------|--------|-----------------|
| **Polygon.io** | Market Data | REST API (API key) | Real-time prices, OHLCV, 50+ technicals, news, options, market breadth |
| **StockNews API** | Intelligence | REST API (24 endpoints) | Sentiment (NLP, 7-day rolling), analyst ratings, earnings, SEC filings, events, trending |
| **CryptoNews API** | Intelligence | REST API (17 endpoints) | Crypto sentiment, whale tracking, institutional flows, regulatory news, DeFi |
| **CoinDesk API** | Crypto Data | REST API + WebSocket | Premium OHLCV+ (minute/hour/day), CADLI/CCIX indices, real-time streaming |
| **WRDS/GKX Panel** | Academic Data | Local parquet (15GB) | 522 features, 20,723 tickers, monthly updated, latest=2024-12-31 |
| **SEC EDGAR** | Filings | REST API + XBRL (free) | 10-K/10-Q filings, 40 fundamental ratios, quality score (A-F), 24 accounting items |
| **FRED** | Macro | REST API | 8 GKX macro variables: treasury yields, credit spreads, inflation, employment, Fed Funds |
| **LuxAlgo** | Technical Signals | TradingView webhooks | Multi-TF signals (1W/1D/4H), buy/sell confirmations (1-12), trend strength |
| **FinBERT** | Sentiment | Local model (PyTorch) | ProsusAI/finbert — financial sentiment NLP, batch inference, GPU/MPS/CPU |
| **HMM Regime** | Regime Detection | Local model (hmmlearn) | 3-state Gaussian HMM (BULL/NEUTRAL/CRISIS), trained on GKX macro panel |
| **CoinGecko** | Crypto Data | REST API (free) | Global crypto market cap, BTC/ETH dominance, coin details, DeFi TVL |
| **Alternative.me** | Sentiment | REST API (free) | Crypto Fear & Greed Index (7-day trend, 0-100 scale) |
| **Lambda API** | Decision Engine | AWS API Gateway | Aggregated intelligence from all above (production Lambda function) |
| **DynamoDB** | Signal Storage | AWS | LuxAlgo signal persistence, decisions, predictions |
| **S3** | Data Storage | AWS | WRDS data backup, model storage (nuble-data-warehouse bucket) |
| **TENK RAG** | SEC Filings | Local DuckDB + embeddings | Sentence-level 10-K/10-Q search with 384-dim sentence-transformer embeddings |

---

## 35. API Endpoint Reference

### ROKET API (FastAPI, Port 8000)

| # | Method | Endpoint | Input | Output |
|---|--------|----------|-------|--------|
| 1 | GET | `/api/health` | — | System status, component readiness, data freshness |
| 2 | GET | `/api/predict/{ticker}` | `?source=auto\|live\|wrds` | ML prediction with signal, score, confidence |
| 3 | GET | `/api/universe` | `?tier=mega\|large\|mid\|small&limit=100` | All stocks with predictions |
| 4 | GET | `/api/regime` | — | HMM macro regime + rule-based |
| 5 | GET | `/api/fundamentals/{ticker}` | — | 522 GKX features |
| 6 | GET | `/api/earnings/{ticker}` | — | Earnings features (SUE, accruals, P/E) |
| 7 | GET | `/api/risk/{ticker}` | — | Risk profile (betas, volatility, momentum) |
| 8 | GET | `/api/insider/{ticker}` | — | Insider + analyst signals |
| 9 | GET | `/api/institutional/{ticker}` | — | Institutional ownership + flows |
| 10 | GET | `/api/analyze/{ticker}` | — | **EVERYTHING** (15 sections) |
| 11 | POST | `/api/analyze` | `{holdings: {AAPL: 0.25}}` | Portfolio batch analysis |
| 12 | POST | `/api/screener` | Filters (score, tier, signal, market cap) | Filtered stock list |
| 13 | GET | `/api/top-picks` | `?n=20&tier=mega&live=false` | Top N picks |
| 14 | GET | `/api/tier/{tier}` | — | Tier-specific predictions |
| 15 | GET | `/api/model-info` | — | Model metadata |
| 16 | GET | `/api/news/{ticker}` | — | News + sentiment |
| 17 | GET | `/api/snapshot/{ticker}` | — | Real-time multi-source snapshot |
| 18 | GET | `/api/sec-quality/{ticker}` | — | SEC quality score (A-F) |
| 19 | GET | `/api/macro` | — | FRED macro environment |
| 20 | GET | `/api/lambda/{ticker}` | — | Lambda decision pass-through |
| 21 | GET | `/api/compare` | `?tickers=AAPL,MSFT` | Side-by-side comparison |
| 22 | POST | `/api/position-size` | `{ticker, portfolio_value, risk_per_trade}` | Kelly criterion sizing |
| 23 | POST | `/webhooks/luxalgo` | TradingView webhook payload | Signal stored |
| 24 | GET | `/signals/{symbol}` | — | Stored LuxAlgo signals |

### Lambda API (AWS Production)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/check/{symbol}` | Full decision engine analysis |
| GET | `/health` | API health check |
| GET | `/signals/{symbol}` | Trading signals |

### Elite Server API (server.py)
| # | Method | Endpoint | Description |
|---|--------|----------|-------------|
| 1 | POST | `/api/chat` (SSE) | Main chat endpoint — streams tokens via Server-Sent Events |
| 2 | POST | `/api/chat/sync` | Synchronous chat (blocking) |
| 3 | WS | `/ws/chat` | WebSocket bidirectional chat |
| 4 | GET | `/api/health` | System health |
| 5 | GET | `/api/status` | Component status |
| 6 | GET | `/api/quote/{symbol}` | Quick quote |
| 7 | GET | `/api/lambda/{symbol}` | Lambda pass-through |
| 8 | GET | `/api/luxalgo/{symbol}` | LuxAlgo signals |
| 9 | GET | `/api/learning/stats` | Learning statistics |
| 10 | GET | `/api/learning/predictions` | All predictions |
| 11 | DEL | `/api/conversation/{id}` | Delete conversation |

### Intelligence API (intelligence.py)
| # | Method | Endpoint | Description |
|---|--------|----------|-------------|
| 1 | GET | `/api/intel/predict/{ticker}` | Single-stock prediction |
| 2 | POST | `/api/intel/predict/batch` | Multi-stock batch prediction (up to 50) |
| 3 | GET | `/api/intel/regime` | HMM regime detection |
| 4 | GET | `/api/intel/top-picks` | Top N ranked stocks |
| 5 | GET | `/api/intel/system-status` | System health + data freshness |
| 6 | GET | `/api/intel/tier-info/{ticker}` | Tier classification + model details |
| 7 | GET | `/api/intel/universe/stats` | Universe coverage statistics |
| 8 | POST | `/api/intel/portfolio/analyze` | Portfolio-level analysis |
| 9 | GET | `/api/intel/tools-schema` | OpenAPI tool definitions for Claude |
| 10 | POST | `/api/intel/chat-with-tools` | Server-side Claude ↔ Tools loop (Pattern B) |

### Multi-Timeframe API (mtf_api.py)
| # | Method | Endpoint | Description |
|---|--------|----------|-------------|
| 1 | POST | `/mtf/webhook` | Receive TradingView signal for any timeframe |
| 2 | GET | `/mtf/decision/{symbol}` | Complete MTF trading decision |
| 3 | GET | `/mtf/signals/{symbol}` | All current signals across timeframes |
| 4 | GET | `/mtf/alignment/{symbol}` | Timeframe alignment score |
| 5 | GET | `/mtf/veto/{symbol}` | Veto engine check |
| 6 | GET | `/mtf/status` | System status for all tracked symbols |
| 7 | POST | `/mtf/cleanup` | Remove expired signals |
| 8 | GET | `/mtf/config` | Current MTF configuration |

### Standalone API (main.py)
| # | Method | Endpoint | Description |
|---|--------|----------|-------------|
| 1 | POST | `/chat` | Orchestrator chat endpoint |
| 2 | POST | `/chat/stream` | SSE streaming chat |
| 3 | WS | `/ws/chat` | WebSocket chat |
| 4 | GET | `/quick/{symbol}` | Quick symbol lookup |
| 5 | GET | `/agents` | List available agents |
| 6 | POST | `/users/profile` | Create/update user profile |
| 7 | GET | `/users/{user_id}/profile` | Get user profile |
| 8 | GET | `/users/{user_id}/conversations` | Get conversation history |
| 9 | GET | `/users/{user_id}/predictions` | Get prediction accuracy |
| 10 | POST | `/feedback` | Submit feedback (1-5 rating) |

---

## 36. File-by-File Index

### Root (`src/nuble/`)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 27 | Package init, v6.0.0, .env loading, console export |
| `__main__.py` | 5 | Entry point → `cli.main()` |
| `cli.py` | 511 | Interactive shell, quick commands, Rich TUI |
| `manager.py` | 1,497 | Central orchestration, fast path, APEX fusion, ML, learning |
| `router.py` | 382 | Smart query routing via regex patterns |
| `services.py` | 949 | Unified services bridge (market data, filings, ML, sentiment, patterns) |
| `helpers.py` | 244 | Token counting, timeout messages, config management |
| `llm.py` | 241 | Claude/OpenAI wrapper with token tracking |
| `lambda_client.py` | 1,113 | Lambda Decision Engine client with circuit breaker |

### Agent (`src/nuble/agent/`)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 0 | Empty init |
| `agent.py` | 240 | LLM planning, action execution, Lambda data injection |
| `prompts.py` | 230 | System prompts (planning, action, answer, summary, compact, ML) |

### Agents (`src/nuble/agents/`)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 51 | Agent exports and registry |
| `orchestrator.py` | 1,732 | Master orchestrator — 9 agents + UDE + ML + Lambda + enrichment |
| `base.py` | 185 | Base agent class, AgentType enum, lazy Anthropic loading |
| `market_analyst.py` | 1,158 | Real-time market analysis (Polygon, StockNews, 50+ indicators) |
| `news_analyst.py` | 867 | News and event analysis (10 StockNews + 7 CryptoNews endpoints) |
| `risk_manager.py` | 569 | Risk assessment — VaR, CVaR, drawdown, correlation |
| `fundamental_analyst.py` | 842 | SEC filings (TENK RAG), valuations, Polygon financials |
| `quant_analyst.py` | 681 | ML signals, factor models, HMM regime |
| `macro_analyst.py` | 651 | FRED macro, yields, sectors, market breadth |
| `portfolio_optimizer.py` | 467 | Risk parity, max-Sharpe, Kelly sizing, trend overlay |
| `crypto_specialist.py` | 653 | Polygon crypto + CoinGecko + Alternative.me + CryptoNews |
| `educator.py` | 152 | Claude-powered financial education |
| `shared_data.py` | 504 | Async SharedDataLayer — 35+ endpoints, per-key dedup cache |

### API (`src/nuble/api/`)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 39 | API module exports |
| `roket.py` | 1,704 | FastAPI REST API — 20+ endpoints |
| `server.py` | 1,143 | Elite server — SSE streaming + WebSocket + mounts all routers |
| `tool_executor.py` | 1,137 | Claude ↔ Tools loop — Pattern B server-side function calling |
| `intelligence.py` | 806 | Intel API — predict, regime, top-picks, universe, portfolio |
| `luxalgo_api.py` | 411 | LuxAlgo webhook router — TradingView alert receiver |
| `mtf_api.py` | 331 | Multi-timeframe API endpoints |
| `main.py` | 483 | Standalone server entry point |
| `roket_tools.py` | 170 | ROKET tool definitions (Python) |
| `roket_tool_definitions.json` | 253 | JSON tool schemas for Claude function calling |

### ML (`src/nuble/ml/`)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 104 | ML module exports and configuration |
| `wrds_predictor.py` | 768 | Multi-tier LightGBM ensemble (4 tier models, 20K+ tickers) |
| `live_predictor.py` | 492 | Live Polygon → LightGBM → composite scoring (0.70×fundamental + 0.30×timing) |
| `hmm_regime.py` | 335 | HMM regime detection (3 states from GKX macro panel, Gaussian HMM) |
| `predictor.py` | 401 | F4 MLPredictor — thread-safe lazy-loading wrapper, 3-model priority chain (CS→Universal→Per-ticker), SHAP, hot-swap |
| `model_manager.py` | 423 | Model lifecycle — 7-day freshness, health checks, background retraining, backtest metrics for CLI |
| `features_v2.py` | 1,075 | Feature engineering v2 — de Prado fractional differentiation, Wilder's RSI, Hurst exponent, cyclical calendar, cross-asset |
| `labeling.py` | 968 | Triple Barrier Labeling (de Prado Ch.3-4) — volatility-scaled barriers, sample weights, meta-labeling |
| `trainer_v2.py` | 1,468 | Training pipeline v2 — PurgedWalkForwardCV (Ch.7), SHAP explanations, isotonic calibration, IC metrics |
| `train_all.py` | 333 | Batch training CLI — default 5 symbols, 3yr Polygon data, TrainingPipeline.run() per symbol |
| `cross_sectional_model.py` | 931 | GKX regression model — cross-sectional rank normalization, Huber loss, Spearman IC evaluation |
| `universal_model.py` | 873 | Universal classifier — ONE LightGBM for ALL stocks, ~45 features (RSI, vol, mean-reversion, Hurst) |
| `universal_features.py` | 1,141 | 120+ features in 6 groups — momentum(25), volatility(20), volume(18), technical(25), microstructure(12), context(12) |
| `validate_pipeline.py` | 324 | 8-test validation suite — synthetic round-trip, SHAP, lazy-load, hot-swap, graceful degradation |
| `backtest/__init__.py` | 19 | Backtest module init |
| `backtest/walk_forward.py` | 1,477 | Walk-forward backtest — expanding window, 21-day rebalance, decile portfolios, per-date scoring |
| `backtest/signal_analysis.py` | 676 | Signal quality — decay (IC at 1-21d horizons), factor exposure, turnover, concentration, regime dependence |

### Data (`src/nuble/data/`)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 37 | Data module exports |
| `data_service.py` | 409 | Unified data access (LRU cache → local disk → S3) |
| `polygon_feature_engine.py` | 1,762 | 600+ WRDS-compatible features from Polygon live data |
| `polygon_feature_engine_v1_backup.py` | 826 | V1 backup of Polygon feature engine (pre-refactor snapshot) |
| `aggregator.py` | 654 | Multi-source parallel data aggregation (MarketSnapshot) |
| `fred_macro.py` | 305 | FRED macro data — 8 GKX variables, yield curve/credit/monetary regimes |
| `sec_edgar.py` | 765 | SEC EDGAR XBRL — 24 accounting items, 40 ratios, quality score A-F |
| `s3_data_manager.py` | 0* | Enterprise S3 data access (transparent local/S3, lazy boto3) |
| `polygon_universe.py` | 560 | Grouped daily bars for entire US universe, monthly parquet files |

*Note: `s3_data_manager.py` is a 0-byte empty file (confirmed via `file` command — not a symlink). The S3 data manager logic (713 lines) is documented in the body of Section 21 from an earlier version of the file.*

### Miscellaneous (`src/nuble/`)
| File | Lines | Purpose |
|------|-------|---------|
| `WRDS` (no .py ext) | 1,561 | Complete WRDS data pipeline specification — AWS RDS + WRDS credentials, 7 dataset specs, 522 GKX feature formulas |
| `core/__init__.py` | 13 | Documents ~3,100 lines of deprecated code removed in v7 (original orchestrator, decision engine, risk manager, helpers, model) |

### Decision (`src/nuble/decision/`)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 54 | Decision module exports |
| `ultimate_engine.py` | 1,662 | 28+ data point institutional decision engine with risk veto |
| `trade_setup.py` | 359 | ATR/Keltner/Kelly trade setup calculator (LONG + SHORT setups) |
| `enrichment_engine.py` | 1,431 | Statistical enrichment (percentiles, z-scores, anomaly detection) |
| `engine_v2.py` | 1,275 | Decision engine v2 — 4-layer architecture (Signal/Context/Validation/Risk) |
| `data_classes.py` | 770 | TradeStrength, Regime, VolatilityState enums + layer score dataclasses |

### Signals (`src/nuble/signals/`)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 114 | Signal module exports |
| `luxalgo_webhook.py` | 603 | LuxAlgo signal receiver + store (TradingView → DynamoDB) |
| `fusion_engine.py` | 792 | Multi-source signal fusion (5 sources: Tech 50%, ML 25%, Sentiment 10%, Regime 10%, Fundamental 5%) |
| `veto_engine.py` | 519 | Institutional multi-TF veto (Weekly has absolute veto power) |
| `position_calculator.py` | 544 | Modified Kelly Criterion — alignment scoring (W40%/D35%/4H25%), ATR stops |
| `timeframe_manager.py` | 765 | Timeframe enum (1W/1D/4h/1h), freshness decay curve, max ages |
| `mtf_fusion.py` | 477 | MTFFusionEngine — TradingDecision + VetoEngine + PositionCalculator |
| `base_source.py` | 251 | NormalizedSignal [-1,+1], SignalSource ABC, CompositeSignalSource |
| `sources/__init__.py` | 17 | Signal sources init |
| `sources/ml_afml.py` | 195 | AFML ML signal source (Advances in Financial Machine Learning) |
| `sources/regime_hmm.py` | 281 | HMM regime signal source (3-state classification) |
| `sources/sentiment_finbert.py` | 236 | FinBERT sentiment source (ProsusAI/finbert NLP) |
| `sources/technical_luxalgo.py` | 174 | LuxAlgo technical source (TradingView signals) |

### Learning (`src/nuble/learning/`)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 28 | Learning module exports |
| `learning_hub.py` | 312 | Singleton coordinator — record/resolve predictions, persist weights |
| `prediction_tracker.py` | 506 | Formal prediction logging with component signals and outcomes |
| `accuracy_monitor.py` | 330 | Rolling accuracy by source/symbol/regime, degradation detection |
| `weight_adjuster.py` | 285 | Dynamic weight optimization (5%-60% bounds, 70/30 smoothing) |
| `resolver.py` | 101 | Background prediction resolution (hourly, fetches prices from Polygon) |

### News (`src/nuble/news/`)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 18 | News module exports |
| `client.py` | 421 | StockNews API client (24 endpoints, async aiohttp) |
| `crypto_client.py` | 441 | CryptoNews API client (17 endpoints, crypto-specific) |
| `coindesk_client.py` | 877 | CoinDesk API integration (articles, categories) |
| `sentiment.py` | 439 | Sentiment analysis utilities (lexicon-based + NLP) |
| `integrator.py` | 541 | Multi-source news integration and deduplication |
| `pipeline.py` | 454 | News processing pipeline (fetch → analyze → score → aggregate) |

### Assets (`src/nuble/assets/`)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 21 | Assets module exports |
| `base.py` | 114 | Base asset class — symbol normalization, validation |
| `detector.py` | 261 | Asset type detection (stock vs crypto vs ETF vs index) |
| `crypto_analyzer.py` | 345 | Crypto-specific analysis utilities |

### Memory (`src/nuble/memory/`)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 15 | Memory module exports |
| `memory_manager.py` | 641 | SQLite-based persistent memory (profiles, conversations, predictions) |

### Cache (`src/nuble/cache/`)
| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 21 | Cache module exports |
| `redis_cache.py` | 499 | Async Redis cache — connection pooling, TTL by timeframe, graceful degradation |

### Infrastructure
| File | Lines | Purpose |
|------|-------|---------|
| `Dockerfile` | 108 | Multi-stage production image (Python 3.11, Gunicorn + Uvicorn) |
| `docker-compose.yml` | 92 | Local dev with Redis + hot reload |
| `infrastructure/aws/cloudformation/master-stack.yaml` | 1,145 | Full AWS stack (VPC, ECS Fargate, ALB, DynamoDB, ElastiCache Redis 7.1, S3, CloudWatch, auto-scaling) |
| `infrastructure/aws/docker-entrypoint.sh` | 113 | Secrets Manager → env vars, S3 data/model sync → exec gunicorn |
| `infrastructure/aws/deploy-production.sh` | 497 | Deploy script (--infra-only, --app-only, --upload-data, --status, --destroy) |
| `.github/workflows/deploy.yml` | 301 | CI/CD pipeline (test → build → push ECR → deploy CF → update ECS → health check) |

### Configuration & Scripts (Root)
| File | Lines | Purpose |
|------|-------|---------|
| `config/production_config.py` | 268 | Production trading config — risk limits, beta hedge, universe, deployment phases |
| `run_production.py` | 274 | Production launch script — configures logging, starts ROKET API |
| `deploy_webhooks.py` | 531 | TradingView webhook deployment + connectivity tests |
| `migrate_to_nuble.py` | 464 | Legacy → NUBLE v6 migration script |
| `patch_server.py` | 100 | Hot-patches for production server |
| `test_connections.py` | 369 | Validates ALL external API connections |
| `test_elite_system.py` | 467 | End-to-end system test (agents, ML, decision engine) |
| `setup.py` | 22 | Package setup entry point |
| `pyproject.toml` | 76 | Package metadata, dependencies, optional extras |
| `system_manifest.json` | 169 | Production certification manifest (80/80 tests passed) |

### TENK SEC Filing RAG (`TENK_SOURCE/src/`)
| File | Lines | Purpose |
|------|-------|---------|
| `db.py` | 157 | DuckDB storage with 384-dim sentence embeddings |
| `agent.py` | 158 | Claude-powered filing analysis agent |
| `tools.py` | 191 | Tool definitions for Claude function calling |
| `terminal.py` | 159 | Interactive terminal interface |
| `export.py` | 190 | Filing export (JSON, CSV, markdown) |
| `utils.py` | 122 | Text chunking, embedding generation |
| `prompts.py` | 60 | System prompts for filing analysis |
| `cli.py` | 16 | CLI entry point |

### Institutional Pro Module (`src/institutional/`) — 57,565 lines
*(See Section 32 for detailed breakdown. Major files listed below.)*

| Category | Key Files | Total Lines |
|----------|-----------|-------------|
| Core (orchestrator, router, intent, synthesizer) | 4 files | ~2,000 |
| Tier 2 Agents (12 specialized sub-agents) | 20+ files | ~6,500 |
| ML/Deep Learning (LSTM, Transformer, TFT, N-BEATS, DeepAR, etc.) | 18 files | ~12,500 |
| Providers (Polygon, Alpha Vantage, Finnhub, SEC EDGAR) | 4 files | ~2,580 |
| Filings (DuckDB, analyzer, search, loader, export) | 6 files | ~2,670 |
| Analytics (technical, anomaly, patterns, sentiment) | 4 files | ~2,590 |
| Signals + Streaming | 3 files | ~2,745 |
| Backtesting + Validation | 5 files | ~3,210 |
| Learning + Hedging + Risk | 3 files | ~2,375 |
| Labels + Features | 2 files | ~2,150 |
| Models (primary + meta) | 2 files | ~1,990 |
| Other (CLI, config, MCP, trading, costs, examples) | 6 files | ~3,760 |

---

## Total Lines of Code (Verified — Fourth-Pass Audit)

| Category | Files | Lines |
|----------|-------|-------|
| **`src/nuble/` Core** (CLI, Manager, Router, Services, Helpers, LLM, Lambda) | 9 | 4,969 |
| **`src/nuble/agent/`** (Agent + Prompts) | 3 | 470 |
| **`src/nuble/agents/`** (Orchestrator + 9 specialists + base + shared) | 13 | 8,512 |
| **`src/nuble/api/`** (ROKET + server + tools + intelligence + luxalgo + mtf + main) | 10 | 6,477 |
| **`src/nuble/ml/`** (LightGBM, HMM, live predictor, training, backtest) | 17 | 11,808 |
| **`src/nuble/data/`** (data service, Polygon feature engine, aggregator, FRED, SEC, S3, universe) | 8 | 4,492 |
| **`src/nuble/decision/`** (UDE, engine_v2, trade setup, enrichment, data classes) | 6 | 5,551 |
| **`src/nuble/signals/`** (LuxAlgo, fusion, veto, position calc, timeframe, MTF, sources) | 13 | 4,968 |
| **`src/nuble/learning/`** (learning hub, prediction tracker, accuracy, weight adjuster, resolver) | 6 | 1,562 |
| **`src/nuble/news/`** (StockNews, CryptoNews, CoinDesk, FinBERT, pipeline, integrator) | 7 | 3,191 |
| **`src/nuble/assets/`** (detector, crypto analyzer, base) | 4 | 741 |
| **`src/nuble/memory/`** (SQLite memory manager) | 2 | 656 |
| **`src/nuble/cache/`** (Redis cache) | 2 | 520 |
| **`src/nuble/` Init files** | 15 | ~550 |
| **`src/nuble/` Subtotal** | **101** | **54,503** |
| | | |
| **`src/institutional/`** (Institutional Pro Module) | ~100 | **57,565** |
| **`TENK_SOURCE/src/`** (SEC Filing RAG) | 9 | **1,062** |
| **Infrastructure** (Dockerfile, CloudFormation, CI/CD, deploy, docker-compose) | 6 | **2,256** |
| **Root Scripts** (production, deploy, test, migrate, config) | 10 | **2,740** |
| | | |
| **GRAND TOTAL** | **~226 files** | **~118,126 lines** |

> **Breakdown**: `src/nuble/` = 54,503 lines (core system), `src/institutional/` = 57,565 lines (institutional pro), `TENK_SOURCE/` = 1,062 lines (SEC RAG), infrastructure = 2,256 lines, root scripts = 2,740 lines.

---

*End of NUBLE Master Documentation — v6.0.0 APEX PREDATOR Edition — Fifth-Pass Deep Audit*
