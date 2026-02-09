# NUBLE-CLI — Complete System Audit & Architecture Document

**Version:** 6.0.0 (APEX PREDATOR Edition)  
**Audit Date:** July 2025  
**Python:** 3.11.14  
**Total Source Files:** 83 Python files  
**Total Lines of Code:** 41,190  
**License:** GPL-3.0  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Complete File Inventory](#3-complete-file-inventory)
4. [Core Runtime Layer](#4-core-runtime-layer)
5. [APEX Dual-Brain Architecture](#5-apex-dual-brain-architecture)
6. [9 Specialized Agents](#6-nine-specialized-agents)
7. [Decision Engine Stack](#7-decision-engine-stack)
8. [ML Pipeline (F1–F4)](#8-ml-pipeline-f1f4)
9. [Signal & Fusion Layer](#9-signal--fusion-layer)
10. [News & Sentiment Layer](#10-news--sentiment-layer)
11. [Learning System](#11-learning-system)
12. [API Server Layer](#12-api-server-layer)
13. [Data, Memory & Cache](#13-data-memory--cache)
14. [External Data Sources](#14-external-data-sources)
15. [Configuration & Environment](#15-configuration--environment)
16. [Data Flow: End-to-End](#16-data-flow-end-to-end)
17. [Key Algorithms & Mathematics](#17-key-algorithms--mathematics)
18. [Known Issues & Technical Debt](#18-known-issues--technical-debt)
19. [Dependency Matrix](#19-dependency-matrix)

---

## 1. Executive Summary

NUBLE-CLI is an institutional-grade AI investment research platform operating as a CLI tool (and optional REST API). It implements an **APEX Dual-Brain Fusion Architecture** where two intelligence paths — a Manager (planning brain) and an Orchestrator (multi-agent brain) — run **in parallel on separate threads**, then their outputs are fused into a single Claude Sonnet 4 response.

### Key Statistics

| Metric | Value |
|--------|-------|
| Total Python files | 83 |
| Total lines of code | 41,190 |
| Specialized AI agents | 9 |
| Decision Engine data points | 28+ |
| External API integrations | 7+ services |
| ML pipeline components | 7 modules (F1-F4) |
| Signal sources | 5+ (LuxAlgo, ML, Sentiment, Regime, Fundamental) |
| LLM | Claude Sonnet 4 (`claude-sonnet-4-20250514`) |

### Architecture at a Glance

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    CLI (cli.py)                              │
│  Banner → Smart Router → Quick Commands → Manager           │
└─────────────────────────────┬───────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │     MANAGER       │
                    │   (Brain #1)      │
                    │  Planning + LLM   │
                    └──┬──────────┬─────┘
            ┌──────────┘          └──────────────┐
            │ (parallel)                          │ (parallel)
            ▼                                     ▼
    ┌───────────────┐                  ┌──────────────────────┐
    │  Fast Path    │                  │    ORCHESTRATOR      │
    │  (no LLM)     │                  │     (Brain #2)       │
    │ Quote/Tech/   │                  │  9 Agents in ||      │
    │ Predict/      │                  │  + DecisionEngine    │
    │ Sentiment     │                  │  + ML Predictor      │
    └───────────────┘                  │  + Enrichment Engine │
                                       │  + Trade Setup       │
                                       │  + Learning Hub      │
                                       └──────────┬───────────┘
                                                  │
                                                  ▼
                                    ┌──────────────────────┐
                                    │ Claude Sonnet 4      │
                                    │ Final Synthesis      │
                                    │ (both paths merged)  │
                                    └──────────────────────┘
                                                  │
                                                  ▼
                                          User Response
```

---

## 2. System Architecture

### 2.1 Directory Structure

```
src/nuble/                          # Main package (41,190 LOC)
├── __init__.py          (27)       # Version, console, .env loading
├── __main__.py          (5)        # Entry point → cli.main()
├── cli.py               (436)      # Interactive shell, commands, banner
├── manager.py           (1,481)    # Brain #1: Planning, fast path, APEX fusion
├── router.py            (382)      # Smart query routing (regex-based intent)
├── services.py          (949)      # Unified services bridge (3 subsystems)
├── helpers.py           (244)      # Token counter, config, commands
├── llm.py               (241)      # LLM wrapper (Anthropic/OpenAI), token tracking
├── lambda_client.py     (1,113)    # AWS Lambda Decision Engine client
│
├── agent/                          # Single-brain agent (Manager's research executor)
│   ├── agent.py         (240)      # Agent class: plan → action → answer
│   └── prompts.py       (230)      # System prompts for planning, answering, actions
│
├── agents/                         # Multi-agent system (Orchestrator's agents)
│   ├── base.py          (185)      # SpecializedAgent ABC, AgentTask, AgentResult
│   ├── orchestrator.py  (1,639)    # Brain #2: Master orchestrator
│   ├── shared_data.py   (504)      # SharedDataLayer: async dedup data fetching
│   ├── market_analyst.py(1,158)    # Polygon price, 50+ indicators, patterns
│   ├── news_analyst.py  (867)      # StockNews, CryptoNews, Fear&Greed
│   ├── quant_analyst.py (681)      # ML signals, factor models, regime
│   ├── fundamental_analyst.py(799) # SEC financials, dividends, TENK RAG
│   ├── macro_analyst.py (615)      # VIX, sectors, yield curve, dollar, commodities
│   ├── risk_manager.py  (569)      # VaR, CVaR, drawdown, correlation, event risk
│   ├── portfolio_optimizer.py(467) # Return analysis, risk parity, rebalancing
│   ├── crypto_specialist.py(653)   # On-chain, DeFi, whale tracking, CoinGecko
│   └── educator.py      (152)      # Financial concept explanations
│
├── decision/                       # Decision & Enrichment Engine stack
│   ├── data_classes.py  (770)      # Enums, dataclasses for all decision layers
│   ├── engine_v2.py     (1,275)    # DecisionEngineV2: 4-layer weighted scoring
│   ├── ultimate_engine.py(1,662)   # UltimateDecisionEngine: 28+ data points
│   ├── enrichment_engine.py(1,370) # Statistical enrichment for Claude
│   └── trade_setup.py   (359)      # ATR stops, Keltner, fractional Kelly sizing
│
├── ml/                             # ML Pipeline (F1-F4, production-grade)
│   ├── __init__.py      (66)       # Public API exports
│   ├── features_v2.py   (1,075)    # F1: Feature engineering pipeline
│   ├── labeling.py      (968)      # F2: Triple-barrier labels, sample weights
│   ├── trainer_v2.py    (1,468)    # F3: Purged WF-CV, LightGBM, calibration
│   ├── predictor.py     (271)      # F4: Thread-safe lazy-loading predictor
│   ├── train_all.py     (333)      # F4: CLI batch training script
│   └── validate_pipeline.py(324)   # F4: 8-test end-to-end validation suite
│
├── signals/                        # Signal source & fusion layer
│   ├── base_source.py   (251)      # SignalSource ABC, NormalizedSignal
│   ├── fusion_engine.py (792)      # Multi-source signal fusion (5 sources)
│   ├── luxalgo_webhook.py(603)     # LuxAlgo signal store (DynamoDB → local)
│   ├── mtf_fusion.py    (477)      # Multi-timeframe fusion
│   ├── position_calculator.py(544) # Risk-based position sizing
│   ├── timeframe_manager.py(765)   # Timeframe hierarchy management
│   ├── veto_engine.py   (519)      # Institutional veto (Weekly has veto power)
│   └── sources/                    # Individual signal source adapters
│       ├── ml_afml.py   (195)      # AFML ML signal source
│       ├── regime_hmm.py(281)      # HMM regime detection source
│       ├── sentiment_finbert.py(236)# FinBERT sentiment source
│       └── technical_luxalgo.py(174)# LuxAlgo technical source
│
├── learning/                       # Continuous learning system
│   ├── learning_hub.py  (312)      # Singleton coordinator
│   ├── prediction_tracker.py(506)  # Prediction logging & storage
│   ├── accuracy_monitor.py(330)    # Accuracy tracking by source/regime
│   ├── weight_adjuster.py(285)     # Dynamic weight optimization
│   └── resolver.py      (101)      # Background prediction resolver
│
├── news/                           # News aggregation & sentiment
│   ├── client.py        (421)      # StockNews API client
│   ├── crypto_client.py (441)      # CryptoNews API client
│   ├── coindesk_client.py(877)     # CoinDesk API client
│   ├── sentiment.py     (439)      # FinBERT sentiment analyzer
│   ├── integrator.py    (541)      # Multi-source news integrator
│   └── pipeline.py      (454)      # News processing pipeline
│
├── api/                            # FastAPI REST server
│   ├── server.py        (1,095)    # v2 Elite API (SSE streaming, quotes, chat)
│   ├── main.py          (483)      # v1 API factory (WebSocket, sessions)
│   ├── luxalgo_api.py   (411)      # LuxAlgo signal endpoints
│   └── mtf_api.py       (331)      # Multi-timeframe API endpoints
│
├── core/                           # Core orchestration utilities
│   ├── unified_orchestrator.py(992)# Alternative orchestrator (tool-use based)
│   ├── tools.py         (415)      # Claude-compatible tool registry
│   ├── tool_handlers.py (894)      # Tool implementations (9 tools)
│   └── memory.py        (499)      # Conversation memory for core orchestrator
│
├── assets/                         # Asset detection & analysis
│   ├── base.py          (114)      # Asset type detection
│   ├── crypto_analyzer.py(344)     # Crypto-specific analysis utilities
│   └── detector.py      (261)      # Ticker/asset detection from text
│
├── data/                           # Data layer
│   └── aggregator.py    (654)      # Multi-source data aggregation
│
├── memory/                         # Persistent memory
│   └── memory_manager.py(641)      # SQLite-based user profiles, conversations
│
└── cache/                          # Caching layer
    └── redis_cache.py   (499)      # Redis async cache with TTL
```

### 2.2 Subsystem Map

| Subsystem | Files | LOC | Purpose |
|-----------|-------|-----|---------|
| Core Runtime | 8 | 3,775 | CLI, Manager, LLM, Router, Services, Helpers |
| Single-Brain Agent | 2 | 470 | Manager's research executor |
| Multi-Agent System | 12 | 8,289 | Orchestrator + 9 specialists + SharedData |
| Decision Engine | 5 | 5,436 | Weighted scoring, enrichment, trade setups |
| ML Pipeline | 7 | 4,505 | Features, labels, training, prediction |
| Signal/Fusion | 11 | 4,837 | Signal sources, fusion, veto, MTF |
| News/Sentiment | 6 | 3,173 | News clients, sentiment, pipeline |
| Learning | 5 | 1,534 | Prediction tracking, accuracy, weights |
| API Server | 4 | 2,320 | FastAPI REST + WebSocket |
| Core Tools | 4 | 2,800 | Tool registry, handlers, memory |
| Assets/Data/Memory | 4 | 1,973 | Asset detection, aggregation, SQLite |
| Cache | 1 | 499 | Redis caching |
| Lambda Client | 1 | 1,113 | AWS Lambda Decision Engine |
| Other | 4 | 166 | __init__, __main__, misc |
| **TOTAL** | **83** | **41,190** | |

---

## 3. Complete File Inventory

### Top 15 Largest Files

| Rank | File | Lines | Purpose |
|------|------|-------|---------|
| 1 | `decision/ultimate_engine.py` | 1,662 | Ultimate Decision Engine (28+ data points) |
| 2 | `agents/orchestrator.py` | 1,639 | Multi-agent orchestrator (Brain #2) |
| 3 | `manager.py` | 1,481 | Manager (Brain #1) — planning, fast path, APEX |
| 4 | `ml/trainer_v2.py` | 1,468 | ML training pipeline (Purged WF-CV, LightGBM) |
| 5 | `decision/enrichment_engine.py` | 1,370 | Statistical enrichment for Claude |
| 6 | `decision/engine_v2.py` | 1,275 | Decision Engine V2 (4-layer scoring) |
| 7 | `agents/market_analyst.py` | 1,158 | Market data + 50+ technicals |
| 8 | `lambda_client.py` | 1,113 | AWS Lambda Decision Engine client |
| 9 | `api/server.py` | 1,095 | FastAPI v2 Elite server |
| 10 | `ml/features_v2.py` | 1,075 | Feature engineering pipeline |
| 11 | `core/unified_orchestrator.py` | 992 | Core tool-use orchestrator |
| 12 | `ml/labeling.py` | 968 | Triple-barrier labeling |
| 13 | `services.py` | 949 | Unified services bridge |
| 14 | `core/tool_handlers.py` | 894 | 9 tool implementations |
| 15 | `news/coindesk_client.py` | 877 | CoinDesk API client |

---

## 4. Core Runtime Layer

### 4.1 `__init__.py` (27 lines)
- Loads `.env` via `python-dotenv` (searches cwd, parent, auto)
- Sets `__version__ = "6.0.0"`
- Creates global `Console()` from Rich library

### 4.2 `cli.py` (436 lines) — Interactive Shell
**Entry point:** `nuble` command (via pyproject.toml `[project.scripts]`)

- **`display_application_banner()`** — Gradient-colored ASCII art banner
- **`show_system_status()`** — Displays component status table (services, Lambda, LuxAlgo, TENK)
- **`handle_quick_command(cmd)`** — Slash commands:
  - `/status` — Component status
  - `/version` — Version + architecture info
  - `/clear` — Clear screen + conversation
  - `/help` — Command reference
  - `/lambda SYMBOL` — Direct Lambda API test
  - `/luxalgo SYMBOL` — LuxAlgo premium signal check
  - `/tenk SYMBOL` — SEC filing RAG search
  - `/quit`, `/exit` — Exit
- **`interactive_shell()`** — Main REPL loop:
  - Creates `Manager()` instance
  - `MAX_CONVERSATION_MESSAGES = 40` — auto-compacts beyond this
  - Routes user input through Manager's `process_prompt()`

### 4.3 `manager.py` (1,481 lines) — Brain #1
**Class:** `Manager`

The Manager is the primary intelligence brain. It handles:

1. **Initialization:**
   - Creates `Agent` (single-brain research executor)
   - Loads ML v2 predictor (F4 pipeline) → fallback to ML v1 (legacy)
   - Loads `UltimateDecisionEngine`
   - Loads `UnifiedServices` + `SmartRouter`
   - Loads `LearningHub` for prediction tracking
   - Loads `OrchestratorAgent` (Brain #2) for APEX tier

2. **Query Processing (`process_prompt()`):**
   - Routes through `SmartRouter` for intent detection
   - **Fast Path** (no LLM): Direct data fetch for quotes, technicals, predictions, sentiment, filings
   - **APEX Path** (research queries):
     1. Launches Orchestrator on background `threading.Thread`
     2. Manager's own Claude planning executes concurrently
     3. Agent executes research steps sequentially (Lambda data injection)
     4. Collects Orchestrator result (waits up to 35s)
     5. Merges both intelligence paths into final Claude prompt
     6. Claude Sonnet 4 synthesizes and responds

3. **Fast Path Methods:**
   - `_fast_quote()` — Polygon prev close, formatted table
   - `_fast_prediction()` — ML model prediction (v2/v1)
   - `_fast_technical()` — RSI, MACD, Bollinger, SMA, ATR table
   - `_fast_patterns()` — Chart pattern detection
   - `_fast_sentiment()` — News sentiment analysis
   - `_fast_filings_search()` — TENK SEC filing search

4. **ML Integration:**
   - `get_ml_prediction(symbol)` — v2 (OHLCV → FeaturePipeline → LightGBM) or v1
   - `_fetch_ohlcv_sync(symbol, days)` — Polygon daily OHLCV fetch
   - `enhance_response_with_ml(symbol, response)` — Injects ML prediction into response

5. **APEX Methods:**
   - `_launch_orchestrator_background(prompt)` — Starts Orchestrator on daemon thread
   - `_collect_orchestrator_result(container, timeout)` — Formats Orchestrator's deep analysis into structured intelligence briefing for final prompt

### 4.4 `router.py` (382 lines) — Smart Query Router
**Class:** `SmartRouter`

Regex-based intent detection that classifies queries into:

| Intent | Fast Path | Example |
|--------|-----------|---------|
| `QUOTE` | ✅ | "AAPL", "AAPL price" |
| `TECHNICAL` | ✅ | "RSI for TSLA" |
| `PREDICTION` | ✅ | "predict AMD" |
| `PATTERN` | ✅ | "patterns for AAPL" |
| `SENTIMENT` | ✅ | "sentiment for NVDA" |
| `FILINGS_SEARCH` | ✅ | "risk factors for AAPL" |
| `FILINGS_ANALYSIS` | ❌ | "analyze AAPL 10-K" |
| `RESEARCH` | ❌ | "Should I buy TSLA?" |
| `COMPARISON` | ❌ | "Compare AAPL vs MSFT" |
| `GENERAL` | ❌ | Anything else |

**Priority order:** Research (checked first to avoid "should I buy" going to prediction) → Prediction → Technical → Pattern → Sentiment → Filings → Comparison → Quote → Research fallback → General.

48 common tickers pre-loaded for fast symbol extraction.

### 4.5 `llm.py` (241 lines) — LLM Wrapper
**Classes:** `LLM`, `TokenTracker`

- Prefers Anthropic (Claude) if `ANTHROPIC_API_KEY` set, falls back to OpenAI
- Default model: `claude-sonnet-4-20250514`
- Methods: `prompt()` (blocking), `prompt_stream()` (streaming)
- Converts OpenAI-style messages to Anthropic format (system prompt extraction, message alternation)
- `TokenTracker` singleton: tracks input/output/cache tokens across session
- Cost estimation: Claude Sonnet 4 pricing ($3/M input, $15/M output)
- JSON retry decorator: up to 3 retries for JSON decode failures

### 4.6 `helpers.py` (244 lines) — Utilities
- `TokenCounter` — tiktoken-based token counting
- `get_timeout_message(elapsed)` — 30+ rotating status messages during processing
- `show_help()`, `handle_help_command()`, `handle_clear_command()`, `handle_compact_command()`
- `load_config()` / `save_config()` — `~/.nuble/config.json` persistence
- `handle_command()` — Legacy command router
- `get_api_key()` — Retrieves API key from config or environment

### 4.7 `lambda_client.py` (1,113 lines) — AWS Lambda Client
**Class:** `NubleLambdaClient`

Production Lambda API client for the NUBLE Decision Engine:
- **Endpoint:** `https://9vyvetp9c7.execute-api.us-east-1.amazonaws.com/production`
- **Retry:** 3 retries with 1.5x exponential backoff, 45s timeout
- **Data classes:** `TechnicalSnapshot`, `IntelligenceSnapshot`, `LambdaAnalysis`
- **Methods:**
  - `health_check()` — API health
  - `get_analysis(symbol)` → `LambdaAnalysis` — Full analysis
  - `get_signals(symbol)` → Dict — Raw signals
- **`LambdaAnalysis`** captures: price, change%, RSI, MACD, SMA stack, ATR, VIX, sentiment, news summary, analyst ratings, LuxAlgo signals (W/D/4H), action, score, confidence, regime
- **Helper functions:** `analyze_symbol()`, `format_analysis_for_context()`, `extract_symbols()`, `is_crypto()`

### 4.8 `services.py` (949 lines) — Unified Services Bridge
**Class:** `UnifiedServices`

Bridges three subsystems (nuble core, institutional, TENK):
- Lazy-initializes: market data, filings, ML, technical, sentiment, patterns
- Service status tracking with `ServiceStatus` dataclass
- Cache with 60s TTL
- Methods: `get_quote()`, `get_technical_indicators()`, `get_prediction()`, `search_filings()`, etc.

---

## 5. APEX Dual-Brain Architecture

### 5.1 How It Works

When a user asks a research question (e.g., "Should I buy TSLA?"):

```
                    Manager.process_prompt("Should I buy TSLA?")
                              │
                    ┌─────────┴──────────┐
                    │                    │
              (main thread)        (daemon thread)
                    │                    │
              ┌─────▼─────┐       ┌──────▼──────┐
              │ Agent.run()│       │ Orchestrator │
              │ Claude     │       │  .process()  │
              │ plans 2-4  │       │              │
              │ research   │       │ 9 agents ||  │
              │ steps      │       │ Decision Eng │
              │            │       │ ML Predictor │
              │ Agent      │       │ Lambda data  │
              │ .action()  │       │ Enrichment   │
              │ executes   │       │ Trade Setup  │
              │ each step  │       │ Learning     │
              │ with Lambda│       │ Claude synth │
              └─────┬──────┘       └──────┬───────┘
                    │                     │
                    └──────────┬──────────┘
                               │
                     ┌─────────▼─────────┐
                     │ Final Claude Call  │
                     │ answer_prompt +    │
                     │ research data +    │
                     │ APEX briefing      │
                     └─────────┬─────────┘
                               │
                        User Response
```

**Key insight:** Zero added latency. The Orchestrator runs **while** the Manager's own planning executes. Results are merged at the end.

### 5.2 Manager (Brain #1) — `manager.py`
- Uses `Agent` (single-brain) for sequential research
- Agent plans research steps as JSON array via Claude
- Each step is executed as an "action" with Lambda data injection
- Results accumulated and fed into final answer prompt

### 5.3 Orchestrator (Brain #2) — `agents/orchestrator.py`
**Class:** `OrchestratorAgent` (1,639 lines)

The Orchestrator's pipeline (executed as `asyncio.run()` on daemon thread):

| Step | Name | Description |
|------|------|-------------|
| 0 | Lambda Data | Fetch real-time intelligence from AWS Lambda |
| 0.5 | Decision Engine | Run UltimateDecisionEngine (28+ data points) |
| 0.7 | ML Predictions | Run v2 ML predictor (OHLCV → features → LightGBM) |
| 1 | Planning | Claude decomposes query into agent tasks |
| 1.5 | SharedDataLayer | Prefetch all data once (async dedup) |
| 2 | Agent Execution | Run 9 agents in parallel (asyncio.gather) |
| 3 | Synthesis | Claude synthesizes all agent outputs |
| 5 | Enrichment | StatisticalEnrichmentEngine processes everything |
| 6 | Trade Setup | TradeSetupCalculator computes entries/stops/targets |
| 7 | Learning | LearningHub records predictions |
| 4 | Response | Claude generates final response |

---

## 6. Nine Specialized Agents

All agents extend `SpecializedAgent` (ABC) from `agents/base.py`.

| Agent | File | Lines | Data Sources | Key Capabilities |
|-------|------|-------|-------------|------------------|
| **Market Analyst** | `market_analyst.py` | 1,158 | Polygon.io | 50+ technical indicators, candlestick patterns, chart patterns, support/resistance, multi-timeframe analysis, volume analysis |
| **News Analyst** | `news_analyst.py` | 867 | StockNews PRO, CryptoNews PRO, Polygon News, Alternative.me FGI, VIX | Multi-source sentiment, analyst ratings, earnings calendar, trending headlines, top mentions, events, sundown digest |
| **Quant Analyst** | `quant_analyst.py` | 681 | Institutional ML models, Polygon | ML signal generation (AFML), factor models, meta-labeling, regime detection, statistical arbitrage |
| **Fundamental Analyst** | `fundamental_analyst.py` | 799 | Polygon Financials, Dividends, News, Details, SMA; StockNews; TENK Filing RAG | SEC financials (income/balance/cashflow), valuation metrics, dividend analysis, SEC filing semantic search |
| **Macro Analyst** | `macro_analyst.py` | 615 | Polygon (VIX, Sectors, Treasury, Dollar, Commodities, Indices, News, SMA), Alt.me FGI, StockNews | Volatility regime, sector rotation, market breadth, yield curve, dollar strength, commodity signals |
| **Risk Manager** | `risk_manager.py` | 569 | Polygon Historical, VIX, Sectors, News; Alt.me FGI; StockNews | VaR (95%, 99%), CVaR, max drawdown, beta, correlation, volatility regime, event risk, sentiment risk |
| **Portfolio Optimizer** | `portfolio_optimizer.py` | 467 | Polygon Historical, SMA, Dividends, News, Prev Close | Return analysis, risk parity, efficient frontier, trend overlay, dividend analysis, rebalancing |
| **Crypto Specialist** | `crypto_specialist.py` | 653 | Polygon, CoinGecko, Alt.me FGI, CryptoNews | On-chain analytics, DeFi protocols, whale tracking, market overview, coin fundamentals, server-side indicators |
| **Educator** | `educator.py` | 152 | Claude LLM | Financial concept explanations, strategy teaching |

### 6.1 SharedDataLayer — `agents/shared_data.py` (504 lines)

Async-native shared data cache that eliminates duplicate HTTP calls:
- All agents read from this layer instead of making their own requests
- Data fetched ONCE per query, shared across all 9 agents
- Uses `aiohttp` with connection pooling (20 connections, 10 per host)
- Per-key `asyncio.Lock` prevents duplicate concurrent fetches
- 60-second TTL cache with `CacheEntry` dataclass
- Tracks fetch count vs cache hit count for monitoring
- **Prefetch method:** `prefetch(symbols, agent_types)` fetches all needed data upfront

---

## 7. Decision Engine Stack

### 7.1 `data_classes.py` (770 lines)
Core data structures for all decision layers:
- **Enums:** `TradeStrength` (5 levels), `Regime` (7 states with alignment/risk multipliers), `VolatilityState` (5 levels)
- **Layer scores:** `SignalLayerScore`, `ContextLayerScore`, `ValidationLayerScore`, `RiskLayerScore`
- **`TradingDecision`** — Complete decision output with all layer scores

### 7.2 DecisionEngineV2 — `engine_v2.py` (1,275 lines)
**4-Layer Weighted Decision System:**

| Layer | Weight | Components |
|-------|--------|-----------|
| Signal Layer | 40% | LuxAlgo multi-TF, momentum (RSI/MACD/Stochastic), trend |
| Context Layer | 30% | Regime detection, sentiment, volatility state, macro |
| Validation Layer | 20% | Historical win rate, backtest confidence, sample size |
| Risk Layer | 10% + VETO | Position limits, drawdown, correlation, liquidity, news blackout, earnings window |

- Confidence thresholds: Very Strong (85+), Strong (70+), Moderate (55+), Weak (40+)
- Position sizing: 0.5%–5.0% of portfolio
- Risk layer has **VETO POWER** — any veto blocks the trade entirely

### 7.3 UltimateDecisionEngine — `ultimate_engine.py` (1,662 lines)
**28+ Data Point Institutional Engine:**

```
DATA SOURCES (28+ data points):
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

+ RISK LAYER (VETO POWER)
  - Max Position Check
  - Drawdown Limit
  - Correlation Check
  - Liquidity Check
  - News Blackout
  - Earnings Window
```

Output: `UltimateDecision` with direction, strength, confidence, trade setup, full reasoning.

### 7.4 EnrichmentEngine — `enrichment_engine.py` (1,370 lines)
**Statistical enrichment that Claude cannot derive from raw numbers:**

The Enrichment Engine does NOT make decisions — it provides statistical context:

| Step | What It Computes |
|------|-----------------|
| 1 | Price & volume enrichment (percentile ranks, z-scores, rates of change) |
| 2 | Technical enrichment (RSI/MACD/Bollinger with historical context) |
| 3 | Sentiment enrichment (NLP scores with percentile ranks) |
| 4 | Fundamental enrichment (valuation metrics with context) |
| 5 | Macro enrichment (VIX, dollar, yield curve context) |
| 6 | Risk enrichment (VaR, drawdown with z-scores) |
| 7 | LuxAlgo preservation (raw signals, no enrichment needed) |
| 8 | ML v1 predictions preservation |
| 8.5 | **ML v2 predictions** (F4 pipeline: OHLCV → features → LightGBM → SHAP) |
| 9 | Anomaly detection (any metric > 2σ flagged) |
| 10 | Divergence detection (price vs sentiment, technicals vs fundamentals) |
| 11 | Conflict detection (cross-source disagreements) |
| 12 | Consensus measurement (direction agreement across all sources) |
| 13 | Data coverage tracking (% of sources reporting) |
| 14 | Intelligence brief generation (formatted text for Claude) |

**Key data structures:**
- `EnrichedMetric` — value + percentile, z-score, rate of change, anomaly flag
- `Divergence` — detected divergence between two sources with severity
- `Anomaly` — statistically unusual observation with historical context
- `ConflictV2` — quantitative conflict measurement (-1 to +1 agreement)
- `ConsensusMetrics` — direction agreement, conviction level
- `EnrichedIntelligence` — complete enriched output with intelligence brief

### 7.5 TradeSetupCalculator — `trade_setup.py` (359 lines)
**Pure mathematical risk management:**

- **ATR-based stops:** Wilder's smoothed ATR(14) × conviction multiplier
  - High conviction: 1.5× ATR (tight)
  - Moderate: 2.0× ATR (standard)
  - Low: 2.5× ATR (wide)
- **Keltner Channel:** SMA(20) ± 2×ATR for dynamic support/resistance
- **Take Profits:** R-multiple targets at 1.5R, 2.5R, 4.0R
- **Position Sizing:** Fractional Kelly Criterion
  - High: 35% Kelly
  - Moderate: 25% Kelly
  - Low: 15% Kelly
- **Volatility checks:** 20d vs 60d vol ratio for regime detection
- **Output:** `TradeSetupResult` with entry, stop, 3 targets, position %, Keltner levels, volatility, risk/reward ratios

---

## 8. ML Pipeline (F1–F4)

### 8.1 Overview

The ML pipeline is a production-grade system built following AFML (Advances in Financial Machine Learning) methodology:

```
Raw OHLCV Data
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ F1: FeaturePipeline (features_v2.py, 1,075 lines)            │
│ • Fractional differentiation (preserve memory, enforce stationarity)│
│ • Cyclical encoding (time-of-day, day-of-week, month)        │
│ • Cross-asset features (SPY, VIX, TLT, GLD, DXY)           │
│ • 85+ technical indicators (returns, vol, momentum, RSI, MACD,│
│   Bollinger, ATR, VWAP, microstructure, regime)              │
│ • Collinearity removal (VIF > 0.95 threshold)                │
│ • Stores fitted_columns for train/predict consistency        │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│ F2: Labeling (labeling.py, 968 lines)                        │
│ • VolatilityEstimator (EWMA, Garman-Klass, Parkinson, Yang-Zhang)│
│ • Triple-barrier labeling (PT × volatility, SL × volatility) │
│ • Sample weighting (uniqueness, return attribution, time decay)│
│ • MetaLabeler (probability of primary model correctness)      │
│ • Diagnostic: label_distribution_report()                     │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│ F3: TrainingPipeline (trainer_v2.py, 1,468 lines)            │
│ • PurgedWalkForwardCV (no look-ahead, gap = max_holding bars)│
│ • LightGBM classifier (GPU if available)                     │
│ • Isotonic calibration per fold                               │
│ • Per-fold metrics: accuracy, precision, recall, profit_factor,│
│   Sharpe, max_drawdown, information_coefficient               │
│ • FinancialMetrics: Sharpe, Sortino, max drawdown, profit factor│
│ • SHAP explanations (handles multiclass shape)                │
│ • Saves: model.txt, calibrators.pkl, feature_pipeline.pkl, metadata.json│
│ • TrainingResults: per-fold + aggregate_metrics               │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│ F4: Integration (predictor.py + train_all.py + validate)      │
│ • MLPredictor: thread-safe, lazy model loading, auto hot-swap │
│ • predict(symbol, df) → {direction, confidence, probabilities,│
│   explanation with SHAP top features, CV IC, training date}   │
│ • train_all.py: CLI batch training across symbols             │
│ • validate_pipeline.py: 8-test end-to-end validation suite    │
│ • Wired into: Orchestrator (step 0.7), Manager (fast path),   │
│   EnrichmentEngine (step 8.5)                                 │
└───────────────────────────────────────────────────────────────┘
```

### 8.2 F1: Feature Engineering — `features_v2.py` (1,075 lines)

**Classes:**
- **`FractionalDifferentiator`** — Fractionally differenced series (d ≈ 0.3-0.5) preserving memory while enforcing stationarity. Uses fixed-width window for production efficiency.
- **`CyclicalEncoder`** — Sin/cos encoding for: hour (if intraday), day-of-week (7), month (12), quarter (4)
- **`CrossAssetFeatures`** — Fetches SPY, VIX, TLT, GLD, DXY from Polygon; computes relative returns, rolling correlations, beta
- **`TechnicalFeatures`** — 85+ features:
  - Multi-horizon returns (1d, 2d, 5d, 10d, 20d)
  - Multi-horizon volatility (5d, 10d, 20d, 60d)
  - Momentum (RSI-14, MACD, Stochastic, Williams %R, CCI, ROC, MFI)
  - Bollinger Band position + width
  - ATR + ATR ratio
  - VWAP ratio
  - Microstructure (Kyle's lambda, Amihud illiquidity, volume imbalance)
  - Regime features (vol ratio, return dispersion)
- **`FeaturePipeline`** — Orchestrates all components:
  - `fit(df)` — Fits pipeline, stores `fitted_columns` after collinearity removal
  - `transform(df)` — Applies pipeline, enforces `fitted_columns` (adds missing cols as 0, drops extras)
  - `fit_transform(df)` — Combined
  - `remove_collinear(threshold=0.95)` — VIF-based collinearity removal

**Critical bugfix (Session 18):** `remove_collinear()` was called during `transform()`, producing different column sets at train vs predict time. Fixed by storing `fitted_columns` during `fit()` and enforcing them during `transform()`.

### 8.3 F2: Labeling — `labeling.py` (968 lines)

**Classes:**
- **`VolatilityEstimator`** — 4 volatility estimators:
  - EWMA (exponentially weighted, span=20)
  - Garman-Klass (uses OHLC, ~7.4× more efficient than close-to-close)
  - Parkinson (uses HL, ~5.2× more efficient)
  - Yang-Zhang (combines overnight and intraday, ~14× more efficient)
  - Ensemble: median of all 4 (robust to outliers)
- **`TripleBarrierLabeler`** — AFML triple-barrier method:
  - Upper barrier: `tp_multiplier × daily_volatility` (default 2.5)
  - Lower barrier: `sl_multiplier × daily_volatility` (default 1.5)
  - Vertical barrier: `max_holding_period` bars (default 10)
  - Labels: +1 (hit TP), -1 (hit SL), 0 (expired at vertical)
- **`SampleWeighter`** — 3-component weights:
  - Uniqueness: average uniqueness of each bar's label
  - Return attribution: |return| as weight proxy
  - Time decay: exponential decay (newer = higher weight)
  - Combined: product of all three, normalized
- **`MetaLabeler`** — Probability of primary model being correct (meta-labeling)
- **`create_labels(df, tp, sl, max_holding)`** — Convenience function
- **`label_distribution_report(labels)`** — Class balance diagnostic

### 8.4 F3: Training — `trainer_v2.py` (1,468 lines)

**Classes:**
- **`PurgedWalkForwardCV`** — Cross-validation with:
  - No look-ahead: purge gap = `max_holding` bars between train/test
  - Minimum training size: 252 bars (1 year)
  - Expanding window by default
- **`FinancialMetrics`** — Computes:
  - Sharpe ratio (annualized)
  - Sortino ratio
  - Maximum drawdown (% and duration)
  - Profit factor (gross profit / gross loss, direction-adjusted)
  - Calmar ratio
  - Information coefficient (correlation of predicted probabilities with actual labels)
- **`ModelTrainer`** — LightGBM training:
  - 3-class classification (short=-1, neutral=0, long=+1)
  - GPU support (auto-detection)
  - Isotonic calibration per fold
  - SHAP explanations (handles multiclass `(n_samples, n_features, n_classes)` shape)
  - Saves: `model.txt`, `calibrators.pkl`, `feature_pipeline.pkl`, `metadata.json`
- **`TrainingPipeline`** — Full pipeline orchestrator:
  - `run(df)` → `TrainingResults` (per-fold + `aggregate_metrics`)
  - `predict_latest(df)` → dict with direction, confidence, probabilities
  - `save(path)` / `load(path)` — Serialization
  - Models saved to `models/{symbol}_{timestamp}/`
- **`TrainingResults`** — Container for:
  - `fold_results`: list of per-fold metrics
  - `aggregate_metrics`: CV aggregate (mean accuracy, Sharpe, IC, etc.)
  - `final_model`: the trained LightGBM model
  - `final_pipeline`: fitted FeaturePipeline

**Bugs fixed (Session 17):**
1. `profit_factor` used raw returns instead of direction-adjusted P&L
2. SHAP multiclass shape handling for SHAP 0.50+ (`(n, features, classes)` → `(n, features)`)

### 8.5 F4: Integration — `predictor.py` + `train_all.py` + `validate_pipeline.py`

**`predictor.py` (271 lines) — MLPredictor:**
- Thread-safe with `threading.Lock`
- Lazy model loading: models loaded on first prediction
- Auto hot-swap: detects new model files and reloads
- `predict(symbol, df)` → dict:
  ```python
  {
      'direction': 'LONG'|'SHORT'|'NEUTRAL',
      'confidence': 0.0-1.0,
      'probabilities': {'short': 0.1, 'neutral': 0.2, 'long': 0.7},
      'explanation': {
          'top_features': [{'feature': 'rsi_14', 'impact': 0.05}, ...],
          'cv_information_coefficient': 0.12,
          'training_date': '2025-07-01'
      }
  }
  ```
- `predict_batch(symbols, df_dict)` — Multi-symbol batch prediction
- Module-level `get_predictor()` singleton

**`train_all.py` (333 lines) — Batch Training CLI:**
- Usage: `python -m nuble.ml.train_all --symbols SPY AAPL --tp 2.5 --sl 1.5`
- Fetches OHLCV from Polygon (falls back to synthetic for testing)
- Configurable: tp/sl multipliers, max holding period, days of data
- Saves models to `models/` directory

**`validate_pipeline.py` (324 lines) — 8-Test Validation Suite:**
1. Synthetic data round-trip (features → labels → train → predict)
2. SHAP explanations present and valid
3. Predictor lazy-loading works
4. Hot-swap detection on new model files
5. Empty/short data handled gracefully
6. Batch prediction across multiple symbols
7. Graceful degradation when no model exists
8. Direction mapping correctness (+1→LONG, -1→SHORT, 0→NEUTRAL)

**Status: All 8/8 tests pass** ✅

### 8.6 ML Integration Points

The ML v2 pipeline is wired into 3 runtime components:

| Component | File | Integration |
|-----------|------|-------------|
| **Orchestrator** | `agents/orchestrator.py` | Step 0.7: fetches OHLCV, runs v2 predictor, SHAP in Claude prompt |
| **Manager** | `manager.py` | Fast path: `_fast_prediction()`, `get_ml_prediction()`, `enhance_response_with_ml()` |
| **Enrichment Engine** | `decision/enrichment_engine.py` | Step 8.5: runs v2 predictor during enrichment, stores in `ml_predictions['v2_prediction']` |

---

## 9. Signal & Fusion Layer

### 9.1 `signals/base_source.py` (251 lines)
- `SignalSource` ABC — interface for all signal sources
- `NormalizedSignal` — direction (-1 to +1), confidence (0-1), freshness

### 9.2 `signals/fusion_engine.py` (792 lines) — Signal Fusion
**5 signal sources with dynamic weights:**

| Source | Default Weight | Description |
|--------|---------------|-------------|
| Technical (LuxAlgo) | 50% | Multi-timeframe technical signals |
| ML (AFML) | 25% | Machine learning model predictions |
| Sentiment (FinBERT) | 10% | NLP-based news sentiment |
| Regime (HMM) | 10% | Hidden Markov Model regime state |
| Fundamental | 5% | Valuation signals |

- Agreement detection: when sources agree, confidence increases
- Regime-adaptive: weights adjust based on market regime
- Continuous learning: weights updated from prediction outcomes
- Output: `FusedSignal` with direction, strength, confidence, position sizing

### 9.3 `signals/luxalgo_webhook.py` (603 lines) — LuxAlgo
- `LuxAlgoSignalStore` — Receives signals from TradingView webhooks via DynamoDB
- Supports: Weekly (1W), Daily (1D), 4-Hour (4H) timeframes
- Signal types: Buy, Sell, Strong Buy, Strong Sell

### 9.4 `signals/veto_engine.py` (519 lines) — Institutional Veto
**Hierarchy rules:**
1. NEVER trade against Weekly trend
2. Daily must align with Weekly or reduce size by 75%
3. 4H triggers entry only after Weekly + Daily alignment
4. Conflicting signals = NO TRADE

Position multipliers: Full alignment (1.0), W+D only (0.75), W neutral (0.50), Counter-trend (0.25), Conflict (0.0)

### 9.5 `signals/timeframe_manager.py` (765 lines)
- Manages timeframe hierarchy and freshness tracking
- `TimeframeSignal` dataclass with direction, confidence, freshness decay

### 9.6 `signals/position_calculator.py` (544 lines)
- Risk-based position sizing using Kelly Criterion
- Volatility-adjusted sizing
- Maximum position limits

### 9.7 Signal Sources (`signals/sources/`)
| Source | File | Lines | Description |
|--------|------|-------|-------------|
| `ml_afml.py` | 195 | AFML ML signal adapter |
| `regime_hmm.py` | 281 | Hidden Markov Model regime detection |
| `sentiment_finbert.py` | 236 | FinBERT sentiment signal |
| `technical_luxalgo.py` | 174 | LuxAlgo technical signal |

---

## 10. News & Sentiment Layer

### 10.1 `news/client.py` (421 lines) — StockNews API Client
- 24 endpoints, PRO tier ($50/mo)
- Ticker news, sentiment stats, top mentions, events, trending headlines, earnings calendar, analyst ratings, sundown digest
- Async with `aiohttp`

### 10.2 `news/crypto_client.py` (441 lines) — CryptoNews API Client
- 17 endpoints, PRO tier ($50/mo)
- Crypto sentiment, top mentions, events, trending headlines
- Topic filtering (DeFi, NFT, Regulations, Mining, etc.)

### 10.3 `news/coindesk_client.py` (877 lines) — CoinDesk Client
- Cryptocurrency news and data from CoinDesk

### 10.4 `news/sentiment.py` (439 lines) — FinBERT Analyzer
- `SentimentAnalyzer` using ProsusAI/finbert model
- Auto-detects device: CUDA → MPS → CPU
- Batch inference with caching
- Output: `SentimentResult` with label, confidence, normalized score (-1 to +1)

### 10.5 `news/integrator.py` (541 lines) — Multi-Source Integrator
- Combines news from all sources
- Cross-source verification and deduplication

### 10.6 `news/pipeline.py` (454 lines) — News Pipeline
- Full news processing pipeline: fetch → deduplicate → sentiment → score → rank

---

## 11. Learning System

### 11.1 Architecture

```
LearningHub (singleton, thread-safe)
    │
    ├── PredictionTracker — logs every prediction with context
    ├── AccuracyMonitor — tracks accuracy by source, symbol, regime
    ├── WeightAdjuster — dynamically adjusts signal weights
    └── PredictionResolver — background hourly resolution
```

### 11.2 `learning/learning_hub.py` (312 lines) — Coordinator
**Thread-safe singleton** coordinating all learning:
- Storage: `~/.nuble/learning/` directory
- `record_prediction(symbol, direction, confidence, price, source, signals)` → prediction_id
- `resolve_predictions(symbol, current_price)` — scores 1d/5d/20d predictions
- `get_signal_weights()` — returns current learned weights
- `get_learning_context()` — formatted track record for Claude prompt
- `get_unresolved()` — outstanding predictions for resolution
- Persists to JSON: `predictions.json`, `raw_predictions.json`, `learned_weights.json`

Default signal weights:
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

### 11.3 `learning/accuracy_monitor.py` (330 lines)
- Tracks accuracy by source, symbol, and regime
- Rolling window (default 100 predictions)
- Degradation detection (accuracy drop > threshold triggers alert)
- `AccuracySnapshot` for trend analysis

### 11.4 `learning/weight_adjuster.py` (285 lines)
- Dynamic weight adjustment based on accuracy performance
- Constraints: min 5%, max 60% per source; smooth adjustments
- Regime-specific weight overrides
- Adjustment history tracking

### 11.5 `learning/prediction_tracker.py` (506 lines)
- `Prediction` dataclass: full prediction with component signals, outcomes
- JSON persistence
- Export for analysis

### 11.6 `learning/resolver.py` (101 lines)
- Background async task (runs every 3600s)
- Fetches current prices from Polygon
- Resolves outstanding predictions (correct/incorrect/partial/expired)

---

## 12. API Server Layer

### 12.1 `api/server.py` (1,095 lines) — v2 Elite API
**FastAPI server** wrapping the full Manager pipeline:

**Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | Full analysis (SSE stream with progress events) |
| POST | `/api/chat/sync` | Full analysis (blocking JSON response) |
| GET | `/api/quote/{symbol}` | Structured quote (clean JSON) |
| GET | `/api/lambda/{symbol}` | Lambda Decision Engine data |
| GET | `/api/luxalgo/{symbol}` | LuxAlgo premium signals |
| GET | `/api/health` | Health check |
| GET | `/api/status` | Full system status |
| DELETE | `/api/conversation/{id}` | Clear conversation |
| WS | `/ws/chat` | WebSocket real-time chat |

**SSE Event Flow:**
```
{"type": "start", "conversation_id": "uuid"}
{"type": "progress", "stage": "routing", "detail": "research"}
{"type": "quote", "data": {"symbol": "TSLA", "price": 411.11}}
{"type": "progress", "stage": "apex_started"}
{"type": "progress", "stage": "agent_done", "agent": "market_analyst"}
{"type": "chunk", "text": "partial markdown..."}
{"type": "response", "text": "full markdown", "metadata": {...}}
{"type": "done", "execution_time": 60.1, "metadata": {...}}
```

**Run:** `uvicorn nuble.api.server:app --host 0.0.0.0 --port 8000` or `nuble-api`

### 12.2 `api/main.py` (483 lines) — v1 API Factory
- Alternative FastAPI app with WebSocket support
- User session management
- Memory integration

### 12.3 `api/luxalgo_api.py` (411 lines)
- LuxAlgo signal CRUD endpoints
- Signal history and alignment status

### 12.4 `api/mtf_api.py` (331 lines)
- Multi-timeframe analysis endpoints
- MTF fusion results

---

## 13. Data, Memory & Cache

### 13.1 `data/aggregator.py` (654 lines)
- Multi-source data aggregation
- Combines Polygon, Lambda, and cached data

### 13.2 `memory/memory_manager.py` (641 lines)
**SQLite-based persistence:**
- `UserProfile` — risk tolerance, portfolio, watchlist, preferences
- `Conversation` — message history with metadata
- `PredictionRecord` — prediction tracking
- CRUD operations with context manager for transactions

### 13.3 `cache/redis_cache.py` (499 lines)
**Async Redis cache:**
- Connection pooling (20 max connections)
- TTL by type: Weekly signals (7d), Daily (24h), 4H (8h), decisions (5min), alignment (1min)
- Key namespaces: `nuble:signal:`, `nuble:decision:`, `nuble:alignment:`, `nuble:veto:`
- Graceful fallback if Redis unavailable

### 13.4 `core/memory.py` (499 lines)
- Conversation memory for core orchestrator
- Message history management

### 13.5 `core/tools.py` (415 lines) + `core/tool_handlers.py` (894 lines)
**Claude-compatible tool registry:**
- 9 tools: `get_stock_quote`, `get_technical_indicators`, `run_ml_prediction`, `search_sec_filings`, `get_news_sentiment`, `analyze_risk`, `get_options_flow`, `get_market_regime`, `compare_stocks`
- Each tool has: name, description, input_schema, handler, cache TTL
- Execution with caching and error handling

### 13.6 `core/unified_orchestrator.py` (992 lines)
- Alternative tool-use based orchestrator
- Pattern matching → execution path routing
- Paths: FAST (no LLM), DECISION (engine), RESEARCH (multi-agent), EDUCATION

### 13.7 `assets/` — Asset Detection (719 lines total)
- `base.py` (114) — Asset type classification
- `crypto_analyzer.py` (344) — Crypto-specific analysis utilities
- `detector.py` (261) — Ticker extraction from text

---

## 14. External Data Sources

### 14.1 API Integrations

| Service | Type | Endpoints Used | Key |
|---------|------|---------------|-----|
| **Polygon.io** | Market Data | Price, OHLCV, technicals (RSI/MACD/SMA/BB), news, financials, dividends, tickers, sector ETFs | `POLYGON_API_KEY` |
| **StockNews API** | News/Sentiment | 24 endpoints — news, sentiment stats, analyst ratings, earnings, events, trending, top mentions | `STOCKNEWS_API_KEY` |
| **CryptoNews API** | Crypto News | 17 endpoints — crypto news, sentiment, events, trending | `CRYPTONEWS_API_KEY` |
| **Alternative.me** | Sentiment | Fear & Greed Index (crypto) | No key |
| **CoinGecko** | Crypto Data | Coin details, global market stats, DeFi TVL | No key |
| **Anthropic** | LLM | Claude Sonnet 4 (messages API) | `ANTHROPIC_API_KEY` |
| **OpenAI** | LLM (fallback) | GPT-4.1 | `OPENAI_API_KEY` |
| **AWS Lambda** | Decision Engine | Custom API Gateway → Lambda → multi-source aggregation | Hardcoded endpoint |
| **TradingView** | LuxAlgo Signals | Webhooks → DynamoDB (external pipeline) | — |

### 14.2 TENK SEC Filing RAG
- Semantic search over 10-K/10-Q SEC filings
- DuckDB + sentence-transformers (384-dim embeddings)
- Topics: risk factors, revenue breakdown, management outlook, competitive position
- Integrated into FundamentalAnalystAgent

---

## 15. Configuration & Environment

### 15.1 Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Yes* | — | Claude Sonnet 4 API access |
| `OPENAI_API_KEY` | No | — | OpenAI fallback |
| `POLYGON_API_KEY` | No | `JHKwAdyIO...` | Polygon.io market data |
| `STOCKNEWS_API_KEY` | No | `zzad9pm...` | StockNews PRO |
| `CRYPTONEWS_API_KEY` | No | `fci3fvh...` | CryptoNews PRO |
| `COINDESK_API_KEY` | No | `78b5a8d...` | CoinDesk API |
| `REDIS_HOST` | No | `localhost` | Redis cache host |
| `REDIS_PORT` | No | `6379` | Redis cache port |
| `REDIS_PASSWORD` | No | — | Redis password |

*One of ANTHROPIC_API_KEY or OPENAI_API_KEY is required.

### 15.2 File-Based Config
- `~/.nuble/config.json` — API keys, preferences
- `~/.nuble/learning/` — Learning system data (predictions, weights)

### 15.3 pyproject.toml Dependencies
```
rich, inquirer, tiktoken, requests, openai, anthropic,
python-dotenv, numpy, aiohttp, fastapi, uvicorn[standard]
```

### 15.4 ML Pipeline Dependencies (installed separately)
```
pandas, numpy, lightgbm, shap, scikit-learn, scipy, joblib, statsmodels, arch
```

---

## 16. Data Flow: End-to-End

### 16.1 Research Query Flow (e.g., "Should I buy TSLA?")

```
1. User types: "Should I buy TSLA?"
   │
2. cli.py → Manager.process_prompt()
   │
3. SmartRouter.route() → QueryIntent.RESEARCH (requires LLM)
   │
4. ┌──── PARALLEL ────────────────────────────────────────────┐
   │                                                           │
   │ BRAIN #1 (main thread):                                  │
   │ a) Agent.run() → Claude plans 3-4 research steps         │
   │ b) For each step:                                        │
   │    - Agent.action() calls Claude with Lambda data        │
   │    - Lambda: get_analysis("TSLA") → price, technicals,   │
   │      sentiment, news, LuxAlgo signals, regime, score     │
   │ c) Research results accumulated                           │
   │                                                           │
   │ BRAIN #2 (daemon thread):                                │
   │ a) Lambda data fetch (Step 0)                            │
   │ b) UltimateDecisionEngine (Step 0.5) → 28+ data points  │
   │ c) ML v2 predictor (Step 0.7) → OHLCV → features →      │
   │    LightGBM → direction, confidence, SHAP features       │
   │ d) Claude plans agent tasks (Step 1)                     │
   │ e) SharedDataLayer prefetch (Step 1.5)                   │
   │ f) 9 agents execute in parallel (Step 2):                │
   │    - MarketAnalyst, NewsAnalyst, QuantAnalyst,           │
   │      FundamentalAnalyst, MacroAnalyst, RiskManager,      │
   │      PortfolioOptimizer, CryptoSpecialist, Educator      │
   │ g) Claude synthesizes agent outputs (Step 3)             │
   │ h) EnrichmentEngine enriches everything (Step 5)         │
   │    - Percentiles, z-scores, anomalies, divergences,      │
   │      conflicts, consensus, ML v2 predictions             │
   │ i) TradeSetupCalculator (Step 6) → entry, stop, targets  │
   │ j) LearningHub records prediction (Step 7)              │
   │ k) Claude generates response (Step 4)                    │
   │                                                           │
   └──────────────────────────────────────────────────────────┘
   │
5. Manager collects Orchestrator result (waits ≤ 35s)
   │
6. Final Claude call with:
   - answer_prompt (system prompt with APEX synthesis protocol)
   - Research data from Brain #1
   - APEX intelligence briefing from Brain #2
   - Original question
   │
7. Claude Sonnet 4 synthesizes everything into response
   │
8. Response streamed to user via Rich markdown rendering
```

### 16.2 Fast Path Flow (e.g., "AAPL")

```
1. User types: "AAPL"
   │
2. SmartRouter.route() → QueryIntent.QUOTE (fast_path=True)
   │
3. Manager._fast_quote("AAPL")
   │
4. Polygon.io /v2/aggs/ticker/AAPL/prev → price data
   │
5. Formatted Rich table displayed immediately (no LLM call)
```

---

## 17. Key Algorithms & Mathematics

### 17.1 Triple-Barrier Labeling (F2)
```
Upper barrier: current_price × (1 + tp × daily_vol)
Lower barrier: current_price × (1 - sl × daily_vol)
Vertical barrier: t + max_holding_period

Label = +1 if upper barrier hit first
Label = -1 if lower barrier hit first
Label =  0 if vertical barrier hit (expired)
```

### 17.2 Fractional Differentiation (F1)
```
d ∈ [0, 1], typically 0.3-0.5
Preserves long-memory while achieving stationarity
Uses fixed-width window (default 100) for O(n) computation
```

### 17.3 Purged Walk-Forward CV (F3)
```
|----TRAIN----|--GAP--|--TEST--|
              ↑
        purge gap = max_holding bars
        
No information from test period leaks into training.
Expanding window: each fold adds more training data.
Minimum training size: 252 bars (1 year of daily data).
```

### 17.4 ATR-Based Trade Setup (Decision)
```
ATR(14) = mean(True Range over 14 bars)
True Range = max(H-L, |H-Cprev|, |L-Cprev|)

Stop Loss = Entry ± (ATR × conviction_multiplier)
TP1 = Entry ± (Stop_Distance × 1.5)  [1.5R]
TP2 = Entry ± (Stop_Distance × 2.5)  [2.5R]
TP3 = Entry ± (Stop_Distance × 4.0)  [4.0R]
```

### 17.5 Fractional Kelly Position Sizing
```
f* = (p × b - q) / b     [Full Kelly]
where p = win rate, b = avg win / avg loss, q = 1 - p

Position = f* × fraction_multiplier
  High conviction:    35% Kelly
  Moderate conviction: 25% Kelly
  Low conviction:      15% Kelly
```

### 17.6 Decision Engine Scoring
```
Final Score = Σ (layer_score × layer_weight)
  Signal Layer:     40% × score
  Context Layer:    30% × score
  Validation Layer: 20% × score
  Risk Layer:       10% × score (+ VETO power)

Confidence thresholds:
  Very Strong: ≥ 85
  Strong:      ≥ 70
  Moderate:    ≥ 55
  Weak:        ≥ 40
  No Trade:    < 40
```

### 17.7 Statistical Enrichment
```
Percentile rank: position in 90-day historical range (0-100)
Z-score: (value - μ_20d) / σ_20d (standard deviations from mean)
Rate of change: (value - value_5d_ago) / value_5d_ago
Anomaly: |z_score| > 2.0 triggers flag
Consensus: weighted direction agreement across all sources
```

---

## 18. Known Issues & Technical Debt

### 18.1 Active Issues

| # | Category | Description | Severity |
|---|----------|-------------|----------|
| 1 | **ML Models** | No pre-trained models ship with the repo. Must run `train_all.py` first. Predictor gracefully degrades (returns empty dict). | Low |
| 2 | **Institutional ML** | Legacy `institutional/ml/` imports fail (`ModuleNotFoundError`). v2 pipeline replaces this but v1 fallback paths still reference it. | Low |
| 3 | **API Keys** | Hardcoded fallback API keys in source code (Polygon, StockNews, CryptoNews, CoinDesk). Should use env-only. | Medium |
| 4 | **Redis** | Redis cache is optional but not integrated into the main pipeline. Only used if Redis is running locally. | Low |
| 5 | **FinBERT** | `sentiment.py` requires `torch` + `transformers` which aren't in `pyproject.toml` dependencies. Will fail at import. | Medium |
| 6 | **asyncio/threading** | Manager uses `asyncio.run()` inside ThreadPoolExecutor for nested event loop scenarios. Works but fragile. | Low |
| 7 | **Token explosion** | Auto-compact at 40 messages, but long APEX briefings can push single prompts near token limits. | Low |
| 8 | **Dual orchestrator** | Both `core/unified_orchestrator.py` and `agents/orchestrator.py` exist. Only `agents/orchestrator.py` is used at runtime. `core/` is an older, unused alternative. | Low |
| 9 | **Missing pyproject deps** | `numpy`, `pandas`, `lightgbm`, `shap`, `scikit-learn`, `scipy`, `joblib`, `statsmodels`, `arch`, `torch`, `transformers`, `sentence-transformers` not in pyproject.toml. | Medium |

### 18.2 Code Quality Notes

| Area | Status | Notes |
|------|--------|-------|
| Error handling | ✅ Good | Try/except throughout with graceful degradation |
| Logging | ✅ Good | Consistent `logger = logging.getLogger(__name__)` |
| Type hints | ⚠️ Partial | Most function signatures typed, some internal methods not |
| Tests | ⚠️ Limited | ML pipeline has 8/8 validation tests; other subsystems lack formal tests |
| Documentation | ✅ Good | Extensive docstrings in most files |
| Thread safety | ✅ Good | MLPredictor uses locks; LearningHub is thread-safe singleton |
| Import guards | ✅ Good | All optional imports wrapped in try/except with availability flags |

### 18.3 Unused/Orphaned Code
- `core/unified_orchestrator.py` (992 lines) — alternative orchestrator, not used at runtime
- `core/tools.py` + `core/tool_handlers.py` (1,309 lines) — tool registry for the unused core orchestrator
- `core/memory.py` (499 lines) — memory for unused core orchestrator
- `api/main.py` (483 lines) — v1 API, superseded by `api/server.py`
- Parts of `signals/` layer (fusion_engine, veto_engine, etc.) — designed for automated trading, not fully wired into CLI research flow

**Estimated dead code:** ~3,500–4,000 lines (~9% of codebase)

---

## 19. Dependency Matrix

### 19.1 Runtime Dependencies (Critical Path)

```
cli.py
  └── manager.py
        ├── agent/agent.py → agent/prompts.py → llm.py
        ├── agents/orchestrator.py
        │     ├── agents/base.py
        │     ├── agents/shared_data.py (aiohttp)
        │     ├── agents/* (9 specialized agents)
        │     ├── decision/enrichment_engine.py
        │     ├── decision/trade_setup.py
        │     ├── decision/ultimate_engine.py
        │     ├── ml/predictor.py → ml/trainer_v2.py → ml/features_v2.py → ml/labeling.py
        │     ├── learning/learning_hub.py
        │     └── lambda_client.py
        ├── router.py
        ├── services.py
        ├── decision/ultimate_engine.py
        ├── ml/predictor.py
        ├── learning/learning_hub.py
        └── lambda_client.py
```

### 19.2 Python Package Dependencies

| Package | Version | Used By |
|---------|---------|---------|
| `anthropic` | — | llm.py, orchestrator, agents |
| `rich` | — | cli.py, manager.py, all display |
| `requests` | — | Lambda client, agents, Polygon API |
| `aiohttp` | — | SharedDataLayer, news clients |
| `numpy` | 2.3.5 | ML pipeline, decision engines, signals |
| `pandas` | 3.0.0 | ML pipeline, data aggregation |
| `lightgbm` | 4.6.0 | trainer_v2.py (model training) |
| `shap` | 0.50.0 | trainer_v2.py (explanations) |
| `scikit-learn` | 1.8.0 | trainer_v2.py (calibration, metrics) |
| `scipy` | 1.17.0 | features_v2.py (statistics) |
| `joblib` | 1.5.3 | trainer_v2.py (serialization) |
| `statsmodels` | 0.14.6 | features_v2.py (ARFIMA) |
| `tiktoken` | — | helpers.py (token counting) |
| `python-dotenv` | — | __init__.py (.env loading) |
| `fastapi` | — | api/server.py |
| `uvicorn` | — | API server runner |

---

## Appendix A: Entry Points

| Entry Point | Command | Description |
|-------------|---------|-------------|
| CLI | `nuble` or `python -m nuble` | Interactive financial research shell |
| API Server | `nuble-api` | FastAPI REST server |
| ML Training | `python -m nuble.ml.train_all --symbols SPY AAPL` | Batch model training |
| ML Validation | `python -m nuble.ml.validate_pipeline` | Run 8-test validation suite |

## Appendix B: Model Storage

```
models/
├── {symbol}_{timestamp}/
│   ├── model.txt              # LightGBM model (text format)
│   ├── calibrators.pkl        # Isotonic calibration per class
│   ├── feature_pipeline.pkl   # Fitted FeaturePipeline (with fitted_columns)
│   └── metadata.json          # Training config, metrics, feature names
│
├── mlp_AMD_20260130.pt       # Legacy PyTorch models (v1)
├── mlp_SLV_20260130.pt
├── mlp_SPY_20260130.pt
├── mlp_TSLA_20260130.pt
└── mlp_XLK_20260130.pt
```

## Appendix C: Session History (Implementation Timeline)

| Session | Prompt | What Was Built |
|---------|--------|----------------|
| 1-10 | Various | Core CLI, agents, APIs, architecture, v2 Elite rewrite |
| 11 | — | LearningHub wired into runtime |
| 12 | — | SharedDataLayer performance optimization |
| 13 | — | Statistical Enrichment Engine |
| 14 | E | trade_setup.py, orchestrator pipeline, Claude prompt, manager APEX |
| 15 | F1 | features_v2.py (85+ features, frac diff, cross-asset) |
| 16 | F1 re-audit + F2 | labeling.py (triple-barrier, sample weights, meta-labeling) |
| 17 | F3 | trainer_v2.py (Purged WF-CV, LightGBM, SHAP, calibration) |
| 18 | F4 | predictor.py, train_all.py, validate_pipeline.py, integration wiring, 8/8 tests pass |

---

*End of Complete System Audit — 41,190 lines across 83 files, audited July 2025*
