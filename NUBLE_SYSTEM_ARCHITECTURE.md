# NUBLE System Architecture — Complete Technical Documentation

> **Version:** 6.0.0 | **Codename:** APEX Dual-Brain Fusion  
> **Generated:** June 2025  
> **Goal:** _"Create the most advanced financial expert in the entire world"_

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Project Structure](#2-project-structure)
3. [Entry Points & Configuration](#3-entry-points--configuration)
4. [Core Architecture: APEX Dual-Brain Fusion](#4-core-architecture-apex-dual-brain-fusion)
5. [Module-by-Module Deep Dive](#5-module-by-module-deep-dive)
   - 5.1 [Manager (`manager.py`)](#51-manager-managerpy)
   - 5.2 [LLM Wrapper (`llm.py`)](#52-llm-wrapper-llmpy)
   - 5.3 [Smart Router (`router.py`)](#53-smart-router-routerpy)
   - 5.4 [Agent (`agent/agent.py`)](#54-agent-agentagentpy)
   - 5.5 [Agent Prompts (`agent/prompts.py`)](#55-agent-prompts-agentpromptspy)
   - 5.6 [CLI Interface (`cli.py`)](#56-cli-interface-clipy)
   - 5.7 [Helpers (`helpers.py`)](#57-helpers-helperspy)
6. [APEX Orchestrator System](#6-apex-orchestrator-system)
   - 6.1 [Orchestrator Agent (`agents/orchestrator.py`)](#61-orchestrator-agent-agentsorchestratopy)
   - 6.2 [Base Agent (`agents/base.py`)](#62-base-agent-agentsbasepy)
   - 6.3 [9 Specialist Agents](#63-9-specialist-agents)
7. [Lambda Decision Engine Client](#7-lambda-decision-engine-client)
8. [Ultimate Decision Engine](#8-ultimate-decision-engine)
9. [Unified Services Layer](#9-unified-services-layer)
10. [Signal Processing Pipeline](#10-signal-processing-pipeline)
11. [News & Sentiment System](#11-news--sentiment-system)
12. [ML & Learning System](#12-ml--learning-system)
13. [API Server (v2 Elite)](#13-api-server-v2-elite)
14. [Data Sources & API Keys](#14-data-sources--api-keys)
15. [Request Lifecycle — End-to-End Flow](#15-request-lifecycle--end-to-end-flow)
16. [Infrastructure](#16-infrastructure)
17. [Token & Cost Management](#17-token--cost-management)
18. [File Reference Matrix](#18-file-reference-matrix)

---

## 1. System Overview

NUBLE is an **institutional-grade AI investment research platform** that fuses multiple intelligence paths into a single, comprehensive financial analysis system. It operates as both a **CLI REPL** and a **production REST API** (FastAPI).

### Core Innovation: APEX Dual-Brain Fusion

The system runs **two independent intelligence paths in parallel** and fuses them into a single response:

| Path | Brain | Components | Latency |
|------|-------|------------|---------|
| **Path 1** (Manager) | Claude Sonnet 4 | Sequential planning → research steps → Lambda data injection → streamed answer | 10-30s |
| **Path 2** (Orchestrator) | Claude Sonnet 4 + 9 Agents | Parallel 9-agent execution → DecisionEngine (28+ data points) → ML Predictor (46M+ params) → Claude synthesis | 15-60s |

The Orchestrator launches in a **background thread** at the start of Path 1's planning phase. By the time Path 1 finishes planning, Path 2 has been running concurrently. The results are injected into the final answer prompt so Claude can synthesize **both** intelligence sources.

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11.14 |
| Primary LLM | Claude Sonnet 4 (`claude-sonnet-4-20250514`) via Anthropic SDK |
| Fallback LLM | OpenAI GPT-4.1 |
| API Framework | FastAPI 0.128+ / Uvicorn |
| Terminal UI | Rich (Console, Live, Panel, Markdown, Spinner, Table) |
| ML | PyTorch (MLP, LSTM, Transformer, N-BEATS ensemble — 46M+ params) |
| NLP | FinBERT (`ProsusAI/finbert`) for financial sentiment |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) for SEC filing RAG |
| Market Data | Polygon.io (prices, technicals, news) |
| News APIs | StockNews PRO (24 endpoints), CryptoNews PRO (17 endpoints) |
| Signals | LuxAlgo Premium (TradingView webhooks → DynamoDB → Lambda) |
| Cloud | AWS Lambda + API Gateway + DynamoDB |
| Database | DuckDB (TENK SEC filing vector store) |

---

## 2. Project Structure

```
NUBLE-CLI/
├── pyproject.toml                    # Package config, dependencies, entry points
├── setup.py                          # Legacy setuptools
├── Dockerfile                        # Container deployment
├── run_production.py                 # Production launcher
├── system_manifest.json              # Component registry
│
├── src/nuble/                        # ═══ MAIN PACKAGE ═══
│   ├── __init__.py                   # Version "6.0.0", Rich Console, dotenv
│   ├── __main__.py                   # python -m nuble
│   ├── cli.py                        # CLI REPL (437 lines) — banner, commands, main loop
│   ├── manager.py                    # Core orchestration (1315 lines) — THE BRAIN
│   ├── llm.py                        # LLM wrapper — Claude/OpenAI streaming
│   ├── router.py                     # SmartRouter (383 lines) — intent detection
│   ├── services.py                   # UnifiedServices (950 lines) — bridges subsystems
│   ├── helpers.py                    # TokenCounter, commands, config persistence
│   ├── lambda_client.py              # Lambda Decision Engine client (1114 lines)
│   │
│   ├── agent/                        # ═══ MANAGER'S OWN AGENT (PATH 1) ═══
│   │   ├── agent.py                  # Agent class — planning, action, answer, summarize
│   │   └── prompts.py               # System prompts (agent, answer, action, summary)
│   │
│   ├── agents/                       # ═══ APEX ORCHESTRATOR AGENTS (PATH 2) ═══
│   │   ├── base.py                   # SpecializedAgent ABC, AgentType enum, data classes
│   │   ├── orchestrator.py           # OrchestratorAgent (1261 lines) — 9-agent coordinator
│   │   ├── market_analyst.py         # Polygon.io technicals, StockNews, earnings (1052 lines)
│   │   ├── news_analyst.py           # StockNews PRO + CryptoNews PRO all endpoints (713 lines)
│   │   ├── risk_manager.py           # Position risk, VaR, correlations, stress tests
│   │   ├── fundamental_analyst.py    # Polygon financials + TENK SEC Filing RAG (799 lines)
│   │   ├── quant_analyst.py          # ML signals, backtests, factor models
│   │   ├── macro_analyst.py          # Fed, economic indicators, rates, geopolitics
│   │   ├── portfolio_optimizer.py    # Asset allocation, rebalancing
│   │   ├── crypto_specialist.py      # On-chain, DeFi, whale tracking
│   │   └── educator.py              # Explanations, strategies, terminology
│   │
│   ├── api/                          # ═══ REST API SERVER ═══
│   │   ├── server.py                 # FastAPI v2 Elite (1038 lines) — SSE, WebSocket
│   │   ├── main.py                   # Legacy API entry
│   │   ├── luxalgo_api.py            # LuxAlgo webhook receiver
│   │   └── mtf_api.py               # Multi-timeframe API
│   │
│   ├── decision/                     # ═══ DECISION ENGINE ═══
│   │   ├── ultimate_engine.py        # UltimateDecisionEngine (1663 lines) — 28+ data points
│   │   ├── engine_v2.py              # Decision Engine V2
│   │   └── data_classes.py           # Enums, dataclasses for all layers (771 lines)
│   │
│   ├── signals/                      # ═══ SIGNAL PROCESSING ═══
│   │   ├── fusion_engine.py          # Signal Fusion Engine (793 lines) — multi-source fusion
│   │   ├── luxalgo_webhook.py        # LuxAlgo TradingView → DynamoDB
│   │   ├── mtf_fusion.py            # Multi-timeframe signal fusion
│   │   ├── position_calculator.py    # Position sizing engine
│   │   ├── timeframe_manager.py      # Timeframe coordination
│   │   ├── veto_engine.py           # Risk veto system
│   │   ├── base_source.py           # Base signal source class
│   │   └── sources/                  # Individual signal sources
│   │       ├── ml_afml.py           # AFML/Triple Barrier ML signals
│   │       ├── regime_hmm.py        # HMM regime detection (3-state)
│   │       ├── sentiment_finbert.py  # FinBERT sentiment signals
│   │       └── technical_luxalgo.py  # LuxAlgo technical signals
│   │
│   ├── news/                         # ═══ NEWS PIPELINE ═══
│   │   ├── client.py                # News aggregation client
│   │   ├── coindesk_client.py       # CoinDesk crypto news
│   │   ├── crypto_client.py         # CryptoNews API client
│   │   ├── integrator.py            # Multi-source news integrator
│   │   ├── pipeline.py              # News processing pipeline
│   │   └── sentiment.py             # FinBERT sentiment analyzer (440 lines)
│   │
│   ├── learning/                     # ═══ ADAPTIVE LEARNING ═══
│   │   ├── accuracy_monitor.py      # Prediction accuracy tracking
│   │   ├── prediction_tracker.py    # Outcome logging & evaluation
│   │   └── weight_adjuster.py       # Dynamic weight adjustment
│   │
│   ├── core/                         # ═══ CORE UTILITIES ═══
│   │   ├── memory.py               # Conversation memory
│   │   ├── tools.py                # Tool definitions
│   │   ├── tool_handlers.py        # Tool execution handlers
│   │   └── unified_orchestrator.py # Unified orchestration layer
│   │
│   ├── memory/                       # ═══ MEMORY MANAGEMENT ═══
│   │   └── memory_manager.py       # Long-term memory storage
│   │
│   ├── data/                         # ═══ DATA AGGREGATION ═══
│   │   └── aggregator.py           # Multi-source data aggregation
│   │
│   └── cache/                        # ═══ CACHING ═══
│       └── redis_cache.py           # Redis caching layer
│
├── src/institutional/                # ═══ INSTITUTIONAL SUBSYSTEM ═══
│   └── ml/                          # ML training, registry, prediction
│       └── registry.py              # Pre-trained model registry
│
├── TENK_SOURCE/                      # ═══ SEC FILING RAG ═══
│   └── src/                         # DuckDB + sentence-transformers
│
├── models/                           # ═══ TRAINED MODELS ═══
│   ├── mlp_AMD_20260130.pt
│   ├── mlp_SLV_20260130.pt
│   ├── mlp_SPY_20260130.pt
│   ├── mlp_TSLA_20260130.pt
│   └── mlp_XLK_20260130.pt
│
├── config/
│   └── production_config.py
├── data/
│   ├── paper_trading_state.json
│   ├── historical/
│   ├── test/
│   └── train/
├── infrastructure/
│   └── aws/                         # CloudFormation, Lambda configs
├── tests/                           # 30+ test files
├── training_results/                # ML training output
└── validation/                      # Validation data
```

---

## 3. Entry Points & Configuration

### Entry Points (from `pyproject.toml`)

| Command | Module | Purpose |
|---------|--------|---------|
| `nuble` | `nuble.cli:main` | Interactive CLI REPL |
| `nuble-api` | `nuble.api.server:run` | FastAPI REST server |

### Dependencies

```toml
dependencies = [
    "rich",                    # Terminal UI (Console, Live, Panel, Markdown)
    "inquirer",                # Interactive prompts
    "tiktoken",                # Token counting (OpenAI tokenizer)
    "requests",                # HTTP client
    "openai>=1.0",             # OpenAI API (fallback)
    "anthropic",               # Claude API (primary)
    "python-dotenv",           # .env loading
    "numpy",                   # Numerical computation
    "aiohttp",                 # Async HTTP client
    "fastapi>=0.100.0",        # REST API framework
    "uvicorn[standard]",       # ASGI server
]
```

### Environment Variables

| Variable | Purpose | Fallback |
|----------|---------|----------|
| `ANTHROPIC_API_KEY` | Claude API access (primary LLM) | Required for APEX |
| `OPENAI_API_KEY` | OpenAI API access (fallback LLM) | Optional |
| `POLYGON_API_KEY` | Market data (prices, technicals, news, financials) | Hardcoded: `JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY` |
| `STOCKNEWS_API_KEY` | StockNews PRO (24 endpoints) | Hardcoded: `zzad9pmlwttixx0fnsenstctzgdk7ysx0ctkgrk0` |
| `CRYPTONEWS_API_KEY` | CryptoNews PRO (17 endpoints) | Hardcoded: `fci3fvhrbxocelhel4ddc7zbmgsxnq1zmwrkxgq2` |
| `NUBLE_API_HOST` | API server bind address | `0.0.0.0` |
| `NUBLE_API_PORT` | API server port | `8000` |
| `AWS_ACCESS_KEY_ID` | DynamoDB access (LuxAlgo signals) | AWS profile |
| `AWS_SECRET_ACCESS_KEY` | DynamoDB access | AWS profile |

### Crypto Ticker Mapping

The system maps common crypto abbreviations to Polygon.io format:

```python
CRYPTO_TICKERS = {
    'BTC': 'X:BTCUSD', 'ETH': 'X:ETHUSD', 'SOL': 'X:SOLUSD',
    'XRP': 'X:XRPUSD', 'ADA': 'X:ADAUSD', 'DOGE': 'X:DOGEUSD',
    'AVAX': 'X:AVAXUSD', 'DOT': 'X:DOTUSD', 'MATIC': 'X:MATICUSD',
    'LINK': 'X:LINKUSD', 'UNI': 'X:UNIUSD', 'AAVE': 'X:AAVEUSD',
    'ATOM': 'X:ATOMUSD', 'NEAR': 'X:NEARUSD', 'SHIB': 'X:SHIBUSD',
    'LTC': 'X:LTCUSD', 'FIL': 'X:FILUSD',
}
```

---

## 4. Core Architecture: APEX Dual-Brain Fusion

This is the heart of NUBLE. When a user asks a complex question like _"Should I buy TSLA?"_, the system activates two independent intelligence paths that run **simultaneously**:

### Architecture Diagram

```
User Query: "Should I buy TSLA?"
              │
              ▼
    ┌─────────────────┐
    │   SmartRouter    │ ← Regex pattern matching, symbol extraction
    │  (router.py)     │ ← Determines: QueryIntent.RESEARCH
    └────────┬────────┘
             │
    ┌────────┴────────────────────────────────────────────┐
    │                     FAST PATH?                       │
    │  If QUOTE/PREDICTION/TECHNICAL/PATTERN/SENTIMENT     │
    │  AND confidence >= 0.8 → Return immediately          │
    │  (No LLM needed — data-only response)                │
    └────────┬────────────────────────────────────────────┘
             │ NOT fast path → FULL APEX PATH
             │
    ┌────────┴──────────────────────────────────────────────────────┐
    │                                                                │
    │         ╔═══════════════╗         ╔═══════════════════╗        │
    │         ║   PATH 1      ║         ║     PATH 2        ║        │
    │         ║ (MANAGER)     ║         ║  (ORCHESTRATOR)   ║        │
    │         ║               ║         ║  Background Thread║        │
    │         ╠═══════════════╣         ╠═══════════════════╣        │
    │         ║               ║         ║                   ║        │
    │         ║ 1. Claude     ║   ║─────║ 1. Extract syms   ║        │
    │         ║    Plans      ║   ║     ║ 2. Lambda fetch   ║        │
    │         ║    research   ║   ║     ║ 3. DecisionEngine ║        │
    │         ║    steps      ║   ║     ║    (28+ data pts) ║        │
    │         ║               ║   ║     ║ 4. ML Predictor   ║        │
    │         ║ 2. Executes   ║   ║     ║    (46M+ params)  ║        │
    │         ║    each step  ║   ║     ║ 5. Claude plans   ║        │
    │         ║    (Lambda    ║   ║     ║    agent tasks     ║        │
    │         ║    data       ║   ║     ║ 6. Parallel exec  ║        │
    │         ║    injected)  ║   ║     ║    (9 agents)     ║        │
    │         ║               ║   ║     ║ 7. Synthesis      ║        │
    │         ║ 3. Summarizes ║   ║     ║ 8. Claude         ║        │
    │         ║    each step  ║   ║     ║    response gen   ║        │
    │         ║               ║   ║     ║                   ║        │
    │         ╚═══════╤═══════╝   ║     ╚════════╤══════════╝        │
    │                 │           ║              │                    │
    │                 │     CONCURRENT           │                    │
    │                 │     EXECUTION            │                    │
    │                 │           ║              │                    │
    │                 ▼           ║              ▼                    │
    │         ┌───────────────┐  ║     ┌────────────────┐            │
    │         │ Collect APEX  │←─╝     │ Orchestrator    │            │
    │         │ results       │←───────│ result dict     │            │
    │         │ (timeout 20s) │        │ (message, data, │            │
    │         │               │        │  agents_used,   │            │
    │         │               │        │  confidence)    │            │
    │         └───────┬───────┘        └────────────────┘            │
    │                 │                                               │
    │                 ▼                                               │
    │    ┌──────────────────────────────────────────┐                │
    │    │  INJECT into conversation:               │                │
    │    │  "CRITICAL ADDITIONAL INTELLIGENCE"      │                │
    │    │  + Orchestrator synthesis                 │                │
    │    │  + DecisionEngine results                │                │
    │    │  + ML predictions                        │                │
    │    │  + Individual agent outputs              │                │
    │    │  + Lambda real-time data                 │                │
    │    └────────────────┬─────────────────────────┘                │
    │                     │                                          │
    │                     ▼                                          │
    │    ┌─────────────────────────────────────────┐                 │
    │    │  FINAL ANSWER (Claude Sonnet 4)         │                 │
    │    │  ─────────────────────────────           │                 │
    │    │  Streamed token-by-token via             │                 │
    │    │  agent.answer() → llm.prompt_stream()   │                 │
    │    │                                         │                 │
    │    │  Has access to BOTH paths' intelligence │                 │
    │    │  Panel title: "Answer — APEX Synthesis" │                 │
    │    └─────────────────────────────────────────┘                 │
    └────────────────────────────────────────────────────────────────┘
```

### Why This Matters

The dual-brain approach means the user gets:
1. **Zero added latency** — the Orchestrator runs concurrently with Manager's own planning
2. **Deeper analysis** — 9 specialist agents + DecisionEngine + ML predictions
3. **Cross-validation** — when both paths agree, confidence is higher
4. **Conflict resolution** — when paths disagree, Claude explains the nuance

---

## 5. Module-by-Module Deep Dive

### 5.1 Manager (`manager.py`)

**Lines:** 1,315 | **Role:** Core orchestration — the central brain

The Manager is the most important module. It coordinates everything: routing, fast paths, APEX dual-brain fusion, and final answer generation.

#### Initialization (`__init__`)

```
Manager.__init__()
├── Agent() — LLM agent for planning/answering (Path 1)
├── TokenCounter() — tiktoken-based token counting
├── ML Predictor (optional) — 46M+ parameter ensemble
├── UltimateDecisionEngine (optional) — 28+ data point scoring
├── UnifiedServices + SmartRouter (optional) — routing & fast paths
└── OrchestratorAgent (optional) — APEX 9-agent system (Path 2)
    └── OrchestratorConfig(use_opus=True, max_parallel_agents=5, default_timeout=25)
```

Each component is wrapped in a `try/except` — if any component fails to import or initialize, the system degrades gracefully and sets `*_enabled = False` flags.

#### Key Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `ml_enabled` | True if ML modules load | ML predictions available |
| `decision_engine_enabled` | True if DecisionEngine loads | Trading decisions available |
| `fast_path_enabled` | True if SmartRouter loads | Can skip LLM for simple queries |
| `apex_enabled` | True if Orchestrator loads | 9-agent APEX system available |

#### `process_prompt(prompt, conversation)` — The Main Entry Point

This is the method called by both CLI and API. Here's the complete flow:

1. **Command Check** — `/help`, `/clear`, `/compact`, `/exit`, `/key` handled immediately
2. **API Key Check** — Requires either `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
3. **Fast Path** — SmartRouter routes the query:
   - If `fast_path=True` and `confidence >= 0.8` → call `_handle_fast_path()` → return immediately
   - Intents: QUOTE, PREDICTION, TECHNICAL, PATTERN, SENTIMENT, FILINGS_SEARCH
4. **APEX Launch** — If apex_enabled, calls `_launch_orchestrator_background(prompt)`:
   - Starts a background thread running `orchestrator.process()`
   - Returns a container with `thread`, `result[0]`, `error[0]`, `completed` Event
5. **Planning Phase** (Path 1) — Claude generates a JSON plan of research steps:
   - `agent.run(conversation)` → returns list of `{title, description}` items
   - Each item is executed via `agent.action()` which fetches Lambda data
   - Results are summarized with `agent.summarize()` and displayed in a Rich Panel
   - Progressive timeout messages ("Retrieving data...", "Cogitating...", etc.)
6. **APEX Collection** — `_collect_orchestrator_result(container, timeout=20)`:
   - Waits for the background thread (already had 15-30s during planning)
   - Formats the 9-agent analysis as a structured intelligence briefing
   - Injects into conversation as `"CRITICAL ADDITIONAL INTELLIGENCE"` message
7. **Final Answer** — `agent.answer(prompt, conversation)`:
   - Claude Sonnet 4 generates the response with access to both paths' data
   - Streamed token-by-token via Rich Live with Markdown rendering
   - Panel title shows "Answer — APEX Synthesis" when APEX data was injected
8. **Cleanup** — Removes `type: "data"` and `type: "apex_data"` messages to reduce tokens

#### Fast Path Handlers

| Intent | Handler | Data Source | LLM Required |
|--------|---------|------------|--------------|
| QUOTE | `_fast_quote()` | Polygon `/v2/aggs/ticker/{sym}/prev` | No |
| PREDICTION | `_fast_prediction()` → `_fallback_prediction()` | Registry or Polygon 90-day historical | No |
| TECHNICAL | `_fast_technical()` | Polygon 90-day historical | No |
| PATTERN | `_fast_patterns()` | Defers to full LLM | Deferred |
| SENTIMENT | `_fast_sentiment()` | Defers to full LLM | Deferred |
| FILINGS_SEARCH | `_fast_filings_search()` | SEC EDGAR links | No |

The `_fast_prediction()` is particularly sophisticated — it performs a **6-signal weighted technical analysis**:
- Signal 1: Price vs SMA20 (weight 2)
- Signal 2: SMA20 vs SMA50 trend (weight 2)
- Signal 3: RSI 14-period (weight 1)
- Signal 4: 5-day momentum (weight 1.5)
- Signal 5: 20-day momentum (weight 1.5)
- Signal 6: Price vs 10-day average (weight 1)

Produces a BULLISH/BEARISH/NEUTRAL direction with 30-85% confidence, RSI, SMAs, and volatility.

---

### 5.2 LLM Wrapper (`llm.py`)

**Role:** Unified LLM interface with Claude primary / OpenAI fallback

#### Architecture

```python
class LLM:
    def __init__(self):
        # Priority: Anthropic > OpenAI
        if ANTHROPIC_AVAILABLE and anthropic_key:
            self.client = Anthropic(api_key=...)
            self.default_model = "claude-sonnet-4-20250514"
        elif OPENAI_AVAILABLE and openai_key:
            self.client = OpenAI(api_key=...)
            self.default_model = "gpt-4.1"
```

#### Key Methods

| Method | Behavior |
|--------|----------|
| `prompt(messages)` | Single response (blocking). Used for planning and summarization. |
| `prompt_stream(messages)` | Yields tokens one by one. Used for final answer streaming. |

#### Streaming Implementation

```
Anthropic:  client.messages.stream() → stream.text_stream → yield text
OpenAI:     client.responses.create(stream=True) → event.delta → yield delta
```

After streaming completes, token usage is tracked via `stream.get_final_message().usage`.

#### Message Conversion

Anthropic requires alternating user/assistant messages with no `system` role in the messages array. The `_convert_messages_for_anthropic()` method:
1. Extracts `developer`/`system` messages → concatenated into `system` parameter
2. Merges consecutive same-role messages
3. Ensures first message is from `user`

#### Token Tracking

The `TokenTracker` singleton tracks all API calls:
- Input/output token counts
- Cache read/creation tokens
- Request count
- Cost estimate (Claude Sonnet 4: $3/M input, $15/M output)
- Session duration

---

### 5.3 Smart Router (`router.py`)

**Lines:** 383 | **Role:** Intent detection and query routing via regex patterns

#### Query Intents

```python
class QueryIntent(Enum):
    QUOTE = "quote"              # "AAPL price" → fast path
    TECHNICAL = "technical"       # "RSI for AAPL" → fast path
    PREDICTION = "prediction"     # "predict AAPL" → fast path
    PATTERN = "pattern"           # "patterns for AAPL" → fast path
    SENTIMENT = "sentiment"       # "sentiment for AAPL" → fast path
    FILINGS_SEARCH = "filings_search"   # "risk factors for AAPL" → fast path
    FILINGS_ANALYSIS = "filings_analysis" # "analyze AAPL 10-K" → LLM
    RESEARCH = "research"         # "Should I buy AAPL?" → full APEX
    COMPARISON = "comparison"     # "Compare AAPL vs MSFT" → LLM
    GENERAL = "general"           # Everything else → LLM
```

#### Critical Routing Order

Research patterns are checked **BEFORE** prediction patterns. This prevents _"Should I buy TSLA?"_ from being routed to the ML-only fast path instead of the full APEX analysis.

```
1. Research (checked first — "should I buy", "your opinion on")
2. Prediction ("predict AAPL", "ml forecast")
3. Technical ("RSI for TSLA", "technicals")
4. Pattern ("patterns for AAPL")
5. Sentiment ("sentiment for AAPL")
6. Filings ("10-K for AAPL", "risk factors")
7. Comparison ("AAPL vs MSFT")
8. Quote (just a ticker, "AAPL price")
9. Symbols present but no intent → Research
10. No symbols → General
```

#### Symbol Extraction

The router extracts symbols from three patterns:
1. `$AAPL` format → dollar-sign tickers
2. Known tickers (60+ common symbols in `COMMON_TICKERS` set)
3. Context-adjacent words (`AAPL stock`, `TSLA price`)

---

### 5.4 Agent (`agent/agent.py`)

**Lines:** 241 | **Role:** Manager's own Claude agent for planning and answering

This is the Agent used by **Path 1** (Manager). Not to be confused with the Orchestrator's specialized agents.

#### Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `run(messages)` | Generate research plan | JSON list of `{title, description}` steps |
| `action(question, title, description)` | Execute a research step | Text analysis from Claude |
| `answer(question, messages)` | Generate final answer | Yields chunks (streaming) |
| `summarize(messages)` | Condense step results | Brief summary (< 50 words) |
| `compact(messages)` | Compress conversation | 1-25 key data points |

#### Lambda Data Injection

Every `answer()` and `action()` call first fetches Lambda data via `get_realtime_analysis()`:
1. Extract symbols from the text
2. For each symbol (up to 3): `lambda_client.get_analysis(symbol)`
3. Format as `REAL-TIME MARKET INTELLIGENCE` and inject into the prompt

This ensures Claude always has **current market data** — not stale training data.

---

### 5.5 Agent Prompts (`agent/prompts.py`)

**Lines:** 231 | **Role:** System prompts defining agent behavior

| Prompt | Purpose | Key Instructions |
|--------|---------|-----------------|
| `agent_prompt` | Planning prompt | Returns JSON array of research steps. 1-2 steps for quotes, 3-4 for buy/sell, 4-5 for portfolio analysis |
| `answer_prompt` | Final answer generation | APEX Synthesis Protocol: convergence/divergence analysis, Decision Engine verdict, LuxAlgo signals, ML predictions, specific numbers |
| `action_prompt` | Research step execution | Execute specific research task with Lambda data |
| `summary_prompt` | Step summarization | < 50 words, preserve numbers/stats |
| `compact_prompt` | Conversation compression | 1-25 key data points, preserve numbers |

The `answer_prompt` is particularly detailed — it defines the **APEX Synthesis Protocol** which tells Claude how to handle data from both intelligence paths, when to declare "HIGH CONVICTION" (all LuxAlgo timeframes aligned), and how to structure the response.

---

### 5.6 CLI Interface (`cli.py`)

**Lines:** 437 | **Role:** Interactive terminal REPL

#### Banner

Displays a gradient ASCII art logo with version number and system status indicators.

#### System Status Checks

At startup, the CLI checks and displays:
- Lambda Decision Engine connection ✓/⚠
- APEX Dual-Brain Fusion availability ✓/⚠

#### Quick Commands

| Command | Action |
|---------|--------|
| `/status` | Show all system components (ML, Lambda, LuxAlgo, TENK) |
| `/version` | Display version number |
| `/clear` | Clear conversation history |
| `/help` | Show available commands and example queries |
| `/lambda SYMBOL` | Direct Lambda Decision Engine test |
| `/luxalgo SYMBOL` | Premium LuxAlgo multi-timeframe signals |
| `/tenk SYMBOL` | SEC Filing RAG search |
| `/quit` or `/exit` | Exit |

#### Main Loop

```
while True:
    input = user_input()
    if input.startswith('/') → handle_quick_command()
    if len(messages) > 40 → auto-compact conversation
    messages.append(user message)
    manager.process_prompt(input, messages)
```

Auto-compaction triggers at `MAX_CONVERSATION_MESSAGES = 40` to prevent context window overflow.

---

### 5.7 Helpers (`helpers.py`)

**Role:** Token counting, command handlers, config persistence

- **TokenCounter**: Uses tiktoken (`o200k_base` encoding) for token counting
- **Config**: Stored at `~/.nuble/config.json` — API keys, preferences
- **Timeout Messages**: 30+ fun status messages that rotate every 10s during long operations ("Cogitating...", "Synthesizing...", "Orchestrating...", etc.)

---

## 6. APEX Orchestrator System

### 6.1 Orchestrator Agent (`agents/orchestrator.py`)

**Lines:** 1,261 | **Role:** 9-agent coordinator powered by Claude Sonnet 4

This is the brain of **Path 2**. It coordinates 9 specialized agents and integrates with the DecisionEngine and ML Predictor.

#### Pipeline

```
process(user_message, conversation_id, user_context)
│
├── 1. Clear _last_agent_outputs (for progress tracking)
├── 2. _initialize_agents() — lazy-load all 9 agents
├── 3. _extract_symbols() — from user message
│
├── 4. STEP 0: Lambda fetch — get_lambda_analysis(symbols)
│      └── Up to 3 symbols via NubleLambdaClient
│
├── 5. STEP 0.5: UltimateDecisionEngine
│      └── make_decision(symbol) → UltimateDecision
│      └── Entry, stop-loss, take-profit, position size, score breakdown
│      └── Only for trading queries (buy/sell/invest/predict keywords)
│
├── 6. STEP 0.7: ML Predictor
│      └── predict(symbol) → direction, confidence, price target
│      └── Run in thread pool executor (CPU-bound)
│
├── 7. STEP 1: _plan_execution()
│      ├── Claude generates JSON plan with agent assignments
│      │   └── Each task: agent type, instruction, priority, dependencies
│      └── Fallback: _simple_plan() — keyword-based rule matching
│
├── 8. STEP 2: _execute_tasks()
│      ├── Topological sort by dependencies
│      ├── Each level runs in parallel via asyncio.gather()
│      ├── Per-agent timeout (default 30s)
│      ├── Results stored in _last_agent_outputs (for API progress tracking)
│      └── Failed agents return error results (non-fatal)
│
├── 9. STEP 3: _synthesize_results()
│      └── Combine: agent_outputs, confidence, errors, lambda_intelligence,
│          decision_engine, ml_predictions, primary_recommendation
│
└── 10. STEP 4: _generate_response()
       ├── Claude builds the final response from ALL data
       ├── Prompt includes: Decision Engine verdict, LuxAlgo signals,
       │   ML predictions, agent analyses, Lambda data
       └── Returns: {message, charts, actions}
```

#### Return Value

```python
{
    'message': str,                    # Claude-generated analysis (markdown)
    'data': {                          # Raw synthesis data
        'decision_engine': {...},      # DecisionEngine recommendation
        'ml_predictions': {...},       # ML model outputs
        'agent_outputs': {...},        # Per-agent data (9 agents)
        'lambda_intelligence': str,    # Lambda formatted context
        'overall_confidence': float,   # Average of successful agents
    },
    'charts': [...],                   # Chart configurations
    'actions': [...],                  # Suggested follow-up actions
    'confidence': float,               # Overall confidence
    'agents_used': ['market_analyst', 'quant_analyst', ...],
    'execution_time_seconds': float,
    'conversation_id': str,
    'symbols': ['TSLA'],
}
```

#### Agent Progress Tracking

The orchestrator exposes `_last_agent_outputs: Dict[str, Any]` which the API server polls every 0.5 seconds to emit real-time `agent_done` progress events.

---

### 6.2 Base Agent (`agents/base.py`)

All 9 specialized agents inherit from `SpecializedAgent`:

```python
class SpecializedAgent(ABC):
    def __init__(self, api_key):
        self.client = Anthropic(api_key=...)  # Lazy loaded
        self.model = "claude-sonnet-4-20250514"

    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]
    
    async def reason_with_claude(self, prompt, max_tokens=2000) -> str
```

Key data classes:
- **AgentType** — 9-variant enum (MARKET_ANALYST, QUANT_ANALYST, etc.)
- **TaskPriority** — CRITICAL, HIGH, MEDIUM, LOW
- **AgentTask** — task_id, agent_type, instruction, context, priority, timeout, dependencies
- **AgentResult** — task_id, agent_type, success, data, confidence, execution_time_ms, error

---

### 6.3 9 Specialist Agents

| Agent | File (lines) | Data Sources | Key Capabilities |
|-------|-------------|-------------|-----------------|
| **MarketAnalyst** | `market_analyst.py` (1052) | Polygon.io (price, OHLCV, 90-day history), StockNews PRO (`/stat`, `/ratings`, `/earnings-calendar`) | 50+ technical indicators, candlestick patterns, support/resistance, multi-timeframe, news sentiment, analyst ratings, earnings calendar |
| **NewsAnalyst** | `news_analyst.py` (713) | Polygon News, StockNews PRO (ALL endpoints), CryptoNews PRO (ALL endpoints), Alternative.me Fear & Greed, Polygon VIX | Multi-source aggregation, quantitative sentiment, trending headlines, breaking events, earnings calendar, sundown digest, whale activity |
| **RiskManager** | `risk_manager.py` | Polygon.io, VIX | Position risk, portfolio VaR, correlation analysis, stress tests, drawdown limits |
| **FundamentalAnalyst** | `fundamental_analyst.py` (799) | Polygon Financials API (`/vX/reference/financials`), Polygon Ticker Details, Polygon Dividends, Polygon SMA, StockNews, **TENK SEC Filing RAG** (DuckDB + embeddings) | Income statement, balance sheet, cash flow, profitability ratios (ROE, ROIC, margins), valuations (P/E, EV/EBITDA), SEC filing semantic search |
| **QuantAnalyst** | `quant_analyst.py` | Pre-trained ML models, historical data | ML signal generation, AFML/Triple Barrier, backtests, factor models, statistical analysis |
| **MacroAnalyst** | `macro_analyst.py` | Economic data | Fed policy, interest rates, CPI, GDP, geopolitics, cross-asset correlation |
| **PortfolioOptimizer** | `portfolio_optimizer.py` | Portfolio data | Asset allocation, rebalancing, optimization, tax efficiency |
| **CryptoSpecialist** | `crypto_specialist.py` | CryptoNews API, on-chain data | On-chain analysis, DeFi protocols, whale tracking, exchange reserves |
| **Educator** | `educator.py` | Knowledge base | Strategy explanations, terminology, examples, beginner guides |

---

## 7. Lambda Decision Engine Client

**File:** `lambda_client.py` | **Lines:** 1,114

The Lambda client provides access to the **NUBLE production decision engine** deployed on AWS Lambda.

### Architecture

```
CLI/API → NubleLambdaClient → API Gateway → Lambda Function → Data Aggregation
                                                            → Decision Engine
                                                            → Response
```

### Endpoint

```
Base URL: https://9vyvetp9c7.execute-api.us-east-1.amazonaws.com/production
```

### Data Sources Aggregated by Lambda

```
├── Polygon.io (Real-time)
│   ├── Price data, OHLCV
│   ├── Technical indicators (RSI, MACD, ATR, Bollinger, SMA stack)
│   ├── Market breadth, sector performance
│   └── Corporate news feed
│
├── StockNews API (24 Endpoints)
│   ├── Sentiment Analysis (NLP-based, 7-day rolling)
│   ├── Analyst Ratings (upgrades, downgrades, price targets)
│   ├── Earnings Calendar + Whisper Numbers
│   ├── SEC Filing Alerts
│   ├── Event Detection (M&A, spinoffs, buybacks)
│   ├── Trending Mentions
│   └── Block Trade Alerts
│
├── CryptoNews API (17 Endpoints)
│   ├── Crypto Sentiment (top 100 coins)
│   ├── Whale Activity Tracking (large wallet movements)
│   ├── Institutional Flow Signals (ETF flows)
│   ├── Regulatory News Detection (SEC, CFTC, global)
│   └── DeFi Protocol Events
│
└── Derived Intelligence
    ├── Multi-Timeframe Regime Detection (BEAR/BULL/VOLATILE/RANGING)
    ├── Cross-Asset Correlation
    ├── Volatility Regime Classification
    └── Composite Decision Scoring (0-100)
```

### Response Structure: `LambdaAnalysis`

```python
@dataclass
class LambdaAnalysis:
    symbol: str
    action: str           # STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL / NEUTRAL
    direction: str        # BULLISH / BEARISH / NEUTRAL
    strength: str         # STRONG / MODERATE / WEAK
    score: float          # 0-100 composite score
    confidence: float     # 0-1
    current_price: float
    regime: MarketRegime  # BULL / BEAR / VOLATILE / RANGING / CRISIS
    
    technicals: TechnicalSnapshot    # RSI, MACD, SMAs, ATR, momentum
    intelligence: IntelligenceSnapshot  # Sentiment, news, analyst, VIX
    
    # LuxAlgo multi-timeframe signals
    luxalgo_weekly_action: str    # BUY/SELL/N/A (only if signal is still valid)
    luxalgo_daily_action: str
    luxalgo_h4_action: str
    luxalgo_direction: str
    luxalgo_score: float
    luxalgo_aligned: bool         # All valid timeframes agree
    luxalgo_valid_count: int      # How many timeframes have fresh signals
```

### LuxAlgo Signal Validation

Lambda validates LuxAlgo signals for freshness. A signal is only shown if Lambda marks it as `valid: True`. Stale signals (e.g., 162+ hours old) are filtered out, preventing false confidence from expired data.

### Reliability Features

- **Retry**: 3 retries with 1.5x exponential backoff
- **Timeout**: 45 seconds per request
- **Connection pooling**: Via `requests.Session()`
- **Error classification**: Network vs API vs parse errors

---

## 8. Ultimate Decision Engine

**File:** `decision/ultimate_engine.py` | **Lines:** 1,663

The crown jewel of NUBLE — a weighted scoring system that integrates **28+ data points** with a **veto-powered risk layer**.

### Layer Architecture

```
TOTAL SCORE (0-100)
│
├── TECHNICAL SIGNALS (35% weight)
│   ├── LuxAlgo Multi-Timeframe (20%) — Premium MTF signals, highest priority
│   ├── ML Signal Generator - AFML (7%) — Triple Barrier method
│   ├── Deep Learning LSTM/Transformer (5%) — Neural network predictions
│   └── Classic TA (3%) — RSI, MACD, Bollinger Bands
│
├── INTELLIGENCE LAYER (30% weight)
│   ├── FinBERT Sentiment (8%) — ProsusAI/finbert model
│   ├── News Analysis (8%) — Aggregate news scoring
│   ├── HMM Regime Detection (7%) — 3-state Hidden Markov Model
│   └── Claude Reasoning (7%) — Optional LLM analysis
│
├── MARKET STRUCTURE (20% weight)
│   ├── Options Flow (6%) — Put/call ratios, GEX
│   ├── Order Flow / Dark Pool (5%)
│   ├── Macro Context (5%) — DXY, VIX, rates
│   └── On-Chain Crypto (4%) — Exchange reserves, whale activity
│
└── VALIDATION (15% weight)
    ├── Historical Win Rate (6%) — Past accuracy for this setup
    ├── Backtest Validation (5%) — Strategy backtesting
    └── Pattern Similarity (4%) — Similar historical patterns
```

### Risk Veto System

The risk layer has **absolute veto power** — it can override any buy/sell signal:

| Risk Check | Veto Condition |
|------------|---------------|
| Max Position | Position exceeds portfolio limit |
| Max Drawdown | Drawdown limit exceeded |
| Correlation | Portfolio correlation too high |
| Liquidity | Insufficient trading volume |
| News Blackout | High-impact news imminent |
| Earnings Window | Within earnings announcement window |
| Conflicting Signals | Critical signal conflicts detected |
| Regime Unfavorable | Bear/Crisis market regime |
| Volatility Spike | Abnormal volatility detected |
| Stale Data | Input data too old to be reliable |

### Output: `UltimateDecision`

```python
@dataclass
class UltimateDecision:
    symbol: str
    direction: TradeDirection        # BUY / SELL / NEUTRAL
    strength: TradeStrength          # STRONG / MODERATE / WEAK / NO_TRADE
    confidence: float                # 0-100
    should_trade: bool
    
    technical_score: LayerScore      # Each layer with score, confidence, components
    intelligence_score: LayerScore
    market_structure_score: LayerScore
    validation_score: LayerScore
    
    risk_checks: List[RiskCheck]     # All risk checks with pass/fail
    veto: bool                       # True if ANY risk check triggered veto
    veto_reason: Optional[str]
    
    trade_setup: TradeSetup          # entry, stop_loss, targets, position_pct, R:R
    reasoning: List[str]             # Human-readable reasoning chain
    raw_signals: Dict[str, Any]      # All raw data for debugging
```

### Strength Classification

| Confidence | Strength |
|-----------|----------|
| ≥ 75 | STRONG |
| ≥ 55 | MODERATE |
| ≥ 40 | WEAK |
| < 40 | NO_TRADE |

### Regime Alignment

The engine applies a ±15% bonus/penalty based on whether the trade direction aligns with the detected market regime.

---

## 9. Unified Services Layer

**File:** `services.py` | **Lines:** 950

Bridges the three subsystems (nuble core, institutional, TENK) into a single interface.

### Service Types

| Service | Status Check | Components |
|---------|-------------|------------|
| MARKET_DATA | Polygon key + Orchestrator | Historical, quotes, multi-provider |
| TECHNICAL | Always available | 50+ indicators (RSI, MACD, Bollinger, SMA, EMA, ATR) |
| FILINGS | FilingsAnalyzer + FilingsSearch | SEC 10-K/10-Q analysis and search |
| ML_PREDICTION | Model registry or RealTimePredictor | Pre-trained `.pt` models |
| SENTIMENT | Always available | Lexicon-based (FinBERT lazy-loaded) |
| PATTERNS | Always available | Classical chart patterns |

### Caching

- 60-second TTL cache on all service results
- Key format: `"{service}:{symbol}"` 

---

## 10. Signal Processing Pipeline

### Signal Fusion Engine (`signals/fusion_engine.py`)

**Lines:** 793

Combines multiple signal sources with **dynamic weight adjustment**:

```
Signal Sources → Normalized Signals → Weighted Fusion → Fused Signal
                                          │
                                    ┌─────┤
                                    │     │
                        ┌───────────┘     └────────────┐
                        │                              │
                  Base Weights:                  Regime-Adaptive:
                  - Technical (LuxAlgo): 50%     Bull → increase technical
                  - ML (AFML): 25%               Bear → increase sentiment
                  - Sentiment (FinBERT): 10%     Volatile → decrease all
                  - Regime (HMM): 10%
                  - Fundamental: 5%
```

### Output: `FusedSignal`

```python
@dataclass
class FusedSignal:
    direction: int              # -1 (sell), 0 (neutral), 1 (buy)
    strength: FusedSignalStrength  # STRONG_BUY → STRONG_SELL (7 levels)
    confidence: float           # 0-1
    source_agreement: float     # -1 to +1 (all agree → 1)
    recommended_size: float     # 0-1 (position sizing)
    stop_loss_pct: float
    take_profit_pct: float
```

### Signal Sources

| Source | File | Weight | Description |
|--------|------|--------|------------|
| LuxAlgo | `sources/technical_luxalgo.py` | 50% base | TradingView Premium MTF signals from DynamoDB |
| ML/AFML | `sources/ml_afml.py` | 25% base | Triple Barrier method signals |
| FinBERT | `sources/sentiment_finbert.py` | 10% base | Financial sentiment from news |
| HMM Regime | `sources/regime_hmm.py` | 10% base | 3-state Hidden Markov Model |

### LuxAlgo Pipeline

```
TradingView (LuxAlgo Premium Indicators)
    → Webhook alerts (BUY/SELL on W/D/4H timeframes)
    → API Gateway
    → Lambda
    → DynamoDB (nuble-production-signals table)
    → Lambda Decision Engine reads signals
    → CLI/API queries via Lambda client
```

Currently active for: **ETH** and **BTC** only (webhook alerts configured in TradingView).

---

## 11. News & Sentiment System

### News Pipeline (`news/`)

| Module | Purpose |
|--------|---------|
| `client.py` | General news aggregation client |
| `coindesk_client.py` | CoinDesk crypto news fetcher |
| `crypto_client.py` | CryptoNews API client |
| `integrator.py` | Multi-source news integration |
| `pipeline.py` | News processing pipeline |
| `sentiment.py` | FinBERT sentiment analyzer (440 lines) |

### FinBERT Sentiment Analyzer

```python
class SentimentAnalyzer:
    MODEL_NAME = "ProsusAI/finbert"  # Pre-trained on financial text
    
    # Input:  "Apple stock surges on strong earnings"
    # Output: SentimentResult(label=POSITIVE, score=0.92, normalized_score=+0.87)
    
    def analyze(self, text) → SentimentResult
    def analyze_batch(self, texts) → List[SentimentResult]
```

Features:
- GPU/MPS/CPU auto-detection
- Batch inference (batch_size=16)
- LRU cache (1000 entries)
- Normalized scores: -1 to +1 scale

---

## 12. ML & Learning System

### Pre-Trained Models

Located in `/models/`:

| Model | Symbol | Architecture |
|-------|--------|-------------|
| `mlp_AMD_20260130.pt` | AMD | MLP |
| `mlp_SLV_20260130.pt` | SLV | MLP |
| `mlp_SPY_20260130.pt` | SPY | MLP |
| `mlp_TSLA_20260130.pt` | TSLA | MLP |
| `mlp_XLK_20260130.pt` | XLK | MLP |

### Model Registry

The `institutional/ml/registry.py` module provides metadata for pre-trained models:
- Model type, training date
- Validation Sharpe ratio, walk-forward Sharpe
- Directional accuracy
- Grade classification

### Adaptive Learning (`learning/`)

| Module | Purpose |
|--------|---------|
| `accuracy_monitor.py` | Tracks prediction accuracy over time |
| `prediction_tracker.py` | Logs predictions and actual outcomes |
| `weight_adjuster.py` | Dynamically adjusts signal source weights based on recent accuracy |

---

## 13. API Server (v2 Elite)

**File:** `api/server.py` | **Lines:** 1,038

A production FastAPI server wrapping the entire NUBLE pipeline with structured metadata, granular progress events, and instant quotes.

### Endpoints

| Method | Path | Purpose | Response |
|--------|------|---------|----------|
| `POST` | `/api/chat` | Full analysis (SSE stream) | Server-Sent Events with progress |
| `POST` | `/api/chat/sync` | Full analysis (blocking) | JSON with metadata |
| `GET` | `/api/quote/{symbol}` | Structured quote | Clean JSON (no markup) |
| `GET` | `/api/lambda/{symbol}` | Lambda Decision Engine | Full analysis data |
| `GET` | `/api/luxalgo/{symbol}` | LuxAlgo premium signals | MTF signal data |
| `GET` | `/api/health` | Health check | `{status, version, uptime}` |
| `GET` | `/api/status` | System component status | All 6 components |
| `DELETE` | `/api/conversation/{id}` | Clear conversation | Confirmation |
| `WS` | `/ws/chat` | WebSocket real-time chat | Same events as SSE |

### SSE Event Flow

```
data: {"type": "start", "conversation_id": "uuid"}

data: {"type": "progress", "stage": "routing", "message": "Detected: research", "symbols": ["TSLA"]}

data: {"type": "quote", "symbol": "TSLA", "price": 411.11, "change_percent": 2.55, ...}

data: {"type": "progress", "stage": "apex_started", "message": "9 specialist agents analyzing..."}

data: {"type": "progress", "stage": "planning", "message": "Claude is planning research steps..."}

data: {"type": "progress", "stage": "agent_done", "agent": "market_analyst", "agents_done": 1}
data: {"type": "progress", "stage": "agent_done", "agent": "quant_analyst", "agents_done": 2}
data: {"type": "progress", "stage": "agent_done", "agent": "news_analyst", "agents_done": 3}
data: {"type": "progress", "stage": "agent_done", "agent": "fundamental_analyst", "agents_done": 4}
data: {"type": "progress", "stage": "agent_done", "agent": "risk_manager", "agents_done": 5}

data: {"type": "progress", "stage": "apex_complete", "agents_used": [...], "execution_time": 61.82}

data: {"type": "response", "text": "## TSLA Analysis...", "metadata": {
    "symbols": ["TSLA"],
    "path": "apex",
    "verdict": "BUY",
    "score": 72.5,
    "confidence": 0.725,
    "price": 411.11,
    "agents_used": ["market_analyst", "quant_analyst", ...],
    "execution_time": 65.3
}}

data: {"type": "done", "execution_time": 65.3, "metadata": {...}}
```

### Key Implementation Details

#### Thread Safety

Manager is NOT thread-safe. The `_ManagerWrapper` class serializes access:
- `_init_lock` (threading.Lock) — protects lazy initialization
- `_call_lock` (threading.Lock) — serializes `process_prompt()` calls
- Console output redirected to `/dev/null` during API calls

#### EventQueue

Thread-safe bridge between sync Manager thread and async SSE generator:

```python
class EventQueue:
    _queue: List[Dict]     # Events buffer
    _lock: threading.Lock  # Thread safety
    _done: threading.Event # Signals completion
    
    put(event)   → thread-safe enqueue
    drain()      → take all pending events (atomic)
    is_done      → True when _result or _error received
```

#### Rich Markup Stripping

The `_strip_rich()` function removes Rich console markup from text before sending to the API:
```python
# Enumerates specific Rich style words to avoid breaking markdown brackets
_RICH_WORDS = r'bold|dim|italic|white|red|green|yellow|blue|cyan|magenta|...'
_RICH_TAG_RE = re.compile(rf'\[/?(?:{_RICH_WORDS})(?:\s+(?:{_RICH_WORDS}))*\]')
```

This avoids breaking markdown links like `[link text](url)` while still stripping `[bold red]text[/bold red]`.

#### Agent Progress Monitoring

The `_monitor_orchestrator_agents()` method runs on a separate thread, polling `orchestrator._last_agent_outputs` every 0.5 seconds. When a new agent name appears in the dict, it emits an `agent_done` event.

#### Metadata Extraction

`_build_metadata()` extracts structured `ResponseMetadata` after processing:
- Symbols, path (fast_path/apex)
- Price and change from Polygon quote
- Decision Engine verdict, score, confidence (skipped for fast-path quotes)
- LuxAlgo signals (aligned, direction, score, valid_count)
- Agents used (from `_last_agent_outputs`)

#### Conversation Store

In-memory, thread-safe:
- Max 1000 conversations
- Max 40 messages per conversation
- Auto-cleanup when limit exceeded (evicts oldest half)

#### CORS

Currently `allow_origins=["*"]` — should be locked to frontend domain in production.

### Response Models (Pydantic)

| Model | Purpose |
|-------|---------|
| `ChatRequest` | Input: message + optional conversation_id |
| `ChatResponse` | Full response with metadata |
| `ResponseMetadata` | Structured: symbols, verdict, score, confidence, price, agents_used, etc. |
| `QuoteData` | Structured quote: price, change_percent, volume, high, low |
| `QuoteResponse` | Quote with optional structured data or fallback text |
| `LambdaResponse` | Full Lambda analysis with LuxAlgo signals |
| `LuxAlgoResponse` | LuxAlgo-only: aligned, direction, timeframe actions, score |
| `HealthResponse` | status, version, uptime |
| `StatusResponse` | Health + component status map |

---

## 14. Data Sources & API Keys

### External APIs

| API | Endpoints | Purpose | Key |
|-----|-----------|---------|-----|
| **Anthropic** | Claude Sonnet 4 | Primary LLM (planning, synthesis, answers) | `ANTHROPIC_API_KEY` |
| **OpenAI** | GPT-4.1 | Fallback LLM | `OPENAI_API_KEY` |
| **Polygon.io** | `/v2/aggs`, `/v3/reference`, `/vX/reference/financials`, `/v1/indicators` | Prices, OHLCV, technicals, company info, financials, dividends, news | `POLYGON_API_KEY` |
| **StockNews PRO** | `/api/v1`, `/stat`, `/top-mention`, `/events`, `/trending-headlines`, `/earnings-calendar`, `/ratings`, `/sundown-digest`, `/category` | Sentiment, analyst ratings, earnings, events, trending | `STOCKNEWS_API_KEY` |
| **CryptoNews PRO** | `/api/v1`, `/stat`, `/top-mention`, `/events`, `/trending-headlines`, `/category` | Crypto sentiment, whale activity, institutional flows, DeFi | `CRYPTONEWS_API_KEY` |
| **AWS Lambda** | Custom API Gateway | Decision Engine, LuxAlgo signals, aggregated intelligence | AWS credentials |
| **Alternative.me** | Fear & Greed Index | Market-wide sentiment | None (public) |
| **SEC EDGAR** | `/submissions/CIK*` | SEC filing metadata | None (public) |

### Internal Services

| Service | Technology | Purpose |
|---------|-----------|---------|
| **DynamoDB** | AWS | LuxAlgo signal storage (table: `nuble-production-signals`) |
| **DuckDB** | TENK_SOURCE | SEC filing vector store with sentence-transformer embeddings |
| **PyTorch** | Local | Pre-trained MLP models for price direction prediction |
| **FinBERT** | `ProsusAI/finbert` | Financial text sentiment analysis |
| **sentence-transformers** | `all-MiniLM-L6-v2` | 384-dim embeddings for SEC filing RAG |

---

## 15. Request Lifecycle — End-to-End Flow

### Example: "Should I buy TSLA?" (Full APEX Path)

```
USER INPUT: "Should I buy TSLA?"
│
│  ┌──────────────────────────────────────────────────┐
│  │ CLI: cli.py:interactive_shell()                  │
│  │ API: server.py:chat_stream() or chat_sync()      │
│  └──────────────────────────────────────────────────┘
│
├── 1. message added to conversation list
│
├── 2. Manager.process_prompt("Should I buy TSLA?", conversation)
│      │
│      ├── 3. SmartRouter.route("Should I buy TSLA?")
│      │      → Matches RESEARCH pattern ("should I ... buy")
│      │      → RoutedQuery(intent=RESEARCH, symbols=["TSLA"], fast_path=False)
│      │
│      ├── 4. NOT fast path → proceed to APEX
│      │
│      ├── 5. _launch_orchestrator_background("Should I buy TSLA?")
│      │      │
│      │      │  ┌── BACKGROUND THREAD ──────────────────────────────┐
│      │      │  │                                                    │
│      │      │  │  orchestrator.process("Should I buy TSLA?")       │
│      │      │  │  │                                                │
│      │      │  │  ├── Lambda: get_analysis("TSLA")                │
│      │      │  │  │   → action=BUY, score=72, price=$411          │
│      │      │  │  │   → RSI=45, MACD=+0.23, VIX=16              │
│      │      │  │  │   → LuxAlgo: W=BUY, D=BUY, 4H=N/A           │
│      │      │  │  │                                                │
│      │      │  │  ├── DecisionEngine.make_decision("TSLA")        │
│      │      │  │  │   → direction=BUY, confidence=72%             │
│      │      │  │  │   → entry=$411, SL=$395, TP=$435              │
│      │      │  │  │   → Technical: +0.32, Intelligence: +0.18    │
│      │      │  │  │   → Risk checks: 8/8 passed, no veto         │
│      │      │  │  │                                                │
│      │      │  │  ├── ML Predictor: predict("TSLA")               │
│      │      │  │  │   → direction=UP, confidence=68%              │
│      │      │  │  │                                                │
│      │      │  │  ├── Claude plans tasks:                          │
│      │      │  │  │   → MARKET_ANALYST: "Analyze TSLA technicals" │
│      │      │  │  │   → QUANT_ANALYST: "ML signals for TSLA"     │
│      │      │  │  │   → NEWS_ANALYST: "TSLA news sentiment"       │
│      │      │  │  │   → FUNDAMENTAL_ANALYST: "TSLA fundamentals"  │
│      │      │  │  │   → RISK_MANAGER: "TSLA risk assessment"      │
│      │      │  │  │                                                │
│      │      │  │  ├── Parallel execution (asyncio.gather):         │
│      │      │  │  │   ┌─ MarketAnalyst (8s) ──┐                   │
│      │      │  │  │   ├─ QuantAnalyst  (5s)   ├─ _last_agent_outputs │
│      │      │  │  │   ├─ NewsAnalyst   (10s)  │  updated per agent │
│      │      │  │  │   ├─ FundAnalyst   (12s)  │                   │
│      │      │  │  │   └─ RiskManager   (4s) ──┘                   │
│      │      │  │  │                                                │
│      │      │  │  ├── Synthesis: combine all outputs               │
│      │      │  │  │                                                │
│      │      │  │  └── Claude generates Orchestrator response       │
│      │      │  │      (with Decision Engine, ML, LuxAlgo data)    │
│      │      │  │                                                    │
│      │      │  └────────────────────────────────────────────────────┘
│      │      │
│      │      └── Returns: container = {thread, result[0], error[0], completed: Event}
│      │
│      ├── 6. PATH 1: Claude planning (CONCURRENT with step 5)
│      │      │
│      │      ├── agent.run(conversation) → JSON plan:
│      │      │   [
│      │      │     {"title": "Fetch TSLA price & technicals", "description": "..."},
│      │      │     {"title": "Check TSLA news sentiment", "description": "..."},
│      │      │     {"title": "Analyze TSLA fundamentals", "description": "..."}
│      │      │   ]
│      │      │
│      │      ├── For each step:
│      │      │   ├── Display in Rich Panel (Planning pane)
│      │      │   ├── agent.action() → Lambda data injection → Claude analysis
│      │      │   ├── agent.summarize() → condense results
│      │      │   └── Update planning display with summary
│      │      │
│      │      └── Total: ~20-30 seconds
│      │
│      ├── 7. _collect_orchestrator_result(container, timeout=20)
│      │      │
│      │      ├── container['completed'].wait(20s)
│      │      │   (Orchestrator has already had 20-30s from planning phase)
│      │      │
│      │      └── Format intelligence briefing:
│      │          ═══════════════════════════════════════
│      │          APEX TIER: MULTI-AGENT DEEP ANALYSIS
│      │          ═══════════════════════════════════════
│      │          Agents: market_analyst, quant_analyst, news_analyst, ...
│      │          Confidence: 72.0%
│      │          --- DECISION ENGINE ---
│      │          Action: BUY, Confidence: 72%, Risk: 0.28
│      │          Entry: $411, SL: $395, TP: $435
│      │          --- ML PREDICTOR ---
│      │          TSLA: UP (68% confidence)
│      │          --- AGENT INTELLIGENCE ---
│      │          [MARKET_ANALYST]: {technicals data...}
│      │          [NEWS_ANALYST]: {sentiment data...}
│      │          ═══════════════════════════════════════
│      │
│      ├── 8. Inject into conversation:
│      │      {"role": "user", "content": "CRITICAL ADDITIONAL INTELLIGENCE...",
│      │       "type": "apex_data"}
│      │
│      ├── 9. agent.answer("Should I buy TSLA?", conversation)
│      │      │
│      │      ├── Lambda data injection (answer-specific)
│      │      ├── llm.prompt_stream(messages) → Claude Sonnet 4
│      │      │   - Has ALL data: Path 1 research + Path 2 APEX intelligence
│      │      │   - Streams token-by-token
│      │      │
│      │      └── Rich Live Panel: "Answer — APEX Synthesis"
│      │          ┌─────────────────────────────────────────┐
│      │          │ ## TSLA: BUY Recommendation             │
│      │          │                                         │
│      │          │ ### Decision Engine Verdict              │
│      │          │ **BUY** with 72% confidence              │
│      │          │ Entry: $411 | SL: $395 | TP: $435       │
│      │          │                                         │
│      │          │ ### Technical Analysis                   │
│      │          │ RSI: 45 (neutral), MACD: bullish...     │
│      │          │ LuxAlgo: Weekly=BUY, Daily=BUY 🔥       │
│      │          │                                         │
│      │          │ ### ML Prediction                        │
│      │          │ Direction: UP (68% confidence)           │
│      │          │ ...                                     │
│      │          └─────────────────────────────────────────┘
│      │
│      ├── 10. Cleanup: remove "data" and "apex_data" messages
│      │
│      └── 11. Display footer:
│             Claude API: 45,230 tokens | ~$0.2834$ | APEX | Requests: 8
│
└── DONE (total: ~60-90 seconds for full APEX path)
```

### Example: "AAPL" (Fast Path)

```
USER INPUT: "AAPL"
│
├── SmartRouter.route("AAPL")
│   → RoutedQuery(intent=QUOTE, symbols=["AAPL"], fast_path=True, confidence=0.95)
│
├── Fast path check: fast_path=True, confidence=0.95 ≥ 0.8 ✓
│
├── _handle_fast_path(routed) → _fast_quote("AAPL")
│   │
│   ├── GET https://api.polygon.io/v2/aggs/ticker/AAPL/prev
│   │
│   └── Output:
│       AAPL
│       $198.45
│       ↑ 1.23%
│       Volume: 45,234,567
│       Range: $196.50 - $199.20
│
└── DONE (total: ~0.5 seconds)
```

---

## 16. Infrastructure

### AWS Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS (us-east-1)                           │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │ API Gateway  │────│ Lambda       │────│ DynamoDB           │  │
│  │ (REST)       │    │ Function     │    │ nuble-production-  │  │
│  │              │    │ (Decision    │    │ signals            │  │
│  │ /production  │    │  Engine)     │    │ (LuxAlgo signals)  │  │
│  └──────┬───────┘    │              │    └────────────────────┘  │
│         │            │  Data Sources│                            │
│         │            │  ├─ Polygon  │                            │
│         │            │  ├─ StockNews│                            │
│         │            │  ├─ CryptoNws│                            │
│         │            │  └─ DynamoDB │                            │
│         │            └──────────────┘                            │
│         │                                                        │
│         │  URL: https://9vyvetp9c7.execute-api.us-east-1.        │
│         │       amazonaws.com/production                         │
└─────────┴────────────────────────────────────────────────────────┘
            │
     ┌──────┴──────┐
     │ NUBLE CLI   │  (local machine / Docker)
     │ or API      │
     └─────────────┘
```

### LuxAlgo Signal Pipeline

```
TradingView (LuxAlgo Premium Indicators)
    │
    │ Webhook Alert (JSON: symbol, action, timeframe)
    │
    ▼
API Gateway (AWS)
    │
    ▼
Lambda Function (webhook receiver)
    │
    ▼
DynamoDB (nuble-production-signals)
    │ Partition: symbol#timeframe
    │ Sort: timestamp
    │
    ▼
Lambda Decision Engine (reads signals)
    │ Validates freshness (valid flag)
    │ Calculates MTF alignment
    │
    ▼
CLI / API (via NubleLambdaClient)
    │
    └── 34% weight in DecisionEngine scoring
```

**Active webhooks:** ETH (3 timeframes: 1W, 1D, 4H) and BTC (3 timeframes)

---

## 17. Token & Cost Management

### Token Tracking

The `TokenTracker` singleton tracks all Claude API calls:

```python
class TokenTracker:
    input_tokens: int       # Total input tokens consumed
    output_tokens: int      # Total output tokens consumed
    cache_read_tokens: int  # Cached input tokens (cheaper)
    cache_creation_tokens: int
    total_requests: int     # Number of API calls
    session_start: datetime
    
    cost_estimate = (input_tokens / 1M * $3) + (output_tokens / 1M * $15)
```

### Context Window Management

| Strategy | Trigger | Action |
|----------|---------|--------|
| Auto-compact | `len(messages) > 40` | Agent compresses to key data points |
| Manual compact | `/compact` command | User-triggered summarization |
| Data cleanup | After every `process_prompt()` | Remove `type: "data"` and `type: "apex_data"` messages |
| Conversation trim | API: after each request | Keep last 10 messages if over 40 |

### Typical Costs Per Query

| Query Type | Tokens (approx) | Cost (approx) |
|-----------|-----------------|---------------|
| Fast path quote | 0 (no LLM) | $0.00 |
| Simple question | 5-10K | $0.02-0.05 |
| APEX full analysis | 30-80K | $0.10-0.40 |

---

## 18. File Reference Matrix

| File | Lines | Primary Responsibility |
|------|-------|----------------------|
| `__init__.py` | 50 | Version, Console, dotenv |
| `cli.py` | 437 | CLI REPL, banner, commands |
| `manager.py` | 1,315 | **Core brain** — routing, APEX fusion, fast paths |
| `llm.py` | ~250 | Claude/OpenAI wrapper, streaming, token tracking |
| `router.py` | 383 | SmartRouter, intent detection, symbol extraction |
| `services.py` | 950 | UnifiedServices, bridges subsystems |
| `helpers.py` | ~300 | TokenCounter, commands, config |
| `lambda_client.py` | 1,114 | Lambda Decision Engine client, LambdaAnalysis |
| `agent/agent.py` | 241 | Path 1 agent — plan, action, answer, summarize |
| `agent/prompts.py` | 231 | System prompts (agent, answer, action, summary) |
| `agents/base.py` | ~200 | SpecializedAgent ABC, enums, dataclasses |
| `agents/orchestrator.py` | 1,261 | **APEX brain** — 9-agent coordination |
| `agents/market_analyst.py` | 1,052 | Polygon technicals, StockNews, earnings |
| `agents/news_analyst.py` | 713 | StockNews PRO + CryptoNews PRO |
| `agents/fundamental_analyst.py` | 799 | Polygon financials + TENK SEC RAG |
| `agents/risk_manager.py` | ~400 | Risk assessment, VaR, stress tests |
| `agents/quant_analyst.py` | ~400 | ML signals, backtests |
| `agents/macro_analyst.py` | ~300 | Economic indicators, Fed, rates |
| `agents/portfolio_optimizer.py` | ~300 | Asset allocation, rebalancing |
| `agents/crypto_specialist.py` | ~400 | On-chain, DeFi, whales |
| `agents/educator.py` | ~200 | Explanations, terminology |
| `api/server.py` | 1,038 | **FastAPI v2 Elite** — SSE, WebSocket, metadata |
| `decision/ultimate_engine.py` | 1,663 | **DecisionEngine** — 28+ data points, veto |
| `decision/data_classes.py` | 771 | Enums and dataclasses for decision layers |
| `signals/fusion_engine.py` | 793 | Multi-source signal fusion |
| `news/sentiment.py` | 440 | FinBERT sentiment analyzer |
| **Total** | **~15,000+** | **50+ Python files** |

---

## Appendix: Verified API Endpoints

All endpoints tested and working as of the latest session:

| Endpoint | Test | Result |
|----------|------|--------|
| `GET /api/health` | `curl localhost:8000/api/health` | `{"status":"healthy","version":"6.0.0"}` ✅ |
| `GET /api/status` | `curl localhost:8000/api/status` | All 6 components reported ✅ |
| `GET /api/quote/TSLA` | `curl localhost:8000/api/quote/TSLA` | `{"price":411.11,"change_percent":2.55}` ✅ |
| `GET /api/quote/BTC` | `curl localhost:8000/api/quote/BTC` | Crypto mapping works ✅ |
| `GET /api/lambda/TSLA` | `curl localhost:8000/api/lambda/TSLA` | Full DecisionEngine data ✅ |
| `POST /api/chat` (fast) | `"AAPL"` | `start→routing→quote→response→done` (0.48s) ✅ |
| `POST /api/chat` (APEX) | `"Should I buy TSLA?"` | Full granular: 5 agents, 61.82s ✅ |
| `POST /api/chat/sync` | Blocking | Metadata included ✅ |

---

> _"This system represents the convergence of real-time market data, multi-agent AI orchestration, machine learning predictions, and institutional-grade risk management into a single, cohesive intelligence platform."_
>
> **— NUBLE v6.0.0 APEX Dual-Brain Fusion**
