# NUBLE-CLI: Complete System Capabilities

## Executive Summary

**NUBLE** is an institutional-grade AI investment research platform that combines the power of **Claude Opus 4.5** (Anthropic's most advanced AI) with real-time market data, SEC filings analysis, and state-of-the-art machine learning models. It functions as an **AI-powered junior analyst** that can research companies, analyze SEC filings, generate trading signals, and provide comprehensive investment insights through natural language conversation.

The system has **three major subsystems**:
1. **NUBLE Core** - Claude-powered conversational AI for investment research
2. **Institutional Platform** - Professional-grade ML models, analytics, and data infrastructure  
3. **TENK Integration** - Specialized SEC filings Q&A with vector database

Unlike simple chatbots, NUBLE can execute multi-step research plans, fetch real-time data, run ML predictions, and synthesize comprehensive analysis reports - all through natural language queries like "What are Tesla's biggest risk factors and should I buy the stock?"

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NUBLE-CLI                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐     ┌─────────────────────┐                        │
│  │   src/nuble/     │     │  src/institutional/  │                        │
│  │   (Core CLI)        │     │  (Pro Platform)      │                        │
│  ├─────────────────────┤     ├─────────────────────┤                        │
│  │ • cli.py            │     │ • cli.py (Advanced) │                        │
│  │ • manager.py        │◄────┤ • core/orchestrator │                        │
│  │ • llm.py (Claude)   │     │ • ml/ (46M params)  │                        │
│  │ • agent/agent.py    │     │ • filings/          │                        │
│  │ • agent/prompts.py  │     │ • streaming/        │                        │
│  └─────────────────────┘     │ • backtesting/      │                        │
│           │                  │ • analytics/        │                        │
│           │                  │ • providers/        │                        │
│           ▼                  │ • validation/       │                        │
│  ┌─────────────────────┐     └─────────────────────┘                        │
│  │  TENK_SOURCE/       │              │                                      │
│  │  (SEC Filings)      │              │                                      │
│  │ • Vector DB         │◄─────────────┘                                      │
│  │ • RAG Search        │                                                     │
│  │ • Filing Loader     │                                                     │
│  └─────────────────────┘                                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │ Polygon.io│   │ SEC EDGAR │   │ Anthropic │
            │(Market)   │   │(Filings)  │   │(Claude)   │
            └───────────┘   └───────────┘   └───────────┘
```

### Design Patterns Used:
- **Orchestrator Pattern**: Central coordination of data providers and analytics
- **Agent Architecture**: Multi-step planning and execution with LLM reasoning
- **Provider Abstraction**: Unified interface for multiple data sources
- **Lazy Loading**: Models loaded on-demand to reduce memory footprint
- **Event-Driven**: Backtesting engine uses event queues for realism

---

## 2. AI Research Assistant Capabilities

### LLM Integration
| Model | Purpose | Usage |
|-------|---------|-------|
| **Claude Opus 4.5** | Primary AI (most advanced) | Deep analysis, research reports |
| **Claude Sonnet 4.5** | Fast queries | Quick quotes, simple analysis |
| **Claude Sonnet 4** | Default | Day-to-day queries |
| **GPT-4 (fallback)** | Alternative | If Claude unavailable |

### What Users Can ASK:

**Market Data & Quotes:**
```
> What's Tesla trading at?
> Show me Apple's performance this week
> Compare NVDA vs AMD stock price
```

**Technical Analysis:**
```
> What's the RSI for TSLA?
> Show technical indicators for AAPL
> Is MSFT overbought or oversold?
> Detect chart patterns for NVDA
```

**SEC Filings & Research:**
```
> What are Tesla's main risk factors?
> Summarize Apple's latest 10-K
> What did NVIDIA say about competition?
> Compare Microsoft and Google's revenue segments
```

**ML Predictions:**
```
> Give me a prediction for TSLA
> What's your ML forecast for AMD?
> Run the transformer model on AAPL
```

**Options & Institutional:**
```
> Show options activity for SPY
> What's unusual options flow for AMD?
> Show 13F holdings for Berkshire
```

### Multi-Step Research Planning

The agent breaks complex queries into executable plans:

```python
User: "Should I buy Tesla stock?"

Agent Plans:
1. Get current price and technicals
2. Analyze recent SEC filings for risks
3. Run ML prediction model
4. Synthesize buy/hold/sell recommendation
```

---

## 3. Quantitative Trading Capabilities

### Deep Learning Models (46.2M+ Parameters)

| Model | Architecture | Parameters | Purpose |
|-------|-------------|------------|---------|
| **Temporal Fusion Transformer** | Attention + LSTM | 8.5M | Interpretable multi-horizon forecasting |
| **N-BEATS** | Residual stacks | 6.2M | Trend/seasonality decomposition |
| **N-HiTS** | Hierarchical interpolation | 4.8M | Long-horizon efficiency |
| **Informer** | ProbSparse attention | 12.1M | Very long sequences (O(L log L)) |
| **DeepAR** | Autoregressive RNN | 5.6M | Probabilistic forecasting |
| **Financial LSTM** | LSTM + Attention | 3.2M | Classic sequence modeling |
| **Market Transformer** | Multi-head attention | 5.8M | Pattern recognition |

### Financial Loss Functions
```python
from institutional.ml.losses import (
    SharpeRatioLoss,      # Optimize Sharpe ratio directly
    SortinoRatioLoss,     # Focus on downside risk
    DirectionalLoss,       # 3x penalty for wrong direction
    ICLoss,               # Information coefficient (rank corr)
    MaxDrawdownLoss,      # Minimize drawdowns
    CombinedLoss,         # Multi-objective optimization
)
```

### Validation Framework
- **Out-of-sample testing** (train/val/test splits)
- **Walk-forward validation** (monthly retraining)
- **Statistical significance** (t-tests, p-values)
- **Grading system** (A+ to F based on Sharpe, direction accuracy)

### Backtesting Engine
- Event-driven architecture
- Transaction cost modeling (10bps default)
- Slippage simulation
- Walk-forward optimization
- Performance attribution

### Real-Time Streaming
- WebSocket connections to Polygon.io
- Live feature engineering
- Real-time signal generation
- Kelly criterion position sizing

---

## 4. Data Infrastructure

### Data Providers

| Provider | Data Types | Features |
|----------|-----------|----------|
| **Polygon.io** | Quotes, OHLCV, options, news | Real-time WebSocket, tick data |
| **SEC EDGAR** | 10-K, 10-Q, 8-K, 13F, Form 4 | Official filings, XBRL data |
| **Alpha Vantage** | Quotes, fundamentals, forex | Free tier available |
| **Finnhub** | News, sentiment, earnings | Social sentiment |

### Databases

| Database | Purpose | Technology |
|----------|---------|------------|
| **DuckDB** | SEC filings vector store | Embedded SQL + vectors |
| **Sentence Transformers** | Semantic embeddings | `all-MiniLM-L6-v2` (384 dims) |
| **File Cache** | Model checkpoints | PyTorch `.pt` files |

### Data Types Supported
```python
class DataType(Enum):
    QUOTE = "quote"           # Real-time prices
    OHLCV = "ohlcv"           # Historical bars
    TICK = "tick"             # Tick-level data
    OPTIONS = "options"       # Options chains + Greeks
    FUNDAMENTALS = "fundamentals"  # Financial statements
    NEWS = "news"             # News articles
    FILING = "filing"         # SEC filings
    HOLDINGS = "holdings"     # 13F institutional
    TRANSACTIONS = "transactions"  # Insider trades
```

---

## 5. User Interaction Methods

### CLI Commands

**Basic:**
```bash
nuble              # Launch interactive shell
nuble "query"      # One-shot query
```

**Institutional CLI Commands:**
```
help                  # Show all commands
<SYMBOL>              # Quick quote (e.g., "AAPL")
analyze <symbol>      # Comprehensive analysis
predict <symbol>      # ML price prediction
technical <symbol>    # 50+ technical indicators
patterns <symbol>     # Chart pattern detection
sentiment <symbol>    # News/social sentiment
filings <symbol>      # SEC filings summary
insider <symbol>      # Insider transactions
options <symbol>      # Options chain + Greeks
ml <symbol>           # Full ML analysis
train <symbol>        # Train custom model
stream <symbol>       # Real-time streaming
```

**Natural Language Queries:**
```
> What happened to Tesla today?
> Compare Apple and Microsoft earnings
> Show me the risk factors for NVIDIA
> What's the RSI for AMD?
> Give me a prediction for SPY
```

---

## 6. Complete Feature List

### AI/LLM Features
- ✅ Claude Opus 4.5 integration (most advanced AI)
- ✅ Claude Sonnet 4.5/4 for faster queries
- ✅ OpenAI GPT-4 fallback
- ✅ Multi-step research planning
- ✅ Token tracking and cost estimation
- ✅ Conversation context management
- ✅ Financial-specific system prompts

### Research Capabilities
- ✅ SEC filings analysis (10-K, 10-Q, 8-K, 13F)
- ✅ Risk factors extraction
- ✅ Management discussion analysis
- ✅ Competitive landscape review
- ✅ Segment analysis
- ✅ Guidance/earnings analysis
- ✅ Semantic search over filings

### Market Data
- ✅ Real-time quotes (Polygon.io)
- ✅ Historical OHLCV data
- ✅ Options chains with Greeks
- ✅ News articles
- ✅ WebSocket streaming

### Technical Analysis (50+ Indicators)
- ✅ Moving averages (SMA, EMA, WMA, DEMA, TEMA, KAMA)
- ✅ Momentum (RSI, Stochastic, MACD, CCI, MFI, Williams %R)
- ✅ Volatility (Bollinger Bands, ATR, Keltner, Donchian)
- ✅ Volume (OBV, VWAP, CMF, Force Index, A/D Line)
- ✅ Support/Resistance (Pivot Points, Fibonacci)

### Pattern Recognition
- ✅ Chart patterns (H&S, triangles, wedges, flags)
- ✅ Candlestick patterns (doji, hammer, engulfing, etc.)
- ✅ Price target calculation
- ✅ Confidence scoring

### Machine Learning
- ✅ 7 production DL models (TFT, N-BEATS, N-HiTS, Informer, DeepAR, LSTM, Transformer)
- ✅ 46.2M+ total parameters
- ✅ Financial loss functions (Sharpe, Sortino, Directional)
- ✅ Walk-forward validation
- ✅ Statistical significance testing
- ✅ Model checkpointing

### Backtesting
- ✅ Event-driven engine
- ✅ Transaction costs (10bps)
- ✅ Slippage modeling
- ✅ Portfolio tracking
- ✅ Performance metrics (Sharpe, Sortino, max DD, win rate)

### Real-Time
- ✅ WebSocket streaming
- ✅ Live feature engineering
- ✅ Real-time signal generation
- ✅ Position sizing (Kelly criterion)

### Export & Storage
- ✅ DuckDB vector database
- ✅ Model checkpoint saving
- ✅ Training results logging
- ✅ Validation reports

---

## 7. Technical Specifications

### Requirements
```python
# Core
python >= 3.8
anthropic  # Claude API
rich       # Terminal UI
duckdb     # Vector database
sentence-transformers  # Embeddings
torch      # Deep learning
numpy, pandas  # Data

# Optional
openai     # GPT fallback
aiohttp    # Async HTTP
websockets # Streaming
edgartools  # SEC EDGAR
```

### Model Configurations

**Temporal Fusion Transformer:**
```python
TFTConfig(
    hidden_size=256,
    num_heads=4,
    num_encoder_layers=1,
    num_decoder_layers=1,
    context_length=60,
    prediction_length=20,
    num_quantiles=7,
    dropout=0.1,
)
```

**N-BEATS:**
```python
NBeatsConfig(
    input_size=60,
    output_size=20,
    num_stacks=2,
    num_blocks=3,
    hidden_size=256,
)
```

### API Keys Required
| Provider | Environment Variable | Purpose |
|----------|---------------------|---------|
| Anthropic | `ANTHROPIC_API_KEY` | Claude AI (required) |
| Polygon.io | `POLYGON_API_KEY` | Market data (recommended) |
| OpenAI | `OPENAI_API_KEY` | GPT fallback (optional) |

---

## 8. What's Production-Ready vs In-Development

### ✅ Production-Ready
| Component | Status | Notes |
|-----------|--------|-------|
| Claude Integration | ✅ Ready | Full Opus 4.5 support |
| Natural Language Queries | ✅ Ready | Multi-step planning |
| Market Data (Polygon) | ✅ Ready | Real-time quotes |
| SEC Filings Search | ✅ Ready | Vector DB with semantic search |
| Technical Indicators | ✅ Ready | 50+ indicators |
| Pattern Recognition | ✅ Ready | Classical patterns |
| Basic ML Predictions | ✅ Ready | Validated on real data |
| Walk-Forward Validation | ✅ Ready | Monthly retraining |

### ⚠️ In Development / Needs Work
| Component | Status | Notes |
|-----------|--------|-------|
| Multi-Agent System | ⚠️ Partial | Infrastructure exists, agents not fully implemented |
| Options Analytics | ⚠️ Partial | Basic chains, Greeks need work |
| Portfolio Optimization | ⚠️ Minimal | Position sizing exists, no full optimizer |
| WebSocket Streaming | ⚠️ Partial | Code exists, not fully tested |
| CNN Pattern Recognition | ⚠️ Placeholder | Infrastructure ready, no trained model |
| FinBERT Sentiment | ⚠️ Placeholder | Falls back to lexicon-based |

### ❌ Not Yet Implemented
| Component | Status |
|-----------|--------|
| Web Interface | ❌ CLI only |
| API Server | ❌ No REST API |
| Order Execution | ❌ No broker integration |
| Live Trading | ❌ Research/backtest only |

---

## 9. Competitive Positioning

| Feature | NUBLE | Bloomberg Terminal | QuantConnect | ChatGPT + Plugins |
|---------|----------|-------------------|--------------|-------------------|
| **Price** | Free/API costs | $25,000/yr | Free-$250/mo | $20/mo |
| **AI Analysis** | ✅ Claude Opus 4.5 | ❌ Limited | ❌ None | ✅ GPT-4 |
| **SEC Filings RAG** | ✅ Vector search | ✅ Basic | ❌ No | ❌ No |
| **Deep Learning** | ✅ 7 models, 46M params | ❌ No | ✅ Yes | ❌ No |
| **Backtesting** | ✅ Event-driven | ✅ Yes | ✅ Excellent | ❌ No |
| **Real-Time Data** | ✅ Polygon | ✅ Best | ✅ Yes | ❌ No |
| **Natural Language** | ✅ Excellent | ❌ Limited | ❌ No | ✅ Excellent |
| **Open Source** | ✅ Yes | ❌ No | ⚠️ Partial | ❌ No |
| **Local Deployment** | ✅ Yes | ❌ No | ❌ No | ❌ No |

**NUBLE's Unique Value:**
1. **AI-First Research** - Claude Opus 4.5 understands complex financial questions
2. **SEC Filings as a First-Class Citizen** - Deep RAG over 10-K/10-Q filings
3. **Production ML Models** - Validated with walk-forward testing
4. **Open Source** - Fully customizable, no vendor lock-in
5. **Local First** - Your data stays on your machine

---

## 10. Recommended Use Cases

### ✅ Best For:

1. **Fundamental Research**
   - "What are Tesla's biggest risk factors?"
   - "Compare Apple and Microsoft's revenue segments"
   - "Summarize NVIDIA's latest earnings call"

2. **Technical Analysis**
   - "Is AMD overbought based on RSI?"
   - "Show me support and resistance for SPY"
   - "What patterns are forming on AAPL chart?"

3. **Due Diligence**
   - "What did the 10-K say about competition?"
   - "Any concerning language in the risk factors?"
   - "How has management guidance changed?"

4. **Quantitative Research**
   - "Backtest a momentum strategy on SPY"
   - "Train and validate a prediction model"
   - "What's the Sharpe ratio of this strategy?"

5. **Market Monitoring**
   - "What's moving today?"
   - "Any unusual options activity in AMD?"
   - "What's the current market sentiment?"

### ⚠️ Not Recommended For:

1. **Live Trading** - No broker integration, research only
2. **High-Frequency Trading** - Not designed for sub-second latency
3. **Cryptocurrency DeFi** - Limited crypto support
4. **Forex** - Basic support only

---

## Quick Start

```bash
# Install
git clone https://github.com/Hlobo-dev/NUBLE-CLI.git
cd NUBLE-CLI
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Configure
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
echo 'POLYGON_API_KEY=...' >> .env

# Run
nuble
```

```
> What are Tesla's main risk factors from their latest 10-K?

[Planning...]
● Let me check Tesla's SEC filings
● Searching for risk factors section
● Analyzing key risks with AI

[Answer]
Based on Tesla's 2024 10-K filing, the main risk factors are:

1. **Production/Manufacturing Risks**
   - Gigafactory ramp challenges
   - Supply chain dependencies
   - Battery cell constraints

2. **Competition**
   - Traditional automakers investing heavily in EVs
   - New entrants from China (BYD, NIO)
   
3. **Regulatory**
   - EV credit phase-outs
   - Autonomous driving regulations
   
[Full analysis continues...]
```

---

## Summary

NUBLE is a **comprehensive AI investment research platform** that brings together:

- 🧠 **Claude Opus 4.5** for natural language financial analysis
- 📊 **Real-time market data** via Polygon.io
- 📑 **SEC filings** with semantic search
- 🤖 **Production ML models** (46M+ parameters)
- 📈 **Backtesting** with proper validation
- 🔧 **50+ technical indicators**

It's designed for **research and analysis**, not live trading. Think of it as having an AI-powered junior analyst that can fetch data, analyze filings, run models, and synthesize insights - all through natural conversation.

**Total Lines of Code:** ~25,000+
**Total ML Parameters:** 46.2M+
**Supported Data Types:** 10+
**Technical Indicators:** 50+
**SEC Filing Types:** 6+

---

*Generated: January 30, 2026*
*Version: 2.0.0*
