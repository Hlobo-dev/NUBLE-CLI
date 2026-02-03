# 🚀 NUBLE ELITE: Signal Fusion System

## The World's Most Intelligent Multi-Source Trading System

---

## Executive Summary

NUBLE Elite is a **multi-source signal fusion system** that combines:

| Source | Weight | Description |
|--------|--------|-------------|
| **LuxAlgo** (TradingView) | 50% | Proven technical signals via webhook |
| **ML Pipeline** (AFML) | 25% | Your trained machine learning models |
| **Sentiment** (FinBERT) | 10% | News and social sentiment analysis |
| **Regime** (HMM) | 10% | Market regime detection |
| **Fundamental** | 5% | Optional valuations |

**No single source dominates. Intelligence comes from fusion.**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NUBLE ELITE: SIGNAL FUSION ARCHITECTURE                │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         SIGNAL SOURCES                                │   │
│  │                                                                       │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │  │  TECHNICAL  │ │     ML      │ │  SENTIMENT  │ │   REGIME    │    │   │
│  │  │  (LuxAlgo)  │ │   (AFML)    │ │  (FinBERT)  │ │   (HMM)     │    │   │
│  │  │             │ │             │ │             │ │             │    │   │
│  │  │ • Webhooks  │ │ • Momentum  │ │ • News      │ │ • Bull/Bear │    │   │
│  │  │ • 1-12 Conf │ │ • Mean Rev  │ │ • Headlines │ │ • Sideways  │    │   │
│  │  │ • Trend     │ │ • Factors   │ │ • Social    │ │ • Volatile  │    │   │
│  │  │             │ │             │ │             │ │             │    │   │
│  │  │ Weight: 50% │ │ Weight: 25% │ │ Weight: 10% │ │ Weight: 10% │    │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                       │                                      │
│                                       ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      SIGNAL FUSION ENGINE                             │   │
│  │                                                                       │   │
│  │  1. NORMALIZE all signals to [-1, +1] scale                          │   │
│  │  2. WEIGHT by source confidence and regime                           │   │
│  │  3. DETECT agreement/disagreement between sources                    │   │
│  │  4. ADJUST weights dynamically based on accuracy                     │   │
│  │  5. COMBINE into single conviction score                             │   │
│  │                                                                       │   │
│  │  Key: Sources that AGREE boost confidence                            │   │
│  │       Sources that DISAGREE reduce position size                     │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                       │                                      │
│                                       ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      DECISION OUTPUT                                  │   │
│  │                                                                       │   │
│  │  • Direction: BUY / SELL / HOLD                                      │   │
│  │  • Strength: STRONG_BUY, BUY, WEAK_BUY, NEUTRAL, WEAK_SELL, etc.    │   │
│  │  • Confidence: 0% - 100%                                             │   │
│  │  • Position Size: Kelly-based with regime adjustment                 │   │
│  │  • Risk Levels: Stop-loss and take-profit percentages               │   │
│  │  • Reasoning: Full explanation of decision                           │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 New Files Created

```
src/nuble/
├── signals/
│   ├── __init__.py                    # Signal fusion exports
│   ├── base_source.py                 # Base class for signal sources
│   ├── luxalgo_webhook.py             # LuxAlgo webhook receiver
│   ├── fusion_engine.py               # Main fusion engine
│   └── sources/
│       ├── __init__.py
│       ├── technical_luxalgo.py       # LuxAlgo source adapter
│       ├── ml_afml.py                 # ML pipeline source adapter
│       ├── sentiment_finbert.py       # Sentiment source adapter
│       └── regime_hmm.py              # Regime source adapter
│
├── learning/
│   ├── __init__.py                    # Learning system exports
│   ├── prediction_tracker.py          # Track all predictions
│   ├── accuracy_monitor.py            # Monitor source accuracy
│   └── weight_adjuster.py             # Dynamic weight adjustment
│
└── api/
    └── luxalgo_api.py                 # Webhook API endpoints

tests/
└── test_luxalgo_integration.py        # Integration tests
```

---

## 🚀 Quick Start

### 1. Configure TradingView Alert

In TradingView, set up your LuxAlgo alert:

**Webhook URL:**
```
https://your-server.com/webhooks/luxalgo
```

**Alert Message (JSON):**
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

For SELL signals:
```json
{
    "action": "SELL",
    "signal_type": "Bearish Confirmation",
    ...
}
```

### 2. Start the API Server

```bash
cd NUBLE-CLI
python -m uvicorn nuble.api.main:app --host 0.0.0.0 --port 8000
```

### 3. Test the Webhook

```bash
curl -X POST "http://localhost:8000/webhooks/luxalgo" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "BUY",
    "symbol": "ETHUSD",
    "exchange": "COINBASE",
    "price": 2340.61,
    "timeframe": "4h",
    "confirmations": 12,
    "trend_strength": 54
  }'
```

### 4. Get Fused Signal

```bash
curl "http://localhost:8000/fusion/ETHUSD?regime=BULL&sentiment=0.3"
```

---

## 📊 API Endpoints

### LuxAlgo Webhook

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/webhooks/luxalgo` | POST | Receive LuxAlgo signal from TradingView |
| `/signals/{symbol}` | GET | Get recent signals for a symbol |
| `/signals/{symbol}/latest` | GET | Get the latest signal |
| `/signals/{symbol}/consensus` | GET | Get signal consensus |
| `/signals/{symbol}/strong` | GET | Get only strong signals |
| `/signals/status` | GET | Get status of all tracked symbols |

### Signal Fusion

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/fusion/{symbol}` | GET | Get fused signal from all sources |
| `/fusion/stats` | GET | Get signal source statistics |

---

## 🧪 Running Tests

```bash
cd NUBLE-CLI
python tests/test_luxalgo_integration.py
```

Expected output:
```
============================================================
NUBLE LUXALGO INTEGRATION TESTS
============================================================

✅ Webhook parsing tests PASSED
✅ Signal store tests PASSED
✅ Signal fusion tests PASSED
✅ Signal sources tests PASSED
✅ Prediction tracking tests PASSED
✅ Accuracy monitoring tests PASSED
✅ Weight adjustment tests PASSED

============================================================
TEST SUMMARY
============================================================
   Passed: 7/7
   Failed: 0/7

🎉 ALL TESTS PASSED!
```

---

## 🔧 Python Usage

### Basic Usage

```python
from nuble.signals import (
    SignalFusionEngine,
    get_signal_store,
    parse_luxalgo_webhook
)

# Create fusion engine
engine = SignalFusionEngine()

# Add a LuxAlgo signal (normally from webhook)
store = get_signal_store()
signal = parse_luxalgo_webhook({
    "action": "BUY",
    "symbol": "ETHUSD",
    "price": 2340.61,
    "timeframe": "4h",
    "confirmations": 12
})
store.add_signal(signal)

# Generate fused signal
fused = engine.generate_fused_signal(
    symbol="ETHUSD",
    prices=price_dataframe,    # Optional: for ML signals
    sentiment=0.3,             # Optional: pre-computed sentiment
    regime="BULL"              # Optional: pre-detected regime
)

# Check if actionable
if fused.is_actionable:
    print(f"Trade: {fused.action_str} {fused.symbol}")
    print(f"Size: {fused.recommended_size:.0%}")
    print(f"Stop: {fused.stop_loss_pct:.2%}")
    print(f"Take Profit: {fused.take_profit_pct:.2%}")
```

### With Continuous Learning

```python
from nuble.signals import SignalFusionEngine
from nuble.learning import PredictionTracker, WeightAdjuster

# Setup
engine = SignalFusionEngine()
tracker = PredictionTracker(storage_path="data/predictions.json")
adjuster = WeightAdjuster(base_weights=engine.weights)

# Generate signal
fused = engine.generate_fused_signal("ETHUSD")

# Log prediction
if fused.is_actionable:
    pred_id = tracker.log_prediction(fused, price=current_price)
    
    # Later, resolve with outcome
    tracker.resolve_prediction(pred_id, outcome_price=exit_price)

# Update weights based on accuracy
stats = tracker.get_accuracy_stats()
for source, data in stats['source_accuracy'].items():
    was_correct = data['accuracy'] > 0.5
    adjuster.record_outcome(source, was_correct)

# Get adjusted weights
new_weights = adjuster.get_weights()
```

---

## 🎯 Signal Fusion Rules

### Agreement Boost

When multiple sources agree, confidence increases:

| Scenario | Effect |
|----------|--------|
| LuxAlgo BUY + ML BUY | +15% confidence |
| All sources agree | +20% confidence, larger position |
| Mixed signals | Neutral |
| LuxAlgo BUY + ML SELL | -15% confidence, smaller position |

### Regime Adaptation

Weights adjust based on market regime:

| Regime | LuxAlgo | ML | Sentiment |
|--------|---------|-----|-----------|
| BULL | +20% | +10% | -10% |
| BEAR | +10% | -10% | +20% |
| SIDEWAYS | -10% | +20% | Normal |
| VOLATILE | -20% | -30% | -20% |

### Position Sizing

Position size is calculated based on:
- Signal confidence
- Source agreement
- Market regime
- Historical accuracy

```
Size = BaseSize × ConfidenceMult × AgreementMult × RegimeMult
```

---

## 📈 Continuous Learning

The system learns from outcomes and adjusts weights:

1. **Track** every prediction with timestamp and context
2. **Resolve** when outcome is known
3. **Calculate** accuracy by source, symbol, regime
4. **Adjust** weights based on performance
5. **Repeat** for continuous improvement

```
┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
│  Predict  │ ──► │   Track   │ ──► │  Resolve  │ ──► │  Adjust   │
└───────────┘     └───────────┘     └───────────┘     └───────────┘
                                                            │
                                                            ▼
                                                    ┌───────────┐
                                                    │  Better   │
                                                    │  Weights  │
                                                    └───────────┘
```

---

## 🛡️ Risk Management

Built-in risk controls:

- **Minimum confidence threshold**: 40%
- **Maximum position size**: Capped at 100%
- **Stop-loss levels**: ATR-based or fixed
- **Take-profit levels**: Risk-reward ratio based
- **Regime adjustment**: Smaller positions in volatile markets

---

## 📝 Configuration

### Fusion Engine

```python
engine = SignalFusionEngine(
    luxalgo_weight=0.50,          # Primary source
    ml_weight=0.25,               # Secondary source
    sentiment_weight=0.10,        # Context
    regime_weight=0.10,           # Context
    fundamental_weight=0.05,      # Optional
    min_confidence_to_trade=0.40, # Threshold
    min_agreement_to_trade=0.50   # Agreement threshold
)
```

### Weight Adjuster

```python
adjuster = WeightAdjuster(
    base_weights=weights,
    min_weight=0.05,              # Never go below 5%
    max_weight=0.60,              # Never exceed 60%
    adjustment_rate=0.1,          # How fast to adjust
    rolling_window=100            # Window for accuracy
)
```

---

## 🔮 Future Enhancements

- [ ] On-chain signals for crypto (whale movements, exchange flows)
- [ ] Fundamental signals (P/E, growth, quality factors)
- [ ] Real-time news streaming
- [ ] Multi-timeframe confirmation
- [ ] Portfolio-level optimization
- [ ] Automated trade execution

---

## 📄 License

Proprietary - NUBLE Institutional

---

*Built with 🧠 by NUBLE Elite*
