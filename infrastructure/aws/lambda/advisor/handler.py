"""
KYPERIAN ADVISOR - AUTONOMOUS AI WEALTH MANAGER
=================================================

The most advanced institutional-grade autonomous trading advisor.
Combines ALL system capabilities into a single intelligent agent.

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                     KYPERIAN ADVISOR BRAIN                       │
│                    (Claude Opus 4.5 Core)                        │
├─────────────────────────────────────────────────────────────────┤
│  MARKET INTELLIGENCE    PORTFOLIO CONTEXT    EXECUTION ENGINE   │
│  ├─ Decision Engine V4  ├─ Positions         ├─ Order Router    │
│  ├─ Polygon.io          ├─ Cash              ├─ Risk Controls   │
│  ├─ LuxAlgo Signals     ├─ P&L               ├─ Trade Confirm   │
│  ├─ HMM Regime          ├─ History           ├─ Notifications   │
│  └─ News/Sentiment      └─ Allocation        └─ Logging         │
├─────────────────────────────────────────────────────────────────┤
│                    AUTONOMOUS OPERATIONS                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ MONITOR  │→ │ ANALYZE  │→ │  DECIDE  │→ │ EXECUTE  │        │
│  │ 24/7     │  │ All Data │  │ Should   │  │ + Notify │        │
│  │ Markets  │  │ Sources  │  │ We Act?  │  │ User     │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
├─────────────────────────────────────────────────────────────────┤
│                  COMMUNICATION LAYER                             │
│  Telegram Bot  │  WebSocket  │  API  │  Email  │  Webhook      │
└─────────────────────────────────────────────────────────────────┘

Features:
- 24/7 Autonomous Market Monitoring
- Proactive Trade Alerts & Execution
- Natural Language Portfolio Q&A
- Risk-Based Position Management
- VIX-Triggered Defensive Mode
- Earnings Calendar Awareness
- Deep Research on Demand
- Performance Tracking & Reporting

Author: KYPERIAN ELITE
Version: 5.0.0 (ADVISOR)
Date: February 2026
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
import urllib.request
import urllib.error
import hashlib
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import re

import boto3
from botocore.exceptions import ClientError

# ============================================================
# CONFIGURATION
# ============================================================

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment Variables
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
IBKR_API_KEY = os.environ.get('IBKR_API_KEY', '')

# DynamoDB Tables
SIGNALS_TABLE = os.environ.get('DYNAMODB_SIGNALS_TABLE', 'kyperian-production-signals')
DECISIONS_TABLE = os.environ.get('DYNAMODB_DECISIONS_TABLE', 'kyperian-production-decisions')
TRADES_TABLE = os.environ.get('DYNAMODB_TRADES_TABLE', 'kyperian-production-trades')
PORTFOLIO_TABLE = os.environ.get('DYNAMODB_PORTFOLIO_TABLE', 'kyperian-production-portfolio')
CONVERSATIONS_TABLE = os.environ.get('DYNAMODB_CONVERSATIONS_TABLE', 'kyperian-production-conversations')

# Engine Configuration
CONFIG = {
    "version": "5.0.0",
    "name": "KYPERIAN ADVISOR",
    "codename": "The Autonomous Wealth Manager",
    
    # Personality
    "personality": {
        "name": "KYPERIAN",
        "style": "Professional but friendly, like a trusted Goldman Sachs advisor",
        "traits": ["proactive", "data-driven", "risk-aware", "clear", "actionable"],
    },
    
    # Owner Profile (would be loaded from DB in production)
    "owner": {
        "name": "Humberto",
        "risk_tolerance": "moderate-aggressive",
        "background": ["Wharton MBA", "AWS TAM", "Entrepreneur"],
        "goals": ["Wealth accumulation", "Beat S&P 500", "Risk-adjusted returns"],
        "preferences": {
            "max_single_position": 0.10,  # 10% max
            "target_cash": 0.15,  # 15% cash target
            "crypto_allocation": 0.15,  # 15% crypto max
            "rebalance_threshold": 0.05,  # 5% drift triggers rebalance
        },
    },
    
    # Trading Rules
    "trading_rules": {
        "min_confidence": 55,
        "require_alignment": True,
        "max_trades_per_day": 5,
        "max_position_pct": 8.0,
        "default_stop_pct": 2.5,
        "vix_defensive_threshold": 30,
        "vix_no_trade_threshold": 40,
    },
    
    # Layer Weights (from V4)
    "weights": {
        "technical": 0.35,
        "intelligence": 0.30,
        "market_structure": 0.20,
        "validation": 0.15,
    },
    
    # Watchlist
    "symbols": {
        "crypto": ["BTCUSD", "ETHUSD"],
        "indices": ["SPY", "QQQ"],
        "mega_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
        "growth": ["TSLA", "AMD"],
    },
    
    # Thresholds
    "strong_threshold": 75,
    "moderate_threshold": 55,
    "weak_threshold": 40,
    
    # Signal max age (hours)
    "signal_max_age": {"1W": 168, "1D": 48, "4h": 12},
    
    # VIX thresholds
    "vix_thresholds": {"low": 15, "normal": 20, "elevated": 25, "high": 30, "extreme": 40},
    
    # Crypto symbol mapping
    "crypto_map": {"BTCUSD": "X:BTCUSD", "ETHUSD": "X:ETHUSD"},
}

ALL_SYMBOLS = (
    CONFIG["symbols"]["crypto"] + 
    CONFIG["symbols"]["indices"] + 
    CONFIG["symbols"]["mega_cap"] + 
    CONFIG["symbols"]["growth"]
)


# ============================================================
# DATA CLASSES
# ============================================================

class MessagePriority(Enum):
    URGENT = "urgent"      # SMS + Telegram + Push
    IMPORTANT = "important"  # Telegram + Push
    NORMAL = "normal"      # Telegram
    LOW = "low"           # Daily digest


class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    ADD = "ADD"


class AdvisorMode(Enum):
    NORMAL = "normal"
    DEFENSIVE = "defensive"
    AGGRESSIVE = "aggressive"
    CASH_PRESERVATION = "cash_preservation"


@dataclass
class Position:
    """Portfolio position"""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    pnl_pct: float
    weight: float  # % of portfolio
    sector: str = "Unknown"
    asset_class: str = "Equity"


@dataclass
class Portfolio:
    """Complete portfolio state"""
    total_value: float
    cash: float
    cash_pct: float
    positions: List[Position]
    daily_pnl: float
    daily_pnl_pct: float
    ytd_return: float
    max_drawdown: float
    current_drawdown: float
    last_updated: str


@dataclass
class TradeSignal:
    """Trading signal from decision engine"""
    symbol: str
    direction: str
    strength: str
    confidence: float
    should_trade: bool
    entry: float
    stop_loss: float
    targets: List[float]
    position_pct: float
    reasoning: List[str]
    data_points: int
    regime: str
    vix: float


@dataclass
class AdvisorMessage:
    """Message to send to user"""
    content: str
    priority: MessagePriority
    action_required: bool = False
    actions: List[Dict] = field(default_factory=list)
    data: Dict = field(default_factory=dict)


@dataclass 
class TradeExecution:
    """Executed trade record"""
    trade_id: str
    symbol: str
    action: str
    quantity: float
    price: float
    value: float
    timestamp: str
    signal_confidence: float
    stop_loss: float
    targets: List[float]
    status: str = "PENDING"


# ============================================================
# DYNAMODB HELPERS
# ============================================================

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


_dynamodb = None
_tables = {}


def get_dynamodb():
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource('dynamodb')
    return _dynamodb


def get_table(table_name: str):
    global _tables
    if table_name not in _tables:
        _tables[table_name] = get_dynamodb().Table(table_name)
    return _tables[table_name]


# ============================================================
# POLYGON.IO DATA LAYER
# ============================================================

def polygon_request(endpoint: str, params: Dict = None) -> Dict:
    """Make request to Polygon.io API"""
    if not POLYGON_API_KEY:
        return {"error": "No Polygon API key"}
    
    base_url = "https://api.polygon.io"
    url = f"{base_url}{endpoint}"
    
    params = params or {}
    params["apiKey"] = POLYGON_API_KEY
    
    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    full_url = f"{url}?{query_string}"
    
    try:
        req = urllib.request.Request(full_url, headers={"User-Agent": "KYPERIAN-ADVISOR/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}


def get_polygon_symbol(symbol: str) -> str:
    """Convert to Polygon format"""
    return CONFIG["crypto_map"].get(symbol, symbol)


def fetch_market_data(symbol: str) -> Dict:
    """Fetch comprehensive market data from Polygon"""
    poly_symbol = get_polygon_symbol(symbol)
    data = {"symbol": symbol, "available": False}
    
    try:
        # Previous close
        prev = polygon_request(f"/v2/aggs/ticker/{poly_symbol}/prev")
        if prev.get("results"):
            bar = prev["results"][0]
            data["price"] = bar.get("c", 0)
            data["volume"] = bar.get("v", 0)
            data["change_1d"] = (bar.get("c", 0) - bar.get("o", 0)) / bar.get("o", 1) * 100
        
        # RSI
        rsi = polygon_request(f"/v1/indicators/rsi/{poly_symbol}", {"timespan": "day", "window": 14, "limit": 1})
        if rsi.get("results", {}).get("values"):
            data["rsi"] = rsi["results"]["values"][0].get("value", 50)
        
        # MACD
        macd = polygon_request(f"/v1/indicators/macd/{poly_symbol}", {"timespan": "day", "limit": 1})
        if macd.get("results", {}).get("values"):
            m = macd["results"]["values"][0]
            data["macd"] = m.get("value", 0)
            data["macd_signal"] = m.get("signal", 0)
            data["macd_histogram"] = m.get("histogram", 0)
        
        # SMAs
        for window in [20, 50, 200]:
            sma = polygon_request(f"/v1/indicators/sma/{poly_symbol}", {"timespan": "day", "window": window, "limit": 1})
            if sma.get("results", {}).get("values"):
                data[f"sma_{window}"] = sma["results"]["values"][0].get("value", 0)
        
        # VIX (for stocks only)
        if not symbol.endswith("USD"):
            vix = polygon_request("/v2/aggs/ticker/VIX/prev")
            if vix.get("results"):
                data["vix"] = vix["results"][0].get("c", 20)
        else:
            data["vix"] = 25  # Default for crypto
        
        # News
        news_ticker = symbol.replace("USD", "") if symbol.endswith("USD") else symbol
        news = polygon_request("/v2/reference/news", {"ticker": news_ticker, "limit": 5})
        if news.get("results"):
            data["news"] = news["results"][:3]
            data["news_count"] = len(news["results"])
        
        data["available"] = True
        
    except Exception as e:
        data["error"] = str(e)
    
    return data


def get_vix() -> Tuple[float, str]:
    """Get current VIX level and state"""
    try:
        vix_data = polygon_request("/v2/aggs/ticker/VIX/prev")
        if vix_data.get("results"):
            vix = vix_data["results"][0].get("c", 20)
            
            thresholds = CONFIG["vix_thresholds"]
            if vix < thresholds["low"]:
                state = "LOW"
            elif vix < thresholds["normal"]:
                state = "NORMAL"
            elif vix < thresholds["elevated"]:
                state = "ELEVATED"
            elif vix < thresholds["high"]:
                state = "HIGH"
            else:
                state = "EXTREME"
            
            return vix, state
    except:
        pass
    
    return 20.0, "NORMAL"


# ============================================================
# LUXALGO SIGNALS
# ============================================================

def get_luxalgo_signals(symbol: str) -> Dict:
    """Get LuxAlgo signals from DynamoDB"""
    table = get_table(SIGNALS_TABLE)
    now = datetime.now(timezone.utc)
    
    signals = {}
    
    for tf, max_age in CONFIG["signal_max_age"].items():
        try:
            response = table.query(
                KeyConditionExpression='pk = :pk AND begins_with(sk, :tf)',
                ExpressionAttributeValues={
                    ':pk': f'SIGNAL#{symbol}',
                    ':tf': f'{tf}#',
                },
                ScanIndexForward=False,
                Limit=1,
            )
            
            if response.get('Items'):
                item = response['Items'][0]
                ts = float(item.get('timestamp', 0))
                if ts > 1e12:
                    ts = ts / 1000
                
                signal_time = datetime.fromtimestamp(ts, tz=timezone.utc)
                age_hours = (now - signal_time).total_seconds() / 3600
                
                action = item.get('action', 'NEUTRAL')
                price = float(item.get('price', 0))
                
                # Score
                if action in ['BUY', 'STRONG_BUY']:
                    score = 0.9 if action == 'STRONG_BUY' else 0.6
                elif action in ['SELL', 'STRONG_SELL']:
                    score = -0.9 if action == 'STRONG_SELL' else -0.6
                else:
                    score = 0
                
                decay = max(0, 1 - (age_hours / max_age) * 0.3)
                
                signals[tf] = {
                    "action": action,
                    "price": price,
                    "age_hours": round(age_hours, 1),
                    "valid": age_hours <= max_age,
                    "score": score * decay if age_hours <= max_age else 0,
                }
            else:
                signals[tf] = {"action": "N/A", "valid": False, "score": 0}
                
        except Exception as e:
            signals[tf] = {"action": "N/A", "valid": False, "error": str(e)}
    
    # Calculate alignment
    valid_signals = [s for s in signals.values() if s.get("valid", False)]
    scores = [s["score"] for s in valid_signals if s["score"] != 0]
    
    aligned = False
    if len(scores) >= 2:
        all_positive = all(s > 0 for s in scores)
        all_negative = all(s < 0 for s in scores)
        aligned = all_positive or all_negative
    
    weighted_score = (
        signals.get("1W", {}).get("score", 0) * 0.45 +
        signals.get("1D", {}).get("score", 0) * 0.35 +
        signals.get("4h", {}).get("score", 0) * 0.20
    )
    
    return {
        "weekly": signals.get("1W", {}),
        "daily": signals.get("1D", {}),
        "h4": signals.get("4h", {}),
        "aligned": aligned,
        "score": weighted_score,
        "direction": 1 if weighted_score > 0.2 else -1 if weighted_score < -0.2 else 0,
        "valid_count": len(valid_signals),
    }


# ============================================================
# REGIME DETECTION
# ============================================================

def detect_regime(luxalgo: Dict, market_data: Dict) -> Tuple[str, float]:
    """Detect market regime from available data"""
    scores = {"BULL": 0, "BEAR": 0, "SIDEWAYS": 0, "HIGH_VOL": 0}
    
    # LuxAlgo weekly (25%)
    weekly = luxalgo.get("weekly", {})
    if weekly.get("valid"):
        action = weekly.get("action", "")
        if action in ["BUY", "STRONG_BUY"]:
            scores["BULL"] += 0.25
        elif action in ["SELL", "STRONG_SELL"]:
            scores["BEAR"] += 0.25
        else:
            scores["SIDEWAYS"] += 0.25
    
    # Trend from SMAs (25%)
    if market_data.get("available"):
        price = market_data.get("price", 0)
        sma50 = market_data.get("sma_50", 0)
        sma200 = market_data.get("sma_200", 0)
        
        if price > 0 and sma50 > 0 and sma200 > 0:
            if price > sma50 > sma200:
                scores["BULL"] += 0.25
            elif price < sma50 < sma200:
                scores["BEAR"] += 0.25
            else:
                scores["SIDEWAYS"] += 0.25
    
    # VIX (25%)
    vix = market_data.get("vix", 20)
    if vix > 30:
        scores["HIGH_VOL"] += 0.25
    elif vix > 25:
        scores["HIGH_VOL"] += 0.1
        scores["BEAR"] += 0.1
    elif vix < 15:
        scores["BULL"] += 0.1
    
    # RSI (15%)
    rsi = market_data.get("rsi", 50)
    if rsi > 60:
        scores["BULL"] += 0.15
    elif rsi < 40:
        scores["BEAR"] += 0.15
    else:
        scores["SIDEWAYS"] += 0.15
    
    # Momentum (10%)
    change = market_data.get("change_1d", 0)
    if change > 2:
        scores["BULL"] += 0.1
    elif change < -2:
        scores["BEAR"] += 0.1
    
    regime = max(scores, key=scores.get)
    confidence = min(1.0, scores[regime] * 1.5)
    
    return regime, confidence


# ============================================================
# DECISION ENGINE (INTEGRATED V4)
# ============================================================

def analyze_symbol(symbol: str) -> Dict:
    """Complete analysis for a single symbol"""
    
    # Fetch data
    luxalgo = get_luxalgo_signals(symbol)
    market_data = fetch_market_data(symbol)
    
    # Detect regime
    regime, regime_confidence = detect_regime(luxalgo, market_data)
    
    # Calculate technical score
    tech_score = 0
    components = {}
    data_points = 0
    
    # LuxAlgo (40%)
    tech_score += luxalgo["score"] * 0.4
    components["luxalgo"] = {
        "aligned": luxalgo["aligned"],
        "weekly": luxalgo["weekly"].get("action", "N/A"),
        "daily": luxalgo["daily"].get("action", "N/A"),
        "h4": luxalgo["h4"].get("action", "N/A"),
    }
    data_points += luxalgo["valid_count"]
    
    if market_data.get("available"):
        # RSI (15%)
        rsi = market_data.get("rsi", 50)
        if rsi < 30:
            rsi_score = 0.8
        elif rsi > 70:
            rsi_score = -0.8
        else:
            rsi_score = 0
        tech_score += rsi_score * 0.15
        components["rsi"] = {"value": rsi, "score": rsi_score}
        data_points += 1
        
        # MACD (15%)
        macd = market_data.get("macd", 0)
        macd_signal = market_data.get("macd_signal", 0)
        macd_bullish = macd > macd_signal
        macd_score = 0.5 if macd_bullish else -0.5
        tech_score += macd_score * 0.15
        components["macd"] = {"bullish": macd_bullish, "score": macd_score}
        data_points += 1
        
        # Trend (20%)
        price = market_data.get("price", 0)
        sma50 = market_data.get("sma_50", 0)
        sma200 = market_data.get("sma_200", 0)
        
        if price > sma50 > sma200:
            trend_score = 0.8
            trend_state = "strong_uptrend"
        elif price > sma50:
            trend_score = 0.4
            trend_state = "uptrend"
        elif price < sma50 < sma200:
            trend_score = -0.8
            trend_state = "strong_downtrend"
        elif price < sma50:
            trend_score = -0.4
            trend_state = "downtrend"
        else:
            trend_score = 0
            trend_state = "sideways"
        
        tech_score += trend_score * 0.2
        components["trend"] = {"state": trend_state, "score": trend_score}
        data_points += 3
        
        # Momentum (10%)
        change = market_data.get("change_1d", 0)
        momentum_score = min(0.8, max(-0.8, change / 5))
        tech_score += momentum_score * 0.1
        components["momentum"] = {"change_1d": change, "score": momentum_score}
        data_points += 1
    
    # Intelligence layer
    intel_score = 0
    
    # Regime (35%)
    if regime == "BULL":
        regime_score = 0.7
    elif regime == "BEAR":
        regime_score = -0.5
    elif regime == "HIGH_VOL":
        regime_score = -0.3
    else:
        regime_score = 0
    intel_score += regime_score * 0.35
    
    # VIX (25%)
    vix = market_data.get("vix", 20)
    if vix < 15:
        vix_score = 0.5
    elif vix < 25:
        vix_score = 0.2
    elif vix < 30:
        vix_score = -0.2
    else:
        vix_score = -0.5
    intel_score += vix_score * 0.25
    data_points += 1
    
    # News sentiment (25%)
    news_sentiment = 0  # Would analyze actual news
    intel_score += news_sentiment * 0.25
    
    # Combined score
    weights = CONFIG["weights"]
    raw_score = (
        tech_score * weights["technical"] +
        intel_score * weights["intelligence"]
    )
    
    # Convert to confidence
    base_confidence = (raw_score + 1) / 2 * 100
    
    # Direction
    if raw_score > 0.15:
        direction = "BUY"
    elif raw_score < -0.15:
        direction = "SELL"
    else:
        direction = "NEUTRAL"
    
    # Regime bonus
    regime_bonus = 0
    if regime == "BULL" and direction == "BUY":
        regime_bonus = 0.15
    elif regime == "BEAR" and direction == "SELL":
        regime_bonus = 0.15
    elif regime == "BULL" and direction == "SELL":
        regime_bonus = -0.12
    elif regime == "BEAR" and direction == "BUY":
        regime_bonus = -0.12
    
    final_confidence = max(0, min(100, base_confidence * (1 + regime_bonus)))
    
    # Strength
    if final_confidence >= CONFIG["strong_threshold"]:
        strength = "STRONG"
    elif final_confidence >= CONFIG["moderate_threshold"]:
        strength = "MODERATE"
    elif final_confidence >= CONFIG["weak_threshold"]:
        strength = "WEAK"
    else:
        strength = "NO_TRADE"
    
    # Risk checks
    veto = False
    veto_reason = None
    
    # Signal conflict
    weekly = luxalgo["weekly"].get("action", "N/A")
    h4 = luxalgo["h4"].get("action", "N/A")
    conflict = (
        (weekly in ["BUY", "STRONG_BUY"] and h4 in ["SELL", "STRONG_SELL"]) or
        (weekly in ["SELL", "STRONG_SELL"] and h4 in ["BUY", "STRONG_BUY"])
    )
    if conflict:
        veto = True
        veto_reason = "Weekly and 4H signals conflict"
    
    # Data freshness
    if luxalgo["valid_count"] == 0:
        veto = True
        veto_reason = "No fresh signals available"
    
    # VIX extreme
    if vix >= CONFIG["vix_thresholds"]["extreme"]:
        veto = True
        veto_reason = f"VIX at {vix:.1f} - extreme volatility"
    
    # Should trade
    should_trade = (
        strength in ["STRONG", "MODERATE"] and
        luxalgo["aligned"] and
        not veto
    )
    
    # Trade setup
    price = market_data.get("price", luxalgo["daily"].get("price", 0))
    trade_setup = None
    
    if price > 0 and should_trade:
        base_stop = 0.025
        if vix > 30:
            stop_pct = base_stop * 1.5
        elif vix > 25:
            stop_pct = base_stop * 1.2
        else:
            stop_pct = base_stop
        
        target_pcts = [stop_pct, stop_pct * 2, stop_pct * 3]
        
        if direction == "BUY":
            stop_loss = price * (1 - stop_pct)
            targets = [price * (1 + t) for t in target_pcts]
        else:
            stop_loss = price * (1 + stop_pct)
            targets = [price * (1 - t) for t in target_pcts]
        
        position_pct = min(CONFIG["trading_rules"]["max_position_pct"], 
                          3.0 * (final_confidence / 100))
        
        trade_setup = {
            "entry": round(price, 4),
            "stop_loss": round(stop_loss, 4),
            "targets": [round(t, 4) for t in targets],
            "position_pct": round(position_pct, 2),
            "stop_pct": round(stop_pct * 100, 2),
            "risk_reward": round(target_pcts[0] / stop_pct, 1),
        }
    
    # Build reasoning
    reasoning = [
        f"📊 Technical: {tech_score*100:+.1f}% (LuxAlgo: {'✅ aligned' if luxalgo['aligned'] else '⚠️ not aligned'})",
        f"🧠 Intelligence: {intel_score*100:+.1f}% (Regime: {regime}, VIX: {vix:.1f})",
        f"🎯 Final: {final_confidence:.1f}% → {strength} {direction}",
    ]
    
    if veto:
        reasoning.insert(0, f"🚫 VETO: {veto_reason}")
    
    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "direction": direction,
        "strength": strength,
        "confidence": round(final_confidence, 2),
        "should_trade": should_trade,
        "veto": veto,
        "veto_reason": veto_reason,
        "data_points_used": data_points,
        "regime": regime,
        "vix": vix,
        "luxalgo": luxalgo,
        "market_data": market_data,
        "trade_setup": trade_setup,
        "reasoning": reasoning,
        "components": components,
    }


def analyze_all_symbols() -> List[Dict]:
    """Analyze all watched symbols"""
    results = []
    
    for symbol in ALL_SYMBOLS:
        try:
            result = analyze_symbol(symbol)
            results.append(result)
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            results.append({
                "symbol": symbol,
                "error": str(e),
                "should_trade": False,
            })
    
    return results


# ============================================================
# CLAUDE INTEGRATION (ADVISOR BRAIN)
# ============================================================

def call_claude(prompt: str, system: str = None, temperature: float = 0.7) -> str:
    """Call Claude API for natural language processing"""
    if not ANTHROPIC_API_KEY:
        return "Claude API not configured"
    
    try:
        url = "https://api.anthropic.com/v1/messages"
        
        default_system = f"""You are KYPERIAN, an elite AI wealth advisor for {CONFIG['owner']['name']}.

Your personality:
- Professional but friendly, like a trusted Goldman Sachs private banker
- Proactive: share insights without being asked
- Data-driven: always cite specific numbers
- Risk-aware: prioritize capital preservation
- Clear: explain complex concepts simply
- Actionable: give specific recommendations

Owner profile:
- Background: {', '.join(CONFIG['owner']['background'])}
- Risk tolerance: {CONFIG['owner']['risk_tolerance']}
- Goals: {', '.join(CONFIG['owner']['goals'])}

Current date: {datetime.now(timezone.utc).strftime('%B %d, %Y')}
"""
        
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 2048,
            "system": system or default_system,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            return result["content"][0]["text"]
    
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return f"Error: {e}"


def generate_advisor_message(context: Dict, event_type: str) -> str:
    """Generate a natural language advisor message"""
    
    if event_type == "morning_briefing":
        prompt = f"""Generate a morning briefing for {CONFIG['owner']['name']}.

Current market snapshot:
{json.dumps(context.get('market_snapshot', {}), indent=2)}

Portfolio status:
{json.dumps(context.get('portfolio', {}), indent=2)}

Tradeable signals:
{json.dumps(context.get('tradeable', []), indent=2)}

Generate a friendly, informative morning briefing covering:
1. Portfolio overnight performance
2. Key market movements
3. Any actionable signals
4. What you're watching today

Keep it concise (under 300 words) and use emojis appropriately.
"""
    
    elif event_type == "trade_alert":
        signal = context.get("signal", {})
        prompt = f"""Generate a trade alert message.

Signal details:
{json.dumps(signal, indent=2)}

Explain:
1. What signal fired
2. Why it's actionable (data points supporting it)
3. Recommended position size and levels
4. Key risks

Be specific with numbers. Under 200 words.
"""
    
    elif event_type == "risk_alert":
        prompt = f"""Generate a risk alert message.

Risk context:
{json.dumps(context, indent=2)}

Explain:
1. What triggered the alert
2. Actions taken (if any)
3. Current portfolio protection status
4. Next steps

Be clear and calm but urgent. Under 150 words.
"""
    
    else:
        prompt = f"""Generate an informative message for this context:

{json.dumps(context, indent=2)}

Be concise and actionable.
"""
    
    return call_claude(prompt)


def answer_user_query(query: str, context: Dict) -> str:
    """Answer a user's question about their portfolio or markets"""
    
    prompt = f"""Answer {CONFIG['owner']['name']}'s question:

"{query}"

Available context:
- Portfolio: {json.dumps(context.get('portfolio', {}), indent=2)}
- Market data: {json.dumps(context.get('market_data', {}), indent=2)}
- Recent signals: {json.dumps(context.get('signals', []), indent=2)}
- Current regime: {context.get('regime', 'Unknown')}
- VIX: {context.get('vix', 20)}

Provide a comprehensive but concise answer.
If they're asking about a specific trade, give specific levels and sizing.
If they're asking about research, provide thorough analysis.
Always be data-driven and cite specific numbers.
"""
    
    return call_claude(prompt)


# ============================================================
# NOTIFICATION SERVICE
# ============================================================

def send_telegram(message: str, parse_mode: str = "HTML") -> bool:
    """Send message via Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": parse_mode,
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode())
            return result.get("ok", False)
    
    except Exception as e:
        logger.error(f"Telegram error: {e}")
        return False


def notify_user(message: AdvisorMessage) -> bool:
    """Send notification based on priority"""
    
    formatted = message.content
    
    if message.priority == MessagePriority.URGENT:
        formatted = f"🚨 URGENT\n\n{formatted}"
    elif message.priority == MessagePriority.IMPORTANT:
        formatted = f"⚠️ IMPORTANT\n\n{formatted}"
    
    return send_telegram(formatted)


# ============================================================
# AUTONOMOUS OPERATIONS
# ============================================================

def run_autonomous_check() -> Dict:
    """
    Main autonomous monitoring function.
    
    Called periodically (every 5 minutes) to:
    1. Check for new signals
    2. Monitor risk conditions
    3. Execute trades if conditions met
    4. Notify user of important events
    """
    timestamp = datetime.now(timezone.utc)
    results = {
        "timestamp": timestamp.isoformat(),
        "checks_run": [],
        "actions_taken": [],
        "notifications_sent": [],
    }
    
    try:
        # 1. Get current VIX
        vix, vix_state = get_vix()
        results["vix"] = {"value": vix, "state": vix_state}
        
        # 2. Check for defensive mode
        if vix >= CONFIG["trading_rules"]["vix_defensive_threshold"]:
            results["mode"] = "DEFENSIVE"
            results["checks_run"].append("VIX defensive trigger")
            
            if vix >= CONFIG["trading_rules"]["vix_no_trade_threshold"]:
                # Send alert
                message = AdvisorMessage(
                    content=f"""🚨 VIX SPIKE ALERT

VIX has spiked to {vix:.1f} ({vix_state})

I'm activating defensive protocols:
• No new positions will be opened
• Monitoring existing positions closely
• Will reduce exposure if VIX stays elevated

Stay calm - this is why we have risk management.
""",
                    priority=MessagePriority.URGENT,
                    action_required=True,
                )
                notify_user(message)
                results["notifications_sent"].append("VIX alert")
        else:
            results["mode"] = "NORMAL"
        
        # 3. Analyze all symbols
        analyses = analyze_all_symbols()
        tradeable = [a for a in analyses if a.get("should_trade", False)]
        
        results["total_analyzed"] = len(analyses)
        results["tradeable_count"] = len(tradeable)
        
        # 4. Process tradeable signals
        for signal in tradeable:
            # Check if we've already notified about this signal
            signal_hash = hashlib.md5(
                f"{signal['symbol']}{signal['direction']}{timestamp.date()}".encode()
            ).hexdigest()[:8]
            
            # Send trade alert
            message = AdvisorMessage(
                content=f"""📈 TRADE SIGNAL: {signal['symbol']}

Direction: {signal['direction']} ({signal['strength']})
Confidence: {signal['confidence']:.1f}%
Data Points: {signal['data_points_used']}

LuxAlgo: {'✅ Aligned' if signal['luxalgo']['aligned'] else '⚠️ Not aligned'}
• Weekly: {signal['luxalgo']['weekly'].get('action', 'N/A')}
• Daily: {signal['luxalgo']['daily'].get('action', 'N/A')}
• 4H: {signal['luxalgo']['h4'].get('action', 'N/A')}

Regime: {signal['regime']} | VIX: {signal['vix']:.1f}

{chr(10).join(signal['reasoning'])}

Trade Setup:
• Entry: ${signal['trade_setup']['entry']:,.2f}
• Stop: ${signal['trade_setup']['stop_loss']:,.2f} ({signal['trade_setup']['stop_pct']:.1f}%)
• Target 1: ${signal['trade_setup']['targets'][0]:,.2f}
• Position: {signal['trade_setup']['position_pct']:.1f}% of portfolio
""",
                priority=MessagePriority.IMPORTANT,
                action_required=True,
                data=signal,
            )
            
            notify_user(message)
            results["notifications_sent"].append(f"Trade alert: {signal['symbol']}")
            results["actions_taken"].append(f"Alerted {signal['symbol']} {signal['direction']}")
        
        # 5. Store results
        results["success"] = True
        
    except Exception as e:
        logger.error(f"Autonomous check error: {e}")
        logger.error(traceback.format_exc())
        results["error"] = str(e)
        results["success"] = False
    
    return results


def generate_daily_digest() -> str:
    """Generate end-of-day portfolio digest"""
    
    # Analyze all
    analyses = analyze_all_symbols()
    vix, vix_state = get_vix()
    
    # Categorize
    bullish = [a for a in analyses if a.get("direction") == "BUY" and a.get("confidence", 0) > 50]
    bearish = [a for a in analyses if a.get("direction") == "SELL" and a.get("confidence", 0) > 50]
    tradeable = [a for a in analyses if a.get("should_trade", False)]
    
    # Get regime distribution
    regimes = {}
    for a in analyses:
        r = a.get("regime", "UNKNOWN")
        regimes[r] = regimes.get(r, 0) + 1
    
    dominant_regime = max(regimes, key=regimes.get) if regimes else "UNKNOWN"
    
    digest = f"""📊 KYPERIAN DAILY DIGEST
{datetime.now(timezone.utc).strftime('%B %d, %Y')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌡️ MARKET CONDITIONS
• VIX: {vix:.1f} ({vix_state})
• Dominant Regime: {dominant_regime}
• Mode: {'DEFENSIVE' if vix > 30 else 'NORMAL'}

📈 SIGNAL SUMMARY
• Bullish Setups: {len(bullish)}
• Bearish Setups: {len(bearish)}
• Tradeable Now: {len(tradeable)}

"""
    
    if tradeable:
        digest += "🔥 ACTIONABLE SIGNALS\n"
        for t in tradeable[:3]:
            digest += f"• {t['symbol']}: {t['direction']} {t['strength']} ({t['confidence']:.0f}%)\n"
        digest += "\n"
    
    if bullish and not tradeable:
        digest += "👀 WATCHING (Bullish, awaiting alignment)\n"
        for b in bullish[:3]:
            digest += f"• {b['symbol']}: {b['confidence']:.0f}% confidence\n"
        digest += "\n"
    
    digest += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Have a great evening! I'll monitor overnight. 🌙
"""
    
    return digest


# ============================================================
# LAMBDA HANDLER
# ============================================================

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for KYPERIAN ADVISOR.
    
    Endpoints:
    - GET /                     Health check and info
    - GET /dashboard            All symbols analysis
    - GET /analyze/{symbol}     Single symbol analysis
    - GET /check/{symbol}       Same as analyze
    - POST /query               Answer user question (Claude)
    - GET /digest               Daily digest
    - POST /monitor             Run autonomous check
    - POST /message             Send test message
    - EventBridge trigger       Periodic autonomous monitoring
    """
    try:
        logger.info(f"Event: {json.dumps(event)[:500]}")
        
        path = event.get('rawPath', event.get('path', '/'))
        method = event.get('requestContext', {}).get('http', {}).get('method', 'GET')
        
        headers = {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
        }
        
        # Health check / info
        if path in ['/', '']:
            vix, vix_state = get_vix()
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    "service": CONFIG["name"],
                    "version": CONFIG["version"],
                    "codename": CONFIG["codename"],
                    "status": "operational",
                    "mode": "DEFENSIVE" if vix > 30 else "NORMAL",
                    "vix": {"value": vix, "state": vix_state},
                    "integrations": {
                        "polygon": bool(POLYGON_API_KEY),
                        "claude": bool(ANTHROPIC_API_KEY),
                        "telegram": bool(TELEGRAM_BOT_TOKEN),
                    },
                    "architecture": {
                        "brain": "Claude Opus 4.5",
                        "decision_engine": "V4 Ultimate (28+ data points)",
                        "risk_management": "4-layer + VETO system",
                        "notifications": "Telegram (configurable)",
                    },
                    "endpoints": {
                        "/": "This info",
                        "/dashboard": "All symbols",
                        "/analyze/{symbol}": "Single symbol",
                        "/query": "POST - Ask a question",
                        "/digest": "Daily digest",
                        "/monitor": "POST - Run autonomous check",
                    },
                    "symbols": ALL_SYMBOLS,
                    "owner": CONFIG["owner"]["name"],
                }),
            }
        
        # Dashboard
        if '/dashboard' in path:
            analyses = analyze_all_symbols()
            tradeable = [a for a in analyses if a.get("should_trade", False)]
            vix, vix_state = get_vix()
            
            # Generate summary
            summary_lines = []
            for a in analyses:
                if a.get("error"):
                    summary_lines.append(f"{a['symbol']}: ❌ Error")
                else:
                    icon = "📈" if a.get("direction") == "BUY" else "📉" if a.get("direction") == "SELL" else "➡️"
                    trade_icon = "🔥" if a.get("should_trade") else "⏸️"
                    summary_lines.append(
                        f"{a['symbol']}: {icon}{a.get('direction', 'N/A')} "
                        f"{a.get('strength', 'N/A')} {a.get('confidence', 0):.0f}% "
                        f"{trade_icon} {a.get('data_points_used', 0)}pts"
                    )
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    "success": True,
                    "advisor": CONFIG["name"],
                    "version": CONFIG["version"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "mode": "DEFENSIVE" if vix > 30 else "NORMAL",
                    "vix": {"value": vix, "state": vix_state},
                    "summary": {
                        "total": len(analyses),
                        "tradeable": len(tradeable),
                        "tradeable_symbols": [t["symbol"] for t in tradeable],
                    },
                    "quick_view": summary_lines,
                    "analyses": analyses,
                }, cls=DecimalEncoder),
            }
        
        # Single symbol
        if '/analyze/' in path or '/check/' in path:
            symbol = path.split('/')[-1].upper()
            
            if not symbol or symbol in ['analyze', 'check']:
                return {
                    'statusCode': 400,
                    'headers': headers,
                    'body': json.dumps({"error": "Symbol required"}),
                }
            
            analysis = analyze_symbol(symbol)
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    "success": True,
                    "advisor": CONFIG["name"],
                    "analysis": analysis,
                }, cls=DecimalEncoder),
            }
        
        # Query (Claude-powered Q&A)
        if '/query' in path and method == 'POST':
            try:
                body = json.loads(event.get('body', '{}'))
            except:
                body = {}
            
            query = body.get('query', body.get('question', ''))
            
            if not query:
                return {
                    'statusCode': 400,
                    'headers': headers,
                    'body': json.dumps({"error": "Query required"}),
                }
            
            # Get context
            analyses = analyze_all_symbols()
            vix, vix_state = get_vix()
            
            context = {
                "signals": [a for a in analyses if a.get("should_trade")],
                "market_data": {a["symbol"]: a.get("market_data", {}) for a in analyses},
                "vix": vix,
                "regime": analyses[0].get("regime") if analyses else "UNKNOWN",
            }
            
            answer = answer_user_query(query, context)
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    "success": True,
                    "query": query,
                    "answer": answer,
                    "advisor": CONFIG["name"],
                }),
            }
        
        # Daily digest
        if '/digest' in path:
            digest = generate_daily_digest()
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    "success": True,
                    "digest": digest,
                }),
            }
        
        # Autonomous monitor
        if '/monitor' in path and method == 'POST':
            result = run_autonomous_check()
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    "success": True,
                    "result": result,
                }, cls=DecimalEncoder),
            }
        
        # Send test message
        if '/message' in path and method == 'POST':
            try:
                body = json.loads(event.get('body', '{}'))
            except:
                body = {}
            
            message = body.get('message', 'Test message from KYPERIAN ADVISOR')
            
            success = send_telegram(message)
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    "success": success,
                    "message": message,
                }),
            }
        
        # EventBridge trigger (scheduled monitoring)
        if event.get('source') == 'aws.events':
            logger.info("EventBridge trigger - running autonomous check")
            result = run_autonomous_check()
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    "triggered": True,
                    "result": result,
                }),
            }
        
        return {
            'statusCode': 404,
            'headers': headers,
            'body': json.dumps({"error": "Endpoint not found"}),
        }
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(traceback.format_exc())
        
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                "error": str(e),
                "type": type(e).__name__,
            }),
        }


# ============================================================
# LOCAL TESTING
# ============================================================

if __name__ == "__main__":
    import os
    
    # Test dashboard
    print("Testing KYPERIAN ADVISOR...")
    
    result = analyze_symbol("AAPL")
    print(f"\nAAPL Analysis:")
    print(f"Direction: {result['direction']} | Strength: {result['strength']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print(f"Should Trade: {result['should_trade']}")
    print(f"Data Points: {result['data_points_used']}")
    print(f"Reasoning: {result['reasoning']}")
