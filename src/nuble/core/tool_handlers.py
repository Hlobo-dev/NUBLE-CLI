"""
NUBLE Tool Handlers
===================

Actual implementations of tool functions that execute real API calls,
ML predictions, and data fetching.
"""

import os
import logging
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


def get_stock_quote(symbol: str) -> Dict[str, Any]:
    """Get real-time stock quote from Polygon.io."""
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        return {"error": "POLYGON_API_KEY not set"}
    
    symbol = symbol.upper().replace('$', '')
    
    try:
        # Get previous day's data
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
        response = requests.get(url, params={'apiKey': api_key}, timeout=10)
        
        if response.status_code != 200:
            return {"error": f"API error: {response.status_code}"}
        
        data = response.json()
        if not data.get('results'):
            return {"error": f"No data for {symbol}"}
        
        result = data['results'][0]
        change = result['c'] - result['o']
        change_pct = (change / result['o']) * 100 if result['o'] else 0
        
        # Get ticker details
        details_url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
        details_resp = requests.get(details_url, params={'apiKey': api_key}, timeout=10)
        company_name = symbol
        market_cap = None
        
        if details_resp.status_code == 200:
            details = details_resp.json().get('results', {})
            company_name = details.get('name', symbol)
            market_cap = details.get('market_cap')
        
        return {
            "symbol": symbol,
            "company_name": company_name,
            "price": result['c'],
            "open": result['o'],
            "high": result['h'],
            "low": result['l'],
            "close": result['c'],
            "volume": result['v'],
            "vwap": result.get('vw'),
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "market_cap": market_cap,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Quote fetch failed for {symbol}: {e}")
        return {"error": str(e)}


def get_technical_indicators(
    symbol: str, 
    indicators: List[str] = None, 
    period: int = 14
) -> Dict[str, Any]:
    """Calculate technical indicators for a symbol."""
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        return {"error": "POLYGON_API_KEY not set"}
    
    symbol = symbol.upper().replace('$', '')
    indicators = indicators or ['rsi', 'macd', 'bollinger', 'sma', 'ema']
    
    try:
        # Fetch historical data (need 100 days for proper indicator calculation)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        response = requests.get(url, params={'apiKey': api_key, 'limit': 100}, timeout=10)
        
        if response.status_code != 200:
            return {"error": f"API error: {response.status_code}"}
        
        data = response.json()
        results = data.get('results', [])
        
        if len(results) < 20:
            return {"error": f"Insufficient data for {symbol}"}
        
        # Convert to numpy arrays
        closes = np.array([r['c'] for r in results])
        highs = np.array([r['h'] for r in results])
        lows = np.array([r['l'] for r in results])
        volumes = np.array([r['v'] for r in results])
        
        output = {
            "symbol": symbol,
            "current_price": float(closes[-1]),
            "indicators": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # RSI
        if 'rsi' in indicators:
            rsi = _calculate_rsi(closes, period)
            rsi_value = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50
            output["indicators"]["rsi"] = {
                "value": round(rsi_value, 2),
                "signal": "overbought" if rsi_value > 70 else "oversold" if rsi_value < 30 else "neutral",
                "period": period
            }
        
        # MACD
        if 'macd' in indicators:
            macd, signal, histogram = _calculate_macd(closes)
            output["indicators"]["macd"] = {
                "macd": round(float(macd[-1]), 4),
                "signal": round(float(signal[-1]), 4),
                "histogram": round(float(histogram[-1]), 4),
                "bullish": histogram[-1] > 0 and histogram[-1] > histogram[-2]
            }
        
        # Bollinger Bands
        if 'bollinger' in indicators:
            upper, middle, lower = _calculate_bollinger(closes, 20, 2)
            current = closes[-1]
            position = (current - lower[-1]) / (upper[-1] - lower[-1]) if (upper[-1] - lower[-1]) > 0 else 0.5
            output["indicators"]["bollinger"] = {
                "upper": round(float(upper[-1]), 2),
                "middle": round(float(middle[-1]), 2),
                "lower": round(float(lower[-1]), 2),
                "position": round(position, 2),
                "signal": "near_upper" if position > 0.8 else "near_lower" if position < 0.2 else "middle"
            }
        
        # SMA
        if 'sma' in indicators:
            sma_20 = _calculate_sma(closes, 20)
            sma_50 = _calculate_sma(closes, 50)
            sma_200 = _calculate_sma(closes, 200) if len(closes) >= 200 else None
            output["indicators"]["sma"] = {
                "sma_20": round(float(sma_20[-1]), 2),
                "sma_50": round(float(sma_50[-1]), 2) if len(sma_50) > 0 else None,
                "sma_200": round(float(sma_200[-1]), 2) if sma_200 is not None else None,
                "trend": "bullish" if closes[-1] > sma_20[-1] > sma_50[-1] else "bearish" if closes[-1] < sma_20[-1] < sma_50[-1] else "mixed"
            }
        
        # EMA
        if 'ema' in indicators:
            ema_12 = _calculate_ema(closes, 12)
            ema_26 = _calculate_ema(closes, 26)
            output["indicators"]["ema"] = {
                "ema_12": round(float(ema_12[-1]), 2),
                "ema_26": round(float(ema_26[-1]), 2),
                "bullish": ema_12[-1] > ema_26[-1]
            }
        
        # ATR (Average True Range)
        if 'atr' in indicators:
            atr = _calculate_atr(highs, lows, closes, period)
            atr_pct = (atr[-1] / closes[-1]) * 100
            output["indicators"]["atr"] = {
                "value": round(float(atr[-1]), 2),
                "percent": round(float(atr_pct), 2),
                "volatility": "high" if atr_pct > 3 else "low" if atr_pct < 1 else "normal"
            }
        
        # Stochastic
        if 'stochastic' in indicators:
            k, d = _calculate_stochastic(highs, lows, closes, 14, 3)
            output["indicators"]["stochastic"] = {
                "k": round(float(k[-1]), 2),
                "d": round(float(d[-1]), 2),
                "signal": "overbought" if k[-1] > 80 else "oversold" if k[-1] < 20 else "neutral"
            }
        
        # Overall technical signal
        bullish_signals = 0
        bearish_signals = 0
        
        for name, ind in output["indicators"].items():
            if name == 'rsi':
                if ind['value'] < 30: bullish_signals += 1
                elif ind['value'] > 70: bearish_signals += 1
            elif name == 'macd':
                if ind['bullish']: bullish_signals += 1
                else: bearish_signals += 1
            elif name == 'bollinger':
                if ind['position'] < 0.3: bullish_signals += 1
                elif ind['position'] > 0.7: bearish_signals += 1
            elif name == 'sma':
                if ind['trend'] == 'bullish': bullish_signals += 1
                elif ind['trend'] == 'bearish': bearish_signals += 1
            elif name == 'stochastic':
                if ind['signal'] == 'oversold': bullish_signals += 1
                elif ind['signal'] == 'overbought': bearish_signals += 1
        
        total = bullish_signals + bearish_signals
        if total > 0:
            score = (bullish_signals - bearish_signals) / total
            output["overall_signal"] = {
                "direction": "bullish" if score > 0.3 else "bearish" if score < -0.3 else "neutral",
                "strength": abs(score),
                "bullish_count": bullish_signals,
                "bearish_count": bearish_signals
            }
        else:
            output["overall_signal"] = {"direction": "neutral", "strength": 0}
        
        return output
        
    except Exception as e:
        logger.error(f"Technical indicators failed for {symbol}: {e}")
        return {"error": str(e)}


def run_ml_prediction(
    symbol: str,
    model: str = "ensemble",
    horizon: str = "5d"
) -> Dict[str, Any]:
    """Run ML prediction models for a symbol."""
    symbol = symbol.upper().replace('$', '')
    
    try:
        # Try to use the real ML predictor
        from ...institutional.ml import get_predictor
        
        predictor = get_predictor()
        prediction = predictor.predict(symbol, model_name=model)
        
        return {
            "symbol": symbol,
            "model": model,
            "horizon": horizon,
            "prediction": {
                "direction": prediction.direction,
                "confidence": round(prediction.direction_confidence, 2),
                "predictions": prediction.predictions,
            },
            "regime": {
                "current": prediction.current_regime,
                "confidence": round(prediction.regime_confidence, 2)
            },
            "uncertainty": round(prediction.uncertainty, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except ImportError:
        logger.warning("ML predictor not available, using fallback")
    except Exception as e:
        logger.warning(f"ML prediction failed: {e}, using fallback")
    
    # Fallback: use technical analysis for prediction
    technicals = get_technical_indicators(symbol, ['rsi', 'macd', 'sma'])
    
    if "error" in technicals:
        return technicals
    
    # Generate prediction from technicals
    overall = technicals.get("overall_signal", {})
    direction = overall.get("direction", "neutral")
    strength = overall.get("strength", 0.5)
    
    # Map technical signal to prediction
    current_price = technicals.get("current_price", 0)
    
    if direction == "bullish":
        target = current_price * (1 + 0.02 * strength)
        stop = current_price * (1 - 0.015)
    elif direction == "bearish":
        target = current_price * (1 - 0.02 * strength)
        stop = current_price * (1 + 0.015)
    else:
        target = current_price
        stop = current_price * 0.98
    
    return {
        "symbol": symbol,
        "model": "technical_fallback",
        "horizon": horizon,
        "prediction": {
            "direction": direction,
            "confidence": round(strength, 2),
            "current_price": round(current_price, 2),
            "target": round(target, 2),
            "stop_loss": round(stop, 2),
        },
        "technicals": technicals.get("indicators", {}),
        "note": "Prediction based on technical analysis (ML models unavailable)",
        "timestamp": datetime.now().isoformat()
    }


def search_sec_filings(
    symbol: str,
    query: str,
    filing_type: str = "all"
) -> Dict[str, Any]:
    """Search SEC filings for relevant information."""
    symbol = symbol.upper().replace('$', '')
    
    try:
        # Try to use TENK integration
        from ...institutional.filings.search import FilingsSearch
        
        search = FilingsSearch()
        results = search.search(symbol, query, filing_type=filing_type if filing_type != "all" else None)
        
        return {
            "symbol": symbol,
            "query": query,
            "filing_type": filing_type,
            "results": results[:5],  # Top 5 results
            "total_matches": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except ImportError:
        logger.warning("Filings search not available")
    except Exception as e:
        logger.warning(f"Filings search failed: {e}")
    
    # Fallback: basic SEC EDGAR API
    try:
        cik_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={symbol}&type=10&dateb=&owner=include&count=10&output=json"
        response = requests.get(cik_url, timeout=10, headers={'User-Agent': 'NUBLE/1.0'})
        
        if response.status_code == 200:
            return {
                "symbol": symbol,
                "query": query,
                "note": "Full filing search unavailable, showing recent filings",
                "filings": response.json().get('filings', {}).get('recent', {}),
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        pass
    
    return {
        "symbol": symbol,
        "query": query,
        "error": "SEC filings search not available",
        "suggestion": "Try asking about specific risk factors or financial metrics"
    }


def get_news_sentiment(
    symbol: str,
    days: int = 7
) -> Dict[str, Any]:
    """Get news and sentiment for a symbol using Lambda API."""
    symbol = symbol.upper().replace('$', '')
    
    try:
        from ..lambda_client import get_lambda_client, analyze_symbol
        
        client = get_lambda_client()
        analysis = client.get_analysis(symbol)
        
        return {
            "symbol": symbol,
            "sentiment": {
                "score": analysis.sentiment_score,
                "label": analysis.sentiment_label,
                "direction": "positive" if analysis.sentiment_score > 0.1 else "negative" if analysis.sentiment_score < -0.1 else "neutral"
            },
            "news_count": analysis.news_count,
            "analyst_rating": analysis.analyst_consensus,
            "price_target": analysis.price_target,
            "action": analysis.action,
            "confidence": analysis.score,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.warning(f"Lambda analysis failed: {e}")
    
    # Fallback: Polygon news
    api_key = os.getenv('POLYGON_API_KEY')
    if api_key:
        try:
            url = f"https://api.polygon.io/v2/reference/news"
            response = requests.get(url, params={
                'ticker': symbol,
                'limit': 10,
                'apiKey': api_key
            }, timeout=10)
            
            if response.status_code == 200:
                news = response.json().get('results', [])
                return {
                    "symbol": symbol,
                    "news": [{
                        "title": n.get('title'),
                        "published": n.get('published_utc'),
                        "source": n.get('publisher', {}).get('name')
                    } for n in news[:5]],
                    "count": len(news),
                    "sentiment": {"note": "Advanced sentiment analysis unavailable"},
                    "timestamp": datetime.now().isoformat()
                }
        except Exception:
            pass
    
    return {
        "symbol": symbol,
        "error": "News/sentiment analysis not available",
        "days": days
    }


def analyze_risk(
    symbol: str,
    position_size: float = None
) -> Dict[str, Any]:
    """Analyze risk metrics for a symbol."""
    symbol = symbol.upper().replace('$', '')
    api_key = os.getenv('POLYGON_API_KEY')
    
    if not api_key:
        return {"error": "POLYGON_API_KEY not set"}
    
    try:
        # Fetch historical data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=252)).strftime('%Y-%m-%d')
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        response = requests.get(url, params={'apiKey': api_key, 'limit': 252}, timeout=10)
        
        if response.status_code != 200:
            return {"error": f"API error: {response.status_code}"}
        
        data = response.json()
        results = data.get('results', [])
        
        if len(results) < 30:
            return {"error": f"Insufficient data for {symbol}"}
        
        closes = np.array([r['c'] for r in results])
        returns = np.diff(closes) / closes[:-1]
        
        # Calculate metrics
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)
        
        # VaR (95% confidence)
        var_95 = np.percentile(returns, 5)
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = np.min(drawdowns)
        
        # Beta (vs SPY)
        try:
            spy_url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{start_date}/{end_date}"
            spy_response = requests.get(spy_url, params={'apiKey': api_key}, timeout=10)
            spy_data = spy_response.json().get('results', [])
            
            if len(spy_data) >= len(results):
                spy_closes = np.array([r['c'] for r in spy_data[:len(results)]])
                spy_returns = np.diff(spy_closes) / spy_closes[:-1]
                
                if len(spy_returns) == len(returns):
                    covariance = np.cov(returns, spy_returns)[0, 1]
                    spy_var = np.var(spy_returns)
                    beta = covariance / spy_var if spy_var > 0 else 1.0
                else:
                    beta = 1.0
            else:
                beta = 1.0
        except Exception:
            beta = 1.0
        
        risk_result = {
            "symbol": symbol,
            "metrics": {
                "daily_volatility": round(float(daily_vol), 4),
                "annual_volatility": round(float(annual_vol), 4),
                "var_95": round(float(var_95), 4),
                "max_drawdown": round(float(max_dd), 4),
                "beta": round(float(beta), 2),
                "sharpe_proxy": round(float(np.mean(returns) / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0, 2)
            },
            "risk_level": "high" if annual_vol > 0.4 else "low" if annual_vol < 0.15 else "medium",
            "current_price": float(closes[-1]),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add position-specific risk if position_size provided
        if position_size:
            risk_result["position_risk"] = {
                "size": position_size,
                "daily_var_95": round(position_size * abs(var_95), 2),
                "max_loss_1y": round(position_size * abs(max_dd), 2),
                "suggested_stop": round(closes[-1] * (1 - 2 * daily_vol), 2)
            }
        
        return risk_result
        
    except Exception as e:
        logger.error(f"Risk analysis failed for {symbol}: {e}")
        return {"error": str(e)}


def get_options_flow(symbol: str) -> Dict[str, Any]:
    """Get options flow data for a symbol."""
    symbol = symbol.upper().replace('$', '')
    api_key = os.getenv('POLYGON_API_KEY')
    
    if not api_key:
        return {"error": "POLYGON_API_KEY not set"}
    
    try:
        # Get options contracts
        url = f"https://api.polygon.io/v3/reference/options/contracts"
        response = requests.get(url, params={
            'underlying_ticker': symbol,
            'limit': 50,
            'apiKey': api_key
        }, timeout=10)
        
        if response.status_code != 200:
            return {"error": f"API error: {response.status_code}"}
        
        contracts = response.json().get('results', [])
        
        # Analyze put/call ratio
        calls = [c for c in contracts if c.get('contract_type') == 'call']
        puts = [c for c in contracts if c.get('contract_type') == 'put']
        
        pc_ratio = len(puts) / len(calls) if len(calls) > 0 else 1.0
        
        return {
            "symbol": symbol,
            "total_contracts": len(contracts),
            "calls": len(calls),
            "puts": len(puts),
            "put_call_ratio": round(pc_ratio, 2),
            "sentiment": "bearish" if pc_ratio > 1.5 else "bullish" if pc_ratio < 0.7 else "neutral",
            "note": "Detailed flow data requires premium subscription",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Options flow failed for {symbol}: {e}")
        return {"error": str(e)}


def get_market_regime(symbol: str = "SPY") -> Dict[str, Any]:
    """Detect current market regime."""
    symbol = symbol.upper() if symbol else "SPY"
    
    try:
        # Try HMM detector
        from ...institutional.ml.regime import HMMRegimeModel
        
        detector = HMMRegimeModel(n_regimes=4)
        # Would need data here...
        
    except ImportError:
        pass
    
    # Fallback: use VIX and technicals
    api_key = os.getenv('POLYGON_API_KEY')
    
    if not api_key:
        return {"error": "POLYGON_API_KEY not set"}
    
    try:
        # Get VIX
        vix_url = f"https://api.polygon.io/v2/aggs/ticker/VIX/prev"
        vix_response = requests.get(vix_url, params={'apiKey': api_key}, timeout=10)
        
        vix = 20  # Default
        if vix_response.status_code == 200:
            vix_data = vix_response.json().get('results', [{}])
            if vix_data:
                vix = vix_data[0].get('c', 20)
        
        # Get SPY trend
        spy_technicals = get_technical_indicators("SPY", ['sma', 'rsi'])
        
        trend = spy_technicals.get("indicators", {}).get("sma", {}).get("trend", "mixed")
        rsi = spy_technicals.get("indicators", {}).get("rsi", {}).get("value", 50)
        
        # Determine regime
        if vix > 30:
            regime = "CRISIS"
        elif vix > 25:
            regime = "VOLATILE"
        elif trend == "bullish" and rsi < 70:
            regime = "BULL"
        elif trend == "bearish" and rsi > 30:
            regime = "BEAR"
        else:
            regime = "RANGING"
        
        return {
            "regime": regime,
            "vix": round(vix, 2),
            "trend": trend,
            "rsi": round(rsi, 2),
            "recommendation": {
                "CRISIS": "Reduce exposure, increase hedges",
                "VOLATILE": "Smaller positions, wider stops",
                "BULL": "Trend following, momentum strategies",
                "BEAR": "Short bias, defensive positions",
                "RANGING": "Mean reversion, range trading"
            }.get(regime, "Balanced approach"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Regime detection failed: {e}")
        return {"error": str(e)}


def compare_stocks(
    symbols: List[str],
    metrics: List[str] = None
) -> Dict[str, Any]:
    """Compare multiple stocks."""
    if not symbols or len(symbols) < 2:
        return {"error": "Need at least 2 symbols to compare"}
    
    symbols = [s.upper().replace('$', '') for s in symbols[:5]]  # Max 5
    metrics = metrics or ['performance', 'technicals', 'risk']
    
    comparison = {
        "symbols": symbols,
        "comparison": {},
        "timestamp": datetime.now().isoformat()
    }
    
    for symbol in symbols:
        comparison["comparison"][symbol] = {}
        
        # Get quote
        quote = get_stock_quote(symbol)
        if "error" not in quote:
            comparison["comparison"][symbol]["quote"] = {
                "price": quote.get("price"),
                "change_percent": quote.get("change_percent"),
                "volume": quote.get("volume")
            }
        
        # Get technicals if requested
        if 'technicals' in metrics:
            tech = get_technical_indicators(symbol, ['rsi', 'macd'])
            if "error" not in tech:
                comparison["comparison"][symbol]["technicals"] = {
                    "rsi": tech.get("indicators", {}).get("rsi", {}).get("value"),
                    "signal": tech.get("overall_signal", {}).get("direction")
                }
        
        # Get risk if requested
        if 'risk' in metrics:
            risk = analyze_risk(symbol)
            if "error" not in risk:
                comparison["comparison"][symbol]["risk"] = {
                    "volatility": risk.get("metrics", {}).get("annual_volatility"),
                    "beta": risk.get("metrics", {}).get("beta"),
                    "risk_level": risk.get("risk_level")
                }
    
    # Add winner analysis
    if len(comparison["comparison"]) >= 2:
        scores = {}
        for sym, data in comparison["comparison"].items():
            score = 0
            if data.get("technicals", {}).get("signal") == "bullish":
                score += 1
            if data.get("risk", {}).get("risk_level") == "low":
                score += 1
            if data.get("quote", {}).get("change_percent", 0) > 0:
                score += 1
            scores[sym] = score
        
        comparison["rankings"] = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    
    return comparison


# Helper functions for technical indicators
def _calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros(len(prices))
    avg_loss = np.zeros(len(prices))
    
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def _calculate_macd(prices: np.ndarray) -> tuple:
    """Calculate MACD."""
    ema_12 = _calculate_ema(prices, 12)
    ema_26 = _calculate_ema(prices, 26)
    
    macd = ema_12 - ema_26
    signal = _calculate_ema(macd, 9)
    histogram = macd - signal
    
    return macd, signal, histogram


def _calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA."""
    multiplier = 2 / (period + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


def _calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate SMA."""
    if len(prices) < period:
        return np.array([np.mean(prices)])
    return np.convolve(prices, np.ones(period)/period, mode='valid')


def _calculate_bollinger(prices: np.ndarray, period: int = 20, std_dev: int = 2) -> tuple:
    """Calculate Bollinger Bands."""
    middle = _calculate_sma(prices, period)
    
    # Pad middle to match prices length
    pad_length = len(prices) - len(middle)
    middle = np.pad(middle, (pad_length, 0), mode='edge')
    
    rolling_std = np.zeros(len(prices))
    for i in range(period - 1, len(prices)):
        rolling_std[i] = np.std(prices[i-period+1:i+1])
    
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    
    return upper, middle, lower


def _calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR."""
    tr = np.zeros(len(closes))
    tr[0] = highs[0] - lows[0]
    
    for i in range(1, len(closes)):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
    
    atr = np.zeros(len(closes))
    atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(closes)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def _calculate_stochastic(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, k_period: int = 14, d_period: int = 3) -> tuple:
    """Calculate Stochastic Oscillator."""
    k = np.zeros(len(closes))
    
    for i in range(k_period - 1, len(closes)):
        low_min = np.min(lows[i-k_period+1:i+1])
        high_max = np.max(highs[i-k_period+1:i+1])
        
        if high_max - low_min > 0:
            k[i] = ((closes[i] - low_min) / (high_max - low_min)) * 100
        else:
            k[i] = 50
    
    d = _calculate_sma(k, d_period)
    d = np.pad(d, (len(k) - len(d), 0), mode='edge')
    
    return k, d


class ToolHandlers:
    """
    Central handler dispatcher for all tools.
    
    Provides static method dispatch for tool execution.
    """
    
    HANDLERS = {
        'get_stock_quote': get_stock_quote,
        'get_technical_indicators': get_technical_indicators,
        'run_ml_prediction': run_ml_prediction,
        'search_sec_filings': search_sec_filings,
        'get_news_sentiment': get_news_sentiment,
        'analyze_risk': analyze_risk,
        'get_options_flow': get_options_flow,
        'get_market_regime': get_market_regime,
        'compare_stocks': compare_stocks,
    }
    
    @classmethod
    def dispatch(cls, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Dispatch a tool call to the appropriate handler.
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Arguments to pass to the handler
            
        Returns:
            Result from the tool handler
        """
        handler = cls.HANDLERS.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            return handler(**kwargs)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"error": str(e)}
    
    @classmethod
    def get_available_tools(cls) -> List[str]:
        """Get list of available tool names."""
        return list(cls.HANDLERS.keys())
