import os
import asyncio
import logging
from . import console
from .agent.agent import Agent
from .agent.prompts import agent_prompt
from rich.spinner import Spinner
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
import time
import threading
from .helpers import get_timeout_message, TokenCounter, handle_command, get_api_key

# Setup logging
logger = logging.getLogger(__name__)

# Default Polygon API key (fallback if not in env)
_DEFAULT_POLYGON_KEY = 'JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY'


def _get_polygon_key() -> str:
    """Get Polygon API key from environment or fallback."""
    return os.getenv('POLYGON_API_KEY', _DEFAULT_POLYGON_KEY)


def run_async(coro):
    """
    Run an async coroutine synchronously.
    Handles nested event loop scenarios gracefully.
    """
    try:
        # Try to get existing event loop
        loop = asyncio.get_running_loop()
        # If we're already in an async context, use nest_asyncio or run in thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result(timeout=30)
    except RuntimeError:
        # No running event loop, we can use asyncio.run directly
        return asyncio.run(coro)


# ML Integration (optional, graceful fallback if not available)
try:
    from institutional.ml import MLPredictor, MLIntegration, get_predictor
    ML_AVAILABLE = True
except ImportError:
    try:
        # Try relative import if absolute fails
        from ..institutional.ml import MLPredictor, MLIntegration, get_predictor
        ML_AVAILABLE = True
    except ImportError:
        ML_AVAILABLE = False
        logger.info("ML module not available, running without ML predictions")

# Unified Services Integration
try:
    from .services import get_services, UnifiedServices
    from .router import get_router, SmartRouter, QueryIntent
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False
    logger.info("Unified services not available")

# Ultimate Decision Engine Integration (NEW!)
try:
    from .decision.ultimate_engine import UltimateDecisionEngine
    DECISION_ENGINE_AVAILABLE = True
except ImportError:
    DECISION_ENGINE_AVAILABLE = False
    logger.info("Ultimate Decision Engine not available")


class Manager:
    def __init__(self, enable_ml: bool = True, enable_fast_path: bool = True, enable_decision_engine: bool = True):
        self.tier = "free"
        api_key = get_api_key()
        self.agent = Agent(api_key=api_key)
        self.system_prompt = agent_prompt
        self.token_counter = TokenCounter()
        
        # Initialize ML components if available and enabled
        self.ml_enabled = enable_ml and ML_AVAILABLE
        self._ml_predictor = None
        self._ml_integration = None
        
        if self.ml_enabled:
            try:
                self._ml_predictor = get_predictor()
                self._ml_integration = MLIntegration(self._ml_predictor)
                logger.info("ML components initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize ML: {e}")
                self.ml_enabled = False
        
        # Initialize Ultimate Decision Engine (NEW!)
        self.decision_engine_enabled = enable_decision_engine and DECISION_ENGINE_AVAILABLE
        self._decision_engine = None
        
        if self.decision_engine_enabled:
            try:
                self._decision_engine = UltimateDecisionEngine()
                logger.info("Ultimate Decision Engine initialized (28+ data points)")
            except Exception as e:
                logger.warning(f"Failed to initialize Decision Engine: {e}")
                self.decision_engine_enabled = False
        
        # Initialize unified services and smart router
        self.fast_path_enabled = enable_fast_path and SERVICES_AVAILABLE
        self._services = None
        self._router = None
        
        if self.fast_path_enabled:
            try:
                self._services = get_services()
                self._router = get_router()
                logger.info("Unified services initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize unified services: {e}")
                self.fast_path_enabled = False
    
    @property
    def services(self) -> 'UnifiedServices':
        """Get unified services instance."""
        return self._services
    
    @property
    def router(self) -> 'SmartRouter':
        """Get smart router instance."""
        return self._router
    
    @property
    def decision_engine(self) -> 'UltimateDecisionEngine':
        """Get Ultimate Decision Engine instance."""
        return self._decision_engine
    
    @property
    def ml_predictor(self):
        """Get ML predictor instance."""
        return self._ml_predictor
    
    @property 
    def ml_integration(self):
        """Get ML integration instance."""
        return self._ml_integration
    
    def get_ml_prediction(self, symbol: str) -> str:
        """
        Get ML prediction for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Formatted prediction string or error message
        """
        if not self.ml_enabled:
            return "[dim]ML predictions not available[/dim]"
        
        try:
            prediction = self._ml_predictor.predict(symbol)
            return self._ml_integration.format_prediction_for_display(prediction)
        except Exception as e:
            logger.warning(f"ML prediction failed for {symbol}: {e}")
            return f"[dim]ML prediction unavailable: {str(e)[:50]}[/dim]"
    
    def enhance_response_with_ml(self, symbol: str, response: dict) -> dict:
        """
        Enhance a response with ML predictions.
        
        Args:
            symbol: Stock ticker
            response: Base response dict
            
        Returns:
            Enhanced response with ML predictions
        """
        if not self.ml_enabled:
            return response
        
        try:
            return self._ml_integration.enhance_query_response(symbol, response)
        except Exception as e:
            logger.warning(f"Failed to enhance response: {e}")
            return response

    # =========================================================================
    # Fast Path Handling (No LLM required)
    # =========================================================================
    
    def _handle_fast_path(self, routed) -> str:
        """
        Handle queries that don't need LLM.
        Returns formatted response or empty string if can't handle.
        """
        if not self.fast_path_enabled or not routed.fast_path:
            return ""
        
        symbol = routed.symbols[0] if routed.symbols else None
        if not symbol:
            return ""
        
        try:
            if routed.intent == QueryIntent.QUOTE:
                return self._fast_quote(symbol)
            elif routed.intent == QueryIntent.PREDICTION:
                return self._fast_prediction(symbol, routed.parameters)
            elif routed.intent == QueryIntent.TECHNICAL:
                return self._fast_technical(symbol, routed.parameters)
            elif routed.intent == QueryIntent.PATTERN:
                return self._fast_patterns(symbol)
            elif routed.intent == QueryIntent.SENTIMENT:
                return self._fast_sentiment(symbol)
            elif routed.intent == QueryIntent.FILINGS_SEARCH:
                return self._fast_filings_search(symbol, routed.parameters)
            # Check for trading decision intents
            elif routed.intent in [QueryIntent.PREDICTION] and self.decision_engine_enabled:
                return self._fast_decision(symbol, routed.parameters)
        except Exception as e:
            logger.warning(f"Fast path failed: {e}")
            return ""
        
        return ""
    
    def _fast_quote(self, symbol: str) -> str:
        """Fast quote without LLM."""
        try:
            import requests
            
            api_key = _get_polygon_key()
            if not api_key:
                return f"[yellow]Quote unavailable - POLYGON_API_KEY not set[/yellow]"
            
            # Map crypto tickers to Polygon format
            CRYPTO_TICKERS = {
                'BTC': 'X:BTCUSD', 'ETH': 'X:ETHUSD', 'SOL': 'X:SOLUSD',
                'XRP': 'X:XRPUSD', 'ADA': 'X:ADAUSD', 'DOT': 'X:DOTUSD',
                'DOGE': 'X:DOGEUSD', 'AVAX': 'X:AVAXUSD', 'LINK': 'X:LINKUSD',
                'MATIC': 'X:MATICUSD', 'LTC': 'X:LTCUSD',
                'BTCUSD': 'X:BTCUSD', 'ETHUSD': 'X:ETHUSD', 'SOLUSD': 'X:SOLUSD',
            }
            
            display_symbol = symbol.upper()
            polygon_symbol = CRYPTO_TICKERS.get(symbol.upper(), symbol.upper())
            is_crypto = symbol.upper() in CRYPTO_TICKERS
            
            # Use Polygon.io for real-time quote
            url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/prev"
            response = requests.get(url, params={'apiKey': api_key}, timeout=5)
            
            if response.status_code != 200:
                return f"[yellow]Could not get quote for {display_symbol}[/yellow]"
            
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                return f"[yellow]No data available for {display_symbol}[/yellow]"
            
            quote = results[0]
            close_price = quote.get('c', 0)
            open_price = quote.get('o', 0)
            high = quote.get('h', 0)
            low = quote.get('l', 0)
            volume = quote.get('v', 0)
            
            # Calculate change
            change_pct = ((close_price - open_price) / open_price * 100) if open_price > 0 else 0
            
            # Build rich display
            change_color = "green" if change_pct >= 0 else "red"
            change_arrow = "↑" if change_pct >= 0 else "↓"
            
            # Format prices appropriately for crypto vs stocks
            price_fmt = f"${close_price:,.2f}" if is_crypto else f"${close_price:.2f}"
            range_fmt = f"${low:,.2f} - ${high:,.2f}" if is_crypto else f"${low:.2f} - ${high:.2f}"
            
            output = []
            output.append(f"\n[bold bright_cyan]{display_symbol}[/bold bright_cyan]")
            output.append(f"[bold white]{price_fmt}[/bold white]")
            output.append(f"[{change_color}]{change_arrow} {change_pct:.2f}%[/{change_color}]")
            output.append(f"[dim]Volume: {volume:,.0f}[/dim]")
            output.append(f"[dim]Range: {range_fmt}[/dim]")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.warning(f"Fast quote failed for {symbol}: {e}")
            return f"[dim]Quote unavailable for {symbol.upper()}. Try asking: 'What is {symbol.upper()} trading at?'[/dim]"
    
    def _fast_prediction(self, symbol: str, params: dict) -> str:
        """Fast ML prediction without LLM.
        
        Uses direct import of registry module to avoid triggering 
        heavy ML dependencies (torch, pandas) through the package __init__.py.
        """
        try:
            import importlib
            import sys
            import os
            
            registry = None
            
            # Strategy: Import registry.py DIRECTLY to avoid institutional/__init__.py
            # which pulls in torch, pandas, aiohttp, etc.
            # We only need ModelMetadata and PreTrainedModelRegistry from registry.py
            registry_module = None
            
            # Find the registry module file path
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            registry_path = os.path.join(base_path, 'institutional', 'ml', 'registry.py')
            
            if os.path.exists(registry_path):
                # Direct file import - bypasses all __init__.py chains
                spec = importlib.util.spec_from_file_location(
                    "institutional_ml_registry", registry_path
                )
                registry_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(registry_module)
                registry = registry_module.get_registry()
            
            if registry is None:
                # Fallback: try normal import (may work if all deps are installed)
                try:
                    from institutional.ml.registry import get_registry
                    registry = get_registry()
                except ImportError:
                    pass
            
            if registry is None:
                return f"[dim]Registry not available. Try full query for {symbol}.[/dim]"
            
            # Check if model has metadata (even without .pt file, we can show validated metrics)
            meta = registry.get_metadata(symbol)
            if meta:
                direction = 'up' if meta.validation_sharpe > 0 else 'down'
                dir_color = "green" if direction == 'up' else "red"
                dir_arrow = "↑" if direction == 'up' else "↓"
                
                sharpe_display = meta.wf_sharpe if meta.wf_sharpe else meta.validation_sharpe
                
                output = []
                output.append(f"\n[bold bright_cyan]ML Prediction: {symbol}[/bold bright_cyan]")
                output.append(f"[{dir_color} bold]{dir_arrow} {direction.upper()}[/{dir_color} bold]")
                output.append(f"[white]Model Grade: {meta.grade}[/white]")
                output.append(f"[white]Walk-Forward Sharpe: {sharpe_display:.2f}[/white]")
                output.append(f"[white]Directional Accuracy: {meta.directional_accuracy*100:.1f}%[/white]")
                output.append(f"[dim]Model: {meta.model_type}[/dim]")
                output.append(f"[dim]Trained: {meta.trained_at[:10]}[/dim]")
                
                return "\n".join(output)
            
            # No pre-trained model available - try to get a quote-based prediction instead
            return self._fallback_prediction(symbol)
            
        except Exception as e:
            logger.warning(f"Fast prediction failed for {symbol}: {e}")
            return self._fallback_prediction(symbol)
    
    def _fallback_prediction(self, symbol: str) -> str:
        """Fallback prediction using Polygon price data and basic technical analysis.
        
        Uses /prev endpoint for current price (most up-to-date) and
        range endpoint for historical analysis (may be delayed on free tier).
        """
        try:
            import requests
            
            api_key = _get_polygon_key()
            if not api_key:
                return f"[dim]Prediction unavailable for {symbol}. Try: 'predict {symbol}' for full analysis.[/dim]"
            
            # Map crypto tickers to Polygon format
            CRYPTO_TICKERS = {
                'BTC': 'X:BTCUSD', 'ETH': 'X:ETHUSD', 'SOL': 'X:SOLUSD',
                'XRP': 'X:XRPUSD', 'ADA': 'X:ADAUSD', 'DOT': 'X:DOTUSD',
                'DOGE': 'X:DOGEUSD', 'AVAX': 'X:AVAXUSD', 'LINK': 'X:LINKUSD',
                'MATIC': 'X:MATICUSD', 'LTC': 'X:LTCUSD', 'UNI': 'X:UNIUSD',
                'ATOM': 'X:ATOMUSD', 'SHIB': 'X:SHIBUSD', 'NEAR': 'X:NEARUSD',
                'ARB': 'X:ARBUSD', 'OP': 'X:OPUSD',
                'BTCUSD': 'X:BTCUSD', 'ETHUSD': 'X:ETHUSD', 'SOLUSD': 'X:SOLUSD',
            }
            
            display_symbol = symbol.upper()
            polygon_symbol = CRYPTO_TICKERS.get(symbol.upper(), symbol.upper())
            is_crypto = symbol.upper() in CRYPTO_TICKERS
            
            # Step 1: Get CURRENT price from /prev endpoint (most up-to-date)
            prev_url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/prev"
            prev_resp = requests.get(prev_url, params={'apiKey': api_key}, timeout=10)
            
            current_price = None
            if prev_resp.status_code == 200:
                prev_data = prev_resp.json()
                prev_results = prev_data.get('results', [])
                if prev_results:
                    current_price = prev_results[0].get('c', 0)
            
            # Step 2: Get historical data for technical analysis
            # IMPORTANT: Do NOT use 'limit' param — it truncates from the START,
            # cutting off the most recent data on free tier
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/range/1/day/{start_date}/{end_date}"
            response = requests.get(url, params={
                'apiKey': api_key, 
                'adjusted': 'true',
                'sort': 'asc'
            }, timeout=10)
            
            if response.status_code != 200:
                return f"[dim]Prediction unavailable for {display_symbol}. Try: 'predict {display_symbol}' for full analysis.[/dim]"
            
            data = response.json()
            results = data.get('results', [])
            
            if len(results) < 20:
                return f"[dim]Insufficient data for {display_symbol} prediction. Try: 'predict {display_symbol}' for full analysis.[/dim]"
            
            import numpy as np
            closes = np.array([r.get('c', 0) for r in results])
            
            # Use /prev price as current if available (it's always more current than range data)
            # Range data on free tier may be delayed by days/weeks
            if current_price and current_price > 0:
                # Append current price to closes array for accurate analysis
                hist_latest = closes[-1]
                if abs(current_price - hist_latest) / hist_latest > 0.001:
                    # Current price differs from last historical bar — use current
                    closes = np.append(closes, current_price)
            else:
                current_price = closes[-1]
            
            current = current_price
            
            # Calculate technical signals
            # SMA 20 and SMA 50
            sma_20 = np.mean(closes[-20:])
            sma_50 = np.mean(closes[-min(50, len(closes)):])
            
            # RSI (14-period) using Wilder's smoothing method
            if len(closes) >= 15:
                deltas = np.diff(closes[-(14+1):])  # Need 15 values for 14 deltas
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100.0  # All gains, no losses
            else:
                rsi = 50.0  # Not enough data
            
            # Momentum calculations
            momentum_5d = ((current - closes[-6]) / closes[-6] * 100) if len(closes) >= 6 else 0
            momentum_20d = ((current - closes[-21]) / closes[-21] * 100) if len(closes) >= 21 else 0
            
            # Volatility (20-day standard deviation of returns)
            if len(closes) >= 21:
                returns = np.diff(closes[-21:]) / closes[-21:-1]
                volatility = np.std(returns) * 100
            else:
                volatility = 0
            
            # Determine direction from signals (weighted scoring)
            bull_score = 0
            bear_score = 0
            total_weight = 0
            
            # Signal 1: Price vs SMA20 (weight: 2)
            w = 2
            total_weight += w
            if current > sma_20:
                bull_score += w
            else:
                bear_score += w
            
            # Signal 2: SMA20 vs SMA50 trend (weight: 2)
            w = 2
            total_weight += w
            if sma_20 > sma_50:
                bull_score += w
            else:
                bear_score += w
            
            # Signal 3: RSI (weight: 1)
            w = 1
            total_weight += w
            if rsi < 30:
                bull_score += w  # Oversold = reversal potential
            elif rsi > 70:
                bear_score += w  # Overbought = reversal potential
            else:
                # Neutral RSI — slight lean based on 50 threshold
                if rsi >= 50:
                    bull_score += w * 0.3
                else:
                    bear_score += w * 0.3
            
            # Signal 4: Short-term momentum 5D (weight: 1.5)
            w = 1.5
            total_weight += w
            if momentum_5d > 0:
                bull_score += w
            else:
                bear_score += w
            
            # Signal 5: Medium-term momentum 20D (weight: 1.5)
            w = 1.5
            total_weight += w
            if momentum_20d > 0:
                bull_score += w
            else:
                bear_score += w
            
            # Signal 6: Price momentum (current vs 10-day avg, weight: 1)
            if len(closes) >= 10:
                w = 1
                total_weight += w
                avg_10 = np.mean(closes[-10:])
                if current > avg_10:
                    bull_score += w
                else:
                    bear_score += w
            
            # Determine direction
            bull_pct = bull_score / total_weight if total_weight > 0 else 0.5
            
            if bull_pct > 0.6:
                direction = "BULLISH"
                dir_color = "green"
                dir_arrow = "↑"
            elif bull_pct < 0.4:
                direction = "BEARISH"
                dir_color = "red"
                dir_arrow = "↓"
            else:
                direction = "NEUTRAL"
                dir_color = "yellow"
                dir_arrow = "→"
            
            confidence = abs(bull_pct - 0.5) * 2  # 0 to 1 scale, 0 = neutral, 1 = strong
            confidence = max(0.3, min(0.85, confidence + 0.3))  # Clamp to reasonable range
            
            # Build output
            price_fmt = f"${current:,.2f}" if is_crypto else f"${current:.2f}"
            
            # Individual indicator colors
            mom5_color = "green" if momentum_5d > 0 else "red"
            mom20_color = "green" if momentum_20d > 0 else "red"
            
            output = []
            output.append(f"\n[bold bright_cyan]📊 Technical Prediction: {display_symbol}[/bold bright_cyan]")
            output.append(f"[bold white]Current Price: {price_fmt}[/bold white]")
            output.append(f"[{dir_color} bold]{dir_arrow} {direction}[/{dir_color} bold] (Confidence: {confidence:.0%})")
            output.append(f"")
            output.append(f"[white]RSI (14):[/white] {rsi:.1f} {'[green](Oversold — bounce likely)[/green]' if rsi < 30 else '[red](Overbought — pullback likely)[/red]' if rsi > 70 else '[dim](Neutral)[/dim]'}")
            output.append(f"[white]SMA 20:[/white] ${sma_20:,.2f} {'[green](Price Above)[/green]' if current > sma_20 else '[red](Price Below)[/red]'}")
            output.append(f"[white]SMA 50:[/white] ${sma_50:,.2f} {'[green](Uptrend)[/green]' if sma_20 > sma_50 else '[red](Downtrend)[/red]'}")
            output.append(f"[white]Momentum 5D:[/white] [{mom5_color}]{momentum_5d:+.2f}%[/{mom5_color}]")
            output.append(f"[white]Momentum 20D:[/white] [{mom20_color}]{momentum_20d:+.2f}%[/{mom20_color}]")
            if volatility > 0:
                vol_label = "High" if volatility > 3 else "Moderate" if volatility > 1.5 else "Low"
                output.append(f"[white]Volatility:[/white] {volatility:.2f}% daily ({vol_label})")
            output.append(f"[dim]Based on {len(results)} days of data • Price from Polygon.io[/dim]")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.warning(f"Fallback prediction failed for {symbol}: {e}")
            return f"[dim]Prediction unavailable for {symbol.upper()}. Try: 'predict {symbol.upper()}' for full analysis.[/dim]"
    
    def _fast_technical(self, symbol: str, params: dict) -> str:
        """Fast technical analysis without LLM."""
        try:
            import requests
            import numpy as np
            
            api_key = _get_polygon_key()
            if not api_key:
                return f"[yellow]Technical analysis unavailable - POLYGON_API_KEY not set[/yellow]"
            
            # Map crypto tickers to Polygon format
            CRYPTO_TICKERS = {
                'BTC': 'X:BTCUSD', 'ETH': 'X:ETHUSD', 'SOL': 'X:SOLUSD',
                'XRP': 'X:XRPUSD', 'ADA': 'X:ADAUSD', 'DOT': 'X:DOTUSD',
                'DOGE': 'X:DOGEUSD', 'AVAX': 'X:AVAXUSD', 'LINK': 'X:LINKUSD',
                'MATIC': 'X:MATICUSD', 'LTC': 'X:LTCUSD',
            }
            
            display_symbol = symbol.upper()
            polygon_symbol = CRYPTO_TICKERS.get(symbol.upper(), symbol.upper())
            is_crypto = symbol.upper() in CRYPTO_TICKERS
            
            # Get 90 days of data for technical indicators (no limit to avoid truncation)
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/range/1/day/{start_date}/{end_date}"
            response = requests.get(url, params={
                'apiKey': api_key,
                'adjusted': 'true',
                'sort': 'asc'
            }, timeout=10)
            
            if response.status_code != 200:
                return f"[yellow]Could not get technical data for {display_symbol}[/yellow]"
            
            data = response.json()
            results = data.get('results', [])
            
            if len(results) < 14:
                return f"[yellow]Insufficient data for technical analysis of {display_symbol}[/yellow]"
            
            # Get current price from /prev (most up-to-date)
            prev_url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/prev"
            prev_resp = requests.get(prev_url, params={'apiKey': api_key}, timeout=5)
            current_price = None
            if prev_resp.status_code == 200:
                prev_data = prev_resp.json()
                prev_results = prev_data.get('results', [])
                if prev_results:
                    current_price = prev_results[0].get('c', 0)
            
            # Extract close prices
            closes = np.array([r.get('c', 0) for r in results])
            highs = np.array([r.get('h', 0) for r in results])
            lows = np.array([r.get('l', 0) for r in results])
            
            # Append current price if more recent
            if current_price and current_price > 0:
                if abs(current_price - closes[-1]) / closes[-1] > 0.001:
                    closes = np.append(closes, current_price)
            else:
                current_price = closes[-1]
            
            # Calculate RSI (14-period)
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate SMAs
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
            sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else closes[-1]
            
            # Calculate MACD (simplified)
            ema_12 = np.mean(closes[-12:])
            ema_26 = np.mean(closes[-26:]) if len(closes) >= 26 else closes[-1]
            macd = ema_12 - ema_26
            
            # Current price
            cp = current_price
            
            # Price formatting
            pfmt = lambda p: f"${p:,.2f}" if is_crypto else f"${p:.2f}"
            
            # Build technical table
            table = Table(title=f"📊 Technical Analysis: {display_symbol}", border_style="bright_cyan")
            table.add_column("Indicator", style="white")
            table.add_column("Value", style="bright_white")
            table.add_column("Signal", style="white")
            
            # Price
            table.add_row("Current Price", pfmt(cp), "")
            
            # RSI
            rsi_signal = "[green]Oversold[/green]" if rsi < 30 else "[red]Overbought[/red]" if rsi > 70 else "Neutral"
            table.add_row("RSI (14)", f"{rsi:.1f}", rsi_signal)
            
            # MACD
            macd_signal = "[green]Bullish[/green]" if macd > 0 else "[red]Bearish[/red]"
            table.add_row("MACD", f"{macd:.3f}", macd_signal)
            
            # Moving Averages
            ma_signal = "[green]Bullish[/green]" if sma_20 > sma_50 else "[red]Bearish[/red]"
            table.add_row("SMA 20/50", f"{pfmt(sma_20)} / {pfmt(sma_50)}", ma_signal)
            
            # Price vs MAs
            price_ma_signal = "[green]Above[/green]" if cp > sma_20 else "[red]Below[/red]"
            table.add_row("Price vs SMA20", pfmt(cp), price_ma_signal)
            
            # Overall signal
            bullish_count = sum([
                rsi < 50,
                macd > 0,
                sma_20 > sma_50,
                cp > sma_20
            ])
            overall = 'bullish' if bullish_count >= 3 else 'bearish' if bullish_count <= 1 else 'neutral'
            overall_color = "green" if overall == 'bullish' else "red" if overall == 'bearish' else "yellow"
            
            # Render table
            from io import StringIO
            from rich.console import Console as RichConsole
            
            buffer = StringIO()
            temp_console = RichConsole(file=buffer, force_terminal=True, width=80)
            temp_console.print(table)
            output = [buffer.getvalue()]
            output.append(f"[{overall_color} bold]Overall: {overall.upper()}[/{overall_color} bold]")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.warning(f"Fast technical analysis failed for {symbol}: {e}")
            return f"[dim]Technical analysis unavailable for {symbol.upper()}. Try: 'technical analysis for {symbol.upper()}'[/dim]"
    
    def _fast_patterns(self, symbol: str) -> str:
        """Fast pattern detection without LLM."""
        # Pattern detection requires price data analysis - defer to full LLM path
        return f"[dim]Pattern detection for {symbol} requires full analysis. Try asking: 'What patterns do you see in {symbol}?'[/dim]"
    
    def _fast_sentiment(self, symbol: str) -> str:
        """Fast sentiment analysis without LLM."""
        # Sentiment analysis requires news/social data - defer to full LLM path
        return f"[dim]Sentiment analysis for {symbol} requires full analysis. Try asking: 'What is the sentiment for {symbol}?'[/dim]"
    
    def _fast_filings_search(self, symbol: str, params: dict) -> str:
        """Fast filings search without deep analysis."""
        try:
            import requests
            
            # Get company info from SEC EDGAR
            headers = {'User-Agent': 'VibeFi Research contact@vibefi.com'}
            
            # Try to get recent filings via SEC EDGAR
            url = f"https://data.sec.gov/submissions/CIK{symbol.zfill(10)}.json"
            
            # For ticker symbols, we'll return a simple message with link
            output = [f"\n[bold bright_cyan]📄 SEC Filings: {symbol}[/bold bright_cyan]"]
            output.append(f"  • [white]View all filings:[/white] https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={symbol}&type=10-K&dateb=&owner=include&count=40")
            output.append(f"  • [dim]Common forms: 10-K (Annual), 10-Q (Quarterly), 8-K (Events)[/dim]")
            output.append(f"\n[dim]Use 'analyze {symbol} 10-K' for deep AI analysis[/dim]")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.warning(f"Fast filings search failed for {symbol}: {e}")
            return f"[dim]Filings search unavailable for {symbol}. Try: 'search filings for {symbol}'[/dim]"
    
    def _fast_decision(self, symbol: str, params: dict) -> str:
        """Fast trading decision using Ultimate Decision Engine."""
        if not self.decision_engine_enabled or not self._decision_engine:
            return ""
        
        try:
            # Get decision from Ultimate Decision Engine
            decision = self._decision_engine.make_decision(symbol)
            
            if not decision:
                return f"[yellow]Could not generate decision for {symbol}[/yellow]"
            
            # Extract decision data
            action = decision.get('action', 'HOLD')
            confidence = decision.get('confidence', 0)
            risk_score = decision.get('risk_score', 0.5)
            risk_veto = decision.get('risk_veto', False)
            entry_price = decision.get('entry_price')
            stop_loss = decision.get('stop_loss')
            take_profit = decision.get('take_profit')
            position_size = decision.get('position_size')
            reasoning = decision.get('reasoning', '')
            data_sources = decision.get('data_sources', [])
            score_breakdown = decision.get('score_breakdown', {})
            
            # Determine colors
            if action == 'BUY':
                action_color = 'bold green'
                action_icon = '📈'
            elif action == 'SELL':
                action_color = 'bold red'
                action_icon = '📉'
            else:
                action_color = 'bold yellow'
                action_icon = '⏸️'
            
            # Build rich output
            output = []
            output.append(f"\n[bold bright_cyan]🎯 NUBLE Ultimate Decision: {symbol}[/bold bright_cyan]")
            output.append(f"[bold white]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold white]")
            output.append(f"{action_icon} [{action_color}]{action}[/{action_color}]  |  Confidence: [white]{confidence:.1%}[/white]")
            
            # Risk warning if vetoed
            if risk_veto:
                output.append(f"\n[bold red]⚠️  RISK VETO ACTIVE - HIGH RISK CONDITIONS DETECTED[/bold red]")
            
            output.append(f"\n[white]Risk Score:[/white] [{'red' if risk_score > 0.7 else 'yellow' if risk_score > 0.4 else 'green'}]{risk_score:.2f}[/]")
            
            # Entry levels if available
            if entry_price is not None:
                output.append(f"\n[bold white]📊 Trade Setup:[/bold white]")
                output.append(f"  • Entry: [white]${entry_price:.2f}[/white]")
                if stop_loss:
                    output.append(f"  • Stop Loss: [red]${stop_loss:.2f}[/red]")
                if take_profit:
                    output.append(f"  • Take Profit: [green]${take_profit:.2f}[/green]")
                if position_size:
                    output.append(f"  • Position Size: [white]{position_size:.1%} of portfolio[/white]")
            
            # Score breakdown if available
            if score_breakdown:
                output.append(f"\n[bold white]📈 Score Breakdown:[/bold white]")
                for category, score in score_breakdown.items():
                    bar_len = int(score * 10) if isinstance(score, (int, float)) else 5
                    bar = "█" * bar_len + "░" * (10 - bar_len)
                    output.append(f"  • {category}: [{bar}] {score:.2f}" if isinstance(score, (int, float)) else f"  • {category}: {score}")
            
            # Reasoning
            if reasoning:
                output.append(f"\n[bold white]💡 Reasoning:[/bold white]")
                output.append(f"[dim white]{reasoning[:200]}{'...' if len(reasoning) > 200 else ''}[/dim white]")
            
            # Data sources
            if data_sources:
                output.append(f"\n[dim]Data Sources: {', '.join(data_sources[:5])}[/dim]")
            
            output.append(f"\n[dim]Powered by Ultimate Decision Engine (28+ data points)[/dim]")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.warning(f"Fast decision failed for {symbol}: {e}")
            return f"[dim]Decision engine unavailable for {symbol}. Error: {str(e)[:50]}[/dim]"

    def execute_plan(
        self, planning_live, planning_content, item, prompt
    ):
        """Execute agent action with progressive timeout messages"""
        result = None
        start_time = time.time()

        # Flag to track if the action is complete
        action_complete = threading.Event()
        def run_action():
            nonlocal result
            try:
                result = self.agent.action(prompt, item["title"], item["description"])
            except Exception as e:
                result = str(e)
            finally:
                action_complete.set()

        # Start the action in a separate thread
        action_thread = threading.Thread(target=run_action)
        action_thread.start()

        # Update the display while waiting
        while not action_complete.is_set():
            elapsed_time = time.time() - start_time
            elapsed_seconds = int(elapsed_time)
            timeout_message = get_timeout_message(elapsed_time)
            
            # Add elapsed time in brackets to the timeout message
            timeout_message_with_time = f"{timeout_message} ({elapsed_seconds}s)"

            # Update the last item in planning_content with current timeout message
            planning_content[-1] = timeout_message_with_time
            planning_live.update(
                Panel("\n".join(planning_content), title="Planning", style="magenta")
            )

            # Wait 1 second before checking again to update the timer
            time.sleep(1.0)

        # Wait for the thread to complete
        action_thread.join()

        return result

    def process_prompt(self, prompt: str, conversation: list) -> str:
        # Handle commands using helpers
        if handle_command(prompt, conversation, self.agent, console):
            return ""
        
        # Check for API keys - either OpenAI or Anthropic works now
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            console.print("[red]⚠ No API key found. Please set either:[/red]")
            console.print("[dim white]  • ANTHROPIC_API_KEY for Claude (recommended)[/dim white]")
            console.print("[dim white]  • OPENAI_API_KEY for OpenAI[/dim white]")
            console.print()
            console.print("[dim white]Add to .env file: ANTHROPIC_API_KEY=sk-ant-...[/dim white]")
            console.print()
            exit()

        # =====================================================================
        # FAST PATH: Route simple queries without LLM planning
        # =====================================================================
        if self.fast_path_enabled and self._router:
            routed = self._router.route(prompt)
            
            if routed.fast_path and routed.confidence >= 0.8:
                console.print(f"[dim]⚡ Fast path: {routed.intent.value}[/dim]")
                
                fast_response = self._handle_fast_path(routed)
                if fast_response:
                    console.print(fast_response)
                    console.print()
                    console.print("[dim]powered by [/dim][bright_cyan]NUBLE[/bright_cyan]", justify="right")
                    
                    # Add to conversation for context
                    conversation.append({"role": "user", "content": prompt})
                    conversation.append({"role": "assistant", "content": fast_response})
                    
                    return fast_response

        # =====================================================================
        # FULL PATH: Complex queries need LLM planning
        # =====================================================================

        # Show initial planning spinner
        console.print()
        plan_spinner = Spinner(
            "dots", text="[bright_magenta]Planning...[/bright_magenta]"
        )
        with Live(plan_spinner, console=console, refresh_per_second=10):
            pass  # Initial planning display

        # Planning pane content that streams live
        planning_content = []
        with Live(console=console, refresh_per_second=10) as planning_live:
            while True:
                # Get plan from the agent
                plan = self.agent.run(conversation)
                if len(plan) == 0:
                    break

                # Add to conversation
                conversation.append({"role": "assistant", "content": str(plan)})

                # Process each plan item
                for item in plan:
                    # Add description and update planning pane immediately
                    planning_content.append(
                        f"[bright_green]●[/bright_green] [white]{item['description']}[/white]"
                    )
                    planning_live.update(
                        Panel(
                            "\n".join(planning_content),
                            title="Planning",
                            style="magenta",
                        )
                    )

                    # Add initial spinner message
                    planning_content.append("[yellow] Retrieving data... (0s)[/yellow]")
                    planning_live.update(
                        Panel(
                            "\n".join(planning_content),
                            title="Planning",
                            style="magenta",
                        )
                    )

                    # Execute with progressive timeout display
                    result = self.execute_plan(
                        planning_live, planning_content, item, prompt
                    )

                    if "[red]⚠" in result:
                        # Clear the planning pane and show only the error message
                        planning_live.stop()
                        console.print(result)
                        console.print()
                        console.print(f"[dim white]Please check your API keys and try again[/dim white]", justify="right")
                        return ""

                    # Add to conversation
                    conversation.append(
                        {
                            "role": "user",
                            "content": f"{item['title']} - {item['description']}",
                        }
                    )
                    conversation.append({"role": "user", "content": str(result), "type": "data"})

                    # Get summary and add to conversation
                    summary = self.agent.summarize(conversation)
                    conversation.append({"role": "user", "content": str(summary)})

                    # Replace spinner with summary and add new line
                    planning_content[-1] = (
                        f"[white]└─[/white] [bright_black]{summary}[/bright_black]"
                    )
                    planning_content.append("")  # Add empty line for spacing
                    planning_live.update(
                        Panel(
                            "\n".join(planning_content),
                            title="Planning",
                            style="magenta",
                        )
                    )

        # Stream the answer as markdown in answer pane
        answer_text = ""
        with Live(console=console, refresh_per_second=10) as live:
            for chunk in self.agent.answer(prompt, conversation):
                answer_text += chunk
                markdown_answer = Markdown(answer_text)
                live.update(
                    Panel(markdown_answer, title="Answer", border_style="bright_cyan")
                )
        conversation.append({"role": "assistant", "content": answer_text})

        # build footer with usage and token info
        tokens = self.token_counter.count_conversation_tokens(conversation)

        # remove large amounts of raw data to reduce token usage
        conversation = [item for item in conversation if "type" not in item or item["type"] != "data"]
        
        # Get Claude API usage from token tracker
        tracker = self.agent.token_tracker
        usage_info = f"[dim white]Claude API: [/dim white][bright_green]{tracker.total_tokens:,}[/bright_green][dim white] tokens[/dim white] | "
        cost_info = f"[dim white]~[/dim white][yellow]{tracker.get_cost_estimate():.4f}[/yellow][dim white]$[/dim white] | "
        
        console.print(f"{usage_info}{cost_info}[dim white]Requests: [/dim white][white]{tracker.total_requests}[/white] | [dim white]powered by [/dim white][bright_cyan]NUBLE[/bright_cyan]", justify="right")

        return answer_text 

