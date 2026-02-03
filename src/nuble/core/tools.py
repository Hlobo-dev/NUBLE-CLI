"""
NUBLE Tool Registry
===================

Provides Claude-compatible tool definitions for agentic execution.
Each tool can be called by the LLM and executed with real data.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time_ms: int = 0
    cached: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'execution_time_ms': self.execution_time_ms,
            'cached': self.cached
        }


class ToolCategory(Enum):
    """Tool categories for organization."""
    MARKET_DATA = "market_data"
    TECHNICAL = "technical_analysis"
    ML_PREDICTION = "ml_prediction"
    SEC_FILINGS = "sec_filings"
    NEWS_SENTIMENT = "news_sentiment"
    PORTFOLIO = "portfolio"
    RISK = "risk_management"


@dataclass
class Tool:
    """
    A tool that can be called by the LLM.
    
    Compatible with Claude's tool_use format.
    """
    name: str
    description: str
    category: ToolCategory
    input_schema: Dict[str, Any]
    handler: Callable[..., ToolResult]
    requires_api_key: Optional[str] = None
    cache_ttl_seconds: int = 60
    
    def to_claude_format(self) -> Dict:
        """Convert to Claude's tool definition format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        start = datetime.now()
        try:
            # Check API key if required
            if self.requires_api_key:
                if not os.getenv(self.requires_api_key):
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Missing API key: {self.requires_api_key}"
                    )
            
            result = await self.handler(**kwargs) if hasattr(self.handler, '__await__') else self.handler(**kwargs)
            execution_time = int((datetime.now() - start).total_seconds() * 1000)
            
            if isinstance(result, ToolResult):
                result.execution_time_ms = execution_time
                return result
            
            return ToolResult(
                success=True,
                data=result,
                execution_time_ms=execution_time
            )
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time_ms=int((datetime.now() - start).total_seconds() * 1000)
            )


class ToolRegistry:
    """
    Registry of all available tools.
    
    Provides:
    - Tool registration and lookup
    - Claude-compatible tool definitions
    - Execution with caching
    """
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._cache: Dict[str, tuple] = {}  # (timestamp, result)
        self._register_default_tools()
    
    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_all(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_all_tools(self) -> List[Dict]:
        """Get all tools as dictionaries (alias for get_claude_tools)."""
        return self.get_claude_tools()
    
    def get_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get tools by category."""
        return [t for t in self._tools.values() if t.category == category]
    
    def get_claude_tools(self) -> List[Dict]:
        """Get all tools in Claude's format."""
        return [t.to_claude_format() for t in self._tools.values()]
    
    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}"
            )
        
        # Check cache
        cache_key = f"{tool_name}:{json.dumps(kwargs, sort_keys=True, default=str)}"
        if cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            age = (datetime.now() - cached_time).total_seconds()
            if age < tool.cache_ttl_seconds:
                cached_result.cached = True
                return cached_result
        
        # Execute
        result = await tool.execute(**kwargs)
        
        # Cache successful results
        if result.success:
            self._cache[cache_key] = (datetime.now(), result)
        
        return result
    
    def _register_default_tools(self):
        """Register all default NUBLE tools."""
        # Import handlers
        from .tool_handlers import (
            get_stock_quote,
            get_technical_indicators,
            run_ml_prediction,
            search_sec_filings,
            get_news_sentiment,
            analyze_risk,
            get_options_flow,
            get_market_regime,
            compare_stocks,
        )
        
        # Market Data Tools
        self.register(Tool(
            name="get_stock_quote",
            description="Get real-time stock quote including price, volume, and change. Use this for current price information.",
            category=ToolCategory.MARKET_DATA,
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, TSLA, MSFT)"
                    }
                },
                "required": ["symbol"]
            },
            handler=get_stock_quote,
            requires_api_key="POLYGON_API_KEY",
            cache_ttl_seconds=30
        ))
        
        # Technical Analysis Tools
        self.register(Tool(
            name="get_technical_indicators",
            description="Get technical indicators for a stock including RSI, MACD, Bollinger Bands, SMA, EMA, ATR. Use for technical analysis.",
            category=ToolCategory.TECHNICAL,
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of indicators: rsi, macd, bollinger, sma, ema, atr, stochastic"
                    },
                    "period": {
                        "type": "integer",
                        "description": "Lookback period (default 14)"
                    }
                },
                "required": ["symbol"]
            },
            handler=get_technical_indicators,
            requires_api_key="POLYGON_API_KEY",
            cache_ttl_seconds=60
        ))
        
        # ML Prediction Tools
        self.register(Tool(
            name="run_ml_prediction",
            description="Run ML prediction models to forecast stock price direction and targets. Uses ensemble of LSTM, Transformer, and N-BEATS models.",
            category=ToolCategory.ML_PREDICTION,
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["ensemble", "lstm", "transformer", "all"],
                        "description": "Which model to use (default: ensemble)"
                    },
                    "horizon": {
                        "type": "string",
                        "enum": ["1d", "5d", "10d", "20d"],
                        "description": "Prediction horizon"
                    }
                },
                "required": ["symbol"]
            },
            handler=run_ml_prediction,
            requires_api_key="POLYGON_API_KEY",
            cache_ttl_seconds=300
        ))
        
        # SEC Filings Tools
        self.register(Tool(
            name="search_sec_filings",
            description="Search SEC filings (10-K, 10-Q, 8-K) for specific information about a company. Use for fundamental research, risk factors, financial data.",
            category=ToolCategory.SEC_FILINGS,
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "query": {
                        "type": "string",
                        "description": "What to search for (e.g., 'risk factors', 'revenue breakdown', 'competition')"
                    },
                    "filing_type": {
                        "type": "string",
                        "enum": ["10-K", "10-Q", "8-K", "all"],
                        "description": "Type of filing to search"
                    }
                },
                "required": ["symbol", "query"]
            },
            handler=search_sec_filings,
            cache_ttl_seconds=3600
        ))
        
        # News & Sentiment Tools
        self.register(Tool(
            name="get_news_sentiment",
            description="Get recent news and sentiment analysis for a stock. Includes news headlines, sentiment scores, and analyst ratings.",
            category=ToolCategory.NEWS_SENTIMENT,
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look back (default 7)"
                    }
                },
                "required": ["symbol"]
            },
            handler=get_news_sentiment,
            cache_ttl_seconds=300
        ))
        
        # Risk Analysis Tools
        self.register(Tool(
            name="analyze_risk",
            description="Analyze risk metrics for a stock or portfolio. Includes VaR, volatility, beta, max drawdown.",
            category=ToolCategory.RISK,
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "position_size": {
                        "type": "number",
                        "description": "Position size in dollars for risk calculation"
                    }
                },
                "required": ["symbol"]
            },
            handler=analyze_risk,
            requires_api_key="POLYGON_API_KEY",
            cache_ttl_seconds=300
        ))
        
        # Options Flow
        self.register(Tool(
            name="get_options_flow",
            description="Get unusual options activity and flow data for a stock. Shows large trades, unusual volume, put/call ratio.",
            category=ToolCategory.MARKET_DATA,
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    }
                },
                "required": ["symbol"]
            },
            handler=get_options_flow,
            requires_api_key="POLYGON_API_KEY",
            cache_ttl_seconds=300
        ))
        
        # Market Regime
        self.register(Tool(
            name="get_market_regime",
            description="Detect current market regime (bull, bear, volatile, ranging) using HMM and volatility analysis.",
            category=ToolCategory.TECHNICAL,
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock or index symbol (default SPY for market)"
                    }
                },
                "required": []
            },
            handler=get_market_regime,
            requires_api_key="POLYGON_API_KEY",
            cache_ttl_seconds=3600
        ))
        
        # Comparison Tool
        self.register(Tool(
            name="compare_stocks",
            description="Compare two or more stocks across multiple metrics: price performance, technicals, fundamentals, sentiment.",
            category=ToolCategory.MARKET_DATA,
            input_schema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock symbols to compare"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metrics to compare: performance, technicals, valuation, sentiment"
                    }
                },
                "required": ["symbols"]
            },
            handler=compare_stocks,
            requires_api_key="POLYGON_API_KEY",
            cache_ttl_seconds=300
        ))


def get_all_tools() -> ToolRegistry:
    """Get a fully initialized tool registry."""
    return ToolRegistry()
