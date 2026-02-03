"""
Smart Query Router
==================

Routes user queries to the appropriate service based on intent detection.
Avoids unnecessary LLM calls for simple queries.
"""

import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class QueryIntent(Enum):
    """Detected query intent."""
    # Direct data queries (no LLM needed)
    QUOTE = "quote"                   # "AAPL price", "What's TSLA at?"
    TECHNICAL = "technical"           # "RSI for AAPL", "technicals for TSLA"
    PREDICTION = "prediction"         # "predict AAPL", "ML forecast for TSLA"
    PATTERN = "pattern"               # "patterns for AAPL"
    SENTIMENT = "sentiment"           # "sentiment for AAPL"
    
    # Filings queries (may need LLM)
    FILINGS_SEARCH = "filings_search"  # "risk factors for AAPL"
    FILINGS_ANALYSIS = "filings_analysis"  # "analyze AAPL 10-K"
    
    # Complex queries (need full LLM planning)
    RESEARCH = "research"              # "Should I buy AAPL?"
    COMPARISON = "comparison"          # "Compare AAPL vs MSFT"
    GENERAL = "general"                # Anything else


@dataclass
class RoutedQuery:
    """Result of query routing."""
    intent: QueryIntent
    symbols: List[str]
    parameters: Dict[str, Any]
    confidence: float
    requires_llm: bool
    fast_path: bool  # Can be answered without LLM


class SmartRouter:
    """
    Routes queries to the most efficient handler.
    
    Fast path (no LLM):
    - Simple quotes: "AAPL price" -> get_quote
    - Technical: "RSI for TSLA" -> get_technical_indicators
    - Predictions: "predict AMD" -> get_prediction
    
    Medium path (targeted LLM):
    - Filings: "risk factors for AAPL" -> search + summarize
    
    Full LLM path:
    - Research: "Should I buy TSLA?" -> full agent planning
    """
    
    # Patterns for fast path detection
    QUOTE_PATTERNS = [
        r'^(\$?[A-Z]{1,5})\s*$',  # Just a ticker: "AAPL" or "$AAPL"
        r'^(\$?[A-Z]{1,5})\s+(price|quote|trading at|stock)\s*$',
        r'^(what\'?s?|show|get)\s+(\$?[A-Z]{1,5})\s*(trading at|price|quote)?',
        r'^(price|quote)\s+(for|of)?\s*(\$?[A-Z]{1,5})',
    ]
    
    TECHNICAL_PATTERNS = [
        r'(rsi|macd|bollinger|sma|ema|atr|stochastic|momentum|overbought|oversold).*(for|of)?\s*(\$?[A-Z]{1,5})',
        r'(\$?[A-Z]{1,5})\s+(rsi|macd|bollinger|technicals?|indicators?)',
        r'(technical|technicals|indicators?)\s+(for|of)?\s*(\$?[A-Z]{1,5})',
        r'(is|show)\s+(\$?[A-Z]{1,5})\s+(overbought|oversold|bullish|bearish)',
    ]
    
    PREDICTION_PATTERNS = [
        r'(predict|prediction|forecast|ml)\s+(for|of)?\s*(\$?[A-Z]{1,5})',
        r'(\$?[A-Z]{1,5})\s+(prediction|forecast|ml|model)',
        r'(what will|where will|should i buy)\s+(\$?[A-Z]{1,5})',
        r'(transformer|lstm|ensemble|nbeats)\s+(for|of|model)?\s*(\$?[A-Z]{1,5})',
    ]
    
    PATTERN_PATTERNS = [
        r'(patterns?|chart patterns?)\s+(for|of|in)?\s*(\$?[A-Z]{1,5})',
        r'(\$?[A-Z]{1,5})\s+(patterns?|chart patterns?)',
        r'(head and shoulders|double top|triangle|wedge|flag).*(for|in)?\s*(\$?[A-Z]{1,5})',
    ]
    
    SENTIMENT_PATTERNS = [
        r'(sentiment|mood|feeling|bullish or bearish)\s+(for|of|on)?\s*(\$?[A-Z]{1,5})',
        r'(\$?[A-Z]{1,5})\s+(sentiment|mood|news sentiment)',
        r'(what\'?s the sentiment|how do people feel)\s+(for|about|on)?\s*(\$?[A-Z]{1,5})',
    ]
    
    FILINGS_PATTERNS = [
        r'(risk factors?|risks?)\s+(for|of|in)?\s*(\$?[A-Z]{1,5})',
        r'(10-?k|10-?q|8-?k|13f|sec filings?|filings?)\s+(for|of)?\s*(\$?[A-Z]{1,5})',
        r'(\$?[A-Z]{1,5})\s+(10-?k|10-?q|filings?|sec|annual report)',
        r'(what did|what does)\s+(\$?[A-Z]{1,5})\s+(say|report|disclose)',
        r'(analyze|summarize|read)\s+(\$?[A-Z]{1,5})\s*(10-?k|10-?q|filing)?',
    ]
    
    COMPARISON_PATTERNS = [
        r'(compare|versus|vs\.?)\s+(\$?[A-Z]{1,5})\s+(and|vs\.?|versus|to|with)\s+(\$?[A-Z]{1,5})',
        r'(\$?[A-Z]{1,5})\s+(vs\.?|versus|or)\s+(\$?[A-Z]{1,5})',
        r'(which is better|difference between)\s+(\$?[A-Z]{1,5})\s+(and|vs\.?)\s+(\$?[A-Z]{1,5})',
    ]
    
    # Common tickers for fast recognition
    COMMON_TICKERS = {
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',
        'AMD', 'INTC', 'NFLX', 'DIS', 'JPM', 'BAC', 'WMT', 'V', 'MA',
        'HD', 'PG', 'JNJ', 'UNH', 'XOM', 'CVX', 'PFE', 'ABBV', 'MRK',
        'KO', 'PEP', 'COST', 'AVGO', 'ADBE', 'CRM', 'ORCL', 'CSCO',
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'GLD', 'SLV', 'XLK',
        'BTC', 'ETH', 'COIN', 'HOOD', 'PLTR', 'SOFI', 'RIVN', 'LCID',
    }
    
    def __init__(self):
        self._compiled_patterns = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for speed."""
        self._compiled_patterns['quote'] = [
            re.compile(p, re.IGNORECASE) for p in self.QUOTE_PATTERNS
        ]
        self._compiled_patterns['technical'] = [
            re.compile(p, re.IGNORECASE) for p in self.TECHNICAL_PATTERNS
        ]
        self._compiled_patterns['prediction'] = [
            re.compile(p, re.IGNORECASE) for p in self.PREDICTION_PATTERNS
        ]
        self._compiled_patterns['pattern'] = [
            re.compile(p, re.IGNORECASE) for p in self.PATTERN_PATTERNS
        ]
        self._compiled_patterns['sentiment'] = [
            re.compile(p, re.IGNORECASE) for p in self.SENTIMENT_PATTERNS
        ]
        self._compiled_patterns['filings'] = [
            re.compile(p, re.IGNORECASE) for p in self.FILINGS_PATTERNS
        ]
        self._compiled_patterns['comparison'] = [
            re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS
        ]
    
    def route(self, query: str) -> RoutedQuery:
        """
        Route a query to the appropriate handler.
        
        Returns RoutedQuery with intent, symbols, and routing decision.
        """
        query = query.strip()
        
        # Extract symbols first
        symbols = self._extract_symbols(query)
        
        # Try fast path patterns in order of specificity
        
        # 1. Prediction queries
        if self._matches_patterns(query, 'prediction'):
            return RoutedQuery(
                intent=QueryIntent.PREDICTION,
                symbols=symbols,
                parameters=self._extract_prediction_params(query),
                confidence=0.9,
                requires_llm=False,
                fast_path=True
            )
        
        # 2. Technical queries
        if self._matches_patterns(query, 'technical'):
            return RoutedQuery(
                intent=QueryIntent.TECHNICAL,
                symbols=symbols,
                parameters=self._extract_technical_params(query),
                confidence=0.9,
                requires_llm=False,
                fast_path=True
            )
        
        # 3. Pattern queries
        if self._matches_patterns(query, 'pattern'):
            return RoutedQuery(
                intent=QueryIntent.PATTERN,
                symbols=symbols,
                parameters={},
                confidence=0.9,
                requires_llm=False,
                fast_path=True
            )
        
        # 4. Sentiment queries
        if self._matches_patterns(query, 'sentiment'):
            return RoutedQuery(
                intent=QueryIntent.SENTIMENT,
                symbols=symbols,
                parameters={},
                confidence=0.9,
                requires_llm=False,
                fast_path=True
            )
        
        # 5. Filings queries
        if self._matches_patterns(query, 'filings'):
            analysis_type = self._extract_filings_type(query)
            return RoutedQuery(
                intent=QueryIntent.FILINGS_SEARCH if analysis_type == 'search' else QueryIntent.FILINGS_ANALYSIS,
                symbols=symbols,
                parameters={'analysis_type': analysis_type},
                confidence=0.85,
                requires_llm=analysis_type != 'search',  # Analysis needs LLM
                fast_path=analysis_type == 'search'
            )
        
        # 6. Comparison queries
        if self._matches_patterns(query, 'comparison'):
            return RoutedQuery(
                intent=QueryIntent.COMPARISON,
                symbols=symbols,
                parameters={},
                confidence=0.85,
                requires_llm=True,  # Comparisons need LLM synthesis
                fast_path=False
            )
        
        # 7. Simple quote (just a ticker)
        if self._matches_patterns(query, 'quote') or (len(symbols) == 1 and len(query.replace('$', '').strip()) <= 5):
            return RoutedQuery(
                intent=QueryIntent.QUOTE,
                symbols=symbols,
                parameters={},
                confidence=0.95,
                requires_llm=False,
                fast_path=True
            )
        
        # 8. If we have symbols but no clear intent, default to research
        if symbols:
            return RoutedQuery(
                intent=QueryIntent.RESEARCH,
                symbols=symbols,
                parameters={'query': query},
                confidence=0.6,
                requires_llm=True,
                fast_path=False
            )
        
        # 9. General query - full LLM
        return RoutedQuery(
            intent=QueryIntent.GENERAL,
            symbols=[],
            parameters={'query': query},
            confidence=0.5,
            requires_llm=True,
            fast_path=False
        )
    
    def _matches_patterns(self, query: str, pattern_type: str) -> bool:
        """Check if query matches any pattern of given type."""
        patterns = self._compiled_patterns.get(pattern_type, [])
        return any(p.search(query) for p in patterns)
    
    def _extract_symbols(self, query: str) -> List[str]:
        """Extract stock symbols from query."""
        symbols = []
        
        # Match $AAPL format
        dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', query.upper())
        symbols.extend(dollar_tickers)
        
        # Match known tickers as standalone words
        words = re.findall(r'\b([A-Z]{1,5})\b', query.upper())
        for word in words:
            if word in self.COMMON_TICKERS and word not in symbols:
                symbols.append(word)
        
        # Match tickers followed by context words
        context_matches = re.findall(
            r'\b([A-Z]{1,5})\s+(?:stock|price|shares?|quote|trading)',
            query.upper()
        )
        for match in context_matches:
            if match not in symbols:
                symbols.append(match)
        
        return symbols
    
    def _extract_technical_params(self, query: str) -> Dict[str, Any]:
        """Extract technical indicator parameters."""
        params = {'indicators': []}
        
        query_lower = query.lower()
        
        if 'rsi' in query_lower:
            params['indicators'].append('rsi')
        if 'macd' in query_lower:
            params['indicators'].append('macd')
        if 'bollinger' in query_lower:
            params['indicators'].append('bollinger')
        if 'sma' in query_lower or 'moving average' in query_lower:
            params['indicators'].append('sma')
        if 'ema' in query_lower:
            params['indicators'].append('ema')
        if 'atr' in query_lower:
            params['indicators'].append('atr')
        
        # If no specific indicator, return all
        if not params['indicators']:
            params['indicators'] = None  # Will calculate all
        
        return params
    
    def _extract_prediction_params(self, query: str) -> Dict[str, Any]:
        """Extract prediction parameters."""
        params = {'model_type': 'mlp'}  # Default
        
        query_lower = query.lower()
        
        if 'transformer' in query_lower or 'tft' in query_lower:
            params['model_type'] = 'transformer'
        elif 'lstm' in query_lower:
            params['model_type'] = 'lstm'
        elif 'ensemble' in query_lower:
            params['model_type'] = 'ensemble'
        elif 'nbeats' in query_lower:
            params['model_type'] = 'nbeats'
        
        return params
    
    def _extract_filings_type(self, query: str) -> str:
        """Determine what type of filings analysis is needed."""
        query_lower = query.lower()
        
        if 'risk' in query_lower:
            return 'risk_factors'
        elif 'competition' in query_lower or 'competitor' in query_lower:
            return 'competitive_landscape'
        elif 'revenue' in query_lower or 'segment' in query_lower:
            return 'segment_analysis'
        elif 'guidance' in query_lower or 'outlook' in query_lower:
            return 'guidance'
        elif 'analyze' in query_lower or 'summarize' in query_lower:
            return 'financial_highlights'
        else:
            return 'search'  # Default to search


# Global router instance
_router: Optional[SmartRouter] = None


def get_router() -> SmartRouter:
    """Get the global router instance."""
    global _router
    if _router is None:
        _router = SmartRouter()
    return _router
