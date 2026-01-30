"""
Intent Engine - LLM-powered query understanding and classification.
Analyzes user queries to determine what data sources and analytics are needed.
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import openai

from ..config import get_config


class QueryCategory(Enum):
    """High-level query categories"""
    PRICE_QUOTE = "price_quote"
    TECHNICAL_ANALYSIS = "technical_analysis"
    OPTIONS = "options"
    FUNDAMENTALS = "fundamentals"
    NEWS_SENTIMENT = "news_sentiment"
    INSTITUTIONAL = "institutional"
    INSIDER = "insider"
    REGULATORY = "regulatory"
    ECONOMIC = "economic"
    COMPARISON = "comparison"
    SCREENING = "screening"
    CRYPTO = "crypto"
    FOREX = "forex"
    GENERAL = "general"


class DataNeed(Enum):
    """Specific data requirements"""
    # Price Data
    REALTIME_QUOTE = "realtime_quote"
    HISTORICAL_PRICES = "historical_prices"
    INTRADAY_BARS = "intraday_bars"
    TICK_DATA = "tick_data"
    
    # Options
    OPTIONS_CHAIN = "options_chain"
    OPTIONS_FLOW = "options_flow"
    IMPLIED_VOLATILITY = "implied_volatility"
    GREEKS = "greeks"
    UNUSUAL_ACTIVITY = "unusual_activity"
    
    # Technical
    TECHNICAL_INDICATORS = "technical_indicators"
    CHART_PATTERNS = "chart_patterns"
    SUPPORT_RESISTANCE = "support_resistance"
    VOLUME_ANALYSIS = "volume_analysis"
    
    # Fundamentals
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    KEY_RATIOS = "key_ratios"
    EARNINGS = "earnings"
    DIVIDENDS = "dividends"
    
    # Institutional
    INSTITUTIONAL_HOLDINGS = "institutional_holdings"
    FUND_HOLDINGS = "fund_holdings"
    POSITION_CHANGES = "position_changes"
    
    # Insider
    INSIDER_TRADES = "insider_trades"
    INSIDER_SENTIMENT = "insider_sentiment"
    CONGRESSIONAL_TRADES = "congressional_trades"
    
    # Regulatory
    SEC_FILINGS = "sec_filings"
    FORM_10K = "form_10k"
    FORM_10Q = "form_10q"
    FORM_8K = "form_8k"
    FORM_13F = "form_13f"
    FORM_4 = "form_4"
    
    # News & Sentiment
    NEWS = "news"
    SENTIMENT_SCORES = "sentiment_scores"
    SOCIAL_SENTIMENT = "social_sentiment"
    PRESS_RELEASES = "press_releases"
    
    # Company Info
    COMPANY_PROFILE = "company_profile"
    PEERS = "peers"
    SUPPLY_CHAIN = "supply_chain"
    
    # Economic
    ECONOMIC_CALENDAR = "economic_calendar"
    MACRO_INDICATORS = "macro_indicators"
    TREASURY_RATES = "treasury_rates"


@dataclass
class ParsedIntent:
    """Parsed intent from user query"""
    original_query: str
    category: QueryCategory
    data_needs: List[DataNeed]
    symbols: List[str]
    timeframe: Optional[str] = None
    date_range: Optional[Tuple[str, str]] = None
    comparison_mode: bool = False
    screening_criteria: Optional[Dict] = None
    confidence: float = 1.0
    
    # Additional context extracted
    specific_indicators: List[str] = field(default_factory=list)
    specific_patterns: List[str] = field(default_factory=list)
    filing_types: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


# Alias for backward compatibility
QueryIntent = ParsedIntent


class IntentEngine:
    """
    LLM-powered intent classification engine.
    Combines rule-based pattern matching with LLM understanding for robust intent detection.
    """
    
    # Rule-based patterns for fast classification
    CATEGORY_PATTERNS = {
        QueryCategory.PRICE_QUOTE: [
            r'\b(price|quote|trading at|current|last|bid|ask|spread)\b',
            r'\b(premarket|pre-market|afterhours|after-hours|extended)\b',
        ],
        QueryCategory.TECHNICAL_ANALYSIS: [
            r'\b(rsi|macd|moving average|sma|ema|bollinger|stochastic)\b',
            r'\b(technical|chart|pattern|support|resistance|trend)\b',
            r'\b(oversold|overbought|crossover|divergence)\b',
            r'\b(cup and handle|head and shoulders|double top|double bottom)\b',
            r'\b(inside day|outside day|doji|hammer|engulfing)\b',
        ],
        QueryCategory.OPTIONS: [
            r'\b(options?|calls?|puts?|strike|expir|chain)\b',
            r'\b(iv|implied volatility|greeks?|delta|gamma|theta|vega)\b',
            r'\b(unusual|flow|sweep|block|dark pool)\b',
            r'\b(gex|gamma exposure|max pain|oi|open interest)\b',
        ],
        QueryCategory.FUNDAMENTALS: [
            r'\b(earnings?|eps|revenue|profit|margin|growth)\b',
            r'\b(balance sheet|income statement|cash flow|financials?)\b',
            r'\b(p/e|pe ratio|p/b|p/s|valuation|multiple)\b',
            r'\b(dividend|payout|yield)\b',
            r'\b(guidance|forecast|estimate)\b',
        ],
        QueryCategory.NEWS_SENTIMENT: [
            r'\b(news|headline|breaking|announcement)\b',
            r'\b(sentiment|bullish|bearish|mood|feeling)\b',
            r'\b(social|twitter|reddit|wsb|stocktwits)\b',
        ],
        QueryCategory.INSTITUTIONAL: [
            r'\b(13f|institutional|hedge fund|whale|smart money)\b',
            r'\b(berkshire|blackrock|vanguard|citadel|bridgewater)\b',
            r'\b(accumulating|distribution|position)\b',
        ],
        QueryCategory.INSIDER: [
            r'\b(insider|executive|ceo|cfo|director|officer)\b',
            r'\b(form 4|buying|selling|transaction)\b',
            r'\b(congress|senator|representative|politician)\b',
        ],
        QueryCategory.REGULATORY: [
            r'\b(sec|filing|10-?k|10-?q|8-?k|proxy)\b',
            r'\b(annual report|quarterly report|disclosure)\b',
        ],
        QueryCategory.ECONOMIC: [
            r'\b(fed|fomc|interest rate|treasury|bond)\b',
            r'\b(gdp|inflation|cpi|ppi|employment|jobs)\b',
            r'\b(economic|macro|recession)\b',
        ],
        QueryCategory.COMPARISON: [
            r'\b(compare|versus|vs\.?|against|better than)\b',
            r'\b(which is|difference between)\b',
        ],
        QueryCategory.SCREENING: [
            r'\b(find|screen|filter|scan|search for)\b',
            r'\b(stocks? with|companies? with|tickers? with)\b',
            r'\b(top|best|worst|highest|lowest)\b',
        ],
        QueryCategory.CRYPTO: [
            r'\b(bitcoin|btc|ethereum|eth|crypto|blockchain)\b',
            r'\b(defi|nft|altcoin|binance|coinbase)\b',
        ],
        QueryCategory.FOREX: [
            r'\b(forex|fx|currency|eur/?usd|gbp/?usd)\b',
            r'\b(dollar|euro|yen|pound)\b',
        ],
    }
    
    # Symbol extraction patterns
    SYMBOL_PATTERNS = [
        r'\$([A-Z]{1,5})\b',  # $AAPL format
        r'\b([A-Z]{1,5})\b(?=\s+(?:stock|share|price|option|call|put))',  # AAPL stock
        r'\bfor\s+([A-Z]{1,5})\b',  # for AAPL
        r'\b([A-Z]{2,5})\b(?=\s*\?)',  # AAPL?
    ]
    
    # Common stock symbols to validate against
    COMMON_SYMBOLS = {
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
        'BRK.A', 'BRK.B', 'V', 'JNJ', 'WMT', 'JPM', 'MA', 'PG', 'UNH',
        'DIS', 'HD', 'PYPL', 'BAC', 'VZ', 'ADBE', 'NFLX', 'CRM', 'CMCSA',
        'KO', 'PFE', 'TMO', 'ABT', 'PEP', 'COST', 'AVGO', 'NKE', 'MRK',
        'CSCO', 'ACN', 'LLY', 'DHR', 'CVX', 'XOM', 'TXN', 'MDT', 'NEE',
        'WFC', 'BMY', 'AMT', 'QCOM', 'HON', 'UNP', 'LOW', 'UPS', 'INTC',
        'AMD', 'COIN', 'PLTR', 'RIVN', 'LCID', 'SOFI', 'HOOD', 'GME', 'AMC',
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'ARKK', 'XLF', 'XLE',
        # Crypto
        'BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'ADA', 'DOT', 'MATIC',
    }
    
    # Data need inference rules
    DATA_NEED_RULES = {
        QueryCategory.PRICE_QUOTE: [DataNeed.REALTIME_QUOTE],
        QueryCategory.TECHNICAL_ANALYSIS: [
            DataNeed.HISTORICAL_PRICES, 
            DataNeed.TECHNICAL_INDICATORS
        ],
        QueryCategory.OPTIONS: [
            DataNeed.OPTIONS_CHAIN, 
            DataNeed.IMPLIED_VOLATILITY,
            DataNeed.GREEKS
        ],
        QueryCategory.FUNDAMENTALS: [
            DataNeed.KEY_RATIOS,
            DataNeed.EARNINGS,
            DataNeed.INCOME_STATEMENT
        ],
        QueryCategory.NEWS_SENTIMENT: [
            DataNeed.NEWS,
            DataNeed.SENTIMENT_SCORES
        ],
        QueryCategory.INSTITUTIONAL: [
            DataNeed.INSTITUTIONAL_HOLDINGS,
            DataNeed.FORM_13F,
            DataNeed.POSITION_CHANGES
        ],
        QueryCategory.INSIDER: [
            DataNeed.INSIDER_TRADES,
            DataNeed.FORM_4,
            DataNeed.INSIDER_SENTIMENT
        ],
        QueryCategory.REGULATORY: [
            DataNeed.SEC_FILINGS
        ],
    }
    
    # LLM prompt for complex intent parsing
    INTENT_PROMPT = """You are an expert financial query parser. Analyze the user's query and extract structured intent.

User Query: {query}

Respond with a JSON object containing:
{{
    "category": "one of: price_quote, technical_analysis, options, fundamentals, news_sentiment, institutional, insider, regulatory, economic, comparison, screening, crypto, forex, general",
    "symbols": ["list", "of", "ticker", "symbols"],
    "data_needs": ["list of specific data types needed"],
    "timeframe": "intraday/daily/weekly/monthly/yearly or null",
    "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} or null,
    "comparison_mode": true/false,
    "specific_indicators": ["RSI", "MACD", etc if mentioned],
    "specific_patterns": ["cup and handle", etc if mentioned],
    "filing_types": ["10-K", "13F", etc if mentioned],
    "keywords": ["key", "terms", "extracted"],
    "confidence": 0.0-1.0
}}

Be precise with symbol extraction. Common patterns:
- $AAPL or AAPL mentioned with stock/share/price context
- Company names should be converted to tickers
- Crypto symbols like BTC, ETH

For data_needs, choose from:
realtime_quote, historical_prices, intraday_bars, tick_data,
options_chain, options_flow, implied_volatility, greeks, unusual_activity,
technical_indicators, chart_patterns, support_resistance, volume_analysis,
income_statement, balance_sheet, cash_flow, key_ratios, earnings, dividends,
institutional_holdings, fund_holdings, position_changes,
insider_trades, insider_sentiment, congressional_trades,
sec_filings, form_10k, form_10q, form_8k, form_13f, form_4,
news, sentiment_scores, social_sentiment, press_releases,
company_profile, peers, supply_chain,
economic_calendar, macro_indicators, treasury_rates

Only return valid JSON, no explanation."""

    def __init__(self):
        self.config = get_config()
        self._init_llm_client()
    
    def _init_llm_client(self):
        """Initialize LLM client"""
        if self.config.llm.api_key:
            self.llm_client = openai.OpenAI(api_key=self.config.llm.api_key)
        else:
            self.llm_client = None
    
    def parse(self, query: str) -> ParsedIntent:
        """
        Parse user query into structured intent.
        Uses hybrid approach: rule-based for speed, LLM for complex queries.
        """
        query_lower = query.lower().strip()
        
        # Step 1: Quick rule-based classification
        category = self._classify_category(query_lower)
        symbols = self._extract_symbols(query)
        
        # Step 2: Determine if LLM parsing is needed
        needs_llm = (
            category == QueryCategory.GENERAL or
            len(symbols) == 0 or
            self._is_complex_query(query_lower)
        )
        
        if needs_llm and self.llm_client:
            return self._parse_with_llm(query, category, symbols)
        else:
            return self._parse_with_rules(query, category, symbols)
    
    def _classify_category(self, query_lower: str) -> QueryCategory:
        """Rule-based category classification"""
        scores = {}
        
        for category, patterns in self.CATEGORY_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                score += len(matches)
            scores[category] = score
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return QueryCategory.GENERAL
    
    def _extract_symbols(self, query: str) -> List[str]:
        """Extract ticker symbols from query"""
        symbols = set()
        
        # Pattern-based extraction
        for pattern in self.SYMBOL_PATTERNS:
            matches = re.findall(pattern, query.upper())
            symbols.update(matches)
        
        # Also check for known symbols in the query
        words = re.findall(r'\b[A-Z]{1,5}\b', query.upper())
        for word in words:
            if word in self.COMMON_SYMBOLS:
                symbols.add(word)
        
        # Filter out common words that might look like symbols
        noise_words = {'A', 'I', 'AN', 'THE', 'FOR', 'AND', 'OR', 'TO', 'IN', 'ON', 'AT', 'IS', 'IT'}
        symbols = symbols - noise_words
        
        return list(symbols)
    
    def _is_complex_query(self, query_lower: str) -> bool:
        """Determine if query needs LLM parsing"""
        complexity_indicators = [
            len(query_lower.split()) > 10,
            '?' in query_lower,
            'compare' in query_lower,
            'versus' in query_lower or ' vs ' in query_lower,
            'find' in query_lower or 'screen' in query_lower,
            query_lower.count(' and ') > 1,
            query_lower.count(' or ') > 0,
        ]
        return sum(complexity_indicators) >= 2
    
    def _parse_with_rules(
        self, 
        query: str, 
        category: QueryCategory, 
        symbols: List[str]
    ) -> ParsedIntent:
        """Rule-based intent parsing"""
        
        # Determine data needs based on category
        data_needs = list(self.DATA_NEED_RULES.get(category, [DataNeed.NEWS]))
        
        # Add additional data needs based on query content
        query_lower = query.lower()
        
        if 'pattern' in query_lower:
            data_needs.append(DataNeed.CHART_PATTERNS)
        if 'support' in query_lower or 'resistance' in query_lower:
            data_needs.append(DataNeed.SUPPORT_RESISTANCE)
        if 'volume' in query_lower:
            data_needs.append(DataNeed.VOLUME_ANALYSIS)
        if 'unusual' in query_lower or 'flow' in query_lower:
            data_needs.append(DataNeed.UNUSUAL_ACTIVITY)
        if 'social' in query_lower or 'twitter' in query_lower or 'reddit' in query_lower:
            data_needs.append(DataNeed.SOCIAL_SENTIMENT)
        if 'congress' in query_lower or 'senator' in query_lower:
            data_needs.append(DataNeed.CONGRESSIONAL_TRADES)
        
        # Extract timeframe
        timeframe = self._extract_timeframe(query_lower)
        
        # Extract specific indicators
        indicators = self._extract_indicators(query_lower)
        
        # Extract patterns
        patterns = self._extract_patterns(query_lower)
        
        return ParsedIntent(
            original_query=query,
            category=category,
            data_needs=list(set(data_needs)),
            symbols=symbols,
            timeframe=timeframe,
            specific_indicators=indicators,
            specific_patterns=patterns,
            confidence=0.85
        )
    
    def _parse_with_llm(
        self, 
        query: str, 
        fallback_category: QueryCategory,
        fallback_symbols: List[str]
    ) -> ParsedIntent:
        """LLM-powered intent parsing for complex queries"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm.model,
                messages=[
                    {"role": "system", "content": "You are a financial query parser. Return only valid JSON."},
                    {"role": "user", "content": self.INTENT_PROMPT.format(query=query)}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Map string category to enum
            category_str = result.get("category", "general")
            try:
                category = QueryCategory(category_str)
            except ValueError:
                category = fallback_category
            
            # Map string data needs to enums
            data_needs = []
            for need_str in result.get("data_needs", []):
                try:
                    data_needs.append(DataNeed(need_str))
                except ValueError:
                    pass
            
            if not data_needs:
                data_needs = list(self.DATA_NEED_RULES.get(category, [DataNeed.NEWS]))
            
            # Extract date range
            date_range = None
            if result.get("date_range"):
                dr = result["date_range"]
                if dr.get("start") and dr.get("end"):
                    date_range = (dr["start"], dr["end"])
            
            return ParsedIntent(
                original_query=query,
                category=category,
                data_needs=data_needs,
                symbols=result.get("symbols", fallback_symbols),
                timeframe=result.get("timeframe"),
                date_range=date_range,
                comparison_mode=result.get("comparison_mode", False),
                specific_indicators=result.get("specific_indicators", []),
                specific_patterns=result.get("specific_patterns", []),
                filing_types=result.get("filing_types", []),
                keywords=result.get("keywords", []),
                confidence=result.get("confidence", 0.9)
            )
            
        except Exception as e:
            # Fallback to rule-based parsing
            print(f"LLM parsing failed: {e}, falling back to rules")
            return self._parse_with_rules(query, fallback_category, fallback_symbols)
    
    def _extract_timeframe(self, query_lower: str) -> Optional[str]:
        """Extract timeframe from query"""
        timeframe_patterns = {
            'intraday': r'\b(intraday|1\s*min|5\s*min|15\s*min|hourly|today)\b',
            'daily': r'\b(daily|day|1d)\b',
            'weekly': r'\b(weekly|week|1w)\b',
            'monthly': r'\b(monthly|month|1m)\b',
            'yearly': r'\b(yearly|year|annual|1y|ytd)\b',
            'quarterly': r'\b(quarterly|quarter|q[1-4])\b',
        }
        
        for tf, pattern in timeframe_patterns.items():
            if re.search(pattern, query_lower):
                return tf
        
        return None
    
    def _extract_indicators(self, query_lower: str) -> List[str]:
        """Extract specific technical indicators mentioned"""
        indicators = []
        indicator_patterns = {
            'RSI': r'\brsi\b',
            'MACD': r'\bmacd\b',
            'SMA': r'\b(sma|simple moving average)\b',
            'EMA': r'\b(ema|exponential moving average)\b',
            'Bollinger Bands': r'\bbollinger\b',
            'Stochastic': r'\bstochastic\b',
            'ADX': r'\badx\b',
            'ATR': r'\batr\b',
            'VWAP': r'\bvwap\b',
            'OBV': r'\bobv\b',
            'CCI': r'\bcci\b',
            'Williams %R': r'\bwilliams\b',
            'Ichimoku': r'\bichimoku\b',
            'Fibonacci': r'\b(fib|fibonacci)\b',
        }
        
        for name, pattern in indicator_patterns.items():
            if re.search(pattern, query_lower):
                indicators.append(name)
        
        return indicators
    
    def _extract_patterns(self, query_lower: str) -> List[str]:
        """Extract specific chart patterns mentioned"""
        patterns = []
        pattern_patterns = {
            'Cup and Handle': r'\bcup\s*(and|&)?\s*handle\b',
            'Head and Shoulders': r'\bhead\s*(and|&)?\s*shoulders\b',
            'Double Top': r'\bdouble\s*top\b',
            'Double Bottom': r'\bdouble\s*bottom\b',
            'Triple Top': r'\btriple\s*top\b',
            'Triple Bottom': r'\btriple\s*bottom\b',
            'Ascending Triangle': r'\bascending\s*triangle\b',
            'Descending Triangle': r'\bdescending\s*triangle\b',
            'Symmetrical Triangle': r'\bsymmetrical\s*triangle\b',
            'Bull Flag': r'\bbull\s*flag\b',
            'Bear Flag': r'\bbear\s*flag\b',
            'Rising Wedge': r'\brising\s*wedge\b',
            'Falling Wedge': r'\bfalling\s*wedge\b',
            'Inside Day': r'\binside\s*day\b',
            'Outside Day': r'\boutside\s*day\b',
            'Doji': r'\bdoji\b',
            'Hammer': r'\bhammer\b',
            'Engulfing': r'\bengulfing\b',
        }
        
        for name, pattern in pattern_patterns.items():
            if re.search(pattern, query_lower):
                patterns.append(name)
        
        return patterns
