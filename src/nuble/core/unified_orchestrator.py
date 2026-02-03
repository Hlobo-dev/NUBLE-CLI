"""
NUBLE Unified Orchestrator
==========================
 
This replaces the fragmented architecture with a unified, optimal design.
"""

import os
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_anthropic = None
def _get_anthropic():
    global _anthropic
    if _anthropic is None:
        try:
            import anthropic
            _anthropic = anthropic
        except ImportError:
            pass
    return _anthropic


class ExecutionPath(Enum):
    """Execution path for a query."""
    FAST = "fast"           # No LLM needed, direct data
    DECISION = "decision"   # Use UltimateDecisionEngine
    RESEARCH = "research"   # Multi-agent research
    EDUCATION = "education" # Explanation/teaching


@dataclass
class OrchestratorConfig:
    """Configuration for the unified orchestrator."""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    enable_tools: bool = True
    enable_memory: bool = True
    enable_decision_engine: bool = True
    enable_ml: bool = True
    max_tool_calls: int = 5
    timeout_seconds: int = 60


@dataclass
class QueryResult:
    """Result from query processing."""
    success: bool
    message: str
    path: ExecutionPath
    data: Dict[str, Any] = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)
    confidence: float = 0.0
    execution_time_ms: int = 0
    conversation_id: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'message': self.message,
            'path': self.path.value,
            'data': self.data,
            'tools_used': self.tools_used,
            'confidence': self.confidence,
            'execution_time_ms': self.execution_time_ms
        }


class UnifiedOrchestrator:
    """
    The unified brain of NUBLE.
    
    Architecture:
    
    User Query
        │
        ▼
    ┌───────────────────────────────────────┐
    │          QUERY ROUTER                 │
    │  (Pattern matching + intent detection) │
    └───────────────────────────────────────┘
        │
        ├── FAST PATH (no LLM)
        │   └── Direct tool execution
        │       - get_stock_quote
        │       - get_technical_indicators
        │       - get_news_sentiment
        │
        ├── DECISION PATH
        │   └── UltimateDecisionEngine
        │       - 28+ data points
        │       - Weighted layer scoring
        │       - Risk veto system
        │       - Trade setup generation
        │
        ├── RESEARCH PATH (with LLM)
        │   └── Claude + Tools
        │       - Multi-step reasoning
        │       - Tool execution
        │       - SEC filings search
        │       - Synthesis
        │
        └── EDUCATION PATH
            └── Claude explanation
                - Concept teaching
                - Examples
                - Strategy education
    
    Example:
        orchestrator = UnifiedOrchestrator()
        result = await orchestrator.process("Should I buy TSLA?")
        print(result.message)
    """
    
    # Fast path patterns (no LLM needed)
    FAST_PATTERNS = {
        'quote': [
            r'^(\$?[A-Z]{1,5})\s*$',
            r'^(price|quote)\s+(of|for)?\s*(\$?[A-Z]{1,5})',
            r'^(what\'?s?|show)\s+(\$?[A-Z]{1,5})\s*(trading|price|at)?',
        ],
        'technical': [
            r'(rsi|macd|bollinger|technicals?|indicators?)\s+(for|of)?\s*(\$?[A-Z]{1,5})',
            r'(\$?[A-Z]{1,5})\s+(rsi|macd|technicals?)',
            r'(is|show)\s+(\$?[A-Z]{1,5})\s+(overbought|oversold)',
        ],
        'sentiment': [
            r'(sentiment|news)\s+(for|of|on)?\s*(\$?[A-Z]{1,5})',
            r'(\$?[A-Z]{1,5})\s+(sentiment|news)',
        ],
    }
    
    # Decision path patterns (use UltimateDecisionEngine)
    DECISION_PATTERNS = [
        r'(should i|do you recommend).*(buy|sell|hold)',
        r'(predict|prediction|forecast|signal)\s+(for|of)?\s*\$?[A-Z]{1,5}',
        r'(entry|stop.?loss|target).*(for|of|on)\s*\$?[A-Z]{1,5}',
        r'(trade setup|trading signal)\s+(for|of)?\s*\$?[A-Z]{1,5}',
        r'(ml|machine learning|model)\s+(for|prediction)',
    ]
    
    # Education patterns
    EDUCATION_PATTERNS = [
        r'^(what is|explain|how does|teach me|tell me about)',
        r'(define|meaning of|definition)',
        r'(strategy|how to trade|when should i)',
    ]
    
    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()
        
        # Claude client
        self.client = None
        anthropic = _get_anthropic()
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic and api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        
        # Components (lazy loaded)
        self._tools = None
        self._memory = None
        self._decision_engine = None
        self._ml_predictor = None
        
        # Compile patterns
        self._compiled_patterns = {}
        self._compile_patterns()
        
        logger.info(f"UnifiedOrchestrator initialized with model: {self.config.model}")
    
    @property
    def tools(self):
        """Lazy load tool registry."""
        if self._tools is None and self.config.enable_tools:
            try:
                from .tools import ToolRegistry
                self._tools = ToolRegistry()
                logger.info(f"Loaded {len(self._tools.get_all())} tools")
            except Exception as e:
                logger.warning(f"Failed to load tools: {e}")
        return self._tools
    
    @property
    def _tool_registry(self):
        """Alias for tools property (compatibility)."""
        return self.tools
    
    @property
    def memory(self):
        """Lazy load memory manager."""
        if self._memory is None and self.config.enable_memory:
            try:
                from .memory import MemoryManager
                self._memory = MemoryManager()
                logger.info("Memory manager loaded")
            except Exception as e:
                logger.warning(f"Failed to load memory: {e}")
        return self._memory
    
    @property
    def decision_engine(self):
        """Lazy load Ultimate Decision Engine."""
        if self._decision_engine is None and self.config.enable_decision_engine:
            try:
                from ..decision.ultimate_engine import UltimateDecisionEngine
                self._decision_engine = UltimateDecisionEngine()
                logger.info("Decision engine loaded")
            except Exception as e:
                logger.warning(f"Failed to load decision engine: {e}")
        return self._decision_engine
    
    @property
    def ml_predictor(self):
        """Lazy load ML predictor."""
        if self._ml_predictor is None and self.config.enable_ml:
            try:
                from ...institutional.ml import get_predictor
                self._ml_predictor = get_predictor()
                logger.info("ML predictor loaded")
            except Exception as e:
                logger.warning(f"Failed to load ML predictor: {e}")
        return self._ml_predictor
    
    def _compile_patterns(self):
        """Pre-compile regex patterns."""
        for category, patterns in self.FAST_PATTERNS.items():
            self._compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        self._compiled_patterns['decision'] = [
            re.compile(p, re.IGNORECASE) for p in self.DECISION_PATTERNS
        ]
        self._compiled_patterns['education'] = [
            re.compile(p, re.IGNORECASE) for p in self.EDUCATION_PATTERNS
        ]
    
    async def process(
        self,
        query: str,
        conversation_id: str = "default",
        user_context: Dict = None
    ) -> QueryResult:
        """
        Process a user query through the optimal path.
        
        This is the main entry point.
        
        Args:
            query: User's question or request
            conversation_id: Unique conversation identifier
            user_context: Optional context (portfolio, preferences)
            
        Returns:
            QueryResult with response and metadata
        """
        start_time = datetime.now()
        
        # Extract symbols from query
        symbols = self._extract_symbols(query)
        
        # Route query to appropriate path
        path, fast_category = self._route_query(query)
        
        logger.info(f"Processing: '{query[:50]}...' | Path: {path.value} | Symbols: {symbols}")
        
        # Add to conversation memory
        if self.memory:
            self.memory.conversations.add_message(conversation_id, 'user', query)
        
        try:
            if path == ExecutionPath.FAST:
                result = await self._handle_fast_path(query, symbols, fast_category)
            elif path == ExecutionPath.DECISION:
                result = await self._handle_decision_path(query, symbols, user_context)
            elif path == ExecutionPath.EDUCATION:
                result = await self._handle_education_path(query, conversation_id)
            else:
                result = await self._handle_research_path(query, symbols, conversation_id, user_context)
            
            # Calculate execution time
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            result.execution_time_ms = execution_time
            result.conversation_id = conversation_id
            
            # Add to memory
            if self.memory and result.success:
                self.memory.conversations.add_message(
                    conversation_id, 'assistant', result.message
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return QueryResult(
                success=False,
                message=f"I encountered an error: {str(e)}",
                path=path,
                execution_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
    
    def _route_query(self, query: str) -> Tuple[ExecutionPath, Optional[str]]:
        """Route query to the optimal execution path."""
        query_lower = query.lower().strip()
        
        # Check fast path patterns
        for category, patterns in self._compiled_patterns.items():
            if category in ('decision', 'education'):
                continue
            for pattern in patterns:
                if pattern.search(query_lower):
                    return ExecutionPath.FAST, category
        
        # Check decision patterns
        for pattern in self._compiled_patterns.get('decision', []):
            if pattern.search(query_lower):
                return ExecutionPath.DECISION, None
        
        # Check education patterns
        for pattern in self._compiled_patterns.get('education', []):
            if pattern.search(query_lower):
                return ExecutionPath.EDUCATION, None
        
        # Default to research path
        return ExecutionPath.RESEARCH, None
    
    async def _handle_fast_path(
        self,
        query: str,
        symbols: List[str],
        category: str
    ) -> QueryResult:
        """Handle fast path queries (no LLM needed)."""
        if not symbols:
            return QueryResult(
                success=False,
                message="I couldn't identify a stock symbol in your query. Try something like 'AAPL price' or 'TSLA technicals'.",
                path=ExecutionPath.FAST
            )
        
        symbol = symbols[0]
        tools_used = []
        data = {}
        
        try:
            if category == 'quote':
                result = await self.tools.execute('get_stock_quote', symbol=symbol)
                tools_used.append('get_stock_quote')
                
                if result.success:
                    data = result.data
                    quote = result.data
                    message = self._format_quote_response(quote)
                else:
                    message = f"Could not get quote for {symbol}: {result.error}"
            
            elif category == 'technical':
                result = await self.tools.execute('get_technical_indicators', symbol=symbol)
                tools_used.append('get_technical_indicators')
                
                if result.success:
                    data = result.data
                    message = self._format_technical_response(result.data)
                else:
                    message = f"Could not get technicals for {symbol}: {result.error}"
            
            elif category == 'sentiment':
                result = await self.tools.execute('get_news_sentiment', symbol=symbol)
                tools_used.append('get_news_sentiment')
                
                if result.success:
                    data = result.data
                    message = self._format_sentiment_response(result.data)
                else:
                    message = f"Could not get sentiment for {symbol}: {result.error}"
            
            else:
                # Default to quote
                result = await self.tools.execute('get_stock_quote', symbol=symbol)
                tools_used.append('get_stock_quote')
                
                if result.success:
                    data = result.data
                    message = self._format_quote_response(result.data)
                else:
                    message = f"Could not get data for {symbol}"
            
            return QueryResult(
                success=True,
                message=message,
                path=ExecutionPath.FAST,
                data=data,
                tools_used=tools_used,
                confidence=0.95
            )
            
        except Exception as e:
            logger.error(f"Fast path failed: {e}")
            return QueryResult(
                success=False,
                message=f"Error fetching data: {str(e)}",
                path=ExecutionPath.FAST
            )
    
    async def _handle_decision_path(
        self,
        query: str,
        symbols: List[str],
        user_context: Dict = None
    ) -> QueryResult:
        """Handle decision path using UltimateDecisionEngine."""
        if not symbols:
            return QueryResult(
                success=False,
                message="I need a stock symbol to provide a trading decision. Try 'Should I buy TSLA?'",
                path=ExecutionPath.DECISION
            )
        
        symbol = symbols[0]
        tools_used = ['decision_engine']
        
        # Try to use the full decision engine
        if self.decision_engine:
            try:
                await self.decision_engine.initialize()
                decision = await self.decision_engine.make_decision(symbol)
                
                message = self._format_decision_response(decision)
                
                # Track prediction
                if self.memory:
                    self.memory.predictions.record_prediction(
                        symbol=symbol,
                        prediction_type='direction',
                        predicted_value=decision.direction.value,
                        confidence=decision.confidence,
                        model='ultimate_engine'
                    )
                
                return QueryResult(
                    success=True,
                    message=message,
                    path=ExecutionPath.DECISION,
                    data=decision.to_dict(),
                    tools_used=tools_used,
                    confidence=decision.confidence
                )
                
            except Exception as e:
                logger.warning(f"Decision engine failed: {e}")
        
        # Fallback: use tools to build decision
        tools_used = []
        data = {}
        
        # Get quote
        quote_result = await self.tools.execute('get_stock_quote', symbol=symbol)
        if quote_result.success:
            data['quote'] = quote_result.data
            tools_used.append('get_stock_quote')
        
        # Get technicals
        tech_result = await self.tools.execute('get_technical_indicators', symbol=symbol)
        if tech_result.success:
            data['technicals'] = tech_result.data
            tools_used.append('get_technical_indicators')
        
        # Get ML prediction
        ml_result = await self.tools.execute('run_ml_prediction', symbol=symbol)
        if ml_result.success:
            data['prediction'] = ml_result.data
            tools_used.append('run_ml_prediction')
        
        # Get sentiment
        sentiment_result = await self.tools.execute('get_news_sentiment', symbol=symbol)
        if sentiment_result.success:
            data['sentiment'] = sentiment_result.data
            tools_used.append('get_news_sentiment')
        
        # Get risk
        risk_result = await self.tools.execute('analyze_risk', symbol=symbol)
        if risk_result.success:
            data['risk'] = risk_result.data
            tools_used.append('analyze_risk')
        
        # Synthesize decision with Claude
        if self.client:
            message = await self._synthesize_decision(query, symbol, data, user_context)
        else:
            message = self._format_fallback_decision(symbol, data)
        
        # Calculate confidence from tools
        confidence = 0.7 if data.get('prediction') else 0.5
        
        return QueryResult(
            success=True,
            message=message,
            path=ExecutionPath.DECISION,
            data=data,
            tools_used=tools_used,
            confidence=confidence
        )
    
    async def _handle_research_path(
        self,
        query: str,
        symbols: List[str],
        conversation_id: str,
        user_context: Dict = None
    ) -> QueryResult:
        """Handle research path with Claude and tools."""
        if not self.client:
            return QueryResult(
                success=False,
                message="Claude AI is not available. Please set ANTHROPIC_API_KEY.",
                path=ExecutionPath.RESEARCH
            )
        
        # Prepare context
        context = self._build_context(conversation_id, user_context)
        
        # Build system prompt
        system_prompt = self._build_system_prompt(symbols)
        
        # Get Claude tools
        claude_tools = self.tools.get_claude_tools() if self.tools else []
        
        # Build messages
        messages = context.get('messages', [])
        if not messages or messages[-1].get('content') != query:
            messages.append({'role': 'user', 'content': query})
        
        tools_used = []
        collected_data = {}
        
        try:
            # Call Claude with tools
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                messages=messages,
                tools=claude_tools if claude_tools else None
            )
            
            # Process response with tool calls
            final_response = await self._process_tool_calls(
                response, messages, claude_tools, tools_used, collected_data
            )
            
            return QueryResult(
                success=True,
                message=final_response,
                path=ExecutionPath.RESEARCH,
                data=collected_data,
                tools_used=tools_used,
                confidence=0.85
            )
            
        except Exception as e:
            logger.error(f"Research path failed: {e}")
            return QueryResult(
                success=False,
                message=f"Research failed: {str(e)}",
                path=ExecutionPath.RESEARCH
            )
    
    async def _handle_education_path(
        self,
        query: str,
        conversation_id: str
    ) -> QueryResult:
        """Handle education/explanation queries."""
        if not self.client:
            return QueryResult(
                success=False,
                message="Claude AI is not available for explanations.",
                path=ExecutionPath.EDUCATION
            )
        
        system_prompt = """You are NUBLE, an expert financial educator.
        
Your role is to explain financial concepts clearly and accurately.
Use examples, analogies, and practical applications.
Keep explanations accessible but don't oversimplify.
If relevant, mention how NUBLE can help with the topic.

Guidelines:
- Use bullet points for clarity
- Include real-world examples
- Mention risks and caveats
- Suggest follow-up questions
"""
        
        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=2000,
                system=system_prompt,
                messages=[{'role': 'user', 'content': query}]
            )
            
            message = response.content[0].text
            
            return QueryResult(
                success=True,
                message=message,
                path=ExecutionPath.EDUCATION,
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"Education path failed: {e}")
            return QueryResult(
                success=False,
                message=f"Could not generate explanation: {str(e)}",
                path=ExecutionPath.EDUCATION
            )
    
    async def _process_tool_calls(
        self,
        response,
        messages: List[Dict],
        tools: List[Dict],
        tools_used: List[str],
        collected_data: Dict
    ) -> str:
        """Process Claude's tool calls iteratively."""
        max_iterations = self.config.max_tool_calls
        iteration = 0
        
        while iteration < max_iterations:
            # Check if response has tool use
            has_tool_use = any(
                block.type == 'tool_use' 
                for block in response.content
            )
            
            if not has_tool_use:
                # No more tool calls, extract text
                for block in response.content:
                    if hasattr(block, 'text'):
                        return block.text
                return "Analysis complete."
            
            # Process tool calls
            tool_results = []
            
            for block in response.content:
                if block.type == 'tool_use':
                    tool_name = block.name
                    tool_input = block.input
                    tool_id = block.id
                    
                    logger.info(f"Executing tool: {tool_name}")
                    tools_used.append(tool_name)
                    
                    # Execute tool
                    result = await self.tools.execute(tool_name, **tool_input)
                    
                    # Store data
                    collected_data[tool_name] = result.data if result.success else result.error
                    
                    tool_results.append({
                        'type': 'tool_result',
                        'tool_use_id': tool_id,
                        'content': json.dumps(result.data if result.success else {'error': result.error})
                    })
            
            # Add assistant message with tool use
            messages.append({'role': 'assistant', 'content': response.content})
            
            # Add tool results
            messages.append({'role': 'user', 'content': tool_results})
            
            # Get next response
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                messages=messages,
                tools=tools
            )
            
            iteration += 1
        
        # Max iterations reached
        for block in response.content:
            if hasattr(block, 'text'):
                return block.text
        
        return "Analysis complete (max tool calls reached)."
    
    async def _synthesize_decision(
        self,
        query: str,
        symbol: str,
        data: Dict,
        user_context: Dict = None
    ) -> str:
        """Use Claude to synthesize a trading decision from collected data."""
        
        prompt = f"""You are NUBLE, an institutional-grade trading advisor.

Based on the following data, provide a clear trading decision for {symbol}.

USER QUERY: {query}

COLLECTED DATA:
{json.dumps(data, indent=2, default=str)}

USER CONTEXT:
{json.dumps(user_context or {}, indent=2)}

Provide:
1. CLEAR RECOMMENDATION (BUY/SELL/HOLD with strength)
2. KEY REASONS (top 3 factors)
3. TRADE SETUP (entry, stop-loss, targets if applicable)
4. RISKS TO WATCH
5. CONFIDENCE LEVEL

Be direct and actionable. Use bullet points for clarity.
"""

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=2000,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Decision synthesis failed: {e}")
            return self._format_fallback_decision(symbol, data)
    
    def _build_context(self, conversation_id: str, user_context: Dict = None) -> Dict:
        """Build context for Claude."""
        context = {}
        
        if self.memory:
            context = self.memory.conversations.get_context(conversation_id)
            prefs = self.memory.get_user_preferences()
            context['preferences'] = prefs.to_dict()
        
        if user_context:
            context['user_context'] = user_context
        
        return context
    
    def _build_system_prompt(self, symbols: List[str] = None) -> str:
        """Build system prompt for Claude."""
        
        tools_desc = ""
        if self.tools:
            tools_desc = "\n".join([
                f"- {t.name}: {t.description}"
                for t in self.tools.get_all()
            ])
        
        return f"""You are NUBLE, the world's most advanced AI financial advisor.

You have access to real-time market data, ML predictions, technical analysis, SEC filings, and more.

AVAILABLE TOOLS:
{tools_desc}

GUIDELINES:
1. Use tools to get real data - don't make up numbers
2. Be specific with prices, percentages, and targets
3. Consider risk always - include stop-losses with buy recommendations
4. Be direct - give clear recommendations when data supports it
5. Cite your sources - mention which tools/data you used
6. Be honest about uncertainty

{"Current symbols in focus: " + ", ".join(symbols) if symbols else ""}

Format responses in clean Markdown. Use bullet points for key insights.
Include specific numbers from the data.
"""
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text."""
        patterns = [
            r'\$([A-Z]{1,5})\b',
            r'\b([A-Z]{2,5})\b',
        ]
        
        symbols = set()
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            symbols.update(matches)
        
        # Filter common words
        common = {'I', 'A', 'THE', 'AND', 'OR', 'FOR', 'TO', 'IN', 'IS', 'IT', 'MY', 
                  'BE', 'ARE', 'DO', 'OF', 'BUY', 'SELL', 'HOLD', 'GET', 'SET', 'ALL'}
        symbols -= common
        
        # Known valid symbols
        valid = {'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',
                 'AMD', 'INTC', 'CRM', 'ORCL', 'IBM', 'NFLX', 'DIS', 'JPM', 'BAC',
                 'SPY', 'QQQ', 'IWM', 'VTI', 'BTC', 'ETH', 'GLD', 'SLV', 'COIN'}
        
        matched = symbols & valid
        return list(matched) if matched else list(symbols)[:5]
    
    # Response formatters
    def _format_quote_response(self, quote: Dict) -> str:
        """Format quote data into readable response."""
        symbol = quote.get('symbol', 'Unknown')
        name = quote.get('company_name', symbol)
        price = quote.get('price', 0)
        change = quote.get('change', 0)
        change_pct = quote.get('change_percent', 0)
        volume = quote.get('volume', 0)
        
        direction = "🟢" if change >= 0 else "🔴"
        
        return f"""## {symbol} - {name}

**Price:** ${price:,.2f} {direction} {change:+.2f} ({change_pct:+.2f}%)

| Metric | Value |
|--------|-------|
| Open | ${quote.get('open', 0):,.2f} |
| High | ${quote.get('high', 0):,.2f} |
| Low | ${quote.get('low', 0):,.2f} |
| Volume | {volume:,.0f} |
| VWAP | ${quote.get('vwap', 0):,.2f} |

*Updated: {quote.get('timestamp', 'now')}*
"""
    
    def _format_technical_response(self, data: Dict) -> str:
        """Format technical indicators into readable response."""
        symbol = data.get('symbol', 'Unknown')
        price = data.get('current_price', 0)
        indicators = data.get('indicators', {})
        signal = data.get('overall_signal', {})
        
        # Build indicator table
        rows = []
        for name, ind in indicators.items():
            if name == 'rsi':
                rows.append(f"| RSI | {ind['value']:.1f} | {ind['signal']} |")
            elif name == 'macd':
                rows.append(f"| MACD | {ind['macd']:.4f} | {'Bullish' if ind['bullish'] else 'Bearish'} |")
            elif name == 'bollinger':
                rows.append(f"| Bollinger | {ind['position']:.2f} | {ind['signal']} |")
            elif name == 'sma':
                rows.append(f"| SMA Trend | - | {ind['trend']} |")
            elif name == 'stochastic':
                rows.append(f"| Stochastic | {ind['k']:.1f} | {ind['signal']} |")
        
        indicator_table = "\n".join(rows) if rows else "| - | - | - |"
        
        # Overall signal
        direction = signal.get('direction', 'neutral')
        strength = signal.get('strength', 0)
        emoji = "🟢" if direction == 'bullish' else "🔴" if direction == 'bearish' else "⚪"
        
        return f"""## {symbol} Technical Analysis

**Price:** ${price:,.2f}

### Indicators
| Indicator | Value | Signal |
|-----------|-------|--------|
{indicator_table}

### Overall Signal
{emoji} **{direction.upper()}** (strength: {strength:.0%})

- Bullish signals: {signal.get('bullish_count', 0)}
- Bearish signals: {signal.get('bearish_count', 0)}
"""
    
    def _format_sentiment_response(self, data: Dict) -> str:
        """Format sentiment data into readable response."""
        symbol = data.get('symbol', 'Unknown')
        sentiment = data.get('sentiment', {})
        
        score = sentiment.get('score', 0)
        label = sentiment.get('label', 'neutral')
        direction = sentiment.get('direction', 'neutral')
        
        emoji = "🟢" if direction == 'positive' else "🔴" if direction == 'negative' else "⚪"
        
        return f"""## {symbol} Sentiment Analysis

{emoji} **{label.upper()}** (score: {score:.2f})

| Metric | Value |
|--------|-------|
| Sentiment Direction | {direction} |
| News Count | {data.get('news_count', 'N/A')} |
| Analyst Rating | {data.get('analyst_rating', 'N/A')} |
| Price Target | {data.get('price_target', 'N/A')} |
| Action Signal | {data.get('action', 'N/A')} |
"""
    
    def _format_decision_response(self, decision) -> str:
        """Format UltimateDecision into readable response."""
        direction_emoji = {
            'BUY': '🟢',
            'SELL': '🔴',
            'NEUTRAL': '⚪'
        }
        
        strength_emoji = {
            'STRONG': '💪',
            'MODERATE': '👍',
            'WEAK': '👌',
            'NO_TRADE': '⛔'
        }
        
        d = decision
        emoji = direction_emoji.get(d.direction.value, '⚪')
        strength = strength_emoji.get(d.strength.value, '')
        
        return f"""## {d.symbol} Trading Decision

{emoji} **{d.direction.value}** {strength} {d.strength.value}

**Confidence:** {d.confidence:.0%}
**Should Trade:** {'✅ Yes' if d.should_trade else '❌ No'}

### Layer Scores
| Layer | Score | Confidence | Weight |
|-------|-------|------------|--------|
| Technical | {d.technical_score.score:.2f} | {d.technical_score.confidence:.0%} | {d.technical_score.weight:.0%} |
| Intelligence | {d.intelligence_score.score:.2f} | {d.intelligence_score.confidence:.0%} | {d.intelligence_score.weight:.0%} |
| Market Structure | {d.market_structure_score.score:.2f} | {d.market_structure_score.confidence:.0%} | {d.market_structure_score.weight:.0%} |
| Validation | {d.validation_score.score:.2f} | {d.validation_score.confidence:.0%} | {d.validation_score.weight:.0%} |

### Trade Setup
- **Entry:** ${d.trade_setup.entry:.2f}
- **Stop Loss:** ${d.trade_setup.stop_loss:.2f} ({d.trade_setup.stop_pct:.1%})
- **Targets:** {', '.join([f'${t:.2f}' for t in d.trade_setup.targets])}
- **Risk/Reward:** {d.trade_setup.risk_reward:.1f}x

### Reasoning
{chr(10).join(['- ' + r for r in d.reasoning[:5]])}

{"### ⚠️ VETO: " + d.veto_reason if d.veto else ""}
"""
    
    def _format_fallback_decision(self, symbol: str, data: Dict) -> str:
        """Format fallback decision when Claude is not available."""
        quote = data.get('quote', {})
        technicals = data.get('technicals', {})
        prediction = data.get('prediction', {})
        sentiment = data.get('sentiment', {})
        risk = data.get('risk', {})
        
        price = quote.get('price', 0)
        direction = prediction.get('prediction', {}).get('direction', 
                   technicals.get('overall_signal', {}).get('direction', 'neutral'))
        
        return f"""## {symbol} Analysis Summary

**Current Price:** ${price:,.2f}
**Signal:** {direction.upper()}

### Technical Overview
{json.dumps(technicals.get('indicators', {}), indent=2)}

### ML Prediction
{json.dumps(prediction.get('prediction', {}), indent=2)}

### Risk Metrics
{json.dumps(risk.get('metrics', {}), indent=2)}

*Note: Full AI synthesis unavailable. Raw data shown above.*
"""


# Convenience functions
async def process_query(query: str, conversation_id: str = "default") -> QueryResult:
    """Process a query using the default orchestrator."""
    orchestrator = UnifiedOrchestrator()
    return await orchestrator.process(query, conversation_id)
