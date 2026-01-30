"""
Claude Opus 4.5 Powered Synthesizer
====================================

Uses Anthropic's Claude Opus 4.5 (the most advanced model) for:
- Natural language query understanding
- Financial data synthesis and analysis
- Research report generation
- Trading recommendations
- Risk assessment narratives
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class ClaudeModel(Enum):
    """Available Claude models - Updated January 2026"""
    # Latest and most capable
    OPUS_4_5 = "claude-opus-4-5-20251101"     # Most capable, best for complex analysis
    SONNET_4_5 = "claude-sonnet-4-5-20250929" # Excellent balance of capability/cost
    HAIKU_4_5 = "claude-haiku-4-5-20251001"   # Fast and efficient
    
    # Previous generation
    OPUS_4_1 = "claude-opus-4-1-20250805"     # Opus 4.1
    OPUS_4 = "claude-opus-4-20250514"         # Opus 4
    SONNET_4 = "claude-sonnet-4-20250514"     # Sonnet 4
    HAIKU_3_5 = "claude-3-5-haiku-20241022"   # Haiku 3.5
    HAIKU_3 = "claude-3-haiku-20240307"       # Haiku 3 (legacy)


@dataclass
class ClaudeResponse:
    """Response from Claude API"""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    latency_ms: float = 0.0


@dataclass
class AnalysisContext:
    """Context for financial analysis"""
    symbol: str
    query: str
    market_data: Dict[str, Any] = field(default_factory=dict)
    technical_data: Dict[str, Any] = field(default_factory=dict)
    sentiment_data: Dict[str, Any] = field(default_factory=dict)
    news_data: List[Dict[str, Any]] = field(default_factory=list)
    fundamentals: Dict[str, Any] = field(default_factory=dict)


class ClaudeSynthesizer:
    """
    Claude Opus 4.5 powered synthesizer for institutional-grade analysis.
    
    Features:
    - Advanced reasoning with Claude Opus 4.5
    - Financial domain expertise prompts
    - Structured output generation
    - Multi-turn analysis conversations
    """
    
    # System prompt for financial analysis
    FINANCIAL_ANALYST_PROMPT = """You are an elite institutional financial analyst with expertise in:
- Technical analysis (chart patterns, indicators, price action)
- Fundamental analysis (valuation, financial statements, ratios)
- Quantitative analysis (statistical models, risk metrics)
- Market microstructure (order flow, liquidity, market making)
- Options and derivatives (Greeks, volatility, strategies)
- Macroeconomics (Fed policy, economic indicators, global markets)

Your analysis style:
1. Be concise but comprehensive
2. Use specific numbers and data points
3. Provide actionable insights
4. Quantify risks and opportunities
5. Consider multiple scenarios
6. Acknowledge uncertainty when appropriate

Format your responses with clear sections using markdown.
Always include:
- Key takeaways at the top
- Specific price levels and targets
- Risk factors and stop-loss considerations
- Confidence level (low/medium/high) with reasoning
"""

    TRADING_SIGNAL_PROMPT = """You are a professional quantitative trader generating trading signals.
Analyze the provided data and generate a clear trading recommendation.

Your output must include:
1. SIGNAL: BUY / SELL / HOLD
2. CONFIDENCE: 0-100%
3. ENTRY: Specific price or range
4. STOP LOSS: Risk management level
5. TARGETS: T1, T2, T3 price targets
6. TIMEFRAME: Expected holding period
7. RISK/REWARD: Calculated ratio
8. KEY FACTORS: Bulleted list of reasons

Be specific with numbers. No vague language.
"""

    RESEARCH_REPORT_PROMPT = """You are a senior equity research analyst at a top-tier investment bank.
Generate an institutional-quality research report.

Structure your report as:
1. EXECUTIVE SUMMARY (2-3 sentences)
2. INVESTMENT THESIS
3. KEY METRICS (table format)
4. VALUATION ANALYSIS
5. RISK FACTORS
6. CATALYSTS (upcoming events that could move the stock)
7. RECOMMENDATION with price target

Write in a professional, institutional tone.
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: ClaudeModel = ClaudeModel.OPUS_4_5,
        max_tokens: int = 4096,
    ):
        """
        Initialize Claude Synthesizer.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use (default: Opus 4.5)
            max_tokens: Maximum response tokens
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        
        # Conversation history for multi-turn analysis
        self._conversation_history: List[Dict[str, str]] = []
    
    def _call_claude(
        self,
        user_message: str,
        system_prompt: str = None,
        temperature: float = 0.7,
    ) -> ClaudeResponse:
        """
        Call Claude API.
        
        Args:
            user_message: User's message/query
            system_prompt: System prompt for context
            temperature: Response randomness (0-1)
            
        Returns:
            ClaudeResponse with content and metadata
        """
        import time
        start = time.time()
        
        system = system_prompt or self.FINANCIAL_ANALYST_PROMPT
        
        response = self.client.messages.create(
            model=self.model.value,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
            temperature=temperature,
        )
        
        latency = (time.time() - start) * 1000
        
        return ClaudeResponse(
            content=response.content[0].text,
            model=self.model.value,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            latency_ms=latency,
        )
    
    def analyze(self, context: AnalysisContext) -> ClaudeResponse:
        """
        Perform comprehensive financial analysis.
        
        Args:
            context: AnalysisContext with all relevant data
            
        Returns:
            ClaudeResponse with analysis
        """
        # Build context message
        message = self._build_analysis_message(context)
        return self._call_claude(message, self.FINANCIAL_ANALYST_PROMPT)
    
    def generate_trading_signal(self, context: AnalysisContext) -> ClaudeResponse:
        """
        Generate trading signal with specific recommendations.
        
        Args:
            context: AnalysisContext with market data
            
        Returns:
            ClaudeResponse with trading signal
        """
        message = self._build_analysis_message(context)
        message += "\n\nGenerate a specific trading signal based on this data."
        return self._call_claude(message, self.TRADING_SIGNAL_PROMPT, temperature=0.3)
    
    def generate_research_report(self, context: AnalysisContext) -> ClaudeResponse:
        """
        Generate institutional research report.
        
        Args:
            context: AnalysisContext with comprehensive data
            
        Returns:
            ClaudeResponse with research report
        """
        message = self._build_analysis_message(context)
        message += "\n\nGenerate a comprehensive institutional research report."
        return self._call_claude(message, self.RESEARCH_REPORT_PROMPT, temperature=0.5)
    
    def answer_query(self, query: str, data: Dict[str, Any] = None) -> ClaudeResponse:
        """
        Answer a natural language financial query.
        
        Args:
            query: User's question
            data: Optional supporting data
            
        Returns:
            ClaudeResponse with answer
        """
        message = query
        if data:
            message += f"\n\nRelevant data:\n```json\n{self._format_data(data)}\n```"
        
        return self._call_claude(message)
    
    def synthesize_data(
        self,
        symbol: str,
        data_sources: Dict[str, Any],
        query: str = None,
    ) -> ClaudeResponse:
        """
        Synthesize data from multiple sources into cohesive analysis.
        
        Args:
            symbol: Stock symbol
            data_sources: Dict of data from various providers
            query: Optional specific question to answer
            
        Returns:
            ClaudeResponse with synthesized analysis
        """
        message = f"# Analysis Request for {symbol}\n\n"
        
        if query:
            message += f"**User Query:** {query}\n\n"
        
        message += "## Available Data\n\n"
        
        for source, data in data_sources.items():
            message += f"### {source}\n```json\n{self._format_data(data)}\n```\n\n"
        
        message += """
Please synthesize this data into a comprehensive analysis that:
1. Identifies key patterns and trends
2. Highlights important anomalies or signals
3. Provides actionable insights
4. Quantifies risks and opportunities
"""
        
        return self._call_claude(message)
    
    def explain_indicator(self, indicator: str, value: float, context: str = "") -> str:
        """
        Get Claude's interpretation of a technical indicator.
        
        Args:
            indicator: Name of the indicator (RSI, MACD, etc.)
            value: Current value
            context: Additional context
            
        Returns:
            Interpretation string
        """
        message = f"""
Interpret this technical indicator reading:
- Indicator: {indicator}
- Value: {value}
{f'- Context: {context}' if context else ''}

Provide a brief (2-3 sentences) interpretation of what this means for trading.
"""
        response = self._call_claude(message, temperature=0.3)
        return response.content
    
    def _build_analysis_message(self, context: AnalysisContext) -> str:
        """Build a comprehensive analysis message from context"""
        message = f"# Financial Analysis Request\n\n"
        message += f"**Symbol:** {context.symbol}\n"
        message += f"**Query:** {context.query}\n"
        message += f"**Timestamp:** {datetime.now().isoformat()}\n\n"
        
        if context.market_data:
            message += "## Market Data\n"
            message += f"```json\n{self._format_data(context.market_data)}\n```\n\n"
        
        if context.technical_data:
            message += "## Technical Indicators\n"
            message += f"```json\n{self._format_data(context.technical_data)}\n```\n\n"
        
        if context.sentiment_data:
            message += "## Sentiment Analysis\n"
            message += f"```json\n{self._format_data(context.sentiment_data)}\n```\n\n"
        
        if context.news_data:
            message += "## Recent News\n"
            for news in context.news_data[:5]:
                message += f"- {news.get('headline', news.get('title', 'N/A'))}\n"
            message += "\n"
        
        if context.fundamentals:
            message += "## Fundamentals\n"
            message += f"```json\n{self._format_data(context.fundamentals)}\n```\n\n"
        
        return message
    
    def _format_data(self, data: Any, indent: int = 2) -> str:
        """Format data as JSON string"""
        import json
        
        def default_serializer(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return str(obj)
        
        try:
            return json.dumps(data, indent=indent, default=default_serializer)
        except:
            return str(data)
    
    def set_model(self, model: ClaudeModel):
        """Switch Claude model"""
        self.model = model
    
    def use_opus(self):
        """Use Opus 4.5 (most capable)"""
        self.model = ClaudeModel.OPUS_4_5
    
    def use_sonnet(self):
        """Use Sonnet 4.5 (balanced)"""
        self.model = ClaudeModel.SONNET_4_5
    
    def use_haiku(self):
        """Use Haiku 4.5 (fastest)"""
        self.model = ClaudeModel.HAIKU_4_5


# Convenience function
def create_claude_synthesizer(
    model: str = "opus",
    api_key: str = None,
) -> ClaudeSynthesizer:
    """
    Create a Claude synthesizer with specified model.
    
    Args:
        model: "opus", "sonnet", or "haiku"
        api_key: Optional API key
        
    Returns:
        ClaudeSynthesizer instance
    """
    model_map = {
        "opus": ClaudeModel.OPUS_4_5,
        "sonnet": ClaudeModel.SONNET_4_5,
        "haiku": ClaudeModel.HAIKU_4_5,
    }
    return ClaudeSynthesizer(
        api_key=api_key,
        model=model_map.get(model.lower(), ClaudeModel.OPUS_4_5),
    )
