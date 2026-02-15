"""
SEC Filings Agent
=================

Interactive AI agent for SEC filings research using Claude Opus 4.5.
Supports streaming responses and multi-turn conversations.
"""

import os
from typing import Optional, List, Dict, Any, Generator, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .search import FilingsSearch, SearchResult
from .loader import FilingsLoader, LoadedFiling
from .analyzer import FilingsAnalyzer, AnalysisType


@dataclass
class AgentMessage:
    """A message in the conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: List[str] = field(default_factory=list)
    tool_calls: List[str] = field(default_factory=list)


@dataclass
class AgentState:
    """Current state of the agent"""
    messages: List[AgentMessage] = field(default_factory=list)
    loaded_filings: Dict[str, LoadedFiling] = field(default_factory=dict)
    current_ticker: Optional[str] = None
    last_search_results: List[SearchResult] = field(default_factory=list)


class SECFilingsAgent:
    """
    Interactive SEC filings research agent powered by Claude Opus 4.5.
    
    Features:
    - Multi-turn conversations
    - Streaming responses
    - Tool calling (search, load, analyze)
    - Context management
    - Export capabilities
    """
    
    SYSTEM_PROMPT = """You are the SEC Filings Research Agent, an elite AI analyst specializing in 
SEC regulatory filings analysis for institutional investors.

You have access to the following capabilities:
1. SEARCH: Search across indexed SEC filings using semantic search
2. LOAD: Download specific filings (10-K, 10-Q, 8-K) from SEC EDGAR
3. ANALYZE: Perform deep analysis on filings (risk factors, financials, MD&A)
4. COMPARE: Compare filings across time periods or companies
5. ANSWER: Answer specific questions about filing content

When the user asks a question:
1. Determine which tools/data you need
2. If filing content is needed but not loaded, explain you'll need to load it
3. Provide thorough, specific answers with citations
4. Highlight key insights an institutional investor would care about

You speak with the authority of a senior hedge fund analyst with deep experience
in fundamental analysis and regulatory filings. Be direct, specific, and actionable.

Current loaded filings will be provided in context when available."""

    TOOL_DESCRIPTIONS = """
Available Tools:
- search(query, tickers, forms): Search filings for relevant content
- load_filing(ticker, form, year, quarter): Load a specific filing
- load_latest(ticker, form): Load the most recent filing
- analyze(analysis_type): Analyze loaded filing (risk_factors, financials, md&a)
- compare(filings): Compare multiple filings
- answer(question): Answer question about loaded content
"""
    
    def __init__(
        self,
        api_key: str = None,
        search: FilingsSearch = None,
        loader: FilingsLoader = None,
        analyzer: FilingsAnalyzer = None,
        model: str = "claude-opus-4-5-20251101",
    ):
        """
        Initialize the SEC Filings Agent.
        
        Args:
            api_key: Anthropic API key
            search: FilingsSearch instance
            loader: FilingsLoader instance
            analyzer: FilingsAnalyzer instance
            model: Claude model to use
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic not installed. Run: pip install anthropic")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        
        # Components
        self.search = search
        self.loader = loader or FilingsLoader()
        self.analyzer = analyzer or FilingsAnalyzer(api_key=self.api_key, search=search)
        
        # State
        self.state = AgentState()
    
    def chat(
        self,
        message: str,
        stream: bool = False,
    ) -> "Union[str, Generator[str, None, None]]":
        """
        Send a message and get a response.
        
        Args:
            message: User message
            stream: Whether to stream the response
            
        Returns:
            Response string or generator for streaming
        """
        # Add user message to history
        self.state.messages.append(AgentMessage(
            role="user",
            content=message,
        ))
        
        # Build context
        context = self._build_context()
        
        # Build messages for API
        api_messages = [
            {"role": m.role, "content": m.content}
            for m in self.state.messages[-20:]  # Last 20 messages
        ]
        
        if stream:
            return self._stream_response(context, api_messages)
        else:
            return self._get_response(context, api_messages)
    
    def _get_response(
        self,
        context: str,
        messages: List[Dict[str, str]],
    ) -> str:
        """Get a complete response"""
        system = f"{self.SYSTEM_PROMPT}\n\n{self.TOOL_DESCRIPTIONS}\n\n{context}"
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=messages,
        )
        
        content = response.content[0].text
        
        # Add assistant message to history
        self.state.messages.append(AgentMessage(
            role="assistant",
            content=content,
        ))
        
        # Check for tool invocations in response
        self._process_tool_calls(content)
        
        return content
    
    def _stream_response(
        self,
        context: str,
        messages: List[Dict[str, str]],
    ) -> Generator[str, None, None]:
        """Stream response chunks"""
        system = f"{self.SYSTEM_PROMPT}\n\n{self.TOOL_DESCRIPTIONS}\n\n{context}"
        
        full_response = ""
        
        with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                yield text
        
        # Add complete message to history
        self.state.messages.append(AgentMessage(
            role="assistant",
            content=full_response,
        ))
        
        # Process tool calls
        self._process_tool_calls(full_response)
    
    def _build_context(self) -> str:
        """Build context string for the agent"""
        parts = []
        
        # Loaded filings
        if self.state.loaded_filings:
            parts.append("LOADED FILINGS:")
            for key, filing in self.state.loaded_filings.items():
                parts.append(f"- {key}: {filing.word_count:,} words")
        
        # Current ticker
        if self.state.current_ticker:
            parts.append(f"\nCurrent ticker focus: {self.state.current_ticker}")
        
        # Last search results
        if self.state.last_search_results:
            parts.append(f"\nLast search returned {len(self.state.last_search_results)} results")
        
        # Indexed tickers if search available
        if self.search:
            tickers = self.search.indexed_tickers
            if tickers:
                parts.append(f"\nIndexed tickers: {', '.join(tickers[:20])}")
        
        return "\n".join(parts) if parts else "No filings loaded yet."
    
    def _process_tool_calls(self, response: str):
        """Process any tool calls in the response"""
        # This is a simplified version - in production you'd parse structured tool calls
        response_lower = response.lower()
        
        # Detect intent to load filings
        if "load" in response_lower and ("10-k" in response_lower or "10-q" in response_lower):
            # Log that loading may be needed
            pass
    
    def load_filing(
        self,
        ticker: str,
        form: str = "10-K",
        year: int = None,
        quarter: int = None,
    ) -> LoadedFiling:
        """
        Load a SEC filing.
        
        Args:
            ticker: Stock ticker
            form: Filing form type
            year: Fiscal year (None for latest)
            quarter: Quarter for 10-Q
            
        Returns:
            LoadedFiling object
        """
        if year:
            filing = self.loader.load_filing(ticker, form, year, quarter)
        else:
            filing = self.loader.load_latest(ticker, form)
        
        if filing:
            key = f"{ticker}_{form}_{filing.year}"
            if form == "10-Q":
                key += f"_Q{filing.quarter}"
            self.state.loaded_filings[key] = filing
            self.state.current_ticker = ticker
        
        return filing
    
    def search_filings(
        self,
        query: str,
        tickers: List[str] = None,
        forms: List[str] = None,
        limit: int = 10,
    ) -> List[SearchResult]:
        """
        Search indexed filings.
        
        Args:
            query: Search query
            tickers: Filter tickers
            forms: Filter forms
            limit: Max results
            
        Returns:
            List of SearchResult
        """
        if not self.search:
            raise ValueError("Search not initialized")
        
        results = self.search.search(
            query=query,
            tickers=tickers,
            forms=forms,
            limit=limit,
        )
        
        self.state.last_search_results = results
        return results
    
    def analyze(
        self,
        analysis_type: AnalysisType = AnalysisType.RISK_FACTORS,
        ticker: str = None,
    ):
        """
        Analyze a loaded filing.
        
        Args:
            analysis_type: Type of analysis
            ticker: Ticker to analyze (uses current if not specified)
        """
        ticker = ticker or self.state.current_ticker
        if not ticker:
            raise ValueError("No ticker specified and none currently loaded")
        
        # Find loaded filing
        filing = None
        for key, f in self.state.loaded_filings.items():
            if f.ticker == ticker.upper():
                filing = f
                break
        
        if not filing:
            raise ValueError(f"No filing loaded for {ticker}")
        
        return self.analyzer.analyze_filing(filing, analysis_type)
    
    def clear_history(self):
        """Clear conversation history"""
        self.state.messages = []
    
    def clear_filings(self):
        """Clear loaded filings"""
        self.state.loaded_filings = {}
        self.state.current_ticker = None
    
    def reset(self):
        """Reset all state"""
        self.state = AgentState()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get agent state summary"""
        return {
            "message_count": len(self.state.messages),
            "loaded_filings": list(self.state.loaded_filings.keys()),
            "current_ticker": self.state.current_ticker,
            "last_search_count": len(self.state.last_search_results),
            "model": self.model,
        }
    
    def export_conversation(self) -> str:
        """Export conversation as markdown"""
        lines = ["# SEC Filings Research Session\n"]
        lines.append(f"Generated: {datetime.now().isoformat()}\n")
        lines.append(f"Model: {self.model}\n\n")
        
        for msg in self.state.messages:
            role_label = "**User:**" if msg.role == "user" else "**Analyst:**"
            lines.append(f"{role_label}\n\n{msg.content}\n\n---\n")
        
        return "\n".join(lines)
