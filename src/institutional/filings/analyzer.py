"""
SEC Filings Analyzer
====================

Claude Opus 4.5 powered analysis of SEC filings.
Provides institutional-grade insights from regulatory documents.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .search import FilingsSearch, SearchResult
from .loader import LoadedFiling


class AnalysisType(Enum):
    """Types of filing analysis"""
    RISK_FACTORS = "risk_factors"
    FINANCIAL_HIGHLIGHTS = "financial_highlights"
    MANAGEMENT_DISCUSSION = "management_discussion"
    SEGMENT_ANALYSIS = "segment_analysis"
    COMPETITIVE_LANDSCAPE = "competitive_landscape"
    GUIDANCE = "guidance"
    LEGAL_PROCEEDINGS = "legal_proceedings"
    RELATED_PARTY = "related_party"
    CUSTOM_QUERY = "custom_query"


@dataclass
class AnalysisResult:
    """Result of a filing analysis"""
    ticker: str
    form: str
    year: int
    analysis_type: str
    content: str
    confidence: float
    sources: List[str] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    model: str = "claude-opus-4-5-20251101"
    latency_ms: int = 0
    tokens_used: int = 0


class FilingsAnalyzer:
    """
    AI-powered SEC filings analysis using Claude Opus 4.5.
    
    Provides:
    - Risk factor analysis
    - Financial highlights extraction
    - Management discussion insights
    - Segment analysis
    - Competitive landscape review
    - Custom query analysis
    """
    
    # System prompts for different analysis types
    SYSTEM_PROMPTS = {
        "risk_factors": """You are an elite SEC filings analyst specializing in risk assessment.
        
Your task is to analyze the risk factors disclosed in SEC filings and provide:
1. The TOP 5 most material risks, ranked by potential impact
2. Changes from prior filings if available
3. Industry-specific vs company-specific risks
4. Hidden risks that may be understated
5. Overall risk profile assessment

Be specific, cite exact language from the filings, and provide actionable insights.
Format your response with clear headers and bullet points.""",

        "financial_highlights": """You are an elite SEC filings analyst specializing in financial analysis.

Your task is to extract and analyze key financial information from SEC filings:
1. Revenue trends and segment breakdown
2. Profitability metrics and margin analysis
3. Balance sheet health indicators
4. Cash flow analysis
5. Key accounting policies and changes
6. Non-GAAP reconciliations and adjustments

Provide specific numbers, calculate growth rates, and highlight anomalies.
Format your response with clear headers and include relevant metrics.""",

        "management_discussion": """You are an elite SEC filings analyst specializing in MD&A analysis.

Your task is to analyze the Management Discussion & Analysis section:
1. Key strategic priorities and initiatives
2. Management's view on business performance
3. Forward-looking statements and guidance
4. Capital allocation plans
5. Competitive positioning commentary
6. Tone and sentiment analysis

Look for what management emphasizes vs. downplays. Identify key themes.
Format your response with clear headers and specific citations.""",

        "segment_analysis": """You are an elite SEC filings analyst specializing in business segment analysis.

Your task is to analyze the company's business segments:
1. Segment revenue breakdown and trends
2. Segment profitability comparison
3. Growth drivers by segment
4. Geographic distribution
5. Key customers and concentration
6. Segment outlook and strategic importance

Provide specific numbers and calculate segment contributions.
Format your response with clear headers and comparative analysis.""",

        "competitive_landscape": """You are an elite SEC filings analyst specializing in competitive analysis.

Your task is to analyze competitive positioning from SEC filings:
1. Identified competitors mentioned
2. Competitive advantages cited
3. Market position and share
4. Barriers to entry discussed
5. Competitive risks and threats
6. Industry dynamics

Extract both explicit and implicit competitive intelligence.
Format your response with clear headers and strategic insights.""",

        "guidance": """You are an elite SEC filings analyst specializing in guidance analysis.

Your task is to extract and analyze forward guidance from SEC filings:
1. Revenue and earnings guidance
2. Margin expectations
3. Capital expenditure plans
4. Strategic initiatives timeline
5. Key assumptions underlying guidance
6. Track record on prior guidance

Compare to analyst expectations where context permits.
Format your response with specific numbers and confidence levels.""",

        "custom_query": """You are an elite SEC filings analyst with comprehensive expertise.

Analyze the SEC filings based on the specific query provided. Be thorough, specific,
and cite exact language from the documents. Provide actionable insights suitable
for institutional investors.""",
    }
    
    def __init__(
        self,
        api_key: str = None,
        search: FilingsSearch = None,
        model: str = "claude-opus-4-5-20251101",
    ):
        """
        Initialize the analyzer.
        
        Args:
            api_key: Anthropic API key
            search: FilingsSearch instance for context retrieval
            model: Claude model to use
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic not installed. Run: pip install anthropic")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        self.client = Anthropic(api_key=self.api_key)
        self.search = search
        self.model = model
    
    def analyze_filing(
        self,
        filing: LoadedFiling,
        analysis_type: AnalysisType = AnalysisType.RISK_FACTORS,
        custom_query: str = None,
        max_context: int = 100000,
    ) -> AnalysisResult:
        """
        Analyze a loaded SEC filing.
        
        Args:
            filing: LoadedFiling object
            analysis_type: Type of analysis
            custom_query: Query for custom analysis
            max_context: Max characters of filing to include
            
        Returns:
            AnalysisResult
        """
        analysis_key = analysis_type.value
        system_prompt = self.SYSTEM_PROMPTS.get(
            analysis_key,
            self.SYSTEM_PROMPTS["custom_query"]
        )
        
        # Prepare filing context
        filing_text = filing.text[:max_context] if len(filing.text) > max_context else filing.text
        
        # Build user prompt
        if analysis_type == AnalysisType.CUSTOM_QUERY and custom_query:
            user_prompt = f"""Analyze the following {filing.ticker} {filing.form} filing for {filing.year}.

QUERY: {custom_query}

FILING CONTENT:
{filing_text}"""
        else:
            user_prompt = f"""Analyze the following {filing.ticker} {filing.form} filing for {filing.year}.

FILING CONTENT:
{filing_text}"""
        
        # Call Claude
        start_time = datetime.now()
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        # Extract key points
        key_points = self._extract_key_points(content)
        
        return AnalysisResult(
            ticker=filing.ticker,
            form=filing.form,
            year=filing.year,
            analysis_type=analysis_key,
            content=content,
            confidence=0.9,
            sources=[f"{filing.ticker} {filing.form} {filing.year}"],
            key_points=key_points,
            model=self.model,
            latency_ms=latency_ms,
            tokens_used=tokens,
        )
    
    def analyze_search_results(
        self,
        query: str,
        results: List[SearchResult],
        analysis_type: AnalysisType = AnalysisType.CUSTOM_QUERY,
    ) -> AnalysisResult:
        """
        Analyze search results from multiple filings.
        
        Args:
            query: Original search query
            results: List of SearchResult objects
            analysis_type: Type of analysis
            
        Returns:
            AnalysisResult
        """
        if not results:
            raise ValueError("No search results to analyze")
        
        # Collect unique tickers and forms
        tickers = list(set(r.ticker for r in results))
        sources = list(set(r.source_label for r in results))
        
        # Build context from results
        context_parts = []
        for r in results[:20]:  # Limit context
            context_parts.append(f"[{r.source_label}] (similarity: {r.similarity:.2f})\n{r.text}\n")
        
        context = "\n---\n".join(context_parts)
        
        system_prompt = self.SYSTEM_PROMPTS.get(
            analysis_type.value,
            self.SYSTEM_PROMPTS["custom_query"]
        )
        
        user_prompt = f"""Analyze the following SEC filings content related to: {query}

Companies analyzed: {', '.join(tickers)}

RELEVANT EXCERPTS:
{context}

Provide comprehensive analysis with specific citations to the source documents."""
        
        # Call Claude
        start_time = datetime.now()
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        key_points = self._extract_key_points(content)
        
        return AnalysisResult(
            ticker=", ".join(tickers),
            form="multiple",
            year=0,
            analysis_type=analysis_type.value,
            content=content,
            confidence=0.85,
            sources=sources,
            key_points=key_points,
            model=self.model,
            latency_ms=latency_ms,
            tokens_used=tokens,
        )
    
    def compare_filings(
        self,
        filings: List[LoadedFiling],
        focus_area: str = "risk factors and key changes",
    ) -> AnalysisResult:
        """
        Compare multiple filings (e.g., YoY comparison).
        
        Args:
            filings: List of LoadedFiling objects to compare
            focus_area: What to focus the comparison on
            
        Returns:
            AnalysisResult with comparative analysis
        """
        if len(filings) < 2:
            raise ValueError("Need at least 2 filings to compare")
        
        # Sort by year (most recent first)
        filings = sorted(filings, key=lambda f: (f.year, f.quarter), reverse=True)
        
        system_prompt = """You are an elite SEC filings analyst specializing in comparative analysis.

Your task is to compare SEC filings across time periods and identify:
1. Material changes in language, disclosures, or risk factors
2. New risks or disclosures added
3. Removed or softened language
4. Trends in financial metrics
5. Shifts in strategic focus
6. Changes in tone and confidence

Be specific about what changed and when. Highlight the most significant changes."""
        
        # Build context
        context_parts = []
        for f in filings:
            label = f"{f.ticker} {f.form} {f.year}"
            if f.form == "10-Q":
                label += f" Q{f.quarter}"
            
            # Truncate each filing
            text = f.text[:40000] if len(f.text) > 40000 else f.text
            context_parts.append(f"=== {label} ===\n{text}")
        
        context = "\n\n".join(context_parts)
        
        user_prompt = f"""Compare the following filings with focus on: {focus_area}

{context}

Provide a detailed comparative analysis highlighting key changes and trends."""
        
        # Call Claude
        start_time = datetime.now()
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        sources = [f"{f.ticker} {f.form} {f.year}" for f in filings]
        key_points = self._extract_key_points(content)
        
        return AnalysisResult(
            ticker=filings[0].ticker,
            form=filings[0].form,
            year=0,  # Multiple years
            analysis_type="comparative",
            content=content,
            confidence=0.85,
            sources=sources,
            key_points=key_points,
            model=self.model,
            latency_ms=latency_ms,
            tokens_used=tokens,
        )
    
    def answer_question(
        self,
        question: str,
        filing: LoadedFiling = None,
        search_results: List[SearchResult] = None,
    ) -> AnalysisResult:
        """
        Answer a specific question using filing context.
        
        Args:
            question: Natural language question
            filing: Optional specific filing to query
            search_results: Optional search results for context
            
        Returns:
            AnalysisResult
        """
        system_prompt = """You are an elite SEC filings analyst. Answer the question based solely on 
the provided SEC filing content. Be specific, cite sources, and indicate confidence level.
If the information is not available in the provided content, say so clearly."""
        
        # Build context
        if filing:
            context = filing.text[:80000]
            sources = [f"{filing.ticker} {filing.form} {filing.year}"]
            ticker = filing.ticker
        elif search_results:
            context_parts = [f"[{r.source_label}]\n{r.text}" for r in search_results[:15]]
            context = "\n\n---\n\n".join(context_parts)
            sources = list(set(r.source_label for r in search_results))
            ticker = ", ".join(set(r.ticker for r in search_results))
        else:
            raise ValueError("Provide either filing or search_results")
        
        user_prompt = f"""QUESTION: {question}

SEC FILING CONTENT:
{context}

Answer the question based on the above content. Be specific and cite relevant sections."""
        
        # Call Claude
        start_time = datetime.now()
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        return AnalysisResult(
            ticker=ticker,
            form="multiple" if search_results else filing.form,
            year=filing.year if filing else 0,
            analysis_type="question_answer",
            content=content,
            confidence=0.85,
            sources=sources,
            key_points=[],
            model=self.model,
            latency_ms=latency_ms,
            tokens_used=tokens,
        )
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from analysis content"""
        key_points = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            # Look for numbered or bulleted items
            if line and (
                line.startswith('1.') or 
                line.startswith('•') or 
                line.startswith('-') or
                line.startswith('*')
            ):
                # Clean up the point
                point = line.lstrip('0123456789.-•* ')
                if len(point) > 20 and len(point) < 200:
                    key_points.append(point)
        
        return key_points[:10]  # Top 10 points
