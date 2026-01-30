"""
Synthesizer - LLM-powered response synthesis and natural language generation.
Transforms aggregated data into comprehensive, actionable insights.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class SynthesisResult:
    """Result from LLM synthesis"""
    summary: str
    key_insights: List[str]
    recommendation: Optional[str]
    risk_factors: List[str]
    data_sources: List[str]
    confidence: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class Synthesizer:
    """
    LLM-powered synthesis engine.
    
    Transforms raw data and analytics into:
    - Natural language summaries
    - Key insights and takeaways
    - Actionable recommendations
    - Risk assessments
    
    Uses OpenAI GPT-4 for synthesis (configurable).
    """
    
    SYSTEM_PROMPT = """You are an expert institutional equity research analyst. 
Your role is to synthesize financial data into clear, actionable insights.

Guidelines:
1. Be precise and data-driven - cite specific numbers
2. Highlight what's unusual or noteworthy
3. Connect technical, fundamental, and sentiment signals
4. Identify potential catalysts and risks
5. Provide balanced, objective analysis
6. Flag any data limitations or caveats

Format your response with clear sections:
- Executive Summary (2-3 sentences)
- Key Insights (bullet points)
- Technical View
- Fundamental View (if data available)
- Sentiment/Flow Analysis (if data available)
- Risk Factors
- Recommendation (if appropriate)

Avoid generic statements - focus on what makes THIS stock interesting RIGHT NOW."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview"
    ):
        """
        Initialize synthesizer.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for synthesis
        """
        self.api_key = api_key
        self.model = model
        self._client = None
        
        if api_key:
            self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except ImportError:
            print("OpenAI package not installed. Install with: pip install openai")
        except Exception as e:
            print(f"Could not initialize OpenAI client: {e}")
    
    def _format_data_for_prompt(self, data: Dict[str, Any]) -> str:
        """Format data dictionary for LLM prompt"""
        formatted_sections = []
        
        for symbol, symbol_data in data.items():
            sections = [f"=== {symbol} ==="]
            
            # Quote data
            if "quote" in symbol_data:
                quote = symbol_data["quote"]
                if hasattr(quote, "price"):
                    sections.append(f"""
CURRENT QUOTE:
- Price: ${quote.price:.2f}
- Change: {quote.change:+.2f} ({quote.change_percent:+.2f}%)
- Volume: {quote.volume:,}
- Day Range: ${quote.low:.2f} - ${quote.high:.2f}""")
                elif isinstance(quote, dict):
                    sections.append(f"""
CURRENT QUOTE:
- Price: ${quote.get('price', 0):.2f}
- Change: {quote.get('change', 0):+.2f} ({quote.get('change_percent', 0):+.2f}%)""")
            
            # Technical analytics
            if "analytics" in symbol_data:
                analytics = symbol_data["analytics"]
                
                if "technical" in analytics:
                    tech = analytics["technical"]
                    sections.append(f"""
TECHNICAL ANALYSIS:
- Trend Direction: {tech.direction}
- Trend Strength: {tech.strength:.2f}
- Support Levels: {', '.join(f'${s:.2f}' for s in tech.support_levels[:3])}
- Resistance Levels: {', '.join(f'${r:.2f}' for r in tech.resistance_levels[:3])}
- RSI: {tech.momentum_indicators.get('rsi', 'N/A')}
- MACD Histogram: {tech.momentum_indicators.get('macd_histogram', 'N/A')}""")
                
                if "signals" in analytics:
                    signals = analytics["signals"]
                    if signals:
                        signal_strs = [
                            f"{s.name}: {s.signal} (strength: {s.strength:.2f})"
                            for s in signals[:5]
                        ]
                        sections.append(f"""
TRADING SIGNALS:
{chr(10).join('- ' + s for s in signal_strs)}""")
                
                if "patterns" in analytics:
                    patterns = analytics["patterns"]
                    if patterns.get("patterns"):
                        pattern_strs = [
                            f"{p.pattern_type.value}: {p.direction} (confidence: {p.confidence:.2f})"
                            for p in patterns["patterns"][:5]
                        ]
                        sections.append(f"""
CHART PATTERNS:
{chr(10).join('- ' + p for p in pattern_strs)}
Overall Bias: {patterns.get('overall_bias', 'neutral')}""")
                
                if "anomalies" in analytics:
                    anomalies = analytics["anomalies"]
                    if anomalies.anomalies:
                        anom_strs = [
                            f"{a.anomaly_type.value}: {a.description} (severity: {a.severity:.2f})"
                            for a in anomalies.anomalies[:5]
                        ]
                        sections.append(f"""
ANOMALIES DETECTED (Risk Score: {anomalies.risk_score:.2f}):
{chr(10).join('- ' + a for a in anom_strs)}
Alert Level: {anomalies.alert_level}""")
                
                if "sentiment" in analytics:
                    sent = analytics["sentiment"]
                    sections.append(f"""
SENTIMENT ANALYSIS:
- Overall Score: {sent.overall_score:.2f} ({sent.overall_sentiment})
- Articles Analyzed: {sent.article_count}
- Bullish: {sent.bullish_count}, Bearish: {sent.bearish_count}, Neutral: {sent.neutral_count}""")
            
            # Fundamentals
            if "fundamentals" in symbol_data:
                fund = symbol_data["fundamentals"]
                if isinstance(fund, dict):
                    sections.append(f"""
FUNDAMENTALS:
{json.dumps(fund, indent=2, default=str)[:1000]}""")
            
            # Options
            if "options" in symbol_data:
                opts = symbol_data["options"]
                if opts:
                    sections.append(f"""
OPTIONS DATA:
- Contracts Available: {len(opts) if isinstance(opts, list) else 'See data'}""")
            
            # Holdings
            if "holdings" in symbol_data:
                holdings = symbol_data["holdings"]
                sections.append(f"""
INSTITUTIONAL HOLDINGS:
- Data Available: Yes""")
            
            # Filings
            if "filing" in symbol_data:
                filings = symbol_data["filing"]
                if isinstance(filings, list) and filings:
                    filing_strs = [
                        f"{f.form_type} ({f.filed_date})"
                        for f in filings[:5]
                    ]
                    sections.append(f"""
RECENT SEC FILINGS:
{chr(10).join('- ' + f for f in filing_strs)}""")
            
            formatted_sections.append("\n".join(sections))
        
        return "\n\n".join(formatted_sections)
    
    async def synthesize(
        self,
        query: str,
        data: Dict[str, Any],
        context: Optional[str] = None
    ) -> SynthesisResult:
        """
        Synthesize data into natural language insights.
        
        Args:
            query: Original user query
            data: Aggregated data from orchestrator
            context: Additional context
            
        Returns:
            SynthesisResult with summary and insights
        """
        if not self._client:
            return self._fallback_synthesis(query, data)
        
        # Format data for prompt
        formatted_data = self._format_data_for_prompt(data)
        
        # Build prompt
        user_prompt = f"""User Query: {query}

Available Data:
{formatted_data}

{f"Additional Context: {context}" if context else ""}

Please provide a comprehensive analysis based on this data."""

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Parse response into structured format
            return self._parse_synthesis_response(content, data)
            
        except Exception as e:
            print(f"Synthesis error: {e}")
            return self._fallback_synthesis(query, data)
    
    def _parse_synthesis_response(
        self,
        content: str,
        data: Dict[str, Any]
    ) -> SynthesisResult:
        """Parse LLM response into structured result"""
        # Extract key sections
        lines = content.split('\n')
        
        # Find insights (bullet points)
        insights = []
        risks = []
        
        in_insights = False
        in_risks = False
        
        for line in lines:
            line = line.strip()
            
            if 'insight' in line.lower() or 'key point' in line.lower():
                in_insights = True
                in_risks = False
                continue
            elif 'risk' in line.lower():
                in_risks = True
                in_insights = False
                continue
            elif any(header in line.lower() for header in 
                    ['technical', 'fundamental', 'recommendation', 'summary']):
                in_insights = False
                in_risks = False
            
            if line.startswith(('-', '•', '*', '·')):
                clean_line = line.lstrip('-•*· ').strip()
                if in_insights:
                    insights.append(clean_line)
                elif in_risks:
                    risks.append(clean_line)
        
        # Extract recommendation if present
        recommendation = None
        for line in lines:
            if 'recommendation' in line.lower():
                idx = lines.index(line)
                if idx + 1 < len(lines):
                    recommendation = lines[idx + 1].strip()
                    if recommendation.startswith(('-', '•', '*')):
                        recommendation = recommendation.lstrip('-•* ').strip()
                break
        
        # Get data sources
        data_sources = list(data.keys())
        
        return SynthesisResult(
            summary=content,
            key_insights=insights[:10],
            recommendation=recommendation,
            risk_factors=risks[:5],
            data_sources=data_sources,
            confidence=0.85
        )
    
    def _fallback_synthesis(
        self,
        query: str,
        data: Dict[str, Any]
    ) -> SynthesisResult:
        """Generate synthesis without LLM (fallback)"""
        insights = []
        risks = []
        
        for symbol, symbol_data in data.items():
            # Extract key points from analytics
            if "analytics" in symbol_data:
                analytics = symbol_data["analytics"]
                
                if "technical" in analytics:
                    tech = analytics["technical"]
                    insights.append(
                        f"{symbol} is showing {tech.direction} momentum "
                        f"with {tech.strength:.0%} strength"
                    )
                
                if "sentiment" in analytics:
                    sent = analytics["sentiment"]
                    insights.append(
                        f"News sentiment is {sent.overall_sentiment} "
                        f"based on {sent.article_count} articles"
                    )
                
                if "anomalies" in analytics:
                    anom = analytics["anomalies"]
                    if anom.anomalies:
                        insights.append(
                            f"Detected {len(anom.anomalies)} anomalies "
                            f"(alert level: {anom.alert_level})"
                        )
                        for a in anom.anomalies[:3]:
                            risks.append(a.description)
                
                if "patterns" in analytics:
                    patterns = analytics["patterns"]
                    if patterns.get("patterns"):
                        p = patterns["patterns"][0]
                        insights.append(
                            f"Detected {p.pattern_type.value} pattern "
                            f"({p.direction}, confidence: {p.confidence:.0%})"
                        )
        
        summary = f"""Analysis for query: "{query}"

Based on available data:
{chr(10).join('• ' + i for i in insights)}

Risk factors:
{chr(10).join('• ' + r for r in risks) if risks else '• No significant risks detected'}

Data sources: {', '.join(data.keys())}"""
        
        return SynthesisResult(
            summary=summary,
            key_insights=insights,
            recommendation=None,
            risk_factors=risks,
            data_sources=list(data.keys()),
            confidence=0.6  # Lower confidence for fallback
        )
    
    def format_for_cli(self, result: SynthesisResult) -> str:
        """Format synthesis result for CLI display"""
        output = []
        output.append("=" * 60)
        output.append("ANALYSIS SUMMARY")
        output.append("=" * 60)
        output.append("")
        output.append(result.summary)
        output.append("")
        
        if result.key_insights:
            output.append("-" * 40)
            output.append("KEY INSIGHTS")
            output.append("-" * 40)
            for insight in result.key_insights:
                output.append(f"  • {insight}")
            output.append("")
        
        if result.risk_factors:
            output.append("-" * 40)
            output.append("RISK FACTORS")
            output.append("-" * 40)
            for risk in result.risk_factors:
                output.append(f"  ⚠ {risk}")
            output.append("")
        
        if result.recommendation:
            output.append("-" * 40)
            output.append("RECOMMENDATION")
            output.append("-" * 40)
            output.append(f"  → {result.recommendation}")
            output.append("")
        
        output.append("-" * 40)
        output.append(f"Data Sources: {', '.join(result.data_sources)}")
        output.append(f"Confidence: {result.confidence:.0%}")
        output.append(f"Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("=" * 60)
        
        return "\n".join(output)
