"""
AI Agent Framework for Financial Research
==========================================

Multi-agent system for institutional-grade financial analysis.
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum

try:
    import openai
except ImportError:
    openai = None


class AgentRole(Enum):
    RESEARCH = "research"
    TRADING = "trading"
    NEWS = "news"
    RISK = "risk"


@dataclass
class AgentContext:
    ticker: Optional[str] = None
    tickers: List[str] = field(default_factory=list)
    timeframe: str = "1D"
    analysis_depth: str = "standard"
    user_query: str = ""


class BaseAgent(ABC):
    def __init__(self, name: str, role: AgentRole, model: str = "gpt-4o"):
        self.name = name
        self.role = role
        self.model = model
        self.api_key = os.environ.get('OPENAI_API_KEY')
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass
    
    async def think(self, query: str, context: AgentContext, data: Optional[Dict] = None) -> str:
        if not self.api_key or not openai:
            return f"[{self.name}] Would analyze: {query}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context: {context.ticker}\nData: {json.dumps(data or {}, default=str)[:2000]}\n\nQuery: {query}"}
        ]
        
        try:
            openai.api_key = self.api_key
            response = openai.chat.completions.create(model=self.model, messages=messages, temperature=0.3, max_tokens=2000)
            return response.choices[0].message.content
        except Exception as e:
            return f"[{self.name}] Error: {e}"


class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("Research Analyst", AgentRole.RESEARCH)
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert institutional research analyst. Provide data-driven, balanced analysis with specific metrics, price targets, and risk assessment. Use professional financial terminology."""


class TradingAgent(BaseAgent):
    def __init__(self):
        super().__init__("Trading Strategist", AgentRole.TRADING)
    
    @property
    def system_prompt(self) -> str:
        return """You are a quantitative trading strategist. Provide clear BUY/SELL/HOLD signals with entry/exit prices, stop-losses, and risk/reward ratios."""


class NewsAgent(BaseAgent):
    def __init__(self):
        super().__init__("News Analyst", AgentRole.NEWS)
    
    @property
    def system_prompt(self) -> str:
        return """You are a financial news analyst. Assess sentiment, identify material news, and estimate market impact (1-10 scale)."""


class RiskAgent(BaseAgent):
    def __init__(self):
        super().__init__("Risk Manager", AgentRole.RISK)
    
    @property
    def system_prompt(self) -> str:
        return """You are a quantitative risk manager. Provide VaR, beta, volatility analysis, tail risk assessment, and risk mitigation strategies."""


class AgentOrchestrator:
    def __init__(self):
        self.agents = {
            AgentRole.RESEARCH: ResearchAgent(),
            AgentRole.TRADING: TradingAgent(),
            AgentRole.NEWS: NewsAgent(),
            AgentRole.RISK: RiskAgent()
        }
    
    async def analyze(self, query: str, ticker: str = None) -> Dict[str, Any]:
        context = AgentContext(ticker=ticker, user_query=query)
        data = {"ticker": ticker, "placeholder": "data_from_mcp"}
        
        results = {}
        for role, agent in self.agents.items():
            results[role.value] = await agent.think(query, context, data)
        
        return {"query": query, "ticker": ticker, "results": results, "timestamp": datetime.now().isoformat()}


class FinancialAssistant:
    def __init__(self):
        self.orchestrator = AgentOrchestrator()
    
    async def ask(self, question: str) -> str:
        import re
        ticker = None
        match = re.search(r'\$?([A-Z]{1,5})\b', question.upper())
        if match:
            ticker = match.group(1)
        
        result = await self.orchestrator.analyze(question, ticker)
        return json.dumps(result, indent=2, default=str)
