"""
KYPERIAN Elite - Multi-Agent Cognitive System

The world's most advanced AI financial advisor.
Powered by Claude Opus 4.5 orchestration with 9 specialized agents.
"""

from .base import (
    SpecializedAgent,
    AgentType,
    AgentTask,
    AgentResult,
    TaskPriority
)
from .orchestrator import OrchestratorAgent, OrchestratorConfig, ConversationContext

# Specialized Agents
from .market_analyst import MarketAnalystAgent
from .quant_analyst import QuantAnalystAgent
from .news_analyst import NewsAnalystAgent
from .fundamental_analyst import FundamentalAnalystAgent
from .macro_analyst import MacroAnalystAgent
from .risk_manager import RiskManagerAgent
from .portfolio_optimizer import PortfolioOptimizerAgent
from .crypto_specialist import CryptoSpecialistAgent
from .educator import EducatorAgent

__all__ = [
    # Base
    'SpecializedAgent',
    'AgentType',
    'AgentTask',
    'AgentResult',
    'TaskPriority',
    
    # Orchestrator
    'OrchestratorAgent',
    'OrchestratorConfig',
    'ConversationContext',
    
    # Specialized Agents
    'MarketAnalystAgent',
    'QuantAnalystAgent',
    'NewsAnalystAgent',
    'FundamentalAnalystAgent',
    'MacroAnalystAgent',
    'RiskManagerAgent',
    'PortfolioOptimizerAgent',
    'CryptoSpecialistAgent',
    'EducatorAgent'
]
