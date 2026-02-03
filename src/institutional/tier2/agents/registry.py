"""
Agent Registry
==============

Factory and registry for agent instantiation and discovery.
"""

from typing import Dict, List, Optional, Type

from .base import BaseAgent


class AgentRegistry:
    """Registry for managing agent types and instances."""
    
    _agents: Dict[str, Type[BaseAgent]] = {}
    _instances: Dict[str, BaseAgent] = {}
    
    @classmethod
    def register(cls, agent_class: Type[BaseAgent]) -> Type[BaseAgent]:
        """Register an agent class. Can be used as decorator."""
        cls._agents[agent_class.agent_id] = agent_class
        return agent_class
    
    @classmethod
    def get_agent_class(cls, agent_id: str) -> Optional[Type[BaseAgent]]:
        """Get agent class by ID."""
        return cls._agents.get(agent_id)
    
    @classmethod
    def create_agent(cls, agent_id: str, **kwargs) -> Optional[BaseAgent]:
        """Create an agent instance by ID."""
        agent_class = cls._agents.get(agent_id)
        if agent_class:
            return agent_class(**kwargs)
        return None
    
    @classmethod
    def get_or_create(cls, agent_id: str, **kwargs) -> Optional[BaseAgent]:
        """Get existing instance or create new one."""
        if agent_id not in cls._instances:
            instance = cls.create_agent(agent_id, **kwargs)
            if instance:
                cls._instances[agent_id] = instance
        return cls._instances.get(agent_id)
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """List all registered agent IDs."""
        return list(cls._agents.keys())
    
    @classmethod
    def get_agents_by_tag(cls, tag: str) -> List[Type[BaseAgent]]:
        """Get all agents that have a specific expertise tag."""
        return [
            agent for agent in cls._agents.values()
            if tag in agent.expertise_tags
        ]


def _register_all_agents():
    """Register all agents on module load."""
    from .technical.mtf_dominance import MTFDominanceAgent
    from .technical.trend_integrity import TrendIntegrityAgent
    from .technical.reversal_pullback import ReversalPullbackAgent
    from .technical.volatility_state import VolatilityStateAgent
    from .market.regime_transition import RegimeTransitionAgent
    from .market.event_window import EventWindowAgent
    from .risk.risk_gatekeeper import RiskGatekeeperAgent
    from .risk.concentration import ConcentrationAgent
    from .risk.liquidity import LiquidityAgent
    from .quality.data_integrity import DataIntegrityAgent
    from .quality.timing import TimingAgent
    from .adversarial.red_team import RedTeamAgent
    
    for agent_class in [
        MTFDominanceAgent,
        TrendIntegrityAgent,
        ReversalPullbackAgent,
        VolatilityStateAgent,
        RegimeTransitionAgent,
        EventWindowAgent,
        RiskGatekeeperAgent,
        ConcentrationAgent,
        LiquidityAgent,
        DataIntegrityAgent,
        TimingAgent,
        RedTeamAgent,
    ]:
        AgentRegistry.register(agent_class)


# Convenience functions
def create_agent(agent_id: str, **kwargs) -> Optional[BaseAgent]:
    """Create an agent by ID."""
    return AgentRegistry.create_agent(agent_id, **kwargs)


def get_all_agents() -> List[BaseAgent]:
    """Get instances of all registered agents."""
    return [
        AgentRegistry.get_or_create(agent_id)
        for agent_id in AgentRegistry.list_agents()
    ]


# Register on import
_register_all_agents()
