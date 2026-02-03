"""
Tier 2 Agent Registry
======================

Manages the council of expert agents.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .config import Tier2Config, AgentConfig, DEFAULT_CONFIG
from .agents import BaseAgent, create_agent


@dataclass
class AgentRegistration:
    """A registered agent with its config and instance."""
    config: AgentConfig
    instance: BaseAgent
    enabled: bool = True


class AgentRegistry:
    """
    Registry of all Tier 2 expert agents.
    
    Manages:
    - Agent instantiation
    - Configuration
    - Enable/disable
    - Weight management
    """
    
    def __init__(self, config: Tier2Config = None):
        self.config = config or DEFAULT_CONFIG
        self._agents: Dict[str, AgentRegistration] = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all configured agents."""
        for name, agent_config in self.config.agents.items():
            try:
                instance = create_agent(name)
                self._agents[name] = AgentRegistration(
                    config=agent_config,
                    instance=instance,
                    enabled=agent_config.enabled,
                )
            except Exception as e:
                print(f"Warning: Failed to initialize agent {name}: {e}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        reg = self._agents.get(name)
        if reg and reg.enabled:
            return reg.instance
        return None
    
    def get_config(self, name: str) -> Optional[AgentConfig]:
        """Get agent config by name."""
        reg = self._agents.get(name)
        return reg.config if reg else None
    
    def get_enabled_agents(self) -> List[str]:
        """Get list of enabled agent names."""
        return [name for name, reg in self._agents.items() if reg.enabled]
    
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """Get all enabled agents."""
        return {
            name: reg.instance 
            for name, reg in self._agents.items() 
            if reg.enabled
        }
    
    def get_agent_weights(self) -> Dict[str, float]:
        """Get weights for all enabled agents."""
        return {
            name: reg.config.weight
            for name, reg in self._agents.items()
            if reg.enabled
        }
    
    def enable_agent(self, name: str) -> bool:
        """Enable an agent."""
        if name in self._agents:
            self._agents[name].enabled = True
            return True
        return False
    
    def disable_agent(self, name: str) -> bool:
        """Disable an agent."""
        if name in self._agents:
            self._agents[name].enabled = False
            return True
        return False
    
    def set_weight(self, name: str, weight: float) -> bool:
        """Update agent weight."""
        if name in self._agents:
            self._agents[name].config.weight = weight
            return True
        return False
    
    def get_status(self) -> Dict:
        """Get registry status."""
        return {
            "total_agents": len(self._agents),
            "enabled_agents": len(self.get_enabled_agents()),
            "agents": {
                name: {
                    "enabled": reg.enabled,
                    "weight": reg.config.weight,
                    "can_veto": reg.config.can_veto,
                    "always_deep": reg.config.always_run_deep,
                }
                for name, reg in self._agents.items()
            }
        }


# Global registry instance
_registry: Optional[AgentRegistry] = None


def get_default_registry() -> AgentRegistry:
    """Get the default global registry."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry


def create_default_registry(config: Tier2Config = None) -> AgentRegistry:
    """Create a new registry with optional config."""
    return AgentRegistry(config=config)


def reset_registry():
    """Reset the global registry."""
    global _registry
    _registry = None
