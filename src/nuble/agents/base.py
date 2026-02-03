#!/usr/bin/env python3
"""
Base class for all specialized agents.

Each agent:
- Has specific expertise
- Has access to specific tools/data sources
- Returns structured data
- Can use Claude Sonnet for speed
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from enum import Enum
import os

# Lazy import anthropic to avoid module-level blocking
_anthropic_module = None
HAS_ANTHROPIC = False

def _get_anthropic():
    """Lazy load anthropic module."""
    global _anthropic_module, HAS_ANTHROPIC
    if _anthropic_module is None:
        try:
            import anthropic as _anth
            _anthropic_module = _anth
            HAS_ANTHROPIC = True
        except ImportError:
            HAS_ANTHROPIC = False
    return _anthropic_module

# Use Claude Sonnet for speed in specialized agents
AGENT_MODEL = "claude-sonnet-4-20250514"


class AgentType(Enum):
    """Specialized agent types."""
    MARKET_ANALYST = "market_analyst"
    QUANT_ANALYST = "quant_analyst"
    NEWS_ANALYST = "news_analyst"
    FUNDAMENTAL_ANALYST = "fundamental_analyst"
    MACRO_ANALYST = "macro_analyst"
    RISK_MANAGER = "risk_manager"
    PORTFOLIO_OPTIMIZER = "portfolio_optimizer"
    CRYPTO_SPECIALIST = "crypto_specialist"
    EDUCATOR = "educator"


class TaskPriority(Enum):
    """Task execution priority."""
    CRITICAL = 1    # Must complete before response
    HIGH = 2        # Important for quality
    MEDIUM = 3      # Adds value
    LOW = 4         # Nice to have


@dataclass
class AgentTask:
    """Task assigned to a specialized agent."""
    task_id: str
    agent_type: AgentType
    instruction: str
    context: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout_seconds: int = 30
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result from a specialized agent."""
    task_id: str
    agent_type: AgentType
    success: bool
    data: Dict[str, Any]
    confidence: float
    execution_time_ms: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'agent_type': self.agent_type.value,
            'success': self.success,
            'data': self.data,
            'confidence': self.confidence,
            'execution_time_ms': self.execution_time_ms,
            'error': self.error
        }


class SpecializedAgent(ABC):
    """
    Base class for specialized agents.
    
    Each agent:
    - Has specific expertise
    - Has access to specific tools/data sources  
    - Returns structured data
    - Can optionally use Claude for reasoning
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.client = None
        # Lazy load anthropic client only when needed
        anthropic = _get_anthropic()
        if anthropic and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = AGENT_MODEL
        self.agent_type: AgentType = None
        self.name: str = "Base Agent"
        self.description: str = "Base specialized agent"
    
    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute a task and return results."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities."""
        pass
    
    def _create_error_result(
        self, 
        task: AgentTask, 
        error: str,
        execution_time_ms: int = 0
    ) -> AgentResult:
        """Create an error result."""
        return AgentResult(
            task_id=task.task_id,
            agent_type=self.agent_type,
            success=False,
            data={},
            confidence=0,
            execution_time_ms=execution_time_ms,
            error=error
        )
    
    def _create_success_result(
        self,
        task: AgentTask,
        data: Dict[str, Any],
        confidence: float,
        execution_time_ms: int = 0
    ) -> AgentResult:
        """Create a success result."""
        return AgentResult(
            task_id=task.task_id,
            agent_type=self.agent_type,
            success=True,
            data=data,
            confidence=confidence,
            execution_time_ms=execution_time_ms
        )
    
    async def reason_with_claude(
        self,
        prompt: str,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """Use Claude for reasoning if available."""
        if not self.client:
            return None
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return None
