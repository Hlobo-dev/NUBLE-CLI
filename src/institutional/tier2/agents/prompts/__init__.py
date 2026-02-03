"""
Prompt Management System
========================

Version-controlled prompts for agent improvement and A/B testing.

Structure:
    prompts/
    ├── __init__.py           # This file
    ├── loader.py             # YAML/JSON prompt loader
    ├── tracker.py            # Performance tracking per version
    ├── ab_test.py            # A/B testing infrastructure
    │
    └── {agent_id}/           # Per-agent prompt directories
        ├── v1.0.yaml
        ├── v1.1.yaml
        └── current.yaml -> v1.1.yaml

Usage:
    from .prompts import PromptManager
    
    manager = PromptManager()
    prompt = manager.get_prompt("mtf_dominance", "system")
    
    # For A/B testing
    prompt = manager.get_prompt_ab("mtf_dominance", "system", user_id)
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from pathlib import Path
import hashlib


@dataclass
class PromptVersion:
    """A versioned prompt configuration."""
    agent_id: str
    version: str
    prompt_type: str  # system, light, deep
    content: str
    metadata: Dict
    
    @property
    def version_hash(self) -> str:
        """Hash for tracking this exact version."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:12]


class PromptManager:
    """
    Manages versioned prompts for all agents.
    
    Supports:
    - Loading prompts from files or inline definitions
    - Version tracking and rollback
    - A/B testing between versions
    - Performance metrics per version
    """
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        self.prompts_dir = prompts_dir or Path(__file__).parent
        self._cache: Dict[str, PromptVersion] = {}
        self._active_versions: Dict[str, str] = {}  # agent_id -> version
        self._ab_tests: Dict[str, 'ABTest'] = {}
    
    def get_prompt(
        self,
        agent_id: str,
        prompt_type: str,
        version: Optional[str] = None
    ) -> Optional[PromptVersion]:
        """Get a prompt by agent and type."""
        version = version or self._active_versions.get(agent_id, "current")
        cache_key = f"{agent_id}:{prompt_type}:{version}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try to load from file
        prompt = self._load_prompt_file(agent_id, prompt_type, version)
        if prompt:
            self._cache[cache_key] = prompt
        
        return prompt
    
    def _load_prompt_file(
        self,
        agent_id: str,
        prompt_type: str,
        version: str
    ) -> Optional[PromptVersion]:
        """Load prompt from YAML file."""
        prompt_path = self.prompts_dir / agent_id / f"{version}.yaml"
        
        if not prompt_path.exists():
            return None
        
        try:
            import yaml
            with open(prompt_path) as f:
                data = yaml.safe_load(f)
            
            content = data.get("prompts", {}).get(prompt_type, "")
            metadata = data.get("metadata", {})
            
            return PromptVersion(
                agent_id=agent_id,
                version=version,
                prompt_type=prompt_type,
                content=content,
                metadata=metadata
            )
        except Exception:
            return None
    
    def set_active_version(self, agent_id: str, version: str):
        """Set the active version for an agent."""
        self._active_versions[agent_id] = version
    
    def get_prompt_ab(
        self,
        agent_id: str,
        prompt_type: str,
        user_id: str
    ) -> Optional[PromptVersion]:
        """Get prompt with A/B test routing."""
        ab_test = self._ab_tests.get(agent_id)
        if ab_test and ab_test.is_active:
            version = ab_test.get_variant(user_id)
            return self.get_prompt(agent_id, prompt_type, version)
        
        return self.get_prompt(agent_id, prompt_type)
    
    def create_ab_test(
        self,
        agent_id: str,
        control_version: str,
        treatment_version: str,
        traffic_split: float = 0.5
    ) -> 'ABTest':
        """Create an A/B test between two versions."""
        ab_test = ABTest(
            agent_id=agent_id,
            control=control_version,
            treatment=treatment_version,
            split=traffic_split
        )
        self._ab_tests[agent_id] = ab_test
        return ab_test


@dataclass
class ABTest:
    """An A/B test configuration."""
    agent_id: str
    control: str
    treatment: str
    split: float = 0.5
    is_active: bool = True
    
    def get_variant(self, user_id: str) -> str:
        """Deterministically assign user to variant."""
        hash_value = int(hashlib.md5(f"{self.agent_id}:{user_id}".encode()).hexdigest(), 16)
        if (hash_value % 100) / 100 < self.split:
            return self.treatment
        return self.control


class PromptMetrics:
    """
    Tracks performance metrics per prompt version.
    
    Metrics:
    - accuracy: % of correct predictions
    - latency: Average response time
    - token_usage: Average tokens consumed
    - user_feedback: Explicit feedback scores
    """
    
    def __init__(self):
        self._metrics: Dict[str, Dict] = {}
    
    def record(
        self,
        version_hash: str,
        accuracy: Optional[float] = None,
        latency_ms: Optional[float] = None,
        tokens: Optional[int] = None
    ):
        """Record a metric observation for a prompt version."""
        if version_hash not in self._metrics:
            self._metrics[version_hash] = {
                "observations": 0,
                "accuracy_sum": 0.0,
                "latency_sum": 0.0,
                "tokens_sum": 0
            }
        
        m = self._metrics[version_hash]
        m["observations"] += 1
        
        if accuracy is not None:
            m["accuracy_sum"] += accuracy
        if latency_ms is not None:
            m["latency_sum"] += latency_ms
        if tokens is not None:
            m["tokens_sum"] += tokens
    
    def get_stats(self, version_hash: str) -> Dict:
        """Get aggregate statistics for a version."""
        m = self._metrics.get(version_hash, {})
        n = m.get("observations", 0)
        
        if n == 0:
            return {"observations": 0}
        
        return {
            "observations": n,
            "avg_accuracy": m["accuracy_sum"] / n,
            "avg_latency_ms": m["latency_sum"] / n,
            "avg_tokens": m["tokens_sum"] / n
        }


# Global instances
prompt_manager = PromptManager()
prompt_metrics = PromptMetrics()

__all__ = [
    "PromptManager",
    "PromptVersion",
    "ABTest",
    "PromptMetrics",
    "prompt_manager",
    "prompt_metrics",
]
