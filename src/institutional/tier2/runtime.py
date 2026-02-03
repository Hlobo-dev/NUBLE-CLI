"""
Tier 2 Agent Runtime
=====================

Executes agents against AWS Bedrock (Claude) with:
- Token budgets
- Timeouts
- Parallel execution
- JSON validation
- Retry logic
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

try:
    import boto3
    from botocore.config import Config as BotoConfig
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

from .config import Tier2Config, AgentConfig, DEFAULT_CONFIG
from .schemas import AgentOutput, Claim, RecommendedDeltas, Tier1DecisionPack
from .agents import BaseAgent, AgentContext, create_agent
from .registry import AgentRegistry, get_default_registry

logger = logging.getLogger(__name__)


@dataclass
class RuntimeMetrics:
    """Metrics from agent runtime execution."""
    agent_name: str
    mode: str  # light or deep
    latency_ms: float
    tokens_input: int
    tokens_output: int
    model: str
    success: bool
    error: Optional[str] = None


@dataclass
class RuntimeResult:
    """Result from running an agent."""
    output: AgentOutput
    metrics: RuntimeMetrics
    raw_response: Optional[str] = None


class AgentRuntime:
    """
    Runtime for executing Tier 2 agents against Bedrock.
    
    Features:
    - Parallel agent execution
    - Token budget enforcement
    - Timeout management
    - JSON validation
    - Retry with backoff
    """
    
    def __init__(
        self,
        config: Tier2Config = None,
        registry: AgentRegistry = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.registry = registry or get_default_registry()
        
        # Bedrock client
        self._bedrock_client = None
        self._executor = ThreadPoolExecutor(max_workers=12)
        
        # Metrics
        self.last_run_metrics: List[RuntimeMetrics] = []
    
    @property
    def bedrock_client(self):
        """Lazy-load Bedrock client."""
        if self._bedrock_client is None and HAS_BOTO3:
            boto_config = BotoConfig(
                connect_timeout=5,
                read_timeout=30,
                retries={'max_attempts': self.config.bedrock_max_retries}
            )
            self._bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=self.config.bedrock_region,
                config=boto_config,
            )
        return self._bedrock_client
    
    def _invoke_bedrock(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        max_tokens: int,
    ) -> Tuple[str, int, int]:
        """
        Invoke Bedrock Claude model.
        
        Returns: (response_text, input_tokens, output_tokens)
        """
        if not self.bedrock_client:
            # Fallback for testing without AWS
            return self._mock_response(system_prompt, user_prompt), 0, 0
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,  # Deterministic for consistency
        }
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            
            result = json.loads(response['body'].read())
            
            text = result['content'][0]['text']
            input_tokens = result.get('usage', {}).get('input_tokens', 0)
            output_tokens = result.get('usage', {}).get('output_tokens', 0)
            
            return text, input_tokens, output_tokens
            
        except Exception as e:
            logger.error(f"Bedrock invocation error: {e}")
            raise
    
    def _mock_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate mock response for testing."""
        # Extract agent name from system prompt
        agent_name = "unknown"
        if "MTF" in system_prompt:
            agent_name = "mtf_dominance"
        elif "Trend Integrity" in system_prompt:
            agent_name = "trend_integrity"
        elif "RISK GATEKEEPER" in system_prompt:
            agent_name = "risk_gatekeeper"
        elif "Red Team" in system_prompt:
            agent_name = "red_team"
        elif "Volatility" in system_prompt:
            agent_name = "volatility_state"
        
        return json.dumps({
            "agent": agent_name,
            "mode": "light",
            "verdict": "NEUTRAL",
            "confidence": 0.5,
            "claims": [
                {
                    "id": f"{agent_name.upper()}_01",
                    "type": "risk",
                    "stance": "neutral",
                    "strength": 0.5,
                    "statement": "Mock claim for testing",
                    "evidence_keys": ["rsi", "price"],
                    "conditions": [],
                }
            ],
            "recommended_deltas": {
                "confidence_delta": 0,
                "position_pct_cap": None,
                "wait_minutes": 0,
                "risk_posture": "normal",
            }
        })
    
    def run_agent_sync(
        self,
        agent_name: str,
        context: AgentContext,
    ) -> RuntimeResult:
        """
        Run a single agent synchronously.
        
        Args:
            agent_name: Name of the agent to run
            context: Context including Tier1 pack and escalation info
            
        Returns:
            RuntimeResult with output and metrics
        """
        start_time = time.time()
        
        # Get agent
        agent = self.registry.get_agent(agent_name)
        if not agent:
            return RuntimeResult(
                output=AgentOutput(
                    agent=agent_name,
                    mode="light",
                    verdict="NEUTRAL",
                    confidence=0.0,
                    claims=[],
                    recommended_deltas=RecommendedDeltas(),
                    valid=False,
                    validation_errors=[f"Agent {agent_name} not found"],
                ),
                metrics=RuntimeMetrics(
                    agent_name=agent_name,
                    mode="light",
                    latency_ms=0,
                    tokens_input=0,
                    tokens_output=0,
                    model="",
                    success=False,
                    error="Agent not found",
                ),
            )
        
        # Get config
        agent_config = self.registry.get_config(agent_name)
        if not agent_config:
            agent_config = AgentConfig(name=agent_name)
        
        # Determine mode and tokens
        mode = "deep" if context.is_deep else "light"
        max_tokens = agent_config.deep_max_tokens if context.is_deep else agent_config.light_max_tokens
        model_id = agent_config.bedrock_model_deep if context.is_deep else agent_config.bedrock_model_light
        
        # Get prompts
        system_prompt = agent.get_system_prompt()
        user_prompt = agent.get_prompt(context)
        
        try:
            # Invoke Bedrock
            response_text, input_tokens, output_tokens = self._invoke_bedrock(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_id=model_id,
                max_tokens=max_tokens,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse response
            output = self._parse_response(response_text, agent_name, mode)
            output.latency_ms = latency_ms
            output.tokens_used = output_tokens
            output.model = model_id
            
            # Validate claims against available evidence
            evidence_keys = context.tier1_pack.get_evidence_keys()
            output.validate_claims(evidence_keys)
            
            return RuntimeResult(
                output=output,
                metrics=RuntimeMetrics(
                    agent_name=agent_name,
                    mode=mode,
                    latency_ms=latency_ms,
                    tokens_input=input_tokens,
                    tokens_output=output_tokens,
                    model=model_id,
                    success=output.valid,
                ),
                raw_response=response_text if self.config.log_raw_llm_output else None,
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Agent {agent_name} failed: {e}")
            
            return RuntimeResult(
                output=AgentOutput(
                    agent=agent_name,
                    mode=mode,
                    verdict="NEUTRAL",
                    confidence=0.0,
                    claims=[],
                    recommended_deltas=RecommendedDeltas(),
                    valid=False,
                    validation_errors=[str(e)],
                ),
                metrics=RuntimeMetrics(
                    agent_name=agent_name,
                    mode=mode,
                    latency_ms=latency_ms,
                    tokens_input=0,
                    tokens_output=0,
                    model=model_id,
                    success=False,
                    error=str(e),
                ),
            )
    
    def _parse_response(self, text: str, agent_name: str, mode: str) -> AgentOutput:
        """Parse agent response text into AgentOutput."""
        # Try to find JSON in the response
        text = text.strip()
        
        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```json") or line.startswith("```"):
                    in_json = not in_json
                    continue
                if in_json:
                    json_lines.append(line)
            text = "\n".join(json_lines)
        
        # Try to parse
        try:
            data = json.loads(text)
            output = AgentOutput.from_dict(data)
            output.agent = agent_name
            output.mode = mode
            return output
            
        except json.JSONDecodeError as e:
            return AgentOutput(
                agent=agent_name,
                mode=mode,
                verdict="NEUTRAL",
                confidence=0.0,
                claims=[],
                recommended_deltas=RecommendedDeltas(),
                valid=False,
                validation_errors=[f"JSON parse error: {str(e)}"],
            )
    
    async def run_agents_parallel(
        self,
        agent_names: List[str],
        context: AgentContext,
    ) -> List[RuntimeResult]:
        """
        Run multiple agents in parallel.
        
        Args:
            agent_names: List of agent names to run
            context: Shared context for all agents
            
        Returns:
            List of RuntimeResults in same order as agent_names
        """
        loop = asyncio.get_event_loop()
        
        # Create tasks
        futures = [
            loop.run_in_executor(
                self._executor,
                self.run_agent_sync,
                name,
                context,
            )
            for name in agent_names
        ]
        
        # Execute with timeout
        timeout = self.config.max_tier2_latency_ms / 1000
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Agent execution timed out after {timeout}s")
            results = []
            for name in agent_names:
                results.append(RuntimeResult(
                    output=AgentOutput(
                        agent=name,
                        mode=context.is_deep and "deep" or "light",
                        verdict="NEUTRAL",
                        confidence=0.0,
                        claims=[],
                        recommended_deltas=RecommendedDeltas(),
                        valid=False,
                        validation_errors=["Timeout"],
                    ),
                    metrics=RuntimeMetrics(
                        agent_name=name,
                        mode=context.is_deep and "deep" or "light",
                        latency_ms=timeout * 1000,
                        tokens_input=0,
                        tokens_output=0,
                        model="",
                        success=False,
                        error="Timeout",
                    ),
                ))
        
        # Process results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(RuntimeResult(
                    output=AgentOutput(
                        agent=agent_names[i],
                        mode=context.is_deep and "deep" or "light",
                        verdict="NEUTRAL",
                        confidence=0.0,
                        claims=[],
                        recommended_deltas=RecommendedDeltas(),
                        valid=False,
                        validation_errors=[str(result)],
                    ),
                    metrics=RuntimeMetrics(
                        agent_name=agent_names[i],
                        mode=context.is_deep and "deep" or "light",
                        latency_ms=0,
                        tokens_input=0,
                        tokens_output=0,
                        model="",
                        success=False,
                        error=str(result),
                    ),
                ))
            else:
                final_results.append(result)
        
        # Store metrics
        self.last_run_metrics = [r.metrics for r in final_results]
        
        return final_results
    
    def run_agents_sync(
        self,
        agent_names: List[str],
        context: AgentContext,
    ) -> List[RuntimeResult]:
        """Synchronous wrapper for run_agents_parallel."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.run_agents_parallel(agent_names, context)
            )
        finally:
            loop.close()
