#!/usr/bin/env python3
"""
NUBLE Orchestrator Agent

The master brain powered by Claude Sonnet 4.
Coordinates all specialized agents, integrates with UltimateDecisionEngine,
and synthesizes responses.

This is the core of the multi-agent system.

Architecture:
1. Intent Understanding - Parse user query
2. Decision Engine Check - Use UltimateDecisionEngine for trading decisions
3. Lambda API Data Fetch - Get real-time market intelligence
4. Task Planning - Decompose into agent tasks  
5. Parallel Execution - Run agents concurrently
6. Result Synthesis - Combine outputs
7. Response Generation - Create final response

INTEGRATION NOTES:
- UltimateDecisionEngine provides 28+ data point analysis with weighted scoring
- ML Predictor provides 46M+ parameter model predictions
- Lambda provides real-time data from 40+ endpoints
"""

import asyncio
import json
import re
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import logging

from .base import (
    SpecializedAgent, 
    AgentType, 
    AgentTask, 
    AgentResult,
    TaskPriority,
    _get_anthropic
)

# Import Lambda client for real-time data
try:
    from ..lambda_client import (
        get_lambda_client,
        analyze_symbol,
        format_analysis_for_context,
        extract_symbols as extract_symbols_from_text,
        is_crypto,
        LambdaAnalysis
    )
    LAMBDA_AVAILABLE = True
except ImportError:
    LAMBDA_AVAILABLE = False
    get_lambda_client = None
    analyze_symbol = None
    format_analysis_for_context = None
    extract_symbols_from_text = None

# Import Ultimate Decision Engine for trading decisions
try:
    from ..decision.ultimate_engine import UltimateDecisionEngine
    DECISION_ENGINE_AVAILABLE = True
except ImportError:
    DECISION_ENGINE_AVAILABLE = False
    UltimateDecisionEngine = None

# Import ML Predictor for predictions
try:
    from ...institutional.ml import get_predictor
    ML_PREDICTOR_AVAILABLE = True
except ImportError:
    ML_PREDICTOR_AVAILABLE = False
    get_predictor = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy anthropic import - use _get_anthropic() when needed
HAS_ANTHROPIC = False  # Will be set to True when first loaded

# Claude Models - Using Sonnet 4 (most capable widely available model)
CLAUDE_OPUS_MODEL = "claude-sonnet-4-20250514"
CLAUDE_SONNET_MODEL = "claude-sonnet-4-20250514"


@dataclass
class ConversationContext:
    """Full context for a conversation."""
    conversation_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    active_symbols: List[str] = field(default_factory=list)
    pending_decisions: List[Dict] = field(default_factory=list)
    last_analysis: Optional[Dict] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_recent_messages(self, n: int = 10) -> List[Dict]:
        """Get the n most recent messages."""
        return self.messages[-n:]


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    use_opus: bool = True                    # Use Opus for orchestration
    max_parallel_agents: int = 5             # Max concurrent agents
    default_timeout: int = 30                # Default agent timeout
    enable_caching: bool = True              # Cache agent results
    verbose_logging: bool = False            # Detailed logging
    max_retries: int = 2                     # Retry failed agents
    enable_decision_engine: bool = True      # Use UltimateDecisionEngine
    enable_ml_predictor: bool = True         # Use ML Predictor


class OrchestratorAgent:
    """
    The Master Orchestrator - powered by Claude Sonnet 4.
    
    Now INTEGRATED with:
    - UltimateDecisionEngine (28+ data points, weighted scoring, risk veto)
    - ML Predictor (46M+ parameters across LSTM, Transformer, Ensemble)
    - Lambda Real-Time Data (40+ endpoints)
    
    Responsibilities:
    1. Deep understanding of user intent
    2. Query decomposition into agent tasks
    3. Decision engine integration for trading decisions
    4. Parallel agent coordination
    5. Result synthesis
    6. Response generation
    
    This is the brain of NUBLE.
    
    Example:
        orchestrator = OrchestratorAgent(api_key="...")
        
        result = await orchestrator.process(
            user_message="Should I buy AAPL?",
            conversation_id="conv_123",
            user_context={"portfolio": {"AAPL": 100}}
        )
        
        print(result['message'])
    """
    
    def __init__(
        self, 
        api_key: str = None,
        config: OrchestratorConfig = None
    ):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.config = config or OrchestratorConfig()
        
        # Initialize Claude client - lazy load anthropic
        self.client = None
        anthropic = _get_anthropic()
        if anthropic and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Initialize Lambda client for real-time market intelligence
        self._lambda_client = None
        if LAMBDA_AVAILABLE:
            try:
                self._lambda_client = get_lambda_client()
                logger.info("Lambda Decision Engine connected")
            except Exception as e:
                logger.warning(f"Lambda client init failed: {e}")
        
        # Initialize Ultimate Decision Engine (NEW!)
        self._decision_engine = None
        if self.config.enable_decision_engine and DECISION_ENGINE_AVAILABLE:
            try:
                self._decision_engine = UltimateDecisionEngine()
                logger.info("Ultimate Decision Engine connected")
            except Exception as e:
                logger.warning(f"Decision Engine init failed: {e}")
        
        # Initialize ML Predictor (NEW!)
        self._ml_predictor = None
        if self.config.enable_ml_predictor and ML_PREDICTOR_AVAILABLE:
            try:
                self._ml_predictor = get_predictor()
                logger.info("ML Predictor connected (46M+ parameters)")
            except Exception as e:
                logger.warning(f"ML Predictor init failed: {e}")
        
        # Model selection
        self.orchestrator_model = CLAUDE_OPUS_MODEL if self.config.use_opus else CLAUDE_SONNET_MODEL
        
        # Specialized agents - lazy loaded
        self._agents: Dict[AgentType, SpecializedAgent] = {}
        self._agents_initialized = False
        
        # Conversations
        self.conversations: Dict[str, ConversationContext] = {}
        
        # Cache
        self._cache: Dict[str, Any] = {}
        
        # Live agent output tracking (for API progress events)
        self._last_agent_outputs: Dict[str, Any] = {}
        
        # Learning system integration
        self._learning_hub = None
        try:
            from nuble.learning.learning_hub import LearningHub
            self._learning_hub = LearningHub()
            logger.info("Learning system connected to Orchestrator")
        except Exception:
            logger.info("Learning system not available in Orchestrator")
        
        logger.info(f"OrchestratorAgent initialized with model: {self.orchestrator_model}")
    
    def get_lambda_analysis(self, symbols: List[str]) -> str:
        """
        Fetch real-time analysis from Lambda for given symbols.
        Returns formatted context for LLM prompts.
        """
        if not self._lambda_client or not LAMBDA_AVAILABLE:
            return ""
        
        analyses = []
        for symbol in symbols[:3]:  # Limit to 3 symbols
            try:
                analysis = self._lambda_client.get_analysis(symbol)
                if analysis.action != 'ERROR':
                    analyses.append(format_analysis_for_context(analysis))
                    logger.info(f"Lambda: {symbol} -> {analysis.action} (score: {analysis.score})")
            except Exception as e:
                logger.warning(f"Lambda analysis failed for {symbol}: {e}")
        
        return "\n\n---\n\n".join(analyses) if analyses else ""
    
    def _initialize_agents(self):
        """Initialize all specialized agents (lazy loading)."""
        if self._agents_initialized:
            return
        
        try:
            from .market_analyst import MarketAnalystAgent
            self._agents[AgentType.MARKET_ANALYST] = MarketAnalystAgent(self.api_key)
        except ImportError:
            logger.warning("MarketAnalystAgent not available")
        
        try:
            from .quant_analyst import QuantAnalystAgent
            self._agents[AgentType.QUANT_ANALYST] = QuantAnalystAgent(self.api_key)
        except ImportError:
            logger.warning("QuantAnalystAgent not available")
        
        try:
            from .news_analyst import NewsAnalystAgent
            self._agents[AgentType.NEWS_ANALYST] = NewsAnalystAgent(self.api_key)
        except ImportError:
            logger.warning("NewsAnalystAgent not available")
        
        try:
            from .fundamental_analyst import FundamentalAnalystAgent
            self._agents[AgentType.FUNDAMENTAL_ANALYST] = FundamentalAnalystAgent(self.api_key)
        except ImportError:
            logger.warning("FundamentalAnalystAgent not available")
        
        try:
            from .macro_analyst import MacroAnalystAgent
            self._agents[AgentType.MACRO_ANALYST] = MacroAnalystAgent(self.api_key)
        except ImportError:
            logger.warning("MacroAnalystAgent not available")
        
        try:
            from .risk_manager import RiskManagerAgent
            self._agents[AgentType.RISK_MANAGER] = RiskManagerAgent(self.api_key)
        except ImportError:
            logger.warning("RiskManagerAgent not available")
        
        try:
            from .portfolio_optimizer import PortfolioOptimizerAgent
            self._agents[AgentType.PORTFOLIO_OPTIMIZER] = PortfolioOptimizerAgent(self.api_key)
        except ImportError:
            logger.warning("PortfolioOptimizerAgent not available")
        
        try:
            from .crypto_specialist import CryptoSpecialistAgent
            self._agents[AgentType.CRYPTO_SPECIALIST] = CryptoSpecialistAgent(self.api_key)
        except ImportError:
            logger.warning("CryptoSpecialistAgent not available")
        
        try:
            from .educator import EducatorAgent
            self._agents[AgentType.EDUCATOR] = EducatorAgent(self.api_key)
        except ImportError:
            logger.warning("EducatorAgent not available")
        
        self._agents_initialized = True
        logger.info(f"Initialized {len(self._agents)} specialized agents")
    
    async def process(
        self,
        user_message: str,
        conversation_id: str,
        user_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process a user message through the full pipeline.
        
        This is the main entry point.
        
        Pipeline:
        1. Extract symbols & fetch Lambda data (real-time intelligence)
        2. Intent Understanding (Claude Opus)
        3. Task Planning (decompose into agent tasks)
        4. Parallel Execution (run agents concurrently)
        5. Result Synthesis (combine agent outputs)
        6. Response Generation (Claude Opus)
        
        Args:
            user_message: The user's query
            conversation_id: Unique conversation identifier
            user_context: Optional user context (portfolio, risk tolerance, etc.)
            
        Returns:
            Dictionary with message, data, charts, actions, etc.
        """
        start_time = datetime.now()
        
        # Clear live tracking from previous run
        self._last_agent_outputs = {}
        
        # Initialize agents if needed
        self._initialize_agents()
        
        # Get or create conversation context
        context = self._get_or_create_context(conversation_id, user_context)
        
        # Add user message to history
        context.add_message("user", user_message)
        
        # Extract symbols from message
        symbols = self._extract_symbols(user_message)
        context.active_symbols = list(set(context.active_symbols + symbols))
        
        logger.info(f"Processing: '{user_message[:50]}...' | Symbols: {symbols}")
        
        # STEP 0: Fetch real-time data from Lambda Decision Engine
        lambda_context = ""
        if symbols and LAMBDA_AVAILABLE:
            try:
                lambda_context = self.get_lambda_analysis(symbols)
                if lambda_context:
                    logger.info(f"Lambda data fetched for {len(symbols)} symbol(s)")
            except Exception as e:
                logger.warning(f"Lambda fetch failed: {e}")
        
        # STEP 0.5: Use UltimateDecisionEngine for trading decisions (NEW!)
        decision_engine_result = None
        message_lower = user_message.lower()
        is_trading_query = any(word in message_lower for word in [
            'buy', 'sell', 'trade', 'invest', 'position', 'should i', 
            'predict', 'forecast', 'recommendation', 'what do you think'
        ])
        
        if self._decision_engine and symbols and is_trading_query:
            try:
                # Use the UltimateDecisionEngine for comprehensive analysis
                for symbol in symbols[:2]:  # Limit to 2 symbols for speed
                    logger.info(f"Running UltimateDecisionEngine for {symbol}...")
                    # make_decision is async, so await it directly
                    decision = await self._decision_engine.make_decision(symbol)
                    if decision:
                        # UltimateDecision is a dataclass — access attributes directly
                        decision_engine_result = {
                            'symbol': decision.symbol,
                            'action': decision.direction.value,  # TradeDirection enum → "BUY"/"SELL"/"NEUTRAL"
                            'confidence': decision.confidence,
                            'risk_score': 1.0 - decision.confidence,  # Inverse of confidence as proxy
                            'entry_price': decision.trade_setup.entry if decision.trade_setup else None,
                            'stop_loss': decision.trade_setup.stop_loss if decision.trade_setup else None,
                            'take_profit': decision.trade_setup.targets[0] if decision.trade_setup and decision.trade_setup.targets else None,
                            'position_size': decision.trade_setup.position_pct if decision.trade_setup else None,
                            'reasoning': '; '.join(decision.reasoning) if decision.reasoning else '',
                            'data_sources': list(decision.raw_signals.keys()) if decision.raw_signals else [],
                            'score_breakdown': {
                                'technical': decision.technical_score.score,
                                'intelligence': decision.intelligence_score.score,
                                'market_structure': decision.market_structure_score.score,
                                'validation': decision.validation_score.score,
                            },
                            'luxalgo': decision.technical_score.components.get('luxalgo', {}),
                            'ml_signals': decision.raw_signals.get('ml_signals', {}),
                            'risk_veto': decision.veto,
                            'veto_reason': decision.veto_reason,
                            'strength': decision.strength.value,
                            'data_points_used': decision.data_points_used,
                            'should_trade': decision.should_trade,
                        }
                        logger.info(f"DecisionEngine: {symbol} -> {decision_engine_result['action']} ({decision_engine_result['confidence']:.1%})")
                        break  # Use first symbol's decision as primary
            except Exception as e:
                logger.warning(f"UltimateDecisionEngine failed: {e}")
        
        # STEP 0.7: Get ML Predictions if predictor available (NEW!)
        ml_predictions = {}
        if self._ml_predictor and symbols and is_trading_query:
            try:
                for symbol in symbols[:2]:
                    prediction = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda s=symbol: self._ml_predictor.predict(s)
                    )
                    if prediction:
                        ml_predictions[symbol] = prediction
                        logger.info(f"ML Prediction for {symbol}: {prediction.get('direction', 'N/A')}")
            except Exception as e:
                logger.warning(f"ML Predictor failed: {e}")
        
        # STEP 1: Intent Understanding & Task Planning
        planning_result = await self._plan_execution(user_message, context)
        
        # STEP 2: Execute Agent Tasks (in parallel where possible)
        agent_results = await self._execute_tasks(planning_result['tasks'], context)
        
        # STEP 3: Synthesize Results
        synthesis = await self._synthesize_results(
            user_message,
            planning_result,
            agent_results,
            context
        )
        
        # Add Lambda data to synthesis for response generation
        if lambda_context:
            synthesis['lambda_intelligence'] = lambda_context
        
        # Add Decision Engine results to synthesis (NEW!)
        if decision_engine_result:
            synthesis['decision_engine'] = decision_engine_result
            # If we have a high-confidence decision from the engine, highlight it
            if decision_engine_result.get('confidence', 0) > 0.7:
                synthesis['primary_recommendation'] = {
                    'action': decision_engine_result['action'],
                    'symbol': decision_engine_result['symbol'],
                    'confidence': decision_engine_result['confidence'],
                    'entry': decision_engine_result.get('entry_price'),
                    'stop_loss': decision_engine_result.get('stop_loss'),
                    'take_profit': decision_engine_result.get('take_profit'),
                    'source': 'UltimateDecisionEngine (28+ data points)'
                }
        
        # Add ML predictions to synthesis (NEW!)
        if ml_predictions:
            synthesis['ml_predictions'] = ml_predictions
        
        # STEP 4: Generate Final Response
        response = await self._generate_response(
            user_message,
            synthesis,
            context
        )
        
        # STEP 4.5: Record predictions in Learning System
        if self._learning_hub and synthesis.get('symbols'):
            for symbol in synthesis['symbols'][:3]:
                de = synthesis.get('decision_engine')
                if de and isinstance(de, dict):
                    try:
                        price = de.get('current_price') or de.get('entry_price', 0)
                        if price:
                            self._learning_hub.record_prediction(
                                symbol=symbol,
                                direction=de.get('direction', de.get('action', 'NEUTRAL')),
                                confidence=de.get('confidence', 0.5),
                                price_at_prediction=float(price),
                                source='decision_engine',
                                signal_snapshot={
                                    'technical_score': de.get('technical_score'),
                                    'intelligence_score': de.get('intelligence_score'),
                                    'risk_score': de.get('risk_score'),
                                    'action': de.get('action'),
                                    'regime': de.get('regime', 'UNKNOWN'),
                                    'ml_predictions': synthesis.get('ml_predictions'),
                                    'agents_used': list(synthesis.get('agent_outputs', {}).keys()),
                                },
                            )
                    except Exception:
                        pass  # Learning should never break the main flow
        
        # Update context
        context.add_message("assistant", response['message'])
        context.last_analysis = synthesis
        
        # Calculate timing
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'message': response['message'],
            'data': synthesis,
            'charts': response.get('charts', []),
            'actions': response.get('actions', []),
            'confidence': synthesis.get('overall_confidence', 0),
            'agents_used': [r.agent_type.value for r in agent_results if r.success],
            'execution_time_seconds': execution_time,
            'conversation_id': conversation_id,
            'symbols': symbols
        }
        
        logger.info(f"Completed in {execution_time:.2f}s | Agents: {result['agents_used']}")
        
        return result
    
    async def _plan_execution(
        self,
        user_message: str,
        context: ConversationContext
    ) -> Dict:
        """
        Use Claude Opus to understand intent and plan execution.
        
        This is where the intelligence happens.
        """
        # If no Claude client, use simple rule-based planning
        if not self.client:
            return self._simple_plan(user_message, context)
        
        planning_prompt = self._build_planning_prompt(user_message, context)
        
        try:
            response = self.client.messages.create(
                model=self.orchestrator_model,
                max_tokens=2000,
                messages=[{"role": "user", "content": planning_prompt}]
            )
            
            response_text = response.content[0].text
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                plan = self._simple_plan(user_message, context)
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            plan = self._simple_plan(user_message, context)
        
        # Convert to AgentTask objects
        tasks = []
        for i, task_def in enumerate(plan.get('tasks', [])):
            try:
                agent_type = AgentType[task_def['agent']]
                priority = TaskPriority[task_def.get('priority', 'MEDIUM')]
                
                task = AgentTask(
                    task_id=f"task_{i}",
                    agent_type=agent_type,
                    instruction=task_def['instruction'],
                    context={
                        'symbols': plan.get('symbols', []),
                        'user_profile': context.user_profile,
                        'query': user_message
                    },
                    priority=priority,
                    dependencies=task_def.get('depends_on', [])
                )
                tasks.append(task)
            except (KeyError, ValueError) as e:
                logger.warning(f"Invalid task definition: {e}")
        
        plan['tasks'] = tasks
        return plan
    
    def _build_planning_prompt(
        self,
        user_message: str,
        context: ConversationContext
    ) -> str:
        """Build the planning prompt for Claude."""
        
        portfolio_str = json.dumps(context.user_profile.get('portfolio', {}), indent=2)
        risk_tolerance = context.user_profile.get('risk_tolerance', 'moderate')
        recent_symbols = context.active_symbols[-5:] if context.active_symbols else []
        
        return f"""You are the Orchestrator of NUBLE, the world's most advanced financial AI system.

Your job is to:
1. Deeply understand what the user is asking
2. Determine which specialized agents need to be consulted
3. Create a plan for gathering the information needed

USER MESSAGE: "{user_message}"

USER CONTEXT:
- Portfolio: {portfolio_str}
- Risk Tolerance: {risk_tolerance}
- Recent Topics: {recent_symbols}

AVAILABLE AGENTS:
1. MARKET_ANALYST - Real-time prices, charts, technical analysis, patterns
2. QUANT_ANALYST - ML signals, backtests, factor models, statistical analysis
3. NEWS_ANALYST - Breaking news, sentiment analysis, event detection
4. FUNDAMENTAL_ANALYST - Financial statements, valuations, earnings, SEC filings, TENK SEC Filing RAG (10-K/10-Q semantic search)
5. MACRO_ANALYST - Fed policy, economic indicators, geopolitics, rates
6. RISK_MANAGER - Position risk, portfolio VaR, correlations, stress tests
7. PORTFOLIO_OPTIMIZER - Asset allocation, rebalancing, tax optimization
8. CRYPTO_SPECIALIST - On-chain data, DeFi, protocol analysis, whale tracking
9. EDUCATOR - Explanations, strategies, terminology, examples

NOTE: LuxAlgo Premium Signals (34% weight, multi-timeframe W/D/4H) are automatically 
fetched via the UltimateDecisionEngine and Lambda — no agent assignment needed.
TENK SEC Filing RAG is integrated into FUNDAMENTAL_ANALYST — include it for 10-K/10-Q/SEC queries.

Analyze the user's query and respond with a JSON plan:

{{
    "intent": "brief description of what user wants",
    "complexity": "simple|moderate|complex",
    "symbols": ["list", "of", "symbols"],
    "requires_portfolio_context": true/false,
    "tasks": [
        {{
            "agent": "AGENT_TYPE",
            "instruction": "specific instruction for this agent",
            "priority": "CRITICAL|HIGH|MEDIUM|LOW",
            "depends_on": []
        }}
    ],
    "synthesis_strategy": "how to combine agent outputs"
}}

Guidelines:
- For price checks: use MARKET_ANALYST only
- For "should I buy X": use MARKET_ANALYST + QUANT_ANALYST + NEWS_ANALYST + FUNDAMENTAL_ANALYST
- For portfolio questions: use PORTFOLIO_OPTIMIZER + RISK_MANAGER
- For explanations: use EDUCATOR
- For 10-K/10-Q/SEC filing analysis: ALWAYS include FUNDAMENTAL_ANALYST (has TENK RAG)
- For complex analysis: use 3-5 agents in parallel

Respond with ONLY the JSON plan, no other text.
"""
    
    def _simple_plan(
        self,
        user_message: str,
        context: ConversationContext
    ) -> Dict:
        """Simple rule-based planning when Claude is not available."""
        message_lower = user_message.lower()
        symbols = self._extract_symbols(user_message)
        
        tasks = []
        
        # Determine which agents to use based on keywords
        if any(word in message_lower for word in ['price', 'quote', 'trading', 'chart', 'technical']):
            tasks.append({
                'agent': 'MARKET_ANALYST',
                'instruction': f'Analyze price and technicals for {", ".join(symbols) if symbols else "the market"}',
                'priority': 'HIGH'
            })
        
        if any(word in message_lower for word in ['signal', 'predict', 'forecast', 'ml', 'model']):
            tasks.append({
                'agent': 'QUANT_ANALYST',
                'instruction': f'Get ML signals for {", ".join(symbols) if symbols else "the market"}',
                'priority': 'HIGH'
            })
        
        if any(word in message_lower for word in ['news', 'sentiment', 'headlines', 'event']):
            tasks.append({
                'agent': 'NEWS_ANALYST',
                'instruction': f'Get news and sentiment for {", ".join(symbols) if symbols else "the market"}',
                'priority': 'MEDIUM'
            })
        
        if any(word in message_lower for word in ['buy', 'sell', 'should', 'recommend']):
            if not tasks:
                tasks.append({
                    'agent': 'MARKET_ANALYST',
                    'instruction': f'Analyze {", ".join(symbols) if symbols else "the market"}',
                    'priority': 'CRITICAL'
                })
            tasks.append({
                'agent': 'QUANT_ANALYST',
                'instruction': f'Get ML signals for {", ".join(symbols) if symbols else "the market"}',
                'priority': 'HIGH'
            })
            tasks.append({
                'agent': 'FUNDAMENTAL_ANALYST',
                'instruction': f'Get fundamentals and SEC filing insights for {", ".join(symbols) if symbols else "the market"}',
                'priority': 'HIGH'
            })
        
        if any(word in message_lower for word in ['10-k', '10-q', '10k', '10q', 'sec', 'filing', 'annual report', 'risk factor']):
            if not any(t['agent'] == 'FUNDAMENTAL_ANALYST' for t in tasks):
                tasks.append({
                    'agent': 'FUNDAMENTAL_ANALYST',
                    'instruction': f'Search SEC filings via TENK RAG for {", ".join(symbols) if symbols else "the company"}',
                    'priority': 'HIGH'
                })
        
        if any(word in message_lower for word in ['portfolio', 'allocation', 'rebalance']):
            tasks.append({
                'agent': 'PORTFOLIO_OPTIMIZER',
                'instruction': 'Analyze portfolio allocation',
                'priority': 'HIGH'
            })
        
        if any(word in message_lower for word in ['risk', 'var', 'drawdown', 'exposure']):
            tasks.append({
                'agent': 'RISK_MANAGER',
                'instruction': 'Analyze risk metrics',
                'priority': 'HIGH'
            })
        
        if any(word in message_lower for word in ['explain', 'what is', 'how does', 'teach']):
            tasks.append({
                'agent': 'EDUCATOR',
                'instruction': f'Explain: {user_message}',
                'priority': 'HIGH'
            })
        
        if any(word in message_lower for word in ['crypto', 'bitcoin', 'btc', 'eth', 'defi']):
            tasks.append({
                'agent': 'CRYPTO_SPECIALIST',
                'instruction': f'Analyze crypto: {user_message}',
                'priority': 'HIGH'
            })
        
        # Default: use market analyst
        if not tasks:
            tasks.append({
                'agent': 'MARKET_ANALYST',
                'instruction': f'Answer: {user_message}',
                'priority': 'HIGH'
            })
        
        return {
            'intent': user_message,
            'complexity': 'moderate' if len(tasks) > 1 else 'simple',
            'symbols': symbols,
            'requires_portfolio_context': 'portfolio' in message_lower,
            'tasks': tasks,
            'synthesis_strategy': 'combine all agent outputs'
        }
    
    async def _execute_tasks(
        self,
        tasks: List[AgentTask],
        context: ConversationContext
    ) -> List[AgentResult]:
        """
        Execute agent tasks with dependency awareness.
        
        Runs independent tasks in parallel.
        """
        if not tasks:
            return []
        
        results: Dict[str, AgentResult] = {}
        
        # Group tasks by dependency level
        levels = self._topological_sort(tasks)
        
        for level in levels:
            # Execute all tasks at this level in parallel
            level_tasks = [t for t in tasks if t.task_id in level]
            
            async def execute_single(task: AgentTask) -> AgentResult:
                agent = self._agents.get(task.agent_type)
                if not agent:
                    logger.warning(f"Agent {task.agent_type.value} not available")
                    return AgentResult(
                        task_id=task.task_id,
                        agent_type=task.agent_type,
                        success=False,
                        data={},
                        confidence=0,
                        execution_time_ms=0,
                        error=f"Agent {task.agent_type.value} not available"
                    )
                
                # Add dependency results to context
                dep_results = {
                    dep_id: results[dep_id].data
                    for dep_id in task.dependencies
                    if dep_id in results
                }
                task.context['dependency_results'] = dep_results
                
                start = datetime.now()
                try:
                    result = await asyncio.wait_for(
                        agent.execute(task),
                        timeout=task.timeout_seconds
                    )
                    execution_time = int((datetime.now() - start).total_seconds() * 1000)
                    result.execution_time_ms = execution_time
                    return result
                except asyncio.TimeoutError:
                    logger.warning(f"Agent {task.agent_type.value} timed out")
                    return AgentResult(
                        task_id=task.task_id,
                        agent_type=task.agent_type,
                        success=False,
                        data={},
                        confidence=0,
                        execution_time_ms=task.timeout_seconds * 1000,
                        error="Timeout"
                    )
                except Exception as e:
                    logger.error(f"Agent {task.agent_type.value} failed: {e}")
                    return AgentResult(
                        task_id=task.task_id,
                        agent_type=task.agent_type,
                        success=False,
                        data={},
                        confidence=0,
                        execution_time_ms=int((datetime.now() - start).total_seconds() * 1000),
                        error=str(e)
                    )
            
            # Run level in parallel
            level_results = await asyncio.gather(
                *[execute_single(t) for t in level_tasks],
                return_exceptions=True
            )
            
            # Store results
            for i, result in enumerate(level_results):
                if isinstance(result, Exception):
                    result = AgentResult(
                        task_id=level_tasks[i].task_id,
                        agent_type=level_tasks[i].agent_type,
                        success=False,
                        data={},
                        confidence=0,
                        execution_time_ms=0,
                        error=str(result)
                    )
                results[result.task_id] = result
                # Update live tracking for API progress events
                if result.success:
                    self._last_agent_outputs[result.agent_type.value] = result.data
        
        return list(results.values())
    
    async def _synthesize_results(
        self,
        user_message: str,
        plan: Dict,
        results: List[AgentResult],
        context: ConversationContext
    ) -> Dict:
        """
        Synthesize all agent results into coherent analysis.
        """
        # Organize results by agent type
        by_agent = {}
        for result in results:
            if result.success:
                by_agent[result.agent_type.value] = result.data
        
        # Calculate overall confidence
        confidences = [r.confidence for r in results if r.success]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Collect errors
        errors = [
            {'agent': r.agent_type.value, 'error': r.error}
            for r in results if not r.success
        ]
        
        return {
            'intent': plan.get('intent'),
            'symbols': plan.get('symbols', []),
            'agent_outputs': by_agent,
            'overall_confidence': overall_confidence,
            'errors': errors if errors else None,
            'synthesis_strategy': plan.get('synthesis_strategy'),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _generate_response(
        self,
        user_message: str,
        synthesis: Dict,
        context: ConversationContext
    ) -> Dict:
        """
        Generate the final user-facing response using Claude Opus.
        
        This is where all the intelligence comes together.
        """
        # If no Claude client, generate simple response
        if not self.client:
            return self._simple_response(user_message, synthesis, context)
        
        response_prompt = self._build_response_prompt(user_message, synthesis, context)
        
        try:
            response = self.client.messages.create(
                model=self.orchestrator_model,
                max_tokens=4000,
                messages=[{"role": "user", "content": response_prompt}]
            )
            
            message = response.content[0].text
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._simple_response(user_message, synthesis, context)
        
        # Generate charts based on analysis
        charts = self._generate_charts(synthesis)
        
        # Generate action buttons
        actions = self._generate_actions(synthesis, user_message)
        
        return {
            'message': message,
            'charts': charts,
            'actions': actions
        }
    
    def _build_response_prompt(
        self,
        user_message: str,
        synthesis: Dict,
        context: ConversationContext
    ) -> str:
        """Build the response generation prompt."""
        
        risk_tolerance = context.user_profile.get('risk_tolerance', 'moderate')
        portfolio = context.user_profile.get('portfolio', {})
        agent_outputs = json.dumps(synthesis['agent_outputs'], indent=2, default=str)
        
        # Get Lambda real-time intelligence if available
        lambda_intel = synthesis.get('lambda_intelligence', '')
        lambda_section = ""
        if lambda_intel:
            lambda_section = f"""

REAL-TIME MARKET INTELLIGENCE (from NUBLE Decision Engine):
{lambda_intel}

This includes live data from:
- StockNews API: sentiment, analyst ratings, earnings, SEC filings, events, price targets
- CryptoNews API: sentiment, whale activity, institutional flows, regulatory, staking
- Polygon.io: prices, volume, technicals (RSI, MACD, SMA, ATR, VIX)
"""

        # Get UltimateDecisionEngine results (NEW!)
        decision_engine_section = ""
        decision_engine = synthesis.get('decision_engine')
        if decision_engine:
            # Extract LuxAlgo signals for explicit surfacing
            luxalgo = decision_engine.get('luxalgo', {})
            luxalgo_section = ""
            if luxalgo:
                aligned = luxalgo.get('aligned', False)
                aligned_label = "✅ ALL TIMEFRAMES ALIGNED" if aligned else "⚠️ Mixed signals"
                luxalgo_section = f"""
- LuxAlgo Premium Signals: {aligned_label}
  - Weekly (1W): {luxalgo.get('weekly', 'N/A')} (highest conviction)
  - Daily (1D): {luxalgo.get('daily', 'N/A')}
  - 4-Hour (4H): {luxalgo.get('h4', 'N/A')} (most responsive)
  - Direction: {luxalgo.get('direction', 'N/A')}
  - Score: {luxalgo.get('score', 0):.3f}"""
            
            decision_engine_section = f"""

ULTIMATE DECISION ENGINE ANALYSIS (28+ data points):
- Symbol: {decision_engine.get('symbol', 'N/A')}
- Recommended Action: {decision_engine.get('action', 'N/A')}
- Confidence: {decision_engine.get('confidence', 0):.1%}
- Risk Score: {decision_engine.get('risk_score', 0.5):.2f}
- Entry Price: {decision_engine.get('entry_price', 'N/A')}
- Stop Loss: {decision_engine.get('stop_loss', 'N/A')}
- Take Profit: {decision_engine.get('take_profit', 'N/A')}
- Position Size: {decision_engine.get('position_size', 'N/A')}
- Risk Veto: {'YES - HIGH RISK' if decision_engine.get('risk_veto') else 'No'}
- Reasoning: {decision_engine.get('reasoning', 'N/A')}
- Data Sources Used: {', '.join(decision_engine.get('data_sources', []))}
{luxalgo_section}
"""

        # Get ML predictions (NEW!)
        ml_section = ""
        ml_predictions = synthesis.get('ml_predictions', {})
        if ml_predictions:
            ml_parts = ["", "ML MODEL PREDICTIONS (46M+ parameters):"]
            for symbol, pred in ml_predictions.items():
                ml_parts.append(f"- {symbol}:")
                ml_parts.append(f"  - Direction: {pred.get('direction', 'N/A')}")
                ml_parts.append(f"  - Confidence: {pred.get('confidence', 0):.1%}")
                ml_parts.append(f"  - Price Target: {pred.get('price_target', 'N/A')}")
                ml_parts.append(f"  - Model: {pred.get('model', 'ensemble')}")
            ml_section = "\n".join(ml_parts)

        return f"""You are NUBLE, the world's most intelligent financial advisor.

You have access to comprehensive analysis from multiple specialized agents AND real-time market intelligence.
Your job is to synthesize this into an excellent, actionable response.

USER QUESTION: "{user_message}"

USER PROFILE:
- Risk Tolerance: {risk_tolerance}
- Portfolio: {json.dumps(portfolio, indent=2) if portfolio else 'Not provided'}
{decision_engine_section}
{ml_section}
{lambda_section}
AGENT ANALYSIS:
{agent_outputs}

OVERALL CONFIDENCE: {synthesis['overall_confidence']:.1%}

---

Generate a response that:

1. DIRECTLY answers the user's question first
2. If the ULTIMATE DECISION ENGINE provided a recommendation, LEAD with that - it's your most sophisticated analysis
3. If LuxAlgo signals are present, highlight alignment or divergence — they carry 34% weight. When all timeframes align, declare HIGH CONVICTION.
4. Uses the REAL-TIME data from the Lambda Decision Engine - you HAVE access to current market data
5. Incorporates ML model predictions when available (these come from 46M+ parameters across LSTM, Transformer, Ensemble)
6. If TENK SEC filing data is present in agent outputs (sec_filing_insights), incorporate risk factors, revenue breakdowns, management outlook
7. Provides specific, actionable insights with actual numbers from the data
8. Includes current prices, sentiment scores, analyst actions from the real-time feed
9. Is personalized to the user's risk tolerance and portfolio (if known)
10. Includes specific numbers, prices, percentages from the data
11. Suggests next steps or follow-up questions

IMPORTANT GUIDELINES:
- If Decision Engine says RISK VETO, ALWAYS mention the high risk warning prominently
- If LuxAlgo shows all timeframes aligned, treat it as a strong confirmation signal
- Be confident but honest about uncertainty
- Don't hedge excessively - give clear recommendations when the data supports it
- Use bullet points for key takeaways
- Include risk warnings where appropriate
- Make it conversational, not robotic
- If recommending a position, include entry, stop-loss, and target levels from the Decision Engine

Format your response in Markdown for best readability.
"""
    
    def _simple_response(
        self,
        user_message: str,
        synthesis: Dict,
        context: ConversationContext
    ) -> Dict:
        """Generate a simple response without Claude."""
        
        agent_outputs = synthesis.get('agent_outputs', {})
        symbols = synthesis.get('symbols', [])
        lambda_intel = synthesis.get('lambda_intelligence', '')
        decision_engine = synthesis.get('decision_engine')
        ml_predictions = synthesis.get('ml_predictions', {})
        
        # Build response from agent data
        parts = []
        
        parts.append(f"## Analysis for: {', '.join(symbols) if symbols else 'Your Query'}\n")
        
        # Include Decision Engine results first (most valuable) - NEW!
        if decision_engine:
            parts.append("\n### 🎯 Ultimate Decision Engine Recommendation\n")
            parts.append(f"**Symbol:** {decision_engine.get('symbol', 'N/A')}")
            parts.append(f"**Recommended Action:** {decision_engine.get('action', 'N/A')}")
            parts.append(f"**Confidence:** {decision_engine.get('confidence', 0):.1%}")
            if decision_engine.get('risk_veto'):
                parts.append("\n⚠️ **RISK VETO ACTIVE** - High risk conditions detected!")
            if decision_engine.get('entry_price') is not None:
                parts.append(f"\n**Entry Price:** ${decision_engine.get('entry_price'):.2f}")
            if decision_engine.get('stop_loss') is not None:
                parts.append(f"**Stop Loss:** ${decision_engine.get('stop_loss'):.2f}")
            if decision_engine.get('take_profit') is not None:
                parts.append(f"**Take Profit:** ${decision_engine.get('take_profit'):.2f}")
            if decision_engine.get('reasoning'):
                parts.append(f"\n**Reasoning:** {decision_engine.get('reasoning')}")
            parts.append("\n")
        
        # Include ML Predictions - NEW!
        if ml_predictions:
            parts.append("\n### 🤖 ML Model Predictions (46M+ Parameters)\n")
            for symbol, pred in ml_predictions.items():
                parts.append(f"**{symbol}:**")
                parts.append(f"  - Direction: {pred.get('direction', 'N/A')}")
                parts.append(f"  - Confidence: {pred.get('confidence', 0):.1%}")
                if pred.get('price_target'):
                    parts.append(f"  - Price Target: ${pred.get('price_target'):.2f}")
            parts.append("\n")
        
        # Include Lambda intelligence
        if lambda_intel:
            parts.append("\n### 📊 Real-Time Market Intelligence\n")
            parts.append(lambda_intel)
            parts.append("\n")
        
        for agent, data in agent_outputs.items():
            parts.append(f"\n### {agent.replace('_', ' ').title()}\n")
            
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        parts.append(f"- **{key}**: {value:.4f}" if isinstance(value, float) else f"- **{key}**: {value}")
                    elif isinstance(value, str):
                        parts.append(f"- **{key}**: {value}")
        
        if not agent_outputs and not lambda_intel:
            parts.append("\nI wasn't able to gather specific data for your query.")
            parts.append("Please try rephrasing or specify a stock symbol.")
        
        parts.append(f"\n\n*Confidence: {synthesis['overall_confidence']:.1%}*")
        
        return {
            'message': '\n'.join(parts),
            'charts': self._generate_charts(synthesis),
            'actions': self._generate_actions(synthesis, user_message)
        }
    
    def _topological_sort(self, tasks: List[AgentTask]) -> List[List[str]]:
        """Sort tasks by dependency level for parallel execution."""
        # Build dependency graph
        task_deps = {t.task_id: set(t.dependencies) for t in tasks}
        
        levels = []
        remaining = set(task_deps.keys())
        
        while remaining:
            # Find tasks with no remaining dependencies
            ready = {
                t for t in remaining
                if not task_deps[t] & remaining
            }
            
            if not ready:
                # Circular dependency - just take remaining
                ready = remaining
            
            levels.append(list(ready))
            remaining -= ready
        
        return levels
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock/crypto symbols from text using the robust lambda_client extractor."""
        if LAMBDA_AVAILABLE and extract_symbols_from_text:
            return extract_symbols_from_text(text)
        
        # Fallback: simple pattern-based extraction
        patterns = [
            r'\$([A-Z]{1,5})\b',      # $AAPL format
            r'\b([A-Z]{2,5})\b',       # Uppercase 2-5 letters
        ]
        
        symbols = set()
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            symbols.update(matches)
        
        # Filter out common words
        common_words = {
            'I', 'A', 'THE', 'AND', 'OR', 'FOR', 'TO', 'IN', 'ON', 'AT', 'IS', 'IT', 'MY',
            'OF', 'BE', 'ARE', 'WAS', 'AN', 'AS', 'IF', 'DO', 'SO', 'UP', 'WE', 'BY',
            'WHAT', 'WHEN', 'HOW', 'WHY', 'WHO', 'CAN', 'WILL', 'ALL', 'GET', 'HAS',
            'BUY', 'SELL', 'NOW', 'TODAY', 'GOOD', 'BAD', 'YES', 'NO', 'NOT', 'THIS',
            'THAT', 'WITH', 'FROM', 'THEY', 'HAVE', 'BEEN', 'WOULD', 'COULD', 'SHOULD'
        }
        symbols -= common_words
        
        # Known valid symbols
        valid_symbols = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',
            'AMD', 'INTC', 'CRM', 'ORCL', 'IBM', 'CSCO', 'ADBE', 'PYPL',
            'NFLX', 'DIS', 'NKE', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
            'XOM', 'CVX', 'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'LLY',
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO',
            'BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'ADA',
            'GLD', 'SLV', 'USO', 'TLT', 'HYG'
        }
        
        # Return intersection with valid symbols, or all if none match
        matched = symbols & valid_symbols
        return list(matched) if matched else list(symbols)[:5]
    
    def _get_or_create_context(
        self,
        conversation_id: str,
        user_context: Optional[Dict]
    ) -> ConversationContext:
        """Get or create conversation context."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationContext(
                conversation_id=conversation_id,
                user_profile=user_context or {}
            )
        
        # Update user profile if provided
        if user_context:
            self.conversations[conversation_id].user_profile.update(user_context)
        
        return self.conversations[conversation_id]
    
    def _generate_charts(self, synthesis: Dict) -> List[Dict]:
        """Generate chart configurations from synthesis."""
        charts = []
        
        agent_outputs = synthesis.get('agent_outputs', {})
        
        # Price chart from market analyst
        if 'market_analyst' in agent_outputs:
            market = agent_outputs['market_analyst']
            if 'price_history' in market:
                charts.append({
                    'type': 'candlestick',
                    'title': 'Price Chart',
                    'data': market['price_history'],
                    'indicators': market.get('indicators', [])
                })
        
        # Sentiment chart from news analyst
        if 'news_analyst' in agent_outputs:
            news = agent_outputs['news_analyst']
            if 'sentiment_timeline' in news:
                charts.append({
                    'type': 'line',
                    'title': 'Sentiment Over Time',
                    'data': news['sentiment_timeline']
                })
        
        return charts
    
    def _generate_actions(self, synthesis: Dict, query: str) -> List[Dict]:
        """Generate actionable buttons."""
        actions = []
        
        symbols = synthesis.get('symbols', [])
        
        if symbols:
            symbol = symbols[0]
            actions.append({
                'label': f'Add {symbol} to Watchlist',
                'action': 'add_watchlist',
                'params': {'symbol': symbol}
            })
            
            actions.append({
                'label': 'Set Price Alert',
                'action': 'set_alert',
                'params': {'symbol': symbol}
            })
            
            actions.append({
                'label': 'Deep Dive Analysis',
                'action': 'deep_analysis',
                'params': {'symbol': symbol}
            })
        
        return actions
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents."""
        self._initialize_agents()
        return [agent.value for agent in self._agents.keys()]
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)
    
    def clear_conversation(self, conversation_id: str):
        """Clear a conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]


# Export for convenience
__all__ = [
    'OrchestratorAgent',
    'OrchestratorConfig',
    'ConversationContext',
    'AgentType',
    'AgentTask',
    'AgentResult',
    'TaskPriority'
]
