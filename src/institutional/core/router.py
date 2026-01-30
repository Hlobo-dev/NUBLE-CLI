"""
Data Router - Intelligent routing of data requests to appropriate providers.
Maps data needs to providers and handles fallback logic.
"""

from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from .intent_engine import DataNeed, ParsedIntent, QueryCategory
from ..config import get_config, ProviderPriority


@dataclass
class ProviderCapability:
    """Defines what a provider can deliver"""
    provider_name: str
    data_needs: Set[DataNeed]
    quality_score: float = 1.0  # 0-1, higher is better
    latency_ms: int = 100  # Typical response time
    cost_per_call: float = 0.0  # For cost optimization
    rate_limit_weight: int = 1  # How much it counts against rate limit


@dataclass
class DataRequest:
    """A request to be sent to a provider"""
    provider: str
    data_need: DataNeed
    symbols: List[str]
    params: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout_ms: int = 30000


@dataclass
class RoutingPlan:
    """Complete routing plan for a query"""
    intent: ParsedIntent
    requests: List[DataRequest]
    parallel_groups: List[List[DataRequest]]  # Requests that can run in parallel
    estimated_time_ms: int = 0
    estimated_cost: float = 0.0


# Alias for backward compatibility
RoutingResult = RoutingPlan


class DataRouter:
    """
    Routes data requests to the most appropriate providers.
    Handles provider capabilities, fallbacks, and optimization.
    """
    
    # Provider capabilities mapping
    PROVIDER_CAPABILITIES = {
        "polygon": ProviderCapability(
            provider_name="polygon",
            data_needs={
                DataNeed.REALTIME_QUOTE,
                DataNeed.HISTORICAL_PRICES,
                DataNeed.INTRADAY_BARS,
                DataNeed.TICK_DATA,
                DataNeed.OPTIONS_CHAIN,
                DataNeed.IMPLIED_VOLATILITY,
                DataNeed.GREEKS,
                DataNeed.OPTIONS_FLOW,
                DataNeed.UNUSUAL_ACTIVITY,
                DataNeed.DIVIDENDS,
            },
            quality_score=0.95,
            latency_ms=50,
            cost_per_call=0.0001
        ),
        "alpha_vantage": ProviderCapability(
            provider_name="alpha_vantage",
            data_needs={
                DataNeed.REALTIME_QUOTE,
                DataNeed.HISTORICAL_PRICES,
                DataNeed.INTRADAY_BARS,
                DataNeed.TECHNICAL_INDICATORS,
                DataNeed.INCOME_STATEMENT,
                DataNeed.BALANCE_SHEET,
                DataNeed.CASH_FLOW,
                DataNeed.KEY_RATIOS,
                DataNeed.EARNINGS,
                DataNeed.NEWS,
                DataNeed.SENTIMENT_SCORES,
                DataNeed.ECONOMIC_CALENDAR,
                DataNeed.MACRO_INDICATORS,
                DataNeed.TREASURY_RATES,
            },
            quality_score=0.85,
            latency_ms=200,
            cost_per_call=0.0002
        ),
        "finnhub": ProviderCapability(
            provider_name="finnhub",
            data_needs={
                DataNeed.REALTIME_QUOTE,
                DataNeed.HISTORICAL_PRICES,
                DataNeed.NEWS,
                DataNeed.SENTIMENT_SCORES,
                DataNeed.SOCIAL_SENTIMENT,
                DataNeed.COMPANY_PROFILE,
                DataNeed.PEERS,
                DataNeed.KEY_RATIOS,
                DataNeed.EARNINGS,
                DataNeed.INSIDER_TRADES,
                DataNeed.INSIDER_SENTIMENT,
                DataNeed.INSTITUTIONAL_HOLDINGS,
                DataNeed.FUND_HOLDINGS,
                DataNeed.CONGRESSIONAL_TRADES,
                DataNeed.SUPPLY_CHAIN,
                DataNeed.PRESS_RELEASES,
                DataNeed.ECONOMIC_CALENDAR,
            },
            quality_score=0.90,
            latency_ms=100,
            cost_per_call=0.0001
        ),
        "sec_edgar": ProviderCapability(
            provider_name="sec_edgar",
            data_needs={
                DataNeed.SEC_FILINGS,
                DataNeed.FORM_10K,
                DataNeed.FORM_10Q,
                DataNeed.FORM_8K,
                DataNeed.FORM_13F,
                DataNeed.FORM_4,
                DataNeed.INSTITUTIONAL_HOLDINGS,
                DataNeed.INSIDER_TRADES,
                DataNeed.INCOME_STATEMENT,
                DataNeed.BALANCE_SHEET,
                DataNeed.CASH_FLOW,
            },
            quality_score=1.0,  # Authoritative source
            latency_ms=500,
            cost_per_call=0.0
        ),
        "yfinance": ProviderCapability(
            provider_name="yfinance",
            data_needs={
                DataNeed.REALTIME_QUOTE,
                DataNeed.HISTORICAL_PRICES,
                DataNeed.OPTIONS_CHAIN,
                DataNeed.DIVIDENDS,
                DataNeed.KEY_RATIOS,
                DataNeed.COMPANY_PROFILE,
                DataNeed.NEWS,
                DataNeed.INCOME_STATEMENT,
                DataNeed.BALANCE_SHEET,
                DataNeed.CASH_FLOW,
            },
            quality_score=0.70,
            latency_ms=300,
            cost_per_call=0.0
        ),
        "computed": ProviderCapability(
            provider_name="computed",
            data_needs={
                DataNeed.TECHNICAL_INDICATORS,
                DataNeed.CHART_PATTERNS,
                DataNeed.SUPPORT_RESISTANCE,
                DataNeed.VOLUME_ANALYSIS,
                DataNeed.POSITION_CHANGES,
            },
            quality_score=0.95,
            latency_ms=100,
            cost_per_call=0.0
        ),
    }
    
    # Priority order for each data need (first available wins)
    DATA_NEED_PRIORITY = {
        # Quotes - Real-time first
        DataNeed.REALTIME_QUOTE: ["polygon", "finnhub", "alpha_vantage", "yfinance"],
        DataNeed.HISTORICAL_PRICES: ["polygon", "alpha_vantage", "yfinance"],
        DataNeed.INTRADAY_BARS: ["polygon", "alpha_vantage", "yfinance"],
        DataNeed.TICK_DATA: ["polygon"],
        
        # Options - Polygon is best
        DataNeed.OPTIONS_CHAIN: ["polygon", "yfinance"],
        DataNeed.OPTIONS_FLOW: ["polygon"],
        DataNeed.IMPLIED_VOLATILITY: ["polygon", "yfinance"],
        DataNeed.GREEKS: ["polygon", "yfinance"],
        DataNeed.UNUSUAL_ACTIVITY: ["polygon", "computed"],
        
        # Technical - Computed preferred, Alpha Vantage for standard
        DataNeed.TECHNICAL_INDICATORS: ["computed", "alpha_vantage"],
        DataNeed.CHART_PATTERNS: ["computed"],
        DataNeed.SUPPORT_RESISTANCE: ["computed"],
        DataNeed.VOLUME_ANALYSIS: ["computed"],
        
        # Fundamentals - Multiple sources
        DataNeed.INCOME_STATEMENT: ["sec_edgar", "alpha_vantage", "finnhub", "yfinance"],
        DataNeed.BALANCE_SHEET: ["sec_edgar", "alpha_vantage", "finnhub", "yfinance"],
        DataNeed.CASH_FLOW: ["sec_edgar", "alpha_vantage", "finnhub", "yfinance"],
        DataNeed.KEY_RATIOS: ["finnhub", "alpha_vantage", "yfinance"],
        DataNeed.EARNINGS: ["finnhub", "alpha_vantage", "yfinance"],
        DataNeed.DIVIDENDS: ["polygon", "alpha_vantage", "yfinance"],
        
        # Institutional - SEC is authoritative
        DataNeed.INSTITUTIONAL_HOLDINGS: ["sec_edgar", "finnhub"],
        DataNeed.FUND_HOLDINGS: ["sec_edgar", "finnhub"],
        DataNeed.POSITION_CHANGES: ["computed"],  # Computed from 13F history
        
        # Insider
        DataNeed.INSIDER_TRADES: ["sec_edgar", "finnhub"],
        DataNeed.INSIDER_SENTIMENT: ["finnhub"],
        DataNeed.CONGRESSIONAL_TRADES: ["finnhub"],
        
        # Regulatory - SEC only
        DataNeed.SEC_FILINGS: ["sec_edgar"],
        DataNeed.FORM_10K: ["sec_edgar"],
        DataNeed.FORM_10Q: ["sec_edgar"],
        DataNeed.FORM_8K: ["sec_edgar"],
        DataNeed.FORM_13F: ["sec_edgar"],
        DataNeed.FORM_4: ["sec_edgar"],
        
        # News & Sentiment
        DataNeed.NEWS: ["finnhub", "alpha_vantage", "yfinance"],
        DataNeed.SENTIMENT_SCORES: ["finnhub", "alpha_vantage", "computed"],
        DataNeed.SOCIAL_SENTIMENT: ["finnhub"],
        DataNeed.PRESS_RELEASES: ["finnhub", "sec_edgar"],
        
        # Company Info
        DataNeed.COMPANY_PROFILE: ["finnhub", "yfinance", "alpha_vantage"],
        DataNeed.PEERS: ["finnhub"],
        DataNeed.SUPPLY_CHAIN: ["finnhub"],
        
        # Economic
        DataNeed.ECONOMIC_CALENDAR: ["finnhub", "alpha_vantage"],
        DataNeed.MACRO_INDICATORS: ["alpha_vantage"],
        DataNeed.TREASURY_RATES: ["alpha_vantage"],
    }
    
    # Data needs that can be fetched in parallel
    PARALLEL_GROUPS = [
        # Price data group
        {DataNeed.REALTIME_QUOTE, DataNeed.HISTORICAL_PRICES, DataNeed.INTRADAY_BARS},
        # Fundamentals group
        {DataNeed.INCOME_STATEMENT, DataNeed.BALANCE_SHEET, DataNeed.CASH_FLOW, DataNeed.KEY_RATIOS},
        # News/Sentiment group
        {DataNeed.NEWS, DataNeed.SENTIMENT_SCORES, DataNeed.SOCIAL_SENTIMENT},
        # Institutional group
        {DataNeed.INSTITUTIONAL_HOLDINGS, DataNeed.INSIDER_TRADES, DataNeed.FORM_13F},
    ]
    
    def __init__(self):
        self.config = get_config()
        self._available_providers = self._get_available_providers()
    
    def _get_available_providers(self) -> Set[str]:
        """Get set of available (configured) providers"""
        available = {"computed", "yfinance"}  # Always available (no API key needed)
        
        validation = self.config.validate()
        for provider, is_valid in validation.items():
            if is_valid and provider != "llm":
                available.add(provider)
        
        return available
    
    def create_routing_plan(self, intent: ParsedIntent) -> RoutingPlan:
        """
        Create an optimal routing plan for the parsed intent.
        Considers provider availability, quality, latency, and cost.
        """
        requests = []
        provider_assignments = {}
        
        # For each data need, find the best available provider
        for data_need in intent.data_needs:
            provider = self._select_provider(data_need)
            
            if provider:
                # Track which data needs go to which provider
                if provider not in provider_assignments:
                    provider_assignments[provider] = []
                provider_assignments[provider].append(data_need)
                
                # Create request
                request = DataRequest(
                    provider=provider,
                    data_need=data_need,
                    symbols=intent.symbols,
                    params=self._build_params(data_need, intent),
                    priority=self._get_priority(data_need),
                    timeout_ms=self._get_timeout(provider)
                )
                requests.append(request)
        
        # Organize into parallel groups
        parallel_groups = self._organize_parallel_groups(requests)
        
        # Calculate estimates
        estimated_time = self._estimate_time(parallel_groups)
        estimated_cost = self._estimate_cost(requests)
        
        return RoutingPlan(
            intent=intent,
            requests=requests,
            parallel_groups=parallel_groups,
            estimated_time_ms=estimated_time,
            estimated_cost=estimated_cost
        )
    
    def _select_provider(self, data_need: DataNeed) -> Optional[str]:
        """Select the best available provider for a data need"""
        priority_list = self.DATA_NEED_PRIORITY.get(data_need, [])
        
        for provider in priority_list:
            if provider in self._available_providers:
                # Check if provider has this capability
                capability = self.PROVIDER_CAPABILITIES.get(provider)
                if capability and data_need in capability.data_needs:
                    return provider
        
        return None
    
    def _build_params(self, data_need: DataNeed, intent: ParsedIntent) -> Dict[str, Any]:
        """Build request parameters based on data need and intent"""
        params = {}
        
        # Add timeframe if relevant
        if intent.timeframe:
            params["timeframe"] = intent.timeframe
        
        # Add date range if specified
        if intent.date_range:
            params["start_date"] = intent.date_range[0]
            params["end_date"] = intent.date_range[1]
        
        # Add specific indicators
        if data_need == DataNeed.TECHNICAL_INDICATORS and intent.specific_indicators:
            params["indicators"] = intent.specific_indicators
        
        # Add pattern types
        if data_need == DataNeed.CHART_PATTERNS and intent.specific_patterns:
            params["patterns"] = intent.specific_patterns
        
        # Add filing types
        if data_need in {DataNeed.SEC_FILINGS, DataNeed.FORM_10K, DataNeed.FORM_10Q, 
                         DataNeed.FORM_8K, DataNeed.FORM_13F, DataNeed.FORM_4}:
            if intent.filing_types:
                params["filing_types"] = intent.filing_types
        
        return params
    
    def _get_priority(self, data_need: DataNeed) -> int:
        """Get execution priority for a data need (lower = higher priority)"""
        # Real-time data has highest priority
        if data_need in {DataNeed.REALTIME_QUOTE, DataNeed.OPTIONS_CHAIN}:
            return 1
        # Historical/fundamental data is medium priority
        if data_need in {DataNeed.HISTORICAL_PRICES, DataNeed.KEY_RATIOS, DataNeed.EARNINGS}:
            return 2
        # Computed analytics depend on other data
        if data_need in {DataNeed.TECHNICAL_INDICATORS, DataNeed.CHART_PATTERNS}:
            return 3
        return 2
    
    def _get_timeout(self, provider: str) -> int:
        """Get timeout for a provider in milliseconds"""
        provider_config = self.config.get_provider(provider)
        if provider_config:
            return provider_config.timeout_seconds * 1000
        return 30000
    
    def _organize_parallel_groups(self, requests: List[DataRequest]) -> List[List[DataRequest]]:
        """Organize requests into groups that can be executed in parallel"""
        # First, group by priority
        priority_groups = {}
        for request in requests:
            if request.priority not in priority_groups:
                priority_groups[request.priority] = []
            priority_groups[request.priority].append(request)
        
        # Within each priority level, all requests can run in parallel
        # (except computed analytics which may depend on fetched data)
        parallel_groups = []
        
        for priority in sorted(priority_groups.keys()):
            group = priority_groups[priority]
            
            # Separate computed requests (they need input data first)
            computed = [r for r in group if r.provider == "computed"]
            non_computed = [r for r in group if r.provider != "computed"]
            
            if non_computed:
                parallel_groups.append(non_computed)
            if computed:
                parallel_groups.append(computed)
        
        return parallel_groups
    
    def _estimate_time(self, parallel_groups: List[List[DataRequest]]) -> int:
        """Estimate total execution time in milliseconds"""
        total = 0
        
        for group in parallel_groups:
            # For parallel execution, time is max of all requests in group
            if group:
                group_time = max(
                    self.PROVIDER_CAPABILITIES.get(
                        r.provider, 
                        ProviderCapability("default", set())
                    ).latency_ms
                    for r in group
                )
                total += group_time
        
        return total
    
    def _estimate_cost(self, requests: List[DataRequest]) -> float:
        """Estimate total cost of requests"""
        total = 0.0
        
        for request in requests:
            capability = self.PROVIDER_CAPABILITIES.get(request.provider)
            if capability:
                total += capability.cost_per_call * len(request.symbols)
        
        return total
    
    def get_provider_status(self) -> Dict[str, Dict]:
        """Get status of all providers"""
        status = {}
        
        for name, capability in self.PROVIDER_CAPABILITIES.items():
            provider_config = self.config.get_provider(name)
            
            status[name] = {
                "available": name in self._available_providers,
                "quality_score": capability.quality_score,
                "typical_latency_ms": capability.latency_ms,
                "data_needs_supported": len(capability.data_needs),
                "api_key_configured": (
                    provider_config.api_key is not None 
                    if provider_config else name in {"yfinance", "computed"}
                )
            }
        
        return status
    
    def suggest_providers(self, data_needs: List[DataNeed]) -> Dict[str, List[str]]:
        """Suggest which providers to configure for given data needs"""
        suggestions = {}
        
        for need in data_needs:
            providers = self.DATA_NEED_PRIORITY.get(need, [])
            available = [p for p in providers if p in self._available_providers]
            unavailable = [p for p in providers if p not in self._available_providers]
            
            suggestions[need.value] = {
                "available": available,
                "recommended_to_add": unavailable[:2] if not available else []
            }
        
        return suggestions
