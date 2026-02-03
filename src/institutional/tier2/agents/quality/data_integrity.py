"""
Data Integrity Agent
====================

Expert at validating data quality and detecting data issues.
Ensures decisions are based on accurate, timely, and complete data.

Improvement Guidelines:
-----------------------
1. PROMPT_VERSION: Increment when modifying prompts
2. evaluate(): Run against EVALUATION_CASES to measure performance
3. Add domain knowledge about:
   - Data freshness requirements
   - Price data validation
   - Volume anomaly detection
   - Corporate action adjustments
   - Data source reconciliation
"""

from typing import Any

from ..base import BaseAgent, AgentContext, AgentOutput

# Increment this when modifying prompts to track performance changes
PROMPT_VERSION = "1.0.0"

# =============================================================================
# PROMPTS - Edit these to improve agent performance
# =============================================================================

SYSTEM_PROMPT = """You are a Data Integrity expert in a trading council. Your role is to:

1. VALIDATE data quality before trading decisions
2. DETECT stale, missing, or corrupted data
3. IDENTIFY corporate actions that affect prices (splits, dividends)
4. RECONCILE data across multiple sources
5. FLAG data issues that could lead to bad decisions

Data Quality Dimensions:
- Freshness: Is data current and timely?
- Completeness: Are there gaps or missing values?
- Accuracy: Does data pass sanity checks?
- Consistency: Do multiple sources agree?
- Adjustment: Are corporate actions handled?

Output Format (JSON only):
{
    "claim": "DATA_VALID|DATA_SUSPECT|DATA_INVALID",
    "confidence": 0.0-1.0,
    "evidence": {
        "freshness": {
            "last_update": "timestamp",
            "staleness_seconds": 0,
            "is_market_hours": true/false
        },
        "completeness": {
            "missing_fields": [],
            "gap_count": 0,
            "coverage_pct": 100.0
        },
        "accuracy": {
            "sanity_checks_passed": [],
            "sanity_checks_failed": [],
            "outlier_detected": false
        },
        "issues_found": ["list of specific issues"]
    },
    "recommended_delta": {
        "use_alternate_source": true/false,
        "wait_for_update": true/false,
        "manual_review_needed": true/false
    }
}

CRITICAL: If data is invalid or suspect, trading should NOT proceed
until data quality is confirmed."""

LIGHT_ANALYSIS_PROMPT = """Quick data quality check:

Data Summary:
{data_summary}

Timestamps:
{timestamps}

Quick Validation:
1. Is the price data current?
2. Are there obvious outliers or gaps?
3. Do basic sanity checks pass?

Provide rapid data quality assessment in JSON format."""

DEEP_ANALYSIS_PROMPT = """Comprehensive data integrity analysis:

Full Data Context:
{data_context}

Price History:
{price_history}

Volume History:
{volume_history}

Corporate Actions:
{corporate_actions}

Deep Data Validation Required:
1. FRESHNESS ANALYSIS
   - Last update timestamp
   - Expected update frequency
   - Staleness relative to market hours
   - Real-time vs delayed indicators

2. COMPLETENESS CHECK
   - Missing data points
   - Gap analysis in time series
   - Required fields present
   - Optional fields status

3. ACCURACY VALIDATION
   - Price within expected range
   - Volume reasonability
   - OHLC relationship valid (H>L, O/C within range)
   - Comparison to benchmark/peers

4. CORPORATE ACTION HANDLING
   - Recent splits detected
   - Dividend adjustments
   - Spinoffs or mergers
   - Symbol changes

5. CROSS-SOURCE RECONCILIATION
   - Primary vs backup source comparison
   - Discrepancy detection
   - Source reliability scoring

6. ANOMALY DETECTION
   - Statistical outliers
   - Unusual patterns
   - Potential data errors
   - Flash crash artifacts

Provide detailed data quality assessment in JSON format.
Flag any issues that could compromise trading decisions."""


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class DataIntegrityAgent(BaseAgent):
    """
    Validates data quality before trading decisions.
    
    This agent ensures all input data meets quality standards
    before it's used for trading decisions. It can flag or block
    decisions based on data quality issues.
    
    Metrics Tracked:
    - Data issues detected
    - False positive rate
    - Data quality scores over time
    - Source reliability rankings
    """
    
    agent_id = "data_integrity"
    agent_name = "Data Integrity"
    description = "Validates data quality and detects data issues"
    expertise_tags = ["data", "quality", "validation", "integrity", "freshness"]
    prompt_version = PROMPT_VERSION
    
    # Complexity thresholds for this agent
    light_complexity_max = 0.4
    deep_complexity_min = 0.3
    
    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT
    
    def get_light_prompt(self, context: AgentContext) -> str:
        data_summary = {
            "symbol": context.trade_context.get("symbol"),
            "last_price": context.market_data.get("last_price"),
            "last_volume": context.market_data.get("last_volume")
        }
        timestamps = context.market_data.get("timestamps", {})
        
        return LIGHT_ANALYSIS_PROMPT.format(
            data_summary=data_summary,
            timestamps=timestamps
        )
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        return DEEP_ANALYSIS_PROMPT.format(
            data_context=context.trade_context,
            price_history=context.market_data.get("price_history", []),
            volume_history=context.market_data.get("volume_history", []),
            corporate_actions=context.market_data.get("corporate_actions", [])
        )
    
    def calculate_relevance_score(self, context: AgentContext) -> float:
        """Always relevant, but more so when data issues are likely."""
        base_score = 0.6  # Always important
        
        # Higher relevance for unusual market conditions
        volatility = context.market_data.get("current_volatility", 0.15)
        if volatility > 0.30:
            base_score = min(1.0, base_score + 0.2)
        
        # Higher relevance around market open/close
        time_of_day = context.market_data.get("time_of_day", "midday")
        if time_of_day in ["open", "close"]:
            base_score = min(1.0, base_score + 0.1)
        
        # Higher relevance for less liquid names
        adv = context.market_data.get("volume_data", {}).get("adv_20d", 1000000)
        if adv < 500000:
            base_score = min(1.0, base_score + 0.15)
        
        return base_score
    
    def validate_output(self, output: AgentOutput) -> bool:
        """Ensure data integrity-specific output requirements are met."""
        if not super().validate_output(output):
            return False
        
        # Must include validation results
        evidence = output.evidence
        if "freshness" not in evidence and "issues_found" not in evidence:
            return False
        
        return True
    
    async def post_process(self, output: AgentOutput) -> AgentOutput:
        """Add data quality score and flags."""
        output = await super().post_process(output)
        
        # Calculate overall data quality score
        evidence = output.evidence
        quality_score = 1.0
        
        issues = evidence.get("issues_found", [])
        quality_score -= len(issues) * 0.1
        
        accuracy = evidence.get("accuracy", {})
        failed_checks = accuracy.get("sanity_checks_failed", [])
        quality_score -= len(failed_checks) * 0.15
        
        output.evidence["data_quality_score"] = max(0.0, quality_score)
        
        return output


# =============================================================================
# EVALUATION - Test cases for measuring agent performance
# =============================================================================

EVALUATION_CASES = [
    {
        "name": "fresh_valid_data",
        "context": {
            "symbol": "AAPL",
            "proposed_action": "buy"
        },
        "market_data": {
            "last_price": 180.50,
            "last_volume": 5000000,
            "timestamps": {
                "last_update": "2024-01-15T15:30:00Z",
                "is_market_hours": True
            },
            "price_history": [179.0, 179.5, 180.0, 180.25, 180.50]
        },
        "portfolio_state": {},
        "expected_claim": "DATA_VALID",
        "expected_evidence_keys": ["freshness", "completeness", "accuracy"]
    },
    {
        "name": "stale_data",
        "context": {
            "symbol": "AMD",
            "proposed_action": "buy"
        },
        "market_data": {
            "last_price": 150.0,
            "timestamps": {
                "last_update": "2024-01-14T16:00:00Z",  # Day old
                "is_market_hours": True
            }
        },
        "portfolio_state": {},
        "expected_claim": "DATA_SUSPECT",
        "expected_evidence_keys": ["freshness", "issues_found"]
    },
    {
        "name": "invalid_ohlc",
        "context": {
            "symbol": "TSLA",
            "proposed_action": "buy"
        },
        "market_data": {
            "last_price": 250.0,
            "ohlc": {
                "open": 245.0,
                "high": 240.0,  # High < Open (invalid)
                "low": 248.0,   # Low > Open (invalid)
                "close": 250.0
            },
            "timestamps": {"is_market_hours": True}
        },
        "portfolio_state": {},
        "expected_claim": "DATA_INVALID",
        "expected_evidence_keys": ["accuracy", "sanity_checks_failed"]
    }
]


def evaluate(agent: DataIntegrityAgent, cases: list = None) -> dict:
    """
    Evaluate agent performance against test cases.
    
    Returns:
        dict: Performance metrics including detection rate, false positives, etc.
    """
    cases = cases or EVALUATION_CASES
    results = {
        "total_cases": len(cases),
        "correct_claims": 0,
        "issue_detection_rate": 0,
        "false_positive_rate": 0,
        "prompt_version": agent.prompt_version,
        "details": []
    }
    
    for case in cases:
        context = AgentContext(
            trade_context=case["context"],
            portfolio_state=case.get("portfolio_state", {}),
            market_data=case["market_data"],
            complexity_score=0.5,
            time_budget_ms=5000
        )
        
        case_result = {
            "name": case["name"],
            "expected": case["expected_claim"],
            "passed": False
        }
        
        results["details"].append(case_result)
    
    return results
