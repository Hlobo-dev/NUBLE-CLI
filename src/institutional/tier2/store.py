"""
Tier 2 Decision Store
======================

DynamoDB storage for decisions, agent runs, and outcomes.

Tables:
- nuble-decisions: Decision records
- nuble-agent-runs: Individual agent outputs
- nuble-outcomes: Trade outcomes for calibration
"""

import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """A stored decision record."""
    decision_id: str
    symbol: str
    timestamp: str  # ISO format
    tier1_action: str
    tier1_confidence: float
    tier2_delta: Dict[str, Any]
    final_action: str
    final_confidence: float
    escalation_reasons: List[str]
    agent_count: int
    latency_ms: float
    ttl: int  # Unix timestamp for TTL


@dataclass
class AgentRunRecord:
    """A stored agent run record."""
    decision_id: str
    agent_name: str
    round: str  # "light" or "deep"
    input_tokens: int
    output_tokens: int
    latency_ms: float
    claims: List[Dict]
    raw_output: Optional[str]  # Only for debugging, TTL'd
    timestamp: str
    ttl: int


@dataclass
class OutcomeRecord:
    """Trade outcome for calibration."""
    decision_id: str
    symbol: str
    entry_price: float
    exit_price: Optional[float]
    pnl_pct: Optional[float]
    holding_hours: Optional[float]
    outcome_label: Optional[str]  # "win", "loss", "scratch"
    tier1_confidence: float
    tier2_confidence: float
    timestamp: str


class DecisionStore:
    """
    Abstract interface for decision storage.
    """
    
    def save_decision(self, record: DecisionRecord) -> bool:
        """Save a decision record."""
        raise NotImplementedError
    
    def save_agent_run(self, record: AgentRunRecord) -> bool:
        """Save an agent run record."""
        raise NotImplementedError
    
    def save_outcome(self, record: OutcomeRecord) -> bool:
        """Save an outcome record."""
        raise NotImplementedError
    
    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        """Get a decision by ID."""
        raise NotImplementedError
    
    def get_decisions_for_symbol(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[DecisionRecord]:
        """Get recent decisions for a symbol."""
        raise NotImplementedError
    
    def get_agent_runs(self, decision_id: str) -> List[AgentRunRecord]:
        """Get all agent runs for a decision."""
        raise NotImplementedError
    
    def get_outcomes_for_calibration(
        self,
        days: int = 30,
    ) -> List[OutcomeRecord]:
        """Get outcomes for agent calibration."""
        raise NotImplementedError


class DynamoDBStore(DecisionStore):
    """
    DynamoDB implementation of DecisionStore.
    
    Schema:
    - nuble-decisions:
        PK: DECISION#{symbol}
        SK: {timestamp_iso}
        GSI1: decision_id
        
    - nuble-agent-runs:
        PK: RUN#{decision_id}
        SK: AGENT#{name}#{round}
        
    - nuble-outcomes:
        PK: OUTCOME#{symbol}
        SK: {timestamp_iso}
        GSI1: decision_id
    """
    
    def __init__(
        self,
        decisions_table: str = "nuble-decisions",
        runs_table: str = "nuble-agent-runs",
        outcomes_table: str = "nuble-outcomes",
        dynamodb_client = None,
    ):
        self.decisions_table = decisions_table
        self.runs_table = runs_table
        self.outcomes_table = outcomes_table
        
        # Lazy load boto3
        self._client = dynamodb_client
    
    @property
    def client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("dynamodb")
        return self._client
    
    def save_decision(self, record: DecisionRecord) -> bool:
        """Save a decision to DynamoDB."""
        try:
            item = {
                "PK": {"S": f"DECISION#{record.symbol}"},
                "SK": {"S": record.timestamp},
                "decision_id": {"S": record.decision_id},
                "symbol": {"S": record.symbol},
                "tier1_action": {"S": record.tier1_action},
                "tier1_confidence": {"N": str(record.tier1_confidence)},
                "tier2_delta": {"S": json.dumps(record.tier2_delta)},
                "final_action": {"S": record.final_action},
                "final_confidence": {"N": str(record.final_confidence)},
                "escalation_reasons": {"SS": record.escalation_reasons or ["NONE"]},
                "agent_count": {"N": str(record.agent_count)},
                "latency_ms": {"N": str(record.latency_ms)},
                "ttl": {"N": str(record.ttl)},
            }
            
            self.client.put_item(
                TableName=self.decisions_table,
                Item=item,
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to save decision: {e}")
            return False
    
    def save_agent_run(self, record: AgentRunRecord) -> bool:
        """Save an agent run to DynamoDB."""
        try:
            item = {
                "PK": {"S": f"RUN#{record.decision_id}"},
                "SK": {"S": f"AGENT#{record.agent_name}#{record.round}"},
                "decision_id": {"S": record.decision_id},
                "agent_name": {"S": record.agent_name},
                "round": {"S": record.round},
                "input_tokens": {"N": str(record.input_tokens)},
                "output_tokens": {"N": str(record.output_tokens)},
                "latency_ms": {"N": str(record.latency_ms)},
                "claims": {"S": json.dumps(record.claims)},
                "timestamp": {"S": record.timestamp},
                "ttl": {"N": str(record.ttl)},
            }
            
            # Raw output only if present (TTL'd for debugging)
            if record.raw_output:
                item["raw_output"] = {"S": record.raw_output}
            
            self.client.put_item(
                TableName=self.runs_table,
                Item=item,
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to save agent run: {e}")
            return False
    
    def save_outcome(self, record: OutcomeRecord) -> bool:
        """Save an outcome to DynamoDB."""
        try:
            item = {
                "PK": {"S": f"OUTCOME#{record.symbol}"},
                "SK": {"S": record.timestamp},
                "decision_id": {"S": record.decision_id},
                "symbol": {"S": record.symbol},
                "entry_price": {"N": str(record.entry_price)},
                "tier1_confidence": {"N": str(record.tier1_confidence)},
                "tier2_confidence": {"N": str(record.tier2_confidence)},
                "timestamp": {"S": record.timestamp},
            }
            
            if record.exit_price is not None:
                item["exit_price"] = {"N": str(record.exit_price)}
            if record.pnl_pct is not None:
                item["pnl_pct"] = {"N": str(record.pnl_pct)}
            if record.holding_hours is not None:
                item["holding_hours"] = {"N": str(record.holding_hours)}
            if record.outcome_label:
                item["outcome_label"] = {"S": record.outcome_label}
            
            self.client.put_item(
                TableName=self.outcomes_table,
                Item=item,
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to save outcome: {e}")
            return False
    
    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        """Query by decision_id using GSI."""
        try:
            response = self.client.query(
                TableName=self.decisions_table,
                IndexName="decision_id-index",
                KeyConditionExpression="decision_id = :did",
                ExpressionAttributeValues={
                    ":did": {"S": decision_id}
                },
                Limit=1,
            )
            
            items = response.get("Items", [])
            if not items:
                return None
            
            item = items[0]
            return DecisionRecord(
                decision_id=item["decision_id"]["S"],
                symbol=item["symbol"]["S"],
                timestamp=item["SK"]["S"],
                tier1_action=item["tier1_action"]["S"],
                tier1_confidence=float(item["tier1_confidence"]["N"]),
                tier2_delta=json.loads(item["tier2_delta"]["S"]),
                final_action=item["final_action"]["S"],
                final_confidence=float(item["final_confidence"]["N"]),
                escalation_reasons=list(item.get("escalation_reasons", {}).get("SS", [])),
                agent_count=int(item["agent_count"]["N"]),
                latency_ms=float(item["latency_ms"]["N"]),
                ttl=int(item["ttl"]["N"]),
            )
            
        except Exception as e:
            logger.error(f"Failed to get decision: {e}")
            return None
    
    def get_decisions_for_symbol(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[DecisionRecord]:
        """Get recent decisions for a symbol."""
        try:
            response = self.client.query(
                TableName=self.decisions_table,
                KeyConditionExpression="PK = :pk",
                ExpressionAttributeValues={
                    ":pk": {"S": f"DECISION#{symbol}"}
                },
                ScanIndexForward=False,  # Newest first
                Limit=limit,
            )
            
            records = []
            for item in response.get("Items", []):
                records.append(DecisionRecord(
                    decision_id=item["decision_id"]["S"],
                    symbol=item["symbol"]["S"],
                    timestamp=item["SK"]["S"],
                    tier1_action=item["tier1_action"]["S"],
                    tier1_confidence=float(item["tier1_confidence"]["N"]),
                    tier2_delta=json.loads(item["tier2_delta"]["S"]),
                    final_action=item["final_action"]["S"],
                    final_confidence=float(item["final_confidence"]["N"]),
                    escalation_reasons=list(item.get("escalation_reasons", {}).get("SS", [])),
                    agent_count=int(item["agent_count"]["N"]),
                    latency_ms=float(item["latency_ms"]["N"]),
                    ttl=int(item["ttl"]["N"]),
                ))
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to get decisions for symbol: {e}")
            return []
    
    def get_agent_runs(self, decision_id: str) -> List[AgentRunRecord]:
        """Get all agent runs for a decision."""
        try:
            response = self.client.query(
                TableName=self.runs_table,
                KeyConditionExpression="PK = :pk",
                ExpressionAttributeValues={
                    ":pk": {"S": f"RUN#{decision_id}"}
                },
            )
            
            records = []
            for item in response.get("Items", []):
                records.append(AgentRunRecord(
                    decision_id=item["decision_id"]["S"],
                    agent_name=item["agent_name"]["S"],
                    round=item["round"]["S"],
                    input_tokens=int(item["input_tokens"]["N"]),
                    output_tokens=int(item["output_tokens"]["N"]),
                    latency_ms=float(item["latency_ms"]["N"]),
                    claims=json.loads(item["claims"]["S"]),
                    raw_output=item.get("raw_output", {}).get("S"),
                    timestamp=item["timestamp"]["S"],
                    ttl=int(item["ttl"]["N"]),
                ))
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to get agent runs: {e}")
            return []
    
    def get_outcomes_for_calibration(
        self,
        days: int = 30,
    ) -> List[OutcomeRecord]:
        """
        Get outcomes for calibration.
        Note: This does a scan - use sparingly.
        """
        try:
            from datetime import timedelta
            
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            
            response = self.client.scan(
                TableName=self.outcomes_table,
                FilterExpression="timestamp >= :cutoff AND attribute_exists(outcome_label)",
                ExpressionAttributeValues={
                    ":cutoff": {"S": cutoff}
                },
            )
            
            records = []
            for item in response.get("Items", []):
                records.append(OutcomeRecord(
                    decision_id=item["decision_id"]["S"],
                    symbol=item["symbol"]["S"],
                    entry_price=float(item["entry_price"]["N"]),
                    exit_price=float(item["exit_price"]["N"]) if "exit_price" in item else None,
                    pnl_pct=float(item["pnl_pct"]["N"]) if "pnl_pct" in item else None,
                    holding_hours=float(item["holding_hours"]["N"]) if "holding_hours" in item else None,
                    outcome_label=item.get("outcome_label", {}).get("S"),
                    tier1_confidence=float(item["tier1_confidence"]["N"]),
                    tier2_confidence=float(item["tier2_confidence"]["N"]),
                    timestamp=item["timestamp"]["S"],
                ))
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to get outcomes: {e}")
            return []


class InMemoryStore(DecisionStore):
    """
    In-memory store for testing and local development.
    """
    
    def __init__(self):
        self.decisions: Dict[str, DecisionRecord] = {}
        self.agent_runs: Dict[str, List[AgentRunRecord]] = {}
        self.outcomes: Dict[str, OutcomeRecord] = {}
    
    def save_decision(self, record: DecisionRecord) -> bool:
        self.decisions[record.decision_id] = record
        return True
    
    def save_agent_run(self, record: AgentRunRecord) -> bool:
        if record.decision_id not in self.agent_runs:
            self.agent_runs[record.decision_id] = []
        self.agent_runs[record.decision_id].append(record)
        return True
    
    def save_outcome(self, record: OutcomeRecord) -> bool:
        self.outcomes[record.decision_id] = record
        return True
    
    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        return self.decisions.get(decision_id)
    
    def get_decisions_for_symbol(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[DecisionRecord]:
        matching = [
            d for d in self.decisions.values()
            if d.symbol == symbol
        ]
        matching.sort(key=lambda x: x.timestamp, reverse=True)
        return matching[:limit]
    
    def get_agent_runs(self, decision_id: str) -> List[AgentRunRecord]:
        return self.agent_runs.get(decision_id, [])
    
    def get_outcomes_for_calibration(
        self,
        days: int = 30,
    ) -> List[OutcomeRecord]:
        return list(self.outcomes.values())


def generate_decision_id(symbol: str, timestamp: str) -> str:
    """Generate a unique decision ID."""
    hash_input = f"{symbol}:{timestamp}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
