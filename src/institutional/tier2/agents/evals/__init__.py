"""
Agent Evaluation System
=======================

Framework for evaluating and benchmarking agent performance.

Components:
- EvalRunner: Runs evaluation cases against agents
- EvalCase: Standardized test case format
- EvalResult: Structured evaluation results
- EvalReport: Aggregate reporting

Usage:
    from .evals import EvalRunner, load_eval_cases
    
    runner = EvalRunner()
    cases = load_eval_cases("mtf_dominance")
    results = await runner.run(agent, cases)
    report = runner.generate_report(results)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path


@dataclass
class EvalCase:
    """
    A single evaluation test case.
    
    Attributes:
        name: Unique identifier for this case
        description: Human-readable description
        context: Input context for the agent
        expected_claim: Expected claim output
        expected_evidence_keys: Evidence keys that should be present
        acceptable_claims: Alternative acceptable claims
        tags: Categorization tags (e.g., "edge_case", "regression")
    """
    name: str
    description: str
    context: Dict[str, Any]
    expected_claim: str
    expected_evidence_keys: List[str] = field(default_factory=list)
    acceptable_claims: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EvalCase':
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            context=data["context"],
            expected_claim=data["expected_claim"],
            expected_evidence_keys=data.get("expected_evidence_keys", []),
            acceptable_claims=data.get("acceptable_claims", []),
            tags=data.get("tags", [])
        )


@dataclass
class EvalResult:
    """Result of evaluating a single case."""
    case_name: str
    passed: bool
    claim_correct: bool
    evidence_complete: bool
    actual_claim: Optional[str] = None
    actual_evidence: Optional[Dict] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class EvalReport:
    """Aggregate evaluation report."""
    agent_id: str
    prompt_version: str
    timestamp: datetime
    total_cases: int
    passed_cases: int
    failed_cases: int
    accuracy: float
    avg_latency_ms: float
    results: List[EvalResult]
    
    # Breakdown by tag
    tag_accuracy: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "prompt_version": self.prompt_version,
            "timestamp": self.timestamp.isoformat(),
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "accuracy": self.accuracy,
            "avg_latency_ms": self.avg_latency_ms,
            "tag_accuracy": self.tag_accuracy,
            "results": [
                {
                    "case_name": r.case_name,
                    "passed": r.passed,
                    "claim_correct": r.claim_correct,
                    "actual_claim": r.actual_claim,
                    "latency_ms": r.latency_ms,
                    "error": r.error
                }
                for r in self.results
            ]
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Evaluation Report: {self.agent_id}",
            f"",
            f"**Prompt Version:** {self.prompt_version}",
            f"**Timestamp:** {self.timestamp.isoformat()}",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Cases | {self.total_cases} |",
            f"| Passed | {self.passed_cases} |",
            f"| Failed | {self.failed_cases} |",
            f"| Accuracy | {self.accuracy:.1%} |",
            f"| Avg Latency | {self.avg_latency_ms:.1f}ms |",
            f"",
        ]
        
        if self.tag_accuracy:
            lines.extend([
                "## Accuracy by Tag",
                "",
                "| Tag | Accuracy |",
                "|-----|----------|",
            ])
            for tag, acc in self.tag_accuracy.items():
                lines.append(f"| {tag} | {acc:.1%} |")
            lines.append("")
        
        # Failed cases details
        failed = [r for r in self.results if not r.passed]
        if failed:
            lines.extend([
                "## Failed Cases",
                "",
            ])
            for result in failed:
                lines.extend([
                    f"### {result.case_name}",
                    f"- **Expected:** {result.metadata.get('expected_claim', 'N/A')}",
                    f"- **Actual:** {result.actual_claim}",
                    f"- **Error:** {result.error or 'None'}",
                    "",
                ])
        
        return "\n".join(lines)


class EvalRunner:
    """
    Runs evaluation cases against agents.
    
    Supports:
    - Batch evaluation
    - Parallel execution
    - Progress reporting
    - Result aggregation
    """
    
    def __init__(self, timeout_ms: int = 30000):
        self.timeout_ms = timeout_ms
    
    async def run_case(
        self,
        agent,
        case: EvalCase
    ) -> EvalResult:
        """Run a single evaluation case."""
        import time
        from ..base import AgentContext
        
        start = time.time()
        
        try:
            # Build context
            context = AgentContext(
                trade_context=case.context.get("context", case.context),
                portfolio_state=case.context.get("portfolio_state", {}),
                market_data=case.context.get("market_data", {}),
                complexity_score=0.5,
                time_budget_ms=self.timeout_ms
            )
            
            # Run agent
            output = await agent.analyze(context)
            
            latency = (time.time() - start) * 1000
            
            # Check claim
            all_acceptable = [case.expected_claim] + case.acceptable_claims
            claim_correct = output.claim in all_acceptable
            
            # Check evidence
            evidence_complete = all(
                key in output.evidence
                for key in case.expected_evidence_keys
            )
            
            passed = claim_correct and evidence_complete
            
            return EvalResult(
                case_name=case.name,
                passed=passed,
                claim_correct=claim_correct,
                evidence_complete=evidence_complete,
                actual_claim=output.claim,
                actual_evidence=output.evidence,
                latency_ms=latency,
                metadata={"expected_claim": case.expected_claim}
            )
            
        except Exception as e:
            latency = (time.time() - start) * 1000
            return EvalResult(
                case_name=case.name,
                passed=False,
                claim_correct=False,
                evidence_complete=False,
                latency_ms=latency,
                error=str(e),
                metadata={"expected_claim": case.expected_claim}
            )
    
    async def run(
        self,
        agent,
        cases: List[EvalCase],
        parallel: bool = False
    ) -> List[EvalResult]:
        """Run all evaluation cases."""
        import asyncio
        
        if parallel:
            tasks = [self.run_case(agent, case) for case in cases]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for case in cases:
                result = await self.run_case(agent, case)
                results.append(result)
            return results
    
    def generate_report(
        self,
        agent,
        results: List[EvalResult],
        cases: List[EvalCase]
    ) -> EvalReport:
        """Generate aggregate report from results."""
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        avg_latency = sum(r.latency_ms for r in results) / total if total else 0
        
        # Calculate accuracy by tag
        tag_results: Dict[str, List[bool]] = {}
        for case, result in zip(cases, results):
            for tag in case.tags:
                if tag not in tag_results:
                    tag_results[tag] = []
                tag_results[tag].append(result.passed)
        
        tag_accuracy = {
            tag: sum(results) / len(results) if results else 0
            for tag, results in tag_results.items()
        }
        
        return EvalReport(
            agent_id=agent.agent_id,
            prompt_version=agent.prompt_version,
            timestamp=datetime.utcnow(),
            total_cases=total,
            passed_cases=passed,
            failed_cases=total - passed,
            accuracy=passed / total if total else 0,
            avg_latency_ms=avg_latency,
            results=results,
            tag_accuracy=tag_accuracy
        )


def load_eval_cases(agent_id: str, evals_dir: Optional[Path] = None) -> List[EvalCase]:
    """Load evaluation cases from JSON file."""
    evals_dir = evals_dir or Path(__file__).parent
    cases_file = evals_dir / f"{agent_id}_cases.json"
    
    if not cases_file.exists():
        return []
    
    with open(cases_file) as f:
        data = json.load(f)
    
    return [EvalCase.from_dict(case) for case in data.get("cases", [])]


def save_eval_cases(
    agent_id: str,
    cases: List[EvalCase],
    evals_dir: Optional[Path] = None
):
    """Save evaluation cases to JSON file."""
    evals_dir = evals_dir or Path(__file__).parent
    cases_file = evals_dir / f"{agent_id}_cases.json"
    
    data = {
        "agent_id": agent_id,
        "version": "1.0",
        "cases": [
            {
                "name": c.name,
                "description": c.description,
                "context": c.context,
                "expected_claim": c.expected_claim,
                "expected_evidence_keys": c.expected_evidence_keys,
                "acceptable_claims": c.acceptable_claims,
                "tags": c.tags
            }
            for c in cases
        ]
    }
    
    with open(cases_file, "w") as f:
        json.dump(data, f, indent=2)


__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalReport",
    "EvalRunner",
    "load_eval_cases",
    "save_eval_cases",
]
