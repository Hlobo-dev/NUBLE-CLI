#!/usr/bin/env python3
"""
NUBLE Educator Agent

Specialized agent for explanations, education, and strategy guidance.
"""

import os
from datetime import datetime
from typing import Dict, Any
import logging

from .base import SpecializedAgent, AgentTask, AgentResult, AgentType

logger = logging.getLogger(__name__)


class EducatorAgent(SpecializedAgent):
    """
    Educator Agent - Learning & Explanation Expert
    
    Capabilities:
    - Concept explanations
    - Strategy tutorials
    - Term definitions
    - Best practices
    - Learning paths
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Educator",
            "description": "Education and explanations",
            "capabilities": [
                "concept_explanations",
                "strategy_tutorials",
                "term_definitions",
                "best_practices",
                "learning_paths"
            ]
        }
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute educational task."""
        start = datetime.now()
        
        try:
            query = task.context.get('query', task.instruction)
            
            # Use Claude for explanations
            explanation = await self._generate_explanation(query)
            
            data = {
                'explanation': explanation,
                'related_topics': self._get_related_topics(query),
                'resources': self._get_resources(query),
                'next_steps': self._suggest_next_steps(query)
            }
            
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.EDUCATOR,
                success=True,
                data=data,
                confidence=0.85,
                execution_time_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.EDUCATOR,
                success=False,
                data={},
                confidence=0,
                execution_time_ms=int((datetime.now() - start).total_seconds() * 1000),
                error=str(e)
            )
    
    async def _generate_explanation(self, query: str) -> Dict:
        """Generate educational explanation using Claude."""
        if self.client:
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    messages=[{
                        "role": "user",
                        "content": f"""You are a financial educator. Explain the following in clear, accessible terms:

{query}

Provide:
1. A simple explanation
2. A more detailed explanation
3. A real-world example
4. Common misconceptions

Format as JSON with keys: simple, detailed, example, misconceptions"""
                    }]
                )
                return {'text': response.content[0].text, 'source': 'claude'}
            except Exception as e:
                logger.warning(f"Claude explanation failed: {e}")
        
        return self._mock_explanation(query)
    
    def _mock_explanation(self, query: str) -> Dict:
        """Generate mock explanation."""
        return {
            'simple': f"A brief explanation of {query}...",
            'detailed': f"A more comprehensive look at {query} involves understanding...",
            'example': "For instance, consider a scenario where...",
            'misconceptions': ["Common myth 1", "Common myth 2"],
            'source': 'template'
        }
    
    def _get_related_topics(self, query: str) -> list:
        """Get related educational topics."""
        topic_map = {
            'rsi': ['MACD', 'Stochastic', 'Overbought/Oversold', 'Divergence'],
            'options': ['Greeks', 'Implied Volatility', 'Theta Decay', 'Put-Call Ratio'],
            'etf': ['Index Funds', 'Expense Ratio', 'NAV', 'Tracking Error'],
            'default': ['Risk Management', 'Diversification', 'Position Sizing']
        }
        
        query_lower = query.lower()
        for key, topics in topic_map.items():
            if key in query_lower:
                return topics
        return topic_map['default']
    
    def _get_resources(self, query: str) -> list:
        """Get educational resources."""
        return [
            {'type': 'article', 'title': 'Getting Started Guide', 'difficulty': 'beginner'},
            {'type': 'video', 'title': 'Visual Tutorial', 'difficulty': 'intermediate'},
            {'type': 'book', 'title': 'Advanced Concepts', 'difficulty': 'advanced'}
        ]
    
    def _suggest_next_steps(self, query: str) -> list:
        """Suggest next learning steps."""
        return [
            'Practice with paper trading',
            'Explore related indicators',
            'Test on historical data'
        ]


__all__ = ['EducatorAgent']
