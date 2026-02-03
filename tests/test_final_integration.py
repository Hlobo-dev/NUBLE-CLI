#!/usr/bin/env python3
"""
Phase 8: Final Integration Test

End-to-end user journeys testing the complete NUBLE system.
These tests simulate real user interactions from start to finish.
"""
import sys
import os
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv()


class FinalIntegrationTester:
    """Complete end-to-end integration tests."""
    
    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.orchestrator = None
    
    async def setup(self):
        """Initialize the orchestrator."""
        if not self.api_key:
            return False
        
        try:
            from src.nuble.agents.orchestrator import OrchestratorAgent
            self.orchestrator = OrchestratorAgent(api_key=self.api_key)
            return True
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    async def run_journey(self, name: str, messages: List[str], 
                          validations: List[Dict[str, Any]]) -> bool:
        """
        Run a complete user journey with multiple messages.
        
        Args:
            name: Journey name
            messages: List of user messages in sequence
            validations: List of validation rules for each response
        """
        print(f"\n  🧭 {name}")
        
        conversation_id = f"journey_{int(time.time())}"
        
        for i, (message, validation) in enumerate(zip(messages, validations), 1):
            try:
                result = await asyncio.wait_for(
                    self.orchestrator.process(
                        user_message=message,
                        conversation_id=conversation_id,
                        user_context={"journey": name, "step": i}
                    ),
                    timeout=90
                )
                
                if not result:
                    print(f"     Step {i}: ❌ No response")
                    return False
                
                response = result.get('message', '')
                
                # Validate response
                if validation.get('min_length'):
                    if len(response) < validation['min_length']:
                        print(f"     Step {i}: ❌ Response too short ({len(response)} < {validation['min_length']})")
                        return False
                
                if validation.get('contains'):
                    for term in validation['contains']:
                        if term.lower() not in response.lower():
                            # Warning, not failure
                            print(f"     Step {i}: ⚠️ Missing term '{term}'")
                
                if validation.get('not_contains'):
                    for term in validation['not_contains']:
                        if term.lower() in response.lower():
                            print(f"     Step {i}: ❌ Should not contain '{term}'")
                            return False
                
                print(f"     Step {i}: ✅ ({len(response)} chars)")
                
            except asyncio.TimeoutError:
                print(f"     Step {i}: ❌ Timeout")
                return False
            except Exception as e:
                print(f"     Step {i}: ❌ {str(e)[:40]}")
                return False
        
        return True
    
    async def test_new_investor_journey(self) -> bool:
        """Journey: New investor learning the basics."""
        return await self.run_journey(
            name="New Investor Journey",
            messages=[
                "Hi, I'm new to investing. What should I know?",
                "What's a stock?",
                "How do I start with $1000?",
                "What's the difference between stocks and ETFs?",
            ],
            validations=[
                {"min_length": 100},
                {"min_length": 50, "contains": ["stock", "share", "company"]},
                {"min_length": 100},
                {"min_length": 100, "contains": ["ETF"]},
            ]
        )
    
    async def test_research_journey(self) -> bool:
        """Journey: User researching a specific stock."""
        return await self.run_journey(
            name="Stock Research Journey",
            messages=[
                "Tell me about Apple stock",
                "What are the technicals for AAPL?",
                "Is it a good time to buy?",
                "What are the risks?",
            ],
            validations=[
                {"min_length": 100, "contains": ["AAPL", "Apple"]},
                {"min_length": 50},
                {"min_length": 50},
                {"min_length": 50, "contains": ["risk"]},
            ]
        )
    
    async def test_portfolio_journey(self) -> bool:
        """Journey: User building a portfolio."""
        return await self.run_journey(
            name="Portfolio Building Journey",
            messages=[
                "I have $50,000 to invest. How should I allocate it?",
                "What about adding some tech stocks?",
                "How much should I put in bonds?",
            ],
            validations=[
                {"min_length": 100, "contains": ["allocation", "diversif"]},
                {"min_length": 50},
                {"min_length": 50},
            ]
        )
    
    async def test_comparison_journey(self) -> bool:
        """Journey: Comparing two investments."""
        return await self.run_journey(
            name="Stock Comparison Journey",
            messages=[
                "Compare NVDA and AMD",
                "Which one has better technicals?",
                "Which would you recommend for growth?",
            ],
            validations=[
                {"min_length": 100, "contains": ["NVDA", "AMD"]},
                {"min_length": 50},
                {"min_length": 50},
            ]
        )
    
    async def test_crypto_journey(self) -> bool:
        """Journey: Exploring cryptocurrency."""
        return await self.run_journey(
            name="Crypto Exploration Journey",
            messages=[
                "Should I invest in Bitcoin?",
                "What about Ethereum?",
                "Is crypto too risky for retirement?",
            ],
            validations=[
                {"min_length": 50, "contains": ["Bitcoin", "BTC"]},
                {"min_length": 50},
                {"min_length": 50, "contains": ["risk"]},
            ]
        )
    
    async def test_market_analysis_journey(self) -> bool:
        """Journey: Understanding market conditions."""
        return await self.run_journey(
            name="Market Analysis Journey",
            messages=[
                "What's happening in the market today?",
                "Are we in a bull or bear market?",
                "What sectors are performing well?",
            ],
            validations=[
                {"min_length": 50},
                {"min_length": 50},
                {"min_length": 50},
            ]
        )
    
    async def test_technical_deep_dive(self) -> bool:
        """Journey: Technical analysis deep dive."""
        return await self.run_journey(
            name="Technical Analysis Deep Dive",
            messages=[
                "Show me RSI for MSFT",
                "What are the support and resistance levels?",
                "Is there a MACD crossover signal?",
            ],
            validations=[
                {"min_length": 50, "contains": ["RSI"]},
                {"min_length": 50},
                {"min_length": 50, "contains": ["MACD"]},
            ]
        )
    
    async def test_error_recovery_journey(self) -> bool:
        """Journey: System recovers from unclear queries."""
        return await self.run_journey(
            name="Error Recovery Journey",
            messages=[
                "asdfasdf",
                "Sorry, I meant what's Tesla's price?",
                "And what about their earnings?",
            ],
            validations=[
                {"min_length": 20},  # Should still respond
                {"min_length": 50, "contains": ["Tesla", "TSLA"]},
                {"min_length": 50},
            ]
        )
    
    async def test_context_memory_journey(self) -> bool:
        """Journey: System remembers context."""
        return await self.run_journey(
            name="Context Memory Journey",
            messages=[
                "I'm interested in GOOGL",
                "What's the P/E ratio?",
                "And what about revenue growth?",
                "Should I buy it?",
            ],
            validations=[
                {"min_length": 50, "contains": ["GOOGL", "Google"]},
                {"min_length": 30},
                {"min_length": 30},
                {"min_length": 50},
            ]
        )
    
    async def test_risk_management_journey(self) -> bool:
        """Journey: Focus on risk management."""
        return await self.run_journey(
            name="Risk Management Journey",
            messages=[
                "How do I manage risk in my portfolio?",
                "What's a stop loss?",
                "How much of my portfolio should be in one stock?",
            ],
            validations=[
                {"min_length": 100, "contains": ["risk"]},
                {"min_length": 50, "contains": ["stop"]},
                {"min_length": 50, "contains": ["position", "concentration", "percent"]},
            ]
        )
    
    async def run_all(self) -> bool:
        """Run all integration journeys."""
        print("\n" + "="*70)
        print("PHASE 8: FINAL INTEGRATION TEST")
        print("="*70)
        
        if not await self.setup():
            print("❌ Failed to initialize orchestrator")
            return False
        
        journeys = [
            ("New Investor", self.test_new_investor_journey),
            ("Stock Research", self.test_research_journey),
            ("Portfolio Building", self.test_portfolio_journey),
            ("Stock Comparison", self.test_comparison_journey),
            ("Crypto Exploration", self.test_crypto_journey),
            ("Market Analysis", self.test_market_analysis_journey),
            ("Technical Deep Dive", self.test_technical_deep_dive),
            ("Error Recovery", self.test_error_recovery_journey),
            ("Context Memory", self.test_context_memory_journey),
            ("Risk Management", self.test_risk_management_journey),
        ]
        
        print(f"\n🎯 Running {len(journeys)} user journeys...\n")
        
        for name, journey in journeys:
            try:
                result = await journey()
                if result:
                    print(f"  ✅ {name} Journey: PASSED")
                    self.passed += 1
                else:
                    print(f"  ❌ {name} Journey: FAILED")
                    self.failed += 1
            except Exception as e:
                print(f"  ❌ {name} Journey: ERROR - {str(e)[:40]}")
                self.failed += 1
        
        # Summary
        total = self.passed + self.failed
        print("\n" + "="*70)
        print(f"FINAL INTEGRATION RESULTS: {self.passed}/{total}")
        print("="*70)
        
        if self.failed == 0:
            print("🏆 ALL USER JOURNEYS COMPLETED SUCCESSFULLY")
        else:
            print(f"❌ {self.failed} journeys need attention")
        
        print("="*70 + "\n")
        
        return self.failed == 0


async def main():
    tester = FinalIntegrationTester()
    success = await tester.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
