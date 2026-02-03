#!/usr/bin/env python3
"""
PHASE 7: Debug Integration Failure
Find exactly which component is failing.
"""

import asyncio
import sys
import os
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()


class IntegrationDebugger:
    """Debug integration failures in detail."""
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.results = []
    
    def log(self, component: str, status: str, detail: str = ""):
        """Log a result."""
        icon = "✅" if status == "OK" else "❌" if status == "FAIL" else "⚠️"
        self.results.append((component, status, detail))
        print(f"  {icon} {component}: {detail[:80] if detail else status}")
    
    async def test_orchestrator_init(self):
        """Test OrchestratorAgent initialization in detail."""
        print("\n" + "="*60)
        print("DEBUG: OrchestratorAgent Initialization")
        print("="*60)
        
        try:
            from src.nuble.agents.orchestrator import OrchestratorAgent
            self.log("Import OrchestratorAgent", "OK")
        except Exception as e:
            self.log("Import OrchestratorAgent", "FAIL", str(e))
            return
        
        try:
            o = OrchestratorAgent(api_key=self.api_key)
            self.log("Create OrchestratorAgent", "OK")
        except Exception as e:
            self.log("Create OrchestratorAgent", "FAIL", str(e))
            traceback.print_exc()
            return
        
        # Check attributes
        attrs_to_check = [
            '_agents', '_agents_initialized', 'api_key', 'model', 
            'client', 'memory', 'process'
        ]
        
        for attr in attrs_to_check:
            if hasattr(o, attr):
                val = getattr(o, attr)
                if attr == '_agents':
                    self.log(f"Attribute: {attr}", "OK", f"Dict with {len(val)} agents")
                elif attr == '_agents_initialized':
                    self.log(f"Attribute: {attr}", "OK", f"Value: {val}")
                elif callable(val):
                    self.log(f"Attribute: {attr}", "OK", "Method exists")
                else:
                    self.log(f"Attribute: {attr}", "OK", f"Type: {type(val).__name__}")
            else:
                self.log(f"Attribute: {attr}", "FAIL", "Missing")
        
        # Test lazy initialization
        try:
            o._initialize_agents()
            self.log("Lazy agent initialization", "OK", f"{len(o._agents)} agents created")
            
            # List agents
            for agent_type, agent in o._agents.items():
                self.log(f"  Agent: {agent_type.name}", "OK", type(agent).__name__)
        except Exception as e:
            self.log("Lazy agent initialization", "FAIL", str(e))
            traceback.print_exc()
    
    async def test_each_agent_individually(self):
        """Test each specialized agent individually."""
        print("\n" + "="*60)
        print("DEBUG: Individual Agent Tests")
        print("="*60)
        
        agents_to_test = [
            ("MarketAnalystAgent", "src.nuble.agents.market_analyst"),
            ("QuantAnalystAgent", "src.nuble.agents.quant_analyst"),
            ("NewsAnalystAgent", "src.nuble.agents.news_analyst"),
            ("FundamentalAnalystAgent", "src.nuble.agents.fundamental_analyst"),
            ("MacroAnalystAgent", "src.nuble.agents.macro_analyst"),
            ("RiskManagerAgent", "src.nuble.agents.risk_manager"),
            ("PortfolioOptimizerAgent", "src.nuble.agents.portfolio_optimizer"),
            ("CryptoSpecialistAgent", "src.nuble.agents.crypto_specialist"),
            ("EducatorAgent", "src.nuble.agents.educator"),
        ]
        
        for agent_name, module_path in agents_to_test:
            print(f"\n  Testing: {agent_name}")
            
            # Test import
            try:
                module = __import__(module_path, fromlist=[agent_name])
                AgentClass = getattr(module, agent_name)
                self.log(f"{agent_name} import", "OK")
            except Exception as e:
                self.log(f"{agent_name} import", "FAIL", str(e))
                continue
            
            # Test instantiation
            try:
                agent = AgentClass(api_key=self.api_key)
                self.log(f"{agent_name} init", "OK")
            except Exception as e:
                self.log(f"{agent_name} init", "FAIL", str(e))
                continue
            
            # Check required attributes
            required = ['agent_type', 'execute', 'api_key']
            for attr in required:
                if hasattr(agent, attr):
                    self.log(f"{agent_name}.{attr}", "OK")
                else:
                    self.log(f"{agent_name}.{attr}", "FAIL", "Missing")
    
    async def test_ml_components(self):
        """Test ML components."""
        print("\n" + "="*60)
        print("DEBUG: ML Components")
        print("="*60)
        
        # Test HMMRegimeModel
        try:
            from src.institutional.ml.regime import HMMRegimeModel
            model = HMMRegimeModel(n_regimes=3)
            self.log("HMMRegimeModel", "OK", "Loaded")
        except Exception as e:
            self.log("HMMRegimeModel", "FAIL", str(e))
        
        # Test TripleBarrierLabeling
        try:
            from src.institutional.ml.features import TripleBarrierLabeling
            labeler = TripleBarrierLabeling()
            self.log("TripleBarrierLabeling", "OK", "Loaded")
        except Exception as e:
            self.log("TripleBarrierLabeling", "FAIL", str(e))
        
        # Test EnsembleMLStrategy
        try:
            from src.institutional.ml.ensemble import EnsembleMLStrategy
            self.log("EnsembleMLStrategy import", "OK")
        except Exception as e:
            self.log("EnsembleMLStrategy import", "FAIL", str(e))
        
        # Test Analytics
        try:
            from src.institutional.analytics.technical import TechnicalAnalyzer
            analyzer = TechnicalAnalyzer()
            self.log("TechnicalAnalyzer", "OK")
        except Exception as e:
            self.log("TechnicalAnalyzer", "FAIL", str(e))
        
        try:
            from src.institutional.analytics.sentiment import SentimentAnalyzer
            analyzer = SentimentAnalyzer()
            self.log("SentimentAnalyzer", "OK")
        except Exception as e:
            self.log("SentimentAnalyzer", "FAIL", str(e))
    
    async def test_live_query(self):
        """Test a live query to verify full integration."""
        print("\n" + "="*60)
        print("DEBUG: Live Query Test")
        print("="*60)
        
        try:
            from src.nuble.agents.orchestrator import OrchestratorAgent
            o = OrchestratorAgent(api_key=self.api_key)
            
            result = await o.process(
                user_message="What is Apple's stock symbol?",
                conversation_id="debug_live",
                user_context={}
            )
            
            if result and 'message' in result:
                msg_len = len(result.get('message', ''))
                agents_used = result.get('agents_used', [])
                self.log("Live query", "OK", f"{msg_len} chars, agents: {agents_used}")
            else:
                self.log("Live query", "FAIL", "No message in result")
                
        except Exception as e:
            self.log("Live query", "FAIL", str(e))
            traceback.print_exc()
    
    def summary(self):
        """Print summary."""
        print("\n" + "="*70)
        print("INTEGRATION DEBUG SUMMARY")
        print("="*70)
        
        passed = sum(1 for r in self.results if r[1] == "OK")
        failed = sum(1 for r in self.results if r[1] == "FAIL")
        
        print(f"\n  Total checks: {len(self.results)}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        
        if failed > 0:
            print("\n  FAILURES:")
            for comp, status, detail in self.results:
                if status == "FAIL":
                    print(f"    ❌ {comp}: {detail}")
        
        return failed == 0


async def main():
    """Run debug."""
    print("="*70)
    print("PHASE 7: INTEGRATION FAILURE DEBUG")
    print(f"Time: {datetime.now().isoformat()}")
    print("="*70)
    
    debugger = IntegrationDebugger()
    
    await debugger.test_orchestrator_init()
    await debugger.test_each_agent_individually()
    await debugger.test_ml_components()
    await debugger.test_live_query()
    
    success = debugger.summary()
    
    if success:
        print("\n✅ ALL INTEGRATIONS VERIFIED")
    else:
        print("\n❌ INTEGRATION FAILURES FOUND - FIX REQUIRED")
    
    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
