#!/usr/bin/env python3
"""
NUBLE Elite - Comprehensive Test Suite

Tests all components of the multi-agent cognitive system.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add source path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_base_components():
    """Test base agent components."""
    print("\n" + "="*60)
    print("TEST 1: Base Agent Components")
    print("="*60)
    
    try:
        from nuble.agents.base import (
            AgentType, AgentTask, AgentResult, TaskPriority
        )
        
        # Test AgentType enum
        assert len(AgentType) == 9, f"Expected 9 agent types, got {len(AgentType)}"
        print(f"✓ AgentType enum has {len(AgentType)} agents:")
        for agent in AgentType:
            print(f"  - {agent.value}")
        
        # Test AgentTask
        task = AgentTask(
            task_id="test_1",
            agent_type=AgentType.MARKET_ANALYST,
            instruction="Analyze AAPL",
            context={'symbols': ['AAPL']},
            priority=TaskPriority.HIGH
        )
        assert task.task_id == "test_1"
        print("✓ AgentTask creation works")
        
        # Test AgentResult
        result = AgentResult(
            task_id="test_1",
            agent_type=AgentType.MARKET_ANALYST,
            success=True,
            data={'price': 178.50},
            confidence=0.85,
            execution_time_ms=150
        )
        assert result.success == True
        assert result.confidence == 0.85
        print("✓ AgentResult creation works")
        
        print("\n✅ BASE COMPONENTS: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ BASE COMPONENTS: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_specialized_agents():
    """Test all 9 specialized agents."""
    print("\n" + "="*60)
    print("TEST 2: Specialized Agents (All 9)")
    print("="*60)
    
    try:
        from nuble.agents import (
            MarketAnalystAgent,
            QuantAnalystAgent,
            NewsAnalystAgent,
            FundamentalAnalystAgent,
            MacroAnalystAgent,
            RiskManagerAgent,
            PortfolioOptimizerAgent,
            CryptoSpecialistAgent,
            EducatorAgent
        )
        
        agents = [
            ("MarketAnalyst", MarketAnalystAgent),
            ("QuantAnalyst", QuantAnalystAgent),
            ("NewsAnalyst", NewsAnalystAgent),
            ("FundamentalAnalyst", FundamentalAnalystAgent),
            ("MacroAnalyst", MacroAnalystAgent),
            ("RiskManager", RiskManagerAgent),
            ("PortfolioOptimizer", PortfolioOptimizerAgent),
            ("CryptoSpecialist", CryptoSpecialistAgent),
            ("Educator", EducatorAgent),
        ]
        
        for name, AgentClass in agents:
            agent = AgentClass()
            capabilities = agent.get_capabilities()
            
            assert 'name' in capabilities
            assert 'capabilities' in capabilities
            
            print(f"✓ {name}: {len(capabilities.get('capabilities', []))} capabilities")
        
        print(f"\n✓ All 9 specialized agents initialized successfully")
        
        print("\n✅ SPECIALIZED AGENTS: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ SPECIALIZED AGENTS: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orchestrator():
    """Test the Orchestrator Agent."""
    print("\n" + "="*60)
    print("TEST 3: Orchestrator Agent")
    print("="*60)
    
    try:
        from nuble.agents import OrchestratorAgent, OrchestratorConfig
        
        # Create with config
        config = OrchestratorConfig(
            use_opus=True,
            max_parallel_agents=5,
            verbose_logging=False
        )
        
        orchestrator = OrchestratorAgent(config=config)
        
        # Check initialization
        assert orchestrator.orchestrator_model in [
            "claude-opus-4-5-20250514",
            "claude-sonnet-4-20250514"
        ], f"Unexpected model: {orchestrator.orchestrator_model}"
        print(f"✓ Orchestrator uses {orchestrator.orchestrator_model}")
        
        # Check agent initialization
        orchestrator._initialize_agents()
        available = orchestrator.get_available_agents()
        print(f"✓ Available agents: {len(available)}")
        
        # Test symbol extraction
        symbols = orchestrator._extract_symbols("Should I buy AAPL and MSFT?")
        assert 'AAPL' in symbols or 'MSFT' in symbols
        print(f"✓ Symbol extraction: {symbols}")
        
        # Test simple planning (without Claude)
        from nuble.agents.base import AgentType
        context_mock = type('Context', (), {
            'user_profile': {'risk_tolerance': 'moderate'},
            'active_symbols': []
        })()
        
        plan = orchestrator._simple_plan("Should I buy AAPL?", context_mock)
        assert 'tasks' in plan
        print(f"✓ Simple planning: {len(plan['tasks'])} tasks generated")
        
        print("\n✅ ORCHESTRATOR: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ ORCHESTRATOR: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_manager():
    """Test the Memory Manager."""
    print("\n" + "="*60)
    print("TEST 4: Memory Manager")
    print("="*60)
    
    try:
        from nuble.memory import MemoryManager, UserProfile, Conversation, Prediction
        import tempfile
        import os
        
        # Use temp file for test
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            test_db = f.name
        
        memory = MemoryManager(db_path=test_db)
        
        # Test user profile
        profile = UserProfile(
            user_id="test_user_123",
            name="Test User",
            risk_tolerance="aggressive",
            portfolio={"AAPL": 10000, "MSFT": 5000},
            watchlist=["NVDA", "TSLA"]
        )
        memory.save_user_profile(profile)
        
        retrieved = memory.get_user_profile("test_user_123")
        assert retrieved is not None
        assert retrieved.name == "Test User"
        assert retrieved.risk_tolerance == "aggressive"
        assert retrieved.portfolio["AAPL"] == 10000
        print("✓ User profile: save and retrieve works")
        
        # Test conversation
        conv = Conversation(
            conversation_id="conv_456",
            user_id="test_user_123"
        )
        conv.add_message("user", "Should I buy AAPL?")
        conv.add_message("assistant", "Based on my analysis...")
        memory.save_conversation(conv)
        
        retrieved_conv = memory.get_conversation("conv_456")
        assert retrieved_conv is not None
        assert len(retrieved_conv.messages) == 2
        print("✓ Conversation: save and retrieve works")
        
        # Test prediction
        pred = Prediction(
            prediction_id="pred_789",
            user_id="test_user_123",
            symbol="AAPL",
            prediction_type="DIRECTION",
            predicted_value="UP",
            confidence=0.75,
            horizon_days=5
        )
        memory.save_prediction(pred)
        
        retrieved_pred = memory.get_prediction("pred_789")
        assert retrieved_pred is not None
        assert retrieved_pred.symbol == "AAPL"
        print("✓ Prediction: save and retrieve works")
        
        # Test resolve prediction
        memory.resolve_prediction("pred_789", "UP", True)
        resolved = memory.get_prediction("pred_789")
        assert resolved.was_correct == True
        print("✓ Prediction resolution works")
        
        # Test accuracy
        accuracy = memory.get_prediction_accuracy(user_id="test_user_123")
        assert accuracy['total'] == 1
        assert accuracy['correct'] == 1
        assert accuracy['accuracy'] == 1.0
        print("✓ Prediction accuracy calculation works")
        
        # Test feedback
        memory.save_feedback("test_user_123", "conv_456", 5, "Great!")
        avg = memory.get_average_rating("test_user_123")
        assert avg == 5.0
        print("✓ Feedback system works")
        
        # Test stats
        stats = memory.get_stats()
        assert stats['users'] == 1
        assert stats['conversations'] == 1
        assert stats['predictions'] == 1
        print(f"✓ Stats: {stats}")
        
        # Cleanup
        os.unlink(test_db)
        
        print("\n✅ MEMORY MANAGER: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ MEMORY MANAGER: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_execution():
    """Test async agent execution."""
    print("\n" + "="*60)
    print("TEST 5: Agent Execution (Async)")
    print("="*60)
    
    try:
        from nuble.agents import MarketAnalystAgent, AgentTask, AgentType
        
        agent = MarketAnalystAgent()
        
        task = AgentTask(
            task_id="exec_test_1",
            agent_type=AgentType.MARKET_ANALYST,
            instruction="Analyze AAPL price and technicals",
            context={'symbols': ['AAPL'], 'query': 'analyze AAPL'}
        )
        
        result = await agent.execute(task)
        
        assert result.success == True
        assert result.task_id == "exec_test_1"
        assert result.agent_type == AgentType.MARKET_ANALYST
        assert result.confidence > 0
        
        print(f"✓ Execution successful")
        print(f"  - Task ID: {result.task_id}")
        print(f"  - Success: {result.success}")
        print(f"  - Confidence: {result.confidence:.2%}")
        print(f"  - Execution time: {result.execution_time_ms}ms")
        
        # Check data structure
        data = result.data
        if 'AAPL' in data:
            aapl_data = data['AAPL']
            if 'technicals' in aapl_data:
                print(f"  - Has technicals: Yes")
            if 'signal' in aapl_data:
                print(f"  - Signal: {aapl_data['signal'].get('recommendation', 'N/A')}")
        
        print("\n✅ AGENT EXECUTION: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ AGENT EXECUTION: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_orchestration():
    """Test full orchestration flow."""
    print("\n" + "="*60)
    print("TEST 6: Full Orchestration Flow")
    print("="*60)
    
    try:
        from nuble.agents import OrchestratorAgent
        
        orchestrator = OrchestratorAgent()
        
        # Simple query
        result = await orchestrator.process(
            user_message="What's the price of AAPL?",
            conversation_id="test_conv_001",
            user_context={'risk_tolerance': 'moderate'}
        )
        
        assert 'message' in result
        assert 'data' in result
        assert 'conversation_id' in result
        
        print(f"✓ Orchestration completed")
        print(f"  - Response length: {len(result['message'])} chars")
        print(f"  - Agents used: {result.get('agents_used', [])}")
        print(f"  - Symbols: {result.get('symbols', [])}")
        print(f"  - Confidence: {result.get('confidence', 0):.2%}")
        print(f"  - Execution time: {result.get('execution_time_seconds', 0):.2f}s")
        
        print("\n✅ FULL ORCHESTRATION: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ FULL ORCHESTRATION: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_components():
    """Test API components (without running server)."""
    print("\n" + "="*60)
    print("TEST 7: API Components")
    print("="*60)
    
    try:
        # Check if FastAPI available
        try:
            import fastapi
            print("✓ FastAPI is installed")
        except ImportError:
            print("⚠️ FastAPI not installed - skipping API tests")
            return True
        
        from nuble.api.main import create_app
        
        app = create_app(enable_memory=True, enable_cors=True)
        
        assert app is not None
        assert app.title == "NUBLE Elite API"
        print("✓ API app created successfully")
        
        # Check routes
        routes = [route.path for route in app.routes]
        expected = ['/health', '/chat', '/quick/{symbol}', '/agents']
        
        for expected_route in expected:
            found = any(expected_route in r for r in routes)
            print(f"  - {expected_route}: {'✓' if found else '✗'}")
        
        print("\n✅ API COMPONENTS: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ API COMPONENTS: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("   NUBLE ELITE - MULTI-AGENT COGNITIVE SYSTEM")
    print("   Comprehensive Test Suite")
    print("="*70)
    
    start_time = datetime.now()
    
    results = []
    
    # Sync tests
    results.append(("Base Components", test_base_components()))
    results.append(("Specialized Agents", test_specialized_agents()))
    results.append(("Orchestrator", test_orchestrator()))
    results.append(("Memory Manager", test_memory_manager()))
    
    # Async tests
    results.append(("Agent Execution", asyncio.run(test_agent_execution())))
    results.append(("Full Orchestration", asyncio.run(test_full_orchestration())))
    
    # API tests
    results.append(("API Components", test_api_components()))
    
    # Summary
    print("\n" + "="*70)
    print("   TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name:30} {status}")
    
    duration = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "-"*70)
    print(f"  Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print(f"  Duration: {duration:.2f}s")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! NUBLE ELITE IS READY!")
        print("\nNext Steps:")
        print("  1. Set ANTHROPIC_API_KEY environment variable")
        print("  2. Run: python -m nuble.api.main")
        print("  3. Open: http://localhost:8000/docs")
        print("\n" + "="*70)
        return 0
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
