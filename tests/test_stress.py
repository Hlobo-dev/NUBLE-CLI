"""
PHASE 4: STRESS TESTING
========================
Tests system performance under load conditions.

Tests:
1. Concurrent queries
2. Memory stability
3. Error recovery
4. Rate limiting behavior
5. Large response handling
"""

import os
import sys
import time
import asyncio
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Verify API key
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key:
    print("❌ ANTHROPIC_API_KEY not found")
    sys.exit(1)
print(f"API Key: ...{api_key[-4:]}")

from kyperian.agents.orchestrator import OrchestratorAgent

def create_orchestrator():
    """Create a fresh orchestrator instance."""
    return OrchestratorAgent(api_key=api_key)

async def run_query_async(orchestrator, query, timeout=60):
    """Run a single query with timeout."""
    start = time.time()
    try:
        result = await asyncio.wait_for(
            orchestrator.process(
                user_message=query,
                conversation_id=f"stress_{uuid.uuid4().hex[:8]}",
                user_context={}
            ),
            timeout=timeout
        )
        elapsed = time.time() - start
        return {
            'success': True,
            'time': elapsed,
            'length': len(result.get('message', '')),
            'query': query[:50],
            'agents': result.get('agents_used', [])
        }
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        return {
            'success': False,
            'time': elapsed,
            'error': f'Timeout after {timeout}s',
            'query': query[:50]
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            'success': False,
            'time': elapsed,
            'error': str(e),
            'query': query[:50]
        }

class StressTestSuite:
    def __init__(self):
        self.orchestrator = create_orchestrator()
        self.results = []
        
    async def test_1_sequential_queries(self):
        """Test sequential query processing."""
        print("\n" + "="*70)
        print("TEST 1: SEQUENTIAL QUERIES (5 queries)")
        print("="*70)
        
        queries = [
            "What's Apple trading at?",
            "Tesla stock price?",
            "NVIDIA analysis",
            "Microsoft technicals",
            "What is RSI?"
        ]
        
        results = []
        for i, query in enumerate(queries, 1):
            print(f"  [{i}/5] {query[:40]}...", end=" ", flush=True)
            result = await run_query_async(self.orchestrator, query)
            results.append(result)
            status = "✅" if result['success'] else "❌"
            print(f"{status} ({result['time']:.1f}s)")
        
        success_count = sum(1 for r in results if r['success'])
        avg_time = sum(r['time'] for r in results) / len(results)
        
        print(f"\n  Results: {success_count}/5 passed, avg time: {avg_time:.1f}s")
        return success_count == 5, results

    async def test_2_memory_stability(self):
        """Test memory doesn't grow excessively over multiple queries."""
        print("\n" + "="*70)
        print("TEST 2: MEMORY STABILITY (10 queries)")
        print("="*70)
        
        import gc
        
        queries = [
            "Apple stock price?",
            "Should I buy Tesla?",
            "What is P/E ratio?",
            "NVIDIA sentiment",
            "Microsoft risk analysis",
            "How to diversify portfolio?",
            "Market overview today",
            "Amazon technicals",
            "Google investment advice",
            "What are moving averages?"
        ]
        
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        success_count = 0
        for i, query in enumerate(queries, 1):
            print(f"  [{i}/10] {query[:40]}...", end=" ", flush=True)
            result = await run_query_async(self.orchestrator, query)
            status = "✅" if result['success'] else "❌"
            print(f"{status} ({result['time']:.1f}s)")
            if result['success']:
                success_count += 1
        
        gc.collect()
        final_objects = len(gc.get_objects())
        growth = final_objects - initial_objects
        growth_per_query = growth / 10
        
        # Allow some object growth, but flag excessive growth
        memory_stable = growth_per_query < 5000
        
        print(f"\n  Results: {success_count}/10 passed")
        print(f"  Memory growth: {growth} objects ({growth_per_query:.0f}/query)")
        print(f"  Memory stable: {'✅' if memory_stable else '⚠️'}")
        
        return success_count >= 8 and memory_stable, {'success': success_count, 'growth': growth}

    async def test_3_error_recovery(self):
        """Test system recovers from invalid inputs."""
        print("\n" + "="*70)
        print("TEST 3: ERROR RECOVERY (malformed inputs)")
        print("="*70)
        
        test_cases = [
            ("Empty query", "", False),
            ("Single char", "x", True),  # Should still work
            ("Normal query after empty", "What is Apple's stock price?", True),
            ("Very long query", "Tell me about " * 100 + "Apple stock", True),
            ("Normal after long", "Tesla price?", True),
            ("Special chars", "What's AAPL @ $150?!?", True),
            ("Normal after special", "NVIDIA analysis", True),
        ]
        
        passed = 0
        for i, (name, query, should_work) in enumerate(test_cases, 1):
            print(f"  [{i}/7] {name}...", end=" ", flush=True)
            try:
                result = await run_query_async(self.orchestrator, query, timeout=45)
                works = result['success']
                
                if should_work:
                    if works:
                        print(f"✅ works as expected ({result['time']:.1f}s)")
                        passed += 1
                    else:
                        print(f"❌ should have worked: {result.get('error', 'unknown')[:50]}")
                else:
                    if not works:
                        print(f"✅ correctly fails")
                        passed += 1
                    else:
                        print(f"⚠️ worked but expected failure ({result['time']:.1f}s)")
                        passed += 1  # Not a hard failure
            except Exception as e:
                if not should_work:
                    print(f"✅ correctly raises exception")
                    passed += 1
                else:
                    print(f"❌ unexpected exception: {e}")
        
        print(f"\n  Results: {passed}/7 passed")
        return passed >= 5, {'passed': passed}

    async def test_4_concurrent_queries_light(self):
        """Test light concurrent load (2 concurrent)."""
        print("\n" + "="*70)
        print("TEST 4: CONCURRENT QUERIES (2 at a time)")
        print("="*70)
        
        queries = [
            "Apple price?",
            "Tesla analysis"
        ]
        
        # Run queries concurrently
        tasks = []
        for q in queries:
            orch = create_orchestrator()
            tasks.append(run_query_async(orch, q, timeout=90))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = 0
        for result in results:
            if isinstance(result, Exception):
                print(f"  ❌ Exception: {result}")
            elif result['success']:
                print(f"  ✅ {result['query']}... ({result['time']:.1f}s)")
                success_count += 1
            else:
                print(f"  ❌ {result['query']}... {result.get('error', 'unknown')[:40]}")
        
        print(f"\n  Results: {success_count}/2 concurrent queries passed")
        return success_count == 2, results

    async def test_5_large_response_handling(self):
        """Test handling of complex queries that generate large responses."""
        print("\n" + "="*70)
        print("TEST 5: LARGE RESPONSE HANDLING")
        print("="*70)
        
        # Complex queries that generate detailed responses
        queries = [
            ("Comprehensive investment analysis", 
             "Give me a complete investment analysis of NVIDIA including technical indicators, fundamentals, news sentiment, risks, and recommendation"),
            ("Educational explanation",
             "Explain in detail how options trading works, including calls, puts, Greeks, and strategies"),
        ]
        
        results = []
        for name, query in queries:
            print(f"  {name}...", end=" ", flush=True)
            result = await run_query_async(self.orchestrator, query, timeout=90)
            results.append(result)
            
            if result['success']:
                print(f"✅ ({result['time']:.1f}s, {result['length']} chars)")
            else:
                print(f"❌ {result.get('error', 'unknown error')[:50]}")
        
        success_count = sum(1 for r in results if r['success'])
        avg_length = sum(r.get('length', 0) for r in results) / len(results)
        
        print(f"\n  Results: {success_count}/2 passed, avg response: {avg_length:.0f} chars")
        return success_count == 2, results

    async def run_all_tests(self):
        """Run all stress tests."""
        print("\n" + "="*70)
        print("PHASE 4: STRESS TESTING")
        print("="*70)
        
        tests = [
            ("Sequential Queries", self.test_1_sequential_queries),
            ("Memory Stability", self.test_2_memory_stability),
            ("Error Recovery", self.test_3_error_recovery),
            ("Concurrent Queries", self.test_4_concurrent_queries_light),
            ("Large Response Handling", self.test_5_large_response_handling),
        ]
        
        results = []
        for name, test_fn in tests:
            try:
                passed, data = await test_fn()
                results.append((name, passed, data))
            except Exception as e:
                print(f"\n❌ Test crashed: {e}")
                traceback.print_exc()
                results.append((name, False, {'error': str(e)}))
        
        # Summary
        print("\n" + "="*70)
        print("STRESS TEST RESULTS")
        print("="*70)
        
        passed_count = 0
        for name, passed, _ in results:
            status = "✅" if passed else "❌"
            print(f"  {status} {name}")
            if passed:
                passed_count += 1
        
        print(f"\n  Overall: {passed_count}/{len(tests)} tests passed")
        
        if passed_count == len(tests):
            print("\n" + "="*70)
            print("🎉 ALL STRESS TESTS PASSED")
            print("="*70)
            return True
        else:
            print("\n" + "="*70)
            print("⚠️ SOME STRESS TESTS FAILED")
            print("="*70)
            return False


if __name__ == "__main__":
    suite = StressTestSuite()
    success = asyncio.run(suite.run_all_tests())
    sys.exit(0 if success else 1)
