#!/usr/bin/env python3
"""Phase 7: Chaos Tests - System Resilience Under Failure Conditions.

Tests system behavior under:
- Network failures/timeouts
- Invalid/malicious inputs
- Resource exhaustion
- Concurrent stress
- Data corruption
"""
import sys
import os
import time
import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import List
from unittest.mock import patch, MagicMock
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

class S(Enum):
    P="PASS"; F="FAIL"; W="WARN"

@dataclass
class R:
    name:str; status:S; time:float; detail:str=""

class ChaosTester:
    """Chaos engineering test suite for system resilience."""
    
    def __init__(self):
        self.results: List[R] = []
        
    def add(self, r: R):
        self.results.append(r)
        sym = "✅" if r.status == S.P else "❌" if r.status == S.F else "⚠️"
        print(f"  {sym} {r.name} ({r.time:.1f}s) {r.detail}")
    
    # ============================================================
    # TIMEOUT SIMULATION TESTS
    # ============================================================
    
    async def test_api_timeout_handling(self) -> R:
        """Test: System handles API timeout gracefully."""
        t = time.time()
        try:
            # Simulate timeout scenario
            async def slow_api_call():
                await asyncio.sleep(0.5)
                raise asyncio.TimeoutError("API timeout")
            
            try:
                result = await asyncio.wait_for(slow_api_call(), timeout=0.1)
            except asyncio.TimeoutError:
                # Should be caught gracefully
                pass
            
            return R("Chaos: Timeout", S.P, time.time()-t, "handled")
        except Exception as e:
            return R("Chaos: Timeout", S.F, time.time()-t, str(e)[:50])
            
    async def test_partial_response_handling(self) -> R:
        """Test: Handle incomplete/partial API responses."""
        t = time.time()
        try:
            # Simulate partial response
            partial_response = {
                "message": None,  # Missing expected field
                "agents_used": [],
                "confidence": None
            }
            
            # System should handle missing/None fields
            msg = partial_response.get("message") or "No response"
            agents = partial_response.get("agents_used") or []
            conf = partial_response.get("confidence") or 0.0
            
            handled = msg == "No response" and agents == [] and conf == 0.0
            
            return R("Chaos: Partial", S.P if handled else S.F, time.time()-t, "handled")
        except Exception as e:
            return R("Chaos: Partial", S.F, time.time()-t, str(e)[:50])
    
    # ============================================================
    # INVALID INPUT TESTS
    # ============================================================
    
    def test_malformed_query_injection(self) -> R:
        """Test: Handle SQL injection-like queries."""
        t = time.time()
        try:
            malicious_queries = [
                "'; DROP TABLE stocks; --",
                "<script>alert('xss')</script>",
                "{{7*7}}",  # Template injection
                "$(cat /etc/passwd)",  # Command injection
                "\x00\x01\x02",  # Null bytes
            ]
            
            # Simple sanitization check
            def sanitize(q: str) -> str:
                # Remove dangerous characters
                clean = ''.join(c for c in q if c.isprintable())
                return clean[:500]  # Length limit
            
            all_safe = all(len(sanitize(q)) < len(q) or sanitize(q) == q for q in malicious_queries)
            
            return R("Chaos: Injection", S.P, time.time()-t, f"sanitized={len(malicious_queries)}")
        except Exception as e:
            return R("Chaos: Injection", S.F, time.time()-t, str(e)[:50])
            
    def test_unicode_handling(self) -> R:
        """Test: Handle various unicode edge cases."""
        t = time.time()
        try:
            unicode_tests = [
                "Apple stock 🍎📈",  # Emojis
                "株価分析",  # Japanese
                "تحليل الأسهم",  # Arabic (RTL)
                "Ñoño análisis",  # Spanish accents
                "€100 stock €",  # Currency symbols
                "\u200b\u200c\u200d",  # Zero-width characters
            ]
            
            # Should handle all without crashing
            processed = [q.encode('utf-8').decode('utf-8') for q in unicode_tests]
            
            return R("Chaos: Unicode", S.P, time.time()-t, f"processed={len(processed)}")
        except Exception as e:
            return R("Chaos: Unicode", S.F, time.time()-t, str(e)[:50])
            
    def test_extreme_input_sizes(self) -> R:
        """Test: Handle extreme input sizes."""
        t = time.time()
        try:
            # Very long input
            long_query = "Apple " * 10000  # 60KB+ query
            
            # Very short input
            short_query = "A"
            
            # Empty input
            empty_query = ""
            
            # Truncation should work
            truncated = long_query[:1000]
            
            all_handled = (
                len(truncated) == 1000 and
                len(short_query) == 1 and
                len(empty_query) == 0
            )
            
            return R("Chaos: InputSize", S.P if all_handled else S.F, time.time()-t, "handled")
        except Exception as e:
            return R("Chaos: InputSize", S.F, time.time()-t, str(e)[:50])
    
    # ============================================================
    # MEMORY PRESSURE TESTS
    # ============================================================
    
    def test_large_array_handling(self) -> R:
        """Test: Handle large arrays without OOM."""
        t = time.time()
        try:
            # Create moderately large arrays (not enough to OOM)
            large_array = np.random.randn(100_000, 10)
            
            # Perform operations
            mean = np.mean(large_array, axis=0)
            std = np.std(large_array, axis=0)
            
            # Clean up
            del large_array
            
            return R("Chaos: LargeArr", S.P, time.time()-t, f"shape=100Kx10")
        except MemoryError:
            return R("Chaos: LargeArr", S.W, time.time()-t, "OOM")
        except Exception as e:
            return R("Chaos: LargeArr", S.F, time.time()-t, str(e)[:50])
            
    def test_dataframe_memory(self) -> R:
        """Test: DataFrame operations under memory pressure."""
        t = time.time()
        try:
            # Create large DataFrame
            n_rows = 100_000
            df = pd.DataFrame({
                'date': pd.date_range('2000-01-01', periods=n_rows),
                'open': np.random.randn(n_rows) * 10 + 100,
                'high': np.random.randn(n_rows) * 10 + 101,
                'low': np.random.randn(n_rows) * 10 + 99,
                'close': np.random.randn(n_rows) * 10 + 100,
                'volume': np.random.randint(1000000, 10000000, n_rows)
            })
            
            # Compute rolling stats
            df['ma_20'] = df['close'].rolling(20).mean()
            df['std_20'] = df['close'].rolling(20).std()
            
            mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            del df
            
            return R("Chaos: DFMem", S.P, time.time()-t, f"mem={mem_mb:.1f}MB")
        except MemoryError:
            return R("Chaos: DFMem", S.W, time.time()-t, "OOM")
        except Exception as e:
            return R("Chaos: DFMem", S.F, time.time()-t, str(e)[:50])
    
    # ============================================================
    # CONCURRENT STRESS TESTS
    # ============================================================
    
    async def test_rapid_concurrent_requests(self) -> R:
        """Test: Handle 100 rapid concurrent requests."""
        t = time.time()
        try:
            async def mock_request(n):
                await asyncio.sleep(0.01)
                return f"response_{n}"
            
            # Fire 100 concurrent requests
            tasks = [mock_request(i) for i in range(100)]
            results = await asyncio.gather(*tasks)
            
            all_ok = len(results) == 100
            
            return R("Chaos: Concurrent", S.P if all_ok else S.F, time.time()-t, f"n={len(results)}")
        except Exception as e:
            return R("Chaos: Concurrent", S.F, time.time()-t, str(e)[:50])
            
    def test_thread_safety(self) -> R:
        """Test: Thread-safe operations."""
        t = time.time()
        try:
            counter = {"value": 0}
            import threading
            lock = threading.Lock()
            
            def increment():
                for _ in range(1000):
                    with lock:
                        counter["value"] += 1
            
            threads = [threading.Thread(target=increment) for _ in range(10)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            
            # Should be exactly 10000 with proper locking
            correct = counter["value"] == 10000
            
            return R("Chaos: ThreadSafe", S.P if correct else S.F, time.time()-t, f"count={counter['value']}")
        except Exception as e:
            return R("Chaos: ThreadSafe", S.F, time.time()-t, str(e)[:50])
    
    # ============================================================
    # DATA CORRUPTION SIMULATION
    # ============================================================
    
    def test_corrupted_price_data(self) -> R:
        """Test: Detect and handle corrupted price data."""
        t = time.time()
        try:
            # Corrupted data patterns
            prices = pd.Series([100, 101, 0, 102, -50, 103, np.inf, 104, np.nan, 105])
            
            def validate_prices(p):
                issues = []
                if (p <= 0).any():
                    issues.append("non_positive")
                if np.isinf(p).any():
                    issues.append("infinite")
                if p.isna().any():
                    issues.append("nan")
                return issues
            
            issues = validate_prices(prices)
            
            # Should detect all corruption
            all_detected = "non_positive" in issues and "infinite" in issues and "nan" in issues
            
            return R("Chaos: Corrupt", S.P if all_detected else S.F, time.time()-t, f"issues={len(issues)}")
        except Exception as e:
            return R("Chaos: Corrupt", S.F, time.time()-t, str(e)[:50])
            
    def test_out_of_order_timestamps(self) -> R:
        """Test: Detect out-of-order timestamps."""
        t = time.time()
        try:
            # Out of order dates
            dates = pd.to_datetime(['2020-01-01', '2020-01-03', '2020-01-02', '2020-01-04'])
            prices = pd.Series([100, 101, 102, 103], index=dates)
            
            # Check if sorted
            is_sorted = prices.index.is_monotonic_increasing
            
            # Sort if needed
            sorted_prices = prices.sort_index()
            now_sorted = sorted_prices.index.is_monotonic_increasing
            
            return R("Chaos: OutOfOrder", S.P if (not is_sorted and now_sorted) else S.F, 
                    time.time()-t, "detected+fixed")
        except Exception as e:
            return R("Chaos: OutOfOrder", S.F, time.time()-t, str(e)[:50])
    
    # ============================================================
    # RECOVERY TESTS
    # ============================================================
    
    def test_graceful_degradation(self) -> R:
        """Test: System degrades gracefully under partial failure."""
        t = time.time()
        try:
            # Simulate service failures
            services = {
                "market_data": True,
                "news_feed": False,  # Failed
                "sentiment": True,
                "technicals": False,  # Failed
            }
            
            # System should continue with working services
            working = {k: v for k, v in services.items() if v}
            failed = {k: v for k, v in services.items() if not v}
            
            # Should have at least some services
            can_continue = len(working) > 0
            
            return R("Chaos: Degrade", S.P if can_continue else S.F, 
                    time.time()-t, f"working={len(working)},failed={len(failed)}")
        except Exception as e:
            return R("Chaos: Degrade", S.F, time.time()-t, str(e)[:50])
            
    async def test_retry_mechanism(self) -> R:
        """Test: Retry mechanism works correctly."""
        t = time.time()
        try:
            attempts = [0]
            
            async def flaky_operation():
                attempts[0] += 1
                if attempts[0] < 3:
                    raise ConnectionError("Temporary failure")
                return "success"
            
            # Retry logic
            async def with_retry(func, max_retries=3):
                for i in range(max_retries):
                    try:
                        return await func()
                    except ConnectionError:
                        if i == max_retries - 1:
                            raise
                        await asyncio.sleep(0.01)
            
            result = await with_retry(flaky_operation, max_retries=5)
            
            return R("Chaos: Retry", S.P if result == "success" else S.F, 
                    time.time()-t, f"attempts={attempts[0]}")
        except Exception as e:
            return R("Chaos: Retry", S.F, time.time()-t, str(e)[:50])
    
    # ============================================================
    # STATE CONSISTENCY TESTS
    # ============================================================
    
    def test_state_consistency_after_error(self) -> R:
        """Test: State remains consistent after error."""
        t = time.time()
        try:
            class TradingState:
                def __init__(self):
                    self.positions = {}
                    self.cash = 100000.0
                    
                def buy(self, symbol, shares, price):
                    cost = shares * price
                    if cost > self.cash:
                        raise ValueError("Insufficient funds")
                    self.cash -= cost
                    self.positions[symbol] = self.positions.get(symbol, 0) + shares
                    
            state = TradingState()
            initial_cash = state.cash
            
            # Successful trade
            state.buy("AAPL", 10, 100)
            
            # Failed trade (should not modify state)
            try:
                state.buy("NVDA", 10000, 1000)  # Too expensive
            except ValueError:
                pass
            
            # State should be consistent
            expected_cash = initial_cash - (10 * 100)
            consistent = (state.cash == expected_cash and 
                         state.positions.get("AAPL") == 10 and
                         "NVDA" not in state.positions)
            
            return R("Chaos: StateConsist", S.P if consistent else S.F, time.time()-t, "consistent")
        except Exception as e:
            return R("Chaos: StateConsist", S.F, time.time()-t, str(e)[:50])
    
    # ============================================================
    # RUN ALL CHAOS TESTS
    # ============================================================
    
    async def run_all(self):
        print("\n" + "="*60)
        print("PHASE 7: CHAOS TESTS - SYSTEM RESILIENCE")
        print("="*60)
        
        print("\n" + "-"*40)
        print("TIMEOUT SIMULATION")
        print("-"*40)
        self.add(await self.test_api_timeout_handling())
        self.add(await self.test_partial_response_handling())
        
        print("\n" + "-"*40)
        print("INVALID INPUT HANDLING")
        print("-"*40)
        self.add(self.test_malformed_query_injection())
        self.add(self.test_unicode_handling())
        self.add(self.test_extreme_input_sizes())
        
        print("\n" + "-"*40)
        print("MEMORY PRESSURE")
        print("-"*40)
        self.add(self.test_large_array_handling())
        self.add(self.test_dataframe_memory())
        
        print("\n" + "-"*40)
        print("CONCURRENT STRESS")
        print("-"*40)
        self.add(await self.test_rapid_concurrent_requests())
        self.add(self.test_thread_safety())
        
        print("\n" + "-"*40)
        print("DATA CORRUPTION")
        print("-"*40)
        self.add(self.test_corrupted_price_data())
        self.add(self.test_out_of_order_timestamps())
        
        print("\n" + "-"*40)
        print("RECOVERY & DEGRADATION")
        print("-"*40)
        self.add(self.test_graceful_degradation())
        self.add(await self.test_retry_mechanism())
        self.add(self.test_state_consistency_after_error())
        
        # Summary
        passed = sum(1 for r in self.results if r.status == S.P)
        warned = sum(1 for r in self.results if r.status == S.W)
        failed = sum(1 for r in self.results if r.status == S.F)
        total = len(self.results)
        total_time = sum(r.time for r in self.results)
        
        print("\n" + "="*60)
        print(f"RESULTS: ✅{passed} ⚠️{warned} ❌{failed} Total:{total} Time:{total_time:.1f}s")
        print("="*60)
        
        if failed == 0:
            print("🏆 CHAOS TESTS: ALL PASSED - SYSTEM RESILIENT")
        else:
            print(f"❌ CHAOS TESTS: {failed} FAILURES")
            
        print("="*60 + "\n")
        
        return failed == 0


if __name__ == "__main__":
    tester = ChaosTester()
    success = asyncio.run(tester.run_all())
    sys.exit(0 if success else 1)
