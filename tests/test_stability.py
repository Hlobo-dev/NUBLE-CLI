#!/usr/bin/env python3
"""
Phase 8: Long-Running Stability Test

Run the system continuously and check for:
- Memory leaks
- Performance degradation
- Error accumulation
- Response time consistency
"""
import sys
import os
import asyncio
import time
import gc
import logging
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv()

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️ psutil not installed - memory monitoring disabled")


@dataclass
class StabilityMetrics:
    """Track metrics over the stability test."""
    requests: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    response_times: List[float] = None
    memory_samples: List[float] = None
    errors: List[str] = None
    
    def __post_init__(self):
        self.response_times = []
        self.memory_samples = []
        self.errors = []


class StabilityTester:
    """Long-running stability test suite."""
    
    # Varied queries to exercise different code paths
    QUERIES = [
        "What's AAPL price?",
        "Should I buy NVDA?",
        "MSFT technicals and RSI",
        "Compare AAPL vs GOOGL",
        "What are the risks of Tesla?",
        "Explain P/E ratio",
        "What's the market sentiment?",
        "Show me support and resistance for AMZN",
        "Bitcoin analysis please",
        "Portfolio allocation for $10k",
    ]
    
    def __init__(self, duration_minutes: int = 5, requests_per_minute: int = 4):
        self.duration_minutes = duration_minutes
        self.requests_per_minute = requests_per_minute
        self.metrics = StabilityMetrics()
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.process = psutil.Process() if HAS_PSUTIL else None
        
    def sample_memory(self) -> float:
        """Get current memory usage in MB."""
        if self.process:
            return self.process.memory_info().rss / 1024 / 1024
        return 0.0
    
    async def run_single_request(self, orchestrator, query: str, request_id: int) -> bool:
        """Run a single request and track metrics."""
        start = time.time()
        
        try:
            result = await asyncio.wait_for(
                orchestrator.process(
                    user_message=query,
                    conversation_id=f"stability_{request_id}",
                    user_context={}
                ),
                timeout=90  # 90 second timeout per request
            )
            
            elapsed = time.time() - start
            self.metrics.response_times.append(elapsed)
            
            if result and 'message' in result and len(result['message']) > 20:
                self.metrics.successes += 1
                return True
            else:
                self.metrics.failures += 1
                self.metrics.errors.append(f"Req {request_id}: empty response")
                return False
                
        except asyncio.TimeoutError:
            self.metrics.timeouts += 1
            self.metrics.errors.append(f"Req {request_id}: timeout")
            self.metrics.response_times.append(90.0)
            return False
        except Exception as e:
            self.metrics.failures += 1
            self.metrics.errors.append(f"Req {request_id}: {str(e)[:30]}")
            self.metrics.response_times.append(time.time() - start)
            return False
    
    def calculate_degradation(self) -> float:
        """Calculate performance degradation (first half vs second half)."""
        if len(self.metrics.response_times) < 4:
            return 0.0
        
        mid = len(self.metrics.response_times) // 2
        first_half = self.metrics.response_times[:mid]
        second_half = self.metrics.response_times[mid:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if first_avg > 0:
            return (second_avg - first_avg) / first_avg * 100
        return 0.0
    
    def calculate_memory_growth(self) -> float:
        """Calculate memory growth over the test."""
        if len(self.metrics.memory_samples) < 2:
            return 0.0
        return self.metrics.memory_samples[-1] - self.metrics.memory_samples[0]
    
    async def run(self):
        """Run the stability test."""
        if not self.api_key:
            print("❌ No ANTHROPIC_API_KEY found")
            return False
        
        print("\n" + "="*70)
        print(f"PHASE 8: STABILITY TEST")
        print(f"Duration: {self.duration_minutes} minutes | Rate: {self.requests_per_minute} req/min")
        print("="*70)
        
        from src.kyperian.agents.orchestrator import OrchestratorAgent
        orchestrator = OrchestratorAgent(api_key=self.api_key)
        
        request_interval = 60 / self.requests_per_minute
        start_time = time.time()
        end_time = start_time + (self.duration_minutes * 60)
        
        print(f"\nStarted: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Target end: {datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}")
        print("-"*70)
        
        # Track minute-by-minute
        minute_count = 0
        minute_start = time.time()
        minute_requests = 0
        minute_successes = 0
        
        # Initial memory sample
        self.metrics.memory_samples.append(self.sample_memory())
        
        while time.time() < end_time:
            # Select query (rotate through list)
            query = self.QUERIES[self.metrics.requests % len(self.QUERIES)]
            
            # Run request
            success = await self.run_single_request(
                orchestrator, query, self.metrics.requests
            )
            
            self.metrics.requests += 1
            minute_requests += 1
            if success:
                minute_successes += 1
            
            # Sample memory
            self.metrics.memory_samples.append(self.sample_memory())
            
            # Print minute summary
            if time.time() - minute_start >= 60:
                minute_count += 1
                recent_times = self.metrics.response_times[-minute_requests:] if minute_requests > 0 else [0]
                avg_time = sum(recent_times) / len(recent_times)
                current_mem = self.metrics.memory_samples[-1]
                
                print(f"  Minute {minute_count}: {minute_successes}/{minute_requests} ok, "
                      f"avg {avg_time:.1f}s, mem {current_mem:.0f}MB")
                
                minute_start = time.time()
                minute_requests = 0
                minute_successes = 0
            
            # Wait for next request
            await asyncio.sleep(request_interval)
            
            # Garbage collect periodically
            if self.metrics.requests % 10 == 0:
                gc.collect()
        
        # Final metrics
        total_time = time.time() - start_time
        degradation = self.calculate_degradation()
        memory_growth = self.calculate_memory_growth()
        
        print("\n" + "="*70)
        print("STABILITY TEST RESULTS")
        print("="*70)
        
        print(f"\n📊 REQUEST METRICS:")
        print(f"  Total requests: {self.metrics.requests}")
        print(f"  Successes: {self.metrics.successes} ({self.metrics.successes/self.metrics.requests*100:.1f}%)")
        print(f"  Failures: {self.metrics.failures}")
        print(f"  Timeouts: {self.metrics.timeouts}")
        print(f"  Duration: {total_time/60:.1f} minutes")
        
        if self.metrics.response_times:
            valid_times = [t for t in self.metrics.response_times if t < 90]
            if valid_times:
                print(f"\n⏱️ RESPONSE TIMES:")
                print(f"  Average: {sum(valid_times)/len(valid_times):.2f}s")
                print(f"  Min: {min(valid_times):.2f}s")
                print(f"  Max: {max(valid_times):.2f}s")
                print(f"  P95: {sorted(valid_times)[int(len(valid_times)*0.95)]:.2f}s")
        
        print(f"\n📈 DEGRADATION CHECK:")
        print(f"  Performance change: {degradation:+.1f}%")
        if degradation > 50:
            print("  ⚠️ WARNING: Significant performance degradation")
        elif degradation > 20:
            print("  ⚠️ NOTICE: Moderate performance degradation")
        else:
            print("  ✅ Performance stable")
        
        if self.metrics.memory_samples and HAS_PSUTIL:
            print(f"\n💾 MEMORY:")
            print(f"  Start: {self.metrics.memory_samples[0]:.0f}MB")
            print(f"  End: {self.metrics.memory_samples[-1]:.0f}MB")
            print(f"  Max: {max(self.metrics.memory_samples):.0f}MB")
            print(f"  Growth: {memory_growth:+.0f}MB")
            
            if memory_growth > 500:
                print("  ⚠️ WARNING: Possible memory leak")
            elif memory_growth > 200:
                print("  ⚠️ NOTICE: Moderate memory growth")
            else:
                print("  ✅ Memory stable")
        
        if self.metrics.errors:
            print(f"\n❌ ERRORS ({len(self.metrics.errors)}):")
            for error in self.metrics.errors[:5]:
                print(f"  • {error}")
            if len(self.metrics.errors) > 5:
                print(f"  ... and {len(self.metrics.errors) - 5} more")
        
        # Final verdict
        success_rate = self.metrics.successes / self.metrics.requests if self.metrics.requests > 0 else 0
        passed = (
            success_rate >= 0.90 and  # 90% success rate
            degradation < 100 and  # Less than 100% degradation
            (memory_growth < 500 or not HAS_PSUTIL)  # Memory growth < 500MB
        )
        
        print("\n" + "="*70)
        if passed:
            print("🏆 STABILITY TEST PASSED")
        else:
            print("❌ STABILITY TEST FAILED")
            if success_rate < 0.90:
                print(f"   Success rate {success_rate:.1%} < 90%")
            if degradation >= 100:
                print(f"   Degradation {degradation:.0f}% >= 100%")
            if memory_growth >= 500 and HAS_PSUTIL:
                print(f"   Memory growth {memory_growth:.0f}MB >= 500MB")
        print("="*70 + "\n")
        
        return passed


async def main():
    """Run stability test with configurable duration."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=5, help='Duration in minutes')
    parser.add_argument('--rate', type=int, default=4, help='Requests per minute')
    args = parser.parse_args()
    
    tester = StabilityTester(
        duration_minutes=args.duration,
        requests_per_minute=args.rate
    )
    success = await tester.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
