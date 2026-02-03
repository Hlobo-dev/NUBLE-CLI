#!/usr/bin/env python3
"""
Extended Real User Scenario Tests

These are comprehensive scenarios that test the full system.
Every single one must pass.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add source paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Load .env file
env_file = Path(__file__).parent.parent / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value


# Extended scenarios with stricter validation
EXTENDED_SCENARIOS = [
    # ===========================================
    # CATEGORY 1: SIMPLE PRICE QUERIES
    # ===========================================
    {
        "category": "Simple Price",
        "query": "What's Apple trading at?",
        "must_contain": ["$"],
        "must_contain_any": ["AAPL", "Apple"],
        "must_not_contain": ["error", "cannot", "don't know"],
        "max_time": 30,
        "min_response_length": 50
    },
    {
        "category": "Simple Price",
        "query": "NVDA price",
        "must_contain": ["$"],
        "must_contain_any": ["NVIDIA", "NVDA"],
        "must_not_contain": ["error"],
        "max_time": 30,
        "min_response_length": 30
    },
    
    # ===========================================
    # CATEGORY 2: INVESTMENT DECISIONS
    # ===========================================
    {
        "category": "Investment Decision",
        "query": "Should I buy Tesla?",
        "must_contain_any": ["buy", "sell", "hold", "recommend", "consider", "wait"],
        "must_contain": ["risk"],
        "must_not_contain": ["error", "cannot analyze"],
        "max_time": 60,
        "min_response_length": 300
    },
    {
        "category": "Investment Decision",
        "query": "Is NVIDIA a good investment right now?",
        "must_contain_any": ["yes", "no", "depends", "recommend", "consider", "investment"],
        "must_not_contain": ["error"],
        "max_time": 60,
        "min_response_length": 300
    },
    
    # ===========================================
    # CATEGORY 3: TECHNICAL ANALYSIS
    # ===========================================
    {
        "category": "Technical Analysis",
        "query": "What's the RSI for Apple?",
        "must_contain": ["RSI"],
        "must_contain_any": ["overbought", "oversold", "neutral", "momentum", "value"],
        "must_not_contain": ["error"],
        "max_time": 30,
        "min_response_length": 100
    },
    {
        "category": "Technical Analysis",
        "query": "Show me MSFT technicals",
        "must_contain_any": ["RSI", "MACD", "moving average", "support", "resistance", "technical"],
        "must_not_contain": ["error"],
        "max_time": 30,
        "min_response_length": 200
    },
    
    # ===========================================
    # CATEGORY 4: SENTIMENT/NEWS
    # ===========================================
    {
        "category": "Sentiment",
        "query": "What's the news sentiment on NVIDIA?",
        "must_contain_any": ["positive", "negative", "neutral", "bullish", "bearish", "sentiment", "news"],
        "must_not_contain": ["error"],
        "max_time": 30,
        "min_response_length": 150
    },
    
    # ===========================================
    # CATEGORY 5: RISK
    # ===========================================
    {
        "category": "Risk",
        "query": "What are the risks of investing in Tesla?",
        "must_contain": ["risk"],
        "must_contain_any": ["volatility", "competition", "valuation", "market", "regulation", "concern"],
        "must_not_contain": ["error"],
        "max_time": 45,
        "min_response_length": 300
    },
    
    # ===========================================
    # CATEGORY 6: EDUCATION
    # Education queries often involve fundamental_analyst for context
    # ===========================================
    {
        "category": "Education",
        "query": "Explain what P/E ratio means",
        "must_contain_any": ["price", "earnings", "ratio", "valuation", "multiple"],
        "must_not_contain": ["error", "cannot explain"],
        "max_time": 45,  # Increased: may call educator + fundamental_analyst
        "min_response_length": 150
    },
    {
        "category": "Education",
        "query": "What is a stop loss order?",
        "must_contain_any": ["stop", "loss", "order", "price", "sell", "protect"],
        "must_not_contain": ["error"],
        "max_time": 45,  # Increased for consistency
        "min_response_length": 100
    },
    
    # ===========================================
    # CATEGORY 7: PORTFOLIO
    # ===========================================
    {
        "category": "Portfolio",
        "query": "How should I diversify my portfolio?",
        "must_contain_any": ["diversif", "allocation", "sector", "asset", "spread", "different"],
        "must_not_contain": ["error"],
        "max_time": 45,
        "min_response_length": 300
    },
    
    # ===========================================
    # CATEGORY 8: MARKET OVERVIEW
    # ===========================================
    {
        "category": "Market",
        "query": "How's the stock market doing today?",
        "must_contain_any": ["S&P", "market", "Dow", "Nasdaq", "up", "down", "flat", "trading"],
        "must_not_contain": ["error"],
        "max_time": 45,  # Multi-agent queries need more time
        "min_response_length": 100
    },
]


async def run_extended_scenarios():
    """Run all extended scenarios with strict validation."""
    from nuble.agents.orchestrator import OrchestratorAgent
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return False
    
    print(f"API Key: ...{api_key[-4:]}")
    
    orchestrator = OrchestratorAgent(api_key=api_key)
    
    results_by_category = {}
    total_passed = 0
    total_failed = 0
    failures = []
    
    print("\n" + "="*70)
    print("EXTENDED REAL USER SCENARIO TESTS")
    print(f"Total scenarios: {len(EXTENDED_SCENARIOS)}")
    print("="*70)
    
    for i, scenario in enumerate(EXTENDED_SCENARIOS, 1):
        category = scenario['category']
        query = scenario['query']
        
        if category not in results_by_category:
            results_by_category[category] = {'passed': 0, 'failed': 0}
        
        print(f"\n[{i}/{len(EXTENDED_SCENARIOS)}] {category}: \"{query[:50]}...\"")
        
        start = time.time()
        
        try:
            result = await asyncio.wait_for(
                orchestrator.process(
                    user_message=query,
                    conversation_id=f"extended_{i}",
                    user_context={}
                ),
                timeout=scenario['max_time'] + 30
            )
            
            elapsed = time.time() - start
            message = result.get('message', '').lower()
            
            # Validation checks
            issues = []
            
            # Check timing
            if elapsed > scenario['max_time']:
                issues.append(f"Too slow: {elapsed:.1f}s > {scenario['max_time']}s")
            
            # Check response length
            if len(message) < scenario.get('min_response_length', 0):
                issues.append(f"Too short: {len(message)} < {scenario['min_response_length']}")
            
            # Check must_contain
            if 'must_contain' in scenario:
                for term in scenario['must_contain']:
                    if term.lower() not in message:
                        issues.append(f"Missing required: '{term}'")
            
            # Check must_contain_any
            if 'must_contain_any' in scenario:
                found_any = any(term.lower() in message for term in scenario['must_contain_any'])
                if not found_any:
                    issues.append(f"Missing one of: {scenario['must_contain_any']}")
            
            # Check must_not_contain
            if 'must_not_contain' in scenario:
                for term in scenario['must_not_contain']:
                    if term.lower() in message:
                        issues.append(f"Contains forbidden: '{term}'")
            
            # Verdict
            if not issues:
                print(f"  ✅ PASSED ({elapsed:.1f}s, {len(message)} chars)")
                total_passed += 1
                results_by_category[category]['passed'] += 1
            else:
                print(f"  ❌ FAILED ({elapsed:.1f}s)")
                for issue in issues:
                    print(f"     • {issue}")
                total_failed += 1
                results_by_category[category]['failed'] += 1
                failures.append({
                    'query': query,
                    'category': category,
                    'issues': issues,
                    'response_preview': message[:200] + '...'
                })
                
        except asyncio.TimeoutError:
            print(f"  ❌ TIMEOUT (>{scenario['max_time']}s)")
            total_failed += 1
            results_by_category[category]['failed'] += 1
            failures.append({
                'query': query,
                'category': category,
                'issues': ['Timeout'],
                'response_preview': 'N/A'
            })
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)[:80]}")
            total_failed += 1
            results_by_category[category]['failed'] += 1
            failures.append({
                'query': query,
                'category': category,
                'issues': [str(e)],
                'response_preview': 'N/A'
            })
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS BY CATEGORY")
    print("="*70)
    
    for category, stats in results_by_category.items():
        total = stats['passed'] + stats['failed']
        pct = stats['passed'] / total * 100 if total > 0 else 0
        status = "✅" if stats['failed'] == 0 else "❌"
        print(f"  {status} {category}: {stats['passed']}/{total} ({pct:.0f}%)")
    
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    total = total_passed + total_failed
    pct = total_passed / total * 100 if total > 0 else 0
    print(f"  Passed: {total_passed}/{total} ({pct:.1f}%)")
    print(f"  Failed: {total_failed}/{total}")
    
    if failures:
        print("\n" + "="*70)
        print("FAILURE DETAILS")
        print("="*70)
        for f in failures:
            print(f"\n  Query: {f['query']}")
            print(f"  Category: {f['category']}")
            print(f"  Issues: {f['issues']}")
            print(f"  Response: {f['response_preview']}")
    
    print("\n" + "="*70)
    if total_failed == 0:
        print("🎉 ALL EXTENDED SCENARIOS PASSED")
        return True
    else:
        print(f"❌ {total_failed} SCENARIOS FAILED - FIX BEFORE PRODUCTION")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_extended_scenarios())
    exit(0 if success else 1)
