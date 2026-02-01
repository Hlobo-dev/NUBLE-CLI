#!/usr/bin/env python3
"""
Debug the complex query failure.
Find exactly why quality indicators are missing.
"""

import asyncio
import os
import sys
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


async def debug_complex_query():
    from kyperian.agents.orchestrator import OrchestratorAgent
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return None
        
    print(f"API Key: ...{api_key[-4:]}")
    
    orchestrator = OrchestratorAgent(api_key=api_key)
    
    query = "Should I buy NVIDIA? Consider the technicals, news sentiment, and risks."
    
    print("="*70)
    print("DEBUGGING COMPLEX QUERY")
    print("="*70)
    print(f"\nQuery: {query}\n")
    
    # Run with detailed logging
    result = await orchestrator.process(
        user_message=query,
        conversation_id="debug_complex",
        user_context={'risk_tolerance': 'moderate'}
    )
    
    print("\n" + "="*70)
    print("RESULT ANALYSIS")
    print("="*70)
    
    # 1. Check which agents were used
    agents_used = result.get('agents_used', [])
    print(f"\n1. AGENTS USED: {agents_used}")
    print(f"   Count: {len(agents_used)}")
    
    if len(agents_used) < 3:
        print("   ⚠️ ISSUE: Should use at least 3 agents for complex query")
        print("   FIX: Check orchestrator planning logic")
    else:
        print("   ✅ Good agent count")
    
    # 2. Check the data returned
    data = result.get('data', {})
    agent_outputs = data.get('agent_outputs', {})
    
    print(f"\n2. AGENT OUTPUTS:")
    for agent, output in agent_outputs.items():
        if isinstance(output, dict):
            print(f"   {agent}: {list(output.keys())}")
        else:
            print(f"   {agent}: {type(output).__name__}")
    
    # 3. Check the message content
    message = result.get('message', '')
    print(f"\n3. MESSAGE LENGTH: {len(message)} chars")
    print(f"   First 500 chars:\n   {message[:500]}...")
    
    # 4. Check for expected quality indicators
    quality_indicators = [
        'bullish', 'bearish', 'neutral', 'buy', 'sell', 'hold',
        'risk', 'price', 'target', 'stop', 'technical', 'sentiment',
        'rsi', 'macd', 'support', 'resistance', 'valuation', 'recommend'
    ]
    
    message_lower = message.lower()
    found = [ind for ind in quality_indicators if ind in message_lower]
    missing = [ind for ind in quality_indicators if ind not in message_lower]
    
    print(f"\n4. QUALITY INDICATORS:")
    print(f"   Found ({len(found)}): {found}")
    print(f"   Missing ({len(missing)}): {missing}")
    
    # 5. Verdict
    print(f"\n" + "="*70)
    if len(found) >= 5:
        print("✅ QUALITY CHECK PASSED")
    else:
        print("❌ QUALITY CHECK FAILED")
        print("\nPOTENTIAL FIXES:")
        
        if len(agents_used) < 3:
            print("  1. Orchestrator planning needs to request more agents")
        
        if 'market_analyst' not in agents_used:
            print("  2. Market analyst should be included for technical analysis")
        
        if 'news_analyst' not in agents_used:
            print("  3. News analyst should be included for sentiment")
        
        if 'risk_manager' not in agents_used:
            print("  4. Risk manager should be included for risk assessment")
        
        if len(message) < 500:
            print("  5. Response too short - synthesis prompt may need improvement")
    
    print("="*70)
    
    return result


if __name__ == "__main__":
    asyncio.run(debug_complex_query())
