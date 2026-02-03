#!/usr/bin/env python3
"""Quick import test - minimal."""

import sys
sys.path.insert(0, 'src')

print("Step 1: Testing basic Python...")
import os
print("✅ os imported")

print("Step 2: Testing dataclasses...")
from dataclasses import dataclass
print("✅ dataclasses imported")

print("Step 3: Testing base module...")
from nuble.agents.base import AgentType, SpecializedAgent
print("✅ base imported")

print("Step 4: Testing market_analyst...")
from nuble.agents.market_analyst import MarketAnalystAgent
print("✅ market_analyst imported")

print("Step 5: Creating MarketAnalystAgent...")
agent = MarketAnalystAgent()
print(f"✅ MarketAnalystAgent created: {agent.name}")

print("\n🎉 ALL IMPORTS PASSED!")
