"""NUBLE APEX Integration Test — Production Validation"""
import os, time, sys, logging

logging.disable(logging.INFO)
import warnings
warnings.filterwarnings('ignore')

from nuble.manager import Manager

print("=" * 60)
print("NUBLE APEX INTEGRATION TEST")
print("=" * 60)

# 1. Init
print("\n[1] Initializing Manager...")
m = Manager()
print(f"    APEX enabled:  {m.apex_enabled}")
print(f"    DE enabled:    {m.decision_engine_enabled}")
print(f"    Orchestrator:  {m._orchestrator is not None}")
print("    ✅ Init OK")

# =====================================================================
# TEST A: Fast-path query (SmartRouter Tier 0)
# =====================================================================
query_a = "Should I buy AAPL right now?"
print(f"\n[2a] Fast-path query: \"{query_a}\"")

messages_a = [{"role": "user", "content": query_a}]
start = time.time()
result_a = None
try:
    result_a = m.process_prompt(query_a, messages_a)
    elapsed = time.time() - start
    print(f"    Elapsed: {elapsed:.1f}s")
    print(f"    Result length: {len(result_a) if result_a else 0} chars")
    if result_a and len(result_a) > 50:
        print("    ✅ Fast-path response OK")
    else:
        print("    ❌ Fast-path response too short")
except Exception as e:
    elapsed = time.time() - start
    print(f"    ❌ EXCEPTION after {elapsed:.1f}s: {e}")
    import traceback
    traceback.print_exc()

# Check cleanup
data_msgs_a = [msg for msg in messages_a if msg.get("type") in ("data", "apex_data")]
print(f"    Residual data msgs: {len(data_msgs_a)} {'✅' if len(data_msgs_a) == 0 else '⚠️'}")

# =====================================================================
# TEST B: Full APEX dual-brain query (bypass fast path)
# =====================================================================
query_b = "Give me a comprehensive analysis of TSLA including technical outlook, fundamental valuation, macro risks, and a complete trade setup with entry, stop loss, and targets."
print(f"\n[2b] Full APEX query: \"{query_b[:60]}...\"")
print("    Processing... (30-90s expected)")

# Disable fast path to force full APEX pipeline
m.fast_path_enabled = False

messages_b = [{"role": "user", "content": query_b}]
start = time.time()
result_b = None
try:
    result_b = m.process_prompt(query_b, messages_b)
    elapsed = time.time() - start
    print(f"    Elapsed: {elapsed:.1f}s")
    print(f"    Result length: {len(result_b) if result_b else 0} chars")
    if result_b and len(result_b) > 100:
        print("    ✅ Full APEX response OK")
    else:
        print(f"    ❌ Response too short or None")
except Exception as e:
    elapsed = time.time() - start
    print(f"    ❌ EXCEPTION after {elapsed:.1f}s: {e}")
    import traceback
    traceback.print_exc()

# Check cleanup
data_msgs_b = [msg for msg in messages_b if msg.get("type") in ("data", "apex_data")]
print(f"    Residual data msgs: {len(data_msgs_b)} {'✅' if len(data_msgs_b) == 0 else '⚠️'}")

# =====================================================================
# SUMMARY
# =====================================================================
print(f"\n{'=' * 60}")
passed = 0
total = 2

if result_a and len(result_a) > 50:
    passed += 1
    print("✅ Test A (Fast-path): PASSED")
else:
    print("❌ Test A (Fast-path): FAILED")

if result_b and len(result_b) > 100:
    passed += 1
    print("✅ Test B (Full APEX): PASSED")
else:
    print("❌ Test B (Full APEX): FAILED")

print(f"\nResult: {passed}/{total} tests passed")

if result_b:
    print(f"\nFull APEX response (first 800 chars):")
    print("-" * 40)
    print(result_b[:800])

print("=" * 60)
