"""Patch server.py on disk — add learning resolver startup + API endpoints."""
import os

server_path = os.path.join(os.path.dirname(__file__), "src", "nuble", "api", "server.py")

with open(server_path, "r") as f:
    content = f.read()

print(f"Before: {content.count(chr(10))} lines")

# ── PATCH 1: Wire resolver into startup ──────────────────────────────────
old_startup = (
    '@app.on_event("startup")\n'
    'async def startup():\n'
    '    logger.info("NUBLE API starting up...")\n'
    '    loop = asyncio.get_event_loop()\n'
    '    await loop.run_in_executor(None, _mgr._ensure_init)\n'
    '    logger.info("NUBLE API ready")'
)

new_startup = (
    '@app.on_event("startup")\n'
    'async def startup():\n'
    '    logger.info("NUBLE API starting up...")\n'
    '    loop = asyncio.get_event_loop()\n'
    '    await loop.run_in_executor(None, _mgr._ensure_init)\n'
    '\n'
    '    # ── Start Learning Resolver (background prediction resolution) ───────\n'
    '    try:\n'
    '        from ..learning.resolver import PredictionResolver\n'
    '        _resolver = PredictionResolver()\n'
    '        await _resolver.start()\n'
    '        logger.info("Learning resolver started — resolving predictions hourly")\n'
    '    except Exception as exc:\n'
    '        logger.warning("Learning resolver unavailable: %s", exc)\n'
    '\n'
    '    logger.info("NUBLE API ready")'
)

if old_startup in content:
    content = content.replace(old_startup, new_startup)
    print("PATCH 1: Startup event patched")
else:
    print("PATCH 1: Startup block NOT found (may already be patched)")

# ── PATCH 2: Add learning endpoints before conversation management ───────
learning_endpoints = (
    '\n'
    '# ── Learning System ─────────────────────────────────────────────────────\n'
    '\n'
    '@app.get("/api/learning/stats")\n'
    'async def learning_stats():\n'
    '    """Get learning system statistics: accuracy, predictions, current weights."""\n'
    '    try:\n'
    '        from ..learning.learning_hub import LearningHub\n'
    '        hub = LearningHub()\n'
    '        return {\n'
    '            "status": "active",\n'
    '            "accuracy": hub.get_accuracy_report(),\n'
    '            "predictions": hub.get_prediction_stats(),\n'
    '            "current_weights": hub.get_weights(),\n'
    '        }\n'
    '    except Exception as e:\n'
    '        return {"status": "unavailable", "error": str(e)}\n'
    '\n'
    '\n'
    '@app.get("/api/learning/predictions")\n'
    'async def learning_predictions():\n'
    '    """Get all raw predictions (for debugging/analysis)."""\n'
    '    try:\n'
    '        from ..learning.learning_hub import LearningHub\n'
    '        hub = LearningHub()\n'
    '        unresolved = hub.get_unresolved()\n'
    '        return {\n'
    '            "total": len(hub._raw_predictions),\n'
    '            "unresolved_count": len(unresolved),\n'
    '            "recent_predictions": list(hub._raw_predictions.values())[-20:],\n'
    '        }\n'
    '    except Exception as e:\n'
    '        return {"status": "unavailable", "error": str(e)}\n'
    '\n'
    '\n'
)

conversation_marker = "# ── Conversation management"

if "/api/learning/stats" not in content:
    if conversation_marker in content:
        content = content.replace(conversation_marker, learning_endpoints + conversation_marker)
        print("PATCH 2: Learning endpoints added")
    else:
        print("PATCH 2: Conversation management marker NOT found")
else:
    print("PATCH 2: Learning endpoints already present")

with open(server_path, "w") as f:
    f.write(content)

print(f"After: {content.count(chr(10))} lines")
print("Done.")
