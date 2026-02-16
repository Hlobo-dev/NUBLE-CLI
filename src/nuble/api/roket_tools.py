"""
ROKET Tool Handler — async Python handler for frontend integration (Pattern A).

This module dispatches Claude tool_use calls to the ROKET REST API via httpx.
Use this when your frontend (or a standalone client) handles the Claude ↔ Tools
loop itself and needs to call the ROKET API over HTTP.

Usage:
    from roket_tools import dispatch_tool

    result = await dispatch_tool("roket_predict", {"ticker": "AAPL"})
"""

import httpx
import json
import os
import logging
from typing import Any, Dict

logger = logging.getLogger("roket_tools")

# ── Configuration ───────────────────────────────────────────────────────
ROKET_BASE_URL = os.environ.get("ROKET_BASE_URL", "http://localhost:8000")


async def dispatch_tool(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route a Claude tool_use call to the correct ROKET API endpoint.

    Args:
        tool_name:  The tool name from Claude's tool_use block (e.g. "roket_predict")
        tool_input: The input dict from Claude's tool_use block

    Returns:
        JSON-serializable dict with the tool result

    Raises:
        ValueError: If the tool name is unknown
    """

    # Map tool names to ROKET API endpoints
    # Routes work for both standalone (/api/...) and mounted (/api/roket/...)
    # The ROKET_BASE_URL env var controls which mode:
    #   Standalone: ROKET_BASE_URL=http://localhost:8000  → /api/predict/AAPL
    #   Mounted:    ROKET_BASE_URL=http://localhost:8000/api/roket → /predict/AAPL
    TOOL_ROUTES = {
        # Single-ticker tools (GET with ticker in path)
        "roket_predict":       ("GET",  "/api/predict/{ticker}"),
        "roket_analyze":       ("GET",  "/api/analyze/{ticker}"),
        "roket_fundamentals":  ("GET",  "/api/fundamentals/{ticker}"),
        "roket_earnings":      ("GET",  "/api/earnings/{ticker}"),
        "roket_risk":          ("GET",  "/api/risk/{ticker}"),
        "roket_insider":       ("GET",  "/api/insider/{ticker}"),
        "roket_institutional": ("GET",  "/api/institutional/{ticker}"),
        "roket_news":          ("GET",  "/api/news/{ticker}"),
        "roket_snapshot":      ("GET",  "/api/snapshot/{ticker}"),
        "roket_sec_quality":   ("GET",  "/api/sec-quality/{ticker}"),
        "roket_lambda":        ("GET",  "/api/lambda/{ticker}"),

        # No-input tools
        "roket_regime":        ("GET",  "/api/regime"),
        "roket_macro":         ("GET",  "/api/macro"),

        # Query-param tools
        "roket_screener":      ("POST", "/api/screener"),
        "roket_universe":      ("GET",  "/api/universe"),
        "roket_compare":       ("GET",  "/api/compare"),
        "roket_position_size": ("POST", "/api/position-size"),
    }

    if tool_name not in TOOL_ROUTES:
        return {"error": f"Unknown tool: {tool_name}"}

    method, path_template = TOOL_ROUTES[tool_name]
    ticker = tool_input.get("ticker", "").upper()

    # Substitute {ticker} in path
    path = path_template.replace("{ticker}", ticker) if "{ticker}" in path_template else path_template

    url = f"{ROKET_BASE_URL}{path}"

    # Transform tool input for screener (tool sends tier/signal singular,
    # but the ScreenerRequest expects tiers/signals as lists)
    body = dict(tool_input)
    if tool_name == "roket_screener":
        if "tier" in body and "tiers" not in body:
            body["tiers"] = [body.pop("tier")]
        if "signal" in body and "signals" not in body:
            body["signals"] = [body.pop("signal").upper()]

    # Transform compare tool input (tickers as query param)
    if tool_name == "roket_compare":
        # tickers can be a string or list
        tickers_val = body.get("tickers", "")
        if isinstance(tickers_val, list):
            body["tickers"] = ",".join(tickers_val)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            if method == "GET":
                # For GET with query params (universe, screener)
                params = {k: v for k, v in tool_input.items() if k != "ticker"}
                response = await client.get(url, params=params if params else None)
            else:
                # POST with JSON body
                response = await client.post(url, json=body)

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"ROKET API error for {tool_name}: {e.response.status_code} {e.response.text[:200]}")
            return {"error": f"API error {e.response.status_code}", "detail": e.response.text[:500]}
        except httpx.ConnectError:
            logger.error(f"Cannot connect to ROKET API at {ROKET_BASE_URL}")
            return {"error": f"Cannot connect to ROKET API at {ROKET_BASE_URL}. Is the server running?"}
        except Exception as e:
            logger.error(f"Tool dispatch error for {tool_name}: {e}")
            return {"error": str(e)}


def dispatch_tool_sync(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous wrapper for dispatch_tool.
    Useful for simple scripts that don't use asyncio.
    """
    import asyncio
    return asyncio.run(dispatch_tool(tool_name, tool_input))


# ── Load tool definitions from JSON ─────────────────────────────────────

def load_tool_definitions() -> list:
    """Load the canonical ROKET tool definitions from the JSON file."""
    json_path = os.path.join(os.path.dirname(__file__), "roket_tool_definitions.json")
    with open(json_path) as f:
        return json.load(f)


# ── Quick test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def _test():
        print("Testing ROKET tool dispatch...")
        print(f"Base URL: {ROKET_BASE_URL}")
        print()

        # Test 1: Predict
        print("1. roket_predict(AAPL)")
        result = await dispatch_tool("roket_predict", {"ticker": "AAPL"})
        print(f"   → {json.dumps(result, default=str)[:200]}")
        print()

        # Test 2: Regime
        print("2. roket_regime()")
        result = await dispatch_tool("roket_regime", {})
        print(f"   → {json.dumps(result, default=str)[:200]}")
        print()

        # Test 3: Unknown tool
        print("3. unknown_tool()")
        result = await dispatch_tool("unknown_tool", {})
        print(f"   → {result}")
        print()

        print("Done.")

    asyncio.run(_test())
