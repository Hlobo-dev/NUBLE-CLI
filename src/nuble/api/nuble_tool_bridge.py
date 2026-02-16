"""
NUBLE Tool Bridge — Universal Tool Integration for Any LLM Frontend
=====================================================================
Provides NUBLE's 17 institutional-grade financial tools in BOTH formats:

  1. Claude / Anthropic format    → input_schema (for direct Anthropic API)
  2. Nova Sonic / Bedrock format  → toolSpec with inputSchema.json (for AWS Bedrock)

This module is the SINGLE SOURCE OF TRUTH for tool definitions.
Import this from any frontend (Node.js via REST, Python directly, etc.)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │              Your Frontend (Nova Sonic / Chat UI)            │
    │                                                              │
    │   Voice Input ──► Nova Sonic ──► toolUse event               │
    │   Text Input  ──► Claude API ──► tool_use block              │
    └──────────────────────┬───────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Tool Bridge │ ◄── THIS MODULE
                    │  (dispatch)  │
                    └──────┬──────┘
                           │
    ┌──────────────────────▼───────────────────────────────────────┐
    │                    NUBLE ROKET API                            │
    │  /api/predict  /api/analyze  /api/regime  /api/snapshot ...   │
    │  (FastAPI on port 8000)                                       │
    └──────────────────────────────────────────────────────────────┘

Usage:
    # Python — get tools for Claude (Anthropic format)
    from nuble.api.nuble_tool_bridge import get_anthropic_tools
    tools = get_anthropic_tools()

    # Python — get tools for Nova Sonic / Bedrock (toolSpec format)
    from nuble.api.nuble_tool_bridge import get_bedrock_tools
    tools = get_bedrock_tools()

    # Python — dispatch a tool call
    from nuble.api.nuble_tool_bridge import dispatch_tool
    result = await dispatch_tool("roket_predict", {"ticker": "AAPL"})

    # REST — get tools in either format
    GET /api/tools?format=anthropic     → Anthropic input_schema format
    GET /api/tools?format=bedrock       → Bedrock toolSpec format
    GET /api/tools?format=both          → Both formats side-by-side

    # REST — dispatch a tool call
    POST /api/tools/execute
    { "tool_name": "roket_predict", "tool_input": {"ticker": "AAPL"} }

    # REST — full chat-with-tools loop (server-side)
    POST /api/tools/chat
    { "message": "Should I buy NVDA?", "provider": "anthropic" }
"""

import os
import json
import time
import logging
import copy
from typing import Any, Dict, List, Optional, Literal
from pathlib import Path

logger = logging.getLogger("nuble.tool_bridge")

# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────

ROKET_BASE_URL = os.environ.get("ROKET_BASE_URL", "http://localhost:8000")

# Path to canonical tool definitions
_TOOL_DEFS_PATH = Path(__file__).parent / "roket_tool_definitions.json"


# ─────────────────────────────────────────────────────────────────────────
# 1. CANONICAL TOOL DEFINITIONS (Source of Truth)
# ─────────────────────────────────────────────────────────────────────────

def _load_canonical_tools() -> List[Dict[str, Any]]:
    """Load canonical tool definitions from roket_tool_definitions.json."""
    with open(_TOOL_DEFS_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────
# 2. FORMAT CONVERTERS
# ─────────────────────────────────────────────────────────────────────────

def get_anthropic_tools() -> List[Dict[str, Any]]:
    """
    Return tools in Claude / Anthropic format.

    Anthropic format:
    {
        "name": "roket_predict",
        "description": "...",
        "input_schema": {
            "type": "object",
            "properties": { "ticker": { "type": "string", ... } },
            "required": ["ticker"]
        }
    }
    """
    return _load_canonical_tools()


def _convert_to_bedrock_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single Anthropic-format tool to Bedrock toolSpec format.

    Bedrock toolSpec format (used by Nova Sonic):
    {
        "toolSpec": {
            "name": "roket_predict",
            "description": "...",
            "inputSchema": {
                "json": "{\"type\":\"object\",\"properties\":{...}}"   ← JSON STRING
            }
        }
    }
    """
    schema = tool.get("input_schema", {"type": "object", "properties": {}})

    return {
        "toolSpec": {
            "name": tool["name"],
            "description": tool["description"],
            "inputSchema": {
                "json": json.dumps(schema)
            }
        }
    }


def get_bedrock_tools() -> List[Dict[str, Any]]:
    """
    Return tools in AWS Bedrock / Nova Sonic toolSpec format.

    This is what goes into the promptStart event's toolConfiguration.tools array.

    Usage in Nova Sonic (Node.js local-server.js):
        const tools = await fetch('/api/tools?format=bedrock').then(r => r.json());
        // tools = [ { toolSpec: { name, description, inputSchema: { json: "..." } } }, ... ]
        // Pass directly to enqueuePromptStart() or setupPromptStartEvent()
    """
    canonical = _load_canonical_tools()
    return [_convert_to_bedrock_tool(t) for t in canonical]


def get_bedrock_tool_configuration() -> Dict[str, Any]:
    """
    Return the complete toolConfiguration object for Bedrock's promptStart event.

    This is the exact shape needed for Nova Sonic's promptStart:
    {
        "toolConfiguration": {
            "tools": [ { "toolSpec": { ... } }, ... ]
        },
        "toolUseOutputConfiguration": {
            "mediaType": "application/json"
        }
    }
    """
    return {
        "toolUseOutputConfiguration": {
            "mediaType": "application/json"
        },
        "toolConfiguration": {
            "tools": get_bedrock_tools()
        }
    }


def get_tools(format: Literal["anthropic", "bedrock", "both"] = "anthropic") -> Any:
    """
    Get tools in the requested format.

    Args:
        format: "anthropic" for Claude, "bedrock" for Nova Sonic, "both" for both
    """
    if format == "anthropic":
        return get_anthropic_tools()
    elif format == "bedrock":
        return get_bedrock_tools()
    elif format == "both":
        return {
            "anthropic": get_anthropic_tools(),
            "bedrock": get_bedrock_tools(),
            "bedrock_tool_configuration": get_bedrock_tool_configuration(),
        }
    else:
        raise ValueError(f"Unknown format: {format}. Use 'anthropic', 'bedrock', or 'both'.")


# ─────────────────────────────────────────────────────────────────────────
# 3. SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────

NUBLE_SYSTEM_PROMPT = """You are NUBLE — an elite institutional-grade financial advisor, researcher, and investment analyst. You have access to the most advanced financial intelligence system ever built.

YOUR CAPABILITIES:
- Multi-tier LightGBM ensemble trained on 3.76M observations of 539 academic features (Gu-Kelly-Xiu 2020)
- Real-time market data from Polygon.io (price, volume, 50+ technical indicators, options flow)
- News sentiment analysis via StockNews API (NLP-scored, 7-day rolling) and FinBERT (ProsusAI/finbert)
- SEC EDGAR XBRL filings with 40 fundamental ratios and quality scoring (A through F)
- FRED macroeconomic data (Treasury yields, credit spreads, inflation, employment, Fed Funds)
- Hidden Markov Model regime detection (bull/neutral/crisis from 420 months of macro data)
- LuxAlgo multi-timeframe technical signals from TradingView (Weekly/Daily/4H confluence)
- Lambda Decision Engine aggregating ALL sources into a single institutional-grade verdict
- Modified Kelly Criterion position sizing with ATR-based stop losses
- 20,723-ticker universe coverage across mega/large/mid/small cap tiers

YOUR TOOLS:
You have 17 powerful tools at your disposal. ALWAYS use them to back your analysis with real data:
- roket_predict: ML prediction for any stock (composite score, signal, confidence)
- roket_analyze: Full deep-dive (prediction + fundamentals + earnings + risk + insider + institutional)
- roket_snapshot: LIVE real-time data (price, technicals, options flow, news, regime, LuxAlgo signals)
- roket_lambda: The MOST POWERFUL tool — combines ALL live data sources into one verdict
- roket_news: Real-time news and sentiment analysis
- roket_fundamentals, roket_earnings, roket_risk, roket_insider, roket_institutional: Specific data slices
- roket_regime: Market regime detection (bull/neutral/crisis)
- roket_macro: Macroeconomic environment (yields, credit, inflation, monetary policy)
- roket_sec_quality: Live SEC filing quality score (A-F grade, 40 ratios)
- roket_screener: Screen 20,723 stocks with filters
- roket_universe: Browse top-ranked stocks
- roket_compare: Side-by-side comparison of 2-5 stocks
- roket_position_size: Kelly Criterion position sizing with stop-loss levels

BEHAVIORAL GUIDELINES:
- ALWAYS call tools before giving investment opinions — never guess
- For any stock question, use roket_analyze or roket_lambda for comprehensive data
- For market overview, combine roket_regime + roket_macro
- For "should I buy X?", use roket_lambda (most comprehensive) then roket_position_size
- Present data clearly: use specific numbers, percentages, and scores
- Acknowledge uncertainty and risks — never promise returns
- When comparing stocks, use roket_compare
- For portfolio questions, combine multiple tools
- Be concise but thorough — quality over quantity
- Express confidence levels based on data agreement across sources"""


NUBLE_VOICE_SYSTEM_PROMPT = """You are NUBLE — an elite institutional-grade financial advisor with a natural, conversational speaking style. You have access to the world's most advanced financial intelligence system.

VOICE INTERACTION GUIDELINES:
- Speak naturally and conversationally — you're having a real-time voice conversation
- Keep responses concise (2-4 sentences for simple questions, expand for complex analysis)
- Use clear, jargon-free language unless the user demonstrates expertise
- When presenting numbers, round appropriately for spoken delivery (say "about 15 percent" not "14.73 percent")
- Signal when you're looking up data: "Let me check that for you..." or "Looking at the latest data..."
- If multiple tools are needed, summarize progressively rather than dumping all data at once
- Express appropriate emotion: excitement for strong opportunities, caution for risky situations
- ALWAYS use your tools — never guess about market data, prices, or predictions

YOUR CAPABILITIES (use these tools via function calling):
- roket_predict: ML stock prediction (use for any stock outlook question)
- roket_analyze: Full deep-dive analysis (use for "tell me about X")
- roket_snapshot: Live market data (use for current prices, technicals)
- roket_lambda: Most powerful live decision tool (use for "should I buy X?")
- roket_news: News and sentiment (use for "what's happening with X?")
- roket_regime: Market regime (use for "how's the market?")
- roket_macro: Economic environment (use for macro questions)
- roket_compare: Compare stocks (use for "X vs Y")
- roket_position_size: Position sizing (use for "how much should I buy?")
- Plus: roket_fundamentals, roket_earnings, roket_risk, roket_insider, roket_institutional, roket_sec_quality, roket_screener, roket_universe"""


def get_system_prompt(mode: Literal["text", "voice"] = "text") -> str:
    """Get the appropriate system prompt for the given interaction mode."""
    return NUBLE_VOICE_SYSTEM_PROMPT if mode == "voice" else NUBLE_SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────────────────
# 4. TOOL DISPATCH (Async — calls ROKET API over HTTP)
# ─────────────────────────────────────────────────────────────────────────

# Tool name → (HTTP method, path template)
TOOL_ROUTES = {
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
    "roket_regime":        ("GET",  "/api/regime"),
    "roket_macro":         ("GET",  "/api/macro"),
    "roket_screener":      ("POST", "/api/screener"),
    "roket_universe":      ("GET",  "/api/universe"),
    "roket_compare":       ("GET",  "/api/compare"),
    "roket_position_size": ("POST", "/api/position-size"),
}


async def dispatch_tool(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch a tool call to the ROKET API and return the result.

    Works with both Anthropic tool_use blocks and Bedrock toolUse events.
    The tool_name and tool_input are identical regardless of which LLM called them.

    Args:
        tool_name:  e.g. "roket_predict"
        tool_input: e.g. {"ticker": "AAPL"}

    Returns:
        JSON-serializable dict with the tool result
    """
    import httpx

    if tool_name not in TOOL_ROUTES:
        return {"error": f"Unknown tool: {tool_name}", "available_tools": list(TOOL_ROUTES.keys())}

    method, path_template = TOOL_ROUTES[tool_name]
    ticker = tool_input.get("ticker", "").upper().strip()

    # Substitute {ticker} in path
    path = path_template.replace("{ticker}", ticker) if "{ticker}" in path_template else path_template
    url = f"{ROKET_BASE_URL}{path}"

    # Transform inputs for specific tools
    body = dict(tool_input)
    if tool_name == "roket_screener":
        if "tier" in body and "tiers" not in body:
            body["tiers"] = [body.pop("tier")]
        if "signal" in body and "signals" not in body:
            body["signals"] = [body.pop("signal").upper()]
    if tool_name == "roket_compare":
        tickers_val = body.get("tickers", "")
        if isinstance(tickers_val, list):
            body["tickers"] = ",".join(tickers_val)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            if method == "GET":
                params = {k: v for k, v in tool_input.items() if k != "ticker"}
                response = await client.get(url, params=params if params else None)
            else:
                response = await client.post(url, json=body)

            response.raise_for_status()
            result = response.json()

            # Truncate very large results to prevent token overflow
            result_str = json.dumps(result, default=str)
            if len(result_str) > 50000:
                logger.warning(f"Tool {tool_name} result truncated from {len(result_str)} chars")
                result["_truncated"] = True
                result["_original_size"] = len(result_str)

            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"ROKET API error for {tool_name}: {e.response.status_code}")
            return {"error": f"API error {e.response.status_code}", "detail": e.response.text[:500]}
        except httpx.ConnectError:
            return {"error": f"Cannot connect to ROKET API at {ROKET_BASE_URL}. Is the server running?"}
        except Exception as e:
            logger.error(f"Tool dispatch error for {tool_name}: {e}")
            return {"error": str(e)}


def dispatch_tool_sync(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous wrapper for dispatch_tool."""
    import asyncio
    return asyncio.run(dispatch_tool(tool_name, tool_input))


# ─────────────────────────────────────────────────────────────────────────
# 5. BEDROCK / NOVA SONIC TOOL USE PARSING
# ─────────────────────────────────────────────────────────────────────────

def parse_bedrock_tool_use(tool_use_event: Dict[str, Any]) -> tuple:
    """
    Parse a Bedrock/Nova Sonic toolUse event into (tool_name, tool_input).

    Bedrock sends toolUse events like:
    {
        "toolUseId": "uuid",
        "toolName": "roket_predict",
        "content": "{\"ticker\":\"AAPL\"}"     ← JSON string
    }

    Returns:
        (tool_name: str, tool_input: dict, tool_use_id: str)
    """
    tool_name = tool_use_event.get("toolName", "")
    tool_use_id = tool_use_event.get("toolUseId", "")
    content = tool_use_event.get("content", "{}")

    # Content can be a string (JSON) or already a dict
    if isinstance(content, str):
        try:
            tool_input = json.loads(content)
        except json.JSONDecodeError:
            tool_input = {}
    else:
        tool_input = content if isinstance(content, dict) else {}

    return tool_name, tool_input, tool_use_id


def parse_anthropic_tool_use(tool_use_block: Dict[str, Any]) -> tuple:
    """
    Parse a Claude/Anthropic tool_use content block into (tool_name, tool_input).

    Anthropic sends tool_use blocks like:
    {
        "type": "tool_use",
        "id": "toolu_...",
        "name": "roket_predict",
        "input": {"ticker": "AAPL"}        ← already a dict
    }

    Returns:
        (tool_name: str, tool_input: dict, tool_use_id: str)
    """
    tool_name = tool_use_block.get("name", "")
    tool_use_id = tool_use_block.get("id", "")
    tool_input = tool_use_block.get("input", {})

    return tool_name, tool_input, tool_use_id


def format_bedrock_tool_result(tool_use_id: str, result: Any) -> Dict[str, Any]:
    """
    Format a tool result for sending back to Bedrock/Nova Sonic.

    This creates the contentStart + toolResult + contentEnd event sequence
    that Nova Sonic expects.

    Returns a dict with the events to enqueue.
    """
    result_str = json.dumps(result, default=str) if not isinstance(result, str) else result

    return {
        "tool_use_id": tool_use_id,
        "content": result_str,
        "events": {
            "contentStart": {
                "type": "TOOL",
                "role": "TOOL",
                "toolResultInputConfiguration": {
                    "toolUseId": tool_use_id,
                    "type": "TEXT",
                    "textInputConfiguration": {
                        "mediaType": "text/plain"
                    }
                }
            },
            "toolResult": result_str,
            "contentEnd": {}
        }
    }


def format_anthropic_tool_result(tool_use_id: str, result: Any) -> Dict[str, Any]:
    """
    Format a tool result for sending back to Claude/Anthropic.

    Returns the tool_result content block for the messages API.
    """
    result_str = json.dumps(result, default=str) if not isinstance(result, str) else result

    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": result_str,
    }


# ─────────────────────────────────────────────────────────────────────────
# 6. FASTAPI ROUTER (REST endpoints for frontend consumption)
# ─────────────────────────────────────────────────────────────────────────

try:
    from fastapi import APIRouter, Query as FastAPIQuery
    from pydantic import BaseModel, Field

    router = APIRouter(prefix="/api/tools", tags=["NUBLE Tool Bridge"])

    class ToolExecuteRequest(BaseModel):
        """Execute a single tool call."""
        tool_name: str = Field(..., description="Tool name (e.g. roket_predict)")
        tool_input: Dict[str, Any] = Field(default_factory=dict, description="Tool input parameters")

    class ToolExecuteResponse(BaseModel):
        """Tool execution result."""
        tool_name: str
        result: Any
        execution_time_ms: float

    @router.get("/")
    async def get_tool_definitions(
        format: str = FastAPIQuery("anthropic", description="Format: 'anthropic', 'bedrock', or 'both'")
    ):
        """
        Get NUBLE tool definitions in the requested format.

        - **anthropic**: Claude/Anthropic input_schema format
        - **bedrock**: AWS Bedrock/Nova Sonic toolSpec format
        - **both**: Both formats plus the complete toolConfiguration object
        """
        return get_tools(format)

    @router.get("/system-prompt")
    async def get_system_prompt_endpoint(
        mode: str = FastAPIQuery("text", description="Mode: 'text' or 'voice'")
    ):
        """Get the NUBLE system prompt for the specified interaction mode."""
        return {"prompt": get_system_prompt(mode), "mode": mode}

    @router.post("/execute", response_model=ToolExecuteResponse)
    async def execute_tool(request: ToolExecuteRequest):
        """
        Execute a single NUBLE tool call.

        This is the universal tool dispatch endpoint. Send the tool name and input,
        get the result. Works for both Anthropic and Bedrock tool call formats.
        """
        start = time.time()
        result = await dispatch_tool(request.tool_name, request.tool_input)
        elapsed = (time.time() - start) * 1000
        return ToolExecuteResponse(
            tool_name=request.tool_name,
            result=result,
            execution_time_ms=round(elapsed, 1)
        )

    @router.get("/health")
    async def tool_bridge_health():
        """Check NUBLE Tool Bridge health and ROKET API connectivity."""
        import httpx
        roket_status = "unknown"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{ROKET_BASE_URL}/api/health")
                if resp.status_code == 200:
                    roket_status = "connected"
                else:
                    roket_status = f"error ({resp.status_code})"
        except Exception as e:
            roket_status = f"disconnected ({e})"

        return {
            "status": "ok",
            "roket_api_url": ROKET_BASE_URL,
            "roket_api_status": roket_status,
            "tools_count": len(TOOL_ROUTES),
            "tools_available": list(TOOL_ROUTES.keys()),
            "formats_supported": ["anthropic", "bedrock"],
        }

except ImportError:
    router = None
    logger.debug("FastAPI not available — REST endpoints disabled")


# ─────────────────────────────────────────────────────────────────────────
# 7. NODE.JS / JAVASCRIPT HELPERS (code generators for frontend)
# ─────────────────────────────────────────────────────────────────────────

def generate_node_tool_handler() -> str:
    """
    Generate a complete Node.js tool handler module.

    This produces a .js file that your Nova Sonic local-server.js can import
    to handle tool calls against the NUBLE ROKET API.
    """
    tools = _load_canonical_tools()
    tool_routes = json.dumps(TOOL_ROUTES, indent=2)

    return f'''/**
 * NUBLE Tool Handler — Auto-generated for Nova Sonic / Bedrock integration
 * Generated from: nuble_tool_bridge.py
 *
 * Usage in local-server.js:
 *   const {{ processNubleTool, getNubleTools, NUBLE_SYSTEM_PROMPT }} = require('./nuble-tools');
 *
 *   // Get tools for promptStart
 *   const tools = getNubleTools();
 *
 *   // Process a tool call from Bedrock
 *   const result = await processNubleTool(toolName, toolContent);
 */

const ROKET_BASE_URL = process.env.ROKET_BASE_URL || 'http://localhost:8000';

// Tool name → [method, pathTemplate]
const TOOL_ROUTES = {tool_routes};

/**
 * Execute a NUBLE tool call against the ROKET API.
 * @param {{string}} toolName - e.g. "roket_predict"
 * @param {{object|string}} toolContent - The tool input (parsed or JSON string)
 * @returns {{Promise<object>}} Tool result
 */
async function processNubleTool(toolName, toolContent) {{
  // Parse content if it's a JSON string
  let input;
  if (typeof toolContent === 'string') {{
    try {{ input = JSON.parse(toolContent); }} catch {{ input = {{}}; }}
  }} else if (toolContent && typeof toolContent.content === 'string') {{
    try {{ input = JSON.parse(toolContent.content); }} catch {{ input = {{}}; }}
  }} else {{
    input = toolContent || {{}};
  }}

  const route = TOOL_ROUTES[toolName];
  if (!route) {{
    return {{ error: `Unknown tool: ${{toolName}}`, available: Object.keys(TOOL_ROUTES) }};
  }}

  const [method, pathTemplate] = route;
  const ticker = (input.ticker || '').toUpperCase().trim();
  const path = pathTemplate.replace('{{ticker}}', ticker);
  const url = `${{ROKET_BASE_URL}}${{path}}`;

  try {{
    const options = {{ method, headers: {{ 'Content-Type': 'application/json' }} }};

    if (method === 'GET') {{
      // Build query params for non-ticker fields
      const params = new URLSearchParams();
      for (const [k, v] of Object.entries(input)) {{
        if (k !== 'ticker' && v !== undefined && v !== null) {{
          params.set(k, String(v));
        }}
      }}
      const queryStr = params.toString();
      const fullUrl = queryStr ? `${{url}}?${{queryStr}}` : url;
      const resp = await fetch(fullUrl, options);
      if (!resp.ok) throw new Error(`HTTP ${{resp.status}}: ${{await resp.text()}}`);
      return await resp.json();
    }} else {{
      // POST with JSON body
      const body = {{ ...input }};
      // Transform screener inputs
      if (toolName === 'roket_screener') {{
        if (body.tier && !body.tiers) {{ body.tiers = [body.tier]; delete body.tier; }}
        if (body.signal && !body.signals) {{ body.signals = [body.signal.toUpperCase()]; delete body.signal; }}
      }}
      options.body = JSON.stringify(body);
      const resp = await fetch(url, options);
      if (!resp.ok) throw new Error(`HTTP ${{resp.status}}: ${{await resp.text()}}`);
      return await resp.json();
    }}
  }} catch (err) {{
    console.error(`[NUBLE] Tool ${{toolName}} error:`, err.message);
    return {{ error: err.message, tool: toolName }};
  }}
}}

/**
 * Get NUBLE tools in Bedrock toolSpec format for Nova Sonic promptStart.
 * @returns {{Array}} Array of {{ toolSpec: {{ name, description, inputSchema: {{ json }} }} }}
 */
function getNubleTools() {{
  return {json.dumps(get_bedrock_tools(), indent=2)};
}}

/**
 * Get the NUBLE system prompt.
 * @param {{"text"|"voice"}} mode - Interaction mode
 * @returns {{string}} System prompt
 */
function getNubleSystemPrompt(mode = 'voice') {{
  if (mode === 'voice') {{
    return {json.dumps(NUBLE_VOICE_SYSTEM_PROMPT)};
  }}
  return {json.dumps(NUBLE_SYSTEM_PROMPT)};
}}

module.exports = {{
  processNubleTool,
  getNubleTools,
  getNubleSystemPrompt,
  ROKET_BASE_URL,
  TOOL_ROUTES,
}};
'''


# ─────────────────────────────────────────────────────────────────────────
# 8. CLI / MAIN — Generate integration files
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NUBLE Tool Bridge — Generate integration files")
    parser.add_argument("--format", choices=["anthropic", "bedrock", "both", "node", "all"],
                        default="both", help="Output format")
    parser.add_argument("--output", "-o", help="Output file path")
    args = parser.parse_args()

    if args.format == "node":
        code = generate_node_tool_handler()
        if args.output:
            with open(args.output, "w") as f:
                f.write(code)
            print(f"✅ Node.js tool handler written to {args.output}")
        else:
            print(code)

    elif args.format == "all":
        # Print everything
        print("=" * 80)
        print("ANTHROPIC FORMAT (Claude)")
        print("=" * 80)
        print(json.dumps(get_anthropic_tools(), indent=2))
        print()
        print("=" * 80)
        print("BEDROCK FORMAT (Nova Sonic)")
        print("=" * 80)
        print(json.dumps(get_bedrock_tools(), indent=2))
        print()
        print("=" * 80)
        print("BEDROCK TOOL CONFIGURATION (complete promptStart fragment)")
        print("=" * 80)
        print(json.dumps(get_bedrock_tool_configuration(), indent=2))
        print()
        print(f"Total tools: {len(TOOL_ROUTES)}")
        print(f"Tool names: {', '.join(TOOL_ROUTES.keys())}")

    else:
        result = get_tools(args.format)
        output = json.dumps(result, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"✅ Written to {args.output}")
        else:
            print(output)
