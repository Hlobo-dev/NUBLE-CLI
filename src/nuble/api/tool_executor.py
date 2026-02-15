#!/usr/bin/env python3
"""
NUBLE Tool Executor — Server-side tool dispatch for Claude function calling
============================================================================
This module receives tool_use requests from Claude and dispatches them to
the correct Intelligence API endpoint. Run this as middleware between your
frontend Claude instance and the NUBLE backend.

Two integration patterns supported:

Pattern A — Direct (recommended):
    Your frontend calls Claude with tools → Claude returns tool_use →
    your frontend calls the matching /api/intel/* endpoint → feeds result
    back to Claude → Claude generates final answer.

Pattern B — Server-side dispatch (this module):
    Your frontend sends the raw user message → this endpoint handles the
    full Claude ↔ tools loop internally → returns the final answer.

Usage (Pattern B):
    POST /api/intel/chat-with-tools
    {
        "message": "Should I buy NVDA?",
        "conversation_id": "optional"
    }

    This will:
    1. Send message to Claude Opus with tools schema
    2. If Claude calls a tool, execute it against the intelligence API
    3. Feed tool result back to Claude
    4. Return Claude's final synthesized answer
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/intel", tags=["Intelligence Chat"])


class ChatWithToolsRequest(BaseModel):
    """Request for tool-augmented chat."""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = None
    max_tool_rounds: int = Field(3, description="Max tool call rounds before forcing a response")


class ChatWithToolsResponse(BaseModel):
    """Response from tool-augmented chat."""
    message: str
    tools_used: List[str] = []
    tool_results: List[Dict[str, Any]] = []
    execution_time_seconds: float


# ── Tool dispatch map ───────────────────────────────────────────────────

def _dispatch_tool(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool call locally (no HTTP, direct function call).
    This is faster than routing through HTTP when running server-side.
    """
    try:
        if tool_name == "get_stock_prediction":
            from ..ml.live_predictor import get_live_predictor
            lp = get_live_predictor()
            return lp.predict(tool_input['ticker'].upper())

        elif tool_name == "get_batch_predictions":
            from ..ml.live_predictor import get_live_predictor
            lp = get_live_predictor()
            return {'predictions': lp.predict_batch(tool_input['tickers'])}

        elif tool_name == "get_market_regime":
            from ..ml.hmm_regime import get_regime_detector
            det = get_regime_detector()
            regime = det.detect_regime()
            regime['vix_exposure'] = det.get_vix_exposure()
            return regime

        elif tool_name == "get_top_picks":
            from ..ml.live_predictor import get_live_predictor
            lp = get_live_predictor()
            n = tool_input.get('n', 10)
            tier = tool_input.get('tier')
            picks = lp.get_live_top_picks(n=n, tier=tier)
            return {'picks': picks, 'count': len(picks)}

        elif tool_name == "analyze_portfolio":
            from ..ml.live_predictor import get_live_predictor
            from ..ml.hmm_regime import get_regime_detector
            lp = get_live_predictor()
            det = get_regime_detector()
            holdings = tool_input['holdings']
            results = []
            for ticker in holdings:
                try:
                    pred = lp.predict(ticker.upper())
                    pred['weight'] = holdings[ticker]
                    results.append(pred)
                except Exception as e:
                    results.append({'ticker': ticker, 'error': str(e)})
            regime = det.detect_regime()
            return {'holdings': results, 'regime': regime}

        elif tool_name == "get_tier_info":
            from ..ml.wrds_predictor import get_wrds_predictor, TIER_CONFIG
            wrds = get_wrds_predictor()
            wrds._ensure_loaded()
            ticker = tool_input['ticker'].upper()
            # Use predict to get full tier info
            pred = wrds.predict(ticker)
            tier = pred.get('tier', 'small')
            tc = TIER_CONFIG.get(tier, {})
            return {
                'ticker': ticker, 'tier': tier,
                'label': tc.get('label', ''), 'ic': tc.get('ic', 0),
                'weight': tc.get('weight', 0), 'strategy': tc.get('strategy', ''),
                'market_cap_millions': pred.get('market_cap_millions', 0),
            }

        elif tool_name == "get_system_status":
            return {'status': 'operational', 'message': 'Use GET /api/intel/system-status for full details'}

        else:
            return {'error': f'Unknown tool: {tool_name}'}

    except Exception as e:
        logger.error(f"Tool dispatch failed for {tool_name}: {e}", exc_info=True)
        return {'error': str(e)}


# ── Tool definitions (same as intelligence.py/tools-schema) ─────────────

def _get_tools_for_claude() -> List[Dict[str, Any]]:
    """Get tools in Anthropic API format."""
    return [
        {
            "name": "get_stock_prediction",
            "description": (
                "Get an institutional-grade ML prediction for a stock. Returns composite "
                "score (70% fundamental + 30% timing), signal, confidence, tier, top drivers. "
                "The model is trained on 3.76M observations of 539 academic features (WRDS/GKX). "
                "Use for any question about a specific stock."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker (e.g. AAPL)"}
                },
                "required": ["ticker"]
            },
        },
        {
            "name": "get_batch_predictions",
            "description": (
                "Get ML predictions for multiple stocks at once. Max 50 tickers."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ticker symbols"
                    }
                },
                "required": ["tickers"]
            },
        },
        {
            "name": "get_market_regime",
            "description": (
                "Detect current market regime (bull/neutral/crisis) using a Hidden Markov Model "
                "trained on 420 months of macro data. Returns state, probabilities, VIX exposure factor. "
                "Use for market condition questions."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "get_top_picks",
            "description": (
                "Get top N stocks ranked by ML composite score. Can filter by tier "
                "(mega/large/mid/small). Use when user asks for recommendations."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Number of picks (default 10)"},
                    "tier": {"type": "string", "enum": ["mega", "large", "mid", "small"]}
                },
            },
        },
        {
            "name": "analyze_portfolio",
            "description": (
                "Analyze a portfolio against the ML models and regime detector. "
                "Input: ticker → weight mapping."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "holdings": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "Ticker → weight"
                    }
                },
                "required": ["holdings"]
            },
        },
        {
            "name": "get_tier_info",
            "description": "Get tier classification for a stock (mega/large/mid/small).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["ticker"]
            },
        },
    ]


# ── Server-side Claude ↔ Tools loop ─────────────────────────────────────

@router.post("/chat-with-tools", response_model=ChatWithToolsResponse)
async def chat_with_tools(request: ChatWithToolsRequest):
    """
    Full Claude ↔ Tools loop server-side.

    1. Sends user message to Claude Opus with tool definitions
    2. If Claude returns tool_use, executes the tool and feeds result back
    3. Repeats until Claude returns a text response (max N rounds)
    4. Returns the final synthesized answer

    This is Pattern B — your frontend just sends the message and gets back
    a rich, data-backed answer. No tool handling needed in the frontend.
    """
    t0 = time.time()

    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not set")

    try:
        import anthropic
    except ImportError:
        raise HTTPException(status_code=503, detail="anthropic package not installed")

    client = anthropic.Anthropic(api_key=api_key)
    tools = _get_tools_for_claude()
    tools_used = []
    tool_results_log = []

    system_prompt = (
        "You are NUBLE, an institutional-grade AI financial advisor powered by "
        "quantitative intelligence. You have access to:\n"
        "• Multi-tier LightGBM ensemble trained on 3.76M observations (539 academic features)\n"
        "• HMM-based market regime detection (bull/neutral/crisis)\n"
        "• Live feature computation from Polygon market data\n"
        "• 20,723-ticker universe with per-tier models (mega/large/mid/small)\n\n"
        "ALWAYS use your tools to back up your analysis with data. Never guess when "
        "you can look it up. When analyzing a stock, ALWAYS call get_stock_prediction. "
        "When discussing market conditions, ALWAYS call get_market_regime.\n\n"
        "Present your analysis professionally with clear structure:\n"
        "1. Lead with the ML signal and key metrics\n"
        "2. Explain what's driving the signal (top feature drivers)\n"
        "3. Put it in regime context (bull/neutral/crisis)\n"
        "4. Give a clear, actionable recommendation with caveats\n\n"
        "Be confident but honest about limitations (feature coverage, data staleness)."
    )

    messages = [{"role": "user", "content": request.message}]

    for round_num in range(request.max_tool_rounds + 1):
        response = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            # Process all tool calls in this response
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            # Add assistant response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool and build results
            tool_result_content = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block.name
                tool_input = tool_block.input
                tools_used.append(tool_name)

                logger.info(f"Tool call [{round_num}]: {tool_name}({json.dumps(tool_input)[:100]})")

                result = _dispatch_tool(tool_name, tool_input)
                tool_results_log.append({
                    'tool': tool_name,
                    'input': tool_input,
                    'result_preview': str(result)[:500],
                })

                tool_result_content.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": json.dumps(result, default=str),
                })

            messages.append({"role": "user", "content": tool_result_content})

        else:
            # Claude returned a text response — we're done
            final_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    final_text += block.text

            return ChatWithToolsResponse(
                message=final_text,
                tools_used=tools_used,
                tool_results=tool_results_log,
                execution_time_seconds=round(time.time() - t0, 2),
            )

    # Max rounds reached — return whatever we have
    return ChatWithToolsResponse(
        message="Analysis incomplete — max tool rounds reached. Please try a more specific question.",
        tools_used=tools_used,
        tool_results=tool_results_log,
        execution_time_seconds=round(time.time() - t0, 2),
    )
