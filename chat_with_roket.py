#!/usr/bin/env python3
"""
ROKET Chat Client — standalone Claude ↔ ROKET tool-use loop.

This is a self-contained terminal chat client that:
1. Sends user messages to Claude with ROKET tool definitions
2. When Claude wants to use tools, dispatches them to the ROKET API
3. Feeds tool results back to Claude for synthesis
4. Prints Claude's final answer

Usage:
    # Make sure ROKET API is running on :8000
    # Make sure ANTHROPIC_API_KEY is set
    python chat_with_roket.py

    # Or specify a different API base
    ROKET_BASE_URL=http://localhost:9000 python chat_with_roket.py
"""

import os
import sys
import json
import asyncio
import logging

# Add project root to path so imports work
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, _root)

from nuble.api.roket_tools import dispatch_tool, load_tool_definitions

logger = logging.getLogger("chat_with_roket")

# ── Configuration ───────────────────────────────────────────────────────

MODEL = "claude-sonnet-4-20250514"
MAX_TOOL_ROUNDS = 5
MAX_TOKENS = 4096

SYSTEM_PROMPT = """You are ROKET — the Robust Quantitative Knowledge Engine for Trading.

You are the conversational interface to NUBLE's institutional-grade ML system.
You have 17 tools at your disposal that give you access to:

SYSTEM A — Historical ML Intelligence (WRDS/GKX):
• A multi-tier LightGBM ensemble trained on 3.76M observations (539 GKX features)
• 20,723-ticker universe with per-tier models (mega/large/mid/small)
• HMM-based market regime detection (bull/neutral/crisis, 420 months macro)
• Deep fundamental, earnings, risk, insider, and institutional metrics

SYSTEM B — Real-Time Live Intelligence:
• StockNews API: Live news articles, sentiment scores (-1.5 to +1.5), trending headlines
• Polygon.io: Real-time quotes, technicals (SMA, RSI, MACD, Bollinger, ATR), options flow
• SEC EDGAR XBRL: Live fundamental ratios, quality scores (A-F) from actual filings
• FRED: Macro data — yield curve, credit spreads, monetary policy stance
• Lambda Decision Engine: Production composite decisions aggregating ALL live sources
• Kelly Criterion: Position sizing with stop-loss and take-profit levels

TOOL SELECTION RULES:
1. ALWAYS use tools to back up your analysis. Never guess when you can look it up.
2. For single stock questions → call roket_predict or roket_analyze.
3. For market conditions → call roket_regime and/or roket_macro.
4. For 'best stocks' / 'top picks' → call roket_universe or roket_screener.
5. For deep dives → call roket_analyze (WRDS data) AND roket_lambda (live data).
6. For news/sentiment/catalysts → call roket_news.
7. For real-time price/technicals/options → call roket_snapshot.
8. For fundamental quality from SEC filings → call roket_sec_quality.
9. For comparing stocks → call roket_compare.
10. For position sizing / stop loss / risk management → call roket_position_size.
11. For the most authoritative LIVE trading decision → call roket_lambda.
12. COMBINE System A + System B tools for the most complete analysis.

RESPONSE FORMAT:
• Lead with the ML signal and key metrics
• Combine historical (WRDS) and live (Lambda/news/technicals) perspectives
• Place analysis in regime and macro context
• Give clear, actionable recommendation with caveats
• Include position sizing guidance when appropriate
• Be confident but honest about limitations (feature coverage, data recency)

You speak with institutional authority. Your data comes from WRDS academic-grade sources, SEC EDGAR filings, and real-time market data feeds."""


# ── Chat Loop ───────────────────────────────────────────────────────────

async def chat_with_roket(user_message: str, conversation_history: list = None) -> str:
    """
    Send a message through the full Claude ↔ ROKET tool loop.

    Args:
        user_message: The user's natural language query
        conversation_history: Optional list of prior messages for multi-turn

    Returns:
        Claude's final synthesized text response
    """
    try:
        import anthropic
    except ImportError:
        return "Error: anthropic package not installed. Run: pip install anthropic"

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY environment variable not set"

    client = anthropic.Anthropic(api_key=api_key)
    tools = load_tool_definitions()

    # Build messages
    messages = conversation_history or []
    messages.append({"role": "user", "content": user_message})

    tools_used = []

    for round_num in range(MAX_TOOL_ROUNDS + 1):
        # Call Claude
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            # Extract tool calls
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            # Add assistant response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Dispatch each tool call
            tool_results = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block.name
                tool_input = tool_block.input
                tools_used.append(tool_name)

                print(f"  🔧 {tool_name}({json.dumps(tool_input)[:80]})", flush=True)

                # Dispatch to ROKET API
                result = await dispatch_tool(tool_name, tool_input)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": json.dumps(result, default=str),
                })

            messages.append({"role": "user", "content": tool_results})

        else:
            # Claude returned text — we're done
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            return final_text

    return "Analysis incomplete — max tool rounds reached."


# ── Interactive REPL ────────────────────────────────────────────────────

async def interactive_loop():
    """Run an interactive chat session with ROKET."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        🚀 ROKET — Quantitative Knowledge Engine        ║")
    print("║   Robust Quantitative Knowledge Engine for Trading      ║")
    print("║                                                        ║")
    print("║   Powered by NUBLE ML • 20,723 tickers • 539 features  ║")
    print("║   Type 'quit' to exit  •  'clear' to reset history     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 🚀")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! 🚀")
            break

        if user_input.lower() == "clear":
            conversation_history = []
            print("  (conversation cleared)\n")
            continue

        print()  # spacing before tool calls

        try:
            response = await chat_with_roket(user_input, conversation_history)
            print(f"\nROKET: {response}\n")

            # Maintain conversation history
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})

        except Exception as e:
            print(f"\n  ❌ Error: {e}\n")


# ── One-shot mode ───────────────────────────────────────────────────────

async def one_shot(query: str):
    """Run a single query and print the result."""
    print(f"\n📊 Query: {query}\n")
    response = await chat_with_roket(query)
    print(f"\n{response}\n")


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # One-shot mode: python chat_with_roket.py "analyze AAPL"
        query = " ".join(sys.argv[1:])
        asyncio.run(one_shot(query))
    else:
        # Interactive mode
        asyncio.run(interactive_loop())
