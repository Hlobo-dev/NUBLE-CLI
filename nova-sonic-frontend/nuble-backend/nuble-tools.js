/**
 * NUBLE Tool Handler — Nova Sonic / Bedrock Integration
 * ======================================================
 * Connects AWS Bedrock Nova Sonic to the NUBLE ROKET financial intelligence API.
 * 
 * This gives Nova Sonic access to 17 institutional-grade financial tools:
 *   - ML predictions (LightGBM ensemble, 3.76M observations, 539 GKX features)
 *   - Real-time market data (Polygon.io: prices, technicals, options flow)
 *   - News sentiment (StockNews API + FinBERT NLP)
 *   - SEC EDGAR fundamentals (XBRL filings, 40 ratios, quality grading)
 *   - Macro environment (FRED: yields, credit, inflation, employment)
 *   - HMM regime detection (bull/neutral/crisis)
 *   - LuxAlgo multi-timeframe signals (TradingView)
 *   - Lambda Decision Engine (aggregates ALL sources)
 *   - Kelly Criterion position sizing
 *   - 20,723-ticker universe screening
 *
 * Architecture:
 *   User speaks → Nova Sonic STT → LLM decides to call tool →
 *   toolUse event → processNubleTool() → ROKET API → result →
 *   sendToolResult() → Nova Sonic LLM → TTS → User hears answer
 *
 * Usage:
 *   const { processNubleTool, getNubleTools, getNubleSystemPrompt } = require('./nuble-tools');
 *
 *   // In your session setup:
 *   const tools = getNubleTools();  // Pass to promptStart toolConfiguration
 *   const systemPrompt = getNubleSystemPrompt('voice');
 *
 *   // When Bedrock sends a toolUse event:
 *   const result = await processNubleTool(toolName, toolContent);
 *   // Send result back via sendToolResult()
 */

const ROKET_BASE_URL = process.env.ROKET_BASE_URL || 'http://localhost:8000';

// ─── Tool Routes: name → [HTTP method, path template] ─────────────────

const TOOL_ROUTES = {
  // Single-ticker tools (GET with ticker in path)
  "roket_predict":       ["GET",  "/api/predict/{ticker}"],
  "roket_analyze":       ["GET",  "/api/analyze/{ticker}"],
  "roket_fundamentals":  ["GET",  "/api/fundamentals/{ticker}"],
  "roket_earnings":      ["GET",  "/api/earnings/{ticker}"],
  "roket_risk":          ["GET",  "/api/risk/{ticker}"],
  "roket_insider":       ["GET",  "/api/insider/{ticker}"],
  "roket_institutional": ["GET",  "/api/institutional/{ticker}"],
  "roket_news":          ["GET",  "/api/news/{ticker}"],
  "roket_snapshot":      ["GET",  "/api/snapshot/{ticker}"],
  "roket_sec_quality":   ["GET",  "/api/sec-quality/{ticker}"],
  "roket_lambda":        ["GET",  "/api/lambda/{ticker}"],

  // No-input tools
  "roket_regime":        ["GET",  "/api/regime"],
  "roket_macro":         ["GET",  "/api/macro"],

  // Body/query-param tools
  "roket_screener":      ["POST", "/api/screener"],
  "roket_universe":      ["GET",  "/api/universe"],
  "roket_compare":       ["GET",  "/api/compare"],
  "roket_position_size": ["POST", "/api/position-size"],

  // Additional intelligence tools
  "roket_top_picks":     ["GET",  "/api/top-picks"],
  "roket_tier":          ["GET",  "/api/tier/{ticker}"],  // {ticker} = tier name (mega/large/mid/small)
  "roket_model_info":    ["GET",  "/api/model-info"],
};

// ─── Tool Definitions in Bedrock toolSpec Format ───────────────────────

const NUBLE_TOOL_SPECS = [
  {
    toolSpec: {
      name: "roket_predict",
      description: "Get an institutional-grade ML prediction for a single stock. Returns composite score (70% fundamental + 30% timing), signal (strong_buy/buy/hold/sell/strong_sell), confidence, tier, top feature drivers. The model is a multi-tier LightGBM ensemble trained on 3.76M observations of 539 academic features from WRDS/GKX. Use for any question about a specific stock's outlook.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            ticker: { type: "string", description: "Stock ticker symbol (e.g. AAPL, TSLA, SPY)" }
          },
          required: ["ticker"]
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_analyze",
      description: "Full deep-dive analysis on a single stock. Returns prediction + fundamentals + earnings + risk + insider + institutional + regime in one call. Use when the user says 'analyze AAPL' or 'tell me everything about TSLA'. This is the most comprehensive tool.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            ticker: { type: "string", description: "Stock ticker symbol" }
          },
          required: ["ticker"]
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_fundamentals",
      description: "Get fundamental valuation factors for a stock from the WRDS/GKX panel. Returns E/P, B/M, S/P, CF/P, D/P, ROE, ROA, gross margin, operating profitability, leverage, sales growth, asset growth, R&D intensity. Use for valuation questions.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            ticker: { type: "string", description: "Stock ticker symbol" }
          },
          required: ["ticker"]
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_earnings",
      description: "Get earnings quality and surprise metrics. Returns SUE (standardized unexpected earnings), earnings yield, ROE, ROA, accruals, cash flow ratios, analyst dispersion, forecast growth.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            ticker: { type: "string", description: "Stock ticker symbol" }
          },
          required: ["ticker"]
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_risk",
      description: "Get risk and volatility metrics. Returns beta, idiosyncratic volatility, return volatility, max daily return, momentum factors (1m, 6m, 12m, 36m), realized vol, GARCH vol, coskewness, illiquidity.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            ticker: { type: "string", description: "Stock ticker symbol" }
          },
          required: ["ticker"]
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_insider",
      description: "Get insider and analyst sentiment metrics. Returns analyst count, forecast growth, dispersion, standardized forecast error, secured/convertible indicators.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            ticker: { type: "string", description: "Stock ticker symbol" }
          },
          required: ["ticker"]
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_institutional",
      description: "Get institutional ownership and market structure metrics. Returns organizational capital, Herfindahl index, dollar volume, turnover, market cap, firm age, S&P membership.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            ticker: { type: "string", description: "Stock ticker symbol" }
          },
          required: ["ticker"]
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_regime",
      description: "Detect current market regime using a Hidden Markov Model trained on 420 months of macro data. Returns regime state (bull/neutral/crisis), state probabilities, transition matrix, VIX-based exposure adjustment. Use for market conditions or macro environment questions.",
      inputSchema: {
        json: JSON.stringify({ type: "object", properties: {} })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_screener",
      description: "Screen the 20,723-ticker universe with filters. Filter by tier (mega/large/mid/small), signal (strong_buy/buy/hold/sell/strong_sell), minimum decile ranking. Returns ranked list of matching stocks.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            tier:  { type: "string", enum: ["mega","large","mid","small"], description: "Filter by market cap tier" },
            signal: { type: "string", enum: ["strong_buy","buy","hold","sell","strong_sell"], description: "Filter by signal" },
            min_decile: { type: "integer", description: "Minimum decile ranking (1-10, 10=top)" },
            limit: { type: "integer", description: "Max results (default 20)" }
          }
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_universe",
      description: "Browse the ranked stock universe. Returns top stocks ranked by ML composite score. Can filter by tier. Use for 'what are your top picks' or 'show me the best stocks'.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            tier: { type: "string", enum: ["mega","large","mid","small"], description: "Filter by tier" },
            limit: { type: "integer", description: "Max results (default 50)" }
          }
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_news",
      description: "Get REAL-TIME news and sentiment for a stock from StockNews API. Returns today's articles, 7-day daily sentiment scores (-1.5 to +1.5), and trending headlines. Use for news, sentiment, or recent catalysts.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            ticker: { type: "string", description: "Stock ticker symbol" }
          },
          required: ["ticker"]
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_snapshot",
      description: "Get a LIVE real-time market snapshot from Polygon.io. Returns current price, bid/ask, volume, technicals (SMA 20/50/200, RSI 14, MACD, Bollinger Bands, ATR), options flow, news sentiment, market regime, and LuxAlgo signals. Use for real-time market data or technical analysis.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            ticker: { type: "string", description: "Stock ticker symbol" }
          },
          required: ["ticker"]
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_sec_quality",
      description: "Get LIVE fundamental quality score from SEC EDGAR XBRL filings. Returns composite quality score (0-100), letter grade (A-F), and 40 GKX fundamental ratios from actual SEC filings. Use for financial health or quality questions.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            ticker: { type: "string", description: "Stock ticker symbol" }
          },
          required: ["ticker"]
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_macro",
      description: "Get macroeconomic environment from FRED. Returns Treasury yields, credit spreads, breakeven inflation, industrial production, unemployment, Fed Funds rate, plus yield curve state, credit cycle, and monetary policy stance.",
      inputSchema: {
        json: JSON.stringify({ type: "object", properties: {} })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_lambda",
      description: "Get a LIVE composite decision from the Lambda Decision Engine. Aggregates Polygon.io, StockNews, CryptoNews, LuxAlgo signals, and HMM regime into a single action (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL) with score 0-100. This is the MOST POWERFUL real-time tool. Use for the most authoritative live trading decision.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            ticker: { type: "string", description: "Stock or crypto ticker symbol" }
          },
          required: ["ticker"]
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_compare",
      description: "Side-by-side comparison of 2-5 stocks. Returns ML prediction, fundamentals, and risk metrics for each stock. Use when comparing stocks or choosing between options.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            tickers: { type: "string", description: "Comma-separated ticker symbols (e.g. 'AAPL,MSFT,GOOGL')" }
          },
          required: ["tickers"]
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_position_size",
      description: "Calculate optimal position size using modified Kelly Criterion. Returns recommended shares, dollar amount, stop-loss price, take-profit levels (1:1, 2:1, 3:1 R:R), and max loss. Use for 'how much should I buy' or position sizing.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            ticker: { type: "string", description: "Stock ticker symbol" },
            portfolio_value: { type: "number", description: "Total portfolio value in dollars (default 100000)" },
            risk_per_trade: { type: "number", description: "Max risk per trade as fraction (default 0.02 = 2%)" }
          },
          required: ["ticker"]
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_top_picks",
      description: "Get the top-ranked stock picks from the ML model. Returns the highest-scoring stocks from the entire 20,723-ticker universe with composite scores, signals, and confidence. Use for 'what should I buy?', 'best stocks right now', or building a portfolio.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            n: { type: "number", description: "Number of top picks to return (default 10, max 50)" },
            tier: { type: "string", description: "Filter by market cap tier: mega, large, mid, small (optional)" }
          }
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_tier",
      description: "Get ML predictions for all stocks in a specific market cap tier. Returns rankings within mega-cap, large-cap, mid-cap, or small-cap. Use for tier-specific analysis.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {
            tier: { type: "string", description: "Market cap tier: mega, large, mid, or small" }
          },
          required: ["tier"]
        })
      }
    }
  },
  {
    toolSpec: {
      name: "roket_model_info",
      description: "Get metadata about the ML models powering ROKET predictions. Returns model type, feature count, training data info, backtest metrics (IC, Sharpe, hit rate, max drawdown), and deployment status. Use when asked about model performance or methodology.",
      inputSchema: {
        json: JSON.stringify({
          type: "object",
          properties: {}
        })
      }
    }
  }
];

// ─── System Prompts ────────────────────────────────────────────────────

const NUBLE_VOICE_SYSTEM_PROMPT = `You are NUBLE — an elite institutional-grade financial advisor with a natural, conversational speaking style. You have access to the world's most advanced financial intelligence system.

VOICE INTERACTION GUIDELINES:
- Speak naturally and conversationally — you're having a real-time voice conversation
- Keep responses concise (2-4 sentences for simple questions, expand for complex analysis)
- Use clear, jargon-free language unless the user demonstrates expertise
- When presenting numbers, round appropriately for spoken delivery (say "about 15 percent" not "14.73 percent")
- Signal when you're looking up data: "Let me check that for you..." or "One moment while I pull up the latest data..."
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
- roket_screener: Screen stocks with filters
- roket_universe: Browse top-ranked stocks
- Plus: roket_fundamentals, roket_earnings, roket_risk, roket_insider, roket_institutional, roket_sec_quality`;

const NUBLE_TEXT_SYSTEM_PROMPT = `You are NUBLE — an elite institutional-grade financial advisor, researcher, and investment analyst. You have access to the most advanced financial intelligence system ever built.

YOUR CAPABILITIES:
- Multi-tier LightGBM ensemble trained on 3.76M observations of 539 academic features (Gu-Kelly-Xiu 2020)
- Real-time market data from Polygon.io (price, volume, 50+ technical indicators, options flow)
- News sentiment analysis via StockNews API (NLP-scored, 7-day rolling) and FinBERT
- SEC EDGAR XBRL filings with 40 fundamental ratios and quality scoring (A-F)
- FRED macroeconomic data (Treasury yields, credit spreads, inflation, employment)
- Hidden Markov Model regime detection (bull/neutral/crisis from 420 months of macro data)
- LuxAlgo multi-timeframe technical signals from TradingView
- Lambda Decision Engine aggregating ALL sources into a single verdict
- Modified Kelly Criterion position sizing with ATR-based stop losses
- 20,723-ticker universe coverage across mega/large/mid/small cap tiers

BEHAVIORAL GUIDELINES:
- ALWAYS call tools before giving investment opinions — never guess
- For any stock question, use roket_analyze or roket_lambda for comprehensive data
- For market overview, combine roket_regime + roket_macro
- For "should I buy X?", use roket_lambda then roket_position_size
- Present data clearly with specific numbers, percentages, and scores
- Acknowledge uncertainty and risks — never promise returns`;

// ─── Tool Execution ────────────────────────────────────────────────────

/**
 * Execute a NUBLE tool call against the ROKET API.
 * 
 * This function handles the full lifecycle:
 * 1. Parse the tool input (handles both string and object formats)
 * 2. Map tool name to ROKET API endpoint
 * 3. Make the HTTP request
 * 4. Return the parsed result
 *
 * @param {string} toolName - Tool name from Bedrock toolUse event (e.g. "roket_predict")
 * @param {object|string} toolContent - Tool input from Bedrock (content field may be JSON string)
 * @returns {Promise<object>} Tool result as JSON object
 */
async function processNubleTool(toolName, toolContent) {
  const startTime = Date.now();
  
  // ── Parse input ──────────────────────────────────────────────────
  let input;
  if (typeof toolContent === 'string') {
    try { input = JSON.parse(toolContent); } catch { input = {}; }
  } else if (toolContent && typeof toolContent.content === 'string') {
    // Bedrock sends { toolUseId, toolName, content: "{\"ticker\":\"AAPL\"}" }
    try { input = JSON.parse(toolContent.content); } catch { input = {}; }
  } else {
    input = toolContent || {};
  }

  console.log(`[NUBLE] Executing tool: ${toolName}`, JSON.stringify(input));

  // ── Route lookup ─────────────────────────────────────────────────
  const route = TOOL_ROUTES[toolName];
  if (!route) {
    console.error(`[NUBLE] Unknown tool: ${toolName}`);
    return { 
      error: `Unknown tool: ${toolName}`,
      available_tools: Object.keys(TOOL_ROUTES),
      hint: "Available tools: " + Object.keys(TOOL_ROUTES).join(', ')
    };
  }

  const [method, pathTemplate] = route;
  // For most tools, {ticker} is the stock ticker (uppercase). For roket_tier, it's the tier name (lowercase).
  let pathValue = '';
  if (toolName === 'roket_tier') {
    pathValue = (input.tier || '').toLowerCase().trim();
  } else {
    pathValue = (input.ticker || '').toUpperCase().trim();
  }
  const path = pathTemplate.replace('{ticker}', pathValue);
  const url = `${ROKET_BASE_URL}${path}`;

  // ── Execute HTTP request ─────────────────────────────────────────
  try {
    const options = { 
      method, 
      headers: { 'Content-Type': 'application/json' },
    };

    let fullUrl = url;

    if (method === 'GET') {
      // Build query params for non-path fields
      const params = new URLSearchParams();
      const pathFields = toolName === 'roket_tier' ? ['tier'] : ['ticker'];
      for (const [k, v] of Object.entries(input)) {
        if (!pathFields.includes(k) && v !== undefined && v !== null) {
          params.set(k, String(v));
        }
      }
      const queryStr = params.toString();
      if (queryStr) fullUrl = `${url}?${queryStr}`;
    } else {
      // POST with JSON body — transform specific tool inputs
      const body = { ...input };
      
      if (toolName === 'roket_screener') {
        if (body.tier && !body.tiers) { body.tiers = [body.tier]; delete body.tier; }
        if (body.signal && !body.signals) { body.signals = [body.signal.toUpperCase()]; delete body.signal; }
      }
      
      options.body = JSON.stringify(body);
    }

    console.log(`[NUBLE] ${method} ${fullUrl}`);
    
    // 60-second timeout to prevent hanging tool calls (analyze can take 15s+)
    const toolController = new AbortController();
    const toolTimeout = setTimeout(() => toolController.abort(), 60000);
    options.signal = toolController.signal;
    
    let resp;
    try {
      resp = await fetch(fullUrl, options);
    } finally {
      clearTimeout(toolTimeout);
    }

    if (!resp.ok) {
      const errorText = await resp.text();
      console.error(`[NUBLE] API error ${resp.status}: ${errorText.substring(0, 200)}`);
      return { error: `ROKET API error ${resp.status}`, detail: errorText.substring(0, 500) };
    }

    const result = await resp.json();
    const elapsed = Date.now() - startTime;
    console.log(`[NUBLE] Tool ${toolName} completed in ${elapsed}ms`);

    return result;

  } catch (err) {
    const isTimeout = err.name === 'AbortError';
    console.error(`[NUBLE] Tool ${toolName} ${isTimeout ? 'timed out (60s)' : 'failed'}:`, err.message);
    return { 
      error: isTimeout ? `Tool ${toolName} timed out after 60 seconds` : err.message, 
      tool: toolName,
      hint: isTimeout 
        ? 'The ROKET API is running but this request took too long. Try a simpler query.'
        : `Make sure the NUBLE ROKET API is running at ${ROKET_BASE_URL}. Start it with: cd NUBLE-CLI && python -m uvicorn nuble.api.roket:app --port 8000`
    };
  }
}

/**
 * Get NUBLE tools in Anthropic Messages API format.
 * Converts Bedrock toolSpec format → Anthropic { name, description, input_schema } format.
 * This is used by server.js for direct Anthropic API calls (chat completions).
 * @returns {Array} Array of { name, description, input_schema: { type, properties, required } }
 */
function getNubleToolsAnthropic() {
  return NUBLE_TOOL_SPECS.map(tool => {
    const spec = tool.toolSpec;
    let inputSchema;
    try {
      inputSchema = typeof spec.inputSchema.json === 'string' 
        ? JSON.parse(spec.inputSchema.json) 
        : spec.inputSchema.json;
    } catch {
      inputSchema = { type: 'object', properties: {} };
    }
    return {
      name: spec.name,
      description: spec.description,
      input_schema: inputSchema,
    };
  });
}

/**
 * Get NUBLE tools in Bedrock toolSpec format.
 * Pass this directly to toolConfiguration.tools in promptStart.
 * @returns {Array} Array of { toolSpec: { name, description, inputSchema: { json } } }
 */
function getNubleTools() {
  return NUBLE_TOOL_SPECS;
}

/**
 * Get the NUBLE system prompt.
 * @param {"text"|"voice"} mode - Interaction mode
 * @returns {string} System prompt
 */
function getNubleSystemPrompt(mode = 'voice') {
  return mode === 'voice' ? NUBLE_VOICE_SYSTEM_PROMPT : NUBLE_TEXT_SYSTEM_PROMPT;
}

/**
 * Check if the ROKET API is healthy and reachable.
 * @returns {Promise<{status: string, details: object}>}
 */
async function checkRoketHealth() {
  try {
    const resp = await fetch(`${ROKET_BASE_URL}/api/health`, { 
      signal: AbortSignal.timeout(5000) 
    });
    if (resp.ok) {
      const data = await resp.json();
      return { status: 'connected', details: data };
    }
    return { status: 'error', details: { statusCode: resp.status } };
  } catch (err) {
    return { status: 'disconnected', details: { error: err.message, url: ROKET_BASE_URL } };
  }
}

module.exports = {
  processNubleTool,
  getNubleTools,
  getNubleToolsAnthropic,
  getNubleSystemPrompt,
  checkRoketHealth,
  ROKET_BASE_URL,
  TOOL_ROUTES,
  NUBLE_TOOL_SPECS,
  NUBLE_VOICE_SYSTEM_PROMPT,
  NUBLE_TEXT_SYSTEM_PROMPT,
};
