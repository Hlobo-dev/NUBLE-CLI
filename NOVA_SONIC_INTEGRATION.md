# NUBLE × Nova Sonic Integration Guide

> **Transform your Nova Sonic voice chatbot into the world's most advanced financial advisor.**

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION ARCHITECTURE                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐    Socket.IO    ┌──────────────────────────────────────┐  │
│   │              │ ◄──────────────►│                                      │  │
│   │   Browser    │                 │    local-server.js (MODIFIED)        │  │
│   │  (NUBLE UI)  │  localhost:3000 │    ├── nuble-tools.js (NEW)         │  │
│   │              │                 │    ├── nova-stream-customer.js       │  │
│   │  🎤 → STT   │                 │    └── Tool execution handler (NEW)  │  │
│   │  TTS → 🔊   │                 │                                      │  │
│   └──────────────┘                 └────────┬──────────────┬─────────────┘  │
│                                             │              │                 │
│                                    HTTP/2   │              │  HTTP/REST      │
│                                    Bidir    │              │                 │
│                                             ▼              ▼                 │
│                              ┌──────────────────┐  ┌──────────────────────┐ │
│                              │  AWS Bedrock      │  │  NUBLE ROKET API    │ │
│                              │  Nova Sonic       │  │  localhost:8000     │ │
│                              │  (STT/LLM/TTS)   │  │  17 financial tools │ │
│                              │                   │  │  ML predictions     │ │
│                              │  Decides WHICH    │  │  Real-time data     │ │
│                              │  tool to call     │  │  News sentiment     │ │
│                              └──────────────────┘  │  SEC filings        │ │
│                                                     │  Macro regime       │ │
│                                                     └──────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Flow:**
1. User speaks → Browser captures audio → Socket.IO → `local-server.js` → Bedrock Nova Sonic (STT)
2. Nova Sonic LLM decides to call a NUBLE tool (e.g. `roket_lambda`)
3. Bedrock sends `toolUse` event → `handleOutputEvent()` catches it
4. Server calls `processNubleTool('roket_lambda', {ticker:'AAPL'})` → HTTP to ROKET API
5. Result comes back → Server sends `sendToolResult()` back to Bedrock
6. Nova Sonic LLM reads the result and speaks the answer (TTS)
7. Audio flows back → Socket.IO → Browser plays audio

---

## Files Created in NUBLE-CLI

| File | Purpose |
|------|---------|
| `src/nuble/api/nuble-tools.js` | Node.js module — drop into nova-sonic-debug. Contains all 17 NUBLE tools in Bedrock toolSpec format, `processNubleTool()` dispatcher, system prompts |
| `src/nuble/api/nuble_tool_bridge.py` | Python universal bridge — serves tools in both Anthropic and Bedrock format via REST, also generates the Node.js module |

---

## Step-by-Step Integration

### Prerequisites

1. NUBLE ROKET API running:
   ```bash
   cd ~/Desktop/NUBLE-CLI
   python -m uvicorn nuble.api.roket:app --port 8000
   ```

2. nova-sonic-debug cloned and on ROKET branch:
   ```bash
   cd ~/Desktop
   git clone https://github.com/Hlobo-dev/nova-sonic-debug.git
   cd nova-sonic-debug
   git checkout ROKET
   ```

---

### Step 1: Copy `nuble-tools.js` into the frontend repo

```bash
cp ~/Desktop/NUBLE-CLI/src/nuble/api/nuble-tools.js ~/Desktop/nova-sonic-debug/src/nuble-tools.js
```

---

### Step 2: Modify `local-server.js` — 4 changes

#### Change 2a: Import nuble-tools at the top

After line 26 (`const { randomUUID } = require('crypto');`), add:

```javascript
const { processNubleTool, getNubleTools, getNubleSystemPrompt, checkRoketHealth } = require('./src/nuble-tools');
```

#### Change 2b: Replace the system prompt

Replace the `DEFAULT_SYSTEM_PROMPT` constant (around line 41) with:

```javascript
const DEFAULT_SYSTEM_PROMPT = getNubleSystemPrompt('voice');
```

#### Change 2c: Pass NUBLE tools to NovaStream

In the `initializeConnection` handler (around line 370), change:

```javascript
// FROM:
const session = new NovaStream(
    socket.id,
    voiceId,
    systemPrompt,
    [],              // ← empty tools array
    null
);

// TO:
const session = new NovaStream(
    socket.id,
    voiceId,
    systemPrompt,
    getNubleTools().map(t => ({
        toolSpec: () => t.toolSpec   // NovaStream expects tools with a .toolSpec() method
    })),
    null
);
```

#### Change 2d: Add tool handling to `handleOutputEvent()`

The current `handleOutputEvent()` function handles `contentStart`, `textOutput`, `audioOutput`, and `contentEnd` — but **NOT tool events**. Add tool handling. After the `if (event.audioOutput) { ... }` block and before the `if (event.contentEnd) { ... }` block, add:

```javascript
    // ─── NUBLE Tool Use Handling ─────────────────────────────────────
    if (event.toolUse) {
        console.log(`[NUBLE] Tool use detected: ${event.toolUse.toolName}`);
        
        // Store tool info on the session for when contentEnd(TOOL) arrives
        if (sessionData) {
            sessionData.pendingToolUse = {
                toolUseId: event.toolUse.toolUseId,
                toolName: event.toolUse.toolName,
                content: event.toolUse.content || '{}'
            };
        }
        
        // Emit to frontend for optional UI display
        socket.emit('toolUse', {
            toolName: event.toolUse.toolName,
            toolUseId: event.toolUse.toolUseId
        });
    }

    // Handle tool content end — this is when we actually execute the tool
    if (event.contentEnd && event.contentEnd.type === 'TOOL' && sessionData?.pendingToolUse) {
        const { toolUseId, toolName, content } = sessionData.pendingToolUse;
        sessionData.pendingToolUse = null;

        console.log(`[NUBLE] Executing tool: ${toolName} (${toolUseId})`);
        
        try {
            // Call NUBLE ROKET API
            const toolResult = await processNubleTool(toolName, content);
            const resultStr = typeof toolResult === 'string' ? toolResult : JSON.stringify(toolResult);

            console.log(`[NUBLE] Tool ${toolName} result: ${resultStr.substring(0, 200)}...`);

            // Send tool result back to Bedrock via NovaStream event queue
            const session = sessionData.session;
            const contentId = randomUUID();

            // 1. contentStart (TOOL)
            session.eventQueue.push({
                event: {
                    contentStart: {
                        promptName: session.promptName,
                        contentName: contentId,
                        interactive: false,
                        type: 'TOOL',
                        role: 'TOOL',
                        toolResultInputConfiguration: {
                            toolUseId: toolUseId,
                            type: 'TEXT',
                            textInputConfiguration: {
                                mediaType: 'text/plain'
                            }
                        }
                    }
                }
            });

            // 2. toolResult (the actual data)
            session.eventQueue.push({
                event: {
                    toolResult: {
                        promptName: session.promptName,
                        contentName: contentId,
                        content: resultStr
                    }
                }
            });

            // 3. contentEnd
            session.eventQueue.push({
                event: {
                    contentEnd: {
                        promptName: session.promptName,
                        contentName: contentId
                    }
                }
            });

            console.log(`[NUBLE] Tool result sent back to Bedrock for ${toolName}`);

            // Emit to frontend
            socket.emit('toolResult', {
                toolName,
                toolUseId,
                resultPreview: resultStr.substring(0, 300)
            });

        } catch (error) {
            console.error(`[NUBLE] Tool execution failed:`, error.message);
            
            // Send error as tool result so Nova Sonic can tell the user
            const session = sessionData.session;
            const contentId = randomUUID();

            session.eventQueue.push({
                event: {
                    contentStart: {
                        promptName: session.promptName,
                        contentName: contentId,
                        interactive: false,
                        type: 'TOOL',
                        role: 'TOOL',
                        toolResultInputConfiguration: {
                            toolUseId: toolUseId,
                            type: 'TEXT',
                            textInputConfiguration: { mediaType: 'text/plain' }
                        }
                    }
                }
            });
            session.eventQueue.push({
                event: {
                    toolResult: {
                        promptName: session.promptName,
                        contentName: contentId,
                        content: JSON.stringify({ error: error.message, tool: toolName })
                    }
                }
            });
            session.eventQueue.push({
                event: {
                    contentEnd: {
                        promptName: session.promptName,
                        contentName: contentId
                    }
                }
            });
        }

        // Don't fall through to the generic contentEnd handler below
        return;
    }
```

---

### Step 3: Add `ROKET_BASE_URL` to `.env`

In your nova-sonic-debug `.env` file, add:

```
ROKET_BASE_URL=http://localhost:8000
```

---

### Step 4: Add health check endpoint (optional but recommended)

In `local-server.js`, after the existing `/health` endpoint, add:

```javascript
app.get('/nuble-health', async (req, res) => {
    const health = await checkRoketHealth();
    res.json({
        nuble: health,
        novaSonic: { status: 'ok', activeSessions: sessionStore.size },
        timestamp: new Date().toISOString()
    });
});
```

---

### Step 5: Update Frontend UI (optional — for branding)

In `public/src/main.js`, replace the `SYSTEM_PROMPT` variable (the BreakThru Beverage sales coach prompt) with:

```javascript
let SYSTEM_PROMPT = null; // Will use server-side NUBLE prompt from getNubleSystemPrompt('voice')
```

The system prompt is now managed server-side via `getNubleSystemPrompt('voice')` which is set as `DEFAULT_SYSTEM_PROMPT` in `local-server.js`.

---

## How Tool Execution Works

```
User says: "What's the outlook for Apple?"

1. Browser → [audio chunks] → Socket.IO → local-server.js → Bedrock Nova Sonic
2. Nova Sonic STT: "What's the outlook for Apple?"
3. Nova Sonic LLM decides: call tool "roket_lambda" with {"ticker": "AAPL"}
4. Bedrock sends event: { event: { toolUse: { toolName: "roket_lambda", content: '{"ticker":"AAPL"}', toolUseId: "abc123" } } }
5. local-server.js handleOutputEvent() stores pendingToolUse
6. Bedrock sends event: { event: { contentEnd: { type: "TOOL" } } }
7. handleOutputEvent() detects contentEnd(TOOL) + pendingToolUse
8. Calls: processNubleTool("roket_lambda", '{"ticker":"AAPL"}')
9. nuble-tools.js → HTTP GET http://localhost:8000/api/lambda/AAPL
10. ROKET API returns: { action: "BUY", score: 78, confidence: "HIGH", ... }
11. local-server.js pushes contentStart(TOOL) + toolResult + contentEnd to NovaStream event queue
12. Bedrock receives tool result
13. Nova Sonic LLM generates response: "Apple is looking strong right now. My Lambda Decision Engine gives it a Buy signal with a score of 78 out of 100 and high confidence. The fundamentals are solid and the technical signals are aligned..."
14. Bedrock sends textOutput + audioOutput events
15. Browser plays the audio response
```

---

## Tool Reference (17 tools available)

| Tool | Trigger Phrases | What It Returns |
|------|----------------|-----------------|
| `roket_predict` | "What's the prediction for X?" | ML score, signal, confidence, tier |
| `roket_analyze` | "Analyze X", "Tell me about X" | Full deep-dive (prediction + fundamentals + risk) |
| `roket_lambda` | "Should I buy X?", "What's the live verdict?" | Composite score 0-100, action, all data sources |
| `roket_snapshot` | "What's the current price of X?" | Live price, technicals, options flow |
| `roket_news` | "What's in the news about X?" | Articles, 7-day sentiment scores |
| `roket_fundamentals` | "What are the fundamentals for X?" | E/P, B/M, ROE, margins, growth |
| `roket_earnings` | "How are X's earnings?" | SUE, earnings yield, analyst estimates |
| `roket_risk` | "How risky is X?" | Beta, volatility, momentum, max drawdown |
| `roket_insider` | "Any insider activity for X?" | Analyst coverage, forecast dispersion |
| `roket_institutional` | "Institutional ownership of X?" | Org capital, Herfindahl, turnover |
| `roket_sec_quality` | "What's the financial health of X?" | Quality score 0-100, letter grade |
| `roket_regime` | "How's the market?", "Market conditions?" | Bull/neutral/crisis, state probs |
| `roket_macro` | "What's the macro environment?" | Treasury yields, credit, inflation |
| `roket_screener` | "Show me strong buy mega caps" | Filtered stock list with scores |
| `roket_universe` | "What are your top picks?" | Ranked stock list by ML score |
| `roket_compare` | "Compare AAPL vs MSFT" | Side-by-side ML + fundamentals |
| `roket_position_size` | "How much should I buy?" | Kelly criterion, shares, stop-loss |

---

## Testing

1. Start the ROKET API:
   ```bash
   cd ~/Desktop/NUBLE-CLI
   python -m uvicorn nuble.api.roket:app --port 8000
   ```

2. Start the frontend:
   ```bash
   cd ~/Desktop/nova-sonic-debug
   node local-server.js
   ```

3. Open http://localhost:3000 in Chrome

4. Click the microphone and say:
   - "What's the outlook for Apple?" → triggers `roket_lambda`
   - "How's the market right now?" → triggers `roket_regime`
   - "Compare Tesla and Nvidia" → triggers `roket_compare`
   - "What are your top picks?" → triggers `roket_universe`

5. Check the ROKET API health:
   ```bash
   curl http://localhost:3000/nuble-health
   ```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Unknown tool: roket_X" | Tool name mismatch | Check TOOL_ROUTES in nuble-tools.js |
| "ROKET API error 404" | Wrong endpoint path | Verify ROKET_BASE_URL in .env |
| Tool called but no result | `handleOutputEvent` not catching toolUse | Ensure the tool handling code is BEFORE the generic contentEnd handler |
| Nova Sonic doesn't call tools | Tools not in promptStart | Verify getNubleTools() returns non-empty array and tools have `.toolSpec()` wrapper |
| "Stream died unexpectedly" | HTTP/2 timeout | Already fixed with 900000ms timeout in nova-stream-customer.js |
| Audio stops after tool call | Tool result format wrong | Ensure contentStart(TOOL) → toolResult → contentEnd sequence is correct |
