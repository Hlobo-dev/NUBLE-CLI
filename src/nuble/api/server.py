#!/usr/bin/env python3
"""
NUBLE API Server v2 — Elite Production API

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                    Frontend                          │
    │    (Input bar, chat bubbles, price cards, badges)    │
    └──────────┬──────────────────────────┬───────────────┘
               │                          │
        WebSocket /ws/chat          REST endpoints
               │                          │
    ┌──────────▼──────────────────────────▼───────────────┐
    │                  NUBLE API Server                    │
    │                                                      │
    │   1. Optimistic Fast Response (instant quote)        │
    │   2. Background APEX Dual-Brain (9 agents)           │
    │   3. Token-by-token streaming (Claude synthesis)     │
    │   4. Granular progress events (agent status)         │
    │   5. Structured metadata (verdict, score, price)     │
    └─────────────────────────────────────────────────────┘

SSE Event Flow (POST /api/chat):
    → start          { conversation_id }
    → quote          { symbol, price, change_percent, ... }     ← instant
    → progress       { stage, message, agents_done, ... }       ← real-time
    → token          { text }                                   ← word-by-word
    → response       { text, metadata }                         ← full result
    → done           { execution_time, metadata }

WebSocket Event Flow (/ws/chat):
    Same events as SSE, fully bidirectional.

REST Endpoints:
    GET  /api/health             → System health
    GET  /api/status             → Full component status
    POST /api/chat               → SSE streaming chat (elite)
    POST /api/chat/sync          → Blocking JSON (simple)
    GET  /api/quote/{symbol}     → Structured quote JSON
    GET  /api/lambda/{symbol}    → Lambda Decision Engine
    GET  /api/luxalgo/{symbol}   → LuxAlgo premium signals
    WS   /ws/chat                → WebSocket real-time chat
    DELETE /api/conversation/{id}→ Clear history
"""

import os
import sys
import re
import json
import time
import uuid
import asyncio
import logging
import threading
import queue
from datetime import datetime
from typing import Optional, Dict, Any, List, Generator
from contextlib import contextmanager

from dotenv import load_dotenv
load_dotenv()

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn[standard]")
    sys.exit(1)

from ..manager import Manager, CRYPTO_TICKERS, _get_polygon_key
from ..router import SmartRouter, QueryIntent
from .. import __version__

# Import Intelligence API router (System A+B endpoints)
try:
    from .intelligence import router as intel_router
    INTEL_ROUTER_AVAILABLE = True
except ImportError:
    INTEL_ROUTER_AVAILABLE = False
    intel_router = None

# Import Tool Executor router (Claude ↔ Tools server-side loop)
try:
    from .tool_executor import router as tool_router
    TOOL_ROUTER_AVAILABLE = True
except ImportError:
    TOOL_ROUTER_AVAILABLE = False
    tool_router = None

logger = logging.getLogger(__name__)


# =============================================================================
# Rich markup stripping
# =============================================================================

_RICH_WORDS = (
    r'bold|dim|italic|underline|blink|reverse|strikethrough|'
    r'white|red|green|yellow|blue|cyan|magenta|black|'
    r'bright_white|bright_red|bright_green|bright_yellow|'
    r'bright_blue|bright_cyan|bright_magenta|bright_black|'
    r'on_\w+'
)
_RICH_TAG_RE = re.compile(
    rf'\[/?(?:{_RICH_WORDS})(?:\s+(?:{_RICH_WORDS}))*\]'
)


def _strip_rich(text: str) -> str:
    """Remove Rich console markup tags, preserving markdown brackets."""
    if not text:
        return text or ""
    return _RICH_TAG_RE.sub('', text).strip()


# =============================================================================
# Pydantic Models
# =============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., description="User query")
    conversation_id: Optional[str] = Field(None)


class ChatMetadata(BaseModel):
    """Structured metadata returned alongside every chat response."""
    symbols: List[str] = Field(default_factory=list)
    path: str = Field("", description="'fast_path', 'apex', or 'full'")
    verdict: Optional[str] = Field(None, description="BUY / SELL / NEUTRAL / HOLD")
    score: Optional[float] = Field(None, description="Composite score 0-100")
    confidence: Optional[float] = Field(None, description="0.0-1.0")
    price: Optional[float] = None
    change_percent: Optional[float] = None
    agents_used: List[str] = Field(default_factory=list)
    decision_engine: Optional[Dict[str, Any]] = None
    ml_prediction: Optional[Dict[str, Any]] = None
    luxalgo: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None


class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    metadata: ChatMetadata
    execution_time_seconds: float


class QuoteData(BaseModel):
    symbol: str
    price: float
    change_percent: float
    volume: float
    high: float
    low: float
    currency: str = "USD"


class QuoteResponse(BaseModel):
    symbol: str
    data: Optional[QuoteData] = None
    text: str = ""
    execution_time_seconds: float


class LambdaResponse(BaseModel):
    symbol: str
    action: str
    score: float
    confidence: float
    current_price: float
    change_percent: float
    rsi: float
    macd: float
    vix: float
    luxalgo_aligned: bool
    luxalgo_direction: Optional[str] = None
    luxalgo_weekly: Optional[str] = None
    luxalgo_daily: Optional[str] = None
    luxalgo_h4: Optional[str] = None
    luxalgo_score: float
    luxalgo_valid_count: int
    stocknews_summary: Optional[str] = None
    cryptonews_summary: Optional[str] = None


class LuxAlgoResponse(BaseModel):
    symbol: str
    aligned: bool
    direction: Optional[str] = None
    weekly: Optional[str] = None
    daily: Optional[str] = None
    h4: Optional[str] = None
    score: float
    valid_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


class StatusResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    components: Dict[str, Any]


# =============================================================================
# Conversation Store
# =============================================================================

class ConversationStore:
    def __init__(self, max_conversations: int = 1000, max_messages: int = 40):
        self._store: Dict[str, list] = {}
        self._lock = threading.Lock()
        self._max = max_conversations
        self._max_messages = max_messages

    def get(self, conversation_id: str) -> list:
        with self._lock:
            if conversation_id not in self._store:
                self._store[conversation_id] = []
            return self._store[conversation_id]

    def trim(self, conversation_id: str):
        with self._lock:
            msgs = self._store.get(conversation_id, [])
            if len(msgs) > self._max_messages:
                self._store[conversation_id] = msgs[-10:]

    def clear(self, conversation_id: str):
        with self._lock:
            self._store.pop(conversation_id, None)

    def cleanup(self):
        with self._lock:
            if len(self._store) > self._max:
                keys = list(self._store.keys())
                for k in keys[: len(keys) // 2]:
                    del self._store[k]


# =============================================================================
# Structured quote helper
# =============================================================================

def _get_structured_quote(symbol: str) -> Optional[QuoteData]:
    """Fetch structured quote from Polygon — clean JSON, no markup."""
    try:
        import requests

        api_key = _get_polygon_key()
        if not api_key:
            return None

        display = symbol.upper()
        polygon_sym = CRYPTO_TICKERS.get(display, display)

        url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_sym}/prev"
        resp = requests.get(url, params={"apiKey": api_key}, timeout=5)
        if resp.status_code != 200:
            return None

        results = resp.json().get("results", [])
        if not results:
            return None

        q = results[0]
        close_p = q.get("c", 0)
        open_p = q.get("o", 0)
        change = ((close_p - open_p) / open_p * 100) if open_p > 0 else 0

        return QuoteData(
            symbol=display,
            price=round(close_p, 2),
            change_percent=round(change, 2),
            volume=q.get("v", 0),
            high=round(q.get("h", 0), 2),
            low=round(q.get("l", 0), 2),
        )
    except Exception as e:
        logger.warning(f"Structured quote failed for {symbol}: {e}")
        return None


# =============================================================================
# Elite Manager Wrapper — Streaming + Progress + Metadata
# =============================================================================

class _EliteManagerWrapper:
    """
    Elite wrapper around Manager that provides:
    1. Thread-safe serialised access
    2. Token-by-token streaming of Claude's answer
    3. Granular progress events during planning/APEX
    4. Structured metadata extraction from results
    5. Optimistic instant quote before deep analysis
    """

    def __init__(self):
        self._manager: Optional[Manager] = None
        self._init_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._router: Optional[SmartRouter] = None

    def _ensure_init(self):
        if self._manager is None:
            with self._init_lock:
                if self._manager is None:
                    logger.info("Initialising Manager (ML + DecisionEngine + APEX)...")
                    self._manager = Manager()
                    self._router = self._manager.router
                    logger.info("Manager ready — all components loaded")

    @property
    def manager(self) -> Manager:
        self._ensure_init()
        return self._manager

    @contextmanager
    def _suppress_console(self):
        """Suppress Rich console output during API calls."""
        from .. import console as nuble_console
        original_file = nuble_console.file
        try:
            nuble_console.file = open(os.devnull, "w")
            yield
        finally:
            try:
                nuble_console.file.close()
            except Exception:
                pass
            nuble_console.file = original_file

    def classify_query(self, prompt: str) -> dict:
        """
        Classify query intent WITHOUT running it.
        Returns: {is_fast_path, intent, symbols}
        """
        self._ensure_init()
        if self._router:
            routed = self._router.route(prompt)
            return {
                "is_fast_path": routed.fast_path and routed.confidence >= 0.8,
                "intent": routed.intent.value,
                "symbols": routed.symbols,
                "confidence": routed.confidence,
            }
        return {"is_fast_path": False, "intent": "general", "symbols": [], "confidence": 0}

    def process_sync(self, prompt: str, messages: list) -> str:
        """Synchronous process — returns full text (used by /chat/sync)."""
        self._ensure_init()
        with self._call_lock:
            with self._suppress_console():
                result = self._manager.process_prompt(prompt, messages)
                return _strip_rich(result) if result else ""

    def process_streaming(self, prompt: str, messages: list, event_queue: queue.Queue):
        """
        Run full Manager pipeline with granular events pushed to queue.

        Events pushed:
            ("progress", {"stage": "routing", "message": "..."})
            ("quote", {QuoteData...})
            ("progress", {"stage": "planning", ...})
            ("progress", {"stage": "apex_started", ...})
            ("progress", {"stage": "agent_done", "agent": "market_analyst", ...})
            ("token", {"text": "chunk"})
            ("response", {"text": "full text", "metadata": {...}})
            ("done", {"execution_time": 42.1, "metadata": {...}})
            ("error", {"message": "..."})
        """
        self._ensure_init()
        t0 = time.time()

        try:
            with self._call_lock:
                self._run_streaming_pipeline(prompt, messages, event_queue, t0)
        except Exception as e:
            logger.error(f"Streaming pipeline error: {e}", exc_info=True)
            event_queue.put(("error", {"message": str(e)}))

    def _run_streaming_pipeline(self, prompt: str, messages: list,
                                 event_queue: queue.Queue, t0: float):
        """Core streaming pipeline — runs under _call_lock."""
        from .. import console as nuble_console
        mgr = self._manager

        # ── Step 1: Route the query ──
        classification = self.classify_query(prompt)
        symbols = classification["symbols"]
        is_fast = classification["is_fast_path"]
        intent = classification["intent"]

        event_queue.put(("progress", {
            "stage": "routing",
            "message": f"Detected: {intent}",
            "symbols": symbols,
            "is_fast_path": is_fast,
        }))

        # ── Step 2: Optimistic instant quote ──
        # For ANY query with a symbol, push the price data immediately
        # so the frontend can show it while the analysis runs.
        quote_data = None
        if symbols:
            try:
                quote_data = _get_structured_quote(symbols[0])
                if quote_data:
                    event_queue.put(("quote", quote_data.model_dump()))
            except Exception:
                pass

        # ── Step 3a: Fast path — return immediately ──
        if is_fast:
            with self._suppress_console():
                result = mgr.process_prompt(prompt, messages)
                clean = _strip_rich(result) if result else ""

            metadata = ChatMetadata(
                symbols=symbols,
                path="fast_path",
                price=quote_data.price if quote_data else None,
                change_percent=quote_data.change_percent if quote_data else None,
                execution_time=round(time.time() - t0, 2),
            )
            event_queue.put(("response", {
                "text": clean,
                "metadata": metadata.model_dump(),
            }))
            event_queue.put(("done", {
                "execution_time": round(time.time() - t0, 2),
                "metadata": metadata.model_dump(),
            }))
            return

        # ── Step 3b: Full APEX path — streaming ──

        # Launch APEX Orchestrator in background
        apex_container = None
        if mgr.apex_enabled and mgr._orchestrator:
            try:
                apex_container = mgr._launch_orchestrator_background(prompt)
                event_queue.put(("progress", {
                    "stage": "apex_started",
                    "message": "9 specialist agents analyzing in parallel...",
                }))
            except Exception as e:
                logger.warning(f"APEX launch failed: {e}")

        # Start a monitoring thread for APEX agent progress
        apex_monitor_stop = threading.Event()
        if apex_container:
            self._start_apex_monitor(apex_container, event_queue, apex_monitor_stop)

        # ── Step 4: Run Manager planning (blocking, under console suppression) ──
        event_queue.put(("progress", {
            "stage": "planning",
            "message": "Claude is planning research steps...",
        }))

        # We need to run the planning + answer phases separately to stream tokens.
        # However, Manager.process_prompt() is monolithic — it does planning AND
        # answer in one call. To stream the answer tokens, we intercept the
        # agent.answer() generator.
        #
        # Strategy: Run process_prompt for planning, then separately call
        # agent.answer() for streaming. But process_prompt modifies `conversation`
        # in place, so after it returns the conversation has all planning data.
        #
        # Better strategy: Run the entire process_prompt but monkey-patch the
        # agent.answer to capture chunks. This is cleaner.

        answer_chunks = []
        original_answer = mgr.agent.answer

        def _intercepting_answer(question, conv_messages):
            """Intercept agent.answer() to capture and stream tokens."""
            event_queue.put(("progress", {
                "stage": "synthesizing",
                "message": "Generating analysis...",
            }))
            for chunk in original_answer(question, conv_messages):
                answer_chunks.append(chunk)
                event_queue.put(("token", {"text": chunk}))
                yield chunk

        # Monkey-patch temporarily
        mgr.agent.answer = _intercepting_answer

        try:
            with self._suppress_console():
                full_result = mgr.process_prompt(prompt, messages)
        finally:
            mgr.agent.answer = original_answer
            apex_monitor_stop.set()

        clean_result = _strip_rich(full_result) if full_result else ""

        # ── Step 5: Collect APEX results for metadata ──
        apex_data = None
        if apex_container:
            try:
                apex_container['completed'].wait(timeout=5.0)
                apex_data = apex_container.get('result', [None])[0]
            except Exception:
                pass

        # ── Step 6: Build structured metadata ──
        metadata = self._extract_metadata(
            clean_result, symbols, apex_data, quote_data, t0
        )

        event_queue.put(("response", {
            "text": clean_result,
            "metadata": metadata.model_dump(),
        }))
        event_queue.put(("done", {
            "execution_time": round(time.time() - t0, 2),
            "metadata": metadata.model_dump(),
        }))

    def _start_apex_monitor(self, apex_container: dict,
                             event_queue: queue.Queue,
                             stop_event: threading.Event):
        """Monitor APEX orchestrator and push agent completion events."""
        def _monitor():
            seen_agents = set()
            while not stop_event.is_set():
                result = apex_container.get('result', [None])[0]
                if result and isinstance(result, dict):
                    agents_used = result.get('agents_used', [])
                    for agent in agents_used:
                        if agent not in seen_agents:
                            seen_agents.add(agent)
                            event_queue.put(("progress", {
                                "stage": "agent_done",
                                "agent": agent,
                                "agents_done": len(seen_agents),
                                "message": f"Agent completed: {agent}",
                            }))
                if apex_container.get('completed', threading.Event()).is_set():
                    if result:
                        event_queue.put(("progress", {
                            "stage": "apex_complete",
                            "message": "Deep analysis complete",
                            "agents_used": result.get('agents_used', []),
                            "execution_time": result.get('execution_time_seconds', 0),
                        }))
                    break
                stop_event.wait(timeout=2.0)

        t = threading.Thread(target=_monitor, daemon=True)
        t.start()

    def _extract_metadata(self, text: str, symbols: list,
                           apex_data: Optional[dict],
                           quote_data: Optional[QuoteData],
                           t0: float) -> ChatMetadata:
        """Extract structured metadata from the analysis results."""
        metadata = ChatMetadata(
            symbols=symbols,
            path="apex" if apex_data else "full",
            execution_time=round(time.time() - t0, 2),
        )

        # Quote data
        if quote_data:
            metadata.price = quote_data.price
            metadata.change_percent = quote_data.change_percent

        # APEX data
        if apex_data and isinstance(apex_data, dict):
            metadata.agents_used = apex_data.get('agents_used', [])
            metadata.confidence = apex_data.get('confidence', None)

            # Decision Engine
            data = apex_data.get('data', {})
            de = data.get('decision_engine')
            if de:
                metadata.decision_engine = de
                metadata.verdict = de.get('action')
                metadata.score = de.get('confidence', 0) * 100

            # ML predictions
            ml = data.get('ml_predictions', {})
            if ml:
                first_sym = next(iter(ml), None)
                if first_sym:
                    metadata.ml_prediction = ml[first_sym]

            # LuxAlgo
            lux = de.get('luxalgo', {}) if de else {}
            if lux:
                metadata.luxalgo = lux

        # Fallback: try to extract verdict from text
        if not metadata.verdict and text:
            text_upper = text.upper()
            if 'NOT RECOMMENDED' in text_upper or 'AVOID' in text_upper:
                metadata.verdict = 'AVOID'
            elif 'STRONG BUY' in text_upper:
                metadata.verdict = 'STRONG BUY'
            elif 'BUY' in text_upper and 'SELL' not in text_upper:
                metadata.verdict = 'BUY'
            elif 'SELL' in text_upper and 'BUY' not in text_upper:
                metadata.verdict = 'SELL'
            elif 'HOLD' in text_upper or 'NEUTRAL' in text_upper:
                metadata.verdict = 'NEUTRAL'

        # Fallback: extract score from Lambda if no APEX
        if metadata.score is None and text:
            import re as _re
            score_match = _re.search(r'(?:score|rating)[:\s]*(\d+(?:\.\d+)?)\s*/?\s*100', text, _re.IGNORECASE)
            if score_match:
                try:
                    metadata.score = float(score_match.group(1))
                except ValueError:
                    pass

        return metadata


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="NUBLE API",
    description=(
        "Institutional-Grade AI Investment Research — APEX Dual-Brain Fusion. "
        "9 specialist agents • DecisionEngine (28+ data points) • ML Predictor • "
        "LuxAlgo Premium • SEC Filing RAG • Token-by-token streaming"
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: lock to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Intelligence API router (System A+B endpoints)
if INTEL_ROUTER_AVAILABLE and intel_router is not None:
    app.include_router(intel_router)
    logger.info("✅ Intelligence API registered at /api/intel/*")

# Register Tool Executor router (Claude ↔ Tools server-side loop)
if TOOL_ROUTER_AVAILABLE and tool_router is not None:
    app.include_router(tool_router)
    logger.info("✅ Tool Executor registered at /api/intel/chat-with-tools")

_start_time = datetime.now()
_conversations = ConversationStore()
_mgr = _EliteManagerWrapper()


@app.on_event("startup")
async def startup():
    logger.info("NUBLE API starting up...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _mgr._ensure_init)

    # ── Start Learning Resolver (background prediction resolution) ───────
    try:
        from ..learning.learning_hub import LearningHub
        from ..learning.resolver import PredictionResolver
        _resolver = PredictionResolver(LearningHub())
        await _resolver.start()
        logger.info("Learning resolver started — resolving predictions hourly")
    except Exception as exc:
        logger.warning("Learning resolver unavailable: %s", exc)

    logger.info("NUBLE API ready")


# =============================================================================
# Routes
# =============================================================================

# ── Health & Status ──────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health():
    uptime = (datetime.now() - _start_time).total_seconds()
    return HealthResponse(status="healthy", version=__version__, uptime_seconds=uptime)


@app.get("/api/status", response_model=StatusResponse)
async def status():
    uptime = (datetime.now() - _start_time).total_seconds()
    mgr = _mgr.manager

    components: Dict[str, Any] = {
        "ml_predictor": {"available": mgr.ml_enabled},
        "decision_engine": {"available": mgr.decision_engine_enabled},
        "smart_router": {"available": mgr.fast_path_enabled},
        "apex_orchestrator": {"available": mgr.apex_enabled},
    }

    try:
        from ..lambda_client import get_lambda_client, NUBLE_API_BASE
        get_lambda_client()
        components["lambda"] = {"available": True, "endpoint": NUBLE_API_BASE}
    except Exception:
        components["lambda"] = {"available": False}

    try:
        from ..agents.fundamental_analyst import FundamentalAnalystAgent
        fa = FundamentalAnalystAgent()
        if fa._init_tenk():
            filings = fa._tenk_db.list_filings()
            components["tenk_rag"] = {"available": True, "filings_loaded": len(filings)}
        else:
            components["tenk_rag"] = {"available": False}
    except Exception:
        components["tenk_rag"] = {"available": False}

    return StatusResponse(
        status="healthy", version=__version__,
        uptime_seconds=uptime, components=components,
    )


# ── Chat (Elite SSE Stream) ─────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    """
    Elite chat endpoint — Server-Sent Events with granular streaming.

    Event types sent to client:
        start     → { conversation_id }
        quote     → { symbol, price, change_percent, volume, ... }  ← instant
        progress  → { stage, message, agents_done, ... }             ← real-time
        token     → { text }                                         ← word-by-word
        response  → { text, metadata }                               ← full result
        done      → { execution_time, metadata }
        error     → { message }

    Frontend integration:

        const evtSource = new EventSource('/api/chat', {method: 'POST', ...});
        // Or use fetch + ReadableStream:

        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: 'Should I buy TSLA?'})
        });
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            const lines = decoder.decode(value).split('\\n');
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const event = JSON.parse(line.slice(6));
                switch (event.type) {
                    case 'quote':    showPriceCard(event);    break;
                    case 'progress': updateSpinner(event);    break;
                    case 'token':    appendToChat(event.text); break;
                    case 'response': showFinalResult(event);  break;
                    case 'done':     showMetadata(event);     break;
                }
            }
        }
    """
    conv_id = request.conversation_id or str(uuid.uuid4())

    async def generate():
        yield f"data: {json.dumps({'type': 'start', 'conversation_id': conv_id})}\n\n"

        messages = _conversations.get(conv_id)
        messages.append({"role": "user", "content": request.message})

        # Create event queue for streaming
        eq: queue.Queue = queue.Queue()

        # Run pipeline in background thread
        loop = asyncio.get_event_loop()
        pipeline_future = loop.run_in_executor(
            None,
            lambda: _mgr.process_streaming(request.message, messages, eq)
        )

        # Drain events from queue as they arrive
        done = False
        while not done:
            try:
                # Check for events with a short timeout
                event_type, event_data = await loop.run_in_executor(
                    None,
                    lambda: eq.get(timeout=0.1)
                )

                payload = {"type": event_type}
                payload.update(event_data)
                yield f"data: {json.dumps(payload, default=str)}\n\n"

                if event_type in ("done", "error"):
                    done = True

            except queue.Empty:
                # Check if pipeline thread finished without pushing done
                if pipeline_future.done():
                    # Drain remaining events
                    while not eq.empty():
                        try:
                            event_type, event_data = eq.get_nowait()
                            payload = {"type": event_type}
                            payload.update(event_data)
                            yield f"data: {json.dumps(payload, default=str)}\n\n"
                            if event_type in ("done", "error"):
                                done = True
                        except queue.Empty:
                            break
                    if not done:
                        # Pipeline finished but no done event — error case
                        exc = pipeline_future.exception()
                        if exc:
                            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                        done = True

        _conversations.trim(conv_id)

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Chat (sync) ─────────────────────────────────────────────────────────────

@app.post("/api/chat/sync", response_model=ChatResponse)
async def chat_sync(request: ChatRequest):
    """Synchronous chat — simpler, blocks until complete."""
    conv_id = request.conversation_id or str(uuid.uuid4())
    t0 = time.time()

    messages = _conversations.get(conv_id)
    messages.append({"role": "user", "content": request.message})

    # Classify for metadata
    classification = _mgr.classify_query(request.message)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _mgr.process_sync(request.message, messages)
    )

    elapsed = time.time() - t0
    _conversations.trim(conv_id)

    # Build metadata
    path = "fast_path" if elapsed < 5 else "apex"
    quote_data = None
    if classification["symbols"]:
        quote_data = await loop.run_in_executor(
            None,
            lambda: _get_structured_quote(classification["symbols"][0])
        )

    metadata = ChatMetadata(
        symbols=classification["symbols"],
        path=path,
        price=quote_data.price if quote_data else None,
        change_percent=quote_data.change_percent if quote_data else None,
        execution_time=round(elapsed, 2),
    )

    return ChatResponse(
        message=result or "",
        conversation_id=conv_id,
        metadata=metadata,
        execution_time_seconds=round(elapsed, 2),
    )


# ── Quote ────────────────────────────────────────────────────────────────────

@app.get("/api/quote/{symbol}", response_model=QuoteResponse)
async def quick_quote(symbol: str):
    t0 = time.time()
    loop = asyncio.get_event_loop()
    quote = await loop.run_in_executor(None, lambda: _get_structured_quote(symbol))

    if quote:
        return QuoteResponse(
            symbol=quote.symbol, data=quote,
            execution_time_seconds=round(time.time() - t0, 2),
        )

    # Fallback through Manager
    messages: list = []
    messages.append({"role": "user", "content": symbol.upper()})
    result = await loop.run_in_executor(
        None,
        lambda: _mgr.process_sync(symbol.upper(), messages)
    )

    return QuoteResponse(
        symbol=symbol.upper(),
        text=result or "No data available",
        execution_time_seconds=round(time.time() - t0, 2),
    )


# ── Lambda ───────────────────────────────────────────────────────────────────

@app.get("/api/lambda/{symbol}", response_model=LambdaResponse)
async def lambda_analysis(symbol: str):
    try:
        from ..lambda_client import get_lambda_client
        client = get_lambda_client()
        a = client.get_analysis(symbol.upper())

        return LambdaResponse(
            symbol=symbol.upper(),
            action=a.action, score=a.score, confidence=a.confidence,
            current_price=a.current_price, change_percent=a.change_percent,
            rsi=a.rsi, macd=a.macd, vix=a.vix,
            luxalgo_aligned=a.luxalgo_aligned,
            luxalgo_direction=a.luxalgo_direction or None,
            luxalgo_weekly=a.luxalgo_weekly_action or None,
            luxalgo_daily=a.luxalgo_daily_action or None,
            luxalgo_h4=a.luxalgo_h4_action or None,
            luxalgo_score=a.luxalgo_score,
            luxalgo_valid_count=a.luxalgo_valid_count,
            stocknews_summary=a.stocknews_summary[:500] if a.stocknews_summary else None,
            cryptonews_summary=a.cryptonews_summary[:500] if a.cryptonews_summary else None,
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="Lambda client not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── LuxAlgo ──────────────────────────────────────────────────────────────────

@app.get("/api/luxalgo/{symbol}", response_model=LuxAlgoResponse)
async def luxalgo_signals(symbol: str):
    try:
        from ..lambda_client import get_lambda_client
        client = get_lambda_client()
        a = client.get_analysis(symbol.upper())

        return LuxAlgoResponse(
            symbol=symbol.upper(),
            aligned=a.luxalgo_aligned,
            direction=a.luxalgo_direction or None,
            weekly=a.luxalgo_weekly_action or None,
            daily=a.luxalgo_daily_action or None,
            h4=a.luxalgo_h4_action or None,
            score=a.luxalgo_score,
            valid_count=a.luxalgo_valid_count,
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="Lambda client not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── WebSocket (Elite) ───────────────────────────────────────────────────────

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    Elite WebSocket chat — same events as SSE, fully bidirectional.

    Send:    {"message": "Should I buy TSLA?", "conversation_id": "optional"}
    Receive: Same event stream as POST /api/chat (start, quote, progress,
             token, response, done, error)

    The WebSocket stays open for multiple messages. Each message triggers
    a full analysis pipeline with streaming events.
    """
    await websocket.accept()
    ws_conv_id = str(uuid.uuid4())
    logger.info(f"WebSocket connected: {ws_conv_id}")

    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            message = payload.get("message", "")
            conv_id = payload.get("conversation_id", ws_conv_id)

            if not message:
                await websocket.send_json({"type": "error", "message": "Empty message"})
                continue

            await websocket.send_json({"type": "start", "conversation_id": conv_id})

            messages = _conversations.get(conv_id)
            messages.append({"role": "user", "content": message})

            # Create event queue
            eq: queue.Queue = queue.Queue()

            # Run pipeline in background thread
            loop = asyncio.get_event_loop()
            pipeline_future = loop.run_in_executor(
                None,
                lambda m=message, msgs=messages, q=eq: _mgr.process_streaming(m, msgs, q)
            )

            # Stream events to WebSocket
            done = False
            while not done:
                try:
                    event_type, event_data = await loop.run_in_executor(
                        None,
                        lambda: eq.get(timeout=0.2)
                    )
                    event_payload = {"type": event_type}
                    event_payload.update(event_data)
                    await websocket.send_json(event_payload)

                    if event_type in ("done", "error"):
                        done = True

                except queue.Empty:
                    if pipeline_future.done():
                        while not eq.empty():
                            try:
                                et, ed = eq.get_nowait()
                                ep = {"type": et}
                                ep.update(ed)
                                await websocket.send_json(ep)
                                if et in ("done", "error"):
                                    done = True
                            except queue.Empty:
                                break
                        if not done:
                            exc = pipeline_future.exception()
                            if exc:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": str(exc)
                                })
                            done = True

            _conversations.trim(conv_id)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {ws_conv_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass



# ── Learning System ─────────────────────────────────────────────────────

@app.get("/api/learning/stats")
async def learning_stats():
    """Get learning system statistics: accuracy, predictions, current weights."""
    try:
        from ..learning.learning_hub import LearningHub
        hub = LearningHub()
        return {
            "status": "active",
            "accuracy": hub.get_accuracy_report(),
            "predictions": hub.get_prediction_stats(),
            "current_weights": hub.get_weights(),
        }
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


@app.get("/api/learning/predictions")
async def learning_predictions():
    """Get all raw predictions (for debugging/analysis)."""
    try:
        from ..learning.learning_hub import LearningHub
        hub = LearningHub()
        unresolved = hub.get_unresolved()
        return {
            "total": len(hub._raw_predictions),
            "unresolved_count": len(unresolved),
            "recent_predictions": list(hub._raw_predictions.values())[-20:],
        }
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


# ── Conversation management ─────────────────────────────────────────────────

@app.delete("/api/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    _conversations.clear(conversation_id)
    return {"status": "cleared", "conversation_id": conversation_id}


# =============================================================================
# CLI runner
# =============================================================================

def run():
    import uvicorn
    host = os.environ.get("NUBLE_API_HOST", "0.0.0.0")
    port = int(os.environ.get("NUBLE_API_PORT", "8000"))
    reload = os.environ.get("NUBLE_API_RELOAD", "false").lower() == "true"
    uvicorn.run("nuble.api.server:app", host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    run()
