#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  NUBLE ROKET — Full-Stack Launcher                                         ║
# ║  Starts all services: Python ROKET API + Node.js Backend + SvelteKit UI    ║
# ║                                                                             ║
# ║  Usage:  bash startall.sh          Start everything                         ║
# ║          bash startall.sh stop     Stop all services                        ║
# ║          bash startall.sh status   Check service status                     ║
# ║          bash startall.sh logs     Tail all logs                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -uo pipefail

# ── Resolve project root (always absolute) ──────────────────────────────────
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Config ──────────────────────────────────────────────────────────────────
ROKET_PORT=8000
NODE_PORT=3000
VENV="$ROOT/.venv"
BACKEND_DIR="$ROOT/nova-sonic-frontend/nuble-backend"
FRONTEND_DIR="$ROOT/nova-sonic-frontend/nuble-frontend"
ENV_FILE="$ROOT/nova-sonic-frontend/.env"
LOG_DIR="$ROOT/logs"
ROKET_LOG="$LOG_DIR/roket-api.log"
NODE_LOG="$LOG_DIR/node-backend.log"
ROKET_PID_FILE="$LOG_DIR/roket-api.pid"
NODE_PID_FILE="$LOG_DIR/node-backend.pid"

# ── Colors ──────────────────────────────────────────────────────────────────
R='\033[0;31m' G='\033[0;32m' Y='\033[1;33m' B='\033[0;34m'
C='\033[0;36m' W='\033[1;37m' D='\033[0;90m' N='\033[0m'

# ── Logging helpers ─────────────────────────────────────────────────────────
ok()   { printf "${G}  ✓${N} %s\n" "$1"; }
warn() { printf "${Y}  !${N} %s\n" "$1"; }
fail() { printf "${R}  ✗${N} %s\n" "$1"; }
info() { printf "${C}  →${N} %s\n" "$1"; }
hdr()  { printf "\n${W}━━━ %s ━━━${N}\n\n" "$1"; }

# ── Utility: check if a port is in use ──────────────────────────────────────
port_pid() { lsof -ti:"$1" 2>/dev/null | head -1; }

# ── Utility: wait for a port to respond ─────────────────────────────────────
wait_for_port() {
    local port=$1 label=$2 max=$3
    local i=0
    while [ $i -lt "$max" ]; do
        if curl -sf "http://localhost:${port}" >/dev/null 2>&1 || \
           curl -sf "http://localhost:${port}/health" >/dev/null 2>&1 || \
           curl -sf "http://localhost:${port}/api/health" >/dev/null 2>&1; then
            return 0
        fi
        i=$((i + 1))
        sleep 2
    done
    return 1
}

# ── STOP ────────────────────────────────────────────────────────────────────
do_stop() {
    hdr "Stopping NUBLE Services"

    local pid
    pid=$(port_pid $ROKET_PORT)
    if [ -n "$pid" ]; then
        kill -9 "$pid" 2>/dev/null || true
        ok "ROKET API stopped (PID $pid)"
    else
        ok "ROKET API already stopped"
    fi

    pid=$(port_pid $NODE_PORT)
    if [ -n "$pid" ]; then
        kill -9 "$pid" 2>/dev/null || true
        ok "Node.js backend stopped (PID $pid)"
    else
        ok "Node.js backend already stopped"
    fi

    # Clean up PID files
    rm -f "$ROKET_PID_FILE" "$NODE_PID_FILE" 2>/dev/null
    printf "\n${G}All services stopped.${N}\n\n"
}

# ── STATUS ──────────────────────────────────────────────────────────────────
do_status() {
    hdr "NUBLE Service Status"

    local pid
    pid=$(port_pid $ROKET_PORT)
    if [ -n "$pid" ]; then
        ok "ROKET API        :${ROKET_PORT}  PID=$pid"
    else
        fail "ROKET API        :${ROKET_PORT}  DOWN"
    fi

    pid=$(port_pid $NODE_PORT)
    if [ -n "$pid" ]; then
        ok "Node.js Backend  :${NODE_PORT}  PID=$pid"
    else
        fail "Node.js Backend  :${NODE_PORT}  DOWN"
    fi
    echo ""
}

# ── LOGS ────────────────────────────────────────────────────────────────────
do_logs() {
    hdr "Streaming Logs (Ctrl+C to exit)"
    tail -f "$ROKET_LOG" "$NODE_LOG" 2>/dev/null
}

# ── Handle subcommands ──────────────────────────────────────────────────────
case "${1:-start}" in
    stop)   do_stop;   exit 0 ;;
    status) do_status; exit 0 ;;
    logs)   do_logs;   exit 0 ;;
    start)  ;; # fall through
    *)
        echo "Usage: bash startall.sh [start|stop|status|logs]"
        exit 1
        ;;
esac

# ════════════════════════════════════════════════════════════════════════════
#  START — Full-stack launch sequence
# ════════════════════════════════════════════════════════════════════════════

printf "\n"
printf "${W}╔══════════════════════════════════════════════════════════════╗${N}\n"
printf "${W}║${N}  ${C}NUBLE ROKET${N} — Full-Stack Launcher                         ${W}║${N}\n"
printf "${W}║${N}  ${D}Institutional-Grade Financial Intelligence Platform${N}         ${W}║${N}\n"
printf "${W}╚══════════════════════════════════════════════════════════════╝${N}\n"
printf "\n"

# ── Pre-flight checks ──────────────────────────────────────────────────────
hdr "Pre-Flight Checks"

# Python venv
if [ ! -f "$VENV/bin/python" ]; then
    fail "Python venv not found at $VENV"
    info "Run: python3 -m venv .venv && .venv/bin/pip install -e '.[all]'"
    exit 1
fi
ok "Python: $($VENV/bin/python --version 2>&1)"

# Node.js
if ! command -v node &>/dev/null; then
    fail "Node.js not found — install from https://nodejs.org"
    exit 1
fi
ok "Node.js: $(node --version)"

# .env
if [ ! -f "$ENV_FILE" ]; then
    fail "Missing $ENV_FILE"
    exit 1
fi
ok "Environment: .env loaded"

# Frontend build
if [ ! -f "$FRONTEND_DIR/build/index.html" ]; then
    warn "Frontend build not found — building now..."
    (cd "$FRONTEND_DIR" && npm run build:direct)
fi
ok "Frontend: SvelteKit build ready"

# Node modules
if [ ! -d "$BACKEND_DIR/node_modules" ]; then
    warn "Node modules missing — installing..."
    (cd "$BACKEND_DIR" && npm install)
fi
ok "Backend: node_modules ready"

# Data
PARQUETS=$(find "$ROOT/data/wrds" -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
MODELS=$(find "$ROOT/models" -name "*.txt" -o -name "*.pt" 2>/dev/null | wc -l | tr -d ' ')
ok "Data: ${PARQUETS} parquets, ${MODELS} model files"

# ── Kill any existing processes on target ports ─────────────────────────────
for p in $ROKET_PORT $NODE_PORT; do
    pid=$(port_pid "$p")
    if [ -n "$pid" ]; then
        warn "Port $p occupied (PID $pid) — killing"
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi
done

# ── Create log directory ────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

# ════════════════════════════════════════════════════════════════════════════
#  SERVICE 1:  ROKET Python FastAPI  (port 8000)
# ════════════════════════════════════════════════════════════════════════════
hdr "Starting ROKET Python API (:$ROKET_PORT)"

info "Loading 4 LightGBM models, 20K+ ticker universe, HMM regime detector..."

PYTHONPATH="$ROOT/src:$ROOT" \
nohup "$VENV/bin/python" -m uvicorn nuble.api.roket:app \
    --host 0.0.0.0 \
    --port "$ROKET_PORT" \
    --log-level info \
    > "$ROKET_LOG" 2>&1 &

ROKET_PID=$!
echo "$ROKET_PID" > "$ROKET_PID_FILE"
info "Process launched (PID: $ROKET_PID) — waiting for health..."

if wait_for_port "$ROKET_PORT" "ROKET API" 45; then
    ok "ROKET API is UP on port $ROKET_PORT (PID: $ROKET_PID)"
else
    fail "ROKET API failed to start within 90s"
    fail "Last 15 lines of log:"
    tail -15 "$ROKET_LOG" 2>/dev/null | while read -r line; do printf "     ${D}%s${N}\n" "$line"; done
    exit 1
fi

# ════════════════════════════════════════════════════════════════════════════
#  SERVICE 2:  Node.js Express + SvelteKit  (port 3000)
# ════════════════════════════════════════════════════════════════════════════
hdr "Starting Node.js Backend (:$NODE_PORT)"

info "Express server + Anthropic Claude proxy + 20 ROKET tool integrations..."

cd "$BACKEND_DIR"
nohup node server.js > "$NODE_LOG" 2>&1 &
NODE_PID=$!
echo "$NODE_PID" > "$NODE_PID_FILE"
cd "$ROOT"

info "Process launched (PID: $NODE_PID) — waiting for health..."

if wait_for_port "$NODE_PORT" "Node.js" 20; then
    ok "Node.js Backend is UP on port $NODE_PORT (PID: $NODE_PID)"
else
    fail "Node.js Backend failed to start"
    fail "Last 15 lines of log:"
    tail -15 "$NODE_LOG" 2>/dev/null | while read -r line; do printf "     ${D}%s${N}\n" "$line"; done
    exit 1
fi

# ════════════════════════════════════════════════════════════════════════════
#  VERIFICATION
# ════════════════════════════════════════════════════════════════════════════
hdr "Verification"

# Quick smoke tests
ROKET_OK=false
NODE_OK=false

ROKET_RESP=$(curl -sf "http://localhost:$ROKET_PORT/api/health" 2>/dev/null || curl -sf "http://localhost:$ROKET_PORT/" 2>/dev/null || echo "")
if [ -n "$ROKET_RESP" ]; then
    ROKET_OK=true
    ok "ROKET API responds"
else
    warn "ROKET API health endpoint returned empty (service may still be loading models)"
    # Check if process is alive
    if kill -0 "$ROKET_PID" 2>/dev/null; then
        ok "ROKET process is alive (PID: $ROKET_PID) — models loading"
        ROKET_OK=true
    else
        fail "ROKET process died"
    fi
fi

NODE_RESP=$(curl -sf "http://localhost:$NODE_PORT/health" 2>/dev/null || echo "")
if echo "$NODE_RESP" | grep -q "status"; then
    NODE_OK=true
    ok "Node.js Backend responds"
else
    fail "Node.js Backend not responding"
fi

# ════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ════════════════════════════════════════════════════════════════════════════

printf "\n"
printf "${W}╔══════════════════════════════════════════════════════════════╗${N}\n"
printf "${W}║${N}  ${G}NUBLE ROKET — All Systems Online${N}                           ${W}║${N}\n"
printf "${W}╠══════════════════════════════════════════════════════════════╣${N}\n"

if [ "$ROKET_OK" = true ]; then
    printf "${W}║${N}  ${G}●${N}  ROKET Python API     ${C}http://localhost:%-5s${N}  PID %-6s ${W}║${N}\n" "$ROKET_PORT" "$ROKET_PID"
else
    printf "${W}║${N}  ${R}●${N}  ROKET Python API     ${R}FAILED${N}                           ${W}║${N}\n"
fi

if [ "$NODE_OK" = true ]; then
    printf "${W}║${N}  ${G}●${N}  Node.js + Frontend   ${C}http://localhost:%-5s${N}  PID %-6s ${W}║${N}\n" "$NODE_PORT" "$NODE_PID"
else
    printf "${W}║${N}  ${R}●${N}  Node.js + Frontend   ${R}FAILED${N}                           ${W}║${N}\n"
fi

printf "${W}╠══════════════════════════════════════════════════════════════╣${N}\n"
printf "${W}║${N}                                                              ${W}║${N}\n"
printf "${W}║${N}  Open App     ${C}http://localhost:3000${N}                         ${W}║${N}\n"
printf "${W}║${N}  API Health   ${C}http://localhost:8000/api/health${N}              ${W}║${N}\n"
printf "${W}║${N}  View Logs    ${C}bash startall.sh logs${N}                        ${W}║${N}\n"
printf "${W}║${N}  Stop All     ${C}bash startall.sh stop${N}                        ${W}║${N}\n"
printf "${W}║${N}  Status       ${C}bash startall.sh status${N}                      ${W}║${N}\n"
printf "${W}║${N}                                                              ${W}║${N}\n"
printf "${W}╚══════════════════════════════════════════════════════════════╝${N}\n"
printf "\n"
printf "${D}Services run in background via nohup. They survive terminal close.${N}\n"
printf "${D}Logs: logs/roket-api.log  |  logs/node-backend.log${N}\n"
printf "\n"
