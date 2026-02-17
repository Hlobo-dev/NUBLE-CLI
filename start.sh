#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# NUBLE Platform — Full Stack Startup Script
# ═══════════════════════════════════════════════════════════════════════
# Starts both services required for the NUBLE financial intelligence platform:
#   1. Python ROKET API (port 8000) — ML predictions, Polygon data, Lambda engine
#   2. Node.js Backend  (port 3000) — SvelteKit frontend + Anthropic Claude proxy
#
# Usage:
#   chmod +x start.sh && ./start.sh
#   ./start.sh --roket-only    # Only start the Python API
#   ./start.sh --node-only     # Only start the Node.js backend
#   ./start.sh --stop          # Stop all running services
# ═══════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

log_info()  { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_step()  { echo -e "${CYAN}[→]${NC} $1"; }
log_header(){ echo -e "\n${BOLD}${BLUE}═══ $1 ═══${NC}\n"; }

# ── Stop function ───────────────────────────────────────────────────
stop_services() {
    log_header "Stopping NUBLE Services"
    
    if lsof -ti:8000 >/dev/null 2>&1; then
        log_step "Stopping ROKET API (port 8000)..."
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true
        log_info "ROKET API stopped"
    else
        log_info "ROKET API not running"
    fi
    
    if lsof -ti:3000 >/dev/null 2>&1; then
        log_step "Stopping Node.js backend (port 3000)..."
        lsof -ti:3000 | xargs kill -9 2>/dev/null || true
        log_info "Node.js backend stopped"
    else
        log_info "Node.js backend not running"
    fi
    
    log_info "All services stopped"
    exit 0
}

# ── Parse arguments ─────────────────────────────────────────────────
START_ROKET=true
START_NODE=true

case "${1:-}" in
    --stop)       stop_services ;;
    --roket-only) START_NODE=false ;;
    --node-only)  START_ROKET=false ;;
    --help|-h)
        echo "Usage: $0 [--roket-only|--node-only|--stop|--help]"
        exit 0
        ;;
esac

# ── Pre-flight checks ──────────────────────────────────────────────
log_header "NUBLE Platform Startup"
echo -e "${BOLD}Project:${NC}  $SCRIPT_DIR"
echo -e "${BOLD}Date:${NC}    $(date)"
echo ""

# Check Python venv
if [ ! -f ".venv/bin/python" ]; then
    log_error "Python virtual environment not found at .venv/"
    log_step "Create it with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

PYTHON_VERSION=$(.venv/bin/python --version 2>&1)
log_info "Python: $PYTHON_VERSION"

# Check Node.js
if ! command -v node &>/dev/null; then
    log_error "Node.js not found. Install from https://nodejs.org"
    exit 1
fi
log_info "Node.js: $(node --version)"

# Check .env file
if [ ! -f "nova-sonic-frontend/.env" ]; then
    log_error "Missing nova-sonic-frontend/.env — copy from .env.example and add your API keys"
    exit 1
fi
log_info "Environment: nova-sonic-frontend/.env loaded"

# Check frontend build
if [ ! -f "nova-sonic-frontend/nuble-frontend/build/index.html" ]; then
    log_warn "Frontend build not found. Building..."
    cd nova-sonic-frontend/nuble-frontend
    npm run build
    cd "$SCRIPT_DIR"
fi
log_info "Frontend: build exists"

# Check data directory
PARQUET_COUNT=$(find data/wrds -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
log_info "Data: $PARQUET_COUNT parquet files in data/wrds/"

# Check models
MODEL_COUNT=$(find models -name "*.txt" -o -name "*.pt" 2>/dev/null | wc -l | tr -d ' ')
log_info "Models: $MODEL_COUNT model files"

# ── Kill any existing processes on target ports ─────────────────────
if lsof -ti:8000 >/dev/null 2>&1; then
    log_warn "Port 8000 in use — killing existing process"
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

if lsof -ti:3000 >/dev/null 2>&1; then
    log_warn "Port 3000 in use — killing existing process"
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# ── Create log directory ────────────────────────────────────────────
mkdir -p logs

# ══════════════════════════════════════════════════════════════════════
# SERVICE 1: ROKET Python API (port 8000)
# ══════════════════════════════════════════════════════════════════════
if [ "$START_ROKET" = true ]; then
    log_header "Starting ROKET Python API"
    log_step "Launching ML prediction engine + financial data services on port 8000..."
    
    PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR" \
    .venv/bin/python -m uvicorn nuble.api.roket:app \
        --host 0.0.0.0 \
        --port 8000 \
        --log-level info \
        > logs/roket-api.log 2>&1 &
    
    ROKET_PID=$!
    echo $ROKET_PID > logs/roket-api.pid
    
    # Wait for ROKET API to be ready
    log_step "Waiting for ROKET API to initialize (loading ML models, data panels)..."
    ROKET_READY=false
    for i in $(seq 1 30); do
        if curl -s http://localhost:8000/api/health >/dev/null 2>&1; then
            ROKET_READY=true
            break
        fi
        sleep 2
    done
    
    if [ "$ROKET_READY" = true ]; then
        HEALTH=$(curl -s http://localhost:8000/api/health 2>/dev/null)
        TICKERS=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('ticker_count', 'N/A'))" 2>/dev/null || echo "N/A")
        log_info "ROKET API ready (PID: $ROKET_PID)"
        log_info "  └─ Universe: $TICKERS tickers, 4-tier LightGBM ensemble"
        log_info "  └─ Endpoints: ML predictions, Polygon live data, Lambda engine"
        log_info "  └─ Log: logs/roket-api.log"
    else
        log_error "ROKET API failed to start within 60s — check logs/roket-api.log"
        cat logs/roket-api.log | tail -20
        exit 1
    fi
fi

# ══════════════════════════════════════════════════════════════════════
# SERVICE 2: Node.js Backend + SvelteKit Frontend (port 3000)
# ══════════════════════════════════════════════════════════════════════
if [ "$START_NODE" = true ]; then
    log_header "Starting Node.js Backend"
    log_step "Launching Express + SvelteKit + Anthropic Claude proxy on port 3000..."
    
    cd nova-sonic-frontend/nuble-backend
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        log_step "Installing Node.js dependencies..."
        npm install
    fi
    
    node server.js > "$SCRIPT_DIR/logs/node-backend.log" 2>&1 &
    NODE_PID=$!
    echo $NODE_PID > "$SCRIPT_DIR/logs/node-backend.pid"
    
    cd "$SCRIPT_DIR"
    
    # Wait for Node.js backend to be ready
    log_step "Waiting for Node.js backend..."
    NODE_READY=false
    for i in $(seq 1 15); do
        if curl -s http://localhost:3000/health >/dev/null 2>&1; then
            NODE_READY=true
            break
        fi
        sleep 1
    done
    
    if [ "$NODE_READY" = true ]; then
        log_info "Node.js backend ready (PID: $NODE_PID)"
        log_info "  └─ Frontend: SvelteKit served at http://localhost:3000"
        log_info "  └─ API: Anthropic Claude + 20 ROKET tool integrations"
        log_info "  └─ Log: logs/node-backend.log"
    else
        log_error "Node.js backend failed to start — check logs/node-backend.log"
        cat logs/node-backend.log | tail -20
        exit 1
    fi
fi

# ── Final status ────────────────────────────────────────────────────
log_header "NUBLE Platform Running"

echo -e "${BOLD}Services:${NC}"
if [ "$START_ROKET" = true ]; then
    echo -e "  ${GREEN}●${NC} ROKET Python API     ${CYAN}http://localhost:8000${NC}  (PID: $ROKET_PID)"
fi
if [ "$START_NODE" = true ]; then
    echo -e "  ${GREEN}●${NC} Node.js Backend      ${CYAN}http://localhost:3000${NC}  (PID: $NODE_PID)"
fi

echo ""
echo -e "${BOLD}Quick Access:${NC}"
echo -e "  🌐 Open App:    ${CYAN}http://localhost:3000${NC}"
echo -e "  📊 API Health:  ${CYAN}http://localhost:8000/api/health${NC}"
echo -e "  📋 Logs:        ${CYAN}tail -f logs/roket-api.log logs/node-backend.log${NC}"
echo -e "  🛑 Stop:        ${CYAN}./start.sh --stop${NC}"
echo ""
echo -e "${GREEN}${BOLD}✓ NUBLE is ready. Open http://localhost:3000 to start analyzing.${NC}"
echo ""

# Keep script running to show logs
if [ "$START_ROKET" = true ] && [ "$START_NODE" = true ]; then
    log_step "Streaming logs (Ctrl+C to stop viewing, services will continue running)..."
    echo ""
    tail -f logs/roket-api.log logs/node-backend.log 2>/dev/null
fi
