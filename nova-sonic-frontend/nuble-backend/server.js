/**
 * ROKET Backend - AI Financial Intelligence Platform
 * Production-grade Node.js server with structured logging, security hardening,
 * and optimized API proxying for Anthropic Claude + Amazon Nova Sonic.
 */

require('dotenv').config({ path: require('path').resolve(__dirname, '..', '.env') });

const express = require('express');
const cors = require('cors');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { v4: uuidv4 } = require('uuid');
const cookieParser = require('cookie-parser');
const fs = require('fs');
const path = require('path');
const http = require('http');
const { Server } = require('socket.io');
const rateLimit = require('express-rate-limit');
const compression = require('compression');
const helmet = require('helmet');
const multer = require('multer');
const XLSX = require('xlsx');
const pdfParse = require('pdf-parse');
const mammoth = require('mammoth');
const { parseOffice } = require('officeparser');
const { NovaSonicBidirectionalStreamClient } = require('./nova-sonic-client');
const { processNubleTool, getNubleTools, getNubleToolsAnthropic, getNubleSystemPrompt, checkRoketHealth } = require('./nuble-tools');

// ==================== Environment & Structured Logging ====================
const NODE_ENV = process.env.NODE_ENV || 'development';
const IS_PRODUCTION = NODE_ENV === 'production';

const pino = require('pino');
const logger = pino({
  level: process.env.LOG_LEVEL || (IS_PRODUCTION ? 'info' : 'debug'),
  transport: IS_PRODUCTION ? undefined : { target: 'pino-pretty', options: { colorize: true } },
  ...(IS_PRODUCTION ? {} : {}),
  serializers: {
    err: pino.stdSerializers.err,
    req: (req) => ({ method: req.method, url: req.url }),
  },
});

// ==================== ROKET Default System Prompt ====================
const ROKET_SYSTEM_PROMPT = `You are ROKET, the world's most advanced AI financial intelligence platform — powered by NUBLE, an institutional-grade dual-brain trading system with 118,000+ lines of production code, 9 specialist AI agents, 4 WRDS-trained LightGBM models, and real-time data from 40+ API endpoints.

IDENTITY:
- Your name is ROKET. You are not a generic AI assistant — you are a specialized financial intelligence system backed by institutional-grade infrastructure.
- You are direct, confident, and precise. You speak like a senior Goldman Sachs strategist who happens to be genuinely helpful and approachable.
- Never use emojis in your responses. Never. Your formatting should be clean, professional, and scannable — like a Bloomberg terminal or a top-tier research report.

THE NUBLE SYSTEM BEHIND YOU:
You are the interface to a dual-brain AI system:

SYSTEM A (Real-Time Intelligence):
- Lambda Decision Engine on AWS (40+ real-time API endpoints)
- Polygon.io real-time market data (prices, OHLCV, technicals, options flow)
- StockNews API (24 endpoints: sentiment, analyst ratings, earnings, SEC filings, events, trending)
- CryptoNews API (17 endpoints: whale tracking, institutional flows, regulatory)
- LuxAlgo multi-timeframe signals from TradingView (weekly/daily/4H confirmations)
- 9 specialist AI agents running in parallel: Market Analyst, Quant Analyst, News Analyst, Fundamental Analyst, Macro Analyst, Risk Manager, Portfolio Optimizer, Crypto Specialist, Educator

SYSTEM B (ML Pipeline — WRDS/GKX Academic Models):
- 4 LightGBM models trained on WRDS/GKX panel data (3.76 million observations, 539 academic features, 20,723 tickers)
- Features include: Fama-French factors, momentum, value, quality, profitability, investment, size, volatility, liquidity, earnings quality, analyst signals, insider activity, institutional ownership
- Live Polygon.io feature engine computing 600+ WRDS-compatible features in real-time
- HMM regime detection (3-state: BULL/SIDEWAYS/BEAR from 420 months of macro data)
- Walk-forward backtested with Information Coefficient, Sharpe ratio, hit rate, and decile monotonicity

CONVERGENCE ENGINES:
- UltimateDecisionEngine: 28+ weighted data points with risk veto (Technical 35%, Intelligence 30%, Market Structure 20%, Validation 15%)
- SignalFusionEngine: Multi-source signal blending (LuxAlgo 50%, ML 25%, Sentiment 10%, Regime 10%, Fundamental 5%)
- VetoEngine: Institutional multi-timeframe hierarchy (NEVER trade against weekly trend)
- TradeSetupCalculator: ATR-based entry/stop/targets with Keltner channels
- PositionCalculator: Modified Kelly Criterion with half-Kelly safety (max 2% risk, max 10% position)
- SEC EDGAR XBRL analysis: 40+ fundamental ratios with quality grading (A-F)
- FRED macroeconomic data: Treasury yields, credit spreads, inflation, employment

CAPABILITIES:
- You have access to the user's connected brokerage accounts, portfolio positions, P&L, cost basis, allocation data, and transaction history.
- You have real-time and near-real-time market data across all asset classes: equities, crypto, forex, commodities, bonds, options, futures, and derivatives.
- You synthesize data from institutional-grade sources: market feeds, SEC filings, earnings transcripts, analyst ratings, insider trading data, institutional flows, and macro indicators.
- You track crypto on-chain metrics, DeFi protocols, whale movements, exchange flows, and tokenomics.
- You understand portfolio theory, risk management, tax optimization, and behavioral finance at an expert level.

RESPONSE STYLE:
- Be specific. Use real numbers, tickers, percentages, and dates. Never say "it went up a lot" — say "up 3.2% today to $67,450."
- Structure your responses clearly with bold headers and bullet points when listing multiple items. No emoji bullets — use dashes or standard bullet points.
- When giving market updates: price, change (% and $), volume context, and a brief "why" if there's a catalyst.
- When analyzing portfolio: reference specific holdings, give allocation percentages, flag concentration risks, and suggest actionable improvements.
- When giving opinions: be direct. State your view, explain the reasoning, and note the key risks. Don't hedge with "it depends" unless it genuinely does, and then explain what it depends on.
- Keep responses appropriately sized. Short answers for simple questions. Deep analysis when the question warrants it.

FINANCIAL EXPERTISE:
- Portfolio analysis: allocation optimization, rebalancing, risk-adjusted returns, Sharpe ratio, beta, drawdown analysis, correlation matrices.
- Options and derivatives: Greeks, IV, spreads, hedging strategies, covered calls, protective puts.
- Tax strategy: tax-loss harvesting, long-term vs short-term capital gains, wash sale rules, Roth conversions, estate planning.
- Macro analysis: Fed policy, yield curve, inflation metrics, employment data, PMI, GDP, geopolitical risk assessment.
- Crypto: on-chain analysis, protocol fundamentals, DeFi yields, staking economics, regulatory landscape.
- Technical analysis: support/resistance, moving averages, RSI, MACD, volume profile, market structure.

RULES:
1. Never use emojis. Ever. Not in headers, not in bullets, not anywhere.
2. Be proactive — if you spot something in their portfolio that needs attention, flag it.
3. You ARE their financial advisor. Don't add disclaimers like "I'm not a financial advisor" or "this is not financial advice." Be transparent about uncertainty when it exists, but own your expertise.
4. Match the user's sophistication. If they ask a beginner question, explain clearly without condescension. If they're a pro, go deep.
5. When you don't have specific real-time data, say so clearly and provide analysis based on what you do know.
6. Format for readability: use bold for key terms, dashes for lists, clear section breaks for long responses.

TOOLS — NUBLE ROKET Financial Intelligence (20 Tools):
You have access to 20 institutional-grade financial tools via the NUBLE ROKET API. USE THEM PROACTIVELY whenever a user asks about a stock, market conditions, portfolio analysis, or any financial topic. Do not guess or fabricate data — call the tools to get real data.

Available tools:
- roket_predict: ML prediction for a stock (LightGBM ensemble, 539 GKX features, composite score + signal + confidence)
- roket_analyze: COMPREHENSIVE deep-dive — prediction + fundamentals + earnings + risk + insider + institutional + regime + live intelligence + trade setup + signal fusion + veto + position sizing — ALL IN ONE CALL
- roket_fundamentals: WRDS/GKX valuation factors (E/P, B/M, S/P, CF/P, ROE, ROA, margins, leverage, growth, R&D intensity)
- roket_earnings: Earnings quality from WRDS (SUE, persistence, smoothness, accruals, cash flow ratios, analyst dispersion)
- roket_risk: Risk profile (6-factor betas, 15 volatility metrics, momentum, VaR, drawdown, Sharpe, Sortino)
- roket_insider: Insider trading activity (buy ratio, CEO buy, cluster buy, analyst consensus, net sentiment)
- roket_institutional: Institutional ownership (top holders, ownership changes, HHI concentration, breadth)
- roket_news: News sentiment (StockNews API + FinBERT NLP scoring, recent headlines)
- roket_snapshot: Real-time market snapshot (price, change, volume, technicals, options flow, LuxAlgo signals)
- roket_sec_quality: SEC EDGAR XBRL filing quality (40+ fundamental ratios, composite quality grade A-F)
- roket_lambda: Lambda Decision Engine — the MOST POWERFUL real-time tool (aggregates Polygon, StockNews, CryptoNews, LuxAlgo, HMM regime into single STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL verdict with score 0-100)
- roket_regime: HMM regime detection (bull/sideways/bear from 420 months of macro history, state probabilities)
- roket_macro: FRED macroeconomic environment (Treasury yields 2Y/10Y/30Y, credit spreads, inflation, employment, VIX)
- roket_screener: Screen the 20,723-ticker universe with custom filters (score, signal, tier, sector, market cap)
- roket_universe: List all tickers in the universe with current predictions
- roket_compare: Side-by-side comparison of 2-5 stocks (ML prediction + fundamentals + risk for each)
- roket_position_size: Kelly Criterion optimal position sizing (shares, dollar amount, stop-loss, 3 take-profit levels, max loss)
- roket_top_picks: Best stock picks from the ML model ranked by composite score (customizable count and tier filter)
- roket_tier: All predictions for a specific market cap tier (mega/large/mid/small)
- roket_model_info: ML model metadata, backtest metrics (IC, Sharpe, hit rate, max drawdown), deployment status

TOOL USAGE GUIDELINES:
- For "what do you think about AAPL?" → call roket_analyze (comprehensive) or roket_lambda (real-time verdict)
- For "is now a good time to buy?" → call roket_regime + roket_macro + roket_predict
- For price/quick data → call roket_snapshot
- For portfolio review → call roket_predict on each holding + roket_risk
- For screening ("find me tech stocks under $50") → call roket_screener
- For "what should I buy?" → call roket_top_picks then roket_analyze on the top results
- For position sizing ("how much should I buy?") → call roket_position_size
- For model performance questions → call roket_model_info
- Always present tool results in your expert analysis — synthesize, don't just dump raw data

MULTI-TOOL ANALYSIS (CRITICAL):
- You can call MULTIPLE tools in a single turn. For complex questions, call all relevant tools at once — they execute in parallel for speed.
- For deep analysis, chain tools across rounds: first gather data (snapshot + fundamentals + risk), then synthesize with ML signals (predict + lambda + regime).
- Example: "Should I buy TSLA?" → Call roket_snapshot, roket_predict, roket_fundamentals, roket_risk, roket_regime, and roket_macro ALL AT ONCE, then synthesize into a comprehensive recommendation with entry/stop/target levels.
- Example: "Compare AAPL vs MSFT vs GOOGL" → Call roket_compare with all three, PLUS roket_predict on each, for a data-rich comparison.
- Example: "Build me a portfolio" → Call roket_top_picks for best stocks, then roket_risk + roket_position_size for each to build an optimized allocation.
- Example: "How is the macro environment?" → Call roket_macro + roket_regime for the complete picture.
- NEVER give a vague answer when you have tools available. Always back your analysis with real data from the tools.
- When presenting results, create a unified narrative — weave together ML predictions, fundamental data, risk metrics, macro context, and regime state into one cohesive expert opinion.
- For the most authoritative answer, use roket_lambda — it aggregates ALL data sources (Polygon, StockNews, LuxAlgo, HMM regime, ML models) into a single institutional-grade verdict.`;

const app = express();
const server = http.createServer(app);

// Production and development origins — locked per environment
const DEV_ORIGINS = [
  'http://localhost:5173',
  'http://localhost:5174',
  'http://localhost:3000',
  'http://127.0.0.1:5173',
];
const PROD_ORIGINS = (process.env.ALLOWED_ORIGINS || 'https://cloudlobo.com,https://www.cloudlobo.com,https://api.cloudlobo.com').split(',');
const ALLOWED_ORIGINS = IS_PRODUCTION ? PROD_ORIGINS : [...DEV_ORIGINS, ...PROD_ORIGINS];

const io = new Server(server, {
  path: '/ws/socket.io',
  cors: {
    origin: ALLOWED_ORIGINS,
    credentials: true
  },
  // Production Socket.IO hardening
  pingTimeout: 30000,
  pingInterval: 25000,
  maxHttpBufferSize: 1e6, // 1MB max payload
  transports: ['websocket', 'polling'],
});

const PORT = process.env.PORT || 8080;

// JWT Secret: MUST be set in production via env var / Secrets Manager
const JWT_SECRET = process.env.JWT_SECRET || (() => {
  if (IS_PRODUCTION) {
    logger.fatal('JWT_SECRET environment variable is required in production');
    process.exit(1);
  }
  const devSecret = 'dev-only-secret-' + require('crypto').randomBytes(16).toString('hex');
  logger.warn({ msg: 'Using auto-generated JWT_SECRET (dev only)' });
  return devSecret;
})();
const JWT_EXPIRY = '7d';

// Anthropic API key - MUST be set via environment variable
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY || '';
if (!ANTHROPIC_API_KEY) {
  logger.warn('ANTHROPIC_API_KEY not set — chat completions will fail');
}

// Request timeout for Anthropic API calls (prevents hanging connections)
const ANTHROPIC_REQUEST_TIMEOUT_MS = parseInt(process.env.ANTHROPIC_TIMEOUT_MS || '120000', 10);
// Static directory for favicons
const STATIC_DIR = path.join(__dirname, 'static');
if (!fs.existsSync(STATIC_DIR)) {
  fs.mkdirSync(STATIC_DIR, { recursive: true });
}

// Create default favicons if they don't exist
const defaultFavicon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" fill="#6366f1"/><text x="50" y="65" font-family="Arial" font-size="50" fill="white" text-anchor="middle">N</text></svg>`;
if (!fs.existsSync(path.join(STATIC_DIR, 'favicon.png'))) {
  fs.writeFileSync(path.join(STATIC_DIR, 'favicon.svg'), defaultFavicon);
}

// Data directory
const DATA_DIR = path.join(__dirname, 'data');
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

// Uploads directory for file uploads
const UPLOADS_DIR = path.join(DATA_DIR, 'uploads');
if (!fs.existsSync(UPLOADS_DIR)) {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

// Configure multer for file uploads (50MB limit)
const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => cb(null, UPLOADS_DIR),
    filename: (req, file, cb) => {
      const uniqueName = `${Date.now()}-${uuidv4()}${path.extname(file.originalname)}`;
      cb(null, uniqueName);
    }
  }),
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB
  fileFilter: (req, file, cb) => {
    // Allow common document types + be lenient with MIME types
    const allowedMimes = [
      // Text
      'text/plain', 'text/csv', 'text/markdown', 'text/html', 'text/css', 'text/xml',
      'text/x-python', 'text/x-java', 'text/x-c', 'text/x-c++', 'text/x-ruby',
      'text/x-sql', 'text/x-shellscript', 'text/x-yaml', 'text/tab-separated-values',
      'text/richtext', 'text/rtf',
      // Documents
      'application/pdf',
      'application/json',
      'application/javascript',
      'application/xml',
      'application/rtf',
      'application/x-rtf',
      // Microsoft Office
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // xlsx
      'application/vnd.ms-excel', // xls
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // docx
      'application/msword', // doc
      'application/vnd.openxmlformats-officedocument.presentationml.presentation', // pptx
      'application/vnd.ms-powerpoint', // ppt
      // OpenDocument
      'application/vnd.oasis.opendocument.text', // odt
      'application/vnd.oasis.opendocument.spreadsheet', // ods
      'application/vnd.oasis.opendocument.presentation', // odp
      // Archives (for reference only)
      'application/zip',
      // Generic binary — check extension instead
      'application/octet-stream',
      // Images
      'image/png', 'image/jpeg', 'image/gif', 'image/webp', 'image/svg+xml',
      'image/bmp', 'image/tiff',
      // Audio
      'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/webm', 'audio/mp4', 'audio/flac',
      // Video
      'video/mp4', 'video/webm', 'video/quicktime',
    ];
    
    // Also allow by file extension for common document types
    const allowedExtensions = [
      // Code & text
      '.txt', '.csv', '.tsv', '.md', '.markdown', '.json', '.jsonl',
      '.js', '.jsx', '.ts', '.tsx', '.py', '.pyw', '.rb', '.java', '.c', '.cpp', '.h', '.hpp',
      '.cs', '.go', '.rs', '.swift', '.kt', '.scala', '.php', '.pl', '.pm', '.r', '.R',
      '.html', '.htm', '.css', '.scss', '.sass', '.less',
      '.xml', '.xsl', '.xslt', '.svg',
      '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.env',
      '.log', '.sql', '.sh', '.bash', '.zsh', '.fish', '.bat', '.ps1', '.psm1',
      '.lua', '.vim', '.tex', '.latex', '.bib',
      '.gitignore', '.dockerignore', '.editorconfig',
      '.makefile', '.cmake',
      // Documents
      '.pdf', '.doc', '.docx', '.rtf',
      '.xls', '.xlsx', '.xlsm',
      '.ppt', '.pptx',
      '.odt', '.ods', '.odp',
      // Images
      '.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.bmp', '.tiff', '.ico',
      // Audio
      '.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac',
      // Video
      '.mp4', '.webm', '.mov', '.avi',
    ];
    
    const ext = path.extname(file.originalname).toLowerCase();
    
    if (allowedMimes.includes(file.mimetype) || file.mimetype.startsWith('text/') || allowedExtensions.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error(`File type ${file.mimetype} (${ext}) not supported`), false);
    }
  }
});

// Simple JSON file-based database with debounced writes
class JsonDB {
  constructor(filename) {
    this.filepath = path.join(DATA_DIR, filename);
    this.data = this.load();
    this._saveTimer = null;
    this._saveDelay = 500; // Debounce writes by 500ms
  }

  load() {
    try {
      if (fs.existsSync(this.filepath)) {
        return JSON.parse(fs.readFileSync(this.filepath, 'utf8'));
      }
    } catch (e) {
      logger.error({ err: e, file: this.filepath }, 'Error loading database file');
    }
    return [];
  }

  save() {
    // Debounce: only write after _saveDelay ms of inactivity
    if (this._saveTimer) clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => {
      try {
        // Write to temp file then rename for atomicity
        const tmpPath = this.filepath + '.tmp';
        fs.writeFileSync(tmpPath, JSON.stringify(this.data, null, 2));
        fs.renameSync(tmpPath, this.filepath);
      } catch (e) {
        logger.error({ err: e, file: this.filepath }, 'Error saving database file');
      }
    }, this._saveDelay);
  }

  // Force immediate save (for graceful shutdown)
  saveSync() {
    if (this._saveTimer) {
      clearTimeout(this._saveTimer);
      this._saveTimer = null;
    }
    fs.writeFileSync(this.filepath, JSON.stringify(this.data, null, 2));
  }

  findOne(predicate) {
    return this.data.find(predicate);
  }

  findAll(predicate) {
    return predicate ? this.data.filter(predicate) : this.data;
  }

  insert(item) {
    this.data.push(item);
    this.save();
    return item;
  }

  update(predicate, updates) {
    const index = this.data.findIndex(predicate);
    if (index !== -1) {
      this.data[index] = { ...this.data[index], ...updates };
      this.save();
      return this.data[index];
    }
    return null;
  }

  delete(predicate) {
    const index = this.data.findIndex(predicate);
    if (index !== -1) {
      const deleted = this.data.splice(index, 1)[0];
      this.save();
      return deleted;
    }
    return null;
  }

  count() {
    return this.data.length;
  }
}

// Initialize databases
const usersDB = new JsonDB('users.json');
const chatsDB = new JsonDB('chats.json');
const configDB = new JsonDB('config.json');
const memoriesDB = new JsonDB('memories.json');
const foldersDB = new JsonDB('folders.json');
const filesDB = new JsonDB('files.json');

// Load or create config
let config = configDB.findOne(c => c.type === 'app_config');
if (!config) {
  config = {
    type: 'app_config',
    signup_enabled: true,
    default_user_role: 'user',
    jwt_expiry: JWT_EXPIRY
  };
  configDB.insert(config);
}

// Security headers (disable CSP for dev; frontend handles its own)
app.use(helmet({
  contentSecurityPolicy: false,
  crossOriginEmbedderPolicy: false,
  crossOriginResourcePolicy: { policy: 'cross-origin' }
}));

// Compress all HTTP responses
app.use(compression());

// Serve static files FIRST - completely bypass CORS middleware
app.use('/static', (req, res, next) => {
  // Handle OPTIONS preflight
  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    return res.sendStatus(204);
  }
  next();
}, express.static(STATIC_DIR, {
  maxAge: '1d', // Cache static assets for 1 day
  setHeaders: (res, path) => {
    // Clear any credentials header and set wildcard origin
    res.removeHeader('Access-Control-Allow-Credentials');
    res.removeHeader('Vary');
    res.setHeader('Access-Control-Allow-Origin', '*');
  }
}));

// CORS middleware for API routes only (skip /static paths)
app.use((req, res, next) => {
  if (req.path.startsWith('/static')) {
    return next();
  }
  cors({
    origin: ALLOWED_ORIGINS,
    credentials: true
  })(req, res, next);
});

app.use(express.json({ limit: '10mb' }));
app.use(cookieParser());

// Rate limiting for auth endpoints (brute force protection)
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 20, // max 20 attempts per window
  message: { detail: 'Too many authentication attempts. Please try again later.' },
  standardHeaders: true,
  legacyHeaders: false,
});

// Rate limiting for API endpoints
const apiLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 60, // 60 requests per minute
  message: { detail: 'Too many requests. Please slow down.' },
  standardHeaders: true,
  legacyHeaders: false,
});

// Request logging (skip health checks and socket.io polling to reduce noise)
app.use((req, res, next) => {
  if (req.url === '/health' || req.url.includes('/socket.io')) {
    return next();
  }
  logger.info({ method: req.method, url: req.url }, 'request');
  next();
});

// Auth middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  
  if (!token) {
    return res.status(401).json({ detail: 'Authentication required' });
  }

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      return res.status(403).json({ detail: 'Invalid or expired token' });
    }
    req.user = user;
    next();
  });
};

// Admin middleware
const requireAdmin = (req, res, next) => {
  if (req.user.role !== 'admin') {
    return res.status(403).json({ detail: 'Admin access required' });
  }
  next();
};

// Helper to create user response (without password)
const userResponse = (user) => {
  const { password_hash, ...safeUser } = user;
  return safeUser;
};

// Generate JWT token
const generateToken = (user) => {
  return jwt.sign(
    { id: user.id, email: user.email, role: user.role, name: user.name },
    JWT_SECRET,
    { expiresIn: JWT_EXPIRY }
  );
};

// ==================== API Routes ====================

// Health check — shallow (for ALB) and deep (for monitoring)
app.get('/health', async (req, res) => {
  const deep = req.query.deep === 'true';
  const health = {
    status: true,
    uptime: Math.floor(process.uptime()),
    timestamp: new Date().toISOString(),
  };
  if (deep) {
    const mem = process.memoryUsage();
    health.memory = {
      rss: Math.round(mem.rss / 1024 / 1024) + 'MB',
      heapUsed: Math.round(mem.heapUsed / 1024 / 1024) + 'MB',
      heapTotal: Math.round(mem.heapTotal / 1024 / 1024) + 'MB',
    };
    health.connections = {
      socketIO: io.engine?.clientsCount || 0,
      novaSonic: socketSessions.size,
    };
    health.data = {
      users: usersDB.count(),
      chats: chatsDB.count(),
    };
    health.env = NODE_ENV;
    health.anthropicConfigured = !!ANTHROPIC_API_KEY;
    
    // Check NUBLE ROKET API health
    health.nuble = await checkRoketHealth();
    health.nubleTools = getNubleTools().length;
  }
  res.json(health);
});

// Get app config/status (called on page load)
app.get('/api/v1/', (req, res) => {
  res.json({
    status: true,
    name: 'ROKET',
    version: '0.1.0',
    auth: true
  });
});

// Backend config (Open WebUI compatibility)
app.get('/api/config', (req, res) => {
  res.json({
    status: true,
    name: 'ROKET',
    version: '0.1.0',
    default_locale: 'en-US',
    oauth: {
      providers: {}
    },
    features: {
      enable_signup: config.signup_enabled,
      enable_login_form: true,
      enable_websocket: true,
      enable_community_sharing: false,
      enable_admin_export: true,
      enable_admin_chat_access: true,
      enable_direct_connections: true
    },
    ui: {
      default_locale: '',
      prompt_suggestions: [
        {
          title: ['Analyze my ', 'portfolio performance'],
          content: 'Can you analyze my current portfolio performance? Look at my positions, P&L, allocation, and give me actionable insights on what\'s working and what needs attention.'
        },
        {
          title: ['What\'s happening in ', 'the markets today?'],
          content: 'Give me a comprehensive market overview for today. Cover the S&P 500, Nasdaq, Bitcoin, and any major movers. What\'s driving price action and what should I watch?'
        },
        {
          title: ['Help me build a ', 'diversified portfolio'],
          content: 'I want to build a well-diversified investment portfolio. Walk me through asset allocation across stocks, bonds, crypto, and alternatives based on a moderate risk tolerance.'
        },
        {
          title: ['Explain the impact of ', 'Fed policy on my investments'],
          content: 'How is current Federal Reserve monetary policy affecting different asset classes? What should I be doing with my portfolio given the interest rate environment?'
        },
        {
          title: ['Find me ', 'tax optimization opportunities'],
          content: 'Review my portfolio for tax-loss harvesting opportunities and capital gains optimization strategies. How can I minimize my tax burden while maintaining my investment thesis?'
        },
        {
          title: ['Compare ', 'BTC vs ETH for long-term hold'],
          content: 'Give me a deep comparison of Bitcoin vs Ethereum as long-term investments. Cover fundamentals, on-chain metrics, institutional adoption, and which has better risk-adjusted return potential.'
        }
      ],
      pending_user_overlay_title: 'Account Pending',
      pending_user_overlay_content: 'Your account is pending approval.'
    },
    audio: {
      tts: {
        engine: '',
        voice: ''
      },
      stt: {
        engine: 'web'
      }
    },
    permissions: {
      chat: {
        deletion: true,
        editing: true,
        temporary: true
      },
      workspace: {
        models: true,
        knowledge: true,
        prompts: true,
        tools: true
      }
    }
  });
});

// Version endpoint
app.get('/api/version', (req, res) => {
  res.json({ version: '0.1.0' });
});

// Version updates endpoint
app.get('/api/version/updates', authenticateToken, (req, res) => {
  res.json({ available: false, current: '0.1.0', latest: '0.1.0' });
});

// Changelog endpoint
app.get('/api/changelog', (req, res) => {
  res.json([]);
});

// ==================== Auth Routes ====================

// Get current user session
app.get('/api/v1/auths/', authenticateToken, (req, res) => {
  const user = usersDB.findOne(u => u.id === req.user.id);
  if (!user) {
    return res.status(404).json({ detail: 'User not found' });
  }
  res.json(userResponse(user));
});

// Sign Up
app.post('/api/v1/auths/signup', authLimiter, async (req, res) => {
  try {
    const { name, email, password, profile_image_url } = req.body;

    if (!name || !email || !password) {
      return res.status(400).json({ detail: 'Name, email and password are required' });
    }

    // Check if signup is enabled (unless first user)
    if (usersDB.count() > 0 && !config.signup_enabled) {
      return res.status(403).json({ detail: 'Signup is currently disabled' });
    }

    // Check if user already exists
    if (usersDB.findOne(u => u.email.toLowerCase() === email.toLowerCase())) {
      return res.status(400).json({ detail: 'Email already registered' });
    }

    // Hash password
    const password_hash = await bcrypt.hash(password, 10);

    // First user becomes admin
    const isFirstUser = usersDB.count() === 0;
    const role = isFirstUser ? 'admin' : config.default_user_role;

    // Create user
    const user = {
      id: uuidv4(),
      name,
      email: email.toLowerCase(),
      password_hash,
      role,
      profile_image_url: profile_image_url || '/user.png',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      settings: {},
      api_key: null
    };

    usersDB.insert(user);

    // Generate token
    const token = generateToken(user);

    logger.info({ email, role }, 'User signed up');

    res.json({
      ...userResponse(user),
      token
    });
  } catch (error) {
    logger.error({ err: error }, 'Signup error');
    res.status(500).json({ detail: 'Internal server error' });
  }
});

// Sign In
app.post('/api/v1/auths/signin', authLimiter, async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ detail: 'Email and password are required' });
    }

    // Find user
    const user = usersDB.findOne(u => u.email.toLowerCase() === email.toLowerCase());
    if (!user) {
      return res.status(401).json({ detail: 'Invalid email or password' });
    }

    // Verify password
    const validPassword = await bcrypt.compare(password, user.password_hash);
    if (!validPassword) {
      return res.status(401).json({ detail: 'Invalid email or password' });
    }

    // Generate token
    const token = generateToken(user);

    logger.info({ email }, 'User signed in');

    res.json({
      ...userResponse(user),
      token
    });
  } catch (error) {
    logger.error({ err: error }, 'Signin error');
    res.status(500).json({ detail: 'Internal server error' });
  }
});

// Sign Out
app.get('/api/v1/auths/signout', (req, res) => {
  res.json({ success: true });
});

// Update profile
app.post('/api/v1/auths/update/profile', authenticateToken, async (req, res) => {
  try {
    const { name, profile_image_url } = req.body;
    
    const updates = { updated_at: new Date().toISOString() };
    if (name) updates.name = name;
    if (profile_image_url !== undefined) updates.profile_image_url = profile_image_url;

    const user = usersDB.update(u => u.id === req.user.id, updates);
    if (!user) {
      return res.status(404).json({ detail: 'User not found' });
    }

    res.json(userResponse(user));
  } catch (error) {
    logger.error({ err: error }, 'Update profile error');
    res.status(500).json({ detail: 'Internal server error' });
  }
});

// Update password
app.post('/api/v1/auths/update/password', authenticateToken, async (req, res) => {
  try {
    const { password, new_password } = req.body;

    const user = usersDB.findOne(u => u.id === req.user.id);
    if (!user) {
      return res.status(404).json({ detail: 'User not found' });
    }

    // Verify current password
    const validPassword = await bcrypt.compare(password, user.password_hash);
    if (!validPassword) {
      return res.status(401).json({ detail: 'Current password is incorrect' });
    }

    // Hash new password
    const password_hash = await bcrypt.hash(new_password, 10);
    
    usersDB.update(u => u.id === req.user.id, {
      password_hash,
      updated_at: new Date().toISOString()
    });

    res.json({ success: true });
  } catch (error) {
    logger.error({ err: error }, 'Update password error');
    res.status(500).json({ detail: 'Internal server error' });
  }
});

// Update timezone
app.post('/api/v1/auths/update/timezone', authenticateToken, (req, res) => {
  const { timezone } = req.body;
  usersDB.update(u => u.id === req.user.id, { timezone, updated_at: new Date().toISOString() });
  res.json({ success: true });
});

// ==================== Admin Auth Config ====================

// Get admin config
app.get('/api/v1/auths/admin/config', authenticateToken, requireAdmin, (req, res) => {
  res.json({
    SHOW_ADMIN_DETAILS: true,
    ENABLE_SIGNUP: config.signup_enabled,
    DEFAULT_USER_ROLE: config.default_user_role,
    JWT_EXPIRY: config.jwt_expiry
  });
});

// Update admin config
app.post('/api/v1/auths/admin/config', authenticateToken, requireAdmin, (req, res) => {
  const { ENABLE_SIGNUP, DEFAULT_USER_ROLE, JWT_EXPIRY } = req.body;
  
  if (ENABLE_SIGNUP !== undefined) config.signup_enabled = ENABLE_SIGNUP;
  if (DEFAULT_USER_ROLE) config.default_user_role = DEFAULT_USER_ROLE;
  if (JWT_EXPIRY) config.jwt_expiry = JWT_EXPIRY;
  
  configDB.update(c => c.type === 'app_config', config);
  
  res.json({
    ENABLE_SIGNUP: config.signup_enabled,
    DEFAULT_USER_ROLE: config.default_user_role,
    JWT_EXPIRY: config.jwt_expiry
  });
});

// Get admin details
app.get('/api/v1/auths/admin/details', authenticateToken, (req, res) => {
  res.json({ show: true });
});

// Signup enabled status
app.get('/api/v1/auths/signup/enabled', (req, res) => {
  res.json(config.signup_enabled);
});

// Toggle signup
app.get('/api/v1/auths/signup/enabled/toggle', authenticateToken, requireAdmin, (req, res) => {
  config.signup_enabled = !config.signup_enabled;
  configDB.update(c => c.type === 'app_config', config);
  res.json(config.signup_enabled);
});

// Get default user role
app.get('/api/v1/auths/signup/user/role', authenticateToken, requireAdmin, (req, res) => {
  res.json(config.default_user_role);
});

// Set default user role
app.post('/api/v1/auths/signup/user/role', authenticateToken, requireAdmin, (req, res) => {
  const { role } = req.body;
  config.default_user_role = role;
  configDB.update(c => c.type === 'app_config', config);
  res.json(config.default_user_role);
});

// Token expiry config
app.get('/api/v1/auths/token/expires', authenticateToken, requireAdmin, (req, res) => {
  res.json({ duration: config.jwt_expiry });
});

app.post('/api/v1/auths/token/expires/update', authenticateToken, requireAdmin, (req, res) => {
  const { duration } = req.body;
  config.jwt_expiry = duration;
  configDB.update(c => c.type === 'app_config', config);
  res.json({ duration: config.jwt_expiry });
});

// LDAP placeholders (return disabled)
app.get('/api/v1/auths/admin/config/ldap', authenticateToken, requireAdmin, (req, res) => {
  res.json({ enable_ldap: false });
});

app.post('/api/v1/auths/admin/config/ldap', authenticateToken, requireAdmin, (req, res) => {
  res.json({ enable_ldap: false });
});

app.get('/api/v1/auths/admin/config/ldap/server', authenticateToken, requireAdmin, (req, res) => {
  res.json({});
});

app.post('/api/v1/auths/admin/config/ldap/server', authenticateToken, requireAdmin, (req, res) => {
  res.json({});
});

// ==================== Users API ====================

// User groups (must be before :id routes)
app.get('/api/v1/users/groups', authenticateToken, (req, res) => {
  res.json([]);
});

// Default permissions
app.get('/api/v1/users/default/permissions', authenticateToken, (req, res) => {
  res.json({
    workspace: { models: true, knowledge: true, prompts: true, tools: true }
  });
});

app.post('/api/v1/users/default/permissions', authenticateToken, requireAdmin, (req, res) => {
  res.json(req.body || {});
});

// Search users
app.get('/api/v1/users/search', authenticateToken, (req, res) => {
  const query = (req.query.q || req.query.query || '').toLowerCase();
  const users = usersDB.findAll(u => 
    u.name.toLowerCase().includes(query) || u.email.toLowerCase().includes(query)
  ).map(userResponse);
  res.json({ items: users, total: users.length });
});

// All users (for admin)
app.get('/api/v1/users/all', authenticateToken, requireAdmin, (req, res) => {
  const users = usersDB.findAll().map(userResponse);
  res.json(users);
});

// Update role
app.post('/api/v1/users/update/role', authenticateToken, requireAdmin, (req, res) => {
  const { id, role } = req.body;
  const user = usersDB.update(u => u.id === id, { role, updated_at: new Date().toISOString() });
  if (!user) return res.status(404).json({ detail: 'User not found' });
  res.json(userResponse(user));
});

// User status update
app.post('/api/v1/users/user/status/update', authenticateToken, (req, res) => {
  const { status } = req.body;
  usersDB.update(u => u.id === req.user.id, { status: status || null, updated_at: new Date().toISOString() });
  res.json({ success: true });
});

// User info
app.get('/api/v1/users/user/info', authenticateToken, (req, res) => {
  const user = usersDB.findOne(u => u.id === req.user.id);
  res.json({ info: user?.info || {} });
});

app.post('/api/v1/users/user/info/update', authenticateToken, (req, res) => {
  const { info } = req.body;
  usersDB.update(u => u.id === req.user.id, { info: info || {}, updated_at: new Date().toISOString() });
  res.json({ info: info || {} });
});

// List all users (admin only)
app.get('/api/v1/users/', authenticateToken, requireAdmin, (req, res) => {
  const users = usersDB.findAll().map(userResponse);
  res.json(users);
});

// Get user by ID
app.get('/api/v1/users/:id', authenticateToken, (req, res) => {
  const user = usersDB.findOne(u => u.id === req.params.id);
  if (!user) {
    return res.status(404).json({ detail: 'User not found' });
  }
  res.json(userResponse(user));
});

// Update user (admin or self)
app.post('/api/v1/users/:id/update', authenticateToken, (req, res) => {
  if (req.user.role !== 'admin' && req.user.id !== req.params.id) {
    return res.status(403).json({ detail: 'Access denied' });
  }

  const { name, email, role, profile_image_url } = req.body;
  const updates = { updated_at: new Date().toISOString() };
  
  if (name) updates.name = name;
  if (email) updates.email = email.toLowerCase();
  if (role && req.user.role === 'admin') updates.role = role;
  if (profile_image_url !== undefined) updates.profile_image_url = profile_image_url;

  const user = usersDB.update(u => u.id === req.params.id, updates);
  if (!user) {
    return res.status(404).json({ detail: 'User not found' });
  }

  res.json(userResponse(user));
});

// Delete user (admin only)
app.delete('/api/v1/users/:id', authenticateToken, requireAdmin, (req, res) => {
  if (req.user.id === req.params.id) {
    return res.status(400).json({ detail: 'Cannot delete yourself' });
  }

  const user = usersDB.delete(u => u.id === req.params.id);
  if (!user) {
    return res.status(404).json({ detail: 'User not found' });
  }

  // Delete user's chats
  const userChats = chatsDB.findAll(c => c.user_id === req.params.id);
  userChats.forEach(chat => chatsDB.delete(c => c.id === chat.id));

  res.json({ success: true });
});

// User profile image (placeholder)
app.get('/api/v1/users/:id/profile/image', (req, res) => {
  // Return a simple SVG avatar
  const user = usersDB.findOne(u => u.id === req.params.id);
  const name = user ? user.name[0].toUpperCase() : '?';
  
  res.setHeader('Content-Type', 'image/svg+xml');
  res.send(`
    <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
      <rect width="100" height="100" fill="#6366f1"/>
      <text x="50" y="50" font-family="Arial" font-size="40" fill="white" text-anchor="middle" dominant-baseline="central">${name}</text>
    </svg>
  `);
});

// User groups by ID
app.get('/api/v1/users/:userId/groups', authenticateToken, (req, res) => {
  res.json([]);
});

// User active status
app.get('/api/v1/users/:userId/active', authenticateToken, (req, res) => {
  const user = usersDB.findOne(u => u.id === req.params.userId);
  res.json({ active: !!user });
});

// ==================== Chats API ====================

// List user's chats (with pagination)
app.get('/api/v1/chats/', authenticateToken, (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 50;
  const skip = (page - 1) * limit;
  
  const allChats = chatsDB.findAll(c => c.user_id === req.user.id)
    .sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));
  
  const paginatedChats = allChats.slice(skip, skip + limit);
  res.json(paginatedChats);
});

// Get pinned chats (must be before :id route)
app.get('/api/v1/chats/pinned', authenticateToken, (req, res) => {
  const chats = chatsDB.findAll(c => c.user_id === req.user.id && c.pinned === true)
    .sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));
  res.json(chats);
});

// Get all tags from user's chats (must be before :id route)
app.get('/api/v1/chats/all/tags', authenticateToken, (req, res) => {
  const chats = chatsDB.findAll(c => c.user_id === req.user.id);
  const allTags = new Set();
  chats.forEach(chat => {
    if (chat.tags && Array.isArray(chat.tags)) {
      chat.tags.forEach(tag => allTags.add(tag));
    }
  });
  res.json(Array.from(allTags));
});

// Search chats by text (must be before :id route)
app.get('/api/v1/chats/search', authenticateToken, (req, res) => {
  const searchText = (req.query.text || '').toLowerCase().trim();
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 50;
  const skip = (page - 1) * limit;

  if (!searchText) {
    return res.json([]);
  }

  const userChats = chatsDB.findAll(c => c.user_id === req.user.id);

  const matchingChats = userChats.filter(chat => {
    // Search in title
    if (chat.title && chat.title.toLowerCase().includes(searchText)) {
      return true;
    }
    // Search in tags
    if (chat.tags && Array.isArray(chat.tags)) {
      if (chat.tags.some(tag => tag.toLowerCase().includes(searchText))) {
        return true;
      }
    }
    // Search in message content
    if (chat.messages && Array.isArray(chat.messages)) {
      for (const msg of chat.messages) {
        const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content || '');
        if (content.toLowerCase().includes(searchText)) {
          return true;
        }
      }
    }
    // Also search in chat.chat.messages (Open WebUI format)
    if (chat.chat && chat.chat.messages) {
      const chatMessages = Object.values(chat.chat.messages || {});
      for (const msg of chatMessages) {
        const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content || '');
        if (content.toLowerCase().includes(searchText)) {
          return true;
        }
      }
    }
    return false;
  });

  const sorted = matchingChats.sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));
  const paginated = sorted.slice(skip, skip + limit);
  res.json(paginated);
});

// Get chat by ID
app.get('/api/v1/chats/:id', authenticateToken, (req, res) => {
  const chat = chatsDB.findOne(c => c.id === req.params.id && c.user_id === req.user.id);
  if (!chat) {
    return res.status(404).json({ detail: 'Chat not found' });
  }
  res.json(chat);
});

// Create chat
app.post('/api/v1/chats/new', authenticateToken, (req, res) => {
  const { chat } = req.body;
  
  const newChat = {
    id: uuidv4(),
    user_id: req.user.id,
    title: chat?.title || 'New Chat',
    messages: chat?.messages || [],
    models: chat?.models || [],
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  };

  chatsDB.insert(newChat);
  res.json(newChat);
});

// Update chat
app.post('/api/v1/chats/:id', authenticateToken, (req, res) => {
  const { chat } = req.body;
  
  const existing = chatsDB.findOne(c => c.id === req.params.id && c.user_id === req.user.id);
  if (!existing) {
    return res.status(404).json({ detail: 'Chat not found' });
  }

  const updates = {
    ...chat,
    updated_at: new Date().toISOString()
  };

  const updated = chatsDB.update(c => c.id === req.params.id, updates);
  res.json(updated);
});

// Delete chat
app.delete('/api/v1/chats/:id', authenticateToken, (req, res) => {
  const chat = chatsDB.findOne(c => c.id === req.params.id && c.user_id === req.user.id);
  if (!chat) {
    return res.status(404).json({ detail: 'Chat not found' });
  }

  chatsDB.delete(c => c.id === req.params.id);
  res.json({ success: true });
});

// Archive/Delete all chats
app.delete('/api/v1/chats/', authenticateToken, (req, res) => {
  const userChats = chatsDB.findAll(c => c.user_id === req.user.id);
  userChats.forEach(chat => chatsDB.delete(c => c.id === chat.id));
  res.json({ success: true });
});

// Get tags for a specific chat
app.get('/api/v1/chats/:id/tags', authenticateToken, (req, res) => {
  const chat = chatsDB.findOne(c => c.id === req.params.id && c.user_id === req.user.id);
  if (!chat) {
    return res.status(404).json({ detail: 'Chat not found' });
  }
  res.json(chat.tags || []);
});

// Update tags for a specific chat
app.post('/api/v1/chats/:id/tags', authenticateToken, (req, res) => {
  const { tags } = req.body;
  const chat = chatsDB.findOne(c => c.id === req.params.id && c.user_id === req.user.id);
  if (!chat) {
    return res.status(404).json({ detail: 'Chat not found' });
  }
  chatsDB.update(c => c.id === req.params.id, { tags: tags || [], updated_at: new Date().toISOString() });
  res.json(tags || []);
});

// ==================== Models API ====================

// Shared model-fetching logic with in-memory cache
let cachedModels = null;
let modelsCacheTime = 0;
const MODELS_CACHE_TTL = 5 * 60 * 1000; // Cache for 5 minutes

// Hardcoded fallback models — always available even if Anthropic API is down or key missing
const FALLBACK_MODELS = [
  { id: 'claude-opus-4-6', display_name: 'Claude Opus 4.6', created_at: '2025-12-01T00:00:00Z' },
  { id: 'claude-opus-4-5-20251101', display_name: 'Claude Opus 4.5', created_at: '2025-11-01T00:00:00Z' },
  { id: 'claude-opus-4-1-20250805', display_name: 'Claude Opus 4.1', created_at: '2025-08-05T00:00:00Z' },
  { id: 'claude-opus-4-20250514', display_name: 'Claude Opus 4', created_at: '2025-05-14T00:00:00Z' },
  { id: 'claude-sonnet-4-5-20250929', display_name: 'Claude Sonnet 4.5', created_at: '2025-09-29T00:00:00Z' },
  { id: 'claude-sonnet-4-20250514', display_name: 'Claude Sonnet 4', created_at: '2025-05-14T00:00:00Z' },
  { id: 'claude-haiku-4-5-20251001', display_name: 'Claude Haiku 4.5', created_at: '2025-10-01T00:00:00Z' },
  { id: 'claude-3-5-haiku-20241022', display_name: 'Claude 3.5 Haiku', created_at: '2024-10-22T00:00:00Z' },
];

function formatModel(m) {
  const id = m.id.startsWith('anthropic.') ? m.id : `anthropic.${m.id}`;
  const name = m.display_name || m.name || m.id;
  return {
    id,
    name,
    object: 'model',
    created: new Date(m.created_at || '2025-01-01').getTime() / 1000,
    owned_by: 'anthropic',
    info: {
      id,
      name,
      meta: {
        profile_image_url: '/static/model-logo.png',
        description: `Anthropic ${name}`,
        capabilities: { vision: true }
      }
    },
    urlIdx: 0
  };
}

async function fetchAnthropicModels() {
  const now = Date.now();
  if (cachedModels && (now - modelsCacheTime) < MODELS_CACHE_TTL) {
    return cachedModels;
  }
  
  try {
    if (!ANTHROPIC_API_KEY) {
      throw new Error('ANTHROPIC_API_KEY not configured');
    }
    const response = await fetch('https://api.anthropic.com/v1/models', {
      headers: {
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      }
    });
    const data = await response.json();
    const apiModels = (data.data || []).map(m => formatModel(m));
    
    // Merge with fallbacks to ensure key models like Opus 4 always appear
    const apiIds = new Set(apiModels.map(m => m.id));
    const fallbackModels = FALLBACK_MODELS
      .filter(fb => !apiIds.has(`anthropic.${fb.id}`))
      .map(fb => formatModel(fb));
    
    cachedModels = [...apiModels, ...fallbackModels];
    modelsCacheTime = now;
    return cachedModels;
  } catch (e) {
    logger.error({ err: e }, 'Failed to fetch Anthropic models — using fallback list');
    // Always return fallback models so the dropdown is never empty
    cachedModels = FALLBACK_MODELS.map(fb => formatModel(fb));
    modelsCacheTime = now;
    return cachedModels;
  }
}

app.get('/api/v1/models/', authenticateToken, async (req, res) => {
  const models = await fetchAnthropicModels();
  res.json(models);
});

// Models list (paginated - workspace admin page)
app.get('/api/v1/models/list', authenticateToken, async (req, res) => {
  const models = await fetchAnthropicModels();
  res.json({ items: models, total: models.length });
});

// Model tags
app.get('/api/v1/models/tags', authenticateToken, (req, res) => {
  res.json([]);
});

// Model CRUD stubs (workspace admin)
app.post('/api/v1/models/create', authenticateToken, (req, res) => {
  res.status(501).json({ detail: 'Model creation not supported — ROKET uses a fixed model configuration' });
});
app.get('/api/v1/models/model', authenticateToken, async (req, res) => {
  const models = await fetchAnthropicModels();
  res.json(models[0] || null);
});
app.post('/api/v1/models/update', authenticateToken, (req, res) => {
  res.status(501).json({ detail: 'Model update not supported' });
});
app.delete('/api/v1/models/delete', authenticateToken, (req, res) => {
  res.status(501).json({ detail: 'Model deletion not supported' });
});
app.post('/api/v1/models/import', authenticateToken, (req, res) => {
  res.status(501).json({ detail: 'Model import not supported' });
});

// Model profile image - serve theme-aware ROKET logo
// Light mode: favicon.png (black R), Dark mode: favicon-dark.png / model-logo.png (white R)
app.get('/api/v1/models/model/profile/image', (req, res) => {
  const theme = req.query.theme || 'dark';
  const isLight = theme === 'light';

  // Light mode → black R (favicon.png), Dark mode → white R (favicon-dark.png / model-logo.png)
  const primaryFile = isLight ? 'favicon.png' : 'favicon-dark.png';
  const fallbackFile = isLight ? 'favicon-dark.png' : 'model-logo.png';

  const primaryPath = path.join(STATIC_DIR, primaryFile);
  const fallbackPath = path.join(STATIC_DIR, fallbackFile);
  const modelLogoPath = path.join(STATIC_DIR, 'model-logo.png');

  if (fs.existsSync(primaryPath)) {
    res.sendFile(primaryPath);
  } else if (fs.existsSync(fallbackPath)) {
    res.sendFile(fallbackPath);
  } else if (fs.existsSync(modelLogoPath)) {
    res.sendFile(modelLogoPath);
  } else {
    // Fallback SVG with ROKET R
    res.setHeader('Content-Type', 'image/svg+xml');
    const fill = isLight ? '#0f172a' : '#ffffff';
    res.send(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text x="50" y="70" font-family="Arial" font-size="65" font-weight="bold" fill="${fill}" text-anchor="middle">R</text></svg>`);
  }
});

// Models API (alternative path)
app.get('/api/models', authenticateToken, async (req, res) => {
  const models = await fetchAnthropicModels();
  res.json({ data: models });
});

// ==================== Config API ====================

app.get('/api/v1/configs/', authenticateToken, (req, res) => {
  res.json({});
});

// Banners config
app.get('/api/v1/configs/banners', authenticateToken, (req, res) => {
  res.json([]);
});

// Models config
app.get('/api/v1/configs/models', authenticateToken, (req, res) => {
  res.json({});
});

app.post('/api/v1/configs/models', authenticateToken, (req, res) => {
  res.json({});
});

// Connections config
app.get('/api/v1/configs/connections', authenticateToken, (req, res) => {
  res.json({
    OPENAI_API_BASE_URLS: [],
    OPENAI_API_KEYS: [],
    OPENAI_API_CONFIGS: {}
  });
});

app.post('/api/v1/configs/connections', authenticateToken, (req, res) => {
  res.json({
    OPENAI_API_BASE_URLS: [],
    OPENAI_API_KEYS: [],
    OPENAI_API_CONFIGS: {}
  });
});

// Tool servers config
app.get('/api/v1/configs/tool_servers', authenticateToken, (req, res) => {
  res.json([]);
});

app.post('/api/v1/configs/tool_servers', authenticateToken, (req, res) => {
  res.json([]);
});

// Code execution config
app.get('/api/v1/configs/code_execution', authenticateToken, (req, res) => {
  res.json({ enabled: false });
});

app.post('/api/v1/configs/code_execution', authenticateToken, (req, res) => {
  res.json({ enabled: false });
});

// Suggestions config
app.get('/api/v1/configs/suggestions', authenticateToken, (req, res) => {
  res.json([]);
});

// ==================== User Settings API ====================

// Get user settings
app.get('/api/v1/users/user/settings', authenticateToken, (req, res) => {
  const user = usersDB.findOne(u => u.id === req.user.id);
  if (!user) {
    return res.status(404).json({ detail: 'User not found' });
  }
  res.json({ ui: user.settings || {} });
});

// Update user settings
app.post('/api/v1/users/user/settings', authenticateToken, (req, res) => {
  const { ui } = req.body;
  usersDB.update(u => u.id === req.user.id, { 
    settings: ui || {},
    updated_at: new Date().toISOString()
  });
  res.json({ ui: ui || {} });
});

// Update user settings (alternative path used by frontend)
app.post('/api/v1/users/user/settings/update', authenticateToken, (req, res) => {
  const settings = req.body;
  usersDB.update(u => u.id === req.user.id, { 
    settings: settings || {},
    updated_at: new Date().toISOString()
  });
  res.json(settings || {});
});

// ==================== Tools API ====================

app.get('/api/v1/tools/', authenticateToken, (req, res) => {
  res.json([]);
});

app.get('/api/v1/tools/list', authenticateToken, (req, res) => {
  res.json({ items: [], total: 0 });
});

app.post('/api/v1/tools/create', authenticateToken, (req, res) => {
  res.status(501).json({ detail: 'Tools are not supported in ROKET' });
});

// ==================== Functions API ====================

app.get('/api/v1/functions/', authenticateToken, (req, res) => {
  res.json([]);
});

// ==================== Groups API ====================

app.get('/api/v1/groups/', authenticateToken, (req, res) => {
  res.json([]);
});

app.post('/api/v1/groups/create', authenticateToken, (req, res) => {
  res.status(501).json({ detail: 'Groups are not supported in ROKET' });
});

app.get('/api/v1/groups/:id', authenticateToken, (req, res) => {
  res.status(404).json({ detail: 'Group not found' });
});

// ==================== Knowledge API ====================

app.get('/api/v1/knowledge/', authenticateToken, (req, res) => {
  res.json([]);
});

app.get('/api/v1/knowledge/search', authenticateToken, (req, res) => {
  res.json({ items: [], total: 0 });
});

app.get('/api/v1/knowledge/search/files', authenticateToken, (req, res) => {
  res.json({ items: [], total: 0 });
});

app.post('/api/v1/knowledge/create', authenticateToken, (req, res) => {
  res.status(501).json({ detail: 'Knowledge bases are not yet supported in ROKET' });
});

app.get('/api/v1/knowledge/:id', authenticateToken, (req, res) => {
  res.status(404).json({ detail: 'Knowledge base not found' });
});

// ==================== Prompts API ====================

app.get('/api/v1/prompts/', authenticateToken, (req, res) => {
  res.json([]);
});

app.get('/api/v1/prompts/list', authenticateToken, (req, res) => {
  res.json([]);
});

app.post('/api/v1/prompts/create', authenticateToken, (req, res) => {
  res.status(501).json({ detail: 'Prompt templates are not yet supported in ROKET' });
});

app.get('/api/v1/prompts/command/:command', authenticateToken, (req, res) => {
  res.status(404).json({ detail: 'Prompt not found' });
});

// ==================== Memories API ====================

// List all memories for current user
app.get('/api/v1/memories/', authenticateToken, (req, res) => {
  const memories = memoriesDB.findAll(m => m.user_id === req.user.id)
    .sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));
  res.json(memories);
});

// Add a new memory
app.post('/api/v1/memories/add', authenticateToken, (req, res) => {
  const { content } = req.body;
  if (!content || !content.trim()) {
    return res.status(400).json({ detail: 'Memory content is required' });
  }

  const memory = {
    id: uuidv4(),
    user_id: req.user.id,
    content: content.trim(),
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  };

  memoriesDB.insert(memory);
  res.json(memory);
});

// Query memories (simple keyword matching)
app.post('/api/v1/memories/query', authenticateToken, (req, res) => {
  const { content } = req.body;
  if (!content) {
    return res.json([]);
  }

  const queryWords = content.toLowerCase().split(/\s+/).filter(w => w.length > 2);
  const userMemories = memoriesDB.findAll(m => m.user_id === req.user.id);

  // Score memories by keyword overlap
  const scored = userMemories.map(memory => {
    const memWords = memory.content.toLowerCase();
    let score = 0;
    for (const word of queryWords) {
      if (memWords.includes(word)) score++;
    }
    return { ...memory, score };
  }).filter(m => m.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);

  res.json(scored);
});

// Update a memory
app.post('/api/v1/memories/:id/update', authenticateToken, (req, res) => {
  const { content } = req.body;
  const memory = memoriesDB.findOne(m => m.id === req.params.id && m.user_id === req.user.id);
  if (!memory) {
    return res.status(404).json({ detail: 'Memory not found' });
  }

  const updated = memoriesDB.update(
    m => m.id === req.params.id,
    { content: content || memory.content, updated_at: new Date().toISOString() }
  );
  res.json(updated);
});

// Delete all memories for current user (must be before :id route)
app.delete('/api/v1/memories/delete/user', authenticateToken, (req, res) => {
  const userMemories = memoriesDB.findAll(m => m.user_id === req.user.id);
  userMemories.forEach(memory => memoriesDB.delete(m => m.id === memory.id));
  res.json({ success: true });
});

// Delete a specific memory
app.delete('/api/v1/memories/:id', authenticateToken, (req, res) => {
  const memory = memoriesDB.findOne(m => m.id === req.params.id && m.user_id === req.user.id);
  if (!memory) {
    return res.status(404).json({ detail: 'Memory not found' });
  }
  memoriesDB.delete(m => m.id === req.params.id);
  res.json({ success: true });
});

// ==================== Files API ====================

// List all files for current user
app.get('/api/v1/files/', authenticateToken, (req, res) => {
  const files = filesDB.findAll(f => f.user_id === req.user.id)
    .sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
  res.json(files);
});

// Helper: Extract text content from a file based on its extension/mimetype
async function extractFileContent(filePath, originalname, mimetype) {
  const ext = path.extname(originalname).toLowerCase();
  
  // All extensions we can read directly as UTF-8 text
  const textExtensions = [
    '.txt', '.csv', '.tsv', '.md', '.markdown', '.json', '.jsonl',
    '.js', '.jsx', '.ts', '.tsx', '.py', '.pyw', '.rb', '.java', '.c', '.cpp', '.h', '.hpp',
    '.cs', '.go', '.rs', '.swift', '.kt', '.scala', '.php', '.pl', '.pm', '.r',
    '.html', '.htm', '.css', '.scss', '.sass', '.less',
    '.xml', '.xsl', '.xslt', '.svg',
    '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.env',
    '.log', '.sql', '.sh', '.bash', '.zsh', '.fish', '.bat', '.ps1', '.psm1',
    '.lua', '.vim', '.tex', '.latex', '.bib',
    '.gitignore', '.dockerignore', '.editorconfig',
    '.makefile', '.cmake',
  ];

  try {
    // ── Plain text / code files ──
    if (textExtensions.includes(ext) || (mimetype && mimetype.startsWith('text/'))) {
      return fs.readFileSync(filePath, 'utf8');
    }

    // ── Excel files (.xlsx, .xls, .xlsm) — rich markdown table extraction ──
    if (['.xlsx', '.xls', '.xlsm'].includes(ext) ||
        mimetype === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' ||
        mimetype === 'application/vnd.ms-excel') {
      const workbook = XLSX.readFile(filePath);
      const sheets = [];
      for (const sheetName of workbook.SheetNames) {
        const sheet = workbook.Sheets[sheetName];
        const jsonData = XLSX.utils.sheet_to_json(sheet, { header: 1, defval: '' });
        
        let sheetText = `## Sheet: "${sheetName}"\n\n`;
        if (jsonData.length > 0) {
          // Format as markdown table for readability
          const headers = jsonData[0].map(h => String(h || '').trim());
          sheetText += '| ' + headers.join(' | ') + ' |\n';
          sheetText += '| ' + headers.map(() => '---').join(' | ') + ' |\n';
          for (let i = 1; i < jsonData.length; i++) {
            const row = jsonData[i].map(c => String(c ?? '').trim());
            sheetText += '| ' + row.join(' | ') + ' |\n';
          }
        }
        sheets.push(sheetText);
      }
      return sheets.join('\n\n');
    }

    // ── PDF files ──
    if (ext === '.pdf' || mimetype === 'application/pdf') {
      try {
        const dataBuffer = fs.readFileSync(filePath);
        const pdfData = await pdfParse(dataBuffer);
        if (pdfData.text && pdfData.text.trim().length > 0) {
          return pdfData.text;
        }
      } catch (pdfErr) {
        logger.warn({ err: pdfErr }, 'pdf-parse failed, trying officeparser fallback');
      }
      // Fallback to officeparser
      try {
        const text = await parseOffice(filePath);
        return text || '';
      } catch (e) {
        logger.warn({ err: e }, 'officeparser PDF fallback also failed');
        return '';
      }
    }

    // ── Word documents (.docx) — mammoth gives clean text ──
    if (ext === '.docx' ||
        mimetype === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
      try {
        const result = await mammoth.extractRawText({ path: filePath });
        if (result.value && result.value.trim().length > 0) {
          return result.value;
        }
      } catch (mamErr) {
        logger.warn({ err: mamErr }, 'mammoth failed, trying officeparser fallback');
      }
      // Fallback
      try {
        const text = await parseOffice(filePath);
        return text || '';
      } catch (e) {
        return '';
      }
    }

    // ── Legacy Word (.doc) — officeparser handles this ──
    if (ext === '.doc' || mimetype === 'application/msword') {
      try {
        const text = await parseOffice(filePath);
        return text || '';
      } catch (e) {
        logger.warn({ err: e }, 'Could not parse .doc file');
        return '';
      }
    }

    // ── PowerPoint (.pptx, .ppt) ──
    if (['.pptx', '.ppt'].includes(ext) ||
        mimetype === 'application/vnd.openxmlformats-officedocument.presentationml.presentation' ||
        mimetype === 'application/vnd.ms-powerpoint') {
      try {
        const text = await parseOffice(filePath);
        return text || '';
      } catch (e) {
        logger.warn({ err: e }, 'Could not parse PowerPoint file');
        return '';
      }
    }

    // ── OpenDocument formats (.odt, .ods, .odp) ──
    if (['.odt', '.ods', '.odp'].includes(ext) ||
        (mimetype && mimetype.includes('opendocument'))) {
      try {
        const text = await parseOffice(filePath);
        return text || '';
      } catch (e) {
        logger.warn({ err: e }, 'Could not parse OpenDocument file');
        return '';
      }
    }

    // ── RTF files ──
    if (ext === '.rtf' || mimetype === 'application/rtf' || mimetype === 'text/rtf') {
      try {
        const rtfContent = fs.readFileSync(filePath, 'utf8');
        // Simple RTF stripping: remove RTF control words and braces
        const stripped = rtfContent
          .replace(/\\[a-z]+[-]?\d*\s?/g, '')  // remove control words
          .replace(/[{}]/g, '')                  // remove braces
          .replace(/\\\n/g, '\n')               // unescape newlines
          .replace(/\\'[0-9a-f]{2}/gi, '')     // remove hex escapes
          .trim();
        if (stripped.length > 0) return stripped;
      } catch (e) {
        logger.warn({ err: e }, 'RTF parsing failed');
      }
      return '';
    }

    // ── Fallback: try reading as UTF-8 text ──
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      // Check if it looks like valid text (no null bytes = not binary)
      if (content && !content.includes('\0')) {
        return content;
      }
    } catch (e) {
      // Not readable as text — that's OK
    }

    // ── Last resort: officeparser for any other Office-like format ──
    try {
      const text = await parseOffice(filePath);
      if (text && text.trim().length > 0) return text;
    } catch (e) {
      // officeparser couldn't handle it either
    }

    return '';
  } catch (e) {
    logger.warn({ err: e, ext, mimetype }, 'Could not extract file content');
    return '';
  }
}

// Upload a file
app.post('/api/v1/files/', authenticateToken, upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ detail: 'No file uploaded' });
    }

    // Extract text content from the uploaded file
    const textContent = await extractFileContent(req.file.path, req.file.originalname, req.file.mimetype);
    
    logger.info({ 
      filename: req.file.originalname, 
      mimetype: req.file.mimetype, 
      size: req.file.size,
      contentLength: textContent.length 
    }, 'File uploaded and content extracted');

    const fileRecord = {
      id: uuidv4(),
      user_id: req.user.id,
      filename: req.file.originalname,
      meta: {
        name: req.file.originalname,
        content_type: req.file.mimetype,
        size: req.file.size,
      },
      data: {
        content: textContent,
      },
      path: req.file.path,
      hash: '',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };

    filesDB.insert(fileRecord);

    res.json(fileRecord);
  } catch (error) {
    logger.error({ err: error }, 'File upload error');
    res.status(500).json({ detail: error.message || 'File upload failed' });
  }
});

// Get file by ID
app.get('/api/v1/files/:id', authenticateToken, (req, res) => {
  const file = filesDB.findOne(f => f.id === req.params.id && f.user_id === req.user.id);
  if (!file) {
    return res.status(404).json({ detail: 'File not found' });
  }
  res.json(file);
});

// Get file content by ID
app.get('/api/v1/files/:id/content', authenticateToken, (req, res) => {
  const file = filesDB.findOne(f => f.id === req.params.id);
  if (!file) {
    return res.status(404).json({ detail: 'File not found' });
  }
  if (file.path && fs.existsSync(file.path)) {
    res.sendFile(file.path);
  } else {
    res.status(404).json({ detail: 'File content not found on disk' });
  }
});

// Update file data content
app.post('/api/v1/files/:id/data/content/update', authenticateToken, (req, res) => {
  const { content } = req.body;
  const file = filesDB.findOne(f => f.id === req.params.id && f.user_id === req.user.id);
  if (!file) {
    return res.status(404).json({ detail: 'File not found' });
  }
  const updated = filesDB.update(
    f => f.id === req.params.id,
    { data: { ...file.data, content }, updated_at: new Date().toISOString() }
  );
  res.json(updated);
});

// Get file processing status (SSE or JSON)
app.get('/api/v1/files/:id/process/status', authenticateToken, (req, res) => {
  const file = filesDB.findOne(f => f.id === req.params.id);
  if (!file) {
    return res.status(404).json({ detail: 'File not found' });
  }

  if (req.query.stream === 'true') {
    // SSE stream — files are processed immediately in our simple setup
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.write(`data: ${JSON.stringify({ status: 'completed', id: file.id, filename: file.filename })}\n\n`);
    res.end();
  } else {
    res.json({ status: 'completed', id: file.id, filename: file.filename });
  }
});

// Delete all files for current user (must be before :id route)
app.delete('/api/v1/files/all', authenticateToken, (req, res) => {
  const userFiles = filesDB.findAll(f => f.user_id === req.user.id);
  userFiles.forEach(file => {
    if (file.path && fs.existsSync(file.path)) {
      try { fs.unlinkSync(file.path); } catch (e) { /* ignore */ }
    }
    filesDB.delete(f => f.id === file.id);
  });
  res.json({ success: true });
});

// Delete file by ID
app.delete('/api/v1/files/:id', authenticateToken, (req, res) => {
  const file = filesDB.findOne(f => f.id === req.params.id && f.user_id === req.user.id);
  if (!file) {
    return res.status(404).json({ detail: 'File not found' });
  }
  // Delete from disk
  if (file.path && fs.existsSync(file.path)) {
    try { fs.unlinkSync(file.path); } catch (e) { /* ignore */ }
  }
  filesDB.delete(f => f.id === req.params.id);
  res.json({ success: true });
});

// ==================== Folders API ====================

// List all folders for current user
app.get('/api/v1/folders/', authenticateToken, (req, res) => {
  const folders = foldersDB.findAll(f => f.user_id === req.user.id)
    .sort((a, b) => new Date(b.updated_at || b.created_at) - new Date(a.updated_at || a.created_at));
  res.json(folders);
});

// Create a new folder
app.post('/api/v1/folders/', authenticateToken, (req, res) => {
  const { name, data, meta } = req.body;
  if (!name || !name.trim()) {
    return res.status(400).json({ detail: 'Folder name is required' });
  }

  const folder = {
    id: uuidv4(),
    user_id: req.user.id,
    name: name.trim(),
    data: data || {},
    meta: meta || {},
    parent_id: null,
    is_expanded: false,
    items: { chat_ids: [], file_ids: [] },
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  };

  foldersDB.insert(folder);
  res.json(folder);
});

// Get folder by ID
app.get('/api/v1/folders/:id', authenticateToken, (req, res) => {
  const folder = foldersDB.findOne(f => f.id === req.params.id && f.user_id === req.user.id);
  if (!folder) {
    return res.status(404).json({ detail: 'Folder not found' });
  }
  res.json(folder);
});

// Update folder
app.post('/api/v1/folders/:id/update', authenticateToken, (req, res) => {
  const { name, data, meta } = req.body;
  const folder = foldersDB.findOne(f => f.id === req.params.id && f.user_id === req.user.id);
  if (!folder) {
    return res.status(404).json({ detail: 'Folder not found' });
  }

  const updates = { updated_at: new Date().toISOString() };
  if (name !== undefined) updates.name = name;
  if (data !== undefined) updates.data = data;
  if (meta !== undefined) updates.meta = meta;

  const updated = foldersDB.update(f => f.id === req.params.id, updates);
  res.json(updated);
});

// Update folder expanded state
app.post('/api/v1/folders/:id/update/expanded', authenticateToken, (req, res) => {
  const { is_expanded } = req.body;
  const folder = foldersDB.findOne(f => f.id === req.params.id && f.user_id === req.user.id);
  if (!folder) {
    return res.status(404).json({ detail: 'Folder not found' });
  }

  const updated = foldersDB.update(f => f.id === req.params.id, {
    is_expanded: !!is_expanded,
    updated_at: new Date().toISOString()
  });
  res.json(updated);
});

// Update folder parent
app.post('/api/v1/folders/:id/update/parent', authenticateToken, (req, res) => {
  const { parent_id } = req.body;
  const folder = foldersDB.findOne(f => f.id === req.params.id && f.user_id === req.user.id);
  if (!folder) {
    return res.status(404).json({ detail: 'Folder not found' });
  }

  // Prevent circular references
  if (parent_id) {
    let current = parent_id;
    const visited = new Set();
    while (current) {
      if (current === req.params.id || visited.has(current)) {
        return res.status(400).json({ detail: 'Circular folder reference detected' });
      }
      visited.add(current);
      const parentFolder = foldersDB.findOne(f => f.id === current);
      current = parentFolder?.parent_id;
    }
  }

  const updated = foldersDB.update(f => f.id === req.params.id, {
    parent_id: parent_id || null,
    updated_at: new Date().toISOString()
  });
  res.json(updated);
});

// Update folder items (chat_ids, file_ids)
app.post('/api/v1/folders/:id/update/items', authenticateToken, (req, res) => {
  const { items } = req.body;
  const folder = foldersDB.findOne(f => f.id === req.params.id && f.user_id === req.user.id);
  if (!folder) {
    return res.status(404).json({ detail: 'Folder not found' });
  }

  const updated = foldersDB.update(f => f.id === req.params.id, {
    items: {
      chat_ids: items?.chat_ids || folder.items?.chat_ids || [],
      file_ids: items?.file_ids || folder.items?.file_ids || [],
    },
    updated_at: new Date().toISOString()
  });
  res.json(updated);
});

// Delete folder
app.delete('/api/v1/folders/:id', authenticateToken, (req, res) => {
  const folder = foldersDB.findOne(f => f.id === req.params.id && f.user_id === req.user.id);
  if (!folder) {
    return res.status(404).json({ detail: 'Folder not found' });
  }

  const deleteContents = req.query.delete_contents === 'true';

  if (deleteContents) {
    // Delete chats in this folder
    if (folder.items?.chat_ids) {
      folder.items.chat_ids.forEach(chatId => {
        chatsDB.delete(c => c.id === chatId && c.user_id === req.user.id);
      });
    }
    // Delete files in this folder
    if (folder.items?.file_ids) {
      folder.items.file_ids.forEach(fileId => {
        const file = filesDB.findOne(f => f.id === fileId);
        if (file?.path && fs.existsSync(file.path)) {
          try { fs.unlinkSync(file.path); } catch (e) { /* ignore */ }
        }
        filesDB.delete(f => f.id === fileId);
      });
    }
    // Delete child folders recursively
    const childFolders = foldersDB.findAll(f => f.parent_id === req.params.id && f.user_id === req.user.id);
    childFolders.forEach(child => {
      foldersDB.delete(f => f.id === child.id);
    });
  } else {
    // Move child folders to parent
    const childFolders = foldersDB.findAll(f => f.parent_id === req.params.id && f.user_id === req.user.id);
    childFolders.forEach(child => {
      foldersDB.update(f => f.id === child.id, { parent_id: folder.parent_id || null });
    });
  }

  foldersDB.delete(f => f.id === req.params.id);
  res.json({ success: true });
});

// ==================== Channels API ====================

app.get('/api/v1/channels/', authenticateToken, (req, res) => {
  res.json([]);
});

// ==================== Retrieval / Web Search API ====================

// RAG config (stub - we don't use a full RAG pipeline)
app.get('/api/v1/retrieval/config', authenticateToken, (req, res) => {
  res.json({
    status: true,
    chunk: { chunk_size: 1500, chunk_overlap: 100 },
    content_extraction: { engine: 'default', tika_server_url: null, document_intelligence_config: null },
    web_loader_ssl_verification: true,
    youtube: { language: ['en'], proxy_url: '' },
    PDF_EXTRACT_IMAGES: false,
    ENABLE_GOOGLE_DRIVE_INTEGRATION: false,
    ENABLE_ONEDRIVE_INTEGRATION: false,
  });
});

app.post('/api/v1/retrieval/config/update', authenticateToken, (req, res) => {
  res.json(req.body);
});

// Query settings
app.get('/api/v1/retrieval/query/settings', authenticateToken, (req, res) => {
  res.json({ k: 4, r: 0.0, template: null });
});

app.post('/api/v1/retrieval/query/settings/update', authenticateToken, (req, res) => {
  res.json(req.body);
});

// Embedding config
app.get('/api/v1/retrieval/embedding', authenticateToken, (req, res) => {
  res.json({ embedding_engine: 'default', embedding_model: 'default', embedding_batch_size: 100 });
});

app.post('/api/v1/retrieval/embedding/update', authenticateToken, (req, res) => {
  res.json(req.body);
});

// Reranking config
app.get('/api/v1/retrieval/reranking', authenticateToken, (req, res) => {
  res.json({ reranking_model: '' });
});

app.post('/api/v1/retrieval/reranking/update', authenticateToken, (req, res) => {
  res.json(req.body);
});

// Web search - uses Anthropic Claude to synthesize web-like search results
app.post('/api/v1/retrieval/process/web/search', authenticateToken, async (req, res) => {
  try {
    const { query, collection_name } = req.body;
    if (!query) {
      return res.status(400).json({ detail: 'Query is required' });
    }

    // Use Anthropic to generate search-like results based on its knowledge
    const searchResult = await callAnthropicForTask(
      `You are a web search engine. Given a search query, provide a comprehensive, factual answer with specific data points, numbers, dates, and sources where applicable. Focus on the most current and relevant information you have. Structure your response clearly with key facts.`,
      `Search query: "${query}"\n\nProvide a comprehensive, factual response with specific data points. Include relevant statistics, dates, and context. Be as specific and accurate as possible.`,
      1000
    );

    const searchContent = searchResult.choices?.[0]?.message?.content || '';

    // Return in the format the frontend expects
    res.json({
      status: true,
      collection_name: collection_name || `web-search-${Date.now()}`,
      filenames: [`web-search: ${query}`],
      loaded_count: 1,
      docs: [{
        content: searchContent,
        metadata: { source: 'web-search', query: query, timestamp: new Date().toISOString() }
      }]
    });
  } catch (error) {
    logger.error({ err: error }, 'Web search error');
    res.status(500).json({ detail: error.message || 'Web search failed' });
  }
});

// Process web page (stub - return success)
app.post('/api/v1/retrieval/process/web', authenticateToken, async (req, res) => {
  const { url, collection_name } = req.body;
  res.json({
    status: true,
    collection_name: collection_name || `web-${Date.now()}`,
    filenames: [url]
  });
});

// Process YouTube (stub)
app.post('/api/v1/retrieval/process/youtube', authenticateToken, (req, res) => {
  res.json({ status: true, collection_name: `youtube-${Date.now()}`, filenames: [req.body.url] });
});

// Query document (stub)
app.post('/api/v1/retrieval/query/doc', authenticateToken, (req, res) => {
  res.json({ status: true, distances: [[]], documents: [[]], metadatas: [[]] });
});

// Query collection (stub)
app.post('/api/v1/retrieval/query/collection', authenticateToken, (req, res) => {
  res.json({ status: true, distances: [[]], documents: [[]], metadatas: [[]] });
});

// ==================== Anthropic API Proxy ====================

// Main chat completions endpoint (called by the frontend)
app.post('/api/chat/completions', authenticateToken, apiLimiter, async (req, res) => {
  try {
    let { model, messages, stream, max_tokens, temperature, top_p, session_id, chat_id, id: responseMessageId } = req.body;
    
    // Strip the anthropic. prefix if present
    if (model && model.startsWith('anthropic.')) {
      model = model.replace('anthropic.', '');
    }
    
    // For the local ROKET model, use Claude as the backend
    if (model === 'roket') {
      model = 'claude-sonnet-4-20250514';
    }
    
    // Validate we have a model
    if (!model) {
      return res.status(400).json({ error: { message: 'Model is required' } });
    }
    
    // Convert OpenAI format messages to Anthropic format
    let systemPrompt = '';
    const anthropicMessages = [];
    
    for (const msg of messages || []) {
      if (msg.role === 'system') {
        systemPrompt += (systemPrompt ? '\n' : '') + (typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content));
      } else {
        // Handle multi-part content (images + text) from the frontend
        let anthropicContent;
        if (Array.isArray(msg.content)) {
          // Convert OpenAI multi-part format to Anthropic format
          const contentParts = [];
          for (const part of msg.content) {
            if (part.type === 'text') {
              contentParts.push({ type: 'text', text: part.text });
            } else if (part.type === 'image_url' && part.image_url?.url) {
              const url = part.image_url.url;
              if (url.startsWith('data:')) {
                // Base64 encoded image
                const match = url.match(/^data:(image\/[^;]+);base64,(.+)$/);
                if (match) {
                  contentParts.push({
                    type: 'image',
                    source: {
                      type: 'base64',
                      media_type: match[1],
                      data: match[2]
                    }
                  });
                }
              } else {
                // URL image — Anthropic supports URL source
                contentParts.push({
                  type: 'image',
                  source: {
                    type: 'url',
                    url: url
                  }
                });
              }
            }
          }
          anthropicContent = contentParts.length === 1 && contentParts[0].type === 'text' 
            ? contentParts[0].text 
            : contentParts;
        } else {
          anthropicContent = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
        }
        
        anthropicMessages.push({
          role: msg.role === 'assistant' ? 'assistant' : 'user',
          content: anthropicContent
        });
      }
    }
    
    // Ensure we have at least one message
    if (anthropicMessages.length === 0) {
      return res.status(400).json({ error: { message: 'At least one message is required' } });
    }
    
    // Generate a task_id for this request
    const taskId = 'task-' + uuidv4();
    
    // Smart max_tokens: Opus 4.6 supports 128K output, Sonnet 4.5 supports 64K
    const isOpus46 = model && (model.includes('opus-4-6') || model.includes('opus-4.6'));
    const isOpus = model && model.includes('opus');
    const defaultMaxTokens = isOpus46 ? 32768 : isOpus ? 16384 : 8192;
    
    const anthropicRequest = {
      model: model,
      max_tokens: max_tokens || defaultMaxTokens,
      messages: anthropicMessages,
      stream: true  // Stream for typewriter effect
    };
    
    // Inject ROKET system prompt with Anthropic prompt caching
    // cache_control: ephemeral tells Anthropic to cache the system prompt across requests
    // This saves ~90% on input token costs for the static system prompt
    const systemBlocks = [
      { type: 'text', text: ROKET_SYSTEM_PROMPT, cache_control: { type: 'ephemeral' } },
    ];
    if (systemPrompt) {
      systemBlocks.push({ type: 'text', text: systemPrompt });
    }
    
    // Inject user memories for personalization
    const userMemories = memoriesDB.findAll(m => m.user_id === req.user.id);
    if (userMemories.length > 0) {
      const memoryContext = userMemories
        .slice(0, 20) // Limit to 20 most recent memories
        .map(m => `- ${m.content}`)
        .join('\n');
      systemBlocks.push({
        type: 'text',
        text: `\n\nUSER MEMORIES & PREFERENCES (use these to personalize your responses):\n${memoryContext}`
      });
    }

    // Inject file context if files are attached
    if (req.body.files && Array.isArray(req.body.files) && req.body.files.length > 0) {
      logger.info({ fileCount: req.body.files.length, fileRefs: req.body.files.map(f => ({ id: f.id, type: f.type, name: f.name })) }, 'Processing attached files for chat');
      
      for (const fileRef of req.body.files) {
        const fileId = fileRef.id || fileRef;
        const file = filesDB.findOne(f => f.id === fileId);
        
        if (!file) {
          logger.warn({ fileId }, 'Attached file not found in database');
          continue;
        }
        
        let content = file.data?.content || '';
        
        // If content is empty, try to re-extract from disk
        if (!content && file.path && fs.existsSync(file.path)) {
          logger.info({ fileId, filename: file.filename, path: file.path }, 'Re-extracting content from file on disk');
          try {
            content = await extractFileContent(file.path, file.filename, file.meta?.content_type);
            // Update the stored content for future use
            if (content) {
              filesDB.update(
                f => f.id === fileId,
                { data: { ...file.data, content }, updated_at: new Date().toISOString() }
              );
            }
          } catch (e) {
            logger.warn({ err: e, fileId }, 'Failed to re-extract file content');
          }
        }
        
        if (content) {
          // Limit to 50K chars per file (generous for spreadsheets/PDFs)
          const truncatedContent = content.length > 50000 ? content.substring(0, 50000) + '\n\n[... content truncated at 50,000 characters ...]' : content;
          systemBlocks.push({
            type: 'text',
            text: `\n\nATTACHED FILE: "${file.filename}" (${file.meta?.content_type || 'unknown type'}, ${file.meta?.size ? Math.round(file.meta.size / 1024) + 'KB' : 'unknown size'})\n---\n${truncatedContent}\n---`
          });
          logger.info({ fileId, filename: file.filename, contentLength: content.length, truncated: content.length > 50000 }, 'File content injected into chat context');
        } else {
          // Even if we can't extract content, tell Claude the file exists
          systemBlocks.push({
            type: 'text',
            text: `\n\nATTACHED FILE: "${file.filename}" (${file.meta?.content_type || 'unknown type'}, ${file.meta?.size ? Math.round(file.meta.size / 1024) + 'KB' : 'unknown size'})\nNote: Unable to extract text content from this file. The file was uploaded but its contents could not be read as text.`
          });
          logger.warn({ fileId, filename: file.filename, contentType: file.meta?.content_type }, 'Could not extract text content from attached file');
        }
      }
    }
    
    anthropicRequest.system = systemBlocks;
    if (temperature !== undefined) {
      anthropicRequest.temperature = temperature;
    }
    if (top_p !== undefined) {
      anthropicRequest.top_p = top_p;
    }
    
    // Inject NUBLE financial tools so Claude can call the ROKET API
    const nubleTools = getNubleToolsAnthropic();
    if (nubleTools.length > 0) {
      anthropicRequest.tools = nubleTools;
    }
    
    // Return task_id immediately, then process in background
    res.json({ task_id: taskId });
    
    // Process the request asynchronously — handles tool calls in a loop
    (async () => {
      const MAX_TOOL_ROUNDS = 5; // Max tool-use round-trips to prevent infinite loops
      let currentMessages = [...anthropicMessages];
      let totalInputTokens = 0;
      let totalOutputTokens = 0;
      const completionId = 'chatcmpl-' + Date.now();
      
      for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), ANTHROPIC_REQUEST_TIMEOUT_MS);
        
        try {
          const isLastPossibleRound = round === MAX_TOOL_ROUNDS - 1;
          const requestBody = {
            ...anthropicRequest,
            messages: currentMessages,
            // On the last possible round, don't allow tools to prevent infinite loops
            ...(isLastPossibleRound ? { tools: undefined } : {}),
            // First round: stream for fast text display. Tool rounds: non-streaming for simplicity
            stream: round === 0,
          };
          
          const response = await fetch('https://api.anthropic.com/v1/messages', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'x-api-key': ANTHROPIC_API_KEY,
              'anthropic-version': '2023-06-01'
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal,
          });
          
          if (!response.ok) {
            const errorText = await response.text();
            let errorData;
            try { errorData = JSON.parse(errorText); } catch { errorData = { message: errorText }; }
            logger.error({ status: response.status, error: errorData, round }, 'Anthropic API error');
            if (session_id) {
              io.to(session_id).emit('events', {
                chat_id, message_id: responseMessageId,
                data: { type: 'chat:completion', data: { error: { message: errorData.error?.message || errorData.message || `Anthropic API error: ${response.status}` } } }
              });
            }
            return;
          }
          
          if (!session_id) {
            logger.warn('No session_id provided, cannot emit response via Socket.IO');
            return;
          }
          
          // ── Round 0: Stream the first response for fast typewriter effect ──
          if (round === 0) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let stopReason = 'end_turn';
            const contentBlocks = []; // Collect all content blocks for tool-use detection
            let currentBlockType = null;
            let currentBlockIndex = -1;
            let currentToolUse = null;
            let currentToolInput = '';
            
            while (true) {
              const { done: readerDone, value } = await reader.read();
              if (readerDone) break;
              
              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split('\n');
              buffer = lines.pop() || '';
              
              for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const dataStr = line.slice(6).trim();
                if (!dataStr || dataStr === '[DONE]') continue;
                
                try {
                  const event = JSON.parse(dataStr);
                  
                  if (event.type === 'content_block_start') {
                    currentBlockIndex = event.index;
                    currentBlockType = event.content_block?.type;
                    if (currentBlockType === 'tool_use') {
                      currentToolUse = {
                        type: 'tool_use',
                        id: event.content_block.id,
                        name: event.content_block.name,
                        input: {}
                      };
                      currentToolInput = '';
                      // Tell the frontend we're calling a tool
                      io.to(session_id).emit('events', {
                        chat_id, message_id: responseMessageId,
                        data: { type: 'chat:completion', data: { id: completionId, done: false, choices: [{ index: 0, delta: { content: `\n\n*Querying ${event.content_block.name.replace('roket_', '').replace(/_/g, ' ')}...*\n` }, finish_reason: null }] } }
                      });
                    } else if (currentBlockType === 'text') {
                      contentBlocks.push({ type: 'text', text: '' });
                    }
                  } else if (event.type === 'content_block_delta') {
                    if (currentBlockType === 'text' && event.delta?.text) {
                      // Stream text to frontend
                      const lastBlock = contentBlocks[contentBlocks.length - 1];
                      if (lastBlock && lastBlock.type === 'text') {
                        lastBlock.text += event.delta.text;
                      }
                      io.to(session_id).emit('events', {
                        chat_id, message_id: responseMessageId,
                        data: { type: 'chat:completion', data: { id: completionId, done: false, choices: [{ index: 0, delta: { content: event.delta.text }, finish_reason: null }] } }
                      });
                    } else if (currentBlockType === 'tool_use' && event.delta?.partial_json) {
                      currentToolInput += event.delta.partial_json;
                    }
                  } else if (event.type === 'content_block_stop') {
                    if (currentBlockType === 'tool_use' && currentToolUse) {
                      try { currentToolUse.input = JSON.parse(currentToolInput); } catch { currentToolUse.input = {}; }
                      contentBlocks.push(currentToolUse);
                      currentToolUse = null;
                      currentToolInput = '';
                    }
                    currentBlockType = null;
                  } else if (event.type === 'message_delta') {
                    if (event.usage) totalOutputTokens += event.usage.output_tokens || 0;
                    if (event.delta?.stop_reason) stopReason = event.delta.stop_reason;
                  } else if (event.type === 'message_start' && event.message?.usage) {
                    totalInputTokens += event.message.usage.input_tokens || 0;
                  } else if (event.type === 'error') {
                    logger.error({ event }, 'Anthropic stream error event');
                    io.to(session_id).emit('events', {
                      chat_id, message_id: responseMessageId,
                      data: { type: 'chat:completion', data: { error: { message: event.error?.message || 'Unknown streaming error' } } }
                    });
                    return;
                  }
                } catch (e) {
                  // Skip non-JSON lines
                }
              }
            }
            
            // Check if Claude wants to use tools
            const toolUseBlocks = contentBlocks.filter(b => b.type === 'tool_use');
            
            if (stopReason === 'tool_use' && toolUseBlocks.length > 0) {
              // Claude wants to call tools — add assistant message and process tools
              currentMessages.push({ role: 'assistant', content: contentBlocks });
              
              // Execute all tool calls in parallel
              const toolResults = await Promise.all(toolUseBlocks.map(async (toolBlock) => {
                logger.info({ tool: toolBlock.name, input: toolBlock.input, round }, 'Executing NUBLE tool for chat');
                try {
                  const result = await processNubleTool(toolBlock.name, toolBlock.input);
                  return {
                    type: 'tool_result',
                    tool_use_id: toolBlock.id,
                    content: typeof result === 'string' ? result : JSON.stringify(result),
                  };
                } catch (err) {
                  logger.error({ err, tool: toolBlock.name }, 'Tool execution failed');
                  return {
                    type: 'tool_result',
                    tool_use_id: toolBlock.id,
                    content: JSON.stringify({ error: err.message }),
                    is_error: true,
                  };
                }
              }));
              
              currentMessages.push({ role: 'user', content: toolResults });
              
              // Continue the loop — next round will NOT stream (simpler for tool follow-ups)
              clearTimeout(timeout);
              continue;
            }
            
            // No tools — we're done. Emit final done event.
            io.to(session_id).emit('events', {
              chat_id, message_id: responseMessageId,
              data: { type: 'chat:completion', data: { id: completionId, done: true, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: totalInputTokens, completion_tokens: totalOutputTokens, total_tokens: totalInputTokens + totalOutputTokens } } }
            });
            return;
          }
          
          // ── Rounds 1+: Non-streaming (tool follow-up rounds) ──
          const result = await response.json();
          totalInputTokens += result.usage?.input_tokens || 0;
          totalOutputTokens += result.usage?.output_tokens || 0;
          
          // Stream the text content to frontend
          const textBlocks = (result.content || []).filter(b => b.type === 'text');
          for (const block of textBlocks) {
            if (block.text) {
              // Send in chunks for a more natural typewriter effect
              const chunks = block.text.match(/.{1,80}/gs) || [block.text];
              for (const chunk of chunks) {
                io.to(session_id).emit('events', {
                  chat_id, message_id: responseMessageId,
                  data: { type: 'chat:completion', data: { id: completionId, done: false, choices: [{ index: 0, delta: { content: chunk }, finish_reason: null }] } }
                });
              }
            }
          }
          
          // Check if more tools are needed
          const toolBlocks = (result.content || []).filter(b => b.type === 'tool_use');
          if (result.stop_reason === 'tool_use' && toolBlocks.length > 0) {
            currentMessages.push({ role: 'assistant', content: result.content });
            
            // Notify frontend about tool calls
            for (const tb of toolBlocks) {
              io.to(session_id).emit('events', {
                chat_id, message_id: responseMessageId,
                data: { type: 'chat:completion', data: { id: completionId, done: false, choices: [{ index: 0, delta: { content: `\n\n*Querying ${tb.name.replace('roket_', '').replace(/_/g, ' ')}...*\n` }, finish_reason: null }] } }
              });
            }
            
            const toolResults = await Promise.all(toolBlocks.map(async (toolBlock) => {
              logger.info({ tool: toolBlock.name, input: toolBlock.input, round }, 'Executing NUBLE tool for chat (follow-up)');
              try {
                const toolResult = await processNubleTool(toolBlock.name, toolBlock.input);
                return {
                  type: 'tool_result',
                  tool_use_id: toolBlock.id,
                  content: typeof toolResult === 'string' ? toolResult : JSON.stringify(toolResult),
                };
              } catch (err) {
                logger.error({ err, tool: toolBlock.name }, 'Tool execution failed');
                return { type: 'tool_result', tool_use_id: toolBlock.id, content: JSON.stringify({ error: err.message }), is_error: true };
              }
            }));
            
            currentMessages.push({ role: 'user', content: toolResults });
            clearTimeout(timeout);
            continue; // Next round
          }
          
          // Done — emit final event
          io.to(session_id).emit('events', {
            chat_id, message_id: responseMessageId,
            data: { type: 'chat:completion', data: { id: completionId, done: true, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: totalInputTokens, completion_tokens: totalOutputTokens, total_tokens: totalInputTokens + totalOutputTokens } } }
          });
          return;
          
        } catch (error) {
          logger.error({ err: error, round }, 'Async chat completion error');
          if (session_id) {
            io.to(session_id).emit('events', {
              chat_id, message_id: responseMessageId,
              data: { type: 'chat:completion', data: { error: { message: error.name === 'AbortError' ? 'Request timed out' : (error.message || 'Failed to process chat completion') } } }
            });
          }
          return;
        } finally {
          clearTimeout(timeout);
        }
      }
      
      // If we exhausted all rounds, emit done
      if (session_id) {
        io.to(session_id).emit('events', {
          chat_id, message_id: responseMessageId,
          data: { type: 'chat:completion', data: { id: completionId, done: true, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: totalInputTokens, completion_tokens: totalOutputTokens, total_tokens: totalInputTokens + totalOutputTokens } } }
        });
      }
    })();
    
  } catch (error) {
    logger.error({ err: error }, 'Chat completion error');
    res.status(500).json({ error: { message: error.message || 'Failed to process chat completion' } });
  }
});

// List available Anthropic models (allow direct-connection key or JWT)
app.get('/api/anthropic/models', async (req, res) => {
  // Check for direct-connection API key or JWT token
  const authHeader = req.headers['authorization'];
  const apiKey = authHeader && authHeader.startsWith('Bearer ') ? authHeader.slice(7) : null;
  
  // Allow direct-connection or valid JWT
  if (apiKey !== 'direct-connection') {
    try {
      jwt.verify(apiKey, JWT_SECRET);
    } catch (err) {
      return res.status(403).json({ detail: 'Invalid or expired token' });
    }
  }
  
  try {
    const models = await fetchAnthropicModels();
    // Return in the simpler format this endpoint expects
    res.json({ data: models.map(m => ({
      id: m.id.replace('anthropic.', ''),
      object: 'model',
      created: m.created,
      owned_by: 'anthropic',
      name: m.name
    })) });
  } catch (error) {
    logger.error({ err: error }, 'Anthropic models endpoint error');
    res.status(500).json({ error: 'Failed to fetch models' });
  }
});

// Proxy chat completions to Anthropic (allow direct-connection key or JWT)
app.post('/api/anthropic/chat/completions', async (req, res) => {
  // Check for direct-connection API key or JWT token
  const authHeader = req.headers['authorization'];
  const apiKey = authHeader && authHeader.startsWith('Bearer ') ? authHeader.slice(7) : null;
  
  // Allow direct-connection or valid JWT
  if (apiKey !== 'direct-connection') {
    try {
      jwt.verify(apiKey, JWT_SECRET);
    } catch (err) {
      return res.status(403).json({ detail: 'Invalid or expired token' });
    }
  }
  
  try {
    let { model, messages, stream, max_tokens, temperature, top_p } = req.body;
    
    // Strip "anthropic." prefix if present (Open WebUI adds this)
    if (model && model.startsWith('anthropic.')) {
      model = model.replace('anthropic.', '');
    }
    
    // Convert OpenAI format messages to Anthropic format
    let systemPrompt = '';
    const anthropicMessages = [];
    
    for (const msg of messages) {
      if (msg.role === 'system') {
        systemPrompt += (systemPrompt ? '\n' : '') + msg.content;
      } else {
        anthropicMessages.push({
          role: msg.role === 'assistant' ? 'assistant' : 'user',
          content: msg.content
        });
      }
    }
    
    const isOpus46_2 = model && (model.includes('opus-4-6') || model.includes('opus-4.6'));
    const isOpus_2 = model && model.includes('opus');
    const defaultMaxTokens_2 = isOpus46_2 ? 32768 : isOpus_2 ? 16384 : 8192;

    const anthropicRequest = {
      model: model,
      max_tokens: max_tokens || defaultMaxTokens_2,
      messages: anthropicMessages,
      stream: stream || false
    };
    
    // Inject ROKET system prompt with prompt caching
    const systemBlocks = [
      { type: 'text', text: ROKET_SYSTEM_PROMPT, cache_control: { type: 'ephemeral' } },
    ];
    if (systemPrompt) {
      systemBlocks.push({ type: 'text', text: systemPrompt });
    }
    anthropicRequest.system = systemBlocks;
    if (temperature !== undefined) {
      anthropicRequest.temperature = temperature;
    }
    if (top_p !== undefined) {
      anthropicRequest.top_p = top_p;
    }
    
    // Inject NUBLE financial tools
    const nubleTools = getNubleToolsAnthropic();
    if (nubleTools.length > 0) {
      anthropicRequest.tools = nubleTools;
    }
    
    // ── Server-side tool execution loop (mirrors /api/chat/completions logic) ──
    const MAX_TOOL_ROUNDS = 5;
    let currentMessages = [...anthropicMessages];
    let totalInputTokens = 0;
    let totalOutputTokens = 0;
    
    for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), ANTHROPIC_REQUEST_TIMEOUT_MS);
      
      try {
        const isLastPossibleRound = round === MAX_TOOL_ROUNDS - 1;
        const requestBody = {
          ...anthropicRequest,
          messages: currentMessages,
          // On the last round, remove tools to force a text response
          ...(isLastPossibleRound ? { tools: undefined } : {}),
          // Always non-streaming for tool loop to simplify handling; stream the final round
          stream: false,
        };
        
        const anthropicResponse = await fetch('https://api.anthropic.com/v1/messages', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01'
          },
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        });
        
        if (!anthropicResponse.ok) {
          const errorText = await anthropicResponse.text();
          let errorData;
          try { errorData = JSON.parse(errorText); } catch { errorData = { message: errorText }; }
          logger.error({ status: anthropicResponse.status, error: errorData, round }, 'Anthropic API error (proxy)');
          return res.status(anthropicResponse.status).json({ 
            error: { message: errorData.error?.message || errorData.message || `Anthropic API error: ${anthropicResponse.status}` },
            detail: errorData
          });
        }
        
        const data = await anthropicResponse.json();
        totalInputTokens += data.usage?.input_tokens || 0;
        totalOutputTokens += data.usage?.output_tokens || 0;
        
        if (data.error) {
          logger.error({ error: data.error, round }, 'Anthropic response error (proxy)');
          return res.status(400).json({ error: data.error });
        }
        
        // Check if Claude wants to use tools
        const toolBlocks = (data.content || []).filter(b => b.type === 'tool_use');
        
        if (data.stop_reason === 'tool_use' && toolBlocks.length > 0) {
          logger.info({ tools: toolBlocks.map(t => t.name), round }, 'Executing NUBLE tools (proxy)');
          
          // Add assistant message with tool calls
          currentMessages.push({ role: 'assistant', content: data.content });
          
          // Execute all tool calls in parallel
          const toolResults = await Promise.all(toolBlocks.map(async (toolBlock) => {
            logger.info({ tool: toolBlock.name, input: toolBlock.input, round }, 'Executing NUBLE tool for proxy chat');
            try {
              const result = await processNubleTool(toolBlock.name, toolBlock.input);
              return {
                type: 'tool_result',
                tool_use_id: toolBlock.id,
                content: typeof result === 'string' ? result : JSON.stringify(result),
              };
            } catch (err) {
              logger.error({ err, tool: toolBlock.name }, 'Tool execution failed (proxy)');
              return {
                type: 'tool_result',
                tool_use_id: toolBlock.id,
                content: JSON.stringify({ error: err.message }),
                is_error: true,
              };
            }
          }));
          
          currentMessages.push({ role: 'user', content: toolResults });
          clearTimeout(timeout);
          continue; // Next round — Claude will synthesize tool results
        }
        
        // No tool calls — this is the final response. Extract text content.
        const textContent = (data.content || [])
          .filter(b => b.type === 'text')
          .map(b => b.text)
          .join('');
        
        if (stream) {
          // Stream the final text as OpenAI-format SSE chunks
          res.setHeader('Content-Type', 'text/event-stream');
          res.setHeader('Cache-Control', 'no-cache');
          res.setHeader('Connection', 'keep-alive');
          
          // Send text in chunks for typewriter effect
          const chunks = textContent.match(/.{1,100}/gs) || [textContent];
          for (const chunk of chunks) {
            const openaiChunk = {
              id: 'chatcmpl-' + Date.now(),
              object: 'chat.completion.chunk',
              created: Math.floor(Date.now() / 1000),
              model: model,
              choices: [{ index: 0, delta: { content: chunk }, finish_reason: null }]
            };
            res.write(`data: ${JSON.stringify(openaiChunk)}\n\n`);
          }
          
          // Send final stop event
          const stopChunk = {
            id: 'chatcmpl-' + Date.now(),
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model: model,
            choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
            usage: { prompt_tokens: totalInputTokens, completion_tokens: totalOutputTokens, total_tokens: totalInputTokens + totalOutputTokens }
          };
          res.write(`data: ${JSON.stringify(stopChunk)}\n\n`);
          res.write('data: [DONE]\n\n');
          return res.end();
        } else {
          // Non-streaming: convert to OpenAI format
          return res.json({
            id: data.id || 'chatcmpl-' + Date.now(),
            object: 'chat.completion',
            created: Math.floor(Date.now() / 1000),
            model: data.model,
            choices: [{
              index: 0,
              message: { role: 'assistant', content: textContent },
              finish_reason: data.stop_reason === 'end_turn' ? 'stop' : data.stop_reason
            }],
            usage: {
              prompt_tokens: totalInputTokens,
              completion_tokens: totalOutputTokens,
              total_tokens: totalInputTokens + totalOutputTokens
            }
          });
        }
        
      } catch (error) {
        logger.error({ err: error, round }, 'Proxy chat completion error');
        return res.status(500).json({ error: { message: error.name === 'AbortError' ? 'Request timed out' : (error.message || 'Failed to process chat completion') } });
      } finally {
        clearTimeout(timeout);
      }
    }
    
    // Exhausted all rounds — return what we have
    return res.json({
      id: 'chatcmpl-' + Date.now(),
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: model,
      choices: [{
        index: 0,
        message: { role: 'assistant', content: 'I reached the maximum number of tool calls. Please try a more specific question.' },
        finish_reason: 'stop'
      }],
      usage: { prompt_tokens: totalInputTokens, completion_tokens: totalOutputTokens, total_tokens: totalInputTokens + totalOutputTokens }
    });
    
  } catch (error) {
    logger.error({ err: error }, 'Anthropic proxy error');
    res.status(500).json({ error: { message: 'Failed to proxy request to Anthropic: ' + error.message } });
  }
});

// ==================== Socket.IO ====================

io.on('connection', (socket) => {
  // Join a room with the socket's own ID so we can send targeted messages
  socket.join(socket.id);
  
  socket.on('disconnect', () => {
    // Socket cleanup handled automatically
  });
  
  // Handle user-usage event
  socket.on('user-usage', (data) => {
    // Echo back or handle as needed
  });
});

// ==================== Nova Sonic Direct Bedrock Integration ====================

const AWS_REGION = process.env.AWS_REGION || 'us-east-1';

// Session states
const SessionState = {
  INITIALIZING: 'initializing',
  READY: 'ready',
  ACTIVE: 'active',
  CLOSED: 'closed',
};

// Store clients per region
const regionClients = new Map();
const socketSessions = new Map();
const socketClients = new Map();
const socketConfigs = new Map();
const sessionStates = new Map();
const cleanupInProgress = new Map();
const sessionTranscripts = new Map(); // Track transcripts for voice sessions

/**
 * Get or create a Bedrock client for a specific region.
 * AWS SDK automatically uses credential chain:
 * 1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
 * 2. Shared credentials file (~/.aws/credentials)
 * 3. ECS container credentials
 * 4. EC2 instance metadata (IAM role)
 */
function getClientForRegion(region) {
  if (!regionClients.has(region)) {
    logger.info({ region }, 'Nova Sonic: creating Bedrock client');
    const client = new NovaSonicBidirectionalStreamClient({
      requestHandlerConfig: {
        maxConcurrentStreams: 10,
      },
      clientConfig: {
        region: region,
      },
    });
    regionClients.set(region, client);
  }
  return regionClients.get(region);
}

// Initialize default region client
const defaultNovaSonicClient = getClientForRegion(AWS_REGION);

// Periodically clean up inactive sessions (every 60s)
setInterval(() => {
  const now = Date.now();
  regionClients.forEach((client, region) => {
    client.getActiveSessions().forEach((sessionId) => {
      const lastActivity = client.getLastActivityTime(sessionId);
      if (now - lastActivity > 5 * 60 * 1000) {
        logger.info({ sessionId, region }, 'Nova Sonic: closing inactive session');
        try {
          client.forceCloseSession(sessionId);
          socketSessions.delete(sessionId);
          socketClients.delete(sessionId);
          socketConfigs.delete(sessionId);
          sessionStates.set(sessionId, SessionState.CLOSED);
        } catch (error) {
          logger.error({ sessionId, err: error }, 'Nova Sonic: error force closing inactive session');
        }
      }
    });
  });
}, 60000);

/**
 * Set up event handlers to relay Bedrock responses to the client via Socket.IO
 */
function setupSessionEventHandlers(session, socket) {
  session.onEvent('usageEvent', (data) => {
    socket.emit('usageEvent', data);
  });

  session.onEvent('completionStart', (data) => {
    socket.emit('completionStart', data);
  });

  session.onEvent('contentStart', (data) => {
    socket.emit('contentStart', data);
  });

  session.onEvent('textOutput', (data) => {
    socket.emit('textOutput', data);
    // Accumulate transcript for voice session saving
    if (data.content) {
      if (!sessionTranscripts.has(socket.id)) {
        sessionTranscripts.set(socket.id, []);
      }
      sessionTranscripts.get(socket.id).push({
        role: data.role || 'ASSISTANT',
        content: data.content,
        timestamp: new Date().toISOString()
      });
    }
  });

  session.onEvent('audioOutput', (data) => {
    socket.emit('audioOutput', data);
  });

  session.onEvent('error', (data) => {
    logger.error({ data }, 'Nova Sonic: session error');
    socket.emit('error', data);
    
    // On bidirectional stream errors, the session is dead - clean up and notify frontend
    if (data.source === 'bidirectionalStream' || data.source === 'responseStream') {
      logger.error({ socketId: socket.id }, 'Nova Sonic: fatal stream error, cleaning up session');
      sessionStates.set(socket.id, SessionState.CLOSED);
      socketSessions.delete(socket.id);
      socketClients.delete(socket.id);
      socketConfigs.delete(socket.id);
      socket.emit('sessionClosed');
    }
  });

  session.onEvent('toolUse', async (data) => {
    // NUBLE ROKET: Execute tool server-side and send result back to Bedrock
    const toolName = data.toolName;
    const toolUseId = data.toolUseId;
    const toolContent = data.content;
    
    logger.info({ toolName, toolUseId }, 'Nova Sonic: NUBLE tool call received from Bedrock');
    socket.emit('toolUse', data);  // Forward to frontend for UI status display
    
    try {
      // Execute the tool against the NUBLE ROKET API
      const result = await processNubleTool(toolName, toolContent);
      logger.info({ toolName, toolUseId, resultKeys: Object.keys(result || {}) }, 'Nova Sonic: NUBLE tool executed successfully');
      
      // Send the tool result back to Bedrock so it can continue generating
      const session = socketSessions.get(socket.id);
      if (session) {
        await session.sendToolResult(toolUseId, result);
        logger.info({ toolName, toolUseId }, 'Nova Sonic: Tool result sent back to Bedrock');
        socket.emit('toolResult', { toolName, toolUseId, result });
      } else {
        logger.error({ toolUseId }, 'Nova Sonic: Session not found for tool result');
      }
    } catch (err) {
      logger.error({ err, toolName, toolUseId }, 'Nova Sonic: NUBLE tool execution failed');
      // Send error result back to Bedrock so it doesn't hang waiting for tool result
      const session = socketSessions.get(socket.id);
      if (session) {
        await session.sendToolResult(toolUseId, { 
          error: err.message, 
          tool: toolName,
          hint: 'The NUBLE ROKET API may not be running. Please try again.' 
        });
      }
      socket.emit('toolResult', { toolName, toolUseId, error: err.message });
    }
  });

  session.onEvent('toolResult', (data) => {
    socket.emit('toolResult', data);
  });

  session.onEvent('contentEnd', (data) => {
    socket.emit('contentEnd', data);
  });

  session.onEvent('bargeIn', (data) => {
    socket.emit('bargeIn', data);
  });

  session.onEvent('streamComplete', () => {
    socket.emit('streamComplete');
    sessionStates.set(socket.id, SessionState.CLOSED);
  });

  session.onEvent('streamInterrupted', (data) => {
    socket.emit('streamInterrupted', data);
  });
}

/**
 * Create and initialize a new Bedrock streaming session
 */
async function createNewNovaSonicSession(socket, config = {}) {
  const sessionId = socket.id;
  const region = config.region || AWS_REGION;
  const client = getClientForRegion(region);

  try {
    sessionStates.set(sessionId, SessionState.INITIALIZING);

    const sessionConfig = {};

    if (config.inferenceConfig) {
      sessionConfig.inferenceConfig = {
        maxTokens: config.inferenceConfig.maxTokens || 1024,
        topP: config.inferenceConfig.topP || 0.9,
        temperature: config.inferenceConfig.temperature || 0.7,
      };
    }

    if (config.turnDetectionConfig?.endpointingSensitivity) {
      sessionConfig.turnDetectionConfig = {
        endpointingSensitivity: config.turnDetectionConfig.endpointingSensitivity,
      };
    }

    const session = client.createStreamSession(
      sessionId,
      Object.keys(sessionConfig).length > 0 ? sessionConfig : undefined
    );

    setupSessionEventHandlers(session, socket);

    socketSessions.set(sessionId, session);
    socketClients.set(sessionId, client);
    socketConfigs.set(sessionId, config);
    sessionStates.set(sessionId, SessionState.READY);

    return session;
  } catch (error) {
    logger.error({ sessionId, err: error }, 'Nova Sonic: error creating session');
    sessionStates.set(sessionId, SessionState.CLOSED);
    throw error;
  }
}

// Create Nova Sonic namespace for voice features
const novaSonicNamespace = io.of('/nova-sonic');

// Authenticate Nova Sonic WebSocket connections via JWT
novaSonicNamespace.use((socket, next) => {
  const token = socket.handshake.auth?.token;
  if (!token) {
    return next(new Error('Authentication required'));
  }
  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      return next(new Error('Invalid or expired token'));
    }
    socket.user = user;
    next();
  });
});

novaSonicNamespace.on('connection', (socket) => {
  sessionStates.set(socket.id, SessionState.CLOSED);

  // Handle session initialization
  socket.on('initializeConnection', async (data, callback) => {
    try {
      let config = {};
      let cb = callback;

      if (typeof data === 'function') {
        cb = data;
      } else if (data && typeof data === 'object') {
        config = data;
      }

      const currentState = sessionStates.get(socket.id);

      if (currentState === SessionState.INITIALIZING || currentState === SessionState.READY || currentState === SessionState.ACTIVE) {
        if (cb) cb({ success: true });
        return;
      }

      await createNewNovaSonicSession(socket, config);
      sessionStates.set(socket.id, SessionState.READY);
      if (cb) cb({ success: true });

    } catch (error) {
      logger.error({ err: error }, 'Nova Sonic: error initializing session');
      sessionStates.set(socket.id, SessionState.CLOSED);
      const cb = typeof data === 'function' ? data : callback;
      if (cb) cb({ success: false, error: error instanceof Error ? error.message : String(error) });
      socket.emit('error', {
        message: 'Failed to initialize session',
        details: error instanceof Error ? error.message : String(error),
      });
    }
  });

  // Handle starting a new chat
  socket.on('startNewChat', async (config = {}) => {
    try {
      const currentState = sessionStates.get(socket.id);

      const existingSession = socketSessions.get(socket.id);
      const client = socketClients.get(socket.id) || defaultNovaSonicClient;

      if (existingSession && client.isSessionActive(socket.id)) {
        try {
          await existingSession.endAudioContent();
          await existingSession.endPrompt();
          await existingSession.close();
        } catch (cleanupError) {
          logger.error({ err: cleanupError }, 'Nova Sonic: error during chat cleanup');
          client.forceCloseSession(socket.id);
        }
        socketSessions.delete(socket.id);
      }

      await createNewNovaSonicSession(socket, config);
    } catch (error) {
      logger.error({ err: error }, 'Nova Sonic: error starting new chat');
      socket.emit('error', {
        message: 'Failed to start new chat',
        details: error instanceof Error ? error.message : String(error),
      });
    }
  });

  // Handle prompt start
  socket.on('promptStart', async (data) => {
    try {
      const session = socketSessions.get(socket.id);
      if (!session) {
        socket.emit('error', { message: 'No active session for prompt start' });
        return;
      }
      const voiceId = data?.voiceId;
      const outputSampleRate = data?.outputSampleRate || 24000;
      
      // Build tool configuration with all 20 NUBLE ROKET financial tools
      const nubleTools = getNubleTools();
      const toolConfiguration = {
        tools: nubleTools,
        toolChoice: { auto: {} }  // Let the model decide when to use tools
      };
      logger.info({ toolCount: nubleTools.length }, 'Nova Sonic: injecting NUBLE ROKET tool configuration');
      
      await session.setupSessionAndPromptStart(voiceId, outputSampleRate, toolConfiguration);
    } catch (error) {
      logger.error({ err: error }, 'Nova Sonic: error processing prompt start');
      socket.emit('error', {
        message: 'Error processing prompt start',
        details: error instanceof Error ? error.message : String(error),
      });
    }
  });

  // Handle system prompt
  socket.on('systemPrompt', async (data) => {
    try {
      const session = socketSessions.get(socket.id);
      if (!session) {
        socket.emit('error', { message: 'No active session for system prompt' });
        return;
      }

      let promptContent;
      let voiceId;

      if (typeof data === 'string') {
        promptContent = data;
      } else if (data && typeof data === 'object') {
        promptContent = data.content || data;
        voiceId = data.voiceId;
      } else {
        promptContent = data;
      }

      await session.setupSystemPrompt(undefined, promptContent, voiceId);
    } catch (error) {
      logger.error({ err: error }, 'Nova Sonic: error processing system prompt');
      socket.emit('error', {
        message: 'Error processing system prompt',
        details: error instanceof Error ? error.message : String(error),
      });
    }
  });

  // Handle audio start - this triggers the bidirectional stream
  socket.on('audioStart', async () => {
    try {
      const session = socketSessions.get(socket.id);
      if (!session) {
        socket.emit('error', { message: 'No active session for audio start' });
        return;
      }

      await session.setupStartAudio();

      // Now that all setup events are queued, start bidirectional streaming
      const client = socketClients.get(socket.id) || defaultNovaSonicClient;
      client.initiateBidirectionalStreaming(socket.id);

      sessionStates.set(socket.id, SessionState.ACTIVE);
      socket.emit('audioReady');
    } catch (error) {
      logger.error({ err: error }, 'Nova Sonic: error processing audio start');
      sessionStates.set(socket.id, SessionState.CLOSED);
      socket.emit('error', {
        message: 'Error processing audio start',
        details: error instanceof Error ? error.message : String(error),
      });
    }
  });

  // Handle audio input from the client (rate-limited: max ~100 packets/sec)
  let audioPacketCount = 0;
  let audioPacketWindowStart = Date.now();
  const AUDIO_RATE_LIMIT = 150; // max packets per second
  
  socket.on('audioInput', async (audioData) => {
    try {
      // Rate limiting for audio packets
      const now = Date.now();
      if (now - audioPacketWindowStart > 1000) {
        audioPacketCount = 0;
        audioPacketWindowStart = now;
      }
      audioPacketCount++;
      if (audioPacketCount > AUDIO_RATE_LIMIT) {
        return; // Silently drop excess audio packets
      }

      const session = socketSessions.get(socket.id);
      const currentState = sessionStates.get(socket.id);

      if (!session || currentState !== SessionState.ACTIVE) {
        // Don't spam errors - just silently drop audio before session is active
        return;
      }

      const audioBuffer = typeof audioData === 'string'
        ? Buffer.from(audioData, 'base64')
        : Buffer.from(audioData);

      await session.streamAudio(audioBuffer);
    } catch (error) {
      logger.error({ err: error }, 'Nova Sonic: error processing audio input');
      socket.emit('error', {
        message: 'Error processing audio',
        details: error instanceof Error ? error.message : String(error),
      });
    }
  });

  // Handle stop audio - graceful session shutdown
  socket.on('stopAudio', async () => {
    try {
      const session = socketSessions.get(socket.id);
      const client = socketClients.get(socket.id) || defaultNovaSonicClient;

      if (!session || cleanupInProgress.get(socket.id)) {
        socket.emit('sessionClosed');
        return;
      }

      cleanupInProgress.set(socket.id, true);
      sessionStates.set(socket.id, SessionState.CLOSED);

      const cleanupPromise = Promise.race([
        (async () => {
          await session.endAudioContent();
          await session.endPrompt();
          await session.close();
        })(),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Session cleanup timeout')), 5000)
        ),
      ]);

      try {
        await cleanupPromise;
      } catch (timeoutErr) {
        logger.warn('Nova Sonic: Cleanup timed out, force closing');
        client.forceCloseSession(socket.id);
      }

      socketSessions.delete(socket.id);
      socketClients.delete(socket.id);
      socketConfigs.delete(socket.id);
      cleanupInProgress.delete(socket.id);

      // Emit accumulated voice transcript so frontend can save to chat history
      const transcript = sessionTranscripts.get(socket.id);
      if (transcript && transcript.length > 0) {
        socket.emit('voiceTranscript', { messages: transcript });
        sessionTranscripts.delete(socket.id);
      }

      socket.emit('sessionClosed');
    } catch (error) {
      logger.error({ err: error }, 'Nova Sonic: error processing streaming end');
      socket.emit('error', {
        message: 'Error processing streaming end events',
        details: error instanceof Error ? error.message : String(error),
      });

      try {
        const client = socketClients.get(socket.id) || defaultNovaSonicClient;
        client.forceCloseSession(socket.id);
        socketSessions.delete(socket.id);
        socketClients.delete(socket.id);
        socketConfigs.delete(socket.id);
        cleanupInProgress.delete(socket.id);
        sessionStates.set(socket.id, SessionState.CLOSED);
        // Emit transcript even on error cleanup
        const errTranscript = sessionTranscripts.get(socket.id);
        if (errTranscript && errTranscript.length > 0) {
          socket.emit('voiceTranscript', { messages: errTranscript });
        }
        sessionTranscripts.delete(socket.id);
      } catch (forceError) {
        logger.error({ err: forceError }, 'Nova Sonic: error during force cleanup');
      }

      socket.emit('sessionClosed');
    }
  });

  // Handle client disconnect
  socket.on('disconnect', async () => {
    const session = socketSessions.get(socket.id);
    const client = socketClients.get(socket.id) || defaultNovaSonicClient;
    const sessionId = socket.id;

    if (session && client.isSessionActive(sessionId) && !cleanupInProgress.get(socket.id)) {
      try {
        cleanupInProgress.set(socket.id, true);

        const cleanupPromise = Promise.race([
          (async () => {
            await session.endAudioContent();
            await session.endPrompt();
            await session.close();
          })(),
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Disconnect cleanup timeout')), 5000)
          ),
        ]);

        try {
          await cleanupPromise;
        } catch (timeoutErr) {
          logger.warn('Nova Sonic: disconnect cleanup timed out, force closing');
          client.forceCloseSession(sessionId);
        }
      } catch (error) {
        logger.error({ err: error }, 'Nova Sonic: error during disconnect cleanup');
        client.forceCloseSession(sessionId);
      }
    }

    socketSessions.delete(sessionId);
    socketClients.delete(sessionId);
    socketConfigs.delete(sessionId);
    sessionStates.delete(sessionId);
    cleanupInProgress.delete(sessionId);
    sessionTranscripts.delete(sessionId);
  });
});

// ==================== Serve Frontend (Production) ====================

if (process.env.SERVE_FRONTEND === 'true' && process.env.FRONTEND_PATH) {
  const frontendPath = path.resolve(__dirname, process.env.FRONTEND_PATH);
  logger.info({ frontendPath }, 'Serving frontend');
  
  // Serve static frontend files
  app.use(express.static(frontendPath, {
    index: false // We'll handle index.html ourselves for SPA routing
  }));
  
  // SPA fallback - serve index.html for all non-API routes
  app.get('*', (req, res, next) => {
    // Skip API routes
    if (req.path.startsWith('/api') || req.path.startsWith('/ws') || req.path.startsWith('/static')) {
      return next();
    }
    res.sendFile(path.join(frontendPath, 'index.html'));
  });
}

// ==================== Chat Completed / Actions (Pipeline hooks) ====================

// Called by frontend after each assistant response finishes
app.post('/api/chat/completed', authenticateToken, async (req, res) => {
  try {
    const { messages, chat_id, session_id, model, id } = req.body;
    // No pipelines/filters - just echo back the messages unchanged
    res.json({ messages: messages || [] });
  } catch (error) {
    logger.error({ err: error }, 'Chat completed error');
    res.status(500).json({ detail: error.message });
  }
});

// Chat actions (no-op - we don't have custom actions)
app.post('/api/chat/actions/:action_id', authenticateToken, (req, res) => {
  res.json({ messages: [] });
});

// ==================== Tasks API ====================

// Helper: call Anthropic for task completions
async function callAnthropicForTask(systemPrompt, userContent, maxTokens = 200) {
  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01'
    },
    body: JSON.stringify({
      model: 'claude-sonnet-4-20250514',
      max_tokens: maxTokens,
      system: systemPrompt,
      messages: [{ role: 'user', content: userContent }]
    })
  });
  
  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Anthropic API error ${response.status}: ${errText}`);
  }
  
  const data = await response.json();
  const content = data.content?.[0]?.text || '';
  
  // Return in OpenAI-compatible format that the frontend expects
  return {
    id: 'chatcmpl-' + Date.now(),
    choices: [{
      index: 0,
      message: { role: 'assistant', content },
      finish_reason: 'stop'
    }],
    usage: {
      prompt_tokens: data.usage?.input_tokens || 0,
      completion_tokens: data.usage?.output_tokens || 0,
      total_tokens: (data.usage?.input_tokens || 0) + (data.usage?.output_tokens || 0)
    }
  };
}

// Task config
let taskConfig = {
  TASK_MODEL: '',
  TASK_MODEL_EXTERNAL: '',
  TITLE_GENERATION_PROMPT_TEMPLATE: 'Create a concise, 3-5 word title for this chat. Return ONLY a JSON object: {"title": "Your Title Here"}',
  TAGS_GENERATION_PROMPT_TEMPLATE: 'Generate 1-3 short tags for this chat. Return ONLY a JSON object: {"tags": ["tag1", "tag2"]}',
  ENABLE_TITLE_GENERATION: true,
  ENABLE_TAGS_GENERATION: true,
  ENABLE_FOLLOW_UP_GENERATION: true,
  ENABLE_AUTOCOMPLETE_GENERATION: false
};

app.get('/api/v1/tasks/config', authenticateToken, (req, res) => {
  res.json(taskConfig);
});

app.post('/api/v1/tasks/config/update', authenticateToken, (req, res) => {
  taskConfig = { ...taskConfig, ...req.body };
  res.json(taskConfig);
});

// Generate title for chat
app.post('/api/v1/tasks/title/completions', authenticateToken, async (req, res) => {
  try {
    const { model, messages, chat_id } = req.body;
    
    // Build a summary of the conversation for title generation
    const conversationSummary = (messages || [])
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .slice(0, 4)
      .map(m => `${m.role}: ${typeof m.content === 'string' ? m.content.substring(0, 200) : JSON.stringify(m.content).substring(0, 200)}`)
      .join('\n');
    
    const result = await callAnthropicForTask(
      'You generate short, concise chat titles. Return ONLY a JSON object with a "title" key. No other text.',
      `Generate a concise 3-5 word title for this conversation:\n\n${conversationSummary}\n\nReturn ONLY: {"title": "Your Title"}`
    );
    
    res.json(result);
  } catch (error) {
    logger.error({ err: error }, 'Title generation error');
    // Return a fallback so the frontend doesn't break
    res.json({
      choices: [{ message: { role: 'assistant', content: '{"title": "New Chat"}' } }]
    });
  }
});

// Generate tags for chat
app.post('/api/v1/tasks/tags/completions', authenticateToken, async (req, res) => {
  try {
    const { model, messages, chat_id } = req.body;
    
    const conversationSummary = (messages || [])
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .slice(0, 4)
      .map(m => `${m.role}: ${typeof m.content === 'string' ? m.content.substring(0, 200) : JSON.stringify(m.content).substring(0, 200)}`)
      .join('\n');
    
    const result = await callAnthropicForTask(
      'You generate short tags for chat conversations. Return ONLY a JSON object with a "tags" key containing an array of 1-3 lowercase tags. No other text.',
      `Generate 1-3 short tags for this conversation:\n\n${conversationSummary}\n\nReturn ONLY: {"tags": ["tag1", "tag2"]}`
    );
    
    res.json(result);
  } catch (error) {
    logger.error({ err: error }, 'Tags generation error');
    res.json({
      choices: [{ message: { role: 'assistant', content: '{"tags": []}' } }]
    });
  }
});

// Generate emoji for chat
app.post('/api/v1/tasks/emoji/completions', authenticateToken, async (req, res) => {
  try {
    const { model, prompt, chat_id } = req.body;
    
    const result = await callAnthropicForTask(
      'You respond with a single emoji that best represents the topic. Return ONLY the emoji character, nothing else.',
      `What single emoji best represents this topic: ${prompt || 'general chat'}`,
      10
    );
    
    res.json(result);
  } catch (error) {
    logger.error({ err: error }, 'Emoji generation error');
    res.json({
      choices: [{ message: { role: 'assistant', content: '💬' } }]
    });
  }
});

// Generate follow-up suggestions
app.post('/api/v1/tasks/follow_ups/completions', authenticateToken, async (req, res) => {
  try {
    const { model, messages, chat_id } = req.body;
    
    const conversationSummary = (typeof messages === 'string' ? messages : (messages || [])
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .slice(-4)
      .map(m => `${m.role}: ${typeof m.content === 'string' ? m.content.substring(0, 200) : ''}`)
      .join('\n'));
    
    const result = await callAnthropicForTask(
      'You suggest follow-up questions. Return ONLY a JSON object: {"follow_ups": ["question1", "question2", "question3"]}',
      `Based on this conversation, suggest 3 brief follow-up questions:\n\n${conversationSummary}\n\nReturn ONLY: {"follow_ups": ["q1", "q2", "q3"]}`
    );
    
    res.json(result);
  } catch (error) {
    logger.error({ err: error }, 'Follow-ups generation error');
    res.json({
      choices: [{ message: { role: 'assistant', content: '{"follow_ups": []}' } }]
    });
  }
});

// Queries completions (search query generation - no-op)
app.post('/api/v1/tasks/queries/completions', authenticateToken, (req, res) => {
  res.json({
    choices: [{ message: { role: 'assistant', content: '{"queries": []}' } }]
  });
});

// Auto completions (autocomplete suggestions - no-op)
app.post('/api/v1/tasks/auto/completions', authenticateToken, (req, res) => {
  res.json({
    choices: [{ message: { role: 'assistant', content: '' } }]
  });
});

// MOA completions (mixture of agents - no-op)
app.post('/api/v1/tasks/moa/completions', authenticateToken, (req, res) => {
  res.json({
    choices: [{ message: { role: 'assistant', content: '' } }]
  });
});

// Task management
app.get('/api/tasks/chat/:chat_id', authenticateToken, (req, res) => {
  res.json([]);
});

app.post('/api/tasks/stop/:id', authenticateToken, (req, res) => {
  res.json({ status: 'ok' });
});

// ==================== Catch-all for unimplemented endpoints ====================

// Ollama API stubs - this platform doesn't use Ollama, return proper JSON so frontend doesn't get HTML parse errors
app.all('/ollama/*', (req, res) => {
  res.status(404).json({ detail: 'Ollama is not available. This platform uses Amazon Bedrock.' });
});

app.all('/api/*', (req, res) => {
  res.status(501).json({ detail: 'Not implemented' });
});

// Error handler
app.use((err, req, res, next) => {
  logger.error({ err }, 'Unhandled server error');
  res.status(500).json({ detail: 'Internal server error' });
});

// Start server with Socket.IO
server.listen(PORT, () => {
  logger.info({
    port: PORT,
    env: NODE_ENV,
    users: usersDB.count(),
    ws: `/ws/socket.io`,
    health: `/health`,
  }, `ROKET backend started`);
});

// Graceful shutdown - flush pending DB writes, close connections
const gracefulShutdown = (signal) => {
  logger.info({ signal }, 'Shutting down gracefully...');

  // Stop accepting new connections
  server.close(() => {
    logger.info('HTTP server closed');
  });

  // Close all Socket.IO connections
  io.close(() => {
    logger.info('Socket.IO closed');
  });

  // Flush databases
  usersDB.saveSync();
  chatsDB.saveSync();
  configDB.saveSync();
  memoriesDB.saveSync();
  foldersDB.saveSync();
  filesDB.saveSync();
  logger.info('Database writes flushed');

  // Force exit after 10s if graceful shutdown hangs
  setTimeout(() => {
    logger.warn('Forced exit after timeout');
    process.exit(1);
  }, 10000).unref();

  process.exit(0);
};
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// Catch unhandled rejections and exceptions in production
process.on('unhandledRejection', (reason, promise) => {
  logger.error({ reason }, 'Unhandled promise rejection');
});
process.on('uncaughtException', (error) => {
  logger.fatal({ err: error }, 'Uncaught exception — shutting down');
  gracefulShutdown('uncaughtException');
});
