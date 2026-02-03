<div align="center">
  
  # 🔮 NUBLE
  
  **Institutional-Grade AI Investment Research Platform**
  
  *Powered by Claude Opus 4.5 • Real-Time Market Data • SEC Filings Analysis*

  [![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
  [![Claude](https://img.shields.io/badge/AI-Claude%20Opus%204.5-purple.svg)](https://anthropic.com)

  ![NUBLE Demo](demo/demo.png) 
  
</div>

## ✨ What is NUBLE?

NUBLE is an institutional-grade AI investment research platform that combines the power of Claude Opus 4.5 with real-time financial data, SEC filings analysis, and advanced machine learning. Built for professional traders, analysts, and investors who demand the best.

### Key Features

- 🤖 **Claude Opus 4.5 Integration** - The most advanced AI for financial analysis
- 📊 **Real-Time Market Data** - Live prices, volume, technicals via Polygon.io
- 📑 **SEC Filings Analysis** - Deep analysis of 10-K, 10-Q, 8-K with semantic search
- 🧠 **Machine Learning** - Transformers, LSTM, ensemble models for prediction
- 🔍 **Multi-Agent System** - Specialized AI agents for research, trading, risk
- 📈 **Technical Analysis** - 50+ indicators, pattern recognition, anomaly detection

## 🚀 Quick Start

### Installation

\`\`\`bash
# Clone the repository
git clone https://github.com/Hlobo-dev/NUBLE-CLI.git
cd NUBLE-CLI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install
pip install -e .
\`\`\`

### Setup

1. **Get your Anthropic API key** from [Anthropic Console](https://console.anthropic.com/)
2. **Create \`.env\` file**:
   \`\`\`bash
   echo 'ANTHROPIC_API_KEY=sk-ant-your-key-here' > .env
   \`\`\`
3. **Launch NUBLE**:
   \`\`\`bash
   nuble
   \`\`\`

## 💡 Example Queries

\`\`\`
> What happened to Tesla stock today?
> Analyze Apple's risk factors from their latest 10-K
> Compare NVIDIA and AMD technical indicators
> Find stocks with unusual options activity
> What are the key risks in Microsoft's SEC filings?
\`\`\`

## 🏗️ Architecture

\`\`\`
src/
├── nuble/         # CLI Application
│   ├── cli.py        # Interactive shell
│   ├── llm.py        # Claude/OpenAI integration
│   ├── manager.py    # Query orchestration
│   └── agent/        # AI agent logic
│
├── institutional/    # Institutional Platform
│   ├── filings/      # SEC Filings (TENK integration)
│   ├── ml/           # Machine Learning models
│   ├── analytics/    # Technical & sentiment analysis
│   ├── providers/    # Data providers
│   └── agents/       # Multi-agent system
│
└── TENK_SOURCE/      # SEC Filings reference
\`\`\`

## 🔐 API Keys Required

| Provider | Purpose | Get Key |
|----------|---------|---------|
| Anthropic | Claude AI (primary) | [console.anthropic.com](https://console.anthropic.com) |
| Polygon.io | Market data | [polygon.io](https://polygon.io) |

## 📦 Dependencies

- Python 3.8+
- anthropic (Claude API)
- rich (Terminal UI)
- duckdb (Vector database)
- sentence-transformers (Embeddings)
- edgartools (SEC EDGAR)
- torch (Deep Learning)

## 📄 License

GPL-3.0 License - See [LICENSE](LICENSE) for details.

---

<div align="center">
  <b>NUBLE</b> - Institutional-Grade AI Investment Research
</div>
