<div align="center">

<img src="static/banner.jpg" alt="tenk banner" width="100%">

# [tenk](https://rallies.ai)

**Talk to SEC filings with AI**

*Ask questions about 10-K and 10-Q filings, get answers with citations*

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Personal%20Use-blue.svg)](LICENSE)
[![Rallies.ai](https://img.shields.io/badge/Built%20by-Rallies.ai-ff6b6b.svg)](https://rallies.ai)

</div>

## What is tenk?

tenk lets you have a conversation with SEC filings. Instead of manually reading through hundreds of pages of 10-K and 10-Q reports, just ask questions in plain English and get answers with direct citations to the source.

<div align="center">

<img src="static/demo.gif" alt="tenk demo" width="100%">

</div>

#### Why not just use ChatGPT?

You could copy-paste filings into ChatGPT, but:

- 10-Ks are 100+ pages - you can't paste them all
- You definitely can't paste multiple filings to compare companies
- You'd have to manually find and download each filing first

tenk automates all of that. It fetches filings from SEC EDGAR, indexes them locally, and lets you query across multiple filings at once.

## Features

- ✅ **RAG** over SEC filings with a local vector database (10-K, 10-Q)
- ✅ **Auto-downloads** filings from SEC EDGAR
- ✅ **Citations** with links to source documents
- ✅ **Stock data** via Yahoo Finance
- ✅ **Web search** for data not in filings
- ✅ **Excel generation** for tables and financial models
- ✅ **Export** answers to PDF, DOCX, or Excel
- ✅ **Conversation memory** with auto-summarization

## Requirements

- Python 3.10+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)

## Installation

```bash
pip install tenk
```

Or from source:

```bash
git clone https://github.com/rallies-ai/tenk.git
cd tenk
pip install -e .
```

## Usage

**Interactive mode:**

```bash
tenk
```

**One-shot mode:**

```bash
tenk "What are Apple's main risk factors?"
```

## Examples

```
> What are NVIDIA's revenue segments?

> Compare AAPL and MSFT gross margins for 2024

> What does Tesla say about competition in their latest 10-K?

> Create an Excel file with Amazon's quarterly revenue for the past 2 years
```

## How it works

1. **Check available filings** - Queries SEC EDGAR to see what's available
2. **Download & index** - Downloads filings and chunks them into a local vector database
3. **Semantic search** - Finds relevant passages using sentence-transformers
4. **AI analysis** - Generates answers with citations
5. **Export** - Optionally export to PDF, DOCX, or Excel

## Configuration

Config file is at `src/config.yaml`:

```yaml
model:
  name: gpt-5.2
  max_turns: 50

search:
  top_k: 10
  chunk_size: 3000
  chunk_overlap: 500

embeddings:
  model: all-MiniLM-L6-v2

edgar:
  identity: "Your Name your@email.com"  # Required by SEC EDGAR

database:
  path: db/filings.db

exports:
  dir: exports
```

**Note:** SEC EDGAR requires a valid user-agent identity. Update the `edgar.identity` field with your name and email.

## Data Storage

- **Filings database:** `db/filings.db` (DuckDB)
- **Exports:** `exports/` directory

Both are created automatically in your current working directory.

## License

Personal Use License - free for personal and non-commercial use. Commercial use prohibited. See [LICENSE](LICENSE) for details.

---

<div align="center">
  Built with ❤️ by <a href="https://rallies.ai">Rallies.ai</a>
</div>
