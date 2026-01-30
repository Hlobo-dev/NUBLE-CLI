import os
import yaml
import yfinance as yf
from agents import function_tool
from edgar import Company
from src import db
from src import terminal as term

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)


@function_tool
def load_filing(ticker: str, form: str, year: int, quarter: int = None) -> dict:
    """
    Download and index a single SEC filing.

    Args:
        ticker: Stock ticker (e.g., "AAPL")
        form: "10-K" for annual or "10-Q" for quarterly
        year: Fiscal year
        quarter: 1-4 for 10-Q (ignored for 10-K)
    """
    ticker = ticker.upper()

    if form == "10-Q":
        if quarter not in [1, 2, 3, 4]:
            return {"error": "quarter must be 1, 2, 3, or 4 for 10-Q"}
        q = quarter
    else:
        q = 0

    if db.has_filing(ticker, form, year, q):
        return {"status": "already_loaded", "ticker": ticker, "form": form, "year": year, "quarter": quarter if form == "10-Q" else None}

    term.print_dim(f"Fetching {ticker} {form} {year}{f' Q{quarter}' if quarter else ''}...")
    company = Company(ticker)
    filings = list(company.get_filings(form=form))

    for f in filings:
        report_date = getattr(f, 'report_date', None)
        if not report_date:
            continue

        parts = str(report_date).split("-")
        f_year = int(parts[0])
        f_month = int(parts[1])
        f_quarter = 1 if f_month <= 3 else 2 if f_month <= 6 else 3 if f_month <= 9 else 4

        if form == "10-K" and f_year == year:
            term.print_dim(f"Downloading {ticker} {form} {year}...")
            text = f.text()
            if text and len(text.strip()) > 100:
                db.add_filing(ticker, form, year, 0, text, getattr(f, 'url', None))
                return {"status": "loaded", "ticker": ticker, "form": form, "year": year}

        if form == "10-Q" and f_year == year and f_quarter == quarter:
            term.print_dim(f"Downloading {ticker} {form} {year} Q{quarter}...")
            text = f.text()
            if text and len(text.strip()) > 100:
                db.add_filing(ticker, form, year, quarter, text, getattr(f, 'url', None))
                return {"status": "loaded", "ticker": ticker, "form": form, "year": year, "quarter": quarter}

    return {"status": "not_found", "ticker": ticker, "form": form, "year": year, "quarter": quarter if form == "10-Q" else None}


@function_tool
def search(queries: list[str], ticker: str, year: int = None, quarter: int = None) -> dict:
    """
    Semantic search over loaded SEC filings (RAG retrieval).

    Args:
        queries: List of natural language queries - use varied phrasings for better results
                 (e.g., ["risk factors", "Item 1A", "competitive threats"])
        ticker: Stock ticker
        year: Optional - filter to specific year
        quarter: Optional - filter to specific quarter (0 for 10-K, 1-4 for 10-Q)

    Tips: Pass multiple query variations to get better coverage. Use section names like
    "Item 1A Risk Factors", "Item 7 MD&A", "Item 1 Business" for targeted search.
    """
    output = {}
    for query in queries:
        results = db.search(query, k=config["search"]["top_k"], ticker=ticker.upper(), year=year, quarter=quarter)
        output[query] = []
        for r in results:
            q = r["quarter"]
            output[query].append({
                "text": r["text"],
                "ticker": r["ticker"],
                "form": r["form"],
                "year": r["year"],
                "quarter": q if q != 0 else None,
                "score": round(r["score"], 3),
                "url": r["url"]
            })
    return output


@function_tool
def list_loaded(ticker: str = None) -> list[dict]:
    """
    List filings in the local database.

    Args:
        ticker: Optional - filter by ticker, or None for all
    """
    filings = db.list_filings(ticker=ticker.upper() if ticker else None)
    output = []
    for f in filings:
        q = f["quarter"]
        output.append({
            "ticker": f["ticker"],
            "form": f["form"],
            "year": f["year"],
            "quarter": q if q != 0 else None,
            "chunks": f["chunks"]
        })
    return output


@function_tool
def check_available(tickers: list[str]) -> dict:
    """
    Check what filings are available on SEC EDGAR for one or more companies.

    Args:
        tickers: List of stock tickers (e.g., ["AAPL", "TSLA"])
    """
    results = {}

    for ticker in tickers:
        ticker = ticker.upper()
        term.print_dim(f"Checking SEC EDGAR for {ticker}...")
        company = Company(ticker)
        results[ticker] = []

        for form in ["10-K", "10-Q"]:
            filings = list(company.get_filings(form=form))[:5]
            for f in filings:
                report_date = getattr(f, 'report_date', None)
                if not report_date:
                    continue
                parts = str(report_date).split("-")
                if len(parts) < 2:
                    continue
                year = int(parts[0])
                month = int(parts[1])
                q = 1 if month <= 3 else 2 if month <= 6 else 3 if month <= 9 else 4

                loaded = db.has_filing(ticker, form, year, q)

                results[ticker].append({
                    "form": form,
                    "year": year,
                    "quarter": q if form == "10-Q" else None,
                    "report_date": str(report_date),
                    "loaded": loaded
                })

    return results


@function_tool
def get_stock_price(ticker: str) -> dict:
    """
    Get latest stock price and key metrics for a ticker using Yahoo Finance.

    Args:
        ticker: Stock ticker (e.g., "AAPL")
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        return {
            "ticker": ticker.upper(),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "previous_close": info.get("previousClose"),
            "open": info.get("open"),
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "volume": info.get("volume"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "currency": info.get("currency", "USD")
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker.upper()}
