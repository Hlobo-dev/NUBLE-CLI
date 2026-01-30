import duckdb
import numpy as np
import threading
import os
import yaml
from sentence_transformers import SentenceTransformer
from edgar import set_identity

from src import terminal as term

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

term.display_banner()

loading_done = False
def done_flag():
    return loading_done

loading_thread = threading.Thread(target=term.animate_loading, args=(done_flag,))
loading_thread.start()

set_identity(config["edgar"]["identity"])

db_path = config["database"]["path"]
if not os.path.isabs(db_path):
    db_path = os.path.join(os.getcwd(), db_path)
os.makedirs(os.path.dirname(db_path), exist_ok=True)

conn = duckdb.connect(db_path)
model = SentenceTransformer(config["embeddings"]["model"])

loading_done = True
loading_thread.join()

def show_ready_message(has_query=False):
    if has_query:
        term.show_working()
    else:
        term.show_ready()


def init_db():
    conn.execute("""
        CREATE TABLE IF NOT EXISTS filings (
            ticker VARCHAR,
            form VARCHAR,
            year INTEGER,
            quarter INTEGER,
            chunk_index INTEGER,
            chunk_text TEXT,
            embedding FLOAT[384],
            url VARCHAR,
            PRIMARY KEY (ticker, form, year, quarter, chunk_index)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_filing ON filings (ticker, form, year, quarter)")

init_db()


def chunk_text(text: str) -> list[str]:
    chunk_size = config["search"]["chunk_size"]
    overlap = config["search"]["chunk_overlap"]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


def has_filing(ticker: str, form: str, year: int, quarter: int) -> bool:
    result = conn.execute("""
        SELECT 1 FROM filings
        WHERE ticker = ? AND form = ? AND year = ? AND quarter = ?
        LIMIT 1
    """, [ticker.upper(), form, year, quarter]).fetchone()
    return result is not None


def add_filing(ticker: str, form: str, year: int, quarter: int, text: str, url: str = None):
    ticker = ticker.upper()
    if has_filing(ticker, form, year, quarter):
        return

    term.print_dim(f"Indexing {ticker} {form} {year}...")
    chunks = chunk_text(text)
    embeddings = model.encode(chunks, show_progress_bar=False)
    conn.executemany(
        "INSERT INTO filings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(ticker, form, year, quarter, i, chunk, emb.tolist(), url)
         for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
    )
    term.console.print(f"[green]✓[/green] Indexed {len(chunks)} chunks for {ticker} {form} {year}")


def search(query: str, k: int = 5, ticker: str = None, form: str = None, year: int = None, quarter: int = None) -> list[dict]:
    where = []
    params = []

    if ticker:
        where.append("ticker = ?")
        params.append(ticker.upper())
    if form:
        where.append("form = ?")
        params.append(form)
    if year:
        where.append("year = ?")
        params.append(year)
    if quarter is not None:
        where.append("quarter = ?")
        params.append(quarter)

    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    rows = conn.execute(f"""
        SELECT chunk_text, ticker, form, year, quarter, embedding, url
        FROM filings {where_clause}
    """, params).fetchall()

    if not rows:
        return []

    texts = [r[0] for r in rows]
    meta = [{"ticker": r[1], "form": r[2], "year": r[3], "quarter": r[4], "url": r[6]} for r in rows]
    embeddings = np.array([r[5] for r in rows])

    query_emb = model.encode(query)
    sims = np.dot(embeddings, query_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
    )

    top_k = np.argsort(sims)[-k:][::-1]
    return [{"text": texts[i], **meta[i], "score": float(sims[i])} for i in top_k]


def list_filings(ticker: str = None) -> list[dict]:
    if ticker:
        rows = conn.execute("""
            SELECT ticker, form, year, quarter, COUNT(*) as chunks
            FROM filings
            WHERE ticker = ?
            GROUP BY ticker, form, year, quarter
            ORDER BY year DESC, quarter DESC
        """, [ticker.upper()]).fetchall()
    else:
        rows = conn.execute("""
            SELECT ticker, form, year, quarter, COUNT(*) as chunks
            FROM filings
            GROUP BY ticker, form, year, quarter
            ORDER BY ticker, year DESC, quarter DESC
        """).fetchall()

    return [{"ticker": r[0], "form": r[1], "year": r[2], "quarter": r[3], "chunks": r[4]} for r in rows]
