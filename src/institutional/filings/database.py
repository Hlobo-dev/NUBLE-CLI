"""
SEC Filings Database Module
===========================

Vector database for SEC filings using DuckDB and sentence-transformers.
Stores filing chunks with embeddings for semantic search.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class FilingChunk:
    """A chunk of a SEC filing with embedding"""
    ticker: str
    form: str
    year: int
    quarter: int
    chunk_index: int
    chunk_text: str
    embedding: List[float] = field(default_factory=list)
    url: Optional[str] = None
    score: float = 0.0


@dataclass
class FilingMetadata:
    """Metadata about an indexed filing"""
    ticker: str
    form: str
    year: int
    quarter: int
    chunks: int
    indexed_at: datetime = field(default_factory=datetime.now)


class FilingsDatabase:
    """
    Vector database for SEC filings.
    
    Uses DuckDB for storage and sentence-transformers for embeddings.
    Supports semantic search over filing text.
    """
    
    # Default settings
    DEFAULT_DB_PATH = "~/.institutional/filings.db"
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_CHUNK_SIZE = 3000
    DEFAULT_CHUNK_OVERLAP = 500
    
    def __init__(
        self,
        db_path: str = None,
        embedding_model: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        """
        Initialize the filings database.
        
        Args:
            db_path: Path to DuckDB database file
            embedding_model: Sentence transformer model name
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between chunks
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("duckdb not installed. Run: pip install duckdb")
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        # Configuration
        self.db_path = Path(db_path or self.DEFAULT_DB_PATH).expanduser()
        self.embedding_model_name = embedding_model or self.DEFAULT_EMBEDDING_MODEL
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or self.DEFAULT_CHUNK_OVERLAP
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database and model
        self.conn = duckdb.connect(str(self.db_path))
        self._embedding_model = None  # Lazy load
        
        # Create tables
        self._init_db()
    
    @property
    def embedding_model(self) -> 'SentenceTransformer':
        """Lazy load embedding model"""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model
    
    def _init_db(self):
        """Initialize database tables"""
        # Main filings table with vector embeddings
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS filings (
                ticker VARCHAR,
                form VARCHAR,
                year INTEGER,
                quarter INTEGER,
                chunk_index INTEGER,
                chunk_text TEXT,
                embedding FLOAT[384],
                url VARCHAR,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, form, year, quarter, chunk_index)
            )
        """)
        
        # Index for faster queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_filing 
            ON filings (ticker, form, year, quarter)
        """)
        
        # Metadata table for tracking
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS filing_metadata (
                ticker VARCHAR,
                form VARCHAR,
                year INTEGER,
                quarter INTEGER,
                total_chunks INTEGER,
                source_url VARCHAR,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, form, year, quarter)
            )
        """)
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for embedding.
        
        Args:
            text: Full filing text
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)
            
            # Move forward with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def has_filing(
        self,
        ticker: str,
        form: str,
        year: int,
        quarter: int = 0,
    ) -> bool:
        """
        Check if a filing is already indexed.
        
        Args:
            ticker: Stock ticker
            form: Filing form type (10-K, 10-Q, etc.)
            year: Fiscal year
            quarter: Quarter (0 for annual, 1-4 for quarterly)
            
        Returns:
            True if filing exists in database
        """
        result = self.conn.execute("""
            SELECT 1 FROM filings
            WHERE ticker = ? AND form = ? AND year = ? AND quarter = ?
            LIMIT 1
        """, [ticker.upper(), form, year, quarter]).fetchone()
        
        return result is not None
    
    def add_filing(
        self,
        ticker: str,
        form: str,
        year: int,
        quarter: int,
        text: str,
        url: str = None,
    ) -> int:
        """
        Index a filing into the database.
        
        Args:
            ticker: Stock ticker
            form: Filing form type
            year: Fiscal year
            quarter: Quarter (0 for annual)
            text: Full filing text
            url: Optional source URL
            
        Returns:
            Number of chunks indexed
        """
        ticker = ticker.upper()
        
        # Skip if already indexed
        if self.has_filing(ticker, form, year, quarter):
            return 0
        
        # Chunk the text
        chunks = self.chunk_text(text)
        if not chunks:
            return 0
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
        
        # Insert chunks
        self.conn.executemany(
            "INSERT INTO filings VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            [
                (ticker, form, year, quarter, i, chunk, emb.tolist(), url)
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
            ]
        )
        
        # Add metadata
        self.conn.execute("""
            INSERT OR REPLACE INTO filing_metadata 
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [ticker, form, year, quarter, len(chunks), url])
        
        return len(chunks)
    
    def search(
        self,
        query: str,
        k: int = 10,
        ticker: str = None,
        form: str = None,
        year: int = None,
        quarter: int = None,
    ) -> List[FilingChunk]:
        """
        Semantic search over indexed filings.
        
        Args:
            query: Search query text
            k: Number of results to return
            ticker: Filter by ticker
            form: Filter by form type
            year: Filter by year
            quarter: Filter by quarter
            
        Returns:
            List of matching FilingChunk objects
        """
        # Build WHERE clause
        where_parts = []
        params = []
        
        if ticker:
            where_parts.append("ticker = ?")
            params.append(ticker.upper())
        if form:
            where_parts.append("form = ?")
            params.append(form)
        if year:
            where_parts.append("year = ?")
            params.append(year)
        if quarter is not None:
            where_parts.append("quarter = ?")
            params.append(quarter)
        
        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        
        # Fetch all matching rows
        rows = self.conn.execute(f"""
            SELECT chunk_text, ticker, form, year, quarter, chunk_index, embedding, url
            FROM filings {where_clause}
        """, params).fetchall()
        
        if not rows:
            return []
        
        # Compute similarities
        texts = [r[0] for r in rows]
        embeddings = np.array([r[6] for r in rows])
        
        query_embedding = self.embedding_model.encode(query)
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        
        # Get top-k
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for i in top_indices:
            row = rows[i]
            results.append(FilingChunk(
                ticker=row[1],
                form=row[2],
                year=row[3],
                quarter=row[4],
                chunk_index=row[5],
                chunk_text=row[0],
                embedding=[],  # Don't return embeddings to save memory
                url=row[7],
                score=float(similarities[i]),
            ))
        
        return results
    
    def multi_search(
        self,
        queries: List[str],
        ticker: str,
        k: int = 5,
        year: int = None,
        quarter: int = None,
    ) -> Dict[str, List[FilingChunk]]:
        """
        Search with multiple query variations for better RAG coverage.
        
        Args:
            queries: List of query variations
            ticker: Filter by ticker
            k: Results per query
            year: Filter by year
            quarter: Filter by quarter
            
        Returns:
            Dict mapping query -> results
        """
        results = {}
        for query in queries:
            results[query] = self.search(
                query=query,
                k=k,
                ticker=ticker,
                year=year,
                quarter=quarter,
            )
        return results
    
    def list_filings(self, ticker: str = None) -> List[FilingMetadata]:
        """
        List all indexed filings.
        
        Args:
            ticker: Optional filter by ticker
            
        Returns:
            List of FilingMetadata
        """
        if ticker:
            rows = self.conn.execute("""
                SELECT ticker, form, year, quarter, total_chunks, indexed_at
                FROM filing_metadata
                WHERE ticker = ?
                ORDER BY year DESC, quarter DESC
            """, [ticker.upper()]).fetchall()
        else:
            rows = self.conn.execute("""
                SELECT ticker, form, year, quarter, total_chunks, indexed_at
                FROM filing_metadata
                ORDER BY ticker, year DESC, quarter DESC
            """).fetchall()
        
        return [
            FilingMetadata(
                ticker=r[0],
                form=r[1],
                year=r[2],
                quarter=r[3],
                chunks=r[4],
                indexed_at=r[5],
            )
            for r in rows
        ]
    
    def delete_filing(
        self,
        ticker: str,
        form: str,
        year: int,
        quarter: int = 0,
    ) -> int:
        """
        Delete a filing from the database.
        
        Args:
            ticker: Stock ticker
            form: Filing form type
            year: Fiscal year
            quarter: Quarter
            
        Returns:
            Number of chunks deleted
        """
        ticker = ticker.upper()
        
        # Count chunks first
        result = self.conn.execute("""
            SELECT COUNT(*) FROM filings
            WHERE ticker = ? AND form = ? AND year = ? AND quarter = ?
        """, [ticker, form, year, quarter]).fetchone()
        
        count = result[0] if result else 0
        
        # Delete
        self.conn.execute("""
            DELETE FROM filings
            WHERE ticker = ? AND form = ? AND year = ? AND quarter = ?
        """, [ticker, form, year, quarter])
        
        self.conn.execute("""
            DELETE FROM filing_metadata
            WHERE ticker = ? AND form = ? AND year = ? AND quarter = ?
        """, [ticker, form, year, quarter])
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        total_chunks = self.conn.execute(
            "SELECT COUNT(*) FROM filings"
        ).fetchone()[0]
        
        total_filings = self.conn.execute(
            "SELECT COUNT(*) FROM filing_metadata"
        ).fetchone()[0]
        
        tickers = self.conn.execute(
            "SELECT COUNT(DISTINCT ticker) FROM filings"
        ).fetchone()[0]
        
        return {
            "total_chunks": total_chunks,
            "total_filings": total_filings,
            "unique_tickers": tickers,
            "db_path": str(self.db_path),
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
        }
    
    def close(self):
        """Close database connection"""
        self.conn.close()
