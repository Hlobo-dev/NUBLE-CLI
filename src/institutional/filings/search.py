"""
SEC Filings Search
==================

Semantic search over SEC filings using sentence transformers and DuckDB.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import os

from .database import FilingsDatabase, FilingChunk


@dataclass
class SearchResult:
    """A search result from the filings database"""
    chunk_id: str
    ticker: str
    form: str
    year: int
    quarter: int
    text: str
    similarity: float
    chunk_index: int = 0
    url: Optional[str] = None
    
    @property
    def source_label(self) -> str:
        """Human-readable source label"""
        if self.form == "10-Q":
            return f"{self.ticker} {self.form} Q{self.quarter} {self.year}"
        return f"{self.ticker} {self.form} {self.year}"


class FilingsSearch:
    """
    Semantic search over indexed SEC filings.
    
    Uses sentence-transformers embeddings stored in DuckDB for
    fast similarity search across filing content.
    """
    
    def __init__(
        self,
        database: FilingsDatabase = None,
        db_path: str = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize search engine.
        
        Args:
            database: Existing FilingsDatabase instance
            db_path: Path to create new database if not provided
            embedding_model: Sentence transformer model name
        """
        self.db = database or FilingsDatabase(
            db_path=db_path or "./filings_db",
            embedding_model=embedding_model,
        )
    
    def search(
        self,
        query: str,
        tickers: List[str] = None,
        forms: List[str] = None,
        years: List[int] = None,
        limit: int = 10,
        min_similarity: float = 0.3,
    ) -> List[SearchResult]:
        """
        Search filings with a natural language query.
        
        Args:
            query: Search query
            tickers: Filter by tickers
            forms: Filter by form types (10-K, 10-Q, etc.)
            years: Filter by fiscal years
            limit: Max results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of SearchResult objects
        """
        # For now, search one ticker/form at a time if specified
        ticker = tickers[0] if tickers else None
        form = forms[0] if forms else None
        year = years[0] if years else None
        
        # Perform vector search
        raw_results = self.db.search(
            query=query,
            ticker=ticker,
            form=form,
            year=year,
            k=limit,
        )
        
        # Convert to SearchResult objects, filter by min_similarity
        results = []
        for chunk in raw_results:
            if chunk.score >= min_similarity:
                results.append(SearchResult(
                    chunk_id=f"{chunk.ticker}_{chunk.form}_{chunk.year}_{chunk.chunk_index}",
                    ticker=chunk.ticker,
                    form=chunk.form,
                    year=chunk.year,
                    quarter=chunk.quarter,
                    text=chunk.chunk_text,
                    similarity=chunk.score,
                    chunk_index=chunk.chunk_index,
                    url=chunk.url,
                ))
        
        return results
    
    def search_by_keywords(
        self,
        keywords: List[str],
        tickers: List[str] = None,
        forms: List[str] = None,
        operator: str = "OR",
        limit: int = 20,
    ) -> List[SearchResult]:
        """
        Search using keyword matching.
        
        Uses semantic search with combined keywords as query.
        
        Args:
            keywords: List of keywords to search
            tickers: Filter by tickers
            forms: Filter by form types
            operator: AND or OR for combining keywords
            limit: Max results
            
        Returns:
            List of SearchResult objects
        """
        # Combine keywords into a search query
        if operator == "AND":
            query = " ".join(keywords)
        else:
            query = " OR ".join(keywords)
        
        return self.search(
            query=query,
            tickers=tickers,
            forms=forms,
            limit=limit,
        )
    
    def find_similar_sections(
        self,
        ticker: str,
        section_name: str,
        compare_tickers: List[str] = None,
        limit: int = 5,
    ) -> List[SearchResult]:
        """
        Find similar sections across filings.
        
        Useful for comparing risk factors, MD&A, etc. across companies.
        
        Args:
            ticker: Source ticker
            section_name: Section to find (e.g., "Risk Factors")
            compare_tickers: Tickers to compare against
            limit: Max results per ticker
            
        Returns:
            List of similar sections
        """
        # Search for the section type in source ticker first
        source_results = self.search(
            query=section_name,
            tickers=[ticker],
            limit=3,
        )
        
        if not source_results:
            return []
        
        # Combine source text
        source_text = " ".join(r.text for r in source_results[:3])
        
        # Search in comparison tickers
        tickers = compare_tickers or []
        tickers = [t for t in tickers if t.upper() != ticker.upper()]
        
        if not tickers:
            # Get all indexed tickers except source
            all_tickers = self.indexed_tickers
            tickers = [t for t in all_tickers if t.upper() != ticker.upper()][:10]
        
        return self.search(
            query=source_text[:1000],  # Limit query length
            tickers=tickers,
            limit=limit,
            min_similarity=0.4,
        )
    
    def get_filing_context(
        self,
        chunk_id: str,
        context_chunks: int = 2,
    ) -> Dict[str, Any]:
        """
        Get extended context around a search result.
        
        Args:
            chunk_id: The chunk ID to get context for
            context_chunks: Number of chunks before/after
            
        Returns:
            Dict with full context
        """
        # Parse chunk_id to get ticker, form, year, chunk_index
        parts = chunk_id.split("_")
        if len(parts) >= 4:
            ticker = parts[0]
            form = parts[1]
            year = int(parts[2])
            chunk_index = int(parts[3])
            
            # Get surrounding chunks
            results = self.db.search(
                query="",  # Empty query to get all
                ticker=ticker,
                form=form,
                year=year,
                k=100,
            )
            
            # Find context
            context_before = []
            context_after = []
            main_chunk = None
            
            for chunk in results:
                if chunk.chunk_index == chunk_index:
                    main_chunk = chunk
                elif chunk.chunk_index >= chunk_index - context_chunks and chunk.chunk_index < chunk_index:
                    context_before.append(chunk)
                elif chunk.chunk_index <= chunk_index + context_chunks and chunk.chunk_index > chunk_index:
                    context_after.append(chunk)
            
            return {
                "main": main_chunk.chunk_text if main_chunk else "",
                "before": [c.chunk_text for c in sorted(context_before, key=lambda x: x.chunk_index)],
                "after": [c.chunk_text for c in sorted(context_after, key=lambda x: x.chunk_index)],
            }
        
        return {"main": "", "before": [], "after": []}
    
    def aggregate_by_topic(
        self,
        query: str,
        tickers: List[str],
        forms: List[str] = None,
    ) -> Dict[str, List[SearchResult]]:
        """
        Search and aggregate results by ticker.
        
        Useful for comparative analysis across companies.
        
        Args:
            query: Search query
            tickers: Tickers to search
            forms: Form types to filter
            
        Returns:
            Dict mapping ticker -> search results
        """
        aggregated = {}
        
        for ticker in tickers:
            results = self.search(
                query=query,
                tickers=[ticker],
                forms=forms,
                limit=5,
            )
            if results:
                aggregated[ticker] = results
        
        return aggregated
    
    @property
    def indexed_tickers(self) -> List[str]:
        """Get list of indexed tickers"""
        filings = self.db.list_filings()
        return list(set(f.ticker for f in filings))
    
    @property
    def indexed_count(self) -> int:
        """Get total number of indexed chunks"""
        stats = self.db.get_stats()
        return stats.get("total_chunks", 0)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the search index.
        
        Returns:
            Dict with index statistics
        """
        stats = self.db.get_stats()
        return {
            "total_chunks": stats.get("total_chunks", 0),
            "indexed_tickers": self.indexed_tickers,
            "ticker_count": stats.get("unique_tickers", 0),
            "embedding_model": stats.get("embedding_model", ""),
            "db_path": stats.get("db_path", ""),
        }
