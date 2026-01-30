"""
SEC Filings Module - TENK Integration
======================================

AI-powered SEC filings analysis using Claude Opus 4.5 and semantic search.
Enhanced for KYPERIAN institutional research.

Features:
- Download and index SEC filings (10-K, 10-Q, 8-K, 13F, etc.)
- Semantic search with sentence transformers
- Claude Opus 4.5 powered analysis
- Vector database with DuckDB
- Export to PDF, DOCX, Excel

This module integrates the TENK project's core functionality with our
institutional research platform, replacing OpenAI with Claude Opus 4.5
for superior analysis.
"""

from .database import FilingsDatabase, FilingChunk, FilingMetadata
from .search import FilingsSearch, SearchResult
from .loader import FilingsLoader, LoadedFiling, FilingInfo, FilingForm
from .analyzer import FilingsAnalyzer, AnalysisType, AnalysisResult
from .agent import SECFilingsAgent, AgentMessage, AgentState
from .export import FilingsExporter, ExportResult

__all__ = [
    # Database
    "FilingsDatabase",
    "FilingChunk",
    "FilingMetadata",
    # Search
    "FilingsSearch", 
    "SearchResult",
    # Loader
    "FilingsLoader",
    "LoadedFiling",
    "FilingInfo",
    "FilingForm",
    # Analyzer
    "FilingsAnalyzer",
    "AnalysisType",
    "AnalysisResult",
    # Agent
    "SECFilingsAgent",
    "AgentMessage",
    "AgentState",
    # Export
    "FilingsExporter",
    "ExportResult",
]
