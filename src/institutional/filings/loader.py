"""
SEC Filings Loader
==================

Download and parse SEC filings from EDGAR using edgartools.
Supports 10-K, 10-Q, 8-K, 13F, and other filing types.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum

try:
    from edgar import Company, set_identity
    EDGAR_AVAILABLE = True
except ImportError:
    EDGAR_AVAILABLE = False


class FilingForm(Enum):
    """SEC filing form types"""
    ANNUAL_10K = "10-K"
    QUARTERLY_10Q = "10-Q"
    CURRENT_8K = "8-K"
    PROXY = "DEF 14A"
    INSIDER_HOLDINGS = "13F-HR"
    BENEFICIAL_OWNERSHIP = "SC 13D"
    BENEFICIAL_OWNERSHIP_G = "SC 13G"
    REGISTRATION = "S-1"
    AMENDMENT_10KA = "10-K/A"
    AMENDMENT_10QA = "10-Q/A"


@dataclass
class FilingInfo:
    """Information about an available SEC filing"""
    ticker: str
    form: str
    year: int
    quarter: Optional[int]
    report_date: date
    filing_date: Optional[date]
    url: Optional[str]
    loaded: bool = False


@dataclass
class LoadedFiling:
    """A downloaded and parsed SEC filing"""
    ticker: str
    form: str
    year: int
    quarter: int
    text: str
    url: Optional[str]
    report_date: date
    filing_date: Optional[date]
    sections: Dict[str, str] = None
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        return len(self.text)


class FilingsLoader:
    """
    Load SEC filings from EDGAR.
    
    Uses edgartools library to fetch filings from the SEC EDGAR database.
    """
    
    # Default identity for SEC EDGAR API
    DEFAULT_IDENTITY = "NUBLE Research research@nuble.ai"
    
    def __init__(self, identity: str = None):
        """
        Initialize the filings loader.
        
        Args:
            identity: Email/name for SEC EDGAR API (required by SEC)
        """
        if not EDGAR_AVAILABLE:
            raise ImportError("edgar not installed. Run: pip install edgartools")
        
        # Set identity for EDGAR API
        set_identity(identity or self.DEFAULT_IDENTITY)
    
    def check_available(
        self,
        tickers: List[str],
        forms: List[str] = None,
        limit: int = 10,
    ) -> Dict[str, List[FilingInfo]]:
        """
        Check what filings are available on SEC EDGAR.
        
        Args:
            tickers: List of stock tickers to check
            forms: Form types to check (default: 10-K, 10-Q)
            limit: Max filings per form type
            
        Returns:
            Dict mapping ticker -> list of FilingInfo
        """
        forms = forms or ["10-K", "10-Q"]
        results = {}
        
        for ticker in tickers:
            ticker = ticker.upper()
            results[ticker] = []
            
            try:
                company = Company(ticker)
                
                for form in forms:
                    filings = list(company.get_filings(form=form))[:limit]
                    
                    for f in filings:
                        report_date = getattr(f, 'report_date', None)
                        if not report_date:
                            continue
                        
                        # Parse date
                        parts = str(report_date).split("-")
                        if len(parts) < 2:
                            continue
                        
                        year = int(parts[0])
                        month = int(parts[1])
                        quarter = self._month_to_quarter(month)
                        
                        filing_date = getattr(f, 'filing_date', None)
                        url = getattr(f, 'url', None)
                        
                        results[ticker].append(FilingInfo(
                            ticker=ticker,
                            form=form,
                            year=year,
                            quarter=quarter if form == "10-Q" else None,
                            report_date=report_date if isinstance(report_date, date) else date.fromisoformat(str(report_date)),
                            filing_date=filing_date if isinstance(filing_date, date) else None,
                            url=url,
                            loaded=False,
                        ))
                        
            except Exception as e:
                results[ticker] = [{"error": str(e)}]
        
        return results
    
    def load_filing(
        self,
        ticker: str,
        form: str,
        year: int,
        quarter: int = None,
    ) -> Optional[LoadedFiling]:
        """
        Download a specific SEC filing.
        
        Args:
            ticker: Stock ticker
            form: Filing form type (10-K, 10-Q, etc.)
            year: Fiscal year
            quarter: Quarter for 10-Q (1-4)
            
        Returns:
            LoadedFiling object or None if not found
        """
        ticker = ticker.upper()
        
        # Validate quarter for 10-Q
        if form == "10-Q" and quarter not in [1, 2, 3, 4]:
            raise ValueError("quarter must be 1-4 for 10-Q filings")
        
        try:
            company = Company(ticker)
            filings = list(company.get_filings(form=form))
            
            for f in filings:
                report_date = getattr(f, 'report_date', None)
                if not report_date:
                    continue
                
                parts = str(report_date).split("-")
                if len(parts) < 2:
                    continue
                
                f_year = int(parts[0])
                f_month = int(parts[1])
                f_quarter = self._month_to_quarter(f_month)
                
                # Match year and quarter
                if form == "10-K" and f_year == year:
                    text = f.text()
                    if text and len(text.strip()) > 100:
                        return LoadedFiling(
                            ticker=ticker,
                            form=form,
                            year=year,
                            quarter=0,
                            text=text,
                            url=getattr(f, 'url', None),
                            report_date=report_date if isinstance(report_date, date) else date.fromisoformat(str(report_date)),
                            filing_date=getattr(f, 'filing_date', None),
                        )
                
                elif form == "10-Q" and f_year == year and f_quarter == quarter:
                    text = f.text()
                    if text and len(text.strip()) > 100:
                        return LoadedFiling(
                            ticker=ticker,
                            form=form,
                            year=year,
                            quarter=quarter,
                            text=text,
                            url=getattr(f, 'url', None),
                            report_date=report_date if isinstance(report_date, date) else date.fromisoformat(str(report_date)),
                            filing_date=getattr(f, 'filing_date', None),
                        )
            
            return None
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {ticker} {form} {year}: {e}")
    
    def load_latest(
        self,
        ticker: str,
        form: str = "10-K",
    ) -> Optional[LoadedFiling]:
        """
        Load the most recent filing of a given type.
        
        Args:
            ticker: Stock ticker
            form: Filing form type
            
        Returns:
            LoadedFiling or None
        """
        ticker = ticker.upper()
        
        try:
            company = Company(ticker)
            filings = list(company.get_filings(form=form))
            
            if not filings:
                return None
            
            # Get the most recent
            f = filings[0]
            report_date = getattr(f, 'report_date', None)
            
            if not report_date:
                return None
            
            parts = str(report_date).split("-")
            year = int(parts[0])
            month = int(parts[1])
            quarter = self._month_to_quarter(month)
            
            text = f.text()
            if text and len(text.strip()) > 100:
                return LoadedFiling(
                    ticker=ticker,
                    form=form,
                    year=year,
                    quarter=quarter if form == "10-Q" else 0,
                    text=text,
                    url=getattr(f, 'url', None),
                    report_date=report_date if isinstance(report_date, date) else date.fromisoformat(str(report_date)),
                    filing_date=getattr(f, 'filing_date', None),
                )
            
            return None
            
        except Exception as e:
            raise RuntimeError(f"Failed to load latest {ticker} {form}: {e}")
    
    def load_multiple(
        self,
        ticker: str,
        form: str = "10-K",
        years: List[int] = None,
        count: int = 3,
    ) -> List[LoadedFiling]:
        """
        Load multiple filings for trend analysis.
        
        Args:
            ticker: Stock ticker
            form: Filing form type
            years: Specific years to load
            count: Number of recent filings if years not specified
            
        Returns:
            List of LoadedFiling objects
        """
        ticker = ticker.upper()
        results = []
        
        try:
            company = Company(ticker)
            filings = list(company.get_filings(form=form))[:count * 2]  # Buffer
            
            loaded = 0
            for f in filings:
                if loaded >= count:
                    break
                
                report_date = getattr(f, 'report_date', None)
                if not report_date:
                    continue
                
                parts = str(report_date).split("-")
                year = int(parts[0])
                
                # Filter by years if specified
                if years and year not in years:
                    continue
                
                month = int(parts[1])
                quarter = self._month_to_quarter(month)
                
                text = f.text()
                if text and len(text.strip()) > 100:
                    results.append(LoadedFiling(
                        ticker=ticker,
                        form=form,
                        year=year,
                        quarter=quarter if form == "10-Q" else 0,
                        text=text,
                        url=getattr(f, 'url', None),
                        report_date=report_date if isinstance(report_date, date) else date.fromisoformat(str(report_date)),
                        filing_date=getattr(f, 'filing_date', None),
                    ))
                    loaded += 1
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Failed to load multiple filings: {e}")
    
    def _month_to_quarter(self, month: int) -> int:
        """Convert month to fiscal quarter"""
        if month <= 3:
            return 1
        elif month <= 6:
            return 2
        elif month <= 9:
            return 3
        else:
            return 4
    
    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get basic company information from SEC.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Dict with company info
        """
        try:
            company = Company(ticker.upper())
            
            return {
                "ticker": ticker.upper(),
                "name": getattr(company, 'name', None),
                "cik": getattr(company, 'cik', None),
                "sic": getattr(company, 'sic', None),
                "sic_description": getattr(company, 'sic_description', None),
                "state": getattr(company, 'state', None),
                "fiscal_year_end": getattr(company, 'fiscal_year_end', None),
            }
        except Exception as e:
            return {"error": str(e), "ticker": ticker.upper()}
