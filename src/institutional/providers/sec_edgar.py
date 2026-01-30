"""
SEC EDGAR Provider - Official SEC filings and financial data.
Authoritative source for 13F, 10-K, 10-Q, 8-K, Form 4, and XBRL data.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
from dataclasses import dataclass
import asyncio
import json
import re

from .base import (
    BaseProvider, ProviderResponse, DataType,
    Filing, InstitutionalHolding, InsiderTransaction, CompanyProfile
)


@dataclass
class SECCompanyInfo:
    """Company information from SEC"""
    cik: str
    name: str
    ticker: Optional[str] = None
    sic: Optional[str] = None
    sic_description: Optional[str] = None
    fiscal_year_end: Optional[str] = None
    state: Optional[str] = None
    state_of_incorporation: Optional[str] = None


class SECEdgarProvider(BaseProvider):
    """
    SEC EDGAR data provider.
    
    Features:
    - All SEC filings (10-K, 10-Q, 8-K, etc.)
    - 13F institutional holdings (authoritative source)
    - Form 4 insider transactions
    - XBRL standardized financial data
    - Company facts and submissions
    
    Note: SEC has fair access policy - max 10 requests/second
    """
    
    BASE_URL = "https://data.sec.gov"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions"
    COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts"
    COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    FULL_INDEX_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    
    def __init__(self, user_agent: str = "InstitutionalResearch/1.0 (research@example.com)", **kwargs):
        super().__init__(
            api_key=None,  # No API key required
            base_url=self.BASE_URL,
            rate_limit=10,  # SEC fair access policy
            **kwargs
        )
        self.user_agent = user_agent
        self._ticker_to_cik: Dict[str, str] = {}
        self._cik_to_ticker: Dict[str, str] = {}
        self._ticker_cache_loaded = False
    
    @property
    def name(self) -> str:
        return "sec_edgar"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [
            DataType.FILING,
            DataType.FUNDAMENTALS,
            DataType.HOLDINGS,
            DataType.TRANSACTIONS,
            DataType.PROFILE,
        ]
    
    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Host": "data.sec.gov"
        }
    
    async def _load_ticker_mapping(self):
        """Load ticker to CIK mapping from SEC"""
        if self._ticker_cache_loaded:
            return
        
        try:
            result = await self._request("GET", self.COMPANY_TICKERS_URL)
            if "data" in result:
                for _, company in result["data"].items():
                    ticker = company.get("ticker", "").upper()
                    cik = str(company.get("cik_str", "")).zfill(10)
                    if ticker and cik:
                        self._ticker_to_cik[ticker] = cik
                        self._cik_to_ticker[cik] = ticker
                self._ticker_cache_loaded = True
        except Exception as e:
            print(f"Warning: Could not load SEC ticker mapping: {e}")
    
    def _get_cik(self, symbol: str) -> Optional[str]:
        """Get CIK for a symbol"""
        return self._ticker_to_cik.get(symbol.upper())
    
    async def get_quote(self, symbol: str) -> ProviderResponse:
        """SEC doesn't provide real-time quotes"""
        return ProviderResponse(
            success=False,
            data=None,
            data_type=DataType.QUOTE,
            provider=self.name,
            symbol=symbol,
            error="SEC EDGAR does not provide real-time quotes"
        )
    
    async def get_historical(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeframe: str = "1d"
    ) -> ProviderResponse:
        """SEC doesn't provide price history"""
        return ProviderResponse(
            success=False,
            data=None,
            data_type=DataType.OHLCV,
            provider=self.name,
            symbol=symbol,
            error="SEC EDGAR does not provide price history"
        )
    
    async def get_company_submissions(self, symbol: str) -> ProviderResponse:
        """Get all submissions/filings for a company"""
        await self._load_ticker_mapping()
        
        cik = self._get_cik(symbol)
        if not cik:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.FILING,
                provider=self.name,
                symbol=symbol,
                error=f"Could not find CIK for symbol {symbol}"
            )
        
        try:
            result = await self._request(
                "GET",
                f"{self.SUBMISSIONS_URL}/CIK{cik}.json"
            )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.FILING,
                    provider=self.name,
                    symbol=symbol,
                    error=result["error"]
                )
            
            data = result["data"]
            company_info = SECCompanyInfo(
                cik=cik,
                name=data.get("name", ""),
                ticker=symbol,
                sic=data.get("sic"),
                sic_description=data.get("sicDescription"),
                fiscal_year_end=data.get("fiscalYearEnd"),
                state=data.get("stateOfIncorporation"),
            )
            
            # Parse recent filings
            filings_data = data.get("filings", {}).get("recent", {})
            filings = []
            
            if filings_data:
                forms = filings_data.get("form", [])
                accessions = filings_data.get("accessionNumber", [])
                filing_dates = filings_data.get("filingDate", [])
                primary_docs = filings_data.get("primaryDocument", [])
                descriptions = filings_data.get("primaryDocDescription", [])
                
                for i in range(min(100, len(forms))):
                    accession = accessions[i].replace("-", "")
                    filings.append(Filing(
                        accession_number=accessions[i],
                        form_type=forms[i],
                        filed_date=datetime.strptime(filing_dates[i], "%Y-%m-%d").date(),
                        cik=cik,
                        company_name=company_info.name,
                        symbol=symbol,
                        url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_docs[i]}",
                        description=descriptions[i] if i < len(descriptions) else None
                    ))
            
            return ProviderResponse(
                success=True,
                data={
                    "company_info": company_info,
                    "filings": filings
                },
                data_type=DataType.FILING,
                provider=self.name,
                symbol=symbol,
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.FILING,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_filings(
        self,
        symbol: str,
        form_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> ProviderResponse:
        """Get specific filing types for a company"""
        response = await self.get_company_submissions(symbol)
        
        if not response.success:
            return response
        
        filings = response.data.get("filings", [])
        
        # Filter by form type if specified
        if form_types:
            form_types_upper = [ft.upper() for ft in form_types]
            filings = [f for f in filings if f.form_type.upper() in form_types_upper]
        
        return ProviderResponse(
            success=True,
            data=filings[:limit],
            data_type=DataType.FILING,
            provider=self.name,
            symbol=symbol,
            latency_ms=response.latency_ms
        )
    
    async def get_company_facts(self, symbol: str) -> ProviderResponse:
        """
        Get XBRL company facts - standardized financial data.
        This is the best source for parsed financial statements.
        """
        await self._load_ticker_mapping()
        
        cik = self._get_cik(symbol)
        if not cik:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.FUNDAMENTALS,
                provider=self.name,
                symbol=symbol,
                error=f"Could not find CIK for symbol {symbol}"
            )
        
        try:
            result = await self._request(
                "GET",
                f"{self.COMPANY_FACTS_URL}/CIK{cik}.json"
            )
            
            if "error" in result:
                return ProviderResponse(
                    success=False,
                    data=None,
                    data_type=DataType.FUNDAMENTALS,
                    provider=self.name,
                    symbol=symbol,
                    error=result["error"]
                )
            
            data = result["data"]
            
            # Extract key financial facts
            facts = {
                "cik": cik,
                "entity_name": data.get("entityName"),
                "facts": {}
            }
            
            # Parse US-GAAP facts
            us_gaap = data.get("facts", {}).get("us-gaap", {})
            dei = data.get("facts", {}).get("dei", {})
            
            # Key metrics to extract
            key_metrics = [
                "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "NetIncomeLoss", "GrossProfit", "OperatingIncomeLoss",
                "Assets", "Liabilities", "StockholdersEquity",
                "CashAndCashEquivalentsAtCarryingValue",
                "LongTermDebt", "ShortTermBorrowings",
                "EarningsPerShareBasic", "EarningsPerShareDiluted",
                "CommonStockSharesOutstanding",
                "OperatingCashFlow", "CapitalExpenditures",
            ]
            
            for metric in key_metrics:
                if metric in us_gaap:
                    metric_data = us_gaap[metric]
                    units = list(metric_data.get("units", {}).keys())
                    if units:
                        values = metric_data["units"][units[0]]
                        # Get most recent values
                        recent_values = sorted(
                            values, 
                            key=lambda x: x.get("end", ""), 
                            reverse=True
                        )[:20]  # Last 20 periods
                        facts["facts"][metric] = {
                            "unit": units[0],
                            "values": recent_values
                        }
            
            # DEI facts (shares outstanding, etc.)
            for metric in ["EntityCommonStockSharesOutstanding", "EntityPublicFloat"]:
                if metric in dei:
                    metric_data = dei[metric]
                    units = list(metric_data.get("units", {}).keys())
                    if units:
                        values = metric_data["units"][units[0]]
                        recent_values = sorted(
                            values,
                            key=lambda x: x.get("end", ""),
                            reverse=True
                        )[:4]
                        facts["facts"][metric] = {
                            "unit": units[0],
                            "values": recent_values
                        }
            
            return ProviderResponse(
                success=True,
                data=facts,
                data_type=DataType.FUNDAMENTALS,
                provider=self.name,
                symbol=symbol,
                latency_ms=result.get("latency_ms", 0)
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.FUNDAMENTALS,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_fundamentals(
        self,
        symbol: str,
        statement_type: str = "income"
    ) -> ProviderResponse:
        """Get financial statement data from XBRL"""
        return await self.get_company_facts(symbol)
    
    async def get_institutional_holdings(self, symbol: str) -> ProviderResponse:
        """
        Get institutional holdings from 13F filings.
        This is the authoritative source for institutional ownership.
        """
        await self._load_ticker_mapping()
        
        try:
            # First, get filings to find recent 13F-HR filings
            filings_response = await self.get_filings(
                symbol, 
                form_types=["13F-HR", "13F-HR/A"],
                limit=4  # Last 4 quarters
            )
            
            if not filings_response.success:
                return filings_response
            
            # For now, return filing info (parsing 13F XML is complex)
            # In production, you'd parse the XML to extract holdings
            holdings_data = {
                "symbol": symbol,
                "filings": filings_response.data,
                "note": "13F filings found. Full parsing requires XML processing."
            }
            
            return ProviderResponse(
                success=True,
                data=holdings_data,
                data_type=DataType.HOLDINGS,
                provider=self.name,
                symbol=symbol,
                latency_ms=filings_response.latency_ms
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.HOLDINGS,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_insider_transactions(
        self,
        symbol: str,
        limit: int = 100
    ) -> ProviderResponse:
        """
        Get insider transactions from Form 4 filings.
        """
        await self._load_ticker_mapping()
        
        try:
            # Get Form 4 filings
            filings_response = await self.get_filings(
                symbol,
                form_types=["4", "4/A"],
                limit=limit
            )
            
            if not filings_response.success:
                return filings_response
            
            # Return filing info
            # Full Form 4 parsing would extract transaction details
            transactions_data = {
                "symbol": symbol,
                "filings": filings_response.data,
                "note": "Form 4 filings found. Full parsing extracts transaction details."
            }
            
            return ProviderResponse(
                success=True,
                data=transactions_data,
                data_type=DataType.TRANSACTIONS,
                provider=self.name,
                symbol=symbol,
                latency_ms=filings_response.latency_ms
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                data_type=DataType.TRANSACTIONS,
                provider=self.name,
                symbol=symbol,
                error=str(e)
            )
    
    async def get_company_profile(self, symbol: str) -> ProviderResponse:
        """Get company profile from SEC submissions"""
        response = await self.get_company_submissions(symbol)
        
        if not response.success:
            return response
        
        company_info = response.data.get("company_info")
        
        if company_info:
            profile = CompanyProfile(
                symbol=symbol,
                name=company_info.name,
                sector=company_info.sic_description,
                cik=company_info.cik,
            )
            
            return ProviderResponse(
                success=True,
                data=profile,
                data_type=DataType.PROFILE,
                provider=self.name,
                symbol=symbol,
                latency_ms=response.latency_ms
            )
        
        return ProviderResponse(
            success=False,
            data=None,
            data_type=DataType.PROFILE,
            provider=self.name,
            symbol=symbol,
            error="Could not extract company profile"
        )
    
    async def get_13f_holders(self, symbol: str) -> ProviderResponse:
        """
        Get list of institutional holders from 13F filings.
        Queries SEC for institutions holding this security.
        """
        # This would require searching 13F filings across all institutions
        # which is a more complex operation. For now, return guidance.
        return ProviderResponse(
            success=True,
            data={
                "symbol": symbol,
                "note": "For comprehensive 13F holder data, use Finnhub or a specialized 13F aggregator.",
                "sec_13f_search": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=13F&dateb=&owner=include&count=40&search_text={symbol}"
            },
            data_type=DataType.HOLDINGS,
            provider=self.name,
            symbol=symbol,
        )
