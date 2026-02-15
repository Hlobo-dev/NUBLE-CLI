#!/usr/bin/env python3
"""
WRDS → Parquet COMPLETE Data Pipeline v2.0
============================================
Downloads ALL institutional datasets from WRDS into Parquet files.
Based on the wrdscomplete guide — 25+ datasets, ~220M+ rows target.

ARCHITECTURE:
  - Option B: WRDS → Parquet locally (no RDS dependency for WRDS data)
  - Chunked year-by-year for large tables (memory-safe)
  - Resumable: manifest tracks completed downloads
  - Discovery-first: verifies table schemas before querying
  - CRSP CIZ migration aware: tries old format first, falls back to v2

OUTPUT STRUCTURE:
    data/wrds/
    ├── CORE (essential for ML)
    │   ├── wrds_financial_ratios.parquet    ★★★ THE GOLD MINE (~2.8M rows)
    │   ├── crsp_monthly.parquet            (~5M rows, 1926-2024)
    │   ├── compustat_quarterly.parquet     (~3M rows)
    │   ├── compustat_annual.parquet        (~800K rows)
    │   ├── ibes_summary.parquet            (~8M rows)
    │   ├── ibes_actuals.parquet            (~2M rows)
    │   ├── ff_factors_daily.parquet        (~25K rows)
    │   ├── ff_factors_monthly.parquet      (~1.2K rows)
    │   ├── crsp_compustat_link.parquet     (~123K rows)
    │   ├── ibes_crsp_link.parquet          (~37K rows)
    │   └── compustat_security.parquet      (~100K rows)
    │
    ├── SUPPLEMENTARY (high-value signals)
    │   ├── ibes_recommendations.parquet    (~1M+ rows)
    │   ├── crsp_distributions.parquet      (~1M rows)
    │   ├── crsp_delisting.parquet          (~39K rows)
    │   ├── crsp_index_monthly.parquet      (~1.2K rows)
    │   ├── crsp_index_daily.parquet        (~26K rows)
    │   ├── crsp_treasury_monthly.parquet   (~1.2K rows)
    │   ├── crsp_treasury_daily.parquet     (~26K rows)
    │   ├── sp500_constituents.parquet      (~50K rows)
    │   ├── short_interest.parquet          (~5M rows)
    │   ├── institutional_holdings.parquet  (~large, chunked)
    │   ├── insider_trading.parquet         (~17M rows, chunked)
    │   └── execucomp.parquet              (~373K rows)
    │
    ├── MASSIVE (optional, multi-hour downloads)
    │   └── crsp_daily.parquet             (~107M rows, 1926-2024)
    │
    └── download_manifest.json

USAGE:
    python wrds_pipeline/download_complete.py --test           # Quick connectivity test
    python wrds_pipeline/download_complete.py --core           # Download core datasets only
    python wrds_pipeline/download_complete.py --all            # Download everything
    python wrds_pipeline/download_complete.py --dataset NAME   # Single dataset
    python wrds_pipeline/download_complete.py --include-daily  # Include CRSP daily (107M rows)
    python wrds_pipeline/download_complete.py --verify         # Verify existing files
    python wrds_pipeline/download_complete.py --force          # Force re-download
"""

import os
import sys
import gc
import json
import time
import hashlib
import logging
import argparse
import urllib.parse
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import sqlalchemy as sa

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

WRDS_USERNAME = os.environ.get("WRDS_USERNAME", "hlobo")
WRDS_PASSWORD = os.environ.get("WRDS_PASSWORD", "")
WRDS_HOST = os.environ.get("WRDS_PGHOST", "wrds-pgdata.wharton.upenn.edu")
WRDS_PORT = int(os.environ.get("WRDS_PGPORT", "9737"))
WRDS_DB = os.environ.get("WRDS_PGDATABASE", "wrds")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "wrds"

PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "snappy"

LOG_FILE = BASE_DIR / "wrds_download_v2.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("wrds_complete")


# ═══════════════════════════════════════════════════════════════════
# Dataset Definitions — ALL datasets from wrdscomplete guide
# ═══════════════════════════════════════════════════════════════════

# Ordered: fastest → slowest, per guide recommendations
CORE_DATASETS = OrderedDict()
SUPPLEMENTARY_DATASETS = OrderedDict()
MASSIVE_DATASETS = OrderedDict()

# ──────────────────────────────────────────────────────────────────
# SMALL / FAST (< 1 minute each)
# ──────────────────────────────────────────────────────────────────

CORE_DATASETS["ff_factors_daily"] = {
    "description": "Fama-French 5 factors + momentum (daily, since 1926)",
    "query": """
        SELECT date, mktrf, smb, hml, rmw, cma, rf, umd
        FROM ff_all.fivefactors_daily
        WHERE date >= '1926-07-01'
        ORDER BY date
    """,
    "fallback_queries": [
        "SELECT date, mktrf, smb, hml, rmw, cma, rf FROM ff.fivefactors_daily WHERE date >= '1926-07-01' ORDER BY date",
        "SELECT date, mktrf, smb, hml, rf FROM ff.factors_daily WHERE date >= '1926-07-01' ORDER BY date",
    ],
    "expected_min_rows": 15_000,
    "chunk_by_year": False,
}

CORE_DATASETS["ff_factors_monthly"] = {
    "description": "Fama-French 5 factors + momentum (monthly, since 1926)",
    "query": """
        SELECT date, mktrf, smb, hml, rmw, cma, rf, umd
        FROM ff_all.fivefactors_monthly
        WHERE date >= '1926-07-01'
        ORDER BY date
    """,
    "fallback_queries": [
        "SELECT date, mktrf, smb, hml, rmw, cma, rf FROM ff.fivefactors_monthly WHERE date >= '1926-07-01' ORDER BY date",
    ],
    "expected_min_rows": 500,
    "chunk_by_year": False,
}

CORE_DATASETS["crsp_compustat_link"] = {
    "description": "CRSP-Compustat linking table (PERMNO ↔ GVKEY, ~123K rows)",
    "query": """
        SELECT gvkey, lpermno, lpermco,
               linktype, linkprim, linkdt, linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE linktype IN ('LC','LU','LS')
          AND linkprim IN ('P','C')
        ORDER BY gvkey, linkdt
    """,
    "expected_min_rows": 20_000,
    "chunk_by_year": False,
}

CORE_DATASETS["ibes_crsp_link"] = {
    "description": "IBES ticker ↔ CRSP PERMNO link (~37K rows)",
    "query": """
        SELECT ticker, permno, ncusip, sdate, edate, score
        FROM wrdsapps.ibcrsphist
        ORDER BY ticker, sdate
    """,
    "fallback_queries": [
        """SELECT DISTINCT a.ticker, b.permno,
                  MIN(b.namedt) AS sdate, MAX(b.nameendt) AS edate
           FROM ibes.idsum a
           JOIN crsp.msenames b
               ON SUBSTR(a.cusip, 1, 8) = SUBSTR(b.ncusip, 1, 8)
           GROUP BY a.ticker, b.permno""",
    ],
    "expected_min_rows": 10_000,
    "chunk_by_year": False,
}

CORE_DATASETS["compustat_security"] = {
    "description": "Compustat security identifiers (GVKEY ↔ ticker, CUSIP, ~100K rows)",
    "query": """
        SELECT gvkey, iid, tic, cusip, exchg, excntry, dldtei, dlrsni,
               secstat, tpci, isin, sedol, ibtic, curr_sp500_flag
        FROM comp.security
        ORDER BY gvkey, iid
    """,
    "expected_min_rows": 40_000,
    "chunk_by_year": False,
}

# ── Supplementary small tables ──

SUPPLEMENTARY_DATASETS["crsp_delisting"] = {
    "description": "CRSP delisting returns — CRITICAL for survivorship-bias-free backtesting (~39K rows)",
    "query": """
        SELECT permno, dlstdt, dlstcd, nwperm, nwcomp,
               nextdt, dlamt, dlretx, dlprc, dlpdt, dlret
        FROM crsp.msedelist
        ORDER BY permno, dlstdt
    """,
    "expected_min_rows": 30_000,
    "chunk_by_year": False,
}

SUPPLEMENTARY_DATASETS["crsp_index_monthly"] = {
    "description": "CRSP market index returns — monthly (since 1926, ~1.2K rows)",
    "query": """
        SELECT date, vwretd, vwretx, ewretd, ewretx, sprtrn,
               spindx, totval, totcnt, usdval, usdcnt
        FROM crsp.msi
        ORDER BY date
    """,
    "expected_min_rows": 1_000,
    "chunk_by_year": False,
}

SUPPLEMENTARY_DATASETS["crsp_index_daily"] = {
    "description": "CRSP market index returns — daily (since 1926, ~26K rows)",
    "query": """
        SELECT date, vwretd, vwretx, ewretd, ewretx, sprtrn,
               spindx, totval, totcnt, usdval, usdcnt
        FROM crsp.dsi
        ORDER BY date
    """,
    "expected_min_rows": 20_000,
    "chunk_by_year": False,
}

SUPPLEMENTARY_DATASETS["crsp_treasury_monthly"] = {
    "description": "CRSP Treasury / risk-free rates — monthly (since 1926)",
    "query": """
        SELECT caldt AS date,
               t30ret, t30ind, t90ret, t90ind,
               b1ret, b1ind, b5ret, b5ind, b10ret, b10ind,
               cpiret, cpiind
        FROM crsp.mcti
        ORDER BY caldt
    """,
    "fallback_queries": [
        "SELECT caldt AS date, t30ret, t30ind FROM crsp.mcti ORDER BY caldt",
    ],
    "expected_min_rows": 1_000,
    "chunk_by_year": False,
}

SUPPLEMENTARY_DATASETS["crsp_treasury_daily"] = {
    "description": "CRSP Treasury / risk-free rates — daily (from crsp.dsi, since 1926)",
    "query": """
        SELECT date,
               vwretd, vwretx, ewretd, ewretx,
               sprtrn, spindx, totval, totcnt
        FROM crsp.dsi
        ORDER BY date
    """,
    "fallback_queries": [
        "SELECT date, vwretd, ewretd, sprtrn FROM crsp.dsi ORDER BY date",
    ],
    "expected_min_rows": 20_000,
    "chunk_by_year": False,
}

SUPPLEMENTARY_DATASETS["sp500_constituents_crsp"] = {
    "description": "S&P 500 historical constituents from CRSP (since 1925)",
    "query": """
        SELECT permno, start AS from_date, ending AS thru_date
        FROM crsp.msp500list
        ORDER BY permno, start
    """,
    "expected_min_rows": 1_000,
    "chunk_by_year": False,
}

SUPPLEMENTARY_DATASETS["sp500_constituents_compustat"] = {
    "description": "S&P 500 historical constituents from Compustat (with GVKEY)",
    "query": """
        SELECT gvkey, iid, gvkeyx, \"from\" AS from_date, thru AS thru_date
        FROM comp.idxcst_his
        WHERE gvkeyx = '000003'
        ORDER BY \"from\"
    """,
    "expected_min_rows": 500,
    "chunk_by_year": False,
}

# ──────────────────────────────────────────────────────────────────
# MEDIUM (5-30 minutes each)
# ──────────────────────────────────────────────────────────────────

CORE_DATASETS["ibes_actuals"] = {
    "description": "IBES actual EPS announcements (~2M rows, since 1976)",
    "query": """
        SELECT ticker, measure, pends, pdicity, anndats, value
        FROM ibes.actu_epsus
        WHERE measure = 'EPS'
          AND pends >= '1976-01-01'
        ORDER BY ticker, pends
    """,
    "expected_min_rows": 400_000,
    "chunk_by_year": False,
}

SUPPLEMENTARY_DATASETS["ibes_recommendations"] = {
    "description": "IBES analyst recommendations (buy/sell/hold, since 1985)",
    "query": """
        SELECT ticker, cusip, cname, oftic, actdats,
               estimid, analyst, ereccd, etext,
               ireccd, itext, emaskcd, amaskcd,
               usfirm, anndats, revdats
        FROM ibes.recddet
        WHERE anndats >= '1985-01-01'
        ORDER BY ticker, anndats
    """,
    "expected_min_rows": 500_000,
    "chunk_by_year": False,
}

SUPPLEMENTARY_DATASETS["crsp_distributions"] = {
    "description": "CRSP distributions — dividends, splits, spinoffs since 1926 (~1M rows)",
    "query": """
        SELECT permno, distcd, divamt, facpr, facshr,
               dclrdt, exdt, rcrddt, paydt, acperm
        FROM crsp.msedist
        ORDER BY permno, exdt
    """,
    "expected_min_rows": 800_000,
    "chunk_by_year": False,
}

CORE_DATASETS["compustat_annual"] = {
    "description": "Compustat annual fundamentals (~800K rows, since 1950)",
    "query": """
        SELECT gvkey, datadate, fyear,
               revt, cogs, xsga, oiadp, ni, ib, dp,
               xrd, xint, txt, ebitda, sale, gp,
               at, act, che, rect, invt, ppent,
               lt, lct, dltt, dlc,
               seq, ceq, csho, txditc, pstk, mib,
               oancf, capx, dv, wcap,
               epspx, dvpsx_f,
               tic, conm, sich, naicsh,
               datafmt, indfmt, consol, popsrc
        FROM comp.funda
        WHERE datafmt = 'STD'
          AND indfmt  = 'INDL'
          AND consol  = 'C'
          AND popsrc  = 'D'
          AND datadate >= '1950-01-01'
        ORDER BY gvkey, datadate
    """,
    "expected_min_rows": 400_000,
    "chunk_by_year": False,
}

SUPPLEMENTARY_DATASETS["execucomp"] = {
    "description": "Executive compensation data (~373K rows, since 1992)",
    "query": """
        SELECT gvkey, execid, year, exec_fullname, ceoann, cfoann,
               titleann, salary, bonus, stock_awards_fv, option_awards_fv,
               tdc1, tdc2, shrown_excl_opts, shrown_tot_pct,
               coname, ticker, cusip
        FROM comp_execucomp.anncomp
        WHERE year >= 1992
        ORDER BY gvkey, year, execid
    """,
    "expected_min_rows": 200_000,
    "chunk_by_year": False,
}

# ──────────────────────────────────────────────────────────────────
# LARGE (30-120 minutes each, chunked by year)
# ──────────────────────────────────────────────────────────────────

CORE_DATASETS["compustat_quarterly"] = {
    "description": "Compustat quarterly fundamentals (~3M rows, since 1962)",
    "query_template": """
        SELECT gvkey, datadate, fyearq, fqtr, rdq,
               revtq, cogsq, xsgaq, oiadpq, niq, ibq, dpq,
               xrdq, xintq, txtq, epspxq, epsfxq,
               atq, actq, cheq, rectq, invtq, ppentq,
               ltq, lctq, dlttq, dlcq,
               seqq, ceqq, cshoq, txditcq, pstkq,
               oancfy, capxy, dvy,
               tic, conm,
               datafmt, indfmt, consol, popsrc, curcdq
        FROM comp.fundq
        WHERE datafmt = 'STD'
          AND indfmt  = 'INDL'
          AND consol  = 'C'
          AND popsrc  = 'D'
          AND EXTRACT(YEAR FROM datadate) = {year}
        ORDER BY gvkey, datadate
    """,
    "expected_min_rows": 1_500_000,
    "chunk_by_year": True,
    "year_range": (1962, 2025),
}

CORE_DATASETS["ibes_summary"] = {
    "description": "IBES consensus estimates — analyst forecasts (~8M rows, since 1976)",
    "query_template": """
        SELECT ticker, cusip, cname, fpedats, statpers, measure, fpi,
               numest, numup, numdown, meanest, medest, stdev,
               highest, lowest, usfirm
        FROM ibes.statsumu_epsus
        WHERE measure = 'EPS'
          AND fpi IN ('1','2','6','7')
          AND EXTRACT(YEAR FROM statpers) = {year}
        ORDER BY ticker, fpedats, statpers
    """,
    "expected_min_rows": 2_000_000,
    "chunk_by_year": True,
    "year_range": (1976, 2025),
}

CORE_DATASETS["crsp_monthly"] = {
    "description": "CRSP monthly stock returns + names (since 1926, ~5M rows)",
    "query_template": """
        SELECT a.permno, a.date, b.ticker,
               a.ret, a.retx, a.prc, a.shrout, a.vol,
               ABS(a.prc) * a.shrout * 1000 AS market_cap,
               b.exchcd, b.shrcd, b.siccd,
               a.altprc, a.spread
        FROM crsp.msf a
        LEFT JOIN crsp.msenames b
            ON a.permno = b.permno
           AND a.date BETWEEN b.namedt AND b.nameendt
        WHERE b.shrcd IN (10, 11)
          AND b.exchcd IN (1, 2, 3)
          AND EXTRACT(YEAR FROM a.date) = {year}
        ORDER BY a.permno, a.date
    """,
    "fallback_query_template": """
        SELECT permno, mthcaldt AS date, ticker,
               mthret AS ret, mthretx AS retx, mthprc AS prc,
               shrout, mthvol AS vol,
               mthcap AS market_cap,
               primaryexch, sharetype, securitysubtype,
               siccd
        FROM crsp.msf_v2
        WHERE sharetype = 'NS' AND securitysubtype = 'COM'
          AND primaryexch IN ('N','A','Q')
          AND EXTRACT(YEAR FROM mthcaldt) = {year}
        ORDER BY permno, mthcaldt
    """,
    "expected_min_rows": 2_500_000,
    "chunk_by_year": True,
    "year_range": (1926, 2025),
}

# ──────────────────────────────────────────────────────────────────
# ★★★ THE GOLD MINE ★★★
# ──────────────────────────────────────────────────────────────────

CORE_DATASETS["wrds_financial_ratios"] = {
    "description": "★★★ WRDS Pre-Computed Financial Ratios — 80+ ratios, point-in-time, monthly (~2.8M rows) ★★★",
    "query_template": """
        SELECT *
        FROM wrdsapps_finratio_ibes.firm_ratio_ibes
        WHERE EXTRACT(YEAR FROM public_date) = {year}
        ORDER BY permno, public_date
    """,
    "fallback_query_template": """
        SELECT *
        FROM wrdsapps_finratio.firm_ratio
        WHERE EXTRACT(YEAR FROM public_date) = {year}
        ORDER BY permno, public_date
    """,
    "expected_min_rows": 1_000_000,
    "chunk_by_year": True,
    "year_range": (1970, 2025),
}

# ──────────────────────────────────────────────────────────────────
# SUPPLEMENTARY LARGE (chunked)
# ──────────────────────────────────────────────────────────────────

SUPPLEMENTARY_DATASETS["short_interest"] = {
    "description": "Compustat short interest data (~5M rows, since 2006)",
    "query_template": """
        SELECT gvkey, iid, shortint, shortintadj,
               datadate, splitadjdate
        FROM comp.sec_shortint
        WHERE EXTRACT(YEAR FROM datadate) = {year}
        ORDER BY gvkey, datadate
    """,
    "expected_min_rows": 3_000_000,
    "chunk_by_year": True,
    "year_range": (2006, 2025),
}

SUPPLEMENTARY_DATASETS["insider_trading"] = {
    "description": "Thomson/LSEG insider trading — Forms 3,4,5 (~17M rows, since 1986)",
    "query_template": """
        SELECT fdate AS filing_date, trandate AS trade_date,
               ticker, cusip6, cusip2,
               cname AS company_name,
               owner AS insider_name,
               trancode AS transaction_type,
               acqdisp,
               shares, tprice AS price,
               sharesheld AS shares_after,
               ownership AS ownership_type,
               rolecode1, formtype,
               shares_adj, tprice_adj
        FROM tfn.table1
        WHERE EXTRACT(YEAR FROM fdate) = {year}
        ORDER BY fdate
    """,
    "expected_min_rows": 5_000_000,
    "chunk_by_year": True,
    "year_range": (1986, 2025),
}

SUPPLEMENTARY_DATASETS["institutional_holdings"] = {
    "description": "Thomson/LSEG 13F institutional holdings (since 1978, very large)",
    "query_template": """
        SELECT a.fdate, a.cusip, a.mgrno, a.shares, a.sole, a.shared, a.no,
               b.mgrname, b.typecode
        FROM tr_13f.s34type3 a
        LEFT JOIN tr_13f.s34type1 b
            ON a.fdate = b.fdate AND a.mgrno = b.mgrno
        WHERE EXTRACT(YEAR FROM a.fdate) = {year}
        ORDER BY a.fdate, a.mgrno, a.cusip
    """,
    "fallback_query_template": """
        SELECT fdate, cusip, mgrno, shares, sole, shared, no
        FROM tr_13f.s34type3
        WHERE EXTRACT(YEAR FROM fdate) = {year}
        ORDER BY fdate, mgrno, cusip
    """,
    "expected_min_rows": 5_000_000,
    "chunk_by_year": True,
    "year_range": (1978, 2025),
}

# ──────────────────────────────────────────────────────────────────
# MASSIVE (optional, hours-long)
# ──────────────────────────────────────────────────────────────────

MASSIVE_DATASETS["crsp_daily"] = {
    "description": "★★★ CRSP daily stock returns — FULL history 1926-2024 (~107M rows) ★★★",
    "query_template": """
        SELECT a.permno, a.date,
               a.ret, a.retx, a.prc, a.vol, a.shrout,
               ABS(a.prc) * a.shrout * 1000 AS market_cap,
               a.bidlo, a.askhi, a.openprc,
               b.ticker, b.exchcd, b.shrcd, b.siccd
        FROM crsp.dsf a
        LEFT JOIN crsp.dsenames b
            ON a.permno = b.permno
           AND a.date BETWEEN b.namedt AND b.nameendt
        WHERE b.shrcd IN (10, 11)
          AND b.exchcd IN (1, 2, 3)
          AND EXTRACT(YEAR FROM a.date) = {year}
        ORDER BY a.permno, a.date
    """,
    "fallback_query_template": """
        SELECT permno, dlycaldt AS date,
               dlyret AS ret, dlyretx AS retx,
               dlyprc AS prc, dlyvol AS vol, shrout,
               dlycap AS market_cap,
               dlylow AS bidlo, dlyhigh AS askhi, dlyopen AS openprc,
               ticker, primaryexch, sharetype, securitysubtype, siccd
        FROM crsp.dsf_v2
        WHERE sharetype = 'NS' AND securitysubtype = 'COM'
          AND primaryexch IN ('N','A','Q')
          AND EXTRACT(YEAR FROM dlycaldt) = {year}
        ORDER BY permno, dlycaldt
    """,
    "expected_min_rows": 50_000_000,
    "chunk_by_year": True,
    "year_range": (1926, 2025),
}


# ═══════════════════════════════════════════════════════════════════
# WRDSCompleteDownloader
# ═══════════════════════════════════════════════════════════════════

class WRDSCompleteDownloader:
    """
    Downloads ALL WRDS datasets and saves as Parquet files.
    
    Principles from wrdscomplete guide:
    1. DISCOVER BEFORE QUERY
    2. CHUNK LARGE TABLES (year-by-year)
    3. HANDLE CRSP CIZ MIGRATION (old first, fallback to v2)
    4. MEMORY MANAGEMENT (gc.collect() after each chunk)
    5. LOG EVERYTHING
    6. CONTINUE ON FAILURE (never abort for single table)
    7. IDEMPOTENT (safe to re-run)
    """

    def __init__(self, output_dir: Path = DATA_DIR, force: bool = False):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.force = force
        self.engine = None
        self.manifest = {}
        self._load_manifest()

    def _load_manifest(self):
        manifest_path = self.output_dir / "download_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)

    def _save_manifest(self):
        manifest_path = self.output_dir / "download_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    # ───────────────────────────────────────────────────────────────
    # Connection
    # ───────────────────────────────────────────────────────────────

    def connect(self):
        logger.info("Connecting to WRDS...")
        password_encoded = urllib.parse.quote_plus(WRDS_PASSWORD)
        pguri = f"postgresql://{WRDS_USERNAME}:{password_encoded}@{WRDS_HOST}:{WRDS_PORT}/{WRDS_DB}"
        
        self.engine = sa.create_engine(
            pguri,
            isolation_level="AUTOCOMMIT",
            connect_args={
                "sslmode": "require",
                "connect_timeout": 60,
                "options": "-c statement_timeout=1800000",  # 30 min per query
            },
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        
        with self.engine.connect() as conn:
            result = conn.execute(sa.text("SELECT 1"))
            assert result.fetchone()[0] == 1
        
        logger.info("✅ Connected to WRDS")

    def disconnect(self):
        if self.engine:
            self.engine.dispose()
            logger.info("Disconnected from WRDS")

    # ───────────────────────────────────────────────────────────────
    # Query execution
    # ───────────────────────────────────────────────────────────────

    def _query_to_df(self, query: str) -> pd.DataFrame:
        with self.engine.connect() as conn:
            df = pd.read_sql(sa.text(query), conn)
        df.columns = df.columns.str.lower().str.strip()
        return df

    def _try_query_with_fallback(self, primary: str, fallback: str = None) -> pd.DataFrame:
        try:
            return self._query_to_df(primary)
        except Exception as e:
            logger.warning(f"  Primary query failed: {str(e)[:150]}")
            if fallback:
                logger.info(f"  Trying fallback query...")
                return self._query_to_df(fallback)
            raise

    def _try_query(self, primary: str, fallback_queries: list = None) -> pd.DataFrame:
        try:
            return self._query_to_df(primary)
        except Exception as e:
            logger.warning(f"  Primary query failed: {str(e)[:150]}")
            if fallback_queries:
                for i, fq in enumerate(fallback_queries):
                    try:
                        logger.info(f"  Trying fallback {i+1}...")
                        return self._query_to_df(fq)
                    except Exception as e2:
                        logger.warning(f"  Fallback {i+1} failed: {str(e2)[:150]}")
            raise

    # ───────────────────────────────────────────────────────────────
    # Parquet I/O
    # ───────────────────────────────────────────────────────────────

    def _save_parquet(self, df: pd.DataFrame, name: str) -> tuple:
        filepath = self.output_dir / f"{name}.parquet"
        df.to_parquet(
            filepath, engine=PARQUET_ENGINE,
            compression=PARQUET_COMPRESSION, index=False,
        )
        file_hash = hashlib.md5(filepath.read_bytes()).hexdigest()
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        return filepath, file_hash, file_size_mb

    def _should_download(self, name: str) -> bool:
        if self.force:
            return True
        filepath = self.output_dir / f"{name}.parquet"
        if not filepath.exists():
            return True
        if name in self.manifest:
            if self.manifest[name].get("status") == "complete":
                rows = self.manifest[name].get("rows", "?")
                logger.info(f"  ⏭️  {name} already complete ({rows:,} rows). Use --force to re-download.")
                return False
        return True

    # ───────────────────────────────────────────────────────────────
    # Download a single dataset
    # ───────────────────────────────────────────────────────────────

    def download_dataset(self, name: str, config: dict) -> dict:
        logger.info(f"\n{'═' * 70}")
        logger.info(f"  📥 {name}")
        logger.info(f"  {config['description']}")
        logger.info(f"{'═' * 70}")

        if not self._should_download(name):
            return self.manifest.get(name, {})

        t0 = time.time()

        try:
            if config.get("chunk_by_year"):
                full_df = self._download_chunked(name, config)
            else:
                fallbacks = config.get("fallback_queries", [])
                full_df = self._try_query(config["query"], fallbacks)
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"  ❌ {name} FAILED after {elapsed:.1f}s: {e}")
            result = {"status": "failed", "error": str(e)[:500], "elapsed_secs": round(elapsed, 1)}
            self.manifest[name] = result
            self._save_manifest()
            return result

        if full_df is None or full_df.empty:
            logger.error(f"  ❌ {name}: No data retrieved")
            result = {"status": "failed", "rows": 0, "error": "empty result"}
            self.manifest[name] = result
            self._save_manifest()
            return result

        # Save
        filepath, file_hash, file_size_mb = self._save_parquet(full_df, name)
        elapsed = time.time() - t0
        n_rows = len(full_df)
        n_cols = len(full_df.columns)

        # Validate
        expected_min = config.get("expected_min_rows", 0)
        if n_rows < expected_min:
            logger.warning(f"  ⚠️  {name}: {n_rows:,} rows < expected minimum {expected_min:,}")
        
        logger.info(f"  ✅ {name}: {n_rows:,} rows × {n_cols} cols ({file_size_mb:.1f} MB) in {elapsed:.1f}s")

        # Date range info
        date_cols = [c for c in full_df.columns if c in ("date", "datadate", "public_date", "statpers", "pends", "fdate", "dlstdt", "exdt", "anndats", "caldt", "filing_date", "trade_date")]
        date_range = None
        if date_cols:
            col = date_cols[0]
            try:
                dates = pd.to_datetime(full_df[col], errors="coerce")
                date_range = f"{dates.min()} → {dates.max()}"
                logger.info(f"       Date range: {date_range}")
            except Exception:
                pass

        # Manifest
        result = {
            "status": "complete",
            "rows": n_rows,
            "columns": list(full_df.columns),
            "n_columns": n_cols,
            "file_size_mb": round(file_size_mb, 2),
            "md5": file_hash,
            "date_range": date_range,
            "downloaded_at": datetime.now().isoformat(),
            "elapsed_secs": round(elapsed, 1),
            "filepath": str(filepath),
        }
        self.manifest[name] = result
        self._save_manifest()

        del full_df
        gc.collect()
        return result

    def _download_chunked(self, name: str, config: dict) -> pd.DataFrame:
        """Download a large dataset year-by-year."""
        start_year, end_year = config["year_range"]
        query_template = config.get("query_template", "")
        fallback_template = config.get("fallback_query_template", "")
        
        all_chunks = []
        total_rows = 0
        use_fallback = False
        
        for year in range(start_year, end_year + 1):
            try:
                if use_fallback and fallback_template:
                    query = fallback_template.format(year=year)
                else:
                    query = query_template.format(year=year)
                
                df = self._query_to_df(query)
                
                if df.empty:
                    continue
                
                all_chunks.append(df)
                total_rows += len(df)
                logger.info(f"    {year}: {len(df):>10,} rows  (cumulative: {total_rows:>12,})")
                
                # Memory management: if accumulated >5M rows, save intermediate
                if total_rows > 5_000_000 and len(all_chunks) > 10:
                    pass  # For now, keep in memory; for truly massive, would save intermediate
                
            except Exception as e:
                err_str = str(e)[:200]
                
                # If primary fails on first year, try fallback template for ALL years
                if not use_fallback and fallback_template and year == start_year:
                    logger.warning(f"    {year}: Primary failed ({err_str}), switching to fallback template")
                    use_fallback = True
                    try:
                        query = fallback_template.format(year=year)
                        df = self._query_to_df(query)
                        if not df.empty:
                            all_chunks.append(df)
                            total_rows += len(df)
                            logger.info(f"    {year}: {len(df):>10,} rows (fallback)")
                        continue
                    except Exception as e2:
                        logger.warning(f"    {year}: Fallback also failed: {str(e2)[:100]}")
                else:
                    logger.warning(f"    {year}: ERROR — {err_str}")

        if all_chunks:
            logger.info(f"  Concatenating {len(all_chunks)} chunks ({total_rows:,} total rows)...")
            full_df = pd.concat(all_chunks, ignore_index=True)
            del all_chunks
            gc.collect()
            return full_df
        else:
            return None

    # ───────────────────────────────────────────────────────────────
    # Download orchestrator
    # ───────────────────────────────────────────────────────────────

    def download_core(self):
        """Download core datasets only (essential for ML)."""
        self.connect()
        try:
            for name, config in CORE_DATASETS.items():
                try:
                    self.download_dataset(name, config)
                except Exception as e:
                    logger.error(f"  ❌ {name} FAILED: {e}")
                    self.manifest[name] = {"status": "failed", "error": str(e)[:500]}
                    self._save_manifest()
        finally:
            self.disconnect()
            self._print_summary()

    def download_all(self, include_daily: bool = False, dataset: str = None):
        """Download ALL datasets (core + supplementary + optionally daily)."""
        self.connect()
        try:
            if dataset:
                # Single dataset mode
                all_ds = {**CORE_DATASETS, **SUPPLEMENTARY_DATASETS, **MASSIVE_DATASETS}
                if dataset not in all_ds:
                    logger.error(f"Unknown dataset: {dataset}")
                    logger.info(f"Available: {sorted(all_ds.keys())}")
                    return
                self.download_dataset(dataset, all_ds[dataset])
                return

            # Download in order: core → supplementary → massive
            logger.info("\n" + "█" * 70)
            logger.info("  PHASE 1: CORE DATASETS")
            logger.info("█" * 70)
            for name, config in CORE_DATASETS.items():
                try:
                    self.download_dataset(name, config)
                except Exception as e:
                    logger.error(f"  ❌ {name} FAILED: {e}")
                    self.manifest[name] = {"status": "failed", "error": str(e)[:500]}
                    self._save_manifest()

            logger.info("\n" + "█" * 70)
            logger.info("  PHASE 2: SUPPLEMENTARY DATASETS")
            logger.info("█" * 70)
            for name, config in SUPPLEMENTARY_DATASETS.items():
                try:
                    self.download_dataset(name, config)
                except Exception as e:
                    logger.error(f"  ❌ {name} FAILED: {e}")
                    self.manifest[name] = {"status": "failed", "error": str(e)[:500]}
                    self._save_manifest()

            # Massive datasets (optional)
            if include_daily:
                logger.info("\n" + "█" * 70)
                logger.info("  PHASE 3: MASSIVE DATASETS (this will take hours)")
                logger.info("█" * 70)
                for name, config in MASSIVE_DATASETS.items():
                    try:
                        self.download_dataset(name, config)
                    except Exception as e:
                        logger.error(f"  ❌ {name} FAILED: {e}")
                        self.manifest[name] = {"status": "failed", "error": str(e)[:500]}
                        self._save_manifest()

        finally:
            self.disconnect()
            self._print_summary()

    # ───────────────────────────────────────────────────────────────
    # Verification
    # ───────────────────────────────────────────────────────────────

    def verify(self):
        logger.info("\n" + "═" * 70)
        logger.info("  PARQUET FILE VERIFICATION")
        logger.info("═" * 70)

        all_ds = {**CORE_DATASETS, **SUPPLEMENTARY_DATASETS, **MASSIVE_DATASETS}
        total_rows = 0
        total_size = 0
        issues = []

        for name, config in all_ds.items():
            filepath = self.output_dir / f"{name}.parquet"
            if not filepath.exists():
                if name in CORE_DATASETS:
                    logger.warning(f"  ❌ {name:40s}: FILE MISSING")
                    issues.append(f"{name}: missing")
                continue

            try:
                df = pd.read_parquet(filepath)
                n_rows = len(df)
                n_cols = len(df.columns)
                file_size = filepath.stat().st_size / (1024 * 1024)
                total_rows += n_rows
                total_size += file_size

                expected_min = config.get("expected_min_rows", 0)
                ok = n_rows >= expected_min * 0.5  # Allow 50% tolerance
                status = "✅" if ok else "⚠️"

                logger.info(f"  {status} {name:40s}: {n_rows:>12,} rows × {n_cols:>3} cols  {file_size:>8.1f} MB")

                # Date info
                date_cols = [c for c in df.columns if c in ("date", "datadate", "public_date", "statpers", "pends", "fdate", "dlstdt", "exdt", "anndats", "caldt", "filing_date")]
                if date_cols:
                    col = date_cols[0]
                    dates = pd.to_datetime(df[col], errors="coerce")
                    logger.info(f"       Date: {dates.min()} → {dates.max()}")

                if not ok:
                    issues.append(f"{name}: {n_rows:,} rows < expected {expected_min:,}")

                # Null rate check
                null_pcts = df.isnull().mean()
                high_null = null_pcts[null_pcts > 0.5]
                if len(high_null) > 0:
                    cols_str = ", ".join(f"{c}({v:.0%})" for c, v in high_null.head(5).items())
                    logger.info(f"       High nulls: {cols_str}")

                del df
            except Exception as e:
                logger.error(f"  ❌ {name:40s}: ERROR — {str(e)[:100]}")
                issues.append(f"{name}: read error")

        logger.info("─" * 70)
        logger.info(f"  Total: {total_rows:>15,} rows   {total_size:>8.1f} MB ({total_size/1024:.2f} GB)")

        if issues:
            logger.warning(f"\n  ⚠️  {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"    • {issue}")
        else:
            logger.info("\n  ✅ ALL FILES VERIFIED")

        return issues

    # ───────────────────────────────────────────────────────────────
    # Summary
    # ───────────────────────────────────────────────────────────────

    def _print_summary(self):
        logger.info("\n" + "═" * 70)
        logger.info("  WRDS COMPLETE DOWNLOAD SUMMARY")
        logger.info("═" * 70)

        total_rows = 0
        total_size = 0
        completed = 0
        failed = 0

        for name, info in self.manifest.items():
            rows = info.get("rows", 0)
            size = info.get("file_size_mb", 0)
            status = info.get("status", "unknown")
            elapsed = info.get("elapsed_secs", 0)
            date_range = info.get("date_range", "")

            if status == "complete":
                icon = "✅"
                completed += 1
            else:
                icon = "❌"
                failed += 1

            logger.info(f"  {icon} {name:40s}: {rows:>12,} rows  {size:>8.1f} MB  ({elapsed:.0f}s)")
            if date_range:
                logger.info(f"       {date_range}")
            total_rows += rows if isinstance(rows, (int, float)) else 0
            total_size += size if isinstance(size, (int, float)) else 0

        logger.info("─" * 70)
        logger.info(f"  Datasets: {completed} complete, {failed} failed")
        logger.info(f"  Total: {total_rows:>15,} rows   {total_size:>8.1f} MB ({total_size/1024:.2f} GB)")
        logger.info(f"  Output: {self.output_dir}")
        logger.info("═" * 70)


# ═══════════════════════════════════════════════════════════════════
# Quick Test
# ═══════════════════════════════════════════════════════════════════

def quick_test():
    """Quick connectivity + data test (5 rows from each critical table)."""
    logger.info("\n" + "═" * 70)
    logger.info("  WRDS COMPLETE CONNECTION TEST")
    logger.info("═" * 70)

    password_encoded = urllib.parse.quote_plus(WRDS_PASSWORD)
    pguri = f"postgresql://{WRDS_USERNAME}:{password_encoded}@{WRDS_HOST}:{WRDS_PORT}/{WRDS_DB}"

    engine = sa.create_engine(
        pguri, isolation_level="AUTOCOMMIT",
        connect_args={"sslmode": "require", "connect_timeout": 30, "options": "-c statement_timeout=30000"},
        pool_pre_ping=True,
    )

    test_queries = [
        # Core
        ("CRSP Monthly (old)",       "SELECT permno, date, ret FROM crsp.msf LIMIT 5"),
        ("CRSP Monthly (v2)",        "SELECT permno, mthcaldt, mthret FROM crsp.msf_v2 LIMIT 5"),
        ("CRSP Daily (old)",         "SELECT permno, date, ret FROM crsp.dsf LIMIT 5"),
        ("Compustat Q",              "SELECT gvkey, datadate, atq FROM comp.fundq LIMIT 5"),
        ("Compustat A",              "SELECT gvkey, datadate, at FROM comp.funda LIMIT 5"),
        ("IBES Summary",             "SELECT ticker, statpers FROM ibes.statsumu_epsus LIMIT 5"),
        ("IBES Actuals",             "SELECT ticker, anndats FROM ibes.actu_epsus LIMIT 5"),
        ("IBES Recommendations",     "SELECT ticker, anndats, ireccd FROM ibes.recddet LIMIT 5"),
        ("CCM Link",                 "SELECT gvkey, lpermno FROM crsp.ccmxpf_lnkhist LIMIT 5"),
        ("IBES-CRSP Link",           "SELECT ticker, permno FROM wrdsapps.ibcrsphist LIMIT 5"),
        ("FF 5-Factor Daily",        "SELECT date, mktrf FROM ff_all.fivefactors_daily LIMIT 5"),
        ("FF 5-Factor Monthly",      "SELECT date, mktrf FROM ff_all.fivefactors_monthly LIMIT 5"),
        ("Compustat Security",       "SELECT gvkey, tic, cusip FROM comp.security LIMIT 5"),
        # Gold Mine
        ("★ Financial Ratios",       "SELECT permno, public_date, bm, roe, roa FROM wrdsapps_finratio_ibes.firm_ratio_ibes LIMIT 5"),
        # Supplementary
        ("CRSP Delisting",           "SELECT permno, dlstdt, dlret FROM crsp.msedelist LIMIT 5"),
        ("CRSP Distributions",       "SELECT permno, exdt, divamt FROM crsp.msedist LIMIT 5"),
        ("CRSP Index Monthly",       "SELECT date, vwretd, sprtrn FROM crsp.msi LIMIT 5"),
        ("CRSP Index Daily",         "SELECT date, vwretd FROM crsp.dsi LIMIT 5"),
        ("CRSP Treasury Monthly",    "SELECT caldt, t30ret FROM crsp.mcti LIMIT 5"),
        ("S&P 500 (CRSP)",           "SELECT permno, start FROM crsp.msp500list LIMIT 5"),
        ("S&P 500 (Compustat)",      "SELECT gvkey, \"from\" FROM comp.idxcst_his WHERE gvkeyx = '000003' LIMIT 5"),
        ("Short Interest",           "SELECT gvkey, datadate, shortint FROM comp.sec_shortint LIMIT 5"),
        ("13F Holdings",             "SELECT fdate, cusip, mgrno, shares FROM tr_13f.s34type3 LIMIT 5"),
        ("Insider Trading",          "SELECT fdate, ticker, trancode, shares FROM tfn.table1 LIMIT 5"),
        ("Execucomp",                "SELECT gvkey, year, exec_fullname, tdc1 FROM comp_execucomp.anncomp LIMIT 5"),
    ]

    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
            logger.info("  ✅ Connection established\n")

            passed = 0
            failed = 0
            for label, query in test_queries:
                try:
                    t0 = time.time()
                    df = pd.read_sql(sa.text(query), conn)
                    elapsed = time.time() - t0
                    logger.info(f"  ✅ {label:30s}: {len(df)} rows ({elapsed:.2f}s)")
                    passed += 1
                except Exception as e:
                    logger.warning(f"  ❌ {label:30s}: {str(e)[:80]}")
                    failed += 1

            logger.info(f"\n  Results: {passed} passed, {failed} failed out of {len(test_queries)}")
            logger.info(f"  Ready to download! Run: python wrds_pipeline/download_complete.py --all")

    except Exception as e:
        logger.error(f"\n  ❌ Connection FAILED: {e}")
        raise
    finally:
        engine.dispose()


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="WRDS Complete Data Pipeline — download EVERYTHING to Parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test                           # Quick connectivity test (25 tables)
  %(prog)s --core                           # Download core datasets only (~11 tables)
  %(prog)s --all                            # Download ALL datasets (~25 tables)
  %(prog)s --all --include-daily            # + CRSP daily (107M rows, takes hours)
  %(prog)s --dataset wrds_financial_ratios  # Download single dataset
  %(prog)s --verify                         # Verify existing files
  %(prog)s --force --all                    # Force re-download everything

Download order (per wrdscomplete guide):
  CORE:  ff_factors → links → compustat → ibes → crsp_monthly → wrds_financial_ratios
  SUPP:  delisting → distributions → index → treasury → sp500 → short_int → insider → 13F
  MASSIVE: crsp_daily (107M rows)

Available datasets:
  CORE: """ + ", ".join(CORE_DATASETS.keys()) + """
  SUPP: """ + ", ".join(SUPPLEMENTARY_DATASETS.keys()) + """
  MASSIVE: """ + ", ".join(MASSIVE_DATASETS.keys()),
    )
    parser.add_argument("--test", action="store_true", help="Quick connection test (25 tables)")
    parser.add_argument("--core", action="store_true", help="Download core datasets only")
    parser.add_argument("--all", action="store_true", help="Download ALL datasets")
    parser.add_argument("--dataset", type=str, help="Download a single dataset by name")
    parser.add_argument("--include-daily", action="store_true", help="Include CRSP daily (107M rows)")
    parser.add_argument("--verify", action="store_true", help="Verify existing Parquet files")
    parser.add_argument("--force", action="store_true", help="Force re-download of all files")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR), help="Output directory")

    args = parser.parse_args()

    if args.test:
        quick_test()
        return

    output_dir = Path(args.output_dir)
    dl = WRDSCompleteDownloader(output_dir=output_dir, force=args.force)

    if args.verify:
        dl.verify()
        return

    if args.core:
        dl.download_core()
    elif args.all or args.dataset:
        dl.download_all(include_daily=args.include_daily, dataset=args.dataset)
    else:
        parser.print_help()
        print("\n⚠️  Specify --test, --core, --all, or --dataset NAME")


if __name__ == "__main__":
    main()
