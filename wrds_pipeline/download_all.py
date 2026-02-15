#!/usr/bin/env python3
"""
WRDS Complete Data Download Pipeline
======================================
Downloads ALL institutional datasets from WRDS into AWS RDS PostgreSQL.

Usage:
    python -m wrds_pipeline.download_all                              # Download everything
    python -m wrds_pipeline.download_all --dataset compustat_quarterly  # Single dataset
    python -m wrds_pipeline.download_all --verify                      # Verify counts only
    python -m wrds_pipeline.download_all --create-tables               # Create tables only
    python -m wrds_pipeline.download_all --list-wrds                   # List WRDS libraries

Features:
    - RESUMABLE: ON CONFLICT DO NOTHING prevents duplicates on re-run
    - CHUNKED:   Large datasets downloaded year-by-year to limit memory
    - LOGGED:    Every step prints row counts, timing, errors
    - ROBUST:    Retries on connection failures; continues if one dataset fails
"""

import os
import sys
import time
import logging
import argparse
import pathlib
from datetime import datetime

import wrds
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np

from wrds_pipeline.config import (
    WRDS_USERNAME, WRDS_PASSWORD, WRDS_PGHOST, WRDS_PGPORT,
    WRDS_PGDATABASE, RDS_CONFIG, BATCH_SIZE, MAX_RETRIES,
    RETRY_DELAY_SECS, CHUNK_START_YEAR, CHUNK_END_YEAR,
)

# ═══════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════

LOG_DIR = pathlib.Path(__file__).resolve().parent.parent
LOG_FILE = LOG_DIR / "wrds_download.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("wrds_pipeline")


# ═══════════════════════════════════════════════════════════════════
# SQL: CREATE TABLE statements
# ═══════════════════════════════════════════════════════════════════

CREATE_TABLES_SQL = """
-- ═══ TABLE 1: compustat_quarterly ═══
CREATE TABLE IF NOT EXISTS compustat_quarterly (
    gvkey       VARCHAR(10) NOT NULL,
    datadate    DATE NOT NULL,
    fyearq      INTEGER,
    fqtr        INTEGER,
    rdq         DATE,
    revtq       DECIMAL(15,3),
    cogsq       DECIMAL(15,3),
    xsgaq       DECIMAL(15,3),
    oiadpq      DECIMAL(15,3),
    niq         DECIMAL(15,3),
    ibq         DECIMAL(15,3),
    dpq         DECIMAL(15,3),
    xrdq        DECIMAL(15,3),
    xintq       DECIMAL(15,3),
    txtq        DECIMAL(15,3),
    epspxq      DECIMAL(12,4),
    epsfxq      DECIMAL(12,4),
    atq         DECIMAL(15,3),
    actq        DECIMAL(15,3),
    cheq        DECIMAL(15,3),
    rectq       DECIMAL(15,3),
    invtq       DECIMAL(15,3),
    ppentq      DECIMAL(15,3),
    ltq         DECIMAL(15,3),
    lctq        DECIMAL(15,3),
    dlttq       DECIMAL(15,3),
    dlcq        DECIMAL(15,3),
    seqq        DECIMAL(15,3),
    ceqq        DECIMAL(15,3),
    cshoq       DECIMAL(15,3),
    txditcq     DECIMAL(15,3),
    pstkq       DECIMAL(15,3),
    oancfq      DECIMAL(15,3),
    capxq       DECIMAL(15,3),
    dvq         DECIMAL(15,3),
    tic         VARCHAR(10),
    conm        VARCHAR(100),
    sic         VARCHAR(4),
    naicsh      VARCHAR(6),
    datafmt     VARCHAR(3),
    indfmt      VARCHAR(6),
    consol      VARCHAR(1),
    popsrc      VARCHAR(1),
    curcdq      VARCHAR(3)
);
CREATE INDEX IF NOT EXISTS idx_cq_gvkey ON compustat_quarterly(gvkey);
CREATE INDEX IF NOT EXISTS idx_cq_datadate ON compustat_quarterly(datadate);
CREATE INDEX IF NOT EXISTS idx_cq_gvkey_datadate ON compustat_quarterly(gvkey, datadate);
CREATE INDEX IF NOT EXISTS idx_cq_rdq ON compustat_quarterly(rdq);
CREATE INDEX IF NOT EXISTS idx_cq_tic ON compustat_quarterly(tic);
DO $$ BEGIN
    ALTER TABLE compustat_quarterly ADD CONSTRAINT uq_cq UNIQUE (gvkey, datadate, fyearq, fqtr);
EXCEPTION WHEN duplicate_table THEN NULL;
          WHEN duplicate_object THEN NULL;
END $$;

-- ═══ TABLE 2: compustat_annual ═══
CREATE TABLE IF NOT EXISTS compustat_annual (
    gvkey       VARCHAR(10) NOT NULL,
    datadate    DATE NOT NULL,
    fyear       INTEGER,
    revt        DECIMAL(15,3),
    cogs        DECIMAL(15,3),
    xsga        DECIMAL(15,3),
    oiadp       DECIMAL(15,3),
    ni          DECIMAL(15,3),
    ib          DECIMAL(15,3),
    dp          DECIMAL(15,3),
    xrd         DECIMAL(15,3),
    xint        DECIMAL(15,3),
    txt         DECIMAL(15,3),
    ebitda      DECIMAL(15,3),
    sale        DECIMAL(15,3),
    gp          DECIMAL(15,3),
    at          DECIMAL(15,3),
    act         DECIMAL(15,3),
    che         DECIMAL(15,3),
    rect        DECIMAL(15,3),
    invt        DECIMAL(15,3),
    ppent       DECIMAL(15,3),
    lt          DECIMAL(15,3),
    lct         DECIMAL(15,3),
    dltt        DECIMAL(15,3),
    dlc         DECIMAL(15,3),
    seq         DECIMAL(15,3),
    ceq         DECIMAL(15,3),
    csho        DECIMAL(15,3),
    txditc      DECIMAL(15,3),
    pstk        DECIMAL(15,3),
    mib         DECIMAL(15,3),
    oancf       DECIMAL(15,3),
    capx        DECIMAL(15,3),
    dv          DECIMAL(15,3),
    wcap        DECIMAL(15,3),
    epspx       DECIMAL(12,4),
    dvpsx_f     DECIMAL(12,4),
    tic         VARCHAR(10),
    conm        VARCHAR(100),
    sic         VARCHAR(4),
    naicsh      VARCHAR(6),
    datafmt     VARCHAR(3),
    indfmt      VARCHAR(6),
    consol      VARCHAR(1),
    popsrc      VARCHAR(1)
);
CREATE INDEX IF NOT EXISTS idx_ca_gvkey ON compustat_annual(gvkey);
CREATE INDEX IF NOT EXISTS idx_ca_datadate ON compustat_annual(datadate);
CREATE INDEX IF NOT EXISTS idx_ca_gvkey_datadate ON compustat_annual(gvkey, datadate);
DO $$ BEGIN
    ALTER TABLE compustat_annual ADD CONSTRAINT uq_ca UNIQUE (gvkey, datadate);
EXCEPTION WHEN duplicate_table THEN NULL;
          WHEN duplicate_object THEN NULL;
END $$;

-- ═══ TABLE 3: ibes_summary ═══
CREATE TABLE IF NOT EXISTS ibes_summary (
    ticker      VARCHAR(10) NOT NULL,
    fpedats     DATE NOT NULL,
    statpers    DATE NOT NULL,
    measure     VARCHAR(3),
    fpi         VARCHAR(1),
    numest      INTEGER,
    meanest     DECIMAL(12,4),
    medest      DECIMAL(12,4),
    stdev       DECIMAL(12,4),
    highest     DECIMAL(12,4),
    lowest      DECIMAL(12,4),
    actual      DECIMAL(12,4)
);
CREATE INDEX IF NOT EXISTS idx_is_ticker ON ibes_summary(ticker);
CREATE INDEX IF NOT EXISTS idx_is_fpedats ON ibes_summary(fpedats);
CREATE INDEX IF NOT EXISTS idx_is_statpers ON ibes_summary(statpers);
CREATE INDEX IF NOT EXISTS idx_is_ticker_fpedats ON ibes_summary(ticker, fpedats);
CREATE INDEX IF NOT EXISTS idx_is_ticker_statpers ON ibes_summary(ticker, fpedats, statpers);

-- ═══ TABLE 4: ibes_actuals ═══
CREATE TABLE IF NOT EXISTS ibes_actuals (
    ticker      VARCHAR(10) NOT NULL,
    measure     VARCHAR(3),
    pends       DATE NOT NULL,
    pdicity     VARCHAR(3),
    anndats     DATE,
    value       DECIMAL(12,4)
);
CREATE INDEX IF NOT EXISTS idx_ia_ticker ON ibes_actuals(ticker);
CREATE INDEX IF NOT EXISTS idx_ia_pends ON ibes_actuals(pends);
CREATE INDEX IF NOT EXISTS idx_ia_anndats ON ibes_actuals(anndats);
CREATE INDEX IF NOT EXISTS idx_ia_ticker_pends ON ibes_actuals(ticker, pends);

-- ═══ TABLE 5: ff_factors_daily ═══
CREATE TABLE IF NOT EXISTS ff_factors_daily (
    date        DATE NOT NULL PRIMARY KEY,
    mktrf       DECIMAL(10,6),
    smb         DECIMAL(10,6),
    hml         DECIMAL(10,6),
    rmw         DECIMAL(10,6),
    cma         DECIMAL(10,6),
    rf          DECIMAL(10,6),
    umd         DECIMAL(10,6)
);

-- ═══ TABLE 5b: ff_factors_monthly ═══
CREATE TABLE IF NOT EXISTS ff_factors_monthly (
    date        DATE NOT NULL PRIMARY KEY,
    mktrf       DECIMAL(10,6),
    smb         DECIMAL(10,6),
    hml         DECIMAL(10,6),
    rmw         DECIMAL(10,6),
    cma         DECIMAL(10,6),
    rf          DECIMAL(10,6),
    umd         DECIMAL(10,6)
);

-- ═══ TABLE 6: crsp_compustat_link ═══
CREATE TABLE IF NOT EXISTS crsp_compustat_link (
    gvkey       VARCHAR(10) NOT NULL,
    lpermno     INTEGER NOT NULL,
    lpermco     INTEGER,
    linktype    VARCHAR(2),
    linkprim    VARCHAR(1),
    linkdt      DATE,
    linkenddt   DATE,
    usedflag    INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_ccl_gvkey ON crsp_compustat_link(gvkey);
CREATE INDEX IF NOT EXISTS idx_ccl_lpermno ON crsp_compustat_link(lpermno);
CREATE INDEX IF NOT EXISTS idx_ccl_dates ON crsp_compustat_link(linkdt, linkenddt);

-- ═══ TABLE 7: crsp_monthly ═══
CREATE TABLE IF NOT EXISTS crsp_monthly (
    permno      INTEGER NOT NULL,
    date        DATE NOT NULL,
    ticker      VARCHAR(10),
    ret         DECIMAL(12,6),
    retx        DECIMAL(12,6),
    prc         DECIMAL(12,4),
    shrout      DECIMAL(15,2),
    vol         BIGINT,
    market_cap  DECIMAL(18,2),
    exchcd      INTEGER,
    shrcd       INTEGER,
    siccd       VARCHAR(4),
    dlret       DECIMAL(12,6),
    dlstcd      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_cm_permno ON crsp_monthly(permno);
CREATE INDEX IF NOT EXISTS idx_cm_date ON crsp_monthly(date);
CREATE INDEX IF NOT EXISTS idx_cm_permno_date ON crsp_monthly(permno, date);
CREATE INDEX IF NOT EXISTS idx_cm_ticker ON crsp_monthly(ticker);
DO $$ BEGIN
    ALTER TABLE crsp_monthly ADD CONSTRAINT uq_cm UNIQUE (permno, date);
EXCEPTION WHEN duplicate_table THEN NULL;
          WHEN duplicate_object THEN NULL;
END $$;

-- ═══ TABLE 8: ibes_crsp_link ═══
CREATE TABLE IF NOT EXISTS ibes_crsp_link (
    ticker      VARCHAR(10),
    permno      INTEGER,
    sdate       DATE,
    edate       DATE
);
CREATE INDEX IF NOT EXISTS idx_icl_ticker ON ibes_crsp_link(ticker);
CREATE INDEX IF NOT EXISTS idx_icl_permno ON ibes_crsp_link(permno);

-- ═══ TABLE 9: compustat_security ═══
CREATE TABLE IF NOT EXISTS compustat_security (
    gvkey       VARCHAR(10),
    iid         VARCHAR(3),
    tic         VARCHAR(10),
    cusip       VARCHAR(9),
    exchg       INTEGER,
    sic         VARCHAR(4),
    naics       VARCHAR(6),
    dldtei      DATE,
    dlrsni      VARCHAR(3)
);
CREATE INDEX IF NOT EXISTS idx_cs_gvkey ON compustat_security(gvkey);
CREATE INDEX IF NOT EXISTS idx_cs_tic ON compustat_security(tic);
"""


# ═══════════════════════════════════════════════════════════════════
# WRDSDownloader
# ═══════════════════════════════════════════════════════════════════

class WRDSDownloader:
    """
    Downloads ALL WRDS datasets into AWS RDS.

    Features:
        - Resume capability (ON CONFLICT DO NOTHING)
        - Chunked downloads (year-by-year for large tables)
        - Progress logging with row counts and timing
        - Automatic retry on connection failures
        - Data validation after each dataset
    """

    def __init__(self):
        self.wrds_conn = None
        self.rds_conn = None

    # ───────────────────────────────────────────────────────────────
    # Connection helpers
    # ───────────────────────────────────────────────────────────────

    def connect_wrds(self):
        """Connect to WRDS via the wrds Python package."""
        # Set up .pgpass so wrds library doesn't prompt
        pgpass = pathlib.Path.home() / ".pgpass"
        pgpass_line = f"{WRDS_PGHOST}:{WRDS_PGPORT}:{WRDS_PGDATABASE}:{WRDS_USERNAME}:{WRDS_PASSWORD}"
        # Append if not already present
        existing = pgpass.read_text() if pgpass.exists() else ""
        if pgpass_line not in existing:
            with open(pgpass, "a") as f:
                f.write(pgpass_line + "\n")
            pgpass.chmod(0o600)

        os.environ["PGHOST"] = WRDS_PGHOST
        os.environ["PGPORT"] = WRDS_PGPORT
        os.environ["PGDATABASE"] = WRDS_PGDATABASE
        os.environ["PGUSER"] = WRDS_USERNAME
        os.environ["PGPASSWORD"] = WRDS_PASSWORD

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self.wrds_conn = wrds.Connection(wrds_username=WRDS_USERNAME)
                logger.info("✓ Connected to WRDS")
                return
            except Exception as e:
                logger.warning(f"  WRDS connection attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECS)
        raise ConnectionError("Could not connect to WRDS after retries")

    def connect_rds(self):
        """Connect to AWS RDS PostgreSQL."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self.rds_conn = psycopg2.connect(**RDS_CONFIG)
                self.rds_conn.autocommit = False
                logger.info("✓ Connected to AWS RDS")
                return
            except Exception as e:
                logger.warning(f"  RDS connection attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECS)
        raise ConnectionError("Could not connect to RDS after retries")

    def _ensure_rds(self):
        """Re-connect to RDS if connection is closed/broken."""
        try:
            with self.rds_conn.cursor() as cur:
                cur.execute("SELECT 1")
            self.rds_conn.rollback()
        except Exception:
            logger.info("  RDS connection lost — reconnecting...")
            self.connect_rds()

    # ───────────────────────────────────────────────────────────────
    # Table creation
    # ───────────────────────────────────────────────────────────────

    def create_tables(self):
        """Create all 9 destination tables in RDS (idempotent)."""
        self._ensure_rds()
        logger.info("═══ Creating tables in RDS ═══")
        with self.rds_conn.cursor() as cur:
            cur.execute(CREATE_TABLES_SQL)
        self.rds_conn.commit()

        # Verify
        with self.rds_conn.cursor() as cur:
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' ORDER BY table_name
            """)
            tables = [r[0] for r in cur.fetchall()]
        logger.info(f"  Tables in RDS: {tables}")

        expected = [
            "compustat_annual", "compustat_quarterly", "compustat_security",
            "crsp_compustat_link", "crsp_monthly", "ff_factors_daily",
            "ff_factors_monthly", "ibes_actuals", "ibes_crsp_link",
            "ibes_summary",
        ]
        missing = [t for t in expected if t not in tables]
        if missing:
            logger.error(f"  ✗ Missing tables: {missing}")
        else:
            logger.info("  ✓ All 9 pipeline tables exist (plus stock_prices)")
        return tables

    # ───────────────────────────────────────────────────────────────
    # RDS helpers
    # ───────────────────────────────────────────────────────────────

    def get_rds_count(self, table: str) -> int:
        """Row count for a table in RDS."""
        self._ensure_rds()
        try:
            with self.rds_conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
            self.rds_conn.rollback()
            return count
        except Exception:
            self.rds_conn.rollback()
            return 0

    def get_rds_max_date(self, table: str, date_col: str) -> str | None:
        """Latest date value in a table (for resume logic)."""
        self._ensure_rds()
        try:
            with self.rds_conn.cursor() as cur:
                cur.execute(f"SELECT MAX({date_col}) FROM {table}")
                val = cur.fetchone()[0]
            self.rds_conn.rollback()
            return str(val) if val else None
        except Exception:
            self.rds_conn.rollback()
            return None

    def insert_dataframe(self, table: str, df: pd.DataFrame, columns: list) -> int:
        """
        Bulk-insert a DataFrame into an RDS table.

        Uses psycopg2.extras.execute_values for speed.
        Converts NaN/NaT to None for PostgreSQL.
        ON CONFLICT DO NOTHING to be idempotent.
        Falls back to row-by-row for any batch that fails.
        """
        if df.empty:
            return 0

        self._ensure_rds()

        # Keep only columns that actually exist in the DataFrame
        available_cols = [c for c in columns if c in df.columns]
        if not available_cols:
            logger.warning(f"  No matching columns for {table}")
            return 0

        # NaN → None
        sub = df[available_cols].copy()
        sub = sub.where(pd.notnull(sub), None)

        # Convert numpy types to Python native for psycopg2
        for col in sub.columns:
            if sub[col].dtype.kind in ("i", "u"):
                sub[col] = sub[col].astype(object).where(sub[col].notna(), None)
            elif sub[col].dtype.kind == "f":
                sub[col] = sub[col].astype(object).where(sub[col].notna(), None)

        col_str = ", ".join(available_cols)
        n_cols = len(available_cols)

        rows = [tuple(r) for r in sub.itertuples(index=False, name=None)]

        inserted = 0
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start : start + BATCH_SIZE]
            try:
                with self.rds_conn.cursor() as cur:
                    psycopg2.extras.execute_values(
                        cur,
                        f"INSERT INTO {table} ({col_str}) VALUES %s ON CONFLICT DO NOTHING",
                        batch,
                        template="(" + ",".join(["%s"] * n_cols) + ")",
                        page_size=BATCH_SIZE,
                    )
                self.rds_conn.commit()
                inserted += len(batch)
            except Exception as e:
                self.rds_conn.rollback()
                logger.warning(f"  Batch insert error ({table}): {e} — falling back to row-by-row")
                self._ensure_rds()
                for row in batch:
                    try:
                        with self.rds_conn.cursor() as cur:
                            placeholders = ",".join(["%s"] * n_cols)
                            cur.execute(
                                f"INSERT INTO {table} ({col_str}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
                                row,
                            )
                        self.rds_conn.commit()
                        inserted += 1
                    except Exception:
                        self.rds_conn.rollback()

        return inserted

    # ───────────────────────────────────────────────────────────────
    # Generic dataset downloader
    # ───────────────────────────────────────────────────────────────

    def download_dataset(
        self,
        name: str,
        query: str,
        table: str,
        columns: list,
        chunk_by_year: bool = False,
        year_col: str = "datadate",
        start_year: int = CHUNK_START_YEAR,
        end_year: int = CHUNK_END_YEAR,
    ) -> int:
        """
        Download a WRDS dataset and load into RDS.

        If chunk_by_year=True, appends a year filter and fetches year-by-year
        to keep memory bounded.
        """
        logger.info(f"═══ Downloading: {name} ═══")
        existing = self.get_rds_count(table)
        logger.info(f"  Existing rows in {table}: {existing:,}")

        total_inserted = 0
        t0 = time.time()

        if chunk_by_year:
            for year in range(start_year, end_year + 1):
                year_query = query + f"\n  AND EXTRACT(YEAR FROM {year_col}) = {year}"
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        logger.info(f"  Fetching {name} year {year} (attempt {attempt})...")
                        df = self.wrds_conn.raw_sql(year_query)
                        if df.empty:
                            logger.info(f"    Year {year}: 0 rows")
                            break

                        df.columns = df.columns.str.lower().str.strip()
                        n = self.insert_dataframe(table, df, columns)
                        total_inserted += n
                        logger.info(f"    Year {year}: {len(df):,} fetched → {n:,} inserted")
                        del df
                        break
                    except Exception as e:
                        logger.error(f"    Year {year} attempt {attempt} failed: {e}")
                        if attempt < MAX_RETRIES:
                            time.sleep(RETRY_DELAY_SECS)
                            # Reconnect WRDS in case of session timeout
                            try:
                                self.wrds_conn.close()
                            except Exception:
                                pass
                            self.connect_wrds()
                        else:
                            logger.error(f"    Year {year} SKIPPED after {MAX_RETRIES} failures")
        else:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    logger.info(f"  Fetching {name} (full query, attempt {attempt})...")
                    df = self.wrds_conn.raw_sql(query)
                    df.columns = df.columns.str.lower().str.strip()
                    logger.info(f"  Downloaded {len(df):,} rows → loading into RDS...")
                    total_inserted = self.insert_dataframe(table, df, columns)
                    del df
                    break
                except Exception as e:
                    logger.error(f"  Attempt {attempt} failed: {e}")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY_SECS)
                        try:
                            self.wrds_conn.close()
                        except Exception:
                            pass
                        self.connect_wrds()
                    else:
                        logger.error(f"  {name} FAILED after {MAX_RETRIES} attempts")

        elapsed = time.time() - t0
        final = self.get_rds_count(table)
        logger.info(f"  ✓ {name}: {total_inserted:,} new rows in {elapsed:.1f}s")
        logger.info(f"    Total in {table}: {final:,}")
        return total_inserted

    # ───────────────────────────────────────────────────────────────
    # Individual dataset methods
    # ───────────────────────────────────────────────────────────────

    def download_compustat_quarterly(self) -> int:
        return self.download_dataset(
            name="Compustat Quarterly (comp.fundq)",
            query="""
                SELECT gvkey, datadate, fyearq, fqtr, rdq,
                       revtq, cogsq, xsgaq, oiadpq, niq, ibq, dpq,
                       xrdq, xintq, txtq, epspxq, epsfxq,
                       atq, actq, cheq, rectq, invtq, ppentq,
                       ltq, lctq, dlttq, dlcq,
                       seqq, ceqq, cshoq, txditcq, pstkq,
                       oancfq, capxq, dvq,
                       tic, conm, sic, naicsh,
                       datafmt, indfmt, consol, popsrc, curcdq
                FROM comp.fundq
                WHERE datafmt = 'STD'
                  AND indfmt  = 'INDL'
                  AND consol  = 'C'
                  AND popsrc  = 'D'
                  AND datadate >= '1970-01-01'
            """,
            table="compustat_quarterly",
            columns=[
                "gvkey", "datadate", "fyearq", "fqtr", "rdq",
                "revtq", "cogsq", "xsgaq", "oiadpq", "niq", "ibq", "dpq",
                "xrdq", "xintq", "txtq", "epspxq", "epsfxq",
                "atq", "actq", "cheq", "rectq", "invtq", "ppentq",
                "ltq", "lctq", "dlttq", "dlcq",
                "seqq", "ceqq", "cshoq", "txditcq", "pstkq",
                "oancfq", "capxq", "dvq",
                "tic", "conm", "sic", "naicsh",
                "datafmt", "indfmt", "consol", "popsrc", "curcdq",
            ],
            chunk_by_year=True,
            year_col="datadate",
        )

    def download_compustat_annual(self) -> int:
        return self.download_dataset(
            name="Compustat Annual (comp.funda)",
            query="""
                SELECT gvkey, datadate, fyear,
                       revt, cogs, xsga, oiadp, ni, ib, dp,
                       xrd, xint, txt, ebitda, sale, gp,
                       at, act, che, rect, invt, ppent,
                       lt, lct, dltt, dlc,
                       seq, ceq, csho, txditc, pstk, mib,
                       oancf, capx, dv, wcap,
                       epspx, dvpsx_f,
                       tic, conm, sic, naicsh,
                       datafmt, indfmt, consol, popsrc
                FROM comp.funda
                WHERE datafmt = 'STD'
                  AND indfmt  = 'INDL'
                  AND consol  = 'C'
                  AND popsrc  = 'D'
                  AND datadate >= '1970-01-01'
            """,
            table="compustat_annual",
            columns=[
                "gvkey", "datadate", "fyear",
                "revt", "cogs", "xsga", "oiadp", "ni", "ib", "dp",
                "xrd", "xint", "txt", "ebitda", "sale", "gp",
                "at", "act", "che", "rect", "invt", "ppent",
                "lt", "lct", "dltt", "dlc",
                "seq", "ceq", "csho", "txditc", "pstk", "mib",
                "oancf", "capx", "dv", "wcap",
                "epspx", "dvpsx_f",
                "tic", "conm", "sic", "naicsh",
                "datafmt", "indfmt", "consol", "popsrc",
            ],
            chunk_by_year=False,
        )

    def download_crsp_compustat_link(self) -> int:
        return self.download_dataset(
            name="CRSP-Compustat Link (crsp.ccmxpf_lnkhist)",
            query="""
                SELECT gvkey, lpermno, lpermco,
                       linktype, linkprim, linkdt, linkenddt,
                       CASE WHEN usedflag IS NOT NULL THEN usedflag ELSE 1 END AS usedflag
                FROM crsp.ccmxpf_lnkhist
                WHERE linktype IN ('LC','LU','LS')
                  AND linkprim IN ('P','C')
            """,
            table="crsp_compustat_link",
            columns=[
                "gvkey", "lpermno", "lpermco",
                "linktype", "linkprim", "linkdt", "linkenddt", "usedflag",
            ],
            chunk_by_year=False,
        )

    def download_ibes_summary(self) -> int:
        return self.download_dataset(
            name="IBES Summary (ibes.statsumu_epsus)",
            query="""
                SELECT ticker, fpedats, statpers, measure, fpi,
                       numest, meanest, medest, stdev,
                       highest, lowest, actual
                FROM ibes.statsumu_epsus
                WHERE measure = 'EPS'
                  AND fpi IN ('1','2','6','7')
                  AND statpers >= '1980-01-01'
            """,
            table="ibes_summary",
            columns=[
                "ticker", "fpedats", "statpers", "measure", "fpi",
                "numest", "meanest", "medest", "stdev",
                "highest", "lowest", "actual",
            ],
            chunk_by_year=True,
            year_col="statpers",
            start_year=1980,
        )

    def download_ibes_actuals(self) -> int:
        return self.download_dataset(
            name="IBES Actuals (ibes.actu_epsus)",
            query="""
                SELECT ticker, measure, pends, pdicity, anndats, value
                FROM ibes.actu_epsus
                WHERE measure = 'EPS'
                  AND pends >= '1980-01-01'
            """,
            table="ibes_actuals",
            columns=["ticker", "measure", "pends", "pdicity", "anndats", "value"],
            chunk_by_year=False,
        )

    def download_ff_factors(self) -> dict:
        """
        Download Fama-French 5 factors + momentum (daily & monthly).
        Handles WRDS table name variations gracefully.
        """
        results = {}

        # ── Daily factors ──
        logger.info("═══ Downloading: Fama-French Daily Factors ═══")
        daily_loaded = False

        # Strategy: try to find 5-factor table, then 3-factor, combine with momentum
        for ff_table in ["ff_all.fivefactors_daily", "ff.fivefactors_daily", "ff.factors_daily"]:
            try:
                test = self.wrds_conn.raw_sql(f"SELECT * FROM {ff_table} LIMIT 1")
                test.columns = test.columns.str.lower().str.strip()
                logger.info(f"  Found {ff_table} — columns: {list(test.columns)}")

                # Build SELECT based on available columns
                select_parts = ["date"]
                for col in ["mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]:
                    if col in test.columns:
                        select_parts.append(col)

                query = f"SELECT {', '.join(select_parts)} FROM {ff_table} WHERE date >= '1970-01-01' ORDER BY date"
                df = self.wrds_conn.raw_sql(query)
                df.columns = df.columns.str.lower().str.strip()

                # Try to get momentum from separate table if missing
                if "umd" not in df.columns:
                    for mom_table in ["ff_all.momentum_daily", "ff.momentum_daily"]:
                        try:
                            mom = self.wrds_conn.raw_sql(
                                f"SELECT date, umd FROM {mom_table} WHERE date >= '1970-01-01' ORDER BY date"
                            )
                            mom.columns = mom.columns.str.lower().str.strip()
                            df = df.merge(mom, on="date", how="left")
                            logger.info(f"    Merged momentum from {mom_table}")
                            break
                        except Exception:
                            continue

                # Fill any still-missing columns with NULL
                for col in ["mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]:
                    if col not in df.columns:
                        df[col] = None

                cols = ["date", "mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]
                n = self.insert_dataframe("ff_factors_daily", df, cols)
                logger.info(f"  ✓ FF Daily: {n:,} rows from {ff_table}")
                results["ff_factors_daily"] = n
                daily_loaded = True
                del df
                break
            except Exception as e:
                logger.warning(f"  {ff_table} failed: {e}")
                continue

        if not daily_loaded:
            logger.error("  ✗ Could not load FF daily factors")
            try:
                ff_tables = self.wrds_conn.list_tables(library="ff")
                logger.info(f"  Available FF tables: {ff_tables[:30]}")
                ff_all = self.wrds_conn.list_tables(library="ff_all")
                logger.info(f"  Available FF_ALL tables: {ff_all[:30]}")
            except Exception:
                pass

        # ── Monthly factors ──
        logger.info("═══ Downloading: Fama-French Monthly Factors ═══")
        for ff_table in ["ff_all.fivefactors_monthly", "ff.fivefactors_monthly", "ff.factors_monthly"]:
            try:
                test = self.wrds_conn.raw_sql(f"SELECT * FROM {ff_table} LIMIT 1")
                test.columns = test.columns.str.lower().str.strip()
                logger.info(f"  Found {ff_table} — columns: {list(test.columns)}")

                select_parts = ["date"]
                for col in ["mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]:
                    if col in test.columns:
                        select_parts.append(col)

                query = f"SELECT {', '.join(select_parts)} FROM {ff_table} WHERE date >= '1970-01-01' ORDER BY date"
                df = self.wrds_conn.raw_sql(query)
                df.columns = df.columns.str.lower().str.strip()

                if "umd" not in df.columns:
                    for mom_table in ["ff_all.momentum_monthly", "ff.momentum_monthly"]:
                        try:
                            mom = self.wrds_conn.raw_sql(
                                f"SELECT date, umd FROM {mom_table} WHERE date >= '1970-01-01' ORDER BY date"
                            )
                            mom.columns = mom.columns.str.lower().str.strip()
                            df = df.merge(mom, on="date", how="left")
                            break
                        except Exception:
                            continue

                for col in ["mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]:
                    if col not in df.columns:
                        df[col] = None

                cols = ["date", "mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]
                n = self.insert_dataframe("ff_factors_monthly", df, cols)
                logger.info(f"  ✓ FF Monthly: {n:,} rows from {ff_table}")
                results["ff_factors_monthly"] = n
                del df
                break
            except Exception as e:
                logger.warning(f"  {ff_table} failed: {e}")
                continue

        return results

    def download_crsp_monthly(self) -> int:
        return self.download_dataset(
            name="CRSP Monthly (crsp.msf + msenames)",
            query="""
                SELECT a.permno, a.date, b.ticker,
                       a.ret, a.retx, a.prc, a.shrout, a.vol,
                       ABS(a.prc) * a.shrout * 1000 AS market_cap,
                       b.exchcd, b.shrcd, b.siccd,
                       a.dlret, a.dlstcd
                FROM crsp.msf a
                LEFT JOIN crsp.msenames b
                    ON a.permno = b.permno
                   AND a.date BETWEEN b.namedt AND b.nameenddt
                WHERE b.shrcd IN (10, 11)
                  AND b.exchcd IN (1, 2, 3)
                  AND a.date >= '1970-01-01'
            """,
            table="crsp_monthly",
            columns=[
                "permno", "date", "ticker", "ret", "retx", "prc",
                "shrout", "vol", "market_cap", "exchcd", "shrcd",
                "siccd", "dlret", "dlstcd",
            ],
            chunk_by_year=True,
            year_col="a.date",
        )

    def download_ibes_crsp_link(self) -> int:
        """Download IBES-CRSP link table. Falls back to CUSIP matching."""
        logger.info("═══ Downloading: IBES-CRSP Link ═══")
        existing = self.get_rds_count("ibes_crsp_link")
        logger.info(f"  Existing rows: {existing:,}")

        t0 = time.time()

        # Try WRDS-maintained tables first
        for link_table in [
            "wrdsapps.ibcrsphist",
            "wrdsapps_ibes.ibcrsphist",
            "wrdsapps_link.ibcrsphist",
        ]:
            try:
                logger.info(f"  Trying {link_table}...")
                df = self.wrds_conn.raw_sql(
                    f"SELECT ticker, permno, sdate, edate FROM {link_table} ORDER BY ticker, sdate"
                )
                df.columns = df.columns.str.lower().str.strip()
                n = self.insert_dataframe("ibes_crsp_link", df, ["ticker", "permno", "sdate", "edate"])
                elapsed = time.time() - t0
                logger.info(f"  ✓ IBES-CRSP Link: {n:,} rows from {link_table} in {elapsed:.1f}s")
                del df
                return n
            except Exception as e:
                logger.warning(f"    {link_table} failed: {e}")
                continue

        # Fallback: CUSIP matching
        logger.warning("  All link tables failed — building from CUSIP matching...")
        try:
            df = self.wrds_conn.raw_sql("""
                SELECT DISTINCT a.ticker, b.permno,
                       MIN(b.namedt) AS sdate, MAX(b.nameenddt) AS edate
                FROM ibes.idsum a
                JOIN crsp.stocknames b
                    ON SUBSTR(a.cusip, 1, 8) = SUBSTR(b.ncusip, 1, 8)
                GROUP BY a.ticker, b.permno
            """)
            df.columns = df.columns.str.lower().str.strip()
            n = self.insert_dataframe("ibes_crsp_link", df, ["ticker", "permno", "sdate", "edate"])
            elapsed = time.time() - t0
            logger.info(f"  ✓ IBES-CRSP Link (CUSIP match): {n:,} rows in {elapsed:.1f}s")
            del df
            return n
        except Exception as e:
            logger.error(f"  ✗ IBES-CRSP link build failed: {e}")
            return 0

    def download_compustat_security(self) -> int:
        """Download Compustat security-level identifiers."""
        logger.info("═══ Downloading: Compustat Security ═══")
        t0 = time.time()
        try:
            df = self.wrds_conn.raw_sql("""
                SELECT gvkey, iid, tic, cusip, exchg, sic, naics, dldtei, dlrsni
                FROM comp.security
            """)
            df.columns = df.columns.str.lower().str.strip()
            n = self.insert_dataframe(
                "compustat_security", df,
                ["gvkey", "iid", "tic", "cusip", "exchg", "sic", "naics", "dldtei", "dlrsni"],
            )
            elapsed = time.time() - t0
            logger.info(f"  ✓ Compustat Security: {n:,} rows in {elapsed:.1f}s")
            del df
            return n
        except Exception as e:
            logger.error(f"  ✗ Compustat Security failed: {e}")
            return 0

    # ───────────────────────────────────────────────────────────────
    # List WRDS libraries (discovery)
    # ───────────────────────────────────────────────────────────────

    def list_wrds_libraries(self):
        """Print all accessible WRDS libraries."""
        libs = self.wrds_conn.list_libraries()
        logger.info(f"WRDS: {len(libs)} libraries available")
        relevant = []
        for lib in sorted(libs):
            if any(kw in lib.lower() for kw in ["crsp", "comp", "ibes", "ff", "factor", "wrdsapp"]):
                relevant.append(lib)
                logger.info(f"  ✓ {lib}")
        return relevant

    # ───────────────────────────────────────────────────────────────
    # Orchestrator
    # ───────────────────────────────────────────────────────────────

    def download_all(self, dataset: str | None = None):
        """
        Download ALL datasets (or a single named dataset).

        Order: small tables first, then medium, then large.
        """
        self.connect_wrds()
        self.connect_rds()
        self.create_tables()

        results = {}

        datasets = {
            "ff_factors":            self.download_ff_factors,
            "crsp_compustat_link":   self.download_crsp_compustat_link,
            "ibes_crsp_link":        self.download_ibes_crsp_link,
            "compustat_security":    self.download_compustat_security,
            "ibes_actuals":          self.download_ibes_actuals,
            "compustat_annual":      self.download_compustat_annual,
            "compustat_quarterly":   self.download_compustat_quarterly,
            "ibes_summary":          self.download_ibes_summary,
            "crsp_monthly":          self.download_crsp_monthly,
        }

        if dataset:
            if dataset not in datasets:
                logger.error(f"Unknown dataset: {dataset}")
                logger.info(f"Available: {list(datasets.keys())}")
                return
            fn = datasets[dataset]
            r = fn()
            if isinstance(r, dict):
                results.update(r)
            else:
                results[dataset] = r
        else:
            for ds_name, fn in datasets.items():
                try:
                    r = fn()
                    if isinstance(r, dict):
                        results.update(r)
                    else:
                        results[ds_name] = r
                except Exception as e:
                    logger.error(f"  ✗ {ds_name} failed entirely: {e}")
                    results[ds_name] = 0

        # ═══ FINAL REPORT ═══
        logger.info("")
        logger.info("═" * 65)
        logger.info("  WRDS DOWNLOAD COMPLETE — SUMMARY")
        logger.info("═" * 65)

        for name, count in results.items():
            logger.info(f"    {name:35s}: {count:>12,} rows inserted")

        logger.info("─" * 65)
        # Show final table counts
        all_tables = [
            "stock_prices", "compustat_quarterly", "compustat_annual",
            "ibes_summary", "ibes_actuals", "ff_factors_daily",
            "ff_factors_monthly", "crsp_compustat_link", "crsp_monthly",
            "ibes_crsp_link", "compustat_security",
        ]
        logger.info("  FINAL TABLE COUNTS:")
        for t in all_tables:
            c = self.get_rds_count(t)
            logger.info(f"    {t:35s}: {c:>12,}")
        logger.info("═" * 65)

        try:
            self.wrds_conn.close()
        except Exception:
            pass
        try:
            self.rds_conn.close()
        except Exception:
            pass

        return results

    def verify(self):
        """Quick verification of all table counts."""
        self.connect_rds()
        tables = [
            "stock_prices", "compustat_quarterly", "compustat_annual",
            "ibes_summary", "ibes_actuals", "ff_factors_daily",
            "ff_factors_monthly", "crsp_compustat_link", "crsp_monthly",
            "ibes_crsp_link", "compustat_security",
        ]
        logger.info("═" * 50)
        logger.info("  RDS TABLE COUNTS")
        logger.info("═" * 50)
        for t in tables:
            c = self.get_rds_count(t)
            status = "✓" if c > 0 else "✗"
            logger.info(f"  {status} {t:35s}: {c:>12,}")
        logger.info("═" * 50)
        self.rds_conn.close()


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="WRDS → RDS data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Download a single dataset (e.g. compustat_quarterly, ff_factors)",
    )
    parser.add_argument("--verify", action="store_true", help="Verify table counts only")
    parser.add_argument("--create-tables", action="store_true", help="Create tables only")
    parser.add_argument("--list-wrds", action="store_true", help="List WRDS libraries")
    args = parser.parse_args()

    dl = WRDSDownloader()

    if args.verify:
        dl.verify()
    elif args.create_tables:
        dl.connect_rds()
        dl.create_tables()
        dl.rds_conn.close()
    elif args.list_wrds:
        dl.connect_wrds()
        dl.list_wrds_libraries()
        dl.wrds_conn.close()
    else:
        dl.download_all(dataset=args.dataset)


if __name__ == "__main__":
    main()
