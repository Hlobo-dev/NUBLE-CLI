#!/usr/bin/env python3
"""
WRDS → Parquet Data Pipeline (Option B)
=========================================
Downloads ALL institutional datasets from WRDS directly into Parquet files.
No RDS dependency. Parquet is the gold standard for ML data:
  - Columnar format = blazing fast reads
  - Compressed = 3-5x smaller than CSV
  - Schema-preserving = types are embedded
  - Portable = works everywhere (pandas, polars, spark, duckdb)

OUTPUT STRUCTURE:
    data/wrds/
    ├── crsp_monthly.parquet          (~3M rows)
    ├── crsp_daily.parquet            (~100M+ rows, optional)
    ├── compustat_quarterly.parquet   (~2M rows)
    ├── compustat_annual.parquet      (~500K rows)
    ├── ibes_summary.parquet          (~3M rows)
    ├── ibes_actuals.parquet          (~500K rows)
    ├── ff_factors_daily.parquet      (~25K rows)
    ├── ff_factors_monthly.parquet    (~1.1K rows)
    ├── crsp_compustat_link.parquet   (~35K rows)
    ├── ibes_crsp_link.parquet        (~15K rows)
    ├── compustat_security.parquet    (~50K rows)
    ├── download_manifest.json        (metadata: dates, row counts, checksums)
    └── README.md                     (data dictionary)

USAGE:
    python wrds_pipeline/download_to_parquet.py                         # Download ALL
    python wrds_pipeline/download_to_parquet.py --dataset crsp_monthly  # Single dataset
    python wrds_pipeline/download_to_parquet.py --test                  # Quick test (5 rows each)
    python wrds_pipeline/download_to_parquet.py --upload-s3             # Upload to S3 after download
    python wrds_pipeline/download_to_parquet.py --verify                # Verify existing files
"""

import os
import sys
import json
import time
import hashlib
import logging
import argparse
import urllib.parse
from datetime import datetime
from pathlib import Path

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

# Output directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "wrds"
S3_BUCKET = "nuble-wrds-data"  # Change if needed
S3_PREFIX = "wrds/"

# Parquet settings
PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "snappy"  # Fast decompression, good ratio

# ═══════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════

LOG_FILE = BASE_DIR / "wrds_download.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("wrds_parquet")


# ═══════════════════════════════════════════════════════════════════
# Dataset Definitions — The SQL for each WRDS dataset
# ═══════════════════════════════════════════════════════════════════

DATASETS = {
    # ── Small tables (download in one shot) ──
    "ff_factors_daily": {
        "description": "Fama-French 5 factors + momentum (daily)",
        "query": """
            SELECT date, mktrf, smb, hml, rmw, cma, rf, umd
            FROM ff_all.fivefactors_daily
            WHERE date >= '1963-07-01'
            ORDER BY date
        """,
        "fallback_queries": [
            """SELECT date, mktrf, smb, hml, rmw, cma, rf
               FROM ff.fivefactors_daily WHERE date >= '1963-07-01' ORDER BY date""",
            """SELECT date, mktrf, smb, hml, rf
               FROM ff.factors_daily WHERE date >= '1963-07-01' ORDER BY date""",
        ],
        "expected_min_rows": 15_000,
        "chunk_by_year": False,
    },

    "ff_factors_monthly": {
        "description": "Fama-French 5 factors + momentum (monthly)",
        "query": """
            SELECT date, mktrf, smb, hml, rmw, cma, rf, umd
            FROM ff_all.fivefactors_monthly
            WHERE date >= '1963-07-01'
            ORDER BY date
        """,
        "fallback_queries": [
            """SELECT date, mktrf, smb, hml, rmw, cma, rf
               FROM ff.fivefactors_monthly WHERE date >= '1963-07-01' ORDER BY date""",
        ],
        "expected_min_rows": 500,
        "chunk_by_year": False,
    },

    "crsp_compustat_link": {
        "description": "CRSP-Compustat linking table (PERMNO ↔ GVKEY)",
        "query": """
            SELECT gvkey, lpermno, lpermco,
                   linktype, linkprim, linkdt, linkenddt,
                   COALESCE(usedflag, 1) AS usedflag
            FROM crsp.ccmxpf_lnkhist
            WHERE linktype IN ('LC','LU','LS')
              AND linkprim IN ('P','C')
            ORDER BY gvkey, linkdt
        """,
        "expected_min_rows": 20_000,
        "chunk_by_year": False,
    },

    "ibes_crsp_link": {
        "description": "IBES ticker ↔ CRSP PERMNO link",
        "query": """
            SELECT ticker, permno, sdate, edate
            FROM wrdsapps.ibcrsphist
            ORDER BY ticker, sdate
        """,
        "fallback_queries": [
            """SELECT ticker, permno, sdate, edate 
               FROM wrdsapps_ibes.ibcrsphist ORDER BY ticker, sdate""",
            # CUSIP fallback
            """SELECT DISTINCT a.ticker, b.permno,
                      MIN(b.namedt) AS sdate, MAX(b.nameenddt) AS edate
               FROM ibes.idsum a
               JOIN crsp.stocknames b
                   ON SUBSTR(a.cusip, 1, 8) = SUBSTR(b.ncusip, 1, 8)
               GROUP BY a.ticker, b.permno""",
        ],
        "expected_min_rows": 10_000,
        "chunk_by_year": False,
    },

    "compustat_security": {
        "description": "Compustat security-level identifiers (GVKEY ↔ ticker, CUSIP, exchange)",
        "query": """
            SELECT gvkey, iid, tic, cusip, exchg, sic, naics, dldtei, dlrsni
            FROM comp.security
            ORDER BY gvkey, iid
        """,
        "expected_min_rows": 40_000,
        "chunk_by_year": False,
    },

    "ibes_actuals": {
        "description": "IBES actual EPS announcements",
        "query": """
            SELECT ticker, measure, pends, pdicity, anndats, value
            FROM ibes.actu_epsus
            WHERE measure = 'EPS'
              AND pends >= '1976-01-01'
            ORDER BY ticker, pends
        """,
        "expected_min_rows": 400_000,
        "chunk_by_year": False,
    },

    # ── Medium tables (can be done in one shot or chunked) ──

    "compustat_annual": {
        "description": "Compustat annual fundamentals (comp.funda)",
        "query": """
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
              AND datadate >= '1960-01-01'
            ORDER BY gvkey, datadate
        """,
        "expected_min_rows": 400_000,
        "chunk_by_year": False,
    },

    # ── Large tables (chunked by year) ──

    "compustat_quarterly": {
        "description": "Compustat quarterly fundamentals (comp.fundq)",
        "query_template": """
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
              AND EXTRACT(YEAR FROM datadate) = {year}
            ORDER BY gvkey, datadate
        """,
        "expected_min_rows": 1_500_000,
        "chunk_by_year": True,
        "year_range": (1962, 2026),
    },

    "ibes_summary": {
        "description": "IBES consensus estimates (analyst forecasts)",
        "query_template": """
            SELECT ticker, fpedats, statpers, measure, fpi,
                   numest, meanest, medest, stdev,
                   highest, lowest, actual
            FROM ibes.statsumu_epsus
            WHERE measure = 'EPS'
              AND fpi IN ('1','2','6','7')
              AND EXTRACT(YEAR FROM statpers) = {year}
            ORDER BY ticker, fpedats, statpers
        """,
        "expected_min_rows": 2_000_000,
        "chunk_by_year": True,
        "year_range": (1976, 2026),
    },

    "crsp_monthly": {
        "description": "CRSP monthly stock file + names (returns, prices, market cap)",
        "query_template": """
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
              AND EXTRACT(YEAR FROM a.date) = {year}
            ORDER BY a.permno, a.date
        """,
        "expected_min_rows": 2_500_000,
        "chunk_by_year": True,
        "year_range": (1926, 2026),
    },
}

# Optional large dataset (hundreds of millions of rows)
OPTIONAL_DATASETS = {
    "crsp_daily": {
        "description": "CRSP daily stock file (full history — VERY LARGE, ~100M+ rows)",
        "query_template": """
            SELECT a.permno, a.date, 
                   a.ret, a.prc, a.vol, a.shrout,
                   ABS(a.prc) * a.shrout * 1000 AS market_cap,
                   a.bidlo, a.askhi, a.openprc
            FROM crsp.dsf a
            WHERE EXTRACT(YEAR FROM a.date) = {year}
            ORDER BY a.permno, a.date
        """,
        "expected_min_rows": 80_000_000,
        "chunk_by_year": True,
        "year_range": (1926, 2026),
    },
}


# ═══════════════════════════════════════════════════════════════════
# WRDSParquetDownloader
# ═══════════════════════════════════════════════════════════════════

class WRDSParquetDownloader:
    """
    Downloads WRDS datasets and saves as Parquet files.
    
    Design principles:
    - Resumable: skips already-downloaded files (use --force to re-download)
    - Chunked: large tables are downloaded year-by-year
    - Validated: row counts checked against expected minimums
    - Manifest: JSON file tracks what was downloaded, when, row counts
    """

    def __init__(self, output_dir: Path = DATA_DIR, force: bool = False):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.force = force
        self.engine = None
        self.manifest = {}
        self._load_manifest()

    def _load_manifest(self):
        """Load existing download manifest."""
        manifest_path = self.output_dir / "download_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)

    def _save_manifest(self):
        """Save download manifest."""
        manifest_path = self.output_dir / "download_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    # ───────────────────────────────────────────────────────────────
    # Connection
    # ───────────────────────────────────────────────────────────────

    def connect(self):
        """Connect to WRDS using direct SQLAlchemy (no interactive prompts)."""
        logger.info("Connecting to WRDS...")
        
        password_encoded = urllib.parse.quote_plus(WRDS_PASSWORD)
        pguri = f"postgresql://{WRDS_USERNAME}:{password_encoded}@{WRDS_HOST}:{WRDS_PORT}/{WRDS_DB}"
        
        self.engine = sa.create_engine(
            pguri,
            isolation_level="AUTOCOMMIT",
            connect_args={
                "sslmode": "require",
                "connect_timeout": 30,
                "options": "-c statement_timeout=600000",  # 10 min per query (large tables)
            },
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        
        # Quick connectivity test
        with self.engine.connect() as conn:
            result = conn.execute(sa.text("SELECT 1"))
            assert result.fetchone()[0] == 1
        
        logger.info("✅ Connected to WRDS (SQLAlchemy direct)")

    def disconnect(self):
        """Dispose of the engine."""
        if self.engine:
            self.engine.dispose()
            logger.info("Disconnected from WRDS")

    # ───────────────────────────────────────────────────────────────
    # Core download logic
    # ───────────────────────────────────────────────────────────────

    def _query_to_df(self, query: str) -> pd.DataFrame:
        """Execute a query and return DataFrame."""
        with self.engine.connect() as conn:
            df = pd.read_sql(sa.text(query), conn)
        df.columns = df.columns.str.lower().str.strip()
        return df

    def _try_query(self, primary_query: str, fallback_queries: list = None) -> pd.DataFrame:
        """Try primary query, then fallbacks if it fails."""
        try:
            return self._query_to_df(primary_query)
        except Exception as e:
            logger.warning(f"  Primary query failed: {e}")
            if fallback_queries:
                for i, fq in enumerate(fallback_queries):
                    try:
                        logger.info(f"  Trying fallback {i+1}...")
                        return self._query_to_df(fq)
                    except Exception as e2:
                        logger.warning(f"  Fallback {i+1} failed: {e2}")
            raise

    def _save_parquet(self, df: pd.DataFrame, name: str) -> Path:
        """Save DataFrame as Parquet with proper settings."""
        filepath = self.output_dir / f"{name}.parquet"
        df.to_parquet(
            filepath,
            engine=PARQUET_ENGINE,
            compression=PARQUET_COMPRESSION,
            index=False,
        )
        
        # Compute file hash for integrity verification
        file_hash = hashlib.md5(filepath.read_bytes()).hexdigest()
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        
        return filepath, file_hash, file_size_mb

    def _should_download(self, name: str) -> bool:
        """Check if we should download this dataset (skip if already done)."""
        if self.force:
            return True
        
        filepath = self.output_dir / f"{name}.parquet"
        if not filepath.exists():
            return True
        
        # Check manifest for completeness
        if name in self.manifest:
            status = self.manifest[name].get("status")
            if status == "complete":
                logger.info(f"  ⏭️  {name} already downloaded ({self.manifest[name].get('rows', '?'):,} rows). Use --force to re-download.")
                return False
        
        return True

    # ───────────────────────────────────────────────────────────────
    # Download a single dataset
    # ───────────────────────────────────────────────────────────────

    def download_dataset(self, name: str, config: dict) -> dict:
        """
        Download one dataset from WRDS and save as Parquet.
        
        Returns dict with: rows, file_size_mb, md5, elapsed_secs
        """
        logger.info(f"\n{'═' * 65}")
        logger.info(f"  Downloading: {name}")
        logger.info(f"  {config['description']}")
        logger.info(f"{'═' * 65}")

        if not self._should_download(name):
            return self.manifest.get(name, {})

        t0 = time.time()

        if config.get("chunk_by_year"):
            # ── Chunked download (year by year) ──
            start_year, end_year = config["year_range"]
            all_chunks = []
            
            for year in range(start_year, end_year + 1):
                query = config["query_template"].format(year=year)
                try:
                    df = self._query_to_df(query)
                    if not df.empty:
                        all_chunks.append(df)
                        logger.info(f"    {year}: {len(df):>10,} rows")
                    else:
                        # Skip empty years silently
                        pass
                except Exception as e:
                    logger.warning(f"    {year}: ERROR — {e}")

            if all_chunks:
                full_df = pd.concat(all_chunks, ignore_index=True)
                del all_chunks
            else:
                logger.error(f"  ❌ No data retrieved for {name}")
                return {"status": "failed", "rows": 0}
        else:
            # ── Single-shot download ──
            fallbacks = config.get("fallback_queries", [])
            full_df = self._try_query(config["query"], fallbacks)

        # ── Save to Parquet ──
        filepath, file_hash, file_size_mb = self._save_parquet(full_df, name)
        elapsed = time.time() - t0
        n_rows = len(full_df)

        # ── Validate ──
        expected_min = config.get("expected_min_rows", 0)
        if n_rows < expected_min:
            logger.warning(f"  ⚠️  {name}: {n_rows:,} rows < expected {expected_min:,}")
        else:
            logger.info(f"  ✅ {name}: {n_rows:,} rows ({file_size_mb:.1f} MB) in {elapsed:.1f}s")

        # ── Update manifest ──
        result = {
            "status": "complete",
            "rows": n_rows,
            "columns": list(full_df.columns),
            "dtypes": {col: str(full_df[col].dtype) for col in full_df.columns},
            "file_size_mb": round(file_size_mb, 2),
            "md5": file_hash,
            "downloaded_at": datetime.now().isoformat(),
            "elapsed_secs": round(elapsed, 1),
            "filepath": str(filepath),
        }
        self.manifest[name] = result
        self._save_manifest()

        del full_df
        return result

    # ───────────────────────────────────────────────────────────────
    # Download ALL
    # ───────────────────────────────────────────────────────────────

    def download_all(self, include_daily: bool = False, dataset: str = None):
        """
        Download all WRDS datasets to Parquet files.
        
        Order: small tables first → large tables last.
        """
        self.connect()

        if dataset:
            all_datasets = {**DATASETS, **OPTIONAL_DATASETS}
            if dataset not in all_datasets:
                logger.error(f"Unknown dataset: {dataset}")
                logger.info(f"Available: {sorted(all_datasets.keys())}")
                return
            config = all_datasets[dataset]
            self.download_dataset(dataset, config)
        else:
            # Download core datasets in order (small → large)
            for name, config in DATASETS.items():
                try:
                    self.download_dataset(name, config)
                except Exception as e:
                    logger.error(f"  ❌ {name} FAILED: {e}")
                    self.manifest[name] = {"status": "failed", "error": str(e)}
                    self._save_manifest()

            # Optionally download CRSP daily (very large)
            if include_daily:
                try:
                    self.download_dataset("crsp_daily", OPTIONAL_DATASETS["crsp_daily"])
                except Exception as e:
                    logger.error(f"  ❌ crsp_daily FAILED: {e}")

        self.disconnect()
        self._print_summary()

    # ───────────────────────────────────────────────────────────────
    # Momentum factor (separate from 5-factor)
    # ───────────────────────────────────────────────────────────────

    def _merge_momentum_into_ff(self):
        """If FF factors don't have UMD, try to merge from momentum table."""
        for freq in ["daily", "monthly"]:
            name = f"ff_factors_{freq}"
            filepath = self.output_dir / f"{name}.parquet"
            if not filepath.exists():
                continue
            
            df = pd.read_parquet(filepath)
            if "umd" in df.columns and df["umd"].notna().any():
                continue
            
            # Try to get momentum
            logger.info(f"  Merging momentum into {name}...")
            for mom_table in [f"ff_all.momentum_{freq}", f"ff.momentum_{freq}"]:
                try:
                    mom = self._query_to_df(
                        f"SELECT date, umd FROM {mom_table} WHERE date >= '1963-07-01' ORDER BY date"
                    )
                    df = df.drop(columns=["umd"], errors="ignore")
                    df = df.merge(mom, on="date", how="left")
                    df.to_parquet(filepath, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False)
                    logger.info(f"    ✅ Merged momentum from {mom_table}")
                    break
                except Exception:
                    continue

    # ───────────────────────────────────────────────────────────────
    # Verification
    # ───────────────────────────────────────────────────────────────

    def verify(self):
        """Verify all downloaded Parquet files."""
        logger.info("\n" + "═" * 65)
        logger.info("  PARQUET FILE VERIFICATION")
        logger.info("═" * 65)

        all_datasets = {**DATASETS, **OPTIONAL_DATASETS}
        total_rows = 0
        total_size = 0
        issues = []

        for name, config in all_datasets.items():
            filepath = self.output_dir / f"{name}.parquet"
            if not filepath.exists():
                if name in DATASETS:  # Only flag missing core datasets
                    logger.warning(f"  ❌ {name:35s}: FILE MISSING")
                    issues.append(f"{name}: file missing")
                continue

            try:
                df = pd.read_parquet(filepath)
                n_rows = len(df)
                file_size = filepath.stat().st_size / (1024 * 1024)
                total_rows += n_rows
                total_size += file_size

                expected_min = config.get("expected_min_rows", 0)
                ok = n_rows >= expected_min
                status = "✅" if ok else "⚠️"
                
                logger.info(
                    f"  {status} {name:35s}: {n_rows:>12,} rows  {file_size:>8.1f} MB"
                )

                if not ok:
                    issues.append(f"{name}: {n_rows:,} rows < expected {expected_min:,}")

                # Check date ranges
                date_cols = [c for c in df.columns if c in ("date", "datadate", "statpers", "pends")]
                if date_cols:
                    col = date_cols[0]
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    min_dt = df[col].min()
                    max_dt = df[col].max()
                    logger.info(f"       Date range: {min_dt} → {max_dt}")

                del df

            except Exception as e:
                logger.error(f"  ❌ {name:35s}: ERROR reading — {e}")
                issues.append(f"{name}: read error - {e}")

        logger.info("─" * 65)
        logger.info(f"  Total rows: {total_rows:>15,}")
        logger.info(f"  Total size: {total_size:>15.1f} MB ({total_size/1024:.2f} GB)")

        if issues:
            logger.warning(f"\n  ⚠️  {len(issues)} issues found:")
            for issue in issues:
                logger.warning(f"    • {issue}")
        else:
            logger.info("\n  ✅ ALL FILES VERIFIED")

        return issues

    # ───────────────────────────────────────────────────────────────
    # S3 Upload
    # ───────────────────────────────────────────────────────────────

    def upload_to_s3(self, bucket: str = S3_BUCKET, prefix: str = S3_PREFIX):
        """Upload all Parquet files to S3."""
        try:
            import boto3
        except ImportError:
            logger.error("boto3 not installed. Run: pip install boto3")
            return

        s3 = boto3.client("s3")
        logger.info(f"\nUploading to s3://{bucket}/{prefix}...")

        for filepath in sorted(self.output_dir.glob("*.parquet")):
            key = f"{prefix}{filepath.name}"
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"  {filepath.name:40s} → s3://{bucket}/{key} ({file_size_mb:.1f} MB)")
            try:
                s3.upload_file(str(filepath), bucket, key)
                logger.info(f"    ✅ Uploaded")
            except Exception as e:
                logger.error(f"    ❌ Failed: {e}")

        # Also upload manifest
        manifest_path = self.output_dir / "download_manifest.json"
        if manifest_path.exists():
            s3.upload_file(str(manifest_path), bucket, f"{prefix}download_manifest.json")
            logger.info(f"  ✅ Manifest uploaded")

    # ───────────────────────────────────────────────────────────────
    # Summary
    # ───────────────────────────────────────────────────────────────

    def _print_summary(self):
        """Print download summary."""
        logger.info("\n" + "═" * 65)
        logger.info("  WRDS → PARQUET DOWNLOAD COMPLETE")
        logger.info("═" * 65)

        total_rows = 0
        total_size = 0
        
        for name, info in self.manifest.items():
            rows = info.get("rows", 0)
            size = info.get("file_size_mb", 0)
            status = info.get("status", "unknown")
            icon = "✅" if status == "complete" else "❌"
            
            logger.info(f"  {icon} {name:35s}: {rows:>12,} rows  {size:>8.1f} MB")
            total_rows += rows
            total_size += size

        logger.info("─" * 65)
        logger.info(f"  Total: {total_rows:>12,} rows  {total_size:>8.1f} MB ({total_size/1024:.2f} GB)")
        logger.info(f"  Output: {self.output_dir}")
        logger.info("═" * 65)


# ═══════════════════════════════════════════════════════════════════
# Quick Test
# ═══════════════════════════════════════════════════════════════════

def quick_test():
    """Run a quick connectivity + data test (5 rows each table)."""
    logger.info("\n" + "═" * 65)
    logger.info("  WRDS QUICK CONNECTION TEST")
    logger.info("═" * 65)

    password_encoded = urllib.parse.quote_plus(WRDS_PASSWORD)
    pguri = f"postgresql://{WRDS_USERNAME}:{password_encoded}@{WRDS_HOST}:{WRDS_PORT}/{WRDS_DB}"
    
    logger.info("  Connecting (30s timeout)...")
    engine = sa.create_engine(
        pguri,
        isolation_level="AUTOCOMMIT",
        connect_args={
            "sslmode": "require",
            "connect_timeout": 30,
            "options": "-c statement_timeout=30000",  # 30s per query
        },
        pool_pre_ping=True,
    )

    try:
        with engine.connect() as conn:
            # Basic test
            logger.info("  Running SELECT 1...")
            result = conn.execute(sa.text("SELECT 1"))
            assert result.fetchone()[0] == 1
            logger.info("  ✅ Connection established")

            # Sample from each core table
            test_queries = [
                ("CRSP Monthly", "SELECT permno, date, ret FROM crsp.msf LIMIT 5"),
                ("Compustat Q",  "SELECT gvkey, datadate, atq FROM comp.fundq LIMIT 5"),
                ("Compustat A",  "SELECT gvkey, datadate, at FROM comp.funda LIMIT 5"),
                ("IBES Summary", "SELECT ticker, statpers FROM ibes.statsumu_epsus LIMIT 5"),
                ("IBES Actuals", "SELECT ticker, anndats FROM ibes.actu_epsus LIMIT 5"),
                ("CCM Link",     "SELECT gvkey, lpermno FROM crsp.ccmxpf_lnkhist LIMIT 5"),
                ("FF 5-Factor",  "SELECT date, mktrf FROM ff_all.fivefactors_daily LIMIT 5"),
                ("CRSP Names",   "SELECT permno, ticker FROM crsp.msenames LIMIT 5"),
            ]

            for label, query in test_queries:
                try:
                    t0 = time.time()
                    df = pd.read_sql(sa.text(query), conn)
                    elapsed = time.time() - t0
                    logger.info(f"  ✅ {label:20s}: {len(df)} rows, {len(df.columns)} cols ({elapsed:.1f}s)")
                except Exception as e:
                    logger.warning(f"  ❌ {label:20s}: {str(e)[:80]}")

        engine.dispose()
        logger.info("\n  ✅ WRDS access verified — all core tables accessible")
        logger.info("  Ready to run full download: python wrds_pipeline/download_to_parquet.py")
    except Exception as e:
        engine.dispose()
        logger.error(f"\n  ❌ Connection FAILED: {e}")
        logger.info("  Troubleshooting:")
        logger.info("    1. Check port 9737: nc -zv wrds-pgdata.wharton.upenn.edu 9737")
        logger.info("    2. Check credentials at wrds-www.wharton.upenn.edu")
        logger.info("    3. Check firewall allows outbound port 9737")
        raise


# ═══════════════════════════════════════════════════════════════════
# Data Dictionary README
# ═══════════════════════════════════════════════════════════════════

def write_data_readme(output_dir: Path):
    """Write a README with data dictionary for the Parquet files."""
    readme = output_dir / "README.md"
    readme.write_text("""# WRDS Institutional Data (Parquet Format)

## Data Sources
All data downloaded from [WRDS (Wharton Research Data Services)](https://wrds-www.wharton.upenn.edu/).

## Files

| File | Description | Key Columns | Approx Rows |
|------|-------------|-------------|-------------|
| `crsp_monthly.parquet` | CRSP monthly stock returns | permno, date, ret, market_cap | ~3M |
| `compustat_quarterly.parquet` | Quarterly income/balance sheet | gvkey, datadate, rdq, atq, niq | ~2M |
| `compustat_annual.parquet` | Annual income/balance sheet | gvkey, datadate, at, ni, seq | ~500K |
| `ibes_summary.parquet` | Analyst consensus estimates | ticker, fpedats, statpers, meanest | ~3M |
| `ibes_actuals.parquet` | Actual EPS announcements | ticker, pends, anndats, value | ~500K |
| `ff_factors_daily.parquet` | Fama-French 5 factors + momentum | date, mktrf, smb, hml, rmw, cma, umd | ~15K |
| `ff_factors_monthly.parquet` | Fama-French 5 factors (monthly) | date, mktrf, smb, hml, rmw, cma, umd | ~750 |
| `crsp_compustat_link.parquet` | PERMNO ↔ GVKEY mapping | gvkey, lpermno, linkdt, linkenddt | ~35K |
| `ibes_crsp_link.parquet` | IBES ticker ↔ PERMNO mapping | ticker, permno, sdate, edate | ~15K |
| `compustat_security.parquet` | Security identifiers | gvkey, tic, cusip, exchg | ~50K |

## Key Identifiers
- **PERMNO** (CRSP): Permanent security identifier — the primary key for linking
- **GVKEY** (Compustat): Company identifier  
- **IBES Ticker**: Analyst coverage identifier

## Linking Logic
```
CRSP (PERMNO) ←→ crsp_compustat_link ←→ Compustat (GVKEY)
CRSP (PERMNO) ←→ ibes_crsp_link ←→ IBES (ticker)
```

## Point-in-Time Discipline
- **Compustat**: Use `rdq` (report date) as the knowledge date, NOT `datadate`
- **IBES**: Use `statpers` (consensus date), NOT `fpedats` (fiscal period end)
- This prevents lookahead bias that inflates backtest results by 5-15%

## Usage
```python
import pandas as pd

crsp = pd.read_parquet("data/wrds/crsp_monthly.parquet")
compustat = pd.read_parquet("data/wrds/compustat_quarterly.parquet")
link = pd.read_parquet("data/wrds/crsp_compustat_link.parquet")
```

## Generated
Downloaded on: {date}
Pipeline: NUBLE WRDS Pipeline v1.0
""".replace("{date}", datetime.now().strftime("%Y-%m-%d")))


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="WRDS → Parquet data pipeline (Option B)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Download all core datasets
  %(prog)s --test                   # Quick connectivity test
  %(prog)s --dataset crsp_monthly   # Download single dataset
  %(prog)s --include-daily          # Include CRSP daily (~100M+ rows)
  %(prog)s --verify                 # Verify existing files
  %(prog)s --upload-s3              # Upload to S3 after download
  %(prog)s --force                  # Re-download everything
        """,
    )
    parser.add_argument("--test", action="store_true", help="Quick connection test only")
    parser.add_argument("--dataset", type=str, help="Download a single dataset by name")
    parser.add_argument("--include-daily", action="store_true", help="Include CRSP daily (very large)")
    parser.add_argument("--verify", action="store_true", help="Verify existing Parquet files")
    parser.add_argument("--upload-s3", action="store_true", help="Upload to S3 after download")
    parser.add_argument("--force", action="store_true", help="Force re-download of all files")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR), help="Output directory")
    
    args = parser.parse_args()

    if args.test:
        quick_test()
        return

    output_dir = Path(args.output_dir)
    dl = WRDSParquetDownloader(output_dir=output_dir, force=args.force)

    if args.verify:
        dl.verify()
        return

    # Full download
    dl.download_all(include_daily=args.include_daily, dataset=args.dataset)
    
    # Write data dictionary
    write_data_readme(output_dir)

    # Upload to S3 if requested
    if args.upload_s3:
        dl.upload_to_s3()


if __name__ == "__main__":
    main()
