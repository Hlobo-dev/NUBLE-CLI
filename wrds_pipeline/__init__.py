"""
WRDS Institutional Data Pipeline
=================================
Downloads research-quality datasets from WRDS (Wharton Research Data Services).
Implements the Gu-Kelly-Xiu (2020) framework for ML in finance.

Architecture: Option B — WRDS Python API → Parquet files (local + S3)
  No RDS dependency. Parquet = columnar, compressed, fast, portable.

Modules:
    config                — Credentials and connection settings
    download_to_parquet   — Main pipeline: WRDS → Parquet files (Option B) ★
    parquet_access        — Data access layer reading from Parquet ★
    download_all          — Legacy: WRDS → RDS pipeline (Option A)
    validate              — Data validation and quality checks
    data_access           — Legacy: RDS query interface
    characteristics       — 94 Gu-Kelly-Xiu stock characteristics

Quick Start:
    python wrds_pipeline/download_to_parquet.py --test     # Verify WRDS access
    python wrds_pipeline/download_to_parquet.py            # Download all → data/wrds/
    python wrds_pipeline/parquet_access.py --summary       # Check downloaded data
"""

__version__ = "2.0.0"
