#!/usr/bin/env python3
"""
WRDS Discovery Script
======================
Discovers ALL available libraries, tables, and schemas on WRDS.
This MUST run BEFORE building download queries — WRDS changes schema over time.
"""

import os
import sys
import time
import urllib.parse
import sqlalchemy as sa
import pandas as pd

WRDS_USERNAME = os.environ.get("WRDS_USERNAME", "hlobo")
WRDS_PASSWORD = os.environ.get("WRDS_PASSWORD", "")
WRDS_HOST = os.environ.get("WRDS_PGHOST", "wrds-pgdata.wharton.upenn.edu")
WRDS_PORT = int(os.environ.get("WRDS_PGPORT", "9737"))
WRDS_DB = os.environ.get("WRDS_PGDATABASE", "wrds")


def connect():
    pw = urllib.parse.quote_plus(WRDS_PASSWORD)
    uri = f"postgresql://{WRDS_USERNAME}:{pw}@{WRDS_HOST}:{WRDS_PORT}/{WRDS_DB}"
    engine = sa.create_engine(uri, isolation_level="AUTOCOMMIT", connect_args={
        "sslmode": "require", "connect_timeout": 30,
        "options": "-c statement_timeout=60000",
    })
    with engine.connect() as c:
        c.execute(sa.text("SELECT 1"))
    print("✅ Connected to WRDS\n")
    return engine


def list_tables(engine, library):
    """List tables in a WRDS library (schema)."""
    try:
        with engine.connect() as c:
            q = sa.text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = :lib ORDER BY table_name
            """)
            df = pd.read_sql(q, c, params={"lib": library})
            return df["table_name"].tolist()
    except Exception as e:
        return f"ERROR: {e}"


def describe_table(engine, library, table):
    """Get column names and types for a table."""
    try:
        with engine.connect() as c:
            q = sa.text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = :lib AND table_name = :tbl
                ORDER BY ordinal_position
            """)
            df = pd.read_sql(q, c, params={"lib": library, "tbl": table})
            return df
    except Exception as e:
        return f"ERROR: {e}"


def count_rows(engine, library, table, limit_check=True):
    """Quick row count estimate."""
    try:
        with engine.connect() as c:
            # Use reltuples for fast estimate
            q = sa.text(f"SELECT reltuples::bigint FROM pg_class WHERE relname = :tbl")
            result = c.execute(q, {"tbl": table}).fetchone()
            if result and result[0] > 0:
                return int(result[0])
            # Fallback: actual count (can be slow)
            q2 = sa.text(f"SELECT COUNT(*) FROM {library}.{table}")
            result2 = c.execute(q2).fetchone()
            return result2[0]
    except Exception as e:
        return f"ERROR: {e}"


def main():
    engine = connect()
    
    # ═══════════════════════════════════════════════════════════
    # 1. Check ALL important libraries
    # ═══════════════════════════════════════════════════════════
    LIBRARIES = [
        'comp', 'crsp', 'ibes', 'ff', 'ff_all',
        'wrdsapps_finratio_ibes', 'wrdsapps_finratio', 'wrdsapps',
        'wrdsapps_beta',
        'tfn', 'tr_insiders', 'tr_13f',
        'ciq', 'ciq_keydev', 'ciq_ratings', 'ciq_transcripts',
        'comp_execucomp', 'execcomp',
        'optionm', 'mfl',
    ]
    
    print("=" * 70)
    print("  WRDS LIBRARY DISCOVERY")
    print("=" * 70)
    
    available_libs = {}
    for lib in LIBRARIES:
        tables = list_tables(engine, lib)
        if isinstance(tables, list) and len(tables) > 0:
            available_libs[lib] = tables
            print(f"\n✅ {lib} ({len(tables)} tables):")
            # Show first 20 tables
            for t in tables[:20]:
                print(f"   • {t}")
            if len(tables) > 20:
                print(f"   ... and {len(tables) - 20} more")
        else:
            print(f"❌ {lib}: {tables if isinstance(tables, str) else 'EMPTY / NO ACCESS'}")
    
    # ═══════════════════════════════════════════════════════════
    # 2. Describe KEY tables we need
    # ═══════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("  KEY TABLE SCHEMAS")
    print("=" * 70)
    
    KEY_TABLES = [
        # CRSP — check old vs new (CIZ)
        ('crsp', 'msf'),
        ('crsp', 'msf_v2'),
        ('crsp', 'dsf'),
        ('crsp', 'dsf_v2'),
        ('crsp', 'msenames'),
        ('crsp', 'dsenames'),
        ('crsp', 'ccmxpf_lnkhist'),
        ('crsp', 'msedist'),
        ('crsp', 'msedelist'),
        ('crsp', 'msi'),
        ('crsp', 'dsi'),
        ('crsp', 'mcti'),
        ('crsp', 'dcti'),
        ('crsp', 'msp500list'),
        ('crsp', 'dsp500list'),
        # Compustat
        ('comp', 'fundq'),
        ('comp', 'funda'),
        ('comp', 'security'),
        ('comp', 'shortint'),
        ('comp', 'sec_shortint'),
        ('comp', 'idxcst_his'),
        ('comp', 'adsprate'),
        ('comp', 'fundq_pit'),
        # IBES
        ('ibes', 'statsumu_epsus'),
        ('ibes', 'actu_epsus'),
        ('ibes', 'recddet'),
        ('ibes', 'idsum'),
        # Fama-French
        ('ff_all', 'fivefactors_daily'),
        ('ff_all', 'fivefactors_monthly'),
        ('ff', 'factors_daily'),
        ('ff', 'factors_monthly'),
        # WRDS Pre-computed ★★★
        ('wrdsapps_finratio_ibes', 'firm_ratio_ibes'),
        ('wrdsapps_finratio', 'firm_ratio'),
        # Links
        ('wrdsapps', 'ibcrsphist'),
        # Thomson/LSEG
        ('tfn', 's34type1'),
        ('tfn', 'table1'),
        # Execucomp
        ('comp_execucomp', 'anncomp'),
    ]
    
    for lib, tbl in KEY_TABLES:
        desc = describe_table(engine, lib, tbl)
        if isinstance(desc, pd.DataFrame) and not desc.empty:
            est_rows = count_rows(engine, lib, tbl)
            print(f"\n{'─' * 60}")
            print(f"  {lib}.{tbl}  (~{est_rows:,} rows est.)")
            print(f"{'─' * 60}")
            for _, row in desc.iterrows():
                print(f"    {row['column_name']:30s}  {row['data_type']:20s}  {'NULL' if row['is_nullable'] == 'YES' else 'NOT NULL'}")
        else:
            print(f"\n  ❌ {lib}.{tbl}: NOT FOUND")
    
    # ═══════════════════════════════════════════════════════════
    # 3. Sample data from critical tables
    # ═══════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("  SAMPLE DATA FROM CRITICAL TABLES")
    print("=" * 70)
    
    SAMPLE_QUERIES = [
        ("WRDS Financial Ratios (GOLD MINE)", 
         "SELECT * FROM wrdsapps_finratio_ibes.firm_ratio_ibes LIMIT 3"),
        ("WRDS Financial Ratios (no IBES)",
         "SELECT * FROM wrdsapps_finratio.firm_ratio LIMIT 3"),
        ("CRSP msf (old format)",
         "SELECT permno, date, ret, prc, shrout, vol FROM crsp.msf LIMIT 3"),
        ("CRSP dsf (old format)",
         "SELECT permno, date, ret, prc, vol FROM crsp.dsf LIMIT 3"),
        ("CRSP distributions",
         "SELECT * FROM crsp.msedist LIMIT 3"),
        ("CRSP delisting",
         "SELECT * FROM crsp.msedelist LIMIT 3"),
        ("CRSP index monthly",
         "SELECT * FROM crsp.msi LIMIT 3"),
        ("CRSP treasury monthly",
         "SELECT * FROM crsp.mcti LIMIT 3"),
        ("S&P 500 constituents (Compustat)",
         "SELECT * FROM comp.idxcst_his WHERE gvkeyx = '000003' LIMIT 3"),
        ("S&P 500 (CRSP)",
         "SELECT * FROM crsp.msp500list LIMIT 3"),
        ("IBES recommendations",
         "SELECT * FROM ibes.recddet LIMIT 3"),
        ("Short interest",
         "SELECT * FROM comp.shortint LIMIT 3"),
        ("Thomson 13F",
         "SELECT * FROM tfn.s34type1 LIMIT 3"),
        ("Thomson Insider",
         "SELECT * FROM tfn.table1 LIMIT 3"),
    ]
    
    for label, query in SAMPLE_QUERIES:
        print(f"\n{'─' * 50}")
        print(f"  {label}")
        try:
            with engine.connect() as c:
                df = pd.read_sql(sa.text(query), c)
                print(f"  Columns: {list(df.columns)}")
                if not df.empty:
                    print(df.to_string(index=False))
                else:
                    print("  (empty)")
        except Exception as e:
            err = str(e)[:200]
            print(f"  ❌ {err}")
    
    engine.dispose()
    print("\n\n✅ DISCOVERY COMPLETE")


if __name__ == "__main__":
    main()
