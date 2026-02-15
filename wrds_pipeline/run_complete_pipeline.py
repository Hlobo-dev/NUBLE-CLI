#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════
WRDS COMPLETE PIPELINE — SINGLE SCRIPT
══════════════════════════════════════════════════════════════════════

This script does EVERYTHING from scratch:
    1. Tests connectivity (WRDS + RDS)
    2. Creates all 9 tables in RDS
    3. Downloads ALL datasets from WRDS
    4. Validates all data
    5. Runs end-to-end tests

RUN:
    python wrds_pipeline/run_complete_pipeline.py

If WRDS port 9737 is blocked from your network, run from EC2 or
WRDS cloud (ssh wrds-cloud.wharton.upenn.edu).

This script is IDEMPOTENT — safe to re-run at any point.
══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import logging
import pathlib
import signal

# ═══════════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("wrds_pipeline_run.log", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("wrds_pipeline_main")

# Add project root to path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def timeout_handler(signum, frame):
    raise TimeoutError("Connection timed out")


def test_wrds_connectivity(timeout_secs=30):
    """Test that we can reach WRDS PostgreSQL on port 9737."""
    import socket

    logger.info("Testing WRDS connectivity (wrds-pgdata.wharton.upenn.edu:9737)...")
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_secs)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_secs)
        result = sock.connect_ex(("wrds-pgdata.wharton.upenn.edu", 9737))
        sock.close()

        signal.alarm(0)

        if result == 0:
            logger.info("  ✅ WRDS port 9737 is REACHABLE")
            return True
        else:
            logger.error(f"  ❌ WRDS port 9737 is BLOCKED (error code: {result})")
            logger.error("     Your network/firewall blocks port 9737.")
            logger.error("     Solutions:")
            logger.error("       1. Run from EC2 instance (same VPC as RDS)")
            logger.error("       2. Run from university network / VPN")
            logger.error("       3. SSH to wrds-cloud.wharton.upenn.edu")
            return False
    except (TimeoutError, Exception) as e:
        signal.alarm(0)
        logger.error(f"  ❌ WRDS connectivity test failed: {e}")
        return False


def test_rds_connectivity(timeout_secs=15):
    """Test that we can reach AWS RDS PostgreSQL."""
    import socket

    host = "trading-data-db.ca90y4g2mxtw.us-east-1.rds.amazonaws.com"
    logger.info(f"Testing RDS connectivity ({host}:5432)...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_secs)
        result = sock.connect_ex((host, 5432))
        sock.close()
        if result == 0:
            logger.info("  ✅ RDS port 5432 is REACHABLE")
            return True
        else:
            logger.error(f"  ❌ RDS port 5432 is NOT reachable (error {result})")
            logger.error("     Check RDS security group allows your IP")
            return False
    except Exception as e:
        logger.error(f"  ❌ RDS connectivity failed: {e}")
        logger.error("     DNS resolution failed — you may not be in the VPC")
        return False


def setup_pgpass():
    """Set up .pgpass for WRDS auto-authentication."""
    from wrds_pipeline.config import WRDS_USERNAME, WRDS_PASSWORD, WRDS_PGHOST, WRDS_PGPORT, WRDS_PGDATABASE

    pgpass_path = pathlib.Path.home() / ".pgpass"
    wrds_line = f"{WRDS_PGHOST}:{WRDS_PGPORT}:{WRDS_PGDATABASE}:{WRDS_USERNAME}:{WRDS_PASSWORD}"

    existing = pgpass_path.read_text() if pgpass_path.exists() else ""
    if wrds_line not in existing:
        with open(pgpass_path, "a") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write(wrds_line + "\n")
        pgpass_path.chmod(0o600)
        logger.info("  ✅ .pgpass updated with WRDS credentials")
    else:
        logger.info("  ✅ .pgpass already has WRDS credentials")

    # Also set env vars
    os.environ["PGHOST"] = WRDS_PGHOST
    os.environ["PGPORT"] = WRDS_PGPORT
    os.environ["PGDATABASE"] = WRDS_PGDATABASE
    os.environ["PGUSER"] = WRDS_USERNAME
    os.environ["PGPASSWORD"] = WRDS_PASSWORD


def run_full_pipeline():
    """Execute the complete WRDS pipeline from scratch."""

    logger.info("")
    logger.info("═" * 70)
    logger.info("  WRDS INSTITUTIONAL DATA PIPELINE — FULL EXECUTION")
    logger.info("═" * 70)
    logger.info(f"  Time:    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Project: {PROJECT_ROOT}")
    logger.info("")

    # ─────────────────────────────────────────────────────────────
    # PHASE 0: Connectivity Tests
    # ─────────────────────────────────────────────────────────────
    logger.info("━" * 70)
    logger.info("  PHASE 0: CONNECTIVITY & ENVIRONMENT")
    logger.info("━" * 70)

    setup_pgpass()

    wrds_ok = test_wrds_connectivity()
    rds_ok = test_rds_connectivity()

    if not wrds_ok:
        logger.error("")
        logger.error("  ╔══════════════════════════════════════════════════════════╗")
        logger.error("  ║  WRDS PORT 9737 IS BLOCKED FROM THIS NETWORK           ║")
        logger.error("  ║                                                         ║")
        logger.error("  ║  Copy this entire wrds_pipeline/ folder to your EC2     ║")
        logger.error("  ║  instance and run from there. EC2 can reach both WRDS   ║")
        logger.error("  ║  and RDS.                                               ║")
        logger.error("  ║                                                         ║")
        logger.error("  ║  Or SSH to wrds-cloud.wharton.upenn.edu:                ║")
        logger.error("  ║    ssh hlobo@wrds-cloud.wharton.upenn.edu               ║")
        logger.error("  ╚══════════════════════════════════════════════════════════╝")
        logger.error("")
        return False

    if not rds_ok:
        logger.error("")
        logger.error("  RDS is not reachable. Check:")
        logger.error("    1. Your IP is in the RDS security group")
        logger.error("    2. The RDS instance is running")
        logger.error("")
        return False

    # Test actual connections
    logger.info("")
    logger.info("  Testing WRDS Python connection...")
    try:
        import wrds
        db = wrds.Connection(wrds_username="hlobo")
        libs = db.list_libraries()
        relevant = [l for l in libs if any(k in l.lower() for k in ["crsp", "comp", "ibes", "ff"])]
        logger.info(f"  ✅ WRDS connected — {len(libs)} libraries, {len(relevant)} relevant")
        for lib in sorted(relevant)[:15]:
            logger.info(f"     ✓ {lib}")
        db.close()
    except Exception as e:
        logger.error(f"  ❌ WRDS Python connection failed: {e}")
        return False

    logger.info("")
    logger.info("  Testing RDS PostgreSQL connection...")
    try:
        import psycopg2
        from wrds_pipeline.config import RDS_CONFIG
        conn = psycopg2.connect(**RDS_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM stock_prices")
        sp_count = cur.fetchone()[0]
        logger.info(f"  ✅ RDS connected — stock_prices: {sp_count:,} rows")
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"  ❌ RDS connection failed: {e}")
        return False

    # ─────────────────────────────────────────────────────────────
    # PHASE 1: Create Tables
    # ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("━" * 70)
    logger.info("  PHASE 1: CREATE TABLES")
    logger.info("━" * 70)

    from wrds_pipeline.download_all import WRDSDownloader
    dl = WRDSDownloader()
    dl.connect_rds()
    tables = dl.create_tables()

    # ─────────────────────────────────────────────────────────────
    # PHASE 2: Download ALL Data
    # ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("━" * 70)
    logger.info("  PHASE 2: DOWNLOAD ALL DATASETS FROM WRDS")
    logger.info("━" * 70)
    logger.info("  This will take 1-4 hours depending on network speed.")
    logger.info("  The script is RESUMABLE — safe to interrupt and re-run.")
    logger.info("")

    t0 = time.time()
    results = dl.download_all()
    elapsed = time.time() - t0

    logger.info(f"")
    logger.info(f"  Download complete in {elapsed/3600:.1f} hours")

    # ─────────────────────────────────────────────────────────────
    # PHASE 3: Validate Data
    # ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("━" * 70)
    logger.info("  PHASE 3: DATA VALIDATION")
    logger.info("━" * 70)

    from wrds_pipeline.validate import WRDSValidator
    v = WRDSValidator()
    counts = v.check_row_counts()
    v.check_date_ranges()
    v.check_null_rates(verbose=True)
    v.check_identifiers()
    v.check_join_coverage()
    v.check_storage()

    # ─────────────────────────────────────────────────────────────
    # PHASE 4: End-to-End Test
    # ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("━" * 70)
    logger.info("  PHASE 4: END-TO-END TESTS")
    logger.info("━" * 70)

    from wrds_pipeline.data_access import WRDSDataAccess
    da = WRDSDataAccess()
    da.summary()

    # Test characteristics for AAPL (PERMNO=14593)
    logger.info("")
    logger.info("  Testing characteristics for AAPL (PERMNO=14593)...")
    try:
        chars = da.get_characteristics(permno=14593, as_of_date="2024-06-30")
        non_null = {k: v for k, v in chars.items() if v is not None}
        logger.info(f"    Features computed: {len(non_null)}/{len(chars)}")
        for k, v in sorted(non_null.items()):
            logger.info(f"      {k:35s}: {v:>12.4f}")
    except Exception as e:
        logger.error(f"    Characteristics test failed: {e}")

    # Test earnings surprise
    logger.info("")
    logger.info("  Testing earnings surprise for AAPL...")
    try:
        sue = da.get_earnings_surprise(permno=14593, as_of_date="2024-06-30")
        for k, v in sue.items():
            logger.info(f"    {k}: {v}")
    except Exception as e:
        logger.error(f"    Earnings surprise test failed: {e}")

    da.close()
    v.close()

    # ─────────────────────────────────────────────────────────────
    # FINAL REPORT
    # ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("═" * 70)
    logger.info("  ██  WRDS PIPELINE COMPLETE  ██")
    logger.info("═" * 70)
    logger.info(f"  Elapsed:  {elapsed/3600:.1f} hours")
    logger.info(f"  Issues:   {len(v.issues)}")
    if v.issues:
        for issue in v.issues:
            logger.warning(f"    ⚠ {issue}")
    else:
        logger.info("    ✅ No issues found!")

    logger.info("")
    logger.info("  NEXT STEPS:")
    logger.info("    1. Build training panel:")
    logger.info("       python -m wrds_pipeline.data_access --panel --start 2000-01-01 --end 2024-12-31")
    logger.info("    2. Train Gu-Kelly-Xiu model on WRDS data")
    logger.info("    3. Run walk-forward backtest")
    logger.info("")
    logger.info("═" * 70)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WRDS Complete Pipeline")
    parser.add_argument("--test-only", action="store_true", help="Test connectivity only, don't download")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, run validation only")
    args = parser.parse_args()

    if args.test_only:
        setup_pgpass()
        wrds_ok = test_wrds_connectivity()
        rds_ok = test_rds_connectivity()
        if wrds_ok and rds_ok:
            logger.info("\n  ✅ Both WRDS and RDS are reachable. Ready to run!")
        else:
            logger.error("\n  ❌ Connectivity issues — see above.")
        sys.exit(0 if (wrds_ok and rds_ok) else 1)

    if args.skip_download:
        from wrds_pipeline.validate import WRDSValidator
        v = WRDSValidator()
        v.check_row_counts()
        v.check_date_ranges()
        v.check_null_rates(verbose=True)
        v.check_identifiers()
        v.check_join_coverage()
        v.check_storage()
        v.close()
        sys.exit(0)

    success = run_full_pipeline()
    sys.exit(0 if success else 1)
