#!/usr/bin/env python3
"""
WRDS Data Validation
=====================
Validates all WRDS data loaded into RDS.
Checks row counts, date ranges, null rates, and join coverage.

Usage:
    python -m wrds_pipeline.validate
    python -m wrds_pipeline.validate --verbose
"""

import sys
import logging
import argparse
from datetime import date

import psycopg2
import pandas as pd

from wrds_pipeline.config import RDS_CONFIG, EXPECTED_MINIMUMS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("wrds_validate")


class WRDSValidator:
    """Validates all WRDS data in the RDS warehouse."""

    def __init__(self):
        self.conn = psycopg2.connect(**RDS_CONFIG)
        self.conn.autocommit = True
        self.issues = []

    def _q(self, sql: str):
        """Execute a query and return all rows."""
        with self.conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()

    def _q1(self, sql: str):
        """Execute a query and return the first value."""
        rows = self._q(sql)
        return rows[0][0] if rows else None

    def close(self):
        self.conn.close()

    # ───────────────────────────────────────────────────────────────
    # 1. Row counts
    # ───────────────────────────────────────────────────────────────

    def check_row_counts(self) -> dict:
        """Check row counts against expected minimums."""
        logger.info("")
        logger.info("═" * 65)
        logger.info("  1. ROW COUNTS")
        logger.info("═" * 65)

        counts = {}
        for table, minimum in EXPECTED_MINIMUMS.items():
            try:
                count = self._q1(f"SELECT COUNT(*) FROM {table}")
                counts[table] = count
                ok = count >= minimum
                status = "✓" if ok else "✗"
                logger.info(
                    f"  {status} {table:35s}: {count:>12,}  (min: {minimum:>12,})"
                )
                if not ok:
                    self.issues.append(
                        f"{table}: {count:,} rows < expected {minimum:,}"
                    )
            except Exception as e:
                counts[table] = 0
                logger.error(f"  ✗ {table:35s}: ERROR — {e}")
                self.issues.append(f"{table}: table missing or error")

        return counts

    # ───────────────────────────────────────────────────────────────
    # 2. Date ranges
    # ───────────────────────────────────────────────────────────────

    def check_date_ranges(self):
        """Check that each table covers the expected date range."""
        logger.info("")
        logger.info("═" * 65)
        logger.info("  2. DATE RANGES")
        logger.info("═" * 65)

        checks = [
            ("stock_prices",        "date",     "1990-01-01", "2024-06-01"),
            ("compustat_quarterly",  "datadate", "1975-01-01", "2024-01-01"),
            ("compustat_annual",     "datadate", "1975-01-01", "2024-01-01"),
            ("ibes_summary",        "statpers", "1985-01-01", "2024-01-01"),
            ("ibes_actuals",        "pends",    "1985-01-01", "2024-01-01"),
            ("ff_factors_daily",    "date",     "1975-01-01", "2024-01-01"),
            ("ff_factors_monthly",  "date",     "1975-01-01", "2024-01-01"),
            ("crsp_monthly",        "date",     "1975-01-01", "2024-01-01"),
        ]

        for table, col, exp_min, exp_max in checks:
            try:
                row = self._q(
                    f"SELECT MIN({col}), MAX({col}) FROM {table}"
                )[0]
                min_dt, max_dt = row
                ok_min = str(min_dt) <= exp_min if min_dt else False
                ok_max = str(max_dt) >= exp_max if max_dt else False
                status = "✓" if (ok_min and ok_max) else "⚠"
                logger.info(
                    f"  {status} {table:30s}: {min_dt} → {max_dt}"
                )
                if not ok_min:
                    self.issues.append(f"{table}: min date {min_dt} > expected {exp_min}")
                if not ok_max:
                    self.issues.append(f"{table}: max date {max_dt} < expected {exp_max}")
            except Exception as e:
                logger.error(f"  ✗ {table:30s}: {e}")

    # ───────────────────────────────────────────────────────────────
    # 3. Null rates for critical columns
    # ───────────────────────────────────────────────────────────────

    def check_null_rates(self, verbose: bool = False):
        """Check null rates for critical columns."""
        logger.info("")
        logger.info("═" * 65)
        logger.info("  3. NULL RATES (critical columns)")
        logger.info("═" * 65)

        # (table, column, max_acceptable_null_pct)
        critical = [
            ("compustat_quarterly", "gvkey",    0.0),
            ("compustat_quarterly", "datadate", 0.0),
            ("compustat_quarterly", "atq",     30.0),
            ("compustat_quarterly", "niq",     30.0),
            ("compustat_quarterly", "revtq",   30.0),
            ("compustat_quarterly", "rdq",     20.0),
            ("compustat_annual",    "at",      20.0),
            ("compustat_annual",    "ni",      20.0),
            ("compustat_annual",    "seq",     30.0),
            ("ibes_summary",       "meanest", 10.0),
            ("ibes_actuals",       "value",   10.0),
            ("ibes_actuals",       "anndats", 15.0),
            ("crsp_monthly",       "ret",     10.0),
            ("crsp_monthly",       "market_cap", 15.0),
            ("ff_factors_daily",   "mktrf",    1.0),
            ("stock_prices",       "ret",     10.0),
            ("stock_prices",       "close",    5.0),
        ]

        for table, col, max_null in critical:
            try:
                total = self._q1(f"SELECT COUNT(*) FROM {table}")
                nulls = self._q1(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL")
                pct = 100.0 * nulls / total if total > 0 else 0
                ok = pct <= max_null
                status = "✓" if ok else "⚠"
                if verbose or not ok:
                    logger.info(
                        f"  {status} {table}.{col:20s}: {pct:5.1f}% null ({nulls:,}/{total:,})"
                    )
                if not ok:
                    self.issues.append(
                        f"{table}.{col}: {pct:.1f}% null > max {max_null}%"
                    )
            except Exception as e:
                if verbose:
                    logger.error(f"  ✗ {table}.{col}: {e}")

        if not verbose:
            logger.info("  (use --verbose to see all columns)")

    # ───────────────────────────────────────────────────────────────
    # 4. Unique identifier counts
    # ───────────────────────────────────────────────────────────────

    def check_identifiers(self):
        """Count unique identifiers in each table."""
        logger.info("")
        logger.info("═" * 65)
        logger.info("  4. UNIQUE IDENTIFIERS")
        logger.info("═" * 65)

        checks = [
            ("stock_prices",        "permno", "CRSP PERMNOs (daily)"),
            ("stock_prices",        "ticker", "Tickers (daily)"),
            ("crsp_monthly",        "permno", "CRSP PERMNOs (monthly)"),
            ("compustat_quarterly", "gvkey",  "Compustat GVKEYs (quarterly)"),
            ("compustat_annual",    "gvkey",  "Compustat GVKEYs (annual)"),
            ("ibes_summary",       "ticker", "IBES tickers (summary)"),
            ("ibes_actuals",       "ticker", "IBES tickers (actuals)"),
            ("crsp_compustat_link", "lpermno", "Linked PERMNOs"),
            ("crsp_compustat_link", "gvkey",  "Linked GVKEYs"),
            ("ibes_crsp_link",     "ticker", "IBES→CRSP tickers"),
            ("ibes_crsp_link",     "permno", "IBES→CRSP PERMNOs"),
        ]

        for table, col, label in checks:
            try:
                n = self._q1(f"SELECT COUNT(DISTINCT {col}) FROM {table}")
                logger.info(f"    {label:40s}: {n:>10,}")
            except Exception as e:
                logger.error(f"    {label:40s}: ERROR — {e}")

    # ───────────────────────────────────────────────────────────────
    # 5. Join coverage (the critical test)
    # ───────────────────────────────────────────────────────────────

    def check_join_coverage(self):
        """Test that critical joins produce reasonable coverage."""
        logger.info("")
        logger.info("═" * 65)
        logger.info("  5. JOIN COVERAGE")
        logger.info("═" * 65)

        # 5a: CRSP → Compustat link coverage
        try:
            row = self._q("""
                SELECT
                    COUNT(DISTINCT sp.permno) AS crsp_permnos,
                    COUNT(DISTINCT CASE WHEN ccl.gvkey IS NOT NULL THEN sp.permno END) AS matched,
                    ROUND(100.0 * COUNT(DISTINCT CASE WHEN ccl.gvkey IS NOT NULL THEN sp.permno END)
                          / NULLIF(COUNT(DISTINCT sp.permno), 0), 1) AS pct
                FROM (SELECT DISTINCT permno FROM stock_prices WHERE date >= '2000-01-01') sp
                LEFT JOIN crsp_compustat_link ccl ON sp.permno = ccl.lpermno
            """)[0]
            logger.info(f"  CRSP → Compustat:")
            logger.info(f"    CRSP PERMNOs (post-2000):  {row[0]:>10,}")
            logger.info(f"    Matched to GVKEY:          {row[1]:>10,}")
            logger.info(f"    Match rate:                {row[2]:>9.1f}%")
            if row[2] and row[2] < 40:
                self.issues.append(f"CRSP→Compustat match rate {row[2]}% < 40%")
        except Exception as e:
            logger.error(f"  CRSP → Compustat join check failed: {e}")

        # 5b: Compustat fundamentals coverage for matched stocks
        try:
            row = self._q("""
                SELECT COUNT(DISTINCT cq.gvkey),
                       MIN(cq.datadate),
                       MAX(cq.datadate),
                       COUNT(*)
                FROM compustat_quarterly cq
                JOIN crsp_compustat_link ccl ON cq.gvkey = ccl.gvkey
            """)[0]
            logger.info(f"  Compustat fundamentals (linked):")
            logger.info(f"    Companies:     {row[0]:>10,}")
            logger.info(f"    Date range:    {row[1]} → {row[2]}")
            logger.info(f"    Total quarters:{row[3]:>10,}")
        except Exception as e:
            logger.error(f"  Compustat fundamentals check failed: {e}")

        # 5c: IBES coverage via link
        try:
            row = self._q("""
                SELECT COUNT(DISTINCT icl.permno),
                       COUNT(DISTINCT ia.ticker),
                       MIN(ia.anndats),
                       MAX(ia.anndats)
                FROM ibes_actuals ia
                JOIN ibes_crsp_link icl ON ia.ticker = icl.ticker
            """)[0]
            logger.info(f"  IBES (linked to CRSP):")
            logger.info(f"    PERMNOs with IBES:  {row[0]:>10,}")
            logger.info(f"    IBES tickers:       {row[1]:>10,}")
            logger.info(f"    Announcement range: {row[2]} → {row[3]}")
        except Exception as e:
            logger.error(f"  IBES coverage check failed: {e}")

        # 5d: Triple join — stocks with price + fundamentals + analyst data
        try:
            row = self._q("""
                SELECT COUNT(DISTINCT sp.permno)
                FROM (SELECT DISTINCT permno FROM stock_prices WHERE date >= '2010-01-01') sp
                JOIN crsp_compustat_link ccl ON sp.permno = ccl.lpermno
                JOIN ibes_crsp_link icl ON sp.permno = icl.permno
            """)[0]
            logger.info(f"  Triple coverage (price + fundamentals + analysts):")
            logger.info(f"    PERMNOs (post-2010): {row[0]:>10,}")
        except Exception as e:
            logger.error(f"  Triple join check failed: {e}")

    # ───────────────────────────────────────────────────────────────
    # 6. RDS storage size
    # ───────────────────────────────────────────────────────────────

    def check_storage(self):
        """Check total data size in RDS."""
        logger.info("")
        logger.info("═" * 65)
        logger.info("  6. STORAGE")
        logger.info("═" * 65)

        try:
            rows = self._q("""
                SELECT table_name,
                       pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) AS size,
                       pg_total_relation_size(quote_ident(table_name)) AS bytes
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY pg_total_relation_size(quote_ident(table_name)) DESC
            """)
            total_bytes = 0
            for table_name, size_str, size_bytes in rows:
                logger.info(f"    {table_name:35s}: {size_str}")
                total_bytes += size_bytes

            logger.info(f"    {'TOTAL':35s}: {total_bytes / (1024**3):.2f} GB")
        except Exception as e:
            logger.error(f"  Storage check failed: {e}")

    # ───────────────────────────────────────────────────────────────
    # Full validation
    # ───────────────────────────────────────────────────────────────

    def run_all(self, verbose: bool = False):
        """Run ALL validation checks."""
        logger.info("")
        logger.info("╔" + "═" * 63 + "╗")
        logger.info("║   WRDS DATA VALIDATION REPORT                                ║")
        logger.info("║   " + str(date.today()) + "                                              ║")
        logger.info("╚" + "═" * 63 + "╝")

        counts = self.check_row_counts()
        self.check_date_ranges()
        self.check_null_rates(verbose=verbose)
        self.check_identifiers()
        self.check_join_coverage()
        self.check_storage()

        # ═══ FINAL VERDICT ═══
        logger.info("")
        logger.info("═" * 65)
        if self.issues:
            logger.info(f"  VERDICT: ⚠ {len(self.issues)} ISSUES FOUND")
            logger.info("─" * 65)
            for issue in self.issues:
                logger.info(f"    ⚠ {issue}")
        else:
            logger.info("  VERDICT: ✓ ALL CHECKS PASSED")
        logger.info("═" * 65)

        self.close()
        return len(self.issues)


def main():
    parser = argparse.ArgumentParser(description="Validate WRDS data in RDS")
    parser.add_argument("--verbose", action="store_true", help="Show all null rate checks")
    args = parser.parse_args()

    v = WRDSValidator()
    n_issues = v.run_all(verbose=args.verbose)
    sys.exit(1 if n_issues > 0 else 0)


if __name__ == "__main__":
    main()
