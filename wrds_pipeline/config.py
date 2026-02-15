"""
WRDS Pipeline Configuration
============================
All credentials, connection strings, and constants in one place.
Reads secrets from environment variables — NEVER hardcode credentials.

Setup:
  Create .env in project root (already in .gitignore) with:
    WRDS_USERNAME=hlobo
    WRDS_PASSWORD=your_password
    RDS_HOST=trading-data-db.ca90y4g2mxtw.us-east-1.rds.amazonaws.com
    RDS_PASSWORD=your_password
"""

import os

# Load .env if available (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════
# WRDS (Wharton Research Data Services)
# ═══════════════════════════════════════════════════════════════════

WRDS_USERNAME = os.environ.get("WRDS_USERNAME", "hlobo")
WRDS_PASSWORD = os.environ.get("WRDS_PASSWORD", "")
WRDS_PGHOST = os.environ.get("WRDS_PGHOST", "wrds-pgdata.wharton.upenn.edu")
WRDS_PGPORT = os.environ.get("WRDS_PGPORT", "9737")
WRDS_PGDATABASE = os.environ.get("WRDS_PGDATABASE", "wrds")

if not WRDS_PASSWORD:
    import warnings
    warnings.warn(
        "WRDS_PASSWORD not set. Create a .env file in the project root:\n"
        "  WRDS_PASSWORD=your_password\n"
        "Or set the environment variable directly.",
        stacklevel=2,
    )

# ═══════════════════════════════════════════════════════════════════
# AWS RDS PostgreSQL (destination)
# ═══════════════════════════════════════════════════════════════════

_RDS_HOST = os.environ.get("RDS_HOST", "trading-data-db.ca90y4g2mxtw.us-east-1.rds.amazonaws.com")
_RDS_USER = os.environ.get("RDS_USER", "dbadmin")
_RDS_PASSWORD = os.environ.get("RDS_PASSWORD", "")
_RDS_DATABASE = os.environ.get("RDS_DATABASE", "trading_data")
_RDS_PORT = int(os.environ.get("RDS_PORT", "5432"))

RDS_CONFIG = {
    "host": _RDS_HOST,
    "port": _RDS_PORT,
    "database": _RDS_DATABASE,
    "user": _RDS_USER,
    "password": _RDS_PASSWORD,
}

RDS_CONNECTION_STRING = (
    f"postgresql://{_RDS_USER}:{_RDS_PASSWORD}"
    f"@{_RDS_HOST}:{_RDS_PORT}/{_RDS_DATABASE}"
)

# ═══════════════════════════════════════════════════════════════════
# Download settings
# ═══════════════════════════════════════════════════════════════════

BATCH_SIZE = 10_000          # Rows per INSERT batch
MAX_RETRIES = 3              # Retries per failed query
RETRY_DELAY_SECS = 30        # Seconds between retries
CHUNK_START_YEAR = 1970      # Earliest data year
CHUNK_END_YEAR = 2025        # Latest data year (inclusive)

# ═══════════════════════════════════════════════════════════════════
# Validation thresholds
# ═══════════════════════════════════════════════════════════════════

EXPECTED_MINIMUMS = {
    "stock_prices":          40_000_000,
    "compustat_quarterly":    2_000_000,
    "compustat_annual":         500_000,
    "ibes_summary":           3_000_000,
    "ibes_actuals":             500_000,
    "ff_factors_daily":          10_000,
    "ff_factors_monthly":           500,
    "crsp_compustat_link":       20_000,
    "crsp_monthly":           3_000_000,
    "ibes_crsp_link":            10_000,
    "compustat_security":        50_000,
}
