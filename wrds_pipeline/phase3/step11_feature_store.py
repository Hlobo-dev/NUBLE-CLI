"""
PHASE 3 — STEP 11: Feature Store (DuckDB Local + Parquet)
==========================================================
Point-in-time feature lookups for backtesting and serving.
DuckDB for fast local queries. Parquet for S3/Athena.
Ensures no look-ahead bias in feature construction.
"""

import pandas as pd
import numpy as np
import os
import time
import json

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")
STORE_DIR = os.path.join(_PROJECT_ROOT, "data", "feature_store")


class FeatureStore:
    """Local feature store with point-in-time lookups."""

    def __init__(self, db_path=None):
        os.makedirs(STORE_DIR, exist_ok=True)
        self.db_path = db_path or os.path.join(STORE_DIR, "features.duckdb")
        self._db = None
        self._init_db()

    def _init_db(self):
        """Initialize DuckDB connection."""
        try:
            import duckdb
            self._db = duckdb.connect(self.db_path)
            self._create_tables()
            print(f"  FeatureStore initialized: {self.db_path}")
        except ImportError:
            print("  ⚠️ DuckDB not installed — using Parquet-only mode")
            self._db = None

    def _create_tables(self):
        """Create feature store tables."""
        if self._db is None:
            return

        self._db.execute("""
            CREATE TABLE IF NOT EXISTS feature_registry (
                feature_name VARCHAR PRIMARY KEY,
                description VARCHAR,
                source VARCHAR,
                frequency VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self._db.execute("""
            CREATE TABLE IF NOT EXISTS feature_metadata (
                feature_name VARCHAR,
                date DATE,
                count BIGINT,
                mean DOUBLE,
                std DOUBLE,
                min DOUBLE,
                max DOUBLE,
                null_pct DOUBLE,
                PRIMARY KEY (feature_name, date)
            )
        """)

    def register_features(self, source_parquet: str, source_name: str = ""):
        """Register features from a Parquet file into the store."""
        if not os.path.exists(source_parquet):
            print(f"  ❌ File not found: {source_parquet}")
            return

        df = pd.read_parquet(source_parquet)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        id_cols = ["permno", "date", "cusip", "ticker", "siccd", "year"]
        feature_cols = [c for c in numeric_cols if c not in id_cols]

        # Create symlink in feature store
        link_name = os.path.basename(source_parquet)
        link_path = os.path.join(STORE_DIR, link_name)
        if not os.path.exists(link_path):
            os.symlink(source_parquet, link_path)

        # Register in DuckDB
        if self._db:
            for col in feature_cols:
                self._db.execute("""
                    INSERT OR REPLACE INTO feature_registry
                    (feature_name, source, description, frequency)
                    VALUES (?, ?, ?, ?)
                """, [col, source_name or link_name, f"Feature from {link_name}", "monthly"])

            # Compute metadata if date column exists
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                for col in feature_cols[:50]:  # Limit for speed
                    stats = df.groupby("date")[col].agg(["count", "mean", "std", "min", "max"]).reset_index()
                    stats["null_pct"] = 1 - stats["count"] / df.groupby("date").size().values
                    for _, row in stats.iterrows():
                        try:
                            self._db.execute("""
                                INSERT OR REPLACE INTO feature_metadata
                                (feature_name, date, count, mean, std, min, max, null_pct)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, [col, row["date"], int(row["count"]),
                                  float(row["mean"]) if pd.notna(row["mean"]) else None,
                                  float(row["std"]) if pd.notna(row["std"]) else None,
                                  float(row["min"]) if pd.notna(row["min"]) else None,
                                  float(row["max"]) if pd.notna(row["max"]) else None,
                                  float(row["null_pct"]) if pd.notna(row["null_pct"]) else None])
                        except Exception:
                            pass

        print(f"  ✅ Registered {len(feature_cols)} features from {link_name}")
        return feature_cols

    def get_features_pit(self, permno: int, as_of_date: str, features: list = None) -> dict:
        """Point-in-time feature lookup (no look-ahead bias).

        Returns features available as of the given date.
        """
        result = {"permno": permno, "as_of_date": as_of_date}

        # Load panel
        panel_path = os.path.join(DATA_DIR, "training_panel.parquet")
        if not os.path.exists(panel_path):
            return result

        panel = pd.read_parquet(panel_path)
        panel["date"] = pd.to_datetime(panel["date"])
        as_of = pd.to_datetime(as_of_date)

        # Point-in-time: only data available BEFORE as_of_date
        stock = panel[(panel["permno"] == permno) & (panel["date"] <= as_of)]
        if len(stock) == 0:
            result["error"] = "No data available"
            return result

        latest = stock.sort_values("date").iloc[-1]
        result["feature_date"] = str(latest["date"].date())

        if features is None:
            features = [c for c in latest.index if c not in ["permno", "date", "cusip", "ticker"]]

        for feat in features:
            if feat in latest.index and pd.notna(latest[feat]):
                result[feat] = float(latest[feat])

        return result

    def get_feature_stats(self, feature_name: str) -> dict:
        """Get historical statistics for a feature."""
        if self._db is None:
            return {"error": "DuckDB not available"}

        try:
            stats = self._db.execute("""
                SELECT date, count, mean, std, null_pct
                FROM feature_metadata
                WHERE feature_name = ?
                ORDER BY date
            """, [feature_name]).fetchdf()

            if len(stats) == 0:
                return {"feature": feature_name, "error": "No metadata"}

            return {
                "feature": feature_name,
                "n_periods": len(stats),
                "avg_coverage": float(1 - stats["null_pct"].mean()),
                "avg_mean": float(stats["mean"].mean()),
                "avg_std": float(stats["std"].mean()),
                "latest_date": str(stats["date"].max()),
            }
        except Exception as e:
            return {"feature": feature_name, "error": str(e)[:80]}

    def list_features(self) -> pd.DataFrame:
        """List all registered features."""
        if self._db is None:
            # Fallback: list from Parquet files
            features = []
            for f in os.listdir(STORE_DIR):
                if f.endswith(".parquet"):
                    df = pd.read_parquet(os.path.join(STORE_DIR, f), columns=None)
                    for col in df.select_dtypes(include=[np.number]).columns:
                        features.append({"feature": col, "source": f})
            return pd.DataFrame(features)

        return self._db.execute(
            "SELECT feature_name, source, frequency, updated_at FROM feature_registry ORDER BY feature_name"
        ).fetchdf()

    def close(self):
        """Close DuckDB connection."""
        if self._db:
            self._db.close()


def main():
    print("=" * 70)
    print("PHASE 3 — STEP 11: FEATURE STORE")
    print("=" * 70)
    start = time.time()

    store = FeatureStore()

    # Register all available feature files
    feature_files = [
        ("training_panel.parquet", "Phase 2 Training Panel"),
        ("gkx_panel.parquet", "GKX Feature Panel"),
        ("welch_goyal_macro.parquet", "Welch-Goyal Macro"),
        ("macro_predictors.parquet", "FRED Macro"),
        ("cz_predictors.parquet", "Chen-Zimmermann Predictors"),
        ("daily_features.parquet", "Daily Features"),
        ("rolling_betas.parquet", "Rolling Betas"),
    ]

    total_features = 0
    for fname, source_name in feature_files:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            cols = store.register_features(fpath, source_name)
            if cols:
                total_features += len(cols)

    # Demo: Point-in-time lookup
    print("\n📊 DEMO: Point-in-time feature lookup")
    pit = store.get_features_pit(14593, "2020-06-30",  # AAPL permno
                                  ["log_market_cap", "bm", "mom_12m", "roaq"])
    for k, v in pit.items():
        print(f"  {k}: {v}")

    # List features
    features_df = store.list_features()
    print(f"\n📊 FEATURE REGISTRY: {len(features_df)} features registered")
    if len(features_df) > 0:
        print(features_df.head(10).to_string(index=False))

    store.close()

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"FEATURE STORE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total features registered: {total_features}")
    print(f"  Store location: {STORE_DIR}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  ✅ Feature store operational")


if __name__ == "__main__":
    main()
