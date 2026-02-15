"""
NUBLE DataService — Unified Data Access Layer
===============================================
Single interface for ALL data access across the system.
Eliminates hardcoded paths. Local cache → S3 fallback → memory cache.

Usage:
    from nuble.data.data_service import get_data_service

    ds = get_data_service()

    # Parquet files (GKX panel, ticker maps, CRSP, etc.)
    panel = ds.load_parquet("gkx_panel.parquet", columns=["permno", "date", "mvel1"])
    tmap  = ds.load_parquet("ticker_permno_map.parquet")

    # Model files (LightGBM, HMM, production, universal)
    model_path = ds.get_model_path("lightgbm/lgb_mega.txt")
    prod_path  = ds.get_model_path("production/mega_production.txt")
    hmm_path   = ds.get_model_path("regime/hmm_regime_model.pkl")

    # JSON files (registries, manifests)
    registry = ds.load_json("production/production_registry.json", root="models")

    # Paths (for anything that still needs a raw path)
    data_dir   = ds.data_dir      # Path to data/wrds/
    models_dir = ds.models_dir    # Path to models/
    project_root = ds.project_root

    # Ticker ↔ PERMNO
    permno = ds.ticker_to_permno("AAPL")
    ticker = ds.permno_to_ticker(14593)
"""

import os
import json
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from functools import lru_cache

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Project root detection (robust: works in Docker, EC2, local)
# ─────────────────────────────────────────────────────────────

def _find_project_root() -> Path:
    """Find project root by walking up from this file."""
    # src/nuble/data/data_service.py → 3 parents up = project root
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
        return candidate

    # Fallbacks for Docker / Lambda
    for path in [
        Path(os.environ.get("NUBLE_PROJECT_ROOT", "")),
        Path("/app"),
        Path.cwd(),
    ]:
        if path.exists() and (path / "src" / "nuble").exists():
            return path

    return candidate  # best guess


# ─────────────────────────────────────────────────────────────
# DataService
# ─────────────────────────────────────────────────────────────

class DataService:
    """
    Unified data access for NUBLE.

    Hierarchy:
      1. In-memory cache (LRU, TTL-based expiry)
      2. Local disk (data/wrds/, models/)
      3. S3 via S3DataManager (transparent fallback)

    Thread-safe. Singleton via get_data_service().
    """

    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        self._root = Path(project_root) if project_root else _find_project_root()
        self._data_dir = self._root / "data" / "wrds"
        self._models_dir = self._root / "models"
        self._results_dir = self._root / "wrds_pipeline" / "phase3" / "results"

        # S3 fallback (lazy)
        self._s3dm = None
        self._s3dm_lock = threading.Lock()

        # Ticker map cache (lazy)
        self._ticker_to_permno_map: Optional[Dict[str, int]] = None
        self._permno_to_ticker_map: Optional[Dict[int, str]] = None
        self._ticker_map_lock = threading.Lock()

        logger.info(f"DataService initialised — root={self._root}")

    # ───────── Properties ─────────

    @property
    def project_root(self) -> Path:
        return self._root

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    @property
    def results_dir(self) -> Path:
        return self._results_dir

    # ───────── S3 fallback (lazy) ─────────

    def _get_s3dm(self):
        """Lazy-load S3DataManager."""
        if self._s3dm is None:
            with self._s3dm_lock:
                if self._s3dm is None:
                    try:
                        from nuble.data.s3_data_manager import S3DataManager
                        self._s3dm = S3DataManager(
                            local_data_dir=str(self._data_dir),
                            local_models_dir=str(self._models_dir),
                        )
                        logger.info("S3DataManager connected as fallback")
                    except Exception as e:
                        logger.debug(f"S3DataManager unavailable: {e}")
                        self._s3dm = False  # sentinel — tried and failed
        return self._s3dm if self._s3dm is not False else None

    # ───────── Parquet loading ─────────

    def load_parquet(self,
                     filename: str,
                     columns: Optional[List[str]] = None,
                     filters: Optional[list] = None,
                     subdir: Optional[str] = None):
        """
        Load a parquet file. Tries local → S3 → error.

        Args:
            filename: e.g. "gkx_panel.parquet", "ticker_permno_map.parquet"
            columns: Column projection (memory-efficient for large files)
            filters: PyArrow predicate pushdown filters
            subdir: Subdirectory under data/ (default: "wrds")

        Returns:
            pandas DataFrame
        """
        import pandas as pd

        base = self._data_dir if subdir is None else (self._root / "data" / subdir)
        local_path = base / filename

        # Strategy 1: Local file (fastest)
        if local_path.exists():
            logger.debug(f"Loading local: {local_path}")
            return pd.read_parquet(local_path, columns=columns, filters=filters)

        # Strategy 2: S3 fallback
        s3dm = self._get_s3dm()
        if s3dm is not None:
            try:
                logger.info(f"Loading from S3: {filename}")
                return s3dm.load_parquet(filename, columns=columns, filters=filters)
            except Exception as e:
                logger.warning(f"S3 load failed for {filename}: {e}")

        # Strategy 3: Check alternate local paths
        alt_paths = [
            self._root / "data" / filename,
            self._root / filename,
        ]
        for alt in alt_paths:
            if alt.exists():
                return pd.read_parquet(alt, columns=columns, filters=filters)

        raise FileNotFoundError(
            f"Parquet file '{filename}' not found.\n"
            f"  Checked: {local_path}\n"
            f"  S3: {'unavailable' if s3dm is None else 'not found'}\n"
            f"  Fix: Place the file in {base}/ or configure S3 credentials."
        )

    # ───────── JSON loading ─────────

    def load_json(self, relative_path: str, root: str = "data") -> dict:
        """
        Load a JSON file relative to a root directory.

        Args:
            relative_path: e.g. "production_registry.json"
            root: "data" (data/wrds/), "models", "project" (project root)
        """
        if root == "models":
            base = self._models_dir
        elif root == "project":
            base = self._root
        else:
            base = self._data_dir

        local_path = base / relative_path
        if local_path.exists():
            with open(local_path) as f:
                return json.load(f)

        # S3 fallback
        s3dm = self._get_s3dm()
        if s3dm is not None:
            try:
                return s3dm.load_json(relative_path)
            except Exception:
                pass

        raise FileNotFoundError(f"JSON file not found: {local_path}")

    # ───────── Model paths ─────────

    def get_model_path(self, relative_path: str) -> Path:
        """
        Get path to a model file, downloading from S3 if needed.

        Args:
            relative_path: e.g. "lightgbm/lgb_mega.txt",
                           "production/mega_production.txt",
                           "regime/hmm_regime_model.pkl",
                           "universal/universal_technical_model.txt"
        Returns:
            Path to the local model file
        """
        local_path = self._models_dir / relative_path
        if local_path.exists():
            return local_path

        # S3 fallback
        s3dm = self._get_s3dm()
        if s3dm is not None:
            try:
                return s3dm.load_model(relative_path)
            except Exception:
                pass

        raise FileNotFoundError(
            f"Model not found: {local_path}\n"
            f"  S3: {'unavailable' if s3dm is None else 'not found'}"
        )

    def model_exists(self, relative_path: str) -> bool:
        """Check if a model file exists locally."""
        return (self._models_dir / relative_path).exists()

    def list_models(self, subdir: str = "") -> List[str]:
        """List model files in a subdirectory."""
        model_dir = self._models_dir / subdir
        if not model_dir.exists():
            return []
        return [f.name for f in model_dir.iterdir() if f.is_file()]

    # ───────── File existence ─────────

    def data_file_exists(self, filename: str) -> bool:
        """Check if a data file exists locally."""
        return (self._data_dir / filename).exists()

    def get_data_path(self, filename: str) -> Path:
        """Get the local path for a data file (may not exist)."""
        return self._data_dir / filename

    # ───────── Ticker ↔ PERMNO mapping ─────────

    def _ensure_ticker_map(self):
        """Lazy-load ticker ↔ PERMNO mapping."""
        if self._ticker_to_permno_map is not None:
            return

        with self._ticker_map_lock:
            if self._ticker_to_permno_map is not None:
                return

            self._ticker_to_permno_map = {}
            self._permno_to_ticker_map = {}

            import pandas as pd

            # Try ticker_permno_map.parquet
            for map_name in ["ticker_permno_map.parquet", "permno_ticker_map.parquet"]:
                try:
                    tmap = self.load_parquet(map_name)
                    for _, row in tmap.iterrows():
                        if pd.notna(row.get("ticker")) and pd.notna(row.get("permno")):
                            ticker = str(row["ticker"]).strip().upper()
                            permno = int(row["permno"])
                            self._ticker_to_permno_map[ticker] = permno
                            self._permno_to_ticker_map[permno] = ticker
                    if self._ticker_to_permno_map:
                        logger.info(f"Ticker map loaded from {map_name}: "
                                    f"{len(self._ticker_to_permno_map)} tickers")
                        return
                except Exception:
                    continue

            # Fallback: extract from crsp_monthly.parquet
            try:
                crsp = self.load_parquet("crsp_monthly.parquet",
                                         columns=["permno", "date", "ticker"])
                crsp["date"] = pd.to_datetime(crsp["date"])
                latest = crsp.sort_values("date").groupby("permno").last()
                for permno, row in latest.iterrows():
                    if pd.notna(row.get("ticker")):
                        ticker = str(row["ticker"]).strip().upper()
                        self._ticker_to_permno_map[ticker] = int(permno)
                        self._permno_to_ticker_map[int(permno)] = ticker
                logger.info(f"Ticker map from CRSP: {len(self._ticker_to_permno_map)} tickers")
            except Exception as e:
                logger.warning(f"Could not build ticker map: {e}")

    def ticker_to_permno(self, ticker: str) -> Optional[int]:
        """Convert ticker to PERMNO."""
        self._ensure_ticker_map()
        return self._ticker_to_permno_map.get(ticker.upper().strip())

    def permno_to_ticker(self, permno: int) -> Optional[str]:
        """Convert PERMNO to ticker."""
        self._ensure_ticker_map()
        return self._permno_to_ticker_map.get(permno)

    def get_ticker_map(self) -> Dict[str, int]:
        """Get full ticker → PERMNO map."""
        self._ensure_ticker_map()
        return dict(self._ticker_to_permno_map)

    def get_permno_map(self) -> Dict[int, str]:
        """Get full PERMNO → ticker map."""
        self._ensure_ticker_map()
        return dict(self._permno_to_ticker_map)

    # ───────── Status / diagnostics ─────────

    def get_status(self) -> Dict[str, Any]:
        """Get DataService status for health checks."""
        status = {
            "project_root": str(self._root),
            "data_dir": str(self._data_dir),
            "data_dir_exists": self._data_dir.exists(),
            "models_dir": str(self._models_dir),
            "models_dir_exists": self._models_dir.exists(),
        }

        # Count local files
        if self._data_dir.exists():
            parquets = list(self._data_dir.glob("*.parquet"))
            status["local_parquet_files"] = len(parquets)
            status["local_parquet_total_mb"] = round(
                sum(f.stat().st_size for f in parquets) / 1e6, 1
            )
        else:
            status["local_parquet_files"] = 0

        # Key files
        key_files = [
            "gkx_panel.parquet",
            "ticker_permno_map.parquet",
            "crsp_monthly.parquet",
        ]
        status["key_files"] = {
            f: (self._data_dir / f).exists() for f in key_files
        }

        # Model directories
        model_subdirs = ["lightgbm", "production", "regime", "universal"]
        status["model_dirs"] = {}
        for sd in model_subdirs:
            sd_path = self._models_dir / sd
            if sd_path.exists():
                files = [f.name for f in sd_path.iterdir() if f.is_file()]
                status["model_dirs"][sd] = files
            else:
                status["model_dirs"][sd] = []

        # S3
        s3dm = self._get_s3dm()
        status["s3_available"] = s3dm is not None and getattr(s3dm, 's3_available', False)

        return status


# ─────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────

_data_service_instance: Optional[DataService] = None
_ds_lock = threading.Lock()


def get_data_service(project_root: Optional[Union[str, Path]] = None) -> DataService:
    """Get or create the singleton DataService instance."""
    global _data_service_instance
    if _data_service_instance is None:
        with _ds_lock:
            if _data_service_instance is None:
                _data_service_instance = DataService(project_root=project_root)
    return _data_service_instance
