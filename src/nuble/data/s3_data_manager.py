"""
S3DataManager — S3 Fallback for WRDS Data & Models
====================================================
Provides transparent S3 download when local files are missing.
Used by DataService and WRDSPredictor as a fallback data source.

Environment Variables:
    NUBLE_DATA_BUCKET  — S3 bucket name (set by ECS task definition)
    S3_DATA_BUCKET     — Alias for NUBLE_DATA_BUCKET
    AWS_REGION         — AWS region (default: us-east-1)

Usage:
    from nuble.data.s3_data_manager import get_data_manager
    dm = get_data_manager()
    df = dm.load_parquet("gkx_panel.parquet")
    model_path = dm.load_model("lightgbm/lgb_mega.txt")
"""

import os
import io
import json
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class S3DataManager:
    """
    Manages data and model access via S3 with local caching.

    Data layout in S3:
        s3://{bucket}/data/wrds/*.parquet
        s3://{bucket}/data/historical/*.parquet
        s3://{bucket}/data/cache/*.json
        s3://{bucket}/models/lightgbm/*.txt
        s3://{bucket}/models/production/*.txt
        s3://{bucket}/models/regime/*.pkl
        s3://{bucket}/models/universal/*.txt

    Local cache mirrors S3 layout under project root.
    """

    def __init__(self,
                 bucket: Optional[str] = None,
                 region: Optional[str] = None,
                 local_data_dir: Optional[str] = None,
                 local_models_dir: Optional[str] = None):
        self._bucket = bucket or os.environ.get("NUBLE_DATA_BUCKET") or os.environ.get("S3_DATA_BUCKET", "")
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")

        # Local cache directories
        self._local_data_dir = Path(local_data_dir) if local_data_dir else Path("/app/data/wrds")
        self._local_models_dir = Path(local_models_dir) if local_models_dir else Path("/app/models")

        self._s3_client = None
        self._s3_lock = threading.Lock()
        self._s3_available = False

        # Try to connect on init
        if self._bucket:
            self._init_s3()
        else:
            logger.info("S3DataManager: No bucket configured (NUBLE_DATA_BUCKET not set)")

    def _init_s3(self):
        """Initialize S3 client and verify bucket access."""
        try:
            import boto3
            from botocore.config import Config as BotoConfig
            from botocore.exceptions import ClientError, NoCredentialsError

            config = BotoConfig(
                region_name=self._region,
                retries={"max_attempts": 3, "mode": "adaptive"},
                connect_timeout=5,
                read_timeout=30,
            )
            self._s3_client = boto3.client("s3", config=config)

            # Verify bucket access with a lightweight HeadBucket call
            self._s3_client.head_bucket(Bucket=self._bucket)
            self._s3_available = True
            logger.info(f"✅ S3DataManager connected: s3://{self._bucket} ({self._region})")

        except (ImportError,):
            logger.warning("S3DataManager: boto3 not installed")
            self._s3_available = False
        except Exception as e:
            logger.warning(f"S3DataManager: Cannot access bucket '{self._bucket}': {e}")
            self._s3_available = False

    @property
    def s3_available(self) -> bool:
        return self._s3_available

    # ───────── Parquet Loading ─────────

    def load_parquet(self,
                     filename: str,
                     columns: Optional[List[str]] = None,
                     filters: Optional[list] = None):
        """
        Load a parquet file from S3, caching locally.

        Args:
            filename: e.g. "gkx_panel.parquet"
            columns: Column projection
            filters: PyArrow predicate filters
        """
        import pandas as pd

        # Check local cache first
        local_path = self._local_data_dir / filename
        if local_path.exists():
            return pd.read_parquet(local_path, columns=columns, filters=filters)

        if not self._s3_available:
            raise FileNotFoundError(
                f"'{filename}' not found locally and S3 is unavailable"
            )

        # Download from S3 to local cache
        s3_key = f"data/wrds/{filename}"
        self._download_file(s3_key, local_path)

        return pd.read_parquet(local_path, columns=columns, filters=filters)

    # ───────── Model Loading ─────────

    def load_model(self, relative_path: str) -> Path:
        """
        Get a model file, downloading from S3 if not cached locally.

        Args:
            relative_path: e.g. "lightgbm/lgb_mega.txt",
                           "production/mega_production.txt"
        Returns:
            Path to the local model file
        """
        local_path = self._local_models_dir / relative_path
        if local_path.exists():
            return local_path

        if not self._s3_available:
            raise FileNotFoundError(
                f"Model '{relative_path}' not found locally and S3 is unavailable"
            )

        s3_key = f"models/{relative_path}"
        self._download_file(s3_key, local_path)
        return local_path

    # ───────── JSON Loading ─────────

    def load_json(self, relative_path: str) -> dict:
        """Load a JSON file from S3."""
        local_path = self._local_data_dir / relative_path
        if local_path.exists():
            with open(local_path) as f:
                return json.load(f)

        if not self._s3_available:
            raise FileNotFoundError(
                f"JSON '{relative_path}' not found locally and S3 unavailable"
            )

        s3_key = f"data/wrds/{relative_path}"
        self._download_file(s3_key, local_path)

        with open(local_path) as f:
            return json.load(f)

    # ───────── Sync Operations ─────────

    def sync_data(self, prefix: str = "data/wrds/", local_dir: Optional[Path] = None):
        """
        Sync an S3 prefix to a local directory.
        Skips files that already exist locally with matching size.
        """
        if not self._s3_available:
            logger.warning("S3 not available for sync")
            return 0

        target_dir = local_dir or self._local_data_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        synced = 0
        try:
            paginator = self._s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    filename = key.replace(prefix, "", 1)
                    if not filename or filename.endswith("/"):
                        continue

                    local_file = target_dir / filename
                    # Skip if local file exists with same size
                    if local_file.exists() and local_file.stat().st_size == obj["Size"]:
                        continue

                    self._download_file(key, local_file)
                    synced += 1

            logger.info(f"Synced {synced} files from s3://{self._bucket}/{prefix}")
        except Exception as e:
            logger.error(f"Sync failed for {prefix}: {e}")

        return synced

    def sync_models(self):
        """Sync all models from S3."""
        return self.sync_data(prefix="models/", local_dir=self._local_models_dir)

    # ───────── File Listing ─────────

    def list_files(self, prefix: str = "data/wrds/") -> List[str]:
        """List files in S3 under a prefix."""
        if not self._s3_available:
            return []

        files = []
        try:
            paginator = self._s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    files.append(obj["Key"])
        except Exception as e:
            logger.error(f"Failed to list S3 files: {e}")
        return files

    # ───────── Internal ─────────

    def _download_file(self, s3_key: str, local_path: Path):
        """Download a single file from S3 to local path."""
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Downloading s3://{self._bucket}/{s3_key} → {local_path}")
            self._s3_client.download_file(self._bucket, s3_key, str(local_path))
            size_mb = local_path.stat().st_size / 1e6
            logger.info(f"  ✓ Downloaded {size_mb:.1f} MB")
        except Exception as e:
            logger.error(f"S3 download failed: s3://{self._bucket}/{s3_key} — {e}")
            # Clean up partial download
            if local_path.exists():
                local_path.unlink()
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get S3DataManager status for diagnostics."""
        status = {
            "bucket": self._bucket or "not configured",
            "region": self._region,
            "s3_available": self._s3_available,
            "local_data_dir": str(self._local_data_dir),
            "local_models_dir": str(self._local_models_dir),
        }

        if self._s3_available:
            try:
                # Count objects in data prefix
                data_files = self.list_files("data/wrds/")
                status["s3_data_files"] = len(data_files)
                model_files = self.list_files("models/")
                status["s3_model_files"] = len(model_files)
            except Exception:
                pass

        return status


# ── Singleton ──────────────────────────────────────────────────
_data_manager_instance: Optional[S3DataManager] = None
_dm_lock = threading.Lock()


def get_data_manager(**kwargs) -> S3DataManager:
    """Get or create the singleton S3DataManager instance."""
    global _data_manager_instance
    if _data_manager_instance is None:
        with _dm_lock:
            if _data_manager_instance is None:
                _data_manager_instance = S3DataManager(**kwargs)
    return _data_manager_instance
