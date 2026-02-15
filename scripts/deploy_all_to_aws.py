#!/usr/bin/env python3
"""
NUBLE Data Lake — Enterprise AWS Deployment
=============================================
Deploys ALL data, models, pipeline code, and results to S3 with
a production-grade data lake architecture.

Architecture:
  s3://nuble-data-warehouse/
  ├── raw/                    ← Original WRDS downloads (immutable)
  │   ├── wrds/               ← Compustat, IBES, CRSP monthly, etc.
  │   └── crsp_daily/         ← CRSP daily by year (1926-2024)
  ├── features/               ← Computed feature panels
  │   ├── gkx_panel.parquet   ← Master panel (3.76M × 539)
  │   ├── training_panel.parquet
  │   ├── daily_features.parquet
  │   └── ...
  ├── predictions/            ← ML model outputs per tier
  │   ├── mega/
  │   ├── large/
  │   ├── mid/
  │   ├── small/
  │   └── ensemble/
  ├── hedging/                ← Dynamic hedging results
  ├── models/                 ← Trained model artifacts
  │   ├── lightgbm/
  │   ├── mlp/
  │   ├── cross_sectional/
  │   └── universal/
  ├── pipeline/               ← WRDS pipeline source code
  │   ├── phase2/
  │   ├── phase3/
  │   └── results/
  ├── results/                ← Training & validation results
  │   ├── training/
  │   └── validation/
  └── metadata/               ← Manifests, configs, mappings
      ├── manifests/
      └── mappings/

Storage tiers (automatic):
  - INTELLIGENT_TIERING for all data
  - Versioning enabled (30d → Glacier, 365d → expire)
  - AES-256 server-side encryption

Usage:
  python scripts/deploy_all_to_aws.py                  # Full deploy
  python scripts/deploy_all_to_aws.py --dry-run        # Preview only
  python scripts/deploy_all_to_aws.py --layer raw      # Upload only raw data
  python scripts/deploy_all_to_aws.py --layer models   # Upload only models
  python scripts/deploy_all_to_aws.py --verify         # Verify S3 vs local
"""

import os
import sys
import json
import time
import hashlib
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
BUCKET = "nuble-data-warehouse"
REGION = "us-east-1"
PROJECT_DIR = Path.home() / "Desktop" / "NUBLE-CLI"
DATA_DIR = PROJECT_DIR / "data" / "wrds"
MODELS_DIR = PROJECT_DIR / "models"
PIPELINE_DIR = PROJECT_DIR / "wrds_pipeline"
TRAINING_DIR = PROJECT_DIR / "training_results"
VALIDATION_DIR = PROJECT_DIR / "validation_results"
VALIDATION_DIR2 = PROJECT_DIR / "validation"

# Maximum parallel uploads
MAX_WORKERS = 4
# Multipart threshold (files > this use multipart upload)
MULTIPART_THRESHOLD_MB = 100

# ─────────────────────────────────────────────────────────────
# S3 Key Mapping — Maps local files to organized S3 paths
# ─────────────────────────────────────────────────────────────

# Files that belong in raw/ (original WRDS downloads, immutable)
RAW_FILES = {
    'compustat_annual.parquet', 'compustat_quarterly.parquet',
    'compustat_security.parquet', 'crsp_monthly.parquet',
    'crsp_delisting.parquet', 'crsp_distributions.parquet',
    'crsp_index_daily.parquet', 'crsp_index_monthly.parquet',
    'crsp_treasury_daily.parquet', 'crsp_treasury_monthly.parquet',
    'crsp_compustat_link.parquet', 'execucomp.parquet',
    'ff_factors_daily.parquet', 'ff_factors_monthly.parquet',
    'ibes_actuals.parquet', 'ibes_crsp_link.parquet',
    'ibes_recommendations.parquet', 'ibes_summary.parquet',
    'insider_trading.parquet', 'institutional_holdings.parquet',
    'short_interest.parquet', 'sp500_constituents_compustat.parquet',
    'sp500_constituents_crsp.parquet', 'wrds_financial_ratios.parquet',
}

# Files that belong in features/ (computed panels)
FEATURE_FILES = {
    'gkx_panel.parquet', 'gkx_panel_pre_level3.parquet',
    'training_panel.parquet', 'daily_features.parquet',
    'rolling_betas.parquet', 'cz_predictors.parquet',
    'fred_daily.parquet', 'fred_monthly.parquet',
    'macro_predictors.parquet', 'welch_goyal_macro.parquet',
}

# Prediction files → predictions/{tier}/
PREDICTION_FILES = {
    'predictions_mega.parquet': 'predictions/by_tier/mega/',
    'predictions_large.parquet': 'predictions/by_tier/large/',
    'predictions_mid.parquet': 'predictions/by_tier/mid/',
    'predictions_small.parquet': 'predictions/by_tier/small/',
    'curated_predictions_mega.parquet': 'predictions/curated/mega/',
    'curated_predictions_large.parquet': 'predictions/curated/large/',
    'curated_predictions_mid.parquet': 'predictions/curated/mid/',
    'curated_predictions_small.parquet': 'predictions/curated/small/',
    'multi_universe_predictions.parquet': 'predictions/ensemble/',
    'curated_multi_universe_predictions.parquet': 'predictions/ensemble/',
    'ensemble_predictions.parquet': 'predictions/ensemble/',
    'lgb_predictions.parquet': 'predictions/ensemble/',
}

# Hedging/Strategy files → hedging/
HEDGING_FILES = {
    'hedged_returns_mega.parquet': 'hedging/',
    'hedged_returns_large.parquet': 'hedging/',
    'hedged_returns_mid.parquet': 'hedging/',
    'hedged_returns_small.parquet': 'hedging/',
    'blended_hedged_returns.parquet': 'hedging/',
    'ensemble_final_returns.parquet': 'hedging/',
    'ensemble_method_comparison.parquet': 'hedging/',
    'best_strategy_mega.parquet': 'hedging/strategies/',
    'best_strategy_large.parquet': 'hedging/strategies/',
    'best_strategy_mid.parquet': 'hedging/strategies/',
    'best_strategy_small.parquet': 'hedging/strategies/',
    'dynamic_hedging_summary.json': 'hedging/',
}

# Metadata/mapping files → metadata/
METADATA_FILES = {
    'permno_ticker_map.parquet': 'metadata/mappings/',
    'ticker_permno_map.parquet': 'metadata/mappings/',
    'download_manifest.json': 'metadata/manifests/',
    'crsp_daily_manifest.json': 'metadata/manifests/',
}

# Model files → models/
MODEL_MAPPING = {
    'lgb_latest_model.txt': 'models/lightgbm/',
}


def get_s3_key(filepath: str) -> str:
    """Map a local file path to its proper S3 key in the data lake."""
    filename = os.path.basename(filepath)

    # Check explicit mappings first
    if filename in PREDICTION_FILES:
        return f"{PREDICTION_FILES[filename]}{filename}"
    if filename in HEDGING_FILES:
        return f"{HEDGING_FILES[filename]}{filename}"
    if filename in METADATA_FILES:
        return f"{METADATA_FILES[filename]}{filename}"
    if filename in MODEL_MAPPING:
        return f"{MODEL_MAPPING[filename]}{filename}"

    # Category-based mapping
    if filename in RAW_FILES:
        return f"raw/wrds/{filename}"
    if filename in FEATURE_FILES:
        return f"features/{filename}"

    # CRSP daily files
    if 'crsp_daily' in filepath and filename.startswith('crsp_daily_'):
        return f"raw/crsp_daily/{filename}"

    # Default: put in raw/wrds/
    return f"raw/wrds/{filename}"


# ─────────────────────────────────────────────────────────────
# Upload engine
# ─────────────────────────────────────────────────────────────

class S3Uploader:
    """High-performance S3 uploader with progress tracking."""

    def __init__(self, bucket: str, region: str, dry_run: bool = False):
        self.bucket = bucket
        self.region = region
        self.dry_run = dry_run
        self.uploaded = []
        self.skipped = []
        self.failed = []
        self.total_bytes = 0

    def _get_s3_etag(self, key: str) -> Optional[str]:
        """Get the ETag of an S3 object (None if not exists)."""
        try:
            result = subprocess.run(
                ["aws", "s3api", "head-object", "--bucket", self.bucket, "--key", key],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get('ETag', '').strip('"')
        except Exception:
            pass
        return None

    def _get_local_md5(self, filepath: str) -> str:
        """Compute MD5 of local file for comparison."""
        h = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    def upload_file(self, local_path: str, s3_key: str, force: bool = False) -> dict:
        """Upload a single file to S3."""
        size = os.path.getsize(local_path)
        size_mb = size / (1024 * 1024)
        filename = os.path.basename(local_path)

        result = {
            'file': filename,
            'local': local_path,
            's3_key': s3_key,
            'size_mb': round(size_mb, 1),
            'status': 'pending',
        }

        if self.dry_run:
            result['status'] = 'would_upload'
            self.uploaded.append(result)
            self.total_bytes += size
            return result

        # Check if already exists in S3 with same size
        if not force:
            existing = self._get_s3_etag(s3_key)
            if existing:
                result['status'] = 'exists_in_s3'
                self.skipped.append(result)
                return result

        # Upload with appropriate method
        try:
            cmd = [
                "aws", "s3", "cp", local_path,
                f"s3://{self.bucket}/{s3_key}",
                "--storage-class", "INTELLIGENT_TIERING",
                "--region", self.region,
            ]

            # Add multipart config for large files
            if size_mb > MULTIPART_THRESHOLD_MB:
                cmd.extend([
                    "--expected-size", str(size),
                ])

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if proc.returncode == 0:
                result['status'] = 'uploaded'
                self.uploaded.append(result)
                self.total_bytes += size
            else:
                result['status'] = 'failed'
                result['error'] = proc.stderr.strip()[:200]
                self.failed.append(result)

        except subprocess.TimeoutExpired:
            result['status'] = 'timeout'
            result['error'] = 'Upload timed out (>60 min)'
            self.failed.append(result)
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)[:200]
            self.failed.append(result)

        return result

    def get_summary(self) -> dict:
        """Get upload summary."""
        return {
            'uploaded': len(self.uploaded),
            'skipped': len(self.skipped),
            'failed': len(self.failed),
            'total_gb': round(self.total_bytes / (1024**3), 2),
            'failed_files': [f['file'] for f in self.failed],
        }


# ─────────────────────────────────────────────────────────────
# Collect all files to upload
# ─────────────────────────────────────────────────────────────

def collect_data_files() -> List[Tuple[str, str]]:
    """Collect all data/wrds/ files with their S3 keys."""
    files = []

    if not DATA_DIR.exists():
        print(f"⚠️  Data directory not found: {DATA_DIR}")
        return files

    # Top-level files in data/wrds/
    for f in sorted(DATA_DIR.iterdir()):
        if f.is_file() and not f.name.startswith('.'):
            s3_key = get_s3_key(str(f))
            files.append((str(f), s3_key))

    # CRSP daily sub-directory
    crsp_daily = DATA_DIR / "crsp_daily"
    if crsp_daily.exists():
        for f in sorted(crsp_daily.iterdir()):
            if f.is_file() and not f.name.startswith('.'):
                s3_key = f"raw/crsp_daily/{f.name}"
                files.append((str(f), s3_key))

    return files


def collect_model_files() -> List[Tuple[str, str]]:
    """Collect all model files with their S3 keys."""
    files = []

    if not MODELS_DIR.exists():
        print(f"⚠️  Models directory not found: {MODELS_DIR}")
        return files

    for root, dirs, filenames in os.walk(MODELS_DIR):
        for fname in sorted(filenames):
            if fname.startswith('.'):
                continue
            local_path = os.path.join(root, fname)
            relative = os.path.relpath(local_path, MODELS_DIR)

            # Organize by model type
            if fname.endswith('.pt'):
                s3_key = f"models/mlp/{fname}"
            elif fname.endswith('.txt') and 'lgb' not in fname.lower():
                # LightGBM .txt model files
                parent = Path(root).name
                s3_key = f"models/{parent}/{fname}"
            elif fname.endswith('.pkl'):
                parent = Path(root).name
                s3_key = f"models/{parent}/{fname}"
            else:
                s3_key = f"models/{relative}"

            files.append((local_path, s3_key))

    return files


def collect_pipeline_files() -> List[Tuple[str, str]]:
    """Collect pipeline code and results."""
    files = []

    if not PIPELINE_DIR.exists():
        return files

    for root, dirs, filenames in os.walk(PIPELINE_DIR):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != '__pycache__']

        for fname in sorted(filenames):
            if fname.startswith('.') or fname.endswith('.pyc'):
                continue
            local_path = os.path.join(root, fname)
            relative = os.path.relpath(local_path, PIPELINE_DIR)
            s3_key = f"pipeline/{relative}"
            files.append((local_path, s3_key))

    return files


def collect_result_files() -> List[Tuple[str, str]]:
    """Collect training and validation results."""
    files = []

    for dir_path, s3_prefix in [
        (TRAINING_DIR, "results/training"),
        (VALIDATION_DIR, "results/validation"),
        (VALIDATION_DIR2, "results/validation_scripts"),
    ]:
        if not dir_path.exists():
            continue
        for root, dirs, filenames in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for fname in sorted(filenames):
                if fname.startswith('.') or fname.endswith('.pyc'):
                    continue
                local_path = os.path.join(root, fname)
                relative = os.path.relpath(local_path, dir_path)
                s3_key = f"{s3_prefix}/{relative}"
                files.append((local_path, s3_key))

    return files


# ─────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────

def verify_deployment() -> dict:
    """Verify S3 contents match local data."""
    print("\n🔍 Verifying S3 deployment...")

    result = subprocess.run(
        ["aws", "s3", "ls", f"s3://{BUCKET}/", "--recursive", "--summarize"],
        capture_output=True, text=True
    )

    s3_files = {}
    for line in result.stdout.strip().split('\n'):
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0][0].isdigit():
            size = int(parts[2])
            key = parts[3]
            s3_files[key] = size

    # Compare with what we expect
    expected = collect_data_files() + collect_model_files()
    
    matched = 0
    missing = []
    size_mismatch = []

    for local_path, s3_key in expected:
        local_size = os.path.getsize(local_path)
        if s3_key in s3_files:
            matched += 1
            # Allow small size differences (S3 metadata)
            if abs(s3_files[s3_key] - local_size) > 1024:
                size_mismatch.append((s3_key, local_size, s3_files[s3_key]))
        else:
            missing.append((s3_key, local_size))

    report = {
        's3_total_files': len(s3_files),
        'expected_files': len(expected),
        'matched': matched,
        'missing': len(missing),
        'size_mismatches': len(size_mismatch),
        'missing_files': [m[0] for m in missing[:20]],
        'total_s3_gb': round(sum(s3_files.values()) / (1024**3), 2),
    }

    return report


# ─────────────────────────────────────────────────────────────
# Create deployment manifest
# ─────────────────────────────────────────────────────────────

def create_manifest(all_files: List[Tuple[str, str]]) -> dict:
    """Create a deployment manifest documenting everything in S3."""
    manifest = {
        'deployment_timestamp': datetime.now().isoformat(),
        'bucket': BUCKET,
        'region': REGION,
        'total_files': len(all_files),
        'total_size_bytes': sum(os.path.getsize(f[0]) for f in all_files),
        'layers': {},
        'file_inventory': [],
    }

    # Organize by layer
    for local_path, s3_key in all_files:
        layer = s3_key.split('/')[0]
        if layer not in manifest['layers']:
            manifest['layers'][layer] = {'count': 0, 'size_bytes': 0, 'files': []}
        
        size = os.path.getsize(local_path)
        manifest['layers'][layer]['count'] += 1
        manifest['layers'][layer]['size_bytes'] += size
        manifest['layers'][layer]['files'].append(s3_key)

        manifest['file_inventory'].append({
            'local': local_path,
            's3_key': s3_key,
            's3_uri': f"s3://{BUCKET}/{s3_key}",
            'size_bytes': size,
            'size_mb': round(size / (1024**2), 1),
        })

    # Humanize layer sizes
    for layer in manifest['layers']:
        size_gb = manifest['layers'][layer]['size_bytes'] / (1024**3)
        manifest['layers'][layer]['size_gb'] = round(size_gb, 2)
        monthly_cost = size_gb * 0.023  # S3 IT pricing
        manifest['layers'][layer]['monthly_cost_usd'] = round(monthly_cost, 2)

    total_gb = manifest['total_size_bytes'] / (1024**3)
    manifest['total_size_gb'] = round(total_gb, 2)
    manifest['monthly_storage_cost_usd'] = round(total_gb * 0.023, 2)

    return manifest


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='NUBLE Data Lake — AWS S3 Deployment')
    parser.add_argument('--dry-run', action='store_true', help='Preview without uploading')
    parser.add_argument('--force', action='store_true', help='Re-upload even if file exists in S3')
    parser.add_argument('--layer', choices=['raw', 'features', 'models', 'pipeline', 'predictions', 'results', 'all'],
                        default='all', help='Upload only a specific data layer')
    parser.add_argument('--verify', action='store_true', help='Verify S3 vs local without uploading')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help='Parallel upload threads')
    args = parser.parse_args()

    print("=" * 72)
    print("  NUBLE DATA LAKE — ENTERPRISE AWS DEPLOYMENT")
    print(f"  Bucket:  s3://{BUCKET}")
    print(f"  Region:  {REGION}")
    print(f"  Mode:    {'DRY RUN' if args.dry_run else 'LIVE UPLOAD'}")
    print(f"  Layer:   {args.layer}")
    print(f"  Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # Verify AWS auth
    print("\n🔐 Verifying AWS credentials...")
    auth = subprocess.run(
        ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
        capture_output=True, text=True
    )
    if auth.returncode != 0:
        print("❌ AWS credentials not configured. Run 'aws configure' first.")
        sys.exit(1)
    print(f"   ✅ Account: {auth.stdout.strip()}")

    # Verify-only mode
    if args.verify:
        report = verify_deployment()
        print(f"\n📊 Verification Report:")
        print(f"   S3 files:        {report['s3_total_files']}")
        print(f"   Expected:        {report['expected_files']}")
        print(f"   Matched:         {report['matched']}")
        print(f"   Missing:         {report['missing']}")
        print(f"   Size mismatches: {report['size_mismatches']}")
        print(f"   Total S3 size:   {report['total_s3_gb']} GB")
        if report['missing_files']:
            print(f"\n   Missing files:")
            for f in report['missing_files']:
                print(f"     • {f}")
        return

    # Collect files based on layer
    all_files = []
    
    print("\n📂 Collecting files...")

    if args.layer in ('raw', 'features', 'predictions', 'all'):
        data_files = collect_data_files()
        if args.layer == 'raw':
            data_files = [(l, k) for l, k in data_files if k.startswith('raw/')]
        elif args.layer == 'features':
            data_files = [(l, k) for l, k in data_files if k.startswith('features/')]
        elif args.layer == 'predictions':
            data_files = [(l, k) for l, k in data_files if k.startswith('predictions/')]
        all_files.extend(data_files)
        print(f"   Data files:      {len(data_files)}")

    if args.layer in ('models', 'all'):
        model_files = collect_model_files()
        all_files.extend(model_files)
        print(f"   Model files:     {len(model_files)}")

    if args.layer in ('pipeline', 'all'):
        pipeline_files = collect_pipeline_files()
        all_files.extend(pipeline_files)
        print(f"   Pipeline files:  {len(pipeline_files)}")

    if args.layer in ('results', 'all'):
        result_files = collect_result_files()
        all_files.extend(result_files)
        print(f"   Result files:    {len(result_files)}")

    total_size_gb = sum(os.path.getsize(f[0]) for f in all_files) / (1024**3)
    print(f"\n   📊 Total: {len(all_files)} files, {total_size_gb:.2f} GB")
    print(f"   💰 Monthly S3 cost: ~${total_size_gb * 0.023:.2f}")

    if not all_files:
        print("\n⚠️  No files to upload.")
        return

    # Show upload plan
    print(f"\n📋 Upload Plan:")
    layers = {}
    for local_path, s3_key in all_files:
        layer = s3_key.split('/')[0]
        if layer not in layers:
            layers[layer] = {'count': 0, 'size': 0}
        layers[layer]['count'] += 1
        layers[layer]['size'] += os.path.getsize(local_path)

    for layer, info in sorted(layers.items()):
        size_gb = info['size'] / (1024**3)
        print(f"   {layer:20s}  {info['count']:4d} files  {size_gb:8.2f} GB")

    # Upload
    print(f"\n🚀 {'[DRY RUN] Would upload' if args.dry_run else 'Uploading'}...")
    start = time.time()

    uploader = S3Uploader(BUCKET, REGION, dry_run=args.dry_run)

    # Sort by size (upload small files first for quick wins)
    all_files.sort(key=lambda x: os.path.getsize(x[0]))

    for idx, (local_path, s3_key) in enumerate(all_files):
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        filename = os.path.basename(local_path)

        result = uploader.upload_file(local_path, s3_key, force=args.force)
        status = result['status']

        # Progress indicator
        icon = {'uploaded': '✅', 'exists_in_s3': '⏭️ ', 'would_upload': '📝',
                'failed': '❌', 'timeout': '⏰', 'error': '💥'}.get(status, '❓')

        # Only print large files or every 10th file
        if size_mb > 50 or status in ('failed', 'error') or (idx + 1) % 20 == 0 or idx == len(all_files) - 1:
            print(f"   [{idx+1:3d}/{len(all_files)}] {icon} {filename:<50s} "
                  f"{size_mb:8.1f} MB → {s3_key}")

    elapsed = time.time() - start
    summary = uploader.get_summary()

    # Create and upload manifest
    print(f"\n📋 Creating deployment manifest...")
    manifest = create_manifest(all_files)
    manifest['upload_summary'] = summary
    manifest['upload_duration_seconds'] = round(elapsed)

    manifest_path = PROJECT_DIR / "data" / "s3_deployment_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    if not args.dry_run:
        # Upload manifest itself
        subprocess.run([
            "aws", "s3", "cp", str(manifest_path),
            f"s3://{BUCKET}/metadata/manifests/deployment_manifest.json",
            "--storage-class", "INTELLIGENT_TIERING",
        ], capture_output=True)

    # Final report
    print(f"\n{'=' * 72}")
    print(f"  {'DRY RUN COMPLETE' if args.dry_run else 'DEPLOYMENT COMPLETE'}")
    print(f"{'=' * 72}")
    print(f"  ✅ Uploaded:    {summary['uploaded']} files ({summary['total_gb']} GB)")
    print(f"  ⏭️  Skipped:    {summary['skipped']} (already in S3)")
    print(f"  ❌ Failed:      {summary['failed']}")
    print(f"  ⏱️  Duration:   {elapsed/60:.1f} minutes")
    print(f"  💰 Monthly:    ~${manifest.get('monthly_storage_cost_usd', 0):.2f}")
    print(f"  📋 Manifest:   {manifest_path}")
    print(f"")
    print(f"  S3 Data Lake Structure:")
    for layer, info in sorted(manifest['layers'].items()):
        print(f"    ├── {layer:20s}  {info['count']:3d} files  {info['size_gb']:6.2f} GB  ~${info['monthly_cost_usd']:.2f}/mo")
    print(f"")

    if summary['failed'] > 0:
        print(f"  ⚠️  Failed files:")
        for f in summary['failed_files']:
            print(f"     • {f}")
        print(f"  Re-run with --force to retry failed uploads.")

    print(f"")
    print(f"  🔗 Access your data:")
    print(f"     aws s3 ls s3://{BUCKET}/ --recursive --summarize")
    print(f"     aws s3 sync s3://{BUCKET}/data/wrds/ data/wrds/  (download all)")
    print(f"")
    print(f"  🐍 In Python:")
    print(f"     from nuble.data.s3_data_manager import get_data_manager")
    print(f"     dm = get_data_manager()")
    print(f"     df = dm.load_parquet('gkx_panel.parquet', columns=['permno','date','mvel1'])")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
