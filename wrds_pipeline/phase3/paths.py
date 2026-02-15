"""
Phase 3 Pipeline — Path Configuration
=======================================
Resolves all paths relative to the project root.
Import this module instead of hardcoding absolute paths.

Usage:
    from wrds_pipeline.phase3.paths import DATA_DIR, RESULTS_DIR, MODELS_DIR, PROJECT_ROOT
"""

import os

# Project root: two levels up from this file (wrds_pipeline/phase3/ → project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "wrds")
FEATURE_STORE_DIR = os.path.join(PROJECT_ROOT, "data", "feature_store")

# Model directories
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LGB_MODELS_DIR = os.path.join(MODELS_DIR, "lightgbm")
MLP_MODELS_DIR = os.path.join(MODELS_DIR)  # .pt files in models/
CROSS_SECTIONAL_DIR = os.path.join(MODELS_DIR, "cross_sectional")
UNIVERSAL_DIR = os.path.join(MODELS_DIR, "universal")

# Results
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

# Pipeline
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "wrds_pipeline")
PHASE3_DIR = os.path.dirname(__file__)

# S3
S3_BUCKET = "nuble-data-warehouse"


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [DATA_DIR, LGB_MODELS_DIR, RESULTS_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
