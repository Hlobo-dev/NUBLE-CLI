"""
NUBLE Model Manager — Lifecycle, Freshness & Health
=====================================================

Manages the universal model lifecycle:
- Freshness checks (stale after 7 days)
- Health verification (model files loadable, quality gates)
- Background retraining (non-blocking, threaded)
- CLI/API status integration

Usage:
    from nuble.ml.model_manager import ModelManager
    mgr = ModelManager()
    
    if mgr.needs_retraining():
        mgr.trigger_background_retrain()
    
    health = mgr.check_health()
    print(mgr.get_status_for_cli())
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_STALE_DAYS = 7  # Universal model is stale after this many days
_PER_TICKER_STALE_DAYS = 7  # Per-ticker models stale threshold


class ModelManager:
    """
    Manages model lifecycle: freshness checks, background retraining, health.

    RULES:
    - Universal model: retrain if older than 7 days or missing
    - Per-ticker models: retrain if older than 7 days (legacy support)
    - Health check: verify model files exist and are loadable
    - Background retrain: non-blocking, uses threading
    """

    def __init__(self, model_dir: str = "models/"):
        self.model_dir = Path(model_dir)
        self._universal_dir = self.model_dir / "universal"
        self._retraining_thread: Optional[threading.Thread] = None
        self._retraining_lock = threading.Lock()
        self._freshness_checked = False  # Only check once per session

    # ── Health ────────────────────────────────────────────────

    def check_health(self) -> Dict[str, Any]:
        """
        Returns comprehensive health status of all models.

        Returns dict with:
        - 'universal': {exists, age_days, fresh, n_training_samples, test_ic, ...}
        - 'per_ticker': {SYMBOL: {exists, age_days, fresh}, ...}
        - 'overall': {healthy, stale_count, missing_count}
        """
        result = {
            "universal": self._check_universal_health(),
            "per_ticker": self._check_per_ticker_health(),
            "overall": {},
        }

        # Compute overall
        u = result["universal"]
        stale = 0
        missing = 0

        if not u["exists"]:
            missing += 1
        elif not u["fresh"]:
            stale += 1

        for sym, info in result["per_ticker"].items():
            if not info["exists"]:
                missing += 1
            elif not info["fresh"]:
                stale += 1

        result["overall"] = {
            "healthy": u["exists"] and u["fresh"],
            "stale_count": stale,
            "missing_count": missing,
            "retraining_in_progress": self.is_retraining(),
        }

        return result

    def _check_universal_health(self) -> Dict[str, Any]:
        """Check universal model file, metadata, and backtest results."""
        model_file = self._universal_dir / "universal_technical_model.txt"
        meta_file = self._universal_dir / "universal_metadata.json"
        backtest_file = self._universal_dir / "backtest_results.json"

        info: Dict[str, Any] = {
            "exists": False,
            "age_days": None,
            "fresh": False,
            "path": str(self._universal_dir),
            "n_training_samples": None,
            "test_ic": None,
            "test_accuracy": None,
            "n_features": None,
            "trained_at": None,
            "quality_gates_passed": None,
            "model_size_kb": None,
            "backtest_available": False,
            "backtest_ic_ir": None,
            "backtest_ls_sharpe": None,
            "backtest_mean_ic": None,
            "backtest_warning": None,
        }

        if not model_file.exists():
            return info

        info["exists"] = True
        info["model_size_kb"] = model_file.stat().st_size / 1024

        # Age from file mtime
        mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
        age = datetime.now() - mtime
        info["age_days"] = age.days
        info["fresh"] = age.days < _STALE_DAYS
        info["trained_at"] = mtime.isoformat()

        # Load metadata if available
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                info["n_training_samples"] = meta.get("n_training_samples")
                info["test_ic"] = meta.get("test_ic_mean")
                info["test_accuracy"] = meta.get("test_accuracy")
                info["n_features"] = meta.get("n_features")
                info["quality_gates_passed"] = meta.get("quality_gates_passed")
            except Exception as e:
                logger.warning("Failed to read model metadata: %s", e)

        # Verify model is loadable
        try:
            import lightgbm as lgb
            lgb.Booster(model_file=str(model_file))
            info["loadable"] = True
        except Exception as e:
            info["loadable"] = False
            info["load_error"] = str(e)

        # Check backtest results
        if backtest_file.exists():
            try:
                with open(backtest_file) as f:
                    bt = json.load(f)
                info["backtest_available"] = True
                info["backtest_mean_ic"] = bt.get("mean_ic")
                info["backtest_ic_ir"] = bt.get("ic_ir")
                info["backtest_ls_sharpe"] = bt.get("long_short_sharpe")

                # Warnings
                bt_ic = bt.get("mean_ic", 0) or 0
                bt_sharpe = bt.get("long_short_sharpe", 0) or 0
                if bt_ic < 0.01:
                    info["backtest_warning"] = "Backtest IC < 0.01 — model unreliable"
                elif bt_sharpe < 0.5:
                    info["backtest_warning"] = "Backtest L/S Sharpe < 0.5 — weak signal"
            except Exception:
                pass

        return info

    def _check_per_ticker_health(self) -> Dict[str, Dict[str, Any]]:
        """Check per-ticker model files (legacy MLP/LSTM/etc).
        
        DEPRECATED: Per-ticker .pt models are no longer used for production decisions.
        System B multi-tier LightGBM ensemble + LivePredictor is the sole signal source.
        These models remain for reference only.
        """
        result = {}

        if not self.model_dir.exists():
            return result

        for f in self.model_dir.glob("*.pt"):
            parts = f.stem.split("_")
            if len(parts) >= 2:
                # Format: mlp_SPY_20260130.pt
                symbol = parts[1] if len(parts) >= 3 else parts[0]
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                age = datetime.now() - mtime
                result[symbol] = {
                    "exists": True,
                    "deprecated": True,  # Mark as deprecated
                    "age_days": age.days,
                    "fresh": age.days < _PER_TICKER_STALE_DAYS,
                    "path": str(f),
                    "size_kb": f.stat().st_size / 1024,
                    "note": "DEPRECATED: Use LivePredictor (System B multi-tier ensemble) instead",
                }

        return result

    # ── Freshness ─────────────────────────────────────────────

    def needs_retraining(self) -> bool:
        """True if universal model doesn't exist or is older than 7 days."""
        health = self._check_universal_health()
        return not health["exists"] or not health["fresh"]

    def check_freshness_once(self) -> Optional[str]:
        """
        Check freshness once per session. Returns warning string if stale,
        None if fresh or already checked.

        Use this in manager.py process_prompt() — it only fires once.
        """
        if self._freshness_checked:
            return None

        self._freshness_checked = True
        health = self._check_universal_health()

        if not health["exists"]:
            return (
                "⚠️  Universal model not found. Predictions will use per-ticker fallback.\n"
                "   Train: python scripts/train_universal.py"
            )

        if not health["fresh"]:
            return (
                f"⚠️  Universal model is {health['age_days']}d old (stale after {_STALE_DAYS}d).\n"
                f"   Retrain: python scripts/train_universal.py"
            )

        return None

    # ── Background Retraining ─────────────────────────────────

    def is_retraining(self) -> bool:
        """True if a background retrain is currently running."""
        return (
            self._retraining_thread is not None
            and self._retraining_thread.is_alive()
        )

    def trigger_background_retrain(
        self, n_stocks: int = 500, quick: bool = False
    ) -> bool:
        """
        Start retraining on a background thread. Non-blocking.

        The system continues to use the stale model. When complete,
        the new model replaces the old model atomically (handled by
        UniversalTechnicalModel.train() save logic).

        Returns True if retrain started, False if already in progress.
        """
        with self._retraining_lock:
            if self.is_retraining():
                logger.info("Background retrain already in progress, skipping")
                return False

            log_path = Path(os.path.expanduser("~/.nuble/training.log"))
            log_path.parent.mkdir(parents=True, exist_ok=True)

            def _retrain():
                try:
                    logger.info("Background retrain started (n_stocks=%d)", n_stocks)
                    with open(log_path, "w") as log_f:
                        log_f.write(f"[{datetime.now().isoformat()}] Retrain started\n")
                        log_f.write(f"n_stocks={n_stocks}, quick={quick}\n\n")

                    # Use subprocess to run train_universal.py so it's isolated
                    script = Path(__file__).resolve().parents[2] / "scripts" / "train_universal.py"
                    if not script.exists():
                        # Fallback: try from project root
                        script = Path("scripts/train_universal.py").resolve()

                    cmd = [sys.executable, str(script)]
                    cmd.extend(["--n-stocks", str(n_stocks)])
                    if quick:
                        cmd.append("--quick")

                    with open(log_path, "a") as log_f:
                        proc = subprocess.run(
                            cmd,
                            stdout=log_f,
                            stderr=subprocess.STDOUT,
                            timeout=1800,  # 30 min max
                        )

                    with open(log_path, "a") as log_f:
                        if proc.returncode == 0:
                            log_f.write(f"\n[{datetime.now().isoformat()}] Retrain SUCCEEDED\n")
                            logger.info("Background retrain completed successfully")
                        else:
                            log_f.write(
                                f"\n[{datetime.now().isoformat()}] Retrain FAILED "
                                f"(exit code {proc.returncode})\n"
                            )
                            logger.warning(
                                "Background retrain failed (exit %d). See %s",
                                proc.returncode,
                                log_path,
                            )

                except subprocess.TimeoutExpired:
                    logger.error("Background retrain timed out after 30 min")
                except Exception as e:
                    logger.error("Background retrain error: %s", e)

            self._retraining_thread = threading.Thread(
                target=_retrain, daemon=True, name="nuble-model-retrain"
            )
            self._retraining_thread.start()
            return True

    # ── CLI Integration ───────────────────────────────────────

    def get_status_for_cli(self) -> str:
        """
        Formatted status string for the CLI /status command.
        Shows model health in a readable format.
        """
        health = self.check_health()
        u = health["universal"]
        pt = health["per_ticker"]
        lines = []

        # Universal model
        if u["exists"]:
            fresh_icon = "🟢" if u["fresh"] else "🟡"
            age_str = f"{u['age_days']}d old" if u["age_days"] is not None else "unknown age"

            lines.append(f"  {fresh_icon} Universal Model: {age_str}")

            if u.get("test_ic") is not None:
                ic = u['test_ic'] or 0
                acc = u.get('test_accuracy') or 0
                n_samples = u.get('n_training_samples')
                samples_str = f"{n_samples:,}" if n_samples else "?"
                lines.append(f"     IC={ic:.4f}  "
                             f"Acc={acc:.1%}  "
                             f"Features={u.get('n_features', '?')}  "
                             f"Samples={samples_str}")

            if u.get("model_size_kb"):
                lines.append(f"     Size: {u['model_size_kb']:.0f} KB  "
                             f"Loadable: {'✅' if u.get('loadable') else '❌'}")

            gates = u.get("quality_gates_passed")
            if gates is not None:
                lines.append(f"     Quality gates: {'✅ All passed' if gates else '❌ Failed'}")

            # Backtest results
            if u.get("backtest_available"):
                bt_ic = u.get("backtest_mean_ic") or 0
                bt_ir = u.get("backtest_ic_ir") or 0
                bt_sharpe = u.get("backtest_ls_sharpe") or 0
                lines.append(f"     Backtest: IC={bt_ic:.4f}  IC_IR={bt_ir:.2f}  "
                             f"L/S Sharpe={bt_sharpe:.2f}")
                if u.get("backtest_warning"):
                    lines.append(f"     ⚠️  {u['backtest_warning']}")
            else:
                lines.append(f"     Backtest: not run — use: python scripts/run_backtest.py")
        else:
            lines.append("  ❌ Universal Model: NOT TRAINED")
            lines.append("     Run: python scripts/train_universal.py")

        # Per-ticker models
        if pt:
            fresh_count = sum(1 for v in pt.values() if v["fresh"])
            stale_count = len(pt) - fresh_count
            symbols = ", ".join(sorted(pt.keys()))
            lines.append(
                f"  📦 Per-Ticker Models: {len(pt)} "
                f"({fresh_count} fresh, {stale_count} stale)"
            )
            lines.append(f"     Symbols: {symbols}")
        else:
            lines.append("  📦 Per-Ticker Models: none")

        # Retraining status
        if self.is_retraining():
            lines.append("  🔄 Background retraining IN PROGRESS...")

        return "\n".join(lines)

    def get_compact_status(self) -> str:
        """One-line status for inline display."""
        u = self._check_universal_health()
        if not u["exists"]:
            return "❌ No model"
        ic = u.get("test_ic") or 0
        bt_str = ""
        if u.get("backtest_available"):
            bt_ir = u.get("backtest_ic_ir") or 0
            bt_str = f", BT_IR={bt_ir:.2f}"
        if u["fresh"]:
            return f"🟢 Universal ({u['age_days']}d, IC={ic:.3f}{bt_str})"
        return f"🟡 Stale ({u['age_days']}d)"


# ── Singleton ─────────────────────────────────────────────────

_instance: Optional[ModelManager] = None


def get_model_manager(model_dir: str = "models/") -> ModelManager:
    """Get or create the singleton ModelManager."""
    global _instance
    if _instance is None:
        _instance = ModelManager(model_dir=model_dir)
    return _instance
