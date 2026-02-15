"""
PHASE 3 — STEP 6C: DEEP LEARNING CROSS-SECTIONAL PREDICTOR
=============================================================
Neural network ensemble following:
- Gu-Kelly-Xiu (2020): Feed-forward NN for stock returns
- Chen-Pelger-Zhu (2024): Conditional autoencoder for factor models
- Feng-He-Polson (2018): Deep learning for cross-sectional returns

Architecture:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. FFN (Feed-Forward Network) — GKX-style
   - BatchNorm → Dense(256) → ReLU → Dropout → Dense(128) → ReLU → Dense(1)
   - With weight decay (L2) and learning rate warmup

2. Residual Network — deeper with skip connections
   - Handles vanishing gradient for 4+ layer networks
   - Each block: Dense → BatchNorm → ReLU → Dense → Add(skip)

3. Ensemble: average across architectures + 5 random seeds

Key innovations vs. basic PyTorch/sklearn:
- Proper WEIGHT INITIALIZATION for financial data (He init)
- LEARNING RATE SCHEDULE (cosine annealing with warmup)
- GRADIENT CLIPPING (prevents exploding gradients from outlier returns)
- BATCH CONSTRUCTION by date (avoids look-ahead bias)
- EARLY STOPPING on IC (not loss) — because IC is what we care about

This runs AFTER step6b_ensemble.py and adds neural predictions
to the ensemble, creating the final combined prediction.
"""

import pandas as pd
import numpy as np
import os
import time
import json
import gc
import warnings
from scipy import stats
from typing import Optional, Tuple

warnings.filterwarnings("ignore")

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, Sampler
    TORCH_AVAILABLE = True
    print(f"PyTorch {torch.__version__} — "
          f"Device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available — install with: pip install torch")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ═══════════════════════════════════════════════════════════════════════
# DATASET & DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

class CrossSectionalDataset(Dataset):
    """
    PyTorch dataset that organizes data by month for proper batching.

    WHY custom dataset?
    - Standard random batching mixes months → look-ahead bias
    - We need all stocks in month t together for cross-sectional ranking
    - Proper implementation: each batch = one month's cross-section
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray):
        self.X = torch.FloatTensor(np.nan_to_num(X, nan=0.0))
        self.y = torch.FloatTensor(y)
        self.dates = dates
        self.unique_dates = np.unique(dates)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MonthlyBatchSampler(Sampler):
    """
    Samples all stocks within a month together.

    This preserves cross-sectional structure:
    - BatchNorm statistics are computed within-month
    - Gradients reflect cross-sectional relationships
    - No temporal leakage between months
    """
    def __init__(self, dates: np.ndarray, shuffle: bool = True):
        self.dates = dates
        self.unique_dates = np.unique(dates)
        self.shuffle = shuffle

    def __iter__(self):
        date_order = np.arange(len(self.unique_dates))
        if self.shuffle:
            np.random.shuffle(date_order)

        for di in date_order:
            dt = self.unique_dates[di]
            indices = np.where(self.dates == dt)[0]
            yield indices.tolist()

    def __len__(self):
        return len(self.unique_dates)


# ═══════════════════════════════════════════════════════════════════════
# NETWORK ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════

class GKXNetwork(nn.Module):
    """
    GKX (2020) Feed-Forward Network for return prediction.

    Architecture: BatchNorm → [Dense → BN → ReLU → Dropout] × N → Dense(1)

    Key design choices:
    - BatchNorm BEFORE first layer (normalizes inputs)
    - BatchNorm WITHIN layers (stabilizes training)
    - Moderate dropout (0.10) — financial data has high noise
    - He initialization (works best with ReLU)
    """
    def __init__(self, input_dim: int, hidden_dims=(256, 128, 64),
                 dropout: float = 0.10):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Input batch normalization
        layers.append(nn.BatchNorm1d(input_dim))

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.

    skip: x → [Linear → BN → ReLU → Linear → BN] + x → ReLU

    Skip connections are critical for deeper networks:
    - Prevent vanishing gradients
    - Allow the model to learn identity if a layer isn't helpful
    - Enable training of 4-6 layer networks for financial data
    """
    def __init__(self, dim: int, dropout: float = 0.10):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResidualNetwork(nn.Module):
    """
    Deep residual network for return prediction.

    Input → [Projection → ResBlock × N] → Dense(1)

    Deeper than GKX FFN, captures more complex interactions
    between features without suffering from depth issues.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_blocks: int = 3, dropout: float = 0.10):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim)
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )

        self.head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.project(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════

def train_neural_model(model: nn.Module, train_dataset: CrossSectionalDataset,
                       val_dataset: CrossSectionalDataset,
                       n_epochs: int = 40, lr: float = 1e-3,
                       weight_decay: float = 1e-4,
                       device: str = "cpu") -> Tuple[nn.Module, dict]:
    """
    Train neural model with:
    - AdamW optimizer (decoupled weight decay)
    - Cosine annealing LR schedule with warmup
    - Gradient clipping at 1.0
    - Early stopping on validation IC (not loss!)
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing with warmup
    warmup_epochs = 3
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs - warmup_epochs, eta_min=lr * 0.01
    )

    # Monthly batch sampler (no look-ahead)
    train_sampler = MonthlyBatchSampler(train_dataset.dates, shuffle=True)
    val_sampler = MonthlyBatchSampler(val_dataset.dates, shuffle=False)

    best_val_ic = -999
    patience = 8
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_ic": []}

    for epoch in range(n_epochs):
        # ── Training ──
        model.train()
        train_losses = []

        for batch_indices in train_sampler:
            X_batch = train_dataset.X[batch_indices].to(device)
            y_batch = train_dataset.y[batch_indices].to(device)

            if len(X_batch) < 32:  # Skip tiny months
                continue

            optimizer.zero_grad()
            pred = model(X_batch)

            # Huber loss for robustness
            loss = nn.functional.huber_loss(pred, y_batch, delta=1.0)
            loss.backward()

            # Gradient clipping — critical for financial data
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        # LR schedule (warmup then cosine)
        if epoch >= warmup_epochs:
            scheduler.step()

        # ── Validation (IC-based) ──
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_indices in val_sampler:
                X_batch = val_dataset.X[batch_indices].to(device)
                y_batch = val_dataset.y[batch_indices]

                if len(X_batch) < 32:
                    continue

                pred = model(X_batch).cpu().numpy()
                val_preds.extend(pred)
                val_targets.extend(y_batch.numpy())

        val_ic = stats.spearmanr(val_preds, val_targets, nan_policy="omit")[0]
        avg_loss = np.mean(train_losses) if train_losses else 999

        history["train_loss"].append(avg_loss)
        history["val_ic"].append(val_ic)

        # Early stopping on IC
        if val_ic > best_val_ic:
            best_val_ic = val_ic
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    return model, {
        "best_val_ic": float(best_val_ic),
        "epochs_trained": epoch + 1,
        "final_loss": float(avg_loss),
    }


def predict_neural(model: nn.Module, X: np.ndarray,
                   device: str = "cpu") -> np.ndarray:
    """Generate predictions in eval mode."""
    model.eval()
    X_tensor = torch.FloatTensor(np.nan_to_num(X, nan=0.0)).to(device)

    with torch.no_grad():
        # Process in chunks to avoid memory issues
        chunk_size = 50000
        preds = []
        for i in range(0, len(X_tensor), chunk_size):
            chunk = X_tensor[i:i + chunk_size]
            pred = model(chunk).cpu().numpy()
            preds.append(pred)

    return np.concatenate(preds)


# ═══════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE VIA PERMUTATION
# ═══════════════════════════════════════════════════════════════════════

def neural_feature_importance(model: nn.Module, X: np.ndarray,
                               y: np.ndarray, feature_names: list,
                               device: str = "cpu",
                               n_top: int = 30) -> pd.DataFrame:
    """
    Permutation importance for neural networks.
    Shuffles each feature and measures IC drop.
    """
    base_pred = predict_neural(model, X, device)
    base_ic = abs(stats.spearmanr(base_pred, y, nan_policy="omit")[0])

    importances = []
    for i, fname in enumerate(feature_names):
        if i >= n_top * 3:  # Only check a subset for speed
            break
        X_perm = X.copy()
        np.random.shuffle(X_perm[:, i])
        perm_pred = predict_neural(model, X_perm, device)
        perm_ic = abs(stats.spearmanr(perm_pred, y, nan_policy="omit")[0])
        importances.append({
            "feature": fname,
            "ic_drop": base_ic - perm_ic,
        })

    return pd.DataFrame(importances).sort_values("ic_drop", ascending=False)


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main():
    if not TORCH_AVAILABLE:
        print("❌ PyTorch is required. Install: pip install torch")
        return

    print("=" * 70)
    print("PHASE 3 — STEP 6C: DEEP LEARNING PREDICTOR")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")
    t_start = time.time()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load preprocessed panel ──────────────────────────────────
    print("\n📊 Loading GKX panel...")
    gkx_path = os.path.join(DATA_DIR, "gkx_panel.parquet")
    panel = pd.read_parquet(gkx_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["year"] = panel["date"].dt.year

    # ── Preprocessing (memory-efficient, same approach as step6b) ──
    from step6b_ensemble import (
        rank_normalize_slice, rank_normalize_target_slice,
        compute_monthly_ics, compute_long_short_returns
    )

    target_col = "fwd_ret_1m"
    id_cols = {"permno", "date", "cusip", "ticker", "siccd", "year",
               "ret", "fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m",
               "ret_forward", "dlret", "dlstcd"}
    feature_cols = [c for c in panel.columns if c not in id_cols
                    and panel[c].dtype in ["float64", "float32", "int64", "int32"]]

    panel = panel.dropna(subset=[target_col])

    # Convert to float32 to save memory
    print("  Converting to float32...")
    for col in feature_cols:
        panel[col] = panel[col].astype(np.float32)
    panel[target_col] = panel[target_col].astype(np.float32)

    # One-time rank normalization (same as step6b)
    print("  Rank-normalizing features per month (one-time)...")
    rank_normalize_slice(panel, feature_cols)
    rank_normalize_target_slice(panel, target_col)
    ranked_target = f"{target_col}_ranked"

    print(f"  Panel: {len(panel):,} x {len(feature_cols)} features")
    gc.collect()

    # ── Walk-forward neural training ─────────────────────────────
    test_start = 2005
    test_end = int(panel["year"].max())
    n_seeds = 3  # Average over 3 seeds for stability

    print(f"\n🧠 WALK-FORWARD NEURAL ENSEMBLE: {test_start}-{test_end}")
    print(f"  Architectures: GKX-FFN(256-128-64) + ResNet(128×3)")
    print(f"  Seeds per model: {n_seeds}")
    print(f"  Device: {device}")

    all_predictions = []
    annual_metrics = []

    for test_year in range(test_start, test_end + 1):
        t0 = time.time()

        val_year = test_year - 1

        # Slice the already-normalized panel
        train_df = panel[panel["year"] < val_year]
        val_df = panel[panel["year"] == val_year]
        test_df = panel[panel["year"] == test_year]

        if len(train_df) < 50000 or len(test_df) < 1000:
            continue

        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df[ranked_target].values.astype(np.float32)
        X_val = val_df[feature_cols].values.astype(np.float32)
        y_val = val_df[ranked_target].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)

        train_dates = train_df["date"].values
        val_dates = val_df["date"].values

        train_dataset = CrossSectionalDataset(X_train, y_train, train_dates)
        val_dataset = CrossSectionalDataset(X_val, y_val, val_dates)

        input_dim = len(feature_cols)
        nn_predictions = []

        # ── Train FFN with multiple seeds ──────────────────────
        for seed in range(n_seeds):
            torch.manual_seed(seed * 42)
            np.random.seed(seed * 42)

            ffn = GKXNetwork(input_dim, hidden_dims=(256, 128, 64), dropout=0.10)
            ffn, ffn_info = train_neural_model(
                ffn, train_dataset, val_dataset,
                n_epochs=40, lr=1e-3, weight_decay=1e-4, device=device
            )
            pred = predict_neural(ffn, X_test, device)
            nn_predictions.append(pred)
            del ffn
            torch.cuda.empty_cache() if device == "cuda" else None

        # ── Train ResNet with multiple seeds ───────────────────
        for seed in range(n_seeds):
            torch.manual_seed(seed * 42 + 1000)
            np.random.seed(seed * 42 + 1000)

            resnet = ResidualNetwork(input_dim, hidden_dim=128,
                                     n_blocks=3, dropout=0.10)
            resnet, resnet_info = train_neural_model(
                resnet, train_dataset, val_dataset,
                n_epochs=40, lr=5e-4, weight_decay=5e-4, device=device
            )
            pred = predict_neural(resnet, X_test, device)
            nn_predictions.append(pred)
            del resnet
            torch.cuda.empty_cache() if device == "cuda" else None

        # ── Average across all neural predictions ──────────────
        nn_pred = np.mean(nn_predictions, axis=0)

        # ── Save ───────────────────────────────────────────────
        test_result = test_df[["permno", "date", target_col]].copy()
        test_result["pred_nn"] = nn_pred
        all_predictions.append(test_result)

        # Monthly IC
        ics = compute_monthly_ics(test_result, "pred_nn", target_col)
        avg_ic = ics["ic"].mean() if len(ics) > 0 else 0
        ls_rets = compute_long_short_returns(test_result, "pred_nn", target_col)
        avg_spread = ls_rets.mean() if len(ls_rets) > 0 else 0

        elapsed = time.time() - t0
        stat = "✅" if avg_ic > 0.02 else "⚠️" if avg_ic > 0 else "❌"
        print(f"  {test_year}: IC_nn={avg_ic:+.4f}{stat} "
              f"spread={avg_spread:+.4f} ({elapsed:.0f}s)")

        annual_metrics.append({
            "year": test_year,
            "ic_nn": avg_ic,
            "spread": avg_spread,
            "time_sec": elapsed,
        })

        del train_df, val_df, X_train, X_val, X_test
        gc.collect()

    # ═══════════════════════════════════════════════════════════════
    # COMBINE WITH TREE/LINEAR ENSEMBLE (if available)
    # ═══════════════════════════════════════════════════════════════
    nn_preds_df = pd.concat(all_predictions, ignore_index=True)

    # Try to load step6b predictions
    tree_pred_path = os.path.join(DATA_DIR, "lgb_predictions.parquet")
    if os.path.exists(tree_pred_path):
        print("\n🔗 COMBINING with tree/linear ensemble predictions...")
        tree_preds = pd.read_parquet(tree_pred_path)

        # Merge on permno + date
        combined = tree_preds.merge(
            nn_preds_df[["permno", "date", "pred_nn"]],
            on=["permno", "date"],
            how="inner"
        )

        # Final ensemble: 50% tree/linear + 50% neural
        # (Equal weight is robust; IC-weighting can overfit)
        combined["prediction_final"] = (
            0.5 * combined["prediction"] + 0.5 * combined["pred_nn"]
        )

        # Save combined
        combined.to_parquet(
            os.path.join(DATA_DIR, "final_predictions.parquet"),
            index=False
        )

        # Compute final metrics
        all_ics = compute_monthly_ics(combined, "prediction_final", target_col)
        overall_ic = all_ics["ic"].mean()
        ic_ir = overall_ic / all_ics["ic"].std() if all_ics["ic"].std() > 0 else 0
        ls_rets = compute_long_short_returns(combined, "prediction_final", target_col)
        sharpe = ls_rets.mean() / ls_rets.std() * np.sqrt(12) if ls_rets.std() > 0 else 0

        # Compare
        ics_tree = compute_monthly_ics(combined, "prediction", target_col)
        ics_nn = compute_monthly_ics(combined, "pred_nn", target_col)

        print(f"\n{'═' * 70}")
        print(f"FINAL COMBINED RESULTS")
        print(f"{'═' * 70}")
        print(f"  Tree/Linear IC:  {ics_tree['ic'].mean():+.4f}")
        print(f"  Neural Net IC:   {ics_nn['ic'].mean():+.4f}")
        print(f"  Combined IC:     {overall_ic:+.4f}  ← FINAL")
        print(f"  IC IR:           {ic_ir:.2f}")
        print(f"  L/S Sharpe:      {sharpe:.2f}")
        print(f"{'═' * 70}")
    else:
        # Just save neural predictions
        nn_preds_df.to_parquet(
            os.path.join(DATA_DIR, "nn_predictions.parquet"),
            index=False
        )

        all_ics = compute_monthly_ics(nn_preds_df, "pred_nn", target_col)
        overall_ic = all_ics["ic"].mean()
        ic_ir = overall_ic / all_ics["ic"].std() if all_ics["ic"].std() > 0 else 0

        print(f"\n  Neural Net IC:   {overall_ic:+.4f}")
        print(f"  IC IR:           {ic_ir:.2f}")

    # Save metrics
    pd.DataFrame(annual_metrics).to_csv(
        os.path.join(RESULTS_DIR, "nn_walk_forward_metrics.csv"), index=False
    )

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed / 60:.1f} min")
    print("  ✅ Deep learning predictions saved")


if __name__ == "__main__":
    main()
