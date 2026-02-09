"""
Triple Barrier Labeling, Sample Weights & Meta-Labeling
=========================================================

Replaces naive binary (up/down next-day) targets with de Prado's
Triple Barrier Method (*Advances in Financial Machine Learning*,
Chapters 3–4).

Problems with naive labeling:
    1. A +0.01% and +5% gain both get label=1 — meaningless.
    2. Stop losses not modeled — a −3% drawdown before recovery
       is labeled "correct."
    3. Fixed horizons ignore volatility — 1% in VIX=12 ≠ 1% in VIX=30.
    4. Adjacent labels overlap → massive temporal correlation →
       inflated cross-validation metrics.

This module provides:
    1. **VolatilityEstimator** — dynamic barrier sizing (EWMA & Parkinson).
    2. **TripleBarrierLabeler** — three barriers (take-profit, stop-loss,
       time expiry); label = which barrier is touched first.
    3. **SampleWeighter** — de Prado Ch.4 uniqueness weights for
       overlapping labels + time decay.
    4. **MetaLabeler** — two-stage framework: primary model picks
       direction, secondary model sizes the bet.
    5. Convenience functions: ``create_labels``, ``create_meta_labels``,
       ``label_distribution_report``.

Dependencies:
    numpy, pandas (no external labeling library — built from scratch).

Author: NUBLE ML Pipeline
Version: 2.0.0
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 1. Volatility Estimator
# ══════════════════════════════════════════════════════════════


class VolatilityEstimator:
    """
    Dynamic volatility estimation for barrier sizing.

    De Prado sizes barriers proportional to recent volatility.
    Key insight: barriers should be **wider** in volatile markets
    (so normal noise doesn't trigger premature exits) and **narrower**
    in calm markets (so real signals are captured before they fade).

    Two estimators:
        - ``daily_volatility``: EWMA of absolute log returns (close-to-close).
        - ``parkinson_volatility``: Parkinson (1980) range-based estimator
          using high/low — 5× more efficient than close-to-close.
    """

    @staticmethod
    def daily_volatility(
        close: pd.Series,
        span: int = 21,
        min_periods: int = 10,
    ) -> pd.Series:
        """
        EWMA standard deviation of log returns.

        This is the default volatility estimator used by
        ``TripleBarrierLabeler`` because it's robust and widely
        understood.

        Formula::

            r_t = ln(C_t / C_{t-1})
            σ_t = EWMA_std(r, span=span)

        Args:
            close:       Close price series with DatetimeIndex.
            span:        EWMA half-life in trading days.
            min_periods: Minimum observations before emitting a value.

        Returns:
            pd.Series of daily volatility estimates, NaN-padded at
            the start where ``min_periods`` hasn't been met.
        """
        log_returns = np.log(close / close.shift(1))
        vol = log_returns.ewm(span=span, min_periods=min_periods).std()
        return vol

    @staticmethod
    def parkinson_volatility(
        high: pd.Series,
        low: pd.Series,
        window: int = 21,
    ) -> pd.Series:
        """
        Parkinson (1980) volatility from high–low range.

        Uses intraday price range — theoretically 5× more efficient
        than close-to-close because it captures intrabar information.

        Formula::

            PV = sqrt( (1 / (4 · n · ln2)) · Σ ln(H_i/L_i)² )

        Implemented as a rolling window over *window* days.

        Args:
            high:   High price series.
            low:    Low price series.
            window: Rolling window length.

        Returns:
            pd.Series of Parkinson volatility estimates.
        """
        log_hl_sq = np.log(high / low) ** 2
        factor = 1.0 / (4.0 * np.log(2))
        return np.sqrt(
            factor * log_hl_sq.rolling(window, min_periods=window).mean()
        )


# ══════════════════════════════════════════════════════════════
# 2. Triple Barrier Labeler  (de Prado Ch.3)
# ══════════════════════════════════════════════════════════════


class TripleBarrierLabeler:
    """
    De Prado's Triple Barrier Method (AFML Chapter 3).

    For each entry point three barriers are erected:

    ┌─────────────────────────────────────────────────────────┐
    │  UPPER  (take-profit)  entry × (1 + tp_mult × σ)       │
    │  ─ ─ ─ ─ ─ ─ ─ entry price ─ ─ ─ ─ ─ ─ ─ ─ ─ ─       │
    │  LOWER  (stop-loss)    entry × (1 − sl_mult × σ)       │
    │  VERTICAL              entry + max_holding_period days  │
    └─────────────────────────────────────────────────────────┘

    Label = which barrier is touched **first**:
        +1  upper barrier (profitable trade)
        −1  lower barrier (losing trade)
         0  vertical barrier (time expired) — or sign(return) if
            ``vertical_label_mode="sign"``

    When a ``side`` Series is provided (from a primary model),
    only the barrier matching the predicted direction is active,
    enabling the **meta-labeling** workflow (Chapter 3.6).
    """

    def __init__(
        self,
        tp_multiplier: float = 2.0,
        sl_multiplier: float = 2.0,
        max_holding_period: int = 10,
        vol_span: int = 21,
        min_return: float = 0.0,
        vertical_label_mode: str = "sign",
    ):
        """
        Args:
            tp_multiplier:       Take-profit distance in daily-vol units.
            sl_multiplier:       Stop-loss distance in daily-vol units.
            max_holding_period:  Vertical barrier in **trading days**.
            vol_span:            EWMA span for ``VolatilityEstimator``.
            min_return:          Minimum absolute return to label as ±1;
                                 returns smaller than this → label = 0.
            vertical_label_mode: ``"sign"`` → vertical hit labelled as
                                 sign(return); ``"zero"`` → labelled as 0.
        """
        if vertical_label_mode not in ("sign", "zero"):
            raise ValueError(
                f"vertical_label_mode must be 'sign' or 'zero', "
                f"got '{vertical_label_mode}'"
            )
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.max_holding_period = max_holding_period
        self.vol_span = vol_span
        self.min_return = min_return
        self.vertical_label_mode = vertical_label_mode

    # ── Public API ─────────────────────────────────────────────

    def compute_barriers(
        self,
        df: pd.DataFrame,
        side: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Compute barrier levels for every observation.

        Barriers are sized by **past** volatility only (no lookahead).
        When *side* is provided, only the barrier matching the
        predicted direction is active (meta-labeling mode).

        Args:
            df:   DataFrame with ``close``, ``high``, ``low`` columns
                  and a DatetimeIndex.
            side: Optional {+1, −1} Series from a primary model.
                  +1 → only upper barrier active (long trade).
                  −1 → only lower barrier active (short trade).

        Returns:
            DataFrame indexed like *df* with columns:
            ``entry_price``, ``upper_barrier``, ``lower_barrier``,
            ``vertical_barrier_idx``, ``daily_vol``.
            Rows where vol is NaN are dropped.
        """
        close = df["close"].astype(np.float64)

        # Volatility computed on PAST data only
        daily_vol = VolatilityEstimator.daily_volatility(
            close, span=self.vol_span
        )

        # Build barriers for every valid observation
        entry_price = close.copy()
        upper = entry_price * (1.0 + self.tp_multiplier * daily_vol)
        lower = entry_price * (1.0 - self.sl_multiplier * daily_vol)

        # When side is provided, disable the opposite barrier
        if side is not None:
            side = side.reindex(df.index)
            # Long trades: no lower barrier (set to -inf so it's never hit)
            upper = upper.where(side != -1, np.inf)
            # Short trades: no upper barrier
            lower = lower.where(side != 1, -np.inf)

        # Vertical barrier: index position + holding period
        idx_positions = np.arange(len(df))
        vertical_idx = np.minimum(
            idx_positions + self.max_holding_period,
            len(df) - 1,
        )

        barriers = pd.DataFrame(
            {
                "entry_price": entry_price,
                "upper_barrier": upper,
                "lower_barrier": lower,
                "vertical_barrier_idx": vertical_idx,
                "daily_vol": daily_vol,
            },
            index=df.index,
        )

        # Drop rows where volatility is unreliable
        barriers = barriers.dropna(subset=["daily_vol"])

        # Drop rows too close to end of series (forward window truncated)
        max_valid = len(df) - self.max_holding_period - 1
        if max_valid > 0:
            barriers = barriers.iloc[: max_valid + 1]

        logger.debug(
            "Barriers computed: %d valid observations (vol_span=%d, holding=%d)",
            len(barriers),
            self.vol_span,
            self.max_holding_period,
        )
        return barriers

    def apply_labels(
        self,
        df: pd.DataFrame,
        barriers: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Determine which barrier was touched first for each entry.

        For each entry at index *i*:
            1. Look at prices from i+1 to i + max_holding_period.
            2. Check if **high** ≥ upper barrier → take-profit touched.
            3. Check if **low** ≤ lower barrier → stop-loss touched.
            4. If both high and low breach on the same day, use
               distance-based tiebreak (closer barrier wins).
            5. If neither horizontal barrier hit, vertical barrier
               fires at the holding period end.

        Returns:
            DataFrame with columns: ``label``, ``touched_barrier``,
            ``holding_period``, ``return_pct``, ``barrier_date``.
        """
        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        index = df.index

        labels: list[int] = []
        touched: list[str] = []
        holdings: list[int] = []
        returns: list[float] = []
        barrier_dates: list[Any] = []

        for row_idx, (dt, bar) in enumerate(barriers.iterrows()):
            entry = bar["entry_price"]
            ub = bar["upper_barrier"]
            lb = bar["lower_barrier"]
            vert_idx = int(bar["vertical_barrier_idx"])

            # Position of this entry in the original DataFrame
            pos = index.get_loc(dt)
            if isinstance(pos, slice):
                pos = pos.start

            tp_day: int | None = None
            sl_day: int | None = None

            # Scan forward day-by-day
            for offset in range(1, vert_idx - pos + 1):
                future_pos = pos + offset
                if future_pos >= len(close):
                    break

                h = high[future_pos]
                l = low[future_pos]

                hit_upper = h >= ub and not np.isinf(ub)
                hit_lower = l <= lb and not np.isinf(lb)

                if hit_upper and hit_lower:
                    # Both breached on same bar — tiebreak by distance
                    dist_up = abs(h - ub)
                    dist_dn = abs(l - lb)
                    if dist_up <= dist_dn:
                        tp_day = offset
                    else:
                        sl_day = offset
                    break
                elif hit_upper:
                    tp_day = offset
                    break
                elif hit_lower:
                    sl_day = offset
                    break

            # Determine outcome
            if tp_day is not None:
                exit_pos = pos + tp_day
                lbl = 1
                barrier_type = "upper"
                hold = tp_day
            elif sl_day is not None:
                exit_pos = pos + sl_day
                lbl = -1
                barrier_type = "lower"
                hold = sl_day
            else:
                # Vertical barrier
                exit_pos = min(vert_idx, len(close) - 1)
                hold = exit_pos - pos
                barrier_type = "vertical"
                ret_at_vert = (close[exit_pos] - entry) / entry
                if self.vertical_label_mode == "sign":
                    if abs(ret_at_vert) < self.min_return:
                        lbl = 0
                    else:
                        lbl = int(np.sign(ret_at_vert))
                else:
                    lbl = 0

            ret_pct = (close[exit_pos] - entry) / entry
            exit_date = index[exit_pos]

            labels.append(lbl)
            touched.append(barrier_type)
            holdings.append(hold)
            returns.append(float(ret_pct))
            barrier_dates.append(exit_date)

        result = pd.DataFrame(
            {
                "label": labels,
                "touched_barrier": touched,
                "holding_period": holdings,
                "return_pct": returns,
                "barrier_date": barrier_dates,
            },
            index=barriers.index,
        )

        logger.debug(
            "Labels applied — distribution: %s",
            result["label"].value_counts().to_dict(),
        )
        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        side: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Full pipeline: compute barriers → apply labels.

        Convenience method combining ``compute_barriers`` and
        ``apply_labels``.  Drops rows where barriers couldn't be
        computed (start of series with unreliable vol, end of series
        where forward window extends beyond available data).

        Args:
            df:   OHLCV DataFrame with DatetimeIndex.
            side: Optional primary-model signal for meta-labeling.

        Returns:
            DataFrame with columns from ``apply_labels`` plus
            ``entry_price`` and ``daily_vol`` from barriers.
        """
        barriers = self.compute_barriers(df, side=side)
        labels = self.apply_labels(df, barriers)

        # Merge barrier info into label DataFrame
        result = labels.join(barriers[["entry_price", "daily_vol"]])
        return result


# ══════════════════════════════════════════════════════════════
# 3. Sample Weighter  (de Prado Ch.4)
# ══════════════════════════════════════════════════════════════


class SampleWeighter:
    """
    De Prado Chapter 4: Sample Uniqueness Weights.

    Adjacent triple-barrier labels overlap in time — e.g. the label
    for day *t* spans [t, t+10] and the label for day *t+1* spans
    [t+1, t+11].  Those 10 shared days mean the two labels are
    **not independent**.

    Standard ML assumes i.i.d. samples → overfitting + inflated
    cross-validation scores.

    Solution: weight each sample by its **average uniqueness** — the
    inverse of the mean number of concurrent labels during its
    active period.  Highly overlapping labels get low weight;
    isolated labels get weight ≈ 1.

    An optional **time decay** factor additionally emphasises
    recent data (de Prado §4.5.2).
    """

    @staticmethod
    def compute_concurrent_labels(
        label_start: pd.DatetimeIndex | pd.Index,
        label_end: pd.DatetimeIndex | pd.Index,
        full_index: pd.DatetimeIndex | pd.Index,
    ) -> pd.Series:
        """
        Count how many labels are active at each time point.

        A label is "active" on every trading day in ``[start, end]``.

        Args:
            label_start: Start dates of each label (entry dates).
            label_end:   End dates of each label (barrier touch dates).
            full_index:  Complete date index of the original DataFrame.

        Returns:
            pd.Series indexed by *full_index* with concurrent-label
            counts.  Days with no active labels have count = 0.
        """
        concurrency = pd.Series(0, index=full_index, dtype=np.int64)

        for start, end in zip(label_start, label_end):
            mask = (full_index >= start) & (full_index <= end)
            concurrency.loc[mask] += 1

        return concurrency

    @staticmethod
    def compute_uniqueness(
        label_start: pd.DatetimeIndex | pd.Index,
        label_end: pd.DatetimeIndex | pd.Index,
        concurrent_count: pd.Series,
    ) -> pd.Series:
        """
        Average uniqueness for each label.

        For label *i* active over [start_i, end_i]::

            u_i = mean( 1 / c_t   for t ∈ [start_i, end_i] )

        where c_t = number of concurrent labels at time *t*.

        Interpretation:
            u ≈ 1.0  →  label period doesn't overlap with others.
            u ≈ 0.1  →  ~10 other labels share the same time window.

        Returns:
            pd.Series of uniqueness values, same index as *label_start*.
        """
        uniqueness_values: list[float] = []

        for start, end in zip(label_start, label_end):
            mask = (concurrent_count.index >= start) & (
                concurrent_count.index <= end
            )
            counts = concurrent_count.loc[mask]
            if len(counts) == 0 or (counts == 0).all():
                uniqueness_values.append(1.0)
            else:
                # Replace zeros to avoid division by zero (shouldn't happen
                # if label is active, but guard defensively)
                safe = counts.replace(0, 1)
                uniqueness_values.append(float((1.0 / safe).mean()))

        return pd.Series(
            uniqueness_values,
            index=label_start,
            name="uniqueness",
        )

    @staticmethod
    def compute_sample_weights(
        uniqueness: pd.Series,
        time_decay: float = 0.5,
    ) -> pd.Series:
        """
        Combine uniqueness with time decay for final sample weights.

        Time decay (de Prado §4.5.2)::

            c = 1 / (1 + time_decay)
            decay_i = c + (1 − c) · (i / n) ^ time_decay

        When ``time_decay=0``: uniform decay → weights = uniqueness.
        When ``time_decay=1``: linear decay → recent samples weighted
        more heavily.

        Weights are normalised so ``sum(weights) ≈ len(weights)``.

        Args:
            uniqueness:  Per-sample uniqueness from
                         ``compute_uniqueness``.
            time_decay:  Decay exponent (0 = none, 1 = strong linear).

        Returns:
            pd.Series of normalised sample weights.
        """
        n = len(uniqueness)
        if n == 0:
            return uniqueness.copy()

        if time_decay <= 0:
            decay = np.ones(n)
        else:
            c = 1.0 / (1.0 + time_decay)
            positions = np.arange(n, dtype=np.float64)
            decay = c + (1.0 - c) * (positions / max(n - 1, 1)) ** time_decay

        weights = uniqueness.values * decay

        # Normalise so sum ≈ n
        total = weights.sum()
        if total > 0:
            weights = weights * (n / total)

        return pd.Series(weights, index=uniqueness.index, name="sample_weight")


# ══════════════════════════════════════════════════════════════
# 4. Meta-Labeler  (de Prado Ch.3.6)
# ══════════════════════════════════════════════════════════════


class MetaLabeler:
    """
    De Prado Chapter 3.6: Meta-Labeling.

    Key insight: it's easier to build a model that **sizes** bets
    than one that **picks direction**.

    Two-stage framework:

    1. **Primary model** — any signal that predicts direction
       (technical rule, sentiment, another ML model).
       Outputs ``side ∈ {+1, −1}``.

    2. **Secondary model** (meta-labeler) — learns *when* the primary
       model's signal will be profitable.
       Output: probability *p* ∈ [0, 1].

    The meta-label is: *was the primary model correct?*

    ┌─────────────────────────────────────────────────────┐
    │  primary = +1, triple_barrier = +1  →  meta = 1    │
    │  primary = +1, triple_barrier = −1  →  meta = 0    │
    │  primary = −1, triple_barrier = −1  →  meta = 1    │
    │  primary = −1, triple_barrier = +1  →  meta = 0    │
    └─────────────────────────────────────────────────────┘

    The meta-model's predicted probability becomes the **bet size**:
        p > 0.5 → take the trade (confirmed).
        p = 0.8 → large position (high confidence).
        p < 0.5 → skip the trade (primary signal likely wrong).

    This dramatically improves precision by filtering false positives.
    """

    def __init__(
        self,
        primary_model: Any = None,
        primary_signal_col: str = "primary_signal",
    ):
        """
        Args:
            primary_model:      Object with ``.predict(X) → {+1, −1}``.
            primary_signal_col: Column name in *df* containing
                                pre-computed primary signals.
        """
        self.primary_model = primary_model
        self.primary_signal_col = primary_signal_col

    def compute_meta_labels(
        self,
        triple_barrier_labels: pd.Series,
        primary_side: pd.Series,
    ) -> pd.Series:
        """
        Meta-label = 1 if primary model was correct, 0 if wrong.

        For vertical-barrier labels that are 0 (ambiguous):
        they are set to NaN and should be excluded from training.

        Args:
            triple_barrier_labels: {−1, 0, +1} from ``TripleBarrierLabeler``.
            primary_side:          {+1, −1} from the primary model.

        Returns:
            pd.Series of {0, 1, NaN}.
        """
        # Align indices
        common = triple_barrier_labels.index.intersection(primary_side.index)
        tbl = triple_barrier_labels.reindex(common)
        ps = primary_side.reindex(common)

        meta = pd.Series(np.nan, index=common, name="meta_label")

        # Where triple barrier label ≠ 0: check agreement
        valid = tbl != 0
        meta.loc[valid] = (tbl.loc[valid] == ps.loc[valid]).astype(int)

        # Where triple barrier label == 0: ambiguous → exclude
        # (meta stays NaN — caller should dropna)

        logger.debug(
            "Meta-labels: correct=%d, wrong=%d, ambiguous=%d",
            int((meta == 1).sum()),
            int((meta == 0).sum()),
            int(meta.isna().sum()),
        )
        return meta

    def get_meta_features(
        self,
        features: pd.DataFrame,
        primary_side: pd.Series,
    ) -> pd.DataFrame:
        """
        Add meta-labeling–specific features.

        These help the meta-model learn *when* the primary model
        is reliable vs. unreliable.

        Features added:
            ``primary_signal``  — the primary model's prediction (+1/−1).
            ``signal_streak``   — consecutive count of identical signals
                                  (long streaks may indicate regime).

        Args:
            features:     Existing feature DataFrame.
            primary_side: Primary model predictions.

        Returns:
            DataFrame with meta-specific columns appended.
        """
        result = features.copy()
        side = primary_side.reindex(result.index).fillna(0)
        result["primary_signal"] = side

        # Signal streak: how many consecutive identical signals
        streak = pd.Series(0, index=result.index, dtype=np.int64)
        current_streak = 0
        prev_signal = None
        for i, idx in enumerate(result.index):
            sig = side.iloc[i] if i < len(side) else 0
            if sig == prev_signal:
                current_streak += 1
            else:
                current_streak = 1
                prev_signal = sig
            streak.iloc[i] = current_streak
        result["signal_streak"] = streak

        return result


# ══════════════════════════════════════════════════════════════
# 5. Module-Level Convenience Functions
# ══════════════════════════════════════════════════════════════


def create_labels(
    df: pd.DataFrame,
    tp_multiplier: float = 2.0,
    sl_multiplier: float = 2.0,
    max_holding_period: int = 10,
    vol_span: int = 21,
    min_return: float = 0.0,
    vertical_label_mode: str = "sign",
    time_decay: float = 0.5,
) -> pd.DataFrame:
    """
    High-level API: create triple-barrier labels with sample weights.

    Combines ``TripleBarrierLabeler`` + ``SampleWeighter`` into a
    single call.

    Args:
        df:                   OHLCV DataFrame with DatetimeIndex.
        tp_multiplier:        Take-profit in daily-vol units (1.5–3.0).
        sl_multiplier:        Stop-loss in daily-vol units (1.5–3.0).
        max_holding_period:   Vertical barrier in trading days (5–21).
        vol_span:             EWMA span for volatility (21–63).
        min_return:           Minimum |return| to label ±1.
        vertical_label_mode:  ``"sign"`` or ``"zero"``.
        time_decay:           Time-decay exponent for sample weights.

    Returns:
        DataFrame with columns:
            ``label``, ``touched_barrier``, ``holding_period``,
            ``return_pct``, ``barrier_date``, ``entry_price``,
            ``daily_vol``, ``sample_weight``.

    Usage::

        labeled = create_labels(df, tp_multiplier=2.0, sl_multiplier=2.0)
        X = features.loc[labeled.index]   # align features
        y = labeled["label"]
        w = labeled["sample_weight"]
    """
    labeler = TripleBarrierLabeler(
        tp_multiplier=tp_multiplier,
        sl_multiplier=sl_multiplier,
        max_holding_period=max_holding_period,
        vol_span=vol_span,
        min_return=min_return,
        vertical_label_mode=vertical_label_mode,
    )
    labeled = labeler.fit_transform(df)

    if labeled.empty:
        labeled["sample_weight"] = pd.Series(dtype=np.float64)
        return labeled

    # Compute sample weights
    label_start = labeled.index
    label_end = pd.DatetimeIndex(labeled["barrier_date"])
    concurrent = SampleWeighter.compute_concurrent_labels(
        label_start, label_end, df.index,
    )
    uniqueness = SampleWeighter.compute_uniqueness(
        label_start, label_end, concurrent,
    )
    labeled["sample_weight"] = SampleWeighter.compute_sample_weights(
        uniqueness, time_decay=time_decay,
    )

    return labeled


def create_meta_labels(
    df: pd.DataFrame,
    primary_side: pd.Series,
    tp_multiplier: float = 2.0,
    sl_multiplier: float = 2.0,
    max_holding_period: int = 10,
    vol_span: int = 21,
    time_decay: float = 0.5,
) -> pd.DataFrame:
    """
    High-level API: create meta-labels for a primary model.

    1. Compute triple-barrier labels using ``primary_side`` for
       one-sided barriers (meta-labeling mode).
    2. Compute meta-labels (was primary model correct?).
    3. Compute sample weights.

    Args:
        df:                  OHLCV DataFrame with DatetimeIndex.
        primary_side:        {+1, −1} Series from the primary model.
        tp_multiplier:       Take-profit in daily-vol units.
        sl_multiplier:       Stop-loss in daily-vol units.
        max_holding_period:  Vertical barrier in trading days.
        vol_span:            EWMA span for volatility.
        time_decay:          Time-decay exponent for sample weights.

    Returns:
        DataFrame with columns:
            ``meta_label``, ``primary_side``, ``triple_barrier_label``,
            ``return_pct``, ``sample_weight``, ``holding_period``,
            ``barrier_date``.
    """
    labeler = TripleBarrierLabeler(
        tp_multiplier=tp_multiplier,
        sl_multiplier=sl_multiplier,
        max_holding_period=max_holding_period,
        vol_span=vol_span,
        vertical_label_mode="sign",
    )

    # One-sided barriers based on primary signal
    labeled = labeler.fit_transform(df, side=primary_side)

    if labeled.empty:
        return pd.DataFrame()

    # Meta-labels
    meta_labeler = MetaLabeler()
    meta = meta_labeler.compute_meta_labels(
        labeled["label"], primary_side.reindex(labeled.index),
    )

    # Sample weights
    label_start = labeled.index
    label_end = pd.DatetimeIndex(labeled["barrier_date"])
    concurrent = SampleWeighter.compute_concurrent_labels(
        label_start, label_end, df.index,
    )
    uniqueness = SampleWeighter.compute_uniqueness(
        label_start, label_end, concurrent,
    )
    weights = SampleWeighter.compute_sample_weights(
        uniqueness, time_decay=time_decay,
    )

    result = pd.DataFrame(
        {
            "meta_label": meta,
            "primary_side": primary_side.reindex(labeled.index),
            "triple_barrier_label": labeled["label"],
            "return_pct": labeled["return_pct"],
            "sample_weight": weights,
            "holding_period": labeled["holding_period"],
            "barrier_date": labeled["barrier_date"],
        },
        index=labeled.index,
    )

    # Drop ambiguous (NaN meta-labels)
    result = result.dropna(subset=["meta_label"])
    result["meta_label"] = result["meta_label"].astype(int)

    return result


def label_distribution_report(labels: pd.DataFrame) -> dict:
    """
    Diagnostic report for label quality.

    Prints a human-readable summary and returns a dict of statistics.

    Expected columns in *labels*:
        ``label``, ``touched_barrier``, ``holding_period``,
        ``return_pct``, ``daily_vol``.
        Optionally ``sample_weight``.

    Returns:
        Dict with ``class_counts``, ``class_ratios``,
        ``avg_holding_period``, ``avg_return_by_label``,
        ``avg_uniqueness``, ``vol_range``, ``barrier_hit_rates``.
    """
    report: dict[str, Any] = {}

    # Class distribution
    if "label" in labels.columns:
        counts = labels["label"].value_counts().to_dict()
        total = len(labels)
        ratios = {k: round(v / total, 4) for k, v in counts.items()} if total > 0 else {}
        report["class_counts"] = counts
        report["class_ratios"] = ratios
    else:
        report["class_counts"] = {}
        report["class_ratios"] = {}

    # Holding period
    if "holding_period" in labels.columns:
        report["avg_holding_period"] = round(
            float(labels["holding_period"].mean()), 2
        )
    else:
        report["avg_holding_period"] = None

    # Return by label class
    if "label" in labels.columns and "return_pct" in labels.columns:
        avg_ret = labels.groupby("label")["return_pct"].mean().to_dict()
        report["avg_return_by_label"] = {
            k: round(float(v), 6) for k, v in avg_ret.items()
        }
    else:
        report["avg_return_by_label"] = {}

    # Sample weight / uniqueness
    if "sample_weight" in labels.columns:
        report["avg_uniqueness"] = round(
            float(labels["sample_weight"].mean()), 4
        )
    else:
        report["avg_uniqueness"] = None

    # Volatility range
    if "daily_vol" in labels.columns:
        vol = labels["daily_vol"].dropna()
        report["vol_range"] = (
            round(float(vol.min()), 6),
            round(float(vol.max()), 6),
        ) if len(vol) > 0 else (None, None)
    else:
        report["vol_range"] = (None, None)

    # Barrier hit rates
    if "touched_barrier" in labels.columns:
        barrier_counts = labels["touched_barrier"].value_counts()
        total = len(labels)
        report["barrier_hit_rates"] = {
            k: round(v / total, 4) for k, v in barrier_counts.items()
        } if total > 0 else {}
    else:
        report["barrier_hit_rates"] = {}

    # Print summary
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION REPORT")
    print("=" * 60)
    print(f"  Total samples:        {len(labels)}")

    if report["class_counts"]:
        print(f"  Class counts:         {report['class_counts']}")
        print(f"  Class ratios:         {report['class_ratios']}")

    if report["avg_holding_period"] is not None:
        print(f"  Avg holding period:   {report['avg_holding_period']} days")

    if report["avg_return_by_label"]:
        print(f"  Avg return by label:  {report['avg_return_by_label']}")

    if report["avg_uniqueness"] is not None:
        print(f"  Avg sample weight:    {report['avg_uniqueness']}")

    if report["vol_range"][0] is not None:
        print(f"  Vol range:            [{report['vol_range'][0]}, {report['vol_range'][1]}]")

    if report["barrier_hit_rates"]:
        print(f"  Barrier hit rates:    {report['barrier_hit_rates']}")

    print("=" * 60 + "\n")

    return report
