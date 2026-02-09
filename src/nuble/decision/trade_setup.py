"""
Trade Setup Calculator — Pure mathematical risk management.

Claude decides: "I'm bullish with moderate conviction on TSLA"
This code computes: entry, stop, targets, position size, R-multiples

Math used:
- ATR-based stop placement (adapts to actual volatility)
- Keltner Channel for dynamic support/resistance
- Fractional Kelly Criterion for position sizing
- Volatility-scaled take profits at defined R-multiples
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeSetupResult:
    """Complete trade setup with full mathematical detail."""
    # Core setup
    direction: str                # "LONG" or "SHORT"
    entry_price: float

    # Stop loss
    stop_loss: float
    stop_distance_dollars: float
    stop_distance_pct: float
    stop_basis: str               # "ATR(14) × multiplier" — shows the math

    # Take profit levels (R-multiples)
    tp1: float                    # 1.5R
    tp1_r_multiple: float
    tp2: float                    # 2.5R
    tp2_r_multiple: float
    tp3: float                    # 4.0R
    tp3_r_multiple: float

    # Position sizing
    position_size_pct: float      # % of portfolio
    position_size_basis: str      # "Fractional Kelly (0.25x)" — shows the math
    risk_per_trade_pct: float     # Actual risk in portfolio %

    # Key levels
    atr_14: float
    atr_pct: float                # ATR as % of price
    daily_volatility: float       # 1-day standard deviation
    annualized_volatility: float  # Annualized vol

    # Keltner Channel
    keltner_upper: float          # SMA20 + 2×ATR (resistance proxy)
    keltner_lower: float          # SMA20 - 2×ATR (support proxy)
    keltner_mid: float            # SMA20

    # Risk/Reward summary
    risk_reward_tp1: float
    risk_reward_tp2: float
    risk_reward_tp3: float

    # Context
    conviction: str               # "high", "moderate", "low"
    notes: List[str]              # Mathematical notes and warnings


class TradeSetupCalculator:
    """
    Pure mathematical trade setup computation.

    Usage:
        calc = TradeSetupCalculator()
        setup = calc.compute(
            direction="LONG",
            conviction="moderate",       # from Claude's analysis
            current_price=245.50,
            closes=np.array([...]),      # 90 days of closes
            highs=np.array([...]),       # 90 days of highs
            lows=np.array([...]),        # 90 days of lows
            portfolio_risk_pct=1.5,      # max risk per trade as % of portfolio
        )
    """

    # ATR stop multipliers by conviction level
    # Higher conviction → tighter stop (more confidence in direction)
    # Lower conviction → wider stop (need more room to be right)
    ATR_STOP_MULTIPLIERS = {
        'high': 1.5,       # 1.5 × ATR — tight, confident
        'moderate': 2.0,   # 2.0 × ATR — standard
        'low': 2.5,        # 2.5 × ATR — wide, uncertain
    }

    # R-multiple targets
    # These are fixed ratios — they define the reward structure
    R_TARGETS = [1.5, 2.5, 4.0]

    # Kelly fraction multiplier by conviction
    # Full Kelly is too aggressive; we use fractional Kelly
    KELLY_FRACTIONS = {
        'high': 0.35,      # 35% Kelly — aggressive but controlled
        'moderate': 0.25,  # 25% Kelly — standard
        'low': 0.15,       # 15% Kelly — conservative
    }

    def compute(
        self,
        direction: str,
        conviction: str,
        current_price: float,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        portfolio_risk_pct: float = 1.5,
        sma_20: Optional[float] = None,
    ) -> Optional[TradeSetupResult]:
        """
        Compute complete trade setup.

        Args:
            direction: "LONG" or "SHORT"
            conviction: "high", "moderate", or "low"
            current_price: Current price
            closes: Array of historical close prices (minimum 20)
            highs: Array of historical high prices
            lows: Array of historical low prices
            portfolio_risk_pct: Maximum risk per trade as % of portfolio (default 1.5%)
            sma_20: SMA(20) value if available (computed if not provided)
        """
        try:
            if len(closes) < 20 or len(highs) < 20 or len(lows) < 20:
                logger.warning("Insufficient data for trade setup (need 20+ bars)")
                return None

            if current_price <= 0:
                return None

            direction = direction.upper()
            conviction = conviction.lower()
            if conviction not in self.ATR_STOP_MULTIPLIERS:
                conviction = 'moderate'

            notes = []

            # ─── ATR Computation ───
            # True Range = max(H-L, |H-Cprev|, |L-Cprev|)
            tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1])
                )
            )
            atr_14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
            atr_pct = (atr_14 / current_price) * 100

            # Wilder's smoothed ATR (more responsive than simple average)
            if len(tr) >= 28:
                wilder_atr = float(tr[-14])
                for i in range(-13, 0):
                    wilder_atr = (wilder_atr * 13 + tr[i]) / 14
                # Use Wilder's if materially different (>10%) from simple
                if abs(wilder_atr - atr_14) / atr_14 > 0.1:
                    notes.append(f"Wilder ATR ({wilder_atr:.2f}) differs from simple ATR ({atr_14:.2f}) by >10% — volatility is changing")
                    atr_14 = wilder_atr
                    atr_pct = (atr_14 / current_price) * 100

            # ─── Volatility ───
            returns = np.diff(closes) / closes[:-1]
            daily_vol = float(np.std(returns[-20:])) if len(returns) >= 20 else float(np.std(returns))
            annual_vol = daily_vol * np.sqrt(252)

            # Volatility regime check
            if len(returns) >= 60:
                vol_recent = float(np.std(returns[-20:]))
                vol_longer = float(np.std(returns[-60:]))
                vol_ratio = vol_recent / vol_longer if vol_longer > 0 else 1.0
                if vol_ratio > 1.5:
                    notes.append(f"⚠️ Volatility expanding: 20d vol is {vol_ratio:.1f}x the 60d vol — stops may need to be wider")
                elif vol_ratio < 0.6:
                    notes.append(f"Volatility compressing: 20d vol is {vol_ratio:.1f}x the 60d vol — potential breakout setup")

            # ─── Keltner Channel ───
            if sma_20 is None:
                sma_20 = float(np.mean(closes[-20:]))
            keltner_upper = sma_20 + 2.0 * atr_14
            keltner_lower = sma_20 - 2.0 * atr_14

            # ─── Stop Loss ───
            atr_multiplier = self.ATR_STOP_MULTIPLIERS[conviction]
            stop_distance = atr_14 * atr_multiplier

            if direction == "LONG":
                stop_loss = current_price - stop_distance
                # Don't place stop below Keltner lower if that's further away
                # (Keltner lower is a natural support level)
                if stop_loss > keltner_lower and keltner_lower < current_price:
                    # Check if Keltner lower is reasonably close
                    keltner_distance = current_price - keltner_lower
                    if keltner_distance < stop_distance * 1.5:
                        # Use Keltner lower as a smarter stop level
                        notes.append(f"Stop adjusted to Keltner lower (${keltner_lower:.2f}) — natural support level")
                        stop_loss = keltner_lower - (atr_14 * 0.1)  # Tiny buffer below support
                        stop_distance = current_price - stop_loss
            else:  # SHORT
                stop_loss = current_price + stop_distance
                if stop_loss < keltner_upper and keltner_upper > current_price:
                    keltner_distance = keltner_upper - current_price
                    if keltner_distance < stop_distance * 1.5:
                        notes.append(f"Stop adjusted to Keltner upper (${keltner_upper:.2f}) — natural resistance level")
                        stop_loss = keltner_upper + (atr_14 * 0.1)
                        stop_distance = stop_loss - current_price

            stop_distance_pct = (stop_distance / current_price) * 100

            # Safety: clamp stop distance to 1%-10% of price
            if stop_distance_pct < 1.0:
                stop_distance_pct = 1.0
                stop_distance = current_price * 0.01
                if direction == "LONG":
                    stop_loss = current_price - stop_distance
                else:
                    stop_loss = current_price + stop_distance
                notes.append("Stop clamped to minimum 1% — ATR-based stop was too tight")
            elif stop_distance_pct > 10.0:
                stop_distance_pct = 10.0
                stop_distance = current_price * 0.10
                if direction == "LONG":
                    stop_loss = current_price - stop_distance
                else:
                    stop_loss = current_price + stop_distance
                notes.append("Stop clamped to maximum 10% — extremely high volatility")

            # ─── Take Profit Levels (R-multiples) ───
            r1, r2, r3 = self.R_TARGETS
            if direction == "LONG":
                tp1 = current_price + (stop_distance * r1)
                tp2 = current_price + (stop_distance * r2)
                tp3 = current_price + (stop_distance * r3)
            else:
                tp1 = current_price - (stop_distance * r1)
                tp2 = current_price - (stop_distance * r2)
                tp3 = current_price - (stop_distance * r3)

            # Check if TP levels are realistic given historical volatility
            # TP3 at 4R in a low-vol stock might take months
            days_to_tp3_estimate = (stop_distance * r3) / (daily_vol * current_price) if (daily_vol * current_price) > 0 else 999
            if days_to_tp3_estimate > 60:
                notes.append(f"TP3 ({r3}R) may take ~{days_to_tp3_estimate:.0f} days at current volatility — consider TP2 as primary target")

            # ─── Position Sizing (Fractional Kelly) ───
            kelly_fraction = self.KELLY_FRACTIONS[conviction]

            # Risk per trade: portfolio_risk_pct scaled by conviction
            # Higher conviction → closer to max risk
            # Lower conviction → fraction of max risk
            conviction_scale = {'high': 1.0, 'moderate': 0.7, 'low': 0.4}
            actual_risk_pct = portfolio_risk_pct * conviction_scale[conviction]

            # Position size = (risk $) / (stop distance $)
            # As % of portfolio: risk_pct / stop_distance_pct
            position_size_pct = actual_risk_pct / (stop_distance_pct / 100) * (1 / 100)
            position_size_pct = position_size_pct * 100  # Convert to percentage

            # Apply Kelly-based cap
            # Kelly: f* = (p × b - q) / b where p=win_rate, q=1-p, b=avg_win/avg_loss
            # Without tracked win rate, we estimate based on conviction
            est_win_rates = {'high': 0.60, 'moderate': 0.52, 'low': 0.48}
            p = est_win_rates[conviction]
            q = 1 - p
            b = r1  # Use TP1 as expected win magnitude
            kelly_optimal = (p * b - q) / b if b > 0 else 0
            kelly_optimal = max(0, kelly_optimal)
            kelly_position = kelly_optimal * kelly_fraction * 100  # As %

            # Take the SMALLER of risk-based and Kelly-based sizing
            position_size_pct = min(position_size_pct, kelly_position) if kelly_position > 0 else position_size_pct

            # Absolute clamps
            position_size_pct = max(0.5, min(8.0, position_size_pct))

            # High volatility reduction
            if annual_vol > 0.50:  # >50% annualized vol
                reduction = min(0.5, (annual_vol - 0.50) * 2)  # Up to 50% reduction
                position_size_pct *= (1 - reduction)
                notes.append(f"Position reduced by {reduction:.0%} for high volatility ({annual_vol:.0%} annualized)")

            position_size_pct = max(0.5, round(position_size_pct, 2))

            sizing_basis = f"Fractional Kelly ({kelly_fraction:.0%}) ∩ Risk-based ({actual_risk_pct:.1f}% risk / {stop_distance_pct:.1f}% stop)"

            # ─── Additional Notes ───
            # Price relative to Keltner
            if direction == "LONG" and current_price > keltner_upper:
                notes.append(f"Price (${current_price:.2f}) is ABOVE Keltner upper (${keltner_upper:.2f}) — extended, higher pullback risk")
            elif direction == "SHORT" and current_price < keltner_lower:
                notes.append(f"Price (${current_price:.2f}) is BELOW Keltner lower (${keltner_lower:.2f}) — oversold, higher squeeze risk")

            # ATR context
            if atr_pct > 5:
                notes.append(f"ATR is {atr_pct:.1f}% of price — very high volatility, wider stops and reduced size appropriate")
            elif atr_pct < 1:
                notes.append(f"ATR is {atr_pct:.2f}% of price — very low volatility, consider this may be a coiled spring")

            return TradeSetupResult(
                direction=direction,
                entry_price=round(current_price, 2),
                stop_loss=round(stop_loss, 2),
                stop_distance_dollars=round(stop_distance, 2),
                stop_distance_pct=round(stop_distance_pct, 2),
                stop_basis=f"ATR(14)=${atr_14:.2f} × {atr_multiplier}x ({conviction} conviction)",
                tp1=round(tp1, 2),
                tp1_r_multiple=r1,
                tp2=round(tp2, 2),
                tp2_r_multiple=r2,
                tp3=round(tp3, 2),
                tp3_r_multiple=r3,
                position_size_pct=position_size_pct,
                position_size_basis=sizing_basis,
                risk_per_trade_pct=round(actual_risk_pct, 2),
                atr_14=round(atr_14, 2),
                atr_pct=round(atr_pct, 2),
                daily_volatility=round(daily_vol, 4),
                annualized_volatility=round(annual_vol, 4),
                keltner_upper=round(keltner_upper, 2),
                keltner_lower=round(keltner_lower, 2),
                keltner_mid=round(sma_20, 2),
                risk_reward_tp1=r1,
                risk_reward_tp2=r2,
                risk_reward_tp3=r3,
                conviction=conviction,
                notes=notes,
            )

        except Exception as e:
            logger.error(f"Trade setup computation failed: {e}")
            return None

    def format_for_brief(self, setup: TradeSetupResult) -> str:
        """Format trade setup for the intelligence brief."""
        lines = []
        lines.append(f"TRADE SETUP ({setup.direction}, {setup.conviction} conviction):")
        lines.append(f"  Entry: ${setup.entry_price:.2f}")
        lines.append(f"  Stop Loss: ${setup.stop_loss:.2f} ({setup.stop_distance_pct:.1f}% | ${setup.stop_distance_dollars:.2f})")
        lines.append(f"    Basis: {setup.stop_basis}")
        lines.append(f"  TP1: ${setup.tp1:.2f} ({setup.tp1_r_multiple}R) | TP2: ${setup.tp2:.2f} ({setup.tp2_r_multiple}R) | TP3: ${setup.tp3:.2f} ({setup.tp3_r_multiple}R)")
        lines.append(f"  Position Size: {setup.position_size_pct:.1f}% of portfolio")
        lines.append(f"    Basis: {setup.position_size_basis}")
        lines.append(f"    Risk per trade: {setup.risk_per_trade_pct:.1f}%")
        lines.append(f"  Keltner Channel: ${setup.keltner_lower:.2f} — ${setup.keltner_mid:.2f} — ${setup.keltner_upper:.2f}")
        lines.append(f"  Volatility: ATR={setup.atr_pct:.1f}% | Daily={setup.daily_volatility:.2%} | Annual={setup.annualized_volatility:.0%}")
        if setup.notes:
            lines.append(f"  Notes:")
            for note in setup.notes:
                lines.append(f"    • {note}")
        return "\n".join(lines)


__all__ = ['TradeSetupCalculator', 'TradeSetupResult']
