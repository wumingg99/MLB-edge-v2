"""
model_combine.py - Fold late information (pitcher/lineup) into the base model
                   WITHOUT breaking V2's isotonic calibration.

THE ONE RULE
------------
V2's probabilities are isotonic-calibrated. Calibration must be the LAST
operation, or you undo it. So the pipeline is always:

    raw model prob  ->  logit  ->  + adjustments (logit units)  ->  prob
                    ->  isotonic calibrate  ->  REPORT THIS

Adjusting the already-calibrated number de-calibrates it. Adjusting the raw
number and calibrating last keeps "confidence == winrate" intact.

WHY LOGIT SPACE
---------------
Adding probabilities can push past 0 or 1 and treats effects as linearly
additive. Adding in log-odds (logit) space stays bounded and composes
correctly. A pitcher or lineup signal becomes a logit shift; positive shift
makes the modeled outcome more likely.

CONSERVATISM
------------
`max_abs_logit` clamps the TOTAL shift. This is the single most important
dial: it stops one noisy lineup read from swinging the number. Tune it once
you have graded data. Default 0.5 logit ~= a 12-point swing at a 50% base.
"""

from __future__ import annotations
import math


def prob_to_logit(p: float, eps: float = 1e-9) -> float:
    """Probability -> log-odds. Clipped to avoid infinities at 0/1."""
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def logit_to_prob(z: float) -> float:
    """Log-odds -> probability. Numerically stable sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def combine_adjustments(
    base_raw_prob: float,
    adjustments: dict[str, float],
    max_abs_logit: float = 0.5,
) -> tuple[float, dict]:
    """
    Apply logit-space adjustments to the RAW (pre-calibration) probability.

    base_raw_prob : the model's raw output for the modeled outcome,
                    BEFORE isotonic calibration.
    adjustments   : {name: logit_shift}. Positive => outcome more likely.
                    e.g. {"lineup": -0.30, "pitcher": +0.10}
    max_abs_logit : clamp on the SUMMED shift (conservatism guard).

    Returns (adjusted_raw_prob, detail). The adjusted prob is still RAW --
    you must calibrate it afterward (see calibrate / adjusted_calibrated_prob).
    """
    base_logit = prob_to_logit(base_raw_prob)
    total_shift = sum(adjustments.values())
    clamped = max(min(total_shift, max_abs_logit), -max_abs_logit)
    adjusted_logit = base_logit + clamped
    detail = {
        "base_raw_prob": base_raw_prob,
        "base_logit": base_logit,
        "raw_total_shift": total_shift,
        "clamped_shift": clamped,
        "clamp_hit": abs(total_shift) > max_abs_logit,
        "adjusted_logit": adjusted_logit,
        "adjusted_raw_prob": logit_to_prob(adjusted_logit),
        "components": dict(adjustments),
    }
    return detail["adjusted_raw_prob"], detail


def calibrate(raw_prob: float, calibrator=None) -> float:
    """
    The LAST step. `calibrator` is a callable raw_prob -> calibrated_prob,
    i.e. V2's fitted isotonic model wrapped to take/return a scalar:

        cal = lambda p: float(iso_reg.predict([p])[0])

    If None, returns raw_prob unchanged so you can wire calibration in later.

    NOTE: the isotonic map was fit on the BASE model's raw outputs. Modest
    adjustments keep the adjusted value in-distribution, which is why the
    clamp matters. If you ever widen the clamp a lot, plan to re-fit
    calibration on the ADJUSTED predictions.
    """
    if calibrator is None:
        return raw_prob
    return float(calibrator(raw_prob))


def adjusted_calibrated_prob(
    base_raw_prob: float,
    adjustments: dict[str, float],
    calibrator=None,
    max_abs_logit: float = 0.5,
) -> tuple[float, dict]:
    """Full pipeline: raw -> logit-adjust -> calibrate. Returns (final, detail)."""
    adj_raw, detail = combine_adjustments(base_raw_prob, adjustments, max_abs_logit)
    final = calibrate(adj_raw, calibrator)
    detail["calibrated_prob"] = final
    return final, detail


if __name__ == "__main__":
    # Toy calibrator standing in for V2's fitted isotonic regression.
    # (Real one: wrap your fitted IsotonicRegression.predict.)
    # This toy shrinks 15% toward 0.5 to show calibration is applied LAST.
    toy_cal = lambda p: 0.5 + (p - 0.5) * 0.85

    print("=== Normal case: lineup weaker than expected, small pitcher bump ===")
    final, d = adjusted_calibrated_prob(
        base_raw_prob=0.55,
        adjustments={"lineup": -0.30, "pitcher": +0.10},
        calibrator=toy_cal,
    )
    print(f"base raw prob:        {d['base_raw_prob']:.4f}")
    print(f"summed logit shift:   {d['raw_total_shift']:+.3f}  "
          f"(clamp hit: {d['clamp_hit']})")
    print(f"adjusted RAW prob:    {d['adjusted_raw_prob']:.4f}")
    print(f"calibrated (report):  {d['calibrated_prob']:.4f}")

    print("\n=== Clamp protecting against a wild swing ===")
    final, d = adjusted_calibrated_prob(
        base_raw_prob=0.55,
        adjustments={"lineup": -1.10, "pitcher": -0.10},  # summed -1.20
        calibrator=toy_cal,
        max_abs_logit=0.5,
    )
    print(f"base raw prob:        {d['base_raw_prob']:.4f}")
    print(f"summed logit shift:   {d['raw_total_shift']:+.3f}  "
          f"(clamp hit: {d['clamp_hit']})")
    print(f"clamped to:           {d['clamped_shift']:+.3f}")
    print(f"adjusted RAW prob:    {d['adjusted_raw_prob']:.4f}")
    print(f"calibrated (report):  {d['calibrated_prob']:.4f}")

    print("\n=== No calibrator wired yet (identity) ===")
    final, d = adjusted_calibrated_prob(
        base_raw_prob=0.55,
        adjustments={"lineup": -0.30, "pitcher": +0.10},
        calibrator=None,
    )
    print(f"adjusted RAW prob:    {d['adjusted_raw_prob']:.4f}")
    print(f"calibrated (report):  {d['calibrated_prob']:.4f}  (== raw, identity)")
