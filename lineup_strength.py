"""
lineup_strength.py - Turn "confirmed lineup vs expected lineup" into ONE
                     logit shift that feeds model_combine's adjustments dict.

THE CHAIN
---------
    confirmed vs expected wOBA per batting slot
      -> change in expected runs (via wOBA scale)
      -> change in win prob, expressed as a logit shift (via runs->logits)

Only changed slots matter: identical slots cancel in the difference, so you
only need accurate wOBAs for the players who actually changed.

ALL DOMAIN ASSUMPTIONS LIVE IN TWO CONSTANTS
--------------------------------------------
WOBA_SCALE     : converts (wOBA - league wOBA) into runs per PA. FanGraphs
                 publishes this yearly (~1.20). Update per season.
LOGITS_PER_RUN : converts a change in a team's expected game runs into a
                 logit shift. Derived from a normal model of run differential
                 (sigma ~ 4 runs, near a pick'em game): d(logit)/d(run) ~ 0.40.
                 It is most accurate near 50/50 and conservative for lopsided
                 games (true sensitivity falls off), which the clamp in
                 model_combine covers. Fit this to your own results.

These are the dials you tune to reality. Everything else is bookkeeping.
"""

from __future__ import annotations

LG_WOBA = 0.320        # league-average wOBA (update per season)
WOBA_SCALE = 1.20      # wOBA scale (update per season)
LOGITS_PER_RUN = 0.40  # runs -> logit shift sensitivity (fit to your data)

# Plate appearances per batting slot for a 9-inning game (sums ~38).
# Top of the order bats more. Tune if your run environment differs.
DEFAULT_SLOT_PA = [4.65, 4.55, 4.45, 4.35, 4.25, 4.15, 4.05, 3.95, 3.85]


def lineup_runs_above_avg(
    wobas: list[float],
    slot_pa: list[float] | None = None,
    lg_woba: float = LG_WOBA,
    woba_scale: float = WOBA_SCALE,
) -> float:
    """
    Expected runs ABOVE league average this lineup produces in one game.
    wobas: 9 starter wOBAs in batting order (slot 1..9).
    """
    if slot_pa is None:
        slot_pa = DEFAULT_SLOT_PA
    if len(wobas) != len(slot_pa):
        raise ValueError(f"expected {len(slot_pa)} wOBAs, got {len(wobas)}")
    return sum(pa * (w - lg_woba) / woba_scale for w, pa in zip(wobas, slot_pa))


def lineup_logit_shift(
    expected_wobas: list[float],
    confirmed_wobas: list[float],
    *,
    logits_per_run: float = LOGITS_PER_RUN,
    lg_woba: float = LG_WOBA,
    woba_scale: float = WOBA_SCALE,
    slot_pa: list[float] | None = None,
) -> tuple[float, dict]:
    """
    Returns (logit_shift, detail). Negative shift => confirmed lineup is
    weaker than expected for this team. Feed logit_shift straight into
    model_combine as adjustments["lineup"].
    """
    exp_runs = lineup_runs_above_avg(expected_wobas, slot_pa, lg_woba, woba_scale)
    con_runs = lineup_runs_above_avg(confirmed_wobas, slot_pa, lg_woba, woba_scale)
    delta_runs = con_runs - exp_runs
    shift = logits_per_run * delta_runs
    return shift, {
        "expected_runs_aa": exp_runs,
        "confirmed_runs_aa": con_runs,
        "delta_runs": delta_runs,
        "logit_shift": shift,
    }


if __name__ == "__main__":
    from model_combine import adjusted_calibrated_prob

    # Echoes the PDF scenario: two expected starters scratched, replaced by
    # weaker bench bats (slots 3 and 5). The other 7 slots are unchanged,
    # so they cancel -- here shown identical for clarity.
    base = [0.330, 0.340, 0.360, 0.345, 0.350, 0.325, 0.315, 0.305, 0.300]
    expected = list(base)
    confirmed = list(base)
    confirmed[2] = 0.290  # slot 3: .360 -> .290 (scratched star)
    confirmed[4] = 0.300  # slot 5: .350 -> .300 (scratched starter)

    shift, d = lineup_logit_shift(expected, confirmed)
    print("=== Lineup impact ===")
    print(f"expected runs (vs avg):  {d['expected_runs_aa']:+.3f}")
    print(f"confirmed runs (vs avg): {d['confirmed_runs_aa']:+.3f}")
    print(f"delta runs:              {d['delta_runs']:+.3f}")
    print(f"-> logit shift:          {d['logit_shift']:+.3f}")

    # Compose with the previous module: base model had this team at 48% raw.
    final, cd = adjusted_calibrated_prob(
        base_raw_prob=0.48,
        adjustments={"lineup": shift},
        calibrator=None,  # wire V2's isotonic here in production
    )
    print("\n=== End-to-end through model_combine ===")
    print(f"base raw prob:   {cd['base_raw_prob']:.4f}")
    print(f"lineup shift:    {cd['components']['lineup']:+.3f}")
    print(f"final prob:      {cd['calibrated_prob']:.4f}  "
          f"({(cd['calibrated_prob'] - cd['base_raw_prob']) * 100:+.1f} pts)")
