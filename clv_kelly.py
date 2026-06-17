"""
clv_kelly.py - Two things you do AROUND a bet, both graded with honest numbers.

CLV (Closing Line Value)  -- EVALUATION. Did you beat the market's sharpest
                             number (the close)? This is the best leading
                             indicator of long-run edge, far better than W/L
                             or ROI at small samples.
KELLY                     -- SIZING. How much to stake given a calibrated edge.

Both reuse market_math, so the honest-pricing fixes flow straight through.

CLV is just: your bet's EV graded against the CLOSING no-vig probability
             instead of against your model.
KELLY is just: that edge divided by the net odds.

** KELLY STAYS DORMANT UNTIL CALIBRATION IS VALIDATED. **
Keep flat-betting until your calibrated probabilities are confirmed over 100+
graded predictions. Then turn this on as FRACTIONAL Kelly (quarter), fed the
CALIBRATED probability and the realistic price -- never the inflated EV.
Sizing up on confidence you have not verified turns a calibration problem
into a bankroll problem.
"""

from __future__ import annotations
from market_math import american_to_decimal, devig_proportional, expected_value


# ----------------------------- CLV -----------------------------------------

def closing_no_vig(close_prices: list[float], side_index: int) -> float:
    """No-vig probability of your backed side from the CLOSING prices.
    close_prices: American odds for every side at close; side_index = your side.
    """
    novig, _ = devig_proportional(close_prices)
    return novig[side_index]


def clv_points(bet_no_vig: float, close_no_vig: float) -> float:
    """Probability points the line moved toward your side. Positive = good:
    the market came to agree with you after you got your number."""
    return close_no_vig - bet_no_vig


def closing_line_ev(your_bet_price_american: float, close_no_vig: float) -> float:
    """Your bet's EV graded against the market's CLOSING fair probability.
    This is the headline CLV metric. Average it across all bets -- a positive
    average is the strongest evidence you are a real long-term winner, even
    while game-outcome variance is still hiding it."""
    return expected_value(close_no_vig, your_bet_price_american)


# ----------------------------- KELLY ----------------------------------------

def kelly_fraction(
    prob: float,
    price_american: float,
    kelly_multiplier: float = 0.25,
    max_fraction: float = 0.05,
) -> float:
    """
    Fraction of bankroll to stake. Use the CALIBRATED prob and the real price.
    full Kelly = edge / net_odds = (p*d - 1) / (d - 1).
    Returns 0 when there is no edge. Capped at max_fraction as a hard guard
    against a single overconfident estimate.
    """
    d = american_to_decimal(price_american)
    net = d - 1.0
    full_kelly = (prob * d - 1.0) / net
    if full_kelly <= 0:
        return 0.0
    return min(kelly_multiplier * full_kelly, max_fraction)


def kelly_stake(
    bankroll: float,
    prob: float,
    price_american: float,
    kelly_multiplier: float = 0.25,
    max_fraction: float = 0.05,
) -> float:
    """Concrete stake in currency units."""
    return bankroll * kelly_fraction(prob, price_american, kelly_multiplier, max_fraction)


if __name__ == "__main__":
    # --- CLV: you bet Dodgers +125 when fair was 42.9% (from market_math). ---
    bet_no_vig = 0.429
    your_price = 125
    # Line at close: Dodgers -105 / Mets -115 (market moved toward the Dodgers).
    close = closing_no_vig([-105, -115], side_index=0)
    print("=== CLV ===")
    print(f"fair prob when you bet:  {bet_no_vig:.4f}")
    print(f"fair prob at close:      {close:.4f}")
    print(f"CLV (points line moved): {clv_points(bet_no_vig, close) * 100:+.1f} pts")
    print(f"closing-line EV of bet:  {closing_line_ev(your_price, close) * 100:+.1f}%"
          f"  (the close says your +125 was this good)")

    # --- KELLY: dormant illustration. Calibrated 48% at +125, $1000 roll. ---
    print("\n=== KELLY (illustration only - keep OFF until calibrated) ===")
    p_cal, price, roll = 0.48, 125, 1000
    f = kelly_fraction(p_cal, price)
    print(f"quarter-Kelly fraction:  {f * 100:.2f}% of bankroll")
    print(f"stake on ${roll:.0f}:          ${kelly_stake(roll, p_cal, price):.2f}")

    print("\n=== KELLY with no edge -> no bet ===")
    p_noedge = 0.42  # below the +125 breakeven of 44.4%
    print(f"fraction at p=0.42:      {kelly_fraction(p_noedge, price) * 100:.2f}%  (correctly zero)")
