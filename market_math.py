"""
market_math.py - Market probability utilities for MLB Edge model_v3.

Pipeline for a single market:
    prices (American, ALL sides) --devig--> no-vig market probabilities
    model_prob vs no-vig market prob --> edge (your true signal, in points)
    model_prob + your actual price  --> expected value per unit staked

Why this module exists:
    A single price already contains the bookmaker's margin (the "vig").
    You can NEVER read a fair probability off one side. To get the fair
    number you need every side of the market and normalize to sum to 1.0.
    Comparing your model to a one-sided implied prob, or computing EV at
    the fair price instead of the price you actually get, both inflate
    your apparent edge -- which is what wrecks calibration.
"""

from __future__ import annotations


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def implied_prob(american: float) -> float:
    """Raw implied probability from ONE American price. Still includes vig."""
    return 1.0 / american_to_decimal(american)


def devig_proportional(prices: list[float]) -> tuple[list[float], float]:
    """
    Remove the bookmaker margin by proportional (multiplicative) normalization.

    prices: American odds for EVERY outcome of the SAME market
            (e.g. both sides of a run line, or the over and under of a total).
    Returns (no_vig_probs, overround):
        no_vig_probs -- list of fair probabilities, sums to 1.0
        overround    -- the margin, e.g. 1.035 means a 3.5% book

    Proportional is the standard, robust default. For markets with extreme
    longshots (favorite-longshot bias), the Shin method is a documented
    upgrade -- ask and I'll add a verified implementation.
    """
    raw = [implied_prob(p) for p in prices]
    overround = sum(raw)
    return [r / overround for r in raw], overround


def edge(model_prob: float, novig_prob: float) -> float:
    """
    Model's disagreement with the FAIR market, in probability points.
    Positive => model thinks the outcome is likelier than the fair market.
    This (not raw EV off a vigged price) is your clean signal.
    """
    return model_prob - novig_prob


def expected_value(model_prob: float, price_american: float) -> float:
    """
    EV per 1 unit staked at the price you ACTUALLY get, using your model prob.
    +0.08 means +8% expected return on stake. This is the realistic number
    -- always lower than `model/novig - 1`, by roughly the vig.
    """
    dec = american_to_decimal(price_american)
    return model_prob * (dec - 1.0) - (1.0 - model_prob)


if __name__ == "__main__":
    # Worked example from the V2 plan: Dodgers -1.5 +125, model says 48%.
    # The PDF only lists one side; the other side is needed to de-vig.
    # Assume the book also offers Mets +1.5 at -145.
    dodgers_price = 125
    mets_price = -145
    model_prob = 0.48

    novig, overround = devig_proportional([dodgers_price, mets_price])
    dodgers_fair, mets_fair = novig

    print(f"Raw implied (Dodgers +125):   {implied_prob(dodgers_price):.4f}")
    print(f"Raw implied (Mets -145):      {implied_prob(mets_price):.4f}")
    print(f"Overround (book margin):      {overround:.4f}  "
          f"({(overround - 1) * 100:.2f}% vig)")
    print(f"NO-VIG fair prob (Dodgers):   {dodgers_fair:.4f}")
    print(f"NO-VIG fair prob (Mets):      {mets_fair:.4f}")
    print(f"Model prob:                   {model_prob:.4f}")
    print(f"Edge vs fair market:          {edge(model_prob, dodgers_fair) * 100:+.1f} points")
    print(f"Realistic EV at +125:         {expected_value(model_prob, dodgers_price) * 100:+.1f}%")
    print(f"PDF-style EV (model/fair - 1):"
          f" {(model_prob / dodgers_fair - 1) * 100:+.1f}%  <- optimistic, fair-odds only")
