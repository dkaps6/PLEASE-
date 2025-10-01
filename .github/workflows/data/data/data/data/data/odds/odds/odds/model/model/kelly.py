def kelly_fraction(p: float, american_odds: float) -> float:
    if american_odds > 0: b = american_odds/100.0
    else: b = 100.0/abs(american_odds)
    q = 1.0 - p
    f = (b*p - q) / b
    return max(0.0, f)
