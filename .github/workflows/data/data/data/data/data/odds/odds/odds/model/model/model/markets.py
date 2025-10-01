import numpy as np
from scipy.stats import norm

def weather_penalty(temp_f: float, wind_mph: float, cfg) -> float:
    wth = cfg["volatility"]["wind_threshold_mph"]
    widen = 1.0
    if wind_mph is not None and wind_mph >= wth:
        widen *= 1.0 + cfg["volatility"]["wind_sd_widen"]
    if temp_f is not None and temp_f <= cfg["volatility"]["cold_threshold_f"]:
        widen *= 1.0 + cfg["volatility"]["cold_sd_widen"]
    return widen

def widen_for_pressure(is_top10_pressure: bool, sd: float, cfg) -> float:
    return sd * (1.0 + (cfg["volatility"]["widen_sd_if_pressure_top10"] if is_top10_pressure else 0.0))

def widen_for_low_qb(low_tier_qb: bool, sd: float, cfg) -> float:
    return sd * (1.0 + (cfg["volatility"]["widen_sd_if_qb_lowtier"] if low_tier_qb else 0.0))

def yardage_probs(mean, sd, line):
    z = (line - mean) / max(sd, 1e-6)
    p_under = float(norm.cdf(z))
    p_over  = 1.0 - p_under
    return p_over, p_under
