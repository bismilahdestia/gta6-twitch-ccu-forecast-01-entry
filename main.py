"""gta6-twitch-ccu-forecast-01 entry: comparable-shape forecaster (v1).

Forecasts GTA VI's launch-day Twitch viewers hour by hour (hours 2..24) from
the case's own comparables: past big console/PC launches' hourly Twitch
curves (SullyGnome hourly category averages), each normalised by its hour-1
print into a log-shape, plus the case's hour-1 print.

Two components, blended 50/50 in log space (geometric mean of quantiles):
1. Regression component: per hour, OLS of log-shape on log(hour-1 print)
   across the pool. Launch curves scale sublinearly with their hour-1 print:
   across the 37 launch comparables, log(p1) vs log(peak ratio) correlates
   at -0.938 -- a launch whose hour 1 is already huge (875k, ELDEN RING)
   peaks at ~1.00x its hour 1, while one whose hour 1 is small (23k,
   Marvel's Wolverine) ramps to ~9x. A fixed median ratio misses both.
   The prediction is clamped to the pool's log(p1) range, so a GTA VI
   hour-1 print above every comparable's stays at the pool's top end.
   Spread = residual std of the fit.
2. Median-shape component: per hour, the median log-shape across the pool
   with an IQR-based sigma. Robust to the pool's extreme shapes.

Pool filter (applied to both components): tier "comparable" only (the
comparable's unlock text carries "Tier: comparable"; early-access-
contaminated and weak launches are excluded), and hour-1 print >= 3000
(a launch whose hour-1 print is below that had not started by hour 1 --
simultaneous unlock later in the day, e.g. Silksong 164 viewers at hour 1
ramping 2120x -- a different regime). The Grand Theft Auto VI category's
own pre-release history is not a launch and is excluded (it has no
release-day window). If fewer than 5 comparables survive, falls back to
the house persistence baseline.

Backtest (local, against SullyGnome answers extracted for the 4 practice
cases + a 37-launch leave-one-out over the comparables):
  practice loss 0.170 (house baseline 0.387), per case 0.12-0.22;
  leave-one-out 0.69 overall, 0.16 on the console-only GTA-like subset.
  The blend beats either component alone on every subset tested.
  Spreads are kept at k=0.6 for both components: at k=0.75 the practice
  90% coverage was 1.000 (over-wide); narrowing to 0.6 improved the
  practice loss on all four cases.

Pure standard library, deterministic (no randomness, pool sorted by name),
no network, no model calls. Runtime is milliseconds per case.
"""

import json
import math
import os
import re
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- model constants (tuned on the practice cases + comparable LOO backtest)
P1_MIN = 3000        # comparables with a smaller hour-1 print are excluded
REG_K = 0.60         # spread multiplier, regression component
MED_K = 0.60         # spread multiplier, median-shape component
BLEND_W = 0.5        # weight of the regression component in the log blend
SIGMA_FLOOR = 0.05   # minimum log-spread per hour
MIN_POOL = 5         # below this, fall back to the persistence baseline
BASE_SIGMA = 0.15    # persistence baseline spread (house starter)


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def percentile(sorted_vals, q):
    """Linear-interpolation percentile of an already sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def release_window(comp):
    """The comparable's release-day 24 hourly viewer counts, or None.

    The window starts at the hour whose `from` equals the comparable's
    day_start; all 24 values must be present and positive.
    """
    day_start = comp.get("day_start")
    if not day_start:
        return None
    hourly = comp.get("hourly") or []
    for i, row in enumerate(hourly):
        if row.get("from") == day_start:
            win = hourly[i:i + 24]
            if len(win) < 24:
                return None
            vals = [r.get("viewers") for r in win]
            if any(v is None or v <= 0 for v in vals):
                return None
            return vals
    return None


def tier_of(comp):
    m = re.search(r"Tier:\s*([a-z-]+)", comp.get("unlock", "") or "")
    return m.group(1) if m else "unknown"


def pool_entries(comparables):
    """(p1, log-shape[23]) for each comparable that passes the pool filter."""
    entries = []
    for comp in sorted(comparables, key=lambda c: c.get("name", "")):
        if comp.get("name", "").startswith("Grand Theft Auto VI"):
            continue  # the target's own pre-release history, not a launch
        if tier_of(comp) != "comparable":
            continue  # early-access-contaminated or weak launches
        win = release_window(comp)
        if win is None:
            continue
        if win[0] < P1_MIN:
            continue  # launch had not started by hour 1: different regime
        p1 = win[0]
        shapes = [math.log(v / p1) for v in win[1:]]
        entries.append((p1, shapes))
    return entries


def ols(xs, ys):
    """Least squares y = a + b*x; returns (a, b, residual std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / max(1, n - 2)
    return a, b, math.sqrt(var)


def regression_hours(p1, entries):
    """Per-hour OLS of log-shape on log(p1); returns {hour: [5 quantiles]}."""
    logp = [math.log(v) for v, _ in entries]
    lo, hi = min(logp), max(logp)
    x = min(max(math.log(max(1, p1)), lo), hi)  # clamp to the pool's range
    out = {}
    for hi_ in range(23):
        ys = [shapes[hi_] for _, shapes in entries]
        a, b, sig = ols(logp, ys)
        sig = max(SIGMA_FLOOR, sig)
        med = a + b * x
        h = hi_ + 2
        out[h] = [p1 * math.exp(med + z * sig * REG_K) for z in Z]
    return out


def median_hours(p1, entries):
    """Per-hour median log-shape + IQR sigma; returns {hour: [5 quantiles]}."""
    out = {}
    for hi_ in range(23):
        col = sorted(shapes[hi_] for _, shapes in entries)
        med = percentile(col, 0.5)
        sig = max(SIGMA_FLOOR, (percentile(col, 0.75) - percentile(col, 0.25)) / 1.349)
        h = hi_ + 2
        out[h] = [p1 * math.exp(med + z * sig * MED_K) for z in Z]
    return out


def persistence_hours(p1):
    """House baseline fallback: every hour stays at the hour-1 print."""
    out = {}
    for h in range(2, 25):
        out[h] = [p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1)) for z in Z]
    return out


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    entries = pool_entries(case.get("comparables") or [])
    if len(entries) < MIN_POOL:
        hours = persistence_hours(p1)
    else:
        reg = regression_hours(p1, entries)
        med = median_hours(p1, entries)
        hours = {}
        for h in range(2, 25):
            hours[h] = [
                math.exp(BLEND_W * math.log(reg[h][i]) + (1 - BLEND_W) * math.log(med[h][i]))
                for i in range(5)
            ]
    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = []
        prev = 0.0
        for q in hours[h]:
            v = min(1e9, max(1.0, round2(q)))
            if v < prev:  # keep quantiles non-decreasing after rounding
                v = prev
            qs.append(v)
            prev = v
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
