# GTA VI Twitch CCU forecast — comparable-shape forecaster

Entry for `gta6-twitch-ccu-forecast-01` (make.fun): forecasts the Twitch
concurrent-viewer count for Grand Theft Auto VI in each hour 2–24 of its
release day (2026-11-19 UTC), as five quantiles per hour, from the case's
hour-1 print and its comparables.

## Model (`main.py`, pure Python 3.13 stdlib, deterministic, ~0.03 s/case)

For every comparable launch, take its release-day 24 hourly viewer counts
(SullyGnome), normalise by its hour-1 print into a log-shape. Two
components, blended 50/50 in log space (geometric mean of quantiles):

1. **Regression** — per hour, OLS of log-shape on log(hour-1 print).
   Launch curves scale sublinearly with hour 1: across the 37 launch
   comparables, log(p1) vs log(peak ratio) correlates at −0.938. A huge
   hour 1 (875k, ELDEN RING) peaks at ~1.00×; a small one (23k, Marvel's
   Wolverine) ramps to ~9×. Prediction clamped to the pool's log(p1) range.
2. **Median shape** — per hour, median log-shape with IQR-based sigma.

Pool filter: tier "comparable" only (excludes early-access-contaminated and
weak launches) and hour-1 print ≥ 3000 (excludes launches that had not
started by hour 1 — simultaneous unlock later in the day, e.g. Silksong
164 viewers at hour 1 ramping 2120×). GTA VI's own pre-release history is
not a launch and is excluded. Fewer than 5 comparables → persistence
fallback (house baseline).

## Backtest (local)

Practice answers were extracted from SullyGnome hourly averages by the
anchor method (each case's hour-1 print appears exactly once in its game's
release-month series; the 23 values after it are hours 2–24). The house
baseline scored against those answers reproduces its published practice
loss of 0.387, validating the extraction.

| model | practice (4 cases) | LOO over 37 comparables | GTA-like subset |
|---|---|---|---|
| house baseline (starter) | 0.387 | 0.930 | 0.301 |
| this entry (v2) | **0.170** | **0.69** | **0.158** |

Per practice case (v2): Wolverine 0.224, RDR2 0.139, God of War Ragnarök
0.121, Spider-Man 2 0.194. Typical miss 16–46% (baseline 42–189%).

Development notes and the scorer/backtest harness live in `dev/` (not part
of the submitted repo — only `main.py`, `LICENSE` and this README are).
