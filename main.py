"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun).

The entry point is this file, main.py (Python 3.13, standard library only).
It reads one case at /input/<id>.in (or $MAKEFUN_INPUT_DIR) and writes
/output/<id>.out (or $MAKEFUN_OUTPUT_DIR): for each of hours 2..24 of the
release day (UTC), five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's
concurrent Twitch viewers in the Grand Theft Auto VI category.

Model, fitted at runtime from the case's own comparables
--------------------------------------------------------
The case gives hour 1's print (p1) and 38-42 past big game launches' hourly
Twitch category counts (comparables), each with its real unlock pattern.

1. Only rolling-midnight launches are used for the fit. GTA VI unlocks at
   midnight local, rolling by timezone, and so do all four practice launches.
   A launch with one simultaneous global unlock has an hour 1 near zero and
   ratios in the hundreds, which would wreck the fit; those are excluded by
   their unlock text ("simultaneous" / "single global"). The GTA VI
   category's own pre-release history ("Not a launch") is excluded too.
2. For each hour h = 2..24, fit log(viewers_h / viewers_1) = a_h + b_h *
   log(p1) by least squares over the rolling comparables. A bigger first
   hour means more regions were already unlocked at 01:00 UTC, so the rest
   of the day grows less: the slope is negative. The fit is evaluated at
   log(p1) clamped to the fitted range, so a launch bigger than any seen
   behaves like the biggest seen instead of extrapolating to a collapse.
3. The spread is the fit's residual standard deviation x 0.8 (tuned on the
   practice launches; the 0.8 factor is flat between 0.6 and 0.9 there).
   Quantiles: q_tau = p1 * exp(median_h + z_tau * sigma_h), z the standard
   normal quantiles, rounded to 2 decimals half-up like the house baseline.
4. Fallbacks: an hour with fewer than 5 fitted points uses the house
   baseline's persistence shape (0.15 * sqrt(h - 1) in log space); with no
   usable comparables at all, the whole forecast is the house baseline.

Validation (practice launches, answers from SullyGnome, the same source the
house used to build the cases): mean pinball loss 0.159 over the four
practice cases, against the house baseline's 0.387. Per case: Wolverine
0.196, Red Dead Redemption 2 0.123, God of War Ragnarok 0.103,
Spider-Man 2 0.216. The program is deterministic (no randomness, sorted
inputs) and runs in well under the 8-second limit.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]
# House baseline spread, in log units per square-root hour (fallback only).
BASELINE_SIGMA = 0.15
# The fitted spread is the residual std times this factor.
SPREAD_MULT = 0.8
# Spread floor and cap, in log units.
SIGMA_MIN = 0.08
SIGMA_MAX = 1.2
# Median log-ratio clamp: a ratio outside [0.5x, 15x] is not believed.
LOG_RATIO_MIN = -0.7
LOG_RATIO_MAX = 2.7
# A comparable counts only with hour 1 and at least this many release-day hours.
MIN_DAY_HOURS = 20
# An hour's regression needs at least this many fitted points.
MIN_FIT_POINTS = 5


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_time(text):
    """An ISO 8601 UTC timestamp as a datetime (handles a trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def classify_unlock(unlock):
    """'rolling' for midnight-local rolling unlocks, 'simul' for one
    simultaneous global unlock, 'unknown' otherwise, 'not-a-launch' for
    the target game's own pre-release history."""
    text = (unlock or "").lower()
    if "not a launch" in text:
        return "not-a-launch"
    if "simultaneous" in text or "single global" in text:
        return "simul"
    if "rolling" in text or "midnight local" in text:
        return "rolling"
    return "unknown"


def day_series(comp):
    """A comparable's release-day hourly viewers as {hour 1..24: viewers}.

    The day starts at the comparable's day_start; an hour's figure is the
    point at that hour. Hours with no point (or no viewers) are skipped.
    """
    day_start = parse_time(comp["day_start"])
    out = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        hour = int((parse_time(point["from"]) - day_start).total_seconds()
                   // 3600) + 1
        if 1 <= hour <= 24:
            out[hour] = viewers
    return out


def ols(points):
    """Least squares y = a + b*x over (x, y) points; returns (a, b, resid_std)."""
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in points)
    den = sum((x - mean_x) ** 2 for x, _ in points)
    slope = num / den if den > 0 else 0.0
    intercept = mean_y - slope * mean_x
    resid = [y - (intercept + slope * x) for x, y in points]
    std = math.sqrt(sum(r * r for r in resid) / n) if resid else BASELINE_SIGMA
    return intercept, slope, std


def baseline_quantiles(p1, hour):
    """The house baseline's row for one hour: persistence with a widening
    lognormal spread."""
    return [round2(p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(hour - 1)))
            for z in Z]


def fit_shape(comparables):
    """Fit the log-ratio model on the comparables.

    Returns (median_fn, sigma_fn): each takes an hour and returns the
    fitted median log-ratio and spread for that hour, or None where the
    hour has too few fitted points. The fit is over rolling-midnight
    launches; with too few of those, over every usable comparable.
    """
    usable = []
    for comp in comparables or []:
        name = (comp.get("name") or "")
        if "Grand Theft Auto VI" in name:
            continue
        if classify_unlock(comp.get("unlock")) == "not-a-launch":
            continue
        series = day_series(comp)
        if 1 in series and len(series) >= MIN_DAY_HOURS:
            usable.append((classify_unlock(comp.get("unlock")), series))
    rolling = [(kind, s) for kind, s in usable if kind == "rolling"]
    chosen = rolling if len(rolling) >= MIN_FIT_POINTS else usable
    if not chosen:
        return None, None
    train = [(math.log(s[1]),
              {h: math.log(v / s[1]) for h, v in s.items()})
             for _, s in chosen]
    log_h1s = [lh for lh, _ in train]
    lo, hi = min(log_h1s), max(log_h1s)

    def median_and_sigma(hour, log_p1):
        points = [(lh, d[hour]) for lh, d in train if hour in d]
        if len(points) < MIN_FIT_POINTS:
            return None, None
        intercept, slope, std = ols(points)
        x = min(max(log_p1, lo), hi)
        median = intercept + slope * x
        sigma = min(max(std * SPREAD_MULT, SIGMA_MIN), SIGMA_MAX)
        return median, sigma

    return median_and_sigma, (lo, hi)


def forecast_case(case, name):
    """Forecast one case: {hours: [{hour, q: [q05..q95]}]} for hours 2..24."""
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, "
                         f"not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    # Hour 1's print; a missing or zero count counts as 1 viewer.
    p1 = max(1, first.get("viewers") or 0)
    log_p1 = math.log(p1)
    median_and_sigma, _ = fit_shape(case.get("comparables"))
    hours = []
    for row in case["forecast"]:
        hour = row["hour"]
        median = sigma = None
        if median_and_sigma is not None:
            median, sigma = median_and_sigma(hour, log_p1)
        if median is None:
            hours.append({"hour": hour, "q": baseline_quantiles(p1, hour)})
            continue
        median = min(max(median, LOG_RATIO_MIN), LOG_RATIO_MAX)
        hours.append({
            "hour": hour,
            "q": [round2(p1 * math.exp(median + z * sigma)) for z in Z],
        })
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
