"""Forecast Grand Theft Auto VI (PS5) trophy earn rates on Exophase.

Competition: gta6-trophy-rarity-forecast-01 (make.fun, program/v1).
Entrypoint: `python3 /bot/main.py` (Python 3.13, standard library only).

What the program does
---------------------
For every trophy of the game it forecasts five quantiles (0.05, 0.25, 0.5,
0.75, 0.95) of the percent of Exophase-tracked players who will have earned
it at the case's target, from the house's reads of Exophase's trophy page
before the case's checkpoint. Output: /output/<id>.out, one JSON object
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.

The model: dilution + progression, in log space
----------------------------------------------
A trophy's earn rate is earners / players-tracked. Both move, and in a
launch week the denominator moves fastest: Exophase's tracked population
grew 638 -> 3,868 players in the ~5 hours after launch night (the house's
own recording), and past 13,750 by the next afternoon (live page,
2026-11-19 16:15 UTC), while most trophies' earners grew far slower. So
rates mostly FALL in the first days even as more people earn each trophy:
new players sync their profiles faster than they finish chapters.

The program therefore forecasts the numerator and the denominator
separately and divides:

* players_tracked N(t): log N is fitted against log hours since launch
  (a power law; the tracked population grows fast early and decelerates)
  over the case's distinct reads, then read at the target. When the
  history is flat (the practice cases: every read the same page), no
  trend can be fitted, so a launch-week refresh prior is used instead:
  one Exophase refresh roughly triples the tracked count, and a refresh
  is likely inside any launch-week window.
* earners e_i(t) per trophy: log earners fitted against log hours since
  launch, shrunk toward a prior slope for the trophy's kind (story
  chapters progress fastest, grind/collect slowest), because a handful of
  reads cannot pin a slope per trophy on their own. The fitted slope is
  then extrapolated with saturation, not as a straight line: growth
  bends over as the cohort matures, so the extra log growth over a gap
  dx is slope * 1.5 * (1 - exp(-dx / 1.5)). A straight-line log-log
  extrapolation would claim, for example, that Chapter 1's earners keep
  multiplying at their steepest recovery pace for the whole week.
* median rate = 100 * e_i(target) / N(target).

Structure the data itself forces is enforced, not just hoped for:
the story chain is ordered (Chapter 1 >= Chapter 2 >= Chapter 3 >=
Chapter 4 >= final mission), and the platinum is capped at the rarest
trophy's rate (it requires every trophy). The one trophy every player
earns in the intro ("Just Like Paradise": complete Gone Fishing, 100%
of tracked players at launch night, 99.99% the next afternoon) is
forecast in the complementary space, near 100%.

Spread: quantiles come from a logit-normal around the median. Its sigma
combines the uncertainty of the tracked-count forecast with the
uncertainty of each trophy's earner growth, and grows with the horizon
like the house baseline's random walk (0.5 logit per sqrt day) as a
floor. A forecast that is too narrow pays for every trophy outside it
on the logit scale, so the spread is honest, not tight.

Assumptions and sources (all of them are in the code, none hidden)
------------------------------------------------------------------
* GTA VI released 2026-11-19 on PS5 and Xbox Series X|S (case data).
* Launch-night anchor, from the house's recording in the practice
  cases: 638 players tracked, 943 trophies earned, 2026-11-18T23:50Z;
  first refresh at 2026-11-19T04:30Z: 3,868 tracked, every rate but
  the 100% one lower.
* Live Exophase page, 2026-11-19 16:15 UTC: tracked players >= 13,750
  (solved from the printed rates: every rate is consistent with an
  integer earner count only at N >= 13,750, and earner counts cannot
  fall from launch night); "Just Like Paradise" 99.99%.
* Week-one scale (analyst estimates, not official figures): Newzoo
  ~$3.25B+ first-week sales; Wall Street consensus 28-35M copies in
  week one; ~20.8M digital copies sold (pre-orders included) as of
  19 Nov (appgazers tracker). These set the order of magnitude of the
  player base only; the program itself forecasts from the case's
  history, never from these numbers.
* No "official" trophy data is invented: where the history cannot show
  a trend, the program says so with a wide spread instead of a
  confident number.

Determinism: pure functions of the input file, no randomness, no clock,
no network. Two runs write byte-identical output.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import
# model` finds a module shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

_NORM = statistics.NormalDist()


def _z(level: float) -> float:
    return _NORM.inv_cdf(level)


def clamp(x: float, lo: float = 0.01, hi: float = 99.99) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up (as JavaScript's Math.round does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def _parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _linfit(xs, ys, ws=None):
    """Weighted least-squares y = a + b*x. Returns (a, b, resid_sd, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    if ws is None:
        ws = [1.0] * n
    if n == 1:
        return ys[0], 0.0, 0.0, 1
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0, 0.0, n
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    a = my - b * mx
    resid = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw)
    return a, b, resid, n


def _recency_weights(n):
    """Recent reads count up to twice as much: the launch-week trend is
    what the target looks like, not the flat first night."""
    if n <= 1:
        return [1.0] * n
    return [0.5 + 0.5 * j / (n - 1) for j in range(n)]


# --- trophy kinds, by id (Exophase order). Used only as shrinkage priors
# for the earners-growth slope, and for the ordering constraints. --------
_STORY = {3, 4, 5, 6, 7}          # chapter trophies, in story order
_FINAL = 7                        # complete the final mission
_PLATINUM = 1                     # requires every trophy
_FISHING = 2                      # complete Gone Fishing: ~every player
_EASY = {2, 20, 21, 33, 34, 35, 44, 46}   # quick, early or incidental
_GRIND = {14, 15, 17, 25, 30, 31, 32, 40, 42, 52, 53, 54, 55}


def _prior_slope(trophy_id: int) -> float:
    """Prior slope of log earners against log hours since launch."""
    if trophy_id in _STORY:
        return 0.85
    if trophy_id in _EASY:
        return 0.90
    if trophy_id in _GRIND:
        return 0.40
    if trophy_id == _PLATINUM:
        # The first story completions cluster late: almost nobody can
        # finish a 40+ hour story in the first day, then a wave arrives.
        # The history cannot show that wave yet, so the prior slope is
        # steep and the spread stays wide.
        return 0.80
    if trophy_id == 14:  # Attain 100% Completion: same late surge.
        return 0.80
    return 0.65


# Extrapolation damping lengths, in ln(1 + hours) units. Growth
# saturates: a trend fitted on the recent window cannot run at its
# fitted slope for the whole 4.5-day horizon. The effective extra log
# growth over a gap dx is slope * L * (1 - exp(-dx / L)): the trend
# persists for about L e-folds of time, then flattens. L = 1.5 for
# earners (a cohort matures and its growth bends over), 2.0 for the
# tracked population (new buyers keep arriving longer).
_L_EARNERS = 1.5
_L_TRACKED = 2.0


# How much the tracked population is expected to multiply when the
# history shows no refresh yet (practice cases): Exophase refreshes a
# game's counts every few hours, and one refresh roughly triples the
# tracked count (638 -> 3,868 on launch night). The longer the window,
# the likelier a refresh lands inside it, so the multiplier grows with
# the horizon and saturates once a refresh is near-certain.
def _flat_growth_multiplier(horizon_hours: float) -> float:
    p_refresh = min(horizon_hours / 5.0, 0.8)
    return 1.0 + p_refresh * 3.0


# Caps on the extrapolated tracked count, as a multiple of the last read.
_N_CAP = 8.0


def forecast_case(case: dict, name: str) -> dict:
    levels = list(case["levels"])
    zs = [_z(lv) for lv in levels]
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0

    launch = _parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target = _parse_time(case["target"])
    target_hours = max((target - launch).total_seconds() / 3600.0, 0.25)
    x_target = math.log1p(target_hours)

    # ---- distinct reads (identical consecutive pages carry no new info) --
    # A read keeps the time it was LAST seen: between Exophase refreshes
    # the page is frozen, so a repeated vector is one observation whose
    # freshest time is its last occurrence, not its first.
    reads = []
    seen = set()
    for r in reversed(history):
        key = (r["players_tracked"], tuple(r["rates"]))
        if key in seen:
            continue
        seen.add(key)
        reads.append(r)
    reads.reverse()
    # The tracked population only grows; if a read ever shows fewer
    # tracked players than an earlier one (a recount, a stale page),
    # carry the running maximum forward for the earners arithmetic.
    fixed = []
    _n_max = 0.0
    for r in reads:
        _n_max = max(_n_max, float(r["players_tracked"]))
        rr = dict(r)
        rr["players_tracked"] = _n_max
        fixed.append(rr)
    reads = fixed
    n_reads = len(reads)

    # ---- forecast the tracked population N at the target ---------------
    horizon_hours = case["horizon_minutes"] / 60.0
    if n_reads == 0:
        n_last = 1000.0
        n_target = n_last * _flat_growth_multiplier(horizon_hours)
        s_n = 1.0
    else:
        n_last = float(reads[-1]["players_tracked"])
        xs, ys = [], []
        for r in reads:
            h = max((_parse_time(r["at"]) - launch).total_seconds() / 3600.0, 0.05)
            xs.append(math.log1p(h))
            ys.append(math.log(max(float(r["players_tracked"]), 1.0)))
        a_n, b_n, resid_n, _ = _linfit(xs, ys, _recency_weights(len(xs)))
        dx_n = max(x_target - xs[-1], 0.0)
        if n_reads >= 2 and b_n > 0.02:
            # Saturating extrapolation, anchored at the last read.
            n_target = math.exp(ys[-1] + b_n * _L_TRACKED * (1.0 - math.exp(-dx_n / _L_TRACKED)))
        else:
            # Tracked count flat so far: a refresh is coming, not none.
            n_target = n_last * _flat_growth_multiplier(horizon_hours)
        n_target = min(max(n_target, n_last), n_last * _N_CAP)
        # Uncertainty of log N at the target: fit noise + horizon growth.
        s_n = 0.25 + 0.30 * math.sqrt(horizon_days) + (resid_n if n_reads >= 3 else 0.25)
        s_n = min(max(s_n, 0.35), 1.2)

    # ---- forecast earners per trophy ------------------------------------
    medians = {}
    sigmas = {}
    for i, trophy in enumerate(trophies):
        tid = trophy["id"]
        # Last known rate and earners.
        last_rate = None
        for r in reversed(reads):
            v = r["rates"][i]
            if v is not None:
                last_rate = float(v)
                break
        if last_rate is None:
            last_rate = 0.0
        e_last = last_rate / 100.0 * n_last

        # Fit log earners against log hours since launch.
        xs, ys = [], []
        for r in reads:
            v = r["rates"][i]
            if v is None:
                continue
            h = max((_parse_time(r["at"]) - launch).total_seconds() / 3600.0, 0.05)
            e = float(v) / 100.0 * float(r["players_tracked"])
            xs.append(math.log1p(h))
            ys.append(math.log(e + 0.5))
        a_e, b_e, resid_e, n_fit = _linfit(xs, ys, _recency_weights(len(xs)))
        prior = _prior_slope(tid)
        if n_fit >= 2 and xs[-1] > xs[0]:
            w = n_fit / (n_fit + 3.0)  # shrink the fitted slope to the prior
            g = w * b_e + (1.0 - w) * prior
        else:
            g = prior
            resid_e = 0.0
        g = min(max(g, 0.0), 1.6)
        if n_fit >= 1:
            # Saturating extrapolation, re-anchored at the last read: the
            # fitted slope cannot run unchanged for the whole horizon.
            dx = max(x_target - xs[-1], 0.0)
            log_e_target = ys[-1] + g * _L_EARNERS * (1.0 - math.exp(-dx / _L_EARNERS))
        else:
            log_e_target = math.log(e_last + 0.5) + g * (x_target - math.log1p(1.0))
        e_target = math.exp(log_e_target) - 0.5
        e_target = min(max(e_target, e_last), n_target)

        if tid == _FISHING and last_rate >= 99.9:
            # Everyone earns it in the intro; forecast in the
            # complementary space (share of tracked players who have not).
            medians[tid] = 99.9
            sigmas[tid] = 1.0
            continue

        m = 100.0 * e_target / n_target
        medians[tid] = clamp(m)
        # Fit residuals measure model misfit (curvature), not future
        # uncertainty, so they enter at half weight; the total spread is
        # capped: wider than the house baseline's random walk, but a
        # 90% band spanning 90 points of rate is not a forecast.
        s_e = 0.30 + 0.35 * math.sqrt(horizon_days) + (0.5 * resid_e if n_fit >= 3 else 0.15)
        sigma = math.sqrt(s_n * s_n + s_e * s_e)
        sigmas[tid] = min(max(sigma, 0.45), 1.5)

    # ---- structural constraints on the medians --------------------------
    # Story chain in order: each chapter's rate is at least the next one's.
    chain = [3, 4, 5, 6, 7]
    run = None
    for tid in chain:
        if tid not in medians:
            continue
        if run is None or medians[tid] < run:
            run = medians[tid]
        else:
            medians[tid] = run
    # Platinum cannot exceed the final mission's rate.
    if _PLATINUM in medians and _FINAL in medians:
        medians[_PLATINUM] = min(medians[_PLATINUM], medians[_FINAL])

    # ---- quantiles -------------------------------------------------------
    rows = []
    for trophy in trophies:
        tid = trophy["id"]
        m = medians[tid]
        if tid == _FISHING and m >= 99.9:
            # q_tau of the non-earner share u = 100 - rate, lognormal
            # around u = 0.1 with sigma 1.0: a few tracked players who
            # never launch the game pull it down, never up.
            s = sigmas[tid]
            u_med = 0.1
            qs = []
            for z in zs:
                u = u_med * math.exp(z * s)
                qs.append(clamp(100.0 - u, 0.0, 100.0))
            qs = sorted(qs)
        else:
            lg = logit(m)
            s = sigmas[tid]
            qs = [clamp(inv_logit(lg + z * s)) for z in zs]
        # Enforce non-decreasing after rounding, then round to 4 decimals.
        qs = [round4(q) for q in qs]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tid, "q": qs})

    # Platinum capped at the rarest trophy's quantile, per level: it
    # requires every trophy, so it cannot be more common than any of them.
    if _PLATINUM in medians:
        for row in rows:
            if row["id"] == _PLATINUM:
                for k in range(len(levels)):
                    cap = min(r["q"][k] for r in rows if r["id"] != _PLATINUM)
                    row["q"][k] = min(row["q"][k], cap)
                for k in range(1, len(levels)):
                    if row["q"][k] < row["q"][k - 1]:
                        row["q"][k] = row["q"][k - 1]

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clear stale outputs from the directory's contents only: the grader
    # bind-mounts the output directory, so the directory itself must
    # never be removed, only the files inside it.
    for stale in OUTPUT_DIR.glob("*.out"):
        stale.unlink()
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
