"""GTA VI trophy earn-rate forecaster.

Reads one make.fun case (Exophase reads of the GTA VI PS5 trophy page)
and writes five quantiles per trophy for the case target.

Method, in short:
  * Exophase republishes its counts in steps, so consecutive identical
    reads are one observation, not many.  The history is first
    compressed to the distinct snapshots it actually contains.
  * Each trophy is modelled on the logit scale against log time since
    launch, with a recency-weighted robust linear fit.  A second view
    decomposes the rate into earners and tracked players, extrapolates
    both counts, and recombines them; the two views are averaged in
    logit space, with the count view trusted less when it disagrees
    wildly with the trend view.
  * Extreme extrapolations are shrunk back toward the trophy's own
    last rate and toward what trophies of the same tier are doing, so
    one noisy low-rate trophy cannot run away on its own.
  * The spread is built from the fit residuals, the step size between
    distinct snapshots, and the forecast horizon, and is widened when
    the history shows no refresh at all (a refresh the forecaster has
    not seen is the largest single source of error in that case).

Standard library only, deterministic, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(v):
    if v >= 0:
        return 100.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return 100.0 * e / (1.0 + e)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(moment):
    return max(0.0, (moment - LAUNCH).total_seconds() / 3600.0)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def distinct_snapshots(history):
    """Collapse runs of identical reads into single snapshots."""
    snaps = []
    for read in history:
        key = (read.get("players_tracked"), read.get("trophies_earned"),
               tuple(read.get("rates") or ()))
        if snaps and snaps[-1][0] == key:
            continue
        snaps.append((key, read))
    return [read for _, read in snaps]


def weighted_linear_fit(xs, ys, weights):
    """Weighted least squares y ~ a + b*x; returns (a, b, resid_sd)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return ys[0], 0.0, 0.0
    sw = sum(weights)
    mx = sum(w * x for x, w in zip(xs, weights)) / sw
    my = sum(w * y for y, w in zip(ys, weights)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, weights))
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, weights))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = max(1.0, sw - 2.0)
    var = sum(w * r * r for r, w in zip(resid, weights)) / dof
    return intercept, slope, math.sqrt(max(0.0, var))


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max(0.0, case["horizon_minutes"] / 1440.0)
    horizon_hours = case["horizon_minutes"] / 60.0

    snaps = distinct_snapshots(history)
    last = history[-1]
    n_snaps = len(snaps)

    # Log-time coordinates of every distinct snapshot and of the target.
    snap_times = [parse_time(s["at"]) for s in snaps]
    snap_x = [math.log1p(hours_since_launch(t)) for t in snap_times]
    target_x = math.log1p(hours_since_launch(target))
    # Recency weights in time, not in snapshot count: refreshes arrive
    # irregularly, so a snapshot an hour old and one a day old must not
    # be treated as neighbours.  Half-life of about 36 hours.
    weights = []
    for t in snap_times:
        age_hours = max(0.0, (checkpoint - t).total_seconds() / 3600.0)
        weights.append(0.5 ** (age_hours / 36.0) + 0.05)

    # Player-count extrapolation on log time (used by the count view).
    players = [float(s.get("players_tracked") or 0) for s in snaps]
    player_growth = 0.0
    if n_snaps >= 3 and players[0] > 0 and players[-1] > 0:
        log_players = [math.log(p) for p in players]
        _, player_growth, _ = weighted_linear_fit(snap_x, log_players, weights)
        player_growth = min(1.2, max(-0.2, player_growth))
    players_target = players[-1] if players else 0.0
    if players_target > 0 and player_growth != 0.0:
        players_target = players_target * math.exp(
            player_growth * (target_x - snap_x[-1]))

    # First pass: a trend median and a count median per trophy.
    trend_med = []
    count_med = []
    resid_sds = []
    step_sds = []
    for idx, trophy in enumerate(trophies):
        rates = []
        xs = []
        ws = []
        for j, snap in enumerate(snaps):
            rate = (snap.get("rates") or [None] * len(trophies))[idx]
            if rate is None:
                continue
            rates.append(float(rate))
            xs.append(snap_x[j])
            ws.append(weights[j])
        last_rate = last["rates"][idx]
        if last_rate is None:
            last_rate = rates[-1] if rates else 0.01
        last_rate = float(last_rate)
        base_logit = logit_pct(last_rate)

        if len(rates) >= 3:
            ys = [logit_pct(r) for r in rates]
            intercept, slope, resid_sd = weighted_linear_fit(xs, ys, ws)
            # Damp the slope: a week of launch data does not license an
            # undamped log-time run, and flat trophies keep slope ~ 0.
            slope = min(2.5, max(-2.5, slope))
            raw = intercept + slope * target_x
            # Shrink the extrapolation toward the last observed logit;
            # the longer the horizon relative to the observed span, the
            # stronger the pull back.
            span = max(0.05, xs[-1] - xs[0])
            reach = max(0.0, target_x - xs[-1])
            shrink = span / (span + 1.6 * reach)
            trend = base_logit + shrink * (raw - base_logit)
            steps = [abs(ys[k] - ys[k - 1]) for k in range(1, len(ys))]
            step_sd = median(steps) if steps else 0.0
        elif len(rates) == 2:
            ys = [logit_pct(r) for r in rates]
            move = ys[-1] - ys[0]
            trend = base_logit + 0.35 * move
            resid_sd = abs(move) * 0.5
            step_sd = abs(move)
        else:
            trend = base_logit
            resid_sd = 0.0
            step_sd = 0.0
        trend_med.append(trend)
        resid_sds.append(resid_sd)
        step_sds.append(step_sd)

        # Count view: earners = rate * players; grow earners with the
        # same log-time shape, divide by the extrapolated player count.
        count_view = trend
        if players_target > 0 and players and players[-1] > 0 and len(rates) >= 3:
            earners = []
            exs = []
            ews = []
            for j, snap in enumerate(snaps):
                rate = (snap.get("rates") or [None] * len(trophies))[idx]
                pcount = float(snap.get("players_tracked") or 0)
                if rate is None or pcount <= 0:
                    continue
                earners.append(float(rate) / 100.0 * pcount)
                exs.append(snap_x[j])
                ews.append(weights[j])
            if len(earners) >= 3 and earners[-1] > 0:
                log_e = [math.log(max(0.35, e)) for e in earners]
                a_e, b_e, _ = weighted_linear_fit(exs, log_e, ews)
                b_e = min(3.0, max(-0.5, b_e))
                earners_target = math.exp(a_e + b_e * target_x)
                # Earners cannot exceed the player count, and a trophy
                # already held by everyone stays held by everyone.
                earners_target = min(earners_target, players_target)
                rate_target = 100.0 * earners_target / players_target
                count_view = logit_pct(rate_target)
        count_med.append(count_view)

    # Tier medians of the trend move, for cross-trophy shrinkage.
    tier_moves = {}
    for idx, trophy in enumerate(trophies):
        last_rate = float(last["rates"][idx] or 0.01)
        move = trend_med[idx] - logit_pct(last_rate)
        tier_moves.setdefault(trophy["tier"], []).append(move)
    tier_median_move = {tier: median(moves) for tier, moves in tier_moves.items()}

    rows = []
    for idx, trophy in enumerate(trophies):
        last_rate = float(last["rates"][idx] if last["rates"][idx] is not None else 0.01)
        base_logit = logit_pct(last_rate)

        # Combine the two views; trust the count view less the further
        # it sits from the trend view.
        gap = abs(count_med[idx] - trend_med[idx])
        count_weight = 0.35 / (1.0 + gap)
        centre = (1.0 - count_weight) * trend_med[idx] + count_weight * count_med[idx]

        # Shrink an outlying move toward its tier's median move.
        move = centre - base_logit
        tier_move = tier_median_move.get(trophy["tier"], 0.0)
        if abs(move - tier_move) > 0.9:
            centre = base_logit + tier_move + 0.65 * (move - tier_move)

        # Hard guardrails: pinned trophies stay pinned, the platinum
        # barely moves inside a week, and no median may claim a move
        # of more than about 2.2 logits from its last rate.
        if last_rate >= 99.99:
            centre = base_logit
        elif last_rate <= 0.01:
            centre = min(centre, base_logit + 0.35)
        if n_snaps <= 2 and last_rate < 50.0 and trophy["tier"] != "platinum":
            # No refresh was observed at all.  The next refresh brings
            # in freshly tracked players, who on average hold less than
            # the early sample does, so the unseen-refresh case carries
            # a small dilution shift on top of its wide spread.
            centre -= 0.12
        if trophy["tier"] == "platinum":
            centre = base_logit + 0.3 * (centre - base_logit)
            centre = min(max(centre, base_logit - 0.25), base_logit + 0.45)
        centre = min(max(centre, base_logit - 2.2), base_logit + 2.2)

        # Spread: residuals and snapshot steps set the floor, horizon
        # growth sets the scale, an unseen refresh adds a lump.
        sigma = 0.22 + 0.85 * resid_sds[idx] + 0.55 * step_sds[idx]
        sigma *= math.sqrt(1.0 + horizon_days)
        if n_snaps <= 2:
            # The page never changed inside the observed window: the
            # target almost surely sits after at least one refresh the
            # history gives no size for.
            sigma += 0.85 + 0.25 * math.sqrt(max(0.0, horizon_hours))
        sigma = min(2.6, max(0.18, sigma))
        if last_rate >= 99.99 or last_rate <= 0.01:
            sigma = min(sigma, 0.35)

        qs = []
        for z in Z:
            # Low-rate trophies have more room to run up than down:
            # widen the upper side a little.
            zz = z
            if z > 0 and centre < -2.0:
                zz = z * 1.18
            qs.append(round4(inv_logit_pct(centre + zz * sigma)))
        # Enforce non-decreasing order after rounding.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels") not in (None, LEVELS):
            raise ValueError(f"{path.name} asks for unexpected levels")
        forecast = forecast_case(case)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
