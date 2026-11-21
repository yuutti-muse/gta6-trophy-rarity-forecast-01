# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yuutti-muse/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yuutti-muse/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A dependency-free Python forecaster for GTA VI's week-one trophy earn rates. For each of the 55 trophies it reads the launch-window earn-rate history in the case file, compresses it into distinct snapshots (the source refreshes in steps), and fits a recency-weighted robust line to the logit of the earn rate against log-time since launch. The extrapolation is shrunk toward the last observed rate, cross-checked against a second view that extrapolates earner counts and player counts separately, and pulled toward the median move of trophies in the same rarity tier when it disagrees strongly. Pinned trophies (at 0.01% or 99.99%) stay pinned and the platinum's move is damped. Predictive quantiles (5/25/50/75/95) come from a spread that grows with the forecast horizon, the fit's residual scatter and the size of the steps between snapshots, so flat launch-night histories still produce honestly wide intervals for the post-refresh target.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=s1cajwfk&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
