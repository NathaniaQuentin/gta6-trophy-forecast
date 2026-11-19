# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/NathaniaQuentin/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/NathaniaQuentin/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, the percent of the players Exophase tracks for the game who will have earned it one week after launch (2026-11-26T00:00:00Z) — five quantiles per trophy, scored by pinball loss on the logit scale.

## The model: dilution + progression

A trophy's earn rate is earners / players-tracked, and in a launch week the denominator moves fastest: Exophase's tracked population grew 638 → 3,868 players in the ~5 hours after launch night, and past 13,750 by the next afternoon, while most trophies' earners grew far slower. So rates mostly *fall* in the first days even as more people earn each trophy. The program forecasts the numerator and the denominator separately and divides:

- **Players tracked N(t):** log N fitted against log hours since launch (power law, recency-weighted), extrapolated with saturation — growth bends over, it does not run at its steepest pace for the whole week.
- **Earners e_i(t) per trophy:** log earners fitted against log hours since launch, shrunk toward a prior slope for the trophy's kind (story chapters fastest, grind/collect slowest), extrapolated with the same saturation.
- **Structure enforced, not hoped for:** the story chain is ordered (Chapter 1 ≥ Chapter 2 ≥ Chapter 3 ≥ Chapter 4 ≥ final mission) and the platinum is capped at the rarest trophy's rate. The one trophy everyone earns in the intro ("Just Like Paradise": complete Gone Fishing, 100% at launch night) is forecast near 100% in the complementary space.
- **Spread:** logit-normal quantiles whose sigma combines tracked-count uncertainty with earner-growth uncertainty and grows with the horizon — honest, not tight, because on the logit scale a too-narrow forecast pays for every trophy outside it.

No official trophy data is invented. Where the history cannot show a trend, the program says so with a wide spread. All assumptions and sources are documented in `main.py`.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic (two runs write byte-identical output, ~0.13 s per run). Reads every `<id>.in` case file, writes `<id>.out` (`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`).

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
