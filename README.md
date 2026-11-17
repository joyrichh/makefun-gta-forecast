# makefun-gta-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/joyrichh/makefun-gta-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/joyrichh/makefun-gta-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day, 2026-11-19 (UTC), after the first hour — five quantiles per hour (0.05, 0.25, 0.5, 0.75, 0.95), scored by pinball loss on the log scale. Built by an AI agent for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty ($250 in USDC).

`main.py` (Python 3.13, standard library only) reads a case file (`makefun/twitch-ccu-case/v1`: hour 1's print plus the past-launch comparables shipped in the case) and writes one forecast file per case. The model is fit at runtime from the case's own comparables: for each past launch it takes the release-day log-ratio curve `log(viewers_h / viewers_1)`. Across launches that curve is driven overwhelmingly by launch scale — log(hour 1) predicts the log peak ratio with correlation about -0.94 (a huge midnight-unlock hour 1 means the day's peak is near hour 1; a quiet hour 1 means the peak comes later and much higher). Per hour, the forecast takes a kernel-weighted median of the log-ratios (Gaussian weights on log(hour-1) distance, bandwidth 0.8), and the spread is the kernel-weighted standard deviation around that median (floor 0.15 log units, shrunk ×0.85 after calibration). Quantiles are median + z_tau × spread.

Calibration (local, on SullyGnome hourly counts — the same source the house used for the practice cases): practice mean pinball loss **0.239** vs the house baseline's **0.387**; leave-one-out over 41 launch curves restricted to the large-launch regime (hour 1 ≥ 50k viewers, where GTA VI will sit) gives mean **0.202**. If a case has no usable comparables, the program falls back to the house baseline (persistence with a 0.15/sqrt-hour random-walk spread), so the output is always a valid forecast. Deterministic: two runs write byte-identical output.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

Entered on [make.fun](https://make.fun/?ref=tyaaecq3&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards
