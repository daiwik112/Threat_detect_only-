# 5-minute demo script

Simulation-only research demo. Every result is a *suspected* anomaly that needs field verification.

**Setup:** `make dev` (or `make api` + `make web`), then open http://localhost:5173.

| # | Time | Do | Say / point at |
|---|---|---|---|
| 1 | 0:00 | **Overview** page | 361 km Sharavathi → Peenya corridor (assumed model). Metered substations (◉) and the unmetered Sira junction (dashed). Open **◆ Assumptions**. |
| 2 | 0:40 | **Scenario Lab** → Preset **Normal operation** → **Run** (`R`) | Status *Normal*, residuals below the dashed alarm line. |
| 3 | 1:10 | Preset **Small theft (near the noise floor)** → Run | 3 MW is close to the sensor noise, so the score may stay low or the interval is wide. Be honest about detectability limits. |
| 4 | 1:40 | Preset **Big theft near Shimoga** → Run → **Reveal truth** (`T`) | Suspicious, likely cause *extraction*, estimated km with 80% band and pulsing map marker; truth line and km error. |
| 5 | 2:20 | **Ask the advisor** | RAG advisor: cited guidance (patrol planning, pre-dispatch checks); click `[n]` to see the source passage. Recommendation-only. |
| 6 | 2:50 | Preset **Sensor bias only** → Run | Likely cause flips to *sensor error*; KCL consistency features in SHAP; advice says verify meters before any patrol. |
| 7 | 3:20 | Tools → **Compare profiles** | Same seed under major-only / major+mid / dense metering. |
| 8 | 3:50 | Tools → **Detectability sweep** | Detection rate and km error vs MW (1–40 MW × 3 seeds): where the method works and where it doesn't. |
| 9 | 4:20 | Tools → **Similar past cases** | Vector DB: nearest labelled scenarios (41-d physics-feature embedding). |
| 10 | 4:40 | **Metrics** page; **Live replay** (Connect & play) | Held-out metrics vs baseline; live stream with alert markers and scrubber. Mitigation tab is locked ("Coming soon"). |

Keyboard: `R` run · `T` reveal truth · `←/→` move the injection ±10 km · **Live mode** re-runs on every change.
