# 06 — WORKFLOW 7: FULL-STACK INTERACTIVE FRONTEND ⇄ BACKEND

**Goal:** wire every UI control to a live backend call so the whole demo is interactive: click the corridor → inject theft → backend runs pandapower → ML detects/localizes → UI animates the result.
**Extends:** `05` (pages/components) and `04` (API). Where they conflict, THIS file wins.
**Depends on:** `01`–`04` working, `08_DATA_CONTRACTS.md`. Use the `frontend-design` skill for the visual design.

---
## 0. CHANGES TO OTHER FILES (apply these; do NOT edit the other md files)
Claude Code: treat the items below as overrides/additions to the other specs. Where this file conflicts with `00`, `04`, `05`, THIS file wins.

| File | Change |
|---|---|
| `00_README_MASTER.md` | Build-order table: row 6 is now **06_FULLSTACK_INTERACTIVE** (all UI controls wired to the live backend, E2E passing). The sensor-placement study stays inside `03` + the SensorStudy page. Add to the Makefile: `make dev` (api + web hot reload), `make types` (generate TS types), `make e2e` (Playwright). |
| `04_BACKEND_API.md` | Add endpoints: `POST /api/batch-analyze` (max 50), `GET /api/sample-request`, `GET /api/scenarios?tap=&label=&limit=`. Extend `/ws/replay` with client commands `play`, `pause`, `speed`. Every response includes `latency_ms` and `request_id`. Details in section 2 below. |
| `05_FRONTEND_DASHBOARD.md` | Add features: Live mode, Reveal truth, magnitude Sweep (detectability curve), Compare sensor profiles, Run history, keyboard shortcuts. Page/component details in sections 1, 4, 5 below replace the shorter ones in `05`. Build order is section 7 below. |
| `08_DATA_CONTRACTS.md` | Add schemas: `BatchAnalyzeRequest {requests: SimulateRequest[]}`, `BatchAnalyzeResponse {results: AnalyzeResponse[]}`, `AnalyzeResponse {request_id, latency_ms, simulate, detect, localize}`, and the replay message format in section 2. |
| `01`, `02`, `03`, `07` | No changes. |

---
## 1. Interaction map (UI event → API → UI update)

| User action | Frontend | Backend call | UI result |
|---|---|---|---|
| Page load | `useFeatures`, `useNetwork` | `GET /api/features`, `GET /api/network` | Draw corridor, gate Mitigation tab |
| Click a point on corridor / drag km slider | `store.setTheftKm(km)` (snaps to 10 km tap) | — (local) | Ghost marker on map, slider synced |
| Press **Run** | `useAnalyze.mutate(req)` | `POST /api/analyze` | Skeleton loaders → results panels fill in |
| Change load/noise/profile with **Live mode** on | debounce 300 ms → re-run | `POST /api/analyze` (cancel previous via AbortController) | Charts update smoothly |
| **Reveal truth** | local toggle | — (truth already in response, hidden until clicked) | True-km line + error in km |
| Click a localization bar | `store.focusTap(id)` | `GET /api/scenarios?tap=` optional | Map zooms to tap; side panel shows its stats |
| **Replay** play | open WS | `WS /ws/replay` | Live feed, alert timeline |
| Load a known dataset case | picker | `GET /api/scenarios/{id}` then `/api/analyze` | Same view as Run |
| Compare profiles | button | 3× `POST /api/analyze` same seed, different `sensor_profile` | Side-by-side localization charts |
| Sweep magnitude | button | `POST /api/analyze` ×N (1…40 MW) | "Detectability curve": detection prob & km-error vs MW |
| Mitigation tab | `GET /api/mitigation/strategies` | Disabled controls + 501 message | "Coming soon" panel |
| Export | client | — | Download JSON / PNG |

## 2. Extra backend endpoints needed (add to `04`)
- `POST /api/batch-analyze` `{requests:[SimulateRequest]}` → array of results (cap 50) for sweeps/compare.
- `GET /api/scenarios?tap=&label=&limit=` filters.
- `WS /ws/replay` message format:
  ```json
  {"t":12,"scenario_id":"...","measurements":{...},"detect":{...},"localize":{...},"truth":{"label":"theft","km":150}}
  ```
  Client may send `{"cmd":"pause"|"play"|"speed","value":2}`.
- `GET /api/sample-request?kind=theft|normal|sensor_fault` — preset requests for demo buttons.
- All responses include `latency_ms` and `request_id`.

## 3. State management (Zustand `useScenarioStore`)
```ts
{
  request: {loadScale, theftOn, theftKm, theftMw, faults[], sensorProfile, noiseMult, seed},
  result: AnalyzeResponse | null,
  status: 'idle'|'running'|'done'|'error',
  revealTruth: boolean,
  liveMode: boolean,
  focusedTap: string | null,
  history: AnalyzeResponse[]      // last 20 runs, for compare
}
```
Server data via TanStack Query; UI-only state via Zustand. Never store server data in two places.

## 4. Components — detailed specs
1. **CorridorMap (React Flow or custom SVG):** horizontal km axis 0–361; substations as nodes (metered = solid + sensor icon, unmetered = dashed); 10 km tap ticks; segments colored by loading %; click anywhere → set theft km; hover tooltip (km, bus, V, P); animated pulse at estimated location, red hairline at true km (after reveal), shaded band for interval.
2. **ControlPanel:** sliders (load scale, km, MW, noise), selects (profile, fault type), seed + 🎲 randomize, toggles (theft, live mode), **Run** with spinner; presets dropdown ("Big theft near Shimoga", "Small theft", "Sensor bias only", "Line outage").
3. **StatusHero:** big Normal/Suspicious badge, anomaly-score gauge, one-line plain-English summary ("Suspected unauthorized extraction near km 150 ± 20").
4. **MeasuredVsExpected:** Plotly grouped bars/lines per bus for P, Q, V with residual shading.
5. **ResidualChart:** per-section residual with ±threshold band.
6. **LocalizationChart:** tap probabilities (bars) + est line + interval band + truth line.
7. **CausePanel:** stacked bar (extraction/sensor/other) + SHAP top contributors.
8. **ActionsPanel:** rule-based recommended inspection steps ("Patrol section Shimoga–Sira km 110–215, prioritise km 140–160; verify meters at X").
9. **DetectabilityCurve:** from Sweep: detection rate & km error vs MW.
10. **CompareProfiles:** 3 columns, same seed.
11. **ReplayTimeline:** scrubber + alert markers + play/pause/speed.
12. **AssumptionsDrawer + DisclaimerBanner** (always visible).
13. **RunHistory:** list of past runs; click to restore; diff two runs.

## 5. UX behaviors
- Optimistic skeletons, never blank screens; error toast shows backend `error.message`.
- Cancel in-flight requests when a new one starts.
- Animations: marker slides to estimated km (300 ms), gauge eases, no flashing.
- Keyboard: `R` run, `T` reveal truth, `←/→` nudge km ±10.
- Accessibility: ARIA labels on map elements, color-blind-safe palette, focus rings.
- Responsive: map stacks above panels on mobile.
- Dark control-room theme default + light toggle.

## 6. Dev environment ("workstation") for full-stack
- `docker-compose.yml`: `api` (8000), `web` (5173) with Vite proxy `/api` → `api:8000`, `/ws` → WS proxy.
- `.env`: `VITE_API_URL`, `CORS_ORIGINS`.
- `make dev` runs both with hot reload; `make e2e` runs Playwright against compose.
- Generate TS types: `npx openapi-typescript http://localhost:8000/openapi.json -o src/api/types.ts` (run on `make types`).
- Mock server (MSW) so frontend can be built before the backend is finished.

## 7. Build order for Claude Code (do in this order, commit after each)
1. Vite+TS+Tailwind scaffold, theme, layout, routing, DisclaimerBanner.
2. API client + MSW mocks from `08` schemas.
3. CorridorMap with static network data.
4. ControlPanel + store + Run → `/api/analyze` (mock first, then real).
5. StatusHero, MeasuredVsExpected, ResidualChart.
6. LocalizationChart + truth reveal + map pulse.
7. Cause/SHAP/Actions panels.
8. Live mode, cancel, history.
9. Sweep + CompareProfiles (batch endpoint).
10. Replay (WebSocket).
11. Metrics + SensorStudy pages.
12. Mitigation placeholder tab.
13. Polish, a11y, export, E2E tests.

## 8. Tests / acceptance
- **Playwright E2E:** (a) theft 150 km / 20 MW → Run → Suspicious, est km within interval, truth reveal shows error; (b) clicking map sets slider; (c) live mode updates on slider change; (d) replay plays & pauses; (e) mitigation tab disabled with message; (f) backend down → friendly error.
- **Contract test:** generated TS types compile against `openapi.json`.
- **Perf:** Run → results rendered < 1 s locally; no layout shift.
- **Demo script** (`docs/DEMO.md`): 5-minute walkthrough: normal → small theft (hard) → big theft → sensor fault vs theft → sensor-profile compare → detectability curve.

## 9. Definition of done
Every control in §1 triggers the listed backend call and updates the listed UI; Mitigation tab is gated by flag; `make dev` brings up the whole app; E2E suite passes.
