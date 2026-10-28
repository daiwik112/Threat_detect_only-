# 05 — WORKFLOW: Interactive Frontend (React + Vite + TypeScript)

**Goal:** a demo-grade dashboard that walks the full flow: network → inject theft → detect → localize → compare → sensor study. Includes a visible, disabled **Mitigation** tab.
**Depends on:** `04` API + `08` contracts. **Port:** 5173. Use the `frontend-design` skill for a distinctive, non-template look (dark control-room theme, clean typography).

## Stack
React 18, Vite, TypeScript, Tailwind, Zustand (state), TanStack Query (API), Plotly.js (charts), React Flow (network diagram), `openapi-typescript` (typed client from `openapi.json`).

## Layout
```
frontend/src/
├─ api/            client.ts, types.ts (generated)
├─ store/          useScenarioStore.ts
├─ components/     NetworkMap, TheftSlider, DetectionCard, ResidualChart,
│                  LocalizationChart, MeasurementTable, CauseBar, ShapBar,
│                  SensorToggle, DisclaimerBanner, AssumptionsDrawer
├─ pages/          Overview, ScenarioLab, Replay, Metrics, SensorStudy, Mitigation
└─ App.tsx         sidebar nav + route + feature flags
```

## Pages
**1. Overview** — corridor diagram (React Flow): substations as nodes along a km axis, metered ones with a sensor icon, unmetered ones dashed; line segments colored by loading %. Assumptions drawer lists everything marked "assumed".

**2. Scenario Lab (main demo)**
- Controls: load-scale slider, theft toggle, **km slider 0–361** (or click on the network map), magnitude slider 0–40 MW, sensor profile select, fault injector (bias/drift/missing/outage), noise multiplier, seed, **Run**.
- Calls `POST /api/analyze`. Shows:
  - Status badge (Normal / Suspicious) + anomaly score gauge.
  - **Measured vs expected** bar/line chart per bus (P, Q, V).
  - Residual chart per section with threshold line.
  - **Localization chart:** x = km, y = probability (ranked taps) + vertical lines for *estimated km* (with interval band) and *true km* (revealed after run). Highlight the suspicious section on the NetworkMap with a pulsing marker.
  - Likely-cause stacked bar (extraction / sensor error / other).
  - SHAP top-contributors horizontal bars.
  - Recommended inspection actions (rule-based text from section + confidence).
- "Reveal truth" button: shows ground-truth vs prediction + error in km.

**3. Replay** — WebSocket `/ws/replay`; play/pause; timeline of scenarios with alert markers; live updating charts.

**4. Metrics** — cards (precision, recall, F1, FPR, MAE km); confusion matrix heatmap; error vs magnitude; error vs true km; AI vs baseline comparison table.

**5. Sensor Study** — table + grouped bars for `major_only / major_plus_mid / dense` × noise; heatmap of "indistinguishable location pairs" (from `sensor_study.json`).

**6. Mitigation (placeholder)** — tab visible with a lock/"Coming soon" badge when `mitigation_enabled=false`. Page shows: planned capabilities (load balancing, redispatch, overload prediction), a disabled strategy picker populated from `GET /api/mitigation/strategies`, and an empty "Plan & impact" panel layout. When flag is true, the page calls `POST /api/mitigation/plan` and renders the response (see 07).

## UX rules
- Persistent `DisclaimerBanner`: "Simulated results — suspected anomalies need field verification."
- Always show uncertainty (interval/probabilities), never a single magic number.
- Responsive; keyboard-accessible; color-blind-safe palette; light/dark.
- Loading/error states for every query; toast on API error.
- Export: "Download result JSON/PNG" on Scenario Lab.

## Tests
- Vitest + Testing Library for stores and components.
- Playwright smoke: open Scenario Lab → set theft 150 km/20 MW → Run → badge shows Suspicious → localization marker rendered.

## Acceptance
- Full demo flow works against the real backend; `npm run build` passes; Lighthouse accessibility ≥ 90.
- Mitigation tab present, gated by `/api/features`.
