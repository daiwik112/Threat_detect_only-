/**
 * MSW mock API (06 §6) — lets the UI run without the backend: `VITE_USE_MOCKS=true npm run dev`.
 * Responses follow the generated schema types; the analyze payload is a fixed example, so every
 * Run returns the same result. A banner in the app makes mock mode obvious.
 */
import { http, HttpResponse, delay } from "msw";
import { analyze, network } from "../test/fixtures";

export const handlers = [
  http.get("/api/health", () => HttpResponse.json({
    status: "ok", model_version: "mock", dataset_hash: "mock", selected_localizer: "xgb_reg",
    models_loaded: { baseline_detector: true, xgb_classifier: true, localizer_reg: true, localizer_rank: true, gnn: false },
  })),
  http.get("/api/features", () => HttpResponse.json({ mitigation_enabled: false, gnn_enabled: true, gnn_available: false, rag_enabled: true })),
  http.get("/api/network", () => HttpResponse.json(network)),
  http.get("/api/sample-requests", () => HttpResponse.json([
    { kind: "theft", label: "Big theft near Shimoga", request: { load_scale: 1, theft: { km: 150, magnitude_mw: 25, pf: 0.95 }, fault: null, sensor_profile: "major_only", noise_mult: 1, seed: 42 } },
    { kind: "normal", label: "Normal operation", request: { load_scale: 1, theft: null, fault: null, sensor_profile: "major_only", noise_mult: 1, seed: 42 } },
  ])),
  http.get("/api/scenarios", () => HttpResponse.json({ total: 0, items: [] })),
  http.post("/api/analyze", async () => { await delay(250); return HttpResponse.json(analyze); }),
  http.post("/api/batch-analyze", async ({ request }) => {
    const body = (await request.json()) as { requests: unknown[] };
    await delay(400);
    return HttpResponse.json({ request_id: "mock", latency_ms: 400, n_ok: body.requests.length,
      results: body.requests.map((_, index) => ({ index, result: analyze, error: null })) });
  }),
  http.get("/api/rag/status", () => HttpResponse.json({ enabled: true, llm_available: false, model: null, n_docs: 0, n_chunks: 0, sources: [], built_at: null })),
  http.get("/api/metrics", () => HttpResponse.json({ error: { code: "not_ready", message: "metrics are not mocked — start the backend" } }, { status: 404 })),
  http.get("/api/sensor-study", () => HttpResponse.json({ error: { code: "not_ready", message: "not mocked" } }, { status: 404 })),
];
