import type { AnalyzeResponse, NetworkResponse } from "../api/types";

export const network: NetworkResponse = {
  buses: [
    { id: "SHARAVATHI", km: 0, role: "slack_gen", metered: true },
    { id: "SHIMOGA", km: 110, role: "junction", metered: true },
    { id: "PEENYA", km: 361, role: "load", metered: true },
  ],
  lines: [
    { id: "LINE0", from_bus: "SHARAVATHI", to_bus: "TAP_150", km_len: 150, loading_pct: 50 },
    { id: "LINE1", from_bus: "TAP_150", to_bus: "PEENYA", km_len: 211, loading_pct: 40 },
  ],
  taps: [{ id: "TAP_150", km: 150, line_id: "LINE0" }],
  shunts: [],
  sections: [{ id: "SHARAVATHI-SHIMOGA", from_bus: "SHARAVATHI", to_bus: "SHIMOGA", km_from: 0, km_to: 110 }],
  assumptions: ["test assumption"],
  disclaimer: "Simulated results",
};

export const analyze: AnalyzeResponse = {
  request_id: "abc123",
  latency_ms: 321,
  simulate: {
    scenario_id: "s1",
    measurements: { m_SHIMOGA_p: 60.1, m_SHIMOGA_vm: 1.02 },
    expected: { m_SHIMOGA_p: 60, m_SHIMOGA_vm: 1.021 },
    true_label: "theft", true_km: 150, true_tap_id: "TAP_150", true_magnitude_mw: 20,
    sensor_profile: "major_only", warnings: [], disclaimer: "Simulated results",
  },
  detect: {
    status: "suspicious", anomaly_score: 0.9999, threshold_z: 3.2,
    residuals: [{ section: "SHIMOGA-NELAMANGALA", r_p_mw: 19.5, r_q_mvar: 6.1, z: 18.2 }],
    top_contributors: [{ feature: "f_r_p_total", shap: 4.2 }],
    likely_cause: { extraction: 0.99, sensor_error: 0.01, other: 0 },
  },
  localize: {
    method: "xgb_reg", est_km: 145, interval_km: [119, 170],
    ranked_sections: [{ section: "SHIMOGA-NELAMANGALA", prob: 0.99 }],
    ranked_taps: [{ tap_id: "TAP_160", km: 160, prob: 0.4 }, { tap_id: "TAP_150", km: 150, prob: 0.3 }],
  },
  recommendations: ["Suspected unauthorized extraction (high confidence)."],
  disclaimer: "Simulated results",
};
