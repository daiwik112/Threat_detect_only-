/**
 * API types — aliases over schema.d.ts, which is GENERATED from the backend's
 * openapi.json (`make types`). Do not hand-edit schema.d.ts.
 */
import type { components } from "./schema";

type S = components["schemas"];

export type NetworkResponse = S["NetworkResponse"];
export type Bus = S["Bus"];
export type Line = S["Line"];
export type Tap = S["Tap"];
export type SimulateRequest = S["SimulateRequest"];
export type SimulateResponse = S["SimulateResponse"];
export type FaultSpec = S["FaultSpec"];
export type DetectResponse = S["DetectResponse"];
export type LocalizeResponse = S["LocalizeResponse"];
export type AnalyzeRequest = S["AnalyzeRequest"];
export type AnalyzeResponse = S["AnalyzeResponse"];
export type FeaturesResponse = S["FeaturesResponse"];
export type HealthResponse = S["HealthResponse"];
export type StrategiesResponse = S["StrategiesResponse"];
export type MitigationResponse = S["MitigationResponse"];
export type ScenarioListResponse = S["ScenarioListResponse"];
export type ScenarioSummary = S["ScenarioSummary"];
export type RagStatus = S["RagStatus"];
export type AdviceResponse = S["AdviceResponse"];
export type RagCitation = S["RagCitation"];
export type AdviseRequest = S["AdviseRequest"];
export type BatchAnalyzeResponse = S["BatchAnalyzeResponse"];
export type SampleRequest = S["SampleRequest"];
export type SimilarCasesResponse = S["SimilarCasesResponse"];
export type SensorProfile = NonNullable<SimulateRequest["sensor_profile"]>;
export type Measurements = Record<string, number | null>;

/* metrics.json / sensor_study.json are free-form files served as-is; typed loosely here. */
export interface DetMetrics {
  precision: number; recall: number; f1: number; fpr: number | null;
  roc_auc?: number; recall_at_fpr_1pct?: number | null; recall_at_fpr_5pct?: number | null;
}
export interface LocMetrics {
  method?: string; n?: number; mae_km: number | null; median_km: number | null;
  within_10km: number | null; within_25km: number | null; within_50km?: number | null;
  top3_within_10km?: number; section_accuracy?: number; interval_coverage?: number;
  by_magnitude?: Record<string, { n: number; mae_km: number | null; median_km: number | null }>;
}
export interface MetricsFile {
  detection: DetMetrics;
  classification: { confusion: number[][]; per_class_recall: Record<string, number | null>; labels: string[] };
  localization: LocMetrics & { method: string; split: string };
  baseline: DetMetrics & LocMetrics;
  detection_detail: Record<string, Record<string, DetMetrics>>;
  localization_detail: Record<string, Record<string, LocMetrics>>;
  robustness: Record<string, Record<string, unknown>>;
  beats_baseline: Record<string, boolean>;
  honest_claim: string;
  selected_localizer: string | null;
  gnn_decision?: { keep: boolean; holdout_mae_km: Record<string, unknown>; holdout_top3_within_10km: Record<string, unknown> };
  artifact_meta?: Record<string, unknown>;
}
export interface EvalArrays {
  [key: string]: unknown;
  roc?: Record<string, { fpr: number[]; tpr: number[] }>;
}
export interface SensorConfig {
  name: string; profile: string; noise_factor: number;
  detection_recall: number; fpr: number; mae_km: number;
  recall_at_fpr_1pct?: number; median_km?: number; within_10km?: number; top3_tap_acc?: number;
}
export interface SensorStudyFile {
  configs: SensorConfig[];
  indistinguishability: {
    adjacent_pairs?: { km_a: number; km_b: number; distance: number; separable: boolean }[];
    unresolvable_ranges_km?: [number, number][];
    threshold?: number;
  };
  missing_datasets?: string[];
}
export interface ReplayMessage {
  t: number; scenario_id: string; measurements: Measurements;
  detect: DetectResponse; localize: LocalizeResponse;
  truth: { label: string; km: number | null; magnitude_mw: number };
}

export interface ApiErrorBody { error: { code: string; message: string } }
