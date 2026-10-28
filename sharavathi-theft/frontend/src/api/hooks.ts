/** Server state lives in TanStack Query only (never duplicated into Zustand). */
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "./client";
import type {
  AdviceResponse, AdviseRequest, RagStatus, BatchAnalyzeResponse, SampleRequest, SimilarCasesResponse, Measurements,
  AnalyzeRequest, AnalyzeResponse, EvalArrays, FeaturesResponse, HealthResponse, MetricsFile,
  MitigationResponse, NetworkResponse, ScenarioListResponse, SensorStudyFile, StrategiesResponse,
} from "./types";

const forever = { staleTime: Infinity, retry: 1 } as const;

export const useHealth = () => useQuery({ queryKey: ["health"], queryFn: () => api<HealthResponse>("/api/health"), retry: 0, refetchInterval: 30_000 });
export const useFeatures = () => useQuery({ queryKey: ["features"], queryFn: () => api<FeaturesResponse>("/api/features"), ...forever });
export const useNetwork = () => useQuery({ queryKey: ["network"], queryFn: () => api<NetworkResponse>("/api/network"), ...forever });
export const useMetrics = () => useQuery({ queryKey: ["metrics"], queryFn: () => api<MetricsFile>("/api/metrics"), ...forever });
export const useEvalArrays = () => useQuery({ queryKey: ["eval-arrays"], queryFn: () => api<EvalArrays>("/api/eval-arrays"), ...forever });
export const useSensorStudy = () => useQuery({ queryKey: ["sensor-study"], queryFn: () => api<SensorStudyFile>("/api/sensor-study"), retry: 0, staleTime: 60_000 });
export const useStrategies = () => useQuery({ queryKey: ["strategies"], queryFn: () => api<StrategiesResponse>("/api/mitigation/strategies"), ...forever });
export const useScenarios = (label: string, limit = 25) =>
  useQuery({ queryKey: ["scenarios", label, limit], queryFn: () => api<ScenarioListResponse>(`/api/scenarios?split=test&limit=${limit}${label ? `&label=${label}` : ""}`), staleTime: Infinity });

/** POST /api/analyze. A newer call aborts the in-flight one. */
let inflight: AbortController | null = null;
export const useAnalyze = () =>
  useMutation({
    mutationKey: ["analyze"],
    mutationFn: (req: AnalyzeRequest) => {
      inflight?.abort();
      inflight = new AbortController();
      return api<AnalyzeResponse>("/api/analyze", { method: "POST", json: req, signal: inflight.signal });
    },
  });

export const useMitigationPlan = () =>
  useMutation({
    mutationFn: (body: { scenario_id: string; strategy: string }) =>
      api<MitigationResponse>("/api/mitigation/plan", { method: "POST", json: body }),
  });

/* ---- RAG mitigation advisor (09) ---- */
export const useRagStatus = () => useQuery({ queryKey: ["rag-status"], queryFn: () => api<RagStatus>("/api/rag/status"), staleTime: 60_000, retry: 0 });
export const useAdvise = () =>
  useMutation({ mutationFn: (body: AdviseRequest) => api<AdviceResponse>("/api/rag/advise", { method: "POST", json: body }) });
export const useRagQuery = () =>
  useMutation({ mutationFn: (body: { question: string; k?: number }) => api<AdviceResponse>("/api/rag/query", { method: "POST", json: body }) });

/* ---- 06: presets, batch (sweep / compare), similar cases (vector DB) ---- */
export const useSampleRequests = () => useQuery({ queryKey: ["sample-requests"], queryFn: () => api<SampleRequest[]>("/api/sample-requests"), ...forever });
export const useBatchAnalyze = () =>
  useMutation({ mutationFn: (requests: AnalyzeRequest[]) => api<BatchAnalyzeResponse>("/api/batch-analyze", { method: "POST", json: { requests } }) });
export const useSimilarCases = () =>
  useMutation({ mutationFn: (body: { measurements: Measurements; k?: number }) => api<SimilarCasesResponse>("/api/similar-cases", { method: "POST", json: body }) });
