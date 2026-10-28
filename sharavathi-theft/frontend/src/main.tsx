import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { useToasts } from "./components/ui";
import "./index.css";

document.documentElement.dataset.theme = localStorage.getItem("theme") ?? "dark";

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false } },
});
queryClient.getQueryCache().config.onError = (err, query) => {
  if (query.queryKey[0] === "sensor-study" || query.queryKey[0] === "health") return; // shown inline
  useToasts.getState().push(`${String(query.queryKey[0])}: ${(err as Error).message}`);
};

const USE_MOCKS = import.meta.env.VITE_USE_MOCKS === "true";

async function enableMocking() {
  if (!USE_MOCKS) return;
  const { worker } = await import("./mocks/browser");       // not bundled into normal builds' hot path
  await worker.start({ onUnhandledRequest: "bypass" });
}

enableMocking().then(() =>
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          {USE_MOCKS && (
            <div role="status" className="bg-serious/20 px-4 py-1 text-center text-xs font-semibold text-ink-1">
              MOCK API (MSW) — fixed example responses, backend not used
            </div>
          )}
          <App />
        </BrowserRouter>
      </QueryClientProvider>
    </StrictMode>,
  ),
);
