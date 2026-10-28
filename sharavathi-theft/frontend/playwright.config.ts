import { defineConfig } from "@playwright/test";

const PY = process.env.PYTHON ?? "python";

export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 30_000 },
  use: { baseURL: "http://127.0.0.1:5173", viewport: { width: 1440, height: 900 } },
  webServer: [
    {
      command: `"${PY}" -m uvicorn app.main:app --port 8000`,
      cwd: "../backend",
      url: "http://127.0.0.1:8000/api/health",
      reuseExistingServer: true,
      timeout: 120_000,
    },
    {
      command: "npx vite --port 5173 --strictPort --host 127.0.0.1",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: true,
      timeout: 120_000,
    },
  ],
});
