/**
 * 06 §8 acceptance against the real backend:
 * (a) estimate inside interval, (b) map click sets slider, (c) live mode re-runs on change,
 * (d) replay plays & pauses, (f) backend down → friendly error; plus presets, compare,
 * similar cases (vector DB) and run history.
 */
import { expect, test } from "@playwright/test";

test("(a) estimate lies within the reported interval", async ({ page }) => {
  await page.goto("/lab");
  const resp = page.waitForResponse((r) => r.url().includes("/api/analyze") && r.request().method() === "POST");
  await page.getByTestId("run-button").click();
  const body = await (await resp).json();
  const [lo, hi] = body.localize.interval_km;
  expect(body.localize.est_km).toBeGreaterThanOrEqual(lo);
  expect(body.localize.est_km).toBeLessThanOrEqual(hi);
  await expect(page.getByTestId("status-summary")).toContainText("Suspected unauthorized extraction near km");
  // the corridor must survive the overlay update (rebuilt React Flow nodes used to be hidden after a run)
  await expect(page.getByRole("button", { name: "Tap at km 250" })).toBeVisible();
  await expect(page.locator(".react-flow__edge").first()).toBeVisible();
});

test("(b) clicking a tap on the corridor map sets the km slider", async ({ page }) => {
  await page.goto("/lab");
  await page.getByRole("button", { name: "Tap at km 250" }).click();
  await expect(page.locator('output[for="theft-km"]')).toContainText("250");
  await page.keyboard.press("ArrowRight");                       // ←/→ nudge ±10 km
  await expect(page.locator('output[for="theft-km"]')).toContainText("260");
});

test("(c) live mode re-runs analysis when a slider changes", async ({ page }) => {
  await page.goto("/lab");
  let calls = 0;
  page.on("request", (r) => { if (r.url().includes("/api/analyze")) calls++; });
  await page.getByTestId("live-mode").check();
  await expect(page.getByTestId("status-badge")).toBeVisible();
  const before = calls;
  await page.locator("#theft-mw").fill("35");
  await expect.poll(() => calls, { timeout: 15_000 }).toBeGreaterThan(before);
  await expect(page.getByTestId("status-badge")).toHaveAttribute("data-status", "suspicious");
});

test("(d) replay plays and pauses", async ({ page }) => {
  await page.goto("/replay");
  await page.getByRole("button", { name: /Connect/ }).click();
  await expect(page.getByText(/^Tick \d+/)).toBeVisible();
  await page.getByRole("button", { name: /Pause/ }).click();
  await page.waitForTimeout(400);                                 // let any in-flight tick land
  const tick = await page.locator(".mono", { hasText: /^tick \d+ \/ \d+$/ }).first().textContent();
  await page.waitForTimeout(2500);
  await expect(page.locator(".mono", { hasText: /^tick \d+ \/ \d+$/ }).first()).toHaveText(tick!);
  await page.getByRole("button", { name: /Play/ }).click();
  await expect(page.locator(".mono", { hasText: /^tick \d+ \/ \d+$/ }).first()).not.toHaveText(tick!, { timeout: 10_000 });
  await expect(page.getByTestId("replay-scrubber")).toBeVisible();
});

test("(f) backend down shows a friendly error", async ({ page }) => {
  // only the backend paths — a "**/api/**" glob would also block Vite's /src/api/*.ts modules
  await page.route((url) => url.pathname.startsWith("/api/"), (route) => route.abort("connectionrefused"));
  await page.goto("/lab");
  await expect(page.getByRole("alert").first()).toContainText(/Cannot reach the API|network_error/);
});

test("presets, compare profiles, similar cases and run history", async ({ page }) => {
  await page.goto("/lab");
  await page.locator("#preset").selectOption("sensor_fault");
  await expect(page.locator("#fault")).toHaveValue("bias");
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("status-summary")).toBeVisible();

  await page.getByRole("tab", { name: "Compare profiles" }).click();
  await page.getByTestId("run-compare").click();
  await expect(page.getByTestId("compare-result")).toContainText("Dense", { timeout: 60_000 });

  await page.getByRole("tab", { name: "Similar past cases" }).click();
  await page.getByTestId("run-similar").click();
  await expect(page.getByTestId("similar-result")).toContainText("scenarios indexed");

  await page.locator("#preset").selectOption("theft");
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("status-badge")).toHaveAttribute("data-status", "suspicious");
  await page.getByRole("tab", { name: "Run history" }).click();
  await page.getByLabel("Select run 1 for diff").check();
  await page.getByLabel("Select run 2 for diff").check();
  await expect(page.getByTestId("history-diff")).toContainText("▶");
});

test("activity console shows user action, injection and the suspected-extraction alert", async ({ page }) => {
  await page.goto("/lab");
  await expect(page.getByTestId("run-button")).toBeVisible();      // page + key handlers mounted
  await page.keyboard.press("`");
  await expect(page.getByTestId("console-dock")).toBeVisible();
  await page.keyboard.press("r");                                 // Run shortcut while the dock is open
  const dock = page.getByTestId("console-dock");
  await expect(dock.locator('[data-category="user"]', { hasText: "Ran analysis" }).last()).toBeVisible();
  await expect(dock.locator('[data-category="simulation"]', { hasText: "unmetered load at km 150" }).last()).toBeVisible();
  const alert = dock.locator('[data-level="alert"]', { hasText: "SUSPECTED UNAUTHORIZED EXTRACTION" }).last();
  await expect(alert).toBeVisible({ timeout: 30_000 });
  await alert.getByRole("button").first().click();
  // a second analysis can append another alert after the click, so `.last()` may move; check the row that was expanded
  const expanded = dock.locator('[data-level="alert"]', { has: page.locator('button[aria-expanded="true"]') });
  await expect(expanded.getByRole("button", { name: /Open km \d+ in Scenario Lab/ })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("console-dock")).toBeHidden();
  await page.goto("/console");
  await expect(page.getByRole("heading", { name: "Activity console" })).toBeVisible();
  await expect(page.locator('[data-category="user"]', { hasText: "Opened Console page" }).last()).toBeVisible();
});
