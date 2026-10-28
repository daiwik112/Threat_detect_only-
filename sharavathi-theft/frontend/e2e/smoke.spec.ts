import { expect, test } from "@playwright/test";

const SHOTS = process.env.E2E_SHOTS; // set to a directory to save page screenshots
// the main pane scrolls (not the window), so screenshots use a tall viewport
if (SHOTS) test.use({ viewport: { width: 1440, height: 2600 } });

test("Scenario Lab: theft 150 km / 20 MW → Suspicious + localization marker", async ({ page }) => {
  await page.goto("/lab");
  // set km = 150, MW = 20 via the sliders (range inputs)
  await page.locator("#theft-km").fill("150");
  await page.locator("#theft-mw").fill("20");
  await expect(page.locator('output[for="theft-km"]')).toContainText("150");
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("status-badge")).toHaveAttribute("data-status", "suspicious");
  await expect(page.getByTestId("localization-marker")).toBeVisible();
  await page.getByTestId("reveal-truth").click();
  await expect(page.getByTestId("truth-panel")).toContainText("150 km");
  // RAG advisor: cited guidance for this result (offline extractive mode without credentials)
  await page.getByTestId("ask-advisor").click();
  await expect(page.getByTestId("advice")).toContainText("Requires human approval");
  await expect(page.getByTestId("advice")).toContainText("Planning the patrol");
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/lab.png`, fullPage: true });
});

test("Normal scenario stays normal", async ({ page }) => {
  await page.goto("/lab");
  await page.getByLabel("Inject unmetered extraction").uncheck();
  await page.getByTestId("run-button").click();
  await expect(page.getByTestId("status-badge")).toHaveAttribute("data-status", "normal");
});

test("Every page renders; Mitigation is gated", async ({ page }) => {
  for (const [path, text] of [
    ["/", "Sharavathi → Peenya 220 kV corridor"],
    ["/metrics", "Model metrics"],
    ["/sensors", "Sensor placement"],
    ["/replay", "Live replay"],
    ["/mitigation", "Coming soon"],
  ] as const) {
    await page.goto(path);
    await expect(page.getByText(text).first()).toBeVisible();
    await expect(page.getByText("Simulated results").first()).toBeVisible();
    if (path === "/metrics") await expect(page.locator(".js-plotly-plot").first()).toBeVisible();
    if (SHOTS) { await page.waitForLoadState("networkidle"); await page.waitForTimeout(800); }
    if (SHOTS) await page.screenshot({ path: `${SHOTS}/${path === "/" ? "overview" : path.slice(1)}.png`, fullPage: true });
  }
});

test("Replay streams over the WebSocket", async ({ page }) => {
  await page.goto("/replay");
  await page.getByRole("button", { name: /Connect/ }).click();
  await expect(page.getByText(/^Tick \d+/)).toBeVisible();
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/replay-live.png`, fullPage: true });
});
