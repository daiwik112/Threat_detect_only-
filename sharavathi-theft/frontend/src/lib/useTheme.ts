import { useScenarioStore } from "../store/useScenarioStore";
import { THEMES, type Theme } from "../theme";

/** Concrete colour roles for the active theme (kept out of Plot.tsx so non-chart pages don't load Plotly). */
export function useTheme(): Theme {
  return THEMES[useScenarioStore((s) => s.theme)];
}
