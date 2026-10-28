/**
 * Chart colour roles (Plotly needs concrete hex). Values mirror index.css tokens and the
 * dataviz reference palette: categorical slots in fixed order, status palette reserved.
 */
export type ThemeMode = "dark" | "light";

const shared = {
  status: { good: "#0ca30c", warning: "#fab219", serious: "#ec835a", critical: "#d03b3b" },
};

export const THEMES = {
  dark: {
    ...shared,
    surface: "#1a1a19",
    grid: "#2e2e2b",
    axis: "#383835",
    text: "#f0efec",
    text2: "#c3c2b7",
    muted: "#9a998f",
    ref: "#8f8e86",
    series: ["#3987e5", "#d95926", "#199e70"], // blue, orange, aqua (slots 1–3)
    seq: ["#1a1a19", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4"],
  },
  light: {
    ...shared,
    surface: "#fcfcfb",
    grid: "#ebeae6",
    axis: "#d6d5cf",
    text: "#0b0b0b",
    text2: "#52514e",
    muted: "#63625d",
    ref: "#8a8984",
    series: ["#2a78d6", "#eb6834", "#1baf7a"],
    seq: ["#fcfcfb", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"],
  },
} as const;

export type Theme = (typeof THEMES)[ThemeMode];

/** Fixed series identity — colour follows the entity, never its rank. */
export const SERIES = { xgb: 0, baseline: 1, gnn: 2 } as const;
