/**
 * Plotly wrapper with the dashboard's chart conventions: recessive grid/axes, text in ink
 * tokens (never series colour), unified hover, PNG export via the modebar, theme-aware.
 */
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Config, Data, Layout } from "plotly.js";
import { forwardRef } from "react";
import { useTheme } from "../lib/useTheme";
import type { Theme } from "../theme";

const PlotlyComponent = createPlotlyComponent(Plotly);
export { Plotly };

export { useTheme };

export function baseLayout(t: Theme, extra: Partial<Layout> = {}): Partial<Layout> {
  const axis = {
    gridcolor: t.grid, linecolor: t.axis, zerolinecolor: t.axis, tickcolor: t.axis,
    tickfont: { color: t.text2, size: 11 }, title: { font: { color: t.text2, size: 12 } },
    automargin: true,
  };
  return {
    paper_bgcolor: t.surface,
    plot_bgcolor: t.surface,
    font: { family: '"IBM Plex Sans", "Segoe UI", system-ui, sans-serif', color: t.text, size: 12 },
    margin: { l: 48, r: 16, t: 12, b: 40 },
    hoverlabel: { bgcolor: t.surface, bordercolor: t.axis, font: { color: t.text, size: 12 } },
    legend: { orientation: "h", y: 1.12, x: 0, font: { color: t.text2, size: 11 } },
    bargap: 0.25,
    ...extra,
    xaxis: { ...axis, ...(extra.xaxis ?? {}) },
    yaxis: { ...axis, ...(extra.yaxis ?? {}) },
  };
}

interface Props {
  data: Data[];
  layout?: Partial<Layout>;
  height?: number;
  ariaLabel: string;
  filename?: string;
}

export const Plot = forwardRef<HTMLDivElement, Props>(function Plot(
  { data, layout = {}, height = 260, ariaLabel, filename = "chart" }, ref,
) {
  const t = useTheme();
  const config: Partial<Config> = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
    toImageButtonOptions: { format: "png", filename, scale: 2 },
  };
  return (
    <div ref={ref} role="img" aria-label={ariaLabel}>
      <PlotlyComponent
        data={data}
        layout={{ ...baseLayout(t, layout), height, autosize: true }}
        config={config}
        style={{ width: "100%" }}
        useResizeHandler
      />
    </div>
  );
});
