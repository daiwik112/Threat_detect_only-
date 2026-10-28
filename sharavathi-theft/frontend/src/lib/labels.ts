/** Human-readable names for model features and measurement channels. */

const SHORT: Record<string, string> = {
  SHARAVATHI: "Sharavathi", SHIMOGA: "Shimoga", SIRA_KBCROSS: "Sira/KB-Cross",
  NELAMANGALA: "Nelamangala", PEENYA: "Peenya",
};
export const busName = (id: string) => SHORT[id] ?? id.replace("TAP_", "Tap ");
export const sectionName = (s: string) => s.split("-").map(busName).join(" → ");

export function featureLabel(f: string): string {
  const rules: [RegExp, (m: RegExpMatchArray) => string][] = [
    [/^f_r_p_total$/, () => "Total unexplained P (MW)"],
    [/^f_r_p_(.+)$/, (m) => `P residual · ${sectionName(m[1])}`],
    [/^f_r_q_(.+)$/, (m) => `Q residual · ${sectionName(m[1])}`],
    [/^f_dp_in_(.+)$/, (m) => `Entry-flow deviation · ${sectionName(m[1])}`],
    [/^f_dva_mw_(.+)$/, (m) => `Angle-drop (MW-equiv.) · ${sectionName(m[1])}`],
    [/^f_dva_(.+)$/, (m) => `Angle-drop deviation · ${sectionName(m[1])}`],
    [/^f_dvm_(.+)$/, (m) => `Voltage-drop deviation · ${sectionName(m[1])}`],
    [/^f_xfrac_(.+)$/, (m) => `Position-in-section estimate · ${sectionName(m[1])}`],
    [/^f_vm_dev_(.+)$/, (m) => `Voltage deviation · ${busName(m[1])}`],
    [/^f_va_dev_(.+)$/, (m) => `Angle deviation · ${busName(m[1])}`],
    [/^f_kcl_p_(.+)$/, (m) => `P meter consistency (KCL) · ${busName(m[1])}`],
    [/^f_kcl_q_(.+)$/, (m) => `Q meter consistency (KCL) · ${busName(m[1])}`],
    [/^f_load_scale_est$/, () => "Estimated load scale"],
    [/^f_load_scale_spread$/, () => "Load-ratio spread (meter disagreement)"],
    [/^f_n_missing$/, () => "Missing channels"],
  ];
  for (const [re, fn] of rules) {
    const m = f.match(re);
    if (m) return fn(m);
  }
  return f;
}

export function channelLabel(c: string): string {
  let m = c.match(/^m_LINE(\d+)_([pq])_(from|to)$/);
  if (m) return `Line ${m[1]} ${m[2].toUpperCase()} (${m[3]} end)`;
  m = c.match(/^m_(.+)_(vm|va|p|q)$/);
  if (m) return `${busName(m[1])} ${{ vm: "V (pu)", va: "θ (°)", p: "P (MW)", q: "Q (Mvar)" }[m[2]]}`;
  return c;
}

export const LABEL_NAMES: Record<string, string> = {
  normal: "Normal", theft: "Suspected extraction", sensor_bias: "Sensor bias",
  sensor_drift: "Sensor drift", missing: "Missing data", line_outage: "Line outage",
};
