import { describe, expect, it } from "vitest";
import { channelLabel, featureLabel, sectionName } from "./labels";

describe("labels", () => {
  it("humanises features", () => {
    expect(featureLabel("f_r_p_SHIMOGA-NELAMANGALA")).toBe("P residual · Shimoga → Nelamangala");
    expect(featureLabel("f_kcl_p_PEENYA")).toBe("P meter consistency (KCL) · Peenya");
    expect(featureLabel("f_dva_mw_SHARAVATHI-SHIMOGA")).toMatch(/^Angle-drop \(MW-equiv\.\)/);
    expect(featureLabel("f_unknown")).toBe("f_unknown");
  });
  it("humanises channels and sections", () => {
    expect(channelLabel("m_SHIMOGA_vm")).toBe("Shimoga V (pu)");
    expect(channelLabel("m_LINE10_p_to")).toBe("Line 10 P (to end)");
    expect(sectionName("NELAMANGALA-PEENYA")).toBe("Nelamangala → Peenya");
  });
});
