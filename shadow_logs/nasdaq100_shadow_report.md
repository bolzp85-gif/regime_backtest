# Nasdaq 100 Current-only Risk-State Future Shadow

Generated UTC: `2026-11-11T03:13:18.440847+00:00`

Stage: **COLLECTING**

Verdict: **NO_DECISION_COLLECTING**

Shadow start: **2026-09-13**

Matured 20D observations: **22**

## Absolute Risk-State metrics

- Stress AUC: `0.161`
- Phase-median AUC: `n/a`
- Score<=40 stress rate: `0.0%`
- Score>=60 stress rate: `66.7%`
- Stress-rate gap: `-66.7%`
- Score vs lower FwdVol: `-0.219`
- Score vs better FwdMAE: `-0.688`

## Descriptive early-warning episodes

- Stress episode onsets: **1**
- Prior Score<=40 alert rate: `100.0%`
- Median earliest alert lead: `13.0 eligible observations`

## Gate

- ✅ **Infrastructure** — Frozen model/core/config integrity: `OK`
- ✅ **Infrastructure** — Source-health OK rate >=95%: `95.2%`
- ⏳ **Current Risk-State** — Absolute Stress AUC >=0.60: `0.161`
- ⏳ **Current Risk-State** — Phase-median Stress AUC >=0.55: `n/a`
- ⏳ **Current Risk-State** — Stressrate Score<=40 minus Score>=60 >=5pp: `-66.7%`
- ⏳ **Current Risk-State** — Higher score relates to lower forward realized volatility: `-0.219`
- ⏳ **Current Risk-State** — Higher score relates to better forward MAE: `-0.688`
- ⏳ **Time-block Stability** — AUC >0.50 in >=50% of complete 60-observation blocks: `n/a · blocks=0`
- ⏳ **Time-block Stability** — Positive stress-gap in >=50% of complete 60-observation blocks: `n/a · blocks=0`

## Interpretation

This experiment validates Current only as a Risk-/Stress-State filter. It does not test or claim directional return prediction.

The early-warning episode table is descriptive and is not an optimization input or promotion gate.
