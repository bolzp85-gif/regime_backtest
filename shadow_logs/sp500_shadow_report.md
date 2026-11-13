# S&P 500 Current-only Risk-State Future Shadow

Generated UTC: `2026-11-13T04:41:13.423688+00:00`

Stage: **COLLECTING**

Verdict: **NO_DECISION_COLLECTING**

Shadow start: **2026-09-12**

Matured 20D observations: **24**

## Absolute Risk-State metrics

- Stress AUC: `0.274`
- Phase-median AUC: `n/a`
- Score<=40 stress rate: `0.0%`
- Score>=60 stress rate: `n/a`
- Stress-rate gap: `n/a`
- Score vs lower FwdVol: `-0.111`
- Score vs better FwdMAE: `-0.625`

## Descriptive early-warning episodes

- Stress episode onsets: **2**
- Prior Score<=40 alert rate: `100.0%`
- Median earliest alert lead: `14.0 eligible observations`

## Gate

- ✅ **Infrastructure** — Frozen model/core/config integrity: `OK`
- ✅ **Infrastructure** — Source-health OK rate >=95%: `97.7%`
- ⏳ **Current Risk-State** — Absolute Stress AUC >=0.60: `0.274`
- ⏳ **Current Risk-State** — Phase-median Stress AUC >=0.55: `n/a`
- ⏳ **Current Risk-State** — Stressrate Score<=40 minus Score>=60 >=5pp: `n/a`
- ⏳ **Current Risk-State** — Higher score relates to lower forward realized volatility: `-0.111`
- ⏳ **Current Risk-State** — Higher score relates to better forward MAE: `-0.625`
- ⏳ **Time-block Stability** — AUC >0.50 in >=50% of complete 60-observation blocks: `n/a · blocks=0`
- ⏳ **Time-block Stability** — Positive stress-gap in >=50% of complete 60-observation blocks: `n/a · blocks=0`

## Interpretation

This experiment validates Current only as a Risk-/Stress-State filter. It does not test or claim directional return prediction.

The early-warning episode table is descriptive and is not an optimization input or promotion gate.
