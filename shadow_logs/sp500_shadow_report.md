# S&P 500 Current-only Risk-State Future Shadow

Generated UTC: `2026-09-12T02:43:40.901141+00:00`

Stage: **COLLECTING**

Verdict: **NO_DECISION_COLLECTING**

Shadow start: **2026-09-12**

Matured 20D observations: **0**

## Absolute Risk-State metrics

- Stress AUC: `n/a`
- Phase-median AUC: `n/a`
- Score<=40 stress rate: `n/a`
- Score>=60 stress rate: `n/a`
- Stress-rate gap: `n/a`
- Score vs lower FwdVol: `n/a`
- Score vs better FwdMAE: `n/a`

## Descriptive early-warning episodes

- Stress episode onsets: **0**
- Prior Score<=40 alert rate: `n/a`
- Median earliest alert lead: `n/a`

## Gate

- ✅ **Infrastructure** — Frozen model/core/config integrity: `OK`
- ⏳ **Infrastructure** — Source-health OK rate >=95%: `n/a`
- ⏳ **Current Risk-State** — Absolute Stress AUC >=0.60: `n/a`
- ⏳ **Current Risk-State** — Phase-median Stress AUC >=0.55: `n/a`
- ⏳ **Current Risk-State** — Stressrate Score<=40 minus Score>=60 >=5pp: `n/a`
- ⏳ **Current Risk-State** — Higher score relates to lower forward realized volatility: `n/a`
- ⏳ **Current Risk-State** — Higher score relates to better forward MAE: `n/a`
- ⏳ **Time-block Stability** — AUC >0.50 in >=50% of complete 60-observation blocks: `n/a · blocks=0`
- ⏳ **Time-block Stability** — Positive stress-gap in >=50% of complete 60-observation blocks: `n/a · blocks=0`

## Interpretation

This experiment validates Current only as a Risk-/Stress-State filter. It does not test or claim directional return prediction.

The early-warning episode table is descriptive and is not an optimization input or promotion gate.
