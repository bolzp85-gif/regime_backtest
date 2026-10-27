# Gold D1 + V1 Dual-Output Future Shadow

Generated UTC: `2026-10-27T02:27:45.660263+00:00`

Stage: **COLLECTING**

Verdict: **NO_DECISION_COLLECTING**

Shadow start: **2026-09-13**

## D1 Direction Context

- Matured 20D: **11**
- IC20: `+0.145`
- Direction20: `100.0%`
- Extreme signals: **11**

## V1 Volatility-State

- Matured 20D: **11**
- High-Vol AUC: `n/a`
- Phase-median AUC: `n/a`
- Q20-Q80 High-Vol gap: `n/a`
- Score vs lower FwdVol: `-0.691`

## Gates

- ✅ **Infrastructure** — Frozen core/spec integrity: `OK`
- ⏳ **D1 Direction** — D1 IC20 >= +0.05: `+0.145`
- ⏳ **D1 Direction** — D1 Direction20 >=50%: `100.0%`
- ⏳ **D1 Direction** — Enough extreme D1 signals: `11 / 30`
- ⏳ **D1 Direction** — Positive IC in >=50% complete 60-observation blocks: `n/a · blocks=0`
- ⏳ **V1 Volatility** — V1 High-Vol AUC >=0.60: `n/a`
- ⏳ **V1 Volatility** — V1 phase-median AUC >=0.55: `n/a`
- ⏳ **V1 Volatility** — V1 Q20-Q80 High-Vol gap >=5pp: `n/a`
- ⏳ **V1 Volatility** — V1 score relates >=+0.10 to lower future volatility: `-0.691`
- ⏳ **V1 Volatility** — AUC >0.50 in >=50% complete 60-observation blocks: `n/a · blocks=0`

## Interpretation

D1 and V1 are evaluated as separate outputs. No combined Gold score is inferred from this shadow.

V1 high-vol events use the PIT threshold stored on the original observation date. The threshold is never re-estimated when the future outcome matures.
