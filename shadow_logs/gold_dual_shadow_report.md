# Gold D1 + V1 Dual-Output Future Shadow

Generated UTC: `2026-11-14T03:38:23.969135+00:00`

Stage: **COLLECTING**

Verdict: **NO_DECISION_COLLECTING**

Shadow start: **2026-09-13**

## D1 Direction Context

- Matured 20D: **25**
- IC20: `-0.375`
- Direction20: `100.0%`
- Extreme signals: **25**

## V1 Volatility-State

- Matured 20D: **25**
- High-Vol AUC: `0.015`
- Phase-median AUC: `n/a`
- Q20-Q80 High-Vol gap: `-60.0%`
- Score vs lower FwdVol: `-0.836`

## Gates

- ✅ **Infrastructure** — Frozen core/spec integrity: `OK`
- ⏳ **D1 Direction** — D1 IC20 >= +0.05: `-0.375`
- ⏳ **D1 Direction** — D1 Direction20 >=50%: `100.0%`
- ⏳ **D1 Direction** — Enough extreme D1 signals: `25 / 30`
- ⏳ **D1 Direction** — Positive IC in >=50% complete 60-observation blocks: `n/a · blocks=0`
- ⏳ **V1 Volatility** — V1 High-Vol AUC >=0.60: `0.015`
- ⏳ **V1 Volatility** — V1 phase-median AUC >=0.55: `n/a`
- ⏳ **V1 Volatility** — V1 Q20-Q80 High-Vol gap >=5pp: `-60.0%`
- ⏳ **V1 Volatility** — V1 score relates >=+0.10 to lower future volatility: `-0.836`
- ⏳ **V1 Volatility** — AUC >0.50 in >=50% complete 60-observation blocks: `n/a · blocks=0`

## Interpretation

D1 and V1 are evaluated as separate outputs. No combined Gold score is inferred from this shadow.

V1 high-vol events use the PIT threshold stored on the original observation date. The threshold is never re-estimated when the future outcome matures.
