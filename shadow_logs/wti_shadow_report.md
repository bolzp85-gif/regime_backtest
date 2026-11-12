# WTI Shadow Holdout Evaluation

Generated UTC: `2026-11-12T04:24:09.347064+00:00`

Stage: **COLLECTING**

Frozen model: `WTI_MODEL_D_FROZEN_2026-09-02_v1`

Eligible observations: **48**

## Matured outcomes

- 5D: 42
- 20D: 28
- 60D: 0

## Horizon metrics

| Horizon | Current IC | Model D IC | Δ D−Current | Current Direction | Model D Direction | D Signals | Bootstrap P(Δ>0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5D | +0.024 | -0.009 | -0.033 | 33.3% | 55.6% | 9 | n/a |
| 20D | -0.512 | -0.415 | +0.097 | 25.0% | 0.0% | 6 | n/a |
| 60D | n/a | n/a | n/a | n/a | n/a | 0 | n/a |

## Risk-state metrics

- Current Stress AUC 20D: 0.095
- Model D Stress AUC 20D: 0.102

## Pre-registered evaluation gate

- ✅ Freeze integrity: expected Model-D version only: `WTI_MODEL_D_FROZEN_2026-09-02_v1`
- ❌ Source-health OK rate >= 95%: `94.1%`
- ✅ Model D 20D IC > Current 20D IC: `D -0.415 vs Current -0.512`
- ❌ Model D absolute 20D IC > 0: `-0.415`
- ❌ Model D Direction 20D >= 50% with at least 20 extreme signals: `0.0%; signals=6`
- ❌ Model D Stress AUC >= 0.55 and >= Current with adequate event/non-event counts: `D 0.102 vs Current 0.095; events=7, non-events=21`

## Verdict: **NO_DECISION_COLLECTING**

No model decision is allowed yet. The holdout is still collecting matured 20D outcomes.
