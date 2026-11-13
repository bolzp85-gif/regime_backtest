# WTI Shadow Holdout Evaluation

Generated UTC: `2026-11-13T03:41:10.446487+00:00`

Stage: **COLLECTING**

Frozen model: `WTI_MODEL_D_FROZEN_2026-09-02_v1`

Eligible observations: **49**

## Matured outcomes

- 5D: 43
- 20D: 29
- 60D: 0

## Horizon metrics

| Horizon | Current IC | Model D IC | Δ D−Current | Current Direction | Model D Direction | D Signals | Bootstrap P(Δ>0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5D | +0.024 | -0.007 | -0.031 | 40.0% | 60.0% | 10 | n/a |
| 20D | -0.508 | -0.384 | +0.125 | 25.0% | 0.0% | 6 | n/a |
| 60D | n/a | n/a | n/a | n/a | n/a | 0 | n/a |

## Risk-state metrics

- Current Stress AUC 20D: 0.052
- Model D Stress AUC 20D: 0.071

## Pre-registered evaluation gate

- ✅ Freeze integrity: expected Model-D version only: `WTI_MODEL_D_FROZEN_2026-09-02_v1`
- ❌ Source-health OK rate >= 95%: `94.2%`
- ✅ Model D 20D IC > Current 20D IC: `D -0.384 vs Current -0.508`
- ❌ Model D absolute 20D IC > 0: `-0.384`
- ❌ Model D Direction 20D >= 50% with at least 20 extreme signals: `0.0%; signals=6`
- ❌ Model D Stress AUC >= 0.55 and >= Current with adequate event/non-event counts: `D 0.071 vs Current 0.052; events=7, non-events=22`

## Verdict: **NO_DECISION_COLLECTING**

No model decision is allowed yet. The holdout is still collecting matured 20D outcomes.
