# WTI Shadow Holdout Evaluation

Generated UTC: `2026-10-23T02:03:44.401208+00:00`

Stage: **COLLECTING**

Frozen model: `WTI_MODEL_D_FROZEN_2026-09-02_v1`

Eligible observations: **34**

## Matured outcomes

- 5D: 29
- 20D: 14
- 60D: 0

## Horizon metrics

| Horizon | Current IC | Model D IC | Δ D−Current | Current Direction | Model D Direction | D Signals | Bootstrap P(Δ>0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5D | -0.368 | -0.429 | -0.061 | 25.0% | 33.3% | 6 | n/a |
| 20D | -0.798 | -0.833 | -0.035 | 25.0% | 0.0% | 6 | n/a |
| 60D | n/a | n/a | n/a | n/a | n/a | 0 | n/a |

## Risk-state metrics

- Current Stress AUC 20D: 0.042
- Model D Stress AUC 20D: 0.021

## Pre-registered evaluation gate

- ✅ Freeze integrity: expected Model-D version only: `WTI_MODEL_D_FROZEN_2026-09-02_v1`
- ❌ Source-health OK rate >= 95%: `91.9%`
- ❌ Model D 20D IC > Current 20D IC: `D -0.833 vs Current -0.798`
- ❌ Model D absolute 20D IC > 0: `-0.833`
- ❌ Model D Direction 20D >= 50% with at least 20 extreme signals: `0.0%; signals=6`
- ❌ Model D Stress AUC >= 0.55 and >= Current with adequate event/non-event counts: `D 0.021 vs Current 0.042; events=6, non-events=8`

## Verdict: **NO_DECISION_COLLECTING**

No model decision is allowed yet. The holdout is still collecting matured 20D outcomes.
