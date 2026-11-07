# WTI Shadow Holdout Evaluation

Generated UTC: `2026-11-07T03:17:50.030748+00:00`

Stage: **COLLECTING**

Frozen model: `WTI_MODEL_D_FROZEN_2026-09-02_v1`

Eligible observations: **45**

## Matured outcomes

- 5D: 39
- 20D: 25
- 60D: 0

## Horizon metrics

| Horizon | Current IC | Model D IC | Δ D−Current | Current Direction | Model D Direction | D Signals | Bootstrap P(Δ>0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5D | -0.130 | -0.173 | -0.042 | 25.0% | 33.3% | 6 | n/a |
| 20D | -0.669 | -0.657 | +0.012 | 25.0% | 0.0% | 6 | n/a |
| 60D | n/a | n/a | n/a | n/a | n/a | 0 | n/a |

## Risk-state metrics

- Current Stress AUC 20D: 0.018
- Model D Stress AUC 20D: 0.009

## Pre-registered evaluation gate

- ✅ Freeze integrity: expected Model-D version only: `WTI_MODEL_D_FROZEN_2026-09-02_v1`
- ❌ Source-health OK rate >= 95%: `93.8%`
- ✅ Model D 20D IC > Current 20D IC: `D -0.657 vs Current -0.669`
- ❌ Model D absolute 20D IC > 0: `-0.657`
- ❌ Model D Direction 20D >= 50% with at least 20 extreme signals: `0.0%; signals=6`
- ❌ Model D Stress AUC >= 0.55 and >= Current with adequate event/non-event counts: `D 0.009 vs Current 0.018; events=6, non-events=19`

## Verdict: **NO_DECISION_COLLECTING**

No model decision is allowed yet. The holdout is still collecting matured 20D outcomes.
