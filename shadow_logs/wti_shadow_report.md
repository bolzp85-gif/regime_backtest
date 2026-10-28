# WTI Shadow Holdout Evaluation

Generated UTC: `2026-10-28T02:38:29.294798+00:00`

Stage: **COLLECTING**

Frozen model: `WTI_MODEL_D_FROZEN_2026-09-02_v1`

Eligible observations: **37**

## Matured outcomes

- 5D: 32
- 20D: 17
- 60D: 0

## Horizon metrics

| Horizon | Current IC | Model D IC | Δ D−Current | Current Direction | Model D Direction | D Signals | Bootstrap P(Δ>0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5D | -0.354 | -0.417 | -0.062 | 25.0% | 33.3% | 6 | n/a |
| 20D | -0.875 | -0.882 | -0.007 | 25.0% | 0.0% | 6 | n/a |
| 60D | n/a | n/a | n/a | n/a | n/a | 0 | n/a |

## Risk-state metrics

- Current Stress AUC 20D: 0.030
- Model D Stress AUC 20D: 0.015

## Pre-registered evaluation gate

- ✅ Freeze integrity: expected Model-D version only: `WTI_MODEL_D_FROZEN_2026-09-02_v1`
- ❌ Source-health OK rate >= 95%: `92.5%`
- ❌ Model D 20D IC > Current 20D IC: `D -0.882 vs Current -0.875`
- ❌ Model D absolute 20D IC > 0: `-0.882`
- ❌ Model D Direction 20D >= 50% with at least 20 extreme signals: `0.0%; signals=6`
- ❌ Model D Stress AUC >= 0.55 and >= Current with adequate event/non-event counts: `D 0.015 vs Current 0.030; events=6, non-events=11`

## Verdict: **NO_DECISION_COLLECTING**

No model decision is allowed yet. The holdout is still collecting matured 20D outcomes.
