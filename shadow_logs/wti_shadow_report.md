# WTI Shadow Holdout Evaluation

Generated UTC: `2026-10-29T02:58:00.898076+00:00`

Stage: **COLLECTING**

Frozen model: `WTI_MODEL_D_FROZEN_2026-09-02_v1`

Eligible observations: **38**

## Matured outcomes

- 5D: 32
- 20D: 18
- 60D: 0

## Horizon metrics

| Horizon | Current IC | Model D IC | Δ D−Current | Current Direction | Model D Direction | D Signals | Bootstrap P(Δ>0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5D | -0.357 | -0.415 | -0.059 | 25.0% | 33.3% | 6 | n/a |
| 20D | -0.895 | -0.901 | -0.006 | 25.0% | 0.0% | 6 | n/a |
| 60D | n/a | n/a | n/a | n/a | n/a | 0 | n/a |

## Risk-state metrics

- Current Stress AUC 20D: 0.028
- Model D Stress AUC 20D: 0.014

## Pre-registered evaluation gate

- ✅ Freeze integrity: expected Model-D version only: `WTI_MODEL_D_FROZEN_2026-09-02_v1`
- ❌ Source-health OK rate >= 95%: `92.7%`
- ❌ Model D 20D IC > Current 20D IC: `D -0.901 vs Current -0.895`
- ❌ Model D absolute 20D IC > 0: `-0.901`
- ❌ Model D Direction 20D >= 50% with at least 20 extreme signals: `0.0%; signals=6`
- ❌ Model D Stress AUC >= 0.55 and >= Current with adequate event/non-event counts: `D 0.014 vs Current 0.028; events=6, non-events=12`

## Verdict: **NO_DECISION_COLLECTING**

No model decision is allowed yet. The holdout is still collecting matured 20D outcomes.
