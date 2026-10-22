# WTI Shadow Holdout Evaluation

Generated UTC: `2026-10-22T01:59:27.420085+00:00`

Stage: **COLLECTING**

Frozen model: `WTI_MODEL_D_FROZEN_2026-09-02_v1`

Eligible observations: **33**

## Matured outcomes

- 5D: 28
- 20D: 13
- 60D: 0

## Horizon metrics

| Horizon | Current IC | Model D IC | Δ D−Current | Current Direction | Model D Direction | D Signals | Bootstrap P(Δ>0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5D | -0.380 | -0.425 | -0.044 | 25.0% | 33.3% | 6 | n/a |
| 20D | -0.797 | -0.835 | -0.038 | 25.0% | 0.0% | 6 | n/a |
| 60D | n/a | n/a | n/a | n/a | n/a | 0 | n/a |

## Risk-state metrics

- Current Stress AUC 20D: 0.048
- Model D Stress AUC 20D: 0.024

## Pre-registered evaluation gate

- ✅ Freeze integrity: expected Model-D version only: `WTI_MODEL_D_FROZEN_2026-09-02_v1`
- ❌ Source-health OK rate >= 95%: `91.7%`
- ❌ Model D 20D IC > Current 20D IC: `D -0.835 vs Current -0.797`
- ❌ Model D absolute 20D IC > 0: `-0.835`
- ❌ Model D Direction 20D >= 50% with at least 20 extreme signals: `0.0%; signals=6`
- ❌ Model D Stress AUC >= 0.55 and >= Current with adequate event/non-event counts: `D 0.024 vs Current 0.048; events=6, non-events=7`

## Verdict: **NO_DECISION_COLLECTING**

No model decision is allowed yet. The holdout is still collecting matured 20D outcomes.
