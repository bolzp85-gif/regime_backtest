# WTI Shadow Holdout Evaluation

Generated UTC: `2026-10-09T03:10:35.043058+00:00`

Stage: **COLLECTING**

Frozen model: `WTI_MODEL_D_FROZEN_2026-09-02_v1`

Eligible observations: **25**

## Matured outcomes

- 5D: 19
- 20D: 5
- 60D: 0

## Horizon metrics

| Horizon | Current IC | Model D IC | Δ D−Current | Current Direction | Model D Direction | D Signals | Bootstrap P(Δ>0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5D | -0.218 | -0.202 | +0.016 | 25.0% | 33.3% | 6 | n/a |
| 20D | n/a | n/a | n/a | 0.0% | 0.0% | 2 | n/a |
| 60D | n/a | n/a | n/a | n/a | n/a | 0 | n/a |

## Risk-state metrics

- Current Stress AUC 20D: 0.000
- Model D Stress AUC 20D: 0.000

## Pre-registered evaluation gate

- ✅ Freeze integrity: expected Model-D version only: `WTI_MODEL_D_FROZEN_2026-09-02_v1`
- ❌ Source-health OK rate >= 95%: `92.6%`
- ❌ Model D 20D IC > Current 20D IC: `n/a`
- ❌ Model D absolute 20D IC > 0: `n/a`
- ❌ Model D Direction 20D >= 50% with at least 20 extreme signals: `0.0%; signals=2`
- ❌ Model D Stress AUC >= 0.55 and >= Current with adequate event/non-event counts: `D 0.000 vs Current 0.000; events=1, non-events=4`

## Verdict: **NO_DECISION_COLLECTING**

No model decision is allowed yet. The holdout is still collecting matured 20D outcomes.
