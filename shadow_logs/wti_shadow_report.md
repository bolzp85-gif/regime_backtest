# WTI Shadow Holdout Evaluation

Generated UTC: `2026-11-11T03:12:28.227748+00:00`

Stage: **COLLECTING**

Frozen model: `WTI_MODEL_D_FROZEN_2026-09-02_v1`

Eligible observations: **47**

## Matured outcomes

- 5D: 41
- 20D: 27
- 60D: 0

## Horizon metrics

| Horizon | Current IC | Model D IC | Δ D−Current | Current Direction | Model D Direction | D Signals | Bootstrap P(Δ>0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5D | +0.004 | -0.032 | -0.036 | 25.0% | 50.0% | 8 | n/a |
| 20D | -0.578 | -0.501 | +0.078 | 25.0% | 0.0% | 6 | n/a |
| 60D | n/a | n/a | n/a | n/a | n/a | 0 | n/a |

## Risk-state metrics

- Current Stress AUC 20D: 0.016
- Model D Stress AUC 20D: 0.008

## Pre-registered evaluation gate

- ✅ Freeze integrity: expected Model-D version only: `WTI_MODEL_D_FROZEN_2026-09-02_v1`
- ❌ Source-health OK rate >= 95%: `94.0%`
- ✅ Model D 20D IC > Current 20D IC: `D -0.501 vs Current -0.578`
- ❌ Model D absolute 20D IC > 0: `-0.501`
- ❌ Model D Direction 20D >= 50% with at least 20 extreme signals: `0.0%; signals=6`
- ❌ Model D Stress AUC >= 0.55 and >= Current with adequate event/non-event counts: `D 0.008 vs Current 0.016; events=6, non-events=21`

## Verdict: **NO_DECISION_COLLECTING**

No model decision is allowed yet. The holdout is still collecting matured 20D outcomes.
