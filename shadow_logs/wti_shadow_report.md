# WTI Shadow Holdout Evaluation

Generated UTC: `2026-11-10T04:49:44.872102+00:00`

Stage: **COLLECTING**

Frozen model: `WTI_MODEL_D_FROZEN_2026-09-02_v1`

Eligible observations: **46**

## Matured outcomes

- 5D: 40
- 20D: 26
- 60D: 0

## Horizon metrics

| Horizon | Current IC | Model D IC | Δ D−Current | Current Direction | Model D Direction | D Signals | Bootstrap P(Δ>0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5D | -0.053 | -0.092 | -0.039 | 25.0% | 42.9% | 7 | n/a |
| 20D | -0.582 | -0.556 | +0.025 | 25.0% | 0.0% | 6 | n/a |
| 60D | n/a | n/a | n/a | n/a | n/a | 0 | n/a |

## Risk-state metrics

- Current Stress AUC 20D: 0.017
- Model D Stress AUC 20D: 0.008

## Pre-registered evaluation gate

- ✅ Freeze integrity: expected Model-D version only: `WTI_MODEL_D_FROZEN_2026-09-02_v1`
- ❌ Source-health OK rate >= 95%: `93.9%`
- ✅ Model D 20D IC > Current 20D IC: `D -0.556 vs Current -0.582`
- ❌ Model D absolute 20D IC > 0: `-0.556`
- ❌ Model D Direction 20D >= 50% with at least 20 extreme signals: `0.0%; signals=6`
- ❌ Model D Stress AUC >= 0.55 and >= Current with adequate event/non-event counts: `D 0.008 vs Current 0.017; events=6, non-events=20`

## Verdict: **NO_DECISION_COLLECTING**

No model decision is allowed yet. The holdout is still collecting matured 20D outcomes.
