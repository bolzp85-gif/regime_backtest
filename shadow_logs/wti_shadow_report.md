# WTI Shadow Holdout Evaluation

Generated UTC: `2026-11-20T04:25:36.136284+00:00`

Stage: **COLLECTING**

Frozen model: `WTI_MODEL_D_FROZEN_2026-09-02_v1`

Eligible observations: **54**

## Matured outcomes

- 5D: 48
- 20D: 33
- 60D: 0

## Horizon metrics

| Horizon | Current IC | Model D IC | Δ D−Current | Current Direction | Model D Direction | D Signals | Bootstrap P(Δ>0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5D | +0.079 | +0.057 | -0.022 | 57.1% | 73.3% | 15 | n/a |
| 20D | -0.419 | -0.306 | +0.113 | 25.0% | 0.0% | 6 | n/a |
| 60D | n/a | n/a | n/a | n/a | n/a | 0 | n/a |

## Risk-state metrics

- Current Stress AUC 20D: 0.153
- Model D Stress AUC 20D: 0.178

## Pre-registered evaluation gate

- ✅ Freeze integrity: expected Model-D version only: `WTI_MODEL_D_FROZEN_2026-09-02_v1`
- ❌ Source-health OK rate >= 95%: `94.7%`
- ✅ Model D 20D IC > Current 20D IC: `D -0.306 vs Current -0.419`
- ❌ Model D absolute 20D IC > 0: `-0.306`
- ❌ Model D Direction 20D >= 50% with at least 20 extreme signals: `0.0%; signals=6`
- ❌ Model D Stress AUC >= 0.55 and >= Current with adequate event/non-event counts: `D 0.178 vs Current 0.153; events=11, non-events=22`

## Verdict: **NO_DECISION_COLLECTING**

No model decision is allowed yet. The holdout is still collecting matured 20D outcomes.
