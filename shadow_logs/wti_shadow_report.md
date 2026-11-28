# WTI Shadow Holdout Evaluation

Generated UTC: `2026-11-28T05:02:10.381251+00:00`

Stage: **COLLECTING**

Frozen model: `WTI_MODEL_D_FROZEN_2026-09-02_v1`

Eligible observations: **60**

## Matured outcomes

- 5D: 53
- 20D: 38
- 60D: 1

## Horizon metrics

| Horizon | Current IC | Model D IC | Δ D−Current | Current Direction | Model D Direction | D Signals | Bootstrap P(Δ>0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5D | +0.108 | +0.091 | -0.017 | 61.1% | 75.0% | 20 | n/a |
| 20D | -0.150 | -0.057 | +0.093 | 25.0% | 0.0% | 6 | n/a |
| 60D | n/a | n/a | n/a | n/a | n/a | 0 | n/a |

## Risk-state metrics

- Current Stress AUC 20D: 0.301
- Model D Stress AUC 20D: 0.321

## Pre-registered evaluation gate

- ✅ Freeze integrity: expected Model-D version only: `WTI_MODEL_D_FROZEN_2026-09-02_v1`
- ✅ Source-health OK rate >= 95%: `95.2%`
- ✅ Model D 20D IC > Current 20D IC: `D -0.057 vs Current -0.150`
- ❌ Model D absolute 20D IC > 0: `-0.057`
- ❌ Model D Direction 20D >= 50% with at least 20 extreme signals: `0.0%; signals=6`
- ❌ Model D Stress AUC >= 0.55 and >= Current with adequate event/non-event counts: `D 0.321 vs Current 0.301; events=16, non-events=22`

## Verdict: **NO_DECISION_COLLECTING**

No model decision is allowed yet. The holdout is still collecting matured 20D outcomes.
