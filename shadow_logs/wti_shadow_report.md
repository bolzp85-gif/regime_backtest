# WTI Shadow Holdout Evaluation

Generated UTC: `2026-09-04T13:49:31.028469+00:00`

Stage: **COLLECTING**

Frozen model: `WTI_MODEL_D_FROZEN_2026-09-02_v1`

Eligible observations: **3**

## Matured outcomes

- 5D: 0
- 20D: 0
- 60D: 0

## Horizon metrics

| Horizon | Current IC | Model D IC | Δ D−Current | Current Direction | Model D Direction | D Signals | Bootstrap P(Δ>0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5D | n/a | n/a | n/a | n/a | n/a | 0 | n/a |
| 20D | n/a | n/a | n/a | n/a | n/a | 0 | n/a |
| 60D | n/a | n/a | n/a | n/a | n/a | 0 | n/a |

## Risk-state metrics

- Current Stress AUC 20D: n/a
- Model D Stress AUC 20D: n/a

## Pre-registered evaluation gate

- ✅ Freeze integrity: expected Model-D version only: `WTI_MODEL_D_FROZEN_2026-09-02_v1`
- ✅ Source-health OK rate >= 95%: `100.0%`
- ❌ Model D 20D IC > Current 20D IC: `n/a`
- ❌ Model D absolute 20D IC > 0: `n/a`
- ❌ Model D Direction 20D >= 50% with at least 20 extreme signals: `n/a; signals=0`
- ❌ Model D Stress AUC >= 0.55 and >= Current with adequate event/non-event counts: `n/a; events=0, non-events=0`

## Verdict: **NO_DECISION_COLLECTING**

No model decision is allowed yet. The holdout is still collecting matured 20D outcomes.
