# EUR/USD Frozen A/B/E Shadow Holdout

Generated UTC: `2026-09-12T00:13:09.335141+00:00`

Stage: **COLLECTING**

Verdict: **NO_DECISION_COLLECTING**

Frozen model version: `EURUSD_A_B_E_FROZEN_2026-09-06_v1`

Shadow start: **2026-09-07**

Eligible observations: **5**

## Matured outcomes

- 5D: 0
- 20D: 0
- 60D: 0

## B — absolute and relative Direction evidence

- Absolute IC20: `n/a`
- Absolute IC60: `n/a`
- ΔIC20 B−Current: `n/a`
- ΔIC60 B−Current: `n/a`
- Extreme-score Direction20: `n/a` on `0` signals

## E — Risk-State evidence

- Current Stress AUC: `n/a`
- B Stress AUC: `n/a`
- E Stress AUC: `n/a`
- Stress events / non-events: `0 / 0`

## Fixed 60-observation stability blocks

- Complete blocks: **0**
- B IC20 positive-block rate: `n/a`
- B IC60 positive-block rate: `n/a`
- E AUC > Current block rate: `n/a`

## Gate

- ✅ **Infrastructure** — Frozen model/core/config integrity: `OK`
- ✅ **Infrastructure** — Source-health OK rate >= 95%: `100.0%`
- ⏳ **B Absolute Direction** — B absolute IC20 > 0: `n/a`
- ⏳ **B Absolute Direction** — B absolute IC60 > 0: `n/a`
- ⏳ **B Relative Direction** — B IC20 > Current and bootstrap P >= 75%: `n/a`
- ⏳ **B Relative Direction** — B IC60 > Current and bootstrap P >= 75%: `n/a`
- ⏳ **B Extreme Direction** — B extreme-score Direction20 >= 50% with >=20 signals: `n/a · signals=0`
- ⏳ **B Non-Overlap** — B non-overlap median IC20 > 0 and > Current: `n/a`
- ⏳ **B Non-Overlap** — B non-overlap median IC60 > 0 and > Current: `n/a`
- ⏳ **B Time Blocks** — B absolute IC20 positive in >=50% of complete 60-observation blocks: `n/a · blocks=0`
- ⏳ **B Time Blocks** — B absolute IC60 positive in >=50% of complete 60-observation blocks: `n/a · blocks=0`
- ⏳ **E Risk-State** — E Stress AUC >= 0.55 and >= Current and >= B: `n/a · events=0 · nonevents=0`
- ⏳ **E Time Blocks** — E Stress AUC > Current in >=50% of evaluable complete time blocks: `n/a · blocks=0`

## Interpretation rule

No production change is allowed during COLLECTING or EARLY_READ. INTERIM/FORMAL results must show absolute B quality, not merely relative improvement.

The historical aggregation effect is explicitly guarded against by non-overlap diagnostics and fixed sequential 60-observation time blocks.
