"""
Gold D1 + V1 Dual-Output Future Shadow Evaluator v1.0.1
=======================================================

D1 is evaluated only as a Direction Context.
V1 is evaluated only as a future Volatility-State.

The two outputs are intentionally not recombined into one score.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

LOG_PATH = Path(
    "shadow_logs/gold_dual_shadow_log.csv"
)

OUT_SUMMARY = Path(
    "shadow_logs/gold_dual_shadow_evaluation_summary.csv"
)

OUT_D1_BLOCKS = Path(
    "shadow_logs/gold_d1_time_blocks.csv"
)

OUT_V1_BLOCKS = Path(
    "shadow_logs/gold_v1_time_blocks.csv"
)

OUT_V1_PHASE = Path(
    "shadow_logs/gold_v1_phase_auc.csv"
)

OUT_GATE = Path(
    "shadow_logs/gold_dual_shadow_gate.csv"
)

OUT_HISTORY = Path(
    "shadow_logs/gold_dual_shadow_evaluation_history.csv"
)

OUT_REPORT = Path(
    "shadow_logs/gold_dual_shadow_report.md"
)

EXPECTED_MODEL_VERSION = (
    "GOLD_D1_V1_DUAL_FROZEN_2026-09-12_v1"
)

EXPECTED_CORE_SHA256 = (
    "9e9531c22aa7e690a88afb5eecd3822e32875495ba4d18f0e498ea6cd3a3b49c"
)

EXPECTED_FREEZE_SPEC_SHA256 = (
    "db3d89c5da83769d6989b150b5f902075d8f0f1df1eaa010a2646a6fa9106877"
)

EXPECTED_SHADOW_START = pd.Timestamp(
    "2026-09-13"
)

# Stage sizes
MIN_20D_EARLY = 60
MIN_20D_INTERIM = 125
MIN_20D_FORMAL = 252

# D1 future gates
D1_MIN_IC20 = 0.05
D1_MIN_DIRECTION20 = 0.50
D1_MIN_SIGNAL_COUNT_INTERIM = 30
D1_MIN_SIGNAL_COUNT_FORMAL = 60
D1_MIN_BLOCK_POSITIVE_IC_RATE = 0.50

# V1 future gates
V1_MIN_AUC = 0.60
V1_MIN_PHASE_AUC = 0.55
V1_MIN_Q20_Q80_GAP = 0.05
V1_MIN_VOL_RELATION = 0.10
V1_MIN_BLOCK_AUC_ABOVE_HALF_RATE = 0.50

TIME_BLOCK_OBSERVATIONS = 60


def _to_bool(series):
    return (
        series.astype(
            str
        )
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
                "y",
            ]
        )
    )


def _safe_spearman(
    x,
    y,
):
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(
                x,
                errors="coerce",
            ),
            "y": pd.to_numeric(
                y,
                errors="coerce",
            ),
        }
    ).dropna()

    if (
        len(
            frame
        ) < 10
        or frame[
            "x"
        ].nunique() < 2
        or frame[
            "y"
        ].nunique() < 2
    ):
        return np.nan

    return float(
        spearmanr(
            frame[
                "x"
            ],
            frame[
                "y"
            ],
        ).statistic
    )


def _direction_accuracy(
    score,
    forward_return,
    bull_threshold=60.0,
    bear_threshold=40.0,
):
    frame = pd.DataFrame(
        {
            "score": pd.to_numeric(
                score,
                errors="coerce",
            ),
            "ret": pd.to_numeric(
                forward_return,
                errors="coerce",
            ),
        }
    ).dropna()

    signals = frame[
        (
            frame[
                "score"
            ]
            >= bull_threshold
        )
        |
        (
            frame[
                "score"
            ]
            <= bear_threshold
        )
    ].copy()

    if signals.empty:
        return (
            np.nan,
            0,
        )

    correct = (
        (
            (
                signals[
                    "score"
                ]
                >= bull_threshold
            )
            &
            (
                signals[
                    "ret"
                ]
                > 0
            )
        )
        |
        (
            (
                signals[
                    "score"
                ]
                <= bear_threshold
            )
            &
            (
                signals[
                    "ret"
                ]
                < 0
            )
        )
    )

    return (
        float(
            correct.mean()
        ),
        int(
            len(
                signals
            )
        ),
    )


def _binary_auc(
    score,
    event,
):
    frame = pd.DataFrame(
        {
            "score": pd.to_numeric(
                score,
                errors="coerce",
            ),
            "event": pd.to_numeric(
                event,
                errors="coerce",
            ),
        }
    ).dropna()

    if frame.empty:
        return np.nan

    y = frame[
        "event"
    ].astype(
        int
    ).to_numpy()

    # lower health score = higher volatility-event risk
    predictor = -frame[
        "score"
    ].astype(
        float
    ).to_numpy()

    n_pos = int(
        np.sum(
            y == 1
        )
    )

    n_neg = int(
        np.sum(
            y == 0
        )
    )

    if (
        n_pos == 0
        or n_neg == 0
    ):
        return np.nan

    ranks = rankdata(
        predictor,
        method="average",
    )

    pos_rank_sum = float(
        np.sum(
            ranks[
                y == 1
            ]
        )
    )

    u = (
        pos_rank_sum
        - n_pos
        * (
            n_pos + 1
        )
        / 2.0
    )

    return float(
        u
        / (
            n_pos
            * n_neg
        )
    )


def _phase_median_auc(
    score,
    event,
    horizon=20,
):
    frame = pd.DataFrame(
        {
            "score": pd.to_numeric(
                score,
                errors="coerce",
            ),
            "event": pd.to_numeric(
                event,
                errors="coerce",
            ),
        }
    ).dropna()

    rows = []
    values = []

    if len(
        frame
    ) < horizon * 3:
        return (
            np.nan,
            pd.DataFrame(),
        )

    for phase in range(
        int(
            horizon
        )
    ):
        sample = frame.iloc[
            phase::int(
                horizon
            )
        ]

        auc = _binary_auc(
            sample[
                "score"
            ],
            sample[
                "event"
            ],
        )

        rows.append(
            {
                "phase": int(
                    phase
                ),
                "n": int(
                    len(
                        sample
                    )
                ),
                "auc": auc,
            }
        )

        if np.isfinite(
            auc
        ):
            values.append(
                auc
            )

    median_auc = (
        float(
            np.median(
                values
            )
        )
        if values
        else np.nan
    )

    return (
        median_auc,
        pd.DataFrame(
            rows
        ),
    )


def _stage(
    matured_20d,
):
    if matured_20d < MIN_20D_EARLY:
        return "COLLECTING"

    if matured_20d < MIN_20D_INTERIM:
        return "EARLY_READ"

    if matured_20d < MIN_20D_FORMAL:
        return "INTERIM"

    return "FORMAL"


def _d1_metrics(
    sample,
):
    ic20 = _safe_spearman(
        sample[
            "d1_score"
        ],
        sample[
            "fwd_return_20d"
        ],
    )

    direction20, signals = _direction_accuracy(
        sample[
            "d1_score"
        ],
        sample[
            "fwd_return_20d"
        ],
    )

    return {
        "n": int(
            len(
                sample
            )
        ),
        "ic20": ic20,
        "direction20": direction20,
        "signals20": signals,
    }


def _v1_metrics(
    sample,
):
    auc = _binary_auc(
        sample[
            "v1_score"
        ],
        sample[
            "high_vol_event_20d"
        ],
    )

    phase_auc, phase_df = _phase_median_auc(
        sample[
            "v1_score"
        ],
        sample[
            "high_vol_event_20d"
        ],
        horizon=20,
    )

    vol_relation = _safe_spearman(
        sample[
            "v1_score"
        ],
        -pd.to_numeric(
            sample[
                "fwd_realized_vol_20d"
            ],
            errors="coerce",
        ),
    )

    frame = pd.DataFrame(
        {
            "score": pd.to_numeric(
                sample[
                    "v1_score"
                ],
                errors="coerce",
            ),
            "event": pd.to_numeric(
                sample[
                    "high_vol_event_20d"
                ],
                errors="coerce",
            ),
        }
    ).dropna()

    if len(
        frame
    ) >= 20:
        q20 = float(
            frame[
                "score"
            ].quantile(
                0.20
            )
        )

        q80 = float(
            frame[
                "score"
            ].quantile(
                0.80
            )
        )

        low = frame[
            frame[
                "score"
            ]
            <= q20
        ]

        high = frame[
            frame[
                "score"
            ]
            >= q80
        ]

        low_rate = float(
            low[
                "event"
            ].mean()
        )

        high_rate = float(
            high[
                "event"
            ].mean()
        )

        gap = (
            low_rate
            - high_rate
        )
    else:
        low_rate = np.nan
        high_rate = np.nan
        gap = np.nan

    return (
        {
            "n": int(
                len(
                    sample
                )
            ),
            "auc": auc,
            "phase_auc": phase_auc,
            "q20_highvol_rate": low_rate,
            "q80_highvol_rate": high_rate,
            "q20_q80_gap": gap,
            "score_vs_lower_fwdvol": vol_relation,
        },
        phase_df,
    )


def _d1_blocks(
    sample,
):
    columns = [
        "block",
        "start_date",
        "end_date",
        "n",
        "ic20",
        "direction20",
        "signals20",
        "positive_ic20",
    ]

    # During the initial COLLECTING phase there are legitimately no matured
    # 20D observations yet. Treat this as an empty block table rather than
    # attempting to sort a columnless DataFrame.
    if (
        sample is None
        or sample.empty
        or "observation_date" not in sample.columns
    ):
        return pd.DataFrame(
            columns=columns
        )

    ordered = sample.sort_values(
        "observation_date"
    ).reset_index(
        drop=True
    )

    rows = []

    complete = (
        len(
            ordered
        )
        // TIME_BLOCK_OBSERVATIONS
    )

    for i in range(
        complete
    ):
        block = ordered.iloc[
            i
            * TIME_BLOCK_OBSERVATIONS:
            (
                i
                + 1
            )
            * TIME_BLOCK_OBSERVATIONS
        ]

        metrics = _d1_metrics(
            block
        )

        rows.append(
            {
                "block": (
                    f"Block_{i + 1:03d}"
                ),
                "start_date": pd.Timestamp(
                    block[
                        "observation_date"
                    ].iloc[
                        0
                    ]
                ).date().isoformat(),
                "end_date": pd.Timestamp(
                    block[
                        "observation_date"
                    ].iloc[
                        -1
                    ]
                ).date().isoformat(),
                **metrics,
                "positive_ic20": bool(
                    np.isfinite(
                        metrics[
                            "ic20"
                        ]
                    )
                    and metrics[
                        "ic20"
                    ] > 0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _v1_blocks(
    sample,
):
    columns = [
        "block",
        "start_date",
        "end_date",
        "n",
        "auc",
        "phase_auc",
        "q20_highvol_rate",
        "q80_highvol_rate",
        "q20_q80_gap",
        "score_vs_lower_fwdvol",
        "auc_above_0_50",
    ]

    # Same initial-holdout guard as D1: zero matured outcomes are expected
    # and must yield an empty block table, not an exception.
    if (
        sample is None
        or sample.empty
        or "observation_date" not in sample.columns
    ):
        return pd.DataFrame(
            columns=columns
        )

    ordered = sample.sort_values(
        "observation_date"
    ).reset_index(
        drop=True
    )

    rows = []

    complete = (
        len(
            ordered
        )
        // TIME_BLOCK_OBSERVATIONS
    )

    for i in range(
        complete
    ):
        block = ordered.iloc[
            i
            * TIME_BLOCK_OBSERVATIONS:
            (
                i
                + 1
            )
            * TIME_BLOCK_OBSERVATIONS
        ]

        metrics, _ = _v1_metrics(
            block
        )

        rows.append(
            {
                "block": (
                    f"Block_{i + 1:03d}"
                ),
                "start_date": pd.Timestamp(
                    block[
                        "observation_date"
                    ].iloc[
                        0
                    ]
                ).date().isoformat(),
                "end_date": pd.Timestamp(
                    block[
                        "observation_date"
                    ].iloc[
                        -1
                    ]
                ).date().isoformat(),
                **metrics,
                "auc_above_0_50": bool(
                    np.isfinite(
                        metrics[
                            "auc"
                        ]
                    )
                    and metrics[
                        "auc"
                    ] > 0.50
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    raw = (
        pd.read_csv(
            LOG_PATH
        )
        if LOG_PATH.exists()
        else pd.DataFrame()
    )

    if not raw.empty:
        raw[
            "observation_date"
        ] = pd.to_datetime(
            raw[
                "observation_date"
            ],
            errors="coerce",
        )

        raw = raw[
            raw[
                "observation_date"
            ]
            >= EXPECTED_SHADOW_START
        ].copy()

    freeze_integrity = bool(
        raw.empty
        or (
            raw[
                "model_version"
            ].dropna().eq(
                EXPECTED_MODEL_VERSION
            ).all()
            and raw[
                "core_sha256"
            ].dropna().eq(
                EXPECTED_CORE_SHA256
            ).all()
            and raw[
                "freeze_spec_sha256"
            ].dropna().eq(
                EXPECTED_FREEZE_SPEC_SHA256
            ).all()
        )
    )

    d1_sample = (
        raw[
            _to_bool(
                raw[
                    "eligible_d1"
                ]
            )
            & pd.to_numeric(
                raw[
                    "fwd_return_20d"
                ],
                errors="coerce",
            ).notna()
        ].copy()
        if not raw.empty
        else pd.DataFrame()
    )

    v1_sample = (
        raw[
            _to_bool(
                raw[
                    "eligible_v1"
                ]
            )
            & pd.to_numeric(
                raw[
                    "fwd_realized_vol_20d"
                ],
                errors="coerce",
            ).notna()
            & pd.to_numeric(
                raw[
                    "high_vol_event_20d"
                ],
                errors="coerce",
            ).notna()
        ].copy()
        if not raw.empty
        else pd.DataFrame()
    )

    matured_20d = int(
        min(
            len(
                d1_sample
            ),
            len(
                v1_sample
            ),
        )
    )

    stage = _stage(
        matured_20d
    )

    d1_metrics = (
        _d1_metrics(
            d1_sample
        )
        if not d1_sample.empty
        else {
            "n": 0,
            "ic20": np.nan,
            "direction20": np.nan,
            "signals20": 0,
        }
    )

    if not v1_sample.empty:
        (
            v1_metrics,
            phase_df,
        ) = _v1_metrics(
            v1_sample
        )
    else:
        v1_metrics = {
            "n": 0,
            "auc": np.nan,
            "phase_auc": np.nan,
            "q20_highvol_rate": np.nan,
            "q80_highvol_rate": np.nan,
            "q20_q80_gap": np.nan,
            "score_vs_lower_fwdvol": np.nan,
        }
        phase_df = pd.DataFrame()

    d1_blocks = _d1_blocks(
        d1_sample
    )

    v1_blocks = _v1_blocks(
        v1_sample
    )

    d1_block_positive_rate = (
        float(
            d1_blocks[
                "positive_ic20"
            ]
            .astype(
                bool
            )
            .mean()
        )
        if not d1_blocks.empty
        else np.nan
    )

    v1_block_auc_rate = (
        float(
            v1_blocks[
                "auc_above_0_50"
            ]
            .astype(
                bool
            )
            .mean()
        )
        if not v1_blocks.empty
        else np.nan
    )

    required_signal_count = (
        D1_MIN_SIGNAL_COUNT_FORMAL
        if stage == "FORMAL"
        else D1_MIN_SIGNAL_COUNT_INTERIM
    )

    gate_rows = [
        {
            "group": "Infrastructure",
            "criterion": "Frozen core/spec integrity",
            "applicable": True,
            "passed": freeze_integrity,
            "value": (
                "OK"
                if freeze_integrity
                else "MISMATCH"
            ),
        },
        {
            "group": "D1 Direction",
            "criterion": "D1 IC20 >= +0.05",
            "applicable": stage in {
                "INTERIM",
                "FORMAL",
            },
            "passed": bool(
                np.isfinite(
                    d1_metrics[
                        "ic20"
                    ]
                )
                and d1_metrics[
                    "ic20"
                ] >= D1_MIN_IC20
            ),
            "value": (
                f"{d1_metrics['ic20']:+.3f}"
                if np.isfinite(
                    d1_metrics[
                        "ic20"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "group": "D1 Direction",
            "criterion": "D1 Direction20 >=50%",
            "applicable": stage in {
                "INTERIM",
                "FORMAL",
            },
            "passed": bool(
                np.isfinite(
                    d1_metrics[
                        "direction20"
                    ]
                )
                and d1_metrics[
                    "direction20"
                ] >= D1_MIN_DIRECTION20
            ),
            "value": (
                f"{d1_metrics['direction20']:.1%}"
                if np.isfinite(
                    d1_metrics[
                        "direction20"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "group": "D1 Direction",
            "criterion": "Enough extreme D1 signals",
            "applicable": stage in {
                "INTERIM",
                "FORMAL",
            },
            "passed": bool(
                int(
                    d1_metrics[
                        "signals20"
                    ]
                )
                >= int(
                    required_signal_count
                )
            ),
            "value": (
                f"{int(d1_metrics['signals20'])} / "
                f"{int(required_signal_count)}"
            ),
        },
        {
            "group": "D1 Direction",
            "criterion": "Positive IC in >=50% complete 60-observation blocks",
            "applicable": (
                stage == "FORMAL"
                and len(
                    d1_blocks
                ) >= 4
            ),
            "passed": bool(
                np.isfinite(
                    d1_block_positive_rate
                )
                and d1_block_positive_rate
                >= D1_MIN_BLOCK_POSITIVE_IC_RATE
            ),
            "value": (
                f"{d1_block_positive_rate:.1%} · blocks={len(d1_blocks)}"
                if np.isfinite(
                    d1_block_positive_rate
                )
                else f"n/a · blocks={len(d1_blocks)}"
            ),
        },
        {
            "group": "V1 Volatility",
            "criterion": "V1 High-Vol AUC >=0.60",
            "applicable": stage in {
                "INTERIM",
                "FORMAL",
            },
            "passed": bool(
                np.isfinite(
                    v1_metrics[
                        "auc"
                    ]
                )
                and v1_metrics[
                    "auc"
                ] >= V1_MIN_AUC
            ),
            "value": (
                f"{v1_metrics['auc']:.3f}"
                if np.isfinite(
                    v1_metrics[
                        "auc"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "group": "V1 Volatility",
            "criterion": "V1 phase-median AUC >=0.55",
            "applicable": stage in {
                "INTERIM",
                "FORMAL",
            },
            "passed": bool(
                np.isfinite(
                    v1_metrics[
                        "phase_auc"
                    ]
                )
                and v1_metrics[
                    "phase_auc"
                ] >= V1_MIN_PHASE_AUC
            ),
            "value": (
                f"{v1_metrics['phase_auc']:.3f}"
                if np.isfinite(
                    v1_metrics[
                        "phase_auc"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "group": "V1 Volatility",
            "criterion": "V1 Q20-Q80 High-Vol gap >=5pp",
            "applicable": stage in {
                "INTERIM",
                "FORMAL",
            },
            "passed": bool(
                np.isfinite(
                    v1_metrics[
                        "q20_q80_gap"
                    ]
                )
                and v1_metrics[
                    "q20_q80_gap"
                ] >= V1_MIN_Q20_Q80_GAP
            ),
            "value": (
                f"{v1_metrics['q20_q80_gap']:+.1%}"
                if np.isfinite(
                    v1_metrics[
                        "q20_q80_gap"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "group": "V1 Volatility",
            "criterion": "V1 score relates >=+0.10 to lower future volatility",
            "applicable": stage in {
                "INTERIM",
                "FORMAL",
            },
            "passed": bool(
                np.isfinite(
                    v1_metrics[
                        "score_vs_lower_fwdvol"
                    ]
                )
                and v1_metrics[
                    "score_vs_lower_fwdvol"
                ] >= V1_MIN_VOL_RELATION
            ),
            "value": (
                f"{v1_metrics['score_vs_lower_fwdvol']:+.3f}"
                if np.isfinite(
                    v1_metrics[
                        "score_vs_lower_fwdvol"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "group": "V1 Volatility",
            "criterion": "AUC >0.50 in >=50% complete 60-observation blocks",
            "applicable": (
                stage == "FORMAL"
                and len(
                    v1_blocks
                ) >= 4
            ),
            "passed": bool(
                np.isfinite(
                    v1_block_auc_rate
                )
                and v1_block_auc_rate
                >= V1_MIN_BLOCK_AUC_ABOVE_HALF_RATE
            ),
            "value": (
                f"{v1_block_auc_rate:.1%} · blocks={len(v1_blocks)}"
                if np.isfinite(
                    v1_block_auc_rate
                )
                else f"n/a · blocks={len(v1_blocks)}"
            ),
        },
    ]

    gate_df = pd.DataFrame(
        gate_rows
    )

    applicable_model_gates = gate_df[
        (
            gate_df[
                "group"
            ]
            != "Infrastructure"
        )
        &
        gate_df[
            "applicable"
        ].astype(
            bool
        )
    ]

    if stage == "COLLECTING":
        verdict = "NO_DECISION_COLLECTING"
    elif stage == "EARLY_READ":
        verdict = "NO_DECISION_EARLY_READ"
    elif applicable_model_gates.empty:
        verdict = "NO_DECISION_INSUFFICIENT_GATES"
    elif (
        freeze_integrity
        and applicable_model_gates[
            "passed"
        ].all()
    ):
        verdict = (
            "INTERIM_PASS"
            if stage == "INTERIM"
            else "FORMAL_PASS"
        )
    else:
        passed_rate = float(
            applicable_model_gates[
                "passed"
            ].mean()
        )

        if (
            freeze_integrity
            and passed_rate >= 0.75
        ):
            verdict = (
                "INTERIM_MIXED"
                if stage == "INTERIM"
                else "FORMAL_MIXED"
            )
        else:
            verdict = (
                "INTERIM_NOT_CONFIRMED"
                if stage == "INTERIM"
                else "FORMAL_NOT_CONFIRMED"
            )

    generated_utc = datetime.now(
        timezone.utc
    ).isoformat()

    last_date = (
        pd.Timestamp(
            raw[
                "observation_date"
            ].max()
        ).date().isoformat()
        if not raw.empty
        else ""
    )

    summary = pd.DataFrame(
        [
            {
                "generated_utc": generated_utc,
                "stage": stage,
                "verdict": verdict,
                "model_version": EXPECTED_MODEL_VERSION,
                "shadow_start_date": EXPECTED_SHADOW_START.date().isoformat(),
                "freeze_integrity": freeze_integrity,
                "last_observation_date": last_date,
                "d1_matured_20d": int(
                    len(
                        d1_sample
                    )
                ),
                "d1_ic20": d1_metrics[
                    "ic20"
                ],
                "d1_direction20": d1_metrics[
                    "direction20"
                ],
                "d1_signals20": d1_metrics[
                    "signals20"
                ],
                "d1_complete_blocks": int(
                    len(
                        d1_blocks
                    )
                ),
                "d1_positive_block_rate": d1_block_positive_rate,
                "v1_matured_20d": int(
                    len(
                        v1_sample
                    )
                ),
                "v1_high_vol_auc": v1_metrics[
                    "auc"
                ],
                "v1_phase_auc": v1_metrics[
                    "phase_auc"
                ],
                "v1_q20_q80_gap": v1_metrics[
                    "q20_q80_gap"
                ],
                "v1_score_vs_lower_fwdvol": v1_metrics[
                    "score_vs_lower_fwdvol"
                ],
                "v1_complete_blocks": int(
                    len(
                        v1_blocks
                    )
                ),
                "v1_auc_above_half_block_rate": v1_block_auc_rate,
            }
        ]
    )

    OUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    d1_blocks.to_csv(
        OUT_D1_BLOCKS,
        index=False,
    )

    v1_blocks.to_csv(
        OUT_V1_BLOCKS,
        index=False,
    )

    phase_df.to_csv(
        OUT_V1_PHASE,
        index=False,
    )

    gate_df.to_csv(
        OUT_GATE,
        index=False,
    )

    if OUT_HISTORY.exists():
        history = pd.read_csv(
            OUT_HISTORY
        )

        if "last_observation_date" in history.columns:
            history = history[
                history[
                    "last_observation_date"
                ].astype(
                    str
                )
                != str(
                    last_date
                )
            ]

        history = pd.concat(
            [
                history,
                summary,
            ],
            ignore_index=True,
        )
    else:
        history = summary.copy()

    history.to_csv(
        OUT_HISTORY,
        index=False,
    )

    report = [
        "# Gold D1 + V1 Dual-Output Future Shadow",
        "",
        f"Generated UTC: `{generated_utc}`",
        "",
        f"Stage: **{stage}**",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"Shadow start: **{EXPECTED_SHADOW_START.date().isoformat()}**",
        "",
        "## D1 Direction Context",
        "",
        (
            f"- Matured 20D: **{len(d1_sample)}**"
        ),
        (
            f"- IC20: `{d1_metrics['ic20']:+.3f}`"
            if np.isfinite(
                d1_metrics[
                    "ic20"
                ]
            )
            else "- IC20: `n/a`"
        ),
        (
            f"- Direction20: `{d1_metrics['direction20']:.1%}`"
            if np.isfinite(
                d1_metrics[
                    "direction20"
                ]
            )
            else "- Direction20: `n/a`"
        ),
        (
            f"- Extreme signals: **{int(d1_metrics['signals20'])}**"
        ),
        "",
        "## V1 Volatility-State",
        "",
        (
            f"- Matured 20D: **{len(v1_sample)}**"
        ),
        (
            f"- High-Vol AUC: `{v1_metrics['auc']:.3f}`"
            if np.isfinite(
                v1_metrics[
                    "auc"
                ]
            )
            else "- High-Vol AUC: `n/a`"
        ),
        (
            f"- Phase-median AUC: `{v1_metrics['phase_auc']:.3f}`"
            if np.isfinite(
                v1_metrics[
                    "phase_auc"
                ]
            )
            else "- Phase-median AUC: `n/a`"
        ),
        (
            f"- Q20-Q80 High-Vol gap: `{v1_metrics['q20_q80_gap']:+.1%}`"
            if np.isfinite(
                v1_metrics[
                    "q20_q80_gap"
                ]
            )
            else "- Q20-Q80 High-Vol gap: `n/a`"
        ),
        (
            f"- Score vs lower FwdVol: "
            f"`{v1_metrics['score_vs_lower_fwdvol']:+.3f}`"
            if np.isfinite(
                v1_metrics[
                    "score_vs_lower_fwdvol"
                ]
            )
            else "- Score vs lower FwdVol: `n/a`"
        ),
        "",
        "## Gates",
        "",
    ]

    for _, row in gate_df.iterrows():
        applicable = bool(
            row[
                "applicable"
            ]
        )

        passed = bool(
            row[
                "passed"
            ]
        )

        icon = (
            "✅"
            if (
                applicable
                and passed
            )
            else (
                "❌"
                if applicable
                else "⏳"
            )
        )

        report.append(
            f"- {icon} **{row['group']}** — "
            f"{row['criterion']}: `{row['value']}`"
        )

    report.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "D1 and V1 are evaluated as separate outputs. "
                "No combined Gold score is inferred from this shadow."
            ),
            "",
            (
                "V1 high-vol events use the PIT threshold stored on the "
                "original observation date. The threshold is never "
                "re-estimated when the future outcome matures."
            ),
            "",
        ]
    )

    OUT_REPORT.write_text(
        "\n".join(
            report
        ),
        encoding="utf-8",
    )

    print(
        "Gold D1 + V1 dual-output evaluator completed."
    )

    print(
        f"Stage: {stage}"
    )

    print(
        f"Verdict: {verdict}"
    )


if __name__ == "__main__":
    main()
