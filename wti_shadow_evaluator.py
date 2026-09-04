"""
WTI Shadow Evaluator
====================

Read-only evaluator for the true forward WTI shadow holdout.

It does NOT:
- calculate or alter model weights,
- change Current or Model D,
- rewrite shadow observations,
- backfill pre-selection history.

It only reads:
    shadow_logs/wti_shadow_log.csv

and writes evaluation artifacts derived from matured holdout outcomes.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

LOG_PATH = Path("shadow_logs/wti_shadow_log.csv")

OUT_SUMMARY = Path(
    "shadow_logs/wti_shadow_evaluation_summary.csv"
)
OUT_HORIZONS = Path(
    "shadow_logs/wti_shadow_horizon_metrics.csv"
)
OUT_GATE = Path(
    "shadow_logs/wti_shadow_gate.csv"
)
OUT_HISTORY = Path(
    "shadow_logs/wti_shadow_evaluation_history.csv"
)
OUT_REPORT = Path(
    "shadow_logs/wti_shadow_report.md"
)

EXPECTED_MODEL_VERSION = (
    "WTI_MODEL_D_FROZEN_2026-09-02_v1"
)
EXPECTED_SHADOW_START = pd.Timestamp(
    "2026-09-02"
)

# ------------------------------------------------------------
# PRE-REGISTERED SHADOW INTERPRETATION RULES
# ------------------------------------------------------------
#
# These rules govern INTERIM / FORMAL evaluation only.
# Before enough 20D outcomes exist, the evaluator explicitly returns
# COLLECTING rather than PASS/FAIL.
#
MIN_MATURED_20D_EARLY = 60
MIN_MATURED_20D_INTERIM = 125
MIN_MATURED_20D_FORMAL = 252

MIN_DIRECTION_20D = 0.50
MIN_DIRECTION_SIGNALS = 20

MIN_STRESS_AUC = 0.55
MIN_STRESS_EVENTS = 5
MIN_STRESS_NONEVENTS = 20

MIN_SOURCE_OK_RATE = 0.95

BULL_THRESHOLD = 60.0
BEAR_THRESHOLD = 40.0

BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260902


def _to_bool(series):
    return (
        series.astype(str)
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


def _safe_spearman(x, y):
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
        len(frame) < 10
        or frame["x"].nunique() < 2
        or frame["y"].nunique() < 2
    ):
        return np.nan

    return float(
        spearmanr(
            frame["x"],
            frame["y"],
        ).statistic
    )


def _directional_accuracy(
    score,
    forward_return,
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
            frame["score"]
            >= BULL_THRESHOLD
        )
        |
        (
            frame["score"]
            <= BEAR_THRESHOLD
        )
    ].copy()

    if signals.empty:
        return np.nan, 0, 0

    correct = (
        (
            (
                signals["score"]
                >= BULL_THRESHOLD
            )
            &
            (
                signals["ret"] > 0
            )
        )
        |
        (
            (
                signals["score"]
                <= BEAR_THRESHOLD
            )
            &
            (
                signals["ret"] < 0
            )
        )
    )

    return (
        float(
            correct.mean()
        ),
        int(
            len(signals)
        ),
        int(
            correct.sum()
        ),
    )


def _binary_auc_lower_score_means_event(
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
        return np.nan, 0, 0

    y = frame["event"].astype(int).to_numpy()
    x = -frame["score"].astype(float).to_numpy()

    n_pos = int(
        np.sum(y == 1)
    )
    n_neg = int(
        np.sum(y == 0)
    )

    if n_pos == 0 or n_neg == 0:
        return np.nan, n_pos, n_neg

    ranks = rankdata(
        x,
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
        -
        n_pos
        * (
            n_pos + 1
        )
        / 2.0
    )

    auc = (
        u
        /
        (
            n_pos
            * n_neg
        )
    )

    return (
        float(auc),
        n_pos,
        n_neg,
    )


def _moving_block_bootstrap_ic_delta(
    current_score,
    model_d_score,
    target,
    block_length,
    n_bootstrap=BOOTSTRAP_REPS,
    seed=BOOTSTRAP_SEED,
):
    frame = pd.DataFrame(
        {
            "current": pd.to_numeric(
                current_score,
                errors="coerce",
            ),
            "model_d": pd.to_numeric(
                model_d_score,
                errors="coerce",
            ),
            "target": pd.to_numeric(
                target,
                errors="coerce",
            ),
        }
    ).dropna()

    n = len(frame)

    observed_current = _safe_spearman(
        frame["current"],
        frame["target"],
    )
    observed_d = _safe_spearman(
        frame["model_d"],
        frame["target"],
    )

    observed_delta = (
        observed_d - observed_current
        if (
            np.isfinite(observed_d)
            and np.isfinite(
                observed_current
            )
        )
        else np.nan
    )

    if n < max(
        80,
        int(block_length) * 4,
    ):
        return {
            "observed_delta": observed_delta,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "prob_positive": np.nan,
            "n": n,
        }

    rng = np.random.default_rng(
        int(seed)
    )

    block_length = int(
        max(
            1,
            block_length,
        )
    )

    max_start = (
        n - block_length
    )

    values = []

    for _ in range(
        int(n_bootstrap)
    ):
        indices = []

        while len(indices) < n:
            start = int(
                rng.integers(
                    0,
                    max_start + 1,
                )
            )

            indices.extend(
                range(
                    start,
                    start
                    + block_length,
                )
            )

        indices = np.asarray(
            indices[:n],
            dtype=int,
        )

        sample = frame.iloc[
            indices
        ]

        current_ic = _safe_spearman(
            sample["current"],
            sample["target"],
        )

        d_ic = _safe_spearman(
            sample["model_d"],
            sample["target"],
        )

        if (
            np.isfinite(current_ic)
            and np.isfinite(d_ic)
        ):
            values.append(
                d_ic - current_ic
            )

    if not values:
        return {
            "observed_delta": observed_delta,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "prob_positive": np.nan,
            "n": n,
        }

    arr = np.asarray(
        values,
        dtype=float,
    )

    return {
        "observed_delta": observed_delta,
        "ci_low": float(
            np.quantile(
                arr,
                0.025,
            )
        ),
        "ci_high": float(
            np.quantile(
                arr,
                0.975,
            )
        ),
        "prob_positive": float(
            np.mean(
                arr > 0
            )
        ),
        "n": n,
    }


def _phase_median_nonoverlap_ic(
    score,
    target,
    horizon,
):
    """
    Daily h-day forward returns overlap heavily.

    To avoid pretending that all daily rows are independent, compute
    non-overlapping samples for every possible phase 0..h-1 and report
    the median phase IC as a robustness diagnostic.
    """
    frame = pd.DataFrame(
        {
            "score": pd.to_numeric(
                score,
                errors="coerce",
            ),
            "target": pd.to_numeric(
                target,
                errors="coerce",
            ),
        }
    ).dropna()

    if len(frame) < (
        int(horizon) * 2
    ):
        return np.nan, 0

    values = []

    for phase in range(
        int(horizon)
    ):
        sample = frame.iloc[
            phase::int(
                horizon
            )
        ]

        ic = _safe_spearman(
            sample["score"],
            sample["target"],
        )

        if np.isfinite(ic):
            values.append(ic)

    if not values:
        return np.nan, 0

    return (
        float(
            np.median(
                values
            )
        ),
        int(
            len(values)
        ),
    )


def _evaluation_stage(
    matured_20d,
):
    matured_20d = int(
        matured_20d
    )

    if matured_20d < MIN_MATURED_20D_EARLY:
        return "COLLECTING"

    if matured_20d < MIN_MATURED_20D_INTERIM:
        return "EARLY_READ"

    if matured_20d < MIN_MATURED_20D_FORMAL:
        return "INTERIM"

    return "FORMAL_1Y"


def _write_markdown_report(
    summary,
    horizon_df,
    gate_df,
):
    lines = []

    lines.append(
        "# WTI Shadow Holdout Evaluation"
    )
    lines.append("")
    lines.append(
        f"Generated UTC: `{summary['generated_utc']}`"
    )
    lines.append("")
    lines.append(
        f"Stage: **{summary['evaluation_stage']}**"
    )
    lines.append("")
    lines.append(
        f"Frozen model: `{summary['model_version']}`"
    )
    lines.append("")
    lines.append(
        f"Eligible observations: **{summary['eligible_observations']}**"
    )
    lines.append("")
    lines.append(
        "## Matured outcomes"
    )
    lines.append("")
    lines.append(
        f"- 5D: {summary['matured_5d']}"
    )
    lines.append(
        f"- 20D: {summary['matured_20d']}"
    )
    lines.append(
        f"- 60D: {summary['matured_60d']}"
    )
    lines.append("")
    lines.append(
        "## Horizon metrics"
    )
    lines.append("")
    lines.append(
        "| Horizon | Current IC | Model D IC | Δ D−Current | Current Direction | Model D Direction | D Signals | Bootstrap P(Δ>0) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    for _, row in horizon_df.iterrows():
        def fmt_num(
            value,
            fmt,
        ):
            return (
                format(
                    float(value),
                    fmt,
                )
                if pd.notna(value)
                else "n/a"
            )

        lines.append(
            "| "
            + str(
                row["horizon"]
            )
            + " | "
            + fmt_num(
                row["current_ic"],
                "+.3f",
            )
            + " | "
            + fmt_num(
                row["model_d_ic"],
                "+.3f",
            )
            + " | "
            + fmt_num(
                row[
                    "ic_delta_d_minus_current"
                ],
                "+.3f",
            )
            + " | "
            + fmt_num(
                row[
                    "current_direction_accuracy"
                ],
                ".1%",
            )
            + " | "
            + fmt_num(
                row[
                    "model_d_direction_accuracy"
                ],
                ".1%",
            )
            + " | "
            + str(
                int(
                    row[
                        "model_d_direction_signals"
                    ]
                )
            )
            + " | "
            + fmt_num(
                row[
                    "bootstrap_prob_delta_positive"
                ],
                ".1%",
            )
            + " |"
        )

    lines.append("")
    lines.append(
        "## Risk-state metrics"
    )
    lines.append("")
    lines.append(
        f"- Current Stress AUC 20D: "
        f"{summary['current_stress_auc_20d']:.3f}"
        if pd.notna(
            summary[
                "current_stress_auc_20d"
            ]
        )
        else "- Current Stress AUC 20D: n/a"
    )
    lines.append(
        f"- Model D Stress AUC 20D: "
        f"{summary['model_d_stress_auc_20d']:.3f}"
        if pd.notna(
            summary[
                "model_d_stress_auc_20d"
            ]
        )
        else "- Model D Stress AUC 20D: n/a"
    )
    lines.append("")
    lines.append(
        "## Pre-registered evaluation gate"
    )
    lines.append("")

    for _, row in gate_df.iterrows():
        lines.append(
            f"- {'✅' if bool(row['passed']) else '❌'} "
            f"{row['criterion']}: `{row['value']}`"
        )

    lines.append("")
    lines.append(
        f"## Verdict: **{summary['verdict']}**"
    )
    lines.append("")

    if summary[
        "evaluation_stage"
    ] == "COLLECTING":
        lines.append(
            "No model decision is allowed yet. The holdout is still collecting matured 20D outcomes."
        )
    elif summary[
        "evaluation_stage"
    ] == "EARLY_READ":
        lines.append(
            "Early descriptive read only. No production decision is allowed at this stage."
        )
    elif summary[
        "evaluation_stage"
    ] == "INTERIM":
        lines.append(
            "Interim evidence may be reviewed, but Model D remains frozen and Current remains production."
        )
    else:
        lines.append(
            "This is the first formal one-year-style review stage. Production changes still require a deliberate human decision."
        )

    OUT_REPORT.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    if not LOG_PATH.exists():
        raise FileNotFoundError(
            f"Missing shadow log: {LOG_PATH}"
        )

    raw = pd.read_csv(
        LOG_PATH
    )

    required = [
        "observation_date",
        "model_version",
        "asset_price",
        "current_score",
        "model_d_score",
        "eligible_for_holdout",
        "source_health",
        "fwd_return_5d",
        "fwd_return_20d",
        "fwd_return_60d",
        "fwd_mae_20d",
        "fwd_realized_vol_20d",
        "stress_event_20d",
    ]

    missing = [
        col
        for col in required
        if col not in raw.columns
    ]

    if missing:
        raise RuntimeError(
            "Shadow log is missing required columns: "
            + ", ".join(
                missing
            )
        )

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
        not raw.empty
        and raw[
            "model_version"
        ].dropna().nunique()
        == 1
        and str(
            raw[
                "model_version"
            ].dropna().iloc[
                0
            ]
        )
        == EXPECTED_MODEL_VERSION
    )

    eligible = _to_bool(
        raw[
            "eligible_for_holdout"
        ]
    )

    source_ok = (
        raw[
            "source_health"
        ]
        .astype(str)
        .str.upper()
        .eq(
            "OK"
        )
    )

    source_ok_rate = (
        float(
            source_ok.mean()
        )
        if len(
            source_ok
        ) > 0
        else np.nan
    )

    sample = raw[
        eligible
        & source_ok
    ].copy()

    sample = sample.sort_values(
        "observation_date"
    )

    horizon_rows = []

    horizon_map = {
        5: "fwd_return_5d",
        20: "fwd_return_20d",
        60: "fwd_return_60d",
    }

    for horizon, target_col in horizon_map.items():
        matured = sample[
            pd.to_numeric(
                sample[
                    target_col
                ],
                errors="coerce",
            ).notna()
        ].copy()

        current_ic = _safe_spearman(
            matured[
                "current_score"
            ],
            matured[
                target_col
            ],
        )

        d_ic = _safe_spearman(
            matured[
                "model_d_score"
            ],
            matured[
                target_col
            ],
        )

        current_direction, current_signals, current_correct = (
            _directional_accuracy(
                matured[
                    "current_score"
                ],
                matured[
                    target_col
                ],
            )
        )

        d_direction, d_signals, d_correct = (
            _directional_accuracy(
                matured[
                    "model_d_score"
                ],
                matured[
                    target_col
                ],
            )
        )

        boot = _moving_block_bootstrap_ic_delta(
            matured[
                "current_score"
            ],
            matured[
                "model_d_score"
            ],
            matured[
                target_col
            ],
            block_length=horizon,
            seed=(
                BOOTSTRAP_SEED
                + horizon
            ),
        )

        current_nonoverlap_ic, current_phases = (
            _phase_median_nonoverlap_ic(
                matured[
                    "current_score"
                ],
                matured[
                    target_col
                ],
                horizon,
            )
        )

        d_nonoverlap_ic, d_phases = (
            _phase_median_nonoverlap_ic(
                matured[
                    "model_d_score"
                ],
                matured[
                    target_col
                ],
                horizon,
            )
        )

        horizon_rows.append(
            {
                "horizon": f"{horizon}D",
                "matured_observations": int(
                    len(
                        matured
                    )
                ),
                "current_ic": current_ic,
                "model_d_ic": d_ic,
                "ic_delta_d_minus_current": (
                    d_ic - current_ic
                    if (
                        np.isfinite(d_ic)
                        and np.isfinite(
                            current_ic
                        )
                    )
                    else np.nan
                ),
                "current_direction_accuracy": (
                    current_direction
                ),
                "current_direction_signals": (
                    current_signals
                ),
                "current_direction_correct": (
                    current_correct
                ),
                "model_d_direction_accuracy": (
                    d_direction
                ),
                "model_d_direction_signals": (
                    d_signals
                ),
                "model_d_direction_correct": (
                    d_correct
                ),
                "current_nonoverlap_phase_median_ic": (
                    current_nonoverlap_ic
                ),
                "model_d_nonoverlap_phase_median_ic": (
                    d_nonoverlap_ic
                ),
                "nonoverlap_phases_available": int(
                    min(
                        current_phases,
                        d_phases,
                    )
                ),
                "bootstrap_prob_delta_positive": (
                    boot[
                        "prob_positive"
                    ]
                ),
                "bootstrap_ci_low": boot[
                    "ci_low"
                ],
                "bootstrap_ci_high": boot[
                    "ci_high"
                ],
            }
        )

    horizon_df = pd.DataFrame(
        horizon_rows
    )

    matured_5d = int(
        horizon_df.loc[
            horizon_df[
                "horizon"
            ]
            == "5D",
            "matured_observations",
        ].iloc[
            0
        ]
    )

    matured_20d = int(
        horizon_df.loc[
            horizon_df[
                "horizon"
            ]
            == "20D",
            "matured_observations",
        ].iloc[
            0
        ]
    )

    matured_60d = int(
        horizon_df.loc[
            horizon_df[
                "horizon"
            ]
            == "60D",
            "matured_observations",
        ].iloc[
            0
        ]
    )

    stage = _evaluation_stage(
        matured_20d
    )

    risk_sample = sample[
        pd.to_numeric(
            sample[
                "stress_event_20d"
            ],
            errors="coerce",
        ).notna()
    ].copy()

    (
        current_auc,
        current_events,
        current_nonevents,
    ) = _binary_auc_lower_score_means_event(
        risk_sample[
            "current_score"
        ],
        risk_sample[
            "stress_event_20d"
        ],
    )

    (
        d_auc,
        d_events,
        d_nonevents,
    ) = _binary_auc_lower_score_means_event(
        risk_sample[
            "model_d_score"
        ],
        risk_sample[
            "stress_event_20d"
        ],
    )

    current_mae_relation = _safe_spearman(
        risk_sample[
            "current_score"
        ],
        risk_sample[
            "fwd_mae_20d"
        ],
    )

    d_mae_relation = _safe_spearman(
        risk_sample[
            "model_d_score"
        ],
        risk_sample[
            "fwd_mae_20d"
        ],
    )

    current_vol_relation = _safe_spearman(
        risk_sample[
            "current_score"
        ],
        -pd.to_numeric(
            risk_sample[
                "fwd_realized_vol_20d"
            ],
            errors="coerce",
        ),
    )

    d_vol_relation = _safe_spearman(
        risk_sample[
            "model_d_score"
        ],
        -pd.to_numeric(
            risk_sample[
                "fwd_realized_vol_20d"
            ],
            errors="coerce",
        ),
    )

    h20 = horizon_df[
        horizon_df[
            "horizon"
        ]
        == "20D"
    ].iloc[
        0
    ]

    enough_direction_signals = bool(
        int(
            h20[
                "model_d_direction_signals"
            ]
        )
        >= MIN_DIRECTION_SIGNALS
    )

    enough_auc_classes = bool(
        d_events
        >= MIN_STRESS_EVENTS
        and d_nonevents
        >= MIN_STRESS_NONEVENTS
    )

    gate_rows = [
        {
            "criterion": (
                "Freeze integrity: expected Model-D version only"
            ),
            "passed": freeze_integrity,
            "value": EXPECTED_MODEL_VERSION,
        },
        {
            "criterion": (
                "Source-health OK rate >= 95%"
            ),
            "passed": bool(
                np.isfinite(
                    source_ok_rate
                )
                and source_ok_rate
                >= MIN_SOURCE_OK_RATE
            ),
            "value": (
                f"{source_ok_rate:.1%}"
                if np.isfinite(
                    source_ok_rate
                )
                else "n/a"
            ),
        },
        {
            "criterion": (
                "Model D 20D IC > Current 20D IC"
            ),
            "passed": bool(
                np.isfinite(
                    h20[
                        "model_d_ic"
                    ]
                )
                and np.isfinite(
                    h20[
                        "current_ic"
                    ]
                )
                and h20[
                    "model_d_ic"
                ]
                > h20[
                    "current_ic"
                ]
            ),
            "value": (
                f"D {h20['model_d_ic']:+.3f} vs "
                f"Current {h20['current_ic']:+.3f}"
                if (
                    pd.notna(
                        h20[
                            "model_d_ic"
                        ]
                    )
                    and pd.notna(
                        h20[
                            "current_ic"
                        ]
                    )
                )
                else "n/a"
            ),
        },
        {
            "criterion": (
                "Model D absolute 20D IC > 0"
            ),
            "passed": bool(
                np.isfinite(
                    h20[
                        "model_d_ic"
                    ]
                )
                and h20[
                    "model_d_ic"
                ] > 0
            ),
            "value": (
                f"{h20['model_d_ic']:+.3f}"
                if pd.notna(
                    h20[
                        "model_d_ic"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "criterion": (
                "Model D Direction 20D >= 50% "
                "with at least 20 extreme signals"
            ),
            "passed": bool(
                enough_direction_signals
                and np.isfinite(
                    h20[
                        "model_d_direction_accuracy"
                    ]
                )
                and h20[
                    "model_d_direction_accuracy"
                ]
                >= MIN_DIRECTION_20D
            ),
            "value": (
                (
                    f"{h20['model_d_direction_accuracy']:.1%}; "
                    f"signals={int(h20['model_d_direction_signals'])}"
                )
                if pd.notna(
                    h20[
                        "model_d_direction_accuracy"
                    ]
                )
                else (
                    f"n/a; signals="
                    f"{int(h20['model_d_direction_signals'])}"
                )
            ),
        },
        {
            "criterion": (
                "Model D Stress AUC >= 0.55 and >= Current "
                "with adequate event/non-event counts"
            ),
            "passed": bool(
                enough_auc_classes
                and np.isfinite(
                    d_auc
                )
                and np.isfinite(
                    current_auc
                )
                and d_auc
                >= MIN_STRESS_AUC
                and d_auc
                >= current_auc
            ),
            "value": (
                (
                    f"D {d_auc:.3f} vs Current {current_auc:.3f}; "
                    f"events={d_events}, non-events={d_nonevents}"
                )
                if (
                    np.isfinite(
                        d_auc
                    )
                    and np.isfinite(
                        current_auc
                    )
                )
                else (
                    f"n/a; events={d_events}, "
                    f"non-events={d_nonevents}"
                )
            ),
        },
    ]

    gate_df = pd.DataFrame(
        gate_rows
    )

    # No PASS/FAIL decision is permitted before INTERIM.
    model_gate_pass = bool(
        gate_df.iloc[
            2:
        ][
            "passed"
        ].all()
    )

    infrastructure_gate_pass = bool(
        gate_df.iloc[
            :2
        ][
            "passed"
        ].all()
    )

    if stage == "COLLECTING":
        verdict = "NO_DECISION_COLLECTING"

    elif stage == "EARLY_READ":
        verdict = "NO_DECISION_EARLY_READ"

    elif stage == "INTERIM":
        if (
            infrastructure_gate_pass
            and model_gate_pass
        ):
            verdict = "INTERIM_PASS"
        else:
            verdict = "INTERIM_NOT_CONFIRMED"

    else:
        if (
            infrastructure_gate_pass
            and model_gate_pass
        ):
            verdict = "FORMAL_PASS"
        else:
            verdict = "FORMAL_NOT_CONFIRMED"

    last_observation = (
        sample[
            "observation_date"
        ].max()
        if not sample.empty
        else pd.NaT
    )

    summary = {
        "generated_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "evaluation_stage": stage,
        "verdict": verdict,
        "model_version": (
            EXPECTED_MODEL_VERSION
        ),
        "shadow_start_date": (
            EXPECTED_SHADOW_START.date().isoformat()
        ),
        "last_observation_date": (
            last_observation.date().isoformat()
            if pd.notna(
                last_observation
            )
            else ""
        ),
        "eligible_observations": int(
            len(
                sample
            )
        ),
        "matured_5d": matured_5d,
        "matured_20d": matured_20d,
        "matured_60d": matured_60d,
        "source_ok_rate": source_ok_rate,
        "freeze_integrity": freeze_integrity,
        "current_ic_20d": h20[
            "current_ic"
        ],
        "model_d_ic_20d": h20[
            "model_d_ic"
        ],
        "ic_delta_20d": h20[
            "ic_delta_d_minus_current"
        ],
        "current_direction_20d": h20[
            "current_direction_accuracy"
        ],
        "model_d_direction_20d": h20[
            "model_d_direction_accuracy"
        ],
        "model_d_direction_signals_20d": int(
            h20[
                "model_d_direction_signals"
            ]
        ),
        "current_stress_auc_20d": (
            current_auc
        ),
        "model_d_stress_auc_20d": d_auc,
        "stress_events_20d": d_events,
        "stress_nonevents_20d": d_nonevents,
        "current_score_vs_better_mae": (
            current_mae_relation
        ),
        "model_d_score_vs_better_mae": (
            d_mae_relation
        ),
        "current_score_vs_lower_vol": (
            current_vol_relation
        ),
        "model_d_score_vs_lower_vol": (
            d_vol_relation
        ),
    }

    summary_df = pd.DataFrame(
        [
            summary
        ]
    )

    OUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    horizon_df.to_csv(
        OUT_HORIZONS,
        index=False,
    )

    gate_df.to_csv(
        OUT_GATE,
        index=False,
    )

    # Keep one evaluation snapshot per latest observation date.
    history_row = summary_df.copy()

    if OUT_HISTORY.exists():
        history = pd.read_csv(
            OUT_HISTORY
        )

        if (
            "last_observation_date"
            in history.columns
        ):
            history = history[
                history[
                    "last_observation_date"
                ].astype(str)
                != str(
                    summary[
                        "last_observation_date"
                    ]
                )
            ]

        history = pd.concat(
            [
                history,
                history_row,
            ],
            ignore_index=True,
        )

    else:
        history = history_row

    history.to_csv(
        OUT_HISTORY,
        index=False,
    )

    _write_markdown_report(
        summary,
        horizon_df,
        gate_df,
    )

    print(
        "WTI shadow evaluator completed."
    )
    print(
        f"Stage: {stage}"
    )
    print(
        f"Verdict: {verdict}"
    )
    print(
        f"Eligible observations: {len(sample)}"
    )
    print(
        f"Matured 20D: {matured_20d}"
    )
    print(
        f"Report: {OUT_REPORT}"
    )


if __name__ == "__main__":
    main()
