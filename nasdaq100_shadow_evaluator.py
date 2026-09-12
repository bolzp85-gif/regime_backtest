"""
Nasdaq 100 Current-only Risk-State Shadow Evaluator
===================================================

Absolute future validation of the frozen Current Nasdaq 100 Regime score.

Primary question:
Does Current remain a useful Risk-/Stress-State filter on truly future data?

It is NOT evaluated as a directional return predictor.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

LOG_PATH = Path(
    "shadow_logs/nasdaq100_shadow_log.csv"
)

OUT_SUMMARY = Path(
    "shadow_logs/nasdaq100_shadow_evaluation_summary.csv"
)

OUT_BLOCKS = Path(
    "shadow_logs/nasdaq100_shadow_time_blocks.csv"
)

OUT_PHASE = Path(
    "shadow_logs/nasdaq100_shadow_phase_auc.csv"
)

OUT_EPISODES = Path(
    "shadow_logs/nasdaq100_shadow_stress_episodes.csv"
)

OUT_GATE = Path(
    "shadow_logs/nasdaq100_shadow_gate.csv"
)

OUT_HISTORY = Path(
    "shadow_logs/nasdaq100_shadow_evaluation_history.csv"
)

OUT_REPORT = Path(
    "shadow_logs/nasdaq100_shadow_report.md"
)

EXPECTED_MODEL_VERSION = (
    "NASDAQ100_CURRENT_RISK_STATE_FROZEN_2026-09-12_v1"
)

EXPECTED_CORE_SHA256 = (
    "76e74f487bc690ece657e654c334960bfd5c05dd683efff467a6ae5641100d81"
)

EXPECTED_CONFIG_SHA256 = (
    "5c973d98ca10c358c83be883575e678837412017f51088e175f44d0fdf55b122"
)

EXPECTED_SHADOW_START = pd.Timestamp(
    "2026-09-13"
)

MIN_SOURCE_OK_RATE = 0.95

MIN_20D_EARLY = 60
MIN_20D_INTERIM = 125
MIN_20D_FORMAL = 252

MIN_STRESS_AUC = 0.60
MIN_PHASE_AUC = 0.55
MIN_STRESS_GAP = 0.05

TIME_BLOCK_OBSERVATIONS = 60
MIN_COMPLETE_BLOCKS = 4
MIN_BLOCK_AUC_ABOVE_HALF_RATE = 0.50
MIN_BLOCK_POSITIVE_GAP_RATE = 0.50

LOW_SCORE_THRESHOLD = 40.0
HIGH_SCORE_THRESHOLD = 60.0

EARLY_WARNING_LOOKBACK = 20


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


def _stress_auc(score, event):
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

    y = frame[
        "event"
    ].astype(int).to_numpy()

    risk = -frame[
        "score"
    ].astype(float).to_numpy()

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
        return np.nan, n_pos, n_neg

    ranks = rankdata(
        risk,
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

    return (
        float(
            u
            / (
                n_pos
                * n_neg
            )
        ),
        n_pos,
        n_neg,
    )


def _phase_auc(score, event, horizon=20):
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

    values = []
    rows = []

    if len(frame) < horizon * 3:
        return np.nan, pd.DataFrame()

    for phase in range(
        int(horizon)
    ):
        sample = frame.iloc[
            phase::int(horizon)
        ]

        auc, events, nonevents = _stress_auc(
            sample["score"],
            sample["event"],
        )

        rows.append(
            {
                "phase": int(phase),
                "n": int(len(sample)),
                "events": events,
                "nonevents": nonevents,
                "stress_auc": auc,
            }
        )

        if np.isfinite(auc):
            values.append(auc)

    median_auc = (
        float(
            np.median(values)
        )
        if values
        else np.nan
    )

    return median_auc, pd.DataFrame(rows)


def _risk_metrics(sample):
    auc, events, nonevents = _stress_auc(
        sample[
            "current_score"
        ],
        sample[
            "stress_event_20d"
        ],
    )

    phase_auc, phase_df = _phase_auc(
        sample[
            "current_score"
        ],
        sample[
            "stress_event_20d"
        ],
        horizon=20,
    )

    low = sample[
        pd.to_numeric(
            sample[
                "current_score"
            ],
            errors="coerce",
        )
        <= LOW_SCORE_THRESHOLD
    ]

    high = sample[
        pd.to_numeric(
            sample[
                "current_score"
            ],
            errors="coerce",
        )
        >= HIGH_SCORE_THRESHOLD
    ]

    low_rate = (
        float(
            pd.to_numeric(
                low[
                    "stress_event_20d"
                ],
                errors="coerce",
            ).mean()
        )
        if not low.empty
        else np.nan
    )

    high_rate = (
        float(
            pd.to_numeric(
                high[
                    "stress_event_20d"
                ],
                errors="coerce",
            ).mean()
        )
        if not high.empty
        else np.nan
    )

    baseline_rate = float(
        pd.to_numeric(
            sample[
                "stress_event_20d"
            ],
            errors="coerce",
        ).mean()
    )

    stress_gap = (
        low_rate
        - high_rate
        if (
            np.isfinite(low_rate)
            and np.isfinite(high_rate)
        )
        else np.nan
    )

    low_uplift = (
        low_rate
        / baseline_rate
        if (
            np.isfinite(low_rate)
            and np.isfinite(baseline_rate)
            and baseline_rate > 0
        )
        else np.nan
    )

    vol_relation = _safe_spearman(
        sample[
            "current_score"
        ],
        -pd.to_numeric(
            sample[
                "fwd_realized_vol_20d"
            ],
            errors="coerce",
        ),
    )

    mae_relation = _safe_spearman(
        sample[
            "current_score"
        ],
        sample[
            "fwd_mae_20d"
        ],
    )

    return (
        {
            "n": int(len(sample)),
            "stress_auc": auc,
            "phase_median_auc": phase_auc,
            "events": events,
            "nonevents": nonevents,
            "baseline_stress_rate": baseline_rate,
            "low_score_stress_rate": low_rate,
            "high_score_stress_rate": high_rate,
            "stress_gap": stress_gap,
            "low_score_stress_uplift": low_uplift,
            "score_vs_lower_fwdvol": vol_relation,
            "score_vs_better_mae": mae_relation,
        },
        phase_df,
    )


def _time_blocks(sample):
    ordered = sample.sort_values(
        "observation_date"
    ).reset_index(
        drop=True
    )

    rows = []

    n_complete = (
        len(ordered)
        // TIME_BLOCK_OBSERVATIONS
    )

    for block_number in range(
        n_complete
    ):
        start = (
            block_number
            * TIME_BLOCK_OBSERVATIONS
        )

        end = (
            start
            + TIME_BLOCK_OBSERVATIONS
        )

        block = ordered.iloc[
            start:end
        ].copy()

        metrics, _ = _risk_metrics(
            block
        )

        rows.append(
            {
                "block": (
                    f"Block_{block_number + 1:03d}"
                ),
                "start_date": pd.Timestamp(
                    block[
                        "observation_date"
                    ].iloc[0]
                ).date().isoformat(),
                "end_date": pd.Timestamp(
                    block[
                        "observation_date"
                    ].iloc[-1]
                ).date().isoformat(),
                **metrics,
                "auc_above_0_50": bool(
                    np.isfinite(
                        metrics[
                            "stress_auc"
                        ]
                    )
                    and metrics[
                        "stress_auc"
                    ] > 0.50
                ),
                "positive_stress_gap": bool(
                    np.isfinite(
                        metrics[
                            "stress_gap"
                        ]
                    )
                    and metrics[
                        "stress_gap"
                    ] > 0
                ),
            }
        )

    return pd.DataFrame(rows)


def _stress_episodes(sample):
    ordered = sample.sort_values(
        "observation_date"
    ).reset_index(
        drop=True
    )

    event = pd.to_numeric(
        ordered[
            "stress_event_20d"
        ],
        errors="coerce",
    )

    rows = []

    for i in range(
        len(ordered)
    ):
        if not (
            np.isfinite(
                event.iloc[i]
            )
            and int(
                event.iloc[i]
            ) == 1
        ):
            continue

        previous = (
            event.iloc[
                i - 1
            ]
            if i > 0
            else 0
        )

        if (
            np.isfinite(previous)
            and int(previous) == 1
        ):
            continue

        left = max(
            0,
            i - EARLY_WARNING_LOOKBACK
        )

        pre = ordered.iloc[
            left:i
        ]

        alert_positions = [
            j
            for j, score in enumerate(
                pd.to_numeric(
                    pre[
                        "current_score"
                    ],
                    errors="coerce",
                )
            )
            if (
                np.isfinite(score)
                and score <= LOW_SCORE_THRESHOLD
            )
        ]

        if alert_positions:
            first_pos = alert_positions[0]
            last_pos = alert_positions[-1]

            first_lead = (
                i
                - (
                    left
                    + first_pos
                )
            )

            last_lead = (
                i
                - (
                    left
                    + last_pos
                )
            )
        else:
            first_lead = np.nan
            last_lead = np.nan

        rows.append(
            {
                "stress_onset_date": pd.Timestamp(
                    ordered[
                        "observation_date"
                    ].iloc[i]
                ).date().isoformat(),
                "score_at_onset": float(
                    ordered[
                        "current_score"
                    ].iloc[i]
                ),
                "pre_alert_score_le40": bool(
                    alert_positions
                ),
                "first_alert_lead_observations": first_lead,
                "last_alert_lead_observations": last_lead,
            }
        )

    return pd.DataFrame(rows)


def _stage(matured_20d):
    if matured_20d < MIN_20D_EARLY:
        return "COLLECTING"

    if matured_20d < MIN_20D_INTERIM:
        return "EARLY_READ"

    if matured_20d < MIN_20D_FORMAL:
        return "INTERIM"

    return "FORMAL"


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
                "config_sha256"
            ].dropna().eq(
                EXPECTED_CONFIG_SHA256
            ).all()
        )
    )

    if raw.empty:
        source_ok_rate = np.nan
        sample = raw.copy()
    else:
        source_ok = (
            raw[
                "source_health"
            ]
            .astype(str)
            .str.upper()
            .eq("OK")
        )

        source_ok_rate = float(
            source_ok.mean()
        )

        sample = raw[
            _to_bool(
                raw[
                    "eligible_for_holdout"
                ]
            )
            & source_ok
            & pd.to_numeric(
                raw[
                    "stress_event_20d"
                ],
                errors="coerce",
            ).notna()
        ].copy()

    matured_20d = int(
        len(sample)
    )

    stage = _stage(
        matured_20d
    )

    if sample.empty:
        metrics = {
            "n": 0,
            "stress_auc": np.nan,
            "phase_median_auc": np.nan,
            "events": 0,
            "nonevents": 0,
            "baseline_stress_rate": np.nan,
            "low_score_stress_rate": np.nan,
            "high_score_stress_rate": np.nan,
            "stress_gap": np.nan,
            "low_score_stress_uplift": np.nan,
            "score_vs_lower_fwdvol": np.nan,
            "score_vs_better_mae": np.nan,
        }
        phase_df = pd.DataFrame()
    else:
        metrics, phase_df = _risk_metrics(
            sample
        )

    block_df = _time_blocks(
        sample
    )

    episode_df = _stress_episodes(
        sample
    )

    n_blocks = int(
        len(block_df)
    )

    if n_blocks > 0:
        block_auc_rate = float(
            block_df[
                "auc_above_0_50"
            ]
            .astype(bool)
            .mean()
        )

        block_gap_rate = float(
            block_df[
                "positive_stress_gap"
            ]
            .astype(bool)
            .mean()
        )
    else:
        block_auc_rate = np.nan
        block_gap_rate = np.nan

    if not episode_df.empty:
        pre_alert_rate = float(
            episode_df[
                "pre_alert_score_le40"
            ]
            .astype(bool)
            .mean()
        )

        median_first_lead = float(
            pd.to_numeric(
                episode_df[
                    "first_alert_lead_observations"
                ],
                errors="coerce",
            ).median()
        )
    else:
        pre_alert_rate = np.nan
        median_first_lead = np.nan

    gate_rows = [
        {
            "group": "Infrastructure",
            "criterion": "Frozen model/core/config integrity",
            "applicable": True,
            "passed": freeze_integrity,
            "value": (
                "OK"
                if freeze_integrity
                else "MISMATCH"
            ),
        },
        {
            "group": "Infrastructure",
            "criterion": "Source-health OK rate >=95%",
            "applicable": np.isfinite(
                source_ok_rate
            ),
            "passed": bool(
                np.isfinite(
                    source_ok_rate
                )
                and source_ok_rate >= MIN_SOURCE_OK_RATE
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
            "group": "Current Risk-State",
            "criterion": "Absolute Stress AUC >=0.60",
            "applicable": (
                stage in {
                    "INTERIM",
                    "FORMAL",
                }
            ),
            "passed": bool(
                np.isfinite(
                    metrics[
                        "stress_auc"
                    ]
                )
                and metrics[
                    "stress_auc"
                ] >= MIN_STRESS_AUC
            ),
            "value": (
                f"{metrics['stress_auc']:.3f}"
                if np.isfinite(
                    metrics[
                        "stress_auc"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "group": "Current Risk-State",
            "criterion": "Phase-median Stress AUC >=0.55",
            "applicable": (
                stage in {
                    "INTERIM",
                    "FORMAL",
                }
            ),
            "passed": bool(
                np.isfinite(
                    metrics[
                        "phase_median_auc"
                    ]
                )
                and metrics[
                    "phase_median_auc"
                ] >= MIN_PHASE_AUC
            ),
            "value": (
                f"{metrics['phase_median_auc']:.3f}"
                if np.isfinite(
                    metrics[
                        "phase_median_auc"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "group": "Current Risk-State",
            "criterion": "Stressrate Score<=40 minus Score>=60 >=5pp",
            "applicable": (
                stage in {
                    "INTERIM",
                    "FORMAL",
                }
            ),
            "passed": bool(
                np.isfinite(
                    metrics[
                        "stress_gap"
                    ]
                )
                and metrics[
                    "stress_gap"
                ] >= MIN_STRESS_GAP
            ),
            "value": (
                f"{metrics['stress_gap']:+.1%}"
                if np.isfinite(
                    metrics[
                        "stress_gap"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "group": "Current Risk-State",
            "criterion": "Higher score relates to lower forward realized volatility",
            "applicable": (
                stage in {
                    "INTERIM",
                    "FORMAL",
                }
            ),
            "passed": bool(
                np.isfinite(
                    metrics[
                        "score_vs_lower_fwdvol"
                    ]
                )
                and metrics[
                    "score_vs_lower_fwdvol"
                ] > 0
            ),
            "value": (
                f"{metrics['score_vs_lower_fwdvol']:+.3f}"
                if np.isfinite(
                    metrics[
                        "score_vs_lower_fwdvol"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "group": "Current Risk-State",
            "criterion": "Higher score relates to better forward MAE",
            "applicable": (
                stage in {
                    "INTERIM",
                    "FORMAL",
                }
            ),
            "passed": bool(
                np.isfinite(
                    metrics[
                        "score_vs_better_mae"
                    ]
                )
                and metrics[
                    "score_vs_better_mae"
                ] > 0
            ),
            "value": (
                f"{metrics['score_vs_better_mae']:+.3f}"
                if np.isfinite(
                    metrics[
                        "score_vs_better_mae"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "group": "Time-block Stability",
            "criterion": "AUC >0.50 in >=50% of complete 60-observation blocks",
            "applicable": (
                stage == "FORMAL"
                and n_blocks >= MIN_COMPLETE_BLOCKS
            ),
            "passed": bool(
                np.isfinite(
                    block_auc_rate
                )
                and block_auc_rate >= MIN_BLOCK_AUC_ABOVE_HALF_RATE
            ),
            "value": (
                f"{block_auc_rate:.1%} · blocks={n_blocks}"
                if np.isfinite(
                    block_auc_rate
                )
                else f"n/a · blocks={n_blocks}"
            ),
        },
        {
            "group": "Time-block Stability",
            "criterion": "Positive stress-gap in >=50% of complete 60-observation blocks",
            "applicable": (
                stage == "FORMAL"
                and n_blocks >= MIN_COMPLETE_BLOCKS
            ),
            "passed": bool(
                np.isfinite(
                    block_gap_rate
                )
                and block_gap_rate >= MIN_BLOCK_POSITIVE_GAP_RATE
            ),
            "value": (
                f"{block_gap_rate:.1%} · blocks={n_blocks}"
                if np.isfinite(
                    block_gap_rate
                )
                else f"n/a · blocks={n_blocks}"
            ),
        },
    ]

    gate_df = pd.DataFrame(
        gate_rows
    )

    infra = gate_df[
        gate_df[
            "group"
        ]
        == "Infrastructure"
    ]

    model_gates = gate_df[
        (
            gate_df[
                "group"
            ]
            != "Infrastructure"
        )
        &
        gate_df[
            "applicable"
        ].astype(bool)
    ]

    infra_pass = bool(
        not infra.empty
        and infra[
            "passed"
        ].all()
    )

    if stage == "COLLECTING":
        verdict = "NO_DECISION_COLLECTING"
    elif stage == "EARLY_READ":
        verdict = "NO_DECISION_EARLY_READ"
    elif model_gates.empty:
        verdict = "NO_DECISION_INSUFFICIENT_GATES"
    else:
        passed_rate = float(
            model_gates[
                "passed"
            ].mean()
        )

        if (
            infra_pass
            and model_gates[
                "passed"
            ].all()
        ):
            verdict = (
                "INTERIM_PASS"
                if stage == "INTERIM"
                else "FORMAL_PASS"
            )
        elif (
            infra_pass
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

    summary_df = pd.DataFrame(
        [
            {
                "generated_utc": generated_utc,
                "evaluation_stage": stage,
                "verdict": verdict,
                "model_version": EXPECTED_MODEL_VERSION,
                "shadow_start_date": EXPECTED_SHADOW_START.date().isoformat(),
                "freeze_integrity": freeze_integrity,
                "source_ok_rate": source_ok_rate,
                "last_observation_date": last_date,
                "matured_20d": matured_20d,
                **metrics,
                "complete_60obs_blocks": n_blocks,
                "block_auc_above_half_rate": block_auc_rate,
                "block_positive_gap_rate": block_gap_rate,
                "stress_episode_count": int(
                    len(
                        episode_df
                    )
                ),
                "pre_alert_rate_le40": pre_alert_rate,
                "median_first_alert_lead_observations": median_first_lead,
            }
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

    block_df.to_csv(
        OUT_BLOCKS,
        index=False,
    )

    phase_df.to_csv(
        OUT_PHASE,
        index=False,
    )

    episode_df.to_csv(
        OUT_EPISODES,
        index=False,
    )

    gate_df.to_csv(
        OUT_GATE,
        index=False,
    )

    history_row = summary_df.copy()

    if OUT_HISTORY.exists():
        history = pd.read_csv(
            OUT_HISTORY
        )

        if "last_observation_date" in history.columns:
            history = history[
                history[
                    "last_observation_date"
                ].astype(str)
                != str(
                    last_date
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

    lines = [
        "# Nasdaq 100 Current-only Risk-State Future Shadow",
        "",
        f"Generated UTC: `{generated_utc}`",
        "",
        f"Stage: **{stage}**",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"Shadow start: **{EXPECTED_SHADOW_START.date().isoformat()}**",
        "",
        f"Matured 20D observations: **{matured_20d}**",
        "",
        "## Absolute Risk-State metrics",
        "",
        (
            f"- Stress AUC: `{metrics['stress_auc']:.3f}`"
            if np.isfinite(
                metrics[
                    "stress_auc"
                ]
            )
            else "- Stress AUC: `n/a`"
        ),
        (
            f"- Phase-median AUC: `{metrics['phase_median_auc']:.3f}`"
            if np.isfinite(
                metrics[
                    "phase_median_auc"
                ]
            )
            else "- Phase-median AUC: `n/a`"
        ),
        (
            f"- Score<=40 stress rate: `{metrics['low_score_stress_rate']:.1%}`"
            if np.isfinite(
                metrics[
                    "low_score_stress_rate"
                ]
            )
            else "- Score<=40 stress rate: `n/a`"
        ),
        (
            f"- Score>=60 stress rate: `{metrics['high_score_stress_rate']:.1%}`"
            if np.isfinite(
                metrics[
                    "high_score_stress_rate"
                ]
            )
            else "- Score>=60 stress rate: `n/a`"
        ),
        (
            f"- Stress-rate gap: `{metrics['stress_gap']:+.1%}`"
            if np.isfinite(
                metrics[
                    "stress_gap"
                ]
            )
            else "- Stress-rate gap: `n/a`"
        ),
        (
            f"- Score vs lower FwdVol: `{metrics['score_vs_lower_fwdvol']:+.3f}`"
            if np.isfinite(
                metrics[
                    "score_vs_lower_fwdvol"
                ]
            )
            else "- Score vs lower FwdVol: `n/a`"
        ),
        (
            f"- Score vs better FwdMAE: `{metrics['score_vs_better_mae']:+.3f}`"
            if np.isfinite(
                metrics[
                    "score_vs_better_mae"
                ]
            )
            else "- Score vs better FwdMAE: `n/a`"
        ),
        "",
        "## Descriptive early-warning episodes",
        "",
        f"- Stress episode onsets: **{len(episode_df)}**",
        (
            f"- Prior Score<=40 alert rate: `{pre_alert_rate:.1%}`"
            if np.isfinite(
                pre_alert_rate
            )
            else "- Prior Score<=40 alert rate: `n/a`"
        ),
        (
            f"- Median earliest alert lead: "
            f"`{median_first_lead:.1f} eligible observations`"
            if np.isfinite(
                median_first_lead
            )
            else "- Median earliest alert lead: `n/a`"
        ),
        "",
        "## Gate",
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

        lines.append(
            f"- {icon} **{row['group']}** — "
            f"{row['criterion']}: `{row['value']}`"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This experiment validates Current only as a Risk-/Stress-State "
                "filter. It does not test or claim directional return prediction."
            ),
            "",
            (
                "The early-warning episode table is descriptive and is not an "
                "optimization input or promotion gate."
            ),
            "",
        ]
    )

    OUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        "Nasdaq 100 Current-only Risk-State evaluator completed."
    )
    print(
        f"Stage: {stage}"
    )
    print(
        f"Verdict: {verdict}"
    )
    print(
        f"Matured 20D: {matured_20d}"
    )


if __name__ == "__main__":
    main()
