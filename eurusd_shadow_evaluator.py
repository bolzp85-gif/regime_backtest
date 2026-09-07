"""
EUR/USD Frozen A/B/E Shadow Evaluator
=====================================

Read-only evaluator for the true future EUR/USD holdout.

It guards against the historical aggregation effect by separating:
- absolute B IC20 / IC60
- relative B improvement over Current
- 20D / 60D non-overlap
- extreme-score Direction Accuracy
- fixed sequential 60-observation block stability
- E Risk-State quality versus Current and B

No model weights are changed by this evaluator.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

LOG_PATH = Path("shadow_logs/eurusd_shadow_log.csv")

OUT_SUMMARY = Path("shadow_logs/eurusd_shadow_evaluation_summary.csv")
OUT_HORIZONS = Path("shadow_logs/eurusd_shadow_horizon_metrics.csv")
OUT_BLOCKS = Path("shadow_logs/eurusd_shadow_time_blocks.csv")
OUT_NONOVERLAP = Path("shadow_logs/eurusd_shadow_nonoverlap.csv")
OUT_GATE = Path("shadow_logs/eurusd_shadow_gate.csv")
OUT_HISTORY = Path("shadow_logs/eurusd_shadow_evaluation_history.csv")
OUT_REPORT = Path("shadow_logs/eurusd_shadow_report.md")

EXPECTED_MODEL_VERSION = "EURUSD_A_B_E_FROZEN_2026-09-06_v1"
EXPECTED_CORE_SHA256 = "c029aee7cc15a64ed5122b2f5c4160def2abe59502a1f20d1ae2680ba1a71695"
EXPECTED_CONFIG_SHA256 = "fb600dcae17347a637e453cac8716abd8c6b3cb44d33c3d53a88ffaec8d9778a"
EXPECTED_SHADOW_START = pd.Timestamp("2026-09-07")

MIN_20D_EARLY = 60
MIN_60D_EARLY = 40
MIN_20D_INTERIM = 125
MIN_60D_INTERIM = 90
MIN_20D_FORMAL = 252
MIN_60D_FORMAL = 240

MIN_SOURCE_OK_RATE = 0.95

BULL_THRESHOLD = 60.0
BEAR_THRESHOLD = 40.0
MIN_EXTREME_SIGNALS = 20
MIN_DIRECTION_20D = 0.50

MIN_BOOTSTRAP_PROB = 0.75

MIN_STRESS_AUC = 0.55
MIN_STRESS_EVENTS = 5
MIN_STRESS_NONEVENTS = 20

TIME_BLOCK_OBSERVATIONS = 60
MIN_COMPLETE_BLOCKS_FOR_STABILITY_GATE = 4
MIN_B_BLOCK_POSITIVE_RATE = 0.50
MIN_E_BLOCK_WIN_RATE = 0.50

BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260906


def _to_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def _safe_spearman(x, y):
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
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


def _extreme_direction_accuracy(score, forward_return):
    frame = pd.DataFrame(
        {
            "score": pd.to_numeric(score, errors="coerce"),
            "ret": pd.to_numeric(forward_return, errors="coerce"),
        }
    ).dropna()

    signals = frame[
        (frame["score"] >= BULL_THRESHOLD)
        | (frame["score"] <= BEAR_THRESHOLD)
    ].copy()

    if signals.empty:
        return np.nan, 0, 0

    correct = (
        (
            (signals["score"] >= BULL_THRESHOLD)
            & (signals["ret"] > 0)
        )
        |
        (
            (signals["score"] <= BEAR_THRESHOLD)
            & (signals["ret"] < 0)
        )
    )

    return (
        float(correct.mean()),
        int(len(signals)),
        int(correct.sum()),
    )


def _stress_auc(score, event):
    frame = pd.DataFrame(
        {
            "score": pd.to_numeric(score, errors="coerce"),
            "event": pd.to_numeric(event, errors="coerce"),
        }
    ).dropna()

    if frame.empty:
        return np.nan, 0, 0

    y = frame["event"].astype(int).to_numpy()
    x = -frame["score"].astype(float).to_numpy()

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    if n_pos == 0 or n_neg == 0:
        return np.nan, n_pos, n_neg

    ranks = rankdata(x, method="average")

    pos_rank_sum = float(
        np.sum(
            ranks[y == 1]
        )
    )

    u = (
        pos_rank_sum
        - n_pos * (n_pos + 1) / 2.0
    )

    return (
        float(u / (n_pos * n_neg)),
        n_pos,
        n_neg,
    )


def _moving_block_bootstrap_delta(
    current_score,
    challenger_score,
    target,
    block_length,
    seed,
):
    frame = pd.DataFrame(
        {
            "current": pd.to_numeric(
                current_score,
                errors="coerce",
            ),
            "challenger": pd.to_numeric(
                challenger_score,
                errors="coerce",
            ),
            "target": pd.to_numeric(
                target,
                errors="coerce",
            ),
        }
    ).dropna()

    n = len(frame)

    current_ic = _safe_spearman(
        frame["current"],
        frame["target"],
    )

    challenger_ic = _safe_spearman(
        frame["challenger"],
        frame["target"],
    )

    observed = (
        challenger_ic - current_ic
        if (
            np.isfinite(challenger_ic)
            and np.isfinite(current_ic)
        )
        else np.nan
    )

    block_length = int(max(1, block_length))

    if n < max(80, block_length * 4):
        return {
            "observed_delta": observed,
            "prob_positive": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": n,
        }

    rng = np.random.default_rng(seed)
    max_start = n - block_length
    values = []

    for _ in range(BOOTSTRAP_REPS):
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
                    start + block_length,
                )
            )

        sample = frame.iloc[
            np.asarray(
                indices[:n],
                dtype=int,
            )
        ]

        a_ic = _safe_spearman(
            sample["current"],
            sample["target"],
        )
        b_ic = _safe_spearman(
            sample["challenger"],
            sample["target"],
        )

        if (
            np.isfinite(a_ic)
            and np.isfinite(b_ic)
        ):
            values.append(
                b_ic - a_ic
            )

    if not values:
        return {
            "observed_delta": observed,
            "prob_positive": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": n,
        }

    arr = np.asarray(
        values,
        dtype=float,
    )

    return {
        "observed_delta": observed,
        "prob_positive": float(
            np.mean(arr > 0)
        ),
        "ci_low": float(
            np.quantile(arr, 0.025)
        ),
        "ci_high": float(
            np.quantile(arr, 0.975)
        ),
        "n": n,
    }


def _phase_nonoverlap(score, target, horizon):
    frame = pd.DataFrame(
        {
            "score": pd.to_numeric(score, errors="coerce"),
            "target": pd.to_numeric(target, errors="coerce"),
        }
    ).dropna()

    if len(frame) < int(horizon) * 2:
        return {
            "median_ic": np.nan,
            "positive_phase_rate": np.nan,
            "phases": 0,
        }

    values = []

    for phase in range(int(horizon)):
        sample = frame.iloc[
            phase::int(horizon)
        ]

        # For non-overlap phase diagnostics each phase is sparse by design.
        # Require at least five independent points per phase; the final
        # statistic is the median across all available phases.
        phase_frame = sample.dropna()

        if (
            len(phase_frame) >= 5
            and phase_frame["score"].nunique() >= 2
            and phase_frame["target"].nunique() >= 2
        ):
            ic = float(
                spearmanr(
                    phase_frame["score"],
                    phase_frame["target"],
                ).statistic
            )
        else:
            ic = np.nan

        if np.isfinite(ic):
            values.append(ic)

    if not values:
        return {
            "median_ic": np.nan,
            "positive_phase_rate": np.nan,
            "phases": 0,
        }

    arr = np.asarray(
        values,
        dtype=float,
    )

    return {
        "median_ic": float(np.median(arr)),
        "positive_phase_rate": float(
            np.mean(arr > 0)
        ),
        "phases": int(len(arr)),
    }


def _evaluation_stage(matured_20d, matured_60d):
    if (
        matured_20d < MIN_20D_EARLY
        or matured_60d < MIN_60D_EARLY
    ):
        return "COLLECTING"

    if (
        matured_20d < MIN_20D_INTERIM
        or matured_60d < MIN_60D_INTERIM
    ):
        return "EARLY_READ"

    if (
        matured_20d < MIN_20D_FORMAL
        or matured_60d < MIN_60D_FORMAL
    ):
        return "INTERIM"

    return "FORMAL"


def _time_block_metrics(sample):
    """
    Fixed sequential blocks of 60 eligible observations.
    Only complete blocks are included.
    """
    matured = sample[
        pd.to_numeric(
            sample["fwd_return_60d"],
            errors="coerce",
        ).notna()
    ].copy()

    matured = matured.sort_values(
        "observation_date"
    ).reset_index(drop=True)

    rows = []

    n_complete = (
        len(matured)
        // TIME_BLOCK_OBSERVATIONS
    )

    for block_number in range(n_complete):
        start = (
            block_number
            * TIME_BLOCK_OBSERVATIONS
        )
        end = start + TIME_BLOCK_OBSERVATIONS
        block = matured.iloc[start:end].copy()

        a_ic20 = _safe_spearman(
            block["current_score"],
            block["fwd_return_20d"],
        )
        b_ic20 = _safe_spearman(
            block["model_b_score"],
            block["fwd_return_20d"],
        )
        a_ic60 = _safe_spearman(
            block["current_score"],
            block["fwd_return_60d"],
        )
        b_ic60 = _safe_spearman(
            block["model_b_score"],
            block["fwd_return_60d"],
        )

        a_auc, events, nonevents = _stress_auc(
            block["current_score"],
            block["stress_event_20d"],
        )
        b_auc, _, _ = _stress_auc(
            block["model_b_score"],
            block["stress_event_20d"],
        )
        e_auc, _, _ = _stress_auc(
            block["model_e_score"],
            block["stress_event_20d"],
        )

        b_direction, b_signals, _ = (
            _extreme_direction_accuracy(
                block["model_b_score"],
                block["fwd_return_20d"],
            )
        )

        rows.append(
            {
                "block": f"Block_{block_number + 1:03d}",
                "start_date": pd.Timestamp(
                    block["observation_date"].iloc[0]
                ).date().isoformat(),
                "end_date": pd.Timestamp(
                    block["observation_date"].iloc[-1]
                ).date().isoformat(),
                "n": len(block),
                "current_ic20": a_ic20,
                "b_ic20": b_ic20,
                "delta_b_minus_current_ic20": (
                    b_ic20 - a_ic20
                    if (
                        np.isfinite(b_ic20)
                        and np.isfinite(a_ic20)
                    )
                    else np.nan
                ),
                "current_ic60": a_ic60,
                "b_ic60": b_ic60,
                "delta_b_minus_current_ic60": (
                    b_ic60 - a_ic60
                    if (
                        np.isfinite(b_ic60)
                        and np.isfinite(a_ic60)
                    )
                    else np.nan
                ),
                "b_direction20_extreme": b_direction,
                "b_extreme_signals": b_signals,
                "current_stress_auc": a_auc,
                "b_stress_auc": b_auc,
                "e_stress_auc": e_auc,
                "stress_events": events,
                "stress_nonevents": nonevents,
                "b_absolute_ic20_positive": bool(
                    np.isfinite(b_ic20)
                    and b_ic20 > 0
                ),
                "b_absolute_ic60_positive": bool(
                    np.isfinite(b_ic60)
                    and b_ic60 > 0
                ),
                "b_beats_current_ic20": bool(
                    np.isfinite(b_ic20)
                    and np.isfinite(a_ic20)
                    and b_ic20 > a_ic20
                ),
                "b_beats_current_ic60": bool(
                    np.isfinite(b_ic60)
                    and np.isfinite(a_ic60)
                    and b_ic60 > a_ic60
                ),
                "e_beats_current_auc": bool(
                    np.isfinite(e_auc)
                    and np.isfinite(a_auc)
                    and e_auc > a_auc
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    raw = (
        pd.read_csv(LOG_PATH)
        if LOG_PATH.exists()
        else pd.DataFrame()
    )

    required = [
        "observation_date",
        "model_version",
        "core_sha256",
        "config_sha256",
        "current_score",
        "model_b_score",
        "model_e_score",
        "eligible_for_holdout",
        "source_health",
        "fwd_return_5d",
        "fwd_return_20d",
        "fwd_return_60d",
        "fwd_mae_20d",
        "fwd_realized_vol_20d",
        "stress_event_20d",
    ]

    if not raw.empty:
        missing = [
            col
            for col in required
            if col not in raw.columns
        ]

        if missing:
            raise RuntimeError(
                "EUR/USD shadow log is missing required columns: "
                + ", ".join(missing)
            )

        raw["observation_date"] = pd.to_datetime(
            raw["observation_date"],
            errors="coerce",
        )

        raw = raw[
            raw["observation_date"]
            >= EXPECTED_SHADOW_START
        ].copy()

    freeze_integrity = bool(
        raw.empty
        or (
            raw["model_version"]
            .dropna()
            .eq(EXPECTED_MODEL_VERSION)
            .all()
            and raw["core_sha256"]
            .dropna()
            .eq(EXPECTED_CORE_SHA256)
            .all()
            and raw["config_sha256"]
            .dropna()
            .eq(EXPECTED_CONFIG_SHA256)
            .all()
        )
    )

    if raw.empty:
        source_ok_rate = np.nan
        sample = raw.copy()
    else:
        source_ok = (
            raw["source_health"]
            .astype(str)
            .str.upper()
            .eq("OK")
        )

        source_ok_rate = float(source_ok.mean())

        sample = raw[
            _to_bool(
                raw["eligible_for_holdout"]
            )
            & source_ok
        ].copy()

        sample = sample.sort_values(
            "observation_date"
        )

    horizon_rows = []

    for horizon in [5, 20, 60]:
        target_col = f"fwd_return_{horizon}d"

        matured = (
            sample[
                pd.to_numeric(
                    sample[target_col],
                    errors="coerce",
                ).notna()
            ].copy()
            if not sample.empty
            else sample.copy()
        )

        target = (
            matured[target_col]
            if not matured.empty
            else pd.Series(dtype=float)
        )

        a_ic = _safe_spearman(
            matured.get(
                "current_score",
                pd.Series(dtype=float),
            ),
            target,
        )
        b_ic = _safe_spearman(
            matured.get(
                "model_b_score",
                pd.Series(dtype=float),
            ),
            target,
        )
        e_ic = _safe_spearman(
            matured.get(
                "model_e_score",
                pd.Series(dtype=float),
            ),
            target,
        )

        a_direction, a_signals, _ = (
            _extreme_direction_accuracy(
                matured.get(
                    "current_score",
                    pd.Series(dtype=float),
                ),
                target,
            )
        )

        b_direction, b_signals, _ = (
            _extreme_direction_accuracy(
                matured.get(
                    "model_b_score",
                    pd.Series(dtype=float),
                ),
                target,
            )
        )

        boot = _moving_block_bootstrap_delta(
            matured.get(
                "current_score",
                pd.Series(dtype=float),
            ),
            matured.get(
                "model_b_score",
                pd.Series(dtype=float),
            ),
            target,
            block_length=horizon,
            seed=BOOTSTRAP_SEED + horizon,
        )

        horizon_rows.append(
            {
                "horizon": f"{horizon}D",
                "matured_observations": len(matured),
                "current_absolute_ic": a_ic,
                "model_b_absolute_ic": b_ic,
                "model_e_absolute_ic": e_ic,
                "delta_b_minus_current_ic": (
                    b_ic - a_ic
                    if (
                        np.isfinite(b_ic)
                        and np.isfinite(a_ic)
                    )
                    else np.nan
                ),
                "current_extreme_direction_accuracy": a_direction,
                "current_extreme_signals": a_signals,
                "model_b_extreme_direction_accuracy": b_direction,
                "model_b_extreme_signals": b_signals,
                "bootstrap_prob_b_better_current": boot[
                    "prob_positive"
                ],
                "bootstrap_delta_ci_low": boot["ci_low"],
                "bootstrap_delta_ci_high": boot["ci_high"],
            }
        )

    horizon_df = pd.DataFrame(horizon_rows)

    def _matured_count(horizon):
        row = horizon_df[
            horizon_df["horizon"] == horizon
        ]
        return (
            int(
                row.iloc[0]["matured_observations"]
            )
            if not row.empty
            else 0
        )

    matured_5d = _matured_count("5D")
    matured_20d = _matured_count("20D")
    matured_60d = _matured_count("60D")

    stage = _evaluation_stage(
        matured_20d,
        matured_60d,
    )

    nonoverlap_rows = []

    for horizon in [20, 60]:
        target_col = f"fwd_return_{horizon}d"

        matured = (
            sample[
                pd.to_numeric(
                    sample[target_col],
                    errors="coerce",
                ).notna()
            ].copy()
            if not sample.empty
            else sample.copy()
        )

        for model_name, score_col in [
            ("A · Current", "current_score"),
            ("B · Full Literature", "model_b_score"),
            ("E · Lit Pillars only", "model_e_score"),
        ]:
            result = _phase_nonoverlap(
                matured.get(
                    score_col,
                    pd.Series(dtype=float),
                ),
                matured.get(
                    target_col,
                    pd.Series(dtype=float),
                ),
                horizon,
            )

            nonoverlap_rows.append(
                {
                    "horizon": f"{horizon}D",
                    "model": model_name,
                    "median_nonoverlap_ic": result["median_ic"],
                    "positive_phase_rate": result[
                        "positive_phase_rate"
                    ],
                    "available_phases": result["phases"],
                }
            )

    nonoverlap_df = pd.DataFrame(
        nonoverlap_rows
    )

    risk = (
        sample[
            pd.to_numeric(
                sample["stress_event_20d"],
                errors="coerce",
            ).notna()
        ].copy()
        if not sample.empty
        else sample.copy()
    )

    a_auc, events, nonevents = _stress_auc(
        risk.get(
            "current_score",
            pd.Series(dtype=float),
        ),
        risk.get(
            "stress_event_20d",
            pd.Series(dtype=float),
        ),
    )
    b_auc, _, _ = _stress_auc(
        risk.get(
            "model_b_score",
            pd.Series(dtype=float),
        ),
        risk.get(
            "stress_event_20d",
            pd.Series(dtype=float),
        ),
    )
    e_auc, _, _ = _stress_auc(
        risk.get(
            "model_e_score",
            pd.Series(dtype=float),
        ),
        risk.get(
            "stress_event_20d",
            pd.Series(dtype=float),
        ),
    )

    block_df = _time_block_metrics(sample)
    n_blocks = len(block_df)

    if n_blocks:
        b_abs_ic20_positive_rate = float(
            block_df[
                "b_absolute_ic20_positive"
            ].astype(bool).mean()
        )
        b_abs_ic60_positive_rate = float(
            block_df[
                "b_absolute_ic60_positive"
            ].astype(bool).mean()
        )
        b_beats_a_ic20_rate = float(
            block_df[
                "b_beats_current_ic20"
            ].astype(bool).mean()
        )
        b_beats_a_ic60_rate = float(
            block_df[
                "b_beats_current_ic60"
            ].astype(bool).mean()
        )

        valid_auc_blocks = block_df[
            pd.to_numeric(
                block_df["e_stress_auc"],
                errors="coerce",
            ).notna()
            &
            pd.to_numeric(
                block_df["current_stress_auc"],
                errors="coerce",
            ).notna()
        ]

        e_auc_block_win_rate = (
            float(
                valid_auc_blocks[
                    "e_beats_current_auc"
                ].astype(bool).mean()
            )
            if not valid_auc_blocks.empty
            else np.nan
        )
    else:
        b_abs_ic20_positive_rate = np.nan
        b_abs_ic60_positive_rate = np.nan
        b_beats_a_ic20_rate = np.nan
        b_beats_a_ic60_rate = np.nan
        e_auc_block_win_rate = np.nan

    h20 = horizon_df[
        horizon_df["horizon"] == "20D"
    ].iloc[0]
    h60 = horizon_df[
        horizon_df["horizon"] == "60D"
    ].iloc[0]

    def _nonoverlap_value(horizon, model):
        rows = nonoverlap_df[
            (
                nonoverlap_df["horizon"] == horizon
            )
            &
            (
                nonoverlap_df["model"] == model
            )
        ]

        return (
            float(
                rows.iloc[0]["median_nonoverlap_ic"]
            )
            if not rows.empty
            else np.nan
        )

    a_no20 = _nonoverlap_value(
        "20D",
        "A · Current",
    )
    b_no20 = _nonoverlap_value(
        "20D",
        "B · Full Literature",
    )
    a_no60 = _nonoverlap_value(
        "60D",
        "A · Current",
    )
    b_no60 = _nonoverlap_value(
        "60D",
        "B · Full Literature",
    )

    enough_blocks = (
        n_blocks
        >= MIN_COMPLETE_BLOCKS_FOR_STABILITY_GATE
    )

    gate_rows = [
        {
            "role": "Infrastructure",
            "criterion": "Frozen model/core/config integrity",
            "applicable": True,
            "passed": freeze_integrity,
            "value": "OK" if freeze_integrity else "MISMATCH",
        },
        {
            "role": "Infrastructure",
            "criterion": "Source-health OK rate >= 95%",
            "applicable": np.isfinite(source_ok_rate),
            "passed": bool(
                np.isfinite(source_ok_rate)
                and source_ok_rate >= MIN_SOURCE_OK_RATE
            ),
            "value": (
                f"{source_ok_rate:.1%}"
                if np.isfinite(source_ok_rate)
                else "n/a"
            ),
        },
        {
            "role": "B Absolute Direction",
            "criterion": "B absolute IC20 > 0",
            "applicable": matured_20d >= MIN_20D_INTERIM,
            "passed": bool(
                np.isfinite(h20["model_b_absolute_ic"])
                and h20["model_b_absolute_ic"] > 0
            ),
            "value": (
                f"{h20['model_b_absolute_ic']:+.3f}"
                if pd.notna(h20["model_b_absolute_ic"])
                else "n/a"
            ),
        },
        {
            "role": "B Absolute Direction",
            "criterion": "B absolute IC60 > 0",
            "applicable": matured_60d >= MIN_60D_INTERIM,
            "passed": bool(
                np.isfinite(h60["model_b_absolute_ic"])
                and h60["model_b_absolute_ic"] > 0
            ),
            "value": (
                f"{h60['model_b_absolute_ic']:+.3f}"
                if pd.notna(h60["model_b_absolute_ic"])
                else "n/a"
            ),
        },
        {
            "role": "B Relative Direction",
            "criterion": "B IC20 > Current and bootstrap P >= 75%",
            "applicable": matured_20d >= MIN_20D_INTERIM,
            "passed": bool(
                np.isfinite(h20["delta_b_minus_current_ic"])
                and h20["delta_b_minus_current_ic"] > 0
                and np.isfinite(
                    h20["bootstrap_prob_b_better_current"]
                )
                and h20[
                    "bootstrap_prob_b_better_current"
                ] >= MIN_BOOTSTRAP_PROB
            ),
            "value": (
                f"Δ={h20['delta_b_minus_current_ic']:+.3f} · "
                f"P={h20['bootstrap_prob_b_better_current']:.1%}"
                if (
                    pd.notna(h20["delta_b_minus_current_ic"])
                    and pd.notna(
                        h20["bootstrap_prob_b_better_current"]
                    )
                )
                else "n/a"
            ),
        },
        {
            "role": "B Relative Direction",
            "criterion": "B IC60 > Current and bootstrap P >= 75%",
            "applicable": matured_60d >= MIN_60D_INTERIM,
            "passed": bool(
                np.isfinite(h60["delta_b_minus_current_ic"])
                and h60["delta_b_minus_current_ic"] > 0
                and np.isfinite(
                    h60["bootstrap_prob_b_better_current"]
                )
                and h60[
                    "bootstrap_prob_b_better_current"
                ] >= MIN_BOOTSTRAP_PROB
            ),
            "value": (
                f"Δ={h60['delta_b_minus_current_ic']:+.3f} · "
                f"P={h60['bootstrap_prob_b_better_current']:.1%}"
                if (
                    pd.notna(h60["delta_b_minus_current_ic"])
                    and pd.notna(
                        h60["bootstrap_prob_b_better_current"]
                    )
                )
                else "n/a"
            ),
        },
        {
            "role": "B Extreme Direction",
            "criterion": (
                "B extreme-score Direction20 >= 50% "
                "with >=20 signals"
            ),
            "applicable": matured_20d >= MIN_20D_INTERIM,
            "passed": bool(
                h20["model_b_extreme_signals"]
                >= MIN_EXTREME_SIGNALS
                and np.isfinite(
                    h20["model_b_extreme_direction_accuracy"]
                )
                and h20[
                    "model_b_extreme_direction_accuracy"
                ] >= MIN_DIRECTION_20D
            ),
            "value": (
                f"{h20['model_b_extreme_direction_accuracy']:.1%} · "
                f"signals={int(h20['model_b_extreme_signals'])}"
                if pd.notna(
                    h20["model_b_extreme_direction_accuracy"]
                )
                else (
                    f"n/a · signals="
                    f"{int(h20['model_b_extreme_signals'])}"
                )
            ),
        },
        {
            "role": "B Non-Overlap",
            "criterion": (
                "B non-overlap median IC20 > 0 and > Current"
            ),
            "applicable": matured_20d >= MIN_20D_INTERIM,
            "passed": bool(
                np.isfinite(b_no20)
                and np.isfinite(a_no20)
                and b_no20 > 0
                and b_no20 > a_no20
            ),
            "value": (
                f"B={b_no20:+.3f} · A={a_no20:+.3f}"
                if (
                    np.isfinite(b_no20)
                    and np.isfinite(a_no20)
                )
                else "n/a"
            ),
        },
        {
            "role": "B Non-Overlap",
            "criterion": (
                "B non-overlap median IC60 > 0 and > Current"
            ),
            "applicable": matured_60d >= MIN_60D_INTERIM,
            "passed": bool(
                np.isfinite(b_no60)
                and np.isfinite(a_no60)
                and b_no60 > 0
                and b_no60 > a_no60
            ),
            "value": (
                f"B={b_no60:+.3f} · A={a_no60:+.3f}"
                if (
                    np.isfinite(b_no60)
                    and np.isfinite(a_no60)
                )
                else "n/a"
            ),
        },
        {
            "role": "B Time Blocks",
            "criterion": (
                "B absolute IC20 positive in >=50% "
                "of complete 60-observation blocks"
            ),
            "applicable": enough_blocks,
            "passed": bool(
                enough_blocks
                and np.isfinite(b_abs_ic20_positive_rate)
                and b_abs_ic20_positive_rate
                >= MIN_B_BLOCK_POSITIVE_RATE
            ),
            "value": (
                f"{b_abs_ic20_positive_rate:.1%} · blocks={n_blocks}"
                if np.isfinite(b_abs_ic20_positive_rate)
                else f"n/a · blocks={n_blocks}"
            ),
        },
        {
            "role": "B Time Blocks",
            "criterion": (
                "B absolute IC60 positive in >=50% "
                "of complete 60-observation blocks"
            ),
            "applicable": enough_blocks,
            "passed": bool(
                enough_blocks
                and np.isfinite(b_abs_ic60_positive_rate)
                and b_abs_ic60_positive_rate
                >= MIN_B_BLOCK_POSITIVE_RATE
            ),
            "value": (
                f"{b_abs_ic60_positive_rate:.1%} · blocks={n_blocks}"
                if np.isfinite(b_abs_ic60_positive_rate)
                else f"n/a · blocks={n_blocks}"
            ),
        },
        {
            "role": "E Risk-State",
            "criterion": (
                "E Stress AUC >= 0.55 and >= Current and >= B"
            ),
            "applicable": (
                matured_20d >= MIN_20D_INTERIM
                and events >= MIN_STRESS_EVENTS
                and nonevents >= MIN_STRESS_NONEVENTS
            ),
            "passed": bool(
                np.isfinite(e_auc)
                and np.isfinite(a_auc)
                and np.isfinite(b_auc)
                and e_auc >= MIN_STRESS_AUC
                and e_auc >= a_auc
                and e_auc >= b_auc
            ),
            "value": (
                f"E={e_auc:.3f} · A={a_auc:.3f} · "
                f"B={b_auc:.3f} · events={events}"
                if (
                    np.isfinite(e_auc)
                    and np.isfinite(a_auc)
                    and np.isfinite(b_auc)
                )
                else (
                    f"n/a · events={events} · "
                    f"nonevents={nonevents}"
                )
            ),
        },
        {
            "role": "E Time Blocks",
            "criterion": (
                "E Stress AUC > Current in >=50% "
                "of evaluable complete time blocks"
            ),
            "applicable": (
                enough_blocks
                and np.isfinite(e_auc_block_win_rate)
            ),
            "passed": bool(
                enough_blocks
                and np.isfinite(e_auc_block_win_rate)
                and e_auc_block_win_rate
                >= MIN_E_BLOCK_WIN_RATE
            ),
            "value": (
                f"{e_auc_block_win_rate:.1%} · blocks={n_blocks}"
                if np.isfinite(e_auc_block_win_rate)
                else f"n/a · blocks={n_blocks}"
            ),
        },
    ]

    gate_df = pd.DataFrame(gate_rows)

    infrastructure = gate_df[
        gate_df["role"] == "Infrastructure"
    ]

    infra_pass = bool(
        not infrastructure.empty
        and infrastructure["passed"].all()
    )

    applicable_model_gates = gate_df[
        gate_df["applicable"].astype(bool)
        & (
            gate_df["role"] != "Infrastructure"
        )
    ]

    if stage == "COLLECTING":
        verdict = "NO_DECISION_COLLECTING"
    elif stage == "EARLY_READ":
        verdict = "NO_DECISION_EARLY_READ"
    elif applicable_model_gates.empty:
        verdict = "NO_DECISION_INSUFFICIENT_GATES"
    else:
        passed_rate = float(
            applicable_model_gates["passed"].mean()
        )

        all_pass = bool(
            applicable_model_gates["passed"].all()
        )

        if infra_pass and all_pass:
            verdict = (
                "INTERIM_PASS"
                if stage == "INTERIM"
                else "FORMAL_PASS"
            )
        elif infra_pass and passed_rate >= 0.75:
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

    last_observation_date = (
        pd.Timestamp(
            sample["observation_date"].max()
        ).date().isoformat()
        if not sample.empty
        else ""
    )

    summary = {
        "generated_utc": generated_utc,
        "evaluation_stage": stage,
        "verdict": verdict,
        "model_version": EXPECTED_MODEL_VERSION,
        "shadow_start_date": (
            EXPECTED_SHADOW_START.date().isoformat()
        ),
        "freeze_integrity": freeze_integrity,
        "source_ok_rate": source_ok_rate,
        "last_observation_date": last_observation_date,
        "eligible_observations": len(sample),
        "matured_5d": matured_5d,
        "matured_20d": matured_20d,
        "matured_60d": matured_60d,
        "current_ic20": h20["current_absolute_ic"],
        "b_absolute_ic20": h20["model_b_absolute_ic"],
        "b_minus_current_ic20": h20[
            "delta_b_minus_current_ic"
        ],
        "current_ic60": h60["current_absolute_ic"],
        "b_absolute_ic60": h60["model_b_absolute_ic"],
        "b_minus_current_ic60": h60[
            "delta_b_minus_current_ic"
        ],
        "b_extreme_direction20": h20[
            "model_b_extreme_direction_accuracy"
        ],
        "b_extreme_signals20": int(
            h20["model_b_extreme_signals"]
        ),
        "current_stress_auc20": a_auc,
        "b_stress_auc20": b_auc,
        "e_stress_auc20": e_auc,
        "stress_events20": events,
        "stress_nonevents20": nonevents,
        "complete_60obs_blocks": n_blocks,
        "b_block_ic20_positive_rate": (
            b_abs_ic20_positive_rate
        ),
        "b_block_ic60_positive_rate": (
            b_abs_ic60_positive_rate
        ),
        "b_block_beats_current_ic20_rate": (
            b_beats_a_ic20_rate
        ),
        "b_block_beats_current_ic60_rate": (
            b_beats_a_ic60_rate
        ),
        "e_block_auc_win_rate": (
            e_auc_block_win_rate
        ),
    }

    summary_df = pd.DataFrame([summary])

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
    block_df.to_csv(
        OUT_BLOCKS,
        index=False,
    )
    nonoverlap_df.to_csv(
        OUT_NONOVERLAP,
        index=False,
    )
    gate_df.to_csv(
        OUT_GATE,
        index=False,
    )

    history_row = summary_df.copy()

    if OUT_HISTORY.exists():
        history = pd.read_csv(OUT_HISTORY)

        if "last_observation_date" in history.columns:
            history = history[
                history["last_observation_date"]
                .astype(str)
                != str(last_observation_date)
            ]

        history = pd.concat(
            [history, history_row],
            ignore_index=True,
        )
    else:
        history = history_row

    history.to_csv(
        OUT_HISTORY,
        index=False,
    )

    lines = [
        "# EUR/USD Frozen A/B/E Shadow Holdout",
        "",
        f"Generated UTC: `{generated_utc}`",
        "",
        f"Stage: **{stage}**",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"Frozen model version: `{EXPECTED_MODEL_VERSION}`",
        "",
        f"Shadow start: **{EXPECTED_SHADOW_START.date().isoformat()}**",
        "",
        f"Eligible observations: **{len(sample)}**",
        "",
        "## Matured outcomes",
        "",
        f"- 5D: {matured_5d}",
        f"- 20D: {matured_20d}",
        f"- 60D: {matured_60d}",
        "",
        "## B — absolute and relative Direction evidence",
        "",
        (
            f"- Absolute IC20: `{h20['model_b_absolute_ic']:+.3f}`"
            if pd.notna(h20["model_b_absolute_ic"])
            else "- Absolute IC20: `n/a`"
        ),
        (
            f"- Absolute IC60: `{h60['model_b_absolute_ic']:+.3f}`"
            if pd.notna(h60["model_b_absolute_ic"])
            else "- Absolute IC60: `n/a`"
        ),
        (
            f"- ΔIC20 B−Current: `{h20['delta_b_minus_current_ic']:+.3f}`"
            if pd.notna(h20["delta_b_minus_current_ic"])
            else "- ΔIC20 B−Current: `n/a`"
        ),
        (
            f"- ΔIC60 B−Current: `{h60['delta_b_minus_current_ic']:+.3f}`"
            if pd.notna(h60["delta_b_minus_current_ic"])
            else "- ΔIC60 B−Current: `n/a`"
        ),
        (
            f"- Extreme-score Direction20: "
            f"`{h20['model_b_extreme_direction_accuracy']:.1%}` "
            f"on `{int(h20['model_b_extreme_signals'])}` signals"
            if pd.notna(
                h20["model_b_extreme_direction_accuracy"]
            )
            else (
                f"- Extreme-score Direction20: `n/a` on "
                f"`{int(h20['model_b_extreme_signals'])}` signals"
            )
        ),
        "",
        "## E — Risk-State evidence",
        "",
        (
            f"- Current Stress AUC: `{a_auc:.3f}`"
            if np.isfinite(a_auc)
            else "- Current Stress AUC: `n/a`"
        ),
        (
            f"- B Stress AUC: `{b_auc:.3f}`"
            if np.isfinite(b_auc)
            else "- B Stress AUC: `n/a`"
        ),
        (
            f"- E Stress AUC: `{e_auc:.3f}`"
            if np.isfinite(e_auc)
            else "- E Stress AUC: `n/a`"
        ),
        f"- Stress events / non-events: `{events} / {nonevents}`",
        "",
        "## Fixed 60-observation stability blocks",
        "",
        f"- Complete blocks: **{n_blocks}**",
        (
            f"- B IC20 positive-block rate: "
            f"`{b_abs_ic20_positive_rate:.1%}`"
            if np.isfinite(b_abs_ic20_positive_rate)
            else "- B IC20 positive-block rate: `n/a`"
        ),
        (
            f"- B IC60 positive-block rate: "
            f"`{b_abs_ic60_positive_rate:.1%}`"
            if np.isfinite(b_abs_ic60_positive_rate)
            else "- B IC60 positive-block rate: `n/a`"
        ),
        (
            f"- E AUC > Current block rate: "
            f"`{e_auc_block_win_rate:.1%}`"
            if np.isfinite(e_auc_block_win_rate)
            else "- E AUC > Current block rate: `n/a`"
        ),
        "",
        "## Gate",
        "",
    ]

    for _, row in gate_df.iterrows():
        applicable = bool(row["applicable"])
        passed = bool(row["passed"])

        icon = (
            "✅"
            if applicable and passed
            else (
                "❌"
                if applicable
                else "⏳"
            )
        )

        lines.append(
            f"- {icon} **{row['role']}** — "
            f"{row['criterion']}: `{row['value']}`"
        )

    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            (
                "No production change is allowed during COLLECTING "
                "or EARLY_READ. INTERIM/FORMAL results must show "
                "absolute B quality, not merely relative improvement."
            ),
            "",
            (
                "The historical aggregation effect is explicitly "
                "guarded against by non-overlap diagnostics and fixed "
                "sequential 60-observation time blocks."
            ),
            "",
        ]
    )

    OUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("EUR/USD shadow evaluator completed.")
    print(f"Stage: {stage}")
    print(f"Verdict: {verdict}")
    print(f"Eligible observations: {len(sample)}")
    print(
        f"Matured 20D / 60D: "
        f"{matured_20d} / {matured_60d}"
    )
    print(f"Report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
