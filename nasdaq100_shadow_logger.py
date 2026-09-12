"""
Nasdaq 100 Current-only Risk-State Shadow Logger
================================================

Frozen on 2026-09-12.
First eligible calendar date: 2026-09-13.

Because 2026-09-13 is a Sunday, the first actual Nasdaq 100 daily market
observation is expected on the next trading day, 2026-09-14.

This logger validates only the frozen Current model. It does not optimize,
compare or modify weights.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from nasdaq100_shadow_core import (
    MODEL_CURRENT,
    STRESS_MAE_THRESHOLDS,
    build_research_dataset,
    build_all_model_scores,
    build_forward_targets,
    current_model_config,
)

ASSET = "Nasdaq 100"

FROZEN_ON_DATE = pd.Timestamp(
    "2026-09-12"
)

SHADOW_START_DATE = pd.Timestamp(
    "2026-09-13"
)

MODEL_VERSION = (
    "NASDAQ100_CURRENT_RISK_STATE_FROZEN_2026-09-12_v1"
)

PIPELINE_VERSION = (
    "ResearchLab_v1.0.18_PIT"
)

EXPECTED_CORE_SHA256 = (
    "76e74f487bc690ece657e654c334960bfd5c05dd683efff467a6ae5641100d81"
)

EXPECTED_CONFIG_SHA256 = (
    "5c973d98ca10c358c83be883575e678837412017f51088e175f44d0fdf55b122"
)

FROZEN_CONFIG_SNAPSHOT = {'pillar_weights': {'Makroökonomie': 0.25, 'Positionierung': 0.15, 'Marktinterna': 0.2, 'Technischer_Trend': 0.2, 'Fundamentale_Faktoren': 0.1, 'Fruehwarnindikatoren': 0.1}, 'sub_weights': {'Makroökonomie': {'fed_policy': 0.2, 'real_yields': 0.3, 'usd_index': 0.2, 'net_liquidity': 0.3}, 'Technischer_Trend': {'distance_200ma': 0.35, 'distance_50ma': 0.35, 'rsi_momentum': 0.3}, 'Fruehwarnindikatoren': {'credit_spreads': 0.5, 'move_index': 0.3, 'vvix_score': 0.2}, 'Positionierung': {'cot_noncommercials': 0.5, 'fear_greed': 0.5}, 'Marktinterna': {'market_momentum': 0.5, 'vix_score': 0.5}, 'Fundamentale_Faktoren': {'pe_valuation': 1.0}}}

MIN_COVERAGE = 60.0
HISTORY_YEARS = 15

# Prevent an unfinished same-day US daily bar from entering the holdout.
SAFE_DAILY_BAR_UTC_HOUR = 22

LOG_DIR = Path(
    "shadow_logs"
)

LOG_PATH = (
    LOG_DIR
    / "nasdaq100_shadow_log.csv"
)

SUMMARY_PATH = (
    LOG_DIR
    / "nasdaq100_shadow_summary.csv"
)

EVENT_NOTE = os.environ.get(
    "NASDAQ100_EVENT_NOTE",
    "",
).strip()

LOG_COLUMNS = [
    "observation_date",
    "logger_run_utc",
    "model_version",
    "pipeline_version",
    "core_sha256",
    "config_sha256",
    "asset",
    "asset_price",
    "current_score",
    "current_coverage",
    "current_regime",
    "eligible_for_holdout",
    "event_note",
    "source_health",
    "source_failures",
    "fwd_return_5d",
    "fwd_return_20d",
    "fwd_return_60d",
    "fwd_mae_20d",
    "fwd_realized_vol_20d",
    "stress_event_20d",
    "matured_5d",
    "matured_20d",
    "matured_60d",
]


def get_regime_label(score):
    try:
        score = float(score)
    except Exception:
        return "n/a"

    if not np.isfinite(score):
        return "n/a"
    if score >= 90:
        return "🟢 Risk-On (Extrem Bullisch)"
    if score >= 75:
        return "🟢 Expansion (Bullisch)"
    if score >= 60:
        return "🟡 Übergangsphase (Leicht Bullisch)"
    if score >= 40:
        return "🟡 Neutral"
    if score >= 25:
        return "🟠 Risk-Off (Bärisch)"
    return "🔴 Stressphase (Stark Bärisch)"


def _safe_float(value):
    try:
        value = float(value)
    except Exception:
        return np.nan

    return (
        value
        if np.isfinite(value)
        else np.nan
    )


def _canonicalize_config(value):
    if isinstance(value, dict):
        return {
            key: _canonicalize_config(value[key])
            for key in sorted(value)
        }

    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return round(float(value), 12)

    if isinstance(value, list):
        return [
            _canonicalize_config(item)
            for item in value
        ]

    return value


def _configs_semantically_equal(actual, expected):
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            return False

        return all(
            _configs_semantically_equal(
                actual[key],
                expected[key],
            )
            for key in actual
        )

    if (
        isinstance(actual, (float, int))
        and not isinstance(actual, bool)
        and isinstance(expected, (float, int))
        and not isinstance(expected, bool)
    ):
        return bool(
            np.isclose(
                float(actual),
                float(expected),
                rtol=0.0,
                atol=1e-12,
            )
        )

    return actual == expected


def _config_fingerprint():
    canonical = _canonicalize_config(
        current_model_config(
            ASSET
        )
    )

    text = json.dumps(
        canonical,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _core_fingerprint():
    path = (
        Path(__file__).resolve().parent
        / "nasdaq100_shadow_core.py"
    )

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _assert_frozen_pipeline():
    actual_core = _core_fingerprint()

    if actual_core != EXPECTED_CORE_SHA256:
        raise RuntimeError(
            "Frozen Nasdaq 100 shadow core changed. "
            "Refusing to contaminate the holdout. "
            f"Expected {EXPECTED_CORE_SHA256}, got {actual_core}."
        )

    actual_config = current_model_config(
        ASSET
    )

    if not _configs_semantically_equal(
        actual_config,
        FROZEN_CONFIG_SNAPSHOT,
    ):
        raise RuntimeError(
            "Frozen Nasdaq 100 Current configuration changed. "
            "Refusing to continue the future holdout. "
            f"Current canonical hash: {_config_fingerprint()}."
        )


def _load_existing_log():
    if not LOG_PATH.exists():
        return pd.DataFrame(
            columns=LOG_COLUMNS
        )

    df = pd.read_csv(
        LOG_PATH
    )

    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df[
        "observation_date"
    ] = pd.to_datetime(
        df[
            "observation_date"
        ],
        errors="coerce",
    )

    return df[
        LOG_COLUMNS
    ].copy()


def _source_health(status):
    if not isinstance(status, dict):
        return (
            "UNKNOWN",
            "status object missing",
        )

    failures = []

    for name, result in status.items():
        if isinstance(result, tuple):
            ok = bool(result[0])
            note = (
                str(result[1])
                if len(result) > 1
                else ""
            )
        else:
            ok = bool(result)
            note = ""

        if not ok:
            failures.append(
                f"{name}: {note}"[:300]
            )

    if not failures:
        return "OK", ""

    return (
        "DEGRADED",
        " | ".join(failures)[:1800],
    )


def _calculate_frames():
    _assert_frozen_pipeline()

    if not os.environ.get(
        "FRED_API_KEY",
        "",
    ).strip():
        raise RuntimeError(
            "FRED_API_KEY is missing. Reuse the existing repository secret."
        )

    today = (
        pd.Timestamp.now(tz="UTC")
        .tz_localize(None)
        .normalize()
    )

    start_date = (
        today
        - pd.DateOffset(
            years=HISTORY_YEARS
        )
        - pd.DateOffset(
            months=3
        )
    ).date()

    raw, status, _pit_quality = (
        build_research_dataset(
            ASSET,
            start_date,
            prefer_first_release=True,
        )
    )

    if raw.empty:
        raise RuntimeError(
            "Nasdaq 100 research dataset is empty."
        )

    (
        _norm_df,
        _configs,
        model_frames,
    ) = build_all_model_scores(
        raw,
        ASSET,
    )

    current_frame = model_frames[
        MODEL_CURRENT
    ]

    targets = build_forward_targets(
        raw[
            "asset_price"
        ],
        ASSET,
    )

    return (
        raw,
        current_frame,
        targets,
        status,
    )


def _append_missing_observations(
    log_df,
    raw,
    current_frame,
    status,
):
    (
        source_health,
        failures,
    ) = _source_health(
        status
    )

    common = pd.DataFrame(
        {
            "asset_price": pd.to_numeric(
                raw["asset_price"],
                errors="coerce",
            ),
            "current_score": pd.to_numeric(
                current_frame[
                    "Final_Regime_Score"
                ],
                errors="coerce",
            ),
            "current_coverage": pd.to_numeric(
                current_frame[
                    "Model_Data_Coverage"
                ],
                errors="coerce",
            ),
        }
    ).dropna(
        subset=[
            "asset_price",
            "current_score",
        ]
    )

    if common.empty:
        raise RuntimeError(
            "No common Nasdaq 100 Current observation is available."
        )

    common.index = pd.to_datetime(
        common.index,
        errors="coerce",
    ).normalize()

    existing_dates = set(
        pd.to_datetime(
            log_df[
                "observation_date"
            ],
            errors="coerce",
        )
        .dropna()
        .dt.normalize()
    )

    now_utc = pd.Timestamp.now(
        tz="UTC"
    )

    today_utc = (
        now_utc
        .tz_localize(None)
        .normalize()
    )

    if int(now_utc.hour) >= SAFE_DAILY_BAR_UTC_HOUR:
        latest_allowed_date = today_utc
    else:
        latest_allowed_date = (
            today_utc
            - pd.Timedelta(days=1)
        )

    candidate = common[
        (
            common.index >= SHADOW_START_DATE
        )
        &
        (
            common.index <= latest_allowed_date
        )
    ]

    run_utc = datetime.now(
        timezone.utc
    ).isoformat()

    rows = []

    for date, row in candidate.iterrows():
        date = pd.Timestamp(
            date
        ).normalize()

        if date in existing_dates:
            continue

        score = _safe_float(
            row[
                "current_score"
            ]
        )

        coverage = _safe_float(
            row[
                "current_coverage"
            ]
        )

        eligible = bool(
            source_health == "OK"
            and np.isfinite(
                coverage
            )
            and coverage >= MIN_COVERAGE
        )

        rows.append(
            {
                "observation_date": date,
                "logger_run_utc": run_utc,
                "model_version": MODEL_VERSION,
                "pipeline_version": PIPELINE_VERSION,
                "core_sha256": EXPECTED_CORE_SHA256,
                "config_sha256": EXPECTED_CONFIG_SHA256,
                "asset": ASSET,
                "asset_price": _safe_float(
                    row[
                        "asset_price"
                    ]
                ),
                "current_score": score,
                "current_coverage": coverage,
                "current_regime": get_regime_label(
                    score
                ),
                "eligible_for_holdout": eligible,
                "event_note": EVENT_NOTE,
                "source_health": source_health,
                "source_failures": failures,
                "fwd_return_5d": np.nan,
                "fwd_return_20d": np.nan,
                "fwd_return_60d": np.nan,
                "fwd_mae_20d": np.nan,
                "fwd_realized_vol_20d": np.nan,
                "stress_event_20d": np.nan,
                "matured_5d": False,
                "matured_20d": False,
                "matured_60d": False,
            }
        )

    if rows:
        log_df = pd.concat(
            [
                log_df,
                pd.DataFrame(rows),
            ],
            ignore_index=True,
        )

    return log_df


def _mature_forward_outcomes(
    log_df,
    targets,
):
    if log_df.empty:
        return log_df

    targets = targets.copy()

    targets.index = pd.to_datetime(
        targets.index,
        errors="coerce",
    ).normalize()

    log_df[
        "observation_date"
    ] = (
        pd.to_datetime(
            log_df[
                "observation_date"
            ],
            errors="coerce",
        )
        .dt.normalize()
    )

    for idx, date in log_df[
        "observation_date"
    ].items():
        if pd.isna(date) or date not in targets.index:
            continue

        t = targets.loc[
            date
        ]

        for log_col, target_col, mature_col in [
            (
                "fwd_return_5d",
                "Fwd_Return_5D",
                "matured_5d",
            ),
            (
                "fwd_return_20d",
                "Fwd_Return_20D",
                "matured_20d",
            ),
            (
                "fwd_return_60d",
                "Fwd_Return_60D",
                "matured_60d",
            ),
        ]:
            value = _safe_float(
                t.get(
                    target_col,
                    np.nan,
                )
            )

            if np.isfinite(value):
                log_df.at[
                    idx,
                    log_col,
                ] = value

                log_df.at[
                    idx,
                    mature_col,
                ] = True

        mae = _safe_float(
            t.get(
                "Fwd_MAE_20D",
                np.nan,
            )
        )

        fwd_vol = _safe_float(
            t.get(
                "Fwd_Realized_Vol_20D",
                np.nan,
            )
        )

        if np.isfinite(mae):
            log_df.at[
                idx,
                "fwd_mae_20d",
            ] = mae

            log_df.at[
                idx,
                "stress_event_20d",
            ] = int(
                mae
                <= STRESS_MAE_THRESHOLDS[
                    ASSET
                ]
            )

        if np.isfinite(fwd_vol):
            log_df.at[
                idx,
                "fwd_realized_vol_20d",
            ] = fwd_vol

    return log_df


def _write_summary(log_df):
    if log_df.empty:
        last_date = ""
        latest_score = np.nan
        latest_coverage = np.nan
        eligible_count = 0

    else:
        ordered = log_df.sort_values(
            "observation_date"
        )

        last = ordered.iloc[
            -1
        ]

        last_date = pd.Timestamp(
            last[
                "observation_date"
            ]
        ).date().isoformat()

        latest_score = _safe_float(
            last[
                "current_score"
            ]
        )

        latest_coverage = _safe_float(
            last[
                "current_coverage"
            ]
        )

        eligible_count = int(
            (
                log_df[
                    "eligible_for_holdout"
                ]
                .astype(str)
                .str.lower()
                .isin(
                    [
                        "true",
                        "1",
                    ]
                )
            ).sum()
        )

    summary = pd.DataFrame(
        [
            {
                "model_version": MODEL_VERSION,
                "frozen_on_date": FROZEN_ON_DATE.date().isoformat(),
                "shadow_start_date": SHADOW_START_DATE.date().isoformat(),
                "last_observation_date": last_date,
                "core_sha256": EXPECTED_CORE_SHA256,
                "config_sha256": EXPECTED_CONFIG_SHA256,
                "observations_total": int(
                    len(log_df)
                ),
                "observations_eligible": eligible_count,
                "matured_5d": int(
                    pd.to_numeric(
                        log_df.get(
                            "fwd_return_5d",
                            pd.Series(dtype=float),
                        ),
                        errors="coerce",
                    ).notna().sum()
                ),
                "matured_20d": int(
                    pd.to_numeric(
                        log_df.get(
                            "fwd_return_20d",
                            pd.Series(dtype=float),
                        ),
                        errors="coerce",
                    ).notna().sum()
                ),
                "matured_60d": int(
                    pd.to_numeric(
                        log_df.get(
                            "fwd_return_60d",
                            pd.Series(dtype=float),
                        ),
                        errors="coerce",
                    ).notna().sum()
                ),
                "latest_current_score": latest_score,
                "latest_current_coverage": latest_coverage,
                "generated_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )


def main():
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        raw,
        current_frame,
        targets,
        status,
    ) = _calculate_frames()

    log_df = _load_existing_log()

    log_df = _append_missing_observations(
        log_df,
        raw,
        current_frame,
        status,
    )

    log_df = _mature_forward_outcomes(
        log_df,
        targets,
    )

    if not log_df.empty:
        log_df = (
            log_df
            .sort_values(
                "observation_date"
            )
            .drop_duplicates(
                subset=[
                    "observation_date",
                    "model_version",
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

    log_df.to_csv(
        LOG_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    _write_summary(
        log_df
    )

    print(
        "Nasdaq 100 Current-only Risk-State shadow logger completed."
    )
    print(
        f"Rows: {len(log_df)}"
    )
    print(
        f"Output: {LOG_PATH}"
    )


if __name__ == "__main__":
    main()
