"""
Gold D1 + V1 Dual-Output Future Shadow Logger
=============================================

Freeze: 2026-09-12
Holdout calendar start: 2026-09-13
First normal Gold daily observation is expected on 2026-09-14.

D1 = Direction Context.
V1 = GVZ Health only, interpreted as Volatility-State health.

No model fitting or weight search occurs in this logger.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from gold_dual_shadow_core import (
    build_research_dataset,
    build_forward_targets,
    build_gold_d1_v1_outputs,
    build_gold_pit_high_vol_threshold,
)

ASSET = "Gold (XAU/USD)"

FROZEN_ON_DATE = pd.Timestamp(
    "2026-09-12"
)

SHADOW_START_DATE = pd.Timestamp(
    "2026-09-13"
)

MODEL_VERSION = (
    "GOLD_D1_V1_DUAL_FROZEN_2026-09-12_v1"
)

PIPELINE_VERSION = (
    "ResearchLab_v1.0.24a_PIT_Fixed"
)

EXPECTED_CORE_SHA256 = (
    "9e9531c22aa7e690a88afb5eecd3822e32875495ba4d18f0e498ea6cd3a3b49c"
)

EXPECTED_FREEZE_SPEC_SHA256 = (
    "db3d89c5da83769d6989b150b5f902075d8f0f1df1eaa010a2646a6fa9106877"
)

FREEZE_SPEC = {'asset': 'Gold (XAU/USD)', 'freeze_date': '2026-09-12', 'holdout_start': '2026-09-13', 'd1_effective_weights': {'g1_macro': 0.35, 'technical_trend': 0.15, 'obv_momentum': 0.075}, 'g1_macro_raw_weights': {'fed_policy': 0.2, 'real_yields': 0.3, 'usd_index': 0.2}, 'g1_transforms': {'fed_policy': '-diff(20)', 'real_yields': '-diff(20)', 'usd_index': '-pct_change(20)', 'net_liquidity': 'excluded'}, 'v1': {'factor': 'vix_score', 'semantic': 'GVZ health only; higher = calmer volatility environment'}, 'pit_high_vol_target': {'backward_realized_vol_window': 20, 'trailing_threshold_lookback': 756, 'threshold_min_periods': 252, 'threshold_quantile': 0.8, 'threshold_shift': 1}, 'min_coverage': 60.0}

MIN_COVERAGE = 60.0
HISTORY_YEARS = 15

# Gold futures update later than the equity cash session.
# A same-day daily bar is only eligible after 23:00 UTC.
SAFE_DAILY_BAR_UTC_HOUR = 23

LOG_DIR = Path(
    "shadow_logs"
)

LOG_PATH = (
    LOG_DIR
    / "gold_dual_shadow_log.csv"
)

SUMMARY_PATH = (
    LOG_DIR
    / "gold_dual_shadow_summary.csv"
)

EVENT_NOTE = os.environ.get(
    "GOLD_EVENT_NOTE",
    "",
).strip()

LOG_COLUMNS = [
    "observation_date",
    "logger_run_utc",
    "model_version",
    "pipeline_version",
    "core_sha256",
    "freeze_spec_sha256",
    "asset",
    "asset_price",
    "d1_score",
    "d1_component_coverage",
    "d1_current_model_coverage",
    "d1_g1_model_coverage",
    "v1_score",
    "v1_coverage",
    "pit_high_vol_threshold",
    "backward_realized_vol_20d",
    "eligible_d1",
    "eligible_v1",
    "event_note",
    "source_health",
    "source_failures",
    "fwd_return_5d",
    "fwd_return_20d",
    "fwd_return_60d",
    "fwd_realized_vol_20d",
    "high_vol_event_20d",
    "matured_5d",
    "matured_20d",
    "matured_60d",
]


def _safe_float(value):
    try:
        value = float(value)
    except Exception:
        return np.nan

    return (
        value
        if np.isfinite(
            value
        )
        else np.nan
    )


def _freeze_spec_fingerprint():
    text = json.dumps(
        FREEZE_SPEC,
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
        / "gold_dual_shadow_core.py"
    )

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _assert_frozen_pipeline():
    actual_core = _core_fingerprint()

    if actual_core != EXPECTED_CORE_SHA256:
        raise RuntimeError(
            "Frozen Gold dual-output core changed. "
            "Refusing to contaminate the holdout. "
            f"Expected {EXPECTED_CORE_SHA256}, got {actual_core}."
        )

    actual_spec = _freeze_spec_fingerprint()

    if actual_spec != EXPECTED_FREEZE_SPEC_SHA256:
        raise RuntimeError(
            "Frozen Gold D1/V1 specification changed. "
            "Refusing to continue the holdout."
        )


def _source_health(status):
    if not isinstance(
        status,
        dict,
    ):
        return (
            "UNKNOWN",
            "status object missing",
        )

    failures = []

    for name, result in status.items():
        if isinstance(
            result,
            tuple,
        ):
            ok = bool(
                result[0]
            )

            note = (
                str(
                    result[1]
                )
                if len(
                    result
                ) > 1
                else ""
            )
        else:
            ok = bool(
                result
            )
            note = ""

        if not ok:
            failures.append(
                f"{name}: {note}"[
                    :300
                ]
            )

    if not failures:
        return (
            "OK",
            "",
        )

    return (
        "DEGRADED",
        " | ".join(
            failures
        )[
            :1800
        ],
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
            df[
                col
            ] = np.nan

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
        pd.Timestamp.now(
            tz="UTC"
        )
        .tz_localize(
            None
        )
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
            "Gold research dataset is empty."
        )

    (
        _norm_df,
        d1,
        v1_score,
        v1_coverage,
    ) = build_gold_d1_v1_outputs(
        raw
    )

    (
        backward_vol,
        pit_threshold,
    ) = build_gold_pit_high_vol_threshold(
        raw[
            "asset_price"
        ]
    )

    targets = build_forward_targets(
        raw[
            "asset_price"
        ],
        ASSET,
    )

    return (
        raw,
        d1,
        v1_score,
        v1_coverage,
        backward_vol,
        pit_threshold,
        targets,
        status,
    )


def _append_missing_observations(
    log_df,
    raw,
    d1,
    v1_score,
    v1_coverage,
    backward_vol,
    pit_threshold,
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
                raw[
                    "asset_price"
                ],
                errors="coerce",
            ),
            "d1_score": pd.to_numeric(
                d1[
                    "d1_score"
                ],
                errors="coerce",
            ),
            "d1_component_coverage": pd.to_numeric(
                d1[
                    "d1_component_coverage"
                ],
                errors="coerce",
            ),
            "d1_current_model_coverage": pd.to_numeric(
                d1[
                    "current_model_coverage"
                ],
                errors="coerce",
            ),
            "d1_g1_model_coverage": pd.to_numeric(
                d1[
                    "g1_model_coverage"
                ],
                errors="coerce",
            ),
            "v1_score": pd.to_numeric(
                v1_score,
                errors="coerce",
            ),
            "v1_coverage": pd.to_numeric(
                v1_coverage,
                errors="coerce",
            ),
            "pit_high_vol_threshold": pd.to_numeric(
                pit_threshold,
                errors="coerce",
            ),
            "backward_realized_vol_20d": pd.to_numeric(
                backward_vol,
                errors="coerce",
            ),
        }
    ).dropna(
        subset=[
            "asset_price"
        ]
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
        .tz_localize(
            None
        )
        .normalize()
    )

    latest_allowed_date = (
        today_utc
        if int(
            now_utc.hour
        )
        >= SAFE_DAILY_BAR_UTC_HOUR
        else (
            today_utc
            - pd.Timedelta(
                days=1
            )
        )
    )

    candidate = common[
        (
            common.index
            >= SHADOW_START_DATE
        )
        &
        (
            common.index
            <= latest_allowed_date
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

        d1_score = _safe_float(
            row[
                "d1_score"
            ]
        )

        d1_component_cov = _safe_float(
            row[
                "d1_component_coverage"
            ]
        )

        current_cov = _safe_float(
            row[
                "d1_current_model_coverage"
            ]
        )

        g1_cov = _safe_float(
            row[
                "d1_g1_model_coverage"
            ]
        )

        v1 = _safe_float(
            row[
                "v1_score"
            ]
        )

        v1_cov = _safe_float(
            row[
                "v1_coverage"
            ]
        )

        threshold = _safe_float(
            row[
                "pit_high_vol_threshold"
            ]
        )

        eligible_d1 = bool(
            np.isfinite(
                d1_score
            )
            and np.isfinite(
                d1_component_cov
            )
            and d1_component_cov
            >= MIN_COVERAGE
            and np.isfinite(
                current_cov
            )
            and current_cov
            >= MIN_COVERAGE
            and np.isfinite(
                g1_cov
            )
            and g1_cov
            >= MIN_COVERAGE
        )

        eligible_v1 = bool(
            np.isfinite(
                v1
            )
            and np.isfinite(
                v1_cov
            )
            and v1_cov
            >= MIN_COVERAGE
            and np.isfinite(
                threshold
            )
        )

        rows.append(
            {
                "observation_date": date,
                "logger_run_utc": run_utc,
                "model_version": MODEL_VERSION,
                "pipeline_version": PIPELINE_VERSION,
                "core_sha256": EXPECTED_CORE_SHA256,
                "freeze_spec_sha256": EXPECTED_FREEZE_SPEC_SHA256,
                "asset": ASSET,
                "asset_price": _safe_float(
                    row[
                        "asset_price"
                    ]
                ),
                "d1_score": d1_score,
                "d1_component_coverage": d1_component_cov,
                "d1_current_model_coverage": current_cov,
                "d1_g1_model_coverage": g1_cov,
                "v1_score": v1,
                "v1_coverage": v1_cov,
                "pit_high_vol_threshold": threshold,
                "backward_realized_vol_20d": _safe_float(
                    row[
                        "backward_realized_vol_20d"
                    ]
                ),
                "eligible_d1": eligible_d1,
                "eligible_v1": eligible_v1,
                "event_note": EVENT_NOTE,
                "source_health": source_health,
                "source_failures": failures,
                "fwd_return_5d": np.nan,
                "fwd_return_20d": np.nan,
                "fwd_return_60d": np.nan,
                "fwd_realized_vol_20d": np.nan,
                "high_vol_event_20d": np.nan,
                "matured_5d": False,
                "matured_20d": False,
                "matured_60d": False,
            }
        )

    if rows:
        log_df = pd.concat(
            [
                log_df,
                pd.DataFrame(
                    rows
                ),
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
    ] = pd.to_datetime(
        log_df[
            "observation_date"
        ],
        errors="coerce",
    ).dt.normalize()

    for idx, date in log_df[
        "observation_date"
    ].items():
        if (
            pd.isna(
                date
            )
            or date not in targets.index
        ):
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

            if np.isfinite(
                value
            ):
                log_df.at[
                    idx,
                    log_col,
                ] = value

                log_df.at[
                    idx,
                    mature_col,
                ] = True

        fwd_vol = _safe_float(
            t.get(
                "Fwd_Realized_Vol_20D",
                np.nan,
            )
        )

        threshold = _safe_float(
            log_df.at[
                idx,
                "pit_high_vol_threshold",
            ]
        )

        if np.isfinite(
            fwd_vol
        ):
            log_df.at[
                idx,
                "fwd_realized_vol_20d",
            ] = fwd_vol

            if np.isfinite(
                threshold
            ):
                log_df.at[
                    idx,
                    "high_vol_event_20d",
                ] = int(
                    fwd_vol
                    >= threshold
                )

    return log_df


def _write_summary(
    log_df,
):
    if log_df.empty:
        last_date = ""
        latest_d1 = np.nan
        latest_v1 = np.nan
        latest_threshold = np.nan
        eligible_d1 = 0
        eligible_v1 = 0
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

        latest_d1 = _safe_float(
            last[
                "d1_score"
            ]
        )

        latest_v1 = _safe_float(
            last[
                "v1_score"
            ]
        )

        latest_threshold = _safe_float(
            last[
                "pit_high_vol_threshold"
            ]
        )

        eligible_d1 = int(
            log_df[
                "eligible_d1"
            ]
            .astype(
                str
            )
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                ]
            )
            .sum()
        )

        eligible_v1 = int(
            log_df[
                "eligible_v1"
            ]
            .astype(
                str
            )
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                ]
            )
            .sum()
        )

    summary = pd.DataFrame(
        [
            {
                "model_version": MODEL_VERSION,
                "frozen_on_date": FROZEN_ON_DATE.date().isoformat(),
                "shadow_start_date": SHADOW_START_DATE.date().isoformat(),
                "last_observation_date": last_date,
                "core_sha256": EXPECTED_CORE_SHA256,
                "freeze_spec_sha256": EXPECTED_FREEZE_SPEC_SHA256,
                "observations_total": int(
                    len(
                        log_df
                    )
                ),
                "eligible_d1": eligible_d1,
                "eligible_v1": eligible_v1,
                "matured_5d": int(
                    pd.to_numeric(
                        log_df.get(
                            "fwd_return_5d",
                            pd.Series(
                                dtype=float
                            ),
                        ),
                        errors="coerce",
                    ).notna().sum()
                ),
                "matured_20d": int(
                    pd.to_numeric(
                        log_df.get(
                            "fwd_return_20d",
                            pd.Series(
                                dtype=float
                            ),
                        ),
                        errors="coerce",
                    ).notna().sum()
                ),
                "matured_60d": int(
                    pd.to_numeric(
                        log_df.get(
                            "fwd_return_60d",
                            pd.Series(
                                dtype=float
                            ),
                        ),
                        errors="coerce",
                    ).notna().sum()
                ),
                "latest_d1_score": latest_d1,
                "latest_v1_score": latest_v1,
                "latest_pit_high_vol_threshold": latest_threshold,
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
        d1,
        v1_score,
        v1_coverage,
        backward_vol,
        pit_threshold,
        targets,
        status,
    ) = _calculate_frames()

    log_df = _load_existing_log()

    log_df = _append_missing_observations(
        log_df,
        raw,
        d1,
        v1_score,
        v1_coverage,
        backward_vol,
        pit_threshold,
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
        "Gold D1 + V1 dual-output shadow logger completed."
    )
    print(
        f"Rows: {len(log_df)}"
    )
    print(
        f"Output: {LOG_PATH}"
    )


if __name__ == "__main__":
    main()
