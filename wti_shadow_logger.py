"""
WTI Shadow Logger
=================

Frozen out-of-sample shadow validation starting 2026-09-02.

Current remains the production reference.
Model D is a challenger only:
    Current pillar weights + Literature Prior subweights.

The logger reconstructs the PIT-safe research scores, appends only
post-selection observations, and matures older rows with realized
5D/20D/60D outcomes.
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from regime_engine import get_regime_label
from wti_shadow_core import (
    MODEL_CURRENT,
    STRESS_MAE_THRESHOLDS,
    build_research_dataset,
    build_all_model_scores,
    build_forward_targets,
    diagnostic_model_configs,
    model_score_frame,
)

ASSET = "WTI Crude Oil"
SHADOW_START_DATE = pd.Timestamp("2026-09-02")

MODEL_VERSION = "WTI_MODEL_D_FROZEN_2026-09-02_v1"
PIPELINE_VERSION = "ResearchLab_v1.0.12_PIT"

MIN_COVERAGE = 60.0
HISTORY_YEARS = 15

LOG_DIR = Path("shadow_logs")
LOG_PATH = LOG_DIR / "wti_shadow_log.csv"
SUMMARY_PATH = LOG_DIR / "wti_shadow_summary.csv"

EVENT_RISK = (
    os.environ.get("WTI_EVENT_RISK", "").strip()
    or "UNCLASSIFIED"
)
EVENT_NOTE = os.environ.get("WTI_EVENT_NOTE", "").strip()

LOG_COLUMNS = [
    "observation_date",
    "logger_run_utc",
    "model_version",
    "pipeline_version",
    "asset",
    "asset_price",
    "current_score",
    "model_d_score",
    "score_delta_d_minus_current",
    "current_coverage",
    "model_d_coverage",
    "current_regime",
    "model_d_regime",
    "eligible_for_holdout",
    "event_risk",
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


def _safe_float(value):
    try:
        value = float(value)
    except Exception:
        return np.nan

    return value if np.isfinite(value) else np.nan


def _load_existing_log():
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)

    df = pd.read_csv(LOG_PATH)

    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["observation_date"] = pd.to_datetime(
        df["observation_date"],
        errors="coerce",
    )

    return df[LOG_COLUMNS].copy()


def _source_health(status):
    if not isinstance(status, dict):
        return "UNKNOWN", "status object missing"

    failures = []

    for name, result in status.items():
        if isinstance(result, tuple):
            ok = bool(result[0])
            note = str(result[1]) if len(result) > 1 else ""
        else:
            ok = bool(result)
            note = ""

        if not ok:
            failures.append(f"{name}: {note}"[:300])

    if not failures:
        return "OK", ""

    return "DEGRADED", " | ".join(failures)[:1800]


def _calculate_frames():
    if not os.environ.get("FRED_API_KEY", "").strip():
        raise RuntimeError(
            "FRED_API_KEY is missing. Add it as a GitHub Actions "
            "repository secret before running the shadow logger."
        )

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()

    start_date = (
        today
        - pd.DateOffset(years=HISTORY_YEARS)
        - pd.DateOffset(months=3)
    ).date()

    raw, status, _pit_quality = build_research_dataset(
        ASSET,
        start_date,
        prefer_first_release=True,
    )

    if raw.empty:
        raise RuntimeError("WTI research dataset is empty.")

    norm_df, _configs, model_frames = build_all_model_scores(
        raw,
        ASSET,
    )

    diagnostic_cfg = diagnostic_model_configs(ASSET)

    model_d_frame = model_score_frame(
        norm_df,
        diagnostic_cfg["D · Lit Subweights only"],
    )

    current_frame = model_frames[MODEL_CURRENT]

    targets = build_forward_targets(
        raw["asset_price"],
        ASSET,
    )

    return raw, current_frame, model_d_frame, targets, status


def _append_missing_observations(
    log_df,
    raw,
    current_frame,
    model_d_frame,
    status,
):
    source_health, failures = _source_health(status)

    common = pd.DataFrame(
        {
            "asset_price": pd.to_numeric(
                raw["asset_price"],
                errors="coerce",
            ),
            "current_score": pd.to_numeric(
                current_frame["Final_Regime_Score"],
                errors="coerce",
            ),
            "model_d_score": pd.to_numeric(
                model_d_frame["Final_Regime_Score"],
                errors="coerce",
            ),
            "current_coverage": pd.to_numeric(
                current_frame["Model_Data_Coverage"],
                errors="coerce",
            ),
            "model_d_coverage": pd.to_numeric(
                model_d_frame["Model_Data_Coverage"],
                errors="coerce",
            ),
        }
    ).dropna(
        subset=[
            "asset_price",
            "current_score",
            "model_d_score",
        ]
    )

    if common.empty:
        raise RuntimeError(
            "No common Current/Model-D observation is available."
        )

    common.index = pd.to_datetime(
        common.index,
        errors="coerce",
    ).normalize()

    existing_dates = set(
        pd.to_datetime(
            log_df["observation_date"],
            errors="coerce",
        )
        .dropna()
        .dt.normalize()
    )

    candidate = common[
        common.index >= SHADOW_START_DATE
    ]

    new_rows = []
    run_utc = datetime.now(timezone.utc).isoformat()

    for date, row in candidate.iterrows():
        date = pd.Timestamp(date).normalize()

        if date in existing_dates:
            continue

        current_score = _safe_float(row["current_score"])
        model_d_score = _safe_float(row["model_d_score"])
        current_cov = _safe_float(row["current_coverage"])
        d_cov = _safe_float(row["model_d_coverage"])

        eligible = bool(
            np.isfinite(current_cov)
            and np.isfinite(d_cov)
            and current_cov >= MIN_COVERAGE
            and d_cov >= MIN_COVERAGE
        )

        new_rows.append(
            {
                "observation_date": date,
                "logger_run_utc": run_utc,
                "model_version": MODEL_VERSION,
                "pipeline_version": PIPELINE_VERSION,
                "asset": ASSET,
                "asset_price": _safe_float(row["asset_price"]),
                "current_score": current_score,
                "model_d_score": model_d_score,
                "score_delta_d_minus_current": (
                    model_d_score - current_score
                ),
                "current_coverage": current_cov,
                "model_d_coverage": d_cov,
                "current_regime": get_regime_label(current_score),
                "model_d_regime": get_regime_label(model_d_score),
                "eligible_for_holdout": eligible,
                "event_risk": EVENT_RISK,
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

    if new_rows:
        log_df = pd.concat(
            [log_df, pd.DataFrame(new_rows)],
            ignore_index=True,
        )

    return log_df


def _mature_forward_outcomes(log_df, targets):
    if log_df.empty:
        return log_df

    targets = targets.copy()
    targets.index = pd.to_datetime(
        targets.index,
        errors="coerce",
    ).normalize()

    log_df["observation_date"] = pd.to_datetime(
        log_df["observation_date"],
        errors="coerce",
    ).dt.normalize()

    for idx, observation_date in log_df[
        "observation_date"
    ].items():
        if pd.isna(observation_date):
            continue

        if observation_date not in targets.index:
            continue

        t = targets.loc[observation_date]

        mappings = {
            "fwd_return_5d": (
                "Fwd_Return_5D",
                "matured_5d",
            ),
            "fwd_return_20d": (
                "Fwd_Return_20D",
                "matured_20d",
            ),
            "fwd_return_60d": (
                "Fwd_Return_60D",
                "matured_60d",
            ),
        }

        for log_col, (target_col, mature_col) in mappings.items():
            value = _safe_float(
                t.get(target_col, np.nan)
            )

            if np.isfinite(value):
                log_df.at[idx, log_col] = value
                log_df.at[idx, mature_col] = True

        mae = _safe_float(
            t.get("Fwd_MAE_20D", np.nan)
        )

        fwd_vol = _safe_float(
            t.get("Fwd_Realized_Vol_20D", np.nan)
        )

        if np.isfinite(mae):
            log_df.at[idx, "fwd_mae_20d"] = mae
            log_df.at[idx, "stress_event_20d"] = int(
                mae <= STRESS_MAE_THRESHOLDS[ASSET]
            )

        if np.isfinite(fwd_vol):
            log_df.at[
                idx,
                "fwd_realized_vol_20d",
            ] = fwd_vol

    return log_df


def _write_summary(log_df):
    eligible_mask = (
        log_df["eligible_for_holdout"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
        if not log_df.empty
        else pd.Series(dtype=bool)
    )

    eligible = (
        log_df[eligible_mask].copy()
        if not log_df.empty
        else log_df.copy()
    )

    if log_df.empty:
        last_date = ""
        current_score = np.nan
        d_score = np.nan
    else:
        ordered = log_df.sort_values("observation_date")
        last = ordered.iloc[-1]

        last_date = pd.Timestamp(
            last["observation_date"]
        ).date().isoformat()

        current_score = _safe_float(
            last["current_score"]
        )
        d_score = _safe_float(
            last["model_d_score"]
        )

    summary = pd.DataFrame(
        [
            {
                "model_version": MODEL_VERSION,
                "shadow_start_date": (
                    SHADOW_START_DATE.date().isoformat()
                ),
                "last_observation_date": last_date,
                "observations_total": int(len(log_df)),
                "observations_eligible": int(len(eligible)),
                "matured_5d": int(
                    pd.to_numeric(
                        log_df["fwd_return_5d"],
                        errors="coerce",
                    ).notna().sum()
                )
                if not log_df.empty
                else 0,
                "matured_20d": int(
                    pd.to_numeric(
                        log_df["fwd_return_20d"],
                        errors="coerce",
                    ).notna().sum()
                )
                if not log_df.empty
                else 0,
                "matured_60d": int(
                    pd.to_numeric(
                        log_df["fwd_return_60d"],
                        errors="coerce",
                    ).notna().sum()
                )
                if not log_df.empty
                else 0,
                "latest_current_score": current_score,
                "latest_model_d_score": d_score,
                "latest_score_delta": (
                    d_score - current_score
                    if (
                        np.isfinite(d_score)
                        and np.isfinite(current_score)
                    )
                    else np.nan
                ),
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
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    (
        raw,
        current_frame,
        model_d_frame,
        targets,
        status,
    ) = _calculate_frames()

    log_df = _load_existing_log()

    log_df = _append_missing_observations(
        log_df,
        raw,
        current_frame,
        model_d_frame,
        status,
    )

    log_df = _mature_forward_outcomes(
        log_df,
        targets,
    )

    if not log_df.empty:
        log_df = (
            log_df
            .sort_values("observation_date")
            .drop_duplicates(
                subset=[
                    "observation_date",
                    "model_version",
                ],
                keep="last",
            )
            .reset_index(drop=True)
        )

    log_df.to_csv(
        LOG_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    _write_summary(log_df)

    latest_date = (
        log_df["observation_date"].max()
        if not log_df.empty
        else None
    )

    print("WTI shadow logger completed.")
    print(f"Model: {MODEL_VERSION}")
    print(f"Rows: {len(log_df)}")
    print(f"Latest observation: {latest_date}")
    print(f"Output: {LOG_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
