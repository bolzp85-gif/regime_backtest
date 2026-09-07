"""
EUR/USD Frozen A/B/E Shadow Logger
==================================

True future holdout frozen on 2026-09-06.
First eligible observation date: 2026-09-07.

A = Current production reference
B = Full Literature Prior, Direction/Regime challenger
E = Literature Pillars only, Risk-State challenger
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from eurusd_shadow_core import (
    MODEL_CURRENT,
    MODEL_LITERATURE,
    STRESS_MAE_THRESHOLDS,
    build_research_dataset,
    build_all_model_scores,
    build_forward_targets,
    current_model_config,
    literature_model_config,
    diagnostic_model_configs,
    model_score_frame,
)

ASSET = "EUR/USD"
FROZEN_ON_DATE = pd.Timestamp("2026-09-06")
SHADOW_START_DATE = pd.Timestamp("2026-09-07")
MODEL_VERSION = "EURUSD_A_B_E_FROZEN_2026-09-06_v1"
PIPELINE_VERSION = "ResearchLab_v1.0.15_PIT_integrityfix1_safeclose"

EXPECTED_CORE_SHA256 = "c029aee7cc15a64ed5122b2f5c4160def2abe59502a1f20d1ae2680ba1a71695"
EXPECTED_CONFIG_SHA256 = "fb600dcae17347a637e453cac8716abd8c6b3cb44d33c3d53a88ffaec8d9778a"

# Semantic freeze snapshot. The exact frozen core hash remains the
# primary integrity lock. This snapshot adds key/weight verification
# without relying on brittle raw-float JSON hashing.
FROZEN_CONFIG_SNAPSHOT = {'A_current': {'pillar_weights': {'Makroökonomie': 0.35, 'Positionierung': 0.2, 'Marktinterna': 0.15, 'Technischer_Trend': 0.2, 'Fruehwarnindikatoren': 0.1}, 'sub_weights': {'Makroökonomie': {'fed_policy': 0.2, 'real_yields': 0.3, 'usd_index': 0.2, 'net_liquidity': 0.3}, 'Technischer_Trend': {'distance_200ma': 0.35, 'distance_50ma': 0.35, 'rsi_momentum': 0.3}, 'Fruehwarnindikatoren': {'credit_spreads': 0.6, 'move_index': 0.4}, 'Positionierung': {'cot_noncommercials': 0.7, 'fear_greed': 0.3}, 'Marktinterna': {'market_momentum': 0.5, 'vix_score': 0.5}, 'Fundamentale_Faktoren': {}}}, 'B_full_literature': {'pillar_weights': {'Makroökonomie': 0.25, 'Positionierung': 0.15, 'Marktinterna': 0.2, 'Technischer_Trend': 0.35, 'Fruehwarnindikatoren': 0.05}, 'sub_weights': {'Makroökonomie': {'fed_policy': 0.4, 'real_yields': 0.35, 'usd_index': 0.05, 'net_liquidity': 0.2}, 'Technischer_Trend': {'distance_200ma': 0.4, 'distance_50ma': 0.4, 'rsi_momentum': 0.2}, 'Fruehwarnindikatoren': {'credit_spreads': 0.6, 'move_index': 0.4}, 'Positionierung': {'cot_noncommercials': 0.9, 'fear_greed': 0.1}, 'Marktinterna': {'market_momentum': 0.7, 'vix_score': 0.3}, 'Fundamentale_Faktoren': {}}}, 'E_literature_pillars_only': {'pillar_weights': {'Makroökonomie': 0.25, 'Positionierung': 0.15, 'Marktinterna': 0.2, 'Technischer_Trend': 0.35, 'Fruehwarnindikatoren': 0.05}, 'sub_weights': {'Makroökonomie': {'fed_policy': 0.2, 'real_yields': 0.3, 'usd_index': 0.2, 'net_liquidity': 0.3}, 'Technischer_Trend': {'distance_200ma': 0.35, 'distance_50ma': 0.35, 'rsi_momentum': 0.3}, 'Fruehwarnindikatoren': {'credit_spreads': 0.6, 'move_index': 0.4}, 'Positionierung': {'cot_noncommercials': 0.7, 'fear_greed': 0.3}, 'Marktinterna': {'market_momentum': 0.5, 'vix_score': 0.5}, 'Fundamentale_Faktoren': {}}}}

MIN_COVERAGE = 60.0
HISTORY_YEARS = 15

# Do not freeze an unfinished same-day Yahoo EUR/USD daily bar.
# Scheduled runs occur at 23:58 UTC. Manual runs before 22:00 UTC
# will therefore wait for the previous completed day.
SAFE_DAILY_BAR_UTC_HOUR = 22

LOG_DIR = Path("shadow_logs")
LOG_PATH = LOG_DIR / "eurusd_shadow_log.csv"
SUMMARY_PATH = LOG_DIR / "eurusd_shadow_summary.csv"

EVENT_RISK = os.environ.get("EURUSD_EVENT_RISK", "").strip() or "UNCLASSIFIED"
EVENT_NOTE = os.environ.get("EURUSD_EVENT_NOTE", "").strip()

LOG_COLUMNS = [
    "observation_date", "logger_run_utc", "model_version", "pipeline_version",
    "core_sha256", "config_sha256", "asset", "asset_price",
    "current_score", "model_b_score", "model_e_score",
    "score_delta_b_minus_current", "score_delta_e_minus_current",
    "current_coverage", "model_b_coverage", "model_e_coverage",
    "current_regime", "model_b_regime", "model_e_regime",
    "eligible_for_holdout", "event_risk", "event_note",
    "source_health", "source_failures",
    "fwd_return_5d", "fwd_return_20d", "fwd_return_60d",
    "fwd_mae_20d", "fwd_realized_vol_20d", "stress_event_20d",
    "matured_5d", "matured_20d", "matured_60d",
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
    return value if np.isfinite(value) else np.nan


def _normalized_config_payload():
    return {
        "A_current": current_model_config(ASSET),
        "B_full_literature": literature_model_config(ASSET),
        "E_literature_pillars_only": diagnostic_model_configs(ASSET)[
            "E · Lit Pillars only"
        ],
    }


def _canonicalize_config(value):
    """
    Deterministic semantic representation of the frozen model config.

    Numeric values are rounded to 12 decimals only to eliminate harmless
    floating-point representation differences. Keys and structure must still
    match exactly.
    """
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


def _config_fingerprint():
    canonical = _canonicalize_config(
        _normalized_config_payload()
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


def _configs_semantically_equal(actual, expected):
    """
    Exact key/structure comparison with a 1e-12 numeric tolerance.
    """
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


def _core_fingerprint():
    core_path = Path(__file__).resolve().parent / "eurusd_shadow_core.py"
    return hashlib.sha256(core_path.read_bytes()).hexdigest()


def _assert_frozen_pipeline():
    actual_core = _core_fingerprint()

    if actual_core != EXPECTED_CORE_SHA256:
        raise RuntimeError(
            "Frozen EUR/USD core changed. Refusing to contaminate the true "
            f"holdout. Expected {EXPECTED_CORE_SHA256}, got {actual_core}."
        )

    actual_payload = _normalized_config_payload()

    if not _configs_semantically_equal(
        actual_payload,
        FROZEN_CONFIG_SNAPSHOT,
    ):
        actual_hash = _config_fingerprint()

        raise RuntimeError(
            "Frozen EUR/USD A/B/E configuration changed semantically. "
            "Refusing to continue. "
            f"Expected semantic hash {EXPECTED_CONFIG_SHA256}, "
            f"got {actual_hash}."
        )


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

    def _ok(result):
        if isinstance(result, tuple):
            return bool(result[0])
        return bool(result)

    yahoo_symbol_results = [
        result
        for name, result in status.items()
        if name.startswith("Yahoo ")
        and name not in {"Yahoo Batch", "Yahoo Research Sample"}
    ]

    yahoo_repaired_ok = bool(
        yahoo_symbol_results
        and all(_ok(result) for result in yahoo_symbol_results)
    )

    failures = []

    for name, result in status.items():
        if isinstance(result, tuple):
            ok = bool(result[0])
            note = str(result[1]) if len(result) > 1 else ""
        else:
            ok = bool(result)
            note = ""

        # A failed batch request is not a data-quality failure when every
        # required Yahoo symbol was successfully repaired individually.
        if (
            name == "Yahoo Batch"
            and not ok
            and yahoo_repaired_ok
        ):
            continue

        if not ok:
            failures.append(f"{name}: {note}"[:300])

    if not failures:
        return "OK", ""

    return "DEGRADED", " | ".join(failures)[:1800]


def _calculate_frames():
    _assert_frozen_pipeline()

    if not os.environ.get("FRED_API_KEY", "").strip():
        raise RuntimeError(
            "FRED_API_KEY is missing. Reuse the existing GitHub Actions secret."
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
        raise RuntimeError("EUR/USD research dataset is empty.")

    norm_df, _configs, model_frames = build_all_model_scores(
        raw,
        ASSET,
    )

    current_frame = model_frames[MODEL_CURRENT]
    model_b_frame = model_frames[MODEL_LITERATURE]

    e_cfg = diagnostic_model_configs(ASSET)["E · Lit Pillars only"]

    model_e_frame = model_score_frame(
        norm_df,
        e_cfg,
    )

    targets = build_forward_targets(
        raw["asset_price"],
        ASSET,
    )

    return raw, current_frame, model_b_frame, model_e_frame, targets, status


def _append_missing_observations(
    log_df,
    raw,
    current_frame,
    model_b_frame,
    model_e_frame,
    status,
):
    source_health, failures = _source_health(status)

    common = pd.DataFrame(
        {
            "asset_price": pd.to_numeric(
                raw["asset_price"], errors="coerce"
            ),
            "current_score": pd.to_numeric(
                current_frame["Final_Regime_Score"], errors="coerce"
            ),
            "model_b_score": pd.to_numeric(
                model_b_frame["Final_Regime_Score"], errors="coerce"
            ),
            "model_e_score": pd.to_numeric(
                model_e_frame["Final_Regime_Score"], errors="coerce"
            ),
            "current_coverage": pd.to_numeric(
                current_frame["Model_Data_Coverage"], errors="coerce"
            ),
            "model_b_coverage": pd.to_numeric(
                model_b_frame["Model_Data_Coverage"], errors="coerce"
            ),
            "model_e_coverage": pd.to_numeric(
                model_e_frame["Model_Data_Coverage"], errors="coerce"
            ),
        }
    ).dropna(
        subset=[
            "asset_price", "current_score", "model_b_score", "model_e_score"
        ]
    )

    if common.empty:
        raise RuntimeError("No common A/B/E EUR/USD observation is available.")

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

    now_utc = pd.Timestamp.now(tz="UTC")
    today_utc = now_utc.tz_localize(None).normalize()

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

    run_utc = datetime.now(timezone.utc).isoformat()
    new_rows = []

    for date, row in candidate.iterrows():
        date = pd.Timestamp(date).normalize()

        if date in existing_dates:
            continue

        current_score = _safe_float(row["current_score"])
        b_score = _safe_float(row["model_b_score"])
        e_score = _safe_float(row["model_e_score"])

        current_cov = _safe_float(row["current_coverage"])
        b_cov = _safe_float(row["model_b_coverage"])
        e_cov = _safe_float(row["model_e_coverage"])

        eligible = bool(
            source_health == "OK"
            and np.isfinite(current_cov)
            and np.isfinite(b_cov)
            and np.isfinite(e_cov)
            and current_cov >= MIN_COVERAGE
            and b_cov >= MIN_COVERAGE
            and e_cov >= MIN_COVERAGE
        )

        new_rows.append(
            {
                "observation_date": date,
                "logger_run_utc": run_utc,
                "model_version": MODEL_VERSION,
                "pipeline_version": PIPELINE_VERSION,
                "core_sha256": EXPECTED_CORE_SHA256,
                "config_sha256": EXPECTED_CONFIG_SHA256,
                "asset": ASSET,
                "asset_price": _safe_float(row["asset_price"]),
                "current_score": current_score,
                "model_b_score": b_score,
                "model_e_score": e_score,
                "score_delta_b_minus_current": b_score - current_score,
                "score_delta_e_minus_current": e_score - current_score,
                "current_coverage": current_cov,
                "model_b_coverage": b_cov,
                "model_e_coverage": e_cov,
                "current_regime": get_regime_label(current_score),
                "model_b_regime": get_regime_label(b_score),
                "model_e_regime": get_regime_label(e_score),
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

    for idx, observation_date in log_df["observation_date"].items():
        if pd.isna(observation_date) or observation_date not in targets.index:
            continue

        t = targets.loc[observation_date]

        mappings = {
            "fwd_return_5d": ("Fwd_Return_5D", "matured_5d"),
            "fwd_return_20d": ("Fwd_Return_20D", "matured_20d"),
            "fwd_return_60d": ("Fwd_Return_60D", "matured_60d"),
        }

        for log_col, (target_col, mature_col) in mappings.items():
            value = _safe_float(t.get(target_col, np.nan))

            if np.isfinite(value):
                log_df.at[idx, log_col] = value
                log_df.at[idx, mature_col] = True

        mae = _safe_float(t.get("Fwd_MAE_20D", np.nan))
        fwd_vol = _safe_float(t.get("Fwd_Realized_Vol_20D", np.nan))

        if np.isfinite(mae):
            log_df.at[idx, "fwd_mae_20d"] = mae
            log_df.at[idx, "stress_event_20d"] = int(
                mae <= STRESS_MAE_THRESHOLDS[ASSET]
            )

        if np.isfinite(fwd_vol):
            log_df.at[idx, "fwd_realized_vol_20d"] = fwd_vol

    return log_df


def _write_summary(log_df):
    if log_df.empty:
        eligible = log_df.copy()
        last_date = ""
        current_score = np.nan
        b_score = np.nan
        e_score = np.nan
    else:
        eligible_mask = (
            log_df["eligible_for_holdout"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )
        eligible = log_df[eligible_mask].copy()

        ordered = log_df.sort_values("observation_date")
        last = ordered.iloc[-1]

        last_date = pd.Timestamp(
            last["observation_date"]
        ).date().isoformat()

        current_score = _safe_float(last["current_score"])
        b_score = _safe_float(last["model_b_score"])
        e_score = _safe_float(last["model_e_score"])

    summary = pd.DataFrame(
        [
            {
                "model_version": MODEL_VERSION,
                "frozen_on_date": FROZEN_ON_DATE.date().isoformat(),
                "shadow_start_date": SHADOW_START_DATE.date().isoformat(),
                "last_observation_date": last_date,
                "core_sha256": EXPECTED_CORE_SHA256,
                "config_sha256": EXPECTED_CONFIG_SHA256,
                "observations_total": int(len(log_df)),
                "observations_eligible": int(len(eligible)),
                "matured_5d": int(
                    pd.to_numeric(
                        log_df["fwd_return_5d"],
                        errors="coerce",
                    ).notna().sum()
                ) if not log_df.empty else 0,
                "matured_20d": int(
                    pd.to_numeric(
                        log_df["fwd_return_20d"],
                        errors="coerce",
                    ).notna().sum()
                ) if not log_df.empty else 0,
                "matured_60d": int(
                    pd.to_numeric(
                        log_df["fwd_return_60d"],
                        errors="coerce",
                    ).notna().sum()
                ) if not log_df.empty else 0,
                "latest_current_score": current_score,
                "latest_model_b_score": b_score,
                "latest_model_e_score": e_score,
                "latest_delta_b_minus_current": (
                    b_score - current_score
                    if (
                        np.isfinite(b_score)
                        and np.isfinite(current_score)
                    )
                    else np.nan
                ),
                "latest_delta_e_minus_current": (
                    e_score - current_score
                    if (
                        np.isfinite(e_score)
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
        model_b_frame,
        model_e_frame,
        targets,
        status,
    ) = _calculate_frames()

    log_df = _load_existing_log()

    log_df = _append_missing_observations(
        log_df,
        raw,
        current_frame,
        model_b_frame,
        model_e_frame,
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

    print("EUR/USD frozen A/B/E shadow logger completed.")
    print(f"Model: {MODEL_VERSION}")
    print(f"Rows: {len(log_df)}")
    print(f"Latest observation: {latest_date}")
    print(f"Output: {LOG_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
