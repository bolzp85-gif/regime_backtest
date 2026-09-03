"""
WTI Shadow Validation Core
==========================

Headless extraction of the PIT-safe research pipeline used in
Regime Backtest Lab v1.0.12.

Purpose:
- compute unchanged Current WTI model
- compute frozen Model D:
  Current pillar weights + Literature Prior subweights
- no Streamlit UI
- no model re-optimization
- no bfill in the research factor pipeline

This module is for shadow validation only and does not alter regime_engine.py.
"""

from __future__ import annotations

import io
import math
import os
import re
from copy import deepcopy
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
import yfinance as yf
from fredapi import Fred

from regime_engine import (
    ASSET_CONFIGS,
    SUB_WEIGHTS_BASE,
    LOOKBACK_CONFIG,
    FRED_API_KEY as ENGINE_FRED_API_KEY,
    fetch_fear_and_greed,
    fetch_sp500_pe_history,
)

FRED_API_KEY = (
    os.environ.get("FRED_API_KEY", "").strip()
    or str(ENGINE_FRED_API_KEY or "").strip()
)

# ============================================================
# 1. RESEARCH CONFIG
# ============================================================

MODEL_CURRENT = "A · Current"
MODEL_LITERATURE = "B · Literature Prior v1"
MODEL_EQUAL = "C · Equal Weight"

MODEL_ORDER = [
    MODEL_CURRENT,
    MODEL_LITERATURE,
    MODEL_EQUAL,
]

PILLARS = [
    "Makroökonomie",
    "Positionierung",
    "Marktinterna",
    "Technischer_Trend",
    "Fundamentale_Faktoren",
    "Fruehwarnindikatoren",
]

FORWARD_HORIZONS = [5, 20, 60]

# ============================================================
# WTI MODEL-D WALK-FORWARD – PRE-REGISTERED RULES
# ============================================================
#
# Model D is frozen:
#   Current pillar weights + Literature Prior subweights.
#
# No weights are re-estimated in the walk-forward section.
# The expanding past is used only as historical context for the
# already point-in-time rolling factor transformations.
#
WTI_WF_BASE_FIRST_TEST_YEAR = 2017
WTI_WF_MIN_IC20_WIN_RATE = 0.60
WTI_WF_MIN_POOLED_IC20 = 0.00
WTI_WF_MIN_DIRECTION_20D = 0.50
WTI_WF_MIN_STRESS_AUC = 0.55
WTI_WF_REQUIRE_STRESS_NOT_WORSE = True
WTI_WF_REQUIRE_IC20_EX_2020_POSITIVE = True

# ============================================================
# WTI EVENT / CRISIS ROBUSTNESS WINDOWS
# ============================================================
#
# These are NOT score inputs and do NOT modify Model D.
# They are fixed exogenous sensitivity windows used only to test
# whether model performance is concentrated in exceptional episodes.
#
# Start dates are anchored to identifiable public events.
# End dates are deliberately simple calendar cut-offs chosen BEFORE
# looking at the event-robustness result; they are not optimized.
#
WTI_EVENT_WINDOWS = [
    {
        "name": "OPEC 2014 / Supply-Glut Regime",
        "category": "Supply / OPEC",
        "start": "2014-11-27",
        "end": "2016-02-29",
        "formal_walk_forward": False,
        "anchor": (
            "OPEC maintained the 30.0 mb/d production level on 27 Nov 2014. "
            "This window is historical context and lies before the formal "
            "2017 walk-forward start."
        ),
    },
    {
        "name": "COVID-19 / Oil-Demand Shock",
        "category": "Pandemic / Demand",
        "start": "2020-03-11",
        "end": "2020-06-30",
        "formal_walk_forward": True,
        "anchor": (
            "WHO characterized COVID-19 as a pandemic on 11 Mar 2020. "
            "The end is fixed at 2020 Q2-end for sensitivity analysis."
        ),
    },
    {
        "name": "Russia-Ukraine / Energy Shock",
        "category": "Geopolitical / Supply",
        "start": "2022-02-24",
        "end": "2022-12-31",
        "formal_walk_forward": True,
        "anchor": (
            "Large-scale Russian military operations in Ukraine began "
            "24 Feb 2022. The sensitivity window is fixed through year-end."
        ),
    },
    {
        "name": "Middle East Escalation 2023",
        "category": "Geopolitical / Regional Risk",
        "start": "2023-10-07",
        "end": "2023-12-31",
        "formal_walk_forward": True,
        "anchor": (
            "The 7 Oct 2023 attacks and subsequent regional escalation "
            "anchor this fixed Q4-2023 sensitivity window."
        ),
    },
]

CFTC_LEGACY_API = (
    "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
)

CFTC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(compatible; QuantRegimeResearch/1.0)"
    ),
    "Accept": "application/json,text/plain,*/*",
}

# Approximation: Tuesday COT report becomes public Friday.
CFTC_PUBLICATION_LAG_DAYS = 3

# Official EIA weekly U.S. crude inventories excluding SPR.
# Important: WCESTUS1 is an EIA petroleum series, NOT a FRED series.
EIA_WTI_INVENTORY_URL = (
    "https://www.eia.gov/dnav/pet/hist/"
    "LeafHandler.ashx?f=W&n=PET&s=WCESTUS1"
)
EIA_WTI_INVENTORY_SERIES = "WCESTUS1"

# Week-ending Friday data are normally released the following Wednesday.
# Use a conservative PIT approximation of +5 calendar days.
EIA_INVENTORY_PUBLICATION_LAG_DAYS = 5

# Conservative fallback availability lags if ALFRED first-release
# metadata cannot be loaded.
FRED_FALLBACK_LAGS = {
    "WALCL": 0,       # weekly balance sheet, date is close to release
    "WTREGEN": 1,     # weekly Treasury account
    "RRPONTSYD": 0,   # daily
    "DFII10": 0,      # daily market yield
    "FEDFUNDS": 35,   # monthly average; conservative fallback
}

RESEARCH_YAHOO_COMMON = [
    "DX-Y.NYB",
    "^MOVE",
    "^VVIX",
    "HYG",
    "LQD",
]

# Literature Prior v1 from the literature review.
LITERATURE_PILLAR_WEIGHTS = {
    "S&P 500": {
        "Makroökonomie": .22,
        "Positionierung": .12,
        "Marktinterna": .18,
        "Technischer_Trend": .25,
        "Fundamentale_Faktoren": .08,
        "Fruehwarnindikatoren": .15,
    },
    "Nasdaq 100": {
        "Makroökonomie": .27,
        "Positionierung": .12,
        "Marktinterna": .20,
        "Technischer_Trend": .26,
        "Fundamentale_Faktoren": 0.00,
        "Fruehwarnindikatoren": .15,
    },
    "Gold (XAU/USD)": {
        "Makroökonomie": .35,
        "Positionierung": .18,
        "Marktinterna": .12,
        "Technischer_Trend": .25,
        "Fundamentale_Faktoren": 0.00,
        "Fruehwarnindikatoren": .10,
    },
    "WTI Crude Oil": {
        "Makroökonomie": .18,
        "Positionierung": .20,
        "Marktinterna": .12,
        "Technischer_Trend": .25,
        "Fundamentale_Faktoren": .25,
        "Fruehwarnindikatoren": 0.00,
    },
    "EUR/USD": {
        "Makroökonomie": .25,
        "Positionierung": .15,
        "Marktinterna": .20,
        "Technischer_Trend": .35,
        "Fundamentale_Faktoren": 0.00,
        "Fruehwarnindikatoren": .05,
    },
}

LITERATURE_MACRO = {
    "S&P 500": {
        "fed_policy": .30,
        "real_yields": .35,
        "usd_index": .15,
        "net_liquidity": .20,
    },
    "Nasdaq 100": {
        "fed_policy": .30,
        "real_yields": .40,
        "usd_index": .10,
        "net_liquidity": .20,
    },
    "Gold (XAU/USD)": {
        "fed_policy": .10,
        "real_yields": .50,
        "usd_index": .25,
        "net_liquidity": .15,
    },
    "WTI Crude Oil": {
        "fed_policy": .15,
        "real_yields": .10,
        "usd_index": .45,
        "net_liquidity": .30,
    },
    "EUR/USD": {
        "fed_policy": .40,
        "real_yields": .35,
        "usd_index": .05,
        "net_liquidity": .20,
    },
}

LITERATURE_POSITIONING = {
    "S&P 500": {
        "cot_noncommercials": .45,
        "fear_greed": .55,
    },
    "Nasdaq 100": {
        "cot_noncommercials": .40,
        "fear_greed": .60,
    },
    "Gold (XAU/USD)": {
        "cot_noncommercials": .85,
        "fear_greed": .15,
    },
    "WTI Crude Oil": {
        "cot_noncommercials": .90,
        "fear_greed": .10,
    },
    "EUR/USD": {
        "cot_noncommercials": .90,
        "fear_greed": .10,
    },
}

LITERATURE_INTERNALS = {
    "S&P 500": {
        "market_momentum": .60,
        "vix_score": .40,
    },
    "Nasdaq 100": {
        "market_momentum": .60,
        "vix_score": .40,
    },
    "Gold (XAU/USD)": {
        "obv_momentum": .65,
        "vix_score": .35,
    },
    "WTI Crude Oil": {
        "obv_momentum": .60,
        "vix_score": .40,
    },
    "EUR/USD": {
        "market_momentum": .70,
        "vix_score": .30,
    },
}

LITERATURE_TECHNICAL = {
    "distance_200ma": .40,
    "distance_50ma": .40,
    "rsi_momentum": .20,
}

LITERATURE_EARLY_US = {
    "credit_spreads": .60,
    "move_index": .20,
    "vvix_score": .20,
}

# For non-US equity assets we keep the same factor set as the
# current architecture and only alter it where Literature Prior
# explicitly justified a change.
LITERATURE_EARLY_OTHER = {
    "credit_spreads": .60,
    "move_index": .40,
}

STRESS_MAE_THRESHOLDS = {
    "S&P 500": -0.05,
    "Nasdaq 100": -0.07,
    "Gold (XAU/USD)": -0.05,
    "WTI Crude Oil": -0.10,
    "EUR/USD": -0.03,
}


# ============================================================
# 2. CONFIG BUILDERS
# ============================================================

def normalize_weight_dict(weights):
    weights = {
        k: float(v)
        for k, v in weights.items()
        if float(v) > 0
    }

    total = sum(weights.values())

    if total <= 0:
        return {}

    return {
        k: v / total
        for k, v in weights.items()
    }


def current_model_config(asset_name):
    cfg = ASSET_CONFIGS[asset_name]

    sub = {
        key: deepcopy(value)
        for key, value in SUB_WEIGHTS_BASE.items()
    }

    for pillar, weights in cfg["Sub_Gewichte"].items():
        sub[pillar] = {
            k: float(v)
            for k, v in weights.items()
            if float(v) > 0
        }

    return {
        "pillar_weights": normalize_weight_dict(
            cfg["Saeulen_Gewichte"]
        ),
        "sub_weights": {
            p: normalize_weight_dict(w)
            for p, w in sub.items()
        },
    }


def literature_model_config(asset_name):
    current = current_model_config(asset_name)

    sub = deepcopy(current["sub_weights"])

    sub["Makroökonomie"] = normalize_weight_dict(
        LITERATURE_MACRO[asset_name]
    )

    sub["Positionierung"] = normalize_weight_dict(
        LITERATURE_POSITIONING[asset_name]
    )

    sub["Marktinterna"] = normalize_weight_dict(
        LITERATURE_INTERNALS[asset_name]
    )

    sub["Technischer_Trend"] = normalize_weight_dict(
        LITERATURE_TECHNICAL
    )

    if asset_name in {
        "S&P 500",
        "Nasdaq 100",
    }:
        sub["Fruehwarnindikatoren"] = normalize_weight_dict(
            LITERATURE_EARLY_US
        )

    elif asset_name != "WTI Crude Oil":
        sub["Fruehwarnindikatoren"] = normalize_weight_dict(
            LITERATURE_EARLY_OTHER
        )

    if asset_name == "S&P 500":
        sub["Fundamentale_Faktoren"] = {
            "pe_valuation": 1.0
        }

    elif asset_name == "WTI Crude Oil":
        sub["Fundamentale_Faktoren"] = {
            "inventories": 1.0
        }

    else:
        sub["Fundamentale_Faktoren"] = {}

    return {
        "pillar_weights": normalize_weight_dict(
            LITERATURE_PILLAR_WEIGHTS[asset_name]
        ),
        "sub_weights": sub,
    }


def equal_model_config(asset_name):
    current = current_model_config(asset_name)
    literature = literature_model_config(asset_name)

    active_pillars = []

    for pillar in PILLARS:
        if (
            current["pillar_weights"].get(pillar, 0) > 0
            or literature["pillar_weights"].get(pillar, 0) > 0
        ):
            active_pillars.append(pillar)

    pillar_weights = {
        pillar: 1.0 / len(active_pillars)
        for pillar in active_pillars
    }

    sub = {}

    for pillar in active_pillars:
        factor_union = []

        for model_cfg in [current, literature]:
            for factor in model_cfg[
                "sub_weights"
            ].get(pillar, {}):
                if factor not in factor_union:
                    factor_union.append(factor)

        if factor_union:
            sub[pillar] = {
                factor: 1.0 / len(factor_union)
                for factor in factor_union
            }
        else:
            sub[pillar] = {}

    return {
        "pillar_weights": pillar_weights,
        "sub_weights": sub,
    }


def all_model_configs(asset_name):
    return {
        MODEL_CURRENT: current_model_config(asset_name),
        MODEL_LITERATURE: literature_model_config(asset_name),
        MODEL_EQUAL: equal_model_config(asset_name),
    }


def diagnostic_model_configs(asset_name):
    """
    Decomposition of the literature-prior change:

    - Subweights only: literature subweights + current pillar weights
    - Pillars only: literature pillar weights + current subweights

    These are diagnostics only, not candidate production models.
    """
    current = current_model_config(asset_name)
    literature = literature_model_config(asset_name)

    return {
        "D · Lit Subweights only": {
            "pillar_weights": deepcopy(current["pillar_weights"]),
            "sub_weights": deepcopy(literature["sub_weights"]),
        },
        "E · Lit Pillars only": {
            "pillar_weights": deepcopy(literature["pillar_weights"]),
            "sub_weights": deepcopy(current["sub_weights"]),
        },
    }


# ============================================================
# 3. PIT-SAFE HELPERS
# ============================================================

def strip_tz_index(index):
    idx = pd.to_datetime(index, errors="coerce")

    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            idx = idx.tz_convert(None)

        return idx.normalize()

    return idx


def pit_reindex(source, target_index):
    """
    Historical-safe reindex:
    only forward-fill information that was already available.
    NEVER backward-fill.
    """
    if not isinstance(source, pd.Series):
        return pd.Series(
            np.nan,
            index=target_index,
            dtype=float,
        )

    s = (
        pd.to_numeric(
            source,
            errors="coerce"
        )
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if s.empty:
        return pd.Series(
            np.nan,
            index=target_index,
            dtype=float,
        )

    s.index = strip_tz_index(s.index)

    s = (
        s[
            ~s.index.duplicated(
                keep="last"
            )
        ]
        .sort_index()
    )

    target_normalized = strip_tz_index(
        target_index
    )

    aligned = s.reindex(
        target_normalized,
        method="ffill"
    )

    aligned.index = target_index

    return aligned


def pit_normalize_to_percentile(
    series,
    lookback=252,
    invert=False,
):
    """
    Rolling z-score -> normal CDF -> 0..100.
    Uses only current/past observations. No bfill.
    """
    if not isinstance(series, pd.Series):
        return pd.Series(dtype=float)

    s = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .ffill()
    )

    if s.dropna().empty:
        return pd.Series(
            np.nan,
            index=series.index,
            dtype=float
        )

    min_periods = max(
        20,
        min(
            60,
            lookback // 4
        )
    )

    mean = s.rolling(
        lookback,
        min_periods=min_periods
    ).mean()

    std = (
        s.rolling(
            lookback,
            min_periods=min_periods
        )
        .std()
        .replace(
            0,
            np.nan
        )
    )

    z = (
        s - mean
    ) / std

    out = pd.Series(
        norm.cdf(z) * 100.0,
        index=series.index,
        dtype=float,
    )

    if invert:
        out = (
            100.0 - out
        )

    return (
        out
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .clip(
            0,
            100
        )
    )


def flatten_yf_field(data, field):
    if data is None or data.empty:
        return pd.DataFrame()

    if not isinstance(
        data.columns,
        pd.MultiIndex
    ):
        if field not in data.columns:
            return pd.DataFrame()

        result = data[
            [field]
        ].copy()

        result.columns = [
            "SINGLE"
        ]

        return result

    level0 = (
        data.columns
        .get_level_values(0)
    )

    level1 = (
        data.columns
        .get_level_values(1)
    )

    if field in level0:
        frame = data[
            field
        ].copy()

    elif field in level1:
        frame = data.xs(
            field,
            axis=1,
            level=1
        ).copy()

    else:
        return pd.DataFrame()

    if isinstance(
        frame,
        pd.Series
    ):
        frame = (
            frame
            .to_frame()
        )

    if isinstance(
        frame.columns,
        pd.MultiIndex
    ):
        frame.columns = (
            frame.columns
            .get_level_values(-1)
        )

    return frame


# ============================================================
# 4. HISTORICAL DATA SOURCES
# ============================================================


def _extract_single_yahoo_series(
    frame,
    field,
):
    """
    Extract a single Close/Volume series from a one-symbol yf.download result.
    Handles normal and MultiIndex column layouts.
    """
    if (
        frame is None
        or frame.empty
    ):
        return pd.Series(
            dtype=float
        )

    if isinstance(
        frame.columns,
        pd.MultiIndex
    ):
        level0 = (
            frame.columns
            .get_level_values(0)
        )

        level1 = (
            frame.columns
            .get_level_values(1)
        )

        if field in level0:
            obj = frame[
                field
            ]

        elif field in level1:
            obj = frame.xs(
                field,
                axis=1,
                level=1
            )

        else:
            return pd.Series(
                dtype=float
            )

        if isinstance(
            obj,
            pd.DataFrame
        ):
            if obj.shape[1] == 0:
                return pd.Series(
                    dtype=float
                )

            obj = obj.iloc[
                :,
                0
            ]

    else:
        if field not in frame.columns:
            return pd.Series(
                dtype=float
            )

        obj = frame[
            field
        ]

    result = pd.to_numeric(
        obj,
        errors="coerce"
    ).dropna()

    if not result.empty:
        result.index = strip_tz_index(
            result.index
        )

    return result


def _download_yahoo_single_with_retry(
    ticker,
    start_date,
    attempts=2,
):
    """
    Per-symbol Yahoo fallback.

    Batch download remains the preferred path. This function is only used
    when one symbol is missing or empty in the batch result.
    """
    last_error = None

    for attempt in range(
        1,
        int(
            attempts
        ) + 1
    ):
        try:
            frame = yf.download(
                ticker,
                start=str(
                    start_date
                ),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="column",
            )

            close = (
                _extract_single_yahoo_series(
                    frame,
                    "Close"
                )
            )

            volume = (
                _extract_single_yahoo_series(
                    frame,
                    "Volume"
                )
            )

            if not close.empty:
                return (
                    close,
                    volume,
                    True,
                    (
                        "Einzelabruf erfolgreich"
                        if attempt == 1
                        else (
                            f"Einzelabruf erfolgreich "
                            f"(Versuch {attempt})"
                        )
                    )
                )

            last_error = (
                "Einzelabruf ohne verwertbare Close-Daten"
            )

        except Exception as exc:
            last_error = (
                f"{type(exc).__name__}: "
                f"{str(exc)[:120]}"
            )

    return (
        pd.Series(
            dtype=float
        ),
        pd.Series(
            dtype=float
        ),
        False,
        (
            last_error
            or "Yahoo Einzelabruf fehlgeschlagen"
        )
    )


def fetch_yahoo_research_bundle(
    asset_name,
    start_date,
):
    """
    Yahoo research bundle with a two-stage retrieval strategy:

    1) Preferred: one batch request for all required symbols.
    2) Fallback: every missing/empty symbol is downloaded individually.

    This avoids losing the whole research sample because a single Yahoo
    symbol failed inside a multi-ticker request.
    """
    cfg = ASSET_CONFIGS[
        asset_name
    ]

    ticker_map = {
        cfg["ticker"]: "asset",
        cfg["volatility_ticker"]: "volatility",
        "DX-Y.NYB": "dxy",
    }

    # Credit/MOVE are active only when the asset actually has a
    # non-zero early-warning pillar. WTI has 0% early-warning weight
    # in both Current and Literature Prior, so unnecessary requests
    # must not be allowed to weaken the WTI research run.
    if asset_name != "WTI Crude Oil":
        ticker_map.update(
            {
                "^MOVE": "move",
                "HYG": "hyg",
                "LQD": "lqd",
            }
        )

    # VVIX is model-active only for S&P 500 and Nasdaq 100.
    # Do not request it for Gold/WTI/EURUSD: an unnecessary Yahoo
    # failure must not weaken an unrelated asset test.
    if asset_name in {
        "S&P 500",
        "Nasdaq 100",
    }:
        ticker_map["^VVIX"] = "vvix"

    tickers = list(
        dict.fromkeys(
            ticker_map.keys()
        )
    )

    source_status = {}

    batch_data = pd.DataFrame()

    try:
        batch_data = yf.download(
            tickers,
            start=str(
                start_date
            ),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="column",
        )

        batch_ok = (
            batch_data is not None
            and not batch_data.empty
        )

    except Exception as exc:
        batch_ok = False

        source_status[
            "Yahoo Batch"
        ] = (
            False,
            (
                "Batch-Download fehlgeschlagen: "
                f"{type(exc).__name__}: "
                f"{str(exc)[:120]}"
            )
        )

    close = (
        flatten_yf_field(
            batch_data,
            "Close"
        )
        if batch_ok
        else pd.DataFrame()
    )

    volume = (
        flatten_yf_field(
            batch_data,
            "Volume"
        )
        if batch_ok
        else pd.DataFrame()
    )

    # Normalize any batch index.
    if not close.empty:
        close.index = strip_tz_index(
            close.index
        )

    if not volume.empty:
        volume.index = strip_tz_index(
            volume.index
        )

    # yfinance normally returns ticker names as columns in multi-symbol mode.
    # Keep only known symbols to avoid accidental column interpretation.
    if not close.empty:
        close = close[
            [
                col
                for col in close.columns
                if col in tickers
            ]
        ]

    if not volume.empty:
        volume = volume[
            [
                col
                for col in volume.columns
                if col in tickers
            ]
        ]

    # --------------------------------------------------------
    # Per-symbol repair pass
    # --------------------------------------------------------

    fallback_used = False

    for ticker in tickers:
        batch_series_ok = (
            ticker in close.columns
            and close[
                ticker
            ].notna().any()
        )

        if batch_series_ok:
            source_status[
                f"Yahoo {ticker}"
            ] = (
                True,
                "Batch-Abruf"
            )
            continue

        fallback_used = True

        (
            single_close,
            single_volume,
            single_ok,
            single_note,
        ) = _download_yahoo_single_with_retry(
            ticker,
            start_date,
            attempts=2,
        )

        source_status[
            f"Yahoo {ticker}"
        ] = (
            single_ok,
            (
                f"{single_note} · "
                "Fallback nach fehlendem Batch-Symbol"
            )
        )

        if single_ok:
            close[
                ticker
            ] = (
                single_close
            )

            if not single_volume.empty:
                volume[
                    ticker
                ] = (
                    single_volume
                )

    if batch_ok:
        source_status[
            "Yahoo Batch"
        ] = (
            True,
            (
                "Batch geladen; fehlende Symbole "
                "wurden einzeln repariert."
                if fallback_used
                else "Alle benötigten Symbole im Batch geladen."
            )
        )

    rename_map = {
        ticker: alias
        for ticker, alias
        in ticker_map.items()
    }

    close = (
        close.rename(
            columns=rename_map
        )
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .sort_index()
    )

    volume = (
        volume.rename(
            columns=rename_map
        )
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .sort_index()
    )

    if (
        "asset" not in close.columns
        or close[
            "asset"
        ].dropna().empty
    ):
        source_status[
            "Yahoo Asset-Preis"
        ] = (
            False,
            (
                f"{cfg['ticker']} konnte weder im Batch "
                "noch im Einzelabruf geladen werden."
            )
        )

        return (
            pd.DataFrame(),
            source_status,
        )

    price = (
        close[
            "asset"
        ]
        .dropna()
    )

    df = pd.DataFrame(
        index=price.index
    )

    df[
        "asset_price"
    ] = price

    # --------------------------------------------------------
    # TECHNICAL TREND
    # --------------------------------------------------------

    ma50 = price.rolling(
        50,
        min_periods=50
    ).mean()

    ma200 = price.rolling(
        200,
        min_periods=200
    ).mean()

    df[
        "distance_50ma"
    ] = (
        (
            price - ma50
        )
        /
        ma50.replace(
            0,
            np.nan
        )
        * 100.0
    )

    df[
        "distance_200ma"
    ] = (
        (
            price - ma200
        )
        /
        ma200.replace(
            0,
            np.nan
        )
        * 100.0
    )

    delta = (
        price.diff()
    )

    gain = (
        delta
        .clip(
            lower=0
        )
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    loss = (
        -delta
        .clip(
            upper=0
        )
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    rs = (
        gain
        /
        loss.replace(
            0,
            np.nan
        )
    )

    df[
        "rsi_momentum"
    ] = (
        100.0
        -
        100.0
        /
        (
            1.0 + rs
        )
    ).clip(
        0,
        100
    )

    df[
        "market_momentum"
    ] = (
        price
        .pct_change()
        .rolling(
            20,
            min_periods=10
        )
        .sum()
        * 100.0
    )

    # --------------------------------------------------------
    # VOLUME / OBV
    # --------------------------------------------------------

    if (
        "asset" in volume.columns
        and volume[
            "asset"
        ].notna().any()
    ):
        asset_volume = (
            pd.to_numeric(
                volume[
                    "asset"
                ],
                errors="coerce"
            )
            .reindex(
                price.index
            )
        )

        signed_volume = np.where(
            delta > 0,
            asset_volume,
            np.where(
                delta < 0,
                -asset_volume,
                0.0
            )
        )

        obv = pd.Series(
            signed_volume,
            index=price.index,
            dtype=float,
        ).cumsum()

        obv_ema = (
            obv.ewm(
                span=50,
                adjust=False
            ).mean()
        )

        df[
            "obv_momentum"
        ] = (
            (
                obv - obv_ema
            )
            /
            obv_ema.abs().replace(
                0,
                np.nan
            )
            * 100.0
        )

    else:
        df[
            "obv_momentum"
        ] = np.nan

    # --------------------------------------------------------
    # VOL / USD / MOVE / VVIX
    # --------------------------------------------------------

    for alias, source_col in [
        (
            "vix_score",
            "volatility"
        ),
        (
            "usd_index",
            "dxy"
        ),
        (
            "move_index",
            "move"
        ),
        (
            "vvix_score",
            "vvix"
        ),
    ]:
        if (
            source_col in close.columns
            and close[
                source_col
            ].notna().any()
        ):
            df[
                alias
            ] = pit_reindex(
                close[
                    source_col
                ],
                df.index
            )

        else:
            df[
                alias
            ] = np.nan

    # --------------------------------------------------------
    # CREDIT PROXY
    # --------------------------------------------------------

    if (
        "lqd" in close.columns
        and "hyg" in close.columns
        and close[
            "lqd"
        ].notna().any()
        and close[
            "hyg"
        ].notna().any()
    ):
        lqd = pit_reindex(
            close[
                "lqd"
            ],
            df.index
        )

        hyg = pit_reindex(
            close[
                "hyg"
            ],
            df.index
        )

        df[
            "credit_spreads"
        ] = (
            lqd
            /
            hyg.replace(
                0,
                np.nan
            )
        )

    else:
        df[
            "credit_spreads"
        ] = np.nan

    source_status[
        "Yahoo Research Sample"
    ] = (
        True,
        (
            f"{len(df):,} Handelstage · "
            f"{df.index.min():%d.%m.%Y}–"
            f"{df.index.max():%d.%m.%Y}"
        )
    )

    return (
        df,
        source_status,
    )


def fetch_cftc_history_research(
    market_code,
):
    params = {
        "cftc_contract_market_code": str(
            market_code
        ),
        "$limit": 5000,
        "$order": (
            "report_date_as_yyyy_mm_dd ASC"
        ),
    }

    try:
        r = requests.get(
            CFTC_LEGACY_API,
            params=params,
            headers=CFTC_HEADERS,
            timeout=30
        )

        r.raise_for_status()

        frame = pd.DataFrame(
            r.json()
        )

        if frame.empty:
            return (
                pd.Series(
                    dtype=float
                ),
                False,
                "CFTC: keine Historie."
            )

        required = [
            "report_date_as_yyyy_mm_dd",
            "noncomm_positions_long_all",
            "noncomm_positions_short_all",
        ]

        missing = [
            col
            for col in required
            if col not in frame.columns
        ]

        if missing:
            return (
                pd.Series(
                    dtype=float
                ),
                False,
                (
                    "CFTC-Felder fehlen: "
                    + ", ".join(
                        missing
                    )
                )
            )

        report_date = pd.to_datetime(
            frame[
                "report_date_as_yyyy_mm_dd"
            ],
            errors="coerce"
        )

        net = (
            pd.to_numeric(
                frame[
                    "noncomm_positions_long_all"
                ],
                errors="coerce"
            )
            -
            pd.to_numeric(
                frame[
                    "noncomm_positions_short_all"
                ],
                errors="coerce"
            )
        )

        result = pd.DataFrame(
            {
                "report_date": report_date,
                "value": net,
            }
        ).dropna()

        # PIT approximation: Tuesday report becomes known Friday.
        result["available_date"] = (
            result["report_date"]
            +
            pd.to_timedelta(
                CFTC_PUBLICATION_LAG_DAYS,
                unit="D"
            )
        )

        result = (
            result
            .sort_values(
                [
                    "available_date",
                    "report_date"
                ]
            )
            .drop_duplicates(
                "available_date",
                keep="last"
            )
        )

        series = (
            result
            .set_index(
                "available_date"
            )["value"]
            .sort_index()
        )

        return (
            series,
            True,
            (
                f"{len(series):,} COT-Berichte; "
                f"+{CFTC_PUBLICATION_LAG_DAYS} Tage "
                "Publikations-Lag approximiert."
            )
        )

    except Exception as exc:
        return (
            pd.Series(
                dtype=float
            ),
            False,
            (
                "CFTC-Fehler: "
                f"{str(exc)[:150]}"
            )
        )


def _fred_first_release_available_series(
    fred,
    series_id,
    realtime_start,
    realtime_end,
    chunk_years=2,
):
    """
    Build an availability-dated series from ALFRED metadata.

    FRED/ALFRED limits the number of vintage dates per observations query.
    Daily series such as DFII10 and RRPONTSYD can therefore fail when
    fredapi uses the default real-time range 1776-07-04 .. 9999-12-31.

    This helper queries bounded two-year real-time windows and concatenates
    the returned release records.
    """
    if not hasattr(fred, "get_series_all_releases"):
        raise AttributeError("fredapi has no get_series_all_releases")

    start_ts = pd.Timestamp(realtime_start).normalize()
    end_ts = pd.Timestamp(realtime_end).normalize()

    if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
        raise ValueError("Ungültiger ALFRED-Zeitraum")

    frames = []
    cursor = start_ts

    while cursor <= end_ts:
        chunk_end = min(
            cursor + pd.DateOffset(years=int(chunk_years)) - pd.Timedelta(days=1),
            end_ts,
        )

        chunk = fred.get_series_all_releases(
            series_id,
            realtime_start=cursor.strftime("%Y-%m-%d"),
            realtime_end=chunk_end.strftime("%Y-%m-%d"),
        )

        if chunk is not None and len(chunk) > 0:
            frames.append(pd.DataFrame(chunk))

        cursor = chunk_end + pd.Timedelta(days=1)

    if not frames:
        raise ValueError("Keine ALFRED-Releases im Research-Zeitraum")

    releases = pd.concat(frames, ignore_index=True)

    required = {"date", "realtime_start", "value"}
    if not required.issubset(releases.columns):
        raise ValueError("Unerwartete ALFRED-Spalten")

    releases["observation_date"] = pd.to_datetime(releases["date"], errors="coerce")
    releases["available_date"] = pd.to_datetime(releases["realtime_start"], errors="coerce")
    releases["value_numeric"] = pd.to_numeric(releases["value"], errors="coerce")

    releases = releases.dropna(
        subset=["observation_date", "available_date", "value_numeric"]
    )

    if releases.empty:
        raise ValueError("Keine verwertbaren ALFRED-Releases")

    releases = (
        releases
        .drop_duplicates(
            subset=["observation_date", "available_date", "value_numeric"],
            keep="last",
        )
        .sort_values(["observation_date", "available_date"])
    )

    first = (
        releases
        .groupby("observation_date", as_index=False)
        .first()
    )

    first = (
        first
        .sort_values(["available_date", "observation_date"])
        .drop_duplicates("available_date", keep="last")
    )

    return (
        first
        .set_index("available_date")["value_numeric"]
        .sort_index()
    )


def fetch_fred_research_series(
    series_id,
    research_start_date,
    prefer_first_release=True,
):
    if not FRED_API_KEY:
        return (
            pd.Series(
                dtype=float
            ),
            False,
            "FRED_API_KEY fehlt.",
            "offline"
        )

    try:
        fred = Fred(
            api_key=FRED_API_KEY
        )

        if prefer_first_release:
            try:
                realtime_start = (
                    pd.Timestamp(research_start_date)
                    - pd.DateOffset(years=1)
                ).date()

                realtime_end = (
                    pd.Timestamp.now()
                    .normalize()
                    .date()
                )

                series = (
                    _fred_first_release_available_series(
                        fred,
                        series_id,
                        realtime_start,
                        realtime_end,
                        chunk_years=2,
                    )
                )

                if not series.empty:
                    return (
                        series,
                        True,
                        (
                            "ALFRED/FRED First-Release "
                            "Availability"
                        ),
                        "first_release"
                    )

            except Exception as pit_exc:
                first_release_note = (
                    f"First-Release nicht nutzbar: "
                    f"{str(pit_exc)[:100]}"
                )

        else:
            first_release_note = (
                "First-Release deaktiviert."
            )

        # Explicit fallback: current-vintage series with conservative lag.
        fallback_observation_start = (
            pd.Timestamp(research_start_date)
            - pd.DateOffset(years=1)
        ).strftime("%Y-%m-%d")

        current = fred.get_series(
            series_id,
            observation_start=fallback_observation_start,
        )

        if (
            current is None
            or len(current) == 0
        ):
            return (
                pd.Series(
                    dtype=float
                ),
                False,
                (
                    "FRED-Serie leer. "
                    + first_release_note
                ),
                "offline"
            )

        current = pd.Series(
            current
        )

        current.index = pd.to_datetime(
            current.index,
            errors="coerce"
        )

        current = pd.to_numeric(
            current,
            errors="coerce"
        ).dropna()

        lag_days = int(
            FRED_FALLBACK_LAGS.get(
                series_id,
                1
            )
        )

        current.index = (
            current.index
            +
            pd.to_timedelta(
                lag_days,
                unit="D"
            )
        )

        return (
            current,
            True,
            (
                "Current-vintage FRED + "
                f"{lag_days}d konservativer Lag. "
                + first_release_note
            ),
            "current_vintage_fallback"
        )

    except Exception as exc:
        return (
            pd.Series(
                dtype=float
            ),
            False,
            (
                "FRED-Fehler: "
                f"{str(exc)[:150]}"
            ),
            "offline"
        )



def fetch_eia_wti_inventory_history(
    cache_version="v1.0.8",
):
    """
    Load official EIA weekly U.S. ending crude-oil stocks excluding SPR.

    Source series:
        WCESTUS1

    Source page:
        EIA Petroleum Navigator / official weekly history.

    Point-in-time handling:
        The source table is indexed by week-ending date (normally Friday).
        For research use, each observation becomes available +5 calendar
        days later, approximating the regular Wednesday EIA release.

    No EIA API key is required because this uses the official public
    historical table directly.
    """
    # Local import is intentional:
    # this makes the cached parser self-contained and guarantees that
    # a stale v1.0.6 NameError cannot recur because of global dependency
    # resolution. `cache_version` also creates a fresh Streamlit cache key.
    import re as regex

    _ = cache_version

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(compatible; QuantRegimeResearch/1.0)"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }

    try:
        response = requests.get(
            EIA_WTI_INVENTORY_URL,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()

        tables = pd.read_html(
            io.StringIO(
                response.text
            )
        )

        observations = []

        month_lookup = {
            "Jan": 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12,
        }

        for table in tables:
            if table is None or table.empty:
                continue

            for _, row in table.iterrows():
                values = [
                    "" if pd.isna(value)
                    else str(value).strip()
                    for value in row.tolist()
                ]

                if not values:
                    continue

                year_month_match = regex.search(
                    r"(?P<year>\d{4})-(?P<month>[A-Za-z]{3})",
                    values[0]
                )

                if not year_month_match:
                    continue

                year = int(
                    year_month_match.group(
                        "year"
                    )
                )

                month_name = (
                    year_month_match.group(
                        "month"
                    )
                    .title()
                )

                row_month = month_lookup.get(
                    month_name
                )

                if row_month is None:
                    continue

                # EIA table rows contain repeating End Date / Value pairs.
                for idx in range(
                    1,
                    len(values) - 1
                ):
                    date_text = (
                        values[idx]
                        .replace(
                            "\xa0",
                            " "
                        )
                        .strip()
                    )

                    if not regex.fullmatch(
                        r"\d{1,2}/\d{1,2}",
                        date_text
                    ):
                        continue

                    value_text = (
                        values[
                            idx + 1
                        ]
                        .replace(
                            ",",
                            ""
                        )
                        .replace(
                            "\xa0",
                            ""
                        )
                        .strip()
                    )

                    try:
                        numeric_value = float(
                            value_text
                        )
                    except Exception:
                        continue

                    try:
                        month_part, day_part = [
                            int(part)
                            for part in date_text.split(
                                "/"
                            )
                        ]

                        # The history table is grouped by year-month.
                        # Use the row's year; the explicit MM/DD is retained.
                        observation_date = pd.Timestamp(
                            year=year,
                            month=month_part,
                            day=day_part,
                        )
                    except Exception:
                        continue

                    observations.append(
                        (
                            observation_date,
                            numeric_value,
                        )
                    )

        if not observations:
            return (
                pd.Series(
                    dtype=float
                ),
                False,
                (
                    "EIA WCESTUS1: keine verwertbaren "
                    "Historienwerte aus offizieller Tabelle."
                ),
            )

        frame = pd.DataFrame(
            observations,
            columns=[
                "observation_date",
                "value",
            ]
        )

        frame = (
            frame
            .dropna()
            .sort_values(
                "observation_date"
            )
            .drop_duplicates(
                "observation_date",
                keep="last"
            )
        )

        frame[
            "available_date"
        ] = (
            frame[
                "observation_date"
            ]
            +
            pd.to_timedelta(
                EIA_INVENTORY_PUBLICATION_LAG_DAYS,
                unit="D"
            )
        )

        series = (
            frame
            .set_index(
                "available_date"
            )[
                "value"
            ]
            .sort_index()
            .astype(
                float
            )
        )

        if series.empty:
            return (
                pd.Series(
                    dtype=float
                ),
                False,
                "EIA WCESTUS1: leere Zeitreihe nach Parsing.",
            )

        return (
            series,
            True,
            (
                f"EIA {EIA_WTI_INVENTORY_SERIES}: "
                f"{len(series):,} Wochenwerte · "
                f"+{EIA_INVENTORY_PUBLICATION_LAG_DAYS} Tage "
                "Publikations-Lag approximiert."
            ),
        )

    except Exception as exc:
        return (
            pd.Series(
                dtype=float
            ),
            False,
            (
                "EIA WCESTUS1 Fehler: "
                f"{type(exc).__name__}: "
                f"{str(exc)[:150]}"
            ),
        )


def fetch_fear_greed_research():
    try:
        series, ok = (
            fetch_fear_and_greed()
        )

        if (
            ok
            and isinstance(
                series,
                pd.Series
            )
            and not series.empty
        ):
            return (
                series,
                True,
                (
                    f"{len(series):,} historische "
                    "CNN Fear-&-Greed-Beobachtungen."
                )
            )

        return (
            pd.Series(
                dtype=float
            ),
            False,
            "CNN Fear & Greed leer."
        )

    except Exception as exc:
        return (
            pd.Series(
                dtype=float
            ),
            False,
            (
                "CNN-Fehler: "
                f"{str(exc)[:120]}"
            )
        )


def fetch_pe_research():
    try:
        pe_result = fetch_sp500_pe_history()

        if (
            not isinstance(pe_result, tuple)
            or len(pe_result) < 2
        ):
            raise ValueError(
                "Unerwartete PE-Rückgabe aus regime_engine"
            )

        series = pe_result[0]
        ok = bool(pe_result[1])
        source_note = (
            str(pe_result[2])
            if len(pe_result) >= 3
            else "Multpl / PE-Historie"
        )

        if (
            ok
            and isinstance(
                series,
                pd.Series
            )
            and not series.empty
        ):
            series = (
                pd.to_numeric(
                    series,
                    errors="coerce"
                )
                .dropna()
            )

            series.index = (
                pd.to_datetime(
                    series.index,
                    errors="coerce"
                )
                +
                pd.to_timedelta(
                    1,
                    unit="D"
                )
            )

            return (
                series,
                True,
                (
                    f"{source_note}; "
                    "+1d Availability-Lag approximiert."
                )
            )

        return (
            pd.Series(
                dtype=float
            ),
            False,
            "PE-Historie leer."
        )

    except Exception as exc:
        return (
            pd.Series(
                dtype=float
            ),
            False,
            (
                "PE-Fehler: "
                f"{str(exc)[:120]}"
            )
        )


# ============================================================
# 5. RAW RESEARCH DATASET
# ============================================================

def build_research_dataset(
    asset_name,
    start_date,
    prefer_first_release=True,
):
    cfg = ASSET_CONFIGS[
        asset_name
    ]

    raw, yahoo_status = (
        fetch_yahoo_research_bundle(
            asset_name,
            start_date,
        )
    )

    status = dict(
        yahoo_status
    )

    pit_quality = []

    if raw.empty:
        return (
            pd.DataFrame(),
            status,
            pd.DataFrame()
        )

    # --------------------------------------------------------
    # CFTC
    # --------------------------------------------------------
    cot, cot_ok, cot_note = (
        fetch_cftc_history_research(
            cfg[
                "cot_market_code"
            ]
        )
    )

    raw[
        "cot_noncommercials"
    ] = pit_reindex(
        cot,
        raw.index
    )

    status["CFTC"] = (
        cot_ok,
        cot_note
    )

    pit_quality.append(
        {
            "Faktor": "CFTC Non-Commercials",
            "PIT-Qualität": "🟡 approximiert",
            "Methode": (
                "Reportdatum + 3 Kalendertage "
                "(Dienstag → Freitag)"
            ),
        }
    )

    # --------------------------------------------------------
    # CNN Fear & Greed
    # --------------------------------------------------------
    fg, fg_ok, fg_note = (
        fetch_fear_greed_research()
    )

    raw[
        "fear_greed"
    ] = pit_reindex(
        fg,
        raw.index
    )

    status[
        "CNN Fear & Greed"
    ] = (
        fg_ok,
        fg_note
    )

    pit_quality.append(
        {
            "Faktor": "CNN Fear & Greed",
            "PIT-Qualität": "🟡 Quellenhistorie",
            "Methode": (
                "Historische CNN-Datumsreihe; "
                "keine Vintage-Rekonstruktion."
            ),
        }
    )

    # PCR stays out of all tested model weights.
    raw[
        "options_put_call"
    ] = np.nan

    # --------------------------------------------------------
    # FRED
    # --------------------------------------------------------
    fred_ids = [
        "WALCL",
        "WTREGEN",
        "RRPONTSYD",
        "FEDFUNDS",
        "DFII10",
    ]

    fred_series = {}
    fred_modes = {}

    for series_id in fred_ids:
        series, ok, note, mode = (
            fetch_fred_research_series(
                series_id,
                start_date,
                prefer_first_release
            )
        )

        fred_series[
            series_id
        ] = series

        fred_modes[
            series_id
        ] = mode

        status[
            f"FRED {series_id}"
        ] = (
            ok,
            note
        )

        pit_quality.append(
            {
                "Faktor": f"FRED {series_id}",
                "PIT-Qualität": (
                    "🟢 First Release"
                    if mode == "first_release"
                    else (
                        "🟡 Current Vintage + Lag"
                        if mode
                        == "current_vintage_fallback"
                        else "🔴 offline"
                    )
                ),
                "Methode": note,
            }
        )

    wal = pit_reindex(
        fred_series.get(
            "WALCL",
            pd.Series(
                dtype=float
            )
        ),
        raw.index
    )

    tga = pit_reindex(
        fred_series.get(
            "WTREGEN",
            pd.Series(
                dtype=float
            )
        ),
        raw.index
    )

    rrp = pit_reindex(
        fred_series.get(
            "RRPONTSYD",
            pd.Series(
                dtype=float
            )
        ),
        raw.index
    )

    fed = pit_reindex(
        fred_series.get(
            "FEDFUNDS",
            pd.Series(
                dtype=float
            )
        ),
        raw.index
    )

    real_yield = pit_reindex(
        fred_series.get(
            "DFII10",
            pd.Series(
                dtype=float
            )
        ),
        raw.index
    )

    # Units as in the production engine:
    # WALCL / WTREGEN = USD millions, RRP = USD billions.
    raw[
        "net_liquidity"
    ] = (
        wal / 1000.0
        -
        tga / 1000.0
        -
        rrp
    )

    raw[
        "fed_policy"
    ] = fed

    raw[
        "real_yields"
    ] = real_yield

    if asset_name == "WTI Crude Oil":
        (
            eia_inventories,
            eia_inventory_ok,
            eia_inventory_note,
        ) = fetch_eia_wti_inventory_history(
            cache_version="v1.0.8"
        )

        raw[
            "inventories"
        ] = pit_reindex(
            eia_inventories,
            raw.index
        )

        status[
            "EIA WCESTUS1"
        ] = (
            eia_inventory_ok,
            eia_inventory_note
        )

        pit_quality.append(
            {
                "Faktor": "EIA WCESTUS1 – Crude Inventories",
                "PIT-Qualität": (
                    "🟡 offizieller EIA-Release-Lag"
                    if eia_inventory_ok
                    else "🔴 offline"
                ),
                "Methode": eia_inventory_note,
            }
        )

    else:
        raw[
            "inventories"
        ] = np.nan

    # --------------------------------------------------------
    # S&P PE
    # --------------------------------------------------------
    if asset_name == "S&P 500":
        pe, pe_ok, pe_note = (
            fetch_pe_research()
        )

        raw[
            "pe_valuation"
        ] = pit_reindex(
            pe,
            raw.index
        )

        status[
            "S&P 500 PE"
        ] = (
            pe_ok,
            pe_note
        )

        pit_quality.append(
            {
                "Faktor": "S&P 500 PE",
                "PIT-Qualität": "🟡 approximiert",
                "Methode": pe_note,
            }
        )

    else:
        raw[
            "pe_valuation"
        ] = np.nan

    # --------------------------------------------------------
    # PIT summary for Yahoo-derived factors.
    # --------------------------------------------------------
    pit_quality.append(
        {
            "Faktor": "Yahoo Markt-/Technikdaten",
            "PIT-Qualität": "🟢 Daily OHLC",
            "Methode": (
                "Historische Tagesdaten; "
                "nur ffill, kein bfill."
            ),
        }
    )

    return (
        raw,
        status,
        pd.DataFrame(
            pit_quality
        )
    )


# ============================================================
# 6. NORMALIZATION & MODEL SCORES
# ============================================================

def normalize_research_factors(
    raw,
    asset_name,
):
    cfg = ASSET_CONFIGS[
        asset_name
    ]

    norm_df = pd.DataFrame(
        index=raw.index
    )

    factor_columns = [
        col
        for col in raw.columns
        if col != "asset_price"
    ]

    for col in factor_columns:
        norm_df[col] = (
            pit_normalize_to_percentile(
                raw[col],
                LOOKBACK_CONFIG.get(
                    col,
                    252
                ),
                col
                in cfg[
                    "invert_inverts"
                ],
            )
        )

    return norm_df


def pillar_score_and_coverage(
    norm_df,
    weights,
):
    factors = [
        factor
        for factor in weights
        if factor in norm_df.columns
    ]

    if not factors:
        return (
            pd.Series(
                50.0,
                index=norm_df.index
            ),
            pd.Series(
                0.0,
                index=norm_df.index
            ),
        )

    w = pd.Series(
        {
            factor: float(
                weights[factor]
            )
            for factor in factors
        },
        dtype=float,
    )

    total_weight = float(
        w.sum()
    )

    if total_weight <= 0:
        return (
            pd.Series(
                50.0,
                index=norm_df.index
            ),
            pd.Series(
                0.0,
                index=norm_df.index
            ),
        )

    sub = (
        norm_df[
            factors
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    valid = sub.notna()

    available_weight = (
        valid.dot(
            w
        )
    )

    weighted_sum = (
        sub
        .fillna(
            0.0
        )
        .dot(
            w
        )
    )

    score = (
        weighted_sum
        .div(
            available_weight.replace(
                0,
                np.nan
            )
        )
        .fillna(
            50.0
        )
        .clip(
            0,
            100
        )
    )

    coverage = (
        available_weight
        /
        total_weight
        * 100.0
    ).clip(
        0,
        100
    )

    return (
        score,
        coverage
    )


def model_score_frame(
    norm_df,
    model_cfg,
):
    result = pd.DataFrame(
        index=norm_df.index
    )

    pillar_weights = (
        model_cfg[
            "pillar_weights"
        ]
    )

    sub_weights = (
        model_cfg[
            "sub_weights"
        ]
    )

    active_pillars = [
        p
        for p, weight
        in pillar_weights.items()
        if float(weight) > 0
    ]

    for pillar in active_pillars:
        score, coverage = (
            pillar_score_and_coverage(
                norm_df,
                sub_weights.get(
                    pillar,
                    {}
                )
            )
        )

        result[
            f"Pillar::{pillar}"
        ] = score

        result[
            f"Coverage::{pillar}"
        ] = coverage

    base_weights = normalize_weight_dict(
        {
            p: pillar_weights[p]
            for p in active_pillars
        }
    )

    score_matrix = pd.DataFrame(
        {
            p: result[
                f"Pillar::{p}"
            ]
            for p in active_pillars
        },
        index=result.index
    )

    coverage_matrix = pd.DataFrame(
        {
            p: (
                result[
                    f"Coverage::{p}"
                ]
                .fillna(
                    0.0
                )
                .clip(
                    0,
                    100
                )
                / 100.0
            )
            for p in active_pillars
        },
        index=result.index,
    )

    base_w = pd.Series(
        base_weights,
        dtype=float,
    )

    effective = (
        coverage_matrix
        .mul(
            base_w,
            axis=1
        )
        .where(
            score_matrix.notna(),
            0.0
        )
    )

    effective_sum = (
        effective
        .sum(
            axis=1
        )
    )

    weighted_score = (
        score_matrix
        .fillna(
            0.0
        )
        .mul(
            effective
        )
        .sum(
            axis=1
        )
    )

    result[
        "Final_Regime_Score"
    ] = (
        weighted_score
        .div(
            effective_sum.replace(
                0,
                np.nan
            )
        )
        .clip(
            0,
            100
        )
    )

    result[
        "Model_Data_Coverage"
    ] = (
        coverage_matrix
        .mul(
            base_w,
            axis=1
        )
        .sum(
            axis=1
        )
        * 100.0
    ).clip(
        0,
        100
    )

    return result


def build_all_model_scores(
    raw,
    asset_name,
):
    norm_df = normalize_research_factors(
        raw,
        asset_name
    )

    configs = all_model_configs(
        asset_name
    )

    frames = {}

    for model_name, cfg in configs.items():
        frames[
            model_name
        ] = model_score_frame(
            norm_df,
            cfg
        )

    return (
        norm_df,
        configs,
        frames
    )


# ============================================================
# 7. FORWARD TARGETS
# ============================================================

def forward_max_adverse_excursion(
    price,
    horizon=20,
):
    """
    Maximum Adverse Excursion (MAE) from today's close over the next
    ``horizon`` trading days.

    This is intentionally NOT a classical peak-to-trough maximum drawdown.
    It measures the worst future price excursion relative to the current
    observation date, which is the relevant risk target for the regime test.
    """
    values = (
        pd.to_numeric(
            price,
            errors="coerce"
        )
        .values
    )

    out = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    for i in range(
        len(values)
    ):
        if (
            not np.isfinite(
                values[i]
            )
            or i + horizon
            >= len(values)
        ):
            continue

        future = values[
            i + 1:
            i + horizon + 1
        ]

        if (
            len(future) < horizon
            or not np.all(
                np.isfinite(
                    future
                )
            )
        ):
            continue

        path_returns = (
            future
            / values[i]
            - 1.0
        )

        out[i] = float(
            np.min(
                path_returns
            )
        )

    return pd.Series(
        out,
        index=price.index,
        dtype=float,
    )


def build_forward_targets(
    price,
    asset_name,
):
    price = pd.to_numeric(
        price,
        errors="coerce"
    )

    targets = pd.DataFrame(
        index=price.index
    )

    daily_returns = (
        price
        .pct_change()
    )

    for horizon in FORWARD_HORIZONS:
        targets[
            f"Fwd_Return_{horizon}D"
        ] = (
            price.shift(
                -horizon
            )
            /
            price
            - 1.0
        )

    targets[
        "Fwd_MAE_20D"
    ] = forward_max_adverse_excursion(
        price,
        20
    )

    # Realized volatility of the next 20 daily returns.
    reversed_future_vol = (
        daily_returns
        .shift(
            -1
        )
        .iloc[::-1]
        .rolling(
            20,
            min_periods=20
        )
        .std()
        .iloc[::-1]
        * np.sqrt(
            252
        )
    )

    targets[
        "Fwd_Realized_Vol_20D"
    ] = reversed_future_vol

    threshold = (
        STRESS_MAE_THRESHOLDS[
            asset_name
        ]
    )

    targets[
        "Stress_Event_20D"
    ] = (
        targets[
            "Fwd_MAE_20D"
        ]
        <= threshold
    ).astype(
        float
    )

    targets.loc[
        targets[
            "Fwd_MAE_20D"
        ].isna(),
        "Stress_Event_20D"
    ] = np.nan

    return targets


