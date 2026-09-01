"""
Regime Backtest Lab v1.0.6
==========================

Separate research environment for the Market Regime Dashboard.

Purpose
-------
Compare three weighting systems on identical historical factor data:

A) Current production weights
B) Literature Prior v1
C) Equal Weight benchmark

Research principles
-------------------
- Production dashboard / TradePilot remain untouched.
- No backward filling (no bfill) in the historical factor pipeline.
- CFTC data is shifted to an approximate publication-availability date.
- FRED attempts first-release / ALFRED-style availability where fredapi
  exposes get_series_all_releases(); otherwise a conservative current-vintage
  fallback with explicit availability lags is used.
- Missing factors are dynamically reweighted and reduce model coverage.
- Results distinguish "real-world sample" and "common sample".
- Main tests:
    * Spearman IC: 5D / 20D / 60D
    * Directional accuracy
    * Quintile monotonicity + Q5-Q1 spread
    * Forward Maximum Adverse Excursion (MAE) / realized volatility
    * Stress-event AUC
    * Rolling IC
    * Block-bootstrap confidence interval for Literature - Current IC
    * Leave-one-pillar-out ablation
"""

from __future__ import annotations

import io
import math
from copy import deepcopy
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm, rankdata, spearmanr

import streamlit as st
import yfinance as yf
from fredapi import Fred

import plotly.graph_objects as go

from regime_engine import (
    ASSET_CONFIGS,
    SUB_WEIGHTS_BASE,
    LOOKBACK_CONFIG,
    FRED_API_KEY,
    fetch_fear_and_greed,
    fetch_sp500_pe_history,
)


# ============================================================
# 0. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Regime Backtest Lab",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Regime Backtest Lab v1.0.6 – WTI / EIA Inventory Fix")
st.caption(
    "Research-Modul – Current vs. Literature Prior v1 vs. Equal Weight. "
    "Der produktive Regime-Code wird nicht verändert."
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


@st.cache_data(
    ttl=3600,
    show_spinner=False
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


@st.cache_data(
    ttl=86400,
    show_spinner=False
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


@st.cache_data(
    ttl=86400,
    show_spinner=False
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



@st.cache_data(
    ttl=21600,
    show_spinner=False
)
def fetch_eia_wti_inventory_history():
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

                year_month_match = re.search(
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

                    if not re.fullmatch(
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


@st.cache_data(
    ttl=14400,
    show_spinner=False
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


@st.cache_data(
    ttl=86400,
    show_spinner=False
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

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
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
        ) = fetch_eia_wti_inventory_history()

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


# ============================================================
# 8. METRICS
# ============================================================

def safe_spearman(
    x,
    y,
):
    x = pd.to_numeric(
        x,
        errors="coerce"
    )

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    valid = (
        x.notna()
        & y.notna()
    )

    if valid.sum() < 20:
        return np.nan

    if (
        x[valid].nunique() < 2
        or y[valid].nunique() < 2
    ):
        return np.nan

    return float(
        spearmanr(
            x[valid],
            y[valid]
        ).statistic
    )


def directional_accuracy(
    score,
    forward_return,
    bull_threshold=60.0,
    bear_threshold=40.0,
):
    df = pd.DataFrame(
        {
            "score": score,
            "ret": forward_return,
        }
    ).dropna()

    signals = df[
        (
            df["score"]
            >= bull_threshold
        )
        |
        (
            df["score"]
            <= bear_threshold
        )
    ].copy()

    if signals.empty:
        return (
            np.nan,
            0
        )

    bull_correct = (
        (
            signals["score"]
            >= bull_threshold
        )
        &
        (
            signals["ret"]
            > 0
        )
    )

    bear_correct = (
        (
            signals["score"]
            <= bear_threshold
        )
        &
        (
            signals["ret"]
            < 0
        )
    )

    correct = (
        bull_correct
        |
        bear_correct
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


def quintile_statistics(
    score,
    forward_return,
):
    df = pd.DataFrame(
        {
            "score": score,
            "ret": forward_return,
        }
    ).dropna()

    if (
        len(df) < 100
        or df[
            "score"
        ].nunique() < 5
    ):
        return (
            pd.DataFrame(),
            np.nan,
            np.nan,
        )

    try:
        df[
            "quintile"
        ] = pd.qcut(
            df[
                "score"
            ],
            q=5,
            labels=[
                "Q1",
                "Q2",
                "Q3",
                "Q4",
                "Q5",
            ],
            duplicates="drop",
        )

    except Exception:
        return (
            pd.DataFrame(),
            np.nan,
            np.nan,
        )

    table = (
        df
        .groupby(
            "quintile",
            observed=True
        )["ret"]
        .agg(
            [
                "mean",
                "median",
                "count",
            ]
        )
    )

    if (
        "Q1" in table.index
        and "Q5" in table.index
    ):
        spread = float(
            table.loc[
                "Q5",
                "mean"
            ]
            -
            table.loc[
                "Q1",
                "mean"
            ]
        )
    else:
        spread = np.nan

    if len(
        table
    ) >= 3:
        monotonicity = safe_spearman(
            pd.Series(
                range(
                    1,
                    len(table) + 1
                )
            ),
            table[
                "mean"
            ].reset_index(
                drop=True
            )
        )
    else:
        monotonicity = np.nan

    return (
        table,
        spread,
        monotonicity,
    )


def binary_auc(
    predictor,
    event,
    higher_predictor_means_event=True,
):
    """
    Mann-Whitney rank AUC without sklearn.
    """
    df = pd.DataFrame(
        {
            "x": predictor,
            "y": event,
        }
    ).dropna()

    if df.empty:
        return np.nan

    y = (
        df[
            "y"
        ]
        .astype(
            int
        )
        .values
    )

    x = (
        df[
            "x"
        ]
        .astype(
            float
        )
        .values
    )

    if not higher_predictor_means_event:
        x = -x

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
        x
    )

    sum_pos = float(
        ranks[
            y == 1
        ].sum()
    )

    auc = (
        sum_pos
        -
        n_pos
        * (
            n_pos + 1
        )
        / 2.0
    ) / (
        n_pos
        * n_neg
    )

    return float(
        auc
    )


def metric_table_for_models(
    model_frames,
    targets,
    common_sample,
    min_coverage,
):
    rows = []

    if common_sample:
        common = pd.Series(
            True,
            index=targets.index
        )

        for frame in model_frames.values():
            common &= (
                frame[
                    "Final_Regime_Score"
                ].notna()
                &
                (
                    frame[
                        "Model_Data_Coverage"
                    ]
                    >= min_coverage
                )
            )

    else:
        common = None

    for model_name in MODEL_ORDER:
        frame = model_frames[
            model_name
        ]

        if common_sample:
            base_mask = common.copy()
        else:
            base_mask = (
                frame[
                    "Final_Regime_Score"
                ].notna()
                &
                (
                    frame[
                        "Model_Data_Coverage"
                    ]
                    >= min_coverage
                )
            )

        score = frame[
            "Final_Regime_Score"
        ].where(
            base_mask
        )

        coverage = frame[
            "Model_Data_Coverage"
        ].where(
            base_mask
        )

        row = {
            "Modell": model_name,
            "N": int(
                base_mask.sum()
            ),
            "Ø Coverage": float(
                coverage.mean()
            ),
        }

        for horizon in FORWARD_HORIZONS:
            ret = targets[
                f"Fwd_Return_{horizon}D"
            ]

            row[
                f"IC {horizon}D"
            ] = safe_spearman(
                score,
                ret
            )

            accuracy, n_signals = (
                directional_accuracy(
                    score,
                    ret
                )
            )

            row[
                f"Direction {horizon}D"
            ] = accuracy

            row[
                f"Signals {horizon}D"
            ] = n_signals

        q_table, spread, mono = (
            quintile_statistics(
                score,
                targets[
                    "Fwd_Return_20D"
                ]
            )
        )

        row[
            "Q5-Q1 20D"
        ] = spread

        row[
            "Quintile Monotonicity"
        ] = mono

        # Low regime score should identify stress, so use -score.
        row[
            "Stress AUC 20D"
        ] = binary_auc(
            score,
            targets[
                "Stress_Event_20D"
            ],
            higher_predictor_means_event=False,
        )

        row[
            "Score vs FwdVol 20D"
        ] = safe_spearman(
            score,
            -targets[
                "Fwd_Realized_Vol_20D"
            ]
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def absolute_validity_assessment(
    metric_row,
):
    """
    Absolute model-validity test.

    Relative superiority is not enough: a model must also demonstrate
    useful absolute predictive characteristics.

    Direction gate:
      - at least 2 of 3 forward ICs > 0
      - 20D directional accuracy >= 50%

    Risk gate:
      - Stress AUC 20D >= 0.50

    Strict combined gate:
      - Direction gate AND Risk gate

    The separate gates are shown because a regime model can be useful as a
    risk-state model even when it is not a directional-return predictor.
    """
    ic_values = [
        float(
            metric_row.get(
                f"IC {horizon}D",
                np.nan
            )
        )
        for horizon in FORWARD_HORIZONS
    ]

    positive_ic_count = sum(
        bool(
            np.isfinite(value)
            and value > 0
        )
        for value in ic_values
    )

    direction_20d = float(
        metric_row.get(
            "Direction 20D",
            np.nan
        )
    )

    stress_auc = float(
        metric_row.get(
            "Stress AUC 20D",
            np.nan
        )
    )

    direction_gate = (
        positive_ic_count >= 2
        and np.isfinite(
            direction_20d
        )
        and direction_20d >= 0.50
    )

    risk_gate = (
        np.isfinite(
            stress_auc
        )
        and stress_auc >= 0.50
    )

    strict_gate = (
        direction_gate
        and risk_gate
    )

    return {
        "positive_ic_count": positive_ic_count,
        "direction_20d": direction_20d,
        "stress_auc": stress_auc,
        "direction_gate": bool(
            direction_gate
        ),
        "risk_gate": bool(
            risk_gate
        ),
        "strict_gate": bool(
            strict_gate
        ),
    }


def absolute_validity_table(
    metric_index,
):
    rows = []

    for model_name in MODEL_ORDER:
        assessment = (
            absolute_validity_assessment(
                metric_index.loc[
                    model_name
                ]
            )
        )

        rows.append(
            {
                "Modell": model_name,
                "Positive ICs": (
                    f"{assessment['positive_ic_count']}/3"
                ),
                "Direction 20D": (
                    assessment[
                        "direction_20d"
                    ]
                ),
                "Stress AUC 20D": (
                    assessment[
                        "stress_auc"
                    ]
                ),
                "Direction-Gate": (
                    "✅"
                    if assessment[
                        "direction_gate"
                    ]
                    else "❌"
                ),
                "Risk-Gate": (
                    "✅"
                    if assessment[
                        "risk_gate"
                    ]
                    else "❌"
                ),
                "Strict Combined Gate": (
                    "✅"
                    if assessment[
                        "strict_gate"
                    ]
                    else "❌"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 9. ROLLING / BOOTSTRAP / ABLATION
# ============================================================

def rolling_ic(
    score,
    target,
    window=504,
    min_periods=252,
):
    joined = pd.DataFrame(
        {
            "score": score,
            "target": target,
        }
    )

    result = pd.Series(
        np.nan,
        index=joined.index,
        dtype=float,
    )

    for i in range(
        len(joined)
    ):
        start = max(
            0,
            i - window + 1
        )

        sample = (
            joined.iloc[
                start:
                i + 1
            ]
            .dropna()
        )

        if len(
            sample
        ) < min_periods:
            continue

        result.iloc[
            i
        ] = safe_spearman(
            sample[
                "score"
            ],
            sample[
                "target"
            ]
        )

    return result


def block_bootstrap_ic_difference(
    current_score,
    literature_score,
    target,
    block_length=20,
    n_boot=500,
    seed=42,
):
    df = pd.DataFrame(
        {
            "current": current_score,
            "literature": literature_score,
            "target": target,
        }
    ).dropna()

    n = len(
        df
    )

    if (
        n < max(
            200,
            block_length * 4
        )
    ):
        return {
            "observed": np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "prob_positive": np.nan,
            "n": n,
        }

    observed = (
        safe_spearman(
            df[
                "literature"
            ],
            df[
                "target"
            ]
        )
        -
        safe_spearman(
            df[
                "current"
            ],
            df[
                "target"
            ]
        )
    )

    rng = np.random.default_rng(
        seed
    )

    values = []

    max_start = (
        n - block_length
    )

    for _ in range(
        int(
            n_boot
        )
    ):
        indices = []

        while len(
            indices
        ) < n:
            start = int(
                rng.integers(
                    0,
                    max_start + 1
                )
            )

            indices.extend(
                range(
                    start,
                    start + block_length
                )
            )

        indices = np.asarray(
            indices[
                :n
            ],
            dtype=int,
        )

        boot = (
            df.iloc[
                indices
            ]
        )

        diff = (
            safe_spearman(
                boot[
                    "literature"
                ],
                boot[
                    "target"
                ]
            )
            -
            safe_spearman(
                boot[
                    "current"
                ],
                boot[
                    "target"
                ]
            )
        )

        if np.isfinite(
            diff
        ):
            values.append(
                diff
            )

    if not values:
        return {
            "observed": observed,
            "lower": np.nan,
            "upper": np.nan,
            "prob_positive": np.nan,
            "n": n,
        }

    arr = np.asarray(
        values,
        dtype=float,
    )

    return {
        "observed": float(
            observed
        ),
        "lower": float(
            np.quantile(
                arr,
                0.025
            )
        ),
        "upper": float(
            np.quantile(
                arr,
                0.975
            )
        ),
        "prob_positive": float(
            np.mean(
                arr > 0
            )
        ),
        "n": n,
    }


def ablated_score(
    model_frame,
    model_cfg,
    excluded_pillar,
):
    active = [
        p
        for p, weight
        in model_cfg[
            "pillar_weights"
        ].items()
        if (
            float(weight) > 0
            and p != excluded_pillar
        )
    ]

    if not active:
        return pd.Series(
            np.nan,
            index=model_frame.index
        )

    weights = normalize_weight_dict(
        {
            p: model_cfg[
                "pillar_weights"
            ][p]
            for p in active
        }
    )

    score_matrix = pd.DataFrame(
        {
            p: model_frame[
                f"Pillar::{p}"
            ]
            for p in active
        },
        index=model_frame.index,
    )

    coverage_matrix = pd.DataFrame(
        {
            p: (
                model_frame[
                    f"Coverage::{p}"
                ]
                / 100.0
            )
            for p in active
        },
        index=model_frame.index,
    )

    w = pd.Series(
        weights,
        dtype=float,
    )

    effective = (
        coverage_matrix
        .mul(
            w,
            axis=1
        )
        .where(
            score_matrix.notna(),
            0.0
        )
    )

    denom = effective.sum(
        axis=1
    )

    numerator = (
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

    return numerator.div(
        denom.replace(
            0,
            np.nan
        )
    )


def literature_ablation_table(
    model_frame,
    model_cfg,
    targets,
    min_coverage,
):
    base_score = (
        model_frame[
            "Final_Regime_Score"
        ]
        .where(
            model_frame[
                "Model_Data_Coverage"
            ]
            >= min_coverage
        )
    )

    base_ic = safe_spearman(
        base_score,
        targets[
            "Fwd_Return_20D"
        ]
    )

    base_acc, _ = (
        directional_accuracy(
            base_score,
            targets[
                "Fwd_Return_20D"
            ]
        )
    )

    rows = []

    for pillar, weight in model_cfg[
        "pillar_weights"
    ].items():
        if float(
            weight
        ) <= 0:
            continue

        score_without = (
            ablated_score(
                model_frame,
                model_cfg,
                pillar
            )
        )

        score_without = (
            score_without
            .where(
                model_frame[
                    "Model_Data_Coverage"
                ]
                >= min_coverage
            )
        )

        ic = safe_spearman(
            score_without,
            targets[
                "Fwd_Return_20D"
            ]
        )

        acc, n_signal = (
            directional_accuracy(
                score_without,
                targets[
                    "Fwd_Return_20D"
                ]
            )
        )

        rows.append(
            {
                "Entfernte Säule": pillar,
                "Basisgewicht": float(
                    weight
                ),
                "IC ohne Säule": ic,
                "Δ IC vs vollständig": (
                    ic - base_ic
                    if (
                        np.isfinite(ic)
                        and np.isfinite(
                            base_ic
                        )
                    )
                    else np.nan
                ),
                "Direction ohne Säule": acc,
                "Δ Direction": (
                    acc - base_acc
                    if (
                        np.isfinite(acc)
                        and np.isfinite(
                            base_acc
                        )
                    )
                    else np.nan
                ),
                "Signals": n_signal,
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "Δ IC vs vollständig",
        ascending=True,
        na_position="last",
    )


# ============================================================
# 10. UI SIDEBAR
# ============================================================

with st.sidebar:
    st.header(
        "⚙️ Research Setup"
    )

    selected_asset = (
        st.selectbox(
            "Asset",
            list(
                ASSET_CONFIGS
            ),
            index=3,
            help=(
                "Für den nächsten Vergleich ist WTI Crude Oil voreingestellt. "
                "Alle anderen Assets bleiben weiterhin auswählbar."
            ),
        )
    )

    history_years = (
        st.select_slider(
            "Yahoo-Historie",
            options=[
                8,
                10,
                12,
                15,
            ],
            value=15,
            help=(
                "Länger ist besser, aber die gemeinsame "
                "Stichprobe wird durch den jüngsten Faktor begrenzt."
            ),
        )
    )

    min_coverage = (
        st.slider(
            "Mindest-Coverage für Auswertung",
            min_value=0,
            max_value=100,
            value=60,
            step=5,
        )
    )

    sample_mode = (
        st.radio(
            "Vergleichsstichprobe",
            [
                "Common Sample",
                "Real-world Sample",
            ],
            index=0,
            help=(
                "Common Sample nutzt nur Tage, an denen alle drei "
                "Modelle die Coverage-Schwelle erfüllen."
            ),
        )
    )

    prefer_first_release = (
        st.checkbox(
            "FRED First-Release / ALFRED bevorzugen",
            value=True,
            help=(
                "Falls fredapi First-Release-Metadaten laden kann, "
                "werden Werte nach ihrem ersten Veröffentlichungsdatum "
                "verfügbar gemacht. Sonst transparenter Fallback."
            ),
        )
    )

    bootstrap_runs = (
        st.select_slider(
            "Bootstrap-Wiederholungen",
            options=[
                200,
                500,
                1000,
            ],
            value=500,
        )
    )

    bootstrap_block = (
        st.select_slider(
            "Bootstrap-Blocklänge (Tage)",
            options=[
                10,
                20,
                40,
                60,
            ],
            value=20,
        )
    )


# ============================================================
# 11. LOAD RESEARCH DATA
# ============================================================

today = (
    pd.Timestamp.now()
    .normalize()
)

start_date = (
    today
    -
    pd.DateOffset(
        years=int(
            history_years
        )
    )
).date()

with st.spinner(
    "Baue point-in-time-orientierten Research-Datensatz …"
):
    raw_df, source_status, pit_quality = (
        build_research_dataset(
            selected_asset,
            start_date,
            prefer_first_release,
        )
    )

if raw_df.empty:
    st.error(
        "Der Research-Datensatz konnte nicht aufgebaut werden."
    )

    with st.expander(
        "Datenquellen"
    ):
        for source, (
            ok,
            note
        ) in source_status.items():
            st.write(
                (
                    "🟢"
                    if ok
                    else "🔴"
                ),
                source,
                "–",
                note,
            )

    st.stop()

with st.spinner(
    "Berechne Current, Literature Prior und Equal Weight …"
):
    (
        norm_df,
        model_configs,
        model_frames,
    ) = build_all_model_scores(
        raw_df,
        selected_asset,
    )

    diagnostic_configs = diagnostic_model_configs(
        selected_asset
    )

    diagnostic_frames = {
        model_name: model_score_frame(
            norm_df,
            cfg
        )
        for model_name, cfg
        in diagnostic_configs.items()
    }

    targets = (
        build_forward_targets(
            raw_df[
                "asset_price"
            ],
            selected_asset,
        )
    )


# ============================================================
# 12. DATA QUALITY / PIT STATUS
# ============================================================

st.markdown("---")

if selected_asset == "Nasdaq 100":
    with st.expander(
        "🔎 Nasdaq-Quellencheck",
        expanded=True
    ):
        st.markdown(
            """
Für den Nasdaq-Test werden die folgenden Quellen getrennt behandelt:

- **Preis/Technik:** `NQ=F`
- **Nasdaq-Volatilität:** `^VXN`
- **Volatility-of-Volatility:** `^VVIX`
- **USD:** `DX-Y.NYB`
- **Bond-Volatilität:** `^MOVE`
- **Credit Proxy:** `LQD / HYG`
- **CFTC Non-Commercials:** Marktcode `209742`
- **Sentiment:** CNN Fear & Greed
- **Makro/FRED:** `WALCL`, `WTREGEN`, `RRPONTSYD`, `FEDFUNDS`, `DFII10`

Yahoo wird zunächst als Batch geladen. Fehlende Yahoo-Reihen werden danach
**einzeln mit separatem Retry nachgeladen**. FRED/ALFRED wird weiterhin in
begrenzten Real-Time-Fenstern geladen, damit Daily-Serien nicht an der
Vintage-Date-Grenze scheitern.

Für Nasdaq wird **kein S&P-500-PE abgerufen**. Die im alten Current-Modell
vorhandene Fundamental-Säule besitzt daher keine Daten-Coverage und wird
durch die Coverage-Logik nicht künstlich mit einem neutralen Wert gefüllt.
"""
        )

elif selected_asset == "Gold (XAU/USD)":
    with st.expander(
        "🔎 Gold-Quellencheck",
        expanded=True
    ):
        st.markdown(
            """
Für den Gold-Test werden die Quellen bewusst getrennt und nur dann geladen,
wenn sie für das Gold-Modell tatsächlich benötigt werden:

- **Preis/Technik & Volumen/OBV:** `GC=F`
- **Gold-Volatilität:** `^GVZ`
- **USD-Index:** `DX-Y.NYB`
- **Bond-Volatilität / Frühwarnung:** `^MOVE`
- **Credit Proxy / Frühwarnung:** `LQD / HYG`
- **CFTC Non-Commercials Gold:** Marktcode `088691`
- **Sentiment-Zusatz:** CNN Fear & Greed
- **Makro/FRED:** `WALCL`, `WTREGEN`, `RRPONTSYD`, `FEDFUNDS`, `DFII10`

**VVIX wird beim Gold-Test nicht geladen**, weil er weder im Current- noch im
Literature-Prior-Goldmodell aktiv ist. Damit kann ein unnötiger `^VVIX`-Abruf
den Gold-Test nicht mehr beeinträchtigen.

Yahoo läuft wie beim Nasdaq zunächst im Batch. Fehlt `GC=F`, `^GVZ`,
`DX-Y.NYB`, `^MOVE`, `HYG` oder `LQD`, wird genau diese Reihe anschließend
**einzeln mit Retry nachgeladen**. FRED/ALFRED bleibt auf begrenzte
2-Jahres-Real-Time-Fenster aufgeteilt.

Für Gold existiert bewusst **keine Fundamentale Säule** (0 %). Es wird daher
weder S&P-PE noch WTI-Inventories abgerufen.

Hinweis zur Interpretation: `GC=F` ist ein kontinuierlicher Yahoo-Futures-
Datensatz. Roll-/Kontrakteffekte können historische Returns beeinflussen;
dieser Test bleibt trotzdem konsistent mit dem produktiven Gold-Ticker des
Regime-Modells.
"""
        )

elif selected_asset == "WTI Crude Oil":
    with st.expander(
        "🛢️ WTI-Quellencheck",
        expanded=True
    ):
        st.markdown(
            """
Für den WTI-Test werden nur die tatsächlich relevanten Quellen geladen:

- **Preis/Technik & Volumen/OBV:** `CL=F`
- **Öl-Volatilität:** `^OVX`
- **USD-Index:** `DX-Y.NYB`
- **CFTC Non-Commercials WTI:** Marktcode `067651`
- **Sentiment-Zusatz:** CNN Fear & Greed
- **Makro/FRED:** `WALCL`, `WTREGEN`, `RRPONTSYD`, `FEDFUNDS`, `DFII10`
- **Fundamental / physischer Markt:** EIA `WCESTUS1` – Weekly U.S. Ending Stocks excluding SPR of Crude Oil

Die WTI-Lagerbestände werden **nicht über FRED** geladen. `WCESTUS1`
ist eine offizielle EIA-Petroleum-Serie. Das Lab lädt die öffentliche
EIA-Historientabelle direkt und verschiebt die Wochenwerte für den
Point-in-Time-Test konservativ um +5 Kalendertage (Freitag → Mittwoch).

Yahoo läuft wie bei Nasdaq und Gold zunächst im Batch. Fehlt `CL=F`, `^OVX`
oder `DX-Y.NYB`, wird genau diese Reihe anschließend **einzeln mit Retry**
nachgeladen.

**MOVE, HYG/LQD und VVIX werden für WTI nicht unnötig abgefragt**, weil die
Frühwarnsäule im WTI-Modell sowohl in Current als auch in Literature Prior
mit 0 % gewichtet ist. Dadurch kann ein irrelevanter Feed-Ausfall den
WTI-Test nicht schwächen.

Der Preisfeed `CL=F` ist ein kontinuierlicher Yahoo-Futures-Datensatz.
Roll-/Kontrakteffekte können historische WTI-Returns beeinflussen. Der Test
bleibt damit konsistent zum produktiven WTI-Ticker des Regime-Modells.
"""
        )

st.markdown("---")
st.subheader(
    "1️⃣ Datenbasis & Point-in-Time-Qualität"
)

q1, q2, q3, q4 = (
    st.columns(4)
)

q1.metric(
    "Historische Handelstage",
    f"{len(raw_df):,}"
)

q2.metric(
    "Start",
    raw_df.index.min().strftime(
        "%d.%m.%Y"
    )
)

q3.metric(
    "Ende",
    raw_df.index.max().strftime(
        "%d.%m.%Y"
    )
)

live_sources = sum(
    1
    for ok, _
    in source_status.values()
    if ok
)

q4.metric(
    "Quellen erfolgreich",
    (
        f"{live_sources}/"
        f"{len(source_status)}"
    )
)

with st.expander(
    "📡 Quellenstatus",
    expanded=False
):
    for source, (
        ok,
        note
    ) in source_status.items():
        st.markdown(
            f"{'🟢' if ok else '⚠️'} "
            f"**{source}:** {note}"
        )

with st.expander(
    "🕒 Point-in-Time / Look-ahead Audit",
    expanded=True
):
    st.info(
        "Research-Regel: Es wird **nirgendwo rückwärts aufgefüllt "
        "(`bfill`)**. Daten dürfen erst ab ihrem bekannten bzw. "
        "approximierten Verfügbarkeitstag in den Score eingehen."
    )

    if not pit_quality.empty:
        st.dataframe(
            pit_quality,
            hide_index=True,
            use_container_width=True,
        )

    st.warning(
        "Der Test ist dadurch deutlich sauberer als eine normale "
        "historische Dashboard-Kurve, aber nicht jede externe Quelle "
        "lässt sich perfekt als institutioneller Vintage-Datensatz "
        "rekonstruieren. Besonders CFTC-Publikationsfeiertage, CNN "
        "Fear & Greed und Multpl-PE bleiben approximative PIT-Komponenten."
    )


# ============================================================
# 13. WEIGHT COMPARISON
# ============================================================

st.markdown("---")

if selected_asset == "Gold (XAU/USD)":
    st.info(
        "🟡 **Gold-Retest v1.0.4:** Faktoren, Gewichtungen und Quellen "
        "sind gegenüber dem vorherigen Gold-Lauf unverändert. Neu sind "
        "nur die strengere Absolute-Validity-Prüfung und die korrekte "
        "Bezeichnung des 20D-Risikoziels als Maximum Adverse Excursion "
        "(MAE). Dadurch bleibt der Vergleich zum vorherigen Gold-Test fair."
    )

elif selected_asset == "WTI Crude Oil":
    st.info(
        "🛢️ **WTI-Test v1.0.6:** Methodik, Faktoren und Gewichte bleiben "
        "unverändert. Korrigiert wurde ausschließlich die Inventarquelle: "
        "`WCESTUS1` ist eine EIA-Serie und wird jetzt direkt aus der "
        "offiziellen EIA-Historie geladen. Relative Modellvergleiche, "
        "Absolute Direction-/Risk-Gates, Common Sample, Block-Bootstrap, "
        "Ablation und MAE bleiben unverändert."
    )

st.subheader(
    "2️⃣ Gewichte der drei Modelle"
)

pillar_weight_rows = []

for model_name in MODEL_ORDER:
    cfg = model_configs[
        model_name
    ]

    row = {
        "Modell": model_name
    }

    for pillar in PILLARS:
        row[pillar] = (
            cfg[
                "pillar_weights"
            ].get(
                pillar,
                0.0
            )
            * 100.0
        )

    pillar_weight_rows.append(
        row
    )

pillar_weight_table = pd.DataFrame(
    pillar_weight_rows
)

st.dataframe(
    pillar_weight_table.style.format(
        {
            pillar: "{:.1f}%"
            for pillar in PILLARS
        }
    ),
    hide_index=True,
    use_container_width=True,
)

with st.expander(
    "Subgewichtungen vergleichen"
):
    selected_pillar = (
        st.selectbox(
            "Säule",
            PILLARS,
            key="research_subweight_pillar",
        )
    )

    factor_union = []

    for model_name in MODEL_ORDER:
        for factor in model_configs[
            model_name
        ][
            "sub_weights"
        ].get(
            selected_pillar,
            {}
        ):
            if factor not in factor_union:
                factor_union.append(
                    factor
                )

    rows = []

    for factor in factor_union:
        row = {
            "Faktor": factor
        }

        for model_name in MODEL_ORDER:
            row[
                model_name
            ] = (
                model_configs[
                    model_name
                ][
                    "sub_weights"
                ].get(
                    selected_pillar,
                    {}
                ).get(
                    factor,
                    0.0
                )
                * 100.0
            )

        rows.append(
            row
        )

    if rows:
        sub_table = pd.DataFrame(
            rows
        )

        st.dataframe(
            sub_table.style.format(
                {
                    model_name: "{:.1f}%"
                    for model_name
                    in MODEL_ORDER
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    else:
        st.info(
            "Diese Säule besitzt für das ausgewählte Asset "
            "keine aktiven Faktoren."
        )


# ============================================================
# 14. MAIN METRICS
# ============================================================

st.markdown("---")
st.subheader(
    "3️⃣ Out-of-Sample-orientierter Modellvergleich"
)

common_sample = (
    sample_mode
    == "Common Sample"
)

metrics = (
    metric_table_for_models(
        model_frames,
        targets,
        common_sample,
        float(
            min_coverage
        ),
    )
)

display_metrics = metrics.copy()

percent_cols = [
    "Ø Coverage",
    "Direction 5D",
    "Direction 20D",
    "Direction 60D",
    "Q5-Q1 20D",
]

for col in [
    "Direction 5D",
    "Direction 20D",
    "Direction 60D",
]:
    display_metrics[
        col
    ] = (
        display_metrics[
            col
        ]
        * 100.0
    )

display_metrics[
    "Q5-Q1 20D"
] = (
    display_metrics[
        "Q5-Q1 20D"
    ]
    * 100.0
)

st.dataframe(
    display_metrics.style.format(
        {
            "Ø Coverage": "{:.1f}%",
            "IC 5D": "{:+.3f}",
            "IC 20D": "{:+.3f}",
            "IC 60D": "{:+.3f}",
            "Direction 5D": "{:.1f}%",
            "Direction 20D": "{:.1f}%",
            "Direction 60D": "{:.1f}%",
            "Q5-Q1 20D": "{:+.2f}%",
            "Quintile Monotonicity": "{:+.3f}",
            "Stress AUC 20D": "{:.3f}",
            "Score vs FwdVol 20D": "{:+.3f}",
        },
        na_rep="n/a",
    ),
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "IC = Spearman-Korrelation zwischen heutigem Score und künftigem "
    "Return. Stress-AUC > 0,50 bedeutet, dass niedrige Scores "
    "bevorstehende Drawdown-Ereignisse besser als Zufall trennen."
)

if selected_asset == "Gold (XAU/USD)":
    with st.expander(
        "🪙 Gold-spezifische Interpretationshilfe",
        expanded=True
    ):
        st.markdown(
            """
Beim Gold-Test achten wir zusätzlich besonders auf drei Punkte:

1. **Real Yields / Fed / USD** – Literature Prior verschiebt innerhalb der
   Makrosäule deutlich mehr Gewicht auf Realrenditen und USD.
2. **CFTC Non-Commercials** – die Literature Prior erhöht deren Anteil in der
   Positionierung von 80 % auf 85 % und reduziert CNN Fear & Greed von 20 %
   auf 15 %.
3. **Trend** – die Literature Prior hebt die gesamte technische Säule von
   15 % auf 25 % an.

Entscheidend ist nicht nur, ob der Forward-Return-IC steigt. Wir prüfen ebenso,
ob niedrige Scores künftige Gold-MAE-/Stressbewegungen und höhere Volatilität zuverlässiger
identifizieren. Dadurch können wir erneut zwischen **Risk-State** und
**Forward-Direction** unterscheiden.
"""
        )

elif selected_asset == "WTI Crude Oil":
    with st.expander(
        "🛢️ WTI-spezifische Interpretationshilfe",
        expanded=True
    ):
        st.markdown(
            """
Beim WTI-Test prüfen wir besonders vier Punkte:

1. **Inventories (`WCESTUS1`)** – Current gewichtet die Fundamentale Säule
   mit 10 %, Literature Prior mit 25 %. Das ist die wichtigste
   asset-spezifische Änderung.
2. **CFTC Non-Commercials** – Literature Prior erhöht den Anteil innerhalb
   der Positionierung auf 90 % und reduziert CNN Fear & Greed auf 10 %.
3. **Makro** – Literature Prior reduziert die Makrosäule von 30 % auf 18 %
   und verschiebt innerhalb der Säule mehr Gewicht auf USD und Net Liquidity.
4. **Trend** – die technische Säule steigt von 20 % auf 25 %.

Für WTI ist deshalb besonders interessant, ob die physischen Lagerdaten
tatsächlich inkrementelle Forward-Information liefern oder primär den
gegenwärtigen Ölmarkt-Zustand beschreiben. Wir trennen weiterhin strikt
zwischen **Forward Direction**, **Risk-State** und relativer Verbesserung
gegenüber Current.
"""
        )


# ============================================================
# 14A. WEIGHT-DECOMPOSITION DIAGNOSTIC
# ============================================================

st.markdown("---")
st.subheader(
    "3A️⃣ Woher kommt eine Veränderung? Subgewichte vs. Säulengewichte"
)

st.caption(
    "Dieser Diagnoseblock trennt die Literature Prior in zwei Schritte: "
    "(D) nur neue Subgewichte bei alten Säulengewichten und "
    "(E) nur neue Säulengewichte bei alten Subgewichten. Dadurch wird "
    "sichtbar, ob ein möglicher Vorteil aus der internen Faktorverteilung "
    "oder aus der Verteilung der sechs Säulen stammt."
)

diag_frames = {
    MODEL_CURRENT: model_frames[MODEL_CURRENT],
    **diagnostic_frames,
    MODEL_LITERATURE: model_frames[MODEL_LITERATURE],
}

if common_sample:
    diag_common = pd.Series(
        True,
        index=targets.index
    )

    for frame in diag_frames.values():
        diag_common &= (
            frame["Final_Regime_Score"].notna()
            &
            (
                frame["Model_Data_Coverage"]
                >= min_coverage
            )
        )
else:
    diag_common = None

diag_rows = []

for model_name, frame in diag_frames.items():
    if diag_common is not None:
        mask = diag_common
    else:
        mask = (
            frame["Final_Regime_Score"].notna()
            &
            (
                frame["Model_Data_Coverage"]
                >= min_coverage
            )
        )

    score_diag = frame["Final_Regime_Score"].where(mask)

    diag_acc, diag_signals = directional_accuracy(
        score_diag,
        targets["Fwd_Return_20D"]
    )

    diag_rows.append(
        {
            "Modellstufe": model_name,
            "N": int(mask.sum()),
            "IC 20D": safe_spearman(
                score_diag,
                targets["Fwd_Return_20D"]
            ),
            "Direction 20D": diag_acc,
            "Signals": diag_signals,
            "Stress AUC 20D": binary_auc(
                score_diag,
                targets["Stress_Event_20D"],
                higher_predictor_means_event=False
            ),
            "Ø Coverage": float(
                frame["Model_Data_Coverage"]
                .where(mask)
                .mean()
            ),
        }
    )

diag_table = pd.DataFrame(diag_rows)

if not diag_table.empty:
    diag_display = diag_table.copy()
    diag_display["Direction 20D"] *= 100.0

    st.dataframe(
        diag_display.style.format(
            {
                "IC 20D": "{:+.3f}",
                "Direction 20D": "{:.1f}%",
                "Stress AUC 20D": "{:.3f}",
                "Ø Coverage": "{:.1f}%",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    current_ic_diag = float(
        diag_table.loc[
            diag_table["Modellstufe"] == MODEL_CURRENT,
            "IC 20D"
        ].iloc[0]
    )

    sub_ic_diag = float(
        diag_table.loc[
            diag_table["Modellstufe"] == "D · Lit Subweights only",
            "IC 20D"
        ].iloc[0]
    )

    pillar_ic_diag = float(
        diag_table.loc[
            diag_table["Modellstufe"] == "E · Lit Pillars only",
            "IC 20D"
        ].iloc[0]
    )

    if all(
        np.isfinite(x)
        for x in [
            current_ic_diag,
            sub_ic_diag,
            pillar_ic_diag,
        ]
    ):
        sub_delta = sub_ic_diag - current_ic_diag
        pillar_delta = pillar_ic_diag - current_ic_diag

        st.caption(
            f"Isolierter ΔIC durch Literature-Subgewichte: "
            f"**{sub_delta:+.3f}** · "
            f"isolierter ΔIC durch Literature-Säulengewichte: "
            f"**{pillar_delta:+.3f}**."
        )


# ============================================================
# 15. QUINTILE TEST
# ============================================================

st.markdown("---")
st.subheader(
    "4️⃣ Quintile-Test – zukünftiger 20D Return"
)

q_cols = st.columns(
    3
)

for i, model_name in enumerate(
    MODEL_ORDER
):
    frame = model_frames[
        model_name
    ]

    score = frame[
        "Final_Regime_Score"
    ].where(
        frame[
            "Model_Data_Coverage"
        ]
        >= min_coverage
    )

    if common_sample:
        common_mask = pd.Series(
            True,
            index=frame.index
        )

        for comparison_frame in model_frames.values():
            common_mask &= (
                comparison_frame[
                    "Final_Regime_Score"
                ].notna()
                &
                (
                    comparison_frame[
                        "Model_Data_Coverage"
                    ]
                    >= min_coverage
                )
            )

        score = score.where(
            common_mask
        )

    q_table, spread, mono = (
        quintile_statistics(
            score,
            targets[
                "Fwd_Return_20D"
            ]
        )
    )

    with q_cols[i]:
        st.markdown(
            f"**{model_name}**"
        )

        if q_table.empty:
            st.info(
                "Nicht genügend Daten."
            )

        else:
            show = q_table.copy()

            show[
                "mean"
            ] = (
                show[
                    "mean"
                ]
                * 100.0
            )

            show[
                "median"
            ] = (
                show[
                    "median"
                ]
                * 100.0
            )

            st.dataframe(
                show.style.format(
                    {
                        "mean": "{:+.2f}%",
                        "median": "{:+.2f}%",
                        "count": "{:.0f}",
                    }
                ),
                use_container_width=True,
            )

            st.caption(
                f"Q5−Q1: "
                f"{spread * 100:+.2f}% · "
                f"Monotonie: {mono:+.3f}"
            )


# ============================================================
# 16. ROLLING IC
# ============================================================

st.markdown("---")
st.subheader(
    "5️⃣ Rolling 2-Jahres-IC (20D Forward Return)"
)

rolling_fig = go.Figure()

for model_name in MODEL_ORDER:
    frame = model_frames[
        model_name
    ]

    score = (
        frame[
            "Final_Regime_Score"
        ]
        .where(
            frame[
                "Model_Data_Coverage"
            ]
            >= min_coverage
        )
    )

    roll = rolling_ic(
        score,
        targets[
            "Fwd_Return_20D"
        ],
        window=504,
        min_periods=252,
    )

    rolling_fig.add_trace(
        go.Scatter(
            x=roll.index,
            y=roll,
            mode="lines",
            name=model_name,
        )
    )

rolling_fig.add_hline(
    y=0.0,
    line_dash="dash",
)

rolling_fig.update_layout(
    height=430,
    yaxis_title="Rolling Spearman IC",
    xaxis_title="Datum",
    hovermode="x unified",
)

st.plotly_chart(
    rolling_fig,
    use_container_width=True,
)


# ============================================================
# 16A. CALENDAR-YEAR STABILITY
# ============================================================

st.markdown("---")
st.subheader(
    "5A️⃣ Stabilität nach Kalenderjahr – 20D IC"
)

st.caption(
    "Ein Modell sollte nicht nur durch ein einzelnes Krisenjahr gut aussehen. "
    "Die Tabelle zeigt den 20D-IC getrennt nach dem Datum, an dem der Score "
    "gebildet wurde. Jahre mit weniger als 80 verwertbaren Beobachtungen "
    "werden nicht bewertet."
)

yearly_rows = []

if common_sample:
    yearly_common = pd.Series(
        True,
        index=targets.index
    )

    for frame in model_frames.values():
        yearly_common &= (
            frame["Final_Regime_Score"].notna()
            &
            (
                frame["Model_Data_Coverage"]
                >= min_coverage
            )
        )
else:
    yearly_common = None

for year in sorted(set(targets.index.year)):
    row = {"Jahr": int(year)}
    sufficient_any = False

    for model_name in MODEL_ORDER:
        frame = model_frames[model_name]

        mask = (
            targets.index.year == year
        )

        if yearly_common is not None:
            mask = mask & yearly_common.values
        else:
            mask = (
                mask
                & frame["Final_Regime_Score"].notna().values
                & (
                    frame["Model_Data_Coverage"].values
                    >= min_coverage
                )
            )

        valid_n = int(
            (
                pd.Series(
                    mask,
                    index=targets.index
                )
                & targets["Fwd_Return_20D"].notna()
            ).sum()
        )

        if valid_n >= 80:
            sufficient_any = True
            score_year = frame["Final_Regime_Score"].where(mask)
            row[model_name] = safe_spearman(
                score_year,
                targets["Fwd_Return_20D"]
            )
        else:
            row[model_name] = np.nan

    if sufficient_any:
        yearly_rows.append(row)

yearly_table = pd.DataFrame(yearly_rows)

if yearly_table.empty:
    st.info(
        "Noch keine Kalenderjahre mit mindestens 80 verwertbaren "
        "20D-Beobachtungen."
    )
else:
    st.dataframe(
        yearly_table.style.format(
            {
                model_name: "{:+.3f}"
                for model_name in MODEL_ORDER
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    yearly_win_counts = {
        model_name: 0
        for model_name in MODEL_ORDER
    }

    comparable_years = 0

    for _, row in yearly_table.iterrows():
        vals = {
            model_name: row[model_name]
            for model_name in MODEL_ORDER
            if np.isfinite(row[model_name])
        }

        if len(vals) == len(MODEL_ORDER):
            comparable_years += 1
            winner = max(vals, key=vals.get)
            yearly_win_counts[winner] += 1

    if comparable_years > 0:
        st.caption(
            "Jahressiege beim höchsten 20D-IC: "
            + " · ".join(
                f"**{model}: {wins}/{comparable_years}**"
                for model, wins in yearly_win_counts.items()
            )
        )


# ============================================================
# 17. BLOCK BOOTSTRAP
# ============================================================

st.markdown("---")
st.subheader(
    "6️⃣ Block-Bootstrap – Literature Prior vs. Current"
)

current_frame = (
    model_frames[
        MODEL_CURRENT
    ]
)

literature_frame = (
    model_frames[
        MODEL_LITERATURE
    ]
)

bootstrap_common = (
    current_frame[
        "Final_Regime_Score"
    ].notna()
    &
    literature_frame[
        "Final_Regime_Score"
    ].notna()
    &
    (
        current_frame[
            "Model_Data_Coverage"
        ]
        >= min_coverage
    )
    &
    (
        literature_frame[
            "Model_Data_Coverage"
        ]
        >= min_coverage
    )
)

bootstrap = (
    block_bootstrap_ic_difference(
        current_frame[
            "Final_Regime_Score"
        ].where(
            bootstrap_common
        ),
        literature_frame[
            "Final_Regime_Score"
        ].where(
            bootstrap_common
        ),
        targets[
            "Fwd_Return_20D"
        ],
        block_length=int(
            bootstrap_block
        ),
        n_boot=int(
            bootstrap_runs
        ),
    )
)

bc1, bc2, bc3, bc4 = (
    st.columns(4)
)

bc1.metric(
    "Δ IC Literature − Current",
    (
        f"{bootstrap['observed']:+.4f}"
        if np.isfinite(
            bootstrap[
                "observed"
            ]
        )
        else "n/a"
    ),
)

bc2.metric(
    "95%-Intervall",
    (
        f"{bootstrap['lower']:+.4f} "
        f"bis {bootstrap['upper']:+.4f}"
        if (
            np.isfinite(
                bootstrap[
                    "lower"
                ]
            )
            and np.isfinite(
                bootstrap[
                    "upper"
                ]
            )
        )
        else "n/a"
    ),
)

bc3.metric(
    "P(Δ IC > 0)",
    (
        f"{bootstrap['prob_positive'] * 100:.1f}%"
        if np.isfinite(
            bootstrap[
                "prob_positive"
            ]
        )
        else "n/a"
    ),
)

bc4.metric(
    "Bootstrap-Sample",
    f"{bootstrap['n']:,}"
)

if (
    np.isfinite(
        bootstrap[
            "lower"
        ]
    )
    and bootstrap[
        "lower"
    ] > 0
):
    st.success(
        "🟢 Das 95%-Bootstrap-Intervall liegt vollständig über 0. "
        "Die Literature Prior zeigt in diesem Test robuste Mehrinformation."
    )

elif (
    np.isfinite(
        bootstrap[
            "upper"
        ]
    )
    and bootstrap[
        "upper"
    ] < 0
):
    st.error(
        "🔴 Das 95%-Bootstrap-Intervall liegt vollständig unter 0. "
        "Die Current-Gewichtung ist in diesem Test robuster."
    )

else:
    st.warning(
        "🟡 Kein eindeutiger statistischer Vorteil: "
        "Das 95%-Intervall umfasst 0."
    )


# ============================================================
# 18. ABLATION
# ============================================================

st.markdown("---")
st.subheader(
    "7️⃣ Ablation – welche Literature-Prior-Säule liefert Zusatznutzen?"
)

ablation = (
    literature_ablation_table(
        literature_frame,
        model_configs[
            MODEL_LITERATURE
        ],
        targets,
        min_coverage,
    )
)

if not ablation.empty:
    ablation_display = (
        ablation.copy()
    )

    ablation_display[
        "Basisgewicht"
    ] *= 100.0

    ablation_display[
        "Direction ohne Säule"
    ] *= 100.0

    ablation_display[
        "Δ Direction"
    ] *= 100.0

    st.dataframe(
        ablation_display.style.format(
            {
                "Basisgewicht": "{:.1f}%",
                "IC ohne Säule": "{:+.3f}",
                "Δ IC vs vollständig": "{:+.3f}",
                "Direction ohne Säule": "{:.1f}%",
                "Δ Direction": "{:+.1f} pp",
                "Signals": "{:.0f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Interpretation: Ein stark negativer Δ-IC nach Entfernen einer Säule "
        "spricht dafür, dass diese Säule inkrementelle Information liefert. "
        "Ein positiver Δ-IC wäre ein Warnsignal, dass die Säule im aktuellen "
        "Research-Sample eher schadet."
    )


# ============================================================
# 18A. GOLD FACTOR DIAGNOSTICS
# ============================================================

if selected_asset == "Gold (XAU/USD)":
    st.markdown("---")
    st.subheader(
        "7A️⃣ Gold-Faktoren – Einzelbeitrag zum 20D-Forward-Verhalten"
    )

    gold_factor_rows = []

    gold_factor_map = {
        "Fed Policy": "fed_policy",
        "Real Yields": "real_yields",
        "USD Index": "usd_index",
        "Net Liquidity": "net_liquidity",
        "CFTC Non-Commercials": "cot_noncommercials",
        "CNN Fear & Greed": "fear_greed",
        "OBV Momentum": "obv_momentum",
        "GVZ": "vix_score",
        "Distance 50MA": "distance_50ma",
        "Distance 200MA": "distance_200ma",
        "RSI": "rsi_momentum",
        "Credit Proxy": "credit_spreads",
        "MOVE": "move_index",
    }

    for label, factor in gold_factor_map.items():
        if factor not in norm_df.columns:
            continue

        factor_score = (
            pd.to_numeric(
                norm_df[factor],
                errors="coerce"
            )
        )

        valid_n = int(
            (
                factor_score.notna()
                & targets["Fwd_Return_20D"].notna()
            ).sum()
        )

        ic20 = safe_spearman(
            factor_score,
            targets["Fwd_Return_20D"]
        )

        stress_auc = binary_auc(
            factor_score,
            targets["Stress_Event_20D"],
            higher_predictor_means_event=False,
        )

        vol_relation = safe_spearman(
            factor_score,
            -targets["Fwd_Realized_Vol_20D"]
        )

        gold_factor_rows.append(
            {
                "Faktor": label,
                "N": valid_n,
                "IC 20D": ic20,
                "Stress AUC 20D": stress_auc,
                "Score vs niedrigere FwdVol": vol_relation,
            }
        )

    if gold_factor_rows:
        gold_factor_table = pd.DataFrame(
            gold_factor_rows
        ).sort_values(
            "Stress AUC 20D",
            ascending=False,
            na_position="last",
        )

        st.dataframe(
            gold_factor_table.style.format(
                {
                    "N": "{:.0f}",
                    "IC 20D": "{:+.3f}",
                    "Stress AUC 20D": "{:.3f}",
                    "Score vs niedrigere FwdVol": "{:+.3f}",
                },
                na_rep="n/a",
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Diese Tabelle ist diagnostisch und verändert keine Gewichte. "
            "Sie hilft zu erkennen, ob z. B. Real Yields, USD, COT oder "
            "Trend eher Forward-Richtung oder Stress-/Volatilitätsinformation liefern."
        )

elif selected_asset == "WTI Crude Oil":
    st.markdown("---")
    st.subheader(
        "7A️⃣ WTI-Faktoren – Einzelbeitrag zum 20D-Forward-Verhalten"
    )

    st.caption(
        "Besonderer Fokus liegt auf `WCESTUS1`. Der Faktor `inventories` "
        "ist im Modell bereits invertiert normalisiert: hohe Lagerbestände "
        "drücken den Faktor-Score, niedrige Lagerbestände erhöhen ihn."
    )

    wti_factor_rows = []

    wti_factor_map = {
        "Fed Policy": "fed_policy",
        "Real Yields": "real_yields",
        "USD Index": "usd_index",
        "Net Liquidity": "net_liquidity",
        "CFTC Non-Commercials": "cot_noncommercials",
        "CNN Fear & Greed": "fear_greed",
        "OBV Momentum": "obv_momentum",
        "OVX": "vix_score",
        "Distance 50MA": "distance_50ma",
        "Distance 200MA": "distance_200ma",
        "RSI": "rsi_momentum",
        "US Crude Inventories (WCESTUS1)": "inventories",
    }

    for label, factor in wti_factor_map.items():
        if factor not in norm_df.columns:
            continue

        factor_score = pd.to_numeric(
            norm_df[factor],
            errors="coerce"
        )

        valid_n = int(
            (
                factor_score.notna()
                & targets["Fwd_Return_20D"].notna()
            ).sum()
        )

        ic20 = safe_spearman(
            factor_score,
            targets["Fwd_Return_20D"]
        )

        stress_auc = binary_auc(
            factor_score,
            targets["Stress_Event_20D"],
            higher_predictor_means_event=False,
        )

        vol_relation = safe_spearman(
            factor_score,
            -targets["Fwd_Realized_Vol_20D"]
        )

        mae_relation = safe_spearman(
            factor_score,
            targets["Fwd_MAE_20D"]
        )

        wti_factor_rows.append(
            {
                "Faktor": label,
                "N": valid_n,
                "IC 20D": ic20,
                "Stress AUC 20D": stress_auc,
                "Score vs niedrigere FwdVol": vol_relation,
                "Score vs bessere FwdMAE": mae_relation,
            }
        )

    if wti_factor_rows:
        wti_factor_table = pd.DataFrame(
            wti_factor_rows
        ).sort_values(
            "IC 20D",
            ascending=False,
            na_position="last",
        )

        st.dataframe(
            wti_factor_table.style.format(
                {
                    "N": "{:.0f}",
                    "IC 20D": "{:+.3f}",
                    "Stress AUC 20D": "{:.3f}",
                    "Score vs niedrigere FwdVol": "{:+.3f}",
                    "Score vs bessere FwdMAE": "{:+.3f}",
                },
                na_rep="n/a",
            ),
            hide_index=True,
            use_container_width=True,
        )

        inventory_row = wti_factor_table[
            wti_factor_table["Faktor"]
            == "US Crude Inventories (WCESTUS1)"
        ]

        if not inventory_row.empty:
            inv = inventory_row.iloc[0]

            st.info(
                "📦 **Inventory-Diagnose:** "
                f"IC20 {inv['IC 20D']:+.3f} · "
                f"Stress-AUC {inv['Stress AUC 20D']:.3f} · "
                f"FwdVol-Bezug {inv['Score vs niedrigere FwdVol']:+.3f} · "
                f"MAE-Bezug {inv['Score vs bessere FwdMAE']:+.3f}. "
                "Damit können wir nach dem Export gesondert beurteilen, "
                "ob die höhere Literature-Prior-Fundamentalgewichtung "
                "empirisch gerechtfertigt ist."
            )

        st.caption(
            "Die Faktortabelle ist rein diagnostisch. Sie verändert keine "
            "Produktionsgewichte und dient dazu, den zusätzlichen Nutzen "
            "von Inventories, COT, OVX, USD, Makro und Trend getrennt zu prüfen."
        )


# ============================================================
# 19. SCORE HISTORY / COVERAGE
# ============================================================

st.markdown("---")
st.subheader(
    "8️⃣ Score-Historie & Coverage"
)

score_fig = go.Figure()

for model_name in MODEL_ORDER:
    frame = model_frames[
        model_name
    ]

    score_fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame[
                "Final_Regime_Score"
            ],
            mode="lines",
            name=model_name,
        )
    )

score_fig.update_layout(
    height=430,
    yaxis=dict(
        title="Regime Score",
        range=[
            0,
            100
        ],
    ),
    hovermode="x unified",
)

st.plotly_chart(
    score_fig,
    use_container_width=True,
)

coverage_rows = []

for model_name in MODEL_ORDER:
    frame = model_frames[
        model_name
    ]

    coverage_rows.append(
        {
            "Modell": model_name,
            "Ø Coverage": (
                frame[
                    "Model_Data_Coverage"
                ].mean()
            ),
            "Median Coverage": (
                frame[
                    "Model_Data_Coverage"
                ].median()
            ),
            f"Tage ≥ {min_coverage}%": int(
                (
                    frame[
                        "Model_Data_Coverage"
                    ]
                    >= min_coverage
                ).sum()
            ),
            "Erster valider Score": (
                frame[
                    "Final_Regime_Score"
                ]
                .dropna()
                .index.min()
                if frame[
                    "Final_Regime_Score"
                ].notna().any()
                else pd.NaT
            ),
        }
    )

coverage_table = pd.DataFrame(
    coverage_rows
)

st.dataframe(
    coverage_table.style.format(
        {
            "Ø Coverage": "{:.1f}%",
            "Median Coverage": "{:.1f}%",
        },
        na_rep="n/a",
    ),
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# 20. PRE-REGISTERED DECISION CHECK
# ============================================================

st.markdown("---")
st.subheader(
    "9️⃣ Vorab definierter Entscheidungscheck"
)

metric_index = (
    metrics
    .set_index(
        "Modell"
    )
)

current_row = (
    metric_index.loc[
        MODEL_CURRENT
    ]
)

literature_row = (
    metric_index.loc[
        MODEL_LITERATURE
    ]
)

equal_row = (
    metric_index.loc[
        MODEL_EQUAL
    ]
)

# ------------------------------------------------------------
# A. RELATIVE MODEL COMPARISON
# ------------------------------------------------------------

st.markdown(
    "### A. Relative Verbesserung gegenüber Current"
)

st.caption(
    "Diese Regeln beantworten nur, ob Literature Prior v1 "
    "**relativ besser** als Current ist. Ein relatives Plus "
    "reicht allein ausdrücklich nicht für eine produktive Übernahme."
)

relative_rules = []

# Rule 1: Literature better in at least 2 of 3 IC horizons.
ic_wins = sum(
    (
        literature_row[
            f"IC {h}D"
        ]
        >
        current_row[
            f"IC {h}D"
        ]
    )
    for h in FORWARD_HORIZONS
    if (
        np.isfinite(
            literature_row[
                f"IC {h}D"
            ]
        )
        and np.isfinite(
            current_row[
                f"IC {h}D"
            ]
        )
    )
)

relative_rules.append(
    (
        "IC: besser bei mindestens 2/3 Horizonten",
        ic_wins >= 2,
        f"{ic_wins}/3",
    )
)

# Rule 2: Directional 20D not worse by >1pp.
direction_delta = (
    literature_row[
        "Direction 20D"
    ]
    -
    current_row[
        "Direction 20D"
    ]
)

relative_rules.append(
    (
        "Directional Accuracy 20D nicht >1pp schlechter",
        (
            np.isfinite(
                direction_delta
            )
            and direction_delta
            >= -0.01
        ),
        (
            f"{direction_delta * 100:+.1f} pp"
            if np.isfinite(
                direction_delta
            )
            else "n/a"
        ),
    )
)

# Rule 3: Stress AUC not worse.
stress_delta = (
    literature_row[
        "Stress AUC 20D"
    ]
    -
    current_row[
        "Stress AUC 20D"
    ]
)

relative_rules.append(
    (
        "Stress-Erkennung nicht schlechter",
        (
            np.isfinite(
                stress_delta
            )
            and stress_delta
            >= -0.01
        ),
        (
            f"{stress_delta:+.3f}"
            if np.isfinite(
                stress_delta
            )
            else "n/a"
        ),
    )
)

# Rule 4: Literature beats equal on 20D IC.
equal_delta = (
    literature_row[
        "IC 20D"
    ]
    -
    equal_row[
        "IC 20D"
    ]
)

relative_rules.append(
    (
        "Literature schlägt Equal Weight beim 20D-IC",
        (
            np.isfinite(
                equal_delta
            )
            and equal_delta > 0
        ),
        (
            f"{equal_delta:+.3f}"
            if np.isfinite(
                equal_delta
            )
            else "n/a"
        ),
    )
)

# Rule 5: Bootstrap probability.
relative_rules.append(
    (
        "Bootstrap P(ΔIC>0) ≥ 75%",
        (
            np.isfinite(
                bootstrap[
                    "prob_positive"
                ]
            )
            and bootstrap[
                "prob_positive"
            ]
            >= .75
        ),
        (
            f"{bootstrap['prob_positive'] * 100:.1f}%"
            if np.isfinite(
                bootstrap[
                    "prob_positive"
                ]
            )
            else "n/a"
        ),
    )
)

relative_rule_df = pd.DataFrame(
    [
        {
            "Kriterium": label,
            "Erfüllt": (
                "✅"
                if passed
                else "❌"
            ),
            "Messwert": detail,
        }
        for label, passed, detail
        in relative_rules
    ]
)

st.dataframe(
    relative_rule_df,
    hide_index=True,
    use_container_width=True,
)

relative_passed_count = sum(
    bool(
        passed
    )
    for _, passed, _
    in relative_rules
)

# ------------------------------------------------------------
# B. ABSOLUTE VALIDITY
# ------------------------------------------------------------

st.markdown(
    "### B. Absolute Validität"
)

st.caption(
    "Dieser zweite Gate verhindert die Fehlinterpretation "
    "„weniger schlecht = gut“. Für einen Direction-Predictor "
    "müssen mindestens zwei der drei Forward-ICs positiv sein "
    "und die 20D Directional Accuracy mindestens 50 % erreichen. "
    "Für einen Risk-State-Filter muss die Stress-AUC mindestens "
    "0,50 erreichen."
)

absolute_table = (
    absolute_validity_table(
        metric_index
    )
)

st.dataframe(
    absolute_table.style.format(
        {
            "Direction 20D": "{:.1%}",
            "Stress AUC 20D": "{:.3f}",
        },
        na_rep="n/a",
    ),
    hide_index=True,
    use_container_width=True,
)

literature_absolute = (
    absolute_validity_assessment(
        literature_row
    )
)

# ------------------------------------------------------------
# C. FINAL MODEL-SELECTION VERDICT
# ------------------------------------------------------------

st.markdown(
    "### C. Gesamturteil Literature Prior v1"
)

relative_gate_passed = (
    relative_passed_count
    == len(
        relative_rules
    )
)

if (
    relative_gate_passed
    and literature_absolute[
        "strict_gate"
    ]
):
    st.success(
        "🟢 **RELATIVE + ABSOLUTE VALIDITÄT BESTANDEN.** "
        "Literature Prior v1 schlägt Current nach den relativen Regeln "
        "und besteht zugleich den strengen absoluten Direction- und "
        "Risk-Gate. Erst dann wäre eine weitere produktionsnahe "
        "Walk-Forward-Validierung gerechtfertigt."
    )

elif (
    relative_gate_passed
    and not literature_absolute[
        "strict_gate"
    ]
):
    st.error(
        "🔴 **RELATIV BESSER, ABER ABSOLUT NICHT VALIDE.** "
        "Literature Prior v1 kann Current statistisch schlagen, "
        "erfüllt aber nicht die Mindestanforderungen an absolute "
        "Direction-/Risk-Prognosequalität. Keine produktive "
        "Gewichtsübernahme."
    )

elif literature_absolute[
    "risk_gate"
] and not literature_absolute[
    "direction_gate"
]:
    st.warning(
        "🟡 **RISK-STATE-NUTZEN, ABER KEIN DIRECTION-PREDICTOR.** "
        "Die Literature Prior erfüllt den Risk-Gate, nicht aber den "
        "Direction-Gate. Der Score wäre eher als Risikozustandsfilter "
        "als als Long-/Short-Prognose zu interpretieren."
    )

elif literature_absolute[
    "direction_gate"
] and not literature_absolute[
    "risk_gate"
]:
    st.warning(
        "🟡 **DIRECTION-NUTZEN, ABER KEIN BELASTBARER RISK-FILTER.** "
        "Die Literature Prior erfüllt den Direction-Gate, nicht aber "
        "den Stress-AUC-Gate."
    )

else:
    st.error(
        "🔴 **KEINE AUSREICHENDE ABSOLUTE VALIDITÄT.** "
        f"Relative Kriterien: {relative_passed_count}/"
        f"{len(relative_rules)} erfüllt. "
        "Weder der vollständige Direction-Gate noch der Risk-Gate "
        "rechtfertigen derzeit eine produktive Neuinterpretation "
        "oder Gewichtsübernahme."
    )

st.info(
    "**Methodische Lesart:** Ein Modell darf nicht allein deshalb als "
    "verbessert gelten, weil z. B. IC −0,07 besser als IC −0,11 ist. "
    "Der neue Absolute-Validity-Gate prüft deshalb zusätzlich, ob das "
    "Modell überhaupt in der beabsichtigten Funktion einen Mindestnutzen "
    "gegenüber Zufall bzw. Null-Information zeigt."
)


# ============================================================
# 21. EXPORT
# ============================================================

st.markdown("---")
st.subheader(
    "🔟 Research-Export"
)

export = pd.DataFrame(
    index=raw_df.index
)

export[
    "Asset_Price"
] = raw_df[
    "asset_price"
]

for model_name in MODEL_ORDER:
    clean_name = (
        model_name
        .replace(
            " · ",
            "_"
        )
        .replace(
            " ",
            "_"
        )
    )

    export[
        f"Score_{clean_name}"
    ] = (
        model_frames[
            model_name
        ][
            "Final_Regime_Score"
        ]
    )

    export[
        f"Coverage_{clean_name}"
    ] = (
        model_frames[
            model_name
        ][
            "Model_Data_Coverage"
        ]
    )

for col in targets.columns:
    export[
        col
    ] = targets[
        col
    ]

csv_bytes = (
    export
    .to_csv()
    .encode(
        "utf-8"
    )
)

st.download_button(
    "⬇️ Research-Zeitreihe als CSV",
    data=csv_bytes,
    file_name=(
        "regime_backtest_"
        + selected_asset
        .replace(
            "/",
            "-"
        )
        .replace(
            " ",
            "_"
        )
        + ".csv"
    ),
    mime="text/csv",
)

st.caption(
    "Der Export enthält Scores, Coverage und Forward-Targets. "
    "Er ist ausdrücklich Research-Datenmaterial und keine "
    "Live-Trading-Freigabe."
)


# ============================================================
# 22. METHODOLOGICAL DISCLOSURE
# ============================================================

st.markdown("---")

with st.expander(
    "📚 Methodische Grenzen dieses Backtests",
    expanded=False,
):
    st.markdown(
        """
**Was dieser Test bereits verhindert**

- kein `bfill()` / keine Rückwärtsauffüllung;
- CFTC erst nach approximiertem Veröffentlichungstag;
- FRED First-Release/Vintage wird bevorzugt und für Daily-Serien in begrenzten Real-Time-Fenstern geladen;
- Coverage-aware Reweighting bei fehlenden Faktoren;
- identischer Rohdatensatz für alle drei Gewichtungsmodelle;
- Common-Sample-Vergleich verfügbar;
- Block-Bootstrap gegen naive Signifikanzinterpretation;
- Equal-Weight-Benchmark;
- Leave-one-pillar-out Ablation.

**Was weiterhin nicht perfekt point-in-time ist**

- CFTC-Feiertagsverschiebungen werden mit dem üblichen Dienstag→Freitag-Lag
  nur approximiert;
- CNN Fear & Greed besitzt keine echte Vintage-Datenbank;
- Multpl-PE wird mit einem konservativen +1-Tag-Lag angenähert;
- Yahoo-End-of-Day-Daten sind historische Marktbeobachtungen, aber keine
  institutionelle Tick-/Session-Rekonstruktion;
- bei Ausfall der ALFRED/First-Release-Abfrage verwendet FRED ausdrücklich
  gekennzeichnete Current-Vintage-Daten mit konservativen Lags;
- survivorship-/methodology changes externer Indizes können nicht vollständig
  rekonstruiert werden.

Deshalb ist das Ergebnis ein **Research-/Model-Selection-Test**, nicht die
Behauptung eines perfekten institutionellen Point-in-Time-Backtests.
"""
    )
