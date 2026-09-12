"""
Regime Backtest Lab v1.0.24
===========================

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
import re
import zipfile
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

st.title("🧪 Regime Backtest Lab v1.0.24 – Gold Volatility-State Robustness Audit")
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
# EUR/USD FROZEN B/E WALK-FORWARD – PRE-DEFINED RULES
# ============================================================
#
# B = Full Literature Prior, frozen Direction challenger.
# E = Literature Pillars only, frozen Risk-State challenger.
#
# No weights are estimated or changed in this validation.
# The rules below are fixed before observing v1.0.15 results.
#
EURUSD_WF_BASE_FIRST_TEST_YEAR = 2017
EURUSD_WF_MIN_YEAR_WIN_RATE = 0.60
EURUSD_WF_MIN_POOLED_IC = 0.00
EURUSD_WF_MIN_DIRECTION_20D = 0.50
EURUSD_WF_MIN_BOOTSTRAP_PROB = 0.75
EURUSD_WF_MIN_STRESS_AUC = 0.55
EURUSD_WF_MIN_E_YEAR_AUC_WIN_RATE = 0.50

# S&P 500 role diagnosis – pre-defined, no weight optimization.
SP500_RISK_MIN_AUC = 0.60
SP500_RISK_MIN_PHASE_AUC = 0.55
SP500_RISK_MIN_STRESS_GAP = 0.05
SP500_DIRECTION_MIN_ACCURACY = 0.50

SP500_DIAGNOSTIC_WINDOWS = [
    ("2015–16 Growth/Oil Scare", "2015-08-01", "2016-02-29"),
    ("Q4 2018 Selloff", "2018-10-01", "2018-12-31"),
    ("COVID Crash", "2020-02-19", "2020-04-30"),
    ("2022 Inflation/Rate Bear", "2022-01-03", "2022-10-31"),
]


# ============================================================
# NASDAQ 100 ROLE DIAGNOSIS – PRE-DEFINED, NO WEIGHT OPTIMIZATION
# ============================================================
NASDAQ_RISK_MIN_AUC = 0.60
NASDAQ_RISK_MIN_PHASE_AUC = 0.55
NASDAQ_RISK_MIN_STRESS_GAP = 0.05
NASDAQ_DIRECTION_MIN_ACCURACY = 0.50

NASDAQ_DIAGNOSTIC_WINDOWS = [
    ("2015–16 Growth/Oil Scare", "2015-08-01", "2016-02-29"),
    ("Q4 2018 Tech Selloff", "2018-10-01", "2018-12-31"),
    ("COVID Crash", "2020-02-19", "2020-04-30"),
    ("2022 Tech/Rate Bear", "2022-01-03", "2022-12-30"),
    ("2023 AI/Rebound Phase", "2023-01-03", "2023-07-31"),
]


# ============================================================
# GOLD ROLE / ORIENTATION DIAGNOSIS – NO WEIGHT OPTIMIZATION
# ============================================================

GOLD_RISK_MIN_AUC = 0.60
GOLD_RISK_MIN_PHASE_AUC = 0.55
GOLD_RISK_MIN_STRESS_GAP = 0.05
GOLD_DIRECTION_MIN_ACCURACY = 0.50

# Orientation audit only. It may diagnose a structural sign problem,
# but it never promotes an inverted score automatically.
GOLD_ORIENTATION_MIN_IC_IMPROVEMENT = 0.05
GOLD_ORIENTATION_MIN_POSITIVE_HORIZONS = 2

GOLD_DIAGNOSTIC_WINDOWS = [
    (
        "2013 Gold Selloff",
        "2013-04-01",
        "2013-06-30",
    ),
    (
        "2015 Bottom / Fed Lift-off",
        "2015-07-01",
        "2016-02-29",
    ),
    (
        "COVID / Safe-Haven Bull",
        "2020-02-19",
        "2020-08-07",
    ),
    (
        "2022 Real-Yield / USD Shock",
        "2022-03-01",
        "2022-11-03",
    ),
    (
        "2024–25 Breakout / Bull",
        "2024-02-01",
        "2025-12-31",
    ),
]


# ============================================================
# GOLD STRUCTURAL DRIVER AUDIT v1.0.20
# ============================================================
#
# These thresholds are diagnostic only. They are deliberately fixed before
# looking at the v1.0.20 output and never trigger an automatic production edit.
#
GOLD_STRUCTURAL_MIN_ABS_IC = 0.03
GOLD_STRUCTURAL_MIN_PERIOD_CONFIRMATIONS = 2
GOLD_STRUCTURAL_BOOTSTRAP_PROB = 0.90

# A level-vs-change transformation is considered worth a separate challenger
# only when the pre-defined 20D change signal improves IC20 materially and
# shows the expected sign in at least two of the three fixed sub-periods.
GOLD_CHANGE_MIN_IC_IMPROVEMENT = 0.05
GOLD_CHANGE_MIN_PERIOD_CONFIRMATIONS = 2

# A leave-one-driver-out improvement is considered structural only if removing
# one driver improves the complete-model IC20 by at least this amount.
GOLD_ABLATION_MIN_IC_IMPROVEMENT = 0.03

GOLD_FIXED_PERIODS = [
    ("2012–2016", 2012, 2016),
    ("2017–2020", 2017, 2020),
    ("2021–2025", 2021, 2025),
]


# ============================================================
# GOLD G1 STRUCTURAL CHALLENGER – FROZEN SPECIFICATION
# ============================================================
#
# G1 changes only the Gold macro pillar representation:
#
# - Fed Policy:       level -> favorable 20D change = -diff(20)
# - Real Yields:      level -> favorable 20D change = -diff(20)
# - USD Index:        level -> favorable 20D pct change = -pct_change(20)
# - Net Liquidity:    removed from G1 macro pillar
#
# Existing Gold macro raw weights 0.20 / 0.30 / 0.20 are renormalized
# internally. All pillar weights and all non-macro subweights remain Current.
#
GOLD_G1_MACRO_RAW_WEIGHTS = {
    "fed_policy": 0.20,
    "real_yields": 0.30,
    "usd_index": 0.20,
}

GOLD_G1_MIN_IC20 = 0.00
GOLD_G1_MIN_DELTA_IC20 = 0.10
GOLD_G1_MIN_DIRECTION20 = 0.50
GOLD_G1_MIN_DIRECTION_IMPROVEMENT = 0.05
GOLD_G1_MIN_POSITIVE_FIXED_PERIODS = 2
GOLD_G1_MIN_YEAR_WIN_RATE = 0.60
GOLD_G1_MIN_POSITIVE_YEAR_RATE = 0.50
GOLD_G1_MIN_BOOTSTRAP_PROB = 0.90

# Risk-state guardrails. G1 is a Direction challenger, but must not destroy
# the existing model's risk information.
GOLD_G1_MAX_STRESS_AUC_DEGRADATION = 0.03
GOLD_G1_MAX_PHASE_AUC_DEGRADATION = 0.03
GOLD_G1_MAX_VOL_REL_DEGRADATION = 0.05
GOLD_G1_MAX_MAE_REL_DEGRADATION = 0.05


# ============================================================
# GOLD DUAL-ROLE ARCHITECTURE – PRE-REGISTERED AUDIT
# ============================================================
#
# No fitted weights.
#
# Direction Context D1 uses the existing effective Current contributions:
#   G1 Macro pillar         = 0.35
#   Technical Trend pillar  = 0.15
#   OBV component           = 0.15 * 0.50 = 0.075
#
# Risk Context R1 uses the existing effective Current contributions:
#   CFTC                    = 0.25 * 0.80 = 0.20
#   Fear & Greed            = 0.25 * 0.20 = 0.05
#   GVZ health score        = 0.15 * 0.50 = 0.075
#   Credit health score     = 0.10 * 0.60 = 0.06
#   MOVE health score       = 0.10 * 0.40 = 0.04
#
# Each role is re-normalized only within itself and dynamically across
# actually available components. No target information enters these weights.
#

GOLD_D1_EFFECTIVE_WEIGHTS = {
    "g1_macro": 0.35,
    "technical_trend": 0.15,
    "obv_momentum": 0.075,
}

GOLD_R1_EFFECTIVE_WEIGHTS = {
    "cot_noncommercials": 0.20,
    "fear_greed": 0.05,
    "vix_score": 0.075,
    "credit_spreads": 0.06,
    "move_index": 0.04,
}

# Direction Context gates
GOLD_D1_MIN_IC20 = 0.05
GOLD_D1_MIN_NONOVERLAP_IC20 = 0.00
GOLD_D1_MIN_DIRECTION20 = 0.50
GOLD_D1_MIN_POSITIVE_FIXED_PERIODS = 2
GOLD_D1_MIN_POSITIVE_YEAR_RATE = 0.50
GOLD_D1_MIN_DELTA_VS_G1_IC20 = 0.03
GOLD_D1_MIN_BOOTSTRAP_PROB_VS_CURRENT = 0.90
GOLD_D1_MIN_BOOTSTRAP_PROB_VS_G1 = 0.75

# Risk Context gates
GOLD_R1_MIN_STRESS_AUC = 0.60
GOLD_R1_MIN_PHASE_AUC = 0.55
GOLD_R1_MIN_QUANTILE_STRESS_GAP = 0.05
GOLD_R1_MIN_POSITIVE_RISK_PERIODS = 2

# Architecture specialization is descriptive, not optimized.
GOLD_DUAL_ROLE_SPLIT_THRESHOLD = 50.0


# ============================================================
# GOLD R1 STRUCTURAL AUDIT – PRE-REGISTERED, D1 FROZEN
# ============================================================
#
# D1 from v1.0.22 is frozen and must not be changed by this audit.
#
# R1 questions:
# 1) Is any risk component structurally oriented the wrong way?
# 2) Does a 20D change in the existing health score work better than its level?
# 3) Does removing one component materially improve R1?
# 4) Is the weak R1 result specific to the -5% MAE stress target?
#
GOLD_RISK_SIGN_MIN_AUC_IMPROVEMENT = 0.05
GOLD_RISK_CHANGE_MIN_AUC_IMPROVEMENT = 0.05
GOLD_RISK_ABLATION_MIN_AUC_IMPROVEMENT = 0.03
GOLD_RISK_ABLATION_MAX_VOL_DAMAGE = 0.05
GOLD_RISK_ABLATION_MAX_MAE_DAMAGE = 0.05
GOLD_RISK_MIN_FIXED_PERIOD_CONFIRMATIONS = 2
GOLD_RISK_MIN_BOOTSTRAP_PROB = 0.90

# Stress-target sensitivity around the frozen Gold -5% MAE target.
# multiplier 0.75 = -3.75%, 1.00 = -5.00%, 1.25 = -6.25%.
GOLD_RISK_MAE_THRESHOLD_MULTIPLIERS = [
    0.75,
    1.00,
    1.25,
]

# Descriptive threshold for "high future realized volatility".
# This is a target audit only, not a production signal threshold.
GOLD_RISK_HIGH_VOL_QUANTILE = 0.80

# If R1 reaches this AUC on future-volatility stress while remaining weak on
# MAE stress, it is treated as evidence that R1 is more a volatility-state
# context than an adverse-excursion context.
GOLD_RISK_VOL_STATE_MIN_AUC = 0.60


# ============================================================
# GOLD VOLATILITY-STATE ROBUSTNESS AUDIT v1.0.24
# ============================================================
#
# D1 remains frozen.
# This audit does NOT create a production V1 model yet.
#
# PIT high-vol target:
# - for every observation date t, calculate a threshold from PAST realized
#   20D volatility only
# - trailing window = 756 observations (~3y)
# - min periods = 252
# - threshold quantile = 80th percentile
# - current/future observation is never included in the threshold window
#
GOLD_VOL_PIT_LOOKBACK = 756
GOLD_VOL_PIT_MIN_PERIODS = 252
GOLD_VOL_PIT_QUANTILE = 0.80

# Pre-registered V-state gates
GOLD_VSTATE_MIN_AUC = 0.60
GOLD_VSTATE_MIN_PHASE_AUC = 0.55
GOLD_VSTATE_MIN_PERIOD_AUC = 0.50
GOLD_VSTATE_MIN_POSITIVE_FIXED_PERIODS = 2
GOLD_VSTATE_MIN_HIGHVOL_GAP = 0.05
GOLD_VSTATE_MIN_VOL_SPEARMAN = 0.10

# MOVE structural comparison
GOLD_MOVE_MIN_AUC_IMPROVEMENT = 0.03
GOLD_MOVE_MIN_BOOTSTRAP_PROB = 0.90

# Fear & Greed is explicitly excluded from a structural volatility challenger
# because the historical series in this research pipeline is too short.
GOLD_VSTATE_EXCLUDE_FEAR_GREED = True

# S&P 500 Frozen A-vs-E Risk-State Walk-Forward – pre-registered rules.
SP500_WF_FIRST_TEST_YEAR = 2017
SP500_WF_MIN_CURRENT_AUC = 0.60
SP500_WF_MIN_CURRENT_PHASE_AUC = 0.55
SP500_WF_MIN_CURRENT_STRESS_GAP = 0.05
SP500_WF_MIN_E_AUC_DELTA = 0.01
SP500_WF_MIN_E_PHASE_AUC_DELTA = 0.01
SP500_WF_MIN_E_YEAR_WIN_RATE = 0.60
SP500_WF_MIN_E_MATERIAL_YEAR_RATE = 0.50
SP500_WF_MIN_E_BOOTSTRAP_PROB = 0.90


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



def build_gold_g1_challenger(
    raw,
    base_norm_df,
):
    """
    Build the single pre-registered Gold G1 challenger.

    PIT rules:
    - 20D changes use only present/past raw observations.
    - Each favorable-change series is normalized with the existing rolling
      PIT percentile engine.
    - No bfill.
    - No weights are fitted from the outcome data.
    """
    asset_name = "Gold (XAU/USD)"

    current_cfg = current_model_config(
        asset_name
    )

    g1_cfg = deepcopy(
        current_cfg
    )

    # Freeze the macro composition to the three pre-registered drivers.
    g1_cfg[
        "sub_weights"
    ][
        "Makroökonomie"
    ] = normalize_weight_dict(
        GOLD_G1_MACRO_RAW_WEIGHTS
    )

    g1_norm = base_norm_df.copy()

    transformed = pd.DataFrame(
        index=raw.index
    )

    transform_specs = {
        "fed_policy": {
            "kind": "diff",
            "sign": -1.0,
        },
        "real_yields": {
            "kind": "diff",
            "sign": -1.0,
        },
        "usd_index": {
            "kind": "pct",
            "sign": -1.0,
        },
    }

    for factor, spec in transform_specs.items():
        if factor not in raw.columns:
            g1_norm[
                factor
            ] = np.nan
            transformed[
                f"{factor}_raw_change20"
            ] = np.nan
            transformed[
                f"{factor}_favorable_change20"
            ] = np.nan
            continue

        raw_series = pd.to_numeric(
            raw[
                factor
            ],
            errors="coerce",
        )

        if spec[
            "kind"
        ] == "pct":
            change = raw_series.pct_change(
                20,
                fill_method=None,
            )
        else:
            change = raw_series.diff(
                20
            )

        favorable_change = (
            float(
                spec[
                    "sign"
                ]
            )
            * change
        )

        transformed[
            f"{factor}_raw_change20"
        ] = change

        transformed[
            f"{factor}_favorable_change20"
        ] = favorable_change

        g1_norm[
            factor
        ] = pit_normalize_to_percentile(
            favorable_change,
            lookback=LOOKBACK_CONFIG.get(
                factor,
                252,
            ),
            invert=False,
        )

    # net_liquidity remains in the research dataframe for diagnostics but has
    # exactly zero weight in the G1 macro pillar by construction.
    g1_frame = model_score_frame(
        g1_norm,
        g1_cfg,
    )

    return (
        g1_norm,
        g1_cfg,
        g1_frame,
        transformed,
    )



def build_coverage_weighted_context(
    component_df,
    weights,
):
    """
    Coverage-aware weighted context score.

    The weights are pre-defined outside this function. Missing components are
    dynamically re-normalized row by row; coverage reports how much of the
    intended weight is actually present.
    """
    cols = [
        col
        for col in weights
        if col in component_df.columns
    ]

    if not cols:
        empty = pd.Series(
            np.nan,
            index=component_df.index,
            dtype=float,
        )
        return empty, empty.copy()

    total_weight = float(
        sum(
            float(
                weights[col]
            )
            for col in cols
        )
    )

    if total_weight <= 0:
        empty = pd.Series(
            np.nan,
            index=component_df.index,
            dtype=float,
        )
        return empty, empty.copy()

    weighted_sum = pd.Series(
        0.0,
        index=component_df.index,
        dtype=float,
    )

    available_weight = pd.Series(
        0.0,
        index=component_df.index,
        dtype=float,
    )

    for col in cols:
        series = pd.to_numeric(
            component_df[col],
            errors="coerce",
        )

        weight = float(
            weights[col]
        )

        valid = series.notna()

        weighted_sum = weighted_sum.add(
            series.fillna(
                0.0
            )
            * weight,
            fill_value=0.0,
        )

        available_weight = available_weight.add(
            valid.astype(
                float
            )
            * weight,
            fill_value=0.0,
        )

    score = (
        weighted_sum
        / available_weight.replace(
            0.0,
            np.nan,
        )
    )

    coverage = (
        available_weight
        / total_weight
        * 100.0
    )

    return score, coverage


def dual_role_risk_metrics(
    score,
    coverage,
    targets,
    mask=None,
):
    """
    Risk-state metrics for a health-oriented score:
    higher score = healthier / less future stress.
    """
    if mask is None:
        mask = pd.Series(
            True,
            index=score.index,
        )

    s = pd.to_numeric(
        score.where(
            mask
        ),
        errors="coerce",
    )

    c = pd.to_numeric(
        coverage.where(
            mask
        ),
        errors="coerce",
    )

    event = pd.to_numeric(
        targets[
            "Stress_Event_20D"
        ],
        errors="coerce",
    )

    valid = (
        s.notna()
        & event.notna()
    )

    s_valid = s.where(
        valid
    )

    auc = binary_auc(
        s_valid,
        event,
        higher_predictor_means_event=False,
    )

    phase_auc, phase_count = phase_median_stress_auc(
        s_valid,
        event,
        horizon=20,
    )

    vol_relation = safe_spearman(
        s_valid,
        -targets[
            "Fwd_Realized_Vol_20D"
        ],
    )

    mae_relation = safe_spearman(
        s_valid,
        targets[
            "Fwd_MAE_20D"
        ],
    )

    frame = pd.DataFrame(
        {
            "score": s,
            "event": event,
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

        low_rate = (
            float(
                low[
                    "event"
                ].mean()
            )
            if not low.empty
            else np.nan
        )

        high_rate = (
            float(
                high[
                    "event"
                ].mean()
            )
            if not high.empty
            else np.nan
        )

        quantile_gap = (
            low_rate
            - high_rate
            if (
                np.isfinite(
                    low_rate
                )
                and np.isfinite(
                    high_rate
                )
            )
            else np.nan
        )
    else:
        q20 = np.nan
        q80 = np.nan
        low_rate = np.nan
        high_rate = np.nan
        quantile_gap = np.nan

    fixed_low = frame[
        frame[
            "score"
        ]
        <= 40
    ]

    fixed_high = frame[
        frame[
            "score"
        ]
        >= 60
    ]

    fixed_low_rate = (
        float(
            fixed_low[
                "event"
            ].mean()
        )
        if not fixed_low.empty
        else np.nan
    )

    fixed_high_rate = (
        float(
            fixed_high[
                "event"
            ].mean()
        )
        if not fixed_high.empty
        else np.nan
    )

    fixed_gap = (
        fixed_low_rate
        - fixed_high_rate
        if (
            np.isfinite(
                fixed_low_rate
            )
            and np.isfinite(
                fixed_high_rate
            )
        )
        else np.nan
    )

    return {
        "N": int(
            valid.sum()
        ),
        "Stress AUC 20D": auc,
        "Phase-Median Stress AUC": phase_auc,
        "AUC-Phasen": phase_count,
        "Q20 Stressrate": low_rate,
        "Q80 Stressrate": high_rate,
        "Q20−Q80 Stress-Gap": quantile_gap,
        "Score≤40 Stressrate": fixed_low_rate,
        "Score≥60 Stressrate": fixed_high_rate,
        "Fixed Stress-Gap": fixed_gap,
        "Score vs niedrigere FwdVol": vol_relation,
        "Score vs bessere FwdMAE": mae_relation,
        "Ø Coverage": float(
            c.mean()
        ),
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



def walk_forward_year_metrics(
    current_score,
    model_d_score,
    current_coverage,
    model_d_coverage,
    targets,
    common_mask,
    first_test_year,
    current_year,
):
    """
    Frozen-parameter expanding-history walk-forward.

    Important methodological point:
    Model D is NOT trained or re-optimized inside the loop. Its weights
    were frozen before this validation step. The historical data before
    each test year merely provide the rolling/z-score lookback context
    that was already calculated point-in-time.

    Each calendar year is treated as a separate chronological test window.
    The current partial calendar year is reported but excluded from the
    formal pass/fail gate.
    """
    rows = []

    available_years = sorted(
        set(
            int(year)
            for year in targets.index.year
            if int(year) >= int(first_test_year)
        )
    )

    for year in available_years:
        year_mask = (
            common_mask
            & (
                targets.index.year
                == int(year)
            )
        )

        if int(
            year_mask.sum()
        ) < 40:
            continue

        current_year_score = (
            current_score.where(
                year_mask
            )
        )

        d_year_score = (
            model_d_score.where(
                year_mask
            )
        )

        row = {
            "Testjahr": int(year),
            "Vollständiges Jahr": (
                int(year)
                < int(current_year)
            ),
            "N Common": int(
                year_mask.sum()
            ),
            "Ø Coverage Current": float(
                pd.to_numeric(
                    current_coverage.where(
                        year_mask
                    ),
                    errors="coerce"
                ).mean()
            ),
            "Ø Coverage Model D": float(
                pd.to_numeric(
                    model_d_coverage.where(
                        year_mask
                    ),
                    errors="coerce"
                ).mean()
            ),
        }

        for horizon in FORWARD_HORIZONS:
            target_h = targets[
                f"Fwd_Return_{horizon}D"
            ]

            row[
                f"Current IC {horizon}D"
            ] = safe_spearman(
                current_year_score,
                target_h
            )

            row[
                f"Model D IC {horizon}D"
            ] = safe_spearman(
                d_year_score,
                target_h
            )

        row[
            "Δ IC20 D−Current"
        ] = (
            row[
                "Model D IC 20D"
            ]
            -
            row[
                "Current IC 20D"
            ]
            if (
                np.isfinite(
                    row[
                        "Model D IC 20D"
                    ]
                )
                and np.isfinite(
                    row[
                        "Current IC 20D"
                    ]
                )
            )
            else np.nan
        )

        (
            current_direction,
            current_signals,
        ) = directional_accuracy(
            current_year_score,
            targets[
                "Fwd_Return_20D"
            ],
        )

        (
            d_direction,
            d_signals,
        ) = directional_accuracy(
            d_year_score,
            targets[
                "Fwd_Return_20D"
            ],
        )

        row[
            "Current Direction 20D"
        ] = current_direction

        row[
            "Model D Direction 20D"
        ] = d_direction

        row[
            "Current Signals"
        ] = current_signals

        row[
            "Model D Signals"
        ] = d_signals

        row[
            "Current Stress AUC"
        ] = binary_auc(
            current_year_score,
            targets[
                "Stress_Event_20D"
            ],
            higher_predictor_means_event=False,
        )

        row[
            "Model D Stress AUC"
        ] = binary_auc(
            d_year_score,
            targets[
                "Stress_Event_20D"
            ],
            higher_predictor_means_event=False,
        )

        row[
            "D besser IC20"
        ] = bool(
            np.isfinite(
                row[
                    "Δ IC20 D−Current"
                ]
            )
            and row[
                "Δ IC20 D−Current"
            ] > 0
        )

        row[
            "D IC20 positiv"
        ] = bool(
            np.isfinite(
                row[
                    "Model D IC 20D"
                ]
            )
            and row[
                "Model D IC 20D"
            ] > 0
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def pooled_walk_forward_metrics(
    score,
    targets,
    mask,
):
    """Pooled metrics for a fixed chronological walk-forward test period."""
    score = (
        pd.to_numeric(
            score,
            errors="coerce"
        )
        .where(
            mask
        )
    )

    result = {}

    for horizon in FORWARD_HORIZONS:
        result[
            f"IC {horizon}D"
        ] = safe_spearman(
            score,
            targets[
                f"Fwd_Return_{horizon}D"
            ]
        )

        (
            direction,
            signal_count,
        ) = directional_accuracy(
            score,
            targets[
                f"Fwd_Return_{horizon}D"
            ],
        )

        result[
            f"Direction {horizon}D"
        ] = direction

        result[
            f"Signals {horizon}D"
        ] = signal_count

    result[
        "Stress AUC 20D"
    ] = binary_auc(
        score,
        targets[
            "Stress_Event_20D"
        ],
        higher_predictor_means_event=False,
    )

    result[
        "Score vs niedrigere FwdVol"
    ] = safe_spearman(
        score,
        -targets[
            "Fwd_Realized_Vol_20D"
        ]
    )

    result[
        "Score vs bessere FwdMAE"
    ] = safe_spearman(
        score,
        targets[
            "Fwd_MAE_20D"
        ]
    )

    return result


def build_event_classification(
    index,
    windows,
):
    """
    Create a fixed exogenous event classification.

    Multiple overlapping windows are supported. No event label is derived
    from prices, scores, volatility, or future returns.
    """
    idx = pd.DatetimeIndex(
        index
    )

    any_event = pd.Series(
        False,
        index=idx,
        dtype=bool,
    )

    labels = pd.Series(
        "Normal",
        index=idx,
        dtype=object,
    )

    categories = pd.Series(
        "Normal",
        index=idx,
        dtype=object,
    )

    for event in windows:
        start = pd.Timestamp(
            event["start"]
        )

        end = pd.Timestamp(
            event["end"]
        )

        mask = (
            (idx >= start)
            & (idx <= end)
        )

        if not mask.any():
            continue

        any_event.loc[
            mask
        ] = True

        for dt in idx[
            mask
        ]:
            if labels.loc[
                dt
            ] == "Normal":
                labels.loc[
                    dt
                ] = event[
                    "name"
                ]

                categories.loc[
                    dt
                ] = event[
                    "category"
                ]

            else:
                labels.loc[
                    dt
                ] = (
                    labels.loc[
                        dt
                    ]
                    + " | "
                    + event[
                        "name"
                    ]
                )

                categories.loc[
                    dt
                ] = (
                    categories.loc[
                        dt
                    ]
                    + " | "
                    + event[
                        "category"
                    ]
                )

    return pd.DataFrame(
        {
            "Event_Any": any_event,
            "Event_Label": labels,
            "Event_Category": categories,
        },
        index=idx,
    )


def event_subset_metrics(
    label,
    mask,
    current_score,
    model_d_score,
    current_coverage,
    model_d_coverage,
    targets,
):
    """Metrics for one externally defined event/normal subset."""
    mask = pd.Series(
        mask,
        index=targets.index,
        dtype=bool,
    )

    n = int(
        mask.sum()
    )

    current_s = (
        pd.to_numeric(
            current_score,
            errors="coerce"
        )
        .where(
            mask
        )
    )

    d_s = (
        pd.to_numeric(
            model_d_score,
            errors="coerce"
        )
        .where(
            mask
        )
    )

    target20 = (
        targets[
            "Fwd_Return_20D"
        ]
    )

    current_ic20 = safe_spearman(
        current_s,
        target20,
    )

    d_ic20 = safe_spearman(
        d_s,
        target20,
    )

    (
        current_direction,
        current_signals,
    ) = directional_accuracy(
        current_s,
        target20,
    )

    (
        d_direction,
        d_signals,
    ) = directional_accuracy(
        d_s,
        target20,
    )

    current_auc = binary_auc(
        current_s,
        targets[
            "Stress_Event_20D"
        ],
        higher_predictor_means_event=False,
    )

    d_auc = binary_auc(
        d_s,
        targets[
            "Stress_Event_20D"
        ],
        higher_predictor_means_event=False,
    )

    return {
        "Teilmenge": label,
        "N": n,
        "Current IC20": current_ic20,
        "Model D IC20": d_ic20,
        "Δ IC20 D−Current": (
            d_ic20 - current_ic20
            if (
                np.isfinite(
                    d_ic20
                )
                and np.isfinite(
                    current_ic20
                )
            )
            else np.nan
        ),
        "Current Direction 20D": current_direction,
        "Model D Direction 20D": d_direction,
        "Current Signals": current_signals,
        "Model D Signals": d_signals,
        "Current Stress AUC": current_auc,
        "Model D Stress AUC": d_auc,
        "Ø Fwd Return 20D": float(
            pd.to_numeric(
                targets[
                    "Fwd_Return_20D"
                ].where(
                    mask
                ),
                errors="coerce"
            ).mean()
        ),
        "Ø Fwd MAE 20D": float(
            pd.to_numeric(
                targets[
                    "Fwd_MAE_20D"
                ].where(
                    mask
                ),
                errors="coerce"
            ).mean()
        ),
        "Ø Fwd Vol 20D": float(
            pd.to_numeric(
                targets[
                    "Fwd_Realized_Vol_20D"
                ].where(
                    mask
                ),
                errors="coerce"
            ).mean()
        ),
        "Ø Coverage Current": float(
            pd.to_numeric(
                current_coverage.where(
                    mask
                ),
                errors="coerce"
            ).mean()
        ),
        "Ø Coverage Model D": float(
            pd.to_numeric(
                model_d_coverage.where(
                    mask
                ),
                errors="coerce"
            ).mean()
        ),
    }


def leave_one_event_out_metrics(
    base_mask,
    event_classification,
    windows,
    current_score,
    model_d_score,
    targets,
):
    """
    Sensitivity table: remove each event window one at a time.

    This is deliberately diagnostic. It was motivated by the previously
    observed 2020 sensitivity and is therefore NOT presented as an
    independent pre-registered model-selection test.
    """
    rows = []

    base_mask = pd.Series(
        base_mask,
        index=targets.index,
        dtype=bool,
    )

    def _row(
        label,
        mask,
    ):
        current_ic20 = safe_spearman(
            current_score.where(
                mask
            ),
            targets[
                "Fwd_Return_20D"
            ],
        )

        d_ic20 = safe_spearman(
            model_d_score.where(
                mask
            ),
            targets[
                "Fwd_Return_20D"
            ],
        )

        return {
            "Sensitivität": label,
            "N": int(
                mask.sum()
            ),
            "Current IC20": current_ic20,
            "Model D IC20": d_ic20,
            "Δ IC20 D−Current": (
                d_ic20 - current_ic20
                if (
                    np.isfinite(
                        current_ic20
                    )
                    and np.isfinite(
                        d_ic20
                    )
                )
                else np.nan
            ),
            "Model D IC20 positiv": bool(
                np.isfinite(
                    d_ic20
                )
                and d_ic20 > 0
            ),
            "Model D besser": bool(
                np.isfinite(
                    current_ic20
                )
                and np.isfinite(
                    d_ic20
                )
                and d_ic20 > current_ic20
            ),
        }

    rows.append(
        _row(
            "Alle vollständigen WF-Jahre",
            base_mask,
        )
    )

    for event in windows:
        if not event.get(
            "formal_walk_forward",
            False
        ):
            continue

        event_mask = (
            event_classification[
                "Event_Label"
            ]
            .astype(
                str
            )
            .str.contains(
                event[
                    "name"
                ],
                regex=False,
            )
        )

        rows.append(
            _row(
                f"Ohne {event['name']}",
                (
                    base_mask
                    & ~event_mask
                ),
            )
        )

    formal_event_mask = (
        base_mask
        & event_classification[
            "Event_Any"
        ]
    )

    rows.append(
        _row(
            "Ohne alle markierten Eventfenster",
            (
                base_mask
                & ~formal_event_mask
            ),
        )
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



def phase_median_stress_auc(score, event, horizon=20):
    """Median stress AUC across phase-shifted non-overlapping samples."""
    frame = pd.DataFrame({
        "score": pd.to_numeric(score, errors="coerce"),
        "event": pd.to_numeric(event, errors="coerce"),
    }).dropna()

    if len(frame) < int(horizon) * 3:
        return np.nan, 0

    values = []
    for phase in range(int(horizon)):
        sample = frame.iloc[phase::int(horizon)]
        auc = binary_auc(
            sample["score"],
            sample["event"],
            higher_predictor_means_event=False,
        )
        if np.isfinite(auc):
            values.append(auc)

    if not values:
        return np.nan, 0

    return float(np.median(values)), int(len(values))


def phase_median_nonoverlap_ic(score, target, horizon):
    """
    Median Spearman IC across all phase-shifted, non-overlapping
    h-day samples. Diagnostic only; scores/weights are untouched.
    """
    frame = pd.DataFrame({
        "score": pd.to_numeric(score, errors="coerce"),
        "target": pd.to_numeric(target, errors="coerce"),
    }).dropna()

    if len(frame) < max(40, int(horizon) * 2):
        return np.nan, 0

    values = []

    for phase in range(int(horizon)):
        sample = frame.iloc[phase::int(horizon)]

        ic = safe_spearman(
            sample["score"],
            sample["target"],
        )

        if np.isfinite(ic):
            values.append(ic)

    if not values:
        return np.nan, 0

    return float(np.median(values)), int(len(values))


def sp500_risk_state_metrics(score, mask, targets):
    score = pd.to_numeric(score.where(mask), errors="coerce")
    auc = binary_auc(
        score,
        targets["Stress_Event_20D"],
        higher_predictor_means_event=False,
    )
    phase_auc, phases = phase_median_stress_auc(
        score,
        targets["Stress_Event_20D"],
        horizon=20,
    )
    vol_rel = safe_spearman(
        score,
        -targets["Fwd_Realized_Vol_20D"],
    )
    mae_rel = safe_spearman(
        score,
        targets["Fwd_MAE_20D"],
    )
    zones = pd.DataFrame({
        "score": score,
        "stress": pd.to_numeric(
            targets["Stress_Event_20D"], errors="coerce"
        ),
    }).dropna()
    low = zones[zones["score"] <= 40]
    high = zones[zones["score"] >= 60]
    low_rate = float(low["stress"].mean()) if not low.empty else np.nan
    high_rate = float(high["stress"].mean()) if not high.empty else np.nan
    gap = (
        low_rate - high_rate
        if np.isfinite(low_rate) and np.isfinite(high_rate)
        else np.nan
    )
    return {
        "N": int((score.notna() & targets["Stress_Event_20D"].notna()).sum()),
        "Stress AUC 20D": auc,
        "Phase-Median Stress AUC": phase_auc,
        "AUC-Phasen": phases,
        "Stressrate Score≤40": low_rate,
        "Stressrate Score≥60": high_rate,
        "Stressrate-Gap": gap,
        "Score vs niedrigere FwdVol": vol_rel,
        "Score vs bessere FwdMAE": mae_rel,
    }


def block_bootstrap_auc_difference(
    score_a,
    score_e,
    event,
    block_length=20,
    n_boot=500,
    seed=20260911,
):
    frame = pd.DataFrame({
        "a": pd.to_numeric(score_a, errors="coerce"),
        "e": pd.to_numeric(score_e, errors="coerce"),
        "event": pd.to_numeric(event, errors="coerce"),
    }).dropna()
    n = len(frame)
    auc_a = binary_auc(frame["a"], frame["event"], higher_predictor_means_event=False)
    auc_e = binary_auc(frame["e"], frame["event"], higher_predictor_means_event=False)
    observed = auc_e - auc_a if np.isfinite(auc_a) and np.isfinite(auc_e) else np.nan
    block_length = max(1, int(block_length))
    if n < max(80, block_length * 4):
        return {"observed": observed, "prob_positive": np.nan, "lower": np.nan, "upper": np.nan, "n": n}
    rng = np.random.default_rng(int(seed))
    max_start = n - block_length
    values = []
    for _ in range(int(n_boot)):
        idx = []
        while len(idx) < n:
            s = int(rng.integers(0, max_start + 1))
            idx.extend(range(s, s + block_length))
        sample = frame.iloc[np.asarray(idx[:n], dtype=int)]
        a = binary_auc(sample["a"], sample["event"], higher_predictor_means_event=False)
        e = binary_auc(sample["e"], sample["event"], higher_predictor_means_event=False)
        if np.isfinite(a) and np.isfinite(e):
            values.append(e - a)
    if not values:
        return {"observed": observed, "prob_positive": np.nan, "lower": np.nan, "upper": np.nan, "n": n}
    arr = np.asarray(values, dtype=float)
    return {
        "observed": observed,
        "prob_positive": float(np.mean(arr > 0)),
        "lower": float(np.quantile(arr, 0.025)),
        "upper": float(np.quantile(arr, 0.975)),
        "n": n,
    }


def confirmation_period_metrics(label, mask, scores, targets):
    """Period metrics for frozen EUR/USD candidates A/B/E."""
    rows = []

    for model_name, score in scores.items():
        score_masked = score.where(mask)

        direction_20d, signals_20d = directional_accuracy(
            score_masked,
            targets["Fwd_Return_20D"],
        )

        rows.append({
            "Periode": label,
            "Modell": model_name,
            "N": int(
                (
                    score_masked.notna()
                    & targets["Fwd_Return_20D"].notna()
                ).sum()
            ),
            "IC 20D": safe_spearman(
                score_masked,
                targets["Fwd_Return_20D"],
            ),
            "IC 60D": safe_spearman(
                score_masked,
                targets["Fwd_Return_60D"],
            ),
            "Direction 20D": direction_20d,
            "Signals 20D": signals_20d,
            "Stress AUC 20D": binary_auc(
                score_masked,
                targets["Stress_Event_20D"],
                higher_predictor_means_event=False,
            ),
            "Score vs bessere FwdMAE": safe_spearman(
                score_masked,
                targets["Fwd_MAE_20D"],
            ),
            "Score vs niedrigere FwdVol": safe_spearman(
                score_masked,
                -targets["Fwd_Realized_Vol_20D"],
            ),
        })

    return rows



def eurusd_be_walk_forward_year_metrics(
    current_score,
    b_score,
    e_score,
    current_coverage,
    b_coverage,
    e_coverage,
    targets,
    common_mask,
    first_test_year,
    current_year,
):
    """
    Calendar-year walk-forward diagnostics for frozen EUR/USD A/B/E.

    The model parameters are not trained inside the loop.
    All rolling factor transforms were already calculated point-in-time.
    The loop only evaluates consecutive chronological test windows.
    """
    rows = []

    available_years = sorted(
        set(
            int(year)
            for year in targets.index.year
            if int(year) >= int(first_test_year)
        )
    )

    for year in available_years:
        year_mask = (
            common_mask
            & (
                targets.index.year
                == int(year)
            )
        )

        if int(
            year_mask.sum()
        ) < 40:
            continue

        current_y = current_score.where(
            year_mask
        )
        b_y = b_score.where(
            year_mask
        )
        e_y = e_score.where(
            year_mask
        )

        row = {
            "Testjahr": int(year),
            "Vollständiges Jahr": (
                int(year)
                < int(current_year)
            ),
            "N Common": int(
                year_mask.sum()
            ),
            "Ø Coverage Current": float(
                pd.to_numeric(
                    current_coverage.where(
                        year_mask
                    ),
                    errors="coerce",
                ).mean()
            ),
            "Ø Coverage B": float(
                pd.to_numeric(
                    b_coverage.where(
                        year_mask
                    ),
                    errors="coerce",
                ).mean()
            ),
            "Ø Coverage E": float(
                pd.to_numeric(
                    e_coverage.where(
                        year_mask
                    ),
                    errors="coerce",
                ).mean()
            ),
        }

        for horizon in FORWARD_HORIZONS:
            target_h = targets[
                f"Fwd_Return_{horizon}D"
            ]

            row[
                f"Current IC {horizon}D"
            ] = safe_spearman(
                current_y,
                target_h,
            )

            row[
                f"B IC {horizon}D"
            ] = safe_spearman(
                b_y,
                target_h,
            )

            row[
                f"E IC {horizon}D"
            ] = safe_spearman(
                e_y,
                target_h,
            )

        row[
            "Δ IC20 B−Current"
        ] = (
            row["B IC 20D"]
            - row["Current IC 20D"]
            if (
                np.isfinite(
                    row["B IC 20D"]
                )
                and np.isfinite(
                    row["Current IC 20D"]
                )
            )
            else np.nan
        )

        row[
            "Δ IC60 B−Current"
        ] = (
            row["B IC 60D"]
            - row["Current IC 60D"]
            if (
                np.isfinite(
                    row["B IC 60D"]
                )
                and np.isfinite(
                    row["Current IC 60D"]
                )
            )
            else np.nan
        )

        (
            current_direction,
            current_signals,
        ) = directional_accuracy(
            current_y,
            targets[
                "Fwd_Return_20D"
            ],
        )

        (
            b_direction,
            b_signals,
        ) = directional_accuracy(
            b_y,
            targets[
                "Fwd_Return_20D"
            ],
        )

        row[
            "Current Direction 20D"
        ] = current_direction

        row[
            "B Direction 20D"
        ] = b_direction

        row[
            "Current Signals"
        ] = current_signals

        row[
            "B Signals"
        ] = b_signals

        current_auc = binary_auc(
            current_y,
            targets[
                "Stress_Event_20D"
            ],
            higher_predictor_means_event=False,
        )

        b_auc = binary_auc(
            b_y,
            targets[
                "Stress_Event_20D"
            ],
            higher_predictor_means_event=False,
        )

        e_auc = binary_auc(
            e_y,
            targets[
                "Stress_Event_20D"
            ],
            higher_predictor_means_event=False,
        )

        row[
            "Current Stress AUC"
        ] = current_auc

        row[
            "B Stress AUC"
        ] = b_auc

        row[
            "E Stress AUC"
        ] = e_auc

        row[
            "B besser IC20"
        ] = bool(
            np.isfinite(
                row[
                    "Δ IC20 B−Current"
                ]
            )
            and row[
                "Δ IC20 B−Current"
            ] > 0
        )

        row[
            "B besser IC60"
        ] = bool(
            np.isfinite(
                row[
                    "Δ IC60 B−Current"
                ]
            )
            and row[
                "Δ IC60 B−Current"
            ] > 0
        )

        row[
            "B IC20 positiv"
        ] = bool(
            np.isfinite(
                row[
                    "B IC 20D"
                ]
            )
            and row[
                "B IC 20D"
            ] > 0
        )

        row[
            "B IC60 positiv"
        ] = bool(
            np.isfinite(
                row[
                    "B IC 60D"
                ]
            )
            and row[
                "B IC 60D"
            ] > 0
        )

        row[
            "E AUC besser Current"
        ] = bool(
            np.isfinite(
                e_auc
            )
            and np.isfinite(
                current_auc
            )
            and e_auc > current_auc
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
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
            index=2,
            help=(
                "Für die nächste Validierungsstufe ist Gold (XAU/USD) voreingestellt. "
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

if selected_asset == "S&P 500":
    with st.expander(
        "🇺🇸 S&P-500-Quellencheck",
        expanded=True
    ):
        st.markdown(
            """
Für die S&P-500-Diagnose bleiben die Referenzquellen unverändert:

- **Preis/Technik:** `^GSPC`
- **Aktienvolatilität:** `^VIX`
- **Volatility-of-Volatility:** `^VVIX`
- **USD-Index:** `DX-Y.NYB`
- **Bond-Volatilität:** `^MOVE`
- **Credit Proxy:** `LQD / HYG`
- **CFTC E-mini S&P 500 Non-Commercials:** Marktcode `13874A`
- **Sentiment:** CNN Fear & Greed
- **Makro/FRED:** `WALCL`, `WTREGEN`, `RRPONTSYD`, `FEDFUNDS`, `DFII10`
- **Bewertung:** historisches S&P-500-KGV über Multpl mit bestehendem Fallback

Yahoo wird zunächst als Batch geladen und fehlende Reihen werden einzeln
mit Retry nachgeladen. FRED/ALFRED, CFTC-Publikationslag und die bestehende
PIT-Logik bleiben unverändert.

**Ziel von v1.0.16:** zuerst sauber bestimmen, ob der bestehende S&P-Score
empirisch eher Direction oder Risk-/Stress-State misst. In diesem Lauf
werden ausdrücklich keine neuen Gewichte optimiert.
"""
        )

elif selected_asset == "Nasdaq 100":
    with st.expander(
        "💻 Nasdaq-100-Quellencheck",
        expanded=True
    ):
        st.markdown(
            """
Für die Nasdaq-100-Diagnose werden die bestehenden Referenzquellen unverändert
verwendet:

- **Preis/Technik:** `^NDX`
- **Nasdaq-Volatilität:** `^VXN`
- **Volatility-of-Volatility:** `^VVIX`
- **USD-Index:** `DX-Y.NYB`
- **Bond-Volatilität:** `^MOVE`
- **Credit Proxy:** `LQD / HYG`
- **CFTC Nasdaq-100 Non-Commercials:** Marktcode `209742`
- **Sentiment:** CNN Fear & Greed
- **Makro/FRED:** `WALCL`, `WTREGEN`, `RRPONTSYD`, `FEDFUNDS`, `DFII10`

Yahoo wird zunächst als Batch geladen; fehlende Reihen werden wie bisher
einzeln mit Retry nachgeladen. FRED/ALFRED, CFTC-Publikationslag und die
bestehende PIT-Logik bleiben unverändert.

**Ziel von v1.0.18:** zuerst bestimmen, ob der Nasdaq-Score empirisch eher
Forward-Direction oder Risk-/Stress-State misst. Es werden in diesem Lauf
ausdrücklich keine neuen Gewichte optimiert.
"""
        )
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


elif selected_asset == "EUR/USD":
    with st.expander(
        "💶 EUR/USD-Quellencheck",
        expanded=True
    ):
        st.markdown(
            """
Für den EUR/USD-Test werden die bestehenden Modellquellen unverändert
verwendet:

- **Spot-/Preisfeed & Technik:** `EURUSD=X`
- **EUR/USD-Volatilität:** `^EVZ`
- **USD-Index:** `DX-Y.NYB`
- **Bond-Volatilität / Frühwarnung:** `^MOVE`
- **Credit Proxy / Frühwarnung:** `LQD / HYG`
- **CFTC Euro FX Non-Commercials:** Marktcode `099741`
- **Sentiment-Zusatz:** CNN Fear & Greed
- **Makro/FRED:** `WALCL`, `WTREGEN`, `RRPONTSYD`, `FEDFUNDS`, `DFII10`

Yahoo wird zunächst als Batch geladen. Fehlende Reihen werden anschließend
**einzeln mit Retry** nachgeladen. `^VVIX` wird für EUR/USD nicht abgefragt,
weil er weder im Current- noch im Literature-Prior-EUR/USD-Modell aktiv ist.

Der `USD Index` wird im EUR/USD-Modell **invertiert** interpretiert:
Dollar-Stärke belastet den EUR/USD-Score, Dollar-Schwäche unterstützt ihn.

**Wichtige Modellgrenze dieses Baseline-Tests:** Die bestehende Architektur
enthält US-Fed/Real-Yield-/Liquiditätsfaktoren, aber noch **keine explizite
EZB-vs.-Fed-Zinsdifferenz bzw. Euro-vs.-US-Realrenditedifferenz**. Das wird
in v1.0.13 bewusst nicht nachträglich ergänzt. Ein schwaches Ergebnis wäre
deshalb nicht automatisch ein Beweis gegen Regime-Modelle bei FX, sondern
könnte auf fehlende relative Makrofaktoren hindeuten.
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

if selected_asset == "Nasdaq 100":
    st.info(
        "💻 **Nasdaq-100-Diagnose v1.0.18:** A/Current, B/Literature, "
        "C/Equal sowie D/E bleiben unverändert. Getrennt geprüft werden "
        "Forward-Direction, Stress-AUC, MAE/Volatilitätsbezug, "
        "Non-Overlap-Stabilität und vorab definierte Stressfenster. "
        "Es findet keine Gewichtsoptimierung statt."
    )

elif selected_asset == "S&P 500":
    st.info(
        "🇺🇸 **S&P-500-Diagnose v1.0.16:** A/Current, B/Literature, "
        "C/Equal sowie D/E bleiben unverändert. Getrennt geprüft werden "
        "Forward-Direction, Stress-AUC, MAE/Volatilitätsbezug, "
        "Non-Overlap-Stabilität und vorab definierte Stressfenster. "
        "Es findet keine Gewichtsoptimierung statt."
    )

elif selected_asset == "Gold (XAU/USD)":
    st.info(
        "🟨 **Gold Volatility-State Audit v1.0.24:** D1 bleibt "
        "vollständig eingefroren. R1 wird nicht weiter auf MAE-Stress getrimmt. "
        "Stattdessen wird geprüft, ob ein separater **Volatility-State V1** "
        "robust ist: PIT-sicheres High-Vol-Ziel, feste Teilperioden inklusive "
        "2017–2020 sowie MOVE Current vs. invertiert vs. entfernt. "
        "Fear & Greed bleibt wegen der kurzen Historie aus einem strukturellen "
        "V1-Challenger ausgeschlossen."
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


elif selected_asset == "EUR/USD":
    st.info(
        "💶 **EUR/USD Baseline v1.0.13:** Current, Literature Prior, "
        "Equal Weight sowie die Diagnosemodelle D/E werden mit exakt "
        "denselben Relative-/Absolute-Validity-Regeln wie Gold und WTI "
        "getestet. Es werden keine neuen FX-Faktoren nachträglich optimiert. "
        "Besonders beobachten wir, ob Subgewichte oder Säulengewichte "
        "den Unterschied verursachen und welche Faktoren Direction- bzw. "
        "Risk-State-Information liefern."
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


elif selected_asset == "EUR/USD":
    with st.expander(
        "💶 EUR/USD-spezifische Interpretationshilfe",
        expanded=True
    ):
        st.markdown(
            """
Beim EUR/USD-Test achten wir besonders auf:

1. **Makro-Säule:** Current 35 % vs. Literature Prior 25 %.
2. **Technischer Trend:** Current 20 % vs. Literature Prior 35 %.
3. **Positionierung:** Current 20 % vs. Literature Prior 15 %.
4. **CFTC innerhalb Positionierung:** Current 70 % vs. Literature 90 %.
5. **USD-Faktor:** im Score invertiert; ein hoher normalisierter
   EUR/USD-freundlicher USD-Score bedeutet relativ schwächeren Dollar.
6. **EVZ / Vola:** vor allem als möglicher Risk-State-Faktor interpretieren,
   nicht automatisch als Direction-Signal.

Ein zentraler Prüfpunkt ist außerdem, ob die aktuell **US-zentrierte
Makroarchitektur** für ein relatives Währungspaar ausreicht. Falls
Fed Policy oder US Real Yields allein schwach sind, wäre ein späterer
separater Test mit **Fed–EZB- bzw. US–Euro-Zinsdifferenzen** ökonomisch
plausibler als bloßes weiteres Gewichtstuning.
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
                "Score vs bessere FwdMAE": safe_spearman(
                    factor_score,
                    targets["Fwd_MAE_20D"],
                ),
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
                    "Score vs bessere FwdMAE": "{:+.3f}",
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


elif selected_asset == "Nasdaq 100":
    st.markdown("---")
    st.subheader(
        "7A️⃣ Nasdaq-100-Faktoren – Direction vs. Risk-State"
    )

    st.caption(
        "Positive IC-Werte sprechen für Forward-Direction. Stress-AUC, "
        "FwdVol- und MAE-Bezug werden separat bewertet. Die Tabelle "
        "verändert keine Gewichte."
    )

    nasdaq_factor_rows = []
    nasdaq_factor_map = {
        "Fed Policy": "fed_policy",
        "US Real Yields": "real_yields",
        "USD Index": "usd_index",
        "Net Liquidity": "net_liquidity",
        "CFTC Nasdaq-100 Non-Commercials": "cot_noncommercials",
        "CNN Fear & Greed": "fear_greed",
        "Market Momentum": "market_momentum",
        "VXN / Nasdaq Volatility": "vix_score",
        "Distance 50MA": "distance_50ma",
        "Distance 200MA": "distance_200ma",
        "RSI": "rsi_momentum",
        "Credit Spreads": "credit_spreads",
        "MOVE": "move_index",
        "VVIX": "vvix_score",
    }

    for label, factor in nasdaq_factor_map.items():
        if factor not in norm_df.columns:
            continue

        factor_score = pd.to_numeric(
            norm_df[factor],
            errors="coerce",
        )

        nasdaq_factor_rows.append(
            {
                "Faktor": label,
                "N": int(
                    (
                        factor_score.notna()
                        & targets["Fwd_Return_20D"].notna()
                    ).sum()
                ),
                "IC 20D": safe_spearman(
                    factor_score,
                    targets["Fwd_Return_20D"],
                ),
                "Stress AUC 20D": binary_auc(
                    factor_score,
                    targets["Stress_Event_20D"],
                    higher_predictor_means_event=False,
                ),
                "Score vs niedrigere FwdVol": safe_spearman(
                    factor_score,
                    -targets["Fwd_Realized_Vol_20D"],
                ),
                "Score vs bessere FwdMAE": safe_spearman(
                    factor_score,
                    targets["Fwd_MAE_20D"],
                ),
            }
        )

    nasdaq_factor_table = pd.DataFrame(nasdaq_factor_rows)

    if not nasdaq_factor_table.empty:
        nasdaq_factor_table = (
            nasdaq_factor_table
            .sort_values(
                "Stress AUC 20D",
                ascending=False,
                na_position="last",
            )
            .reset_index(drop=True)
        )

        st.dataframe(
            nasdaq_factor_table.style.format(
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

        st.caption(
            "VXN, VVIX, Credit und MOVE werden nicht allein anhand des "
            "Return-IC beurteilt: sie können als Risk-State-Faktoren "
            "wertvoll sein, auch wenn ihr Direction-IC schwach ist."
        )

elif selected_asset == "S&P 500":
    st.markdown("---")
    st.subheader(
        "7A️⃣ S&P-500-Faktoren – Direction vs. Risk-State"
    )

    st.caption(
        "Positive IC-Werte sprechen für Forward-Direction. Stress-AUC, "
        "FwdVol- und MAE-Bezug werden separat bewertet. Die Tabelle "
        "verändert keine Gewichte."
    )

    sp500_factor_rows = []
    sp500_factor_map = {
        "Fed Policy": "fed_policy",
        "US Real Yields": "real_yields",
        "USD Index": "usd_index",
        "Net Liquidity": "net_liquidity",
        "CFTC E-mini S&P Non-Commercials": "cot_noncommercials",
        "CNN Fear & Greed": "fear_greed",
        "Market Momentum": "market_momentum",
        "VIX": "vix_score",
        "Distance 50MA": "distance_50ma",
        "Distance 200MA": "distance_200ma",
        "RSI": "rsi_momentum",
        "S&P 500 P/E Valuation": "pe_valuation",
        "Credit Spreads": "credit_spreads",
        "MOVE": "move_index",
        "VVIX": "vvix_score",
    }

    for label, factor in sp500_factor_map.items():
        if factor not in norm_df.columns:
            continue

        factor_score = pd.to_numeric(
            norm_df[factor],
            errors="coerce",
        )

        sp500_factor_rows.append({
            "Faktor": label,
            "N": int((
                factor_score.notna()
                & targets["Fwd_Return_20D"].notna()
            ).sum()),
            "IC 20D": safe_spearman(
                factor_score,
                targets["Fwd_Return_20D"],
            ),
            "Stress AUC 20D": binary_auc(
                factor_score,
                targets["Stress_Event_20D"],
                higher_predictor_means_event=False,
            ),
            "Score vs niedrigere FwdVol": safe_spearman(
                factor_score,
                -targets["Fwd_Realized_Vol_20D"],
            ),
            "Score vs bessere FwdMAE": safe_spearman(
                factor_score,
                targets["Fwd_MAE_20D"],
            ),
        })

    sp500_factor_table = pd.DataFrame(sp500_factor_rows)

    if not sp500_factor_table.empty:
        sp500_factor_table = sp500_factor_table.sort_values(
            "Stress AUC 20D",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

        st.dataframe(
            sp500_factor_table.style.format(
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

        st.caption(
            "VIX, VVIX, Credit und MOVE werden nicht allein anhand des "
            "Return-IC beurteilt: sie können als Risk-State-Faktoren "
            "nützlich sein, auch wenn ihre Direction-ICs schwach sind."
        )

elif selected_asset == "EUR/USD":
    st.markdown("---")
    st.subheader(
        "7A️⃣ EUR/USD-Faktoren – Einzelbeitrag zum 20D-Forward-Verhalten"
    )

    st.caption(
        "Die Faktoren sind bereits in derselben Richtung normalisiert wie "
        "der EUR/USD-Regime-Score. Positive IC-Werte bedeuten daher: "
        "höherer Faktor-Score geht im Mittel mit höherem künftigem EUR/USD "
        "einher. Die Tabelle verändert keine Gewichte."
    )

    eur_factor_rows = []

    eur_factor_map = {
        "Fed Policy": "fed_policy",
        "US Real Yields": "real_yields",
        "USD Index (invertiert für EUR/USD)": "usd_index",
        "Net Liquidity": "net_liquidity",
        "CFTC Euro FX Non-Commercials": "cot_noncommercials",
        "CNN Fear & Greed": "fear_greed",
        "Market Momentum": "market_momentum",
        "EVZ / EURUSD Volatility": "vix_score",
        "Distance 50MA": "distance_50ma",
        "Distance 200MA": "distance_200ma",
        "RSI": "rsi_momentum",
        "Credit Spreads": "credit_spreads",
        "MOVE": "move_index",
    }

    for label, factor in eur_factor_map.items():
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

        eur_factor_rows.append(
            {
                "Faktor": label,
                "N": valid_n,
                "IC 20D": ic20,
                "Stress AUC 20D": stress_auc,
                "Score vs niedrigere FwdVol": vol_relation,
                "Score vs bessere FwdMAE": mae_relation,
            }
        )

    if eur_factor_rows:
        eur_factor_table = pd.DataFrame(
            eur_factor_rows
        ).sort_values(
            "IC 20D",
            ascending=False,
            na_position="last",
        )

        st.dataframe(
            eur_factor_table.style.format(
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

        st.caption(
            "Besonders wichtig ist die Trennung zwischen Direction und "
            "Risk-State. EVZ, MOVE oder Credit können als Risikofaktoren "
            "wertvoll sein, selbst wenn ihr Forward-Return-IC gering ist. "
            "Umgekehrt sollte ein Direction-Faktor nicht allein wegen "
            "einer guten Stress-AUC höher gewichtet werden."
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
# 19G. GOLD ROLE, FACTOR & ORIENTATION DIAGNOSIS
# ============================================================

gold_role_table = pd.DataFrame()
gold_zone_table = pd.DataFrame()
gold_period_table = pd.DataFrame()
gold_nonoverlap_table = pd.DataFrame()
gold_window_table = pd.DataFrame()
gold_gate_table = pd.DataFrame()
gold_orientation_table = pd.DataFrame()

if selected_asset == "Gold (XAU/USD)":
    st.markdown("---")
    st.subheader(
        "🪙 Gold – Direction, Risk-State oder falsche Score-Orientierung?"
    )

    st.caption(
        "Gold unterscheidet sich strukturell von Aktien: Safe-Haven-, "
        "Real-Yield- und USD-Effekte können die Interpretation verändern. "
        "A/Current bleibt Referenz. B, D und E bleiben unverändert. "
        "Ein invertierter Current-Score wird nur diagnostisch angezeigt "
        "und ist ausdrücklich kein neues Modell."
    )

    gold_models = {
        "A · Current": model_frames[
            MODEL_CURRENT
        ],
        "B · Literature Prior": model_frames[
            MODEL_LITERATURE
        ],
        "D · Lit Subweights only": diagnostic_frames[
            "D · Lit Subweights only"
        ],
        "E · Lit Pillars only": diagnostic_frames[
            "E · Lit Pillars only"
        ],
    }

    gold_common = pd.Series(
        True,
        index=targets.index,
    )

    for frame in gold_models.values():
        gold_common &= (
            frame[
                "Final_Regime_Score"
            ].notna()
            &
            (
                frame[
                    "Model_Data_Coverage"
                ]
                >= float(
                    min_coverage
                )
            )
        )

    # --------------------------------------------------------
    # 1. MODEL ROLE MATRIX
    # --------------------------------------------------------

    st.markdown(
        "### 1. Modellrollen – Direction vs. Gold-eigener Risk-State"
    )

    role_rows = []

    for model_name, frame in gold_models.items():
        score = frame[
            "Final_Regime_Score"
        ].where(
            gold_common
        )

        direction20, signals20 = directional_accuracy(
            score,
            targets[
                "Fwd_Return_20D"
            ],
        )

        phase_auc, phase_count = phase_median_stress_auc(
            score,
            targets[
                "Stress_Event_20D"
            ],
            horizon=20,
        )

        role_rows.append(
            {
                "Modell": model_name,
                "N": int(
                    gold_common.sum()
                ),
                "IC 5D": safe_spearman(
                    score,
                    targets[
                        "Fwd_Return_5D"
                    ],
                ),
                "IC 20D": safe_spearman(
                    score,
                    targets[
                        "Fwd_Return_20D"
                    ],
                ),
                "IC 60D": safe_spearman(
                    score,
                    targets[
                        "Fwd_Return_60D"
                    ],
                ),
                "Direction 20D": direction20,
                "Signals 20D": signals20,
                "Stress AUC 20D": binary_auc(
                    score,
                    targets[
                        "Stress_Event_20D"
                    ],
                    higher_predictor_means_event=False,
                ),
                "Phase-Median Stress AUC": phase_auc,
                "AUC-Phasen": phase_count,
                "Score vs niedrigere FwdVol": safe_spearman(
                    score,
                    -targets[
                        "Fwd_Realized_Vol_20D"
                    ],
                ),
                "Score vs bessere FwdMAE": safe_spearman(
                    score,
                    targets[
                        "Fwd_MAE_20D"
                    ],
                ),
                "Ø Coverage": float(
                    frame[
                        "Model_Data_Coverage"
                    ]
                    .where(
                        gold_common
                    )
                    .mean()
                ),
            }
        )

    gold_role_table = pd.DataFrame(
        role_rows
    )

    st.dataframe(
        gold_role_table.style.format(
            {
                "N": "{:.0f}",
                "IC 5D": "{:+.3f}",
                "IC 20D": "{:+.3f}",
                "IC 60D": "{:+.3f}",
                "Direction 20D": "{:.1%}",
                "Signals 20D": "{:.0f}",
                "Stress AUC 20D": "{:.3f}",
                "Phase-Median Stress AUC": "{:.3f}",
                "AUC-Phasen": "{:.0f}",
                "Score vs niedrigere FwdVol": "{:+.3f}",
                "Score vs bessere FwdMAE": "{:+.3f}",
                "Ø Coverage": "{:.1f}%",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 2. ORIENTATION AUDIT
    # --------------------------------------------------------

    st.markdown(
        "### 2. Current Orientation Audit – nur Diagnose"
    )

    current_gold_score = gold_models[
        "A · Current"
    ][
        "Final_Regime_Score"
    ].where(
        gold_common
    )

    inverted_gold_score = (
        100.0
        - current_gold_score
    )

    orientation_rows = []

    for label, score in [
        (
            "Current",
            current_gold_score,
        ),
        (
            "100 − Current (nur Diagnose)",
            inverted_gold_score,
        ),
    ]:
        direction20, signals20 = directional_accuracy(
            score,
            targets[
                "Fwd_Return_20D"
            ],
        )

        phase_auc, phases = phase_median_stress_auc(
            score,
            targets[
                "Stress_Event_20D"
            ],
            horizon=20,
        )

        orientation_rows.append(
            {
                "Orientierung": label,
                "IC 5D": safe_spearman(
                    score,
                    targets[
                        "Fwd_Return_5D"
                    ],
                ),
                "IC 20D": safe_spearman(
                    score,
                    targets[
                        "Fwd_Return_20D"
                    ],
                ),
                "IC 60D": safe_spearman(
                    score,
                    targets[
                        "Fwd_Return_60D"
                    ],
                ),
                "Direction 20D": direction20,
                "Signals": signals20,
                "Stress AUC 20D": binary_auc(
                    score,
                    targets[
                        "Stress_Event_20D"
                    ],
                    higher_predictor_means_event=False,
                ),
                "Phase-Median Stress AUC": phase_auc,
                "Score vs niedrigere FwdVol": safe_spearman(
                    score,
                    -targets[
                        "Fwd_Realized_Vol_20D"
                    ],
                ),
                "Score vs bessere FwdMAE": safe_spearman(
                    score,
                    targets[
                        "Fwd_MAE_20D"
                    ],
                ),
            }
        )

    gold_orientation_table = pd.DataFrame(
        orientation_rows
    )

    st.dataframe(
        gold_orientation_table.style.format(
            {
                "IC 5D": "{:+.3f}",
                "IC 20D": "{:+.3f}",
                "IC 60D": "{:+.3f}",
                "Direction 20D": "{:.1%}",
                "Signals": "{:.0f}",
                "Stress AUC 20D": "{:.3f}",
                "Phase-Median Stress AUC": "{:.3f}",
                "Score vs niedrigere FwdVol": "{:+.3f}",
                "Score vs bessere FwdMAE": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    current_orientation = gold_orientation_table[
        gold_orientation_table[
            "Orientierung"
        ]
        == "Current"
    ].iloc[
        0
    ]

    inverted_orientation = gold_orientation_table[
        gold_orientation_table[
            "Orientierung"
        ]
        == "100 − Current (nur Diagnose)"
    ].iloc[
        0
    ]

    current_ic20 = float(
        current_orientation[
            "IC 20D"
        ]
    )

    inverted_ic20 = float(
        inverted_orientation[
            "IC 20D"
        ]
    )

    inverted_positive_horizons = sum(
        bool(
            np.isfinite(
                inverted_orientation[
                    f"IC {h}D"
                ]
            )
            and inverted_orientation[
                f"IC {h}D"
            ] > 0
        )
        for h in FORWARD_HORIZONS
    )

    orientation_flag = bool(
        np.isfinite(
            current_ic20
        )
        and np.isfinite(
            inverted_ic20
        )
        and (
            inverted_ic20
            - current_ic20
        )
        >= GOLD_ORIENTATION_MIN_IC_IMPROVEMENT
        and inverted_positive_horizons
        >= GOLD_ORIENTATION_MIN_POSITIVE_HORIZONS
    )

    if orientation_flag:
        st.warning(
            "⚠️ **STRUKTURELLER ORIENTIERUNGSHINWEIS:** Die diagnostische "
            "Umkehr `100 − Current` verbessert die Forward-IC-Richtung "
            "deutlich. Das ist **kein Beweis**, dass wir den Produktionsscore "
            "einfach invertieren dürfen. Es zeigt nur, dass vor einem Gold-"
            "Shadow zuerst die Faktorzeichen bzw. die Score-Semantik geprüft "
            "werden müssen."
        )
    else:
        st.info(
            "ℹ️ Der Orientation-Audit liefert keinen ausreichend starken "
            "Hinweis auf eine einfache globale Score-Umkehr."
        )

    # --------------------------------------------------------
    # 3. CURRENT SCORE ZONES
    # --------------------------------------------------------

    st.markdown(
        "### 3. Current – Forward-Verhalten nach Score-Zone"
    )

    zone_frame = pd.DataFrame(
        {
            "score": pd.to_numeric(
                current_gold_score,
                errors="coerce",
            ),
            "stress": pd.to_numeric(
                targets[
                    "Stress_Event_20D"
                ],
                errors="coerce",
            ),
            "ret5": pd.to_numeric(
                targets[
                    "Fwd_Return_5D"
                ],
                errors="coerce",
            ),
            "ret20": pd.to_numeric(
                targets[
                    "Fwd_Return_20D"
                ],
                errors="coerce",
            ),
            "ret60": pd.to_numeric(
                targets[
                    "Fwd_Return_60D"
                ],
                errors="coerce",
            ),
            "fwdvol": pd.to_numeric(
                targets[
                    "Fwd_Realized_Vol_20D"
                ],
                errors="coerce",
            ),
            "mae": pd.to_numeric(
                targets[
                    "Fwd_MAE_20D"
                ],
                errors="coerce",
            ),
        }
    ).dropna(
        subset=[
            "score"
        ]
    )

    zone_rows = []

    for label, mask in [
        (
            "≤40 Low Score",
            zone_frame[
                "score"
            ] <= 40,
        ),
        (
            "40–60 Neutral",
            (
                zone_frame[
                    "score"
                ] > 40
            )
            &
            (
                zone_frame[
                    "score"
                ] < 60
            ),
        ),
        (
            "≥60 High Score",
            zone_frame[
                "score"
            ] >= 60,
        ),
    ]:
        z = zone_frame[
            mask
        ]

        zone_rows.append(
            {
                "Score-Zone": label,
                "N": int(
                    len(
                        z
                    )
                ),
                "Ø Fwd Return 5D": (
                    float(
                        z[
                            "ret5"
                        ].mean()
                    )
                    if z[
                        "ret5"
                    ].notna().any()
                    else np.nan
                ),
                "Ø Fwd Return 20D": (
                    float(
                        z[
                            "ret20"
                        ].mean()
                    )
                    if z[
                        "ret20"
                    ].notna().any()
                    else np.nan
                ),
                "Ø Fwd Return 60D": (
                    float(
                        z[
                            "ret60"
                        ].mean()
                    )
                    if z[
                        "ret60"
                    ].notna().any()
                    else np.nan
                ),
                "Stressrate 20D": (
                    float(
                        z[
                            "stress"
                        ].mean()
                    )
                    if z[
                        "stress"
                    ].notna().any()
                    else np.nan
                ),
                "Ø FwdVol 20D": (
                    float(
                        z[
                            "fwdvol"
                        ].mean()
                    )
                    if z[
                        "fwdvol"
                    ].notna().any()
                    else np.nan
                ),
                "Ø FwdMAE 20D": (
                    float(
                        z[
                            "mae"
                        ].mean()
                    )
                    if z[
                        "mae"
                    ].notna().any()
                    else np.nan
                ),
            }
        )

    gold_zone_table = pd.DataFrame(
        zone_rows
    )

    st.dataframe(
        gold_zone_table.style.format(
            {
                "N": "{:.0f}",
                "Ø Fwd Return 5D": "{:+.2%}",
                "Ø Fwd Return 20D": "{:+.2%}",
                "Ø Fwd Return 60D": "{:+.2%}",
                "Stressrate 20D": "{:.1%}",
                "Ø FwdVol 20D": "{:.2%}",
                "Ø FwdMAE 20D": "{:+.2%}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 4. PERIOD ROBUSTNESS
    # --------------------------------------------------------

    st.markdown(
        "### 4. Perioden-Robustheit"
    )

    score_map = {
        name: frame[
            "Final_Regime_Score"
        ].where(
            gold_common
        )
        for name, frame
        in gold_models.items()
    }

    period_rows = []

    period_rows.extend(
        confirmation_period_metrics(
            "Gesamter Common Sample",
            gold_common,
            score_map,
            targets,
        )
    )

    for label, y1, y2 in [
        (
            "2012–2016",
            2012,
            2016,
        ),
        (
            "2017–2020",
            2017,
            2020,
        ),
        (
            "2021–2025",
            2021,
            2025,
        ),
    ]:
        mask = (
            gold_common
            &
            (
                targets.index.year
                >= y1
            )
            &
            (
                targets.index.year
                <= y2
            )
        )

        period_rows.extend(
            confirmation_period_metrics(
                label,
                mask,
                score_map,
                targets,
            )
        )

    gold_period_table = pd.DataFrame(
        period_rows
    )

    st.dataframe(
        gold_period_table.style.format(
            {
                "N": "{:.0f}",
                "IC 20D": "{:+.3f}",
                "IC 60D": "{:+.3f}",
                "Direction 20D": "{:.1%}",
                "Signals 20D": "{:.0f}",
                "Stress AUC 20D": "{:.3f}",
                "Score vs bessere FwdMAE": "{:+.3f}",
                "Score vs niedrigere FwdVol": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 5. NON-OVERLAP
    # --------------------------------------------------------

    st.markdown(
        "### 5. Non-Overlap-Diagnostik – IC20 / IC60"
    )

    nonoverlap_rows = []

    for horizon in [
        20,
        60,
    ]:
        target_h = targets[
            f"Fwd_Return_{horizon}D"
        ]

        for model_name, score in score_map.items():
            median_ic, phases = phase_median_nonoverlap_ic(
                score,
                target_h,
                horizon,
            )

            nonoverlap_rows.append(
                {
                    "Horizont": f"{horizon}D",
                    "Modell": model_name,
                    "Median Non-Overlap IC": median_ic,
                    "Verfügbare Phasen": phases,
                }
            )

    gold_nonoverlap_table = pd.DataFrame(
        nonoverlap_rows
    )

    st.dataframe(
        gold_nonoverlap_table.style.format(
            {
                "Median Non-Overlap IC": "{:+.3f}",
                "Verfügbare Phasen": "{:.0f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 6. PRE-DEFINED GOLD REGIME WINDOWS
    # --------------------------------------------------------

    st.markdown(
        "### 6. Current in vorab definierten Gold-Regimefenstern"
    )

    window_rows = []

    price_series = pd.to_numeric(
        raw_df[
            "asset_price"
        ],
        errors="coerce",
    )

    for (
        window_name,
        start_text,
        end_text,
    ) in GOLD_DIAGNOSTIC_WINDOWS:
        start = pd.Timestamp(
            start_text
        )

        end = pd.Timestamp(
            end_text
        )

        during_score = current_gold_score[
            (
                current_gold_score.index
                >= start
            )
            &
            (
                current_gold_score.index
                <= end
            )
        ].dropna()

        pre_score = current_gold_score[
            current_gold_score.index
            < start
        ].dropna().tail(
            60
        )

        price_window = price_series[
            (
                price_series.index
                >= start
            )
            &
            (
                price_series.index
                <= end
            )
        ].dropna()

        if not price_window.empty:
            max_drawdown = float(
                (
                    price_window
                    / price_window.cummax()
                    - 1.0
                ).min()
            )

            total_return = float(
                price_window.iloc[
                    -1
                ]
                / price_window.iloc[
                    0
                ]
                - 1.0
            )
        else:
            max_drawdown = np.nan
            total_return = np.nan

        window_rows.append(
            {
                "Regimefenster": window_name,
                "Start": start.date().isoformat(),
                "Ende": end.date().isoformat(),
                "N": int(
                    len(
                        during_score
                    )
                ),
                "Ø Score vorher 60T": (
                    float(
                        pre_score.mean()
                    )
                    if not pre_score.empty
                    else np.nan
                ),
                "Ø Score im Fenster": (
                    float(
                        during_score.mean()
                    )
                    if not during_score.empty
                    else np.nan
                ),
                "Min Score": (
                    float(
                        during_score.min()
                    )
                    if not during_score.empty
                    else np.nan
                ),
                "Max Score": (
                    float(
                        during_score.max()
                    )
                    if not during_score.empty
                    else np.nan
                ),
                "Anteil Score <40": (
                    float(
                        (
                            during_score
                            < 40
                        ).mean()
                    )
                    if not during_score.empty
                    else np.nan
                ),
                "Anteil Score ≥60": (
                    float(
                        (
                            during_score
                            >= 60
                        ).mean()
                    )
                    if not during_score.empty
                    else np.nan
                ),
                "Gold Return Fenster": total_return,
                "Gold MaxDrawdown Fenster": max_drawdown,
            }
        )

    gold_window_table = pd.DataFrame(
        window_rows
    )

    st.dataframe(
        gold_window_table.style.format(
            {
                "N": "{:.0f}",
                "Ø Score vorher 60T": "{:.1f}",
                "Ø Score im Fenster": "{:.1f}",
                "Min Score": "{:.1f}",
                "Max Score": "{:.1f}",
                "Anteil Score <40": "{:.1%}",
                "Anteil Score ≥60": "{:.1%}",
                "Gold Return Fenster": "{:+.1%}",
                "Gold MaxDrawdown Fenster": "{:.1%}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Diese Gold-Regimefenster wurden vor der Auswertung festgelegt. "
        "Sie dienen nur der Interpretation und nicht der Gewichtsanpassung."
    )

    # --------------------------------------------------------
    # 7. ROLE GATES
    # --------------------------------------------------------

    st.markdown(
        "### 7. Vorab festgelegte Rollen-Gates für Current"
    )

    current_role = gold_role_table[
        gold_role_table[
            "Modell"
        ]
        == "A · Current"
    ].iloc[
        0
    ]

    positive_ics = sum(
        bool(
            np.isfinite(
                current_role[
                    f"IC {h}D"
                ]
            )
            and current_role[
                f"IC {h}D"
            ] > 0
        )
        for h in FORWARD_HORIZONS
    )

    low_row = gold_zone_table[
        gold_zone_table[
            "Score-Zone"
        ]
        == "≤40 Low Score"
    ]

    high_row = gold_zone_table[
        gold_zone_table[
            "Score-Zone"
        ]
        == "≥60 High Score"
    ]

    low_stress = (
        float(
            low_row.iloc[
                0
            ][
                "Stressrate 20D"
            ]
        )
        if not low_row.empty
        else np.nan
    )

    high_stress = (
        float(
            high_row.iloc[
                0
            ][
                "Stressrate 20D"
            ]
        )
        if not high_row.empty
        else np.nan
    )

    stress_gap = (
        low_stress
        - high_stress
        if (
            np.isfinite(
                low_stress
            )
            and np.isfinite(
                high_stress
            )
        )
        else np.nan
    )

    direction_gate = bool(
        positive_ics
        >= 2
        and np.isfinite(
            current_role[
                "Direction 20D"
            ]
        )
        and current_role[
            "Direction 20D"
        ]
        >= GOLD_DIRECTION_MIN_ACCURACY
    )

    gate_rows = [
        {
            "Rolle": "Direction",
            "Kriterium": (
                "Mindestens 2/3 Forward-ICs positiv UND Direction20 ≥50%"
            ),
            "Erfüllt": direction_gate,
            "Messwert": (
                f"{positive_ics}/3 positive ICs · "
                f"Direction {current_role['Direction 20D']:.1%}"
            ),
        },
        {
            "Rolle": "Risk-State",
            "Kriterium": "Stress-AUC 20D ≥0.60",
            "Erfüllt": bool(
                np.isfinite(
                    current_role[
                        "Stress AUC 20D"
                    ]
                )
                and current_role[
                    "Stress AUC 20D"
                ]
                >= GOLD_RISK_MIN_AUC
            ),
            "Messwert": (
                f"{current_role['Stress AUC 20D']:.3f}"
                if np.isfinite(
                    current_role[
                        "Stress AUC 20D"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "Rolle": "Risk-State",
            "Kriterium": "Phase-Median Stress-AUC ≥0.55",
            "Erfüllt": bool(
                np.isfinite(
                    current_role[
                        "Phase-Median Stress AUC"
                    ]
                )
                and current_role[
                    "Phase-Median Stress AUC"
                ]
                >= GOLD_RISK_MIN_PHASE_AUC
            ),
            "Messwert": (
                f"{current_role['Phase-Median Stress AUC']:.3f}"
                if np.isfinite(
                    current_role[
                        "Phase-Median Stress AUC"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "Rolle": "Risk-State",
            "Kriterium": (
                "Stressrate Score≤40 minus Score≥60 ≥5 Prozentpunkte"
            ),
            "Erfüllt": bool(
                np.isfinite(
                    stress_gap
                )
                and stress_gap
                >= GOLD_RISK_MIN_STRESS_GAP
            ),
            "Messwert": (
                f"Low {low_stress:.1%} · High {high_stress:.1%} · "
                f"Gap {stress_gap:+.1%}"
                if np.isfinite(
                    stress_gap
                )
                else "n/a"
            ),
        },
        {
            "Rolle": "Risk-State",
            "Kriterium": (
                "Höherer Score korreliert mit niedrigerer FwdVol"
            ),
            "Erfüllt": bool(
                np.isfinite(
                    current_role[
                        "Score vs niedrigere FwdVol"
                    ]
                )
                and current_role[
                    "Score vs niedrigere FwdVol"
                ] > 0
            ),
            "Messwert": (
                f"{current_role['Score vs niedrigere FwdVol']:+.3f}"
                if np.isfinite(
                    current_role[
                        "Score vs niedrigere FwdVol"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "Rolle": "Risk-State",
            "Kriterium": (
                "Höherer Score korreliert mit besserer FwdMAE"
            ),
            "Erfüllt": bool(
                np.isfinite(
                    current_role[
                        "Score vs bessere FwdMAE"
                    ]
                )
                and current_role[
                    "Score vs bessere FwdMAE"
                ] > 0
            ),
            "Messwert": (
                f"{current_role['Score vs bessere FwdMAE']:+.3f}"
                if np.isfinite(
                    current_role[
                        "Score vs bessere FwdMAE"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "Rolle": "Orientation Audit",
            "Kriterium": (
                "Diagnostische Umkehr verbessert IC20 ≥0.05 "
                "und hat ≥2 positive Forward-ICs"
            ),
            "Erfüllt": orientation_flag,
            "Messwert": (
                f"Current IC20 {current_ic20:+.3f} · "
                f"Invertiert {inverted_ic20:+.3f} · "
                f"positive Horizonte {inverted_positive_horizons}/3"
            ),
        },
    ]

    gold_gate_table = pd.DataFrame(
        gate_rows
    )

    gold_gate_table[
        "Status"
    ] = gold_gate_table[
        "Erfüllt"
    ].map(
        {
            True: "✅",
            False: "❌",
        }
    )

    st.dataframe(
        gold_gate_table[
            [
                "Rolle",
                "Kriterium",
                "Status",
                "Messwert",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    risk_rows = gold_gate_table[
        gold_gate_table[
            "Rolle"
        ]
        == "Risk-State"
    ]

    risk_passed = int(
        risk_rows[
            "Erfüllt"
        ].sum()
    )

    st.markdown(
        "### 8. Gold-Diagnoseurteil"
    )

    if orientation_flag:
        st.warning(
            "🟠 **ORIENTIERUNG MUSS VOR EINEM GOLD-SHADOW GEKLÄRT WERDEN.** "
            "Die historische Diagnose spricht für einen möglichen "
            "Score-/Faktorzeichen-Konflikt. Deshalb wird aus diesem Lauf "
            "noch kein Future-Shadow abgeleitet."
        )

    elif (
        risk_passed
        == len(
            risk_rows
        )
        and not direction_gate
    ):
        st.success(
            "🟢 **CURRENT BESTÄTIGT SICH PRIMÄR ALS GOLD-RISK-STATE-FILTER.** "
            "Dann kann als nächster Schritt ein Current-only Future-Shadow "
            "vorbereitet werden."
        )

    elif (
        direction_gate
        and risk_passed
        >= 4
    ):
        st.success(
            "🟢 **CURRENT ZEIGT DIRECTION- UND RISK-STATE-EIGENSCHAFTEN.** "
            "Dann wird im nächsten Schritt geprüft, welche Rolle zeitlich "
            "robuster ist."
        )

    elif risk_passed >= max(
        1,
        len(
            risk_rows
        )
        - 1,
    ):
        st.warning(
            f"🟡 **RISK-STATE TEILWEISE BESTÄTIGT "
            f"({risk_passed}/{len(risk_rows)} Gates).** "
            "Vor einem Shadow wird die schwache Teilprüfung isoliert."
        )

    else:
        st.error(
            f"🔴 **CURRENT IST ALS GOLD-MODELL IN DER BESTEHENDEN "
            f"ORIENTIERUNG NICHT ROBUST GENUG "
            f"({risk_passed}/{len(risk_rows)} Risk-Gates).** "
            "Dann untersuchen wir Faktorzeichen und Säulenarchitektur, "
            "ohne aus diesem Datensatz sofort neue Gewichte zu optimieren."
        )

    st.info(
        "Wichtig: Der Orientation-Audit ist bewusst nur diagnostisch. "
        "Selbst ein klar besseres `100 − Current` wird nicht automatisch "
        "zum neuen Gold-Modell. Eine solche strukturelle Änderung müsste "
        "separat begründet, eingefroren und anschließend out-of-sample "
        "getestet werden."
    )



# ============================================================
# 19G2. GOLD STRUCTURAL DRIVER AUDIT
# ============================================================

gold_structural_sign_table = pd.DataFrame()
gold_level_change_table = pd.DataFrame()
gold_pillar_role_table = pd.DataFrame()
gold_driver_ablation_table = pd.DataFrame()
gold_macro_state_table = pd.DataFrame()
gold_safe_haven_state_table = pd.DataFrame()
gold_structural_gate_table = pd.DataFrame()

if selected_asset == "Gold (XAU/USD)":
    st.markdown("---")
    st.subheader(
        "🧬 Gold Structural Driver Audit"
    )

    st.caption(
        "Ziel: nicht neue Gewichte suchen, sondern klären, **warum** der "
        "bestehende Gold-Score strukturell schwach orientiert ist. "
        "Geprüft werden Faktorzeichen, Level-vs.-Änderung, Säulenrollen, "
        "Leave-one-driver-out sowie Makro- und Safe-Haven-Zustände. "
        "Keine Diagnose verändert den Produktionscode."
    )

    # --------------------------------------------------------
    # A. FACTOR SIGN STABILITY
    # --------------------------------------------------------

    st.markdown(
        "### A. Faktorzeichen – ist die aktuelle Orientierung zeitlich stabil?"
    )

    structural_factor_map = {
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

    sign_rows = []

    for label, factor in structural_factor_map.items():
        if factor not in norm_df.columns:
            continue

        current_factor = pd.to_numeric(
            norm_df[factor],
            errors="coerce",
        )

        opposite_factor = (
            100.0
            - current_factor
        )

        ic5 = safe_spearman(
            current_factor,
            targets["Fwd_Return_5D"],
        )
        ic20 = safe_spearman(
            current_factor,
            targets["Fwd_Return_20D"],
        )
        ic60 = safe_spearman(
            current_factor,
            targets["Fwd_Return_60D"],
        )

        nonoverlap20, phases20 = phase_median_nonoverlap_ic(
            current_factor,
            targets["Fwd_Return_20D"],
            20,
        )

        period_ics = []
        opposite_period_ics = []

        for _, y1, y2 in GOLD_FIXED_PERIODS:
            mask = (
                (targets.index.year >= y1)
                & (targets.index.year <= y2)
            )

            period_ics.append(
                safe_spearman(
                    current_factor.where(mask),
                    targets["Fwd_Return_20D"],
                )
            )

            opposite_period_ics.append(
                safe_spearman(
                    opposite_factor.where(mask),
                    targets["Fwd_Return_20D"],
                )
            )

        current_positive_periods = int(
            sum(
                np.isfinite(value)
                and value > 0
                for value in period_ics
            )
        )

        opposite_positive_periods = int(
            sum(
                np.isfinite(value)
                and value > 0
                for value in opposite_period_ics
            )
        )

        boot = {
            "prob_positive": np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "observed": np.nan,
        }

        # Bootstrap only factors with a non-trivial negative IC. This keeps the
        # Streamlit run tractable and avoids significance fishing on tiny noise.
        if (
            np.isfinite(ic20)
            and ic20 <= -GOLD_STRUCTURAL_MIN_ABS_IC
        ):
            boot = block_bootstrap_ic_difference(
                current_factor,
                opposite_factor,
                targets["Fwd_Return_20D"],
                block_length=int(
                    bootstrap_block
                ),
                n_boot=int(
                    bootstrap_runs
                ),
                seed=20260912,
            )

        sign_conflict = bool(
            np.isfinite(ic20)
            and ic20 <= -GOLD_STRUCTURAL_MIN_ABS_IC
            and np.isfinite(nonoverlap20)
            and nonoverlap20 < 0
            and opposite_positive_periods
            >= GOLD_STRUCTURAL_MIN_PERIOD_CONFIRMATIONS
            and np.isfinite(
                boot.get(
                    "prob_positive",
                    np.nan,
                )
            )
            and boot[
                "prob_positive"
            ]
            >= GOLD_STRUCTURAL_BOOTSTRAP_PROB
        )

        sign_rows.append(
            {
                "Faktor": label,
                "Current IC 5D": ic5,
                "Current IC 20D": ic20,
                "Current IC 60D": ic60,
                "Non-Overlap IC20": nonoverlap20,
                "Non-Overlap Phasen": phases20,
                "Current positive Perioden": current_positive_periods,
                "Opposite positive Perioden": opposite_positive_periods,
                "Bootstrap P(Opposite>Current)": boot.get(
                    "prob_positive",
                    np.nan,
                ),
                "Bootstrap Δ Low": boot.get(
                    "lower",
                    np.nan,
                ),
                "Bootstrap Δ High": boot.get(
                    "upper",
                    np.nan,
                ),
                "Struktureller Sign-Konflikt": sign_conflict,
            }
        )

    gold_structural_sign_table = (
        pd.DataFrame(
            sign_rows
        )
        .sort_values(
            "Current IC 20D",
            ascending=True,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    st.dataframe(
        gold_structural_sign_table.style.format(
            {
                "Current IC 5D": "{:+.3f}",
                "Current IC 20D": "{:+.3f}",
                "Current IC 60D": "{:+.3f}",
                "Non-Overlap IC20": "{:+.3f}",
                "Non-Overlap Phasen": "{:.0f}",
                "Current positive Perioden": "{:.0f}",
                "Opposite positive Perioden": "{:.0f}",
                "Bootstrap P(Opposite>Current)": "{:.1%}",
                "Bootstrap Δ Low": "{:+.3f}",
                "Bootstrap Δ High": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Ein Sign-Konflikt wird nur markiert, wenn der aktuelle Faktor nicht "
        "nur global negativ ist, sondern auch Non-Overlap, Teilperioden und "
        "Block-Bootstrap den Konflikt stützen. Das ist ein Diagnosehinweis, "
        "keine automatische Invertierung."
    )

    # --------------------------------------------------------
    # B. LEVEL VS 20D CHANGE
    # --------------------------------------------------------

    st.markdown(
        "### B. Makro/CFTC – Level oder Veränderung?"
    )

    change_specs = {
        "Fed Policy": {
            "factor": "fed_policy",
            "kind": "diff",
            "sign": -1.0,
            "theory": "fallende Policy Rate = Gold-Tailwind",
        },
        "Real Yields": {
            "factor": "real_yields",
            "kind": "diff",
            "sign": -1.0,
            "theory": "fallende Realrenditen = Gold-Tailwind",
        },
        "USD Index": {
            "factor": "usd_index",
            "kind": "pct",
            "sign": -1.0,
            "theory": "schwächerer USD = Gold-Tailwind",
        },
        "Net Liquidity": {
            "factor": "net_liquidity",
            "kind": "pct",
            "sign": 1.0,
            "theory": "steigende Netto-Liquidität = Gold-Tailwind",
        },
        "CFTC Non-Commercials": {
            "factor": "cot_noncommercials",
            "kind": "diff",
            "sign": 1.0,
            "theory": "steigende Netto-Spekulantenposition = Momentum-Tailwind",
        },
    }

    change_rows = []

    for label, spec in change_specs.items():
        factor = spec["factor"]

        if (
            factor not in raw_df.columns
            or factor not in norm_df.columns
        ):
            continue

        raw_series = pd.to_numeric(
            raw_df[factor],
            errors="coerce",
        )

        level_score = pd.to_numeric(
            norm_df[factor],
            errors="coerce",
        )

        if spec["kind"] == "pct":
            change_raw = raw_series.pct_change(
                20,
                fill_method=None,
            )
        else:
            change_raw = raw_series.diff(
                20
            )

        favorable_change = (
            float(
                spec["sign"]
            )
            * change_raw
        )

        level_ic5 = safe_spearman(
            level_score,
            targets["Fwd_Return_5D"],
        )
        level_ic20 = safe_spearman(
            level_score,
            targets["Fwd_Return_20D"],
        )
        level_ic60 = safe_spearman(
            level_score,
            targets["Fwd_Return_60D"],
        )

        change_ic5 = safe_spearman(
            favorable_change,
            targets["Fwd_Return_5D"],
        )
        change_ic20 = safe_spearman(
            favorable_change,
            targets["Fwd_Return_20D"],
        )
        change_ic60 = safe_spearman(
            favorable_change,
            targets["Fwd_Return_60D"],
        )

        level_nonoverlap, _ = phase_median_nonoverlap_ic(
            level_score,
            targets["Fwd_Return_20D"],
            20,
        )

        change_nonoverlap, _ = phase_median_nonoverlap_ic(
            favorable_change,
            targets["Fwd_Return_20D"],
            20,
        )

        change_period_ics = []

        for _, y1, y2 in GOLD_FIXED_PERIODS:
            mask = (
                (targets.index.year >= y1)
                & (targets.index.year <= y2)
            )

            change_period_ics.append(
                safe_spearman(
                    favorable_change.where(
                        mask
                    ),
                    targets[
                        "Fwd_Return_20D"
                    ],
                )
            )

        change_positive_periods = int(
            sum(
                np.isfinite(value)
                and value > 0
                for value in change_period_ics
            )
        )

        transformation_candidate = bool(
            np.isfinite(
                level_ic20
            )
            and np.isfinite(
                change_ic20
            )
            and (
                change_ic20
                - level_ic20
            )
            >= GOLD_CHANGE_MIN_IC_IMPROVEMENT
            and np.isfinite(
                change_nonoverlap
            )
            and change_nonoverlap > 0
            and change_positive_periods
            >= GOLD_CHANGE_MIN_PERIOD_CONFIRMATIONS
        )

        change_rows.append(
            {
                "Treiber": label,
                "Vorab-Theorie": spec["theory"],
                "Level IC 5D": level_ic5,
                "Level IC 20D": level_ic20,
                "Level IC 60D": level_ic60,
                "Level Non-Overlap IC20": level_nonoverlap,
                "20D-Änderung IC 5D": change_ic5,
                "20D-Änderung IC 20D": change_ic20,
                "20D-Änderung IC 60D": change_ic60,
                "Änderung Non-Overlap IC20": change_nonoverlap,
                "Δ IC20 Änderung−Level": (
                    change_ic20
                    - level_ic20
                    if (
                        np.isfinite(
                            change_ic20
                        )
                        and np.isfinite(
                            level_ic20
                        )
                    )
                    else np.nan
                ),
                "Änderung positive Perioden": change_positive_periods,
                "Transformations-Kandidat": transformation_candidate,
            }
        )

    gold_level_change_table = pd.DataFrame(
        change_rows
    )

    st.dataframe(
        gold_level_change_table.style.format(
            {
                "Level IC 5D": "{:+.3f}",
                "Level IC 20D": "{:+.3f}",
                "Level IC 60D": "{:+.3f}",
                "Level Non-Overlap IC20": "{:+.3f}",
                "20D-Änderung IC 5D": "{:+.3f}",
                "20D-Änderung IC 20D": "{:+.3f}",
                "20D-Änderung IC 60D": "{:+.3f}",
                "Änderung Non-Overlap IC20": "{:+.3f}",
                "Δ IC20 Änderung−Level": "{:+.3f}",
                "Änderung positive Perioden": "{:.0f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Diese Prüfung verändert die bestehende Normalisierung nicht. "
        "Sie testet nur die strukturelle Hypothese, dass Gold auf "
        "**Veränderungen** von Realzinsen, USD, Liquidität oder Positionierung "
        "stärker reagiert als auf deren langfristiges Level/Perzentil."
    )

    # --------------------------------------------------------
    # C. PILLAR ROLE AUDIT
    # --------------------------------------------------------

    st.markdown(
        "### C. Welche Säule misst Direction – welche eher Risiko?"
    )

    current_gold_frame = model_frames[
        MODEL_CURRENT
    ]

    current_gold_cfg = current_model_config(
        "Gold (XAU/USD)"
    )

    pillar_rows = []

    for pillar, weight in current_gold_cfg[
        "pillar_weights"
    ].items():
        if float(
            weight
        ) <= 0:
            continue

        column = (
            f"Pillar::{pillar}"
        )

        if column not in current_gold_frame.columns:
            continue

        pillar_score = pd.to_numeric(
            current_gold_frame[
                column
            ],
            errors="coerce",
        )

        direction20, signals20 = directional_accuracy(
            pillar_score,
            targets[
                "Fwd_Return_20D"
            ],
        )

        phase_auc, phase_count = phase_median_stress_auc(
            pillar_score,
            targets[
                "Stress_Event_20D"
            ],
            horizon=20,
        )

        pillar_rows.append(
            {
                "Säule": pillar,
                "Basisgewicht": float(
                    weight
                ),
                "IC 5D": safe_spearman(
                    pillar_score,
                    targets[
                        "Fwd_Return_5D"
                    ],
                ),
                "IC 20D": safe_spearman(
                    pillar_score,
                    targets[
                        "Fwd_Return_20D"
                    ],
                ),
                "IC 60D": safe_spearman(
                    pillar_score,
                    targets[
                        "Fwd_Return_60D"
                    ],
                ),
                "Direction 20D": direction20,
                "Signals": signals20,
                "Stress AUC 20D": binary_auc(
                    pillar_score,
                    targets[
                        "Stress_Event_20D"
                    ],
                    higher_predictor_means_event=False,
                ),
                "Phase-Median Stress AUC": phase_auc,
                "AUC-Phasen": phase_count,
                "Score vs niedrigere FwdVol": safe_spearman(
                    pillar_score,
                    -targets[
                        "Fwd_Realized_Vol_20D"
                    ],
                ),
                "Score vs bessere FwdMAE": safe_spearman(
                    pillar_score,
                    targets[
                        "Fwd_MAE_20D"
                    ],
                ),
            }
        )

    gold_pillar_role_table = (
        pd.DataFrame(
            pillar_rows
        )
        .sort_values(
            "IC 20D",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    st.dataframe(
        gold_pillar_role_table.style.format(
            {
                "Basisgewicht": "{:.1%}",
                "IC 5D": "{:+.3f}",
                "IC 20D": "{:+.3f}",
                "IC 60D": "{:+.3f}",
                "Direction 20D": "{:.1%}",
                "Signals": "{:.0f}",
                "Stress AUC 20D": "{:.3f}",
                "Phase-Median Stress AUC": "{:.3f}",
                "AUC-Phasen": "{:.0f}",
                "Score vs niedrigere FwdVol": "{:+.3f}",
                "Score vs bessere FwdMAE": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # D. LEAVE-ONE-DRIVER-OUT
    # --------------------------------------------------------

    st.markdown(
        "### D. Leave-one-driver-out – welcher Treiber zieht das Gesamtmodell?"
    )

    base_score = pd.to_numeric(
        current_gold_frame[
            "Final_Regime_Score"
        ],
        errors="coerce",
    )

    base_ic20 = safe_spearman(
        base_score,
        targets[
            "Fwd_Return_20D"
        ],
    )

    base_auc = binary_auc(
        base_score,
        targets[
            "Stress_Event_20D"
        ],
        higher_predictor_means_event=False,
    )

    base_vol_rel = safe_spearman(
        base_score,
        -targets[
            "Fwd_Realized_Vol_20D"
        ],
    )

    base_mae_rel = safe_spearman(
        base_score,
        targets[
            "Fwd_MAE_20D"
        ],
    )

    ablation_rows = []

    for pillar, factor_weights in current_gold_cfg[
        "sub_weights"
    ].items():
        active_factors = [
            factor
            for factor, weight
            in factor_weights.items()
            if float(
                weight
            ) > 0
        ]

        # Avoid deleting the only driver of a pillar.
        if len(
            active_factors
        ) < 2:
            continue

        for factor in active_factors:
            candidate_cfg = deepcopy(
                current_gold_cfg
            )

            candidate_cfg[
                "sub_weights"
            ][
                pillar
            ][
                factor
            ] = 0.0

            candidate_frame = model_score_frame(
                norm_df,
                candidate_cfg,
            )

            candidate_score = pd.to_numeric(
                candidate_frame[
                    "Final_Regime_Score"
                ],
                errors="coerce",
            )

            pair_mask = (
                base_score.notna()
                & candidate_score.notna()
                & (
                    current_gold_frame[
                        "Model_Data_Coverage"
                    ]
                    >= float(
                        min_coverage
                    )
                )
                & (
                    candidate_frame[
                        "Model_Data_Coverage"
                    ]
                    >= float(
                        min_coverage
                    )
                )
            )

            pair_base = base_score.where(
                pair_mask
            )

            pair_candidate = candidate_score.where(
                pair_mask
            )

            pair_ic_base = safe_spearman(
                pair_base,
                targets[
                    "Fwd_Return_20D"
                ],
            )

            pair_ic_candidate = safe_spearman(
                pair_candidate,
                targets[
                    "Fwd_Return_20D"
                ],
            )

            pair_auc_base = binary_auc(
                pair_base,
                targets[
                    "Stress_Event_20D"
                ],
                higher_predictor_means_event=False,
            )

            pair_auc_candidate = binary_auc(
                pair_candidate,
                targets[
                    "Stress_Event_20D"
                ],
                higher_predictor_means_event=False,
            )

            pair_vol_base = safe_spearman(
                pair_base,
                -targets[
                    "Fwd_Realized_Vol_20D"
                ],
            )

            pair_vol_candidate = safe_spearman(
                pair_candidate,
                -targets[
                    "Fwd_Realized_Vol_20D"
                ],
            )

            pair_mae_base = safe_spearman(
                pair_base,
                targets[
                    "Fwd_MAE_20D"
                ],
            )

            pair_mae_candidate = safe_spearman(
                pair_candidate,
                targets[
                    "Fwd_MAE_20D"
                ],
            )

            delta_ic = (
                pair_ic_candidate
                - pair_ic_base
                if (
                    np.isfinite(
                        pair_ic_candidate
                    )
                    and np.isfinite(
                        pair_ic_base
                    )
                )
                else np.nan
            )

            harmful_direction_candidate = bool(
                np.isfinite(
                    delta_ic
                )
                and delta_ic
                >= GOLD_ABLATION_MIN_IC_IMPROVEMENT
            )

            ablation_rows.append(
                {
                    "Säule": pillar,
                    "Faktor entfernt": factor,
                    "Basisgewicht Faktor": float(
                        factor_weights[
                            factor
                        ]
                    ),
                    "N Common": int(
                        pair_mask.sum()
                    ),
                    "Base IC20": pair_ic_base,
                    "Ohne Faktor IC20": pair_ic_candidate,
                    "Δ IC20 ohne−base": delta_ic,
                    "Base Stress AUC": pair_auc_base,
                    "Ohne Faktor Stress AUC": pair_auc_candidate,
                    "Δ Stress AUC": (
                        pair_auc_candidate
                        - pair_auc_base
                        if (
                            np.isfinite(
                                pair_auc_candidate
                            )
                            and np.isfinite(
                                pair_auc_base
                            )
                        )
                        else np.nan
                    ),
                    "Δ FwdVol Relation": (
                        pair_vol_candidate
                        - pair_vol_base
                        if (
                            np.isfinite(
                                pair_vol_candidate
                            )
                            and np.isfinite(
                                pair_vol_base
                            )
                        )
                        else np.nan
                    ),
                    "Δ FwdMAE Relation": (
                        pair_mae_candidate
                        - pair_mae_base
                        if (
                            np.isfinite(
                                pair_mae_candidate
                            )
                            and np.isfinite(
                                pair_mae_base
                            )
                        )
                        else np.nan
                    ),
                    "Direction-Harm-Kandidat": harmful_direction_candidate,
                }
            )

    gold_driver_ablation_table = (
        pd.DataFrame(
            ablation_rows
        )
        .sort_values(
            "Δ IC20 ohne−base",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    st.dataframe(
        gold_driver_ablation_table.style.format(
            {
                "Basisgewicht Faktor": "{:.1%}",
                "N Common": "{:.0f}",
                "Base IC20": "{:+.3f}",
                "Ohne Faktor IC20": "{:+.3f}",
                "Δ IC20 ohne−base": "{:+.3f}",
                "Base Stress AUC": "{:.3f}",
                "Ohne Faktor Stress AUC": "{:.3f}",
                "Δ Stress AUC": "{:+.3f}",
                "Δ FwdVol Relation": "{:+.3f}",
                "Δ FwdMAE Relation": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Auch hier gilt: ein Faktor wird nicht entfernt. Die Ablation zeigt "
        "nur, ob seine Anwesenheit die Direction-Metrik des vollständigen "
        "Modells systematisch belastet und welche Risk-State-Eigenschaften "
        "dabei gleichzeitig verloren oder gewonnen würden."
    )

    # --------------------------------------------------------
    # E. MACRO STATE AUDIT
    # --------------------------------------------------------

    st.markdown(
        "### E. Gold-Makrozustände – Real Yields × USD"
    )

    real_yield_raw = pd.to_numeric(
        raw_df.get(
            "real_yields",
            pd.Series(
                np.nan,
                index=raw_df.index,
            ),
        ),
        errors="coerce",
    )

    usd_raw = pd.to_numeric(
        raw_df.get(
            "usd_index",
            pd.Series(
                np.nan,
                index=raw_df.index,
            ),
        ),
        errors="coerce",
    )

    real_yield_change = real_yield_raw.diff(
        20
    )

    usd_change = usd_raw.pct_change(
        20,
        fill_method=None,
    )

    macro_state = pd.Series(
        "Mixed",
        index=raw_df.index,
        dtype=object,
    )

    macro_state[
        (
            real_yield_change < 0
        )
        &
        (
            usd_change < 0
        )
    ] = "Tailwind: Real Yields↓ + USD↓"

    macro_state[
        (
            real_yield_change > 0
        )
        &
        (
            usd_change > 0
        )
    ] = "Headwind: Real Yields↑ + USD↑"

    macro_rows = []

    for state_name in [
        "Tailwind: Real Yields↓ + USD↓",
        "Mixed",
        "Headwind: Real Yields↑ + USD↑",
    ]:
        mask = (
            macro_state
            == state_name
        )

        sample_score = current_gold_score.where(
            mask
        )

        macro_rows.append(
            {
                "Makrozustand": state_name,
                "N": int(
                    (
                        mask
                        & targets[
                            "Fwd_Return_20D"
                        ].notna()
                    ).sum()
                ),
                "Ø Current Score": float(
                    current_gold_score.where(
                        mask
                    ).mean()
                ),
                "Ø Fwd Return 5D": float(
                    targets[
                        "Fwd_Return_5D"
                    ].where(
                        mask
                    ).mean()
                ),
                "Ø Fwd Return 20D": float(
                    targets[
                        "Fwd_Return_20D"
                    ].where(
                        mask
                    ).mean()
                ),
                "Ø Fwd Return 60D": float(
                    targets[
                        "Fwd_Return_60D"
                    ].where(
                        mask
                    ).mean()
                ),
                "Current IC20 im Zustand": safe_spearman(
                    sample_score,
                    targets[
                        "Fwd_Return_20D"
                    ],
                ),
                "Stressrate 20D": float(
                    targets[
                        "Stress_Event_20D"
                    ].where(
                        mask
                    ).mean()
                ),
            }
        )

    gold_macro_state_table = pd.DataFrame(
        macro_rows
    )

    st.dataframe(
        gold_macro_state_table.style.format(
            {
                "N": "{:.0f}",
                "Ø Current Score": "{:.1f}",
                "Ø Fwd Return 5D": "{:+.2%}",
                "Ø Fwd Return 20D": "{:+.2%}",
                "Ø Fwd Return 60D": "{:+.2%}",
                "Current IC20 im Zustand": "{:+.3f}",
                "Stressrate 20D": "{:.1%}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # F. SAFE-HAVEN STATE AUDIT
    # --------------------------------------------------------

    st.markdown(
        "### F. Safe-Haven-Zustände – Credit Proxy × MOVE"
    )

    credit_raw = pd.to_numeric(
        raw_df.get(
            "credit_spreads",
            pd.Series(
                np.nan,
                index=raw_df.index,
            ),
        ),
        errors="coerce",
    )

    move_raw = pd.to_numeric(
        raw_df.get(
            "move_index",
            pd.Series(
                np.nan,
                index=raw_df.index,
            ),
        ),
        errors="coerce",
    )

    credit_change = credit_raw.pct_change(
        20,
        fill_method=None,
    )

    move_change = move_raw.pct_change(
        20,
        fill_method=None,
    )

    haven_state = pd.Series(
        "Mixed",
        index=raw_df.index,
        dtype=object,
    )

    haven_state[
        (
            credit_change > 0
        )
        &
        (
            move_change > 0
        )
    ] = "Stress steigt: Credit↑ + MOVE↑"

    haven_state[
        (
            credit_change < 0
        )
        &
        (
            move_change < 0
        )
    ] = "Stress fällt: Credit↓ + MOVE↓"

    haven_rows = []

    for state_name in [
        "Stress steigt: Credit↑ + MOVE↑",
        "Mixed",
        "Stress fällt: Credit↓ + MOVE↓",
    ]:
        mask = (
            haven_state
            == state_name
        )

        haven_rows.append(
            {
                "Safe-Haven-Zustand": state_name,
                "N": int(
                    (
                        mask
                        & targets[
                            "Fwd_Return_20D"
                        ].notna()
                    ).sum()
                ),
                "Ø Current Score": float(
                    current_gold_score.where(
                        mask
                    ).mean()
                ),
                "Ø Fwd Return 20D": float(
                    targets[
                        "Fwd_Return_20D"
                    ].where(
                        mask
                    ).mean()
                ),
                "Ø FwdMAE 20D": float(
                    targets[
                        "Fwd_MAE_20D"
                    ].where(
                        mask
                    ).mean()
                ),
                "Ø FwdVol 20D": float(
                    targets[
                        "Fwd_Realized_Vol_20D"
                    ].where(
                        mask
                    ).mean()
                ),
                "Current IC20 im Zustand": safe_spearman(
                    current_gold_score.where(
                        mask
                    ),
                    targets[
                        "Fwd_Return_20D"
                    ],
                ),
            }
        )

    gold_safe_haven_state_table = pd.DataFrame(
        haven_rows
    )

    st.dataframe(
        gold_safe_haven_state_table.style.format(
            {
                "N": "{:.0f}",
                "Ø Current Score": "{:.1f}",
                "Ø Fwd Return 20D": "{:+.2%}",
                "Ø FwdMAE 20D": "{:+.2%}",
                "Ø FwdVol 20D": "{:.2%}",
                "Current IC20 im Zustand": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # G. STRUCTURAL DECISION GATES
    # --------------------------------------------------------

    st.markdown(
        "### G. Strukturelles Diagnoseurteil"
    )

    sign_conflict_count = int(
        gold_structural_sign_table[
            "Struktureller Sign-Konflikt"
        ].astype(
            bool
        ).sum()
    )

    transform_candidate_count = int(
        gold_level_change_table[
            "Transformations-Kandidat"
        ].astype(
            bool
        ).sum()
    )

    ablation_harm_count = int(
        gold_driver_ablation_table[
            "Direction-Harm-Kandidat"
        ].astype(
            bool
        ).sum()
    )

    # Carry forward the v1.0.19 role result. The current model is shadow-ready
    # only if there is no orientation warning and every Risk-State gate passed.
    current_risk_gate_count = int(
        len(
            risk_rows
        )
    )

    current_risk_pass_count = int(
        risk_rows[
            "Erfüllt"
        ].sum()
    )

    shadow_ready = bool(
        not orientation_flag
        and current_risk_gate_count > 0
        and current_risk_pass_count
        == current_risk_gate_count
        and sign_conflict_count == 0
    )

    gold_structural_gate_table = pd.DataFrame(
        [
            {
                "Prüfung": "Globaler Orientation-Hinweis aus v1.0.19",
                "Status": (
                    "⚠️ vorhanden"
                    if orientation_flag
                    else "✅ nein"
                ),
                "Messwert": (
                    f"Current IC20 {current_ic20:+.3f} · "
                    f"100−Current {inverted_ic20:+.3f}"
                ),
            },
            {
                "Prüfung": "Stabile Faktor-Sign-Konflikte",
                "Status": (
                    "⚠️ untersuchen"
                    if sign_conflict_count > 0
                    else "✅ keine"
                ),
                "Messwert": str(
                    sign_conflict_count
                ),
            },
            {
                "Prüfung": "Level→Change Transformations-Kandidaten",
                "Status": (
                    "🧪 separat testen"
                    if transform_candidate_count > 0
                    else "— keine"
                ),
                "Messwert": str(
                    transform_candidate_count
                ),
            },
            {
                "Prüfung": "LODO Direction-Harm-Kandidaten",
                "Status": (
                    "⚠️ untersuchen"
                    if ablation_harm_count > 0
                    else "✅ keine"
                ),
                "Messwert": str(
                    ablation_harm_count
                ),
            },
            {
                "Prüfung": "Current Risk-State Gates aus v1.0.19",
                "Status": (
                    "✅ vollständig"
                    if (
                        current_risk_gate_count > 0
                        and current_risk_pass_count
                        == current_risk_gate_count
                    )
                    else "❌ nicht vollständig"
                ),
                "Messwert": (
                    f"{current_risk_pass_count}/"
                    f"{current_risk_gate_count}"
                ),
            },
            {
                "Prüfung": "Current direkt für Future-Shadow freigeben",
                "Status": (
                    "✅ JA"
                    if shadow_ready
                    else "❌ NEIN"
                ),
                "Messwert": (
                    "nur bei stabiler Current-Struktur"
                ),
            },
        ]
    )

    st.dataframe(
        gold_structural_gate_table,
        hide_index=True,
        use_container_width=True,
    )

    if shadow_ready:
        st.success(
            "🟢 **CURRENT WÄRE STRUKTURELL SHADOW-TAUGLICH.** "
            "Das wäre nur dann der Fall, wenn weder globale noch faktorielle "
            "Orientierungskonflikte bestehen und alle Gold-Risk-State-Gates "
            "getragen werden."
        )
    elif (
        sign_conflict_count > 0
        or transform_candidate_count > 0
        or ablation_harm_count > 0
    ):
        st.warning(
            "🟠 **NOCH KEIN GOLD-SHADOW.** Der Audit identifiziert konkrete "
            "strukturelle Kandidaten. Der nächste Schritt wäre ein **kleiner, "
            "vorab definierter Gold-Challenger**, der ausschließlich diese "
            "diagnostizierten Strukturfragen korrigiert – keine freie "
            "Gewichtsoptimierung."
        )
    else:
        st.error(
            "🔴 **CURRENT IST NICHT SHADOW-TAUGLICH, ABER DER AUDIT FINDET "
            "KEINE EINFACHE EINZELURSACHE.** Dann sollte die Gold-Architektur "
            "als Mehr-Regime-Modell neu konzipiert werden, statt einzelne "
            "Faktorzeichen nachträglich zu drehen."
        )

    st.info(
        "Research-Regel v1.0.20: Sign-Flip, Change-Transformation oder "
        "Faktorentfernung werden **nicht** aus dieser Auswertung heraus "
        "produktiv gemacht. Ein daraus abgeleiteter Challenger muss separat "
        "festgeschrieben und anschließend historisch sowie future-shadow "
        "validiert werden."
    )



# ============================================================
# 19G3. GOLD G1 STRUCTURAL CHALLENGER
# ============================================================

gold_g1_model_table = pd.DataFrame()
gold_g1_macro_table = pd.DataFrame()
gold_g1_fixed_period_table = pd.DataFrame()
gold_g1_yearly_table = pd.DataFrame()
gold_g1_nonoverlap_table = pd.DataFrame()
gold_g1_bootstrap_table = pd.DataFrame()
gold_g1_quintile_table = pd.DataFrame()
gold_g1_gate_table = pd.DataFrame()
gold_g1_research_export = pd.DataFrame()

if selected_asset == "Gold (XAU/USD)":
    st.markdown("---")
    st.subheader(
        "🧪 Gold G1 Structural Challenger – Current vs. G1"
    )

    st.caption(
        "G1 ist **vollständig vorab festgelegt**. Es werden keine Gewichte "
        "gesucht: Die Gold-Makrosäule bleibt bei 35 %. Innerhalb dieser "
        "Säule werden nur Fed Policy, Real Yields und USD in ihre "
        "vorab definierte günstige 20D-Veränderung transformiert; "
        "Net Liquidity wird aus G1-Makro entfernt. Alle anderen Säulen "
        "und Faktoren sind exakt Current."
    )

    (
        gold_g1_norm_df,
        gold_g1_cfg,
        gold_g1_frame,
        gold_g1_transformed,
    ) = build_gold_g1_challenger(
        raw_df,
        norm_df,
    )

    gold_current_frame = model_frames[
        MODEL_CURRENT
    ]

    gold_current_score = pd.to_numeric(
        gold_current_frame[
            "Final_Regime_Score"
        ],
        errors="coerce",
    )

    gold_g1_score = pd.to_numeric(
        gold_g1_frame[
            "Final_Regime_Score"
        ],
        errors="coerce",
    )

    gold_g1_common = (
        gold_current_score.notna()
        & gold_g1_score.notna()
        & (
            pd.to_numeric(
                gold_current_frame[
                    "Model_Data_Coverage"
                ],
                errors="coerce",
            )
            >= float(
                min_coverage
            )
        )
        & (
            pd.to_numeric(
                gold_g1_frame[
                    "Model_Data_Coverage"
                ],
                errors="coerce",
            )
            >= float(
                min_coverage
            )
        )
    )

    # --------------------------------------------------------
    # 1. FROZEN SPECIFICATION SNAPSHOT
    # --------------------------------------------------------

    st.markdown(
        "### 1. Eingefrorene G1-Spezifikation"
    )

    current_macro = current_model_config(
        "Gold (XAU/USD)"
    )[
        "sub_weights"
    ][
        "Makroökonomie"
    ]

    g1_macro = gold_g1_cfg[
        "sub_weights"
    ][
        "Makroökonomie"
    ]

    spec_table = pd.DataFrame(
        [
            {
                "Treiber": "Fed Policy",
                "Current": (
                    f"Level/Perzentil · Gewicht "
                    f"{current_macro.get('fed_policy', 0.0):.1%}"
                ),
                "G1": (
                    f"-Δ20 Fed Policy · Gewicht "
                    f"{g1_macro.get('fed_policy', 0.0):.1%}"
                ),
            },
            {
                "Treiber": "Real Yields",
                "Current": (
                    f"Level/Perzentil · Gewicht "
                    f"{current_macro.get('real_yields', 0.0):.1%}"
                ),
                "G1": (
                    f"-Δ20 Real Yields · Gewicht "
                    f"{g1_macro.get('real_yields', 0.0):.1%}"
                ),
            },
            {
                "Treiber": "USD Index",
                "Current": (
                    f"Level/Perzentil · Gewicht "
                    f"{current_macro.get('usd_index', 0.0):.1%}"
                ),
                "G1": (
                    f"-20D-%Δ USD · Gewicht "
                    f"{g1_macro.get('usd_index', 0.0):.1%}"
                ),
            },
            {
                "Treiber": "Net Liquidity",
                "Current": (
                    f"Level/Perzentil · Gewicht "
                    f"{current_macro.get('net_liquidity', 0.0):.1%}"
                ),
                "G1": "entfernt aus Makro · Gewicht 0.0%",
            },
        ]
    )

    st.dataframe(
        spec_table,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Makro-Säulengewicht selbst bleibt unverändert. Die verbleibenden "
        "0.20 / 0.30 / 0.20 werden ausschließlich innerhalb Makro auf "
        "28.57 % / 42.86 % / 28.57 % renormalisiert."
    )

    # --------------------------------------------------------
    # 2. FULL MODEL COMPARISON
    # --------------------------------------------------------

    st.markdown(
        "### 2. Current vs. G1 – vollständiges Modell"
    )

    model_rows = []

    for label, frame, score in [
        (
            "A · Current",
            gold_current_frame,
            gold_current_score,
        ),
        (
            "G1 · Structural Challenger",
            gold_g1_frame,
            gold_g1_score,
        ),
    ]:
        paired_score = score.where(
            gold_g1_common
        )

        direction20, signals20 = directional_accuracy(
            paired_score,
            targets[
                "Fwd_Return_20D"
            ],
        )

        phase_auc, phase_count = phase_median_stress_auc(
            paired_score,
            targets[
                "Stress_Event_20D"
            ],
            horizon=20,
        )

        model_rows.append(
            {
                "Modell": label,
                "N Common": int(
                    gold_g1_common.sum()
                ),
                "IC 5D": safe_spearman(
                    paired_score,
                    targets[
                        "Fwd_Return_5D"
                    ],
                ),
                "IC 20D": safe_spearman(
                    paired_score,
                    targets[
                        "Fwd_Return_20D"
                    ],
                ),
                "IC 60D": safe_spearman(
                    paired_score,
                    targets[
                        "Fwd_Return_60D"
                    ],
                ),
                "Direction 20D": direction20,
                "Signals 20D": signals20,
                "Stress AUC 20D": binary_auc(
                    paired_score,
                    targets[
                        "Stress_Event_20D"
                    ],
                    higher_predictor_means_event=False,
                ),
                "Phase-Median Stress AUC": phase_auc,
                "AUC-Phasen": phase_count,
                "Score vs niedrigere FwdVol": safe_spearman(
                    paired_score,
                    -targets[
                        "Fwd_Realized_Vol_20D"
                    ],
                ),
                "Score vs bessere FwdMAE": safe_spearman(
                    paired_score,
                    targets[
                        "Fwd_MAE_20D"
                    ],
                ),
                "Ø Coverage": float(
                    pd.to_numeric(
                        frame[
                            "Model_Data_Coverage"
                        ],
                        errors="coerce",
                    )
                    .where(
                        gold_g1_common
                    )
                    .mean()
                ),
            }
        )

    gold_g1_model_table = pd.DataFrame(
        model_rows
    )

    st.dataframe(
        gold_g1_model_table.style.format(
            {
                "N Common": "{:.0f}",
                "IC 5D": "{:+.3f}",
                "IC 20D": "{:+.3f}",
                "IC 60D": "{:+.3f}",
                "Direction 20D": "{:.1%}",
                "Signals 20D": "{:.0f}",
                "Stress AUC 20D": "{:.3f}",
                "Phase-Median Stress AUC": "{:.3f}",
                "AUC-Phasen": "{:.0f}",
                "Score vs niedrigere FwdVol": "{:+.3f}",
                "Score vs bessere FwdMAE": "{:+.3f}",
                "Ø Coverage": "{:.1f}%",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    current_metrics = gold_g1_model_table[
        gold_g1_model_table[
            "Modell"
        ]
        == "A · Current"
    ].iloc[
        0
    ]

    g1_metrics = gold_g1_model_table[
        gold_g1_model_table[
            "Modell"
        ]
        == "G1 · Structural Challenger"
    ].iloc[
        0
    ]

    # --------------------------------------------------------
    # 3. MACRO PILLAR ISOLATION
    # --------------------------------------------------------

    st.markdown(
        "### 3. Isolierte Makrosäule – wurde der eigentliche Fehler repariert?"
    )

    current_macro_score = pd.to_numeric(
        gold_current_frame[
            "Pillar::Makroökonomie"
        ],
        errors="coerce",
    ).where(
        gold_g1_common
    )

    g1_macro_score = pd.to_numeric(
        gold_g1_frame[
            "Pillar::Makroökonomie"
        ],
        errors="coerce",
    ).where(
        gold_g1_common
    )

    macro_rows = []

    for label, score in [
        (
            "Current Macro",
            current_macro_score,
        ),
        (
            "G1 Macro",
            g1_macro_score,
        ),
    ]:
        macro_nonoverlap, macro_phases = phase_median_nonoverlap_ic(
            score,
            targets[
                "Fwd_Return_20D"
            ],
            20,
        )

        macro_rows.append(
            {
                "Makro": label,
                "IC 5D": safe_spearman(
                    score,
                    targets[
                        "Fwd_Return_5D"
                    ],
                ),
                "IC 20D": safe_spearman(
                    score,
                    targets[
                        "Fwd_Return_20D"
                    ],
                ),
                "IC 60D": safe_spearman(
                    score,
                    targets[
                        "Fwd_Return_60D"
                    ],
                ),
                "Non-Overlap IC20": macro_nonoverlap,
                "Non-Overlap Phasen": macro_phases,
                "Stress AUC": binary_auc(
                    score,
                    targets[
                        "Stress_Event_20D"
                    ],
                    higher_predictor_means_event=False,
                ),
                "vs niedrigere FwdVol": safe_spearman(
                    score,
                    -targets[
                        "Fwd_Realized_Vol_20D"
                    ],
                ),
                "vs bessere FwdMAE": safe_spearman(
                    score,
                    targets[
                        "Fwd_MAE_20D"
                    ],
                ),
            }
        )

    gold_g1_macro_table = pd.DataFrame(
        macro_rows
    )

    st.dataframe(
        gold_g1_macro_table.style.format(
            {
                "IC 5D": "{:+.3f}",
                "IC 20D": "{:+.3f}",
                "IC 60D": "{:+.3f}",
                "Non-Overlap IC20": "{:+.3f}",
                "Non-Overlap Phasen": "{:.0f}",
                "Stress AUC": "{:.3f}",
                "vs niedrigere FwdVol": "{:+.3f}",
                "vs bessere FwdMAE": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 4. FIXED PERIOD ROBUSTNESS
    # --------------------------------------------------------

    st.markdown(
        "### 4. Feste Teilperioden – 2012–16 / 2017–20 / 2021–25"
    )

    fixed_rows = []

    for period_label, y1, y2 in GOLD_FIXED_PERIODS:
        mask = (
            gold_g1_common
            & (
                targets.index.year
                >= y1
            )
            & (
                targets.index.year
                <= y2
            )
        )

        current_score_p = gold_current_score.where(
            mask
        )

        g1_score_p = gold_g1_score.where(
            mask
        )

        current_dir, current_signals = directional_accuracy(
            current_score_p,
            targets[
                "Fwd_Return_20D"
            ],
        )

        g1_dir, g1_signals = directional_accuracy(
            g1_score_p,
            targets[
                "Fwd_Return_20D"
            ],
        )

        current_ic = safe_spearman(
            current_score_p,
            targets[
                "Fwd_Return_20D"
            ],
        )

        g1_ic = safe_spearman(
            g1_score_p,
            targets[
                "Fwd_Return_20D"
            ],
        )

        fixed_rows.append(
            {
                "Periode": period_label,
                "N": int(
                    mask.sum()
                ),
                "Current IC20": current_ic,
                "G1 IC20": g1_ic,
                "Δ IC20 G1−A": (
                    g1_ic
                    - current_ic
                    if (
                        np.isfinite(
                            g1_ic
                        )
                        and np.isfinite(
                            current_ic
                        )
                    )
                    else np.nan
                ),
                "Current Direction20": current_dir,
                "G1 Direction20": g1_dir,
                "Current Signals": current_signals,
                "G1 Signals": g1_signals,
                "G1 IC20 positiv": bool(
                    np.isfinite(
                        g1_ic
                    )
                    and g1_ic > 0
                ),
                "G1 schlägt A": bool(
                    np.isfinite(
                        g1_ic
                    )
                    and np.isfinite(
                        current_ic
                    )
                    and g1_ic
                    > current_ic
                ),
            }
        )

    gold_g1_fixed_period_table = pd.DataFrame(
        fixed_rows
    )

    st.dataframe(
        gold_g1_fixed_period_table.style.format(
            {
                "N": "{:.0f}",
                "Current IC20": "{:+.3f}",
                "G1 IC20": "{:+.3f}",
                "Δ IC20 G1−A": "{:+.3f}",
                "Current Direction20": "{:.1%}",
                "G1 Direction20": "{:.1%}",
                "Current Signals": "{:.0f}",
                "G1 Signals": "{:.0f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 5. YEAR-BY-YEAR STABILITY
    # --------------------------------------------------------

    st.markdown(
        "### 5. Kalenderjahr-Stabilität"
    )

    yearly_rows = []

    current_year = int(
        pd.Timestamp.now().year
    )

    for year in sorted(
        set(
            targets.index.year
        )
    ):
        if (
            year < 2012
            or year >= current_year
        ):
            continue

        mask = (
            gold_g1_common
            & (
                targets.index.year
                == int(
                    year
                )
            )
        )

        if int(
            mask.sum()
        ) < 60:
            continue

        current_ic = safe_spearman(
            gold_current_score.where(
                mask
            ),
            targets[
                "Fwd_Return_20D"
            ],
        )

        g1_ic = safe_spearman(
            gold_g1_score.where(
                mask
            ),
            targets[
                "Fwd_Return_20D"
            ],
        )

        yearly_rows.append(
            {
                "Jahr": int(
                    year
                ),
                "N": int(
                    mask.sum()
                ),
                "Current IC20": current_ic,
                "G1 IC20": g1_ic,
                "Δ IC20 G1−A": (
                    g1_ic
                    - current_ic
                    if (
                        np.isfinite(
                            g1_ic
                        )
                        and np.isfinite(
                            current_ic
                        )
                    )
                    else np.nan
                ),
                "G1 positiv": bool(
                    np.isfinite(
                        g1_ic
                    )
                    and g1_ic > 0
                ),
                "G1 schlägt A": bool(
                    np.isfinite(
                        g1_ic
                    )
                    and np.isfinite(
                        current_ic
                    )
                    and g1_ic > current_ic
                ),
            }
        )

    gold_g1_yearly_table = pd.DataFrame(
        yearly_rows
    )

    st.dataframe(
        gold_g1_yearly_table.style.format(
            {
                "N": "{:.0f}",
                "Current IC20": "{:+.3f}",
                "G1 IC20": "{:+.3f}",
                "Δ IC20 G1−A": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 6. NON-OVERLAP
    # --------------------------------------------------------

    st.markdown(
        "### 6. Non-Overlap – IC20 / IC60"
    )

    nonoverlap_rows = []

    for horizon in [
        20,
        60,
    ]:
        for label, score in [
            (
                "A · Current",
                gold_current_score.where(
                    gold_g1_common
                ),
            ),
            (
                "G1 · Structural Challenger",
                gold_g1_score.where(
                    gold_g1_common
                ),
            ),
        ]:
            median_ic, phases = phase_median_nonoverlap_ic(
                score,
                targets[
                    f"Fwd_Return_{horizon}D"
                ],
                horizon,
            )

            nonoverlap_rows.append(
                {
                    "Horizont": f"{horizon}D",
                    "Modell": label,
                    "Median Non-Overlap IC": median_ic,
                    "Phasen": phases,
                }
            )

    gold_g1_nonoverlap_table = pd.DataFrame(
        nonoverlap_rows
    )

    st.dataframe(
        gold_g1_nonoverlap_table.style.format(
            {
                "Median Non-Overlap IC": "{:+.3f}",
                "Phasen": "{:.0f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 7. BLOCK BOOTSTRAP
    # --------------------------------------------------------

    st.markdown(
        "### 7. 20D-Block-Bootstrap – G1 IC20 minus Current"
    )

    g1_boot = block_bootstrap_ic_difference(
        gold_current_score.where(
            gold_g1_common
        ),
        gold_g1_score.where(
            gold_g1_common
        ),
        targets[
            "Fwd_Return_20D"
        ].where(
            gold_g1_common
        ),
        block_length=int(
            bootstrap_block
        ),
        n_boot=int(
            bootstrap_runs
        ),
        seed=20260912,
    )

    gold_g1_bootstrap_table = pd.DataFrame(
        [
            {
                "Vergleich": "G1 − Current IC20",
                "Δ IC20": g1_boot[
                    "observed"
                ],
                "P(G1>A)": g1_boot[
                    "prob_positive"
                ],
                "95% CI Low": g1_boot[
                    "lower"
                ],
                "95% CI High": g1_boot[
                    "upper"
                ],
                "N": g1_boot[
                    "n"
                ],
            }
        ]
    )

    st.dataframe(
        gold_g1_bootstrap_table.style.format(
            {
                "Δ IC20": "{:+.3f}",
                "P(G1>A)": "{:.1%}",
                "95% CI Low": "{:+.3f}",
                "95% CI High": "{:+.3f}",
                "N": "{:.0f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 8. QUINTILE MONOTONICITY
    # --------------------------------------------------------

    st.markdown(
        "### 8. Quintile – steigt Forward-Return mit dem G1-Score?"
    )

    quintile_rows = []

    for label, score in [
        (
            "A · Current",
            gold_current_score.where(
                gold_g1_common
            ),
        ),
        (
            "G1 · Structural Challenger",
            gold_g1_score.where(
                gold_g1_common
            ),
        ),
    ]:
        q_table, spread, monotonic = quintile_statistics(
            score,
            targets[
                "Fwd_Return_20D"
            ],
        )

        quintile_rows.append(
            {
                "Modell": label,
                "Q5−Q1 Mean Return Spread": spread,
                "Quintile-Monotonie": monotonic,
                "Q1 Mean": (
                    float(
                        q_table.loc[
                            "Q1",
                            "mean"
                        ]
                    )
                    if (
                        not q_table.empty
                        and "Q1" in q_table.index
                    )
                    else np.nan
                ),
                "Q5 Mean": (
                    float(
                        q_table.loc[
                            "Q5",
                            "mean"
                        ]
                    )
                    if (
                        not q_table.empty
                        and "Q5" in q_table.index
                    )
                    else np.nan
                ),
            }
        )

    gold_g1_quintile_table = pd.DataFrame(
        quintile_rows
    )

    st.dataframe(
        gold_g1_quintile_table.style.format(
            {
                "Q5−Q1 Mean Return Spread": "{:+.2%}",
                "Quintile-Monotonie": "{:+.3f}",
                "Q1 Mean": "{:+.2%}",
                "Q5 Mean": "{:+.2%}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 9. PRE-REGISTERED GATES
    # --------------------------------------------------------

    st.markdown(
        "### 9. Vorab festgelegte G1-Gates"
    )

    current_ic20 = float(
        current_metrics[
            "IC 20D"
        ]
    )

    g1_ic20 = float(
        g1_metrics[
            "IC 20D"
        ]
    )

    current_direction = float(
        current_metrics[
            "Direction 20D"
        ]
    )

    g1_direction = float(
        g1_metrics[
            "Direction 20D"
        ]
    )

    delta_ic20 = (
        g1_ic20
        - current_ic20
        if (
            np.isfinite(
                g1_ic20
            )
            and np.isfinite(
                current_ic20
            )
        )
        else np.nan
    )

    delta_direction = (
        g1_direction
        - current_direction
        if (
            np.isfinite(
                g1_direction
            )
            and np.isfinite(
                current_direction
            )
        )
        else np.nan
    )

    g1_nonoverlap20_row = gold_g1_nonoverlap_table[
        (
            gold_g1_nonoverlap_table[
                "Horizont"
            ]
            == "20D"
        )
        &
        (
            gold_g1_nonoverlap_table[
                "Modell"
            ]
            == "G1 · Structural Challenger"
        )
    ]

    g1_nonoverlap20 = (
        float(
            g1_nonoverlap20_row.iloc[
                0
            ][
                "Median Non-Overlap IC"
            ]
        )
        if not g1_nonoverlap20_row.empty
        else np.nan
    )

    positive_fixed_periods = int(
        gold_g1_fixed_period_table[
            "G1 IC20 positiv"
        ]
        .astype(
            bool
        )
        .sum()
    )

    evaluable_years = gold_g1_yearly_table[
        gold_g1_yearly_table[
            "G1 IC20"
        ].notna()
        &
        gold_g1_yearly_table[
            "Current IC20"
        ].notna()
    ]

    year_count = int(
        len(
            evaluable_years
        )
    )

    year_win_rate = (
        float(
            evaluable_years[
                "G1 schlägt A"
            ]
            .astype(
                bool
            )
            .mean()
        )
        if year_count > 0
        else np.nan
    )

    positive_year_rate = (
        float(
            evaluable_years[
                "G1 positiv"
            ]
            .astype(
                bool
            )
            .mean()
        )
        if year_count > 0
        else np.nan
    )

    current_auc = float(
        current_metrics[
            "Stress AUC 20D"
        ]
    )

    g1_auc = float(
        g1_metrics[
            "Stress AUC 20D"
        ]
    )

    current_phase_auc = float(
        current_metrics[
            "Phase-Median Stress AUC"
        ]
    )

    g1_phase_auc = float(
        g1_metrics[
            "Phase-Median Stress AUC"
        ]
    )

    current_vol_rel = float(
        current_metrics[
            "Score vs niedrigere FwdVol"
        ]
    )

    g1_vol_rel = float(
        g1_metrics[
            "Score vs niedrigere FwdVol"
        ]
    )

    current_mae_rel = float(
        current_metrics[
            "Score vs bessere FwdMAE"
        ]
    )

    g1_mae_rel = float(
        g1_metrics[
            "Score vs bessere FwdMAE"
        ]
    )

    gate_rows = [
        {
            "Gruppe": "Direction",
            "Kriterium": "G1 IC20 > 0",
            "Erfüllt": bool(
                np.isfinite(
                    g1_ic20
                )
                and g1_ic20
                > GOLD_G1_MIN_IC20
            ),
            "Messwert": (
                f"{g1_ic20:+.3f}"
                if np.isfinite(
                    g1_ic20
                )
                else "n/a"
            ),
        },
        {
            "Gruppe": "Direction",
            "Kriterium": "Δ IC20 G1−Current ≥ +0.10",
            "Erfüllt": bool(
                np.isfinite(
                    delta_ic20
                )
                and delta_ic20
                >= GOLD_G1_MIN_DELTA_IC20
            ),
            "Messwert": (
                f"{delta_ic20:+.3f}"
                if np.isfinite(
                    delta_ic20
                )
                else "n/a"
            ),
        },
        {
            "Gruppe": "Direction",
            "Kriterium": "G1 Non-Overlap IC20 > 0",
            "Erfüllt": bool(
                np.isfinite(
                    g1_nonoverlap20
                )
                and g1_nonoverlap20 > 0
            ),
            "Messwert": (
                f"{g1_nonoverlap20:+.3f}"
                if np.isfinite(
                    g1_nonoverlap20
                )
                else "n/a"
            ),
        },
        {
            "Gruppe": "Direction",
            "Kriterium": (
                "G1 Direction20 ≥50% UND Verbesserung ≥5 Prozentpunkte"
            ),
            "Erfüllt": bool(
                np.isfinite(
                    g1_direction
                )
                and np.isfinite(
                    delta_direction
                )
                and g1_direction
                >= GOLD_G1_MIN_DIRECTION20
                and delta_direction
                >= GOLD_G1_MIN_DIRECTION_IMPROVEMENT
            ),
            "Messwert": (
                f"G1 {g1_direction:.1%} · Δ {delta_direction:+.1%}"
                if (
                    np.isfinite(
                        g1_direction
                    )
                    and np.isfinite(
                        delta_direction
                    )
                )
                else "n/a"
            ),
        },
        {
            "Gruppe": "Stabilität",
            "Kriterium": "G1 IC20 positiv in mindestens 2/3 festen Perioden",
            "Erfüllt": bool(
                positive_fixed_periods
                >= GOLD_G1_MIN_POSITIVE_FIXED_PERIODS
            ),
            "Messwert": (
                f"{positive_fixed_periods}/3"
            ),
        },
        {
            "Gruppe": "Stabilität",
            "Kriterium": "G1 schlägt Current in ≥60% auswertbarer Jahre",
            "Erfüllt": bool(
                np.isfinite(
                    year_win_rate
                )
                and year_win_rate
                >= GOLD_G1_MIN_YEAR_WIN_RATE
            ),
            "Messwert": (
                f"{year_win_rate:.1%} · N={year_count}"
                if np.isfinite(
                    year_win_rate
                )
                else "n/a"
            ),
        },
        {
            "Gruppe": "Stabilität",
            "Kriterium": "G1 IC20 positiv in ≥50% auswertbarer Jahre",
            "Erfüllt": bool(
                np.isfinite(
                    positive_year_rate
                )
                and positive_year_rate
                >= GOLD_G1_MIN_POSITIVE_YEAR_RATE
            ),
            "Messwert": (
                f"{positive_year_rate:.1%} · N={year_count}"
                if np.isfinite(
                    positive_year_rate
                )
                else "n/a"
            ),
        },
        {
            "Gruppe": "Stabilität",
            "Kriterium": "Block-Bootstrap P(G1>Current) ≥90%",
            "Erfüllt": bool(
                np.isfinite(
                    g1_boot[
                        "prob_positive"
                    ]
                )
                and g1_boot[
                    "prob_positive"
                ]
                >= GOLD_G1_MIN_BOOTSTRAP_PROB
            ),
            "Messwert": (
                f"{g1_boot['prob_positive']:.1%}"
                if np.isfinite(
                    g1_boot[
                        "prob_positive"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "Gruppe": "Risk Guardrail",
            "Kriterium": "Stress-AUC nicht >0.03 schlechter als Current",
            "Erfüllt": bool(
                np.isfinite(
                    g1_auc
                )
                and np.isfinite(
                    current_auc
                )
                and g1_auc
                >= (
                    current_auc
                    - GOLD_G1_MAX_STRESS_AUC_DEGRADATION
                )
            ),
            "Messwert": (
                f"A {current_auc:.3f} · G1 {g1_auc:.3f} · "
                f"Δ {g1_auc-current_auc:+.3f}"
                if (
                    np.isfinite(
                        g1_auc
                    )
                    and np.isfinite(
                        current_auc
                    )
                )
                else "n/a"
            ),
        },
        {
            "Gruppe": "Risk Guardrail",
            "Kriterium": "Phase-AUC nicht >0.03 schlechter als Current",
            "Erfüllt": bool(
                np.isfinite(
                    g1_phase_auc
                )
                and np.isfinite(
                    current_phase_auc
                )
                and g1_phase_auc
                >= (
                    current_phase_auc
                    - GOLD_G1_MAX_PHASE_AUC_DEGRADATION
                )
            ),
            "Messwert": (
                f"A {current_phase_auc:.3f} · G1 {g1_phase_auc:.3f} · "
                f"Δ {g1_phase_auc-current_phase_auc:+.3f}"
                if (
                    np.isfinite(
                        g1_phase_auc
                    )
                    and np.isfinite(
                        current_phase_auc
                    )
                )
                else "n/a"
            ),
        },
        {
            "Gruppe": "Risk Guardrail",
            "Kriterium": "FwdVol-Bezug nicht >0.05 schlechter als Current",
            "Erfüllt": bool(
                np.isfinite(
                    g1_vol_rel
                )
                and np.isfinite(
                    current_vol_rel
                )
                and g1_vol_rel
                >= (
                    current_vol_rel
                    - GOLD_G1_MAX_VOL_REL_DEGRADATION
                )
            ),
            "Messwert": (
                f"A {current_vol_rel:+.3f} · G1 {g1_vol_rel:+.3f} · "
                f"Δ {g1_vol_rel-current_vol_rel:+.3f}"
                if (
                    np.isfinite(
                        g1_vol_rel
                    )
                    and np.isfinite(
                        current_vol_rel
                    )
                )
                else "n/a"
            ),
        },
        {
            "Gruppe": "Risk Guardrail",
            "Kriterium": "FwdMAE-Bezug nicht >0.05 schlechter als Current",
            "Erfüllt": bool(
                np.isfinite(
                    g1_mae_rel
                )
                and np.isfinite(
                    current_mae_rel
                )
                and g1_mae_rel
                >= (
                    current_mae_rel
                    - GOLD_G1_MAX_MAE_REL_DEGRADATION
                )
            ),
            "Messwert": (
                f"A {current_mae_rel:+.3f} · G1 {g1_mae_rel:+.3f} · "
                f"Δ {g1_mae_rel-current_mae_rel:+.3f}"
                if (
                    np.isfinite(
                        g1_mae_rel
                    )
                    and np.isfinite(
                        current_mae_rel
                    )
                )
                else "n/a"
            ),
        },
    ]

    gold_g1_gate_table = pd.DataFrame(
        gate_rows
    )

    gold_g1_gate_table[
        "Status"
    ] = gold_g1_gate_table[
        "Erfüllt"
    ].map(
        {
            True: "✅",
            False: "❌",
        }
    )

    st.dataframe(
        gold_g1_gate_table[
            [
                "Gruppe",
                "Kriterium",
                "Status",
                "Messwert",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    direction_gates = gold_g1_gate_table[
        gold_g1_gate_table[
            "Gruppe"
        ]
        == "Direction"
    ]

    stability_gates = gold_g1_gate_table[
        gold_g1_gate_table[
            "Gruppe"
        ]
        == "Stabilität"
    ]

    risk_gates = gold_g1_gate_table[
        gold_g1_gate_table[
            "Gruppe"
        ]
        == "Risk Guardrail"
    ]

    direction_pass = int(
        direction_gates[
            "Erfüllt"
        ].sum()
    )

    stability_pass = int(
        stability_gates[
            "Erfüllt"
        ].sum()
    )

    risk_pass = int(
        risk_gates[
            "Erfüllt"
        ].sum()
    )

    st.markdown(
        "### 10. G1-Urteil"
    )

    g1_full_pass = bool(
        direction_pass
        == len(
            direction_gates
        )
        and stability_pass
        == len(
            stability_gates
        )
        and risk_pass
        == len(
            risk_gates
        )
    )

    if g1_full_pass:
        st.success(
            "🟢 **G1 BESTEHT DEN HISTORISCHEN STRUCTURAL-CHALLENGER-TEST.** "
            "Die gezielte Makro-Reparatur verbessert Direction materiell "
            "und stabil, ohne die Risk-State-Eigenschaften unzulässig zu "
            "verschlechtern. Nächster Schritt wäre ein **eingefrorener "
            "G1-vs-Current Walk-Forward**, nicht sofort Produktion."
        )
    elif (
        direction_pass
        == len(
            direction_gates
        )
        and stability_pass
        >= max(
            1,
            len(
                stability_gates
            )
            - 1
        )
    ):
        st.warning(
            f"🟡 **G1 REPARIERT DIRECTION, ABER DIE ROBUSTHEIT IST NOCH "
            f"NICHT VOLLSTÄNDIG ({stability_pass}/{len(stability_gates)} "
            f"Stabilitäts-Gates; {risk_pass}/{len(risk_gates)} Risk-Gates).** "
            "Dann wird die einzelne schwache Prüfung untersucht, ohne G1 "
            "weiter auf die Historie zu optimieren."
        )
    else:
        st.error(
            f"🔴 **G1 REICHT NICHT AUS.** "
            f"Direction {direction_pass}/{len(direction_gates)}, "
            f"Stabilität {stability_pass}/{len(stability_gates)}, "
            f"Risk-Guardrails {risk_pass}/{len(risk_gates)}. "
            "Dann verwerfen wir die einfache G1-Hypothese und bauen nicht "
            "durch nachträgliches Feintuning weiter."
        )

    st.caption(
        "Methodischer Hinweis: G1 wurde aus der v1.0.20-Strukturdiagnose "
        "abgeleitet. Selbst ein vollständiges Bestehen ist daher noch kein "
        "echter unseen Holdout. Der nächste zulässige Schritt wäre ein "
        "eingefrorener Walk-Forward und danach – nur bei Bestätigung – "
        "ein Future-Shadow."
    )

    # --------------------------------------------------------
    # 11. EXPORT FRAME
    # --------------------------------------------------------

    gold_g1_research_export = pd.DataFrame(
        index=raw_df.index
    )

    gold_g1_research_export[
        "asset_price"
    ] = pd.to_numeric(
        raw_df[
            "asset_price"
        ],
        errors="coerce",
    )

    gold_g1_research_export[
        "current_score"
    ] = gold_current_score

    gold_g1_research_export[
        "g1_score"
    ] = gold_g1_score

    gold_g1_research_export[
        "current_coverage"
    ] = pd.to_numeric(
        gold_current_frame[
            "Model_Data_Coverage"
        ],
        errors="coerce",
    )

    gold_g1_research_export[
        "g1_coverage"
    ] = pd.to_numeric(
        gold_g1_frame[
            "Model_Data_Coverage"
        ],
        errors="coerce",
    )

    gold_g1_research_export[
        "current_macro_score"
    ] = current_macro_score

    gold_g1_research_export[
        "g1_macro_score"
    ] = g1_macro_score

    for col in gold_g1_transformed.columns:
        gold_g1_research_export[
            col
        ] = gold_g1_transformed[
            col
        ]

    for col in [
        "Fwd_Return_5D",
        "Fwd_Return_20D",
        "Fwd_Return_60D",
        "Fwd_MAE_20D",
        "Fwd_Realized_Vol_20D",
        "Stress_Event_20D",
    ]:
        gold_g1_research_export[
            col
        ] = targets[
            col
        ]

    gold_g1_research_export[
        "common_sample"
    ] = gold_g1_common.astype(
        bool
    )



# ============================================================
# 19G4. GOLD DUAL-ROLE ARCHITECTURE AUDIT
# ============================================================

gold_dual_spec_table = pd.DataFrame()
gold_direction_component_table = pd.DataFrame()
gold_direction_period_table = pd.DataFrame()
gold_direction_yearly_table = pd.DataFrame()
gold_direction_bootstrap_table = pd.DataFrame()
gold_risk_component_table = pd.DataFrame()
gold_risk_period_table = pd.DataFrame()
gold_cross_role_table = pd.DataFrame()
gold_quadrant_table = pd.DataFrame()
gold_dual_gate_table = pd.DataFrame()
gold_dual_research_export = pd.DataFrame()

if selected_asset == "Gold (XAU/USD)":
    st.markdown("---")
    st.subheader(
        "🧭 Gold Dual-Role Architecture Audit"
    )

    st.caption(
        "Die Architektur trennt bewusst zwei Fragen: "
        "**Direction Context D1** = bullischer/bärischer Gold-Kontext; "
        "**Risk Context R1** = gesundes/fragiles Gold-Risikoumfeld. "
        "Die Gewichte werden nicht gesucht, sondern aus den bereits "
        "vorhandenen effektiven Current-Gewichten abgeleitet."
    )

    # --------------------------------------------------------
    # 1. PRE-REGISTERED ROLE SPECIFICATION
    # --------------------------------------------------------

    st.markdown(
        "### 1. Eingefrorene Rollen-Spezifikation"
    )

    d1_weight_total = float(
        sum(
            GOLD_D1_EFFECTIVE_WEIGHTS.values()
        )
    )

    r1_weight_total = float(
        sum(
            GOLD_R1_EFFECTIVE_WEIGHTS.values()
        )
    )

    spec_rows = []

    for component, weight in GOLD_D1_EFFECTIVE_WEIGHTS.items():
        spec_rows.append(
            {
                "Rolle": "Direction D1",
                "Komponente": component,
                "Effektives Current-Gewicht": float(
                    weight
                ),
                "Rollen-Gewicht": float(
                    weight
                    / d1_weight_total
                ),
            }
        )

    for component, weight in GOLD_R1_EFFECTIVE_WEIGHTS.items():
        spec_rows.append(
            {
                "Rolle": "Risk R1",
                "Komponente": component,
                "Effektives Current-Gewicht": float(
                    weight
                ),
                "Rollen-Gewicht": float(
                    weight
                    / r1_weight_total
                ),
            }
        )

    gold_dual_spec_table = pd.DataFrame(
        spec_rows
    )

    st.dataframe(
        gold_dual_spec_table.style.format(
            {
                "Effektives Current-Gewicht": "{:.1%}",
                "Rollen-Gewicht": "{:.1%}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "D1 verwendet G1-Makro, den unveränderten technischen Trend und "
        "den bisherigen OBV-Anteil. R1 verwendet CFTC, Fear & Greed, "
        "GVZ-Health, Credit-Health und MOVE-Health. Fehlende Daten werden "
        "zeilenweise coverage-aware renormalisiert."
    )

    # --------------------------------------------------------
    # 2. BUILD D1 / R1
    # --------------------------------------------------------

    direction_components = pd.DataFrame(
        index=raw_df.index
    )

    direction_components[
        "g1_macro"
    ] = pd.to_numeric(
        gold_g1_frame[
            "Pillar::Makroökonomie"
        ],
        errors="coerce",
    )

    direction_components[
        "technical_trend"
    ] = pd.to_numeric(
        gold_current_frame[
            "Pillar::Technischer_Trend"
        ],
        errors="coerce",
    )

    direction_components[
        "obv_momentum"
    ] = pd.to_numeric(
        norm_df[
            "obv_momentum"
        ],
        errors="coerce",
    )

    (
        gold_d1_score,
        gold_d1_coverage,
    ) = build_coverage_weighted_context(
        direction_components,
        GOLD_D1_EFFECTIVE_WEIGHTS,
    )

    risk_components = pd.DataFrame(
        index=raw_df.index
    )

    for factor in GOLD_R1_EFFECTIVE_WEIGHTS:
        risk_components[
            factor
        ] = pd.to_numeric(
            norm_df.get(
                factor,
                pd.Series(
                    np.nan,
                    index=norm_df.index,
                ),
            ),
            errors="coerce",
        )

    (
        gold_r1_score,
        gold_r1_coverage,
    ) = build_coverage_weighted_context(
        risk_components,
        GOLD_R1_EFFECTIVE_WEIGHTS,
    )

    direction_common = (
        gold_d1_score.notna()
        & gold_g1_score.notna()
        & gold_current_score.notna()
        & (
            gold_d1_coverage
            >= float(
                min_coverage
            )
        )
        & (
            pd.to_numeric(
                gold_current_frame[
                    "Model_Data_Coverage"
                ],
                errors="coerce",
            )
            >= float(
                min_coverage
            )
        )
        & (
            pd.to_numeric(
                gold_g1_frame[
                    "Model_Data_Coverage"
                ],
                errors="coerce",
            )
            >= float(
                min_coverage
            )
        )
    )

    risk_common = (
        gold_r1_score.notna()
        & (
            gold_r1_coverage
            >= float(
                min_coverage
            )
        )
    )

    dual_common = (
        direction_common
        & risk_common
    )

    # --------------------------------------------------------
    # 3. DIRECTION COMPONENT AUDIT
    # --------------------------------------------------------

    st.markdown(
        "### 2. Direction-Rolle – welche Komponente trägt Gold-Forward-Returns?"
    )

    direction_candidates = {
        "A · Current Full": gold_current_score,
        "G1 · Full": gold_g1_score,
        "G1 Macro": direction_components[
            "g1_macro"
        ],
        "Current Trend": direction_components[
            "technical_trend"
        ],
        "OBV Momentum": direction_components[
            "obv_momentum"
        ],
        "D1 · Direction Context": gold_d1_score,
    }

    direction_rows = []

    for label, score in direction_candidates.items():
        paired = pd.to_numeric(
            score,
            errors="coerce",
        ).where(
            direction_common
        )

        direction20, signals20 = directional_accuracy(
            paired,
            targets[
                "Fwd_Return_20D"
            ],
        )

        non20, phases20 = phase_median_nonoverlap_ic(
            paired,
            targets[
                "Fwd_Return_20D"
            ],
            20,
        )

        non60, phases60 = phase_median_nonoverlap_ic(
            paired,
            targets[
                "Fwd_Return_60D"
            ],
            60,
        )

        q_table, q_spread, q_mono = quintile_statistics(
            paired,
            targets[
                "Fwd_Return_20D"
            ],
        )

        direction_rows.append(
            {
                "Komponente": label,
                "N": int(
                    (
                        paired.notna()
                        & targets[
                            "Fwd_Return_20D"
                        ].notna()
                    ).sum()
                ),
                "IC 5D": safe_spearman(
                    paired,
                    targets[
                        "Fwd_Return_5D"
                    ],
                ),
                "IC 20D": safe_spearman(
                    paired,
                    targets[
                        "Fwd_Return_20D"
                    ],
                ),
                "IC 60D": safe_spearman(
                    paired,
                    targets[
                        "Fwd_Return_60D"
                    ],
                ),
                "Direction 20D": direction20,
                "Signals": signals20,
                "Non-Overlap IC20": non20,
                "NO20 Phasen": phases20,
                "Non-Overlap IC60": non60,
                "NO60 Phasen": phases60,
                "Q5−Q1 Spread": q_spread,
                "Quintile-Monotonie": q_mono,
            }
        )

    gold_direction_component_table = pd.DataFrame(
        direction_rows
    )

    st.dataframe(
        gold_direction_component_table.style.format(
            {
                "N": "{:.0f}",
                "IC 5D": "{:+.3f}",
                "IC 20D": "{:+.3f}",
                "IC 60D": "{:+.3f}",
                "Direction 20D": "{:.1%}",
                "Signals": "{:.0f}",
                "Non-Overlap IC20": "{:+.3f}",
                "NO20 Phasen": "{:.0f}",
                "Non-Overlap IC60": "{:+.3f}",
                "NO60 Phasen": "{:.0f}",
                "Q5−Q1 Spread": "{:+.2%}",
                "Quintile-Monotonie": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 4. D1 FIXED PERIODS / YEARS
    # --------------------------------------------------------

    st.markdown(
        "### 3. D1 – feste Teilperioden und Kalenderjahre"
    )

    d1_period_rows = []

    for label, y1, y2 in GOLD_FIXED_PERIODS:
        mask = (
            direction_common
            & (
                targets.index.year
                >= y1
            )
            & (
                targets.index.year
                <= y2
            )
        )

        d1_ic = safe_spearman(
            gold_d1_score.where(
                mask
            ),
            targets[
                "Fwd_Return_20D"
            ],
        )

        g1_ic = safe_spearman(
            gold_g1_score.where(
                mask
            ),
            targets[
                "Fwd_Return_20D"
            ],
        )

        current_ic = safe_spearman(
            gold_current_score.where(
                mask
            ),
            targets[
                "Fwd_Return_20D"
            ],
        )

        d1_dir, d1_signals = directional_accuracy(
            gold_d1_score.where(
                mask
            ),
            targets[
                "Fwd_Return_20D"
            ],
        )

        d1_period_rows.append(
            {
                "Periode": label,
                "N": int(
                    mask.sum()
                ),
                "Current IC20": current_ic,
                "G1 IC20": g1_ic,
                "D1 IC20": d1_ic,
                "D1−G1": (
                    d1_ic
                    - g1_ic
                    if (
                        np.isfinite(
                            d1_ic
                        )
                        and np.isfinite(
                            g1_ic
                        )
                    )
                    else np.nan
                ),
                "D1 Direction20": d1_dir,
                "D1 Signals": d1_signals,
                "D1 positiv": bool(
                    np.isfinite(
                        d1_ic
                    )
                    and d1_ic > 0
                ),
            }
        )

    gold_direction_period_table = pd.DataFrame(
        d1_period_rows
    )

    st.dataframe(
        gold_direction_period_table.style.format(
            {
                "N": "{:.0f}",
                "Current IC20": "{:+.3f}",
                "G1 IC20": "{:+.3f}",
                "D1 IC20": "{:+.3f}",
                "D1−G1": "{:+.3f}",
                "D1 Direction20": "{:.1%}",
                "D1 Signals": "{:.0f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    d1_year_rows = []

    current_year = int(
        pd.Timestamp.now().year
    )

    for year in sorted(
        set(
            targets.index.year
        )
    ):
        if (
            year < 2012
            or year >= current_year
        ):
            continue

        mask = (
            direction_common
            & (
                targets.index.year
                == int(
                    year
                )
            )
        )

        if int(
            mask.sum()
        ) < 60:
            continue

        current_ic = safe_spearman(
            gold_current_score.where(
                mask
            ),
            targets[
                "Fwd_Return_20D"
            ],
        )

        g1_ic = safe_spearman(
            gold_g1_score.where(
                mask
            ),
            targets[
                "Fwd_Return_20D"
            ],
        )

        d1_ic = safe_spearman(
            gold_d1_score.where(
                mask
            ),
            targets[
                "Fwd_Return_20D"
            ],
        )

        d1_year_rows.append(
            {
                "Jahr": int(
                    year
                ),
                "N": int(
                    mask.sum()
                ),
                "Current IC20": current_ic,
                "G1 IC20": g1_ic,
                "D1 IC20": d1_ic,
                "D1 positiv": bool(
                    np.isfinite(
                        d1_ic
                    )
                    and d1_ic > 0
                ),
                "D1 schlägt G1": bool(
                    np.isfinite(
                        d1_ic
                    )
                    and np.isfinite(
                        g1_ic
                    )
                    and d1_ic > g1_ic
                ),
                "D1 schlägt Current": bool(
                    np.isfinite(
                        d1_ic
                    )
                    and np.isfinite(
                        current_ic
                    )
                    and d1_ic > current_ic
                ),
            }
        )

    gold_direction_yearly_table = pd.DataFrame(
        d1_year_rows
    )

    st.dataframe(
        gold_direction_yearly_table.style.format(
            {
                "N": "{:.0f}",
                "Current IC20": "{:+.3f}",
                "G1 IC20": "{:+.3f}",
                "D1 IC20": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 5. DIRECTION BOOTSTRAP
    # --------------------------------------------------------

    st.markdown(
        "### 4. D1 Block-Bootstrap – gegenüber Current und G1 Full"
    )

    boot_d1_vs_current = block_bootstrap_ic_difference(
        gold_current_score.where(
            direction_common
        ),
        gold_d1_score.where(
            direction_common
        ),
        targets[
            "Fwd_Return_20D"
        ].where(
            direction_common
        ),
        block_length=int(
            bootstrap_block
        ),
        n_boot=int(
            bootstrap_runs
        ),
        seed=20260912,
    )

    boot_d1_vs_g1 = block_bootstrap_ic_difference(
        gold_g1_score.where(
            direction_common
        ),
        gold_d1_score.where(
            direction_common
        ),
        targets[
            "Fwd_Return_20D"
        ].where(
            direction_common
        ),
        block_length=int(
            bootstrap_block
        ),
        n_boot=int(
            bootstrap_runs
        ),
        seed=20260913,
    )

    gold_direction_bootstrap_table = pd.DataFrame(
        [
            {
                "Vergleich": "D1 − Current",
                "Δ IC20": boot_d1_vs_current[
                    "observed"
                ],
                "P(Challenger>Basis)": boot_d1_vs_current[
                    "prob_positive"
                ],
                "95% CI Low": boot_d1_vs_current[
                    "lower"
                ],
                "95% CI High": boot_d1_vs_current[
                    "upper"
                ],
                "N": boot_d1_vs_current[
                    "n"
                ],
            },
            {
                "Vergleich": "D1 − G1 Full",
                "Δ IC20": boot_d1_vs_g1[
                    "observed"
                ],
                "P(Challenger>Basis)": boot_d1_vs_g1[
                    "prob_positive"
                ],
                "95% CI Low": boot_d1_vs_g1[
                    "lower"
                ],
                "95% CI High": boot_d1_vs_g1[
                    "upper"
                ],
                "N": boot_d1_vs_g1[
                    "n"
                ],
            },
        ]
    )

    st.dataframe(
        gold_direction_bootstrap_table.style.format(
            {
                "Δ IC20": "{:+.3f}",
                "P(Challenger>Basis)": "{:.1%}",
                "95% CI Low": "{:+.3f}",
                "95% CI High": "{:+.3f}",
                "N": "{:.0f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 6. RISK COMPONENT AUDIT
    # --------------------------------------------------------

    st.markdown(
        "### 5. Risk-Rolle – welche Komponente trägt Stress/Volatilität?"
    )

    current_positioning = pd.to_numeric(
        gold_current_frame[
            "Pillar::Positionierung"
        ],
        errors="coerce",
    )

    current_early = pd.to_numeric(
        gold_current_frame[
            "Pillar::Fruehwarnindikatoren"
        ],
        errors="coerce",
    )

    risk_candidates = {
        "A · Current Full": (
            gold_current_score,
            pd.to_numeric(
                gold_current_frame[
                    "Model_Data_Coverage"
                ],
                errors="coerce",
            ),
        ),
        "G1 · Full": (
            gold_g1_score,
            pd.to_numeric(
                gold_g1_frame[
                    "Model_Data_Coverage"
                ],
                errors="coerce",
            ),
        ),
        "Current Positioning": (
            current_positioning,
            pd.Series(
                100.0,
                index=raw_df.index,
            ),
        ),
        "GVZ Health": (
            risk_components[
                "vix_score"
            ],
            pd.Series(
                100.0,
                index=raw_df.index,
            ),
        ),
        "Current Early Warning": (
            current_early,
            pd.Series(
                100.0,
                index=raw_df.index,
            ),
        ),
        "R1 · Risk Context": (
            gold_r1_score,
            gold_r1_coverage,
        ),
    }

    risk_rows = []

    for label, (
        score,
        coverage,
    ) in risk_candidates.items():
        metrics = dual_role_risk_metrics(
            pd.to_numeric(
                score,
                errors="coerce",
            ),
            pd.to_numeric(
                coverage,
                errors="coerce",
            ),
            targets,
            mask=risk_common,
        )

        risk_rows.append(
            {
                "Komponente": label,
                **metrics,
            }
        )

    gold_risk_component_table = pd.DataFrame(
        risk_rows
    )

    st.dataframe(
        gold_risk_component_table.style.format(
            {
                "N": "{:.0f}",
                "Stress AUC 20D": "{:.3f}",
                "Phase-Median Stress AUC": "{:.3f}",
                "AUC-Phasen": "{:.0f}",
                "Q20 Stressrate": "{:.1%}",
                "Q80 Stressrate": "{:.1%}",
                "Q20−Q80 Stress-Gap": "{:+.1%}",
                "Score≤40 Stressrate": "{:.1%}",
                "Score≥60 Stressrate": "{:.1%}",
                "Fixed Stress-Gap": "{:+.1%}",
                "Score vs niedrigere FwdVol": "{:+.3f}",
                "Score vs bessere FwdMAE": "{:+.3f}",
                "Ø Coverage": "{:.1f}%",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 7. R1 PERIOD ROBUSTNESS
    # --------------------------------------------------------

    st.markdown(
        "### 6. R1 – Risk-State in festen Teilperioden"
    )

    risk_period_rows = []

    for label, y1, y2 in GOLD_FIXED_PERIODS:
        mask = (
            risk_common
            & (
                targets.index.year
                >= y1
            )
            & (
                targets.index.year
                <= y2
            )
        )

        metrics = dual_role_risk_metrics(
            gold_r1_score,
            gold_r1_coverage,
            targets,
            mask=mask,
        )

        risk_period_rows.append(
            {
                "Periode": label,
                **metrics,
                "AUC > 0.50": bool(
                    np.isfinite(
                        metrics[
                            "Stress AUC 20D"
                        ]
                    )
                    and metrics[
                        "Stress AUC 20D"
                    ] > 0.50
                ),
            }
        )

    gold_risk_period_table = pd.DataFrame(
        risk_period_rows
    )

    st.dataframe(
        gold_risk_period_table.style.format(
            {
                "N": "{:.0f}",
                "Stress AUC 20D": "{:.3f}",
                "Phase-Median Stress AUC": "{:.3f}",
                "AUC-Phasen": "{:.0f}",
                "Q20 Stressrate": "{:.1%}",
                "Q80 Stressrate": "{:.1%}",
                "Q20−Q80 Stress-Gap": "{:+.1%}",
                "Score≤40 Stressrate": "{:.1%}",
                "Score≥60 Stressrate": "{:.1%}",
                "Fixed Stress-Gap": "{:+.1%}",
                "Score vs niedrigere FwdVol": "{:+.3f}",
                "Score vs bessere FwdMAE": "{:+.3f}",
                "Ø Coverage": "{:.1f}%",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 8. CROSS-ROLE / SPECIALIZATION
    # --------------------------------------------------------

    st.markdown(
        "### 7. Rollen-Spezialisierung – Direction D1 vs. Risk R1"
    )

    cross_rows = []

    for label, score, coverage in [
        (
            "D1 · Direction Context",
            gold_d1_score,
            gold_d1_coverage,
        ),
        (
            "R1 · Risk Context",
            gold_r1_score,
            gold_r1_coverage,
        ),
    ]:
        s = pd.to_numeric(
            score,
            errors="coerce",
        ).where(
            dual_common
        )

        direction20, signals20 = directional_accuracy(
            s,
            targets[
                "Fwd_Return_20D"
            ],
        )

        risk_metrics = dual_role_risk_metrics(
            s,
            pd.to_numeric(
                coverage,
                errors="coerce",
            ),
            targets,
            mask=dual_common,
        )

        cross_rows.append(
            {
                "Output": label,
                "IC 20D": safe_spearman(
                    s,
                    targets[
                        "Fwd_Return_20D"
                    ],
                ),
                "Direction 20D": direction20,
                "Signals": signals20,
                "Stress AUC 20D": risk_metrics[
                    "Stress AUC 20D"
                ],
                "Phase Stress AUC": risk_metrics[
                    "Phase-Median Stress AUC"
                ],
                "Q20−Q80 Stress-Gap": risk_metrics[
                    "Q20−Q80 Stress-Gap"
                ],
                "vs niedrigere FwdVol": risk_metrics[
                    "Score vs niedrigere FwdVol"
                ],
                "vs bessere FwdMAE": risk_metrics[
                    "Score vs bessere FwdMAE"
                ],
            }
        )

    gold_cross_role_table = pd.DataFrame(
        cross_rows
    )

    d1_r1_corr = safe_spearman(
        gold_d1_score.where(
            dual_common
        ),
        gold_r1_score.where(
            dual_common
        ),
    )

    st.dataframe(
        gold_cross_role_table.style.format(
            {
                "IC 20D": "{:+.3f}",
                "Direction 20D": "{:.1%}",
                "Signals": "{:.0f}",
                "Stress AUC 20D": "{:.3f}",
                "Phase Stress AUC": "{:.3f}",
                "Q20−Q80 Stress-Gap": "{:+.1%}",
                "vs niedrigere FwdVol": "{:+.3f}",
                "vs bessere FwdMAE": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        f"Spearman-Korrelation D1 vs. R1 auf dem Dual-Common-Sample: "
        f"{d1_r1_corr:+.3f}. Eine hohe Korrelation ist nicht automatisch "
        "schlecht; entscheidend ist, ob die Outputs in ihren jeweiligen "
        "Zielrollen unterschiedliche Stärke zeigen."
    )

    # --------------------------------------------------------
    # 9. 2x2 STATE QUADRANTS
    # --------------------------------------------------------

    st.markdown(
        "### 8. 2×2 Gold-State-Matrix – Direction × Risk"
    )

    quadrant_frame = pd.DataFrame(
        {
            "d1": pd.to_numeric(
                gold_d1_score,
                errors="coerce",
            ),
            "r1": pd.to_numeric(
                gold_r1_score,
                errors="coerce",
            ),
            "ret20": pd.to_numeric(
                targets[
                    "Fwd_Return_20D"
                ],
                errors="coerce",
            ),
            "stress": pd.to_numeric(
                targets[
                    "Stress_Event_20D"
                ],
                errors="coerce",
            ),
            "vol": pd.to_numeric(
                targets[
                    "Fwd_Realized_Vol_20D"
                ],
                errors="coerce",
            ),
            "mae": pd.to_numeric(
                targets[
                    "Fwd_MAE_20D"
                ],
                errors="coerce",
            ),
        }
    ).where(
        dual_common,
        np.nan,
    )

    quadrant_rows = []

    quadrant_specs = [
        (
            "Bullish + Healthy",
            (
                quadrant_frame[
                    "d1"
                ]
                >= GOLD_DUAL_ROLE_SPLIT_THRESHOLD
            )
            &
            (
                quadrant_frame[
                    "r1"
                ]
                >= GOLD_DUAL_ROLE_SPLIT_THRESHOLD
            ),
        ),
        (
            "Bullish + Fragile",
            (
                quadrant_frame[
                    "d1"
                ]
                >= GOLD_DUAL_ROLE_SPLIT_THRESHOLD
            )
            &
            (
                quadrant_frame[
                    "r1"
                ]
                < GOLD_DUAL_ROLE_SPLIT_THRESHOLD
            ),
        ),
        (
            "Bearish + Healthy",
            (
                quadrant_frame[
                    "d1"
                ]
                < GOLD_DUAL_ROLE_SPLIT_THRESHOLD
            )
            &
            (
                quadrant_frame[
                    "r1"
                ]
                >= GOLD_DUAL_ROLE_SPLIT_THRESHOLD
            ),
        ),
        (
            "Bearish + Fragile",
            (
                quadrant_frame[
                    "d1"
                ]
                < GOLD_DUAL_ROLE_SPLIT_THRESHOLD
            )
            &
            (
                quadrant_frame[
                    "r1"
                ]
                < GOLD_DUAL_ROLE_SPLIT_THRESHOLD
            ),
        ),
    ]

    for label, mask in quadrant_specs:
        q = quadrant_frame[
            mask
        ]

        quadrant_rows.append(
            {
                "State": label,
                "N": int(
                    len(
                        q
                    )
                ),
                "Ø D1": float(
                    q[
                        "d1"
                    ].mean()
                ),
                "Ø R1": float(
                    q[
                        "r1"
                    ].mean()
                ),
                "Ø Fwd Return 20D": float(
                    q[
                        "ret20"
                    ].mean()
                ),
                "Stressrate 20D": float(
                    q[
                        "stress"
                    ].mean()
                ),
                "Ø FwdVol 20D": float(
                    q[
                        "vol"
                    ].mean()
                ),
                "Ø FwdMAE 20D": float(
                    q[
                        "mae"
                    ].mean()
                ),
            }
        )

    gold_quadrant_table = pd.DataFrame(
        quadrant_rows
    )

    st.dataframe(
        gold_quadrant_table.style.format(
            {
                "N": "{:.0f}",
                "Ø D1": "{:.1f}",
                "Ø R1": "{:.1f}",
                "Ø Fwd Return 20D": "{:+.2%}",
                "Stressrate 20D": "{:.1%}",
                "Ø FwdVol 20D": "{:.2%}",
                "Ø FwdMAE 20D": "{:+.2%}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 10. PRE-REGISTERED ARCHITECTURE GATES
    # --------------------------------------------------------

    st.markdown(
        "### 9. Vorab festgelegte Dual-Role-Gates"
    )

    d1_row = gold_direction_component_table[
        gold_direction_component_table[
            "Komponente"
        ]
        == "D1 · Direction Context"
    ].iloc[
        0
    ]

    g1_row = gold_direction_component_table[
        gold_direction_component_table[
            "Komponente"
        ]
        == "G1 · Full"
    ].iloc[
        0
    ]

    r1_row = gold_risk_component_table[
        gold_risk_component_table[
            "Komponente"
        ]
        == "R1 · Risk Context"
    ].iloc[
        0
    ]

    positive_d1_periods = int(
        gold_direction_period_table[
            "D1 positiv"
        ]
        .astype(
            bool
        )
        .sum()
    )

    evaluable_d1_years = gold_direction_yearly_table[
        gold_direction_yearly_table[
            "D1 IC20"
        ].notna()
    ]

    d1_positive_year_rate = (
        float(
            evaluable_d1_years[
                "D1 positiv"
            ]
            .astype(
                bool
            )
            .mean()
        )
        if not evaluable_d1_years.empty
        else np.nan
    )

    positive_risk_periods = int(
        gold_risk_period_table[
            "AUC > 0.50"
        ]
        .astype(
            bool
        )
        .sum()
    )

    d1_delta_vs_g1 = (
        float(
            d1_row[
                "IC 20D"
            ]
        )
        - float(
            g1_row[
                "IC 20D"
            ]
        )
    )

    direction_gate_rows = [
        {
            "Gruppe": "Direction D1",
            "Kriterium": "D1 IC20 ≥ +0.05",
            "Erfüllt": bool(
                np.isfinite(
                    d1_row[
                        "IC 20D"
                    ]
                )
                and d1_row[
                    "IC 20D"
                ]
                >= GOLD_D1_MIN_IC20
            ),
            "Messwert": (
                f"{d1_row['IC 20D']:+.3f}"
            ),
        },
        {
            "Gruppe": "Direction D1",
            "Kriterium": "D1 Non-Overlap IC20 > 0",
            "Erfüllt": bool(
                np.isfinite(
                    d1_row[
                        "Non-Overlap IC20"
                    ]
                )
                and d1_row[
                    "Non-Overlap IC20"
                ]
                > GOLD_D1_MIN_NONOVERLAP_IC20
            ),
            "Messwert": (
                f"{d1_row['Non-Overlap IC20']:+.3f}"
            ),
        },
        {
            "Gruppe": "Direction D1",
            "Kriterium": "D1 Direction20 ≥50%",
            "Erfüllt": bool(
                np.isfinite(
                    d1_row[
                        "Direction 20D"
                    ]
                )
                and d1_row[
                    "Direction 20D"
                ]
                >= GOLD_D1_MIN_DIRECTION20
            ),
            "Messwert": (
                f"{d1_row['Direction 20D']:.1%}"
            ),
        },
        {
            "Gruppe": "Direction D1",
            "Kriterium": "D1 IC20 positiv in mindestens 2/3 festen Perioden",
            "Erfüllt": bool(
                positive_d1_periods
                >= GOLD_D1_MIN_POSITIVE_FIXED_PERIODS
            ),
            "Messwert": (
                f"{positive_d1_periods}/3"
            ),
        },
        {
            "Gruppe": "Direction D1",
            "Kriterium": "D1 IC20 positiv in ≥50% vollständiger Jahre",
            "Erfüllt": bool(
                np.isfinite(
                    d1_positive_year_rate
                )
                and d1_positive_year_rate
                >= GOLD_D1_MIN_POSITIVE_YEAR_RATE
            ),
            "Messwert": (
                f"{d1_positive_year_rate:.1%}"
                if np.isfinite(
                    d1_positive_year_rate
                )
                else "n/a"
            ),
        },
        {
            "Gruppe": "Direction D1",
            "Kriterium": "D1 IC20 mindestens +0.03 über G1 Full",
            "Erfüllt": bool(
                np.isfinite(
                    d1_delta_vs_g1
                )
                and d1_delta_vs_g1
                >= GOLD_D1_MIN_DELTA_VS_G1_IC20
            ),
            "Messwert": (
                f"{d1_delta_vs_g1:+.3f}"
            ),
        },
        {
            "Gruppe": "Direction D1",
            "Kriterium": "Bootstrap P(D1>Current) ≥90%",
            "Erfüllt": bool(
                np.isfinite(
                    boot_d1_vs_current[
                        "prob_positive"
                    ]
                )
                and boot_d1_vs_current[
                    "prob_positive"
                ]
                >= GOLD_D1_MIN_BOOTSTRAP_PROB_VS_CURRENT
            ),
            "Messwert": (
                f"{boot_d1_vs_current['prob_positive']:.1%}"
                if np.isfinite(
                    boot_d1_vs_current[
                        "prob_positive"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "Gruppe": "Direction D1",
            "Kriterium": "Bootstrap P(D1>G1 Full) ≥75%",
            "Erfüllt": bool(
                np.isfinite(
                    boot_d1_vs_g1[
                        "prob_positive"
                    ]
                )
                and boot_d1_vs_g1[
                    "prob_positive"
                ]
                >= GOLD_D1_MIN_BOOTSTRAP_PROB_VS_G1
            ),
            "Messwert": (
                f"{boot_d1_vs_g1['prob_positive']:.1%}"
                if np.isfinite(
                    boot_d1_vs_g1[
                        "prob_positive"
                    ]
                )
                else "n/a"
            ),
        },
    ]

    risk_gate_rows = [
        {
            "Gruppe": "Risk R1",
            "Kriterium": "R1 Stress-AUC ≥0.60",
            "Erfüllt": bool(
                np.isfinite(
                    r1_row[
                        "Stress AUC 20D"
                    ]
                )
                and r1_row[
                    "Stress AUC 20D"
                ]
                >= GOLD_R1_MIN_STRESS_AUC
            ),
            "Messwert": (
                f"{r1_row['Stress AUC 20D']:.3f}"
            ),
        },
        {
            "Gruppe": "Risk R1",
            "Kriterium": "R1 Phase-Median Stress-AUC ≥0.55",
            "Erfüllt": bool(
                np.isfinite(
                    r1_row[
                        "Phase-Median Stress AUC"
                    ]
                )
                and r1_row[
                    "Phase-Median Stress AUC"
                ]
                >= GOLD_R1_MIN_PHASE_AUC
            ),
            "Messwert": (
                f"{r1_row['Phase-Median Stress AUC']:.3f}"
            ),
        },
        {
            "Gruppe": "Risk R1",
            "Kriterium": "R1 Q20−Q80 Stress-Gap ≥5pp",
            "Erfüllt": bool(
                np.isfinite(
                    r1_row[
                        "Q20−Q80 Stress-Gap"
                    ]
                )
                and r1_row[
                    "Q20−Q80 Stress-Gap"
                ]
                >= GOLD_R1_MIN_QUANTILE_STRESS_GAP
            ),
            "Messwert": (
                f"{r1_row['Q20−Q80 Stress-Gap']:+.1%}"
                if np.isfinite(
                    r1_row[
                        "Q20−Q80 Stress-Gap"
                    ]
                )
                else "n/a"
            ),
        },
        {
            "Gruppe": "Risk R1",
            "Kriterium": "R1 höher → niedrigere FwdVol",
            "Erfüllt": bool(
                np.isfinite(
                    r1_row[
                        "Score vs niedrigere FwdVol"
                    ]
                )
                and r1_row[
                    "Score vs niedrigere FwdVol"
                ] > 0
            ),
            "Messwert": (
                f"{r1_row['Score vs niedrigere FwdVol']:+.3f}"
            ),
        },
        {
            "Gruppe": "Risk R1",
            "Kriterium": "R1 höher → bessere FwdMAE",
            "Erfüllt": bool(
                np.isfinite(
                    r1_row[
                        "Score vs bessere FwdMAE"
                    ]
                )
                and r1_row[
                    "Score vs bessere FwdMAE"
                ] > 0
            ),
            "Messwert": (
                f"{r1_row['Score vs bessere FwdMAE']:+.3f}"
            ),
        },
        {
            "Gruppe": "Risk R1",
            "Kriterium": "R1 AUC >0.50 in mindestens 2/3 festen Perioden",
            "Erfüllt": bool(
                positive_risk_periods
                >= GOLD_R1_MIN_POSITIVE_RISK_PERIODS
            ),
            "Messwert": (
                f"{positive_risk_periods}/3"
            ),
        },
    ]

    gold_dual_gate_table = pd.DataFrame(
        direction_gate_rows
        + risk_gate_rows
    )

    gold_dual_gate_table[
        "Status"
    ] = gold_dual_gate_table[
        "Erfüllt"
    ].map(
        {
            True: "✅",
            False: "❌",
        }
    )

    st.dataframe(
        gold_dual_gate_table[
            [
                "Gruppe",
                "Kriterium",
                "Status",
                "Messwert",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    d_gates = gold_dual_gate_table[
        gold_dual_gate_table[
            "Gruppe"
        ]
        == "Direction D1"
    ]

    r_gates = gold_dual_gate_table[
        gold_dual_gate_table[
            "Gruppe"
        ]
        == "Risk R1"
    ]

    d_pass = int(
        d_gates[
            "Erfüllt"
        ].sum()
    )

    r_pass = int(
        r_gates[
            "Erfüllt"
        ].sum()
    )

    st.markdown(
        "### 10. Architektururteil"
    )

    if (
        d_pass
        == len(
            d_gates
        )
        and r_pass
        == len(
            r_gates
        )
    ):
        st.success(
            "🟢 **DUAL-ROLE-ARCHITEKTUR HISTORISCH BESTÄTIGT.** "
            "D1 trägt die Direction-Rolle und R1 die Risk-State-Rolle "
            "mit den vorab definierten Anforderungen. Der nächste Schritt "
            "wäre ein **eingefrorener Dual-Output-Walk-Forward**, nicht "
            "sofort Produktion."
        )

    elif (
        d_pass
        == len(
            d_gates
        )
        and r_pass
        < len(
            r_gates
        )
    ):
        st.warning(
            f"🟡 **DIRECTION-SPLIT BESTÄTIGT, RISK-SPLIT NOCH NICHT "
            f"({r_pass}/{len(r_gates)} Risk-Gates).** "
            "Dann bleibt D1 eingefroren und nur der Risk-Block wird "
            "diagnostisch weiter untersucht – ohne D1 nachzutunen."
        )

    elif (
        r_pass
        == len(
            r_gates
        )
        and d_pass
        < len(
            d_gates
        )
    ):
        st.warning(
            f"🟡 **RISK-SPLIT BESTÄTIGT, DIRECTION-SPLIT NOCH NICHT "
            f"({d_pass}/{len(d_gates)} Direction-Gates).** "
            "Dann bleibt R1 eingefroren und wir verwerfen bzw. überdenken "
            "D1, statt beide Rollen wieder zusammenzumischen."
        )

    else:
        st.error(
            f"🔴 **DUAL-ROLE IN DIESER EINFACHEN FORM NOCH NICHT "
            f"BESTÄTIGT.** Direction {d_pass}/{len(d_gates)}, "
            f"Risk {r_pass}/{len(r_gates)}. Dann verwenden wir die "
            "Komponenten- und Quadrantentabellen, um zu entscheiden, "
            "welche Rolle strukturell anders definiert werden muss."
        )

    st.info(
        "Research-Regel v1.0.22: D1 und R1 sind Diagnose-Outputs. "
        "Es werden weder neue Produktionsgewichte geschrieben noch "
        "bestehende Gold-, WTI-, EUR/USD-, S&P- oder Nasdaq-Shadows verändert."
    )

    # --------------------------------------------------------
    # 11. EXPORT FRAME
    # --------------------------------------------------------

    gold_dual_research_export = pd.DataFrame(
        index=raw_df.index
    )

    gold_dual_research_export[
        "asset_price"
    ] = pd.to_numeric(
        raw_df[
            "asset_price"
        ],
        errors="coerce",
    )

    gold_dual_research_export[
        "current_score"
    ] = gold_current_score

    gold_dual_research_export[
        "g1_full_score"
    ] = gold_g1_score

    gold_dual_research_export[
        "d1_direction_score"
    ] = gold_d1_score

    gold_dual_research_export[
        "d1_coverage"
    ] = gold_d1_coverage

    gold_dual_research_export[
        "r1_risk_score"
    ] = gold_r1_score

    gold_dual_research_export[
        "r1_coverage"
    ] = gold_r1_coverage

    gold_dual_research_export[
        "g1_macro"
    ] = direction_components[
        "g1_macro"
    ]

    gold_dual_research_export[
        "technical_trend"
    ] = direction_components[
        "technical_trend"
    ]

    gold_dual_research_export[
        "obv_momentum"
    ] = direction_components[
        "obv_momentum"
    ]

    for factor in GOLD_R1_EFFECTIVE_WEIGHTS:
        gold_dual_research_export[
            f"risk_component__{factor}"
        ] = risk_components[
            factor
        ]

    for col in [
        "Fwd_Return_5D",
        "Fwd_Return_20D",
        "Fwd_Return_60D",
        "Fwd_MAE_20D",
        "Fwd_Realized_Vol_20D",
        "Stress_Event_20D",
    ]:
        gold_dual_research_export[
            col
        ] = targets[
            col
        ]

    gold_dual_research_export[
        "direction_common"
    ] = direction_common.astype(
        bool
    )

    gold_dual_research_export[
        "risk_common"
    ] = risk_common.astype(
        bool
    )

    gold_dual_research_export[
        "dual_common"
    ] = dual_common.astype(
        bool
    )



# ============================================================
# 19G5. GOLD R1 RISK CONTEXT STRUCTURAL AUDIT
# ============================================================

gold_r1_frozen_d1_table = pd.DataFrame()
gold_r1_component_orientation_table = pd.DataFrame()
gold_r1_level_change_table = pd.DataFrame()
gold_r1_ablation_table = pd.DataFrame()
gold_r1_target_sensitivity_table = pd.DataFrame()
gold_r1_volatility_target_table = pd.DataFrame()
gold_r1_period_component_table = pd.DataFrame()
gold_r1_structural_gate_table = pd.DataFrame()
gold_r1_research_export = pd.DataFrame()

if selected_asset == "Gold (XAU/USD)":
    st.markdown("---")
    st.subheader(
        "🛡️ Gold R1 Risk Context Structural Audit"
    )

    st.caption(
        "D1 wird in diesem Abschnitt **nicht mehr angefasst**. "
        "Der Audit untersucht nur, warum R1 historisch lediglich 3/6 "
        "Risk-Gates bestanden hat. Alle Prüfungen sind diagnostisch; "
        "es wird noch kein neuer R2-Score erzeugt."
    )

    # --------------------------------------------------------
    # 1. FREEZE CHECK FOR D1
    # --------------------------------------------------------

    st.markdown(
        "### 1. D1 Freeze – bestandener Direction Context bleibt unverändert"
    )

    d1_reference = gold_direction_component_table[
        gold_direction_component_table[
            "Komponente"
        ]
        == "D1 · Direction Context"
    ].iloc[
        0
    ]

    d1_positive_periods_frozen = int(
        gold_direction_period_table[
            "D1 positiv"
        ]
        .astype(
            bool
        )
        .sum()
    )

    d1_evaluable_years_frozen = gold_direction_yearly_table[
        gold_direction_yearly_table[
            "D1 IC20"
        ].notna()
    ]

    d1_positive_year_rate_frozen = (
        float(
            d1_evaluable_years_frozen[
                "D1 positiv"
            ]
            .astype(
                bool
            )
            .mean()
        )
        if not d1_evaluable_years_frozen.empty
        else np.nan
    )

    gold_r1_frozen_d1_table = pd.DataFrame(
        [
            {
                "Status": "FROZEN – keine Änderungen in v1.0.23",
                "IC 20D": float(
                    d1_reference[
                        "IC 20D"
                    ]
                ),
                "Non-Overlap IC20": float(
                    d1_reference[
                        "Non-Overlap IC20"
                    ]
                ),
                "Direction 20D": float(
                    d1_reference[
                        "Direction 20D"
                    ]
                ),
                "Positive feste Perioden": d1_positive_periods_frozen,
                "Positive Jahresrate": d1_positive_year_rate_frozen,
                "P(D1>Current)": float(
                    boot_d1_vs_current[
                        "prob_positive"
                    ]
                ),
                "P(D1>G1)": float(
                    boot_d1_vs_g1[
                        "prob_positive"
                    ]
                ),
            }
        ]
    )

    st.dataframe(
        gold_r1_frozen_d1_table.style.format(
            {
                "IC 20D": "{:+.3f}",
                "Non-Overlap IC20": "{:+.3f}",
                "Direction 20D": "{:.1%}",
                "Positive feste Perioden": "{:.0f}",
                "Positive Jahresrate": "{:.1%}",
                "P(D1>Current)": "{:.1%}",
                "P(D1>G1)": "{:.1%}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 2. COMPONENT ORIENTATION AUDIT
    # --------------------------------------------------------

    st.markdown(
        "### 2. R1-Komponenten – stimmt die Health-Orientierung?"
    )

    component_labels = {
        "cot_noncommercials": "CFTC Non-Commercials",
        "fear_greed": "CNN Fear & Greed",
        "vix_score": "GVZ Health",
        "credit_spreads": "Credit Health",
        "move_index": "MOVE Health",
    }

    orientation_rows = []

    for factor, label in component_labels.items():
        current_score = pd.to_numeric(
            risk_components[
                factor
            ],
            errors="coerce",
        ).where(
            risk_common
        )

        opposite_score = (
            100.0
            - current_score
        )

        current_auc = binary_auc(
            current_score,
            targets[
                "Stress_Event_20D"
            ],
            higher_predictor_means_event=False,
        )

        opposite_auc = binary_auc(
            opposite_score,
            targets[
                "Stress_Event_20D"
            ],
            higher_predictor_means_event=False,
        )

        current_phase, current_phases = phase_median_stress_auc(
            current_score,
            targets[
                "Stress_Event_20D"
            ],
            horizon=20,
        )

        opposite_phase, opposite_phases = phase_median_stress_auc(
            opposite_score,
            targets[
                "Stress_Event_20D"
            ],
            horizon=20,
        )

        current_vol = safe_spearman(
            current_score,
            -targets[
                "Fwd_Realized_Vol_20D"
            ],
        )

        opposite_vol = safe_spearman(
            opposite_score,
            -targets[
                "Fwd_Realized_Vol_20D"
            ],
        )

        current_mae = safe_spearman(
            current_score,
            targets[
                "Fwd_MAE_20D"
            ],
        )

        opposite_mae = safe_spearman(
            opposite_score,
            targets[
                "Fwd_MAE_20D"
            ],
        )

        current_period_wins = 0
        opposite_period_wins = 0
        period_aucs = []

        for period_label, y1, y2 in GOLD_FIXED_PERIODS:
            period_mask = (
                risk_common
                & (
                    targets.index.year
                    >= y1
                )
                & (
                    targets.index.year
                    <= y2
                )
            )

            c_auc = binary_auc(
                current_score.where(
                    period_mask
                ),
                targets[
                    "Stress_Event_20D"
                ],
                higher_predictor_means_event=False,
            )

            o_auc = binary_auc(
                opposite_score.where(
                    period_mask
                ),
                targets[
                    "Stress_Event_20D"
                ],
                higher_predictor_means_event=False,
            )

            period_aucs.append(
                {
                    "Komponente": label,
                    "Periode": period_label,
                    "Current AUC": c_auc,
                    "Opposite AUC": o_auc,
                }
            )

            if (
                np.isfinite(
                    c_auc
                )
                and c_auc > 0.50
            ):
                current_period_wins += 1

            if (
                np.isfinite(
                    o_auc
                )
                and o_auc > 0.50
            ):
                opposite_period_wins += 1

        boot = block_bootstrap_auc_difference(
            current_score,
            opposite_score,
            targets[
                "Stress_Event_20D"
            ],
            block_length=int(
                bootstrap_block
            ),
            n_boot=int(
                bootstrap_runs
            ),
            seed=20260923,
        )

        auc_improvement = (
            opposite_auc
            - current_auc
            if (
                np.isfinite(
                    opposite_auc
                )
                and np.isfinite(
                    current_auc
                )
            )
            else np.nan
        )

        sign_conflict = bool(
            np.isfinite(
                auc_improvement
            )
            and auc_improvement
            >= GOLD_RISK_SIGN_MIN_AUC_IMPROVEMENT
            and opposite_period_wins
            >= GOLD_RISK_MIN_FIXED_PERIOD_CONFIRMATIONS
            and np.isfinite(
                boot[
                    "prob_positive"
                ]
            )
            and boot[
                "prob_positive"
            ]
            >= GOLD_RISK_MIN_BOOTSTRAP_PROB
        )

        orientation_rows.append(
            {
                "Komponente": label,
                "Current AUC": current_auc,
                "Opposite AUC": opposite_auc,
                "Δ AUC Opposite−Current": auc_improvement,
                "Current Phase-AUC": current_phase,
                "Opposite Phase-AUC": opposite_phase,
                "Current vs niedrigere FwdVol": current_vol,
                "Opposite vs niedrigere FwdVol": opposite_vol,
                "Current vs bessere FwdMAE": current_mae,
                "Opposite vs bessere FwdMAE": opposite_mae,
                "Current AUC>0.50 Perioden": current_period_wins,
                "Opposite AUC>0.50 Perioden": opposite_period_wins,
                "P(Opposite>Current)": boot[
                    "prob_positive"
                ],
                "95% CI Low": boot[
                    "lower"
                ],
                "95% CI High": boot[
                    "upper"
                ],
                "Struktureller Sign-Konflikt": sign_conflict,
            }
        )

        if period_aucs:
            if gold_r1_period_component_table.empty:
                gold_r1_period_component_table = pd.DataFrame(
                    period_aucs
                )
            else:
                gold_r1_period_component_table = pd.concat(
                    [
                        gold_r1_period_component_table,
                        pd.DataFrame(
                            period_aucs
                        ),
                    ],
                    ignore_index=True,
                )

    gold_r1_component_orientation_table = (
        pd.DataFrame(
            orientation_rows
        )
        .sort_values(
            "Current AUC",
            ascending=True,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    st.dataframe(
        gold_r1_component_orientation_table.style.format(
            {
                "Current AUC": "{:.3f}",
                "Opposite AUC": "{:.3f}",
                "Δ AUC Opposite−Current": "{:+.3f}",
                "Current Phase-AUC": "{:.3f}",
                "Opposite Phase-AUC": "{:.3f}",
                "Current vs niedrigere FwdVol": "{:+.3f}",
                "Opposite vs niedrigere FwdVol": "{:+.3f}",
                "Current vs bessere FwdMAE": "{:+.3f}",
                "Opposite vs bessere FwdMAE": "{:+.3f}",
                "Current AUC>0.50 Perioden": "{:.0f}",
                "Opposite AUC>0.50 Perioden": "{:.0f}",
                "P(Opposite>Current)": "{:.1%}",
                "95% CI Low": "{:+.3f}",
                "95% CI High": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Ein Sign-Konflikt wird nur markiert, wenn die Gegenrichtung "
        "mindestens +0.05 AUC gewinnt, in mindestens 2/3 festen Perioden "
        "AUC>0.50 erreicht und der Block-Bootstrap P(Opposite>Current) "
        "mindestens 90 % beträgt. Es wird trotzdem nichts invertiert."
    )

    # --------------------------------------------------------
    # 3. LEVEL VS 20D CHANGE OF EXISTING HEALTH SCORE
    # --------------------------------------------------------

    st.markdown(
        "### 3. R1-Komponenten – Level oder 20D-Health-Veränderung?"
    )

    change_rows = []

    for factor, label in component_labels.items():
        level_score = pd.to_numeric(
            risk_components[
                factor
            ],
            errors="coerce",
        )

        # Positive change means the existing health-oriented factor improved.
        change20 = level_score.diff(
            20
        )

        level_auc = binary_auc(
            level_score.where(
                risk_common
            ),
            targets[
                "Stress_Event_20D"
            ],
            higher_predictor_means_event=False,
        )

        change_auc = binary_auc(
            change20.where(
                risk_common
            ),
            targets[
                "Stress_Event_20D"
            ],
            higher_predictor_means_event=False,
        )

        level_phase, _ = phase_median_stress_auc(
            level_score.where(
                risk_common
            ),
            targets[
                "Stress_Event_20D"
            ],
            horizon=20,
        )

        change_phase, _ = phase_median_stress_auc(
            change20.where(
                risk_common
            ),
            targets[
                "Stress_Event_20D"
            ],
            horizon=20,
        )

        level_vol = safe_spearman(
            level_score.where(
                risk_common
            ),
            -targets[
                "Fwd_Realized_Vol_20D"
            ],
        )

        change_vol = safe_spearman(
            change20.where(
                risk_common
            ),
            -targets[
                "Fwd_Realized_Vol_20D"
            ],
        )

        level_mae = safe_spearman(
            level_score.where(
                risk_common
            ),
            targets[
                "Fwd_MAE_20D"
            ],
        )

        change_mae = safe_spearman(
            change20.where(
                risk_common
            ),
            targets[
                "Fwd_MAE_20D"
            ],
        )

        positive_change_periods = 0

        for _, y1, y2 in GOLD_FIXED_PERIODS:
            period_mask = (
                risk_common
                & (
                    targets.index.year
                    >= y1
                )
                & (
                    targets.index.year
                    <= y2
                )
            )

            p_auc = binary_auc(
                change20.where(
                    period_mask
                ),
                targets[
                    "Stress_Event_20D"
                ],
                higher_predictor_means_event=False,
            )

            if (
                np.isfinite(
                    p_auc
                )
                and p_auc > 0.50
            ):
                positive_change_periods += 1

        auc_improvement = (
            change_auc
            - level_auc
            if (
                np.isfinite(
                    change_auc
                )
                and np.isfinite(
                    level_auc
                )
            )
            else np.nan
        )

        change_candidate = bool(
            np.isfinite(
                auc_improvement
            )
            and auc_improvement
            >= GOLD_RISK_CHANGE_MIN_AUC_IMPROVEMENT
            and positive_change_periods
            >= GOLD_RISK_MIN_FIXED_PERIOD_CONFIRMATIONS
        )

        change_rows.append(
            {
                "Komponente": label,
                "Level AUC": level_auc,
                "Change20 AUC": change_auc,
                "Δ AUC Change−Level": auc_improvement,
                "Level Phase-AUC": level_phase,
                "Change20 Phase-AUC": change_phase,
                "Level vs niedrigere FwdVol": level_vol,
                "Change20 vs niedrigere FwdVol": change_vol,
                "Level vs bessere FwdMAE": level_mae,
                "Change20 vs bessere FwdMAE": change_mae,
                "Change AUC>0.50 Perioden": positive_change_periods,
                "Change-Kandidat": change_candidate,
            }
        )

    gold_r1_level_change_table = pd.DataFrame(
        change_rows
    )

    st.dataframe(
        gold_r1_level_change_table.style.format(
            {
                "Level AUC": "{:.3f}",
                "Change20 AUC": "{:.3f}",
                "Δ AUC Change−Level": "{:+.3f}",
                "Level Phase-AUC": "{:.3f}",
                "Change20 Phase-AUC": "{:.3f}",
                "Level vs niedrigere FwdVol": "{:+.3f}",
                "Change20 vs niedrigere FwdVol": "{:+.3f}",
                "Level vs bessere FwdMAE": "{:+.3f}",
                "Change20 vs bessere FwdMAE": "{:+.3f}",
                "Change AUC>0.50 Perioden": "{:.0f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 4. LEAVE-ONE-COMPONENT-OUT R1
    # --------------------------------------------------------

    st.markdown(
        "### 4. R1 Leave-one-out – welcher Baustein belastet den Risk Context?"
    )

    full_r1_metrics = dual_role_risk_metrics(
        gold_r1_score,
        gold_r1_coverage,
        targets,
        mask=risk_common,
    )

    ablation_rows = []

    for removed_factor, removed_label in component_labels.items():
        reduced_weights = {
            factor: weight
            for factor, weight
            in GOLD_R1_EFFECTIVE_WEIGHTS.items()
            if factor
            != removed_factor
        }

        (
            reduced_score,
            reduced_coverage,
        ) = build_coverage_weighted_context(
            risk_components,
            reduced_weights,
        )

        pair_mask = (
            risk_common
            & reduced_score.notna()
            & (
                reduced_coverage
                >= float(
                    min_coverage
                )
            )
        )

        base_metrics = dual_role_risk_metrics(
            gold_r1_score,
            gold_r1_coverage,
            targets,
            mask=pair_mask,
        )

        reduced_metrics = dual_role_risk_metrics(
            reduced_score,
            reduced_coverage,
            targets,
            mask=pair_mask,
        )

        delta_auc = (
            reduced_metrics[
                "Stress AUC 20D"
            ]
            - base_metrics[
                "Stress AUC 20D"
            ]
            if (
                np.isfinite(
                    reduced_metrics[
                        "Stress AUC 20D"
                    ]
                )
                and np.isfinite(
                    base_metrics[
                        "Stress AUC 20D"
                    ]
                )
            )
            else np.nan
        )

        delta_phase = (
            reduced_metrics[
                "Phase-Median Stress AUC"
            ]
            - base_metrics[
                "Phase-Median Stress AUC"
            ]
            if (
                np.isfinite(
                    reduced_metrics[
                        "Phase-Median Stress AUC"
                    ]
                )
                and np.isfinite(
                    base_metrics[
                        "Phase-Median Stress AUC"
                    ]
                )
            )
            else np.nan
        )

        delta_vol = (
            reduced_metrics[
                "Score vs niedrigere FwdVol"
            ]
            - base_metrics[
                "Score vs niedrigere FwdVol"
            ]
            if (
                np.isfinite(
                    reduced_metrics[
                        "Score vs niedrigere FwdVol"
                    ]
                )
                and np.isfinite(
                    base_metrics[
                        "Score vs niedrigere FwdVol"
                    ]
                )
            )
            else np.nan
        )

        delta_mae = (
            reduced_metrics[
                "Score vs bessere FwdMAE"
            ]
            - base_metrics[
                "Score vs bessere FwdMAE"
            ]
            if (
                np.isfinite(
                    reduced_metrics[
                        "Score vs bessere FwdMAE"
                    ]
                )
                and np.isfinite(
                    base_metrics[
                        "Score vs bessere FwdMAE"
                    ]
                )
            )
            else np.nan
        )

        harm_candidate = bool(
            np.isfinite(
                delta_auc
            )
            and delta_auc
            >= GOLD_RISK_ABLATION_MIN_AUC_IMPROVEMENT
            and (
                not np.isfinite(
                    delta_vol
                )
                or delta_vol
                >= -GOLD_RISK_ABLATION_MAX_VOL_DAMAGE
            )
            and (
                not np.isfinite(
                    delta_mae
                )
                or delta_mae
                >= -GOLD_RISK_ABLATION_MAX_MAE_DAMAGE
            )
        )

        boot = block_bootstrap_auc_difference(
            gold_r1_score.where(
                pair_mask
            ),
            reduced_score.where(
                pair_mask
            ),
            targets[
                "Stress_Event_20D"
            ].where(
                pair_mask
            ),
            block_length=int(
                bootstrap_block
            ),
            n_boot=int(
                bootstrap_runs
            ),
            seed=20260924,
        )

        ablation_rows.append(
            {
                "Entfernte Komponente": removed_label,
                "N Common": int(
                    pair_mask.sum()
                ),
                "Base AUC": base_metrics[
                    "Stress AUC 20D"
                ],
                "Ohne Komponente AUC": reduced_metrics[
                    "Stress AUC 20D"
                ],
                "Δ AUC": delta_auc,
                "Δ Phase-AUC": delta_phase,
                "Δ FwdVol Relation": delta_vol,
                "Δ FwdMAE Relation": delta_mae,
                "P(ohne > R1)": boot[
                    "prob_positive"
                ],
                "95% CI Low": boot[
                    "lower"
                ],
                "95% CI High": boot[
                    "upper"
                ],
                "Harm-Kandidat": harm_candidate,
            }
        )

    gold_r1_ablation_table = (
        pd.DataFrame(
            ablation_rows
        )
        .sort_values(
            "Δ AUC",
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    st.dataframe(
        gold_r1_ablation_table.style.format(
            {
                "N Common": "{:.0f}",
                "Base AUC": "{:.3f}",
                "Ohne Komponente AUC": "{:.3f}",
                "Δ AUC": "{:+.3f}",
                "Δ Phase-AUC": "{:+.3f}",
                "Δ FwdVol Relation": "{:+.3f}",
                "Δ FwdMAE Relation": "{:+.3f}",
                "P(ohne > R1)": "{:.1%}",
                "95% CI Low": "{:+.3f}",
                "95% CI High": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Auch ein klarer Leave-one-out-Befund entfernt noch keinen Faktor. "
        "Er zeigt nur, ob R1 durch genau diesen Baustein historisch belastet "
        "wird, ohne dass dafür neue Gewichte angepasst werden."
    )

    # --------------------------------------------------------
    # 5. STRESS TARGET SENSITIVITY
    # --------------------------------------------------------

    st.markdown(
        "### 5. Ist das -5%-MAE-Stressziel selbst das Problem?"
    )

    gold_base_mae_threshold = float(
        STRESS_MAE_THRESHOLDS[
            "Gold (XAU/USD)"
        ]
    )

    fwd_mae = pd.to_numeric(
        targets[
            "Fwd_MAE_20D"
        ],
        errors="coerce",
    )

    target_rows = []

    for multiplier in GOLD_RISK_MAE_THRESHOLD_MULTIPLIERS:
        threshold = (
            gold_base_mae_threshold
            * float(
                multiplier
            )
        )

        event = (
            fwd_mae
            <= threshold
        ).where(
            fwd_mae.notna()
        )

        event_numeric = pd.to_numeric(
            event,
            errors="coerce",
        )

        r1_auc = binary_auc(
            gold_r1_score.where(
                risk_common
            ),
            event_numeric,
            higher_predictor_means_event=False,
        )

        r1_phase, phases = phase_median_stress_auc(
            gold_r1_score.where(
                risk_common
            ),
            event_numeric,
            horizon=20,
        )

        event_rate = float(
            event_numeric.where(
                risk_common
            ).mean()
        )

        target_rows.append(
            {
                "MAE-Multiplikator": float(
                    multiplier
                ),
                "MAE-Schwelle": threshold,
                "Eventrate": event_rate,
                "R1 Stress AUC": r1_auc,
                "R1 Phase-AUC": r1_phase,
                "AUC-Phasen": phases,
                "AUC > 0.55": bool(
                    np.isfinite(
                        r1_auc
                    )
                    and r1_auc > 0.55
                ),
            }
        )

    gold_r1_target_sensitivity_table = pd.DataFrame(
        target_rows
    )

    st.dataframe(
        gold_r1_target_sensitivity_table.style.format(
            {
                "MAE-Multiplikator": "{:.2f}",
                "MAE-Schwelle": "{:+.2%}",
                "Eventrate": "{:.1%}",
                "R1 Stress AUC": "{:.3f}",
                "R1 Phase-AUC": "{:.3f}",
                "AUC-Phasen": "{:.0f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 6. FUTURE VOLATILITY TARGET AUDIT
    # --------------------------------------------------------

    st.markdown(
        "### 6. Misst R1 eher zukünftige Volatilität als adverse excursion?"
    )

    fwd_vol = pd.to_numeric(
        targets[
            "Fwd_Realized_Vol_20D"
        ],
        errors="coerce",
    )

    vol_common = (
        risk_common
        & fwd_vol.notna()
    )

    high_vol_threshold = float(
        fwd_vol.where(
            vol_common
        ).quantile(
            GOLD_RISK_HIGH_VOL_QUANTILE
        )
    )

    high_vol_event = (
        fwd_vol
        >= high_vol_threshold
    ).where(
        fwd_vol.notna()
    )

    high_vol_event_numeric = pd.to_numeric(
        high_vol_event,
        errors="coerce",
    )

    volatility_rows = []

    for label, score in [
        (
            "R1 · Risk Context",
            gold_r1_score,
        ),
        (
            "CFTC",
            risk_components[
                "cot_noncommercials"
            ],
        ),
        (
            "Fear & Greed",
            risk_components[
                "fear_greed"
            ],
        ),
        (
            "GVZ Health",
            risk_components[
                "vix_score"
            ],
        ),
        (
            "Credit Health",
            risk_components[
                "credit_spreads"
            ],
        ),
        (
            "MOVE Health",
            risk_components[
                "move_index"
            ],
        ),
    ]:
        s = pd.to_numeric(
            score,
            errors="coerce",
        ).where(
            vol_common
        )

        auc = binary_auc(
            s,
            high_vol_event_numeric,
            higher_predictor_means_event=False,
        )

        phase_auc, phases = phase_median_stress_auc(
            s,
            high_vol_event_numeric,
            horizon=20,
        )

        volatility_rows.append(
            {
                "Output": label,
                "High-Vol-Schwelle": high_vol_threshold,
                "High-Vol-Eventrate": float(
                    high_vol_event_numeric.where(
                        vol_common
                    ).mean()
                ),
                "High-Vol AUC": auc,
                "High-Vol Phase-AUC": phase_auc,
                "AUC-Phasen": phases,
                "Spearman vs niedrigere FwdVol": safe_spearman(
                    s,
                    -fwd_vol,
                ),
            }
        )

    gold_r1_volatility_target_table = pd.DataFrame(
        volatility_rows
    )

    st.dataframe(
        gold_r1_volatility_target_table.style.format(
            {
                "High-Vol-Schwelle": "{:.2%}",
                "High-Vol-Eventrate": "{:.1%}",
                "High-Vol AUC": "{:.3f}",
                "High-Vol Phase-AUC": "{:.3f}",
                "AUC-Phasen": "{:.0f}",
                "Spearman vs niedrigere FwdVol": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 7. STRUCTURAL VERDICT
    # --------------------------------------------------------

    st.markdown(
        "### 7. Strukturelles R1-Urteil"
    )

    sign_conflicts = int(
        gold_r1_component_orientation_table[
            "Struktureller Sign-Konflikt"
        ]
        .astype(
            bool
        )
        .sum()
    )

    change_candidates = int(
        gold_r1_level_change_table[
            "Change-Kandidat"
        ]
        .astype(
            bool
        )
        .sum()
    )

    harm_candidates = int(
        gold_r1_ablation_table[
            "Harm-Kandidat"
        ]
        .astype(
            bool
        )
        .sum()
    )

    target_auc_above_055 = int(
        gold_r1_target_sensitivity_table[
            "AUC > 0.55"
        ]
        .astype(
            bool
        )
        .sum()
    )

    r1_vol_row = gold_r1_volatility_target_table[
        gold_r1_volatility_target_table[
            "Output"
        ]
        == "R1 · Risk Context"
    ].iloc[
        0
    ]

    r1_high_vol_auc = float(
        r1_vol_row[
            "High-Vol AUC"
        ]
    )

    r1_is_volatility_context = bool(
        np.isfinite(
            r1_high_vol_auc
        )
        and r1_high_vol_auc
        >= GOLD_RISK_VOL_STATE_MIN_AUC
        and float(
            full_r1_metrics[
                "Stress AUC 20D"
            ]
        ) < GOLD_R1_MIN_STRESS_AUC
    )

    # D1 must remain frozen regardless of the R1 outcome.
    d1_frozen_ok = bool(
        np.isfinite(
            d1_reference[
                "IC 20D"
            ]
        )
        and float(
            d1_reference[
                "IC 20D"
            ]
        )
        >= GOLD_D1_MIN_IC20
        and np.isfinite(
            d1_reference[
                "Non-Overlap IC20"
            ]
        )
        and float(
            d1_reference[
                "Non-Overlap IC20"
            ]
        )
        > GOLD_D1_MIN_NONOVERLAP_IC20
    )

    gold_r1_structural_gate_table = pd.DataFrame(
        [
            {
                "Prüfung": "D1 Freeze bleibt valide",
                "Status": (
                    "✅ FROZEN"
                    if d1_frozen_ok
                    else "⚠️ Referenzproblem"
                ),
                "Messwert": (
                    f"IC20 {d1_reference['IC 20D']:+.3f} · "
                    f"NO20 {d1_reference['Non-Overlap IC20']:+.3f}"
                ),
            },
            {
                "Prüfung": "R1 stabile Sign-Konflikte",
                "Status": (
                    "⚠️ untersuchen"
                    if sign_conflicts > 0
                    else "✅ keine"
                ),
                "Messwert": str(
                    sign_conflicts
                ),
            },
            {
                "Prüfung": "R1 Level→Change Kandidaten",
                "Status": (
                    "🧪 untersuchen"
                    if change_candidates > 0
                    else "— keine"
                ),
                "Messwert": str(
                    change_candidates
                ),
            },
            {
                "Prüfung": "R1 Leave-one-out Harm-Kandidaten",
                "Status": (
                    "⚠️ untersuchen"
                    if harm_candidates > 0
                    else "✅ keine"
                ),
                "Messwert": str(
                    harm_candidates
                ),
            },
            {
                "Prüfung": "R1 MAE-Zielrobustheit",
                "Status": (
                    "✅ relativ robust"
                    if target_auc_above_055 >= 2
                    else "⚠️ schwach"
                ),
                "Messwert": (
                    f"AUC>0.55 bei "
                    f"{target_auc_above_055}/"
                    f"{len(GOLD_RISK_MAE_THRESHOLD_MULTIPLIERS)} "
                    f"Schwellen"
                ),
            },
            {
                "Prüfung": "R1 ist eher Volatility-State als MAE-Stress",
                "Status": (
                    "🟡 JA"
                    if r1_is_volatility_context
                    else "— nicht bestätigt"
                ),
                "Messwert": (
                    f"High-Vol AUC {r1_high_vol_auc:.3f} · "
                    f"MAE-Stress AUC "
                    f"{full_r1_metrics['Stress AUC 20D']:.3f}"
                ),
            },
            {
                "Prüfung": "R2 jetzt automatisch bauen",
                "Status": "❌ NEIN",
                "Messwert": (
                    "erst strukturelle Befunde auswerten"
                ),
            },
        ]
    )

    st.dataframe(
        gold_r1_structural_gate_table,
        hide_index=True,
        use_container_width=True,
    )

    if (
        sign_conflicts > 0
        or change_candidates > 0
        or harm_candidates > 0
    ):
        st.warning(
            "🟠 **R1 HAT KONKRETE STRUKTURELLE REPARATURKANDIDATEN.** "
            "D1 bleibt eingefroren. Aus den eindeutig bestätigten R1-Befunden "
            "kann als nächster Schritt genau **ein vorab definierter R2-"
            "Challenger** abgeleitet werden – ohne freie Gewichtssuche."
        )
    elif r1_is_volatility_context:
        st.warning(
            "🟡 **R1 IST EHER EIN VOLATILITY-STATE-CONTEXT ALS EIN "
            "MAE-STRESS-CONTEXT.** Dann sollten wir R1 nicht gewaltsam auf "
            "das falsche Ziel optimieren, sondern die Risk-Rolle semantisch "
            "trennen."
        )
    elif target_auc_above_055 < 2:
        st.error(
            "🔴 **R1 BLEIBT SCHWACH UND DAS MAE-STRESSZIEL IST NICHT "
            "ROBUST GENUG.** Dann wäre ein neuer R2-Challenger auf dieser "
            "Basis methodisch nicht gerechtfertigt; zuerst müsste die "
            "Gold-Risk-Zieldefinition überarbeitet werden."
        )
    else:
        st.error(
            "🔴 **R1 IST SCHWACH, ABER ES GIBT KEINE KLARE EINZELURSACHE.** "
            "Wir bauen deshalb nicht durch nachträgliches Feintuning weiter."
        )

    st.info(
        "Research-Regel v1.0.23: D1 wird nicht verändert. "
        "Sign-Flip, Change-Transformation, Faktorentfernung oder ein neues "
        "R2-Modell werden aus diesem Audit nicht automatisch umgesetzt."
    )

    # --------------------------------------------------------
    # 8. EXPORT FRAME
    # --------------------------------------------------------

    gold_r1_research_export = pd.DataFrame(
        index=raw_df.index
    )

    gold_r1_research_export[
        "asset_price"
    ] = pd.to_numeric(
        raw_df[
            "asset_price"
        ],
        errors="coerce",
    )

    gold_r1_research_export[
        "d1_direction_score_FROZEN"
    ] = gold_d1_score

    gold_r1_research_export[
        "d1_coverage_FROZEN"
    ] = gold_d1_coverage

    gold_r1_research_export[
        "r1_risk_score"
    ] = gold_r1_score

    gold_r1_research_export[
        "r1_coverage"
    ] = gold_r1_coverage

    for factor, label in component_labels.items():
        gold_r1_research_export[
            f"r1_component__{factor}"
        ] = risk_components[
            factor
        ]

        gold_r1_research_export[
            f"r1_component_change20__{factor}"
        ] = pd.to_numeric(
            risk_components[
                factor
            ],
            errors="coerce",
        ).diff(
            20
        )

    for col in [
        "Fwd_Return_5D",
        "Fwd_Return_20D",
        "Fwd_Return_60D",
        "Fwd_MAE_20D",
        "Fwd_Realized_Vol_20D",
        "Stress_Event_20D",
    ]:
        gold_r1_research_export[
            col
        ] = targets[
            col
        ]

    gold_r1_research_export[
        "risk_common"
    ] = risk_common.astype(
        bool
    )



# ============================================================
# 19G6. GOLD VOLATILITY-STATE ROBUSTNESS AUDIT
# ============================================================

gold_vol_d1_frozen_table = pd.DataFrame()
gold_vol_target_table = pd.DataFrame()
gold_vol_model_table = pd.DataFrame()
gold_vol_period_table = pd.DataFrame()
gold_vol_2017_2020_table = pd.DataFrame()
gold_vol_move_table = pd.DataFrame()
gold_vol_quintile_table = pd.DataFrame()
gold_vol_gate_table = pd.DataFrame()
gold_vol_research_export = pd.DataFrame()

if selected_asset == "Gold (XAU/USD)":
    st.markdown("---")
    st.subheader(
        "🌡️ Gold Volatility-State Robustness Audit"
    )

    st.caption(
        "D1 bleibt eingefroren. Geprüft wird ausschließlich, ob R1 "
        "bzw. eine minimal veränderte MOVE-Variante zukünftige "
        "Gold-Volatilitätsregime robust erkennt. Das High-Vol-Ziel ist "
        "vollständig PIT-sicher: jede Schwelle wird nur aus der Vergangenheit "
        "berechnet."
    )

    # --------------------------------------------------------
    # 1. D1 FREEZE SNAPSHOT
    # --------------------------------------------------------

    st.markdown(
        "### 1. D1 bleibt eingefroren"
    )

    gold_vol_d1_frozen_table = gold_r1_frozen_d1_table.copy()

    st.dataframe(
        gold_vol_d1_frozen_table.style.format(
            {
                "IC 20D": "{:+.3f}",
                "Non-Overlap IC20": "{:+.3f}",
                "Direction 20D": "{:.1%}",
                "Positive feste Perioden": "{:.0f}",
                "Positive Jahresrate": "{:.1%}",
                "P(D1>Current)": "{:.1%}",
                "P(D1>G1)": "{:.1%}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 2. PIT-SAFE HIGH-VOL TARGET
    # --------------------------------------------------------

    st.markdown(
        "### 2. PIT-sicheres High-Vol-Ziel"
    )

    fwd_vol = pd.to_numeric(
        targets[
            "Fwd_Realized_Vol_20D"
        ],
        errors="coerce",
    )

    # PIT threshold: shift(1) guarantees the threshold at t uses only
    # historical realized-vol observations prior to t.
    pit_high_vol_threshold = (
        fwd_vol.shift(
            1
        )
        .rolling(
            GOLD_VOL_PIT_LOOKBACK,
            min_periods=GOLD_VOL_PIT_MIN_PERIODS,
        )
        .quantile(
            GOLD_VOL_PIT_QUANTILE
        )
    )

    pit_high_vol_event = (
        fwd_vol
        >= pit_high_vol_threshold
    ).where(
        fwd_vol.notna()
        & pit_high_vol_threshold.notna()
    )

    pit_high_vol_event_numeric = pd.to_numeric(
        pit_high_vol_event,
        errors="coerce",
    )

    target_valid = (
        pit_high_vol_event_numeric.notna()
        & fwd_vol.notna()
    )

    gold_vol_target_table = pd.DataFrame(
        [
            {
                "Target": "PIT High-Vol 20D",
                "Trailing Lookback": GOLD_VOL_PIT_LOOKBACK,
                "Min Periods": GOLD_VOL_PIT_MIN_PERIODS,
                "Quantil": GOLD_VOL_PIT_QUANTILE,
                "N": int(
                    target_valid.sum()
                ),
                "Eventrate": float(
                    pit_high_vol_event_numeric.where(
                        target_valid
                    ).mean()
                ),
                "Ø PIT-Schwelle": float(
                    pit_high_vol_threshold.where(
                        target_valid
                    ).mean()
                ),
                "Letzte PIT-Schwelle": (
                    float(
                        pit_high_vol_threshold.dropna().iloc[
                            -1
                        ]
                    )
                    if pit_high_vol_threshold.notna().any()
                    else np.nan
                ),
            }
        ]
    )

    st.dataframe(
        gold_vol_target_table.style.format(
            {
                "Trailing Lookback": "{:.0f}",
                "Min Periods": "{:.0f}",
                "Quantil": "{:.0%}",
                "N": "{:.0f}",
                "Eventrate": "{:.1%}",
                "Ø PIT-Schwelle": "{:.2%}",
                "Letzte PIT-Schwelle": "{:.2%}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Die Schwelle wird für jeden Zeitpunkt separat aus den vorherigen "
        "756 Beobachtungen berechnet. Durch `shift(1)` fließt die aktuelle "
        "oder zukünftige Volatilität niemals in ihre eigene Schwelle ein."
    )

    # --------------------------------------------------------
    # 3. BUILD STRUCTURAL MOVE VARIANTS
    # --------------------------------------------------------

    r1_weights_no_fear = {
        factor: weight
        for factor, weight
        in GOLD_R1_EFFECTIVE_WEIGHTS.items()
        if factor != "fear_greed"
    }

    # Base R1 as already defined in v1.0.22.
    base_r1_score = gold_r1_score.copy()
    base_r1_coverage = gold_r1_coverage.copy()

    # Structural V-base excludes Fear & Greed because of insufficient history.
    (
        vbase_score,
        vbase_coverage,
    ) = build_coverage_weighted_context(
        risk_components,
        r1_weights_no_fear,
    )

    # MOVE inverted.
    risk_components_move_inverted = risk_components.copy()
    risk_components_move_inverted[
        "move_index"
    ] = (
        100.0
        - pd.to_numeric(
            risk_components[
                "move_index"
            ],
            errors="coerce",
        )
    )

    (
        vmove_inv_score,
        vmove_inv_coverage,
    ) = build_coverage_weighted_context(
        risk_components_move_inverted,
        r1_weights_no_fear,
    )

    # MOVE removed.
    vmove_removed_weights = {
        factor: weight
        for factor, weight
        in r1_weights_no_fear.items()
        if factor != "move_index"
    }

    (
        vmove_removed_score,
        vmove_removed_coverage,
    ) = build_coverage_weighted_context(
        risk_components,
        vmove_removed_weights,
    )

    # --------------------------------------------------------
    # 4. MAIN VOLATILITY-STATE COMPARISON
    # --------------------------------------------------------

    st.markdown(
        "### 3. Volatility-State – R1 und strukturelle MOVE-Varianten"
    )

    vol_candidates = {
        "R1 Original": (
            base_r1_score,
            base_r1_coverage,
        ),
        "V-Base ohne Fear&Greed": (
            vbase_score,
            vbase_coverage,
        ),
        "V-MOVE invertiert": (
            vmove_inv_score,
            vmove_inv_coverage,
        ),
        "V-MOVE entfernt": (
            vmove_removed_score,
            vmove_removed_coverage,
        ),
        "GVZ Health": (
            risk_components[
                "vix_score"
            ],
            pd.Series(
                100.0,
                index=raw_df.index,
            ),
        ),
        "CFTC": (
            risk_components[
                "cot_noncommercials"
            ],
            pd.Series(
                100.0,
                index=raw_df.index,
            ),
        ),
        "Credit Health": (
            risk_components[
                "credit_spreads"
            ],
            pd.Series(
                100.0,
                index=raw_df.index,
            ),
        ),
    }

    vol_rows = []

    for label, (
        score,
        coverage,
    ) in vol_candidates.items():
        score = pd.to_numeric(
            score,
            errors="coerce",
        )

        coverage = pd.to_numeric(
            coverage,
            errors="coerce",
        )

        common = (
            target_valid
            & score.notna()
            & (
                coverage
                >= float(
                    min_coverage
                )
            )
        )

        s = score.where(
            common
        )

        auc = binary_auc(
            s,
            pit_high_vol_event_numeric,
            higher_predictor_means_event=False,
        )

        phase_auc, phases = phase_median_stress_auc(
            s,
            pit_high_vol_event_numeric,
            horizon=20,
        )

        vol_rel = safe_spearman(
            s,
            -fwd_vol,
        )

        frame = pd.DataFrame(
            {
                "score": s,
                "event": pit_high_vol_event_numeric,
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

            highvol_gap = (
                low_rate
                - high_rate
            )
        else:
            low_rate = np.nan
            high_rate = np.nan
            highvol_gap = np.nan

        vol_rows.append(
            {
                "Modell": label,
                "N": int(
                    common.sum()
                ),
                "High-Vol AUC": auc,
                "Phase-Median AUC": phase_auc,
                "AUC-Phasen": phases,
                "Q20 High-Vol Rate": low_rate,
                "Q80 High-Vol Rate": high_rate,
                "Q20−Q80 High-Vol Gap": highvol_gap,
                "Score vs niedrigere FwdVol": vol_rel,
                "Ø Coverage": float(
                    coverage.where(
                        common
                    ).mean()
                ),
            }
        )

    gold_vol_model_table = pd.DataFrame(
        vol_rows
    )

    st.dataframe(
        gold_vol_model_table.style.format(
            {
                "N": "{:.0f}",
                "High-Vol AUC": "{:.3f}",
                "Phase-Median AUC": "{:.3f}",
                "AUC-Phasen": "{:.0f}",
                "Q20 High-Vol Rate": "{:.1%}",
                "Q80 High-Vol Rate": "{:.1%}",
                "Q20−Q80 High-Vol Gap": "{:+.1%}",
                "Score vs niedrigere FwdVol": "{:+.3f}",
                "Ø Coverage": "{:.1f}%",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 5. FIXED PERIOD ROBUSTNESS
    # --------------------------------------------------------

    st.markdown(
        "### 4. Feste Teilperioden – insbesondere 2017–2020"
    )

    period_rows = []

    structural_vol_models = {
        "R1 Original": (
            base_r1_score,
            base_r1_coverage,
        ),
        "V-Base ohne Fear&Greed": (
            vbase_score,
            vbase_coverage,
        ),
        "V-MOVE invertiert": (
            vmove_inv_score,
            vmove_inv_coverage,
        ),
        "V-MOVE entfernt": (
            vmove_removed_score,
            vmove_removed_coverage,
        ),
    }

    for period_label, y1, y2 in GOLD_FIXED_PERIODS:
        period_mask = (
            target_valid
            & (
                targets.index.year
                >= y1
            )
            & (
                targets.index.year
                <= y2
            )
        )

        for label, (
            score,
            coverage,
        ) in structural_vol_models.items():
            common = (
                period_mask
                & pd.to_numeric(
                    score,
                    errors="coerce",
                ).notna()
                & (
                    pd.to_numeric(
                        coverage,
                        errors="coerce",
                    )
                    >= float(
                        min_coverage
                    )
                )
            )

            s = pd.to_numeric(
                score,
                errors="coerce",
            ).where(
                common
            )

            auc = binary_auc(
                s,
                pit_high_vol_event_numeric,
                higher_predictor_means_event=False,
            )

            phase_auc, phases = phase_median_stress_auc(
                s,
                pit_high_vol_event_numeric,
                horizon=20,
            )

            period_rows.append(
                {
                    "Periode": period_label,
                    "Modell": label,
                    "N": int(
                        common.sum()
                    ),
                    "High-Vol AUC": auc,
                    "Phase-Median AUC": phase_auc,
                    "AUC-Phasen": phases,
                    "Score vs niedrigere FwdVol": safe_spearman(
                        s,
                        -fwd_vol,
                    ),
                    "AUC > 0.50": bool(
                        np.isfinite(
                            auc
                        )
                        and auc
                        > GOLD_VSTATE_MIN_PERIOD_AUC
                    ),
                }
            )

    gold_vol_period_table = pd.DataFrame(
        period_rows
    )

    st.dataframe(
        gold_vol_period_table.style.format(
            {
                "N": "{:.0f}",
                "High-Vol AUC": "{:.3f}",
                "Phase-Median AUC": "{:.3f}",
                "AUC-Phasen": "{:.0f}",
                "Score vs niedrigere FwdVol": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    gold_vol_2017_2020_table = gold_vol_period_table[
        gold_vol_period_table[
            "Periode"
        ]
        == "2017–2020"
    ].reset_index(
        drop=True
    )

    st.markdown(
        "#### Fokus: 2017–2020"
    )

    st.dataframe(
        gold_vol_2017_2020_table.style.format(
            {
                "N": "{:.0f}",
                "High-Vol AUC": "{:.3f}",
                "Phase-Median AUC": "{:.3f}",
                "AUC-Phasen": "{:.0f}",
                "Score vs niedrigere FwdVol": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 6. MOVE CURRENT VS INVERTED VS REMOVED
    # --------------------------------------------------------

    st.markdown(
        "### 5. MOVE – Current vs. invertiert vs. entfernt"
    )

    move_rows = []

    move_variant_defs = {
        "MOVE Current": (
            vbase_score,
            vbase_coverage,
        ),
        "MOVE invertiert": (
            vmove_inv_score,
            vmove_inv_coverage,
        ),
        "MOVE entfernt": (
            vmove_removed_score,
            vmove_removed_coverage,
        ),
    }

    base_move_score = pd.to_numeric(
        vbase_score,
        errors="coerce",
    )

    base_move_coverage = pd.to_numeric(
        vbase_coverage,
        errors="coerce",
    )

    for label, (
        score,
        coverage,
    ) in move_variant_defs.items():
        score = pd.to_numeric(
            score,
            errors="coerce",
        )

        coverage = pd.to_numeric(
            coverage,
            errors="coerce",
        )

        common = (
            target_valid
            & base_move_score.notna()
            & score.notna()
            & (
                base_move_coverage
                >= float(
                    min_coverage
                )
            )
            & (
                coverage
                >= float(
                    min_coverage
                )
            )
        )

        base_auc = binary_auc(
            base_move_score.where(
                common
            ),
            pit_high_vol_event_numeric,
            higher_predictor_means_event=False,
        )

        variant_auc = binary_auc(
            score.where(
                common
            ),
            pit_high_vol_event_numeric,
            higher_predictor_means_event=False,
        )

        base_phase, _ = phase_median_stress_auc(
            base_move_score.where(
                common
            ),
            pit_high_vol_event_numeric,
            horizon=20,
        )

        variant_phase, _ = phase_median_stress_auc(
            score.where(
                common
            ),
            pit_high_vol_event_numeric,
            horizon=20,
        )

        boot = block_bootstrap_auc_difference(
            base_move_score.where(
                common
            ),
            score.where(
                common
            ),
            pit_high_vol_event_numeric.where(
                common
            ),
            block_length=int(
                bootstrap_block
            ),
            n_boot=int(
                bootstrap_runs
            ),
            seed=20260924,
        )

        move_rows.append(
            {
                "MOVE-Variante": label,
                "N": int(
                    common.sum()
                ),
                "Base AUC": base_auc,
                "Variante AUC": variant_auc,
                "Δ AUC vs MOVE Current": (
                    variant_auc
                    - base_auc
                    if (
                        np.isfinite(
                            variant_auc
                        )
                        and np.isfinite(
                            base_auc
                        )
                    )
                    else np.nan
                ),
                "Base Phase-AUC": base_phase,
                "Variante Phase-AUC": variant_phase,
                "P(Variante>Current)": boot[
                    "prob_positive"
                ],
                "95% CI Low": boot[
                    "lower"
                ],
                "95% CI High": boot[
                    "upper"
                ],
            }
        )

    gold_vol_move_table = pd.DataFrame(
        move_rows
    )

    st.dataframe(
        gold_vol_move_table.style.format(
            {
                "N": "{:.0f}",
                "Base AUC": "{:.3f}",
                "Variante AUC": "{:.3f}",
                "Δ AUC vs MOVE Current": "{:+.3f}",
                "Base Phase-AUC": "{:.3f}",
                "Variante Phase-AUC": "{:.3f}",
                "P(Variante>Current)": "{:.1%}",
                "95% CI Low": "{:+.3f}",
                "95% CI High": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 7. QUINTILE MONOTONICITY
    # --------------------------------------------------------

    st.markdown(
        "### 6. Volatility-State Quintile – fällt High-Vol-Risiko mit dem Score?"
    )

    quintile_rows = []

    for label, (
        score,
        coverage,
    ) in structural_vol_models.items():
        score = pd.to_numeric(
            score,
            errors="coerce",
        )

        coverage = pd.to_numeric(
            coverage,
            errors="coerce",
        )

        common = (
            target_valid
            & score.notna()
            & (
                coverage
                >= float(
                    min_coverage
                )
            )
        )

        frame = pd.DataFrame(
            {
                "score": score.where(
                    common
                ),
                "event": pit_high_vol_event_numeric.where(
                    common
                ),
            }
        ).dropna()

        if len(
            frame
        ) < 100:
            quintile_rows.append(
                {
                    "Modell": label,
                    "N": int(
                        len(
                            frame
                        )
                    ),
                    "Q1 High-Vol Rate": np.nan,
                    "Q5 High-Vol Rate": np.nan,
                    "Q1−Q5 Gap": np.nan,
                    "Monotonie": np.nan,
                }
            )
            continue

        try:
            frame[
                "q"
            ] = pd.qcut(
                frame[
                    "score"
                ].rank(
                    method="first"
                ),
                5,
                labels=[
                    "Q1",
                    "Q2",
                    "Q3",
                    "Q4",
                    "Q5",
                ],
            )

            rates = frame.groupby(
                "q",
                observed=False,
            )[
                "event"
            ].mean()

            q1 = float(
                rates.get(
                    "Q1",
                    np.nan,
                )
            )

            q5 = float(
                rates.get(
                    "Q5",
                    np.nan,
                )
            )

            gap = (
                q1
                - q5
                if (
                    np.isfinite(
                        q1
                    )
                    and np.isfinite(
                        q5
                    )
                )
                else np.nan
            )

            rate_values = rates.reindex(
                [
                    "Q1",
                    "Q2",
                    "Q3",
                    "Q4",
                    "Q5",
                ]
            ).to_numpy(
                dtype=float
            )

            monotonic = (
                float(
                    spearmanr(
                        np.arange(
                            1,
                            6,
                        ),
                        -rate_values,
                    ).statistic
                )
                if np.isfinite(
                    rate_values
                ).sum()
                >= 3
                else np.nan
            )
        except Exception:
            q1 = np.nan
            q5 = np.nan
            gap = np.nan
            monotonic = np.nan

        quintile_rows.append(
            {
                "Modell": label,
                "N": int(
                    len(
                        frame
                    )
                ),
                "Q1 High-Vol Rate": q1,
                "Q5 High-Vol Rate": q5,
                "Q1−Q5 Gap": gap,
                "Monotonie": monotonic,
            }
        )

    gold_vol_quintile_table = pd.DataFrame(
        quintile_rows
    )

    st.dataframe(
        gold_vol_quintile_table.style.format(
            {
                "N": "{:.0f}",
                "Q1 High-Vol Rate": "{:.1%}",
                "Q5 High-Vol Rate": "{:.1%}",
                "Q1−Q5 Gap": "{:+.1%}",
                "Monotonie": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 8. PRE-REGISTERED ROBUSTNESS GATES
    # --------------------------------------------------------

    st.markdown(
        "### 7. Vorab festgelegte Volatility-State-Gates"
    )

    def _row_for(model_name):
        return gold_vol_model_table[
            gold_vol_model_table[
                "Modell"
            ]
            == model_name
        ].iloc[
            0
        ]

    candidate_rows = {
        "R1 Original": _row_for(
            "R1 Original"
        ),
        "V-Base ohne Fear&Greed": _row_for(
            "V-Base ohne Fear&Greed"
        ),
        "V-MOVE invertiert": _row_for(
            "V-MOVE invertiert"
        ),
        "V-MOVE entfernt": _row_for(
            "V-MOVE entfernt"
        ),
    }

    # Candidate is selected by fixed structural priority, not by maximizing
    # outcome metrics:
    # 1) MOVE inverted if it passes the pre-registered MOVE structural test.
    # 2) otherwise MOVE removed if it passes.
    # 3) otherwise V-Base.
    move_inv_row = gold_vol_move_table[
        gold_vol_move_table[
            "MOVE-Variante"
        ]
        == "MOVE invertiert"
    ].iloc[
        0
    ]

    move_removed_row = gold_vol_move_table[
        gold_vol_move_table[
            "MOVE-Variante"
        ]
        == "MOVE entfernt"
    ].iloc[
        0
    ]

    move_inv_structural = bool(
        np.isfinite(
            move_inv_row[
                "Δ AUC vs MOVE Current"
            ]
        )
        and move_inv_row[
            "Δ AUC vs MOVE Current"
        ]
        >= GOLD_MOVE_MIN_AUC_IMPROVEMENT
        and np.isfinite(
            move_inv_row[
                "P(Variante>Current)"
            ]
        )
        and move_inv_row[
            "P(Variante>Current)"
        ]
        >= GOLD_MOVE_MIN_BOOTSTRAP_PROB
    )

    move_removed_structural = bool(
        np.isfinite(
            move_removed_row[
                "Δ AUC vs MOVE Current"
            ]
        )
        and move_removed_row[
            "Δ AUC vs MOVE Current"
        ]
        >= GOLD_MOVE_MIN_AUC_IMPROVEMENT
        and np.isfinite(
            move_removed_row[
                "P(Variante>Current)"
            ]
        )
        and move_removed_row[
            "P(Variante>Current)"
        ]
        >= GOLD_MOVE_MIN_BOOTSTRAP_PROB
    )

    if move_inv_structural:
        selected_v_name = "V-MOVE invertiert"
    elif move_removed_structural:
        selected_v_name = "V-MOVE entfernt"
    else:
        selected_v_name = "V-Base ohne Fear&Greed"

    selected_v = candidate_rows[
        selected_v_name
    ]

    selected_periods = gold_vol_period_table[
        gold_vol_period_table[
            "Modell"
        ]
        == selected_v_name
    ]

    positive_periods = int(
        selected_periods[
            "AUC > 0.50"
        ]
        .astype(
            bool
        )
        .sum()
    )

    period_2017_2020 = selected_periods[
        selected_periods[
            "Periode"
        ]
        == "2017–2020"
    ]

    auc_2017_2020 = (
        float(
            period_2017_2020.iloc[
                0
            ][
                "High-Vol AUC"
            ]
        )
        if not period_2017_2020.empty
        else np.nan
    )

    selected_quintile = gold_vol_quintile_table[
        gold_vol_quintile_table[
            "Modell"
        ]
        == selected_v_name
    ]

    q_gap = (
        float(
            selected_quintile.iloc[
                0
            ][
                "Q1−Q5 Gap"
            ]
        )
        if not selected_quintile.empty
        else np.nan
    )

    gate_rows = [
        {
            "Prüfung": "D1 bleibt eingefroren",
            "Erfüllt": True,
            "Messwert": (
                f"IC20 "
                f"{float(d1_reference['IC 20D']):+.3f}"
            ),
        },
        {
            "Prüfung": "Fear & Greed aus strukturellem V-Kandidaten ausgeschlossen",
            "Erfüllt": bool(
                GOLD_VSTATE_EXCLUDE_FEAR_GREED
            ),
            "Messwert": "JA",
        },
        {
            "Prüfung": "Selected V-State AUC ≥0.60",
            "Erfüllt": bool(
                np.isfinite(
                    selected_v[
                        "High-Vol AUC"
                    ]
                )
                and selected_v[
                    "High-Vol AUC"
                ]
                >= GOLD_VSTATE_MIN_AUC
            ),
            "Messwert": (
                f"{selected_v['High-Vol AUC']:.3f}"
            ),
        },
        {
            "Prüfung": "Selected Phase-AUC ≥0.55",
            "Erfüllt": bool(
                np.isfinite(
                    selected_v[
                        "Phase-Median AUC"
                    ]
                )
                and selected_v[
                    "Phase-Median AUC"
                ]
                >= GOLD_VSTATE_MIN_PHASE_AUC
            ),
            "Messwert": (
                f"{selected_v['Phase-Median AUC']:.3f}"
            ),
        },
        {
            "Prüfung": "AUC >0.50 in mindestens 2/3 festen Perioden",
            "Erfüllt": bool(
                positive_periods
                >= GOLD_VSTATE_MIN_POSITIVE_FIXED_PERIODS
            ),
            "Messwert": (
                f"{positive_periods}/3"
            ),
        },
        {
            "Prüfung": "2017–2020 AUC >0.50",
            "Erfüllt": bool(
                np.isfinite(
                    auc_2017_2020
                )
                and auc_2017_2020
                > GOLD_VSTATE_MIN_PERIOD_AUC
            ),
            "Messwert": (
                f"{auc_2017_2020:.3f}"
                if np.isfinite(
                    auc_2017_2020
                )
                else "n/a"
            ),
        },
        {
            "Prüfung": "Q1−Q5 High-Vol-Gap ≥5pp",
            "Erfüllt": bool(
                np.isfinite(
                    q_gap
                )
                and q_gap
                >= GOLD_VSTATE_MIN_HIGHVOL_GAP
            ),
            "Messwert": (
                f"{q_gap:+.1%}"
                if np.isfinite(
                    q_gap
                )
                else "n/a"
            ),
        },
        {
            "Prüfung": "Score korreliert ≥+0.10 mit niedrigerer FwdVol",
            "Erfüllt": bool(
                np.isfinite(
                    selected_v[
                        "Score vs niedrigere FwdVol"
                    ]
                )
                and selected_v[
                    "Score vs niedrigere FwdVol"
                ]
                >= GOLD_VSTATE_MIN_VOL_SPEARMAN
            ),
            "Messwert": (
                f"{selected_v['Score vs niedrigere FwdVol']:+.3f}"
            ),
        },
    ]

    gold_vol_gate_table = pd.DataFrame(
        gate_rows
    )

    gold_vol_gate_table[
        "Status"
    ] = gold_vol_gate_table[
        "Erfüllt"
    ].map(
        {
            True: "✅",
            False: "❌",
        }
    )

    st.dataframe(
        gold_vol_gate_table[
            [
                "Prüfung",
                "Status",
                "Messwert",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    passed = int(
        gold_vol_gate_table[
            "Erfüllt"
        ]
        .astype(
            bool
        )
        .sum()
    )

    total = int(
        len(
            gold_vol_gate_table
        )
    )

    st.markdown(
        "### 8. Volatility-State-Urteil"
    )

    st.write(
        f"Strukturell ausgewählter Kandidat: **{selected_v_name}**"
    )

    if passed == total:
        st.success(
            "🟢 **GOLD VOLATILITY-STATE HISTORISCH BESTÄTIGT.** "
            "Der PIT-sichere High-Vol-Test, die festen Perioden inklusive "
            "2017–2020 sowie Quintile und FwdVol-Bezug tragen den "
            "vorab strukturell ausgewählten Kandidaten. Nächster Schritt "
            "wäre ein eingefrorener **D1 + V1 Dual-Output Walk-Forward**."
        )
    elif passed >= total - 1:
        st.warning(
            f"🟡 **VOLATILITY-STATE FAST BESTÄTIGT ({passed}/{total}).** "
            "Wir untersuchen genau das eine fehlende Gate, ohne danach "
            "Schwellen oder Gewichte anzupassen."
        )
    else:
        st.error(
            f"🔴 **VOLATILITY-STATE NOCH NICHT ROBUST GENUG "
            f"({passed}/{total}).** "
            "Dann bauen wir keinen V1-Challenger aus dieser Historie."
        )

    st.info(
        "Research-Regel v1.0.24: Fear & Greed wird wegen der kurzen Historie "
        "nicht Bestandteil eines strukturellen V1. MOVE wird nur dann als "
        "invertiert/entfernt ausgewählt, wenn die vorab definierte AUC- und "
        "Bootstrap-Hürde erfüllt ist. D1 bleibt unverändert."
    )

    # --------------------------------------------------------
    # 9. EXPORT FRAME
    # --------------------------------------------------------

    gold_vol_research_export = pd.DataFrame(
        index=raw_df.index
    )

    gold_vol_research_export[
        "asset_price"
    ] = pd.to_numeric(
        raw_df[
            "asset_price"
        ],
        errors="coerce",
    )

    gold_vol_research_export[
        "d1_direction_score_FROZEN"
    ] = gold_d1_score

    gold_vol_research_export[
        "r1_original"
    ] = base_r1_score

    gold_vol_research_export[
        "vbase_no_fear_greed"
    ] = vbase_score

    gold_vol_research_export[
        "vmove_inverted"
    ] = vmove_inv_score

    gold_vol_research_export[
        "vmove_removed"
    ] = vmove_removed_score

    gold_vol_research_export[
        "pit_high_vol_threshold"
    ] = pit_high_vol_threshold

    gold_vol_research_export[
        "pit_high_vol_event"
    ] = pit_high_vol_event_numeric

    gold_vol_research_export[
        "fwd_realized_vol_20d"
    ] = fwd_vol

    gold_vol_research_export[
        "fwd_mae_20d"
    ] = pd.to_numeric(
        targets[
            "Fwd_MAE_20D"
        ],
        errors="coerce",
    )

    gold_vol_research_export[
        "stress_event_20d"
    ] = pd.to_numeric(
        targets[
            "Stress_Event_20D"
        ],
        errors="coerce",
    )

    gold_vol_research_export[
        "selected_v_candidate"
    ] = selected_v_name


# ============================================================
# 19N. NASDAQ 100 ROLE & ROBUSTNESS DIAGNOSIS
# ============================================================

nasdaq_role_table = pd.DataFrame()
nasdaq_zone_table = pd.DataFrame()
nasdaq_period_table = pd.DataFrame()
nasdaq_nonoverlap_table = pd.DataFrame()
nasdaq_crisis_table = pd.DataFrame()
nasdaq_gate_table = pd.DataFrame()

if selected_asset == "Nasdaq 100":
    st.markdown("---")
    st.subheader(
        "🧠 Nasdaq 100 – Ist der Score Direction oder Risk-State?"
    )

    st.caption(
        "A/Current bleibt Referenz; B, D und E werden unverändert "
        "mitgeführt. In dieser Stufe werden keine Gewichte optimiert."
    )

    nasdaq_models = {
        "A · Current": model_frames[MODEL_CURRENT],
        "B · Literature Prior": model_frames[MODEL_LITERATURE],
        "D · Lit Subweights only": diagnostic_frames[
            "D · Lit Subweights only"
        ],
        "E · Lit Pillars only": diagnostic_frames[
            "E · Lit Pillars only"
        ],
    }

    nasdaq_common = pd.Series(
        True,
        index=targets.index,
    )

    for frame in nasdaq_models.values():
        nasdaq_common &= (
            frame["Final_Regime_Score"].notna()
            & (
                frame["Model_Data_Coverage"]
                >= float(min_coverage)
            )
        )

    st.markdown("### 1. Modellrollen – absolute Direction vs. Risk-State")

    role_rows = []

    for model_name, frame in nasdaq_models.items():
        score = frame["Final_Regime_Score"].where(
            nasdaq_common
        )

        direction20, signals20 = directional_accuracy(
            score,
            targets["Fwd_Return_20D"],
        )

        phase_auc, phase_count = phase_median_stress_auc(
            score,
            targets["Stress_Event_20D"],
            horizon=20,
        )

        role_rows.append(
            {
                "Modell": model_name,
                "N": int(nasdaq_common.sum()),
                "IC 5D": safe_spearman(
                    score,
                    targets["Fwd_Return_5D"],
                ),
                "IC 20D": safe_spearman(
                    score,
                    targets["Fwd_Return_20D"],
                ),
                "IC 60D": safe_spearman(
                    score,
                    targets["Fwd_Return_60D"],
                ),
                "Direction 20D": direction20,
                "Signals 20D": signals20,
                "Stress AUC 20D": binary_auc(
                    score,
                    targets["Stress_Event_20D"],
                    higher_predictor_means_event=False,
                ),
                "Phase-Median Stress AUC": phase_auc,
                "AUC-Phasen": phase_count,
                "Score vs niedrigere FwdVol": safe_spearman(
                    score,
                    -targets["Fwd_Realized_Vol_20D"],
                ),
                "Score vs bessere FwdMAE": safe_spearman(
                    score,
                    targets["Fwd_MAE_20D"],
                ),
                "Ø Coverage": float(
                    frame["Model_Data_Coverage"]
                    .where(nasdaq_common)
                    .mean()
                ),
            }
        )

    nasdaq_role_table = pd.DataFrame(role_rows)

    st.dataframe(
        nasdaq_role_table.style.format(
            {
                "N": "{:.0f}",
                "IC 5D": "{:+.3f}",
                "IC 20D": "{:+.3f}",
                "IC 60D": "{:+.3f}",
                "Direction 20D": "{:.1%}",
                "Signals 20D": "{:.0f}",
                "Stress AUC 20D": "{:.3f}",
                "Phase-Median Stress AUC": "{:.3f}",
                "AUC-Phasen": "{:.0f}",
                "Score vs niedrigere FwdVol": "{:+.3f}",
                "Score vs bessere FwdMAE": "{:+.3f}",
                "Ø Coverage": "{:.1f}%",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### 2. Current – Stresswahrscheinlichkeit nach Score-Zone")

    current_nasdaq_score = nasdaq_models[
        "A · Current"
    ]["Final_Regime_Score"].where(
        nasdaq_common
    )

    zone_frame = pd.DataFrame(
        {
            "score": pd.to_numeric(
                current_nasdaq_score,
                errors="coerce",
            ),
            "stress": pd.to_numeric(
                targets["Stress_Event_20D"],
                errors="coerce",
            ),
            "ret20": pd.to_numeric(
                targets["Fwd_Return_20D"],
                errors="coerce",
            ),
            "fwdvol": pd.to_numeric(
                targets["Fwd_Realized_Vol_20D"],
                errors="coerce",
            ),
            "mae": pd.to_numeric(
                targets["Fwd_MAE_20D"],
                errors="coerce",
            ),
        }
    ).dropna(subset=["score"])

    zone_rows = []

    for label, mask in [
        (
            "≤40 Risk-Off/Stress",
            zone_frame["score"] <= 40,
        ),
        (
            "40–60 Neutral",
            (zone_frame["score"] > 40)
            & (zone_frame["score"] < 60),
        ),
        (
            "≥60 Risk-On",
            zone_frame["score"] >= 60,
        ),
    ]:
        z = zone_frame[mask]

        zone_rows.append(
            {
                "Score-Zone": label,
                "N": int(len(z)),
                "Stressrate 20D": (
                    float(z["stress"].mean())
                    if z["stress"].notna().any()
                    else np.nan
                ),
                "Ø Fwd Return 20D": (
                    float(z["ret20"].mean())
                    if z["ret20"].notna().any()
                    else np.nan
                ),
                "Ø FwdVol 20D": (
                    float(z["fwdvol"].mean())
                    if z["fwdvol"].notna().any()
                    else np.nan
                ),
                "Ø FwdMAE 20D": (
                    float(z["mae"].mean())
                    if z["mae"].notna().any()
                    else np.nan
                ),
            }
        )

    nasdaq_zone_table = pd.DataFrame(zone_rows)

    st.dataframe(
        nasdaq_zone_table.style.format(
            {
                "N": "{:.0f}",
                "Stressrate 20D": "{:.1%}",
                "Ø Fwd Return 20D": "{:+.2%}",
                "Ø FwdVol 20D": "{:.2%}",
                "Ø FwdMAE 20D": "{:+.2%}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### 3. Perioden-Robustheit")

    score_map = {
        name: frame["Final_Regime_Score"].where(
            nasdaq_common
        )
        for name, frame in nasdaq_models.items()
    }

    period_rows = []
    period_rows.extend(
        confirmation_period_metrics(
            "Gesamter Common Sample",
            nasdaq_common,
            score_map,
            targets,
        )
    )

    for label, y1, y2 in [
        ("2012–2016", 2012, 2016),
        ("2017–2020", 2017, 2020),
        ("2021–2025", 2021, 2025),
    ]:
        mask = (
            nasdaq_common
            & (targets.index.year >= y1)
            & (targets.index.year <= y2)
        )

        period_rows.extend(
            confirmation_period_metrics(
                label,
                mask,
                score_map,
                targets,
            )
        )

    nasdaq_period_table = pd.DataFrame(
        period_rows
    )

    st.dataframe(
        nasdaq_period_table.style.format(
            {
                "N": "{:.0f}",
                "IC 20D": "{:+.3f}",
                "IC 60D": "{:+.3f}",
                "Direction 20D": "{:.1%}",
                "Signals 20D": "{:.0f}",
                "Stress AUC 20D": "{:.3f}",
                "Score vs bessere FwdMAE": "{:+.3f}",
                "Score vs niedrigere FwdVol": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### 4. Non-Overlap-Diagnostik – IC20 / IC60")

    nonoverlap_rows = []

    for horizon in [20, 60]:
        target_h = targets[
            f"Fwd_Return_{horizon}D"
        ]

        for model_name, score in score_map.items():
            median_ic, phases = phase_median_nonoverlap_ic(
                score,
                target_h,
                horizon,
            )

            nonoverlap_rows.append(
                {
                    "Horizont": f"{horizon}D",
                    "Modell": model_name,
                    "Median Non-Overlap IC": median_ic,
                    "Verfügbare Phasen": phases,
                }
            )

    nasdaq_nonoverlap_table = pd.DataFrame(
        nonoverlap_rows
    )

    st.dataframe(
        nasdaq_nonoverlap_table.style.format(
            {
                "Median Non-Overlap IC": "{:+.3f}",
                "Verfügbare Phasen": "{:.0f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### 5. Current in vorab definierten Nasdaq-Stressfenstern")

    crisis_rows = []
    price_series = pd.to_numeric(
        raw_df["asset_price"],
        errors="coerce",
    )

    for window_name, start_text, end_text in NASDAQ_DIAGNOSTIC_WINDOWS:
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)

        during_score = current_nasdaq_score[
            (current_nasdaq_score.index >= start)
            & (current_nasdaq_score.index <= end)
        ].dropna()

        pre_score = current_nasdaq_score[
            current_nasdaq_score.index < start
        ].dropna().tail(60)

        price_window = price_series[
            (price_series.index >= start)
            & (price_series.index <= end)
        ].dropna()

        max_drawdown = (
            float(
                (
                    price_window
                    / price_window.cummax()
                    - 1.0
                ).min()
            )
            if not price_window.empty
            else np.nan
        )

        crisis_rows.append(
            {
                "Stressfenster": window_name,
                "Start": start.date().isoformat(),
                "Ende": end.date().isoformat(),
                "N": int(len(during_score)),
                "Ø Score vorher 60T": (
                    float(pre_score.mean())
                    if not pre_score.empty
                    else np.nan
                ),
                "Ø Score im Fenster": (
                    float(during_score.mean())
                    if not during_score.empty
                    else np.nan
                ),
                "Min Score im Fenster": (
                    float(during_score.min())
                    if not during_score.empty
                    else np.nan
                ),
                "Anteil Score <40": (
                    float((during_score < 40).mean())
                    if not during_score.empty
                    else np.nan
                ),
                "Preis-MaxDrawdown": max_drawdown,
            }
        )

    nasdaq_crisis_table = pd.DataFrame(
        crisis_rows
    )

    st.dataframe(
        nasdaq_crisis_table.style.format(
            {
                "N": "{:.0f}",
                "Ø Score vorher 60T": "{:.1f}",
                "Ø Score im Fenster": "{:.1f}",
                "Min Score im Fenster": "{:.1f}",
                "Anteil Score <40": "{:.1%}",
                "Preis-MaxDrawdown": "{:.1%}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Die Nasdaq-Stressfenster wurden vor dieser Auswertung festgelegt "
        "und werden nicht zur Gewichtsanpassung verwendet."
    )

    st.markdown("### 6. Vorab festgelegte Rollen-Gates für Current")

    current_role = nasdaq_role_table[
        nasdaq_role_table["Modell"] == "A · Current"
    ].iloc[0]

    positive_ics = sum(
        bool(
            np.isfinite(
                current_role[f"IC {h}D"]
            )
            and current_role[f"IC {h}D"] > 0
        )
        for h in FORWARD_HORIZONS
    )

    low_row = nasdaq_zone_table[
        nasdaq_zone_table["Score-Zone"] == "≤40 Risk-Off/Stress"
    ]
    high_row = nasdaq_zone_table[
        nasdaq_zone_table["Score-Zone"] == "≥60 Risk-On"
    ]

    low_stress = (
        float(low_row.iloc[0]["Stressrate 20D"])
        if not low_row.empty
        else np.nan
    )
    high_stress = (
        float(high_row.iloc[0]["Stressrate 20D"])
        if not high_row.empty
        else np.nan
    )

    stress_gap = (
        low_stress - high_stress
        if (
            np.isfinite(low_stress)
            and np.isfinite(high_stress)
        )
        else np.nan
    )

    direction_gate = bool(
        positive_ics >= 2
        and np.isfinite(current_role["Direction 20D"])
        and current_role["Direction 20D"]
        >= NASDAQ_DIRECTION_MIN_ACCURACY
    )

    gate_rows = [
        {
            "Rolle": "Direction",
            "Kriterium": (
                "Mindestens 2/3 Forward-ICs positiv UND Direction20 ≥50%"
            ),
            "Erfüllt": direction_gate,
            "Messwert": (
                f"{positive_ics}/3 positive ICs · "
                f"Direction {current_role['Direction 20D']:.1%}"
            ),
        },
        {
            "Rolle": "Risk-State",
            "Kriterium": "Stress-AUC 20D ≥0.60",
            "Erfüllt": bool(
                np.isfinite(current_role["Stress AUC 20D"])
                and current_role["Stress AUC 20D"]
                >= NASDAQ_RISK_MIN_AUC
            ),
            "Messwert": (
                f"{current_role['Stress AUC 20D']:.3f}"
                if np.isfinite(current_role["Stress AUC 20D"])
                else "n/a"
            ),
        },
        {
            "Rolle": "Risk-State",
            "Kriterium": "Phase-Median Stress-AUC ≥0.55",
            "Erfüllt": bool(
                np.isfinite(
                    current_role["Phase-Median Stress AUC"]
                )
                and current_role["Phase-Median Stress AUC"]
                >= NASDAQ_RISK_MIN_PHASE_AUC
            ),
            "Messwert": (
                f"{current_role['Phase-Median Stress AUC']:.3f}"
                if np.isfinite(
                    current_role["Phase-Median Stress AUC"]
                )
                else "n/a"
            ),
        },
        {
            "Rolle": "Risk-State",
            "Kriterium": (
                "Stressrate Score≤40 minus Score≥60 ≥5 Prozentpunkte"
            ),
            "Erfüllt": bool(
                np.isfinite(stress_gap)
                and stress_gap >= NASDAQ_RISK_MIN_STRESS_GAP
            ),
            "Messwert": (
                f"Low {low_stress:.1%} · High {high_stress:.1%} · "
                f"Gap {stress_gap:+.1%}"
                if np.isfinite(stress_gap)
                else "n/a"
            ),
        },
        {
            "Rolle": "Risk-State",
            "Kriterium": (
                "Höherer Score korreliert mit niedrigerer FwdVol"
            ),
            "Erfüllt": bool(
                np.isfinite(
                    current_role["Score vs niedrigere FwdVol"]
                )
                and current_role["Score vs niedrigere FwdVol"] > 0
            ),
            "Messwert": (
                f"{current_role['Score vs niedrigere FwdVol']:+.3f}"
                if np.isfinite(
                    current_role["Score vs niedrigere FwdVol"]
                )
                else "n/a"
            ),
        },
        {
            "Rolle": "Risk-State",
            "Kriterium": (
                "Höherer Score korreliert mit besserer FwdMAE"
            ),
            "Erfüllt": bool(
                np.isfinite(
                    current_role["Score vs bessere FwdMAE"]
                )
                and current_role["Score vs bessere FwdMAE"] > 0
            ),
            "Messwert": (
                f"{current_role['Score vs bessere FwdMAE']:+.3f}"
                if np.isfinite(
                    current_role["Score vs bessere FwdMAE"]
                )
                else "n/a"
            ),
        },
    ]

    nasdaq_gate_table = pd.DataFrame(gate_rows)
    nasdaq_gate_table["Status"] = nasdaq_gate_table[
        "Erfüllt"
    ].map(
        {
            True: "✅",
            False: "❌",
        }
    )

    st.dataframe(
        nasdaq_gate_table[
            ["Rolle", "Kriterium", "Status", "Messwert"]
        ],
        hide_index=True,
        use_container_width=True,
    )

    risk_rows = nasdaq_gate_table[
        nasdaq_gate_table["Rolle"] == "Risk-State"
    ]
    risk_passed = int(
        risk_rows["Erfüllt"].sum()
    )

    st.markdown("### 7. Rollen-Urteil")

    if (
        risk_passed == len(risk_rows)
        and not direction_gate
    ):
        st.success(
            "🟢 **CURRENT BESTÄTIGT SICH PRIMÄR ALS NASDAQ-RISK-STATE-FILTER, "
            "NICHT ALS KONTINUIERLICHER DIRECTION-PREDICTOR.**"
        )

    elif (
        direction_gate
        and risk_passed >= 4
    ):
        st.success(
            "🟢 **CURRENT ZEIGT SOWOHL DIRECTION- ALS AUCH "
            "RISK-STATE-EIGENSCHAFTEN.**"
        )

    elif risk_passed >= max(
        1,
        len(risk_rows) - 1,
    ):
        st.warning(
            f"🟡 **RISK-STATE-ROLLE ÜBERWIEGEND, ABER NICHT VOLLSTÄNDIG "
            f"BESTÄTIGT ({risk_passed}/{len(risk_rows)} Risk-Gates).**"
        )

    else:
        st.error(
            f"🔴 **CURRENT IST AUCH ALS NASDAQ-RISK-STATE NICHT STABIL GENUG "
            f"({risk_passed}/{len(risk_rows)} Risk-Gates).**"
        )

    st.info(
        "v1.0.18 ist absichtlich eine Diagnosestufe. "
        "Es werden keine Gewichte automatisch optimiert."
    )


# ============================================================
# 20A. S&P 500 ROLE & ROBUSTNESS DIAGNOSIS
# ============================================================

sp500_role_table = pd.DataFrame()
sp500_zone_table = pd.DataFrame()
sp500_period_table = pd.DataFrame()
sp500_nonoverlap_table = pd.DataFrame()
sp500_crisis_table = pd.DataFrame()
sp500_gate_table = pd.DataFrame()

if selected_asset == "S&P 500":
    st.markdown("---")
    st.subheader(
        "🧠 S&P 500 – Ist der Score Direction oder Risk-State?"
    )

    sp_models = {
        "A · Current": model_frames[MODEL_CURRENT],
        "B · Literature Prior": model_frames[MODEL_LITERATURE],
        "D · Lit Subweights only": diagnostic_frames[
            "D · Lit Subweights only"
        ],
        "E · Lit Pillars only": diagnostic_frames[
            "E · Lit Pillars only"
        ],
    }

    sp_common = pd.Series(True, index=targets.index)

    for frame in sp_models.values():
        sp_common &= (
            frame["Final_Regime_Score"].notna()
            & (
                frame["Model_Data_Coverage"]
                >= float(min_coverage)
            )
        )

    st.markdown("### 1. Modellrollen – absolute Direction vs. Risk-State")

    role_rows = []

    for model_name, frame in sp_models.items():
        score = frame["Final_Regime_Score"].where(sp_common)

        direction20, signals20 = directional_accuracy(
            score,
            targets["Fwd_Return_20D"],
        )

        phase_auc, phase_count = phase_median_stress_auc(
            score,
            targets["Stress_Event_20D"],
            horizon=20,
        )

        role_rows.append({
            "Modell": model_name,
            "N": int(sp_common.sum()),
            "IC 5D": safe_spearman(
                score,
                targets["Fwd_Return_5D"],
            ),
            "IC 20D": safe_spearman(
                score,
                targets["Fwd_Return_20D"],
            ),
            "IC 60D": safe_spearman(
                score,
                targets["Fwd_Return_60D"],
            ),
            "Direction 20D": direction20,
            "Signals 20D": signals20,
            "Stress AUC 20D": binary_auc(
                score,
                targets["Stress_Event_20D"],
                higher_predictor_means_event=False,
            ),
            "Phase-Median Stress AUC": phase_auc,
            "AUC-Phasen": phase_count,
            "Score vs niedrigere FwdVol": safe_spearman(
                score,
                -targets["Fwd_Realized_Vol_20D"],
            ),
            "Score vs bessere FwdMAE": safe_spearman(
                score,
                targets["Fwd_MAE_20D"],
            ),
            "Ø Coverage": float(
                frame["Model_Data_Coverage"]
                .where(sp_common)
                .mean()
            ),
        })

    sp500_role_table = pd.DataFrame(role_rows)

    st.dataframe(
        sp500_role_table.style.format(
            {
                "N": "{:.0f}",
                "IC 5D": "{:+.3f}",
                "IC 20D": "{:+.3f}",
                "IC 60D": "{:+.3f}",
                "Direction 20D": "{:.1%}",
                "Signals 20D": "{:.0f}",
                "Stress AUC 20D": "{:.3f}",
                "Phase-Median Stress AUC": "{:.3f}",
                "AUC-Phasen": "{:.0f}",
                "Score vs niedrigere FwdVol": "{:+.3f}",
                "Score vs bessere FwdMAE": "{:+.3f}",
                "Ø Coverage": "{:.1f}%",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### 2. Current – Stresswahrscheinlichkeit nach Score-Zone")

    current_sp_score = sp_models[
        "A · Current"
    ]["Final_Regime_Score"].where(sp_common)

    zone_frame = pd.DataFrame({
        "score": pd.to_numeric(
            current_sp_score,
            errors="coerce",
        ),
        "stress": pd.to_numeric(
            targets["Stress_Event_20D"],
            errors="coerce",
        ),
        "ret20": pd.to_numeric(
            targets["Fwd_Return_20D"],
            errors="coerce",
        ),
        "fwdvol": pd.to_numeric(
            targets["Fwd_Realized_Vol_20D"],
            errors="coerce",
        ),
        "mae": pd.to_numeric(
            targets["Fwd_MAE_20D"],
            errors="coerce",
        ),
    }).dropna(subset=["score"])

    zone_rows = []

    for label, mask in [
        ("≤40 Risk-Off/Stress", zone_frame["score"] <= 40),
        (
            "40–60 Neutral",
            (zone_frame["score"] > 40)
            & (zone_frame["score"] < 60),
        ),
        ("≥60 Risk-On", zone_frame["score"] >= 60),
    ]:
        z = zone_frame[mask]

        zone_rows.append({
            "Score-Zone": label,
            "N": len(z),
            "Stressrate 20D": (
                float(z["stress"].mean())
                if z["stress"].notna().any()
                else np.nan
            ),
            "Ø Fwd Return 20D": (
                float(z["ret20"].mean())
                if z["ret20"].notna().any()
                else np.nan
            ),
            "Ø FwdVol 20D": (
                float(z["fwdvol"].mean())
                if z["fwdvol"].notna().any()
                else np.nan
            ),
            "Ø FwdMAE 20D": (
                float(z["mae"].mean())
                if z["mae"].notna().any()
                else np.nan
            ),
        })

    sp500_zone_table = pd.DataFrame(zone_rows)

    st.dataframe(
        sp500_zone_table.style.format(
            {
                "N": "{:.0f}",
                "Stressrate 20D": "{:.1%}",
                "Ø Fwd Return 20D": "{:+.2%}",
                "Ø FwdVol 20D": "{:.2%}",
                "Ø FwdMAE 20D": "{:+.2%}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### 3. Perioden-Robustheit")

    score_map = {
        name: frame["Final_Regime_Score"].where(sp_common)
        for name, frame in sp_models.items()
    }

    period_rows = []
    period_rows.extend(
        confirmation_period_metrics(
            "Gesamter Common Sample",
            sp_common,
            score_map,
            targets,
        )
    )

    for label, y1, y2 in [
        ("2012–2016", 2012, 2016),
        ("2017–2020", 2017, 2020),
        ("2021–2025", 2021, 2025),
    ]:
        mask = (
            sp_common
            & (targets.index.year >= y1)
            & (targets.index.year <= y2)
        )

        period_rows.extend(
            confirmation_period_metrics(
                label,
                mask,
                score_map,
                targets,
            )
        )

    sp500_period_table = pd.DataFrame(period_rows)

    st.dataframe(
        sp500_period_table.style.format(
            {
                "N": "{:.0f}",
                "IC 20D": "{:+.3f}",
                "IC 60D": "{:+.3f}",
                "Direction 20D": "{:.1%}",
                "Signals 20D": "{:.0f}",
                "Stress AUC 20D": "{:.3f}",
                "Score vs bessere FwdMAE": "{:+.3f}",
                "Score vs niedrigere FwdVol": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### 4. Non-Overlap-Diagnostik – IC20 / IC60")

    nonoverlap_rows = []

    for horizon in [20, 60]:
        target_h = targets[f"Fwd_Return_{horizon}D"]

        for model_name, score in score_map.items():
            median_ic, phases = phase_median_nonoverlap_ic(
                score,
                target_h,
                horizon,
            )

            nonoverlap_rows.append({
                "Horizont": f"{horizon}D",
                "Modell": model_name,
                "Median Non-Overlap IC": median_ic,
                "Verfügbare Phasen": phases,
            })

    sp500_nonoverlap_table = pd.DataFrame(nonoverlap_rows)

    st.dataframe(
        sp500_nonoverlap_table.style.format(
            {
                "Median Non-Overlap IC": "{:+.3f}",
                "Verfügbare Phasen": "{:.0f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### 5. Current in vorab definierten Stressfenstern")

    crisis_rows = []
    price_series = pd.to_numeric(
        raw_df["asset_price"],
        errors="coerce",
    )

    for window_name, start_text, end_text in SP500_DIAGNOSTIC_WINDOWS:
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)

        during_score = current_sp_score[
            (current_sp_score.index >= start)
            & (current_sp_score.index <= end)
        ].dropna()

        pre_score = current_sp_score[
            current_sp_score.index < start
        ].dropna().tail(60)

        price_window = price_series[
            (price_series.index >= start)
            & (price_series.index <= end)
        ].dropna()

        max_drawdown = (
            float(
                (
                    price_window
                    / price_window.cummax()
                    - 1.0
                ).min()
            )
            if not price_window.empty
            else np.nan
        )

        crisis_rows.append({
            "Stressfenster": window_name,
            "Start": start.date().isoformat(),
            "Ende": end.date().isoformat(),
            "N": len(during_score),
            "Ø Score vorher 60T": (
                float(pre_score.mean())
                if not pre_score.empty
                else np.nan
            ),
            "Ø Score im Fenster": (
                float(during_score.mean())
                if not during_score.empty
                else np.nan
            ),
            "Min Score im Fenster": (
                float(during_score.min())
                if not during_score.empty
                else np.nan
            ),
            "Anteil Score <40": (
                float((during_score < 40).mean())
                if not during_score.empty
                else np.nan
            ),
            "Preis-MaxDrawdown": max_drawdown,
        })

    sp500_crisis_table = pd.DataFrame(crisis_rows)

    st.dataframe(
        sp500_crisis_table.style.format(
            {
                "N": "{:.0f}",
                "Ø Score vorher 60T": "{:.1f}",
                "Ø Score im Fenster": "{:.1f}",
                "Min Score im Fenster": "{:.1f}",
                "Anteil Score <40": "{:.1%}",
                "Preis-MaxDrawdown": "{:.1%}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Die Stressfenster wurden vor dieser Auswertung festgelegt und "
        "werden nicht zur Gewichtsanpassung verwendet."
    )

    st.markdown("### 6. Vorab festgelegte Rollen-Gates für Current")

    current_role = sp500_role_table[
        sp500_role_table["Modell"] == "A · Current"
    ].iloc[0]

    positive_ics = sum(
        bool(
            np.isfinite(current_role[f"IC {h}D"])
            and current_role[f"IC {h}D"] > 0
        )
        for h in FORWARD_HORIZONS
    )

    low_row = sp500_zone_table[
        sp500_zone_table["Score-Zone"] == "≤40 Risk-Off/Stress"
    ]
    high_row = sp500_zone_table[
        sp500_zone_table["Score-Zone"] == "≥60 Risk-On"
    ]

    low_stress = (
        float(low_row.iloc[0]["Stressrate 20D"])
        if not low_row.empty
        else np.nan
    )
    high_stress = (
        float(high_row.iloc[0]["Stressrate 20D"])
        if not high_row.empty
        else np.nan
    )

    stress_gap = (
        low_stress - high_stress
        if (
            np.isfinite(low_stress)
            and np.isfinite(high_stress)
        )
        else np.nan
    )

    direction_gate = bool(
        positive_ics >= 2
        and np.isfinite(current_role["Direction 20D"])
        and current_role["Direction 20D"]
        >= SP500_DIRECTION_MIN_ACCURACY
    )

    gate_rows = [
        {
            "Rolle": "Direction",
            "Kriterium": (
                "Mindestens 2/3 Forward-ICs positiv UND Direction20 ≥50%"
            ),
            "Erfüllt": direction_gate,
            "Messwert": (
                f"{positive_ics}/3 positive ICs · "
                f"Direction {current_role['Direction 20D']:.1%}"
            ),
        },
        {
            "Rolle": "Risk-State",
            "Kriterium": "Stress-AUC 20D ≥0.60",
            "Erfüllt": bool(
                np.isfinite(current_role["Stress AUC 20D"])
                and current_role["Stress AUC 20D"]
                >= SP500_RISK_MIN_AUC
            ),
            "Messwert": (
                f"{current_role['Stress AUC 20D']:.3f}"
                if np.isfinite(current_role["Stress AUC 20D"])
                else "n/a"
            ),
        },
        {
            "Rolle": "Risk-State",
            "Kriterium": "Phase-Median Stress-AUC ≥0.55",
            "Erfüllt": bool(
                np.isfinite(current_role["Phase-Median Stress AUC"])
                and current_role["Phase-Median Stress AUC"]
                >= SP500_RISK_MIN_PHASE_AUC
            ),
            "Messwert": (
                f"{current_role['Phase-Median Stress AUC']:.3f}"
                if np.isfinite(
                    current_role["Phase-Median Stress AUC"]
                )
                else "n/a"
            ),
        },
        {
            "Rolle": "Risk-State",
            "Kriterium": (
                "Stressrate Score≤40 minus Score≥60 ≥5 Prozentpunkte"
            ),
            "Erfüllt": bool(
                np.isfinite(stress_gap)
                and stress_gap >= SP500_RISK_MIN_STRESS_GAP
            ),
            "Messwert": (
                f"Low {low_stress:.1%} · High {high_stress:.1%} · "
                f"Gap {stress_gap:+.1%}"
                if np.isfinite(stress_gap)
                else "n/a"
            ),
        },
        {
            "Rolle": "Risk-State",
            "Kriterium": (
                "Höherer Score korreliert mit niedrigerer FwdVol"
            ),
            "Erfüllt": bool(
                np.isfinite(
                    current_role["Score vs niedrigere FwdVol"]
                )
                and current_role["Score vs niedrigere FwdVol"] > 0
            ),
            "Messwert": (
                f"{current_role['Score vs niedrigere FwdVol']:+.3f}"
                if np.isfinite(
                    current_role["Score vs niedrigere FwdVol"]
                )
                else "n/a"
            ),
        },
        {
            "Rolle": "Risk-State",
            "Kriterium": (
                "Höherer Score korreliert mit besserer FwdMAE"
            ),
            "Erfüllt": bool(
                np.isfinite(
                    current_role["Score vs bessere FwdMAE"]
                )
                and current_role["Score vs bessere FwdMAE"] > 0
            ),
            "Messwert": (
                f"{current_role['Score vs bessere FwdMAE']:+.3f}"
                if np.isfinite(
                    current_role["Score vs bessere FwdMAE"]
                )
                else "n/a"
            ),
        },
    ]

    sp500_gate_table = pd.DataFrame(gate_rows)
    sp500_gate_table["Status"] = sp500_gate_table[
        "Erfüllt"
    ].map({True: "✅", False: "❌"})

    st.dataframe(
        sp500_gate_table[
            ["Rolle", "Kriterium", "Status", "Messwert"]
        ],
        hide_index=True,
        use_container_width=True,
    )

    risk_rows = sp500_gate_table[
        sp500_gate_table["Rolle"] == "Risk-State"
    ]
    risk_passed = int(risk_rows["Erfüllt"].sum())

    st.markdown("### 7. Rollen-Urteil")

    if (
        risk_passed == len(risk_rows)
        and not direction_gate
    ):
        st.success(
            "🟢 **CURRENT BESTÄTIGT SICH PRIMÄR ALS RISK-STATE-FILTER, "
            "NICHT ALS KONTINUIERLICHER DIRECTION-PREDICTOR.** "
            "Im nächsten Schritt definieren wir anhand von 3A/7A einen "
            "getrennten Challenger, ohne Current unnötig zu ersetzen."
        )
    elif risk_passed >= max(1, len(risk_rows) - 1):
        st.warning(
            f"🟡 **RISK-STATE-ROLLE ÜBERWIEGEND, ABER NICHT VOLLSTÄNDIG "
            f"BESTÄTIGT ({risk_passed}/{len(risk_rows)} Risk-Gates).**"
        )
    else:
        st.error(
            f"🔴 **CURRENT IST AUCH ALS RISK-STATE NICHT STABIL GENUG "
            f"({risk_passed}/{len(risk_rows)} Risk-Gates).** "
            "Dann untersuchen wir zuerst die Faktorarchitektur."
        )

    st.info(
        "v1.0.16 ist absichtlich eine Diagnosestufe. Aus den Ergebnissen "
        "werden in diesem Lauf noch keine Gewichte automatisch optimiert."
    )


# ============================================================
# 20A2. S&P 500 FROZEN A-vs-E RISK-STATE WALK-FORWARD
# ============================================================

sp500_wf_yearly_df = pd.DataFrame()
sp500_wf_pooled_df = pd.DataFrame()
sp500_wf_bootstrap_df = pd.DataFrame()
sp500_wf_gate_df = pd.DataFrame()

if selected_asset == "S&P 500":
    st.markdown("---")
    st.subheader("🧭 S&P 500 Frozen A-vs-E Risk-State Walk-Forward")
    st.caption(
        "A = Current bleibt die Referenz. E = Literature Pillars only ist "
        "der eingefrorene Risk-State-Challenger. Keine Gewichte werden "
        "verändert. Ein Mini-Vorsprung reicht ausdrücklich nicht für eine Promotion."
    )

    a_frame = model_frames[MODEL_CURRENT]
    e_frame = diagnostic_frames["E · Lit Pillars only"]
    a_score = a_frame["Final_Regime_Score"]
    e_score = e_frame["Final_Regime_Score"]

    common = (
        a_score.notna()
        & e_score.notna()
        & (a_frame["Model_Data_Coverage"] >= float(min_coverage))
        & (e_frame["Model_Data_Coverage"] >= float(min_coverage))
    )

    current_year = int(pd.Timestamp.now().year)
    yearly_rows = []

    for year in sorted(set(targets.index.year)):
        if int(year) < SP500_WF_FIRST_TEST_YEAR:
            continue
        mask = common & (targets.index.year == int(year))
        if int(mask.sum()) < 40:
            continue

        a = sp500_risk_state_metrics(a_score, mask, targets)
        e = sp500_risk_state_metrics(e_score, mask, targets)
        auc_delta = (
            e["Stress AUC 20D"] - a["Stress AUC 20D"]
            if np.isfinite(e["Stress AUC 20D"]) and np.isfinite(a["Stress AUC 20D"])
            else np.nan
        )
        phase_delta = (
            e["Phase-Median Stress AUC"] - a["Phase-Median Stress AUC"]
            if np.isfinite(e["Phase-Median Stress AUC"]) and np.isfinite(a["Phase-Median Stress AUC"])
            else np.nan
        )
        yearly_rows.append({
            "Testjahr": int(year),
            "Vollständiges Jahr": int(year) < current_year,
            "N": a["N"],
            "A Stress AUC": a["Stress AUC 20D"],
            "E Stress AUC": e["Stress AUC 20D"],
            "Δ AUC E−A": auc_delta,
            "A Phase AUC": a["Phase-Median Stress AUC"],
            "E Phase AUC": e["Phase-Median Stress AUC"],
            "Δ Phase AUC E−A": phase_delta,
            "A Stress-Gap": a["Stressrate-Gap"],
            "E Stress-Gap": e["Stressrate-Gap"],
            "A FwdVol Relation": a["Score vs niedrigere FwdVol"],
            "E FwdVol Relation": e["Score vs niedrigere FwdVol"],
            "A FwdMAE Relation": a["Score vs bessere FwdMAE"],
            "E FwdMAE Relation": e["Score vs bessere FwdMAE"],
            "E AUC besser A": bool(np.isfinite(auc_delta) and auc_delta > 0),
            "E AUC mindestens +0.01": bool(np.isfinite(auc_delta) and auc_delta >= SP500_WF_MIN_E_AUC_DELTA),
        })

    sp500_wf_yearly_df = pd.DataFrame(yearly_rows)

    st.markdown("### 1. Chronologische Testjahre")
    if not sp500_wf_yearly_df.empty:
        st.dataframe(
            sp500_wf_yearly_df.style.format({
                "N": "{:.0f}",
                "A Stress AUC": "{:.3f}",
                "E Stress AUC": "{:.3f}",
                "Δ AUC E−A": "{:+.3f}",
                "A Phase AUC": "{:.3f}",
                "E Phase AUC": "{:.3f}",
                "Δ Phase AUC E−A": "{:+.3f}",
                "A Stress-Gap": "{:+.1%}",
                "E Stress-Gap": "{:+.1%}",
                "A FwdVol Relation": "{:+.3f}",
                "E FwdVol Relation": "{:+.3f}",
                "A FwdMAE Relation": "{:+.3f}",
                "E FwdMAE Relation": "{:+.3f}",
            }, na_rep="n/a"),
            hide_index=True,
            use_container_width=True,
        )

    complete = sp500_wf_yearly_df[
        sp500_wf_yearly_df["Vollständiges Jahr"].astype(bool)
    ].copy()

    if complete.empty:
        st.warning("Keine vollständigen Walk-Forward-Testjahre verfügbar.")
    else:
        years = complete["Testjahr"].astype(int).tolist()
        pooled_mask = common & pd.Series(
            targets.index.year,
            index=targets.index,
        ).isin(years)

        a_pool = sp500_risk_state_metrics(a_score, pooled_mask, targets)
        e_pool = sp500_risk_state_metrics(e_score, pooled_mask, targets)

        sp500_wf_pooled_df = pd.DataFrame([
            {"Modell": "A · Current", **a_pool},
            {"Modell": "E · Lit Pillars only", **e_pool},
        ])

        st.markdown("### 2. Gepoolte vollständige Testjahre")
        st.dataframe(
            sp500_wf_pooled_df.style.format({
                "N": "{:.0f}",
                "Stress AUC 20D": "{:.3f}",
                "Phase-Median Stress AUC": "{:.3f}",
                "AUC-Phasen": "{:.0f}",
                "Stressrate Score≤40": "{:.1%}",
                "Stressrate Score≥60": "{:.1%}",
                "Stressrate-Gap": "{:+.1%}",
                "Score vs niedrigere FwdVol": "{:+.3f}",
                "Score vs bessere FwdMAE": "{:+.3f}",
            }, na_rep="n/a"),
            hide_index=True,
            use_container_width=True,
        )

        boot = block_bootstrap_auc_difference(
            a_score.where(pooled_mask),
            e_score.where(pooled_mask),
            targets["Stress_Event_20D"].where(pooled_mask),
            block_length=int(bootstrap_block),
            n_boot=int(bootstrap_runs),
            seed=20260911,
        )
        sp500_wf_bootstrap_df = pd.DataFrame([{
            "Vergleich": "E − A Stress AUC 20D",
            "Δ AUC": boot["observed"],
            "P(E>A)": boot["prob_positive"],
            "95% CI Low": boot["lower"],
            "95% CI High": boot["upper"],
            "N": boot["n"],
        }])

        st.markdown("### 3. Block-Bootstrap – E Stress-AUC minus A")
        st.dataframe(
            sp500_wf_bootstrap_df.style.format({
                "Δ AUC": "{:+.3f}",
                "P(E>A)": "{:.1%}",
                "95% CI Low": "{:+.3f}",
                "95% CI High": "{:+.3f}",
                "N": "{:.0f}",
            }, na_rep="n/a"),
            hide_index=True,
            use_container_width=True,
        )

        a_auc = float(a_pool["Stress AUC 20D"])
        e_auc = float(e_pool["Stress AUC 20D"])
        a_phase = float(a_pool["Phase-Median Stress AUC"])
        e_phase = float(e_pool["Phase-Median Stress AUC"])
        a_gap = float(a_pool["Stressrate-Gap"])
        n_years = len(complete)
        year_wins = int(complete["E AUC besser A"].astype(bool).sum())
        material_wins = int(complete["E AUC mindestens +0.01"].astype(bool).sum())
        year_win_rate = year_wins / n_years if n_years else np.nan
        material_rate = material_wins / n_years if n_years else np.nan

        current_rows = [
            {"Gruppe": "A Current Validity", "Kriterium": "A gepoolte Stress-AUC ≥0.60", "Erfüllt": bool(np.isfinite(a_auc) and a_auc >= SP500_WF_MIN_CURRENT_AUC), "Messwert": f"{a_auc:.3f}" if np.isfinite(a_auc) else "n/a"},
            {"Gruppe": "A Current Validity", "Kriterium": "A Phase-Median Stress-AUC ≥0.55", "Erfüllt": bool(np.isfinite(a_phase) and a_phase >= SP500_WF_MIN_CURRENT_PHASE_AUC), "Messwert": f"{a_phase:.3f}" if np.isfinite(a_phase) else "n/a"},
            {"Gruppe": "A Current Validity", "Kriterium": "A Stressrate-Gap Score≤40 minus ≥60 ≥5pp", "Erfüllt": bool(np.isfinite(a_gap) and a_gap >= SP500_WF_MIN_CURRENT_STRESS_GAP), "Messwert": f"{a_gap:+.1%}" if np.isfinite(a_gap) else "n/a"},
            {"Gruppe": "A Current Validity", "Kriterium": "A höherer Score → niedrigere FwdVol", "Erfüllt": bool(np.isfinite(a_pool["Score vs niedrigere FwdVol"]) and a_pool["Score vs niedrigere FwdVol"] > 0), "Messwert": f"{a_pool['Score vs niedrigere FwdVol']:+.3f}"},
            {"Gruppe": "A Current Validity", "Kriterium": "A höherer Score → bessere FwdMAE", "Erfüllt": bool(np.isfinite(a_pool["Score vs bessere FwdMAE"]) and a_pool["Score vs bessere FwdMAE"] > 0), "Messwert": f"{a_pool['Score vs bessere FwdMAE']:+.3f}"},
        ]

        promotion_rows = [
            {"Gruppe": "E Promotion", "Kriterium": "E gepoolte AUC mindestens +0.01 über A", "Erfüllt": bool(np.isfinite(e_auc) and np.isfinite(a_auc) and e_auc-a_auc >= SP500_WF_MIN_E_AUC_DELTA), "Messwert": f"E {e_auc:.3f} · A {a_auc:.3f} · Δ {e_auc-a_auc:+.3f}"},
            {"Gruppe": "E Promotion", "Kriterium": "E Phase-AUC mindestens +0.01 über A", "Erfüllt": bool(np.isfinite(e_phase) and np.isfinite(a_phase) and e_phase-a_phase >= SP500_WF_MIN_E_PHASE_AUC_DELTA), "Messwert": f"E {e_phase:.3f} · A {a_phase:.3f} · Δ {e_phase-a_phase:+.3f}"},
            {"Gruppe": "E Promotion", "Kriterium": "E AUC > A in mindestens 60% der vollständigen Testjahre", "Erfüllt": bool(np.isfinite(year_win_rate) and year_win_rate >= SP500_WF_MIN_E_YEAR_WIN_RATE), "Messwert": f"{year_wins}/{n_years} ({year_win_rate:.1%})" if np.isfinite(year_win_rate) else "n/a"},
            {"Gruppe": "E Promotion", "Kriterium": "E erreicht +0.01 AUC-Vorsprung in mindestens 50% der Testjahre", "Erfüllt": bool(np.isfinite(material_rate) and material_rate >= SP500_WF_MIN_E_MATERIAL_YEAR_RATE), "Messwert": f"{material_wins}/{n_years} ({material_rate:.1%})" if np.isfinite(material_rate) else "n/a"},
            {"Gruppe": "E Promotion", "Kriterium": "Bootstrap P(E>A) ≥90%", "Erfüllt": bool(np.isfinite(boot["prob_positive"]) and boot["prob_positive"] >= SP500_WF_MIN_E_BOOTSTRAP_PROB), "Messwert": f"{boot['prob_positive']:.1%}" if np.isfinite(boot["prob_positive"]) else "n/a"},
            {"Gruppe": "E Promotion", "Kriterium": "E behält positiven Bezug zu niedrigerer FwdVol", "Erfüllt": bool(np.isfinite(e_pool["Score vs niedrigere FwdVol"]) and e_pool["Score vs niedrigere FwdVol"] > 0), "Messwert": f"{e_pool['Score vs niedrigere FwdVol']:+.3f}"},
            {"Gruppe": "E Promotion", "Kriterium": "E behält positiven Bezug zu besserer FwdMAE", "Erfüllt": bool(np.isfinite(e_pool["Score vs bessere FwdMAE"]) and e_pool["Score vs bessere FwdMAE"] > 0), "Messwert": f"{e_pool['Score vs bessere FwdMAE']:+.3f}"},
        ]

        sp500_wf_gate_df = pd.DataFrame(current_rows + promotion_rows)
        sp500_wf_gate_df["Status"] = sp500_wf_gate_df["Erfüllt"].map({True: "✅", False: "❌"})

        st.markdown("### 4. Vorab festgelegte Walk-Forward-Gates")
        st.dataframe(
            sp500_wf_gate_df[["Gruppe", "Kriterium", "Status", "Messwert"]],
            hide_index=True,
            use_container_width=True,
        )

        current_gate = sp500_wf_gate_df[sp500_wf_gate_df["Gruppe"] == "A Current Validity"]
        e_gate = sp500_wf_gate_df[sp500_wf_gate_df["Gruppe"] == "E Promotion"]
        current_passed = int(current_gate["Erfüllt"].sum())
        e_passed = int(e_gate["Erfüllt"].sum())

        st.markdown("### 5. Walk-Forward-Urteil")
        if current_passed == len(current_gate) and e_passed == len(e_gate):
            st.success(
                "🟢 **A BLEIBT VALIDE UND E ZEIGT EINEN MATERIELLEN, STABILEN VORTEIL.** "
                "E wäre damit als Frozen Challenger für einen echten zukünftigen Shadow-Holdout gerechtfertigt."
            )
        elif current_passed == len(current_gate):
            st.info(
                f"🔵 **CURRENT BESTÄTIGT ({current_passed}/{len(current_gate)}), "
                f"E ABER NICHT ALS KLAR ÜBERLEGENER NACHFOLGER ({e_passed}/{len(e_gate)} Promotion-Gates).** "
                "A bleibt damit S&P-Referenz."
            )
        else:
            st.warning(
                f"🟡 **CURRENT SELBST NICHT VOLLSTÄNDIG BESTÄTIGT ({current_passed}/{len(current_gate)}).** "
                "Dann wird kein Challenger promoviert."
            )

        st.caption(
            "Dieser Walk-Forward ist eine historische Robustheitsprüfung, kein echter unseen Holdout. "
            "Nur ein klarer E-Vorteil rechtfertigt anschließend einen Future-Shadow."
        )


# ============================================================
# 20A. EUR/USD B-vs-E CONFIRMATION TEST
# ============================================================

eurusd_confirmation_horizons = pd.DataFrame()
eurusd_confirmation_yearly = pd.DataFrame()
eurusd_confirmation_periods = pd.DataFrame()
eurusd_confirmation_nonoverlap = pd.DataFrame()
eurusd_confirmation_gate = pd.DataFrame()

if selected_asset == "EUR/USD":
    st.markdown("---")
    st.subheader(
        "🔬 EUR/USD Bestätigungstest – Current vs. B vs. E"
    )

    st.caption(
        "**B = Full Literature Prior** und "
        "**E = Literature Pillars only** bleiben exakt eingefroren. "
        "Es werden keine Gewichte gesucht oder neu optimiert. "
        "Geprüft werden ausschließlich Robustheit und die zuvor "
        "identifizierten Rollen: B für 20–60D Direction, E für Risk-State."
    )

    current_be_frame = model_frames[MODEL_CURRENT]
    b_frame = model_frames[MODEL_LITERATURE]
    e_frame = diagnostic_frames["E · Lit Pillars only"]

    be_common = (
        current_be_frame["Final_Regime_Score"].notna()
        & b_frame["Final_Regime_Score"].notna()
        & e_frame["Final_Regime_Score"].notna()
        & (
            current_be_frame["Model_Data_Coverage"]
            >= float(min_coverage)
        )
        & (
            b_frame["Model_Data_Coverage"]
            >= float(min_coverage)
        )
        & (
            e_frame["Model_Data_Coverage"]
            >= float(min_coverage)
        )
    )

    current_be_score = current_be_frame[
        "Final_Regime_Score"
    ].where(be_common)

    b_score = b_frame[
        "Final_Regime_Score"
    ].where(be_common)

    e_score = e_frame[
        "Final_Regime_Score"
    ].where(be_common)

    # ========================================================
    # 1. HORIZONS + BOOTSTRAP
    # ========================================================

    st.markdown(
        "### 1. Forward-IC & Bootstrap – 5D / 20D / 60D"
    )

    horizon_rows = []

    b_positive_ic_count = 0
    b_ic_wins_vs_current = 0
    b_bootstrap_75_vs_current = 0
    b_ic_wins_vs_e = 0

    for horizon in FORWARD_HORIZONS:
        target_h = targets[f"Fwd_Return_{horizon}D"]

        current_ic = safe_spearman(
            current_be_score,
            target_h,
        )
        b_ic = safe_spearman(
            b_score,
            target_h,
        )
        e_ic = safe_spearman(
            e_score,
            target_h,
        )

        if np.isfinite(b_ic) and b_ic > 0:
            b_positive_ic_count += 1

        if (
            np.isfinite(b_ic)
            and np.isfinite(current_ic)
            and b_ic > current_ic
        ):
            b_ic_wins_vs_current += 1

        if (
            np.isfinite(b_ic)
            and np.isfinite(e_ic)
            and b_ic > e_ic
        ):
            b_ic_wins_vs_e += 1

        boot_b_current = block_bootstrap_ic_difference(
            current_be_score,
            b_score,
            target_h,
            block_length=int(bootstrap_block),
            n_boot=int(bootstrap_runs),
            seed=2100 + int(horizon),
        )

        boot_e_current = block_bootstrap_ic_difference(
            current_be_score,
            e_score,
            target_h,
            block_length=int(bootstrap_block),
            n_boot=int(bootstrap_runs),
            seed=2200 + int(horizon),
        )

        # Function returns second minus first; E first, B second => B-E.
        boot_b_e = block_bootstrap_ic_difference(
            e_score,
            b_score,
            target_h,
            block_length=int(bootstrap_block),
            n_boot=int(bootstrap_runs),
            seed=2300 + int(horizon),
        )

        if (
            np.isfinite(
                boot_b_current["prob_positive"]
            )
            and boot_b_current["prob_positive"] >= 0.75
        ):
            b_bootstrap_75_vs_current += 1

        horizon_rows.append({
            "Horizont": f"{horizon}D",
            "Current IC": current_ic,
            "B IC": b_ic,
            "E IC": e_ic,
            "Δ B−Current": (
                b_ic - current_ic
                if (
                    np.isfinite(b_ic)
                    and np.isfinite(current_ic)
                )
                else np.nan
            ),
            "P(B>Current)": boot_b_current[
                "prob_positive"
            ],
            "B−Current CI Low": boot_b_current[
                "lower"
            ],
            "B−Current CI High": boot_b_current[
                "upper"
            ],
            "Δ E−Current": (
                e_ic - current_ic
                if (
                    np.isfinite(e_ic)
                    and np.isfinite(current_ic)
                )
                else np.nan
            ),
            "P(E>Current)": boot_e_current[
                "prob_positive"
            ],
            "Δ B−E": (
                b_ic - e_ic
                if (
                    np.isfinite(b_ic)
                    and np.isfinite(e_ic)
                )
                else np.nan
            ),
            "P(B>E)": boot_b_e[
                "prob_positive"
            ],
            "B−E CI Low": boot_b_e[
                "lower"
            ],
            "B−E CI High": boot_b_e[
                "upper"
            ],
        })

    eurusd_confirmation_horizons = pd.DataFrame(
        horizon_rows
    )

    st.dataframe(
        eurusd_confirmation_horizons.style.format(
            {
                "Current IC": "{:+.3f}",
                "B IC": "{:+.3f}",
                "E IC": "{:+.3f}",
                "Δ B−Current": "{:+.3f}",
                "P(B>Current)": "{:.1%}",
                "B−Current CI Low": "{:+.3f}",
                "B−Current CI High": "{:+.3f}",
                "Δ E−Current": "{:+.3f}",
                "P(E>Current)": "{:.1%}",
                "Δ B−E": "{:+.3f}",
                "P(B>E)": "{:.1%}",
                "B−E CI Low": "{:+.3f}",
                "B−E CI High": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # ========================================================
    # 2. ABSOLUTE ROLES
    # ========================================================

    st.markdown(
        "### 2. Absolute Validity & Rollenvergleich"
    )

    current_direction_20d, current_signals_20d = (
        directional_accuracy(
            current_be_score,
            targets["Fwd_Return_20D"],
        )
    )
    b_direction_20d, b_signals_20d = directional_accuracy(
        b_score,
        targets["Fwd_Return_20D"],
    )
    e_direction_20d, e_signals_20d = directional_accuracy(
        e_score,
        targets["Fwd_Return_20D"],
    )

    current_auc = binary_auc(
        current_be_score,
        targets["Stress_Event_20D"],
        higher_predictor_means_event=False,
    )
    b_auc = binary_auc(
        b_score,
        targets["Stress_Event_20D"],
        higher_predictor_means_event=False,
    )
    e_auc = binary_auc(
        e_score,
        targets["Stress_Event_20D"],
        higher_predictor_means_event=False,
    )

    current_mae = safe_spearman(
        current_be_score,
        targets["Fwd_MAE_20D"],
    )
    b_mae = safe_spearman(
        b_score,
        targets["Fwd_MAE_20D"],
    )
    e_mae = safe_spearman(
        e_score,
        targets["Fwd_MAE_20D"],
    )

    role_table = pd.DataFrame([
        {
            "Modell": "A · Current",
            "Direction 20D": current_direction_20d,
            "Signals 20D": current_signals_20d,
            "Stress AUC 20D": current_auc,
            "Score vs bessere FwdMAE": current_mae,
        },
        {
            "Modell": "B · Full Literature",
            "Direction 20D": b_direction_20d,
            "Signals 20D": b_signals_20d,
            "Stress AUC 20D": b_auc,
            "Score vs bessere FwdMAE": b_mae,
        },
        {
            "Modell": "E · Lit Pillars only",
            "Direction 20D": e_direction_20d,
            "Signals 20D": e_signals_20d,
            "Stress AUC 20D": e_auc,
            "Score vs bessere FwdMAE": e_mae,
        },
    ])

    st.dataframe(
        role_table.style.format(
            {
                "Direction 20D": "{:.1%}",
                "Signals 20D": "{:.0f}",
                "Stress AUC 20D": "{:.3f}",
                "Score vs bessere FwdMAE": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # ========================================================
    # 3. NON-OVERLAP
    # ========================================================

    st.markdown(
        "### 3. Non-Overlap-Diagnostik – 20D / 60D"
    )

    nonoverlap_rows = []

    for horizon in [20, 60]:
        target_h = targets[f"Fwd_Return_{horizon}D"]

        for model_name, score in [
            ("A · Current", current_be_score),
            ("B · Full Literature", b_score),
            ("E · Lit Pillars only", e_score),
        ]:
            median_ic, phases = phase_median_nonoverlap_ic(
                score,
                target_h,
                horizon,
            )

            nonoverlap_rows.append({
                "Horizont": f"{horizon}D",
                "Modell": model_name,
                "Median Non-Overlap IC": median_ic,
                "Verfügbare Phasen": phases,
            })

    eurusd_confirmation_nonoverlap = pd.DataFrame(
        nonoverlap_rows
    )

    st.dataframe(
        eurusd_confirmation_nonoverlap.style.format(
            {
                "Median Non-Overlap IC": "{:+.3f}",
                "Verfügbare Phasen": "{:.0f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Die Non-Overlap-Diagnose reduziert die Scheinsicherheit "
        "durch stark überlappende 20D-/60D-Forward-Returns. "
        "Sie verändert weder Scores noch Bestätigungskriterien."
    )

    # ========================================================
    # 4. PERIOD ROBUSTNESS
    # ========================================================

    st.markdown(
        "### 4. Perioden-Robustheit"
    )

    score_map = {
        "A · Current": current_be_score,
        "B · Full Literature": b_score,
        "E · Lit Pillars only": e_score,
    }

    period_rows = []

    period_rows.extend(
        confirmation_period_metrics(
            "Gesamter Common Sample",
            be_common,
            score_map,
            targets,
        )
    )

    mask_2017_2025 = (
        be_common
        & (targets.index.year >= 2017)
        & (targets.index.year <= 2025)
    )

    period_rows.extend(
        confirmation_period_metrics(
            "2017–2025",
            mask_2017_2025,
            score_map,
            targets,
        )
    )

    mask_2021_2025 = (
        be_common
        & (targets.index.year >= 2021)
        & (targets.index.year <= 2025)
    )

    period_rows.extend(
        confirmation_period_metrics(
            "2021–2025",
            mask_2021_2025,
            score_map,
            targets,
        )
    )

    eurusd_confirmation_periods = pd.DataFrame(
        period_rows
    )

    st.dataframe(
        eurusd_confirmation_periods.style.format(
            {
                "N": "{:.0f}",
                "IC 20D": "{:+.3f}",
                "IC 60D": "{:+.3f}",
                "Direction 20D": "{:.1%}",
                "Signals 20D": "{:.0f}",
                "Stress AUC 20D": "{:.3f}",
                "Score vs bessere FwdMAE": "{:+.3f}",
                "Score vs niedrigere FwdVol": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # ========================================================
    # 5. YEARLY STABILITY
    # ========================================================

    st.markdown(
        "### 5. Kalenderjahr-Stabilität – IC20"
    )

    yearly_rows = []

    yearly_frame = pd.DataFrame({
        "Current": current_be_score,
        "B": b_score,
        "E": e_score,
        "Target20": targets["Fwd_Return_20D"],
    })

    for year, year_df in yearly_frame.groupby(
        yearly_frame.index.year
    ):
        usable = year_df.dropna()

        if len(usable) < 40:
            continue

        current_year_ic = safe_spearman(
            usable["Current"],
            usable["Target20"],
        )
        b_year_ic = safe_spearman(
            usable["B"],
            usable["Target20"],
        )
        e_year_ic = safe_spearman(
            usable["E"],
            usable["Target20"],
        )

        yearly_rows.append({
            "Jahr": int(year),
            "Current IC20": current_year_ic,
            "B IC20": b_year_ic,
            "E IC20": e_year_ic,
            "B−Current": (
                b_year_ic - current_year_ic
                if (
                    np.isfinite(b_year_ic)
                    and np.isfinite(current_year_ic)
                )
                else np.nan
            ),
            "E−Current": (
                e_year_ic - current_year_ic
                if (
                    np.isfinite(e_year_ic)
                    and np.isfinite(current_year_ic)
                )
                else np.nan
            ),
            "B−E": (
                b_year_ic - e_year_ic
                if (
                    np.isfinite(b_year_ic)
                    and np.isfinite(e_year_ic)
                )
                else np.nan
            ),
            "B positiv": bool(
                np.isfinite(b_year_ic)
                and b_year_ic > 0
            ),
            "B besser Current": bool(
                np.isfinite(b_year_ic)
                and np.isfinite(current_year_ic)
                and b_year_ic > current_year_ic
            ),
            "E besser Current": bool(
                np.isfinite(e_year_ic)
                and np.isfinite(current_year_ic)
                and e_year_ic > current_year_ic
            ),
        })

    eurusd_confirmation_yearly = pd.DataFrame(
        yearly_rows
    )

    if not eurusd_confirmation_yearly.empty:
        st.dataframe(
            eurusd_confirmation_yearly.style.format(
                {
                    "Current IC20": "{:+.3f}",
                    "B IC20": "{:+.3f}",
                    "E IC20": "{:+.3f}",
                    "B−Current": "{:+.3f}",
                    "E−Current": "{:+.3f}",
                    "B−E": "{:+.3f}",
                },
                na_rep="n/a",
            ),
            hide_index=True,
            use_container_width=True,
        )

        full_years = eurusd_confirmation_yearly[
            eurusd_confirmation_yearly["Jahr"]
            < pd.Timestamp.now().year
        ]

        if not full_years.empty:
            st.caption(
                f"Vollständige Jahre: B schlägt Current in "
                f"**{int(full_years['B besser Current'].sum())} von "
                f"{len(full_years)} Jahren**, E schlägt Current in "
                f"**{int(full_years['E besser Current'].sum())} von "
                f"{len(full_years)} Jahren**; B besitzt in "
                f"**{int(full_years['B positiv'].sum())} von "
                f"{len(full_years)} Jahren** einen positiven IC20."
            )

    # ========================================================
    # 6. FROZEN CONFIRMATION RULES
    # ========================================================

    st.markdown(
        "### 6. Bestätigungskriterien"
    )

    b_direction_gate = (
        b_positive_ic_count >= 2
        and np.isfinite(b_direction_20d)
        and b_direction_20d >= 0.50
    )

    b_relative_gate = (
        b_ic_wins_vs_current >= 2
        and b_bootstrap_75_vs_current >= 2
    )

    b_vs_e_direction_role = (
        b_ic_wins_vs_e >= 2
    )

    e_risk_gate = (
        np.isfinite(e_auc)
        and e_auc >= 0.55
        and np.isfinite(current_auc)
        and e_auc >= current_auc
        and np.isfinite(b_auc)
        and e_auc >= b_auc
    )

    recent_b = eurusd_confirmation_periods[
        (
            eurusd_confirmation_periods["Periode"]
            == "2021–2025"
        )
        & (
            eurusd_confirmation_periods["Modell"]
            == "B · Full Literature"
        )
    ]

    if not recent_b.empty:
        recent_b_ic20 = float(
            recent_b.iloc[0]["IC 20D"]
        )
        recent_b_ic60 = float(
            recent_b.iloc[0]["IC 60D"]
        )

        b_recent_gate = (
            (
                np.isfinite(recent_b_ic20)
                and recent_b_ic20 > 0
            )
            or
            (
                np.isfinite(recent_b_ic60)
                and recent_b_ic60 > 0
            )
        )
    else:
        recent_b_ic20 = np.nan
        recent_b_ic60 = np.nan
        b_recent_gate = False

    gate_rows = [
        {
            "Kriterium": (
                "B Direction-Gate: ≥2/3 positive ICs und Direction20 ≥50%"
            ),
            "Erfüllt": b_direction_gate,
            "Messwert": (
                f"{b_positive_ic_count}/3 positive ICs · "
                f"Direction {b_direction_20d:.1%}"
                if np.isfinite(b_direction_20d)
                else f"{b_positive_ic_count}/3 · n/a"
            ),
        },
        {
            "Kriterium": (
                "B relativ zu Current: IC besser ≥2/3 und Bootstrap ≥75% ≥2/3"
            ),
            "Erfüllt": b_relative_gate,
            "Messwert": (
                f"IC-Wins {b_ic_wins_vs_current}/3 · "
                f"Bootstrap {b_bootstrap_75_vs_current}/3"
            ),
        },
        {
            "Kriterium": (
                "B gegenüber E als Direction-Kandidat: IC besser ≥2/3"
            ),
            "Erfüllt": b_vs_e_direction_role,
            "Messwert": f"{b_ic_wins_vs_e}/3",
        },
        {
            "Kriterium": (
                "E Risk-Gate: AUC ≥0.55 und ≥ Current und ≥ B"
            ),
            "Erfüllt": e_risk_gate,
            "Messwert": (
                f"E {e_auc:.3f} · "
                f"Current {current_auc:.3f} · "
                f"B {b_auc:.3f}"
                if (
                    np.isfinite(e_auc)
                    and np.isfinite(current_auc)
                    and np.isfinite(b_auc)
                )
                else "n/a"
            ),
        },
        {
            "Kriterium": (
                "B jüngere Periode 2021–2025: IC20 oder IC60 positiv"
            ),
            "Erfüllt": b_recent_gate,
            "Messwert": (
                f"IC20 {recent_b_ic20:+.3f} · "
                f"IC60 {recent_b_ic60:+.3f}"
                if (
                    np.isfinite(recent_b_ic20)
                    and np.isfinite(recent_b_ic60)
                )
                else "n/a"
            ),
        },
    ]

    eurusd_confirmation_gate = pd.DataFrame(
        gate_rows
    )

    eurusd_confirmation_gate["Status"] = (
        eurusd_confirmation_gate["Erfüllt"].map(
            {
                True: "✅",
                False: "❌",
            }
        )
    )

    st.dataframe(
        eurusd_confirmation_gate[
            [
                "Kriterium",
                "Status",
                "Messwert",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    passed_count = int(
        eurusd_confirmation_gate["Erfüllt"].sum()
    )
    total_count = int(
        len(eurusd_confirmation_gate)
    )

    st.markdown(
        "### 7. EUR/USD Bestätigungsurteil"
    )

    if passed_count == total_count:
        st.success(
            "🟢 **B UND E IN IHREN GETRENNTEN ROLLEN BESTÄTIGT.** "
            "B ist der stärkere 20–60D-Direction-Kandidat; E bestätigt "
            "den stärkeren Risk-State. Der nächste methodische Schritt "
            "wäre ein chronologischer Walk-Forward-Test mit exakt "
            "eingefrorenen B-/E-Konfigurationen – nicht weiteres "
            "Gewichtstuning."
        )

    elif passed_count >= 4:
        st.warning(
            f"🟡 **EUR/USD BESTÄTIGUNG GEMISCHT POSITIV "
            f"({passed_count}/{total_count}).** "
            "Mindestens ein festes Kriterium ist nicht erfüllt. "
            "Dieses Ergebnis wird zuerst untersucht, bevor ein "
            "Walk-Forward-Test festgelegt wird."
        )

    else:
        st.error(
            f"🔴 **EUR/USD B/E NICHT AUSREICHEND BESTÄTIGT "
            f"({passed_count}/{total_count}).** "
            "Dann sollte nicht weiter an denselben Gewichten optimiert "
            "werden. Stattdessen wäre ein separater relativer "
            "FX-Makrotest sinnvoll."
        )

    st.info(
        "Dieser Bestätigungstest nutzt weiterhin dieselbe Historie, "
        "auf der B/E als Kandidaten identifiziert wurden. Selbst ein "
        "vollständiges Bestehen ist daher noch kein echter unseen Holdout, "
        "sondern rechtfertigt höchstens den nächsten Walk-Forward-Schritt."
    )


# ============================================================
# 20A. WTI MODEL-D CONFIRMATION TEST
# ============================================================
if selected_asset == "WTI Crude Oil":
    st.markdown("---")
    st.subheader("🔬 WTI Bestätigungstest – Current vs. Model D")
    st.caption(
        "Model D = Current-Säulengewichte + die bereits vorab definierten "
        "Literature-Subgewichte. Es werden keine neuen Gewichte gesucht oder optimiert."
    )

    model_d_name = "D · Lit Subweights only"
    current_d_frame = model_frames[MODEL_CURRENT]
    model_d_frame = diagnostic_frames[model_d_name]

    # Strict common sample for A vs D only.
    d_common = (
        current_d_frame["Final_Regime_Score"].notna()
        & model_d_frame["Final_Regime_Score"].notna()
        & (current_d_frame["Model_Data_Coverage"] >= min_coverage)
        & (model_d_frame["Model_Data_Coverage"] >= min_coverage)
    )

    current_d_score = current_d_frame["Final_Regime_Score"].where(d_common)
    model_d_score = model_d_frame["Final_Regime_Score"].where(d_common)

    d_rows = []
    d_positive_ic_count = 0
    d_ic_wins = 0
    d_bootstrap_75 = 0

    for horizon in FORWARD_HORIZONS:
        target_h = targets[f"Fwd_Return_{horizon}D"]
        current_ic_h = safe_spearman(current_d_score, target_h)
        model_d_ic_h = safe_spearman(model_d_score, target_h)

        if np.isfinite(model_d_ic_h) and model_d_ic_h > 0:
            d_positive_ic_count += 1
        if np.isfinite(model_d_ic_h) and np.isfinite(current_ic_h) and model_d_ic_h > current_ic_h:
            d_ic_wins += 1

        boot_h = block_bootstrap_ic_difference(
            current_d_score,
            model_d_score,
            target_h,
            block_length=int(bootstrap_block),
            n_boot=int(bootstrap_runs),
            seed=100 + int(horizon),
        )

        if np.isfinite(boot_h["prob_positive"]) and boot_h["prob_positive"] >= 0.75:
            d_bootstrap_75 += 1

        d_rows.append({
            "Horizont": f"{horizon}D",
            "Current IC": current_ic_h,
            "Model D IC": model_d_ic_h,
            "Δ IC D−Current": (
                model_d_ic_h - current_ic_h
                if np.isfinite(model_d_ic_h) and np.isfinite(current_ic_h)
                else np.nan
            ),
            "Bootstrap P(Δ>0)": boot_h["prob_positive"],
            "Bootstrap CI Low": boot_h["lower"],
            "Bootstrap CI High": boot_h["upper"],
            "Bootstrap N": boot_h["n"],
        })

    d_validation_table = pd.DataFrame(d_rows)

    st.markdown("### 1. Forward-IC & Bootstrap – 5D / 20D / 60D")
    st.dataframe(
        d_validation_table.style.format(
            {
                "Current IC": "{:+.3f}",
                "Model D IC": "{:+.3f}",
                "Δ IC D−Current": "{:+.3f}",
                "Bootstrap P(Δ>0)": "{:.1%}",
                "Bootstrap CI Low": "{:+.3f}",
                "Bootstrap CI High": "{:+.3f}",
                "Bootstrap N": "{:.0f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    d_direction_20d, d_signal_count = directional_accuracy(
        model_d_score,
        targets["Fwd_Return_20D"],
    )
    current_direction_20d, _ = directional_accuracy(
        current_d_score,
        targets["Fwd_Return_20D"],
    )
    d_stress_auc = binary_auc(
        model_d_score,
        targets["Stress_Event_20D"],
        higher_predictor_means_event=False,
    )
    current_stress_auc = binary_auc(
        current_d_score,
        targets["Stress_Event_20D"],
        higher_predictor_means_event=False,
    )

    d_direction_gate = (
        d_positive_ic_count >= 2
        and np.isfinite(d_direction_20d)
        and d_direction_20d >= 0.50
    )
    d_risk_gate = np.isfinite(d_stress_auc) and d_stress_auc >= 0.50
    d_strict_gate = d_direction_gate and d_risk_gate

    st.markdown("### 2. Absolute Validity – Model D")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Positive ICs", f"{d_positive_ic_count} / 3")
    a2.metric("Direction 20D", f"{d_direction_20d:.1%}" if np.isfinite(d_direction_20d) else "n/a")
    a3.metric("Stress AUC 20D", f"{d_stress_auc:.3f}" if np.isfinite(d_stress_auc) else "n/a")
    a4.metric("Strict Gate", "✅ BESTANDEN" if d_strict_gate else "❌ NICHT BESTANDEN")
    st.write(f"**Direction-Gate:** {'✅ bestanden' if d_direction_gate else '❌ nicht bestanden'}")
    st.write(f"**Risk-Gate:** {'✅ bestanden' if d_risk_gate else '❌ nicht bestanden'}")

    st.markdown("### 3. Relative Robustheit gegenüber Current")
    d_direction_delta = d_direction_20d - current_direction_20d
    d_stress_delta = d_stress_auc - current_stress_auc
    d_relative_rules = [
        ("IC besser als Current bei mindestens 2/3 Horizonten", d_ic_wins >= 2, f"{d_ic_wins}/3"),
        ("Bootstrap P(ΔIC>0) ≥75% bei mindestens 2/3 Horizonten", d_bootstrap_75 >= 2, f"{d_bootstrap_75}/3"),
        ("Direction 20D nicht schlechter als Current", np.isfinite(d_direction_delta) and d_direction_delta >= 0, f"{d_direction_delta*100:+.2f} pp" if np.isfinite(d_direction_delta) else "n/a"),
        ("Stress-AUC nicht schlechter als Current", np.isfinite(d_stress_delta) and d_stress_delta >= 0, f"{d_stress_delta:+.3f}" if np.isfinite(d_stress_delta) else "n/a"),
    ]
    d_relative_df = pd.DataFrame([
        {"Kriterium": label, "Erfüllt": "✅" if passed else "❌", "Messwert": detail}
        for label, passed, detail in d_relative_rules
    ])
    st.dataframe(d_relative_df, hide_index=True, use_container_width=True)
    d_relative_passed = sum(bool(passed) for _, passed, _ in d_relative_rules)

    st.markdown("### 4. WTI Model-D Urteil")
    if d_strict_gate and d_relative_passed == len(d_relative_rules):
        st.success(
            "🟢 **MODEL D BESTÄTIGT.** Absolute Direction-/Risk-Gates und alle "
            "relativen Robustheitskriterien gegenüber Current sind erfüllt. "
            "Nächster Schritt wäre ein separater Holdout-/Walk-Forward-Test – "
            "nicht die sofortige Änderung der Produktions-Engine."
        )
    elif d_risk_gate and not d_direction_gate:
        st.warning(
            "🟡 **MODEL D ALS RISK-STATE-KANDIDAT.** Der Risk-Gate ist bestanden, "
            "der Direction-Gate nicht vollständig. Keine Übernahme als Direction-Modell."
        )
    elif d_direction_gate and not d_risk_gate:
        st.warning(
            "🟡 **MODEL D MIT DIRECTION-NUTZEN, ABER OHNE BESTÄTIGTEN RISK-GATE.**"
        )
    else:
        st.error(
            "🔴 **MODEL D NICHT BESTÄTIGT.** Keine ausreichende Kombination aus "
            "absoluter und relativer Robustheit."
        )

    st.markdown("### 5. Kalenderjahr-Stabilität – 20D IC")
    d_year_rows = []
    validation_frame = pd.DataFrame({
        "Current": current_d_score,
        "ModelD": model_d_score,
        "Target20": targets["Fwd_Return_20D"],
    })

    for year, year_df in validation_frame.groupby(validation_frame.index.year):
        if len(year_df.dropna()) < 40:
            continue
        current_ic_year = safe_spearman(year_df["Current"], year_df["Target20"])
        d_ic_year = safe_spearman(year_df["ModelD"], year_df["Target20"])
        d_year_rows.append({
            "Jahr": int(year),
            "Current IC20": current_ic_year,
            "Model D IC20": d_ic_year,
            "D−Current": d_ic_year-current_ic_year if np.isfinite(d_ic_year) and np.isfinite(current_ic_year) else np.nan,
            "D positiv": bool(np.isfinite(d_ic_year) and d_ic_year > 0),
            "D besser": bool(np.isfinite(d_ic_year) and np.isfinite(current_ic_year) and d_ic_year > current_ic_year),
        })

    d_year_df = pd.DataFrame(d_year_rows)
    if not d_year_df.empty:
        st.dataframe(
            d_year_df.style.format(
                {"Current IC20": "{:+.3f}", "Model D IC20": "{:+.3f}", "D−Current": "{:+.3f}"},
                na_rep="n/a",
            ),
            hide_index=True,
            use_container_width=True,
        )
        full_years = d_year_df[d_year_df["Jahr"] < pd.Timestamp.now().year]
        if not full_years.empty:
            st.caption(
                f"Vollständige Kalenderjahre: Model D war in **{int(full_years['D besser'].sum())} "
                f"von {len(full_years)} Jahren** besser als Current und hatte in "
                f"**{int(full_years['D positiv'].sum())} von {len(full_years)} Jahren** "
                "einen positiven 20D-IC."
            )

    st.info(
        "Dieser Bestätigungstest verändert keine Produktionsgewichte. Selbst bei "
        "bestandenem Gate folgt zunächst ein separater Holdout-/Walk-Forward-Test."
    )




# ============================================================
# 20B. EUR/USD WALK-FORWARD – FROZEN B / E
# ============================================================

eurusd_wf_yearly_df = pd.DataFrame()
eurusd_wf_pooled_df = pd.DataFrame()
eurusd_wf_bootstrap_df = pd.DataFrame()
eurusd_wf_nonoverlap_df = pd.DataFrame()
eurusd_wf_gate_df = pd.DataFrame()

if selected_asset == "EUR/USD":
    st.markdown("---")
    st.subheader(
        "🧭 EUR/USD Walk-Forward – Frozen B / E"
    )

    st.caption(
        "B (Full Literature Prior) und E (Literature Pillars only) "
        "sind vollständig eingefroren. B wird als 20–60D-Direction-"
        "Challenger geprüft, E als Risk-State-Challenger. "
        "Es findet **keine Neuoptimierung innerhalb der Testjahre** statt."
    )

    wf_current_frame = model_frames[
        MODEL_CURRENT
    ]

    wf_b_frame = model_frames[
        MODEL_LITERATURE
    ]

    wf_e_frame = diagnostic_frames[
        "E · Lit Pillars only"
    ]

    wf_current_score = wf_current_frame[
        "Final_Regime_Score"
    ]

    wf_b_score = wf_b_frame[
        "Final_Regime_Score"
    ]

    wf_e_score = wf_e_frame[
        "Final_Regime_Score"
    ]

    wf_common = (
        wf_current_score.notna()
        & wf_b_score.notna()
        & wf_e_score.notna()
        & (
            wf_current_frame[
                "Model_Data_Coverage"
            ]
            >= float(
                min_coverage
            )
        )
        & (
            wf_b_frame[
                "Model_Data_Coverage"
            ]
            >= float(
                min_coverage
            )
        )
        & (
            wf_e_frame[
                "Model_Data_Coverage"
            ]
            >= float(
                min_coverage
            )
        )
    )

    first_data_year = int(
        raw_df.index.min().year
    )

    eurusd_wf_first_year = max(
        int(
            EURUSD_WF_BASE_FIRST_TEST_YEAR
        ),
        first_data_year + 6,
    )

    current_calendar_year = int(
        pd.Timestamp.now().year
    )

    last_full_test_year = (
        current_calendar_year - 1
    )

    st.info(
        f"**Festes Schema:** erster formaler Test "
        f"**{eurusd_wf_first_year}**, letztes vollständiges Testjahr "
        f"**{last_full_test_year}**. Das laufende Jahr "
        f"**{current_calendar_year}** wird nur informativ angezeigt."
    )

    st.markdown(
        """
**Vorab eingefrorene Rollen**

- **B / Full Literature:** Direction-Regime auf **20D und 60D**
- **E / Literature Pillars only:** **Stress-/Risk-State**
- **A / Current:** Referenzmodell

Die Historie vor jedem Testjahr dient nur als bereits vorhandener
Point-in-Time-Lookback-Kontext der Faktoren. Gewichte werden weder
trainiert noch rollierend angepasst.
"""
    )

    eurusd_wf_yearly_df = (
        eurusd_be_walk_forward_year_metrics(
            wf_current_score,
            wf_b_score,
            wf_e_score,
            wf_current_frame[
                "Model_Data_Coverage"
            ],
            wf_b_frame[
                "Model_Data_Coverage"
            ],
            wf_e_frame[
                "Model_Data_Coverage"
            ],
            targets,
            wf_common,
            eurusd_wf_first_year,
            current_calendar_year,
        )
    )

    if eurusd_wf_yearly_df.empty:
        st.warning(
            "Nicht genügend EUR/USD-Testjahre für den "
            "Walk-Forward-Test vorhanden."
        )

    else:
        st.markdown(
            "### 1. Chronologische Testjahre"
        )

        st.dataframe(
            eurusd_wf_yearly_df.style.format(
                {
                    "N Common": "{:.0f}",
                    "Ø Coverage Current": "{:.1f}%",
                    "Ø Coverage B": "{:.1f}%",
                    "Ø Coverage E": "{:.1f}%",
                    "Current IC 5D": "{:+.3f}",
                    "B IC 5D": "{:+.3f}",
                    "E IC 5D": "{:+.3f}",
                    "Current IC 20D": "{:+.3f}",
                    "B IC 20D": "{:+.3f}",
                    "E IC 20D": "{:+.3f}",
                    "Current IC 60D": "{:+.3f}",
                    "B IC 60D": "{:+.3f}",
                    "E IC 60D": "{:+.3f}",
                    "Δ IC20 B−Current": "{:+.3f}",
                    "Δ IC60 B−Current": "{:+.3f}",
                    "Current Direction 20D": "{:.1%}",
                    "B Direction 20D": "{:.1%}",
                    "Current Stress AUC": "{:.3f}",
                    "B Stress AUC": "{:.3f}",
                    "E Stress AUC": "{:.3f}",
                },
                na_rep="n/a",
            ),
            hide_index=True,
            use_container_width=True,
        )

        completed_years = eurusd_wf_yearly_df[
            eurusd_wf_yearly_df[
                "Vollständiges Jahr"
            ].astype(bool)
        ].copy()

        if completed_years.empty:
            st.warning(
                "Noch keine vollständigen EUR/USD-Testjahre "
                "für das formale Gate vorhanden."
            )

        else:
            completed_year_list = (
                completed_years[
                    "Testjahr"
                ]
                .astype(int)
                .tolist()
            )

            pooled_mask = (
                wf_common
                & pd.Series(
                    targets.index.year,
                    index=targets.index,
                ).isin(
                    completed_year_list
                )
            )

            # =================================================
            # 2. POOLED COMPLETE-YEAR METRICS
            # =================================================

            st.markdown(
                "### 2. Gepoolte vollständige Testjahre"
            )

            current_pooled = pooled_walk_forward_metrics(
                wf_current_score,
                targets,
                pooled_mask,
            )

            b_pooled = pooled_walk_forward_metrics(
                wf_b_score,
                targets,
                pooled_mask,
            )

            e_pooled = pooled_walk_forward_metrics(
                wf_e_score,
                targets,
                pooled_mask,
            )

            eurusd_wf_pooled_df = pd.DataFrame(
                [
                    {
                        "Modell": "A · Current",
                        **current_pooled,
                    },
                    {
                        "Modell": "B · Full Literature",
                        **b_pooled,
                    },
                    {
                        "Modell": "E · Lit Pillars only",
                        **e_pooled,
                    },
                ]
            )

            st.dataframe(
                eurusd_wf_pooled_df.style.format(
                    {
                        "IC 5D": "{:+.3f}",
                        "IC 20D": "{:+.3f}",
                        "IC 60D": "{:+.3f}",
                        "Direction 5D": "{:.1%}",
                        "Direction 20D": "{:.1%}",
                        "Direction 60D": "{:.1%}",
                        "Signals 5D": "{:.0f}",
                        "Signals 20D": "{:.0f}",
                        "Signals 60D": "{:.0f}",
                        "Stress AUC 20D": "{:.3f}",
                        "Score vs niedrigere FwdVol": "{:+.3f}",
                        "Score vs bessere FwdMAE": "{:+.3f}",
                    },
                    na_rep="n/a",
                ),
                hide_index=True,
                use_container_width=True,
            )

            # =================================================
            # 3. BOOTSTRAP – COMPLETE WF YEARS ONLY
            # =================================================

            st.markdown(
                "### 3. Block-Bootstrap nur über vollständige Testjahre"
            )

            bootstrap_rows = []

            for horizon in [
                20,
                60,
            ]:
                target_h = targets[
                    f"Fwd_Return_{horizon}D"
                ].where(
                    pooled_mask
                )

                boot_b_current = (
                    block_bootstrap_ic_difference(
                        wf_current_score.where(
                            pooled_mask
                        ),
                        wf_b_score.where(
                            pooled_mask
                        ),
                        target_h,
                        block_length=int(
                            bootstrap_block
                        ),
                        n_boot=int(
                            bootstrap_runs
                        ),
                        seed=3100 + int(
                            horizon
                        ),
                    )
                )

                boot_e_current = (
                    block_bootstrap_ic_difference(
                        wf_current_score.where(
                            pooled_mask
                        ),
                        wf_e_score.where(
                            pooled_mask
                        ),
                        target_h,
                        block_length=int(
                            bootstrap_block
                        ),
                        n_boot=int(
                            bootstrap_runs
                        ),
                        seed=3200 + int(
                            horizon
                        ),
                    )
                )

                boot_b_e = (
                    block_bootstrap_ic_difference(
                        wf_e_score.where(
                            pooled_mask
                        ),
                        wf_b_score.where(
                            pooled_mask
                        ),
                        target_h,
                        block_length=int(
                            bootstrap_block
                        ),
                        n_boot=int(
                            bootstrap_runs
                        ),
                        seed=3300 + int(
                            horizon
                        ),
                    )
                )

                bootstrap_rows.extend(
                    [
                        {
                            "Horizont": f"{horizon}D",
                            "Vergleich": "B − Current",
                            "Δ IC": boot_b_current[
                                "observed"
                            ],
                            "P(Δ>0)": boot_b_current[
                                "prob_positive"
                            ],
                            "CI Low": boot_b_current[
                                "lower"
                            ],
                            "CI High": boot_b_current[
                                "upper"
                            ],
                            "N": boot_b_current[
                                "n"
                            ],
                        },
                        {
                            "Horizont": f"{horizon}D",
                            "Vergleich": "E − Current",
                            "Δ IC": boot_e_current[
                                "observed"
                            ],
                            "P(Δ>0)": boot_e_current[
                                "prob_positive"
                            ],
                            "CI Low": boot_e_current[
                                "lower"
                            ],
                            "CI High": boot_e_current[
                                "upper"
                            ],
                            "N": boot_e_current[
                                "n"
                            ],
                        },
                        {
                            "Horizont": f"{horizon}D",
                            "Vergleich": "B − E",
                            "Δ IC": boot_b_e[
                                "observed"
                            ],
                            "P(Δ>0)": boot_b_e[
                                "prob_positive"
                            ],
                            "CI Low": boot_b_e[
                                "lower"
                            ],
                            "CI High": boot_b_e[
                                "upper"
                            ],
                            "N": boot_b_e[
                                "n"
                            ],
                        },
                    ]
                )

            eurusd_wf_bootstrap_df = pd.DataFrame(
                bootstrap_rows
            )

            st.dataframe(
                eurusd_wf_bootstrap_df.style.format(
                    {
                        "Δ IC": "{:+.3f}",
                        "P(Δ>0)": "{:.1%}",
                        "CI Low": "{:+.3f}",
                        "CI High": "{:+.3f}",
                        "N": "{:.0f}",
                    },
                    na_rep="n/a",
                ),
                hide_index=True,
                use_container_width=True,
            )

            # =================================================
            # 4. NON-OVERLAP – COMPLETE WF YEARS ONLY
            # =================================================

            st.markdown(
                "### 4. Non-Overlap-Diagnostik – vollständige Testjahre"
            )

            nonoverlap_rows = []

            for horizon in [
                20,
                60,
            ]:
                target_h = targets[
                    f"Fwd_Return_{horizon}D"
                ].where(
                    pooled_mask
                )

                for model_name, score in [
                    (
                        "A · Current",
                        wf_current_score.where(
                            pooled_mask
                        ),
                    ),
                    (
                        "B · Full Literature",
                        wf_b_score.where(
                            pooled_mask
                        ),
                    ),
                    (
                        "E · Lit Pillars only",
                        wf_e_score.where(
                            pooled_mask
                        ),
                    ),
                ]:
                    median_ic, phases = (
                        phase_median_nonoverlap_ic(
                            score,
                            target_h,
                            horizon,
                        )
                    )

                    nonoverlap_rows.append(
                        {
                            "Horizont": f"{horizon}D",
                            "Modell": model_name,
                            "Median Non-Overlap IC": median_ic,
                            "Verfügbare Phasen": phases,
                        }
                    )

            eurusd_wf_nonoverlap_df = pd.DataFrame(
                nonoverlap_rows
            )

            st.dataframe(
                eurusd_wf_nonoverlap_df.style.format(
                    {
                        "Median Non-Overlap IC": "{:+.3f}",
                        "Verfügbare Phasen": "{:.0f}",
                    },
                    na_rep="n/a",
                ),
                hide_index=True,
                use_container_width=True,
            )

            # =================================================
            # 5. PRE-DEFINED GATE
            # =================================================

            st.markdown(
                "### 5. Vorab festgelegter Walk-Forward-Gate"
            )

            n_years = int(
                len(
                    completed_years
                )
            )

            b_ic20_wins = int(
                completed_years[
                    "B besser IC20"
                ]
                .astype(bool)
                .sum()
            )

            b_ic60_wins = int(
                completed_years[
                    "B besser IC60"
                ]
                .astype(bool)
                .sum()
            )

            b_ic20_win_rate = (
                b_ic20_wins / n_years
                if n_years > 0
                else np.nan
            )

            b_ic60_win_rate = (
                b_ic60_wins / n_years
                if n_years > 0
                else np.nan
            )

            e_auc_valid = (
                completed_years[
                    "E Stress AUC"
                ].notna()
                & completed_years[
                    "Current Stress AUC"
                ].notna()
            )

            e_auc_years = int(
                e_auc_valid.sum()
            )

            e_auc_wins = int(
                completed_years.loc[
                    e_auc_valid,
                    "E AUC besser Current"
                ]
                .astype(bool)
                .sum()
            )

            e_auc_win_rate = (
                e_auc_wins / e_auc_years
                if e_auc_years > 0
                else np.nan
            )

            pooled_b_ic20 = float(
                b_pooled[
                    "IC 20D"
                ]
            )

            pooled_b_ic60 = float(
                b_pooled[
                    "IC 60D"
                ]
            )

            pooled_b_direction20 = float(
                b_pooled[
                    "Direction 20D"
                ]
            )

            pooled_current_auc = float(
                current_pooled[
                    "Stress AUC 20D"
                ]
            )

            pooled_b_auc = float(
                b_pooled[
                    "Stress AUC 20D"
                ]
            )

            pooled_e_auc = float(
                e_pooled[
                    "Stress AUC 20D"
                ]
            )

            def _bootstrap_prob(
                horizon,
                comparison,
            ):
                rows = eurusd_wf_bootstrap_df[
                    (
                        eurusd_wf_bootstrap_df[
                            "Horizont"
                        ]
                        == horizon
                    )
                    & (
                        eurusd_wf_bootstrap_df[
                            "Vergleich"
                        ]
                        == comparison
                    )
                ]

                if rows.empty:
                    return np.nan

                return float(
                    rows.iloc[
                        0
                    ][
                        "P(Δ>0)"
                    ]
                )

            p_b_current_20 = _bootstrap_prob(
                "20D",
                "B − Current",
            )

            p_b_current_60 = _bootstrap_prob(
                "60D",
                "B − Current",
            )

            def _nonoverlap_value(
                horizon,
                model,
            ):
                rows = eurusd_wf_nonoverlap_df[
                    (
                        eurusd_wf_nonoverlap_df[
                            "Horizont"
                        ]
                        == horizon
                    )
                    & (
                        eurusd_wf_nonoverlap_df[
                            "Modell"
                        ]
                        == model
                    )
                ]

                if rows.empty:
                    return np.nan

                return float(
                    rows.iloc[
                        0
                    ][
                        "Median Non-Overlap IC"
                    ]
                )

            no_current_20 = _nonoverlap_value(
                "20D",
                "A · Current",
            )
            no_b_20 = _nonoverlap_value(
                "20D",
                "B · Full Literature",
            )
            no_current_60 = _nonoverlap_value(
                "60D",
                "A · Current",
            )
            no_b_60 = _nonoverlap_value(
                "60D",
                "B · Full Literature",
            )

            b_nonoverlap_gate = bool(
                np.isfinite(
                    no_b_20
                )
                and np.isfinite(
                    no_current_20
                )
                and np.isfinite(
                    no_b_60
                )
                and np.isfinite(
                    no_current_60
                )
                and no_b_20
                > no_current_20
                and no_b_60
                > no_current_60
            )

            b_bootstrap_gate = bool(
                np.isfinite(
                    p_b_current_20
                )
                and np.isfinite(
                    p_b_current_60
                )
                and p_b_current_20
                >= EURUSD_WF_MIN_BOOTSTRAP_PROB
                and p_b_current_60
                >= EURUSD_WF_MIN_BOOTSTRAP_PROB
            )

            e_pooled_risk_gate = bool(
                np.isfinite(
                    pooled_e_auc
                )
                and np.isfinite(
                    pooled_current_auc
                )
                and np.isfinite(
                    pooled_b_auc
                )
                and pooled_e_auc
                >= EURUSD_WF_MIN_STRESS_AUC
                and pooled_e_auc
                >= pooled_current_auc
                and pooled_e_auc
                >= pooled_b_auc
            )

            gate_rows = [
                {
                    "Rolle": "B Direction",
                    "Kriterium": (
                        "B schlägt Current beim IC20 in ≥60% "
                        "der vollständigen Testjahre"
                    ),
                    "Erfüllt": bool(
                        np.isfinite(
                            b_ic20_win_rate
                        )
                        and b_ic20_win_rate
                        >= EURUSD_WF_MIN_YEAR_WIN_RATE
                    ),
                    "Messwert": (
                        f"{b_ic20_wins}/{n_years} "
                        f"({b_ic20_win_rate:.1%})"
                        if np.isfinite(
                            b_ic20_win_rate
                        )
                        else "n/a"
                    ),
                },
                {
                    "Rolle": "B Direction",
                    "Kriterium": (
                        "B schlägt Current beim IC60 in ≥60% "
                        "der vollständigen Testjahre"
                    ),
                    "Erfüllt": bool(
                        np.isfinite(
                            b_ic60_win_rate
                        )
                        and b_ic60_win_rate
                        >= EURUSD_WF_MIN_YEAR_WIN_RATE
                    ),
                    "Messwert": (
                        f"{b_ic60_wins}/{n_years} "
                        f"({b_ic60_win_rate:.1%})"
                        if np.isfinite(
                            b_ic60_win_rate
                        )
                        else "n/a"
                    ),
                },
                {
                    "Rolle": "B Direction",
                    "Kriterium": (
                        "Gepoolter B IC20 > 0"
                    ),
                    "Erfüllt": bool(
                        np.isfinite(
                            pooled_b_ic20
                        )
                        and pooled_b_ic20
                        > EURUSD_WF_MIN_POOLED_IC
                    ),
                    "Messwert": (
                        f"{pooled_b_ic20:+.3f}"
                        if np.isfinite(
                            pooled_b_ic20
                        )
                        else "n/a"
                    ),
                },
                {
                    "Rolle": "B Direction",
                    "Kriterium": (
                        "Gepoolter B IC60 > 0"
                    ),
                    "Erfüllt": bool(
                        np.isfinite(
                            pooled_b_ic60
                        )
                        and pooled_b_ic60
                        > EURUSD_WF_MIN_POOLED_IC
                    ),
                    "Messwert": (
                        f"{pooled_b_ic60:+.3f}"
                        if np.isfinite(
                            pooled_b_ic60
                        )
                        else "n/a"
                    ),
                },
                {
                    "Rolle": "B Direction",
                    "Kriterium": (
                        "Gepoolte B Direction20 ≥50%"
                    ),
                    "Erfüllt": bool(
                        np.isfinite(
                            pooled_b_direction20
                        )
                        and pooled_b_direction20
                        >= EURUSD_WF_MIN_DIRECTION_20D
                    ),
                    "Messwert": (
                        f"{pooled_b_direction20:.1%}"
                        if np.isfinite(
                            pooled_b_direction20
                        )
                        else "n/a"
                    ),
                },
                {
                    "Rolle": "B Direction",
                    "Kriterium": (
                        "Bootstrap P(B>Current) ≥75% auf 20D UND 60D"
                    ),
                    "Erfüllt": b_bootstrap_gate,
                    "Messwert": (
                        f"20D {p_b_current_20:.1%} · "
                        f"60D {p_b_current_60:.1%}"
                        if (
                            np.isfinite(
                                p_b_current_20
                            )
                            and np.isfinite(
                                p_b_current_60
                            )
                        )
                        else "n/a"
                    ),
                },
                {
                    "Rolle": "B Direction",
                    "Kriterium": (
                        "Non-Overlap: B IC > Current auf 20D UND 60D"
                    ),
                    "Erfüllt": b_nonoverlap_gate,
                    "Messwert": (
                        f"20D B {no_b_20:+.3f} vs A {no_current_20:+.3f} · "
                        f"60D B {no_b_60:+.3f} vs A {no_current_60:+.3f}"
                        if (
                            np.isfinite(
                                no_b_20
                            )
                            and np.isfinite(
                                no_current_20
                            )
                            and np.isfinite(
                                no_b_60
                            )
                            and np.isfinite(
                                no_current_60
                            )
                        )
                        else "n/a"
                    ),
                },
                {
                    "Rolle": "E Risk",
                    "Kriterium": (
                        "Gepoolte E Stress-AUC ≥0.55 und ≥ Current und ≥ B"
                    ),
                    "Erfüllt": e_pooled_risk_gate,
                    "Messwert": (
                        f"E {pooled_e_auc:.3f} · "
                        f"A {pooled_current_auc:.3f} · "
                        f"B {pooled_b_auc:.3f}"
                        if (
                            np.isfinite(
                                pooled_e_auc
                            )
                            and np.isfinite(
                                pooled_current_auc
                            )
                            and np.isfinite(
                                pooled_b_auc
                            )
                        )
                        else "n/a"
                    ),
                },
                {
                    "Rolle": "E Risk",
                    "Kriterium": (
                        "E Stress-AUC > Current in ≥50% "
                        "der auswertbaren vollständigen Testjahre"
                    ),
                    "Erfüllt": bool(
                        np.isfinite(
                            e_auc_win_rate
                        )
                        and e_auc_win_rate
                        >= EURUSD_WF_MIN_E_YEAR_AUC_WIN_RATE
                    ),
                    "Messwert": (
                        f"{e_auc_wins}/{e_auc_years} "
                        f"({e_auc_win_rate:.1%})"
                        if np.isfinite(
                            e_auc_win_rate
                        )
                        else "n/a"
                    ),
                },
            ]

            eurusd_wf_gate_df = pd.DataFrame(
                gate_rows
            )

            eurusd_wf_gate_df[
                "Status"
            ] = eurusd_wf_gate_df[
                "Erfüllt"
            ].map(
                {
                    True: "✅",
                    False: "❌",
                }
            )

            st.dataframe(
                eurusd_wf_gate_df[
                    [
                        "Rolle",
                        "Kriterium",
                        "Status",
                        "Messwert",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )

            b_gate = eurusd_wf_gate_df[
                eurusd_wf_gate_df[
                    "Rolle"
                ]
                == "B Direction"
            ]

            e_gate = eurusd_wf_gate_df[
                eurusd_wf_gate_df[
                    "Rolle"
                ]
                == "E Risk"
            ]

            b_passed = int(
                b_gate[
                    "Erfüllt"
                ].sum()
            )
            b_total = int(
                len(
                    b_gate
                )
            )

            e_passed = int(
                e_gate[
                    "Erfüllt"
                ].sum()
            )
            e_total = int(
                len(
                    e_gate
                )
            )

            st.markdown(
                "### 6. Walk-Forward-Urteil"
            )

            if (
                b_passed == b_total
                and e_passed == e_total
            ):
                st.success(
                    "🟢 **EUR/USD B UND E – WALK-FORWARD IN IHREN "
                    "GETRENNTEN ROLLEN BESTÄTIGT.** "
                    "B bestätigt sich als 20–60D Direction-Challenger, "
                    "E als Risk-State-Challenger. Der nächste strengere "
                    "Schritt wäre ein echter zukünftiger Shadow-Holdout "
                    "mit unveränderten Konfigurationen."
                )

            elif (
                b_passed >= max(
                    1,
                    b_total - 1
                )
                and e_passed >= max(
                    1,
                    e_total - 1
                )
            ):
                st.warning(
                    f"🟡 **EUR/USD WALK-FORWARD GEMISCHT POSITIV.** "
                    f"B: {b_passed}/{b_total}, "
                    f"E: {e_passed}/{e_total}. "
                    "Mindestens ein festes Kriterium bleibt offen; "
                    "keine produktive Umstellung allein auf dieser Basis."
                )

            else:
                st.error(
                    f"🔴 **EUR/USD WALK-FORWARD NICHT AUSREICHEND "
                    f"BESTÄTIGT.** "
                    f"B: {b_passed}/{b_total}, "
                    f"E: {e_passed}/{e_total}. "
                    "Dann bleibt Current produktiv und wir untersuchen "
                    "separat die relative FX-Makrostruktur, statt "
                    "weitere Gewichte auf derselben Historie zu tunen."
                )

            st.caption(
                "Auch dieser Walk-Forward ist eine historische "
                "Robustheitsprüfung und kein echter zukünftiger unseen "
                "Holdout, da B/E bereits anhand derselben Gesamthistorie "
                "als Kandidaten identifiziert wurden."
            )


# ============================================================
# 20B. WTI WALK-FORWARD VALIDATION – FROZEN MODEL D
# ============================================================

walk_forward_year_df = pd.DataFrame()
walk_forward_pooled_df = pd.DataFrame()
walk_forward_gate_df = pd.DataFrame()

if selected_asset == "WTI Crude Oil":
    st.markdown("---")
    st.subheader(
        "🧭 WTI Walk-Forward Validation – Frozen Model D"
    )

    st.caption(
        "Model D ist vollständig eingefroren: Current-Säulengewichte + "
        "die bereits zuvor definierten Literature-Subgewichte. "
        "In diesem Abschnitt werden **keine Gewichte neu geschätzt oder "
        "angepasst**. Jedes Kalenderjahr wird chronologisch als separates "
        "Testfenster ausgewertet."
    )

    current_wf_frame = (
        model_frames[
            MODEL_CURRENT
        ]
    )

    model_d_wf_frame = (
        diagnostic_frames[
            "D · Lit Subweights only"
        ]
    )

    current_wf_score = (
        current_wf_frame[
            "Final_Regime_Score"
        ]
    )

    model_d_wf_score = (
        model_d_wf_frame[
            "Final_Regime_Score"
        ]
    )

    wf_common = (
        current_wf_score.notna()
        & model_d_wf_score.notna()
        & (
            current_wf_frame[
                "Model_Data_Coverage"
            ]
            >= float(
                min_coverage
            )
        )
        & (
            model_d_wf_frame[
                "Model_Data_Coverage"
            ]
            >= float(
                min_coverage
            )
        )
    )

    # Guarantee a substantial historical warm-up even if the user selects
    # a shorter data history. With the standard 15-year run this resolves
    # to 2017 (2011-2016 context before the first formal test year).
    first_data_year = int(
        raw_df.index.min().year
    )

    walk_forward_first_year = max(
        int(
            WTI_WF_BASE_FIRST_TEST_YEAR
        ),
        first_data_year + 6,
    )

    current_calendar_year = int(
        pd.Timestamp.now().year
    )

    last_full_test_year = (
        current_calendar_year - 1
    )

    st.info(
        f"**Festes Walk-Forward-Schema:** erster formaler Test "
        f"**{walk_forward_first_year}**, letztes vollständiges Testjahr "
        f"**{last_full_test_year}**. Das laufende Jahr "
        f"**{current_calendar_year}** wird nur informativ angezeigt und "
        "nicht für das Bestanden/Nicht-bestanden-Urteil verwendet."
    )

    st.markdown(
        """
**Methodik**

Für beispielsweise Testjahr 2020 stehen nur die bis dahin bereits
verfügbaren, point-in-time aufgebauten Scorewerte zur Verfügung. Die
vorangegangene Historie dient ausschließlich als Lookback-/Warm-up-Kontext
für Rolling-Z-Scores und andere historische Transformationen.

Es gibt hier **kein Training auf dem Testjahr, kein Re-Fitting und keine
Gewichtsoptimierung**. Deshalb ist dies präziser ein
*Frozen-Parameter Expanding-History Walk-Forward*.
"""
    )

    walk_forward_year_df = (
        walk_forward_year_metrics(
            current_wf_score,
            model_d_wf_score,
            current_wf_frame[
                "Model_Data_Coverage"
            ],
            model_d_wf_frame[
                "Model_Data_Coverage"
            ],
            targets,
            wf_common,
            walk_forward_first_year,
            current_calendar_year,
        )
    )

    if walk_forward_year_df.empty:
        st.error(
            "Für den Walk-Forward-Test stehen nicht genügend "
            "chronologische Testjahre zur Verfügung."
        )

    else:
        st.markdown(
            "### 1. Testjahre"
        )

        st.dataframe(
            walk_forward_year_df.style.format(
                {
                    "Ø Coverage Current": "{:.1f}%",
                    "Ø Coverage Model D": "{:.1f}%",
                    "Current IC 5D": "{:+.3f}",
                    "Model D IC 5D": "{:+.3f}",
                    "Current IC 20D": "{:+.3f}",
                    "Model D IC 20D": "{:+.3f}",
                    "Current IC 60D": "{:+.3f}",
                    "Model D IC 60D": "{:+.3f}",
                    "Δ IC20 D−Current": "{:+.3f}",
                    "Current Direction 20D": "{:.1%}",
                    "Model D Direction 20D": "{:.1%}",
                    "Current Stress AUC": "{:.3f}",
                    "Model D Stress AUC": "{:.3f}",
                },
                na_rep="n/a",
            ),
            hide_index=True,
            use_container_width=True,
        )

        completed_years = (
            walk_forward_year_df[
                walk_forward_year_df[
                    "Vollständiges Jahr"
                ]
            ]
            .copy()
        )

        if completed_years.empty:
            st.error(
                "Es gibt noch keine vollständigen Walk-Forward-Testjahre."
            )

        else:
            completed_year_values = set(
                completed_years[
                    "Testjahr"
                ].astype(
                    int
                )
            )

            pooled_complete_mask = (
                wf_common
                & pd.Series(
                    [
                        int(year)
                        in completed_year_values
                        for year
                        in targets.index.year
                    ],
                    index=targets.index,
                    dtype=bool,
                )
            )

            current_pooled = (
                pooled_walk_forward_metrics(
                    current_wf_score,
                    targets,
                    pooled_complete_mask,
                )
            )

            d_pooled = (
                pooled_walk_forward_metrics(
                    model_d_wf_score,
                    targets,
                    pooled_complete_mask,
                )
            )

            pooled_rows = []

            for label, values in [
                (
                    "A · Current",
                    current_pooled,
                ),
                (
                    "D · Lit Subweights only",
                    d_pooled,
                ),
            ]:
                pooled_rows.append(
                    {
                        "Modell": label,
                        **values,
                    }
                )

            walk_forward_pooled_df = (
                pd.DataFrame(
                    pooled_rows
                )
            )

            st.markdown(
                "### 2. Gepoolte vollständige Testjahre"
            )

            st.dataframe(
                walk_forward_pooled_df.style.format(
                    {
                        "IC 5D": "{:+.3f}",
                        "IC 20D": "{:+.3f}",
                        "IC 60D": "{:+.3f}",
                        "Direction 5D": "{:.1%}",
                        "Direction 20D": "{:.1%}",
                        "Direction 60D": "{:.1%}",
                        "Stress AUC 20D": "{:.3f}",
                        "Score vs niedrigere FwdVol": "{:+.3f}",
                        "Score vs bessere FwdMAE": "{:+.3f}",
                    },
                    na_rep="n/a",
                ),
                hide_index=True,
                use_container_width=True,
            )

            # ------------------------------------------------
            # Crisis-concentration test: remove 2020 entirely.
            # ------------------------------------------------

            pooled_ex_2020_mask = (
                pooled_complete_mask
                & (
                    targets.index.year
                    != 2020
                )
            )

            d_ic20_ex_2020 = safe_spearman(
                model_d_wf_score.where(
                    pooled_ex_2020_mask
                ),
                targets[
                    "Fwd_Return_20D"
                ],
            )

            # ------------------------------------------------
            # Walk-forward bootstrap on completed test years.
            # ------------------------------------------------

            wf_bootstrap_rows = []

            for horizon in FORWARD_HORIZONS:
                wf_boot = (
                    block_bootstrap_ic_difference(
                        current_wf_score.where(
                            pooled_complete_mask
                        ),
                        model_d_wf_score.where(
                            pooled_complete_mask
                        ),
                        targets[
                            f"Fwd_Return_{horizon}D"
                        ],
                        block_length=int(
                            bootstrap_block
                        ),
                        n_boot=int(
                            bootstrap_runs
                        ),
                        seed=800 + int(
                            horizon
                        ),
                    )
                )

                wf_bootstrap_rows.append(
                    {
                        "Horizont": f"{horizon}D",
                        "Δ IC D−Current": (
                            d_pooled[
                                f"IC {horizon}D"
                            ]
                            -
                            current_pooled[
                                f"IC {horizon}D"
                            ]
                        ),
                        "Bootstrap P(Δ>0)": wf_boot[
                            "prob_positive"
                        ],
                        "CI Low": wf_boot[
                            "lower"
                        ],
                        "CI High": wf_boot[
                            "upper"
                        ],
                        "Bootstrap N": wf_boot[
                            "n"
                        ],
                    }
                )

            wf_bootstrap_df = (
                pd.DataFrame(
                    wf_bootstrap_rows
                )
            )

            st.markdown(
                "### 3. Bootstrap nur über die vollständigen Walk-Forward-Testjahre"
            )

            st.dataframe(
                wf_bootstrap_df.style.format(
                    {
                        "Δ IC D−Current": "{:+.3f}",
                        "Bootstrap P(Δ>0)": "{:.1%}",
                        "CI Low": "{:+.3f}",
                        "CI High": "{:+.3f}",
                        "Bootstrap N": "{:.0f}",
                    },
                    na_rep="n/a",
                ),
                hide_index=True,
                use_container_width=True,
            )

            # ------------------------------------------------
            # Pre-registered final gate.
            # ------------------------------------------------

            finite_year_comparisons = (
                completed_years[
                    "D besser IC20"
                ]
                .notna()
            )

            n_test_years = int(
                finite_year_comparisons.sum()
            )

            n_d_wins = int(
                completed_years.loc[
                    finite_year_comparisons,
                    "D besser IC20"
                ]
                .astype(
                    bool
                )
                .sum()
            )

            d_win_rate = (
                n_d_wins
                / n_test_years
                if n_test_years > 0
                else np.nan
            )

            pooled_current_ic20 = float(
                current_pooled[
                    "IC 20D"
                ]
            )

            pooled_d_ic20 = float(
                d_pooled[
                    "IC 20D"
                ]
            )

            pooled_d_direction20 = float(
                d_pooled[
                    "Direction 20D"
                ]
            )

            pooled_current_auc = float(
                current_pooled[
                    "Stress AUC 20D"
                ]
            )

            pooled_d_auc = float(
                d_pooled[
                    "Stress AUC 20D"
                ]
            )

            gate_rows = [
                {
                    "Kriterium": (
                        "Model D schlägt Current beim IC20 "
                        "in mindestens 60% der vollständigen Testjahre"
                    ),
                    "Erfüllt": (
                        bool(
                            np.isfinite(
                                d_win_rate
                            )
                            and d_win_rate
                            >= WTI_WF_MIN_IC20_WIN_RATE
                        )
                    ),
                    "Messwert": (
                        f"{n_d_wins}/{n_test_years} "
                        f"({d_win_rate:.1%})"
                        if np.isfinite(
                            d_win_rate
                        )
                        else "n/a"
                    ),
                },
                {
                    "Kriterium": (
                        "Gepoolter Model-D IC20 > 0"
                    ),
                    "Erfüllt": (
                        bool(
                            np.isfinite(
                                pooled_d_ic20
                            )
                            and pooled_d_ic20
                            > WTI_WF_MIN_POOLED_IC20
                        )
                    ),
                    "Messwert": (
                        f"{pooled_d_ic20:+.3f}"
                        if np.isfinite(
                            pooled_d_ic20
                        )
                        else "n/a"
                    ),
                },
                {
                    "Kriterium": (
                        "Gepoolte Direction 20D ≥ 50%"
                    ),
                    "Erfüllt": (
                        bool(
                            np.isfinite(
                                pooled_d_direction20
                            )
                            and pooled_d_direction20
                            >= WTI_WF_MIN_DIRECTION_20D
                        )
                    ),
                    "Messwert": (
                        f"{pooled_d_direction20:.1%}"
                        if np.isfinite(
                            pooled_d_direction20
                        )
                        else "n/a"
                    ),
                },
                {
                    "Kriterium": (
                        "Gepoolte Stress-AUC ≥ 0.55"
                    ),
                    "Erfüllt": (
                        bool(
                            np.isfinite(
                                pooled_d_auc
                            )
                            and pooled_d_auc
                            >= WTI_WF_MIN_STRESS_AUC
                        )
                    ),
                    "Messwert": (
                        f"{pooled_d_auc:.3f}"
                        if np.isfinite(
                            pooled_d_auc
                        )
                        else "n/a"
                    ),
                },
                {
                    "Kriterium": (
                        "Stress-AUC von Model D nicht schlechter als Current"
                    ),
                    "Erfüllt": (
                        bool(
                            np.isfinite(
                                pooled_d_auc
                            )
                            and np.isfinite(
                                pooled_current_auc
                            )
                            and pooled_d_auc
                            >= pooled_current_auc
                        )
                    ),
                    "Messwert": (
                        f"D {pooled_d_auc:.3f} vs. "
                        f"Current {pooled_current_auc:.3f}"
                        if (
                            np.isfinite(
                                pooled_d_auc
                            )
                            and np.isfinite(
                                pooled_current_auc
                            )
                        )
                        else "n/a"
                    ),
                },
                {
                    "Kriterium": (
                        "Model-D IC20 bleibt auch ohne Krisenjahr 2020 positiv"
                    ),
                    "Erfüllt": (
                        bool(
                            np.isfinite(
                                d_ic20_ex_2020
                            )
                            and d_ic20_ex_2020 > 0
                        )
                    ),
                    "Messwert": (
                        f"{d_ic20_ex_2020:+.3f}"
                        if np.isfinite(
                            d_ic20_ex_2020
                        )
                        else "n/a"
                    ),
                },
            ]

            walk_forward_gate_df = (
                pd.DataFrame(
                    gate_rows
                )
            )

            walk_forward_gate_df[
                "Status"
            ] = walk_forward_gate_df[
                "Erfüllt"
            ].map(
                {
                    True: "✅",
                    False: "❌",
                }
            )

            st.markdown(
                "### 4. Vorab festgelegter Walk-Forward-Gate"
            )

            st.dataframe(
                walk_forward_gate_df[
                    [
                        "Kriterium",
                        "Status",
                        "Messwert",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )

            wf_passed = int(
                walk_forward_gate_df[
                    "Erfüllt"
                ]
                .sum()
            )

            wf_total = int(
                len(
                    walk_forward_gate_df
                )
            )

            if wf_passed == wf_total:
                st.success(
                    "🟢 **WTI MODEL D – HISTORISCHER WALK-FORWARD BESTANDEN.** "
                    "Alle vorab festgelegten Kriterien wurden in den "
                    "chronologischen, vollständigen Testjahren erfüllt. "
                    "Das ist eine starke Bestätigung, aber noch kein echter "
                    "zukünftiger Unseen-Holdout, weil Model D anhand der "
                    "bereits bekannten Gesamthistorie als Kandidat "
                    "identifiziert wurde."
                )

            elif wf_passed >= 4:
                st.warning(
                    f"🟡 **WTI MODEL D – WALK-FORWARD GEMISCHT POSITIV "
                    f"({wf_passed}/{wf_total}).** "
                    "Die Evidenz ist interessant, erfüllt aber nicht alle "
                    "vorab festgelegten Kriterien. Keine produktive "
                    "Gewichtsänderung allein auf dieser Basis."
                )

            else:
                st.error(
                    f"🔴 **WTI MODEL D – WALK-FORWARD NICHT BESTÄTIGT "
                    f"({wf_passed}/{wf_total}).** "
                    "Die historische Robustheit reicht für eine "
                    "Produktionsänderung nicht aus."
                )

            st.caption(
                "Der historische Walk-Forward-Test ist eine "
                "Robustheitsprüfung. Der strengste unabhängige Test ist "
                "anschließend ein echter zukünftiger Shadow-/Holdout-Test "
                "mit unveränderten Gewichten."
            )


# ============================================================
# 20C. WTI EVENT / CRISIS ROBUSTNESS
# ============================================================

event_classification = pd.DataFrame(
    index=targets.index
)
event_robustness_df = pd.DataFrame()
event_leave_one_out_df = pd.DataFrame()
event_gate_df = pd.DataFrame()

if selected_asset == "WTI Crude Oil":
    st.markdown("---")
    st.subheader(
        "🌍 WTI Event-/Krisen-Robustheit"
    )

    st.warning(
        "Dieser Test wurde **nach** der beobachteten 2020-Sensitivität "
        "hinzugefügt. Er ist deshalb eine gezielte Robustheitsdiagnose "
        "und kein vollständig unabhängiger vorab registrierter "
        "Model-Selection-Test. Die Model-D-Gewichte bleiben trotzdem "
        "vollständig eingefroren."
    )

    with st.expander(
        "📅 Fest definierte externe Ereignisfenster",
        expanded=True,
    ):
        event_rows = []

        for event in WTI_EVENT_WINDOWS:
            event_rows.append(
                {
                    "Ereignis": event[
                        "name"
                    ],
                    "Kategorie": event[
                        "category"
                    ],
                    "Start": event[
                        "start"
                    ],
                    "Ende": event[
                        "end"
                    ],
                    "Im formalen WF": (
                        "Ja"
                        if event.get(
                            "formal_walk_forward",
                            False
                        )
                        else "Nein"
                    ),
                    "Begründung": event[
                        "anchor"
                    ],
                }
            )

        st.dataframe(
            pd.DataFrame(
                event_rows
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown(
            """
**Externe Ankerquellen**

- [OPEC – 27.11.2014: Produktionsniveau 30,0 mb/d beibehalten](https://www.opec.org/pr-detail/84-27-nov-2014.html)
- [WHO – 11.03.2020: COVID-19 als Pandemie charakterisiert](https://www.who.int/news-room/speeches/item/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020)
- [UN – 24.02.2022: großangelegte russische Militäroperationen in der Ukraine](https://press.un.org/en/2022/sgsm21158.doc.htm)
- [UN – 07.10.2023: Angriff und Beginn der Eskalation Israel/Gaza](https://press.un.org/en/2023/sgsm21981.doc.htm)

Die **Enddaten** der Fenster sind bewusst einfache Research-Cut-offs und
keine Behauptung, dass das jeweilige Ereignis an diesem Tag ökonomisch
„beendet“ war.
"""
        )

    event_classification = (
        build_event_classification(
            targets.index,
            WTI_EVENT_WINDOWS,
        )
    )

    current_event_frame = (
        model_frames[
            MODEL_CURRENT
        ]
    )

    model_d_event_frame = (
        diagnostic_frames[
            "D · Lit Subweights only"
        ]
    )

    current_event_score = (
        current_event_frame[
            "Final_Regime_Score"
        ]
    )

    d_event_score = (
        model_d_event_frame[
            "Final_Regime_Score"
        ]
    )

    last_full_year = (
        int(
            pd.Timestamp.now().year
        )
        - 1
    )

    event_base_mask = (
        current_event_score.notna()
        & d_event_score.notna()
        & (
            current_event_frame[
                "Model_Data_Coverage"
            ]
            >= float(
                min_coverage
            )
        )
        & (
            model_d_event_frame[
                "Model_Data_Coverage"
            ]
            >= float(
                min_coverage
            )
        )
        & (
            targets.index.year
            >= int(
                WTI_WF_BASE_FIRST_TEST_YEAR
            )
        )
        & (
            targets.index.year
            <= last_full_year
        )
    )

    formal_window_names = [
        event[
            "name"
        ]
        for event
        in WTI_EVENT_WINDOWS
        if event.get(
            "formal_walk_forward",
            False
        )
    ]

    formal_event_mask = pd.Series(
        False,
        index=targets.index,
        dtype=bool,
    )

    for event_name in formal_window_names:
        formal_event_mask |= (
            event_classification[
                "Event_Label"
            ]
            .astype(
                str
            )
            .str.contains(
                event_name,
                regex=False,
            )
        )

    formal_event_mask &= (
        event_base_mask
    )

    normal_mask = (
        event_base_mask
        & ~formal_event_mask
    )

    event_rows = [
        event_subset_metrics(
            "Normal – alle markierten Events entfernt",
            normal_mask,
            current_event_score,
            d_event_score,
            current_event_frame[
                "Model_Data_Coverage"
            ],
            model_d_event_frame[
                "Model_Data_Coverage"
            ],
            targets,
        ),
        event_subset_metrics(
            "Alle formalen Eventfenster zusammen",
            formal_event_mask,
            current_event_score,
            d_event_score,
            current_event_frame[
                "Model_Data_Coverage"
            ],
            model_d_event_frame[
                "Model_Data_Coverage"
            ],
            targets,
        ),
    ]

    for event in WTI_EVENT_WINDOWS:
        if not event.get(
            "formal_walk_forward",
            False
        ):
            continue

        event_mask = (
            event_base_mask
            & event_classification[
                "Event_Label"
            ]
            .astype(
                str
            )
            .str.contains(
                event[
                    "name"
                ],
                regex=False,
            )
        )

        event_rows.append(
            event_subset_metrics(
                event[
                    "name"
                ],
                event_mask,
                current_event_score,
                d_event_score,
                current_event_frame[
                    "Model_Data_Coverage"
                ],
                model_d_event_frame[
                    "Model_Data_Coverage"
                ],
                targets,
            )
        )

    event_robustness_df = (
        pd.DataFrame(
            event_rows
        )
    )

    st.markdown(
        "### 1. Normalregime gegen Eventregime"
    )

    st.dataframe(
        event_robustness_df.style.format(
            {
                "Current IC20": "{:+.3f}",
                "Model D IC20": "{:+.3f}",
                "Δ IC20 D−Current": "{:+.3f}",
                "Current Direction 20D": "{:.1%}",
                "Model D Direction 20D": "{:.1%}",
                "Current Stress AUC": "{:.3f}",
                "Model D Stress AUC": "{:.3f}",
                "Ø Fwd Return 20D": "{:+.2%}",
                "Ø Fwd MAE 20D": "{:+.2%}",
                "Ø Fwd Vol 20D": "{:.1%}",
                "Ø Coverage Current": "{:.1f}%",
                "Ø Coverage Model D": "{:.1f}%",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Leave-one-event-out
    # --------------------------------------------------------

    event_leave_one_out_df = (
        leave_one_event_out_metrics(
            event_base_mask,
            event_classification,
            WTI_EVENT_WINDOWS,
            current_event_score,
            d_event_score,
            targets,
        )
    )

    st.markdown(
        "### 2. Leave-one-event-out Sensitivität"
    )

    st.dataframe(
        event_leave_one_out_df.style.format(
            {
                "Current IC20": "{:+.3f}",
                "Model D IC20": "{:+.3f}",
                "Δ IC20 D−Current": "{:+.3f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Dieser Block zeigt unmittelbar, ob ein einzelnes Ereignis – "
        "insbesondere COVID-2020 – das Ergebnis dominiert. "
        "Die Zeile „Ohne alle markierten Eventfenster“ ist die strengste "
        "Normalregime-Sensitivität."
    )

    # --------------------------------------------------------
    # Interpretation gates: normal-market Direction vs Risk
    # --------------------------------------------------------

    normal_row = (
        event_robustness_df[
            event_robustness_df[
                "Teilmenge"
            ]
            ==
            "Normal – alle markierten Events entfernt"
        ]
        .iloc[
            0
        ]
    )

    normal_d_ic20 = float(
        normal_row[
            "Model D IC20"
        ]
    )

    normal_current_ic20 = float(
        normal_row[
            "Current IC20"
        ]
    )

    normal_d_direction = float(
        normal_row[
            "Model D Direction 20D"
        ]
    )

    normal_current_auc = float(
        normal_row[
            "Current Stress AUC"
        ]
    )

    normal_d_auc = float(
        normal_row[
            "Model D Stress AUC"
        ]
    )

    normal_direction_gate = (
        np.isfinite(
            normal_d_ic20
        )
        and normal_d_ic20 > 0
        and np.isfinite(
            normal_d_direction
        )
        and normal_d_direction >= 0.50
    )

    normal_risk_gate = (
        np.isfinite(
            normal_d_auc
        )
        and normal_d_auc >= 0.55
        and np.isfinite(
            normal_current_auc
        )
        and normal_d_auc >= normal_current_auc
    )

    normal_relative_gate = (
        np.isfinite(
            normal_d_ic20
        )
        and np.isfinite(
            normal_current_ic20
        )
        and normal_d_ic20 > normal_current_ic20
    )

    event_gate_rows = [
        {
            "Prüfung": (
                "Normalregime: Model-D IC20 > Current"
            ),
            "Erfüllt": normal_relative_gate,
            "Messwert": (
                f"D {normal_d_ic20:+.3f} vs. "
                f"Current {normal_current_ic20:+.3f}"
                if (
                    np.isfinite(
                        normal_d_ic20
                    )
                    and np.isfinite(
                        normal_current_ic20
                    )
                )
                else "n/a"
            ),
        },
        {
            "Prüfung": (
                "Normalregime Direction-Gate: IC20 >0 UND Direction ≥50%"
            ),
            "Erfüllt": normal_direction_gate,
            "Messwert": (
                f"IC20 {normal_d_ic20:+.3f} · "
                f"Direction {normal_d_direction:.1%}"
                if (
                    np.isfinite(
                        normal_d_ic20
                    )
                    and np.isfinite(
                        normal_d_direction
                    )
                )
                else "n/a"
            ),
        },
        {
            "Prüfung": (
                "Normalregime Risk-Gate: AUC ≥0.55 und ≥ Current"
            ),
            "Erfüllt": normal_risk_gate,
            "Messwert": (
                f"D {normal_d_auc:.3f} vs. "
                f"Current {normal_current_auc:.3f}"
                if (
                    np.isfinite(
                        normal_d_auc
                    )
                    and np.isfinite(
                        normal_current_auc
                    )
                )
                else "n/a"
            ),
        },
    ]

    event_gate_df = pd.DataFrame(
        event_gate_rows
    )

    event_gate_df[
        "Status"
    ] = event_gate_df[
        "Erfüllt"
    ].map(
        {
            True: "✅",
            False: "❌",
        }
    )

    st.markdown(
        "### 3. Interpretation außerhalb markierter Krisen"
    )

    st.dataframe(
        event_gate_df[
            [
                "Prüfung",
                "Status",
                "Messwert",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    if (
        normal_direction_gate
        and normal_risk_gate
        and normal_relative_gate
    ):
        st.success(
            "🟢 **MODEL D IST AUCH AUSSERHALB DER MARKIERTEN EVENTS "
            "RICHTUNGS- UND RISIKOSEITIG ROBUST.** "
            "Die Krisenfenster erklären den historischen Vorteil nicht allein."
        )

    elif (
        normal_risk_gate
        and normal_relative_gate
        and not normal_direction_gate
    ):
        st.warning(
            "🟡 **MODEL D BLEIBT AUSSERHALB DER EVENTS RELATIV BESSER "
            "UND ALS RISK-STATE-MODELL BRAUCHBAR, ABER NICHT ALS "
            "BELASTBARER DIRECTION-PREDICTOR.** "
            "Das würde die bisherige Interpretation weiter stützen."
        )

    elif (
        normal_relative_gate
        and not normal_risk_gate
    ):
        st.warning(
            "🟠 **RELATIVER SCORE-VORTEIL AUSSERHALB DER EVENTS, "
            "ABER KEIN ROBUSTER NORMALREGIME-RISK-GATE.**"
        )

    else:
        st.error(
            "🔴 **STARKE EVENT-/KRISENABHÄNGIGKEIT.** "
            "Außerhalb der markierten externen Ereignisfenster lässt sich "
            "der Model-D-Vorteil nicht ausreichend bestätigen."
        )

    st.info(
        "Wichtig: Auch ein gutes Ergebnis hier macht aus Geopolitik oder "
        "Pandemien **keinen neuen quantitativen Scorefaktor**. Der Test "
        "entscheidet nur, ob wir künftig zusätzlich eine separate "
        "Event-Risk-Warnschicht im TradePilot/Regime-Dashboard brauchen."
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

# Diagnostic models are essential for offline validation.
# v1.0.9 displayed them in Streamlit but accidentally omitted them
# from the downloaded research CSV.
for diagnostic_name in [
    "D · Lit Subweights only",
    "E · Lit Pillars only",
]:
    if diagnostic_name not in diagnostic_frames:
        continue

    clean_name = (
        diagnostic_name
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
        diagnostic_frames[
            diagnostic_name
        ][
            "Final_Regime_Score"
        ]
    )

    export[
        f"Coverage_{clean_name}"
    ] = (
        diagnostic_frames[
            diagnostic_name
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

if (
    selected_asset == "WTI Crude Oil"
    and not event_classification.empty
):
    export[
        "Event_Any"
    ] = event_classification[
        "Event_Any"
    ]

    export[
        "Event_Label"
    ] = event_classification[
        "Event_Label"
    ]

    export[
        "Event_Category"
    ] = event_classification[
        "Event_Category"
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
        (
            (
                (
                (
                "regime_backtest_Gold_volatility_state_robustness_audit.csv"
                if selected_asset == "Gold (XAU/USD)"
                else "regime_backtest_Nasdaq100_role_factor_diagnosis.csv"
            )
                if selected_asset == "Nasdaq 100"
                else "regime_backtest_SP500_frozen_A_vs_E_risk_state_walk_forward.csv"
            )
                if selected_asset == "S&P 500"
                else "regime_backtest_EUR-USD_frozen_be_walk_forward.csv"
            )
            if selected_asset == "EUR/USD"
            else
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
            + "_model_d_validation.csv"
        )
    ),
    mime="text/csv",
)

st.caption(
    "Der Export enthält A/Current, B/Literature, C/Equal Weight sowie "
    "D/Lit-Subweights-only und E/Lit-Pillars-only inklusive Coverage "
    "und Forward-Targets. Für EUR/USD wird zusätzlich ein eigenes "
    "B-vs-E-Bestätigungs-ZIP erzeugt. Research-Datenmaterial – keine "
    "Live-Trading-Freigabe."
)












if (
    selected_asset == "Gold (XAU/USD)"
    and not gold_vol_gate_table.empty
):
    gold_vol_zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        gold_vol_zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        zf.writestr(
            "gold_volatility_state_research_timeseries.csv",
            gold_vol_research_export.to_csv(),
        )
        zf.writestr(
            "gold_volatility_target_pit.csv",
            gold_vol_target_table.to_csv(
                index=False
            ),
        )
        zf.writestr(
            "gold_volatility_models.csv",
            gold_vol_model_table.to_csv(
                index=False
            ),
        )
        zf.writestr(
            "gold_volatility_fixed_periods.csv",
            gold_vol_period_table.to_csv(
                index=False
            ),
        )
        zf.writestr(
            "gold_volatility_2017_2020.csv",
            gold_vol_2017_2020_table.to_csv(
                index=False
            ),
        )
        zf.writestr(
            "gold_volatility_move_audit.csv",
            gold_vol_move_table.to_csv(
                index=False
            ),
        )
        zf.writestr(
            "gold_volatility_quintiles.csv",
            gold_vol_quintile_table.to_csv(
                index=False
            ),
        )
        zf.writestr(
            "gold_volatility_gate.csv",
            gold_vol_gate_table.to_csv(
                index=False
            ),
        )
        zf.writestr(
            "gold_d1_frozen_reference.csv",
            gold_vol_d1_frozen_table.to_csv(
                index=False
            ),
        )

        # Traceability to v1.0.23.
        zf.writestr(
            "gold_v1023_r1_component_orientation.csv",
            gold_r1_component_orientation_table.to_csv(
                index=False
            ),
        )
        zf.writestr(
            "gold_v1023_r1_volatility_target_audit.csv",
            gold_r1_volatility_target_table.to_csv(
                index=False
            ),
        )

    gold_vol_zip_buffer.seek(
        0
    )

    st.download_button(
        "⬇️ Gold Volatility-State Robustness Audit v1.0.24 als ZIP",
        data=gold_vol_zip_buffer.getvalue(),
        file_name=(
            "Gold_Volatility_State_Robustness_Audit_v1_0_24.zip"
        ),
        mime="application/zip",
    )

    st.caption(
        "Für die nächste Entscheidung genügt dieses ZIP. Enthalten sind "
        "PIT-High-Vol-Ziel, Modellvergleich, feste Teilperioden, separater "
        "2017–2020-Block, MOVE Current/invertiert/entfernt, Quintile, "
        "Gate-Auswertung und der eingefrorene D1-Referenzstand."
    )


if (
    selected_asset == "Nasdaq 100"
    and not nasdaq_role_table.empty
):
    nasdaq_zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        nasdaq_zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        zf.writestr(
            "nasdaq100_role_diagnosis_research_timeseries.csv",
            export.to_csv(),
        )
        zf.writestr(
            "nasdaq100_model_role_matrix.csv",
            nasdaq_role_table.to_csv(index=False),
        )
        zf.writestr(
            "nasdaq100_3a_weight_decomposition.csv",
            diag_table.to_csv(index=False),
        )

        if (
            "nasdaq_factor_table" in globals()
            and not nasdaq_factor_table.empty
        ):
            zf.writestr(
                "nasdaq100_7a_factor_diagnostics.csv",
                nasdaq_factor_table.to_csv(index=False),
            )

        zf.writestr(
            "nasdaq100_period_robustness.csv",
            nasdaq_period_table.to_csv(index=False),
        )
        zf.writestr(
            "nasdaq100_nonoverlap.csv",
            nasdaq_nonoverlap_table.to_csv(index=False),
        )
        zf.writestr(
            "nasdaq100_crisis_windows.csv",
            nasdaq_crisis_table.to_csv(index=False),
        )
        zf.writestr(
            "nasdaq100_role_gate.csv",
            nasdaq_gate_table.to_csv(index=False),
        )

    nasdaq_zip_buffer.seek(0)

    st.download_button(
        "⬇️ Nasdaq 100 Role & Factor Diagnosis als ZIP",
        data=nasdaq_zip_buffer.getvalue(),
        file_name="Nasdaq100_Role_Factor_Diagnosis_v1_0_18.zip",
        mime="application/zip",
    )

    st.caption(
        "Für die nächste Auswertung genügt dieses ZIP."
    )


if (
    selected_asset == "S&P 500"
    and not sp500_wf_yearly_df.empty
):
    sp_wf_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(
        sp_wf_zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        zf.writestr(
            "sp500_frozen_ae_research_timeseries.csv",
            export.to_csv(),
        )
        zf.writestr(
            "sp500_frozen_ae_yearly.csv",
            sp500_wf_yearly_df.to_csv(index=False),
        )
        if not sp500_wf_pooled_df.empty:
            zf.writestr(
                "sp500_frozen_ae_pooled.csv",
                sp500_wf_pooled_df.to_csv(index=False),
            )
        if not sp500_wf_bootstrap_df.empty:
            zf.writestr(
                "sp500_frozen_ae_bootstrap.csv",
                sp500_wf_bootstrap_df.to_csv(index=False),
            )
        if not sp500_wf_gate_df.empty:
            zf.writestr(
                "sp500_frozen_ae_gate.csv",
                sp500_wf_gate_df.to_csv(index=False),
            )
    sp_wf_zip_buffer.seek(0)
    st.download_button(
        "⬇️ S&P 500 Frozen A-vs-E Risk-State Walk-Forward als ZIP",
        data=sp_wf_zip_buffer.getvalue(),
        file_name="SP500_Frozen_A_vs_E_Risk_State_Walk_Forward_v1_0_17.zip",
        mime="application/zip",
    )
    st.caption(
        "Für die Auswertung genügt dieses ZIP: Zeitreihe, Jahresfenster, "
        "gepoolte Risk-State-Metriken, Bootstrap und feste Gates."
    )


if (
    selected_asset == "S&P 500"
    and not sp500_role_table.empty
):
    sp_zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        sp_zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        zf.writestr(
            "sp500_role_diagnosis_research_timeseries.csv",
            export.to_csv(),
        )
        zf.writestr(
            "sp500_model_role_matrix.csv",
            sp500_role_table.to_csv(index=False),
        )
        zf.writestr(
            "sp500_3a_weight_decomposition.csv",
            diag_table.to_csv(index=False),
        )
        if (
            "sp500_factor_table" in globals()
            and not sp500_factor_table.empty
        ):
            zf.writestr(
                "sp500_7a_factor_diagnostics.csv",
                sp500_factor_table.to_csv(index=False),
            )
        zf.writestr(
            "sp500_period_robustness.csv",
            sp500_period_table.to_csv(index=False),
        )
        zf.writestr(
            "sp500_nonoverlap.csv",
            sp500_nonoverlap_table.to_csv(index=False),
        )
        zf.writestr(
            "sp500_crisis_windows.csv",
            sp500_crisis_table.to_csv(index=False),
        )
        zf.writestr(
            "sp500_role_gate.csv",
            sp500_gate_table.to_csv(index=False),
        )

    sp_zip_buffer.seek(0)

    st.download_button(
        "⬇️ S&P 500 Role & Factor Diagnosis als ZIP",
        data=sp_zip_buffer.getvalue(),
        file_name="SP500_Role_Factor_Diagnosis_v1_0_16.zip",
        mime="application/zip",
    )

    st.caption(
        "Für die nächste Auswertung genügt dieses ZIP: Zeitreihe, "
        "A/B/D/E-Rollenmatrix, 3A, 7A, Perioden-, Non-Overlap-, "
        "Stressfenster- und Rollen-Gate-Auswertung."
    )


if (
    selected_asset == "EUR/USD"
    and not eurusd_wf_yearly_df.empty
):
    eur_wf_zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        eur_wf_zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        zf.writestr(
            "eurusd_be_walk_forward_research_timeseries.csv",
            export.to_csv(),
        )

        zf.writestr(
            "eurusd_be_walk_forward_yearly.csv",
            eurusd_wf_yearly_df.to_csv(
                index=False
            ),
        )

        if not eurusd_wf_pooled_df.empty:
            zf.writestr(
                "eurusd_be_walk_forward_pooled.csv",
                eurusd_wf_pooled_df.to_csv(
                    index=False
                ),
            )

        if not eurusd_wf_bootstrap_df.empty:
            zf.writestr(
                "eurusd_be_walk_forward_bootstrap.csv",
                eurusd_wf_bootstrap_df.to_csv(
                    index=False
                ),
            )

        if not eurusd_wf_nonoverlap_df.empty:
            zf.writestr(
                "eurusd_be_walk_forward_nonoverlap.csv",
                eurusd_wf_nonoverlap_df.to_csv(
                    index=False
                ),
            )

        if not eurusd_wf_gate_df.empty:
            zf.writestr(
                "eurusd_be_walk_forward_gate.csv",
                eurusd_wf_gate_df.to_csv(
                    index=False
                ),
            )

    eur_wf_zip_buffer.seek(
        0
    )

    st.download_button(
        "⬇️ EUR/USD Frozen B/E Walk-Forward als ZIP",
        data=eur_wf_zip_buffer.getvalue(),
        file_name=(
            "EURUSD_Frozen_BE_Walk_Forward_v1_0_15.zip"
        ),
        mime="application/zip",
    )

    st.caption(
        "Für die Walk-Forward-Auswertung genügt dieses ZIP. "
        "Es enthält die Research-Zeitreihe, Jahresfenster, gepoolte "
        "Kennzahlen, Bootstrap, Non-Overlap-Diagnose und den festen Gate."
    )


if (
    selected_asset == "EUR/USD"
    and not eurusd_confirmation_horizons.empty
):
    eur_zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        eur_zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        zf.writestr(
            "eurusd_b_vs_e_research_timeseries.csv",
            export.to_csv(),
        )
        zf.writestr(
            "eurusd_b_vs_e_horizon_comparison.csv",
            eurusd_confirmation_horizons.to_csv(
                index=False
            ),
        )
        zf.writestr(
            "eurusd_b_vs_e_yearly_ic20.csv",
            eurusd_confirmation_yearly.to_csv(
                index=False
            ),
        )
        zf.writestr(
            "eurusd_b_vs_e_period_robustness.csv",
            eurusd_confirmation_periods.to_csv(
                index=False
            ),
        )
        zf.writestr(
            "eurusd_b_vs_e_nonoverlap.csv",
            eurusd_confirmation_nonoverlap.to_csv(
                index=False
            ),
        )
        zf.writestr(
            "eurusd_b_vs_e_confirmation_gate.csv",
            eurusd_confirmation_gate.to_csv(
                index=False
            ),
        )

    eur_zip_buffer.seek(0)

    st.download_button(
        "⬇️ EUR/USD B-vs-E Bestätigung als ZIP",
        data=eur_zip_buffer.getvalue(),
        file_name="EURUSD_B_vs_E_Confirmation_v1_0_14.zip",
        mime="application/zip",
    )

    st.caption(
        "Für die spätere Auswertung genügt dieses ZIP. Es enthält "
        "Zeitreihe, Horizon-/Bootstrap-Vergleich, Jahresstabilität, "
        "Perioden-Robustheit, Non-Overlap-Diagnose und den Bestätigungs-Gate."
    )


if (
    selected_asset == "WTI Crude Oil"
    and not walk_forward_year_df.empty
):
    wf_zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        wf_zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        zf.writestr(
            "wti_model_d_walk_forward_yearly.csv",
            walk_forward_year_df.to_csv(
                index=False
            ),
        )

        if not walk_forward_pooled_df.empty:
            zf.writestr(
                "wti_model_d_walk_forward_pooled.csv",
                walk_forward_pooled_df.to_csv(
                    index=False
                ),
            )

        if not walk_forward_gate_df.empty:
            zf.writestr(
                "wti_model_d_walk_forward_gate.csv",
                walk_forward_gate_df.to_csv(
                    index=False
                ),
            )

        if not event_robustness_df.empty:
            zf.writestr(
                "wti_model_d_event_robustness.csv",
                event_robustness_df.to_csv(
                    index=False
                ),
            )

        if not event_leave_one_out_df.empty:
            zf.writestr(
                "wti_model_d_event_leave_one_out.csv",
                event_leave_one_out_df.to_csv(
                    index=False
                ),
            )

        if not event_gate_df.empty:
            zf.writestr(
                "wti_model_d_event_gate.csv",
                event_gate_df.to_csv(
                    index=False
                ),
            )

        zf.writestr(
            "wti_model_d_research_timeseries.csv",
            export.to_csv(),
        )

    wf_zip_buffer.seek(
        0
    )

    st.download_button(
        "⬇️ WTI Walk-Forward + Event-Robustheit als ZIP",
        data=wf_zip_buffer.getvalue(),
        file_name=(
            "WTI_Model_D_Event_Robustness_v1_0_12.zip"
        ),
        mime="application/zip",
    )

    st.caption(
        "Für die spätere Auswertung genügt dieses ZIP: Es enthält "
        "die Tages-Zeitreihe, die Jahresergebnisse, die gepoolten "
        "Walk-Forward-Kennzahlen und den vorab festgelegten Gate."
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
