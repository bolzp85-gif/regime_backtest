"""Multi-Asset Seasonality Lab 1.7.0. Start: streamlit run FX_Seasonality_Lab.py
Dependencies: streamlit, pandas, numpy, pandas_market_calendars;
optional yfinance for online history.
Independent research dashboard. No engine/shadow imports or modifications.
"""

import io
import json
import zipfile
import hashlib
from functools import lru_cache
from datetime import date
import numpy as np
import pandas as pd

VERSION = "1.7.0"
ASSETS = {
    "S&P 500": ("^GSPC", "SP500", "Preisindex; keine Dividendenrendite."),
    "Nasdaq 100": ("^NDX", "NASDAQ100", "Preisindex; keine Dividendenrendite."),
    "Gold": (
        "GC=F",
        "GOLD",
        "Yahoo-Gold-Futures als Proxy, kein Spotgold. Kontraktwechsel können die Saisonalität beeinflussen.",
    ),
    "WTI": (
        "CL=F",
        "WTI",
        "Yahoo-WTI-Futures als Proxy. Kontraktwechsel und negative Preise sind besonders zu prüfen; keine Rollrendite eines handelbaren Portfolios.",
    ),
    "EUR/USD": (
        "EURUSD=X",
        "EURUSD",
        "USD je EUR; steigender Kurs bedeutet stärkeren Euro.",
    ),
    "USD/CHF": (
        "CHF=X",
        "USDCHF",
        "CHF je USD; steigender Kurs bedeutet stärkeren US-Dollar.",
    ),
    "USD/JPY": (
        "JPY=X",
        "USDJPY",
        "JPY je USD; steigender Kurs bedeutet stärkeren US-Dollar.",
    ),
}
SYMBOLS = {name: cfg[0] for name, cfg in ASSETS.items()}

CALENDAR_STRICT = "Strikt: alle Quellenlücken behalten"
CALENDAR_SESSIONS = "Handelssitzungen: Asset-Kalender"
CALENDAR_REFERENCE = "Referenztage: CME-Geschäftstage (Modell)"
ASSET_CALENDARS = {
    "S&P 500": "NYSE",
    "Nasdaq 100": "NASDAQ",
    "Gold": "CMEGlobex_Gold",
    "WTI": "CMEGlobex_CL",
    "EUR/USD": "FOREX",
    "USD/CHF": "FOREX",
    "USD/JPY": "FOREX",
}
WTI_REOPEN_SOURCE = "https://www.cmegroup.com/media-room/press-releases/2005/9/02/_ny_commodity_exchangesannouncewtccommemorationplans.html"


@lru_cache(maxsize=128)
def calendar_sessions(calendar_name, start, end):
    """Provider date labels, not intraday UTC timestamps or session closes."""
    import pandas_market_calendars as mcal

    days = mcal.get_calendar(calendar_name).valid_days(start, end)
    return days.tz_localize(None).normalize()


def calendar_overrides(frame, asset):
    """Explicit source-backed exceptions; never learn closures from NaNs."""
    if frame is None:
        return {}
    if not {"Date", "Asset", "Status", "Source"}.issubset(frame.columns):
        raise ValueError("Kalender-CSV benötigt Date, Asset, Status, Source.")
    result = {}
    for row in frame.itertuples(index=False):
        if row.Asset != asset:
            raise ValueError("Kalender-CSV enthält ein anderes Asset.")
        if row.Status not in ["closed", "no_reference"]:
            raise ValueError("Kalenderstatus muss closed oder no_reference sein.")
        if not isinstance(row.Source, str) or not row.Source.strip():
            raise ValueError("Jede Kalenderausnahme benötigt eine Quellenangabe.")
        if not isinstance(row.Date, str) or len(row.Date) != 10:
            raise ValueError("Kalenderdatum benötigt YYYY-MM-DD.")
        try:
            stamp = pd.Timestamp(date.fromisoformat(row.Date))
        except ValueError as exc:
            raise ValueError("Ungültiges Kalenderdatum.") from exc
        if stamp in result:
            raise ValueError("Doppelte Kalenderausnahme.")
        result[stamp] = (row.Status, row.Source.strip())
    return result


def apply_calendar(
    p, asset, mode, overrides=None, require_all_sessions=False, invalid_dates=()
):
    if mode not in [CALENDAR_STRICT, CALENDAR_SESSIONS, CALENDAR_REFERENCE]:
        raise ValueError("Unbekannter Kalendermodus.")
    exceptions = calendar_overrides(overrides, asset)
    audit = []
    if mode == CALENDAR_STRICT:
        return (
            p,
            {
                "mode": mode,
                "calendar": None,
                "model_verified": False,
                "removed_closed_missing_rows": 0,
                "inserted_missing_sessions": 0,
                "absent_calendar_sessions": 0,
                "require_all_sessions": False,
                "finite_quotes_on_non_session": 0,
                "overrides_applied": 0,
            },
            pd.DataFrame(columns=["Date", "Status", "Source"]),
        )
    if asset not in ASSET_CALENDARS:
        raise ValueError("Asset wird für die Kalenderprüfung benötigt.")
    if mode == CALENDAR_REFERENCE and asset not in ["WTI", "Gold"]:
        raise ValueError("CME-Referenzmodell ist nur für WTI und Gold verfügbar.")
    import pandas_market_calendars as mcal

    calendar_name = (
        "CME_TradeDate" if mode == CALENDAR_REFERENCE else ASSET_CALENDARS[asset]
    )
    expected = calendar_sessions(
        calendar_name, str(p.index.min().date()), str(p.index.max().date())
    )
    model_source = "pandas_market_calendars " + mcal.__version__ + ": " + calendar_name
    if asset == "WTI":
        for stamp in pd.date_range("2001-09-11", "2001-09-13"):
            exceptions.setdefault(stamp, ("closed", WTI_REOPEN_SOURCE))
    applicable = {
        stamp: value
        for stamp, value in exceptions.items()
        if value[0] == "closed" or mode == CALENDAR_REFERENCE
    }
    expected = expected.difference(pd.DatetimeIndex(list(applicable)))
    removed_dates = p.index[
        p.isna() & ~p.index.isin(expected) & ~p.index.isin(invalid_dates)
    ]
    finite_closed = p.index[np.isfinite(p) & ~p.index.isin(expected)]
    inserted_dates = expected.difference(p.index)
    for status, dates in [
        (
            (
                "missing_non_reference_day_removed"
                if mode == CALENDAR_REFERENCE
                else "missing_non_session_removed"
            ),
            removed_dates,
        ),
        (
            (
                "finite_non_reference_day_retained"
                if mode == CALENDAR_REFERENCE
                else "finite_non_session_retained"
            ),
            finite_closed,
        ),
        (
            "invalid_non_session_retained",
            p.index[p.index.isin(invalid_dates) & ~p.index.isin(expected)],
        ),
        (
            (
                "expected_session_absent_inserted_nan"
                if require_all_sessions
                else "calendar_session_absent_unverified_quote_day"
            ),
            inserted_dates,
        ),
        (
            "expected_session_missing_retained",
            p.index[p.isna() & p.index.isin(expected)],
        ),
    ]:
        for stamp in dates:
            audit.append(
                {
                    "Date": str(stamp.date()),
                    "Status": status,
                    "Source": applicable.get(stamp, (None, model_source))[1],
                }
            )
    q = p.drop(removed_dates)
    if require_all_sessions:
        q = q.reindex(q.index.union(expected)).sort_index()
    # Calendar-backed coverage extends only through actual supplied date labels.
    q.attrs["coverage_through"] = str(p.index.max().date())
    return (
        q,
        {
            "mode": mode,
            "calendar": calendar_name,
            "calendar_package_version": mcal.__version__,
            "classification": (
                "business_reference_proxy"
                if mode == CALENDAR_REFERENCE
                else "trading_session"
            ),
            "model_verified": False,
            "reference_model_is_settlement_calendar": False,
            "removed_closed_missing_rows": len(removed_dates),
            "inserted_missing_sessions": (
                len(inserted_dates) if require_all_sessions else 0
            ),
            "absent_calendar_sessions": len(inserted_dates),
            "require_all_sessions": bool(require_all_sessions),
            "finite_quotes_on_non_session": len(finite_closed),
            "overrides_applied": sum(
                p.index.min() <= t <= p.index.max() for t in applicable
            ),
            "source_backed_exceptions": [
                {"Date": str(t.date()), "Status": v[0], "Source": v[1]}
                for t, v in sorted(applicable.items())
                if p.index.min() <= t <= p.index.max()
            ],
            "historical_caveat": "Venue/calendar models are not independently verified for every historic date; CME TradeDate is a business-day proxy, not a confirmed Yahoo settlement calendar.",
        },
        pd.DataFrame(audit, columns=["Date", "Status", "Source"]),
    )


def public_quality(quality):
    return {k: v for k, v in quality.items() if not k.startswith("_")}


def write_source_audit(z, quality):
    for key, name in [
        ("_raw_source_csv", "source_rows.csv"),
        ("_raw_prices_csv", "raw_source_prices.csv"),
        ("_calendar_audit_csv", "calendar_audit.csv"),
    ]:
        if key in quality:
            z.writestr(name, quality[key])


CONTROL_DISJOINT = "Kontrollfenster ohne Überschneidung"
CONTROL_SHIFTED = "Kalender-verschobene Referenzfenster (überlappend)"
HISTORY_OPTIONS = [
    "Gesamte verfügbare Historie",
    "40 Jahre",
    "30 Jahre",
    "20 Jahre",
    "15 Jahre",
    "10 Jahre",
    "Ab Startdatum",
]


def history_request(selection, start_date=None, now=None):
    """Return an explicit Yahoo request; never synthesize older observations."""
    if selection not in HISTORY_OPTIONS:
        raise ValueError("Unbekannte Historienauswahl.")
    if selection == HISTORY_OPTIONS[0]:
        return {"period": "max"}
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if pd.isna(now) or now.tzinfo is None:
        raise ValueError("Zeitstempel benötigt Zeitzone.")
    if selection == "Ab Startdatum":
        start = pd.Timestamp(start_date)
        if (
            pd.isna(start)
            or start.date() < date(1900, 1, 1)
            or start.date() > now.date()
        ):
            raise ValueError("Startdatum muss zwischen 1900 und heute liegen.")
        return {"start": str(start.date())}
    years = int(selection.split()[0])
    return {"start": str((now - pd.DateOffset(years=years)).date())}


def fetch_yahoo_history(asset, request, yf_module=None):
    if yf_module is None:
        import yfinance as yf_module
    if (
        asset not in SYMBOLS
        or not isinstance(request, dict)
        or set(request) not in [{"period"}, {"start"}]
    ):
        raise ValueError("Ungültiges Asset oder Historien-Anfrage.")
    if "period" in request and request["period"] != "max":
        raise ValueError("Period-Anfrage muss max sein.")
    h = yf_module.Ticker(SYMBOLS[asset]).history(
        **request,
        interval="1d",
        auto_adjust=False,
        back_adjust=False,
        repair=False,
        keepna=True,
        actions=False,
        timeout=60,
        raise_errors=True,
    )
    if h is None or h.empty or "Close" not in h:
        raise ValueError("Yahoo lieferte keine Tagespreise für diese Anfrage.")
    frame = h.reset_index()
    raw = frame.to_csv(index=False).encode()
    return frame, {
        "source": "Yahoo " + SYMBOLS[asset],
        "received_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "yfinance": yf_module.__version__,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "history_request": request,
        "raw_first_provider_label": str(h.index.min()),
        "raw_last_provider_label": str(h.index.max()),
    }


def calculation_policy():
    return {
        "nonpositive_price_rule": "raw retained; return requires finite positive endpoints; path holes at nonpositive prices",
        "missing_rule": "no price fill; calendar-classified non-session NaNs may be removed; nonnumeric nonempty prices retained as barriers; absent expected sessions inserted as NaN only when require_all_sessions enabled; intervals containing missing prices and gaps >7 calendar days rejected",
        "path_risk_rule": "MAE/MFE/drawdown unavailable for any path containing a nonpositive price",
        "calendar_coverage_rule": "calendar intervals require coverage through end date; events require all requested observed-row offsets",
        "PIT_verified": False,
    }


def load_bytes(data, name, pair, member=None):
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            candidates = [
                n for n in z.namelist() if n.endswith("research_timeseries.csv")
            ]
            expected = ASSETS[pair][1] + "_research_timeseries.csv"
            if member is not None:
                candidates = [
                    n
                    for n in z.namelist()
                    if n == member and n.lower().endswith(".csv")
                ]
            else:
                candidates = [n for n in candidates if n.split("/")[-1] == expected]
            if len(candidates) != 1:
                raise ValueError(
                    "Keine eindeutige Preis-CSV für das ausgewählte Asset. CSV im Dashboard auswählen oder direkt hochladen."
                )
            info = z.getinfo(candidates[0])
            if info.file_size > 32 * 1024 * 1024:
                raise ValueError("CSV im ZIP ist größer als 32 MiB.")
            frame = pd.read_csv(z.open(candidates[0]))
    else:
        frame = pd.read_csv(io.BytesIO(data))
    return frame


def prepare(
    frame,
    asset=None,
    now=None,
    calendar_mode=CALENDAR_STRICT,
    overrides=None,
    require_all_sessions=False,
):
    """Keep provider date labels and raw nonpositive/missing prices.

    Positive-only assets reject nonpositive prices; WTI and unspecified
    series retain them. Missing prices are never filled. Enabled calendars
    may remove empty non-observation days; other NaNs remain barriers.
    The conservative cutoff excludes the current provider day in every mode.
    """
    dc = next(
        (c for c in ["Date", "date", "Datetime", "provider_timestamp"] if c in frame),
        None,
    )
    pc = next((c for c in ["asset_price", "Close", "close"] if c in frame), None)
    if dc is None or pc is None:
        raise ValueError("Benötigte CSV-Spalten: Date und asset_price oder Close.")
    labels = []
    for value in frame[dc]:
        try:
            stamp = pd.Timestamp(value)
            if pd.isna(stamp):
                raise ValueError("missing")
            labels.append(stamp.date())
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(
                "Fehlender oder ungültiger Datumswert: Quelle zuerst prüfen."
            ) from exc
    idx = pd.DatetimeIndex(labels)
    if idx.has_duplicates:
        raise ValueError("Doppelte Tageslabels: Quelle zuerst prüfen.")
    p = pd.Series(
        pd.to_numeric(frame[pc], errors="coerce").to_numpy(dtype=float),
        index=idx,
        name="Close",
    ).sort_index()
    original_values = pd.Series(frame[pc].to_numpy(), index=idx).sort_index()
    invalid_dates = p.index[
        p.isna()
        & original_values.notna()
        & original_values.astype(str).str.strip().ne("")
    ]
    raw_prices_csv = p.rename_axis("Date").to_csv()
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if pd.isna(now) or now.tzinfo is None:
        raise ValueError("Zeitstempel benötigt Zeitzone.")
    cutoff = min(
        now.tz_convert(t).date() for t in ["UTC", "Europe/London", "America/New_York"]
    )
    keep = (p.index.dayofweek < 5) & (p.index < pd.Timestamp(cutoff))
    removed = int((~keep).sum())
    p = p.loc[keep]
    if p.empty or not np.isfinite(p).any():
        raise ValueError("Keine gültige Historie mit endlichen Preisen.")
    if np.isinf(p).any():
        raise ValueError("Unendliche Preise: Quelle zuerst prüfen.")
    nonpositive = p <= 0
    if asset is not None and asset != "WTI" and nonpositive.any():
        raise ValueError(
            "Null-/Negativpreis für dieses Asset: Quelle und Asset-Zuordnung prüfen."
        )
    original_missing = int(p.isna().sum())
    p, calendar_info, calendar_audit = apply_calendar(
        p, asset, calendar_mode, overrides, require_all_sessions, invalid_dates
    )
    nonpositive = p <= 0
    return p, {
        "cutoff_exclusive": str(cutoff),
        "removed_rows": removed,
        "rows": len(p),
        "first": str(p.index.min().date()),
        "last": str(p.index.max().date()),
        "nonpositive_rows": int(nonpositive.sum()),
        "nonpositive_dates": [str(t.date()) for t in p.index[nonpositive]],
        "missing_price_rows": int(p.isna().sum()),
        "source_missing_price_rows_before_calendar": original_missing,
        "nonnumeric_nonempty_dates": [
            str(t.date()) for t in invalid_dates if t in p.index
        ],
        "missing_price_dates": [str(t.date()) for t in p.index[p.isna()]],
        "calendar": calendar_info,
        "_raw_source_csv": frame.to_csv(index=False),
        "_raw_prices_csv": raw_prices_csv,
        "_calendar_audit_csv": calendar_audit.to_csv(index=False),
        "largest_calendar_gap_days": (
            int(p.index.to_series().diff().dt.days.max()) if len(p) > 1 else 0
        ),
    }


def safe_return(start_price, end_price):
    """Conventional percentage returns require finite, positive endpoints."""
    if not (
        np.isfinite(start_price)
        and np.isfinite(end_price)
        and start_price > 0
        and end_price > 0
    ):
        return np.nan
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value = float(end_price / start_price - 1.0)
    return value if np.isfinite(value) else np.nan


def daily_returns(p):
    previous = p.shift(1)
    valid = (p > 0) & (previous > 0) & np.isfinite(p) & np.isfinite(previous)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        returns = (p / previous - 1.0).where(valid)
    gaps = p.index.to_series().diff().dt.days
    return returns.where(np.isfinite(returns) & (gaps <= 7))


def sequence_valid(seq):
    return (
        len(seq) >= 2
        and np.isfinite(seq).all()
        and seq.index.to_series().diff().dt.days.max() <= 7
    )


def common_path_length(obs):
    """Structural row count, including NaN holes inside a price path."""
    return int(obs["bars"].min()) + 1 if not obs.empty else 0


def requested_seasonal_windows(p, start_md, end_md, years):
    """Block every selected seasonal return interval, including rejected ones."""
    rows = []
    for y in years:
        try:
            start = pd.Timestamp(y, *start_md)
            end = pd.Timestamp(y + int(end_md < start_md), *end_md)
        except ValueError:
            continue
        prior = p.index[p.index < start]
        rows.append(
            {
                "entry": prior[-1] if len(prior) else start - pd.Timedelta(days=1),
                "exit": end,
            }
        )
    return pd.DataFrame(rows, columns=["entry", "exit"])


def interval_record(p, start, end, label, year):
    """Prior close to last inside close; require coverage through calendar end."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    coverage = pd.Timestamp(p.attrs.get("coverage_through", p.index.max()))
    if p.empty or start > end or end > coverage or start <= p.index.min():
        return None
    prior = p.loc[p.index < start]
    inside = p.loc[(p.index >= start) & (p.index <= end)]
    if prior.empty or inside.empty:
        return None
    entry, exit_ = prior.index[-1], inside.index[-1]
    seq = p.loc[entry:exit_]
    if (start - entry).days > 7 or (end - exit_).days > 7 or not sequence_valid(seq):
        return None
    ret = safe_return(seq.iloc[0], seq.iloc[-1])
    if not np.isfinite(ret):
        return None
    r = daily_returns(seq).dropna()
    path_complete = bool((seq > 0).all())
    return {
        "group": label,
        "year": year,
        "planned_start": str(start.date()),
        "planned_end": str(end.date()),
        "entry": str(entry.date()),
        "exit": str(exit_.date()),
        "bars": len(inside),
        "return": ret,
        "nonpositive_inside": int((seq <= 0).sum()),
        "daily_vol": (
            float(r.std(ddof=1) * np.sqrt(252))
            if len(r) > 1 and path_complete
            else np.nan
        ),
    }


def observations(p, mode, custom=None):
    rows = []
    if p.empty:
        return pd.DataFrame(
            columns=["group", "year", "entry", "exit", "bars", "return"]
        )
    if mode == "Wochentage":
        r = daily_returns(p)
        gaps = p.index.to_series().diff().dt.days
        for t, v in r.items():
            if pd.notna(v) and gaps.loc[t] <= 7:
                rows.append(
                    {
                        "group": ["Mo", "Di", "Mi", "Do", "Fr"][t.dayofweek],
                        "year": t.year,
                        "entry": str(p.index[p.index.get_loc(t) - 1].date()),
                        "exit": str(t.date()),
                        "bars": 1,
                        "return": float(v),
                    }
                )
    else:
        year_labels = (
            p.index.isocalendar().year
            if mode in ["Kalenderwochen", "Freie Wochenspanne"]
            else p.index.year
        )
        for y in range(int(min(year_labels)), int(max(year_labels)) + 1):
            intervals = []
            if mode == "Monate":
                for m in range(1, 13):
                    a = pd.Timestamp(y, m, 1)
                    intervals.append((a, a + pd.offsets.MonthEnd(0), f"{m:02d}"))
            elif mode == "Kalenderwochen":
                for w in range(1, 54):
                    try:
                        a = pd.Timestamp(date.fromisocalendar(y, w, 1))
                    except ValueError:
                        continue
                    intervals.append((a, a + pd.Timedelta(days=6), f"KW{w:02d}"))
            elif mode == "Freie Datumsspanne":
                sm, sd, em, ed = custom
                try:
                    a = pd.Timestamp(y, sm, sd)
                    b = pd.Timestamp(y + int((em, ed) < (sm, sd)), em, ed)
                    intervals = [(a, b, f"{sd:02d}.{sm:02d}–{ed:02d}.{em:02d}")]
                except ValueError:
                    continue  # e.g.29 February in a non-leap year
            elif mode == "Freie Wochenspanne":
                sw, ew = custom
                try:
                    a = pd.Timestamp(date.fromisocalendar(y, sw, 1))
                    b = pd.Timestamp(date.fromisocalendar(y + int(ew < sw), ew, 7))
                    intervals = [(a, b, f"KW{sw:02d}–KW{ew:02d}")]
                except ValueError:
                    continue
            else:
                raise ValueError("Unbekannter Analysemodus: " + str(mode))
            for a, b, label in intervals:
                rec = interval_record(p, a, b, label, y)
                if rec:
                    rows.append(rec)
    return pd.DataFrame(rows)


def summarize(obs):
    rows = []
    if obs.empty:
        return pd.DataFrame(
            columns=[
                "Gruppe",
                "N",
                "Jahre",
                "Mittel_%",
                "Median_%",
                "Positiv_%",
                "Minimum_%",
                "Maximum_%",
            ]
        ).set_index("Gruppe")
    for key, g in obs.groupby("group", sort=True):
        r = g["return"]
        rows.append(
            {
                "Gruppe": key,
                "N": len(g),
                "Jahre": g.year.nunique(),
                "Mittel_%": 100 * r.mean(),
                "Median_%": 100 * r.median(),
                "Positiv_%": 100 * (r > 0).mean(),
                "Minimum_%": 100 * r.min(),
                "Maximum_%": 100 * r.max(),
            }
        )
    return pd.DataFrame(rows).set_index("Gruppe")


def event_observations(p, events, before, after):
    """Return segments share boundary closes only; no event-day inference."""
    if (
        not isinstance(before, (int, np.integer))
        or not isinstance(after, (int, np.integer))
        or before < 2
        or after < 1
    ):
        raise ValueError(
            "Event-Fenster: mindestens zwei Zeilen vorher und eine danach."
        )
    rows, rejected = [], []
    for _, e in events.iterrows():
        day = pd.Timestamp(e["Date"]).normalize()
        kind = str(e["Event"])
        reason = None
        if day not in p.index:
            reason = "Kein Kursdatum; keine automatische Verschiebung"
        else:
            i = p.index.get_loc(day)
            if i - before < 0 or i + after >= len(p):
                reason = "Fenster nicht vollständig"
            else:
                path = p.iloc[i - before : i + after + 1]
                if not sequence_valid(path):
                    reason = "Fehlender Preis oder Quellenlücke >7 Kalendertage"
        if reason:
            rejected.append(
                {
                    "Date": str(day.date()),
                    "Event": kind,
                    "window": "Alle",
                    "reason": reason,
                }
            )
            continue
        for label, a, b in [
            ("Vorher", i - before, i - 1),
            ("Ereignistag", i - 1, i),
            ("Nachher", i, i + after),
            ("Gesamt", i - before, i + after),
        ]:
            ret = safe_return(p.iloc[a], p.iloc[b])
            if not np.isfinite(ret):
                rejected.append(
                    {
                        "Date": str(day.date()),
                        "Event": kind,
                        "window": label,
                        "reason": "Null-/Negativpreis oder ungültiger Wert an Renditegrenze",
                    }
                )
                continue
            rows.append(
                {
                    "Date": str(day.date()),
                    "Event": kind,
                    "group": kind + " | " + label,
                    "window": label,
                    "year": day.year,
                    "entry": str(p.index[a].date()),
                    "exit": str(p.index[b].date()),
                    "bars": b - a,
                    "return": ret,
                    "nonpositive_inside": int((p.iloc[a : b + 1] <= 0).sum()),
                    "full_start": str(path.index[0].date()),
                    "full_end": str(path.index[-1].date()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(
        rejected, columns=["Date", "Event", "window", "reason"]
    )


def event_dashboard(st, p, asset, provenance, quality, date_range=None):
    st.subheader("Event-Studie")
    st.warning(
        "Tagesdaten trennen keine unmittelbare Reaktion auf eine Veröffentlichung. Tag0 ist das Anbieter-Datumslabel, nicht eine verifizierte Veröffentlichungssitzung. Keine Kausalitäts- oder Intraday-Aussage."
    )
    source = st.radio(
        "Event-Termine",
        [
            "Offizielle / eigene Termine als CSV",
            "Quartalsverfall: Kalenderregel (ungeprüfte Näherung)",
        ],
    )
    if source.startswith("Quartalsverfall"):
        st.warning(
            "Erzeugt den dritten Freitag im März, Juni, September und Dezember. Feiertagsverschiebungen und produkt-/börsenspezifische Abrechnung fehlen. Für echte Hexensabbat-Termine einen geprüften Kalender als CSV verwenden. Kein Verfallskalender für Gold-, Öl- oder FX-Kontrakte."
        )
        rows = []
        for y in range(p.index.min().year, p.index.max().year + 1):
            for m in [3, 6, 9, 12]:
                first = pd.Timestamp(y, m, 1)
                day = first + pd.Timedelta(days=(4 - first.dayofweek) % 7 + 14)
                rows.append(
                    {
                        "Date": str(day.date()),
                        "Event": "Quartalsverfall-Regel ungeprüft",
                        "Source": "third-Friday calendar rule",
                    }
                )
        events = pd.DataFrame(rows)
        event_source = {"type": "calendar_rule", "verified": False}
    else:
        st.markdown(
            "Terminreferenzen: [Fed-Entscheidungen](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) · [US-CPI](https://www.bls.gov/schedule/news_release/cpi.htm) · [CPI-Archiv](https://www.bls.gov/bls/news-release/cpi.htm) · [Cboe-Verfallskalender](https://www.cboe.com/about/hours/us-options/)"
        )
        st.write(
            "CSV-Spalten: Date, Event, optional Source. Date im Format JJJJ-MM-TT; Event z. B. FOMC, CPI oder Hexensabbat. Bei FOMC den Entscheidungstag verwenden, bei CPI den Veröffentlichungstag – nicht den Berichtsmonat. Uhrzeitinformationen werden in dieser Tagesstudie nicht ausgewertet."
        )
        st.download_button(
            "Leere Event-CSV-Vorlage",
            "Date,Event,Source\n",
            file_name="event_calendar_template.csv",
            mime="text/csv",
        )
        upload = st.file_uploader(
            "Event-Kalender hochladen", type=["csv"], key="event_csv"
        )
        if upload is None:
            return
        raw = upload.getvalue()
        try:
            events = pd.read_csv(io.BytesIO(raw))
            if not {"Date", "Event"}.issubset(events):
                raise ValueError("Date und Event fehlen.")
            if not events.Date.astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
                raise ValueError("Date muss JJJJ-MM-TT sein, ohne Uhrzeit.")
            events["Date"] = pd.to_datetime(
                events.Date, format="%Y-%m-%d", errors="raise"
            )
            if (
                events.Event.isna().any()
                or events.Event.astype(str).str.strip().eq("").any()
            ):
                raise ValueError("Event-Bezeichnung fehlt.")
            events["Event"] = events.Event.astype(str).str.strip()
            if events.duplicated(["Date", "Event"]).any():
                raise ValueError("Doppelte Date/Event-Kombinationen entfernen.")
        except Exception as e:
            st.error(str(e))
            return
        event_source = {
            "type": "uploaded_calendar",
            "filename": upload.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "verified": False,
        }
    if date_range is not None:
        events = events[pd.to_datetime(events.Date).between(*date_range)].copy()
        st.caption(
            "Nur Ereignistage innerhalb des gewählten Von–Bis-Zeitraums; Vor-/Nachfenster dürfen darüber hinausreichen."
        )
    if events.empty:
        st.info("Keine Ereignisse im gewählten Zeitraum.")
        return
    kinds = st.multiselect(
        "Ereignisse",
        sorted(events.Event.unique()),
        default=sorted(events.Event.unique()),
    )
    events = events[events.Event.isin(kinds)].copy()
    before = int(
        st.number_input(
            "Beginn: Handelstage vor Tag0", min_value=2, max_value=60, value=5
        )
    )
    after = int(
        st.number_input(
            "Ende: Handelstage nach Tag0", min_value=1, max_value=60, value=5
        )
    )
    st.caption(
        f"Vorher: Schluss T−{before} bis T−1; Ereignistag: T−1 bis T0; Nachher: T0 bis T+{after}. Die Segmente sind getrennt; „Gesamt“ umfasst alle. Ein Handelstag ist hier eine verfügbare Kurszeile."
    )
    obs, rejected = event_observations(p, events, before, after)
    if obs.empty:
        st.warning("Keine vollständigen Event-Fenster.")
        st.dataframe(rejected)
        return
    windows = (
        obs[["Date", "Event", "full_start", "full_end"]]
        .drop_duplicates()
        .sort_values(["full_start", "full_end"])
    )
    overlap = []
    prior_end = None
    for _, row in windows.iterrows():
        overlap.append(prior_end is not None and row.full_start <= prior_end)
        prior_end = max(prior_end or row.full_end, row.full_end)
    windows["overlaps_prior_window"] = overlap
    st.caption(
        f"{len(windows)} Ereignisse mit mindestens einem gültigen Segment; {len(rejected)} Ausschluss-Einträge (ganze Ereignisse oder einzelne Segmente); {sum(overlap)} Fenster überschneiden ein vorheriges. Ereignisse können zugleich auftreten und sind nicht unabhängig."
    )
    summary = direction_summary(obs)
    st.dataframe(summary.round(3))
    st.bar_chart(summary[["Mittel_%", "Median_%"]])
    annual = (
        obs.pivot_table(index="year", columns="group", values="return", aggfunc="mean")
        * 100
    )
    st.subheader("Jahresmittel in %")
    st.dataframe(annual.round(3))
    with st.expander("Einzelereignisse, Überschneidungen und Ausschlüsse"):
        st.dataframe(obs)
        st.dataframe(windows)
        st.dataframe(rejected)
    protocol = {
        "version": VERSION,
        "asset": asset,
        "analysis": "event study",
        "before": before,
        "after": after,
        "price_source": provenance,
        "quality": public_quality(quality),
        "event_source": event_source,
        "selected_events": kinds,
        "selected_date_range": date_range,
        "date_alignment": "exact provider date; absent dates rejected",
        "timing_verified": False,
        "exploratory": True,
        "multiple_testing_adjusted": False,
        "costs_included": False,
        "promotion_authorized": False,
        "overlap_rule": "inclusive full window overlap flagged; observations retained",
        "interpretation": "calendar association, not causal release surprise or executable strategy",
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in [
            ("prices", p.rename_axis("Date").to_frame()),
            ("events", events),
            ("observations", obs),
            ("summary", summary),
            ("annual", annual),
            ("overlap", windows),
            ("excluded", rejected),
        ]:
            z.writestr(name + ".csv", df.to_csv())
        write_source_audit(z, quality)
        protocol.update(calculation_policy())
        z.writestr("protocol.json", json.dumps(protocol, indent=2, default=str))
    st.download_button(
        "Event-Auswertung als ZIP",
        buf.getvalue(),
        file_name=ASSETS[asset][1] + "_Events_v1_7_0.zip",
        mime="application/zip",
    )


def direction_summary(obs):
    tab = summarize(obs)
    if obs.empty:
        return tab.assign(
            **{
                "Negativ_%": pd.Series(dtype=float),
                "Unverändert_%": pd.Series(dtype=float),
                "Beschreibung": pd.Series(dtype=str),
            }
        )
    grouped = obs.groupby("group")["return"]
    tab["Negativ_%"] = grouped.apply(lambda r: 100 * (r < 0).mean())
    tab["Unverändert_%"] = grouped.apply(lambda r: 100 * (r == 0).mean())

    def label(r):
        if r["Mittel_%"] > 0 and r["Median_%"] > 0 and r["Positiv_%"] > 50:
            return "Historisch eher Long"
        if r["Mittel_%"] < 0 and r["Median_%"] < 0 and r["Negativ_%"] > 50:
            return "Historisch eher Short"
        return "Gemischt / keine eindeutige Richtung"

    tab["Beschreibung"] = tab.apply(label, axis=1)
    return tab


def selected_days(p, start, end):
    r = daily_returns(p)
    gaps = p.index.to_series().diff().dt.days
    rows = []
    for t in p.loc[pd.Timestamp(start) : pd.Timestamp(end)].index:
        if pd.notna(r.loc[t]) and gaps.loc[t] <= 7:
            i = p.index.get_loc(t)
            rows.append(
                {
                    "group": "Ausgewählte Tage",
                    "year": t.year,
                    "entry": str(p.index[i - 1].date()),
                    "exit": str(t.date()),
                    "bars": 1,
                    "return": float(r.loc[t]),
                }
            )
    return pd.DataFrame(rows)


def period_dashboard(st, p, asset, provenance, quality):
    st.subheader("Dein Zeitraum und historische Richtung")
    a, b = st.columns(2)
    start = a.date_input(
        "Von",
        value=max(p.index.min().date(), (p.index.max() - pd.Timedelta(days=90)).date()),
        min_value=p.index.min().date(),
        max_value=p.index.max().date(),
        key="period_start_" + asset,
    )
    end = b.date_input(
        "Bis einschließlich",
        value=p.index.max().date(),
        min_value=p.index.min().date(),
        max_value=p.index.max().date(),
        key="period_end_" + asset,
    )
    mode = st.radio(
        "Was möchtest du untersuchen?",
        [
            "Kursverlauf in genau diesem Zeitraum",
            "Diese Kalenderspanne in früheren Jahren",
            "Ereignisse in diesem Zeitraum",
        ],
    )
    if mode != "Diese Kalenderspanne in früheren Jahren" and start > end:
        st.error("Von muss vor oder auf Bis liegen.")
        return
    st.caption(
        "Long/Short beschreibt historische Kursbewegungen vor Kosten. Eine Mehrheit positiver Tage ist keine Prognose und keine Handelsfreigabe."
    )
    if mode == "Ereignisse in diesem Zeitraum":
        event_dashboard(
            st, p, asset, provenance, quality, (pd.Timestamp(start), pd.Timestamp(end))
        )
        return
    if mode == "Diese Kalenderspanne in früheren Jahren":
        st.caption(
            "Monat und Tag von Von/Bis bilden das jährlich wiederholte Fenster. Die Jahreszahlen wählst du darunter separat. Ein Ende vor dem Startmonat/-tag bedeutet Jahreswechsel."
        )
        lo, hi = p.index.min().year, p.index.max().year
        years = (
            st.slider("Historische Vergleichsjahre (Startjahr)", lo, hi, (lo, hi))
            if lo < hi
            else (lo, hi)
        )
        obs = observations(
            p, "Freie Datumsspanne", (start.month, start.day, end.month, end.day)
        )
        if not obs.empty:
            obs = obs[obs.year.between(*years)]
        if obs.empty:
            st.warning("Keine vollständig abgedeckten Jahresfenster.")
            return
        st.caption(
            "Eine Beobachtung je vollständigem Jahresfenster. Einstieg: Schlusskurs vor Beginn; Ausstieg: letzter Schlusskurs innerhalb des Fensters."
        )
    else:
        obs = selected_days(p, start, end)
        if obs.empty:
            st.warning("Keine auswertbaren Tagesrenditen.")
            return
        total = interval_record(p, start, end, "Gesamter Zeitraum", start.year)
        if total:
            st.metric(
                "Kursveränderung im gesamten Zeitraum", f"{100*total['return']:+.2f} %"
            )
            st.caption(
                f"Vom Schlusskurs {total['entry']} bis Schlusskurs {total['exit']}."
            )
        else:
            st.warning(
                "Gesamtrendite wegen fehlendem Vortag, Quellenlücken, fehlenden Preisen oder ungültiger Renditegrenze nicht ausgewiesen."
            )
        st.line_chart(p.loc[str(start) : str(end)].to_frame())
        st.caption(
            "Tagesrenditen: vorheriger verfügbarer Schlusskurs bis Tagesschluss. Fehlender Vortag, fehlende Preise, Null-/Negativpreise an Renditegrenzen und Lücken über sieben Kalendertage werden ausgeschlossen; keine Intraday-Renditen."
        )
    summary = direction_summary(obs)
    st.dataframe(summary.round(3))
    st.caption(
        "„Eher Long“: Mittel und Median positiv sowie mehr als 50 % positive Beobachtungen. „Eher Short“ entsprechend negativ. Sonst gemischt. Diese Beschreibungsregel ist kein statistischer Bestätigungstest."
    )
    detail = obs.copy()
    detail["Rendite_%"] = 100 * detail["return"]
    detail["Richtung"] = np.where(
        detail["return"] > 0,
        "Long",
        np.where(detail["return"] < 0, "Short", "Unverändert"),
    )
    st.subheader("Einzelne Tage / Jahresfenster")
    st.dataframe(detail.drop(columns=["return"]))
    protocol = {
        "version": VERSION,
        "asset": asset,
        "mode": mode,
        "from": str(start),
        "to": str(end),
        "comparison_years": (
            list(years) if mode == "Diese Kalenderspanne in früheren Jahren" else None
        ),
        "source": provenance,
        "quality": public_quality(quality),
        "exploratory": True,
        "costs_included": False,
        "promotion_authorized": False,
        "multiple_testing_adjusted": False,
        "direction_rule": "mean and median same sign and >50% observations of that sign; otherwise mixed",
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in [
            ("prices", p.rename_axis("Date").to_frame()),
            ("observations", detail),
            ("summary", summary),
        ]:
            z.writestr(name + ".csv", df.to_csv())
        write_source_audit(z, quality)
        protocol.update(calculation_policy())
        z.writestr("protocol.json", json.dumps(protocol, indent=2, default=str))
    st.download_button(
        "Meine Auswertung herunterladen",
        buf.getvalue(),
        file_name=ASSETS[asset][1] + "_Zeitraum_v1_7_0.zip",
        mime="application/zip",
    )


def seasonal_paths(p, start_md, end_md, years, trend="Alle"):
    """Equal-weight yearly windows on observed trading-row offsets; no fill."""
    paths = {}
    rows = []
    excluded = []
    ma = p.rolling(200, min_periods=200).mean()
    for y in years:
        try:
            start = pd.Timestamp(y, *start_md)
            end = pd.Timestamp(y + int(end_md < start_md), *end_md)
        except ValueError:
            excluded.append(
                {"year": y, "reason": "Datum existiert nicht in diesem Jahr"}
            )
            continue
        rec = interval_record(p, start, end, "Saisonales Fenster", y)
        if rec is None:
            excluded.append(
                {
                    "year": y,
                    "reason": "Fenster unvollständig, fehlender Preis, Quellenlücke oder ungültige Renditegrenze",
                }
            )
            continue
        entry = pd.Timestamp(rec["entry"])
        state = (
            "Unbekannt"
            if pd.isna(ma.loc[entry])
            else (
                "Oberhalb MA200"
                if p.loc[entry] > ma.loc[entry]
                else "Unterhalb/gleich MA200"
            )
        )
        if trend != "Alle" and state != trend:
            excluded.append({"year": y, "reason": "Trendfilter: " + state})
            continue
        seq = p.loc[rec["entry"] : rec["exit"]]
        safe_seq = seq.where(seq > 0)
        path = safe_seq / safe_seq.iloc[0] - 1
        complete = bool((seq > 0).all())
        paths[str(y)] = pd.Series(path.to_numpy(), index=range(len(path)))
        rec.update(
            {
                "trend_at_entry": state,
                "path_complete": complete,
                "MAE_Close_Long": float(path.min()) if complete else np.nan,
                "MFE_Close_Long": float(path.max()) if complete else np.nan,
                "Max_Drawdown_Close": (
                    float((safe_seq / safe_seq.cummax() - 1).min())
                    if complete
                    else np.nan
                ),
            }
        )
        rows.append(rec)
    curves = pd.DataFrame(paths)
    curves.index.name = "Handelstage seit Einstieg"
    return pd.DataFrame(rows), curves, pd.DataFrame(excluded)


def path_statistics(curves):
    return pd.DataFrame(
        {
            "Mittel": curves.mean(axis=1) * 100,
            "Median": curves.median(axis=1) * 100,
            "P10": curves.quantile(0.10, axis=1) * 100,
            "P90": curves.quantile(0.90, axis=1) * 100,
            "N": curves.count(axis=1),
        }
    )


def control_windows(p, obs, excluded_windows, trend="Alle"):
    """Fixed disjoint return blocks anchored to each calendar year; no best selection.
    Block length equals that year's seasonal window. Drop blocks overlapping any
    seasonal window's return dates. Shared boundary closes do not share returns.
    """
    rows = []
    ma = p.rolling(200, min_periods=200).mean()
    blocked = np.zeros(len(p), dtype=bool)
    for _, r in excluded_windows.iterrows():
        blocked |= (p.index > pd.Timestamp(r.entry)) & (p.index <= pd.Timestamp(r.exit))
    for _, r in obs.iterrows():
        n = int(r.bars)
        idx = np.flatnonzero(p.index.year == int(r.year))
        if n < 1:
            continue
        for offset in range(0, len(idx) - n + 1, n):
            first = int(idx[offset])
            last = int(idx[offset + n - 1])
            entry = first - 1
            if entry < 0 or blocked[first : last + 1].any():
                continue
            seq = p.iloc[entry : last + 1]
            if not sequence_valid(seq):
                continue
            state = (
                "Unbekannt"
                if pd.isna(ma.iloc[entry])
                else (
                    "Oberhalb MA200"
                    if p.iloc[entry] > ma.iloc[entry]
                    else "Unterhalb/gleich MA200"
                )
            )
            if trend != "Alle" and state != trend:
                continue
            ret = safe_return(seq.iloc[0], seq.iloc[-1])
            if not np.isfinite(ret):
                continue
            rows.append(
                {
                    "year": int(r.year),
                    "entry": str(p.index[entry].date()),
                    "exit": str(p.index[last].date()),
                    "bars": n,
                    "return": ret,
                    "trend_at_entry": state,
                }
            )
    return pd.DataFrame(
        rows, columns=["year", "entry", "exit", "bars", "return", "trend_at_entry"]
    )


def shifted_reference_windows(p, obs, trend="Alle"):
    """Three pre-defined anchors per start year; equal observed-row length.

    References may overlap seasonal windows and one another. They are not
    independent controls and are never searched for best performance.
    """
    rows = []
    ma = p.rolling(200, min_periods=200).mean()
    for r in obs.itertuples():
        n = int(r.bars)
        if n < 1:
            continue
        target_entry, target_exit = pd.Timestamp(r.entry), pd.Timestamp(r.exit)
        for month in [4, 7, 10]:
            anchor = pd.Timestamp(int(r.year), month, 1)
            first = int(p.index.searchsorted(anchor))
            entry, last = first - 1, first + n - 1
            if entry < 0 or last >= len(p):
                continue
            if (p.index[first] - anchor).days > 7 or (anchor - p.index[entry]).days > 7:
                continue
            if p.index[entry] == target_entry and p.index[last] == target_exit:
                continue
            seq = p.iloc[entry : last + 1]
            if not sequence_valid(seq):
                continue
            state = (
                "Unbekannt"
                if pd.isna(ma.iloc[entry])
                else (
                    "Oberhalb MA200"
                    if p.iloc[entry] > ma.iloc[entry]
                    else "Unterhalb/gleich MA200"
                )
            )
            if trend != "Alle" and state != trend:
                continue
            ret = safe_return(seq.iloc[0], seq.iloc[-1])
            if not np.isfinite(ret):
                continue
            overlap = bool(
                ((seq.index[1:] > target_entry) & (seq.index[1:] <= target_exit)).any()
            )
            rows.append(
                {
                    "year": int(r.year),
                    "entry": str(p.index[entry].date()),
                    "exit": str(p.index[last].date()),
                    "bars": n,
                    "return": ret,
                    "trend_at_entry": state,
                    "reference_anchor": str(anchor.date()),
                    "overlaps_target_returns": overlap,
                    "reference_mode": "calendar_shifted",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "year",
            "entry",
            "exit",
            "bars",
            "return",
            "trend_at_entry",
            "reference_anchor",
            "overlaps_target_returns",
            "reference_mode",
        ],
    )


def build_comparison_windows(p, obs, blocked, trend, method):
    if method == CONTROL_DISJOINT:
        controls = control_windows(p, obs, blocked, trend)
        controls["reference_mode"] = "disjoint"
        controls["overlaps_target_returns"] = False
        return controls
    if method == CONTROL_SHIFTED:
        return shifted_reference_windows(p, obs, trend)
    raise ValueError("Unbekannte Vergleichsmethode.")


def paired_controls(obs, controls):
    if controls.empty or obs.empty:
        return pd.DataFrame(
            columns=["year", "return", "control_mean", "control_N", "difference"]
        )
    base = (
        controls.groupby("year")["return"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "control_mean", "count": "control_N"})
    )
    paired = obs[["year", "return"]].merge(base, on="year", how="inner")
    paired["difference"] = paired["return"] - paired["control_mean"]
    return paired


def temporal_split(obs, cutoff):
    """Purge windows sharing any time across the split boundary."""
    if obs.empty:
        return obs.copy(), obs.copy(), obs.copy()
    cut = pd.Timestamp(cutoff)
    left = obs[pd.to_datetime(obs.exit) < cut].copy()
    right = obs[pd.to_datetime(obs.entry) >= cut].copy()
    purge = obs[~obs.index.isin(left.index.union(right.index))].copy()
    return left, right, purge


def comparison_row(label, seasonal, controls):
    pairs = paired_controls(seasonal, controls)
    return {
        "Bereich": label,
        "Saison_N": len(seasonal),
        "Jahre_mit_Kontrolle": len(pairs),
        "Saison_Mittel_%": 100 * seasonal["return"].mean() if len(seasonal) else np.nan,
        "Saison_positiv_%": (
            100 * (seasonal["return"] > 0).mean() if len(seasonal) else np.nan
        ),
        "Saison_Mittel_Common_%": (
            100 * pairs["return"].mean() if len(pairs) else np.nan
        ),
        "Kontrolle_Mittel_Common_%": (
            100 * pairs.control_mean.mean() if len(pairs) else np.nan
        ),
        "Differenz_Common_Prozentpunkte": (
            100 * pairs.difference.mean() if len(pairs) else np.nan
        ),
    }


def seasonal_dashboard(st, p, asset, provenance, quality):
    st.subheader("Saisonale Kurven und Jahresvergleich")
    st.write(
        "Wähle ein jährlich wiederkehrendes Fenster und die Jahre, über die du es vergleichen möchtest."
    )
    a, b = st.columns(2)
    start = a.date_input(
        "Fenster von (nur Monat/Tag)", date(2000, 1, 1), key="curve_start"
    )
    end = b.date_input(
        "Fenster bis (nur Monat/Tag)", date(2000, 12, 31), key="curve_end"
    )
    st.caption(
        "Liegt der Endmonat/-tag vor dem Beginn, läuft das Fenster ins Folgejahr. 29. Februar wird nur in passenden Schaltjahren ausgewertet."
    )
    lo, hi = p.index.min().year, p.index.max().year
    bounds = (
        st.slider("Historie: Startjahre", lo, hi, (lo, hi), key="curve_years")
        if lo < hi
        else (lo, hi)
    )
    trend = st.selectbox(
        "Marktumfeld beim Einstieg",
        ["Alle", "Oberhalb MA200", "Unterhalb/gleich MA200"],
    )
    st.caption(
        "MA200 verwendet die unveränderten 200 Kurszeilen bis zum Einstiegsschluss, einschließlich negativer WTI-Werte. Fehlende Preise machen den MA200 unbekannt. Das ist ein beschreibender Kontextfilter, keine Zusicherung einer Ausführung zu diesem Schlusskurs."
    )
    allobs, _, _ = seasonal_paths(
        p,
        (start.month, start.day),
        (end.month, end.day),
        range(bounds[0], bounds[1] + 1),
    )
    obs, curves, excluded = seasonal_paths(
        p,
        (start.month, start.day),
        (end.month, end.day),
        range(bounds[0], bounds[1] + 1),
        trend,
    )
    st.caption(
        f"{bounds[1]-bounds[0]+1} angefragte Startjahre → {len(allobs)} vollständig abgedeckte Fenster → {len(obs)} nach Kontextfilter."
    )
    if obs.empty:
        st.warning("Keine vollständigen Fenster für diese Auswahl.")
        st.dataframe(excluded)
        return
    st.caption(
        "Einstieg = letzter Schlusskurs vor dem Startdatum; Ausstieg = letzter Schlusskurs innerhalb des Fensters. Jedes Jahr erhält dasselbe Gewicht. Es werden keine Kurse aufgefüllt. Null-/Negativpreise bleiben roh erhalten; prozentuale Kurvenwerte fehlen an diesen Zeilen."
    )
    display = st.radio(
        "Kurvenlänge",
        ["Gemeinsame Länge aller Jahre", "Alle verfügbaren Handelstage"],
        horizontal=True,
    )
    shown = (
        curves.iloc[: common_path_length(obs)]
        if display.startswith("Gemeinsame")
        else curves
    )
    stats = path_statistics(shown)
    graph = stats.reset_index().rename(columns={"Handelstage seit Einstieg": "Tag"})
    # Vega-Lite is rendered by Streamlit; no additional plotting dependency.
    st.vega_lite_chart(
        graph,
        {
            "layer": [
                {
                    "mark": {"type": "area", "opacity": 0.15, "color": "#467ab8"},
                    "encoding": {
                        "x": {"field": "Tag", "type": "quantitative"},
                        "y": {
                            "field": "P10",
                            "type": "quantitative",
                            "title": "Kursveränderung (%)",
                        },
                        "y2": {"field": "P90"},
                    },
                },
                {
                    "mark": {"type": "line", "color": "#167c80"},
                    "encoding": {
                        "x": {"field": "Tag", "type": "quantitative"},
                        "y": {"field": "Mittel", "type": "quantitative"},
                        "tooltip": [
                            {"field": "Tag"},
                            {"field": "Mittel"},
                            {"field": "Median"},
                            {"field": "N"},
                        ],
                    },
                },
                {
                    "mark": {"type": "line", "color": "#e69b38", "strokeDash": [5, 3]},
                    "encoding": {
                        "x": {"field": "Tag", "type": "quantitative"},
                        "y": {"field": "Median", "type": "quantitative"},
                    },
                },
            ]
        },
        width="stretch",
    )
    st.caption(
        "Türkis: Mittel · Orange: Median · Band: historische 10.–90. Perzentile, kein Konfidenz- oder Prognoseintervall. X-Achse = verfügbare Handelstage ab Einstieg, keine identischen Kalenderdaten. Bei gemeinsamer Länge endet die Grafik am kürzesten Jahresfenster; die folgende Ergebnistabelle nutzt stets die vollständigen Fenster."
    )
    st.line_chart(stats[["N"]])
    show_years = st.multiselect(
        "Einzeljahre einblenden", list(curves.columns), default=list(curves.columns)
    )
    if show_years:
        st.line_chart(shown[show_years] * 100)
    summary = direction_summary(obs)
    pos = int((obs["return"] > 0).sum())
    neg = int((obs["return"] < 0).sum())
    st.write(
        f"{pos} von {len(obs)} Jahresfenstern gestiegen; {neg} gefallen; {len(obs)-pos-neg} unverändert."
    )
    st.dataframe(summary.round(3))
    st.caption(
        "Eine hohe Trefferquote allein belegt keinen saisonalen Vorteil. Der folgende Kontrollvergleich ist beschreibend, kein Signifikanztest."
    )
    blocked_windows = requested_seasonal_windows(
        p,
        (start.month, start.day),
        (end.month, end.day),
        range(bounds[0], bounds[1] + 1),
    )
    annual_window = (start.month, start.day, end.month, end.day) == (1, 1, 12, 31)
    control_method = st.selectbox(
        "Vergleichsbasis",
        [CONTROL_DISJOINT, CONTROL_SHIFTED],
        index=1 if annual_window else 0,
        key="control_method_" + asset + "_" + str(annual_window),
    )
    controls = build_comparison_windows(p, obs, blocked_windows, trend, control_method)
    paired = paired_controls(obs, controls)
    comparison = pd.DataFrame([comparison_row("Gesamte Auswahl", obs, controls)])
    st.subheader("Vergleich mit gleich langen Zeitfenstern")
    st.dataframe(comparison.round(3))
    if control_method == CONTROL_DISJOINT:
        st.caption(
            "Kontrollen: fest eingeteilte, gleich lange Blöcke im jeweiligen Startjahr. Keine gemeinsamen Renditetage zwischen Blöcken; alle angefragten Saisonfenster sind ausgeschlossen. Gleicher MA200-Filter am Einstieg; erst je Jahr mitteln, dann gemeinsame Jahre gleich gewichten."
        )
        st.caption(
            "Bei einem Ganzjahresfenster gibt es im selben Jahr keinen freien Kontrollbereich. Auch lange Teiljahresfenster können ohne passende Kontrollen bleiben. Kontrollen können zeitlich abhängig sein; keine Anpassung an Volatilität, Wochentage oder Nachrichten."
        )
    else:
        st.warning(
            "Diese Referenzen überlappen Saisonfenster und einander. Sie sind keine unabhängigen Kontrollbeobachtungen und belegen keinen saisonalen Vorteil."
        )
        st.caption(
            "Feste Referenzstarts: 1. April, 1. Juli und 1. Oktober des jeweiligen Saison-Startjahres. Die Länge entspricht der Zahl beobachteter Renditetage des Saisonfensters. Feiertage können den ersten Kurs verschieben. Identische Ziel-/Referenzfenster werden ausgelassen; fehlende oder ungültige Preise sperren betroffene Referenzen."
        )
        st.caption(
            "Referenzen können ins Folgejahr reichen. Nur vollständig beobachtete Fenster werden verwendet; kein Auffüllen. Gleicher MA200-Filter am Einstieg. Erst gültige Referenzen je Startjahr mitteln, dann gemeinsame Jahre gleich gewichten. Die Anzahl pro Jahr kann kleiner als drei sein."
        )
    if paired.empty:
        st.info(
            "Für diese Auswahl sind keine Kontrollfenster verfügbar. Keine Aussage über einen Vorteil gegenüber der Vergleichsbasis."
        )
    st.subheader("Zeitlich getrennte historische Prüfung")
    st.warning(
        "Diese Aufteilung ist eine historische Diagnose. Bereits betrachtete Daten werden dadurch nicht zu einem unberührten Holdout. Kein bestanden/nicht bestanden und keine automatische Promotion."
    )
    split_year = int(
        st.number_input(
            "Prüfzeitraum beginnt am 1. Januar des Jahres",
            min_value=int(bounds[0]),
            max_value=int(bounds[1] + 1),
            value=int(max(bounds[0], bounds[1] - 4)),
            key="seasonal_split_year",
        )
    )
    cutoff = pd.Timestamp(split_year, 1, 1)
    discovery, validation, purged = temporal_split(obs, cutoff)
    dc, vc, pc = temporal_split(controls, cutoff)
    split_summary = pd.DataFrame(
        [
            comparison_row("Früherer Zeitraum", discovery, dc),
            comparison_row("Späterer Prüfzeitraum", validation, vc),
        ]
    )
    st.dataframe(split_summary.round(3))
    st.caption(
        f"{len(purged)} saisonale Fenster und {len(pc)} Kontrollfenster an der Trennlinie ausgeschlossen. Fenster im Prüfbereich müssen vollständig ab der Grenze liegen, einschließlich Einstiegsschluss."
    )
    with st.expander("Kontrollfenster und gepaarte Jahresvergleiche"):
        st.dataframe(controls)
        st.dataframe(paired)
        st.dataframe(purged)
    r = obs["return"]
    risk = pd.DataFrame(
        [
            {
                "Gewinnmittel_%": 100 * r[r > 0].mean(),
                "Verlustmittel_%": 100 * r[r < 0].mean(),
                "Fenster_mit_vollständigem_positivem_Pfad": int(
                    obs.path_complete.sum()
                ),
                "Median_MAE_Close_Long_%": 100 * obs.MAE_Close_Long.median(),
                "Median_MFE_Close_Long_%": 100 * obs.MFE_Close_Long.median(),
                "Schlechtester_Close_Drawdown_%": 100 * obs.Max_Drawdown_Close.min(),
            }
        ]
    )
    st.subheader("Gewinne, Verluste und Bewegung innerhalb des Fensters")
    st.dataframe(risk.round(3))
    st.caption(
        "MAE/MFE: ungünstigste/günstigste Schlusskursbewegung relativ zum Einstieg einer Long-Position. Bei Null-/Negativpreisen innerhalb eines Pfades entfallen sämtliche MAE/MFE/Drawdown-Kennzahlen dieses Fensters; Kurven behalten an diesen Zeilen Lücken. Keine Intraday-Hochs/-Tiefs, kein Stop-Loss-Backtest; keine Kosten oder Rollkorrektur."
    )
    st.subheader("Stabilität: gleiche Auswahl über verschiedene Historien")
    periods = []
    for label, sub in [
        ("Gesamte Auswahl", obs),
        ("Letzte 10 Startjahre", obs[obs.year >= bounds[1] - 9]),
        ("Letzte 5 Startjahre", obs[obs.year >= bounds[1] - 4]),
    ]:
        if not sub.empty:
            row = direction_summary(sub).reset_index()
            row.insert(0, "Historie", label)
            periods.append(row)
    stability = pd.concat(periods, ignore_index=True)
    st.dataframe(stability.round(3))
    st.caption(
        "Diese Teilmengen überlappen; sie sind keine unabhängigen Bestätigungen. Einzelne Jahre werden nicht nach ihrem Ergebnis entfernt."
    )
    details = obs.copy()
    for c in ["return", "MAE_Close_Long", "MFE_Close_Long", "Max_Drawdown_Close"]:
        details[c + "_pct"] = 100 * details.pop(c)
    st.subheader("Jahresergebnisse")
    st.dataframe(details.round(3))
    st.bar_chart(details.set_index("year")[["return_pct"]])
    months = observations(p, "Monate")
    if not months.empty:
        months = months[months.year.between(*bounds)]
        heat = months.pivot(index="year", columns="group", values="return") * 100
        st.subheader("Monatsübersicht der gewählten Historie")
        st.caption(
            "Gesamte Monate, unabhängig vom gewählten Fenster und MA200-Filter; leere Zellen bleiben fehlend."
        )
        heatlong = months[["year", "group", "return"]].copy()
        heatlong["Rendite"] = heatlong.pop("return") * 100
        st.vega_lite_chart(
            heatlong,
            {
                "mark": "rect",
                "encoding": {
                    "x": {"field": "group", "type": "ordinal", "title": "Monat"},
                    "y": {"field": "year", "type": "ordinal", "title": "Jahr"},
                    "color": {
                        "field": "Rendite",
                        "type": "quantitative",
                        "scale": {"scheme": "redblue", "domainMid": 0},
                    },
                    "tooltip": [
                        {"field": "year"},
                        {"field": "group"},
                        {"field": "Rendite"},
                    ],
                },
            },
            width="stretch",
        )
        st.dataframe(heat.round(2))
    else:
        heat = pd.DataFrame()
    with st.expander("Ausgeschlossene Jahre und Kurvendaten"):
        st.dataframe(excluded)
        st.dataframe(stats)
    protocol = {
        "version": VERSION,
        "asset": asset,
        "analysis": "seasonal paths",
        "start_month_day": [start.month, start.day],
        "end_month_day": [end.month, end.day],
        "years": list(bounds),
        "trend_filter": trend,
        "alignment": "observed trading-row offset; no fill",
        "display_length": display,
        "comparison_method": control_method,
        "comparison_is_independent": False,
        "reference_anchors_month_day": (
            [[4, 1], [7, 1], [10, 1]] if control_method == CONTROL_SHIFTED else None
        ),
        "control_rule": (
            "fixed nonoverlapping return blocks in start year; all requested seasonal intervals excluded"
            if control_method == CONTROL_DISJOINT
            else "fixed April/July/October starts in target start year; equal observed-return-row length; overlaps permitted and flagged; identical target omitted"
        ),
        "control_weight_rule": "mean per start year, then equal weights on common paired years",
        "historical_split": str(cutoff.date()),
        "split_purge": "entry >= cutoff for later sample; exit < cutoff for earlier",
        "equal_year_weights": True,
        "band": "cross-year 10/90 percentiles, not confidence interval",
        "source": provenance,
        "quality": public_quality(quality),
        "exploratory": True,
        "holdout": False,
        "costs_included": False,
        "multiple_testing_adjusted": False,
        "promotion_authorized": False,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in [
            ("prices", p.rename_axis("Date").to_frame()),
            ("yearly_paths_decimal", curves),
            ("display_statistics_pct", stats),
            ("observations", details),
            ("summary", summary),
            ("risk", risk),
            ("stability", stability),
            ("monthly_pct", heat),
            ("excluded", excluded),
            ("controls", controls),
            ("paired_controls", paired),
            ("control_comparison", comparison),
            ("historical_split", split_summary),
            ("split_purged", purged),
            ("control_split_purged", pc),
            ("blocked_seasonal_windows", blocked_windows),
        ]:
            z.writestr(name + ".csv", df.to_csv())
        write_source_audit(z, quality)
        protocol.update(calculation_policy())
        z.writestr("protocol.json", json.dumps(protocol, indent=2, default=str))
    st.download_button(
        "Kurven und Auswertung als ZIP",
        buf.getvalue(),
        file_name=ASSETS[asset][1] + "_Seasonal_Curves_v1_7_0.zip",
        mime="application/zip",
    )


def main():
    import streamlit as st

    st.set_page_config(page_title="Multi-Asset Seasonality Lab", layout="wide")
    st.title("Multi-Asset Seasonality Lab v" + VERSION)
    st.write(
        "Kalendermuster der sieben Projekt-Assets untersuchen – separate Research-App."
    )
    st.warning(
        "Explorative Auswertung: Viele frei gewählte Zeitfenster erzeugen Zufallstreffer. Keine automatischen Signifikanz-, Modell- oder Handelsfreigaben."
    )
    pair = st.selectbox("Asset", list(SYMBOLS))
    source = st.radio(
        "Datenquelle", ["Vorhandene Audit-ZIP / CSV", "Yahoo abrufen"], horizontal=True
    )
    key = "seasonality_" + VERSION + "_" + pair
    if source == "Vorhandene Audit-ZIP / CSV":
        upload = st.file_uploader("Research-ZIP oder Tages-CSV", type=["zip", "csv"])
        if upload is None:
            st.info(
                "Für USD/CHF und USD/JPY funktionieren die bisherigen Audit-ZIPs. Für andere Assets: Preis-CSV oder ZIP mit Preis-CSV. Spalten: Date und asset_price oder Close. Alternativ Yahoo wählen."
            )
            return
        raw = upload.getvalue()
        member = None
        try:
            if upload.name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    members = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not members:
                    raise ValueError("Keine CSV in der ZIP.")
                preferred = ASSETS[pair][1] + "_research_timeseries.csv"
                default = next(
                    (i for i, n in enumerate(members) if n.split("/")[-1] == preferred),
                    0,
                )
                member = st.selectbox("Preis-CSV in der ZIP", members, index=default)
            st.caption(
                "Upload: Asset-Zuordnung und Preisart werden nicht aus den Kurswerten erkannt."
            )
            price_kind = st.selectbox(
                "Preisart der hochgeladenen Reihe",
                [
                    "Ungeprüft",
                    "Spot / Kassakurs",
                    "Preisindex",
                    "Total-Return-Index",
                    "Futures / kontinuierliche Futures-Reihe",
                ],
            )
            if not st.checkbox(
                "Die ausgewählte Datei gehört zum oben gewählten Asset.",
                key="confirm_"
                + pair
                + "_"
                + hashlib.sha256(raw + str(member).encode()).hexdigest()[:12],
            ):
                return
            frame = load_bytes(raw, upload.name, pair, member)
        except Exception as e:
            st.error(str(e))
            return
        provenance = {
            "source": upload.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "csv_member": member,
            "price_kind": price_kind,
            "asset_identity": "user confirmed; not independently verified",
        }
    else:
        st.info("Yahoo-Symbol: " + SYMBOLS[pair] + " · " + ASSETS[pair][2])
        st.caption(
            "Yahoo ist eine aktuelle, potenziell revidierte Historie. Kein Nachweis historischer Veröffentlichungszeitpunkte."
        )
        history_selection = st.selectbox(
            "Historie abrufen", HISTORY_OPTIONS, key="history_selection_" + pair
        )
        history_start = None
        if history_selection == "Ab Startdatum":
            history_start = st.date_input(
                "Gewünschtes Startdatum",
                date(1980, 1, 1),
                min_value=date(1900, 1, 1),
                max_value=pd.Timestamp.now(tz="UTC").date(),
                key="history_start_" + pair,
            )
        request = history_request(history_selection, history_start)
        st.caption(
            "Die maximale Reichweite ist je Yahoo-Symbol unterschiedlich. Ein frühes Startdatum garantiert keine Kurse ab diesem Datum. CSV-/Audit-Uploads können eine längere geprüfte Historie ergänzen; Quellen werden nicht automatisch verbunden."
        )
        if st.button("Tagesdaten abrufen"):
            try:
                with st.spinner("Historie wird abgerufen …"):
                    st.session_state[key] = fetch_yahoo_history(pair, request)
            except Exception as e:
                st.error("Abruf fehlgeschlagen: " + str(e))
                return
        if key not in st.session_state:
            return
        frame, provenance = st.session_state[key]
        if provenance.get("history_request") != request:
            st.info(
                "Die geladene Historie gehört zu einer anderen Auswahl. Bitte Tagesdaten erneut abrufen."
            )
            return
    if pair in ["Gold", "WTI"]:
        st.warning(
            "Futures-Ergebnisse sind Quellen-Saisonalität, keine vollständig rollbereinigte Strategie. Prozentrechnungen mit Null- oder Negativpreisen an einer Renditegrenze werden gesperrt; Werte werden weder entfernt noch künstlich ersetzt."
        )
    calendars = [CALENDAR_STRICT, CALENDAR_SESSIONS]
    if pair in ["WTI", "Gold"]:
        calendars.append(CALENDAR_REFERENCE)
    calendar_mode = st.selectbox(
        "Kalenderprüfung",
        calendars,
        index=1 if source == "Yahoo abrufen" else 0,
        key="calendar_mode_" + pair + "_" + source,
    )
    st.caption(
        "Asset-Kalender: "
        + ASSET_CALENDARS[pair]
        + ". Leere Nicht-Handelstage entfallen; verkürzte Sitzungen bleiben Handelstage. "
        "Historische Kalenderregeln sind Modelle. Für Uploads muss der Kalender zur Preisart passen. "
        "FX verwendet Werktage, keinen US-Börsenfeiertagskalender."
    )
    if calendar_mode == CALENDAR_REFERENCE:
        st.warning(
            "CME-Geschäftstage sind ein Näherungsmodell für Referenztage, kein bestätigter Kalender der Yahoo-Schluss- oder Settlementpreise. "
            "An Feiertagen kann trotzdem Handel stattfinden. Ergebnisse dieses Modus hängen von der Modellannahme ab."
        )
        if not st.checkbox(
            "Das CME-Geschäftstagsmodell für diese Preisreihe verwenden.",
            key="reference_model_" + pair,
        ):
            return
    overrides = None
    require_all_sessions = False
    if calendar_mode != CALENDAR_STRICT:
        require_all_sessions = st.checkbox(
            "Für jeden Kalendertag der gewählten Sitzungsliste einen Kurs erwarten.",
            key="require_all_sessions_" + pair,
        )
        st.caption(
            "Aktivieren, wenn die Quelle für jede gelistete Sitzung einen Tageskurs liefern muss. "
            "Dann sperren auch vollständig fehlende Zeilen betroffene Fenster. "
            "Ohne diese Option werden solche Tage nur protokolliert; vorhandene leere Zeilen bleiben außerhalb von Kalenderausnahmen gesperrt."
        )
    with st.expander("Quellenbelegte Kalenderausnahmen"):
        st.caption(
            "Optionale CSV: Date, Asset, Status, Source. Datum YYYY-MM-DD; Asset exakt wie oben. "
            "closed = kein Handel; no_reference = kein neuer Referenzkurs (nur Referenzmodus). "
            "Die Quellenangabe wird protokolliert, aber nicht automatisch verifiziert. "
            "Keine Ausnahme allein aus einer leeren Kurszeile ableiten. "
            "Im strikten Modus werden Ausnahmen nicht angewendet."
        )
        override_upload = st.file_uploader(
            "Kalenderausnahmen als CSV", type=["csv"], key="calendar_csv_" + pair
        )
        if override_upload is not None:
            try:
                overrides = pd.read_csv(
                    io.BytesIO(override_upload.getvalue()), dtype=str
                )
            except Exception as e:
                st.error("Kalender-CSV nicht lesbar: " + str(e))
                return
    try:
        p, quality = prepare(
            frame,
            pair,
            calendar_mode=calendar_mode,
            overrides=overrides,
            require_all_sessions=require_all_sessions,
        )
    except Exception as e:
        st.error(str(e))
        return
    if quality["nonpositive_rows"]:
        st.warning(
            f"{quality['nonpositive_rows']} Null-/Negativpreise bleiben in der Rohhistorie erhalten. Prozentrechnungen mit diesen Renditegrenzen entfallen."
        )
        with st.expander("Null-/Negativpreise anzeigen"):
            st.dataframe(p[p <= 0].to_frame())
    if quality["missing_price_rows"]:
        st.warning(
            f"{quality['missing_price_rows']} fehlende/nichtnumerische Preise bleiben als Lücken erhalten; betroffene Renditen und Fenster werden ausgeschlossen."
        )
    st.caption(
        f"{len(p):,} Tageswerte · {quality['first']} bis {quality['last']} · ausgeschlossene Zeilen: {quality['removed_rows']}"
    )
    cq = quality["calendar"]
    st.caption(
        f"Kalenderprüfung: {cq['removed_closed_missing_rows']} leere Zeilen an Kalender-Ausnahmetagen entfernt; "
        f"{cq['inserted_missing_sessions']} fehlende erwartete Tageszeilen als Lücken markiert; "
        f"{cq['finite_quotes_on_non_session']} vorhandene Preise außerhalb des Modells erhalten."
    )
    if cq["absent_calendar_sessions"] and not cq["require_all_sessions"]:
        st.warning(
            f"{cq['absent_calendar_sessions']} Kalender-Sitzungstage sind vollständig ohne Quellenzeile. "
            "Ob die Quelle dort einen neuen Tageskurs liefern müsste, ist ungeprüft. "
            "Diese Tage stehen im Prüfprotokoll; die vollständige Sitzungsabdeckung ist derzeit nicht erzwungen."
        )
    with st.expander("Kalender-Prüfprotokoll"):
        st.json(cq)
        audit = pd.read_csv(io.StringIO(quality["_calendar_audit_csv"]))
        st.dataframe(audit, use_container_width=True)
    st.caption(
        "Datum = Anbieter-Tageslabel. Kalenderklassifikation ist im Export dokumentiert; identische Session-Schlüsse, historische Kalenderregeln und PIT-Verfügbarkeit sind nicht unabhängig verifiziert."
    )
    mode = st.selectbox(
        "Analyse",
        [
            "Saisonale Kurven und Filter",
            "Mein Zeitraum + optionales Event",
            "Monate",
            "Wochentage",
            "Kalenderwochen",
            "Freie Datumsspanne",
            "Freie Wochenspanne",
            "Events",
        ],
    )
    if mode == "Saisonale Kurven und Filter":
        seasonal_dashboard(st, p, pair, provenance, quality)
        return
    if mode == "Mein Zeitraum + optionales Event":
        period_dashboard(st, p, pair, provenance, quality)
        return
    if mode == "Events":
        event_dashboard(st, p, pair, provenance, quality)
        return
    custom = None
    if mode == "Freie Datumsspanne":
        a, b = st.columns(2)
        start = a.date_input("Jährlich ab (Jahr wird ignoriert)", date(2000, 11, 15))
        end = b.date_input("Jährlich bis einschließlich", date(2000, 12, 15))
        custom = (start.month, start.day, end.month, end.day)
    elif mode == "Freie Wochenspanne":
        sw = st.number_input("Erste ISO-Kalenderwoche", 1, 53, 45)
        ew = st.number_input("Letzte ISO-Kalenderwoche einschließlich", 1, 53, 50)
        custom = (sw, ew)
    with st.expander("Welche Rendite wird gemessen?", expanded=True):
        st.write(
            "Monat/Woche/Spanne: letzter verfügbarer Schlusskurs VOR dem Beginn bis letzter Schlusskurs innerhalb der Spanne. Jahreswechsel werden unterstützt. Nicht vollständig abgedeckte Kalenderzeiträume werden ausgelassen; ein Wochenende am Datenende kann deshalb auch eine gerade beendete Woche ausschließen."
        )
        st.write(
            "Wochentag: Rendite vom vorherigen verfügbaren Schlusskurs zum Schlusskurs dieses Tages. Montag enthält typischerweise das Wochenende. Das ist keine Open-to-Close- oder Intraday-Rendite."
        )
        st.write(
            "Renditen sind Veränderungen der gewählten Preisreihe: keine Spreads, Finanzierung, Slippage oder Positionsgrößen. Fehlende Preise werden nicht aufgefüllt. Offensichtliche Lücken über sieben Kalendertage werden ausgeschlossen; kürzere vollständig fehlende Kurszeilen können unentdeckt bleiben. Explizit fehlende Preise sperren betroffene Fenster; Null-/Negativpreise bleiben erhalten und sperren konventionelle Renditen an den Endpunkten."
        )
    obs = observations(p, mode, custom)
    if obs.empty:
        st.warning("Keine vollständig auswertbaren Zeiträume.")
        return
    yrs = sorted(obs.year.unique())
    lo, hi = int(min(yrs)), int(max(yrs))
    if lo < hi:
        yrange = st.slider(
            "Auswertungsjahre (Startjahr bzw. ISO-Jahr)", lo, hi, (lo, hi)
        )
    else:
        yrange = (lo, hi)
    obs = obs[obs.year.between(*yrange)]
    if obs.empty:
        st.warning("Keine Beobachtungen im ausgewählten Bereich.")
        return
    summary = summarize(obs)
    st.subheader("Ergebnis je Kalendergruppe")
    st.dataframe(summary.round(3))
    st.bar_chart(summary[["Mittel_%", "Median_%"]])
    annual = (
        obs.pivot_table(index="year", columns="group", values="return", aggfunc="mean")
        * 100
    )
    st.subheader("Einzeljahre – Mittel der Beobachtungen in %")
    st.dataframe(annual.round(3))
    st.caption(
        "Bei Monaten/Wochen/freien Spannen meist eine Beobachtung je Jahr; bei Wochentagen ein Jahresmittel vieler Tagesrenditen. N ist keine Zahl unabhängiger Tests."
    )
    st.subheader("Zeitliche Stabilität")
    parts = []
    edges = np.unique(
        np.linspace(
            yrange[0], yrange[1] + 1, min(3, yrange[1] - yrange[0] + 1) + 1, dtype=int
        )
    )
    for a, next_start in zip(edges[:-1], edges[1:]):
        b = next_start - 1
        # Purge intervals crossing the period boundary.
        sub = obs[
            (pd.to_datetime(obs.entry) >= pd.Timestamp(a, 1, 1))
            & (pd.to_datetime(obs.exit) <= pd.Timestamp(b, 12, 31))
        ]
        if not sub.empty:
            tab = summarize(sub).reset_index()
            tab.insert(0, "Periode", f"{a}–{b}")
            parts.append(tab)
    stability = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    st.dataframe(stability.round(3))
    with st.expander("Alle einzelnen Beobachtungen"):
        st.dataframe(obs)
    protocol = {
        "version": VERSION,
        "asset": pair,
        "mode": mode,
        "custom": custom,
        "years": yrange,
        "yahoo_proxy": (
            {"symbol": SYMBOLS[pair], "description": ASSETS[pair][2]}
            if source == "Yahoo abrufen"
            else None
        ),
        "source": provenance,
        "quality": public_quality(quality),
        "exploratory": True,
        "PIT_verified": False,
        "multiple_testing_adjusted": False,
        "promotion_authorized": False,
        "return_rule": "previous available close before interval to last inside; weekdays previous-close to close",
        "missing_rule": "no fill; reject explicit missing prices; intervals with calendar gaps>7 days excluded",
        "full_calendar_end_required": True,
        "costs_included": False,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, table in [
            ("prices", p.rename_axis("Date").to_frame()),
            ("observations", obs),
            ("summary", summary),
            ("annual", annual),
            ("stability", stability),
        ]:
            z.writestr(name + ".csv", table.to_csv())
        write_source_audit(z, quality)
        protocol.update(calculation_policy())
        z.writestr("protocol.json", json.dumps(protocol, indent=2, default=str))
    st.download_button(
        "Auswertung als ZIP herunterladen",
        buf.getvalue(),
        file_name=ASSETS[pair][1] + "_Seasonality_v1_7_0.zip",
        mime="application/zip",
    )
    st.info(
        "Ein auffälliges Muster ist zunächst eine Hypothese. Für eine Bestätigung müssen Zeitraum, Richtung und Kriterien vor einer separaten Validierung feststehen. Dieses Dashboard optimiert keine Gewichte."
    )


if __name__ == "__main__":
    main()
