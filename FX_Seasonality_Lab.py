"""Multi-Asset Seasonality Lab 1.5.1.
Start: streamlit run FX_Seasonality_Lab.py
Dependencies: streamlit, pandas, numpy; optional yfinance for online history.

Independent research dashboard.
No engine/shadow imports or modifications.
"""
import io
import json
import zipfile
import hashlib
from datetime import date

import numpy as np
import pandas as pd


VERSION = '1.5.1'

ASSETS = {
    'S&P 500': (
        '^GSPC',
        'SP500',
        'Preisindex; keine Dividendenrendite.'
    ),
    'Nasdaq 100': (
        '^NDX',
        'NASDAQ100',
        'Preisindex; keine Dividendenrendite.'
    ),
    'Gold': (
        'GC=F',
        'GOLD',
        'Yahoo-Gold-Futures als Proxy, kein Spotgold. '
        'Kontraktwechsel können die Saisonalität beeinflussen.'
    ),
    'WTI': (
        'CL=F',
        'WTI',
        'Yahoo-WTI-Futures als Proxy. Kontraktwechsel und negative Preise '
        'sind besonders zu prüfen; keine Rollrendite eines handelbaren Portfolios.'
    ),
    'EUR/USD': (
        'EURUSD=X',
        'EURUSD',
        'USD je EUR; steigender Kurs bedeutet stärkeren Euro.'
    ),
    'USD/CHF': (
        'CHF=X',
        'USDCHF',
        'CHF je USD; steigender Kurs bedeutet stärkeren US-Dollar.'
    ),
    'USD/JPY': (
        'JPY=X',
        'USDJPY',
        'JPY je USD; steigender Kurs bedeutet stärkeren US-Dollar.'
    ),
}

SYMBOLS = {name: cfg[0] for name, cfg in ASSETS.items()}


# -------------------------------------------------------------------
# SAFE RETURN FUNCTIONS
# -------------------------------------------------------------------

def safe_return(start_price, end_price):
    """
    Calculate a simple percentage return only when both endpoints are
    finite and strictly positive.

    Zero and negative prices remain in the raw source history but are
    not used as endpoints for conventional percentage-return arithmetic.
    """
    try:
        a = float(start_price)
        b = float(end_price)
    except (TypeError, ValueError):
        return np.nan

    if (
        not np.isfinite(a)
        or not np.isfinite(b)
        or a <= 0
        or b <= 0
    ):
        return np.nan

    return b / a - 1.0


def safe_pct_change(p):
    """
    Previous-close percentage return without calculating through
    zero or negative prices.

    If either the current or previous observation is <= 0, the return
    is NaN. Raw price observations themselves remain untouched.
    """
    prev = p.shift(1)

    valid = (
        np.isfinite(p)
        & np.isfinite(prev)
        & (p > 0)
        & (prev > 0)
    )

    r = pd.Series(
        np.nan,
        index=p.index,
        dtype=float,
        name='return'
    )

    r.loc[valid] = (
        p.loc[valid] / prev.loc[valid] - 1.0
    )

    return r


# -------------------------------------------------------------------
# FILE LOADING
# -------------------------------------------------------------------

def load_bytes(data, name, pair, member=None):
    if name.lower().endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(data)) as z:

            candidates = [
                n for n in z.namelist()
                if n.endswith('research_timeseries.csv')
            ]

            expected = (
                ASSETS[pair][1] + '_research_timeseries.csv'
            )

            if member is not None:
                candidates = [
                    n for n in z.namelist()
                    if n == member
                    and n.lower().endswith('.csv')
                ]
            else:
                candidates = [
                    n for n in candidates
                    if n.split('/')[-1] == expected
                ]

            if len(candidates) != 1:
                raise ValueError(
                    'Keine eindeutige Preis-CSV für das ausgewählte '
                    'Asset. CSV im Dashboard auswählen oder direkt '
                    'hochladen.'
                )

            info = z.getinfo(candidates[0])

            if info.file_size > 32 * 1024 * 1024:
                raise ValueError(
                    'CSV im ZIP ist größer als 32 MiB.'
                )

            frame = pd.read_csv(
                z.open(candidates[0])
            )

    else:
        frame = pd.read_csv(
            io.BytesIO(data)
        )

    return frame


# -------------------------------------------------------------------
# PRICE PREPARATION
# -------------------------------------------------------------------

def prepare(frame, asset, now=None):
    dc = next(
        (
            c for c in [
                'Date',
                'date',
                'Datetime',
                'provider_timestamp'
            ]
            if c in frame
        ),
        None
    )

    pc = next(
        (
            c for c in [
                'asset_price',
                'Close',
                'close'
            ]
            if c in frame
        ),
        None
    )

    if dc is None or pc is None:
        raise ValueError(
            'Benötigte CSV-Spalten: Date und '
            'asset_price oder Close.'
        )

    # Preserve provider daily date labels;
    # do not shift them through UTC.
    idx = pd.DatetimeIndex([
        pd.Timestamp(v).date()
        for v in frame[dc]
    ])

    if idx.hasnans or idx.has_duplicates:
        raise ValueError(
            'Fehlende oder doppelte Datumswerte: '
            'Quelle zuerst prüfen.'
        )

    p = pd.Series(
        pd.to_numeric(
            frame[pc],
            errors='coerce'
        ).to_numpy(),
        index=idx,
        name='Close'
    ).sort_index()

    now = (
        pd.Timestamp.now(tz='UTC')
        if now is None
        else pd.Timestamp(now)
    )

    if now.tzinfo is None:
        raise ValueError(
            'Zeitstempel benötigt Zeitzone.'
        )

    cutoff = min(
        now.tz_convert(t).date()
        for t in [
            'UTC',
            'Europe/London',
            'America/New_York'
        ]
    )

    keep = (
        (p.index.dayofweek < 5)
        & (p.index < pd.Timestamp(cutoff))
    )

    removed = int((~keep).sum())

    p = p.loc[keep]

    if p.empty:
        raise ValueError(
            'Keine gültige Historie.'
        )

    # Missing / infinite observations remain
    # a hard source-quality error.
    if (~np.isfinite(p)).any():
        raise ValueError(
            'Fehlende oder nicht endliche Preise '
            'in der Historie. Keine automatische '
            'Auffüllung.'
        )

    nonpositive = p <= 0

    # WTI historically traded below zero in April 2020.
    # Those raw observations are preserved.
    #
    # For the other assets, a zero/negative price is
    # treated as a source-quality problem.
    if nonpositive.any() and asset != 'WTI':
        raise ValueError(
            'Null- oder Negativpreise für dieses '
            'Asset gefunden. Quelle zuerst prüfen.'
        )

    quality = {
        'cutoff_exclusive': str(cutoff),
        'removed_rows': removed,
        'rows': len(p),
        'first': str(p.index.min().date()),
        'last': str(p.index.max().date()),
        'largest_calendar_gap_days': (
            int(
                p.index
                .to_series()
                .diff()
                .dt.days
                .max()
            )
            if len(p) > 1
            else 0
        ),
        'nonpositive_rows': int(
            nonpositive.sum()
        ),
        'nonpositive_dates': [
            str(t.date())
            for t in p.index[
                nonpositive.to_numpy()
            ]
        ]
    }

    return p, quality


# -------------------------------------------------------------------
# INTERVAL RETURNS
# -------------------------------------------------------------------

def interval_record(p, start, end, label, year):
    """
    Return from last available close BEFORE start
    to last close ON/BEFORE end.

    Require historical coverage through calendar end
    (conservative on weekends).

    Zero/negative prices are retained in the price history.
    A return is not calculated if its entry or exit price
    is <= 0.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if (
        start > end
        or end > p.index.max()
        or start <= p.index.min()
    ):
        return None

    prior = p.loc[
        p.index < start
    ]

    inside = p.loc[
        (p.index >= start)
        & (p.index <= end)
    ]

    if prior.empty or inside.empty:
        return None

    entry = prior.index[-1]
    exit_ = inside.index[-1]

    # Avoid silently spanning obvious source outages;
    # not a holiday-calendar proof.
    seq = p.loc[
        entry:exit_
    ]

    if (
        (start - entry).days > 7
        or (end - exit_).days > 7
        or (
            seq.index
            .to_series()
            .diff()
            .dt.days
            .max()
            > 7
        )
    ):
        return None

    ret = safe_return(
        prior.iloc[-1],
        inside.iloc[-1]
    )

    if not np.isfinite(ret):
        return None

    r = safe_pct_change(
        seq
    ).dropna()

    return {
        'group': label,
        'year': year,
        'planned_start': str(
            start.date()
        ),
        'planned_end': str(
            end.date()
        ),
        'entry': str(
            entry.date()
        ),
        'exit': str(
            exit_.date()
        ),
        'bars': len(inside),
        'return': float(ret),
        'daily_vol': (
            float(
                r.std(ddof=1)
                * np.sqrt(252)
            )
            if len(r) > 1
            else np.nan
        )
    }


# -------------------------------------------------------------------
# STANDARD CALENDAR OBSERVATIONS
# -------------------------------------------------------------------

def observations(p, mode, custom=None):
    rows = []

    if mode == 'Wochentage':

        r = safe_pct_change(p)

        gaps = (
            p.index
            .to_series()
            .diff()
            .dt.days
        )

        for t, v in r.items():

            if (
                pd.notna(v)
                and gaps.loc[t] <= 7
            ):
                rows.append({
                    'group': [
                        'Mo',
                        'Di',
                        'Mi',
                        'Do',
                        'Fr'
                    ][t.dayofweek],
                    'year': t.year,
                    'entry': str(
                        p.index[
                            p.index.get_loc(t) - 1
                        ].date()
                    ),
                    'exit': str(
                        t.date()
                    ),
                    'bars': 1,
                    'return': float(v)
                })

    else:

        for y in range(
            p.index.min().year,
            p.index.max().year + 1
        ):

            intervals = []

            if mode == 'Monate':

                for m in range(1, 13):
                    a = pd.Timestamp(
                        y,
                        m,
                        1
                    )

                    intervals.append((
                        a,
                        a + pd.offsets.MonthEnd(0),
                        f'{m:02d}'
                    ))

            elif mode == 'Kalenderwochen':

                for w in range(1, 54):
                    try:
                        a = pd.Timestamp(
                            date.fromisocalendar(
                                y,
                                w,
                                1
                            )
                        )
                    except ValueError:
                        continue

                    intervals.append((
                        a,
                        a + pd.Timedelta(days=6),
                        f'KW{w:02d}'
                    ))

            elif mode == 'Freie Datumsspanne':

                sm, sd, em, ed = custom

                try:
                    a = pd.Timestamp(
                        y,
                        sm,
                        sd
                    )

                    b = pd.Timestamp(
                        y + int(
                            (em, ed)
                            < (sm, sd)
                        ),
                        em,
                        ed
                    )

                    intervals = [(
                        a,
                        b,
                        (
                            f'{sd:02d}.{sm:02d}'
                            f'–'
                            f'{ed:02d}.{em:02d}'
                        )
                    )]

                except ValueError:
                    # e.g. 29 February
                    # in a non-leap year
                    continue

            else:

                sw, ew = custom

                try:
                    a = pd.Timestamp(
                        date.fromisocalendar(
                            y,
                            sw,
                            1
                        )
                    )

                    b = pd.Timestamp(
                        date.fromisocalendar(
                            y + int(ew < sw),
                            ew,
                            7
                        )
                    )

                    intervals = [(
                        a,
                        b,
                        f'KW{sw:02d}–KW{ew:02d}'
                    )]

                except ValueError:
                    continue

            for a, b, label in intervals:

                rec = interval_record(
                    p,
                    a,
                    b,
                    label,
                    y
                )

                if rec:
                    rows.append(rec)

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------------

def summarize(obs):
    rows = []

    for key, g in obs.groupby(
        'group',
        sort=True
    ):

        r = g['return']

        rows.append({
            'Gruppe': key,
            'N': len(g),
            'Jahre': g.year.nunique(),
            'Mittel_%': 100 * r.mean(),
            'Median_%': 100 * r.median(),
            'Positiv_%': (
                100 * (r > 0).mean()
            ),
            'Minimum_%': (
                100 * r.min()
            ),
            'Maximum_%': (
                100 * r.max()
            )
        })

    return (
        pd.DataFrame(rows)
        .set_index('Gruppe')
    )


# -------------------------------------------------------------------
# EVENT STUDY
# -------------------------------------------------------------------

def event_observations(
    p,
    events,
    before,
    after
):
    """
    Daily labels only; day0 must exist exactly.
    No inferred release time.

    Returns involving a zero/negative endpoint
    are rejected rather than calculated.
    """
    rows = []
    rejected = []

    for _, e in events.iterrows():

        day = pd.Timestamp(
            e['Date']
        ).normalize()

        kind = str(
            e['Event']
        )

        if day not in p.index:

            rejected.append({
                'Date': str(day.date()),
                'Event': kind,
                'reason': (
                    'Kein Kursdatum; '
                    'keine automatische Verschiebung'
                )
            })

            continue

        i = p.index.get_loc(day)

        if (
            i - before < 0
            or i + after >= len(p)
        ):

            rejected.append({
                'Date': str(day.date()),
                'Event': kind,
                'reason': (
                    'Fenster nicht vollständig'
                )
            })

            continue

        path = p.iloc[
            i - before:
            i + after + 1
        ]

        if (
            path.index
            .to_series()
            .diff()
            .dt.days
            .max()
            > 7
        ):

            rejected.append({
                'Date': str(day.date()),
                'Event': kind,
                'reason': (
                    'Quellenlücke >7 Kalendertage'
                )
            })

            continue

        legs = [
            (
                'Vorher',
                i - before,
                i - 1
            ),
            (
                'Ereignistag',
                i - 1,
                i
            ),
            (
                'Nachher',
                i,
                i + after
            ),
            (
                'Gesamt',
                i - before,
                i + after
            )
        ]

        calculated = []
        invalid = []

        for label, a, b in legs:

            ret = safe_return(
                p.iloc[a],
                p.iloc[b]
            )

            if not np.isfinite(ret):
                invalid.append(label)
                continue

            calculated.append({
                'Date': str(day.date()),
                'Event': kind,
                'group': (
                    kind
                    + ' | '
                    + label
                ),
                'window': label,
                'year': day.year,
                'entry': str(
                    p.index[a].date()
                ),
                'exit': str(
                    p.index[b].date()
                ),
                'bars': b - a,
                'return': float(ret),
                'full_start': str(
                    path.index[0].date()
                ),
                'full_end': str(
                    path.index[-1].date()
                )
            })

        # Conservative handling:
        # if any event sub-window has an invalid
        # zero/negative endpoint, reject the
        # complete event observation.
        if invalid:

            rejected.append({
                'Date': str(day.date()),
                'Event': kind,
                'reason': (
                    'Null-/Negativpreis an '
                    'Renditegrenze: '
                    + ', '.join(invalid)
                )
            })

            continue

        rows.extend(
            calculated
        )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(rejected)
    )


def event_dashboard(
    st,
    p,
    asset,
    provenance,
    quality,
    date_range=None
):
    st.subheader(
        'Event-Studie'
    )

    st.warning(
        'Tagesdaten trennen keine unmittelbare Reaktion '
        'auf eine Veröffentlichung. Tag0 ist das '
        'Anbieter-Datumslabel, nicht eine verifizierte '
        'Veröffentlichungssitzung. Keine Kausalitäts- '
        'oder Intraday-Aussage.'
    )

    source = st.radio(
        'Event-Termine',
        [
            'Offizielle / eigene Termine als CSV',
            'Quartalsverfall: Kalenderregel '
            '(ungeprüfte Näherung)'
        ]
    )

    if source.startswith(
        'Quartalsverfall'
    ):

        st.warning(
            'Erzeugt den dritten Freitag im März, Juni, '
            'September und Dezember. Feiertagsverschiebungen '
            'und produkt-/börsenspezifische Abrechnung fehlen. '
            'Für echte Hexensabbat-Termine einen geprüften '
            'Kalender als CSV verwenden. Kein Verfallskalender '
            'für Gold-, Öl- oder FX-Kontrakte.'
        )

        rows = []

        for y in range(
            p.index.min().year,
            p.index.max().year + 1
        ):

            for m in [
                3,
                6,
                9,
                12
            ]:

                first = pd.Timestamp(
                    y,
                    m,
                    1
                )

                day = (
                    first
                    + pd.Timedelta(
                        days=(
                            (4 - first.dayofweek) % 7
                            + 14
                        )
                    )
                )

                rows.append({
                    'Date': str(
                        day.date()
                    ),
                    'Event': (
                        'Quartalsverfall-Regel '
                        'ungeprüft'
                    ),
                    'Source': (
                        'third-Friday calendar rule'
                    )
                })

        events = pd.DataFrame(rows)

        event_source = {
            'type': 'calendar_rule',
            'verified': False
        }

    else:

        st.markdown(
            'Terminreferenzen: '
            '[Fed-Entscheidungen]'
            '(https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)'
            ' · '
            '[US-CPI]'
            '(https://www.bls.gov/schedule/news_release/cpi.htm)'
            ' · '
            '[CPI-Archiv]'
            '(https://www.bls.gov/bls/news-release/cpi.htm)'
            ' · '
            '[Cboe-Verfallskalender]'
            '(https://www.cboe.com/about/hours/us-options/)'
        )

        st.write(
            'CSV-Spalten: Date, Event, optional Source. '
            'Date im Format JJJJ-MM-TT; Event z. B. '
            'FOMC, CPI oder Hexensabbat. Bei FOMC den '
            'Entscheidungstag verwenden, bei CPI den '
            'Veröffentlichungstag – nicht den Berichtsmonat. '
            'Uhrzeitinformationen werden in dieser '
            'Tagesstudie nicht ausgewertet.'
        )

        st.download_button(
            'Leere Event-CSV-Vorlage',
            'Date,Event,Source\n',
            file_name='event_calendar_template.csv',
            mime='text/csv'
        )

        upload = st.file_uploader(
            'Event-Kalender hochladen',
            type=['csv'],
            key='event_csv'
        )

        if upload is None:
            return

        raw = upload.getvalue()

        try:

            events = pd.read_csv(
                io.BytesIO(raw)
            )

            if not {
                'Date',
                'Event'
            }.issubset(events):
                raise ValueError(
                    'Date und Event fehlen.'
                )

            if not (
                events.Date
                .astype(str)
                .str.fullmatch(
                    r'\d{4}-\d{2}-\d{2}'
                )
                .all()
            ):
                raise ValueError(
                    'Date muss JJJJ-MM-TT sein, '
                    'ohne Uhrzeit.'
                )

            events['Date'] = pd.to_datetime(
                events.Date,
                format='%Y-%m-%d',
                errors='raise'
            )

            if (
                events.Event.isna().any()
                or (
                    events.Event
                    .astype(str)
                    .str.strip()
                    .eq('')
                    .any()
                )
            ):
                raise ValueError(
                    'Event-Bezeichnung fehlt.'
                )

            events['Event'] = (
                events.Event
                .astype(str)
                .str.strip()
            )

            if events.duplicated(
                ['Date', 'Event']
            ).any():
                raise ValueError(
                    'Doppelte Date/Event-Kombinationen '
                    'entfernen.'
                )

        except Exception as e:
            st.error(
                str(e)
            )
            return

        event_source = {
            'type': 'uploaded_calendar',
            'filename': upload.name,
            'sha256': hashlib.sha256(
                raw
            ).hexdigest(),
            'verified': False
        }

    if date_range is not None:

        events = events[
            pd.to_datetime(
                events.Date
            ).between(
                *date_range
            )
        ].copy()

        st.caption(
            'Nur Ereignistage innerhalb des gewählten '
            'Von–Bis-Zeitraums; Vor-/Nachfenster dürfen '
            'darüber hinausreichen.'
        )

    if events.empty:
        st.info(
            'Keine Ereignisse im gewählten Zeitraum.'
        )
        return

    kinds = st.multiselect(
        'Ereignisse',
        sorted(
            events.Event.unique()
        ),
        default=sorted(
            events.Event.unique()
        )
    )

    events = events[
        events.Event.isin(
            kinds
        )
    ].copy()

    before = int(
        st.number_input(
            'Beginn: Handelstage vor Tag0',
            min_value=2,
            max_value=60,
            value=5
        )
    )

    after = int(
        st.number_input(
            'Ende: Handelstage nach Tag0',
            min_value=1,
            max_value=60,
            value=5
        )
    )

    st.caption(
        f'Vorher: Schluss T−{before} bis T−1; '
        f'Ereignistag: T−1 bis T0; '
        f'Nachher: T0 bis T+{after}. '
        'Die Segmente sind getrennt; „Gesamt“ umfasst '
        'alle. Ein Handelstag ist hier eine verfügbare '
        'Kurszeile.'
    )

    obs, rejected = event_observations(
        p,
        events,
        before,
        after
    )

    if obs.empty:
        st.warning(
            'Keine vollständigen Event-Fenster.'
        )
        st.dataframe(
            rejected
        )
        return

    windows = (
        obs[
            [
                'Date',
                'Event',
                'full_start',
                'full_end'
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                'full_start',
                'full_end'
            ]
        )
    )

    overlap = []
    prior_end = None

    for _, row in windows.iterrows():

        overlap.append(
            prior_end is not None
            and row.full_start <= prior_end
        )

        prior_end = max(
            prior_end or row.full_end,
            row.full_end
        )

    windows[
        'overlaps_prior_window'
    ] = overlap

    st.caption(
        f'{len(windows)} vollständige Ereignisse; '
        f'{len(rejected)} ausgeschlossen; '
        f'{sum(overlap)} Fenster überschneiden ein '
        'vorheriges. Ereignisse können zugleich auftreten '
        'und sind nicht unabhängig.'
    )

    summary = direction_summary(
        obs
    )

    st.dataframe(
        summary.round(3)
    )

    st.bar_chart(
        summary[
            [
                'Mittel_%',
                'Median_%'
            ]
        ]
    )

    annual = (
        obs.pivot_table(
            index='year',
            columns='group',
            values='return',
            aggfunc='mean'
        )
        * 100
    )

    st.subheader(
        'Jahresmittel in %'
    )

    st.dataframe(
        annual.round(3)
    )

    with st.expander(
        'Einzelereignisse, Überschneidungen '
        'und Ausschlüsse'
    ):
        st.dataframe(obs)
        st.dataframe(windows)
        st.dataframe(rejected)

    protocol = {
        'version': VERSION,
        'asset': asset,
        'analysis': 'event study',
        'before': before,
        'after': after,
        'price_source': provenance,
        'quality': quality,
        'event_source': event_source,
        'selected_events': kinds,
        'selected_date_range': date_range,
        'date_alignment': (
            'exact provider date; '
            'absent dates rejected'
        ),
        'timing_verified': False,
        'exploratory': True,
        'multiple_testing_adjusted': False,
        'costs_included': False,
        'promotion_authorized': False,
        'overlap_rule': (
            'inclusive full window overlap flagged; '
            'observations retained'
        ),
        'nonpositive_price_rule': (
            'raw prices retained; returns with '
            'zero/negative endpoints rejected'
        ),
        'interpretation': (
            'calendar association, not causal '
            'release surprise or executable strategy'
        )
    }

    buf = io.BytesIO()

    with zipfile.ZipFile(
        buf,
        'w',
        zipfile.ZIP_DEFLATED
    ) as z:

        for name, df in [
            ('prices', p.to_frame()),
            ('events', events),
            ('observations', obs),
            ('summary', summary),
            ('annual', annual),
            ('overlap', windows),
            ('excluded', rejected)
        ]:
            z.writestr(
                name + '.csv',
                df.to_csv()
            )

        z.writestr(
            'protocol.json',
            json.dumps(
                protocol,
                indent=2,
                default=str
            )
        )

    st.download_button(
        'Event-Auswertung als ZIP',
        buf.getvalue(),
        file_name=(
            ASSETS[asset][1]
            + '_Events_v1_5_1.zip'
        ),
        mime='application/zip'
    )


# -------------------------------------------------------------------
# DIRECTION SUMMARY
# -------------------------------------------------------------------

def direction_summary(obs):

    tab = summarize(obs)

    grouped = (
        obs.groupby(
            'group'
        )['return']
    )

    tab['Negativ_%'] = grouped.apply(
        lambda r:
        100 * (r < 0).mean()
    )

    tab['Unverändert_%'] = grouped.apply(
        lambda r:
        100 * (r == 0).mean()
    )

    def label(r):

        if (
            r['Mittel_%'] > 0
            and r['Median_%'] > 0
            and r['Positiv_%'] > 50
        ):
            return 'Historisch eher Long'

        if (
            r['Mittel_%'] < 0
            and r['Median_%'] < 0
            and r['Negativ_%'] > 50
        ):
            return 'Historisch eher Short'

        return (
            'Gemischt / keine eindeutige Richtung'
        )

    tab['Beschreibung'] = tab.apply(
        label,
        axis=1
    )

    return tab


# -------------------------------------------------------------------
# SELECTED PERIOD
# -------------------------------------------------------------------

def selected_days(
    p,
    start,
    end
):
    r = safe_pct_change(p)

    gaps = (
        p.index
        .to_series()
        .diff()
        .dt.days
    )

    rows = []

    for t in p.loc[
        pd.Timestamp(start):
        pd.Timestamp(end)
    ].index:

        if (
            pd.notna(r.loc[t])
            and gaps.loc[t] <= 7
        ):

            i = p.index.get_loc(t)

            rows.append({
                'group': 'Ausgewählte Tage',
                'year': t.year,
                'entry': str(
                    p.index[i - 1].date()
                ),
                'exit': str(
                    t.date()
                ),
                'bars': 1,
                'return': float(
                    r.loc[t]
                )
            })

    return pd.DataFrame(rows)


def period_dashboard(
    st,
    p,
    asset,
    provenance,
    quality
):
    st.subheader(
        'Dein Zeitraum und historische Richtung'
    )

    a, b = st.columns(2)

    start = a.date_input(
        'Von',
        value=max(
            p.index.min().date(),
            (
                p.index.max()
                - pd.Timedelta(days=90)
            ).date()
        ),
        min_value=p.index.min().date(),
        max_value=p.index.max().date(),
        key='period_start_' + asset
    )

    end = b.date_input(
        'Bis einschließlich',
        value=p.index.max().date(),
        min_value=p.index.min().date(),
        max_value=p.index.max().date(),
        key='period_end_' + asset
    )

    if start > end:
        st.error(
            'Von muss vor oder auf Bis liegen.'
        )
        return

    mode = st.radio(
        'Was möchtest du untersuchen?',
        [
            'Kursverlauf in genau diesem Zeitraum',
            'Diese Kalenderspanne in früheren Jahren',
            'Ereignisse in diesem Zeitraum'
        ]
    )

    st.caption(
        'Long/Short beschreibt historische Kursbewegungen '
        'vor Kosten. Eine Mehrheit positiver Tage ist keine '
        'Prognose und keine Handelsfreigabe.'
    )

    if mode == 'Ereignisse in diesem Zeitraum':

        event_dashboard(
            st,
            p,
            asset,
            provenance,
            quality,
            (
                pd.Timestamp(start),
                pd.Timestamp(end)
            )
        )

        return

    if mode == (
        'Diese Kalenderspanne in früheren Jahren'
    ):

        st.caption(
            'Monat und Tag von Von/Bis bilden das jährlich '
            'wiederholte Fenster. Die Jahreszahlen wählst du '
            'darunter separat. Ein Ende vor dem '
            'Startmonat/-tag bedeutet Jahreswechsel.'
        )

        lo = p.index.min().year
        hi = p.index.max().year

        years = (
            st.slider(
                'Historische Vergleichsjahre '
                '(Startjahr)',
                lo,
                hi,
                (lo, hi)
            )
            if lo < hi
            else (lo, hi)
        )

        obs = observations(
            p,
            'Freie Datumsspanne',
            (
                start.month,
                start.day,
                end.month,
                end.day
            )
        )

        if not obs.empty:
            obs = obs[
                obs.year.between(
                    *years
                )
            ]

        if obs.empty:
            st.warning(
                'Keine vollständig abgedeckten '
                'Jahresfenster.'
            )
            return

        st.caption(
            'Eine Beobachtung je vollständigem '
            'Jahresfenster. Einstieg: Schlusskurs vor '
            'Beginn; Ausstieg: letzter Schlusskurs '
            'innerhalb des Fensters.'
        )

    else:

        obs = selected_days(
            p,
            start,
            end
        )

        if obs.empty:
            st.warning(
                'Keine auswertbaren Tagesrenditen.'
            )
            return

        total = interval_record(
            p,
            start,
            end,
            'Gesamter Zeitraum',
            start.year
        )

        if total:

            st.metric(
                'Kursveränderung im gesamten Zeitraum',
                f"{100 * total['return']:+.2f} %"
            )

            st.caption(
                f"Vom Schlusskurs {total['entry']} "
                f"bis Schlusskurs {total['exit']}."
            )

        else:
            st.warning(
                'Gesamtrendite wegen fehlendem Vortag, '
                'Quellenlücken oder Null-/Negativpreis '
                'an einer Renditegrenze nicht ausgewiesen.'
            )

        st.line_chart(
            p.loc[
                str(start):
                str(end)
            ].to_frame()
        )

        st.caption(
            'Tagesrenditen: vorheriger verfügbarer '
            'Schlusskurs bis Tagesschluss. Fehlender '
            'Vortag und Lücken über sieben Kalendertage '
            'werden ausgeschlossen. Renditen mit '
            'Null-/Negativpreis als Start- oder Endwert '
            'werden nicht berechnet; keine '
            'Intraday-Renditen.'
        )

    summary = direction_summary(
        obs
    )

    st.dataframe(
        summary.round(3)
    )

    st.caption(
        '„Eher Long“: Mittel und Median positiv sowie '
        'mehr als 50 % positive Beobachtungen. '
        '„Eher Short“ entsprechend negativ. '
        'Sonst gemischt. Diese Beschreibungsregel '
        'ist kein statistischer Bestätigungstest.'
    )

    detail = obs.copy()

    detail['Rendite_%'] = (
        100 * detail['return']
    )

    detail['Richtung'] = np.where(
        detail['return'] > 0,
        'Long',
        np.where(
            detail['return'] < 0,
            'Short',
            'Unverändert'
        )
    )

    st.subheader(
        'Einzelne Tage / Jahresfenster'
    )

    st.dataframe(
        detail.drop(
            columns=['return']
        )
    )

    protocol = {
        'version': VERSION,
        'asset': asset,
        'mode': mode,
        'from': str(start),
        'to': str(end),
        'comparison_years': (
            list(years)
            if mode == (
                'Diese Kalenderspanne '
                'in früheren Jahren'
            )
            else None
        ),
        'source': provenance,
        'quality': quality,
        'exploratory': True,
        'costs_included': False,
        'promotion_authorized': False,
        'multiple_testing_adjusted': False,
        'nonpositive_price_rule': (
            'raw prices retained; returns with '
            'zero/negative endpoints rejected'
        ),
        'direction_rule': (
            'mean and median same sign and >50% '
            'observations of that sign; otherwise mixed'
        )
    }

    buf = io.BytesIO()

    with zipfile.ZipFile(
        buf,
        'w',
        zipfile.ZIP_DEFLATED
    ) as z:

        for name, df in [
            ('prices', p.to_frame()),
            ('observations', detail),
            ('summary', summary)
        ]:

            z.writestr(
                name + '.csv',
                df.to_csv()
            )

        z.writestr(
            'protocol.json',
            json.dumps(
                protocol,
                indent=2,
                default=str
            )
        )

    st.download_button(
        'Meine Auswertung herunterladen',
        buf.getvalue(),
        file_name=(
            ASSETS[asset][1]
            + '_Zeitraum_v1_5_1.zip'
        ),
        mime='application/zip'
    )


# -------------------------------------------------------------------
# SEASONAL PATHS
# -------------------------------------------------------------------

def seasonal_paths(
    p,
    start_md,
    end_md,
    years,
    trend='Alle'
):
    """
    Equal-weight yearly windows on observed trading-row
    offsets; no fill.

    Raw prices remain intact.

    Percentage path values are NaN on dates where the
    raw price is zero or negative.
    """
    paths = {}
    rows = []
    excluded = []

    ma = p.rolling(
        200,
        min_periods=200
    ).mean()

    for y in years:

        try:

            start = pd.Timestamp(
                y,
                *start_md
            )

            end = pd.Timestamp(
                y + int(
                    end_md < start_md
                ),
                *end_md
            )

        except ValueError:

            excluded.append({
                'year': y,
                'reason': (
                    'Datum existiert nicht '
                    'in diesem Jahr'
                )
            })

            continue

        rec = interval_record(
            p,
            start,
            end,
            'Saisonales Fenster',
            y
        )

        if rec is None:

            excluded.append({
                'year': y,
                'reason': (
                    'Fenster unvollständig, '
                    'Quellenlücke oder '
                    'ungültige Renditegrenze'
                )
            })

            continue

        entry = pd.Timestamp(
            rec['entry']
        )

        state = (
            'Unbekannt'
            if pd.isna(
                ma.loc[entry]
            )
            else (
                'Oberhalb MA200'
                if (
                    p.loc[entry]
                    > ma.loc[entry]
                )
                else (
                    'Unterhalb/gleich MA200'
                )
            )
        )

        if (
            trend != 'Alle'
            and state != trend
        ):

            excluded.append({
                'year': y,
                'reason': (
                    'Trendfilter: '
                    + state
                )
            })

            continue

        seq = p.loc[
            rec['entry']:
            rec['exit']
        ]

        # Raw price values are never changed.
        # Only the percentage-path representation
        # suppresses zero/negative observations.
        safe_seq = seq.where(
            seq > 0
        )

        if not np.isfinite(
            safe_seq.iloc[0]
        ):

            excluded.append({
                'year': y,
                'reason': (
                    'Null-/Negativpreis '
                    'am Einstieg'
                )
            })

            continue

        path = (
            safe_seq
            / safe_seq.iloc[0]
            - 1.0
        )

        drawdown = (
            safe_seq
            / safe_seq.cummax()
            - 1.0
        )

        paths[str(y)] = pd.Series(
            path.to_numpy(),
            index=range(
                len(path)
            )
        )

        rec.update({
            'trend_at_entry': state,
            'MAE_Close_Long': float(
                path.min()
            ),
            'MFE_Close_Long': float(
                path.max()
            ),
            'Max_Drawdown_Close': float(
                drawdown.min()
            )
        })

        rows.append(
            rec
        )

    curves = pd.DataFrame(
        paths
    )

    curves.index.name = (
        'Handelstage seit Einstieg'
    )

    return (
        pd.DataFrame(rows),
        curves,
        pd.DataFrame(excluded)
    )


def path_statistics(curves):

    return pd.DataFrame({
        'Mittel': (
            curves.mean(axis=1)
            * 100
        ),
        'Median': (
            curves.median(axis=1)
            * 100
        ),
        'P10': (
            curves.quantile(
                .10,
                axis=1
            )
            * 100
        ),
        'P90': (
            curves.quantile(
                .90,
                axis=1
            )
            * 100
        ),
        'N': curves.count(
            axis=1
        )
    })


# -------------------------------------------------------------------
# CONTROL WINDOWS
# -------------------------------------------------------------------

def control_windows(
    p,
    obs,
    excluded_windows,
    trend='Alle'
):
    """
    Fixed disjoint return blocks anchored to each
    calendar year; no best selection.

    Block length equals that year's seasonal window.

    Drop blocks overlapping any seasonal window's
    return dates. Shared boundary closes do not
    share returns.

    Zero/negative endpoint prices invalidate a
    control return rather than being transformed.
    """
    rows = []

    ma = p.rolling(
        200,
        min_periods=200
    ).mean()

    blocked = np.zeros(
        len(p),
        dtype=bool
    )

    for _, r in excluded_windows.iterrows():

        blocked |= (
            (p.index > pd.Timestamp(r.entry))
            & (
                p.index
                <= pd.Timestamp(r.exit)
            )
        )

    for _, r in obs.iterrows():

        n = int(
            r.bars
        )

        idx = np.flatnonzero(
            p.index.year
            == int(r.year)
        )

        if n < 1:
            continue

        for offset in range(
            0,
            len(idx) - n + 1,
            n
        ):

            first = int(
                idx[offset]
            )

            last = int(
                idx[offset + n - 1]
            )

            entry = first - 1

            if entry < 0:
                continue

            if blocked[
                first:last + 1
            ].any():
                continue

            seq = p.iloc[
                entry:last + 1
            ]

            if (
                seq.index
                .to_series()
                .diff()
                .dt.days
                .max()
                > 7
            ):
                continue

            state = (
                'Unbekannt'
                if pd.isna(
                    ma.iloc[entry]
                )
                else (
                    'Oberhalb MA200'
                    if (
                        p.iloc[entry]
                        > ma.iloc[entry]
                    )
                    else (
                        'Unterhalb/gleich MA200'
                    )
                )
            )

            if (
                trend != 'Alle'
                and state != trend
            ):
                continue

            ret = safe_return(
                seq.iloc[0],
                seq.iloc[-1]
            )

            if not np.isfinite(ret):
                continue

            rows.append({
                'year': int(
                    r.year
                ),
                'entry': str(
                    p.index[entry].date()
                ),
                'exit': str(
                    p.index[last].date()
                ),
                'bars': n,
                'return': float(ret),
                'trend_at_entry': state
            })

    return pd.DataFrame(
        rows,
        columns=[
            'year',
            'entry',
            'exit',
            'bars',
            'return',
            'trend_at_entry'
        ]
    )


def paired_controls(
    obs,
    controls
):
    if controls.empty:
        return pd.DataFrame()

    base = (
        controls.groupby(
            'year'
        )['return']
        .agg(
            [
                'mean',
                'count'
            ]
        )
        .rename(
            columns={
                'mean': 'control_mean',
                'count': 'control_N'
            }
        )
    )

    paired = (
        obs[
            [
                'year',
                'return'
            ]
        ]
        .merge(
            base,
            on='year',
            how='inner'
        )
    )

    paired['difference'] = (
        paired['return']
        - paired['control_mean']
    )

    return paired


def temporal_split(
    obs,
    cutoff
):
    """
    Purge windows sharing any time across
    the split boundary.
    """
    if obs.empty:
        return (
            obs.copy(),
            obs.copy(),
            obs.copy()
        )

    cut = pd.Timestamp(
        cutoff
    )

    left = obs[
        pd.to_datetime(
            obs.exit
        ) < cut
    ].copy()

    right = obs[
        pd.to_datetime(
            obs.entry
        ) >= cut
    ].copy()

    purge = obs[
        ~obs.index.isin(
            left.index.union(
                right.index
            )
        )
    ].copy()

    return (
        left,
        right,
        purge
    )


def comparison_row(
    label,
    seasonal,
    controls
):
    pairs = paired_controls(
        seasonal,
        controls
    )

    return {
        'Bereich': label,
        'Saison_N': len(
            seasonal
        ),
        'Jahre_mit_Kontrolle': len(
            pairs
        ),
        'Saison_Mittel_%': (
            100
            * seasonal['return'].mean()
            if len(seasonal)
            else np.nan
        ),
        'Saison_positiv_%': (
            100
            * (
                seasonal['return'] > 0
            ).mean()
            if len(seasonal)
            else np.nan
        ),
        'Saison_Mittel_Common_%': (
            100
            * pairs['return'].mean()
            if len(pairs)
            else np.nan
        ),
        'Kontrolle_Mittel_Common_%': (
            100
            * pairs.control_mean.mean()
            if len(pairs)
            else np.nan
        ),
        'Differenz_Common_Prozentpunkte': (
            100
            * pairs.difference.mean()
            if len(pairs)
            else np.nan
        )
    }


# -------------------------------------------------------------------
# SEASONAL DASHBOARD
# -------------------------------------------------------------------

def seasonal_dashboard(
    st,
    p,
    asset,
    provenance,
    quality
):
    st.subheader(
        'Saisonale Kurven und Jahresvergleich'
    )

    st.write(
        'Wähle ein jährlich wiederkehrendes Fenster '
        'und die Jahre, über die du es vergleichen '
        'möchtest.'
    )

    a, b = st.columns(2)

    start = a.date_input(
        'Fenster von (nur Monat/Tag)',
        date(
            2000,
            1,
            1
        ),
        key='curve_start'
    )

    end = b.date_input(
        'Fenster bis (nur Monat/Tag)',
        date(
            2000,
            12,
            31
        ),
        key='curve_end'
    )

    st.caption(
        'Liegt der Endmonat/-tag vor dem Beginn, '
        'läuft das Fenster ins Folgejahr. '
        '29. Februar wird nur in passenden '
        'Schaltjahren ausgewertet.'
    )

    lo = p.index.min().year
    hi = p.index.max().year

    bounds = (
        st.slider(
            'Historie: Startjahre',
            lo,
            hi,
            (lo, hi),
            key='curve_years'
        )
        if lo < hi
        else (lo, hi)
    )

    trend = st.selectbox(
        'Marktumfeld beim Einstieg',
        [
            'Alle',
            'Oberhalb MA200',
            'Unterhalb/gleich MA200'
        ]
    )

    st.caption(
        'MA200 verwendet ausschließlich die 200 '
        'verfügbaren Schlusskurse bis zum Einstiegsschluss. '
        'Die Rohpreisreihe bleibt unverändert. Das ist ein '
        'beschreibender Kontextfilter, keine Zusicherung '
        'einer Ausführung zu diesem Schlusskurs.'
    )

    allobs, _, base_excluded = seasonal_paths(
        p,
        (
            start.month,
            start.day
        ),
        (
            end.month,
            end.day
        ),
        range(
            bounds[0],
            bounds[1] + 1
        )
    )

    obs, curves, excluded = seasonal_paths(
        p,
        (
            start.month,
            start.day
        ),
        (
            end.month,
            end.day
        ),
        range(
            bounds[0],
            bounds[1] + 1
        ),
        trend
    )

    st.caption(
        f'{bounds[1] - bounds[0] + 1} '
        f'angefragte Startjahre → '
        f'{len(allobs)} vollständig abgedeckte '
        f'Fenster → {len(obs)} nach Kontextfilter.'
    )

    if obs.empty:

        st.warning(
            'Keine vollständigen Fenster '
            'für diese Auswahl.'
        )

        st.dataframe(
            excluded
        )

        return

    st.caption(
        'Einstieg = letzter Schlusskurs vor dem '
        'Startdatum; Ausstieg = letzter Schlusskurs '
        'innerhalb des Fensters. Jedes Jahr erhält '
        'dasselbe Gewicht. Es werden keine Kurse '
        'aufgefüllt. Null-/Negativpreise bleiben in '
        'der Rohhistorie erhalten, werden aber in '
        'prozentualen Verlaufskurven an der jeweiligen '
        'Stelle nicht berechnet.'
    )

    display = st.radio(
        'Kurvenlänge',
        [
            'Gemeinsame Länge aller Jahre',
            'Alle verfügbaren Handelstage'
        ],
        horizontal=True
    )

    if display.startswith(
        'Gemeinsame'
    ):
        shown = curves.iloc[
            :int(
                curves.count().min()
            )
        ]
    else:
        shown = curves

    stats = path_statistics(
        shown
    )

    graph = (
        stats
        .reset_index()
        .rename(
            columns={
                'Handelstage seit Einstieg':
                'Tag'
            }
        )
    )

    st.vega_lite_chart(
        graph,
        {
            'layer': [
                {
                    'mark': {
                        'type': 'area',
                        'opacity': .15,
                        'color': '#467ab8'
                    },
                    'encoding': {
                        'x': {
                            'field': 'Tag',
                            'type': 'quantitative'
                        },
                        'y': {
                            'field': 'P10',
                            'type': 'quantitative',
                            'title': (
                                'Kursveränderung (%)'
                            )
                        },
                        'y2': {
                            'field': 'P90'
                        }
                    }
                },
                {
                    'mark': {
                        'type': 'line',
                        'color': '#167c80'
                    },
                    'encoding': {
                        'x': {
                            'field': 'Tag',
                            'type': 'quantitative'
                        },
                        'y': {
                            'field': 'Mittel',
                            'type': 'quantitative'
                        },
                        'tooltip': [
                            {'field': 'Tag'},
                            {'field': 'Mittel'},
                            {'field': 'Median'},
                            {'field': 'N'}
                        ]
                    }
                },
                {
                    'mark': {
                        'type': 'line',
                        'color': '#e69b38',
                        'strokeDash': [
                            5,
                            3
                        ]
                    },
                    'encoding': {
                        'x': {
                            'field': 'Tag',
                            'type': 'quantitative'
                        },
                        'y': {
                            'field': 'Median',
                            'type': 'quantitative'
                        }
                    }
                }
            ]
        },
        use_container_width=True
    )

    st.caption(
        'Türkis: Mittel · Orange: Median · '
        'Band: historische 10.–90. Perzentile, '
        'kein Konfidenz- oder Prognoseintervall. '
        'X-Achse = verfügbare Handelstage ab Einstieg, '
        'keine identischen Kalenderdaten. Bei gemeinsamer '
        'Länge endet die Grafik am kürzesten Jahresfenster; '
        'die folgende Ergebnistabelle nutzt stets die '
        'vollständigen Fenster.'
    )

    st.line_chart(
        stats[
            ['N']
        ]
    )

    show_years = st.multiselect(
        'Einzeljahre einblenden',
        list(
            curves.columns
        ),
        default=list(
            curves.columns
        )
    )

    if show_years:

        st.line_chart(
            shown[
                show_years
            ] * 100
        )

    summary = direction_summary(
        obs
    )

    pos = int(
        (
            obs['return'] > 0
        ).sum()
    )

    neg = int(
        (
            obs['return'] < 0
        ).sum()
    )

    st.write(
        f'{pos} von {len(obs)} Jahresfenstern '
        f'gestiegen; {neg} gefallen; '
        f'{len(obs) - pos - neg} unverändert.'
    )

    st.dataframe(
        summary.round(3)
    )

    st.caption(
        'Eine hohe Trefferquote allein belegt keinen '
        'saisonalen Vorteil. Der folgende Kontrollvergleich '
        'ist beschreibend, kein Signifikanztest.'
    )

    controls = control_windows(
        p,
        obs,
        allobs,
        trend
    )

    paired = paired_controls(
        obs,
        controls
    )

    comparison = pd.DataFrame([
        comparison_row(
            'Gesamte Auswahl',
            obs,
            controls
        )
    ])

    st.subheader(
        'Ist das Fenster auffälliger als '
        'andere Zeiträume?'
    )

    st.dataframe(
        comparison.round(3)
    )

    st.caption(
        'Kontrollen: gleich lange Blöcke aus dem '
        'jeweiligen Startjahr, fest ab dem ersten '
        'verfügbaren Handelstag des Jahres eingeteilt. '
        'Keine gemeinsamen Renditetage zwischen '
        'Kontrollblöcken; saisonale Fenster sind '
        'ausgeschlossen. Gleicher MA200-Filter am '
        'jeweiligen Einstieg. Erst je Jahr mitteln, '
        'dann Jahre gleich gewichten. Der Common-Vergleich '
        'nutzt nur Jahre mit beiden Beobachtungen.'
    )

    st.caption(
        'Keine Anpassung an Volatilität, Wochentage '
        'oder Nachrichten. Kontrollen können zeitlich '
        'abhängig sein. Bei langen Fenstern fehlen '
        'häufig passende Blöcke; fehlende Kontrollen '
        'werden nicht ersetzt. Für jahresübergreifende '
        'Saisonfenster liegen Kontrollen nur im Startjahr.'
    )

    if paired.empty:

        st.info(
            'Für diese Auswahl sind keine '
            'Kontrollfenster verfügbar. Keine Aussage '
            'über einen Vorteil gegenüber der '
            'Vergleichsbasis.'
        )

    st.subheader(
        'Zeitlich getrennte historische Prüfung'
    )

    st.warning(
        'Diese Aufteilung ist eine historische Diagnose. '
        'Bereits betrachtete Daten werden dadurch nicht '
        'zu einem unberührten Holdout. Kein '
        'bestanden/nicht bestanden und keine automatische '
        'Promotion.'
    )

    split_year = int(
        st.number_input(
            'Prüfzeitraum beginnt am '
            '1. Januar des Jahres',
            min_value=int(
                bounds[0]
            ),
            max_value=int(
                bounds[1] + 1
            ),
            value=int(
                max(
                    bounds[0],
                    bounds[1] - 4
                )
            ),
            key='seasonal_split_year'
        )
    )

    cutoff = pd.Timestamp(
        split_year,
        1,
        1
    )

    discovery, validation, purged = temporal_split(
        obs,
        cutoff
    )

    dc, vc, pc = temporal_split(
        controls,
        cutoff
    )

    split_summary = pd.DataFrame([
        comparison_row(
            'Früherer Zeitraum',
            discovery,
            dc
        ),
        comparison_row(
            'Späterer Prüfzeitraum',
            validation,
            vc
        )
    ])

    st.dataframe(
        split_summary.round(3)
    )

    st.caption(
        f'{len(purged)} saisonale Fenster und '
        f'{len(pc)} Kontrollfenster an der '
        'Trennlinie ausgeschlossen. Fenster im '
        'Prüfbereich müssen vollständig ab der Grenze '
        'liegen, einschließlich Einstiegsschluss.'
    )

    with st.expander(
        'Kontrollfenster und gepaarte '
        'Jahresvergleiche'
    ):

        st.dataframe(
            controls
        )

        st.dataframe(
            paired
        )

        st.dataframe(
            purged
        )

    r = obs['return']

    risk = pd.DataFrame([{
        'Gewinnmittel_%': (
            100
            * r[
                r > 0
            ].mean()
        ),
        'Verlustmittel_%': (
            100
            * r[
                r < 0
            ].mean()
        ),
        'Median_MAE_Close_Long_%': (
            100
            * obs.MAE_Close_Long.median()
        ),
        'Median_MFE_Close_Long_%': (
            100
            * obs.MFE_Close_Long.median()
        ),
        'Schlechtester_Close_Drawdown_%': (
            100
            * obs.Max_Drawdown_Close.min()
        )
    }])

    st.subheader(
        'Gewinne, Verluste und Bewegung '
        'innerhalb des Fensters'
    )

    st.dataframe(
        risk.round(3)
    )

    st.caption(
        'MAE/MFE: ungünstigste/günstigste '
        'Schlusskursbewegung relativ zum Einstieg '
        'einer Long-Position. Null-/Negativpreise '
        'werden in prozentualen Pfadkennzahlen '
        'nicht berechnet. Keine Intraday-Hochs/-Tiefs, '
        'kein Stop-Loss-Backtest; keine Kosten oder '
        'Rollkorrektur.'
    )

    st.subheader(
        'Stabilität: gleiche Auswahl über '
        'verschiedene Historien'
    )

    periods = []

    for label, sub in [
        (
            'Gesamte Auswahl',
            obs
        ),
        (
            'Letzte 10 Startjahre',
            obs[
                obs.year
                >= bounds[1] - 9
            ]
        ),
        (
            'Letzte 5 Startjahre',
            obs[
                obs.year
                >= bounds[1] - 4
            ]
        )
    ]:

        if not sub.empty:

            row = (
                direction_summary(
                    sub
                )
                .reset_index()
            )

            row.insert(
                0,
                'Historie',
                label
            )

            periods.append(
                row
            )

    stability = pd.concat(
        periods,
        ignore_index=True
    )

    st.dataframe(
        stability.round(3)
    )

    st.caption(
        'Diese Teilmengen überlappen; sie sind keine '
        'unabhängigen Bestätigungen. Einzelne Jahre '
        'werden nicht nach ihrem Ergebnis entfernt.'
    )

    details = obs.copy()

    for c in [
        'return',
        'MAE_Close_Long',
        'MFE_Close_Long',
        'Max_Drawdown_Close'
    ]:

        details[
            c + '_pct'
        ] = (
            100
            * details.pop(c)
        )

    st.subheader(
        'Jahresergebnisse'
    )

    st.dataframe(
        details.round(3)
    )

    st.bar_chart(
        details
        .set_index('year')[
            ['return_pct']
        ]
    )

    months = observations(
        p,
        'Monate'
    )

    if not months.empty:

        months = months[
            months.year.between(
                *bounds
            )
        ]

        heat = (
            months.pivot(
                index='year',
                columns='group',
                values='return'
            )
            * 100
        )

        st.subheader(
            'Monatsübersicht der '
            'gewählten Historie'
        )

        st.caption(
            'Gesamte Monate, unabhängig vom gewählten '
            'Fenster und MA200-Filter; leere Zellen '
            'bleiben fehlend.'
        )

        heatlong = months[
            [
                'year',
                'group',
                'return'
            ]
        ].copy()

        heatlong[
            'Rendite'
        ] = (
            heatlong.pop(
                'return'
            )
            * 100
        )

        st.vega_lite_chart(
            heatlong,
            {
                'mark': 'rect',
                'encoding': {
                    'x': {
                        'field': 'group',
                        'type': 'ordinal',
                        'title': 'Monat'
                    },
                    'y': {
                        'field': 'year',
                        'type': 'ordinal',
                        'title': 'Jahr'
                    },
                    'color': {
                        'field': 'Rendite',
                        'type': 'quantitative',
                        'scale': {
                            'scheme': 'redblue',
                            'domainMid': 0
                        }
                    },
                    'tooltip': [
                        {'field': 'year'},
                        {'field': 'group'},
                        {'field': 'Rendite'}
                    ]
                }
            },
            use_container_width=True
        )

        st.dataframe(
            heat.round(2)
        )

    else:
        heat = pd.DataFrame()

    with st.expander(
        'Ausgeschlossene Jahre '
        'und Kurvendaten'
    ):

        st.dataframe(
            excluded
        )

        st.dataframe(
            stats
        )

    protocol = {
        'version': VERSION,
        'asset': asset,
        'analysis': 'seasonal paths',
        'start_month_day': [
            start.month,
            start.day
        ],
        'end_month_day': [
            end.month,
            end.day
        ],
        'years': list(bounds),
        'trend_filter': trend,
        'alignment': (
            'observed trading-row offset; no fill'
        ),
        'display_length': display,
        'control_rule': (
            'fixed nonoverlapping return blocks in '
            'start calendar year; seasonal return dates '
            'excluded; same entry trend filter; '
            'equal year weight on common sample'
        ),
        'historical_split': str(
            cutoff.date()
        ),
        'split_purge': (
            'entry >= cutoff for later sample; '
            'exit < cutoff for earlier'
        ),
        'equal_year_weights': True,
        'band': (
            'cross-year 10/90 percentiles, '
            'not confidence interval'
        ),
        'nonpositive_price_rule': (
            'raw prices retained; conventional percentage '
            'returns require positive endpoints; '
            'percentage path values are NaN on '
            'zero/negative observations'
        ),
        'source': provenance,
        'quality': quality,
        'exploratory': True,
        'holdout': False,
        'costs_included': False,
        'multiple_testing_adjusted': False,
        'promotion_authorized': False
    }

    buf = io.BytesIO()

    with zipfile.ZipFile(
        buf,
        'w',
        zipfile.ZIP_DEFLATED
    ) as z:

        for name, df in [
            (
                'prices',
                p.to_frame()
            ),
            (
                'yearly_paths_decimal',
                curves
            ),
            (
                'display_statistics_pct',
                stats
            ),
            (
                'observations',
                details
            ),
            (
                'summary',
                summary
            ),
            (
                'risk',
                risk
            ),
            (
                'stability',
                stability
            ),
            (
                'monthly_pct',
                heat
            ),
            (
                'excluded',
                excluded
            ),
            (
                'controls',
                controls
            ),
            (
                'paired_controls',
                paired
            ),
            (
                'control_comparison',
                comparison
            ),
            (
                'historical_split',
                split_summary
            ),
            (
                'split_purged',
                purged
            )
        ]:

            z.writestr(
                name + '.csv',
                df.to_csv()
            )

        z.writestr(
            'protocol.json',
            json.dumps(
                protocol,
                indent=2,
                default=str
            )
        )

    st.download_button(
        'Kurven und Auswertung als ZIP',
        buf.getvalue(),
        file_name=(
            ASSETS[asset][1]
            + '_Seasonal_Curves_v1_5_1.zip'
        ),
        mime='application/zip'
    )


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    import streamlit as st

    st.set_page_config(
        page_title=(
            'Multi-Asset Seasonality Lab'
        ),
        layout='wide'
    )

    st.title(
        'Multi-Asset Seasonality Lab v'
        + VERSION
    )

    st.write(
        'Kalendermuster der sieben Projekt-Assets '
        'untersuchen – separate Research-App.'
    )

    st.warning(
        'Explorative Auswertung: Viele frei gewählte '
        'Zeitfenster erzeugen Zufallstreffer. Keine '
        'automatischen Signifikanz-, Modell- oder '
        'Handelsfreigaben.'
    )

    pair = st.selectbox(
        'Asset',
        list(SYMBOLS)
    )

    source = st.radio(
        'Datenquelle',
        [
            'Vorhandene Audit-ZIP / CSV',
            'Yahoo abrufen'
        ],
        horizontal=True
    )

    key = (
        'seasonality_'
        + pair
    )

    if source == (
        'Vorhandene Audit-ZIP / CSV'
    ):

        upload = st.file_uploader(
            'Research-ZIP oder Tages-CSV',
            type=[
                'zip',
                'csv'
            ]
        )

        if upload is None:

            st.info(
                'Für USD/CHF und USD/JPY funktionieren '
                'die bisherigen Audit-ZIPs. Für andere '
                'Assets: Preis-CSV oder ZIP mit Preis-CSV. '
                'Spalten: Date und asset_price oder Close. '
                'Alternativ Yahoo wählen.'
            )

            return

        raw = upload.getvalue()

        member = None

        try:

            if upload.name.lower().endswith(
                '.zip'
            ):

                with zipfile.ZipFile(
                    io.BytesIO(raw)
                ) as z:

                    members = [
                        n for n in z.namelist()
                        if n.lower().endswith(
                            '.csv'
                        )
                    ]

                if not members:
                    raise ValueError(
                        'Keine CSV in der ZIP.'
                    )

                preferred = (
                    ASSETS[pair][1]
                    + '_research_timeseries.csv'
                )

                default = next(
                    (
                        i
                        for i, n
                        in enumerate(members)
                        if (
                            n.split('/')[-1]
                            == preferred
                        )
                    ),
                    0
                )

                member = st.selectbox(
                    'Preis-CSV in der ZIP',
                    members,
                    index=default
                )

            st.caption(
                'Upload: Asset-Zuordnung und Preisart '
                'werden nicht aus den Kurswerten erkannt.'
            )

            price_kind = st.selectbox(
                'Preisart der hochgeladenen Reihe',
                [
                    'Ungeprüft',
                    'Spot / Kassakurs',
                    'Preisindex',
                    'Total-Return-Index',
                    (
                        'Futures / kontinuierliche '
                        'Futures-Reihe'
                    )
                ]
            )

            confirmed = st.checkbox(
                'Die ausgewählte Datei gehört '
                'zum oben gewählten Asset.',
                key=(
                    'confirm_'
                    + pair
                    + '_'
                    + hashlib.sha256(
                        raw
                    ).hexdigest()[:12]
                )
            )

            if not confirmed:
                return

            frame = load_bytes(
                raw,
                upload.name,
                pair,
                member
            )

        except Exception as e:

            st.error(
                str(e)
            )

            return

        provenance = {
            'source': upload.name,
            'sha256': hashlib.sha256(
                raw
            ).hexdigest(),
            'csv_member': member,
            'price_kind': price_kind,
            'asset_identity': (
                'user confirmed; '
                'not independently verified'
            )
        }

    else:

        st.info(
            'Yahoo-Symbol: '
            + SYMBOLS[pair]
            + ' · '
            + ASSETS[pair][2]
        )

        st.caption(
            'Yahoo ist eine aktuelle, potenziell '
            'revidierte Historie. Kein Nachweis '
            'historischer Veröffentlichungszeitpunkte.'
        )

        if st.button(
            '15 Jahre Tagesdaten abrufen'
        ):

            try:

                import yfinance as yf

                started = pd.Timestamp.now(
                    tz='UTC'
                )

                h = yf.Ticker(
                    SYMBOLS[pair]
                ).history(
                    start=str(
                        (
                            started
                            - pd.DateOffset(
                                years=15
                            )
                        ).date()
                    ),
                    interval='1d',
                    auto_adjust=False,
                    back_adjust=False,
                    repair=False,
                    keepna=True,
                    actions=False,
                    timeout=20,
                    raise_errors=True
                )

                frame = h.reset_index()

                raw = (
                    frame
                    .to_csv(
                        index=False
                    )
                    .encode()
                )

                st.session_state[
                    key
                ] = (
                    frame,
                    {
                        'source': (
                            'Yahoo '
                            + SYMBOLS[pair]
                        ),
                        'received_at_utc': (
                            pd.Timestamp.now(
                                tz='UTC'
                            ).isoformat()
                        ),
                        'yfinance': (
                            yf.__version__
                        ),
                        'sha256': (
                            hashlib.sha256(
                                raw
                            ).hexdigest()
                        )
                    }
                )

            except Exception as e:

                st.error(
                    'Abruf fehlgeschlagen: '
                    + str(e)
                )

                return

        if key not in st.session_state:
            return

        frame, provenance = (
            st.session_state[key]
        )

    if pair in [
        'Gold',
        'WTI'
    ]:

        st.warning(
            'Futures-Ergebnisse sind '
            'Quellen-Saisonalität, keine vollständig '
            'rollbereinigte Strategie. Bei Null- oder '
            'Negativpreisen werden betroffene '
            'Prozentrechnungen gesperrt; die Rohwerte '
            'werden weder entfernt noch künstlich ersetzt.'
        )

    try:

        p, quality = prepare(
            frame,
            pair
        )

    except Exception as e:

        st.error(
            str(e)
        )

        return

    # Dedicated historical WTI negative-price notice.
    if (
        pair == 'WTI'
        and quality.get(
            'nonpositive_rows',
            0
        )
    ):

        st.warning(
            f"{quality['nonpositive_rows']} historische "
            'Null-/Negativpreis-Beobachtung(en) gefunden. '
            'Die Rohpreise bleiben unverändert erhalten. '
            'Prozentuale Berechnungen mit einem '
            'Null-/Negativpreis als Start- oder Endpunkt '
            'werden ausgeschlossen.'
        )

        with st.expander(
            'Null-/Negativpreise anzeigen'
        ):

            st.write(
                quality.get(
                    'nonpositive_dates',
                    []
                )
            )

            affected = p[
                p <= 0
            ].to_frame()

            st.dataframe(
                affected
            )

    st.caption(
        f"{len(p):,} Tageswerte · "
        f"{quality['first']} bis "
        f"{quality['last']} · "
        f"ausgeschlossene Zeilen: "
        f"{quality['removed_rows']}"
    )

    st.caption(
        'Datum = Anbieter-Tageslabel. '
        'Feiertagskalender, identische Session-Schlüsse '
        'und PIT-Verfügbarkeit sind nicht verifiziert.'
    )

    mode = st.selectbox(
        'Analyse',
        [
            'Saisonale Kurven und Filter',
            'Mein Zeitraum + optionales Event',
            'Monate',
            'Wochentage',
            'Kalenderwochen',
            'Freie Datumsspanne',
            'Freie Wochenspanne',
            'Events'
        ]
    )

    if mode == (
        'Saisonale Kurven und Filter'
    ):

        seasonal_dashboard(
            st,
            p,
            pair,
            provenance,
            quality
        )

        return

    if mode == (
        'Mein Zeitraum + optionales Event'
    ):

        period_dashboard(
            st,
            p,
            pair,
            provenance,
            quality
        )

        return

    if mode == 'Events':

        event_dashboard(
            st,
            p,
            pair,
            provenance,
            quality
        )

        return

    custom = None

    if mode == (
        'Freie Datumsspanne'
    ):

        a, b = st.columns(2)

        start = a.date_input(
            'Jährlich ab '
            '(Jahr wird ignoriert)',
            date(
                2000,
                11,
                15
            )
        )

        end = b.date_input(
            'Jährlich bis einschließlich',
            date(
                2000,
                12,
                15
            )
        )

        custom = (
            start.month,
            start.day,
            end.month,
            end.day
        )

    elif mode == (
        'Freie Wochenspanne'
    ):

        sw = st.number_input(
            'Erste ISO-Kalenderwoche',
            1,
            53,
            45
        )

        ew = st.number_input(
            'Letzte ISO-Kalenderwoche '
            'einschließlich',
            1,
            53,
            50
        )

        custom = (
            sw,
            ew
        )

    with st.expander(
        'Welche Rendite wird gemessen?',
        expanded=True
    ):

        st.write(
            'Monat/Woche/Spanne: letzter verfügbarer '
            'Schlusskurs VOR dem Beginn bis letzter '
            'Schlusskurs innerhalb der Spanne. '
            'Jahreswechsel werden unterstützt. '
            'Nicht vollständig abgedeckte '
            'Kalenderzeiträume werden ausgelassen; '
            'ein Wochenende am Datenende kann deshalb '
            'auch eine gerade beendete Woche '
            'ausschließen.'
        )

        st.write(
            'Wochentag: Rendite vom vorherigen '
            'verfügbaren Schlusskurs zum Schlusskurs '
            'dieses Tages. Montag enthält typischerweise '
            'das Wochenende. Das ist keine '
            'Open-to-Close- oder Intraday-Rendite.'
        )

        st.write(
            'Renditen sind Veränderungen der gewählten '
            'Preisreihe: keine Spreads, Finanzierung, '
            'Slippage oder Positionsgrößen. Fehlende '
            'Preise werden nicht aufgefüllt. '
            'Offensichtliche Lücken über sieben '
            'Kalendertage werden ausgeschlossen; '
            'kürzere Quellenlücken können unentdeckt '
            'bleiben. Null-/Negativpreise bleiben in '
            'der Rohreihe erhalten; konventionelle '
            'Prozentrechnungen mit einem solchen '
            'Start- oder Endwert werden nicht '
            'durchgeführt.'
        )

    obs = observations(
        p,
        mode,
        custom
    )

    if obs.empty:

        st.warning(
            'Keine vollständig auswertbaren '
            'Zeiträume.'
        )

        return

    yrs = sorted(
        obs.year.unique()
    )

    lo = int(
        min(yrs)
    )

    hi = int(
        max(yrs)
    )

    if lo < hi:

        yrange = st.slider(
            'Auswertungsjahre '
            '(Startjahr bzw. ISO-Jahr)',
            lo,
            hi,
            (lo, hi)
        )

    else:

        yrange = (
            lo,
            hi
        )

    obs = obs[
        obs.year.between(
            *yrange
        )
    ]

    if obs.empty:

        st.warning(
            'Keine Beobachtungen im '
            'ausgewählten Bereich.'
        )

        return

    summary = summarize(
        obs
    )

    st.subheader(
        'Ergebnis je Kalendergruppe'
    )

    st.dataframe(
        summary.round(3)
    )

    st.bar_chart(
        summary[
            [
                'Mittel_%',
                'Median_%'
            ]
        ]
    )

    annual = (
        obs.pivot_table(
            index='year',
            columns='group',
            values='return',
            aggfunc='mean'
        )
        * 100
    )

    st.subheader(
        'Einzeljahre – '
        'Mittel der Beobachtungen in %'
    )

    st.dataframe(
        annual.round(3)
    )

    st.caption(
        'Bei Monaten/Wochen/freien Spannen '
        'meist eine Beobachtung je Jahr; bei '
        'Wochentagen ein Jahresmittel vieler '
        'Tagesrenditen. N ist keine Zahl '
        'unabhängiger Tests.'
    )

    st.subheader(
        'Zeitliche Stabilität'
    )

    parts = []

    for a, b in [
        (2012, 2016),
        (2017, 2020),
        (2021, 2025)
    ]:

        # Purge intervals crossing
        # the period boundary.
        sub = obs[
            (
                pd.to_datetime(
                    obs.entry
                )
                >= pd.Timestamp(
                    a,
                    1,
                    1
                )
            )
            & (
                pd.to_datetime(
                    obs.exit
                )
                <= pd.Timestamp(
                    b,
                    12,
                    31
                )
            )
        ]

        if not sub.empty:

            tab = (
                summarize(
                    sub
                )
                .reset_index()
            )

            tab.insert(
                0,
                'Periode',
                f'{a}–{b}'
            )

            parts.append(
                tab
            )

    stability = (
        pd.concat(
            parts,
            ignore_index=True
        )
        if parts
        else pd.DataFrame()
    )

    st.dataframe(
        stability.round(3)
    )

    with st.expander(
        'Alle einzelnen Beobachtungen'
    ):
        st.dataframe(
            obs
        )

    protocol = {
        'version': VERSION,
        'asset': pair,
        'mode': mode,
        'custom': custom,
        'years': yrange,
        'yahoo_proxy': (
            {
                'symbol': SYMBOLS[pair],
                'description': ASSETS[pair][2]
            }
            if source == 'Yahoo abrufen'
            else None
        ),
        'source': provenance,
        'quality': quality,
        'exploratory': True,
        'PIT_verified': False,
        'multiple_testing_adjusted': False,
        'promotion_authorized': False,
        'return_rule': (
            'previous available close before interval '
            'to last inside; weekdays previous-close '
            'to close'
        ),
        'missing_rule': (
            'no fill; reject explicit missing prices; '
            'intervals with calendar gaps>7 days excluded'
        ),
        'nonpositive_price_rule': (
            'raw prices retained; conventional percentage '
            'returns require strictly positive start '
            'and end prices'
        ),
        'full_calendar_end_required': True,
        'costs_included': False
    }

    buf = io.BytesIO()

    with zipfile.ZipFile(
        buf,
        'w',
        zipfile.ZIP_DEFLATED
    ) as z:

        for name, table in [
            (
                'prices',
                p.to_frame()
            ),
            (
                'observations',
                obs
            ),
            (
                'summary',
                summary
            ),
            (
                'annual',
                annual
            ),
            (
                'stability',
                stability
            )
        ]:

            z.writestr(
                name + '.csv',
                table.to_csv()
            )

        z.writestr(
            'protocol.json',
            json.dumps(
                protocol,
                indent=2,
                default=str
            )
        )

    st.download_button(
        'Auswertung als ZIP herunterladen',
        buf.getvalue(),
        file_name=(
            ASSETS[pair][1]
            + '_Seasonality_v1_5_1.zip'
        ),
        mime='application/zip'
    )

    st.info(
        'Ein auffälliges Muster ist zunächst eine '
        'Hypothese. Für eine Bestätigung müssen Zeitraum, '
        'Richtung und Kriterien vor einer separaten '
        'Validierung feststehen. Dieses Dashboard '
        'optimiert keine Gewichte.'
    )


if __name__ == '__main__':
    main()