"""Streamlit diagnostic v1.0.0; read-only, no trading/model changes.
Run: streamlit run ALFRED_Release_Diagnose.py
Uses FRED_API_KEY from Streamlit secrets or environment. Never exports the key.
"""
import io
import json
import os
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

CASES = {
    'IR3TIB01CHM156N': ('2021-06-03', '2024-06-02'),
    'IR3TIB01USM156N': ('2013-06-03', '2015-06-02'),
    'IRLTLT01USM156N': ('2013-06-03', '2015-06-02'),
    'IRLTLT01JPM156N': ('2013-06-03', '2015-06-02'),
    'IRLTLT01CHM156N': ('2013-06-03', '2015-06-02'),
}


def retrieve(key, series, start, end):
    """Fetch all pages, preserving original response records and metadata."""
    offset, records, pages = 0, [], []
    while True:
        params = dict(series_id=series, api_key=key, file_type='json',
                      realtime_start=start, realtime_end=end,
                      output_type=1, limit=100000, offset=offset,
                      sort_order='asc')
        try:
            response = requests.get('https://api.stlouisfed.org/fred/series/observations',
                                    params=params, timeout=30)
        except requests.RequestException:
            raise ValueError('FRED connection failed') from None
        if response.status_code != 200:
            raise ValueError(f'FRED HTTP {response.status_code}')
        try:
            payload = response.json()
        except ValueError:
            raise ValueError('FRED returned invalid JSON') from None
        batch = payload.get('observations')
        if not isinstance(batch, list) or 'count' not in payload:
            raise ValueError('FRED response is missing observations/count')
        pages.append(payload)
        records.extend(batch)
        offset += len(batch)
        if offset >= int(payload['count']):
            break
        if not batch:
            raise ValueError('Incomplete FRED pagination')
    return records, pages


def selected_records(records):
    """Reproduce v1.0.29 selection, retaining dates for source comparison."""
    d = pd.DataFrame(records)
    if d.empty:
        return d
    for col in ['date', 'realtime_start']:
        d[col] = pd.to_datetime(d[col], errors='raise')
    d['value_numeric'] = pd.to_numeric(d['value'], errors='coerce')
    d = d.dropna(subset=['date', 'realtime_start', 'value_numeric'])
    d = d.sort_values(['date', 'realtime_start']).groupby('date', as_index=False).first()
    return d.sort_values(['realtime_start', 'date']).drop_duplicates('realtime_start', keep='last')


def main():
    st.title('ALFRED Release-Diagnose v1.0.0')
    st.write('PrÃ¼ft RohverÃ¶ffentlichungen fÃ¼r auffÃ¤llige Zinsdaten. Keine ModellÃ¤nderung.')
    key = os.environ.get('FRED_API_KEY', '')
    try:
        key = st.secrets.get('FRED_API_KEY', key)
    except FileNotFoundError:
        pass
    if not key:
        key = st.text_input('FRED API-Key', type='password')
    if st.button('Diagnose abrufen', disabled=not bool(key)):
        output, summary, errors = io.BytesIO(), [], []
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
            for series, (start, end) in CASES.items():
                try:
                    records, pages = retrieve(key, series, start, end)
                    # Preserve source response before applying research selection.
                    archive.writestr(f'{series}_raw_pages.json', json.dumps(pages, indent=2))
                    chosen = selected_records(records)
                    archive.writestr(f'{series}_selected.csv', chosen.to_csv(index=False))
                    summary.append(dict(series=series, start=start, end=end,
                                        raw_records=len(records), selected_records=len(chosen)))
                except Exception as exc:
                    # Only sanitized errors leave this diagnostic.
                    errors.append(dict(series=series, error=type(exc).__name__))
            archive.writestr('protocol.json', json.dumps(dict(
                version='1.0.0', created_utc=datetime.now(timezone.utc).isoformat(),
                research_only=True, cases=CASES, requests=summary, errors=errors,
                scope='Targeted raw records; not a replacement full-history backtest',
                limitations=['No full PIT certification',
                             'Single-window comparison may expose chunk-boundary differences']), indent=2))
        st.session_state['alfred_diagnostic_zip'] = output.getvalue()
        st.dataframe(pd.DataFrame(summary))
        if errors:
            st.warning('Einige Abrufe sind fehlgeschlagen; Details stehen im Protokoll.')
    if 'alfred_diagnostic_zip' in st.session_state:
        st.download_button('Diagnose-ZIP herunterladen', st.session_state['alfred_diagnostic_zip'],
                           'ALFRED_Release_Diagnose_v1_0_0.zip', 'application/zip')


if __name__ == '__main__':
    main()