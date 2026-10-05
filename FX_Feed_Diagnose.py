"""FX Feed Diagnose v1.0.0 – standalone Streamlit research utility.

Start: streamlit run FX_Feed_Diagnose.py
Dependencies: streamlit, yfinance, pandas (already used by the research lab).
No engine imports, credentials, scheduled jobs or filesystem writes.
API reference: https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import platform
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

VERSION = "1.0.0"
SYMBOLS = {"MOVE": "^MOVE", "USDCHF": "CHF=X"}
PARAMS = dict(period="3mo", interval="1d", auto_adjust=False,
              back_adjust=False, repair=False, keepna=True,
              actions=False, rounding=False, timeout=20, raise_errors=True)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def inspect_history(history, received_at):
    """Conservative date diagnostics, never certification of a session close."""
    now = pd.Timestamp(received_at)
    if now.tzinfo is None:
        raise ValueError("Abrufzeit benötigt eine Zeitzone.")
    rows = []
    tz_known = isinstance(history.index, pd.DatetimeIndex) and history.index.tz is not None
    duplicates = history.index.duplicated(keep=False)
    ordered = history.index.is_monotonic_increasing
    for pos, (stamp, values) in enumerate(history.iterrows()):
        stamp = pd.Timestamp(stamp)
        flags = []
        valid_stamp = not pd.isna(stamp)
        date = stamp.date() if valid_stamp else None
        aware = valid_stamp and stamp.tzinfo is not None
        if not tz_known or not aware:
            flags.append("TIMEZONE_UNKNOWN")
        if not valid_stamp:
            flags.append("INVALID_TIMESTAMP")
        close = pd.to_numeric(values.get("Close"), errors="coerce")
        if pd.isna(close) or not math.isfinite(float(close)) or close <= 0:
            flags.append("INVALID_CLOSE")
        if duplicates[pos]:
            flags.append("DUPLICATE_TIMESTAMP")
        if not ordered:
            flags.append("UNSORTED_INPUT")
        if valid_stamp and stamp.dayofweek >= 5:
            flags.append("WEEKEND_DATE")
        if aware:
            # Local date alone is not evidence of official market close.
            if date >= now.tz_convert(stamp.tzinfo).date():
                flags.append("CURRENT_OR_FUTURE_LOCAL_DATE")
            if stamp.tz_convert("UTC") > now.tz_convert("UTC"):
                flags.append("FUTURE_TIMESTAMP")
        rows.append({"provider_timestamp":str(stamp), "provider_date":str(date) if date else "",
                     "provider_timezone":str(stamp.tzinfo) if aware else "",
                     "Close":close, "observed_at_utc":now.tz_convert("UTC").isoformat(),
                     "candidate_for_manual_review":not flags,
                     "flags":";".join(flags),
                     "session_close_verified":False, "publication_time_verified":False})
    return pd.DataFrame(rows, columns=["provider_timestamp","provider_date","provider_timezone","Close",
                                      "observed_at_utc","candidate_for_manual_review","flags",
                                      "session_close_verified","publication_time_verified"])


def collect(yf):
    run = {"version":VERSION,"run_id":uuid.uuid4().hex,"started_at_utc":utc_now(),
           "parameters":PARAMS,"python":platform.python_version(),"pandas":pd.__version__,
           "yfinance":getattr(yf,"__version__","unknown"),"feeds":{},
           "PIT_verified":False,"shadow_started":False,"status":"DIAGNOSTIC_ONLY"}
    tables = {}
    payload = {}
    for name, symbol in SYMBOLS.items():
        feed = {"symbol":symbol,"history_request_started_at_utc":utc_now()}
        run["feeds"][name] = feed
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(**PARAMS)
            feed["history_received_at_utc"] = utc_now()
            if not isinstance(history,pd.DataFrame) or history.empty:
                raise ValueError("Leere Tageshistorie vom Anbieter.")
            # Preserve library output before diagnostic filtering. This is NOT raw HTTP JSON.
            data = history.to_csv(index_label="provider_timestamp").encode("utf-8")
            payload[name+"_provider_history.csv"] = data
            feed["history_sha256"] = hashlib.sha256(data).hexdigest()
            table = inspect_history(history, feed["history_received_at_utc"])
            tables[name] = table
            payload[name+"_diagnostics.csv"] = table.to_csv(index=False).encode("utf-8")
            feed.update(status="RECEIVED_UNVERIFIED",rows=len(history),
                        candidate_rows=int(table.candidate_for_manual_review.sum()),
                        latest_provider_timestamp=str(history.index.max()),
                        timezone=str(getattr(history.index,"tz",None)))
            try:
                feed["metadata_request_started_at_utc"] = utc_now()
                metadata = ticker.get_history_metadata()
                feed["metadata_received_at_utc"] = utc_now()
                payload[name+"_metadata.json"] = json.dumps(metadata,default=str,indent=2).encode("utf-8")
            except Exception as exc:
                feed["metadata_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            feed["status"] = "FETCH_OR_SCHEMA_ERROR"
            feed["error"] = f"{type(exc).__name__}: {exc}"
            feed["error_at_utc"] = utc_now()
    run["finished_at_utc"] = utc_now()
    run["all_histories_received"] = all(f.get("status")=="RECEIVED_UNVERIFIED" for f in run["feeds"].values())
    # Diagnostic intersection only: equal provider dates do not prove synchronous closes.
    if len(tables)==2:
        sets = [set(t.loc[t.candidate_for_manual_review,"provider_date"]) for t in tables.values()]
        common = sorted(sets[0] & sets[1])
        run["common_candidate_dates"] = len(common)
        run["latest_common_candidate_date"] = common[-1] if common else None
    try:
        run["diagnostic_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        run["diagnostic_code_sha256"] = None
    payload["protocol.json"] = json.dumps(run,indent=2,default=str).encode("utf-8")
    payload["README.txt"] = (
        "FX Feed Diagnose v1.0.0\n"
        "Yahoo daily histories via yfinance; provider_history is library output, not original HTTP payload.\n"
        "No backfill, forward fill, price repair or automatic adjustment requested.\n"
        "received_at proves observation at retrieval, NOT first publication or historical availability.\n"
        "An old weekday date is only a review candidate; it does NOT certify finality or freshness.\n"
        "Exchange metadata can describe the current session, not all historical sessions.\n"
        "Equal MOVE/FX provider dates do NOT prove synchronous session closes.\n"
        "No market calendar or release calendar has been independently verified.\n"
        "Session close and publication verification therefore remain false.\n"
        "Keep this ZIP unchanged. A later independently downloaded ZIP can reveal revisions.\n"
        "This diagnostic starts no shadow, computes no trading score and writes no repository files.\n"
        "API reference: https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html\n"
    ).encode("utf-8")
    payload["checksums.json"] = json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in payload.items()},indent=2).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as z:
        for name, content in payload.items():
            z.writestr(name,content)
    return run,tables,buffer.getvalue()


def main():
    import streamlit as st
    import yfinance as yf
    st.set_page_config(page_title="FX Feed Diagnose",layout="wide")
    st.title("FX Feed Diagnose v1.0.0")
    st.write("Separate Datenprüfung für MOVE und USD/CHF. Kein Modelltest und kein Future Shadow.")
    st.info("1. Diagnose starten. 2. ZIP herunterladen. 3. Diese eine ZIP zur Auswertung schicken.")
    st.caption("Beide Reihen werden gemeinsam abgerufen. Keine API-Schlüssel erforderlich.")
    if st.button("Diagnose starten",type="primary"):
        st.session_state.pop("fx_feed_diagnostic",None)
        with st.spinner("MOVE und USD/CHF werden abgerufen …"):
            st.session_state["fx_feed_diagnostic"] = collect(yf)
    if "fx_feed_diagnostic" in st.session_state:
        run,tables,data = st.session_state["fx_feed_diagnostic"]
        st.write("Abruf abgeschlossen (UTC):",run["finished_at_utc"])
        if not run["all_histories_received"]:
            st.error("Mindestens ein Feed konnte nicht ausgewertet werden. Bitte auch dann die Diagnose-ZIP herunterladen.")
        st.warning("Abrufzeit dokumentiert; offizieller Tagesabschluss und Veröffentlichungszeit bleiben ungeprüft.")
        for name,feed in run["feeds"].items():
            st.subheader(name)
            st.json(feed)
            if name in tables:
                st.dataframe(tables[name].tail(12))
        stamp = pd.Timestamp(run["started_at_utc"]).strftime("%Y%m%dT%H%M%SZ")
        st.download_button("Diagnose-ZIP herunterladen",data=data,
                           file_name=f"FX_Feed_Diagnose_v1_0_0_{stamp}_{run['run_id'][:8]}.zip",
                           mime="application/zip")
    with st.expander("Einrichtung und Grenzen"):
        st.write("Diese Datei als zusätzliche App starten, nicht Regime_Backtest_lab.py oder regime_engine.py ersetzen.")
        st.code("streamlit run FX_Feed_Diagnose.py")
        st.write("Benötigt streamlit, yfinance und pandas. In Streamlit Cloud eine separate App mit dieser Hauptdatei anlegen; vorhandene Apps und Workflows unverändert lassen.")
        st.write("Die ZIP wird erst durch Herunterladen dauerhaft gesichert. Es gibt keine automatische Sammlung. Ein einzelner Abruf beweist weder historische PIT-Sicherheit noch einen endgültigen Tagesabschluss.")


if __name__ == "__main__":
    main()