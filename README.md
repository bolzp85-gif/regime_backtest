# regime_backtest

Regime Backtest Lab

Zweck

Separates Streamlit-Research-Modul für den Vergleich:

1. A · Current – aktueller produktiver Referenzstand
2. B · Literature Prior v1 – literaturbasierte Prior-Gewichte
3. C · Equal Weight – neutrale Benchmark

Das Modul verändert weder das Market Regime Dashboard noch den TradePilot.

GitHub-Dateien

Lege diese Dateien in dasselbe Repository/Verzeichnis:

• regime_backtest_lab.py
• regime_engine.py

Die vorhandene requirements.txt kann unverändert weiterverwendet werden,
sofern sie bereits die Dependencies des Dashboards enthält:

• streamlit
• numpy
• pandas
• requests
• scipy
• yfinance
• fredapi
• plotly

Streamlit

Für eine separate Research-App:

Main file path:

regime_backtest_lab.py

Der vorhandene Streamlit Secret:

FRED_API_KEY = "..."

wird wiederverwendet.

Wichtig

Das Research-Modul verwendet kein bfill(). FRED First-Release/Vintage wird
bevorzugt. Nicht perfekt point-in-time rekonstruierbare Quellen werden in der
App transparent gekennzeichnet.