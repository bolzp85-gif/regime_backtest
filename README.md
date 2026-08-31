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

REGIME BACKTEST LAB – STATIC VALIDATION REPORT
================================================

PASS: Python syntax
PASS: no duplicate top-level functions
PASS: no duplicate static Streamlit widget keys
PASS: no .bfill() in research pipeline
PASS: Literature Prior pillar weights = 100% for every asset
PASS: Literature Prior subweights = 100%
PASS: Current / Literature / Equal benchmark present
PASS: Subweight-only and pillar-only decomposition present
PASS: forward horizons 5D / 20D / 60D configured
PASS: 20D forward max drawdown and realized volatility present
PASS: Spearman IC / directional / quintile / stress AUC present
PASS: rolling IC and calendar-year stability present
PASS: block bootstrap present
PASS: leave-one-pillar-out ablation present
PASS: Common Sample / Real-world Sample modes present
PASS: FRED First-Release / ALFRED attempt present
PASS: CFTC publication-lag approximation present

Production regime_engine SHA256:
f9fec22a70252fcef772aaa7496b8bd76571c2a9410e1a168a76d9bfc7da27ac

Important:
External API behavior cannot be fully runtime-tested in the offline
build environment. The Streamlit app reports source/PIT fallback
status transparently when it runs online.