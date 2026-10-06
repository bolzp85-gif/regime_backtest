"""Multi-Asset Seasonality Lab 1.2.0. Start: streamlit run FX_Seasonality_Lab.py
Dependencies: streamlit, pandas, numpy; optional yfinance for online history.
Independent research dashboard. No engine/shadow imports or modifications.
"""
import io
import json
import zipfile
import hashlib
from datetime import date
import numpy as np
import pandas as pd

VERSION = '1.2.0'
ASSETS = {
    'S&P 500': ('^GSPC','SP500','Preisindex; keine Dividendenrendite.'),
    'Nasdaq 100': ('^NDX','NASDAQ100','Preisindex; keine Dividendenrendite.'),
    'Gold': ('GC=F','GOLD','Yahoo-Gold-Futures als Proxy, kein Spotgold. Kontraktwechsel können die Saisonalität beeinflussen.'),
    'WTI': ('CL=F','WTI','Yahoo-WTI-Futures als Proxy. Kontraktwechsel und negative Preise sind besonders zu prüfen; keine Rollrendite eines handelbaren Portfolios.'),
    'EUR/USD': ('EURUSD=X','EURUSD','USD je EUR; steigender Kurs bedeutet stärkeren Euro.'),
    'USD/CHF': ('CHF=X','USDCHF','CHF je USD; steigender Kurs bedeutet stärkeren US-Dollar.'),
    'USD/JPY': ('JPY=X','USDJPY','JPY je USD; steigender Kurs bedeutet stärkeren US-Dollar.'),
}
SYMBOLS = {name:cfg[0] for name,cfg in ASSETS.items()}


def load_bytes(data, name, pair, member=None):
    if name.lower().endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            candidates=[n for n in z.namelist() if n.endswith('research_timeseries.csv')]
            expected=ASSETS[pair][1]+'_research_timeseries.csv'
            if member is not None:
                candidates=[n for n in z.namelist() if n==member and n.lower().endswith('.csv')]
            else:
                candidates=[n for n in candidates if n.split('/')[-1]==expected]
            if len(candidates)!=1:
                raise ValueError('Keine eindeutige Preis-CSV für das ausgewählte Asset. CSV im Dashboard auswählen oder direkt hochladen.')
            info=z.getinfo(candidates[0])
            if info.file_size>32*1024*1024:
                raise ValueError('CSV im ZIP ist größer als32 MiB.')
            frame=pd.read_csv(z.open(candidates[0]))
    else:
        frame=pd.read_csv(io.BytesIO(data))
    return frame


def prepare(frame, now=None):
    dc=next((c for c in ['Date','date','Datetime','provider_timestamp'] if c in frame),None)
    pc=next((c for c in ['asset_price','Close','close'] if c in frame),None)
    if dc is None or pc is None:
        raise ValueError('Benötigte CSV-Spalten: Date und asset_price oder Close.')
    # Preserve provider daily date labels; do not shift them through UTC.
    idx=pd.DatetimeIndex([pd.Timestamp(v).date() for v in frame[dc]])
    if idx.hasnans or idx.has_duplicates:
        raise ValueError('Fehlende oder doppelte Datumswerte: Quelle zuerst prüfen.')
    p=pd.Series(pd.to_numeric(frame[pc],errors='coerce').to_numpy(),index=idx,name='Close').sort_index()
    now=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('Zeitstempel benötigt Zeitzone.')
    cutoff=min(now.tz_convert(t).date() for t in ['UTC','Europe/London','America/New_York'])
    keep=(p.index.dayofweek<5)&(p.index<pd.Timestamp(cutoff))
    removed=int((~keep).sum());p=p.loc[keep]
    if p.empty or (~np.isfinite(p)).any() or (p<=0).any():
        raise ValueError('Keine gültige Historie oder fehlende/ungültige Preise. Keine automatische Auffüllung.')
    return p,{'cutoff_exclusive':str(cutoff),'removed_rows':removed,'rows':len(p),
              'first':str(p.index.min().date()),'last':str(p.index.max().date()),
              'largest_calendar_gap_days':int(p.index.to_series().diff().dt.days.max()) if len(p)>1 else 0}


def interval_record(p,start,end,label,year):
    """Return from last available close BEFORE start to last close ON/BEFORE end.
    Require historical coverage through calendar end (conservative on weekends).
    """
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    if start> end or end>p.index.max() or start<=p.index.min():return None
    prior=p.loc[p.index<start];inside=p.loc[(p.index>=start)&(p.index<=end)]
    if prior.empty or inside.empty:return None
    entry,exit_=prior.index[-1],inside.index[-1]
    # Avoid silently spanning obvious source outages; not a holiday-calendar proof.
    seq=p.loc[entry:exit_]
    if (start-entry).days>7 or (end-exit_).days>7 or seq.index.to_series().diff().dt.days.max()>7:return None
    r=seq.pct_change(fill_method=None).dropna()
    return {'group':label,'year':year,'planned_start':str(start.date()),'planned_end':str(end.date()),
            'entry':str(entry.date()),'exit':str(exit_.date()),'bars':len(inside),
            'return':float(inside.iloc[-1]/prior.iloc[-1]-1),
            'daily_vol':float(r.std(ddof=1)*np.sqrt(252)) if len(r)>1 else np.nan}


def observations(p,mode,custom=None):
    rows=[]
    if mode=='Wochentage':
        r=p.pct_change(fill_method=None)
        gaps=p.index.to_series().diff().dt.days
        for t,v in r.items():
            if pd.notna(v) and gaps.loc[t]<=7:
                rows.append({'group':['Mo','Di','Mi','Do','Fr'][t.dayofweek], 'year':t.year,
                             'entry':str(p.index[p.index.get_loc(t)-1].date()),'exit':str(t.date()),
                             'bars':1,'return':float(v)})
    else:
        for y in range(p.index.min().year,p.index.max().year+1):
            intervals=[]
            if mode=='Monate':
                for m in range(1,13):
                    a=pd.Timestamp(y,m,1);intervals.append((a,a+pd.offsets.MonthEnd(0),f'{m:02d}'))
            elif mode=='Kalenderwochen':
                for w in range(1,54):
                    try:a=pd.Timestamp(date.fromisocalendar(y,w,1))
                    except ValueError:continue
                    intervals.append((a,a+pd.Timedelta(days=6),f'KW{w:02d}'))
            elif mode=='Freie Datumsspanne':
                sm,sd,em,ed=custom
                try:
                    a=pd.Timestamp(y,sm,sd);b=pd.Timestamp(y+int((em,ed)<(sm,sd)),em,ed)
                    intervals=[(a,b,f'{sd:02d}.{sm:02d}–{ed:02d}.{em:02d}')]
                except ValueError:continue  # e.g.29 February in a non-leap year
            else:
                sw,ew=custom
                try:
                    a=pd.Timestamp(date.fromisocalendar(y,sw,1))
                    b=pd.Timestamp(date.fromisocalendar(y+int(ew<sw),ew,7))
                    intervals=[(a,b,f'KW{sw:02d}–KW{ew:02d}')]
                except ValueError:continue
            for a,b,label in intervals:
                rec=interval_record(p,a,b,label,y)
                if rec:rows.append(rec)
    return pd.DataFrame(rows)


def summarize(obs):
    rows=[]
    for key,g in obs.groupby('group',sort=True):
        r=g['return'];rows.append({'Gruppe':key,'N':len(g),'Jahre':g.year.nunique(),
          'Mittel_%':100*r.mean(),'Median_%':100*r.median(),'Positiv_%':100*(r>0).mean(),
          'Minimum_%':100*r.min(),'Maximum_%':100*r.max()})
    return pd.DataFrame(rows).set_index('Gruppe')


def event_observations(p, events, before, after):
    """Daily labels only; day0 must exist exactly. No inferred release time."""
    rows=[];rejected=[]
    for _,e in events.iterrows():
        day=pd.Timestamp(e['Date']).normalize();kind=str(e['Event'])
        if day not in p.index:
            rejected.append({'Date':str(day.date()),'Event':kind,'reason':'Kein Kursdatum; keine automatische Verschiebung'});continue
        i=p.index.get_loc(day)
        if i-before<0 or i+after>=len(p):
            rejected.append({'Date':str(day.date()),'Event':kind,'reason':'Fenster nicht vollständig'});continue
        path=p.iloc[i-before:i+after+1]
        if path.index.to_series().diff().dt.days.max()>7:
            rejected.append({'Date':str(day.date()),'Event':kind,'reason':'Quellenlücke >7 Kalendertage'});continue
        for label,a,b in [('Vorher',i-before,i-1),('Ereignistag',i-1,i),('Nachher',i,i+after),('Gesamt',i-before,i+after)]:
            rows.append({'Date':str(day.date()),'Event':kind,'group':kind+' | '+label,
                         'window':label,'year':day.year,'entry':str(p.index[a].date()),
                         'exit':str(p.index[b].date()),'bars':b-a,'return':float(p.iloc[b]/p.iloc[a]-1),
                         'full_start':str(path.index[0].date()),'full_end':str(path.index[-1].date())})
    return pd.DataFrame(rows),pd.DataFrame(rejected)


def event_dashboard(st,p,asset,provenance,quality):
    st.subheader('Event-Studie')
    st.warning('Tagesdaten trennen keine unmittelbare Reaktion auf eine Veröffentlichung. Tag0 ist das Anbieter-Datumslabel, nicht eine verifizierte Veröffentlichungssitzung. Keine Kausalitäts- oder Intraday-Aussage.')
    source=st.radio('Event-Termine',['Offizielle / eigene Termine als CSV','Quartalsverfall: Kalenderregel (ungeprüfte Näherung)'])
    if source.startswith('Quartalsverfall'):
        st.warning('Erzeugt den dritten Freitag im März, Juni, September und Dezember. Feiertagsverschiebungen und produkt-/börsenspezifische Abrechnung fehlen. Für echte Hexensabbat-Termine einen geprüften Kalender als CSV verwenden. Kein Verfallskalender für Gold-, Öl- oder FX-Kontrakte.')
        rows=[]
        for y in range(p.index.min().year,p.index.max().year+1):
            for m in [3,6,9,12]:
                first=pd.Timestamp(y,m,1);day=first+pd.Timedelta(days=(4-first.dayofweek)%7+14)
                rows.append({'Date':str(day.date()),'Event':'Quartalsverfall-Regel ungeprüft','Source':'third-Friday calendar rule'})
        events=pd.DataFrame(rows);event_source={'type':'calendar_rule','verified':False}
    else:
        st.markdown('Terminreferenzen: [Fed-Entscheidungen](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) · [US-CPI](https://www.bls.gov/schedule/news_release/cpi.htm) · [CPI-Archiv](https://www.bls.gov/bls/news-release/cpi.htm) · [Cboe-Verfallskalender](https://www.cboe.com/about/hours/us-options/)')
        st.write('CSV-Spalten: Date, Event, optional Source. Date im Format JJJJ-MM-TT; Event z. B. FOMC, CPI oder Hexensabbat. Bei FOMC den Entscheidungstag verwenden, bei CPI den Veröffentlichungstag – nicht den Berichtsmonat. Uhrzeitinformationen werden in dieser Tagesstudie nicht ausgewertet.')
        st.download_button('Leere Event-CSV-Vorlage','Date,Event,Source\n',file_name='event_calendar_template.csv',mime='text/csv')
        upload=st.file_uploader('Event-Kalender hochladen',type=['csv'],key='event_csv')
        if upload is None:return
        raw=upload.getvalue()
        try:
            events=pd.read_csv(io.BytesIO(raw))
            if not {'Date','Event'}.issubset(events):raise ValueError('Date und Event fehlen.')
            if not events.Date.astype(str).str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():raise ValueError('Date muss JJJJ-MM-TT sein, ohne Uhrzeit.')
            events['Date']=pd.to_datetime(events.Date,format='%Y-%m-%d',errors='raise')
            if events.Event.isna().any() or events.Event.astype(str).str.strip().eq('').any():raise ValueError('Event-Bezeichnung fehlt.')
            events['Event']=events.Event.astype(str).str.strip()
            if events.duplicated(['Date','Event']).any():raise ValueError('Doppelte Date/Event-Kombinationen entfernen.')
        except Exception as e:st.error(str(e));return
        event_source={'type':'uploaded_calendar','filename':upload.name,'sha256':hashlib.sha256(raw).hexdigest(),'verified':False}
    if events.empty:st.info('Kalender ist leer.');return
    kinds=st.multiselect('Ereignisse',sorted(events.Event.unique()),default=sorted(events.Event.unique()))
    events=events[events.Event.isin(kinds)].copy()
    before=int(st.number_input('Beginn: Handelstage vor Tag0',min_value=2,max_value=60,value=5))
    after=int(st.number_input('Ende: Handelstage nach Tag0',min_value=1,max_value=60,value=5))
    st.caption(f'Vorher: Schluss T−{before} bis T−1; Ereignistag: T−1 bis T0; Nachher: T0 bis T+{after}. Die Segmente sind getrennt; „Gesamt“ umfasst alle. Ein Handelstag ist hier eine verfügbare Kurszeile.')
    obs,rejected=event_observations(p,events,before,after)
    if obs.empty:st.warning('Keine vollständigen Event-Fenster.');st.dataframe(rejected);return
    windows=obs[['Date','Event','full_start','full_end']].drop_duplicates().sort_values(['full_start','full_end'])
    overlap=[];prior_end=None
    for _,row in windows.iterrows():
        overlap.append(prior_end is not None and row.full_start<=prior_end)
        prior_end=max(prior_end or row.full_end,row.full_end)
    windows['overlaps_prior_window']=overlap
    st.caption(f'{len(windows)} vollständige Ereignisse; {len(rejected)} ausgeschlossen; {sum(overlap)} Fenster überschneiden ein vorheriges. Ereignisse können zugleich auftreten und sind nicht unabhängig.')
    summary=summarize(obs);st.dataframe(summary.round(3));st.bar_chart(summary[['Mittel_%','Median_%']])
    annual=obs.pivot_table(index='year',columns='group',values='return',aggfunc='mean')*100
    st.subheader('Jahresmittel in %');st.dataframe(annual.round(3))
    with st.expander('Einzelereignisse, Überschneidungen und Ausschlüsse'):
        st.dataframe(obs);st.dataframe(windows);st.dataframe(rejected)
    protocol={'version':VERSION,'asset':asset,'analysis':'event study','before':before,'after':after,
              'price_source':provenance,'quality':quality,'event_source':event_source,
              'selected_events':kinds,'date_alignment':'exact provider date; absent dates rejected',
              'timing_verified':False,'exploratory':True,'multiple_testing_adjusted':False,
              'costs_included':False,'promotion_authorized':False,
              'overlap_rule':'inclusive full window overlap flagged; observations retained',
              'interpretation':'calendar association, not causal release surprise or executable strategy'}
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name,df in [('prices',p.to_frame()),('events',events),('observations',obs),('summary',summary),('annual',annual),('overlap',windows),('excluded',rejected)]:z.writestr(name+'.csv',df.to_csv())
        z.writestr('protocol.json',json.dumps(protocol,indent=2,default=str))
    st.download_button('Event-Auswertung als ZIP',buf.getvalue(),file_name=ASSETS[asset][1]+'_Events_v1_2_0.zip',mime='application/zip')


def main():
    import streamlit as st
    st.set_page_config(page_title='Multi-Asset Seasonality Lab',layout='wide')
    st.title('Multi-Asset Seasonality Lab v'+VERSION)
    st.write('Kalendermuster der sieben Projekt-Assets untersuchen – separate Research-App.')
    st.warning('Explorative Auswertung: Viele frei gewählte Zeitfenster erzeugen Zufallstreffer. Keine automatischen Signifikanz-, Modell- oder Handelsfreigaben.')
    pair=st.selectbox('Asset',list(SYMBOLS))
    source=st.radio('Datenquelle',['Vorhandene Audit-ZIP / CSV','Yahoo abrufen'],horizontal=True)
    key='seasonality_'+pair
    if source=='Vorhandene Audit-ZIP / CSV':
        upload=st.file_uploader('Research-ZIP oder Tages-CSV',type=['zip','csv'])
        if upload is None:
            st.info('Für USD/CHF und USD/JPY funktionieren die bisherigen Audit-ZIPs. Für andere Assets: Preis-CSV oder ZIP mit Preis-CSV. Spalten: Date und asset_price oder Close. Alternativ Yahoo wählen.');return
        raw=upload.getvalue()
        member=None
        try:
            if upload.name.lower().endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    members=[n for n in z.namelist() if n.lower().endswith('.csv')]
                if not members:raise ValueError('Keine CSV in der ZIP.')
                preferred=ASSETS[pair][1]+'_research_timeseries.csv'
                default=next((i for i,n in enumerate(members) if n.split('/')[-1]==preferred),0)
                member=st.selectbox('Preis-CSV in der ZIP',members,index=default)
            st.caption('Upload: Asset-Zuordnung und Preisart werden nicht aus den Kurswerten erkannt.')
            price_kind=st.selectbox('Preisart der hochgeladenen Reihe',['Ungeprüft','Spot / Kassakurs','Preisindex','Total-Return-Index','Futures / kontinuierliche Futures-Reihe'])
            if not st.checkbox('Die ausgewählte Datei gehört zum oben gewählten Asset.',key='confirm_'+pair+'_'+hashlib.sha256(raw).hexdigest()[:12]):return
            frame=load_bytes(raw,upload.name,pair,member)
        except Exception as e:st.error(str(e));return
        provenance={'source':upload.name,'sha256':hashlib.sha256(raw).hexdigest(),
                    'csv_member':member,'price_kind':price_kind,'asset_identity':'user confirmed; not independently verified'}
    else:
        st.info('Yahoo-Symbol: '+SYMBOLS[pair]+' · '+ASSETS[pair][2])
        st.caption('Yahoo ist eine aktuelle, potenziell revidierte Historie. Kein Nachweis historischer Veröffentlichungszeitpunkte.')
        if st.button('15 Jahre Tagesdaten abrufen'):
            try:
                import yfinance as yf
                started=pd.Timestamp.now(tz='UTC')
                h=yf.Ticker(SYMBOLS[pair]).history(start=str((started-pd.DateOffset(years=15)).date()),
                    interval='1d',auto_adjust=False,back_adjust=False,repair=False,keepna=True,actions=False,timeout=20,raise_errors=True)
                frame=h.reset_index();raw=frame.to_csv(index=False).encode()
                st.session_state[key]=(frame,{'source':'Yahoo '+SYMBOLS[pair],
                    'received_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),'yfinance':yf.__version__,
                    'sha256':hashlib.sha256(raw).hexdigest()})
            except Exception as e:st.error('Abruf fehlgeschlagen: '+str(e));return
        if key not in st.session_state:return
        frame,provenance=st.session_state[key]
    if pair in ['Gold','WTI']:
        st.warning('Futures-Ergebnisse sind Quellen-Saisonalität, keine vollständig rollbereinigte Strategie. Bei Null- oder Negativpreisen wird die Prozentrechnung gesperrt; Werte werden weder entfernt noch künstlich ersetzt.')
    try:p,quality=prepare(frame)
    except Exception as e:st.error(str(e));return
    st.caption(f"{len(p):,} Tageswerte · {quality['first']} bis {quality['last']} · ausgeschlossene Zeilen: {quality['removed_rows']}")
    st.caption('Datum = Anbieter-Tageslabel. Feiertagskalender, identische Session-Schlüsse und PIT-Verfügbarkeit sind nicht verifiziert.')
    mode=st.selectbox('Analyse',['Monate','Wochentage','Kalenderwochen','Freie Datumsspanne','Freie Wochenspanne','Events'])
    if mode=='Events':
        event_dashboard(st,p,pair,provenance,quality)
        return
    custom=None
    if mode=='Freie Datumsspanne':
        a,b=st.columns(2)
        start=a.date_input('Jährlich ab (Jahr wird ignoriert)',date(2000,11,15))
        end=b.date_input('Jährlich bis einschließlich',date(2000,12,15))
        custom=(start.month,start.day,end.month,end.day)
    elif mode=='Freie Wochenspanne':
        sw=st.number_input('Erste ISO-Kalenderwoche',1,53,45)
        ew=st.number_input('Letzte ISO-Kalenderwoche einschließlich',1,53,50);custom=(sw,ew)
    with st.expander('Welche Rendite wird gemessen?',expanded=True):
        st.write('Monat/Woche/Spanne: letzter verfügbarer Schlusskurs VOR dem Beginn bis letzter Schlusskurs innerhalb der Spanne. Jahreswechsel werden unterstützt. Nicht vollständig abgedeckte Kalenderzeiträume werden ausgelassen; ein Wochenende am Datenende kann deshalb auch eine gerade beendete Woche ausschließen.')
        st.write('Wochentag: Rendite vom vorherigen verfügbaren Schlusskurs zum Schlusskurs dieses Tages. Montag enthält typischerweise das Wochenende. Das ist keine Open-to-Close- oder Intraday-Rendite.')
        st.write('Renditen sind Veränderungen der gewählten Preisreihe: keine Spreads, Finanzierung, Slippage oder Positionsgrößen. Fehlende Preise werden nicht aufgefüllt. Offensichtliche Lücken über sieben Kalendertage werden ausgeschlossen; kürzere Quellenlücken können unentdeckt bleiben.')
    obs=observations(p,mode,custom)
    if obs.empty:st.warning('Keine vollständig auswertbaren Zeiträume.');return
    yrs=sorted(obs.year.unique());lo,hi=int(min(yrs)),int(max(yrs))
    if lo<hi:yrange=st.slider('Auswertungsjahre (Startjahr bzw. ISO-Jahr)',lo,hi,(lo,hi))
    else:yrange=(lo,hi)
    obs=obs[obs.year.between(*yrange)]
    if obs.empty:st.warning('Keine Beobachtungen im ausgewählten Bereich.');return
    summary=summarize(obs)
    st.subheader('Ergebnis je Kalendergruppe')
    st.dataframe(summary.round(3));st.bar_chart(summary[['Mittel_%','Median_%']])
    annual=obs.pivot_table(index='year',columns='group',values='return',aggfunc='mean')*100
    st.subheader('Einzeljahre – Mittel der Beobachtungen in %')
    st.dataframe(annual.round(3))
    st.caption('Bei Monaten/Wochen/freien Spannen meist eine Beobachtung je Jahr; bei Wochentagen ein Jahresmittel vieler Tagesrenditen. N ist keine Zahl unabhängiger Tests.')
    st.subheader('Zeitliche Stabilität')
    parts=[]
    for a,b in [(2012,2016),(2017,2020),(2021,2025)]:
        # Purge intervals crossing the period boundary.
        sub=obs[(pd.to_datetime(obs.entry)>=pd.Timestamp(a,1,1))&(pd.to_datetime(obs.exit)<=pd.Timestamp(b,12,31))]
        if not sub.empty:
            tab=summarize(sub).reset_index();tab.insert(0,'Periode',f'{a}–{b}');parts.append(tab)
    stability=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    st.dataframe(stability.round(3))
    with st.expander('Alle einzelnen Beobachtungen'):st.dataframe(obs)
    protocol={'version':VERSION,'asset':pair,'mode':mode,'custom':custom,'years':yrange,
              'yahoo_proxy':{'symbol':SYMBOLS[pair],'description':ASSETS[pair][2]} if source=='Yahoo abrufen' else None,
              'source':provenance,'quality':quality,'exploratory':True,'PIT_verified':False,
              'multiple_testing_adjusted':False,'promotion_authorized':False,
              'return_rule':'previous available close before interval to last inside; weekdays previous-close to close',
              'missing_rule':'no fill; reject explicit missing prices; intervals with calendar gaps>7 days excluded',
              'full_calendar_end_required':True,'costs_included':False}
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name,table in [('prices',p.to_frame()),('observations',obs),('summary',summary),('annual',annual),('stability',stability)]:
            z.writestr(name+'.csv',table.to_csv())
        z.writestr('protocol.json',json.dumps(protocol,indent=2,default=str))
    st.download_button('Auswertung als ZIP herunterladen',buf.getvalue(),
                       file_name=ASSETS[pair][1]+'_Seasonality_v1_2_0.zip',mime='application/zip')
    st.info('Ein auffälliges Muster ist zunächst eine Hypothese. Für eine Bestätigung müssen Zeitraum, Richtung und Kriterien vor einer separaten Validierung feststehen. Dieses Dashboard optimiert keine Gewichte.')


if __name__=='__main__':main()
