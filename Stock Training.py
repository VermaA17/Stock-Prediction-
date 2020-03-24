import numpy as np 
import pandas as pd
from datetime import date
from pmdarima.arima import auto_arima
from dateutil.rrule import rrule, DAILY

src = 'Stock_AI_Project/'

res1=pd.read_csv(src+'main_data.csv')
res1['Date'] = pd.to_datetime(res1['Date'], format='%Y-%m-%d')
res1['Close']=res1['Close'].fillna(method='ffill')
res1 = res1.sort_values(by ='Stock Symbol')
res1=res1[~(res1['Stock Symbol'].isin(['LHO',
'PAY',
'VVC',
'DM',
'BLH',
'AEUA',
'AED',
'AET',
'ANTX',
'SEMG',
'KND',
'NORD',
'CBS',
'TVPT',
'OA',
'DNB',
'SGF',
'EEP',
'GPT',
'PHH']))]

a = date(2019, 1, 1)
b = date(2019, 12, 20)

pl = [['Date','Stock','next_day_forecast','AIC','BIC','Max','Min','Mean','Median','next_day_value','last_day_value']]
for k in res1['Stock Symbol'].unique():
   
    res1_2 = res1.loc[res1['Stock Symbol'] == k].drop(columns = 'Stock Symbol')
    #a=res1_2['Date'].max() - pd.DateOffset(months=6) 
    #b=res1_2['Date'].max() - pd.DateOffset(days=10)
    for dt in rrule(DAILY, dtstart=a, until=b,interval=10):
        res1_2['Date'] = pd.to_datetime(res1_2['Date'], format='%Y-%m-%d', errors='coerce')
        df2t = res1_2[res1_2['Date'] <=dt]
        df2v = res1_2[res1_2['Date'] > dt]
        df2v = df2v.set_index('Date')
        yo=df2t['Close'].max()
        jo=df2t['Close'].min()
        ko=df2t['Close'].mean()
        fo=df2t['Close'].median()
        df2t = df2t.set_index('Date')
        df2t=df2t.sort_values(by="Date")
        df2v=df2v.sort_values(by="Date")
        model = auto_arima(df2t, trace=True, error_action='ignore', suppress_warnings=True)
        model.fit(df2t)
        next_day_forecast = model.predict(n_periods=1)[0]
        AIC = model.aic()
        BIC =model.bic()
        # df2t.to_csv("Inspect.csv")
        # df2v.to_csv("Inspectt.csv")
        next_day_value = df2v.iloc[:1,].values[0][0]
        last_day_value = df2t.iloc[-1:,].values[0][0]
        pl.append([dt,k,next_day_forecast, AIC,BIC,yo,jo,ko,fo,next_day_value,last_day_value]) 
    Final=pd.DataFrame(pl)
    Final.to_csv(src+'defo.csv')


