#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:49:38 2021

@author: skm
"""
import pandas as pd
import numpy as np
from yahoofinancials import YahooFinancials as yf
from bs4 import BeautifulSoup as bs
import requests
from urllib.request import urlopen
import json
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from yahooquery import Ticker
from millify import millify


def readable_nums(num_list):
    for num in num_list:
        yield millify(num,precision=2)
        
class Indices:
    def __init__(self):
        pass
    
    def get_dia(self):
        """dataframe of info of all tickers in Dow Jones Industrial Average"""
        url = 'https://www.dogsofthedow.com/dow-jones-industrial-average-companies.htm'
        request = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
        soup = bs(request.text, "lxml")
        stats = soup.find('table',class_='tablepress tablepress-id-42 tablepress-responsive')
        pulled_df =pd.read_html(str(stats))[0]
        return pulled_df
    
    
    def get_spy(self):
        """dataframe of info of all tickers in SPY"""
        url = 'https://www.slickcharts.com/sp500'
        request = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
        soup = bs(request.text, "lxml")
        stats = soup.find('table',class_='table table-hover table-borderless table-sm')
        df =pd.read_html(str(stats))[0]
        df['% Chg'] = df['% Chg'].str.strip('()-%')
        df['% Chg'] = pd.to_numeric(df['% Chg'])
        df['Chg'] = pd.to_numeric(df['Chg'])
        return df
    
    def get_qqq(self):
        """dataframe of info of all tickers in QQQ"""
        df = pd.DataFrame()
        urls = ['https://www.dividendmax.com/market-index-constituents/nasdaq-100',
                'https://www.dividendmax.com/market-index-constituents/nasdaq-100?page=2',
                'https://www.dividendmax.com/market-index-constituents/nasdaq-100?page=3']
        for url in urls:
            request = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
            soup = bs(request.text, "lxml")
            stats = soup.find('table',class_='mdc-data-table__table')
            temp =pd.read_html(str(stats))[0]
            df = df.append(temp)
        df.rename(columns={'Market Cap':'Market Cap $bn'},inplace=True)
        df['Market Cap $bn'] =  df['Market Cap $bn'].str.strip("£$bn")
        df['Market Cap $bn'] = pd.to_numeric(df['Market Cap $bn'])
        df = df.sort_values('Market Cap $bn',ascending=False)
        return df
    
    def get_vti(self):
        df = pd.read_csv('vti.csv',header=3)
        return df
    
def get_historical_data(ticker,period,interval):
    yf = Ticker(ticker)
    # pull historical stock data for SPY comparison
    hist = yf.history(period=period,interval=interval).reset_index()
    hist['date'] = list(map(str,hist['date']))
    hist['Day'] = hist['date'].apply(lambda x: x.split()[0])
    hist['avg_price'] = (hist['high']+hist['close']+hist['low'])/3
    
    sma_df = yf.history(period='max',interval='1d').reset_index()
    sma_df['date'] = list(map(str,sma_df['date']))
    sma_df['Day'] = sma_df['date'].apply(lambda x: x.split()[0])
    sma_df['200_sma'] = sma_df['close'].rolling(window=200).mean()
    sma_df['50_sma'] = sma_df['close'].rolling(window=50).mean()

    df = pd.merge(hist,sma_df[['Day','200_sma','50_sma']],on='Day',how='left')
    intraday_df = df.groupby('Day').apply(vwap) 
    df = pd.merge(df,intraday_df,on='date',how='left')
    return df

def get_10_year():
    """get 10-year treasury from Yahoo Finance"""
    ten_yr = yf(['^TNX'])
    ten_yr = ten_yr.get_current_price()
    
    for k,v in ten_yr.items():
        treas = v/100
    return treas

def vwap(x):
    d={}
    d['vwap'] = np.cumsum(x['volume'].values*x['avg_price'].values) / np.cumsum(x['volume'].values)
    d['Day'] = x['Day']
    d['date'] = x['date']
    df = pd.DataFrame(data=d)
    return df

def make_ohlc(ticker,df):
    ohlc_fig = make_subplots(specs=[[{"secondary_y": True}]]) # creates ability to plot vol and $ change within main plot
 
    #include OHLC (already comes with rangeselector)
    ohlc_fig.add_trace(go.Candlestick(x=df['date'],
                     open=df['open'], 
                     high=df['high'],
                     low=df['low'], 
                     close=df['close'],name='OHLC'),secondary_y=True)
    
    ohlc_fig.add_trace(go.Scatter(x=df['date'],y=df['200_sma'],name='200 SMA',line=dict(color='orange')),secondary_y=True)
    ohlc_fig.add_trace(go.Scatter(x=df['date'],y=df['50_sma'],name='50 SMA',line=dict(color='navy')),secondary_y=True)
    
    # include a go.Bar trace for volume
    ohlc_fig.add_trace(go.Bar(x=df['date'], y=df['volume'],name='Volume',marker_color='dodgerblue'),
                    secondary_y=False)
    ohlc_fig.add_trace(go.Scatter(x=df['date'],y=df['vwap'],name='Intraday VWAP',line=dict(color='cyan')),secondary_y=True)
    ohlc_fig.layout.yaxis2.showgrid=False
    ohlc_fig.update_xaxes(type='category',nticks=10,tickangle=15)
    ohlc_fig.update_layout(title_text=f'{ticker} Price Chart',xaxis=dict(rangeslider=dict(visible=False)))
    return ohlc_fig

def make_comp_chart(ticker,df,comps):
    comp_fig = go.Figure()
    comp_fig.add_trace(go.Scatter(x=df['date'],y=df[f'{ticker}_pct_change'],name=f'{ticker}'))
    comp_fig.add_trace(go.Scatter(x=df['date'],y=df['SPY_pct_change'],name='SPY'))
    comp_fig.add_trace(go.Scatter(x=df['date'],y=df['DIA_pct_change'],name='DIA'))
    comp_fig.add_trace(go.Scatter(x=df['date'],y=df['QQQ_pct_change'],name='QQQ'))
    comp_fig.update_xaxes(type='category')
    comp_fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        ),
        yaxis = dict(
            tickformat = '.0%',
            autorange=True, # PLOTLY HAS NO AUTORANGE FEATURE, TRYING TO IMPLEMENT MANUALLY BUT NO DICE
            fixedrange=False, # PLOTLY HAS NO AUTORANGE FEATURE, TRYING TO IMPLEMENT MANUALLY BUT NO DICE
            ),
        title_text=f'{ticker} vs. Indices Historical Prices',
    )

    return comp_fig
    
    

class DDM:
    def __init__(self):
        pass
    
    def get_dividend_growth_rate(self,data):
        deltas=[]
        for col in range(0,4): #average three years of growth
            try:
                div = (data.iloc[:,col]['dividendsPaid']/data.iloc[:,col+1]['dividendsPaid'])-1
                deltas.append(div)
                growth=np.mean(deltas)
                return growth
            except IndexError:
                break
            
    def get_cost_of_equity(self,ticker,beta):
      
        if type(beta) is str:
           beta=1.75
            
        rm= 0.085
        rfr = get_10_year()
        re= rfr+beta*(rm-rfr)
        print("Strategisk Valuation based on the following:")
        print(f"Risk Free Rate: {round(rfr*100,2)}%")
        print(f"Expected Market Return: {round(rm*100,2)}%")
        return re

class DCF:
    def __init__(self):
        pass
    
    def get_jsonparsed_data(self,url):
        response = urlopen(url)
        data = response.read().decode("utf-8")
        return json.loads(data)

    def get_tax_rate(self,ticker,key):
        data = pd.DataFrame(self.get_jsonparsed_data(f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=120&apikey={key}'))
        earnings = data.iloc[0]['ebitda']
        taxes_paid = data.iloc[0]['incomeTaxExpense']
        tax_rate = taxes_paid/earnings
        return tax_rate
    
    def get_wacc(self,total_debt,equity,debt_pmt,tax_rate,beta,rfr,ticker):
        
        if type(beta) is str:
            beta=1.75
            
        rm= 0.085
        re= rfr+beta*(rm-rfr)
        
        if total_debt<1 or debt_pmt<1:
            wacc = re
        else:
            rd= debt_pmt/total_debt
            value = total_debt+equity
            wacc = (equity/value*re) + ((total_debt/value * rd) * (1 - tax_rate))
        
        # print(f"\n{ticker.upper()} Discounted Cash Flows based on the following:")
        # print(f"Market Return Rate: {round(rm*100,2)}%")
        # print(f"Risk Free Rate: {round(rfr*100,2)}%")
        return wacc
        
        
    def intrinsic_value(self,cash_flow_df, total_debt, cash_and_ST_investments, 
                                      data, discount_rate,shares_outstanding,name):
        
        def calc_cashflow():
            cf = cash_flow
            for year in range(1,6):
                cf *= (1 + st_growth)
                dcf = round(cf/((1 + discount_rate)**year),0)   
                yield cf,dcf
            for year in range(6,11):
                cf *= (1 + lt_growth)
                dcf = round(cf/((1 + discount_rate)**year),0)   
                yield cf,dcf
            for year in range(11,21):
                cf *= (1 + terminal_growth)
                dcf = round(cf/((1 + discount_rate)**year),0)   
                yield cf,dcf
            
        try:
           st_growth =  float(data['EPS next 5Y'].str.strip('%')) / 100
        
        except ValueError: # means EPS next 5Y is string, so cannot be divided, onto substitute method
            st_growth = 0.15 # set to 15%, unavailable EPS data means large / volatile growth

        lt_growth = st_growth*0.5 # 1/2 of initial growth
        if lt_growth <= 0.05:
            terminal_growth = 0.03
        elif lt_growth >=0.10:
            terminal_growth = 0.5*lt_growth
        else:
            terminal_growth = 0.5*lt_growth
    
        cash_flow=cash_flow_df.iloc[-1]['FreeCashFlow']
        
        year_list = [i for i in range(1,21)]
        cashflows = list(calc_cashflow())
        cf_list = [i[0] for i in cashflows]
        dcf_list = [i[1] for i in cashflows]
         
        intrinsic_value = (sum(dcf_list) - total_debt + cash_and_ST_investments)/shares_outstanding
        df = pd.DataFrame.from_dict({'Year Out': year_list, 'Future Value': cf_list, 'Present Value': dcf_list})
        
        fig = px.bar(df,x='Year Out',y=['Future Value','Present Value'],barmode='group',color_discrete_sequence=['navy','paleturquoise'])
        fig.update_layout(title=f'{name} Projected Free Cash Flows',yaxis_title='Free Cash Flow ($)',legend_title='')
        y1 = list(readable_nums(df['Future Value']))
        y2 = list(readable_nums(df['Present Value']))
        texts = [y1,y2]
        for i, t in enumerate(texts):
            fig.data[i].text = t
            fig.data[i].textposition = 'outside'
    
        return fig, intrinsic_value, st_growth, lt_growth,terminal_growth
    
    
    def get_fmp_dcf(self,ticker,key):
        dcf=self.get_jsonparsed_data(f'https://www.financialmodelingprep.com/api/v3/discounted-cash-flow/{ticker}?apikey={key}')
        return dcf
    
    def get_q_cf(self,ticker,key):
        base_url = "https://financialmodelingprep.com/api/v3/"                       
        q_cf_stmt = pd.DataFrame(self.get_jsonparsed_data(base_url+f'cash-flow-statement/{ticker}?period=quarter&apikey={key}'))
        q_cf_stmt = q_cf_stmt.set_index('date').iloc[:4] # extract for last 4 quarters
        q_cf_stmt = q_cf_stmt.apply(pd.to_numeric, errors='coerce')
        
        return q_cf_stmt
    
    def get_q_bs(self,ticker,key):
        base_url = "https://financialmodelingprep.com/api/v3/"
        q_bs = pd.DataFrame(self.get_jsonparsed_data(base_url+f'balance-sheet-statement/{ticker}?period=quarter&apikey={key}'))
        q_bs = q_bs.set_index('date')
        q_bs = q_bs.apply(pd.to_numeric, errors='coerce')
        return q_bs
    
    def get_annual_cf(self,ticker,key):
        base_url = "https://financialmodelingprep.com/api/v3/"
        cash_flow_statement = pd.DataFrame(self.get_jsonparsed_data(base_url+f'cash-flow-statement/{ticker}?apikey={key}'))
        cash_flow_statement = cash_flow_statement.set_index('date')
        cash_flow_statement = cash_flow_statement.apply(pd.to_numeric, errors='coerce')
        return cash_flow_statement
    
    
    

class FinViz:
    def __init__(self):
        pass
    
    def fundamentals(self,ticker):
        try:
            url = f'https://www.finviz.com/quote.ashx?t={ticker.lower()}'
            request = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
            soup = bs(request.text, "lxml")
            stats = soup.find('table',class_='snapshot-table2')
            fundamentals =pd.read_html(str(stats))[0]
            
            # Clean up fundamentals dataframe
            fundamentals.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
            colOne = []
            colLength = len(fundamentals)
            for k in np.arange(0, colLength, 2):
                colOne.append(fundamentals[f'{k}'])
            attrs = pd.concat(colOne, ignore_index=True)
        
            colTwo = []
            colLength = len(fundamentals)
            for k in np.arange(1, colLength, 2):
                colTwo.append(fundamentals[f'{k}'])
            vals = pd.concat(colTwo, ignore_index=True)
            
            fundamentals = pd.DataFrame()
            fundamentals['Attributes'] = attrs
            fundamentals[f'{ticker.upper()}'] = vals
            fundamentals = fundamentals.set_index('Attributes')
            fundamentals = fundamentals.T
            
            # catch known duplicate column name EPS next Y
           # fundamentals.rename(columns={fundamentals.columns[28]:'EPS growth next Y'},inplace=True)
            return fundamentals
    
        except Exception as e:
            return e
    
    def get_ratings(self,ticker):
        url = f'https://www.finviz.com/quote.ashx?t={ticker.lower()}'
        request = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
        soup = bs(request.text, "lxml")
        stats = soup.find('table',class_='fullview-ratings-outer')
        ratings =pd.read_html(str(stats))[0]
        ratings['date'] = ratings[0].apply(lambda x: x.split()[0][:9])
        ratings['rating'] = ratings[0].apply(lambda x: x.split()[0][9:])
        ratings['firm'] = ratings[0].apply(lambda x: x.split()[1])
        ratings.drop(columns=0,inplace=True)
        return ratings
    
    def fundamental_metric(self,soup, metric):
        # the table which stores the data in Finviz has html table attribute class of 'snapshot-td2'
        return soup.find(text = metric).find_next(class_='snapshot-td2').text
    
    def get_data(self,ticker,metrics):
        try:
            url = ("http://finviz.com/quote.ashx?t=" + ticker.lower())
            soup = bs(requests.get(url,headers={'User-Agent':\
                                         'Mozilla/5.0'}).content,
                                          features='lxml')
            finviz = {}        
            for m in metrics:   
                finviz[m] = self.fundamental_metric(soup,m)
            for key, value in finviz.items():
                # replace percentages
                if (value[-1]=='%'):
                    finviz[key] = value[:-1]
                    finviz[key] = float(finviz[key])
                # billion
                if (value[-1]=='B'):
                    finviz[key] = value[:-1]
                    finviz[key] = float(finviz[key])*1000000000  
                # million
                if (value[-1]=='M'):
                    finviz[key] = value[:-1]
                    finviz[key] = float(finviz[key])*1000000
                try:
                    finviz[key] = float(finviz[key])
                except:
                    pass 
        except Exception as e:
            print(e)
            print(f'Usuccessful parsing {ticker} data.')        
        return finviz
            

class YahooFin:
    def __init__(self):
        pass
    
    def format_stmt(self,statement):
        new = pd.DataFrame()
        for year in statement:
            temp=pd.DataFrame(year)
            new=new.join(temp,how='outer')
        return new.T.reset_index()