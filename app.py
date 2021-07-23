#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 08:15:10 2021

@author: skm
"""

from utils import DCF, FinViz, Indices,get_10_year, get_historical_data,make_ohlc
from yahooquery import Ticker
import pandas as pd
from functools import reduce
import numpy as np
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from millify import millify




# instantiate dcf and finviz classes
dcf = DCF()
fv = FinViz() 

# obtain tickers in all major index ETFs
index = Indices()
vti = index.get_vti()
tickers = vti['TICKER']


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(children=[
    html.Div([
        html.H4(children='Discounted Cash Flows Model'),
        html.H6(children= 'Code can be found here: https://github.com/skmcwilliams/dcf_app'),
        
        dcc.Dropdown(
            id='ticker',
            options = [
                {'label': i,'value': i} for i in tickers
            ],
            # value='IBM',
            searchable=True,
            clearable=True,
            placeholder='Select or type ticker for valuation'
            ),
        
       # dcc.Graph(id = 'ohlc_plot'),
        ]),

    #html.Div([
     #   dcc.Dropdown(
      #      id='comps',
       #     options = [
        #        {'label': i,'value': i} for i in tickers
         #   ],
          #  # value='IBM',
           # searchable=True,
            #clearable=True,
            #multi=True,
           # placeholder='Select or type tickers for comparison'
           # ),
       # dcc.Dropdown(
        #    id='period',
         #   options = [
          #      {'label': i,'value': i} for i in ['1d', '5d', '7d', '60d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
          #  ],
            # value='5y',
           # searchable=True,
            #clearable=True,
            #placeholder='Select Period'
            #),
        #dcc.Dropdown(
         #   id='interval',
         #   options = [
         #       {'label': i,'value': i} for i in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
          #  ],
            # value='1d',
           # searchable=True,
            #clearable=True,
            #placeholder='Select Interval'
            #),
        #dcc.Graph(id = 'comp_plot'),
   # ]),
    html.Div([
        dcc.Graph(id = 'hist_cashflows'),
    ]),
    html.Div([
        dcc.Graph(id='proj_cashflows'),
        html.H6(id='text'),
    ]),
    html.Div([
        dcc.Graph(id='yahoo_plot'),
    ]),
    html.Div([
        dcc.Graph(id='finviz_plot'),
    ]),
])


""" CALLBACK FOR PRICE CHART"""
@app.callback(dash.dependencies.Output(component_id='ohlc_plot', component_property= 'figure'),
              [dash.dependencies.Input(component_id='ticker', component_property= 'value')])
def update_ohlc_plot(ticker_value):

    # gather historical data for ticker and indices data
    df = get_historical_data(str(ticker_value),'5Y','1d',True)

    ohlc_fig = make_subplots(specs=[[{"secondary_y": True}]]) # creates ability to plot vol and $ change within main plot
 
    #include OHLC (already comes with rangeselector)
    ohlc_fig.add_trace(go.Candlestick(x=df['date'],
                     open=df[f'{ticker_value}_open'], 
                     high=df[f'{ticker_value}_high'],
                     low=df[f'{ticker_value}_low'], 
                     close=df[f'{ticker_value}_close'],name='Daily Candlestick'),secondary_y=True)
    
    ohlc_fig.add_trace(go.Scatter(x=df['date'],y=df[f'{ticker_value}_200_sma'],name='200-day SMA',line=dict(color='cyan')),secondary_y=True)
    ohlc_fig.add_trace(go.Scatter(x=df['date'],y=df[f'{ticker_value}_50_sma'],name='50-day SMA',line=dict(color='navy')),secondary_y=True)
    
    # include a go.Bar trace for volume
    ohlc_fig.add_trace(go.Bar(x=df['date'], y=df[f'{ticker_value}_volume'],name='Volume'),
                    secondary_y=False)
   
    ohlc_fig.layout.yaxis2.showgrid=False
    ohlc_fig.update_xaxes(type='category')
    ohlc_fig.update_layout(
        title_text=f"{vti['HOLDINGS'][vti['TICKER']=='TSLA'].iloc[0]} Price Chart",
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
        )
    )
    return ohlc_fig

""" CALLBACK FOR COMPARISON CHART"""
@app.callback(dash.dependencies.Output(component_id='comp_plot', component_property= 'figure'),
              [dash.dependencies.Input(component_id='ticker', component_property= 'value'),
              dash.dependencies.Input(component_id='comps', component_property= 'value'),
              dash.dependencies.Input(component_id='period', component_property= 'value'),
              dash.dependencies.Input(component_id='interval', component_property= 'value')])
def update_comp_chart(ticker_value,comps_value,period_value,interval_value):
    ticker_hist = get_historical_data(ticker_value,period_value,interval_value,True)
    data_frames = [ticker_hist]
    for comp in comps_value:
        df = get_historical_data(comp,period_value,interval_value,True)
        data_frames = data_frames.append(df)
    
    # marge indices df on ticker df
    df = reduce(lambda  left,right: pd.merge(left,right,on=['date'],
                                            how='inner'), data_frames)

    df[f'{ticker_value}_pct_change'] = (df[f'{ticker_value}_close']-df[f'{ticker_value}_close'].iloc[0])/df[f'{ticker_value}_close'].iloc[0]
    ticks = [i.split('_')[0] for i in df.columns if "symbol" in i]
    comp_fig = go.Figure()
    for comp in ticks:
        df[f'{comp}_pct_change'] = df[f'{comp}_close'].apply(lambda x: (x - df[f'{comp}_close'].iloc[0])/df[f'{comp}_close'].iloc[0])
        comp_fig.add_trace(go.Scatter(x=df['date'],y=df[f'{comp}_pct_change'],name=f'{comp}'))

    comp_fig.update_xaxes(type='category')
    comp_fig.update_layout(
        yaxis = dict(
            tickformat = '.0%',
            autorange=True, # PLOTLY HAS NO AUTORANGE FEATURE, TRYING TO IMPLEMENT MANUALLY BUT NO DICE
            fixedrange=False, # PLOTLY HAS NO AUTORANGE FEATURE, TRYING TO IMPLEMENT MANUALLY BUT NO DICE
            ),
        title_text=f"{ticker_value} vs. {comps_value} Historical Prices",
    )

    return comp_fig

"""CALLBACK FOR HISTORICAL CASHFLOWS BAR CHART"""
@app.callback(dash.dependencies.Output(component_id='hist_cashflows', component_property= 'figure'),
              [dash.dependencies.Input(component_id='ticker', component_property= 'value')])
def update_historical_plot(ticker_value):
    yf = Ticker(ticker_value)
    #GET QUARTERLY CASH FLOW
    cash_flow_df = yf.cash_flow('a',True).reset_index()
    cash_flow_df = cash_flow_df.drop_duplicates(subset='asOfDate')
    cash_flow_df['asOfDate'] = list(map(str,cash_flow_df['asOfDate']))
    cash_flow_df['year'] = cash_flow_df['asOfDate'].apply(lambda x: x.split('-')[0])
    cash_flow_df.insert(0,'Period',cash_flow_df['year']+'-'+cash_flow_df['periodType'])

    # PLOT HISTORICAL CASH FLOWS
    millified = [millify(i,precision=2) for i in cash_flow_df['FreeCashFlow']]
    name = vti['HOLDINGS'][vti['TICKER']==ticker_value].iloc[0]
    cf_fig = px.bar(data_frame=cash_flow_df,x='Period',y='FreeCashFlow',orientation='v',
    title=f"{name} Historical Free Cash Flows",text=millified)
    return cf_fig

@app.callback(dash.dependencies.Output(component_id='proj_cashflows', component_property= 'figure'),
              dash.dependencies.Output(component_id='text', component_property= 'children'),
              [dash.dependencies.Input(component_id='ticker', component_property= 'value')])
def update_pcf_chart(ticker_value):
    if ticker_value=='AAPL':
        key='demo'
    else:
        keys= ['3da65237f17cee96481b2251702509d1','3a1649ceeafc5888ec99181c59cb5f8b']
        key= np.random.choice(keys)
    yf = Ticker(ticker_value)
    finviz_df = fv.fundamentals(ticker_value)
    #GET QUARTERLY CASH FLOW AND BALANCE SHEET
    balance_sheet= yf.balance_sheet('q',False)
    cash_flow_df = yf.cash_flow('a',True).reset_index()


    # CREATE VARIABLES TO PRINT AT BEGINNING
    try:
        total_debt = balance_sheet.iloc[-1]['TotalDebt']
    except KeyError:
        total_debt=0

    try:
        debt_payment = np.nan_to_num(cash_flow_df.iloc[-1]['RepaymentOfDebt']*-1)
    except KeyError:
        debt_payment = 0

        
    try:
        cash_and_ST_investments = balance_sheet.iloc[-1]['CashAndCashEquivalents']
        while pd.isnull(cash_and_ST_investments):
            for i in range(1,len(balance_sheet)):
                cash_and_ST_investments = balance_sheet.iloc[-i]['CashAndCashEquivalents']
    except KeyError:
        cash_and_ST_investments = balance_sheet.iloc[-1]['CashCashEquivalentsAndShortTermInvestments']
        while pd.isnull(cash_and_ST_investments):
            for i in range(1,len(balance_sheet)):
                cash_and_ST_investments = balance_sheet.iloc[-i]['CashCashEquivalentsAndShortTermInvestments']
    
    
    """
    cash_flow = cash_flow_df.iloc[-1]['FreeCashFlow']

    try:
        quick_ratio = balance_sheet.iloc[-1]['CurrentAssets']/balance_sheet.iloc[-1]['CurrentLiabilities']
    except KeyError:
        quick_ratio = 0
    """  
    # SET DCF VARIABLES
    total_equity = yf.summary_detail[ticker_value]['marketCap']
    try:
        beta = yf.summary_detail[ticker_value]['beta']
    except KeyError:
        beta=2.0
    current_price = float(finviz_df['Price'])
    shares_outstanding = total_equity/current_price
    tax_rate = dcf.get_tax_rate(ticker_value,key)
    treasury = get_10_year()
    wacc = dcf.get_wacc(total_debt,total_equity,debt_payment,tax_rate,beta,treasury,ticker_value)

    # DCF VALUATION
    name = vti['HOLDINGS'][vti['TICKER']==ticker_value].iloc[0]
    intrinsic_value = dcf.calculate_intrinsic_value(ticker_value,
                                                cash_flow_df, total_debt, 
                                                cash_and_ST_investments, 
                                                finviz_df, wacc,shares_outstanding,name)
    return intrinsic_value[0],f"Based on the Following:\
        Total Debt: ${millify(total_debt,2)} | \
        Tax Rate: {round(tax_rate*100,0)}% | \
        Cash and Short-Term Investments: ${millify(cash_and_ST_investments,2)} | \
        Beta: {round(beta,2)} | \
        Market Rate of Return: 8.50% | \
        Risk Free Rate (10-year Treasury): {round(treasury*100,2)}% | \
        Resulting Valuation for {ticker_value}: ${round(intrinsic_value[1],2)}/share | \
        Margin to Current Price: {round(((intrinsic_value[1]-current_price)/current_price)*100,2)}%"
    
"""CALLBACK FOR YAHOO RATINGS PLOT"""
@app.callback(dash.dependencies.Output(component_id='yahoo_plot', component_property= 'figure'),
              [dash.dependencies.Input(component_id='ticker', component_property= 'value')])
def update_yahoo(ticker_value):
    yf = Ticker(ticker_value)
    yahoo_ratings = yf.recommendation_trend.reset_index()
    yahoo_ratings.rename(columns={'period':'Period'},inplace=True)
    yahoo_ratings.at[0,'Period'] = 'Current'
    yahoo_ratings.at[1,'Period'] = '1 Month Back' 
    yahoo_ratings.at[2,'Period'] = '2 Months Back'
    yahoo_ratings.at[3,'Period'] = '3 Months Back' 
    name = vti['HOLDINGS'][vti['TICKER']==ticker_value].iloc[0]
    ratings_fig = px.bar(yahoo_ratings,x='Period',y=['strongBuy','buy','hold','sell','strongSell'],
                            title=f"{name} Yahoo Recommendation Trends")
    ratings_fig.update_layout(legend_title='')
    return ratings_fig

"""CALLBACK FOR FINVIZ RATINGS PLOT"""
@app.callback(dash.dependencies.Output(component_id='finviz_plot', component_property= 'figure'),
              [dash.dependencies.Input(component_id='ticker', component_property= 'value')])
def update_finviz(ticker_value):
    name = vti['HOLDINGS'][vti['TICKER']==ticker_value].iloc[0]
    finviz_ratings = fv.get_ratings(ticker_value)
    finviz_ratings = finviz_ratings.drop_duplicates(subset='firm') #ensure latest rating by each firm
    finviz_ratings = finviz_ratings[finviz_ratings['date'].str.endswith('21')] #only recent ratings
    fv_fig = px.histogram(finviz_ratings, x="rating",title=f"{name} 2021 Investment Bank Ratings per FinViz")
    return fv_fig


if __name__=='__main__':
   app.run_server(debug=True)

    