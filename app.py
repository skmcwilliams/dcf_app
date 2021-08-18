#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 08:15:10 2021

@author: skm
"""

from utils import DCF, FinViz,get_10_year, get_historical_data,make_ohlc,readable_nums
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

# obtain tickers in all major index ETFs
ticker_df = pd.read_csv('tickers.csv')
tickers = ticker_df['Symbol']
periods = ['1d', '5d', '7d', '60d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
rates = [0.05,0.075,0.10,0.15,0.2]

# STANDARD DASH APP LANGUAGE
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# CUSTOM DASH APP LANGUAGE
app.layout = html.Div(children=[
    html.Div([
        html.H4(children='Stock Visualization and Discounted Cash Flows Model'),
        html.H6(children= 'If a chart does not update or appears blank, no data is available'),
        dcc.Markdown(children='Select Ticker below for valuation and plotting'),
        
        dcc.Dropdown(
            id='ticker',
            options = [
                {'label': i,'value': i} for i in tickers if '^' not in i and '/' not in i
            ],
            searchable=True,
            clearable=True,
            placeholder='Select or type ticker for valuation'
            ),
        dcc.Markdown(children='Select period and interval for price chart'),
        dcc.Dropdown(
            id='period',
            options = [
                {'label': i,'value': i} for i in periods
            ],
            searchable=True,
            clearable=True,
            placeholder='Select time period of data'
            ),
        
        dcc.Dropdown(
            id='interval',
            options = [
                {'label': i,'value': i} for i in intervals
            ],
            searchable=True,
            clearable=True,
            placeholder='Select interval between data points'
            ),
        
        dcc.Graph(id = 'price_plot'),
        ]),
    html.Div([
        dcc.Graph(id = 'hist_cashflows'),
    ]),
    html.Div([
        dcc.Graph(id='yahoo_earnings_plot'),
    ]),
    html.Div([
        dcc.Markdown(children='Company Snapshot'),
        dcc.Markdown(children='Select desired rate of return for valuation'),
    dcc.Dropdown(
            id='return_rate',
            options = [
                {'label': f"{i*100}%",'value': i} for i in rates
            ],
            searchable=True,
            clearable=True,
            placeholder='Desired rate of return'
            ),
    ]),
    html.Div([
        dcc.Graph(id='data_table'),
    ]),
    html.Div([
        dcc.Graph(id='proj_cashflows'),
    ]),
    html.Div([
        dcc.Markdown(children='Valuation Result'),
    ]),
    html.Div([
        dcc.Graph(id='calcs_table')
    ]),
    html.Div([
        dcc.Graph(id='yahoo_ratings_plot'),
    ]),
    html.Div([
        dcc.Graph(id='finviz_plot'),
        dcc.Markdown(children='All data collected via Yahoo Finance and FinViz, please see code for detail: https://github.com/skmcwilliams/dcf_app'),
    ]),
])


""" CALLBACK FOR PRICE CHART"""
@app.callback(dash.dependencies.Output(component_id='price_plot', component_property= 'figure'),
              [dash.dependencies.Input(component_id='ticker', component_property= 'value'),
              dash.dependencies.Input(component_id='period', component_property= 'value'),
              dash.dependencies.Input(component_id='interval', component_property= 'value')])
def update_price_plot(ticker_value,period_value,interval_value):
    names = ticker_df['Name'][ticker_df['Symbol']==ticker_value].iloc[0].split()[:-2]
    name = ' '.join(names)
    # gather historical data for ticker and indices data
    df = get_historical_data(ticker_value,period_value,interval_value)

    price_fig = make_ohlc(ticker_value,df)
    return price_fig

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
    millified = list(readable_nums(cash_flow_df['FreeCashFlow']))
    names = ticker_df['Name'][ticker_df['Symbol']==ticker_value].iloc[0].split()[:-2]
    name = ' '.join(names)
    cf_fig = px.bar(data_frame=cash_flow_df,x='Period',y='FreeCashFlow',orientation='v',color_discrete_sequence=['navy'],
    title = f"{name} Historical Free Cash Flows",text=millified,labels={'FreeCashFlow':'Free Cash Flow ($)'})
    return cf_fig

"""CALLBACK FOR YAHOO EARNINGS PLOT"""
@app.callback(dash.dependencies.Output(component_id='yahoo_earnings_plot', component_property= 'figure'),
              [dash.dependencies.Input(component_id='ticker', component_property= 'value')])
def update_yahoo_earnings(ticker_value):
    yf = Ticker(ticker_value)
    yahoo_earnings = yf.earning_history.reset_index()
    yahoo_earnings.rename(columns={'period':'Period'},inplace=True)
    yahoo_earnings.at[0,'Period'] = '4 Quarters Back'
    yahoo_earnings.at[1,'Period'] = '3 Quarters Back'
    yahoo_earnings.at[2,'Period'] = '2 Quarters Back'
    yahoo_earnings.at[3,'Period'] = '1 Quarter Back'

    names = ticker_df['Name'][ticker_df['Symbol']==ticker_value].iloc[0].split()[:-2]
    name = ' '.join(names)
    earnings_fig = px.bar(yahoo_earnings,x='Period',y=['epsActual','epsEstimate'],barmode='group',
                            color_discrete_sequence=['navy','paleturquoise'],
                            title=f"{name} Quarterly Earnings Per Share")
    earnings_fig.update_layout(legend_title='',yaxis_title='Earnings ($)')

    y1 = list(readable_nums(yahoo_earnings['epsActual']))
    y2 = list(readable_nums(yahoo_earnings['epsEstimate']))
    texts = [y1,y2]
    for i, t in enumerate(texts):
        earnings_fig.data[i].text = t
        earnings_fig.data[i].textposition = 'inside'

    return earnings_fig

@app.callback(dash.dependencies.Output(component_id='data_table', component_property= 'figure'),
              dash.dependencies.Output(component_id='proj_cashflows', component_property= 'figure'),
              dash.dependencies.Output(component_id='calcs_table', component_property= 'figure'),
              [dash.dependencies.Input(component_id='ticker', component_property= 'value'),
              dash.dependencies.Input(component_id='return_rate', component_property= 'value')])
def update_pcf_chart(ticker_value,return_rate_value):
    dcf = DCF()
    fv = FinViz() 
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
    
    try:
        quick_ratio = balance_sheet.iloc[-1]['CurrentAssets']/balance_sheet.iloc[-1]['CurrentLiabilities']
    except KeyError:
        quick_ratio = 0
    
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
    wacc = dcf.get_wacc(total_debt,total_equity,debt_payment,tax_rate,beta,treasury,return_rate_value)

    # DCF VALUATION
    names = ticker_df['Name'][ticker_df['Symbol']==ticker_value].iloc[0].split()[:-2]
    name = ' '.join(names)
    intrinsic_value = dcf.intrinsic_value(cash_flow_df, total_debt, cash_and_ST_investments,finviz_df, wacc,shares_outstanding,name)
                                                

    metrics = {'Metric':['Total Debt','Tax Rate','Cash and Short-Term Investments','Quick Ratio','WACC','Beta','Risk Free Rate'],
        'Value':[f'${millify(total_debt,2)}',f'{round(tax_rate*100,2)}%',f'${millify(cash_and_ST_investments,2)}',round(quick_ratio,2),f'{round(wacc*100,2)}%',round(beta,2),f'{round(treasury*100,2)}%']}
    
    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_fig = go.Figure(data=[go.Table(
                header=dict(values=list(metrics_df.columns),
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[metrics_df[i] for i in metrics_df.columns],
                        fill_color='silver',
                        align='left'))
                ])

    calculations = {'Assumptions/Valuation': ['Years 1-5 Growth Rate Assumption','Years 6-10 Growth Rate Assumption','Years 11-20 Growth Rate Assumption',f"{ticker_value} Valuation",'Margin to Current Price'],
                    'Value': [f'{round(intrinsic_value[2]*100,2)}%',f'{round(intrinsic_value[3]*100,2)}%',f'{round(intrinsic_value[4]*100,2)}%',f'${round(intrinsic_value[1],2)}/share',f"{round(((intrinsic_value[1]-current_price)/current_price)*100,2)}%"]}

    calcs_df = pd.DataFrame.from_dict(calculations)
    calcs_fig = go.Figure(data=[go.Table(
                header=dict(values=list(calcs_df.columns),
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[calcs_df[i] for i in calcs_df.columns],
                        fill_color='silver',
                        align='left'))
                ])
    return metrics_fig,intrinsic_value[0],calcs_fig
    
"""CALLBACK FOR YAHOO RATINGS PLOT"""
@app.callback(dash.dependencies.Output(component_id='yahoo_ratings_plot', component_property= 'figure'),
              [dash.dependencies.Input(component_id='ticker', component_property= 'value')])
def update_yahoo_ratings(ticker_value):
    yf = Ticker(ticker_value)
    yahoo_ratings = yf.recommendation_trend.reset_index()
    yahoo_ratings.rename(columns={'period':'Period'},inplace=True)
    yahoo_ratings.at[0,'Period'] = 'Current'
    yahoo_ratings.at[1,'Period'] = '1 Month Back' 
    yahoo_ratings.at[2,'Period'] = '2 Months Back'
    yahoo_ratings.at[3,'Period'] = '3 Months Back' 
    names = ticker_df['Name'][ticker_df['Symbol']==ticker_value].iloc[0].split()[:-2]
    name = ' '.join(names)
    ratings_fig = px.bar(yahoo_ratings,x='Period',y=['strongBuy','buy','hold','sell','strongSell'],
                                    title=f"{name} Recommendation Trend",color_discrete_sequence=['green','palegreen','silver','yellow','red'])
    ratings_fig.update_layout(legend_title='',yaxis_title='Count')

    texts = [yahoo_ratings['strongBuy'],yahoo_ratings['buy'],yahoo_ratings['hold'],yahoo_ratings['sell'],yahoo_ratings['strongSell']]
    for i, t in enumerate(texts):
        ratings_fig.data[i].text = t
        ratings_fig.data[i].textposition = 'inside'
    return ratings_fig

"""CALLBACK FOR FINVIZ RATINGS PLOT"""
@app.callback(dash.dependencies.Output(component_id='finviz_plot', component_property= 'figure'),
              [dash.dependencies.Input(component_id='ticker', component_property= 'value')])
def update_finviz(ticker_value):
    fv = FinViz() 
    names = ticker_df['Name'][ticker_df['Symbol']==ticker_value].iloc[0].split()[:-2]
    name = ' '.join(names)
    finviz_ratings = fv.get_ratings(ticker_value)
    finviz_ratings = finviz_ratings[finviz_ratings['rating'].isin(['Upgrade','Downgrade'])]
    year = str(finviz_ratings['date'].iloc[0][-2:])
    finviz_ratings = finviz_ratings[finviz_ratings['date'].str.endswith(year)] #only same-year ratings
    finviz_ratings = finviz_ratings['rating'].value_counts().to_frame()
    # finviz_ratings = finviz_ratings.drop_duplicates(subset='firm') #ensure latest rating by each firm
    
    # fv_fig = px.histogram(finviz_ratings, x="rating",title=f"{name} 2021 Investment Bank Ratings",color_discrete_sequence=['navy'],labels={'ratings':'Rating'})
   #  fv_fig.update_layout(yaxis_title='Count')
    fv_fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = float(round((finviz_ratings.at['Upgrade','rating']/finviz_ratings['rating'].sum())*100,2)),
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{name} 20{year} Sentiment Gauge"},
        gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "black"},
        'bar': {'color': "gray"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "black",
        'steps': [
            {'range': [0, 50], 'color': 'red'},
            {'range': [50, 75], 'color': 'yellow'},
            {'range': [75, 100], 'color': 'lightgreen'}]}))
    
    return fv_fig


if __name__=='__main__':
   app.run_server(debug=True)

    