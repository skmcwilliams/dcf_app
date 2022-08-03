#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 19:31:28 2021

@author: skm
"""
import pandas as pd
from yahoo_fin import stock_info as si

def get_tickers():
       # gather stock symbols from major US exchanges
    df = pd.concat([pd.DataFrame(si.tickers_sp500()), pd.DataFrame(si.tickers_nasdaq()),
           pd.DataFrame(si.tickers_dow()),pd.DataFrame(si.tickers_other())])
    
    def sym_map(syms):
        bad_list = ['W', 'R', 'P', 'Q']
        sav_set = set()
        for s in syms:
            if len(s) > 4 and s[-1] in bad_list:
                pass
            else:
                sav_set.add(s)
        return sav_set
    
    sym = list(df[0])
    symbols = sym_map(sym)
    return symbols

        
    
ticker_df = pd.DataFrame(get_tickers(),columns=['Symbol']).dropna(subset=['Symbol'])
                
    
    
        
        
"""
sym1 = set(df1[0])
sym2 = set(df2[0])
sym3 = set(df3[0])
sym4 = set(df4[0])

# join the 4 sets into one. Because it's a set, there will be no duplicate symbols
symbols = set.union( sym1, sym2, sym3, sym4 )

# Some stocks are 5 characters. Those stocks with the suffixes listed below are not of interest.
my_list = ['W', 'R', 'P', 'Q']
del_set = set()
sav_set = set()

for symbol in symbols:
    if len( symbol ) > 4 and symbol[-1] in my_list:
        del_set.add( symbol )
    else:
        sav_set.add( symbol )

"""


