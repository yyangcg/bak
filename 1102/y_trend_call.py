# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 18:30:09 2017

@author: finup
"""

import numpy as np
import pandas as pd
import pandas.core.algorithms as algos


def get_data(filename):
    data_raw = pd.read_csv(filename)
    return data_raw

def get_label(data):
    data=data[data.plus_date!=0]
    data.fillna({'plus_date':0}, inplace=True)
    data.plus_date[data.plus_date>7]=0
    data.plus_date[data.plus_date!=0]=1
    return data

def merge_score(data_raw, df_score, colname):
    df = pd.merge(df_score, data_raw, how = 'inner', on = colname)
    return df

date = '20171009'

filename = 'D:\\test\\' + 'huoke_' + date + '.csv'

data_raw = get_data(filename)

data_score_name = 'D:\\test\\upto_20171011.csv'

data_with_score = get_data(data_score_name)

df_score = data_with_score[['user_id','score']]

df = merge_score(data_raw, df_score, colname = 'user_id')

data = get_label(df)

use_col=[ 'callstatus'
         ,'plus_date'
         ,'score']

Data = data[use_col]

#Data.to_csv('huoke_' + date + '.csv', encoding='utf-8',index=False,index_label=True)
def bin_fun(data,binset,elt):
    data_new = data.sort_values(elt, ascending=False)
    data_new = data_new.reset_index()
    del data_new['index']
    bins = algos.quantile(data_new.index, np.linspace(0, 1, 10))
    tt = pd.cut(data_new.index, bins, include_lowest=True)
    data_new['index'] = tt
    return data_new

bin_fun(Data,binset,elt = 'score')