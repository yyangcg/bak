# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 17:46:38 2017

@author: finup
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 18:30:09 2017

@author: finup
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 18:30:09 2017

@author: finup
"""

import numpy as np
import pandas as pd
import pandas.core.algorithms as algos
import math


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


def fico_inverse_transform(score, eps=0.00001):
    
    prob = ((1 + eps) * np.exp((score - 632) / 62) - eps) / (1 + np.exp((score - 632) / 62))
    prob = [1 if p > 1 else 0 if p < 0.001 else p for p in prob]

    return prob

filename = 'D:\\test\\data\\data_test_all(1).csv'
#filename = 'D:\\test\\data\\huoke_20170929_label.csv'

data_score_name = 'D:\\test\\data\\test_all_score_prob.csv'
#data_score_name = 'D:\\test\\data\\upto_20171011.csv'

data_raw = get_data(filename)

data_with_score = get_data(data_score_name)

df_score = data_with_score[['user_id','score']]

df = merge_score(data_raw, df_score, colname = 'user_id')

data = get_label(df)

#score = data.score
#
#prob = fico_inverse_transform(score, eps=0.00001)
#
#data['prob'] = prob

data['call'] = 0

data.loc[(data['callstatus']==1.0) & (data['call_effec'] != '失败') , 'call'] = 1

use_col=[ 'call'
         ,'plus_date'
         ,'score']

Data = data[use_col]

Data.to_csv('D:\\test\\huoke_test(prob).csv', encoding='utf-8',index=False,index_label=True)
#def bin_fun(data,binset,elt):
#    data_new = data.sort_values(elt, ascending=False)
#    data_new = data_new.reset_index()
#    del data_new['index']
#    bins = algos.quantile(data_new.index, np.linspace(0, 1, 10))
#    tt = pd.cut(data_new.index, bins, include_lowest=True)
#    data_new['index'] = tt
#    return data_new
#
#bin_fun(Data,binset,elt = 'score')