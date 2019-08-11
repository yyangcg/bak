# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:05:55 2017

@author: finup
"""
import pandas.core.algorithms as algos
import pandas as pd
import numpy as np

def bin_describe_new(data, elt, label_column, binset=20):
    
    data_new = data.sort_values(by=elt)
    data_new = data_new.reset_index()
    del data_new['index']
    bins = algos.quantile(data_new.index, np.linspace(0, 1, binset))
    tt = pd.cut(data_new.index, bins, include_lowest=True)
    data_new['index'] = tt
    grouped_data = data_new.groupby(['index'])
    print(grouped_data.agg({elt: ['count', 'min', 'max', 'mean'], label_column: ['mean']}))
    return data_new, bins


def bin_apply(data_new, label_column):
    grouped_data = data_new.groupby(['index'])
    print(grouped_data.agg({label_column: ['count', 'mean', 'min']}))



result = pd.read_csv('C:\\20170912\\data_train_all(1).csv',encoding='utf-8')

def get_label(data):
    
    data.fillna({'plus_date':0}, inplace=True)
    data.plus_date[data.plus_date>7]=0
    data.plus_date[data.plus_date!=0]=1
    return data

result = get_label(result)

data_new, bins = bin_describe_new(result, elt='staynum', label_column='plus_date', binset=11)

#print(data_new)

describe_label = bin_apply(data_new, label_column='plus_date')

#==============================================================================
# describe_label.columns
# import matplotlib.pyplot as plt
# plt.figure(1)
# # plt.hist(LOSS, bins=20,range=(min(LOSS),max(LOSS)))
# plt.plot(describe_label.index,describe_label.mean,'b')
# plt.show()
#==============================================================================
