#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:40:45 2017

@author: mengshuang
"""

import pandas.core.algorithms as algos
import pandas as pd
import numpy as np

def bin_describe_new(data, elt, label_column, binset=20):

    data_new = data.sort_values(by=elt)
#    print(data_new.head())
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

