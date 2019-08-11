#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:40:45 2017

@author: mengshuang
"""

import pandas.core.algorithms as algos
import pandas as pd
import numpy as np

class BIN_DECILE:
    def __init__(self):
        self.binset = 10
        
    def bin_describe_index(self, data, elt, binset):    
        data_new = data.sort_values(by=elt)
        data_new = data_new.reset_index()
        del data_new['index']
        bins = algos.quantile(data_new.index, np.linspace(0, 1, binset))
        labels_list = [x for x in range(1,binset)]
        tt = pd.cut(data_new.index, bins, include_lowest=True, labels=labels_list)
        data_new['bins'] = tt
        return data_new, bins
    
    def bin_describe_value(self, data_base, data_pred, elt_base, elt_pred, binset):
        bins = algos.quantile(data_base[elt_base].unique(), np.linspace(0, 1, binset))
        tt_base = pd.cut(data_base[elt_base], bins, include_lowest=True)
        tt_pred = pd.cut(data_pred[elt_pred], bins, include_lowest=True)
        data_base['bins'] = tt_base
        data_pred['bins'] = tt_pred
        return data_base, data_pred, bins
        
    
    def bin_apply(self, data_new, label_column):
        grouped_data = data_new.groupby(['bins'])
        print(grouped_data.agg({label_column: ['count', 'mean', 'min', 'max']}))

### example call function
#from BIN_DECILE import BIN_DECILE
#bd = BIN_DECILE()
#df_probas_new3, bins = bd.bin_describe_new(df_probas_new2,elt='predict',binset=10)
#bd.bin_apply(df_probas_new3, label_column= 'is_invest')
#bd.bin_apply(df_probas_new3, label_column= 'ks_score')
#bd.bin_apply(df_probas_new3, label_column= 'predict')