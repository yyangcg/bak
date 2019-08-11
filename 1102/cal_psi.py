#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 18:24:21 2017

@author: finup
"""
import pandas.core.algorithms as algos
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def cal_psi(base, oot, elt):
    df_base = base[elt].as_data_frame(use_pandas=True, header=True)
    df_oot = oot[elt].as_data_frame(use_pandas=True, header=True)
    
    maximum1 = df_base[elt].max()
    minimum1 = df_base[elt].min()
    df_base[elt] = df_base[elt].fillna(int(minimum1) - 2)
    a1 = 1 / (maximum1 - minimum1)
    b1 = -a1 * minimum1
    df_base[elt] = a1 * df_base[elt] + b1
    
    maximum2 = df_oot[elt].max()
    minimum2 = df_oot[elt].min()
    df_oot[elt] = df_oot[elt].fillna(int(minimum2) - 2)
    a2 = 1 / (maximum2 - minimum2)
    b2 = -a2 * minimum2
    df_oot[elt] = a2 * df_oot[elt] + b2
    
    bins = algos.quantile([x/10.0 for x in range(11)], np.linspace(0, 1, 11))
#    bins_new = np.append(bins, [99, -99])
#    bins_sort = np.sort(bins_new)
    bin_base = pd.cut(df_base[elt], bins, include_lowest=True)
    bin_oot = pd.cut(df_oot[elt], bins, include_lowest=True)
    
    df_base['bins'] = bin_base
    df_oot['bins'] = bin_oot
    
    plt.subplot(311)
    plt.hist(df_base[elt], bins=bins)
    plt.title("Base Histogram")
    plt.subplot(313)
    plt.hist(df_oot[elt], bins=bins)
    plt.title("oot Histogram")
    plt.show() 
    
    
    grouped_base = df_base.groupby(['bins']).agg({elt:['count', 'mean']})
    print(grouped_base)
    grouped_oot = df_oot.groupby(['bins']).agg({elt:['count', 'mean']})
    print(grouped_oot)
    
    accsum = 0
    n_base = len(df_base)
    n_oot = len(df_oot)
    for i in range(len(bins)-1):
        base_percent = grouped_base[elt]['count'][i]*1.0/n_base
        oot_percent = grouped_oot[elt]['count'][i]*1.0/n_oot
        accsum += (oot_percent - base_percent) * math.log(oot_percent/base_percent)
    print('psi: '+str(accsum))
        
        
        
        
        
        