# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:55:43 2017

@author: finup
"""
# segmentation


import numpy as np
import pandas as pd
from decision_tree_spliter import tree_to_code
from sklearn.ensemble import RandomForestClassifier
#from sklearn.cross_validation import train_test_split

train_raw=pd.read_csv('C:\\20170912\\data_train_all(1).csv')
pd.set_option('display.max_columns',100)

def get_label(data):
    
    data.fillna({'plus_date':0}, inplace=True)
    data.plus_date[data.plus_date>7]=0
    data.plus_date[data.plus_date!=0]=1
    return data

train = get_label(train_raw)

train.staytime = train.staytime.fillna(-9999)

#==============================================================================
# 
# x = train[['staynum']]
# 
# client_type_id = pd.get_dummies( train.client_type_id , prefix='client_type_id',drop_first=True )
# 
# x = pd.concat( [ x , client_type_id ] , axis=1 )
# 
# x.info()
# 
#==============================================================================



client_dict = { 1: 10, 2 : 30,  3 : 40, 6 : 20}

train['client_type_id_1'] = train.client_type_id.map({ 1: 10, 2 : 30,  3 : 40, 6 : 20})


x = train[['client_type_id_1','staynum']]

x.info()


xtarget = train['plus_date']
from sklearn import tree 
clf = tree.DecisionTreeClassifier(max_depth=1)
obj = clf.fit(x, xtarget)
tree_to_code(obj, ['client_type_id_1','staynum'], train)
#==============================================================================
# tree_to_code(obj, ['client_type_id_2','client_type_id_3','client_type_id_6','staynum'], train)
#==============================================================================
