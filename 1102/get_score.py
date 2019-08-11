# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:13:54 2017

@author: finup
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

# data_train=pd.read_csv('C:\\20170912_aug_data\\data_train_all(1).csv',index_col=0)
# data_test=pd.read_csv('D:\\test\\data\\data_test_all(1).csv')
# data_all = data_train.append(data_test, ignore_index=True)

data_all=pd.read_csv('0601_0930.csv')
#data_all=pd.read_csv('D:\\6-9\\0601_0930.csv')
# pd.set_option('display.max_columns',100)

def get_label(data):
    data=data[data.plus_date!=0]
    data.fillna({'plus_date':0}, inplace=True)
    data.plus_date[data.plus_date>7]=0
    data.plus_date[data.plus_date!=0]=1
    return data

def user_age(data):
    x_train = data
    user_age_list=list(x_train.user_age)
    temp=[x if x >= 0 else np.nan for x in user_age_list]
    x_train.user_age=temp
    return x_train


use_col=['lastday_invite'
         ,'rechargestatus'
         ,'staytime'
         ,'sex_id'
         ,'client_type_id'
         ,'isrecharged'
         ,'user_age' 
         , 'staynum'
         ,'lastamt_invite'
         , 'isinvited'
        ]

data = get_label(data_all)

data.plus_date.value_counts()

y_label = data['plus_date']

#x_train = train.drop(['plus_date'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(data, y_label, test_size=0.3, random_state=1)

data_train = user_age(x_train)
data_test = user_age(x_test)

xx_train =x_train.loc[: ,use_col]
xx_test =x_test.loc[: ,use_col]
#
#cat_colnames=['sex_id', 'client_type_id', 'isrecharged', 'rechargestatus', 'isinvited', 'plus_date']
#
#for colname in set(xx_train.columns).difference(set(cat_colnames)):
#    xx_train[colname + '_na'] = xx_train[colname].isnull().astype(int)
    
xx_train=xx_train.fillna(0)
xx_test=xx_test.fillna(0)


dummy_col=['sex_id','client_type_id','rechargestatus']

xx_train[dummy_col]=xx_train[dummy_col].astype('object')
xx_test[dummy_col]=xx_test[dummy_col].astype('object')

Train=pd.get_dummies(xx_train,drop_first=True)
Test=pd.get_dummies(xx_test,drop_first=True)

X_train,y_train = Train,y_train
X_test,y_test = Test,y_test


clf=RandomForestClassifier(n_estimators=512,max_depth=10,random_state=50,oob_score=True,min_samples_leaf=20,min_samples_split=20)
clf.fit(X_train,y_train)

def data_result(clf, X, data, y, dataset):
    prob=clf.predict_proba(X)
    prediction=prob[:,1]
    Data=pd.DataFrame()
    Data['isinvested']=y
    Data['prediction']=prediction
    Data['call'] = 0
    Data['dataset'] = dataset 
    Data.loc[(data['callstatus']==1.0) & (data['call_effec'] != '失败') , 'call'] = 1
    return Data

result_train = data_result(clf, X_train, x_train, y_train, dataset = 'train')
result_test = data_result(clf, X_test, x_test, y_test, dataset = 'test')
Data = result_train.append(result_test, ignore_index=True)

Data.to_csv('06_09_call_score_rf_512.csv',index=False, index_label=False)