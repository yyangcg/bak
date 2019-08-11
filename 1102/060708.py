# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

# data_train=pd.read_csv('C:\\20170912_aug_data\\data_train_all(1).csv',index_col=0)
# data_test=pd.read_csv('D:\\test\\data\\data_test_all(1).csv')
# data_all = data_train.append(data_test, ignore_index=True)
#start_time = time.time()

#data_all=pd.read_csv('0601_0930.csv')
#df = data_all.loc[(data_all['create_date_id']<20170801), :]
#df.to_csv('06_07_data.csv',index=False, index_label=False)
#df_1 = data_all.loc[(data_all['create_date_id']<20170901) & (data_all['create_date_id']>20170731), :]
#df_1.to_csv('08_data.csv',index=False, index_label=False)


data_all=pd.read_csv('0601_0930.csv')
y = data_all['plus_date']

data_train, data_test, y_train, y_test = train_test_split(data_all, y, test_size=0.3, random_state=1)

df = data_train

df.to_csv('06_09_train_data.csv',index=False, index_label=False)


df_1 = data_test
df_1.to_csv('06_09_test_data.csv',index=False, index_label=False)



