
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
import time

# # ks value
# In[3]:
def calc_ks(data, n, prediction="prediction", label="label"):
    """
    calculate ks value
    :param data: DataFrame[prediction,label]
    :param n : int number of cut
    :param prediction: string name of prediction
    :param label: string name of label
    :return: array
    """
    data = data.sort_values(prediction, ascending=False)
    boolean = True
    while boolean:
        try:
            data[prediction] = pd.Series(pd.qcut(data[prediction], n, labels=np.arange(n).astype(str)))
        except ValueError:
            boolean = True
            n -= 1
        else:
            boolean = False
    count = data.groupby(prediction)[label].agg({'bad': np.count_nonzero, 'obs': np.size})
    count["good"] = count["obs"] - count["bad"]
    t_bad = np.sum(count["bad"])
    t_good = np.sum(count["good"])
    ks_vector = np.abs(np.cumsum(count["bad"]) / t_bad - np.cumsum(count["good"]) / t_good)
    return ks_vector

def calc_continus_ks(data, prediction="prediction", label="label"):
    """

    :param data:
    :param prediction:
    :param label:
    :return:
    """
    data = data.sort_values(prediction, ascending=False)
    count = data.groupby(prediction, sort=False)[label].agg({'bad': np.count_nonzero, 'obs': np.size})
    count["good"] = count["obs"] - count["bad"]
    t_bad = np.sum(count["bad"])
    t_good = np.sum(count["good"])
    ks_vector = np.abs(np.cumsum(count["bad"]) / t_bad - np.cumsum(count["good"]) / t_good)
    return ks_vector


# In[2]:

pd.set_option('display.max_columns',100)

# # split

# In[3]:

def get_label(data):
    data.fillna({'plus_date':0}, inplace=True)
    data.plus_date[data.plus_date>7]=0
    data.plus_date[data.plus_date!=0]=1
    return data


def get_data(filename):
    data_train = pd.read_csv(filename,index_col=0)
    return data_train

def user_age(train):
    x_train = train
    user_age_list=list(x_train.user_age)
    temp=[x if x >= 0 else np.nan for x in user_age_list]
    x_train.user_age=temp
    return x_train

def get_cols(data_train,use_col):
    
    x_train=data_train[use_col]
    return x_train
# In[4]:
training_filename = 'C:\\20170912\\data_train_all(1).csv'
testing_filename = 'C:\\20170912\\data_test.csv'

drop_col=['mobile','create_date_id','create_time','date_id','invest_time',
          'active_num','pno','callstatus','call_effec','gap_calltoinvest',
          'user_id','active_num','last_calltime','call_times','isactive_aftercall',
          'isinvested','lastday_before_invest','rate','call_effec','invest_time',
          'plus_date','lianxuday_last','etl_time']

use_col=['lastday_invite','rechargestatus','staytime','sex_id','client_type_id','isrecharged',
         'user_age' , 'staynum','lastamt_invite', 'isinvited','plus_date'
        ]

can_col=['areacode', 'devicetype','channel_id','staytime_p3','staytime_p6',
         'staytime_p12','staytime_p1','staytime_asset','staytime_hudong','staynum_redbag',
         'staynum_p3','staynum_p6','staynum_p12','staynum_p1','staynum_asset',
          'staynum_banner','staynum_ph','staynum_recharge','staytime_buy',
         'staynum_hudong','staynum_icon1','staynum_icon2','staynum_icon3',
        'staytime_buy','staytime_ph','staytime_recharge','staytime_redbag',]

pre_process=['lastday_invite','lastamt_invite','rechargestatus']

data_train_raw=get_data(training_filename)
train = get_label(data_train_raw)
y_train = train['plus_date']
data_train = user_age(train)
x_train = get_cols(data_train,use_col)

#x_train.plus_date.isnull().sum()

# In[6]:

xx_train = x_train.drop(['plus_date'],axis=1)

# x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.3, random_state=1)
# # preprocessing

# In[10]:

cat_colnames=['sex_id', 'client_type_id', 'isrecharged', 'rechargestatus', 'isinvited', 'plus_date']

getdummy_col=['sex_id','client_type_id','rechargestatus']

cat_colnames=['sex_id', 'client_type_id', 'isrecharged', 'rechargestatus', 'isinvited', 'plus_date']

for colname in set(xx_train.columns).difference(set(cat_colnames)):
    xx_train[colname + '_na'] = xx_train[colname].isnull().astype(int)

xx_train=xx_train.fillna(0)

dummy_col=['sex_id','client_type_id','rechargestatus']

xx_train[dummy_col]=xx_train[dummy_col].astype('object')

xx_train.info()   
# # fillna


# In[12]:

Train=pd.get_dummies(xx_train,drop_first=True)
X_train,y_train = Train,y_train 

# In[12]:
gbdt=GradientBoostingClassifier(learning_rate=0.1, n_estimators=30, subsample=1, min_samples_split=20, min_samples_leaf=10,
                               max_depth=5, max_features=None, verbose=0, max_leaf_nodes=None,
                               warm_start=False)
gbdt.fit(X_train,y_train)
# In[114]:

gbdt.feature_importances_


# In[115]:

importances = gbdt.feature_importances_
indices = np.argsort(importances)[::-1]
indices

#concat importances with column names
importance = pd.DataFrame(importances, index=X_train.columns, 
                          columns=["Importance"])

importance
importance_reorder = importance.sort_values(['Importance'], ascending=[False])
importance_reorder   


# In[116]:



# In[117]:

import matplotlib.pyplot as plt
from sklearn import ensemble,metrics
target=y_train
auc=metrics.roc_auc_score(target, prediction)

print( "AUC:", auc)


# In[ ]:


# final1['y_star'] = prediction2

# Data_vali.isnull().sum()
#
#Data.to_csv('C:\\20170926 baseline\\base_train.csv',index=False, index_label=False)
# # test

# In[14]:

data_test=pd.read_csv('C:\\20170920\\oot_20170922.csv')
pd.set_option('display.max_columns',100)
#
#data=pd.read_csv('C:\\20170912\\oot_20170801_20170831.csv')


# In[16]:

data_test_raw=data_test
test = get_label(data_test_raw)
y_test = test['plus_date']
data_test = user_age(test)
x_test = get_cols(data_test,use_col)

#x_train.plus_date.isnull().sum()

# In[6]:

xx_test = x_test.drop(['plus_date'],axis=1)

# x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.3, random_state=1)
# # preprocessing

# In[10]:

for colname in set(xx_test.columns).difference(set(cat_colnames)):
    xx_test[colname + '_na'] = xx_test[colname].isnull().astype(int)

xx_test=xx_test.fillna(0)

xx_test[dummy_col]=xx_test[dummy_col].astype('object')

xx_test.info()   

# In[21]:

Test = pd.get_dummies(xx_test,drop_first=True)

Test.info()

# In[ ]:
prob=gbdt.predict_proba(Test)
prediction=prob[:,1]
Data=pd.DataFrame()
Data['label']=test_y
Data['prediction']=prediction
#Data = Data.sort_values(by='prediction', ascending=False)
ks_y=calc_continus_ks(Data,prediction="prediction")
max(ks_y)
# In[ ]:
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
import time

start_time = time.time()
auc_train=[]
auc_test=[]
tmp=[]
tmp_2=[]
pre=pd.DataFrame()

n_gbdts = 100

for i in range(n_gbdts):
    X_1, X_2, y_1, y_2 = train_test_split(X_train,y_train, test_size=0.3, random_state=i)
    #clf=RandomForestClassifier(n_estimators=128,max_depth=20,random_state=i,oob_score=True,min_samples_leaf=20,min_samples_split=20)
    gbdt_100=GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=5, random_state=i)
    gbdt_100.fit(X_1,y_1)
    prob=gbdt_100.predict_proba(Test)
    prediction=prob[:,1]
    pre[i]=prediction
    
print("Training {0} GBDTs took {1} seconds.".format(n_gbdts, time.time() - start_time))


# In[ ]:

pre1=pre.T
pre2=pre1.mean()
pre1;
Data=pd.DataFrame()
Data['label']=y_test
Data['prediction']=pre2
ks_y=calc_continus_ks(Data)
max(ks_y)


# In[22]:

test_prob=clf.predict_proba(Test)
test_prediction=test_prob[:,1]
Data_test=pd.DataFrame()
Data_test['label']=test_y
Data_test['prediction']=test_prediction
ks_y=calc_continus_ks(Data_test)
max(ks_y)


# In[119]:

import matplotlib.pyplot as plt
from sklearn import ensemble,metrics
target=test_y
auc=metrics.roc_auc_score(target, test_prediction)

print( "AUC:", auc)


# In[ ]:

final2 = pd.DataFrame()

final2['y_true'] = Data_test['label']

final2['y_hat'] = Data_test['prediction']

Data_test=Data_test.to_csv('C:\\20170926 baseline\\base_test.csv',index=False, index_label=False)
# In[ ]:



