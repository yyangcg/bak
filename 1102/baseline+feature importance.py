
# coding: utf-8

# # ks value

# In[18]:

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


# In[ ]:

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
# result=pd.read_csv('C:\\20170912_aug_data\\data_train_all(1).csv',index_col=0)
result=pd.read_csv('D:\\competition\\06_train.csv')

pd.set_option('display.max_columns',100)


# # split

# In[135]:

def get_label(data):
    data=data[data.plus_date!=0]
    data.fillna({'plus_date':0}, inplace=True)
    data.plus_date[data.plus_date>7]=0
    data.plus_date[data.plus_date!=0]=1
    return data

# def drop_plus_date_rows(df):
#     conditions = df.plus_date == 0
#     return df.loc[~conditions, :]

def get_label_1(df):
    conditions = df.plus_date == 0
    df = df.loc[~conditions, :]
    df['label'] = df.plus_date.copy()
    df.fillna(0, inplace=True)
    df.loc[(df['plus_date']>7), 'label'] = 0
    df.loc[(df['plus_date']<8) & (df['plus_date']!=0), 'label'] = 1
    return df 

def user_age(data):
    x_train = data
    user_age_list=list(x_train.user_age)
    temp=[x if x >= 0 else np.nan for x in user_age_list]
    x_train.user_age=temp
    return x_train

drop_col=['mobile','create_date_id','create_time','date_id','invest_time',
          'active_num','pno','callstatus','call_effec','gap_calltoinvest',
          'user_id','active_num','last_calltime','call_times','isactive_aftercall',
          'isinvested','lastday_before_invest','rate','call_effec','invest_time',
          'plus_date','lianxuday_last','etl_time']

use_col=['lastday_invite','rechargestatus','staytime','sex_id','client_type_id','isrecharged',
         'user_age' , 'staynum','lastamt_invite', 'isinvited',
        ]

can_col=['areacode', 'devicetype','channel_id','staytime_p3','staytime_p6',
         'staytime_p12','staytime_p1','staytime_asset','staytime_hudong','staynum_redbag',
         'staynum_p3','staynum_p6','staynum_p12','staynum_p1','staynum_asset',
          'staynum_banner','staynum_ph','staynum_recharge','staytime_buy',
         'staynum_hudong','staynum_icon1','staynum_icon2','staynum_icon3',
        'staytime_buy','staytime_ph','staytime_recharge','staytime_redbag',]

pre_process=['lastday_invite','lastamt_invite','rechargestatus']


# In[133]:

train = get_label(result)
train.plus_date.value_counts()


# In[136]:

train = get_label_1(result)
train.label.value_counts()


# In[148]:

y_train = train['label']

x_train = train.drop(['plus_date','label'],axis=1)

x_train = user_age(x_train)

xx_train =x_train.loc[: ,use_col]

# xx_train = x_train[use_col]

# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)


# # preprocessing

# In[152]:

cat_colnames=['sex_id', 'client_type_id', 'isrecharged', 'rechargestatus', 'isinvited', 'plus_date']

for colname in set(xx_train.columns).difference(set(cat_colnames)):
    xx_train[colname + '_na'] = xx_train[colname].isnull().astype(int)


# # fillna

# In[ ]:

xx_train=xx_train.fillna(0)

dummy_col=['sex_id','client_type_id','rechargestatus']

xx_train[dummy_col]=xx_train[dummy_col].astype('object')

# xx_train.info()


# In[154]:

Train=pd.get_dummies(xx_train,drop_first=True)

# Train.info()


# In[155]:

X_train,y_train = Train,y_train 
clf=RandomForestClassifier(n_estimators=128,max_depth=10,random_state=50,oob_score=True,min_samples_leaf=20,min_samples_split=20)
clf.fit(X_train,y_train)


# In[156]:

clf.feature_importances_


# In[157]:

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
indices

#concat importances with column names
importance = pd.DataFrame(importances, index=X_train.columns, 
                          columns=["Importance"])

importance
importance_reorder = importance.sort_values(['Importance'], ascending=[False])
importance_reorder   


# In[70]:

# importance_reorder.to_csv('C:\\20170926 baseline\\importance_reorder.csv',index=False, index_label=False)


# In[158]:

prob=clf.predict_proba(X_train)
prediction=prob[:,1]
Data=pd.DataFrame()
Data['label']=y_train
Data['prediction']=prediction
# Data = Data.sort_values(by='prediction', ascending=False)
# ks_y=calc_continus_ks(Data)
# max(ks_y)
Data['user_id']=train.user_id
Data['dataset'] = 'train'

Data['call'] = 0

Data.loc[(train['callstatus']==1.0) & (train['call_effec'] != '失败') , 'call'] = 1


# In[34]:

import matplotlib.pyplot as plt
from sklearn import ensemble,metrics
target=y_train
auc=metrics.roc_auc_score(target, prediction)

print( "AUC:", auc)


# In[159]:

Data.to_csv('D:\\test\\aug_all\\aug_train_score_test.csv',index=False, index_label=False)


# # test

# In[86]:

data=pd.read_csv('D:\\test\\data\\data_test_all(1).csv')
pd.set_option('display.max_columns',100)


# In[80]:

data.callstatus.value_counts()

data=pd.read_csv('C:\\20170912\\oot_20170801_20170831.csv')
pd.set_option('display.max_columns',100)
# In[87]:

test = get_label(data)

test_y = test['plus_date']

test = user_age(test)


# In[88]:

x_test = test[use_col]

cat_colnames=['sex_id', 'client_type_id', 'isrecharged', 'rechargestatus', 'isinvited', 'plus_date']

for colname in set(x_test.columns).difference(set(cat_colnames)):
    x_test[colname + '_na'] = x_test[colname].isnull().astype(int)


# In[89]:

xx_test=x_test.fillna(0)


# In[90]:

xx_test[dummy_col]=xx_test[dummy_col].astype('object')

xx_test.info()


# In[91]:

Test = pd.get_dummies(xx_test,drop_first=True)

Test.info()


# In[92]:

test_prob=clf.predict_proba(Test)
test_prediction=test_prob[:,1]
Data_test=pd.DataFrame()
Data_test['label']=test_y
Data_test['score']=test_prediction
# ks_y=calc_continus_ks(Data_test)
# max(ks_y)
Data_test['user_id']=test.user_id
Data_test['dataset'] = 0

Data_test['call'] = 0

Data_test.loc[(test['callstatus']==1.0) & (test['call_effec'] != '失败') , 'call'] = 1


# In[52]:

Data_test['user_id']=data.user_id
def pdo_transform(score, lower_bound=0.001, eps=0.00001):

    score = np.array([x if x >= lower_bound else lower_bound for x in score])

    score = score + eps

    new_score = 632.0 + 62.0 * (np.log(score) - np.log(1 - score + 2 * eps))

    return new_score
# Data_test['score']=pdo_transform(test_prediction)


# In[54]:

def inv_trans_score(score, lower_bound=0.001, eps=0.00001):
    log_score = exp((score - 632)/62.0)
    score_new = 2 * log_score * ( 1 + eps )/( 1 + 2 * log_score)
    score_new = score_new - eps
    return score_new
    


# In[93]:

Data_test.to_csv('D:\\test\\aug_all\\aug_test_score.csv',index=False, index_label=False)


# In[27]:

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
# In[95]:

data_train=pd.read_csv('D:\\test\\aug_all\\aug_train_score.csv')
data_test=pd.read_csv('D:\\test\\aug_all\\aug_test_score.csv')
data_all = data_train.append(data_test, ignore_index=True)


# In[96]:

data_all.to_csv('D:\\test\\aug_all\\aug_all_score.csv',index=False, index_label=False)


# In[ ]:



