from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from zodbpickle import pickle
from IPython.display import display
import time
from data_preproc_and_feature_engineering import *
pd.set_option('display.max_columns', 100)

start_time = time.time()

# Data folder
data_folder = '../data/'
# Load data
data_train = pd.DataFrame.from_csv(data_folder + 'aug/data_train_stf.csv')
data_test = pd.DataFrame.from_csv(data_folder + 'aug/data_test_all.csv')
oot_data = pd.read_csv(data_folder + 'oot_20170922.csv', warn_bad_lines=False, error_bad_lines=False, engine='python')

## !!! Caution vars values changed !!!
data_train = drop_plus_date_rows(data_train)
data_test = drop_plus_date_rows(data_test)
oot_data = drop_plus_date_rows(oot_data)

# Reset shuffled indices
data_train.reset_index(drop=True, inplace=True)
data_test.reset_index(drop=True, inplace=True)
oot_data.reset_index(drop=True, inplace=True)

# Get feature data/label data and preserve test set user_ids
feature_train, feature_test = data_train.drop('isinvested', axis=1), data_test.drop('isinvested', axis=1)
feature_train_new, feature_oot = data_train.drop('isinvested', axis=1), oot_data.drop('isinvested', axis=1)

label_train, label_test = get_labels(data_train, data_test)
label_oot = get_labels(oot_data)

user_id_test = data_test.user_id
user_id_oot = oot_data.user_id

# Join back for data preprocessing
feature_all_df = feature_train.append(feature_test, ignore_index=True)
feature_all_oot = feature_train_new.append(feature_oot, ignore_index=True)

# Determine features to use and drop the rest
feature_all_dropped = drop_columns(feature_all_df)
feature_all_oot_dropped = drop_columns(feature_all_oot)

# Feature engineering
feature_all_proc = speedy_process(feature_all_dropped, cat_colnames=['sex_id', 'client_type_id', 
                                                        'isrecharged', 'rechargestatus', 'isinvited'])
feature_all_oot_proc = speedy_process(feature_all_oot_dropped, cat_colnames=['sex_id', 'client_type_id', 
                                                        'isrecharged', 'rechargestatus', 'isinvited'])

feature_train = feature_all_proc.iloc[:len(feature_train), :]
feature_test = feature_all_proc.iloc[len(feature_train):, :]

feature_train_oot = feature_all_oot_proc.iloc[:len(feature_train_new), :]
feature_oot = feature_all_oot_proc.iloc[len(feature_train_new):, :]

# Convert labels
label_dict = {'未投资': 0, '已投资': 1}

label_train.replace(label_dict, inplace=True)
label_test.replace(label_dict, inplace=True)
label_oot.replace(label_dict, inplace=True)

# Training / validation / test split
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(feature_train, label_train, test_size=0.30, random_state=13)

# View and save feature importancese
from reporting import get_feature_importance_table

feature_importance_df = get_feature_importance_table(X_train, y_train)
feature_importance_df.to_csv('feature_importances.csv')

# Build model
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=512, min_samples_leaf=25, max_features='sqrt', n_jobs=-1, oob_score=True, random_state=13)
clf.fit(X_train, y_train)

# Predict
# Training set prediction
y_pred_train = clf.predict_proba(X_train)
# Validation set prediction
y_pred_val = clf.predict_proba(X_val)

# Test set prediction
y_pred = clf.predict_proba(feature_test)
y_pred_binary = clf.predict(feature_test)

y_pred_oot = clf.predict_proba(feature_oot)
y_pred_binary_oot = clf.predict(feature_oot)

print("Training and predicting took {} secs.".format(round(time.time() - start_time, 2)))

# Report
from reporting import eval_ks, draw_decile_chart, get_performance_table, create_and_save_roc_curve_to_png

# Create decile chart (training set)
decile_train = pd.DataFrame({'Label': y_train, 'Prediction': y_pred_train[:, 1]})
draw_decile_chart(decile_train, 'Prediction', ['Label', 'Prediction'], bins=5, chart_name='train')

# Create decile chart (test set)
decile_test = pd.DataFrame({'Label': label_test, 'Prediction': y_pred[:, 1]})
draw_decile_chart(decile_test, 'Prediction', ['Label', 'Prediction'], bins=5, chart_name='test')

# Create decile chart (oot)
decile_oot = pd.DataFrame({'Label': label_oot, 'Prediction': y_pred_oot[:, 1]})
draw_decile_chart(decile_oot, 'Prediction', ['Label', 'Prediction'], bins=5, chart_name='oot')

# Create performance table ACC, KS, AUC, F1
performance_df = get_performance_table(label_test, y_pred, y_pred_binary)
performance_df.to_csv('performance.csv')

# create_and_save_roc_curve_to_png(label_test, y_pred)

performance_oot = get_performance_table(label_oot, y_pred_oot, y_pred_binary_oot)
performance_oot.to_csv('performance_oot.csv')

create_and_save_roc_curve_to_png(label_oot, y_pred_oot)

from reporting import BinDecile

xl_df = pd.DataFrame({'user_id': user_id_test, 'y_hat': y_pred[:, 1], 'y': label_test})
xl_oot = pd.DataFrame({'user_id': user_id_oot, 'y_hat': y_pred_oot[:, 1], 'y': label_oot})

bin_obj = BinDecile()
bin_obj_oot = BinDecile()

xl_df_binned, bins = bin_obj.bin_describe_new(xl_df, 'y_hat', binset=10, df_new_bin_name='y_hat_bin')
xl_df_binned_oot, bins_oot = bin_obj_oot.bin_describe_new(xl_oot, 'y_hat', binset=10, df_new_bin_name='y_hat_bin')
# display(xl_df_binned.head())

good_rate_table = bin_obj.bin_apply(xl_df_binned, 'y', df_new_bin_name='y_hat_bin')
good_rate_table.to_csv('good_rate_table.csv')

good_rate_table_oot = bin_obj_oot.bin_apply(xl_df_binned_oot, 'y', df_new_bin_name='y_hat_bin')
good_rate_table_oot.to_csv('good_rate_table_oot.csv')

pivot_table = feature_test.copy()
pivot_table.reset_index(drop=True, inplace=True)

pivot_table.loc[:, 'label'] = label_test
pivot_table.loc[:, 'y_hat'] = y_pred[:, 1]
display(pivot_table.head())
pivot_table.to_csv('pivot_table.csv')

# My old chart
# decile_df = pd.DataFrame({'Label': label_test, 'Prediction': y_pred[:, 1]})
# draw_decile_chart(decile_df, 'Prediction', ['Label', 'Prediction'], bins=10)
