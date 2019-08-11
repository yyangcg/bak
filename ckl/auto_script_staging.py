# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import time

import datetime

from IPython.display import display
import sys
import pickle
from data_preproc_and_feature_engineering import *
from reporting import *
from user_acquisition_io import *

def scheduler(schedule_time):
    from apscheduler.schedulers.blocking import BlockingScheduler
    import logging

    print('Scheduler will start at {}'.format(schedule_time))
    logging.basicConfig()
    scheduler = BlockingScheduler()
    scheduler.add_job(main, 'interval', seconds=86400, next_run_time=schedule_time)
    # scheduler.add_job(main, 'interval', seconds=1000, next_run_time=schedule_time)
    scheduler.start()


def build_model(feature_all, data_train, label_train):
    feature_train = feature_all.iloc[:len(data_train), :]
    feature_test = feature_all.iloc[len(data_train):, :]

    # global label_dict
    label_train.replace(label_dict, inplace=True)

    from sklearn.ensemble import RandomForestClassifier

    # clf = RandomForestClassifier(n_estimators=128, n_jobs=-1, oob_score=True, random_state=13)
    # clf = RandomForestClassifier(n_estimators=128, n_jobs=-1, oob_score=False, random_state=13)
    clf = RandomForestClassifier(n_estimators=512, min_samples_leaf=25, max_features='sqrt', n_jobs=-1, oob_score=True, random_state=13)
    clf.fit(feature_train, np.asarray(label_train, dtype="|S6"))

    return clf, feature_train, feature_test


def predict_and_get_score(clf, feature_test, label_test):
    global label_dict
    
    try:
        label_test.replace(label_dict, inplace=True)
    except TypeError:
        print('label_test already transformed.')
        
    y_pred = clf.predict_proba(feature_test)
    score = y_pred[:, 1]
    return score


def pdo_transform(score, lower_bound=0.001, eps=0.00001):

    score = np.array([x if x >= lower_bound else lower_bound for x in score])

    score = score + eps

    new_score = 632.0 + 62.0 * (np.log(score) - np.log(1 - score + 2 * eps))

    return new_score
    

def create_mysql_csv(data_test, score, filename):
    # mysql_cols = ['user_id', 'mobile']
    mysql_cols = ['user_id', 'mobile', 'areacode', 'active_num', 'sex_id', 'isrecharged', 
                  'rechargestatus', 'user_age', 'channel_id', 'score']
    mysql_df = data_test.loc[:, mysql_cols]
    mysql_df['score'] = score

    mysql_df.to_csv(filename)

    return mysql_df
    
def create_csv_ks(filename, ks):
    datetime_ks = [datetime.date.today(), ks]
    with open(filename, 'ab') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(datetime_ks)

def main():
    if mode == 'live':
        daily_scoring_tablename = 'ck_user_id_score'
        accum_tablename = 'ck_accum'
    elif mode == 'staging':
        daily_scoring_tablename = 'ck_user_id_score_staging'
        accum_tablename = 'ck_accum_staging'
    else:
        raise ValueError("Invalid mode.")

    print("Process running at {}".format(datetime.datetime.now()))
    start_time = time.time()

    # with open('user_acquisition_partition_limited.sql', 'r') as handle:
    #     sql_train = handle.read().replace('\n', ' ')

    with open('main_sql.sql', 'r') as handle:
        sql = handle.read()

    sql_train = sql

    # sql_train = sql_train.replace('partition_date', '''"2017-09-01"''')
    sql_test = sql.replace('statistic_dayx', 'statistic_dayb').replace('statistic_dayy', 'statistic_daye')
    sql_test = sql_test.replace('final_train_baiyang', 'final_test_baiyang').replace('final_train_baiyang', 'final_test_baiyang')

    # Pull data from Hive
    global label_dict

    pull_data(sql_train, sql_test)

    # Get data
    data_train_raw, data_test_raw = get_data(training_filename, testing_filename)
    data_train_raw = drop_plus_date_rows(data_train_raw)
    data_test_raw = drop_plus_date_rows(data_test_raw)
    
    # Reset shuffled indices
    data_train_raw.reset_index(drop=True, inplace=True)
    data_test_raw.reset_index(drop=True, inplace=True)

    data_train, data_test = data_train_raw, data_test_raw

    # Get label
    label_train, label_test = get_labels(data_train, data_test)
    # Combine data
    data_all = combine_data(data_train, data_test)
    # Feature transform
    feature_all = feature_transform(data_all)
    # Build_model
    clf, feature_train, feature_test = build_model(feature_all, data_train, label_train)

    # Save model
    save_folder_path = '/home/cloud/calvinku/sandbox/'
    create_time = datetime.date.today().strftime('%Y-%m-%d')
    with open(save_folder_path + 'rf_' + create_time + '_model.sav', 'wb') as handle:
        pickle.dump(clf, handle)

    # Load model
    with open(save_folder_path + 'rf_2017-09-30_model.sav', 'rb') as handle:
        clf_aug = pickle.load(handle)

    # Get score
    score_aug = predict_and_get_score(clf_aug, feature_test, label_test)
    score_aug = pdo_transform(score_aug)
    score = predict_and_get_score(clf, feature_test, label_test)
    score = pdo_transform(score)
    
    # Save score data to csv
    mysql_df_aug = create_mysql_csv(data_test, score_aug, 'score_aug.csv')
    # mysql_df_aug = create_mysql_csv(data_test, score, 'score_aug.csv')
    mysql_df = create_mysql_csv(data_test, score, 'score.csv')
    
    mysql_df['score_new'] = score_aug
    # mysql_df['score_new'] = score

    print(mysql_df.shape)
    display(mysql_df.head())

    update_db(mysql_df, daily_scoring_tablename)
    update_db(mysql_df, accum_tablename, overwrite=False)

    # Writing lift scores to MySQL
    # lift_main()

    print('Entire process took {} seconds.'.format(time.time() - start_time))
    print('Waiting for next process....')

# def update_db_lift(df, table_name, overwrite=True, host='192.168.143.201', user='phdc_calvinku', password='phdc_calvinku.XLPfZrvjMlsFqSqfxOSBcDc1EOGp5Xy8', port=3306, db="policy_jtyy"):
#     # Connect to server
#     conn = mdb.connect(host=host, user=user, password=password, port=port, db=db, charset="utf8")
#     # Create cursor
#     cur = conn.cursor()
#     # Set encoding
#     cur.execute("SET NAMES utf8")

#     if overwrite:
#         # Drop table if it already exist using execute() method.
#         cur.execute("DROP TABLE IF EXISTS " + table_name)
#         # Create table SQL string
#         create_table_sql = """
#           CREATE TABLE IF NOT EXISTS """ + table_name + """
#           (
#           `id` bigint(100) NOT NULL AUTO_INCREMENT COMMENT 'Primary key',
#           `user_id` varchar(100) COMMENT 'User ID',
#           `lift_score` DOUBLE PRECISION NOT NULL DEFAULT '0' COMMENT 'Baiyang Team Lift Score',
#            PRIMARY KEY (`id`),
#            UNIQUE KEY `user_id_u` (`user_id`)
#            ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='iqj_acquiring_score'
#           """

#         cur.execute(create_table_sql)

#         num_records = len(df)

#         for row in df.iloc[:num_records, :].itertuples():
#             data_tuple = tuple(row[1:])

#             data_list = []

#             for x in data_tuple:
#               if type(x) == type('abc'):
#                 data_list.append(x)
#               elif np.isnan(x):
#                 data_list.append(None)
#               else:
#                 data_list.append(x)            

#             data_tuple = tuple(data_list)
#             print(data_tuple)
#             add_new_record_to_sql_table_lift(conn, cur, data_tuple, table=table_name)
#     else:
#         pass

#     # Close cursor and disconnect
#     cur.close()
#     conn.close()
#     print('lift_main() finished updating MySQL.')


# def add_new_record_to_sql_table_lift(conn, sql_cursor, data_tuple, table, db='policy_jtyy'):
#     sql_cursor.execute('''INSERT INTO ''' + table + ''' (user_id, lift_score) VALUES (%s, %s) ''', tuple(str(x) if x is not None else None for x in data_tuple))
#     # Make commitment to the above query
#     conn.commit()

# def lift_main():
#     from sklearn.ensemble import RandomForestClassifier

#     # Get data
#     data_train_raw, data_test_raw = get_data()
#     data_train_raw = drop_plus_date_rows(data_train_raw)
#     data_test_raw = drop_plus_date_rows(data_test_raw)
    
#     # Reset shuffled indices
#     data_train_raw.reset_index(drop=True, inplace=True)
#     data_test_raw.reset_index(drop=True, inplace=True)

#     data_train, data_test = data_train_raw, data_test_raw

#     feature_train, feature_test = data_train.drop('isinvested', axis=1), data_test.drop('isinvested', axis=1)
#     label_train, label_test = data_train.isinvested, data_test.isinvested

#     user_id_test = data_test.user_id.copy()

#     xs_feature = feature_train.append(feature_test, ignore_index=True)
#     # label_all = label_train.append(label_test, ignore_index=True)

#     xs_feature = drop_stay_cols(xs_feature)

#     # Create succ_call
#     xs_feature['succ_call'] = xs_feature.callstatus
#     no_connection = (xs_feature.call_effec == '失败') & (xs_feature.callstatus == 1)
#     xs_feature.loc[no_connection, 'succ_call'] = 0 

#     xs_feature = drop_columns_x_and_s(xs_feature)

#     xs_feature = speedy_process(xs_feature, cat_colnames=['sex_id', 'client_type_id', 
#       'isrecharged', 'rechargestatus', 'isinvited'], fix_colnames=['succ_call'])

#     # Convert labels
#     label_train.replace(label_dict, inplace=True)
#     label_test.replace(label_dict, inplace=True)
#     # label_all.replace(label_dict, inplace=True)

#     # Split train/test
#     xs_train = xs_feature.iloc[:len(data_train), :]
#     xs_test = xs_feature.iloc[len(data_train):, :]

#     # Train model
#     start_time = time.time()

#     clf = RandomForestClassifier(n_estimators=1024, min_samples_leaf=25, max_features='log2', n_jobs=-1, random_state=13, oob_score=True)
#     clf.fit(xs_train, label_train)

#     print("Training took {} seconds.".format(time.time() - start_time))

#     # Save model
#     with open('rf_lift_model.sav', 'wb') as handle:
#             pickle.dump(clf, handle)

#     # Load model
#     # with open('rf_lift_model.sav', 'rb') as handle:
#     #     clf = pickle.load(handle)

#     y_test_hat = clf.predict_proba(xs_test)
#     y_test_hat = y_test_hat[:, 1]

#     # print(roc_auc_score(label_test, y_test_hat))

#     # Create fake data from _test
#     xs_test_real = xs_test.copy()
#     xs_test_fake = xs_test_real.copy()

#     # Make fake succ_call column
#     succ_call = (xs_test_real.succ_call == 1)
#     fake_succ_call = ~succ_call
#     fake_succ_call = fake_succ_call.astype(int)

#     xs_test_fake.succ_call = fake_succ_call

#     # Reset indices for joining
#     xs_test_real.reset_index(drop=True, inplace=True)
#     xs_test_fake.reset_index(drop=True, inplace=True)

#     xs_test_real['user_id'] = data_test.user_id
#     xs_test_fake['user_id'] = data_test.user_id

#     # Concatenate real and fake
#     real_and_fake_test = xs_test_real.append(xs_test_fake, ignore_index=True)

#     # Separate invested/not_invested
#     y_invested = real_and_fake_test[real_and_fake_test.succ_call == 1]
#     y_not_invested = real_and_fake_test[real_and_fake_test.succ_call == 0]

#     # Sort according to user_id
#     y_invested = y_invested.sort_values('user_id')
#     y_not_invested = y_not_invested.sort_values('user_id')

#     # Drop user_id
#     y_invested = y_invested.drop('user_id', axis=1)
#     y_not_invested = y_not_invested.drop('user_id', axis=1)

#     # Predict y_invested/y_not_invested
#     label_test_invested = clf.predict_proba(y_invested)
#     label_test_not_invested = clf.predict_proba(y_not_invested)

#     # Get lift for everybody
#     lift = label_test_invested[:, 1] - label_test_not_invested[:, 1]

#     # Sort data_test
#     data_test_sorted = data_test.sort_values('user_id')
#     data_test_sorted['lift'] = lift

#     user_lift_score = data_test_sorted.loc[:, ['user_id', 'lift']]

#     # display(user_lift_score.head())
#     update_db_lift(user_lift_score, 'ck_lift_score_staging')

if __name__ == '__main__':
    mode = 'staging'

    training_filename = 'training61.csv'
    testing_filename = 'testing61.csv'

    label_dict = {'未投资': 0, '已投资': 1}

    # schedule_time = datetime.datetime(2017, 9, 29, 7, 30)
    # schedule_time = datetime.datetime.now()
    # scheduler(schedule_time)
    main()
    # lift_main()
