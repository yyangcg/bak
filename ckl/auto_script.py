# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import time
import pyhs2
import sqlparse
import datetime
import pymysql as mdb
from IPython.display import display
import sys
import pickle
from apscheduler.schedulers.blocking import BlockingScheduler
import logging

def time_func(delta):
    return (datetime.date.today() - datetime.timedelta(days=delta)).strftime("%Y-%m-%d")

def scheduler(schedule_time):
    print('Scheduler will start at {}'.format(schedule_time))
    logging.basicConfig()
    scheduler = BlockingScheduler()
    scheduler.add_job(main, 'interval', seconds=86400, next_run_time=schedule_time)
    # scheduler.add_job(main, 'interval', seconds=1000, next_run_time=schedule_time)
    scheduler.start()

def drop_plus_date_rows(df):
    conditions = df.plus_date == 0
    return df.loc[~conditions, :]

def get_data():
    global training_filename
    global testing_filename
    data_train = pd.read_csv(training_filename)
    data_test = pd.read_csv(testing_filename)

    return data_train, data_test

def get_labels(data_train, data_test):
    return data_train.isinvested, data_test.isinvested

def combine_data(data_train, data_test):
    data_all = data_train.append(data_test, ignore_index=True)
    return data_all

def drop_columns(df):
    drop_useless_cols = ['user_id', 'mobile', 'pno']
    drop_might_be_useful_cols = ['channel_id', 'create_date_id', 'create_time', 'areacode', 'devicetype']
    drop_future_cols = ['date_id', 'invest_time', 'active_num', 'lastday_before_invest', 'lianxuday_last', 'rate']
    drop_alter_label = ['plus_date']
    drop_call_cols = ['callstatus', 'call_effec', 'gap_calltoinvest', 'last_calltime',
           'call_times', 'isactive_aftercall']
    # drop_other_cols = ['etl_time']
    drop_other_cols = []

    drop_total = list(set(drop_useless_cols + drop_might_be_useful_cols + drop_future_cols + drop_alter_label + drop_call_cols + drop_other_cols))
#     print(len(drop_useless_cols) + len(drop_might_be_useful_cols) + len(drop_future_cols), len(drop_alter_label), len(drop_call_cols), len(drop_other_cols), len(drop_total))
    df_modified = df.drop(drop_total, axis=1)
#     print("{} cols dropped.".format(len(cols_to_drop_tuple)))

    # Drop the stay-series, to be added back
    df_modified = df_modified.iloc[:, :10]
    
    return df_modified

def speedy_process(df, cat_colnames, fix_colnames=[]):
    # Process numeric features
    for colname in set(df.columns).difference(set(cat_colnames)).difference(set(fix_colnames)):
        # Create NA indicator
        df[colname + '_na'] = df[colname].isnull().astype(int)
        # Fill NA using median
        df[colname] = df[colname].fillna(df[colname].median())
        # Normalization
        df[colname] = (df[colname] - df[colname].min()) / (df[colname].max() - df[colname].min())
        
    # Process categorical features
    for cat_colname in cat_colnames:
        one_hot_df = pd.get_dummies(df[cat_colname], prefix=cat_colname, drop_first=True, dummy_na=True)
        df = df.join(one_hot_df)
        df.drop([cat_colname], axis=1, inplace=True)
    
    return df

def feature_transform(data_all):
    feature_all = data_all.drop('isinvested', axis=1)
    feature_all = drop_columns(feature_all)

    feature_all = speedy_process(feature_all , cat_colnames=['sex_id', 'client_type_id', 
                                                        'isrecharged', 'rechargestatus', 'isinvited'])

    return feature_all

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

# def pdo_transform(score, eps=0.00001):
    
#     score= score + eps
    
#     new_score = 632.0 + 62.0 * (np.log(score) - np.log(1 - score + 2 * eps))
    
#     return new_score
    
    
def create_mysql_csv(data_test, score, filename):
    # mysql_cols = ['user_id', 'mobile']
    mysql_cols = ['user_id', 'mobile', 'areacode', 'active_num', 'sex_id', 'isrecharged', 
                  'rechargestatus', 'user_age', 'channel_id', 'score']
    mysql_df = data_test.loc[:, mysql_cols]
    mysql_df['score'] = score

    mysql_df.to_csv(filename)

    return mysql_df

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

def eval_ks(y_true, y_pred):
    target_oos = y_pred
    # rf_results = pd.DataFrame({'prediction':target_oos[:, 1],"label":y_true})
    rf_results = pd.DataFrame({'prediction':target_oos,"label":y_true})
    ks_dis = calc_ks(rf_results, 10, prediction="prediction")
#     print(max(ks_dis))
    ks_cont = calc_continus_ks(rf_results, prediction="prediction")
#     print(max(ks_cont))
    return max(ks_dis), max(ks_cont)
    
def create_csv_ks(filename, ks):
    datetime_ks = [datetime.date.today(), ks]
    with open(filename, 'ab') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(datetime_ks)

def add_new_record_to_sql_table(conn, sql_cursor, data_tuple, table, db='policy_jtyy'):
    # sql_query = '''INSERT INTO ''' + db + '''.''' + table + ''' (user_id, mobile, areacode, active_num, sex_id, isrecharged, rechargestatus, user_age, channel_Id, score) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''', data_tuple
    # sql_query = ('''INSERT INTO ''' + db + '''.''' + table + ''' (user_id, mobile, areacode, active_num, sex_id, isrecharged, rechargestatus, user_age, channel_Id, score) VALUES ({0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9})''').format(*data_tuple)
    # sql_query = ('''INSERT INTO `ck_user_id_score` (user_id, mobile) VALUES ({0}, {1})''').format(*data_tuple)
    # sql_cursor.execute(sql_query)
    sql_cursor.execute('''INSERT INTO ''' + table + ''' (user_id, mobile, areacode, active_num, sex_id, isrecharged, rechargestatus, user_age, channel_Id, score, score_aug) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ''', tuple(str(x) if x is not None else None for x in data_tuple))
    # Make commitment to the above query
    conn.commit()

def time_func(delta):
    return (datetime.date.today() - datetime.timedelta(days=delta)).strftime("%Y-%m-%d")

class HiveClient:
    # create connection to hive server2
    MAX_BLOCK_SIZE = 2
    def __init__(self, db_host, user, password, database, port=10000, authMechanism="PLAIN"):
        self.conn = pyhs2.connect(host=db_host,
                                  port=port,
                                  authMechanism=authMechanism,
                                  user=user,
                                  password=password,
                                  database=database,
                                  )
       
    def query(self, sql):
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            val = cursor.fetchall()
#            #print val
            columnNames = [a['columnName'] for a in cursor.getSchema()]
            df = pd.DataFrame(data=val, columns=columnNames)
#            # print df.head()
            return df
            # return cursor.fetch()
    def close(self):
        self.conn.close()

def pull_data(sql_train, sql_test):
    print("Begin pulling data...")

    statistic_dayx, statistic_dayy = '"' + time_func(38) + '"', '"' + time_func(8) + '"'
    statistic_dayb, statistic_daye = '"' + time_func(1) + '"', '"' + time_func(1) + '"'

    partition_date_train = '"' + time_func(7) + '"'
    partition_date_test = '"' + time_func(0) + '"'
    # partition_date_label = '''"2017-09-27"'''

    print("Training dates: {0}, {1}".format(statistic_dayx, statistic_dayy))
    print("Prediction dates: {0}, {1}".format(statistic_dayb, statistic_daye))
    print("Partition date: {}".format(partition_date_test))

    start_time = time.time()
    hive_client = HiveClient(db_host='192.168.176.53', port=15000, user='zhouxiaolei',
            password='5bb785da9dec898163091d80dd61d5cd', database='qianjin_test', authMechanism='PLAIN')
    ##pull train data
    # global sql_train
    # global sql_test
    
    sql_train = unicode(sql_train, 'utf-8')
    parsedsql_train = sqlparse.split(sql_train)
    for elt in enumerate(parsedsql_train):
        sqltmp = elt[1].replace(';','').replace('statistic_dayx', statistic_dayx).replace('statistic_dayy', statistic_dayy).replace('partition_date', partition_date_train).encode('utf8')
        print(sqltmp)
        result = hive_client.query(sqltmp)
        print(result.head())
        print("pulling data is done --- %s seconds ---" % round((time.time() - start_time),2))
        filename = 'training' + str(elt[0]) +'.csv' 
        result.to_csv(filename, encoding='utf-8',index=False,index_label=True)
        print("saving data is done --- %s seconds ---" % round((time.time() - start_time),2))
    # hive_client.close()


    # hive_client = HiveClient(db_host='192.168.176.53', port=15000, user='zhouxiaolei',
    #         password='5bb785da9dec898163091d80dd61d5cd', database='qianjin_test', authMechanism='PLAIN')
    ##pull test data    
    sql_test =unicode(sql_test,'utf-8')
    parsedsql_test = sqlparse.split(sql_test)
    for elt in enumerate(parsedsql_test):
        sqltmp = elt[1].replace(';','').replace('statistic_dayb', statistic_dayb).replace('statistic_daye', statistic_daye).replace('partition_date', partition_date_test).encode('utf8')
        print(sqltmp)
        result = hive_client.query(sqltmp)
        print(result.head())
        print("pulling data is done --- %s seconds ---" % round((time.time() - start_time),2))
        filename = 'testing' + str(elt[0]) +'.csv' 
        result.to_csv(filename, encoding='utf-8',index=False,index_label=True)
        print("saving data is done --- %s seconds ---" % round((time.time() - start_time),2))
    hive_client.close()

    print("Finished pulling data...")

def sched_pull():
    from apscheduler.schedulers.blocking import BlockingScheduler
#    from apscheduler.scheduler import Scheduler
    import logging
    logging.basicConfig()
    scheduler = BlockingScheduler()
    scheduler.add_job(main, 'interval', seconds=20)
    scheduler.start()

def main():
    print("Process running at {}".format(datetime.datetime.now()))
    start_time = time.time()

    # with open('user_acquisition_partition_limited.sql', 'r') as handle:
    #     sql_train = handle.read().replace('\n', ' ')

    sql = '''
    DROP TABLE IF EXISTS qianjin_test.test_yimei0719_track;


    DROP TABLE IF EXISTS qianjin_test.test_yimei07191_track;


    DROP TABLE IF EXISTS qianjin_test.test_yimei07192_track;


    DROP TABLE IF EXISTS qianjin_test.test_waihu0719_track;


    DROP TABLE IF EXISTS qianjin_test.test0719_track;


    DROP TABLE IF EXISTS qianjin_test.test07191_track;


    DROP TABLE IF EXISTS qianjin_test.test0719_app_track;


    DROP TABLE IF EXISTS qianjin_test.test0719_waihu_track;


    DROP TABLE IF EXISTS qianjin_test.test0719_gap_1_track;


    DROP TABLE IF EXISTS qianjin_test.test0719_gap_2_track;


    DROP TABLE IF EXISTS qianjin_test.test0719_gap_track;


    DROP TABLE IF EXISTS qianjin_test.test0719_lianxu_track;


    DROP TABLE IF EXISTS qianjin_test.test07191_final_1_track;


    DROP TABLE IF EXISTS qianjin_test.final_train_baiyang;


    DROP TABLE IF EXISTS qianjin_test.app_log_action_track;


    DROP TABLE IF EXISTS qianjin_test.test_finaltrack_new;


    CREATE TABLE qianjin_test.test_finaltrack_new AS
    SELECT *
    FROM qianjin_develop.test_finaltrack_new
    WHERE 0=1;


    CREATE TABLE qianjin_test.test_yimei0719_track (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100));


    CREATE TABLE qianjin_test.test_yimei07191_track (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int);


    CREATE TABLE qianjin_test.test_yimei07192_track (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int, invited_id int, isinvited int, rank_id int, lastday_invite int, lastamt_invite int);


    CREATE TABLE qianjin_test.test_waihu0719_track (user_id varchar(100), status int, max_call_time TIMESTAMP, min_call_time TIMESTAMP, call_times int);


    CREATE TABLE qianjin_test.test0719_track (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int, invited_id int, isinvited int, rank_id int, lastday_invite int, lastamt_invite int, callstatus int);


    CREATE TABLE qianjin_test.test07191_track (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int, invited_id int, isinvited int, rank_id int, lastday_invite int, lastamt_invite int, callstatus int, call_effec varchar(100), Gap_CalltoInvest int, last_calltime TIMESTAMP, call_times int);


    CREATE TABLE qianjin_test.test0719_app_track (user_Id varchar(100), pno varchar(100), staytime INT, staynum int, staytime_recharge int, staynum_recharge int, staytime_p3 int, staynum_p3 int, staytime_p6 int, staynum_p6 int, staytime_p12 int, staynum_p12 int, staytime_ph int, staynum_ph int, staytime_p1 int, staynum_p1 int, staytime_buy int, staynum_buy int, staytime_asset int, staynum_asset int, staytime_redbag int, staynum_redbag int, staytime_hudong int, staynum_hudong int, staynum_banner int, staynum_icon1 int, staynum_icon2 int, staynum_icon3 int , devicetype varchar(100));


    CREATE TABLE qianjin_test.test0719_waihu_track (user_Id varchar(100), pno varchar(100), isactive_aftercall int);


    CREATE TABLE qianjin_test.test0719_gap_1_track (user_Id varchar(100), pno varchar(100), create_date_id int, date_id int, active_dateid varchar(100));


    CREATE TABLE qianjin_test.test0719_gap_2_track (user_Id varchar(100), pno varchar(100), create_date_id int, date_id int, active_dateid varchar(100), rank int);


    CREATE TABLE qianjin_test.test0719_gap_track (user_Id varchar(100), pno varchar(100), active_num int, Lastday_before_invest INT);


    CREATE TABLE qianjin_test.test0719_lianxu_track (user_Id varchar(100), pno varchar(100), lianxuday_last INT);


    CREATE TABLE qianjin_test.test07191_final_1_track (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int, invited_id int, isinvited int, rank_id int, lastday_invite int, lastamt_invite int, callstatus int, call_effec varchar(100), Gap_CalltoInvest int, last_calltime TIMESTAMP, call_times int, staytime INT, staynum int, staytime_recharge int, staynum_recharge int, staytime_p3 int, staynum_p3 int, staytime_p6 int, staynum_p6 int, staytime_p12 int, staynum_p12 int, staytime_ph int, staynum_ph int, staytime_p1 int, staynum_p1 int, staytime_buy int, staynum_buy int, staytime_asset int, staynum_asset int, staytime_redbag int, staynum_redbag int, staytime_hudong int, staynum_hudong int, staynum_banner int, staynum_icon1 int, staynum_icon2 int, staynum_icon3 int , devicetype varchar(100), isactive_aftercall int, active_num int, Lastday_before_invest INT, lianxuday_last INT, rate double, Get_date_id int);


    CREATE TABLE qianjin_test.final_train_baiyang (user_Id varchar(100), mobile varchar(100), IsInvested varchar(10), user_age int, sex_id int, create_date_Id int, create_time TIMESTAMP, date_id int, invest_time TIMESTAMP, areacode varchar(20), plus_date int, client_type_Id int, channel_id int, pno varchar(100), isrecharged int, rechargestatus int, isinvited int, lastday_invite int, lastamt_invite int, callstatus int, call_effec varchar(100), Gap_CalltoInvest int, last_calltime TIMESTAMP, call_times int, staytime INT, staynum int, staytime_recharge int, staynum_recharge int, staytime_p3 int, staynum_p3 int, staytime_p6 int, staynum_p6 int, staytime_p12 int, staynum_p12 int, staytime_ph int, staynum_ph int, staytime_p1 int, staynum_p1 int, staytime_buy int, staynum_buy int, staytime_asset int, staynum_asset int, staytime_redbag int, staynum_redbag int, staytime_hudong int, staynum_hudong int, staynum_banner int, staynum_icon1 int, staynum_icon2 int, staynum_icon3 int , devicetype varchar(100), isactive_aftercall int, active_num int, Lastday_before_invest INT, lianxuday_last INT, rate double);


    CREATE TABLE qianjin_test.app_log_action_track (pid varchar(100), userid int, dateid varchar(20), pageid varchar(100), lab varchar(5), act varchar(200), actiontime varchar(100), exittime varchar(100), clienttype int, devicetype varchar(100));


    INSERT overwrite TABLE qianjin_test.test_yimei0719_track
    SELECT DISTINCT b.user_id ,
                    b.mobile,
                    CASE
                        WHEN e.user_id IS NOT NULL THEN '已投资'
                        ELSE '未投资'
                    END IsInvested,
                    b.user_age,
                    b.sex_id,
                    b.create_date_Id,
                    from_unixtime(b.create_time/1000) create_time,
                    e.date_id,
                    from_unixtime(invest_time/1000) invest_time,
                    b.city_id areacode,
                    CASE
                        WHEN e.date_Id IS NULL THEN 'NA'
                        ELSE datediff(concat(substring(date_Id,1,4),'-',substring(date_Id,5,2),'-',substring(date_Id,7,2)),concat(substring(create_date_Id,1,4),'-',substring(create_date_Id,5,2),'-',substring(create_date_Id,7,2)))
                    END plus_date,
                    b.client_type_id,
                    b.channel_Id ,
                    f.pno
    FROM qianjin_mid.qianjin_mid_user_detail_temp b
    LEFT JOIN
      (SELECT *
       FROM qianjin_mid.qianjin_mid_user_invest_detail -- isinvested
       WHERE rank_id = 1
         AND bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
         AND bdp_type = 's') e ON b.user_id = e.user_Id
    LEFT JOIN
      (SELECT DISTINCT user_id,
                       max(CASE WHEN TYPE = 2 THEN pid ELSE pno END) pno
       FROM qianjin_mid.qianjin_mid_app_token -- pno
       WHERE pno IS NOT NULL
         AND bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
         AND bdp_type = 's'
       GROUP BY user_id) f ON b.user_id = f.user_id
    WHERE concat(substring(b.create_date_id,1,4),'-',substring(b.create_date_id,5,2),'-',substring(b.create_date_id,7,2)) BETWEEN statistic_dayx AND statistic_dayy
      AND b.channel_categ_id != 73
      AND b.bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
      AND b.bdp_type = 's' ;


    INSERT overwrite TABLE qianjin_test.test_yimei07191_track
    SELECT DISTINCT a.*,
                    max(CASE WHEN b.user_id IS NOT NULL THEN 1 ELSE 0 END) isrecharged,
                    max(b.recharge_status_id) rechargestatus
    FROM test_yimei0719_track a
    LEFT JOIN qianjin_mid.qianjin_mid_user_recharge_detail b ON a.user_id = b.user_id
    AND to_date(concat(substring(a.create_date_Id,1,4),'-',substring(a.create_date_Id,5,2),'-',substring(a.create_date_Id,7,2))) = to_date(from_unixtime(b.recharge_time/1000))
    GROUP BY a.user_Id,
             a.mobile,
             a.isinvested,
             a.user_age,
             a.sex_id,
             a.create_date_id,
             a.create_time,
             a.date_id,
             a.invest_time,
             a.areacode,
             a.plus_date,
             a.client_type_Id,
             a.channel_id,
             a.pno ;


    INSERT overwrite TABLE qianjin_test.test_yimei07192_track -- isinvited
    SELECT DISTINCT a.* ,
                    CASE
                        WHEN a.invest_time IS NOT NULL THEN datediff(concat(substring(a.date_id,1,4),'-',substring(a.date_id,5,2),'-',substring(a.date_id,7,2)),concat(substring(b.date_id,1,4),'-',substring(b.date_id,5,2),'-',substring(b.date_id,7,2)))
                        ELSE datediff('partition_date', concat(substring(b.date_id,1,4),'-',substring(b.date_id,5,2),'-',substring(b.date_id,7,2)))
                    END Lastday_invite ,
                    b.invest_amt Lastamt_invite
    FROM
      (SELECT a.user_Id,
              a.mobile,
              a.isinvested,
              a.user_age,
              a.sex_id,
              a.create_date_id,
              a.create_time,
              a.date_id,
              a.invest_time,
              a.areacode,
              a.plus_date,
              a.client_type_Id,
              a.channel_id,
              a.pno,
              a.isrecharged,
              a.rechargestatus,
              b.invited_id,
              CASE
                  WHEN b.invited_Id IS NOT NULL THEN 1
                  ELSE 0
              END IsInvited ,
              max(C.rank_id) rank_id
       FROM test_yimei07191_track a
       JOIN
         (SELECT *
          FROM qianjin_mid.qianjin_mid_user_detail_temp
          WHERE bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
            AND bdp_type = 's') b ON a.user_id = b.user_id
       LEFT JOIN
         (SELECT *
          FROM qianjin_mid.qianjin_mid_user_invest_detail
          WHERE bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
            AND bdp_type = 's') c ON b.invited_id = c.user_Id
       AND (a.invest_time >= from_unixtime(c.invest_time/1000)
            OR a.invest_time IS NULL)
       GROUP BY a.user_Id,
                a.mobile,
                a.isinvested,
                a.user_age,
                a.sex_id,
                a.create_date_id,
                a.create_time,
                a.date_id,
                a.invest_time,
                a.areacode,
                a.plus_date,
                a.client_type_Id,
                a.channel_id,
                a.pno,
                a.isrecharged,
                a.rechargestatus,
                b.invited_id) a
    LEFT JOIN
      (SELECT *
       FROM qianjin_mid.qianjin_mid_user_invest_detail
       WHERE bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
         AND bdp_type = 's') b ON a.invited_id = b.user_id
    AND a.rank_id = b.rank_id;


    INSERT overwrite TABLE qianjin_test.test_waihu0719_track -- call info
    SELECT user_id,
           max(phone_status) status,
           max(from_unixtime(cast(create_time/1000 AS int))) max_call_time ,
           min(from_unixtime(cast(create_time/1000 AS int))) min_call_time,
           count(*) call_times
    FROM
      (SELECT DISTINCT user_id,
                       phone_status,
                       create_time,
                       call_type
       FROM qianjin_crm.qianjin_crm_outcall_work_order
       WHERE bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
         AND bdp_type = 's') a
    WHERE call_type = 1
    GROUP BY user_id;


    INSERT overwrite TABLE qianjin_test.test0719_track
    SELECT DISTINCT a.* ,
                    CASE
                        WHEN b.call_times IS NOT NULL
                             AND a.invest_time IS NOT NULL
                             AND (b.min_call_time > a.create_time
                                  OR a.invest_time > b.max_call_time) THEN 1
                        WHEN b.call_times IS NOT NULL
                             AND a.invest_time IS NULL
                             AND (b.min_call_time > a.create_time
                                  OR b.max_call_time > a.create_time) THEN 1
                        ELSE 0
                    END callstatus
    FROM qianjin_test.test_yimei07192_track a
    LEFT JOIN qianjin_test.test_waihu0719_track b ON a.user_Id = b.user_id;


    INSERT overwrite TABLE qianjin_test.test07191_track
    SELECT DISTINCT a.* ,
                    CASE
                        WHEN b.status = 3
                             AND a.callstatus = 1 THEN '成功'
                        WHEN b.status = 4
                             AND a.callstatus = 1 THEN '预约下次'
                        WHEN b.status = 5
                             AND a.callstatus = 1 THEN '拒访'
                        ELSE '失败'
                    END call_effec ,
                    CASE
                        WHEN a.callstatus = 1 THEN datediff(a.invest_time,max_call_time)
                    END Gap_CalltoInvest ,
                    CASE
                        WHEN a.callstatus = 1
                             AND a.invest_time >= b.max_call_time THEN b.max_call_time
                        WHEN a.callstatus = 1
                             AND a.invest_time < b.max_call_time THEN b.min_call_time
                        WHEN a.callstatus = 1
                             AND a.invest_time IS NULL THEN b.max_call_time
                    END last_calltime ,
                    b.call_times
    FROM qianjin_test.test0719_track a
    LEFT JOIN qianjin_test.test_waihu0719_track b ON a.user_Id = b.user_id ;


    INSERT overwrite TABLE qianjin_test.app_log_action_track
    SELECT pid,
           userid,
           dateid,
           pageid,
           lab,
           act,
           actiontime,
           exittime,
           clienttype,
           devicetype
    FROM qianjin.qianjin_app_log b
    WHERE concat(substring(b.bdp_day,1,4),'-',substring(b.bdp_day,5,2),'-',substring(b.bdp_day,7,2)) BETWEEN statistic_dayx AND statistic_dayy ;


    INSERT overwrite TABLE qianjin_test.test0719_app_track
    SELECT a.user_id,
           a.pno,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND exittime>=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('RechargeActivity','RechargeMainViewController', 'JoinConfirmRechargeNewActivity','BuyConfirmViewController')
               AND exittime >=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_recharge,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('RechargeActivity','RechargeMainViewController', 'JoinConfirmRechargeNewActivity','BuyConfirmViewController')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_recharge,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('DPlanActivity3','ProductionDetailDPViewController_3')
               AND exittime >=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_p3,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('DPlanActivity3','ProductionDetailDPViewController_3')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_p3,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('DPlanActivity6','ProductionDetailDPViewController_6')
               AND exittime >=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_p6,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('DPlanActivity6','ProductionDetailDPViewController_6')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_p6,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('DPlanActivity12','ProductionDetailDPViewController_12')
               AND exittime >=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_p12,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('DPlanActivity12','ProductionDetailDPViewController_12')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_p12,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('IHuoBaoDetailActivity','ProductionDetailCTViewController')
               AND exittime >= actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_ph,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('IHuoBaoDetailActivity','ProductionDetailCTViewController')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_ph,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('DPlanActivity1','ProductionDetailDPViewController_1')
               AND exittime >=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_p1,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('DPlanActivity1','ProductionDetailDPViewController_1')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_p1,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('JoinDepositActivity3','BuyDPViewController_3' ,'JoinDepositActivity6','BuyDPViewController_6' ,'JoinDepositActivity12','BuyDPViewController_12' ,'JoinCurrentActivity','BuyCRTViewController','AddBookingActivity','BuyRSViewController')
               AND exittime>=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_buy,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('JoinDepositActivity3','BuyDPViewController_3' ,'JoinDepositActivity6','BuyDPViewController_6' ,'JoinDepositActivity12','BuyDPViewController_12' ,'JoinCurrentActivity','BuyCRTViewController','AddBookingActivity','BuyRSViewController')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_buy,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('AssetsNewActivity','AssetsMainViewController')
               AND exittime>=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_asset,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('AssetsNewActivity','AssetsMainViewController')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_asset,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('MyRewardActivity','RewardViewController')
               AND exittime>=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_redbag,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('MyRewardActivity','RewardViewController')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_redbag,
           sum(CASE WHEN (actiontime <=invest_time
                          OR invest_time IS NULL
                          OR a.invest_time ='')
               AND pageid IN ('InterActionActivity','InteractionMainViewController')
               AND exittime>=actiontime THEN (unix_timestamp(exittime)-unix_timestamp(actiontime)) END) staytime_hudong,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid IN ('InterActionActivity','InteractionMainViewController')
                 AND substring(pageid,1,1) != '0' THEN pageid END) staynum_hudong,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid ='010000'
                 AND act IN ('点击banner','点击baner') THEN pageid END) staynum_banner,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid ='010000'
                 AND act IN ('点击icon')
                 AND (c.clienttype =2
                      AND c.lab = 1
                      OR c.clienttype =3
                      AND c.lab = 0) THEN pageid END) staynum_icon1,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid ='010000'
                 AND act IN ('点击icon')
                 AND (c.clienttype =2
                      AND c.lab = 2
                      OR c.clienttype =3
                      AND c.lab = 1) THEN pageid END) staynum_icon2,
           count(CASE WHEN (actiontime <=invest_time
                            OR invest_time IS NULL
                            OR a.invest_time ='')
                 AND pageid ='010000'
                 AND act IN ('点击icon')
                 AND (c.clienttype =2
                      AND c.lab = 3
                      OR c.clienttype =3
                      AND c.lab = 2) THEN pageid END) staynum_icon3,
           max(c.devicetype) devicetype
    FROM qianjin_test.test07191_track a
    LEFT JOIN qianjin_test.app_log_action_track c ON a.pno = c.pid
    AND concat(substring(a.create_date_id,1,4),'-',substring(a.create_date_id,5,2),'-',substring(a.create_date_id,7,2)) = concat(substring(c.dateid,1,4),'-',substring(c.dateid,5,2),'-',substring(c.dateid,7,2))
    GROUP BY a.user_id,
             a.pno;


    INSERT overwrite TABLE qianjin_test.test0719_waihu_track
    SELECT a.user_id,
           a.pno,
           MAX (CASE
                    WHEN a.last_calltime IS NOT NULL
                         AND a.last_calltime < b.actiontime
                         AND to_date(a.last_calltime) = to_date(b.actiontime)
                         AND pageid !='060000' THEN 1
                    ELSE 0
                END) isactive_aftercall
    FROM qianjin_test.test07191_track a
    LEFT JOIN qianjin_test.app_log_action_track b ON a.pno = b.pid
    GROUP BY a.user_id,
             a.pno;


    INSERT overwrite TABLE qianjin_test.test0719_gap_1_track
    SELECT DISTINCT a.user_id,
                    a.pno,
                    a.create_date_Id,
                    a.date_id,
                    CASE
                        WHEN b.actiontime BETWEEN a.create_time AND a.invest_time THEN to_date(b.actiontime)
                        WHEN b.actiontime >= a.create_time
                             AND (a.invest_time IS NULL
                                  OR a.invest_time ='')
                             AND to_date(b.actiontime) <= date_add(a.create_time,30) THEN to_date(b.actiontime)
                    END active_dateid
    FROM qianjin_test.test07191_track a
    LEFT JOIN qianjin_test.app_log_action_track b ON a.pno = b.pid;


    INSERT overwrite TABLE qianjin_test.test0719_gap_2_track
    SELECT *,
           row_number() over (partition BY user_id
                              ORDER BY active_dateid) rank
    FROM qianjin_test.test0719_gap_1_track
    WHERE active_dateid IS NOT NULL;


    INSERT overwrite TABLE qianjin_test.test0719_gap_track
    SELECT user_id,
           pno,
           count(CASE WHEN active_dateid IS NOT NULL THEN user_id END) active_num ,
           MAX(CASE
                   WHEN date_id IS NOT NULL
                        AND active_dateid < concat(substring(date_id,1,4),'-',substring(date_id,5,2),'-',substring(date_id,7,2)) THEN date_id
               END) Lastday_before_invest
    FROM qianjin_test.test0719_gap_2_track
    GROUP BY user_id,
             pno;


    INSERT overwrite TABLE qianjin_test.test0719_lianxu_track
    SELECT a.user_id,
           a.pno,
           MAX(a.rank) - MAX(CASE
                                 WHEN c.active_Num =1 THEN 1
                                 WHEN (datediff(a.active_dateid,b.active_dateid) !=1
                                       OR b.active_dateid IS NULL)
                                      AND c.active_Num !=1 THEN a.rank
                             END) lianxuday_last
    FROM qianjin_test.test0719_gap_2_track a
    JOIN qianjin_test.test0719_gap_track c ON a.user_Id = c.user_id
    LEFT JOIN qianjin_test.test0719_gap_2_track b ON a.user_id = b.user_id
    AND a.rank = b.rank+1
    GROUP BY a.user_id,
             a.pno,
             c.active_Num;


    INSERT overwrite TABLE qianjin_test.test07191_final_1_track
    SELECT a.*,
           b.staytime,
           b.staynum,
           b.staytime_recharge,
           b.staynum_recharge,
           b.staytime_p3,
           b.staynum_p3,
           b.staytime_p6,
           b.staynum_p6,
           b.staytime_p12,
           b.staynum_p12,
           b.staytime_ph,
           b.staynum_ph,
           b.staytime_p1,
           b.staynum_p1,
           b.staytime_buy,
           b.staynum_buy,
           b.staytime_asset,
           b.staynum_asset,
           b.staytime_redbag,
           b.staynum_redbag,
           b.staytime_hudong,
           b.staynum_hudong,
           b.staynum_banner,
           b.staynum_icon1,
           b.staynum_icon2,
           b.staynum_icon3,
           b.devicetype,
           c.isactive_aftercall,
           d.active_num,
           d.Lastday_before_invest,
           e.lianxuday_last,
           f.rate,
           '20170818' AS Get_date_id
    FROM qianjin_test.test07191_track a
    JOIN qianjin_test.test0719_app_track b ON a.user_id = b.user_Id
    JOIN qianjin_test.test0719_waihu_track c ON a.user_id = c.user_id
    LEFT JOIN qianjin_test.test0719_gap_track d ON a.user_id = d.user_id
    LEFT JOIN qianjin_test.test0719_lianxu_track e ON a.user_id = e.user_id
    LEFT JOIN
      (SELECT a.channel_id,
              a.num/b.num rate
       FROM
         (SELECT channel_id,
                 count(DISTINCT a.user_id) num
          FROM qianjin_mid.qianjin_mid_user_detail_temp a
          JOIN qianjin_mid.qianjin_mid_user_invest_detail b ON a.user_id = b.user_id
          AND date_id = create_date_id
          WHERE concat(substring(a.create_date_id,1,4),'-',substring(a.create_date_id,5,2),'-',substring(a.create_date_id,7,2)) BETWEEN statistic_dayx AND statistic_dayy
            AND a.bdp_type = 's'
            AND b.bdp_type = 's'
            AND a.bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
            AND b.bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
            AND b.rank_id = 1
          GROUP BY channel_id) a
       JOIN
         (SELECT channel_id,
                 count(DISTINCT a.user_id) num
          FROM qianjin_mid.qianjin_mid_user_detail_temp a
          WHERE concat(substring(a.create_date_id,1,4),'-',substring(a.create_date_id,5,2),'-',substring(a.create_date_id,7,2)) BETWEEN statistic_dayx AND statistic_dayy
            AND a.bdp_type = 's'
            AND a.bdp_day = concat(substring(partition_date,1,4),substring(partition_date,6,2),substring(partition_date,9,2))
          GROUP BY channel_id) b ON a.channel_id = b.channel_id) f ON a.channel_id = f.channel_id;


    INSERT overwrite TABLE qianjin_test.final_train_baiyang
    SELECT a.user_id,
           a.mobile,
           a.isinvested,
           a.user_age,
           a.sex_id,
           a.create_date_id,
           a.create_time,
           a.date_id,
           a.invest_time,
           a.areacode,
           a.plus_date,
           a.client_type_Id,
           a.channel_id,
           a.pno,
           a.isrecharged,
           a.rechargestatus,
           a.isinvited,
           a.lastday_invite,
           a.lastamt_invite,
           a.callstatus,
           a.call_effec,
           a.gap_calltoinvest,
           a.last_calltime,
           a.call_times,
           a.staytime,
           a.staynum,
           a.staytime_recharge,
           a.staynum_recharge,
           a.staytime_p3,
           a.staynum_p3,
           a.staytime_p6,
           a.staynum_p6,
           a.staytime_p12,
           a.staynum_p12,
           a.staytime_ph,
           a.staynum_ph,
           a.staytime_p1,
           a.staynum_p1,
           a.staytime_buy,
           a.staynum_buy,
           a.staytime_asset,
           a.staynum_asset,
           a.staytime_redbag,
           a.staynum_redbag,
           a.staytime_hudong,
           a.staynum_hudong,
           a.staynum_banner,
           a.staynum_icon1,
           a.staynum_icon2,
           a.staynum_icon3,
           a.devicetype,
           a.isactive_aftercall,
           a.active_num,
           a.lastday_before_invest,
           a.lianxuday_last,
           a.rate
    FROM qianjin_test.test07191_final_1_track a;


    DROP TABLE qianjin_test.test_yimei0719_track;


    DROP TABLE qianjin_test.test_yimei07191_track;


    DROP TABLE qianjin_test.test_yimei07192_track;


    DROP TABLE qianjin_test.test_waihu0719_track;


    DROP TABLE qianjin_test.test0719_track;


    DROP TABLE qianjin_test.test07191_track;


    DROP TABLE qianjin_test.test0719_app_track;


    DROP TABLE qianjin_test.test0719_waihu_track;


    DROP TABLE qianjin_test.test0719_gap_1_track;


    DROP TABLE qianjin_test.test0719_gap_2_track;


    DROP TABLE qianjin_test.test0719_gap_track;


    DROP TABLE qianjin_test.test0719_lianxu_track;


    DROP TABLE qianjin_test.test07191_final_1_track;


    DROP TABLE qianjin_test.app_log_action_track;


    SELECT *
    FROM qianjin_test.final_train_baiyang
    '''

    sql_train = sql

    # sql_train = sql_train.replace('partition_date', '''"2017-09-01"''')
    sql_test = sql.replace('statistic_dayx', 'statistic_dayb').replace('statistic_dayy', 'statistic_daye')
    sql_test = sql_test.replace('final_train_baiyang', 'final_test_baiyang').replace('final_train_baiyang', 'final_test_baiyang')

    # Pull data from Hive
    global label_dict

    pull_data(sql_train, sql_test)

    # Get data
    data_train_raw, data_test_raw = get_data()
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

    # # Load model
   #  with open(save_folder_path + 'rf_aug_model.sav', 'rb') as handle:
   #      clf_aug = pickle.load(handle)

    # Get score
    # score_aug = predict_and_get_score(clf_aug, feature_test, label_test)
    # score_aug = pdo_transform(score_aug)
    score = predict_and_get_score(clf, feature_test, label_test)
    score = pdo_transform(score)
    
    # Save score data to csv
    # mysql_df_aug = create_mysql_csv(data_test, score_aug, 'score_aug.csv')
    mysql_df_aug = create_mysql_csv(data_test, score, 'score_aug.csv')
    mysql_df = create_mysql_csv(data_test, score, 'score.csv')
    # mysql_df['score_new'] = score_aug
    mysql_df['score_new'] = score

    print(mysql_df.shape)
    display(mysql_df.head())

    update_db(mysql_df, 'ck_user_id_score')
    update_db(mysql_df, 'ck_accum', overwrite=False)

    # Writing lift scores to MySQL
    # lift_main()

    print('Entire process took {} seconds.'.format(time.time() - start_time))
    print('Waiting for next process....')

def update_db(df, table_name, overwrite=True, host='192.168.143.201', user='phdc_calvinku', password='phdc_calvinku.XLPfZrvjMlsFqSqfxOSBcDc1EOGp5Xy8', port=3306, db="policy_jtyy"):
    # Connect to server
    conn = mdb.connect(host=host, user=user, password=password, port=port, db=db, charset="utf8")
    # Create cursor
    cur = conn.cursor()
    # Set encoding
    cur.execute("SET NAMES utf8")

    if overwrite:
        # Drop table if it already exist using execute() method.
        cur.execute("DROP TABLE IF EXISTS " + table_name)
        # Create table SQL string
        create_table_sql = """
          CREATE TABLE IF NOT EXISTS """ + table_name + """
          (
          `id` bigint(100) NOT NULL AUTO_INCREMENT COMMENT 'Primary key',
          `user_id` varchar(100) COMMENT 'User ID',
          `mobile` varchar(1000) COMMENT 'Mobile Number',
          `areacode` varchar(1000) COMMENT 'Area Code',
          `active_num` bigint(100) COMMENT 'active_num',
          `sex_id` bigint(100) COMMENT 'Gender',
          `isrecharged` DOUBLE PRECISION COMMENT 'isrecharged',
          `rechargestatus` DOUBLE PRECISION COMMENT 'rechargestatus',
          `user_age` DOUBLE PRECISION COMMENT 'User Age',
          `channel_Id` DOUBLE PRECISION COMMENT 'channel_Id',
          `score` DOUBLE PRECISION NOT NULL DEFAULT '0' COMMENT 'Baiyang Team Score',
            `score_aug` DOUBLE PRECISION NOT NULL DEFAULT '0' COMMENT 'Baiyang Team Score Aug',
           PRIMARY KEY (`id`),
           UNIQUE KEY `user_id_u` (`user_id`)
           ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='iqj_acquiring_score'
          """

        cur.execute(create_table_sql)

        # global num_records
        # num_records = 200
        num_records = len(df)

        for row in df.iloc[:num_records, :].itertuples():
            data_tuple = tuple(row[1:])

            data_list = []

            for x in data_tuple:
              if type(x) == type('abc'):
                data_list.append(x)
              elif np.isnan(x):
                data_list.append(None)
              else:
                data_list.append(x)            

            data_tuple = tuple(data_list)
            print(data_tuple)
            add_new_record_to_sql_table(conn, cur, data_tuple, table=table_name)

    else:
        # global num_records
        # num_records = 200
        num_records = len(df)

        for row in df.iloc[:num_records, :].itertuples():
            data_tuple = tuple(row[1:])

            data_list = []

            for x in data_tuple:
              if type(x) == type('abc'):
                data_list.append(x)
              elif np.isnan(x):
                data_list.append(None)
              else:
                data_list.append(x)            

            data_tuple = tuple(data_list)
            print(data_tuple)
            add_new_record_to_sql_table(conn, cur, data_tuple, table=table_name)

    # Close cursor and disconnect
    cur.close()
    conn.close()
    print('Finished updating MySQL.')

def drop_stay_cols(df):
    df = df.drop(['staytime_recharge', 'staynum_recharge', 'staytime_p3', 'staynum_p3', 'staytime_p6', 'staynum_p6', 'staytime_p12', 'staynum_p12', 'staytime_ph', 'staynum_ph', 'staytime_p1', 'staynum_p1', 'staytime_buy', 'staynum_buy', 'staytime_asset', 'staynum_asset', 'staytime_redbag', 'staynum_redbag', 'staytime_hudong', 'staynum_hudong', 'staynum_banner', 'staynum_icon1', 'staynum_icon2', 'staynum_icon3'], axis=1)
    
    return df

def drop_columns_x_and_s(df):
    drop_useless_cols = ['user_id', 'mobile', 'pno']
    drop_might_be_useful_cols = ['channel_id', 'create_date_id', 'create_time', 'areacode', 'devicetype']
    drop_future_cols = ['date_id', 'invest_time', 'active_num', 'lastday_before_invest', 'lianxuday_last', 'rate']
    drop_alter_label = ['plus_date']
    drop_call_cols = ['callstatus', 'call_effec', 'gap_calltoinvest', 'last_calltime',
           'call_times', 'isactive_aftercall']
    # drop_other_cols = ['etl_time']
    drop_other_cols = []
    
    drop_total = list(set(drop_useless_cols + drop_might_be_useful_cols + drop_future_cols + drop_alter_label + drop_call_cols + drop_other_cols))

    df_modified = df.drop(drop_total, axis=1)
    
    return df_modified

def update_db_lift(df, table_name, overwrite=True, host='192.168.143.201', user='phdc_calvinku', password='phdc_calvinku.XLPfZrvjMlsFqSqfxOSBcDc1EOGp5Xy8', port=3306, db="policy_jtyy"):
    # Connect to server
    conn = mdb.connect(host=host, user=user, password=password, port=port, db=db, charset="utf8")
    # Create cursor
    cur = conn.cursor()
    # Set encoding
    cur.execute("SET NAMES utf8")

    if overwrite:
        # Drop table if it already exist using execute() method.
        cur.execute("DROP TABLE IF EXISTS " + table_name)
        # Create table SQL string
        create_table_sql = """
          CREATE TABLE IF NOT EXISTS """ + table_name + """
          (
          `id` bigint(100) NOT NULL AUTO_INCREMENT COMMENT 'Primary key',
          `user_id` varchar(100) COMMENT 'User ID',
          `lift_score` DOUBLE PRECISION NOT NULL DEFAULT '0' COMMENT 'Baiyang Team Lift Score',
           PRIMARY KEY (`id`),
           UNIQUE KEY `user_id_u` (`user_id`)
           ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='iqj_acquiring_score'
          """

        cur.execute(create_table_sql)

        num_records = len(df)

        for row in df.iloc[:num_records, :].itertuples():
            data_tuple = tuple(row[1:])

            data_list = []

            for x in data_tuple:
              if type(x) == type('abc'):
                data_list.append(x)
              elif np.isnan(x):
                data_list.append(None)
              else:
                data_list.append(x)            

            data_tuple = tuple(data_list)
            print(data_tuple)
            add_new_record_to_sql_table_lift(conn, cur, data_tuple, table=table_name)
    else:
        pass

    # Close cursor and disconnect
    cur.close()
    conn.close()
    print('lift_main() finished updating MySQL.')

def add_new_record_to_sql_table_lift(conn, sql_cursor, data_tuple, table, db='policy_jtyy'):
    sql_cursor.execute('''INSERT INTO ''' + table + ''' (user_id, lift_score) VALUES (%s, %s) ''', tuple(str(x) if x is not None else None for x in data_tuple))
    # Make commitment to the above query
    conn.commit()

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
    training_filename = 'training61.csv'
    testing_filename = 'testing61.csv'

    label_dict = {'未投资': 0, '已投资': 1}

    schedule_time = datetime.datetime(2017, 10, 18, 7, 15)
    # schedule_time = datetime.datetime.now()
    scheduler(schedule_time)
    # main()
    # lift_main()
