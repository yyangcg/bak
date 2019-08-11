import pyhs2
import sqlparse
import pymysql as mdb
import datetime
import time
import pandas as pd
import numpy as np

def time_func(delta):
    return (datetime.date.today() - datetime.timedelta(days=delta)).strftime("%Y-%m-%d")


def add_new_record_to_sql_table(conn, sql_cursor, data_tuple, table, db='policy_jtyy'):
    # sql_query = '''INSERT INTO ''' + db + '''.''' + table + ''' (user_id, mobile, areacode, active_num, sex_id, isrecharged, rechargestatus, user_age, channel_Id, score) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''', data_tuple
    # sql_query = ('''INSERT INTO ''' + db + '''.''' + table + ''' (user_id, mobile, areacode, active_num, sex_id, isrecharged, rechargestatus, user_age, channel_Id, score) VALUES ({0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9})''').format(*data_tuple)
    # sql_query = ('''INSERT INTO `ck_user_id_score` (user_id, mobile) VALUES ({0}, {1})''').format(*data_tuple)
    # sql_cursor.execute(sql_query)
    sql_cursor.execute('''INSERT INTO ''' + table + ''' (user_id, mobile, areacode, active_num, sex_id, isrecharged, rechargestatus, user_age, channel_Id, score, score_aug) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ''', tuple(str(x) if x is not None else None for x in data_tuple))
    # Make commitment to the above query
    conn.commit()


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

    print("Training dates: {}".format(statistic_dayx, statistic_dayy))
    print("Prediction dates: {}".format(statistic_dayb, statistic_daye))
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

    # Close cursor and disconnect
    cur.close()
    conn.close()
    print('lift_main() finished updating MySQL.')


def add_new_record_to_sql_table_lift(conn, sql_cursor, data_tuple, table, db='policy_jtyy'):
    sql_cursor.execute('''INSERT INTO ''' + table + ''' (user_id, lift_score) VALUES (%s, %s) ''', tuple(str(x) if x is not None else None for x in data_tuple))
    # Make commitment to the above query
    conn.commit()