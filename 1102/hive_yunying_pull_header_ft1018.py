#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 19:18:59 2017

@author: finup
"""

# -*- coding: utf-8 -*-
# require hive server2
# --------------------------------------------
# Author: Lich_Amnesia <alwaysxiaop@gmail.com>
# Date: 2016-03-09
# --------------------------------------------
import pyhs2
import pandas as pd
import time

sql = '''
SELECT 
    *
FROM qianjin_test.yunying_base_dt0831_ft1018
where 1 = 0

'''

class HiveClient:
    # create connection to hive server2
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
            #print val
            columnNames = [a['columnName'] for a in cursor.getSchema()]
            df = pd.DataFrame(data=val, columns=columnNames)
            # print df.head()
            return df
            # return cursor.fetch()
    def close(self):
        self.conn.close()
def main():
    start_time = time.time()
    hive_client = HiveClient(db_host='192.168.176.53', port=15000, user='zhouxiaolei',
            password='5bb785da9dec898163091d80dd61d5cd', database='qianjin_test', authMechanism='PLAIN')
    global sql
    result = hive_client.query(sql)
    # print result
    print "pulling data is done --- %s seconds ---" % round((time.time() - start_time),2)
    result.to_csv('header_dt0831_ft1018.csv', encoding='utf-8',index=False,index_label=True)
    print "saving data is done --- %s seconds ---" % round((time.time() - start_time),2)
    hive_client.close()

def sched_pull():
    from apscheduler.schedulers.blocking import BlockingScheduler
#    from apscheduler.scheduler import Scheduler
    import logging
    logging.basicConfig()
    scheduler = BlockingScheduler()
    scheduler.add_job(main, 'interval', seconds=20)
    scheduler.start()

if __name__ == '__main__':
    main()


