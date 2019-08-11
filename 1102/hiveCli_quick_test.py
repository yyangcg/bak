# -*- coding: utf-8 -*-
# require hive server2
# --------------------------------------------
# Author: Lich_Amnesia <alwaysxiaop@gmail.com>
# Date: 2016-03-09
# --------------------------------------------
import pyhs2
import pandas as pd
import time
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
            columnNames = [a['columnName'] for a in cursor.getSchema()]
            df = pd.DataFrame(data=val, columns=columnNames)
            return df
#            return cursor.fetch()
    def close(self):
        self.conn.close()
def pull_data():
    import sys

    start_time = time.time()
    print("hive client connecting ...")
#    print start_time.strftime('%l:%M%p %Z on %b %d, %Y')
    reload(sys)
    sys.setdefaultencoding("utf-8")
    hive_client = HiveClient(db_host='192.168.176.53', port=15000, user='zhouxiaolei', password='5bb785da9dec898163091d80dd61d5cd', database='qianjin_test', authMechanism='PLAIN')
    result = hive_client.query("""
SELECT * FROM qianjin_test.app_behave_test limit 20
""")

    print(result)
    print("pulling data is done --- %s seconds ---") % round((time.time() - start_time),2)
#    result.to_csv('result.csv', encoding='utf-8',index=False,index_label=True)
    print("saving data is done --- %s seconds ---") % round((time.time() - start_time),2)
    hive_client.close()

def main():
    from apscheduler.schedulers.blocking import BlockingScheduler
    import logging
    logging.basicConfig()
    scheduler = BlockingScheduler()
    scheduler.add_job(pull_data, 'interval', seconds=20)
    scheduler.start()

if __name__ == '__main__':
    main()
