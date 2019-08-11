# -*- coding: utf-8 -*-
# require hive server2
# --------------------------------------------
# Author: Lich_Amnesia <alwaysxiaop@gmail.com>
# Date: 2016-03-09
# --------------------------------------------
import pyhs2
import pandas as pd
import time
import sqlparse
import sys

sql = '''

DROP TABLE IF EXISTS qianjin_test.yunying_base_dt0831_ft1018;
CREATE TABLE qianjin_test.yunying_base_dt0831_ft1018 AS
select
         case when a.invest1 = 0 
                 and a.invest2 = 0 
                 and a.invest3 = 0 
              then 0
              else 1
         end as is_invest_all_y
         
        , case when (cast(a.invest_zcb1_amt as float) 
                        + cast(a.invest_zcb2_amt as float) 
                        + cast(a.invest_zcb3_amt as float)
                        + cast(a.order1_amt as float) 
                        + cast(a.order2_amt as float) 
                        + cast(a.order3_amt as float)
                     ) > 0 
                then 1
                else 0
            end as is_invest_zcb_y
            
        , case when (cast(a.invest_lcb1_amt as float) 
                        + cast(a.invest_lcb2_amt as float) 
                        + cast(a.invest_lcb3_amt as float)
                    )  > 0
                then 1
                else 0
            end as is_invest_lcb_y
                    
        -- past invest amout as Xs
        ,(cast(a.invest_amt_zcb1_new as float) 
            + cast(a.invest_amt_zcb3_new as float)
            + cast(a.invest_amt_zcb6_new as float)
            + cast(a.invest_amt_zcb12_new as float)
            + cast(a.invest_amt_zcb18_new as float)
        ) as invest_amt_zcb_x
        
        ,(cast(a.invest_amt_total_new as float) 
            - cast(a.invest_amt_lcb_new as float)  
        ) as invest_amt_zcb_backup
         
        -- inest amt as y 
        ,(cast(a.invest1_amt as float) 
            + cast(a.invest2_amt as float) 
            + cast(a.invest3_amt as float)
            + cast(a.order1_amt as float) 
            + cast(a.order2_amt as float) 
            + cast(a.order3_amt as float)
        ) as invest_amt_all_y
        
        -- zcb and order as y
        ,(cast(a.invest_zcb1_amt as float) 
            + cast(a.invest_zcb2_amt as float) 
            + cast(a.invest_zcb3_amt as float)
            + cast(a.order1_amt as float) 
            + cast(a.order2_amt as float) 
            + cast(a.order3_amt as float)
        ) as invest_amt_zcb_y
        
        -- lcb and order as y
        ,(cast(a.invest_lcb1_amt as float) 
            + cast(a.invest_lcb2_amt as float) 
            + cast(a.invest_lcb3_amt as float)
        ) as invest_amt_lcb_y
        
        ,a.*
 from
         (select
                 -- x: zcb & lcb & total invest amount
                 case when invest_amt_total is null then 0
                      else cast(invest_amt_total as int)
                  end as invest_amt_total_new
                  ,case when invest_amt_lcb is null then 0
                      else cast(invest_amt_lcb as int)
                  end as invest_amt_lcb_new
                  ,case when invest_amt_zcb1 is null then 0
                      else cast(invest_amt_zcb1 as int)
                  end as invest_amt_zcb1_new 
                  ,case when invest_amt_zcb3 is null then 0
                      else cast(invest_amt_zcb3 as int)
                  end as invest_amt_zcb3_new 
                  ,case when invest_amt_zcb6 is null then 0
                      else cast(invest_amt_zcb6 as int)
                  end as invest_amt_zcb6_new 
                  ,case when invest_amt_zcb12 is null then 0
                      else cast(invest_amt_zcb12 as int)
                  end as invest_amt_zcb12_new 
                  ,case when invest_amt_zcb18 is null then 0
                      else cast(invest_amt_zcb18 as int)
                  end as invest_amt_zcb18_new 
                 
                 -- y: invest amt 
                 ,case when invest1 is null then 0
                      else cast(invest1 as int)
                  end as invest1_amt
                  ,case when invest2 is null then 0
                      else cast(invest2 as int)
                  end as invest2_amt
                  ,case when invest3 is null then 0
                      else cast(invest3 as int)
                  end as invest3_amt
                  
                 -- y: YUYUE amt 
                  ,case when order1 is null then 0
                      else cast(order1 as int)
                  end as order1_amt
                  ,case when order2 is null then 0
                      else cast(order2 as int)
                  end as order2_amt
                  ,case when order3 is null then 0
                      else cast(order3 as int)
                  end as order3_amt
                 
                 -- y: ZCB amt 
                  ,case when invest_zcb1 is null then 0
                      else cast(invest_zcb1 as int)
                  end as invest_zcb1_amt
                  ,case when invest_zcb2 is null then 0
                      else cast(invest_zcb2 as int)
                  end as invest_zcb2_amt
                  ,case when invest_zcb3 is null then 0
                      else cast(invest_zcb3 as int)
                  end as invest_zcb3_amt
                  
                  --y: LCB amt
                  ,case when invest_lcb1 is null then 0
                      else cast(invest_lcb1 as int)
                  end as invest_lcb1_amt
                  ,case when invest_lcb2 is null then 0
                      else cast(invest_lcb2 as int)
                  end as invest_lcb2_amt
                  ,case when invest_lcb3 is null then 0
                      else cast(invest_lcb3 as int)
                  end as invest_lcb3_amt
                  
                 ,*
         from
                 qianjin_develop.yunying_model_data_new5
         where
               sex_id is not null
        -- limit 20 
         ) a
where
    a.duanchi_days =0 
    and 
    a.login_times_90days > 1

'''
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
            columnNames = [a['columnName'] for a in cursor.getSchema()]
            df = pd.DataFrame(data=val, columns=columnNames)
            return df
            # return cursor.fetch()
    def close(self):
        self.conn.close()
def main():
    start_time = time.time()
    hive_client = HiveClient(db_host='192.168.176.53', port=15000, user='zhouxiaolei',
            password='5bb785da9dec898163091d80dd61d5cd', database='qianjin_test', authMechanism='PLAIN')
    global sql
    sql = unicode(sql,'utf-8')
    parsedsql = sqlparse.split(sql)
    for i, elt in enumerate(parsedsql):
        sqltmp = elt.replace(';','').encode('utf8')
        result = hive_client.query(sqltmp)
        print ("pulling data is done --- %s seconds ---") % round((time.time() - start_time),2)
        filename = 'data_base_0831_' + str(i) +'.csv' 
        result.to_csv(filename, encoding='utf-8',index=False,index_label=True)
        print ("saving data is done --- %s seconds ---") % round((time.time() - start_time),2)
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
