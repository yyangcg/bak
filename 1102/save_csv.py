# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import datetime

filename = 'oot_training61.csv'
mysql_df = pd.read_csv(filename, low_memory = False)
table_name = str(datetime.date.today())
mysql_df.to_csv('yyh_oot_x_' + table_name + '.csv', encoding='utf-8',index=False,index_label=True)
    # hive_client.close()

        
        


    # hive_client = HiveClient(db_host='192.168.176.53', port=15000, user='zhouxiaolei',
    #         password='5bb785da9dec898163091d80dd61d5cd', database='qianjin_test', authMechanism='PLAIN')

    # #pull test data    
    # sql_test = unicode(sql_test,'utf-8')
    # parsedsql_test = sqlparse.split(sql_test)
    # for elt in enumerate(parsedsql_test):
    #     sqltmp = elt[1].replace(';','').replace('statistic_dayb',statistic_dayb).replace('statistic_daye',statistic_daye).replace('partition_date', partition_date_test).encode('utf8')
    #     print(sqltmp)
    #     result = hive_client.query(sqltmp)
    #     print(result.head())
    #     print("pulling data is done --- %s seconds ---" % round((time.time() - start_time),2))
    #     filename = 'oot_testing' + str(elt[0]) +'.csv' 
    #     result.to_csv(filename, encoding='utf-8',index=False,index_label=True)
    #     print("saving data is done --- %s seconds ---" % round((time.time() - start_time),2))

